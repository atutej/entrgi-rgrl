#!/usr/bin/env python3
"""
Aggregate experimental results
Automatically detects methods, datasets, seeds, and temperatures.
Aggregates eval results (LMUnit scores) and reward results (Top@1, Avg@N).
"""

import json
import os
import re
import math
import argparse
from collections import defaultdict


def mean(x):
    return sum(x) / len(x) if x else 0


def std(x, ddof=1):
    if len(x) < 2:
        return 0
    m = mean(x)
    return math.sqrt(sum((xi - m) ** 2 for xi in x) / (len(x) - ddof))


def sem(x):
    """Standard error of the mean."""
    if len(x) < 2:
        return 0
    return std(x) / math.sqrt(len(x))


def extract_method(filename, known_methods=None):
    """Extract method name from filename."""
    if known_methods:
        # Check known methods first (sorted by length, longest first)
        for method in sorted(known_methods, key=len, reverse=True):
            if f"_{method}_seed" in filename:
                return method

    # Fallback: extract dynamically
    patterns = [
        r'_MNT\d+_M\d+_([a-zA-Z][a-zA-Z0-9_]*)_seed\d+',  # entrgi/aps/expectation: MNT128_M3_method
        r'_MNT\d+_([a-zA-Z][a-zA-Z0-9_]*)_seed\d+',       # bon: MNT128_method
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)

    return None


def extract_info(filename):
    """Extract dataset, seed, temperature, and config parameters from filename."""
    info = {}

    # Dataset
    dataset_match = re.search(r'(\w+)_', filename)
    if dataset_match:
        info['dataset'] = dataset_match.group(1)

    # Seed
    seed_match = re.search(r'seed(\d+)', filename)
    if seed_match:
        info['seed'] = int(seed_match.group(1))

    # Temperature
    temp_match = re.search(r'temp([\d.]+)', filename)
    if temp_match:
        info['temperature'] = temp_match.group(1)

    # Config parameters
    # K value (e.g., k1, k2)
    k_match = re.search(r'_k(\d+)_', filename)
    if k_match:
        info['K'] = int(k_match.group(1))

    # T value (e.g., T128)
    t_match = re.search(r'_T(\d+)_', filename)
    if t_match:
        info['T'] = int(t_match.group(1))

    # MNT value (e.g., MNT128)
    mnt_match = re.search(r'_MNT(\d+)_', filename)
    if mnt_match:
        info['MNT'] = int(mnt_match.group(1))

    # M value (e.g., M3)
    m_match = re.search(r'_M(\d+)_', filename)
    if m_match:
        info['M'] = int(m_match.group(1))

    return info


def get_config_key(info, config_params=None):
    """Generate a config key string from extracted info."""
    if config_params is None:
        config_params = ['K', 'T', 'MNT', 'M', 'temperature']

    parts = []
    for param in config_params:
        if param in info:
            # Use 'temp' prefix for temperature to match filename format
            if param == 'temperature':
                parts.append(f"temp{info[param]}")
            else:
                parts.append(f"{param}{info[param]}")

    return "_".join(parts) if parts else "default"


def discover_methods(directory):
    """Discover all unique methods in a directory."""
    methods = set()

    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue

        # Pattern: after config tokens (T128_MNT128 or T128_MNT128_M3), extract method before _seed
        # Match: _MNT128_METHOD_seed or _MNT128_M3_METHOD_seed
        patterns = [
            r'_MNT\d+_M\d+_([a-zA-Z][a-zA-Z0-9_]*)_seed\d+',  # entrgi/aps/expectation: MNT128_M3_method
            r'_MNT\d+_([a-zA-Z][a-zA-Z0-9_]*)_seed\d+',       # bon: MNT128_method
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                methods.add(match.group(1))
                break

    return sorted(methods, key=len, reverse=True)


def load_eval_results(lmunit_path, methods, group_by_config=True):
    """Load evaluation results from lmunit_results folder."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    if not os.path.exists(lmunit_path):
        return results, set()

    all_configs = set()

    for filename in os.listdir(lmunit_path):
        if not filename.endswith('_eval.json'):
            continue

        info = extract_info(filename)
        method = extract_method(filename, methods)

        if not method or 'dataset' not in info or 'seed' not in info:
            continue

        temp = info.get('temperature', 'unknown')
        config_key = get_config_key(info)
        all_configs.add(config_key)

        # Create method key with config if grouping by config
        if group_by_config and config_key != "default":
            method_key = f"{method} ({config_key})"
        else:
            method_key = method

        filepath = os.path.join(lmunit_path, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
                score = data.get('statistics').get('mean_score')
                if score is not None:
                    results[method_key][info['dataset']][temp][info['seed']] = score
        except (json.JSONDecodeError, IOError):
            continue

    return results, all_configs


def load_reward_results(base_path, methods, group_by_config=True):
    """Load reward results from root folder."""
    top1_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    avgN_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    all_configs = set()

    for filename in os.listdir(base_path):
        if not filename.endswith('.json') or filename.endswith('_eval.json'):
            continue
        if os.path.isdir(os.path.join(base_path, filename)):
            continue

        info = extract_info(filename)
        method = extract_method(filename, methods)

        if not method or 'dataset' not in info or 'seed' not in info:
            continue

        temp = info.get('temperature', 'unknown')
        config_key = get_config_key(info)
        all_configs.add(config_key)

        # Create method key with config if grouping by config
        if group_by_config and config_key != "default":
            method_key = f"{method} ({config_key})"
        else:
            method_key = method

        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
                if 'metrics' in data:
                    if 'mean_top1_reward' in data['metrics']:
                        top1_results[method_key][info['dataset']][temp][info['seed']] = data['metrics']['mean_top1_reward']
                    if 'mean_avgN_reward' in data['metrics']:
                        avgN_results[method_key][info['dataset']][temp][info['seed']] = data['metrics']['mean_avgN_reward']
        except Exception as e:
            print(f"Error: {e}")

    return top1_results, avgN_results, all_configs


def get_method_seeds(results, method, datasets, temperature):
    """Get all seeds available for a specific method across its datasets."""
    all_seeds = None
    for dataset in datasets:
        seeds = set(results[method][dataset][temperature].keys())
        if seeds:
            if all_seeds is None:
                all_seeds = seeds
            else:
                all_seeds &= seeds
    return sorted(all_seeds) if all_seeds else []


def print_summary_table(results, methods, datasets, temperature):
    """Print a summary table of results, using all available data per method."""
    print(f"\n{'Method':<35} {'Seeds':<8}", end="")
    for d in datasets:
        print(f"{d.upper():<18} ", end="")
    print(f"{'Overall':<18}")
    print("-" * (43 + 19 * (len(datasets) + 1)))

    for method in methods:
        # Get seeds available for this method
        method_seeds = get_method_seeds(results, method, datasets, temperature)

        row = [f"{method:<35} {len(method_seeds):<8}"]
        all_vals = []

        for dataset in datasets:
            vals = [results[method][dataset][temperature].get(s) for s in method_seeds]
            vals = [v for v in vals if v is not None]
            if vals:
                row.append(f"{mean(vals):.4f}±{sem(vals):.4f}  ")
                all_vals.extend(vals)
            else:
                row.append(f"{'N/A':<18}")

        if all_vals:
            row.append(f"{mean(all_vals):.4f}±{sem(all_vals):.4f}")
        else:
            row.append("N/A")

        print("".join(row))


def main():
    parser = argparse.ArgumentParser(description='Aggregate experimental results')
    parser.add_argument('folder', help='Path to results folder')
    parser.add_argument('--temperature', '-t', help='Filter by temperature (e.g., 0.1, 0.7)')
    parser.add_argument('--methods', '-m', nargs='+', help='Specific methods to include')
    parser.add_argument('--datasets', '-d', nargs='+', help='Specific datasets to include')
    parser.add_argument('--min-seeds', type=int, default=1, help='Minimum number of common seeds required')
    parser.add_argument('--no-reward', action='store_true', help='Skip reward results')
    parser.add_argument('--no-eval', action='store_true', help='Skip eval results')
    parser.add_argument('--no-config', action='store_true', help='Do not group by config (K, T, MNT, M)')

    args = parser.parse_args()

    base_path = os.path.abspath(args.folder)
    lmunit_path = os.path.join(base_path, 'lmunit_results')
    group_by_config = not args.no_config

    print("=" * 100)
    print(f"AGGREGATING RESULTS FROM: {base_path}")
    print("=" * 100)

    # Discover methods
    all_methods = set()
    if os.path.exists(lmunit_path):
        all_methods.update(discover_methods(lmunit_path))
    all_methods.update(discover_methods(base_path))

    methods = sorted(all_methods, key=len, reverse=True)
    if args.methods:
        methods = [m for m in methods if m in args.methods]

    print(f"\nDiscovered methods: {methods}")

    # Load results
    all_configs = set()
    if not args.no_eval:
        eval_results, eval_configs = load_eval_results(lmunit_path, methods, group_by_config)
        all_configs.update(eval_configs)
    else:
        eval_results = {}

    if not args.no_reward:
        top1_results, avgN_results, reward_configs = load_reward_results(base_path, methods, group_by_config)
        all_configs.update(reward_configs)
    else:
        top1_results = {}
        avgN_results = {}

    if all_configs:
        print(f"Discovered configs: {sorted(all_configs)}")

    # Get all method keys from loaded results
    method_keys = set(eval_results.keys()) | set(top1_results.keys()) | set(avgN_results.keys())
    method_keys = sorted(method_keys, key=len, reverse=True)

    # Filter by method name if specified
    if args.methods:
        method_keys = [mk for mk in method_keys if any(m in mk for m in args.methods)]

    print(f"Discovered method configs: {method_keys}")

    # Discover datasets and temperatures
    all_datasets = set()
    all_temps = set()

    for method_key in method_keys:
        for dataset in eval_results.get(method_key, {}):
            all_datasets.add(dataset)
            all_temps.update(eval_results[method_key][dataset].keys())
        for dataset in top1_results.get(method_key, {}):
            all_datasets.add(dataset)
            all_temps.update(top1_results[method_key][dataset].keys())

    datasets = sorted(all_datasets)
    temperatures = sorted(all_temps)

    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]
    if args.temperature:
        temperatures = [t for t in temperatures if t == args.temperature]

    print(f"Discovered datasets: {datasets}")
    print(f"Discovered temperatures: {temperatures}")

    # Process each temperature
    for temp in temperatures:
        print(f"\n{'='*100}")
        print(f"TEMPERATURE: {temp}")
        print(f"{'='*100}")

        # Filter method_keys to only those matching this temperature
        # (when temp is in config key, filter by temp{value} in method name)
        temp_method_keys = [mk for mk in method_keys if f"temp{temp}" in mk or f"temp" not in mk]

        # EVAL RESULTS
        if eval_results and not args.no_eval:
            eval_datasets = [d for d in datasets if any(
                eval_results[m][d][temp] for m in temp_method_keys if d in eval_results.get(m, {})
            )]

            if eval_datasets:
                print(f"\n### EVAL RESULTS (LMUnit Score) ###")

                print_summary_table(eval_results, temp_method_keys, eval_datasets, temp)

        # REWARD RESULTS
        if not args.no_reward:
            # Top@1 results
            if top1_results:
                reward_datasets = [d for d in datasets if any(
                    top1_results[m][d][temp] for m in temp_method_keys if d in top1_results.get(m, {})
                )]

                if reward_datasets:
                    print(f"\n### REWARD RESULTS (Top@1) ###")
                    print_summary_table(top1_results, temp_method_keys, reward_datasets, temp)

            # Avg@N results
            if avgN_results:
                reward_datasets = [d for d in datasets if any(
                    avgN_results[m][d][temp] for m in temp_method_keys if d in avgN_results.get(m, {})
                )]

                if reward_datasets:
                    print(f"\n### REWARD RESULTS (Avg@N) ###")
                    print_summary_table(avgN_results, temp_method_keys, reward_datasets, temp)

    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()