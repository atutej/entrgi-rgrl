"""
Dataset-agnostic multi-GPU sharded runner for run_benchmarks_sampling.py.

Splits whatever benchmark's data list round-robin across --num_gpus GPU-pinned subprocesses of
run_benchmarks_sampling.py (using its own --num_shards/--shard_index), waits for all of them, then
merges the shard result files and recomputes the benchmark's own aggregate_metrics() over the full
(unsharded) set. Never inspects benchmark-specific fields -- only the generic
{"results": [...], "metrics": {...}} envelope every ReasonBenchmark subclass already produces --
so it works for any --benchmark registered in reason_benchmarks/, not just the one used to write
this script.

Usage:
    python run_sharded.py --num_gpus 4 --benchmark ifeval \\
        --model dream/Dream-org/Dream-v0-Instruct-7B \\
        --max_tokens 1280 --temperature 0.1 --top_p 0.9 \\
        --output_dir outputs/dream-v0-instruct-7b --force_overwrite

Every flag other than --num_gpus/--gpu_ids is forwarded verbatim to run_benchmarks_sampling.py,
once per shard, with CUDA_VISIBLE_DEVICES pinned to one GPU each.
"""
import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np

import run_benchmarks_sampling as rbs
from reason_benchmarks.benchmark_utils import load_reason_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated physical GPU ids to pin shards to. Defaults to 0..num_gpus-1.",
    )
    ns, forwarded = parser.parse_known_args()

    gpu_ids = ns.gpu_ids.split(",") if ns.gpu_ids else [str(i) for i in range(ns.num_gpus)]
    num_shards = len(gpu_ids)

    # Parse the forwarded args the same way run_benchmarks_sampling.py does, so the merge step
    # computes the exact same output filenames each subprocess independently derives.
    old_argv = sys.argv
    sys.argv = ["run_benchmarks_sampling.py"] + forwarded
    args = rbs._parse_args()
    sys.argv = old_argv

    if args.benchmark is None or "," in args.benchmark:
        raise ValueError("run_sharded.py handles exactly one --benchmark per invocation")
    if "," in args.max_tokens or "," in args.n:
        raise ValueError("run_sharded.py doesn't support comma-separated --max_tokens/--n sweeps")

    procs = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [sys.executable, "run_benchmarks_sampling.py"] + forwarded + [
            "--num_shards", str(num_shards),
            "--shard_index", str(shard_index),
        ]
        print(f"[shard {shard_index}] GPU {gpu_id}: {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more shards failed: exit codes {exit_codes}")

    if args.skip_eval:
        print("*** --skip_eval set: shards produced model outputs only, nothing to merge ***")
        return

    benchmark_cls = load_reason_benchmark(args.benchmark)
    all_results = []
    for shard_index in range(num_shards):
        shard_args = copy.deepcopy(args)
        shard_args.num_shards = num_shards
        shard_args.shard_index = shard_index
        with open(rbs.output_filename_func(shard_args)) as f:
            all_results.extend(json.load(f)["results"])

    all_metrics = [r["metrics"] for r in all_results]
    if benchmark_cls.implements_aggregation():
        avg_metrics = benchmark_cls.aggregate_metrics(all_metrics)
    else:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = float(np.mean([(0 if key not in m else m[key]) for m in all_metrics]))
    avg_metrics["test_sample_size"] = len(all_metrics)
    if "extraction" in avg_metrics and "accuracy" in avg_metrics:
        avg_metrics["extracted_accuracy"] = (
            avg_metrics["accuracy"] / avg_metrics["extraction"] if avg_metrics["extraction"] > 0 else 0.0
        )

    merged_args = copy.deepcopy(args)
    merged_args.num_shards = 1
    merged_args.shard_index = 0
    output_filename = rbs.output_filename_func(merged_args)
    with open(output_filename, "w") as f:
        json.dump({"avg_metrics": avg_metrics, "args": args.__dict__, "results": all_results}, f, indent=2)
    with open(output_filename.replace(".json", ".json.score"), "w") as f:
        json.dump(avg_metrics, f, indent=2)

    print(f"Merged {num_shards} shards -> {output_filename}")
    print("Average metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
