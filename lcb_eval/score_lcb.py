"""
Scores an eval_lcb.py submission file using the OFFICIAL LiveCodeBench evaluator
(lcb_runner.runner.custom_evaluator, run as a subprocess -- see ../LiveCodeBench, cloned from
https://github.com/LiveCodeBench/LiveCodeBench and `pip install -e --no-deps`'d into this conda
env). No custom sandboxing/execution code lives in this repo; this script only orchestrates the
official one. By default eval_lcb.py's "target" set is every problem in range, so its own
whole-range pass@1 already matches what this prints -- the ("target"-only) restriction here only
actually narrows anything when eval_lcb.py was run with --test_sample_size (a random subset), in
which case this avoids diluting the number with problems that were never really attempted (empty
code submitted just to satisfy the evaluator's count assertion, see eval_lcb.py's own docstring).

Usage:
    python score_lcb.py --custom_output_file outputs/.../lcb_..._lcb_submission.json \\
        --release_version release_v1
(add --start_date/--end_date if the eval_lcb.py run used them -- must match exactly, since the
evaluator reconstructs its own benchmark from these same filters.)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

LCB_DIR = Path(__file__).resolve().parent.parent / "LiveCodeBench"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--custom_output_file", type=str, required=True,
                    help="The *_lcb_submission.json file eval_lcb.py's --merge step wrote.")
    p.add_argument("--release_version", type=str, required=True)
    p.add_argument("--start_date", type=str, default=None)
    p.add_argument("--end_date", type=str, default=None)
    p.add_argument("--num_process_evaluate", type=int, default=12)
    p.add_argument("--timeout", type=int, default=6)
    args = p.parse_args()

    custom_output_file = os.path.abspath(args.custom_output_file)

    cmd = [
        sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
        "--scenario", "codegeneration",
        "--release_version", args.release_version,
        "--custom_output_file", custom_output_file,
        "--num_process_evaluate", str(args.num_process_evaluate),
        "--timeout", str(args.timeout),
    ]
    if args.start_date:
        cmd += ["--start_date", args.start_date]
    if args.end_date:
        cmd += ["--end_date", args.end_date]

    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LCB_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(LCB_DIR), env=env)

    # lcb_runner.runner.custom_evaluator's own naming convention (see custom_evaluator.py):
    # output_path = custom_output_file[:-5] + f"_{scenario}_output.json"
    output_path = custom_output_file[:-5] + "_codegeneration_output.json"
    eval_path = output_path.replace(".json", "_eval.json")
    eval_all_path = output_path.replace(".json", "_eval_all.json")

    with open(eval_path) as f:
        metrics = json.load(f)
    with open(eval_all_path) as f:
        eval_all = json.load(f)

    # Restrict to the target (LeetCode/functional) subset -- eval_lcb.py's own merged results
    # file (same path minus "_lcb_submission") records which question_ids those are.
    merged_results_path = custom_output_file.replace("_lcb_submission.json", ".json")
    with open(merged_results_path) as f:
        merged = json.load(f)["results"]
    target_ids = {r["question_id"] for r in merged if r["is_target"]}

    target_entries = [e for e in eval_all if e["question_id"] in target_ids]
    n = len(target_entries)
    pass_at_1 = sum(e["pass@1"] for e in target_entries) / n if n else 0.0

    print(f"\n=== whole release/date-range (includes non-target problems submitted empty) ===")
    print(f"overall pass@1 (raw, from lcb_runner): {metrics[0].get('pass@1', 'N/A')}")
    print(f"\n=== target (LeetCode/functional) subset only, n={n} ===")
    print(f"pass@1: {pass_at_1:.4f}")

    summary_path = custom_output_file.replace("_lcb_submission.json", "_target_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_target": n,
            "pass_at_1_target": pass_at_1,
            "per_question": [
                {"question_id": e["question_id"], "difficulty": e.get("difficulty"),
                 "pass@1": e["pass@1"]}
                for e in target_entries
            ],
        }, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
