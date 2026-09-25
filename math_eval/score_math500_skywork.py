"""
Separate scoring stage: takes a merged MATH-500 results file produced by eval_math500.py
(--merge) and scores each (problem, candidate) pair with a Skywork reward model, in addition to
the boxed-answer correctness score eval_math500.py already computed. Reports both side by side --
correctness tells you if the final answer was right, the reward model gives a continuous quality
signal over the whole response (reasoning included), independent of whether \\boxed{} matched.

Reuses eval/score_with_skywork.py's exact loading + scoring convention (same reward model family
used elsewhere in this repo for training/guidance) rather than reimplementing it.

Run AFTER generation finishes and all 4 shard processes have exited (frees their GPU memory) --
the reward model (default Skywork-Reward-V2-Qwen3-8B) needs its own GPU. Single GPU, single
process, same pattern as eval/score_wildchat_heldout.py.

Usage:
    python score_math500_skywork.py \\
        --completions_file outputs/full_500/math500_..._42.json \\
        --skywork_reward_model Skywork/Skywork-Reward-V2-Qwen3-8B
"""

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from score_with_skywork import load_reward_model, score as score_completions  # noqa: E402


def load_math500_results(path):
    with open(path) as f:
        data = json.load(f)
    rows = data["results"]
    inst_ids, prompts, completions, correctness = [], [], [], []
    for row in rows:
        inst_ids.append(row["inst_id"])
        prompts.append(row["problem"][-1]["content"])  # chat list -> user turn text
        completions.append(row["candidate"])
        correctness.append(row["score"])
    return inst_ids, prompts, completions, correctness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions_file", type=str, required=True,
                         help="Merged math500 results JSON from eval_math500.py --merge.")
    parser.add_argument("--skywork_reward_model", type=str,
                         default="Skywork/Skywork-Reward-V2-Qwen3-8B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--output_suffix", type=str, default=".skywork_score.json")
    args = parser.parse_args()

    inst_ids, prompts, completions, correctness = load_math500_results(args.completions_file)
    print(f"[{args.completions_file}] loaded {len(prompts)} completions")

    print(f"Loading reward model {args.skywork_reward_model} ...")
    rm, tok = load_reward_model(args.skywork_reward_model)

    print("Scoring ...")
    reward_scores = score_completions(
        rm, tok, prompts, completions,
        batch_size=args.batch_size, max_length=args.max_length,
    )

    n = len(reward_scores)
    reward_mean = sum(reward_scores) / n
    reward_std = (sum((s - reward_mean) ** 2 for s in reward_scores) / n) ** 0.5
    accuracy = sum(correctness) / n

    print(f"MATH-500 boxed-answer accuracy: {accuracy:.4f} (n={n})")
    print(f"{args.skywork_reward_model} reward: mean={reward_mean:.4f} std={reward_std:.4f} (n={n})")

    out_path = args.completions_file + args.output_suffix
    with open(out_path, "w") as f:
        json.dump({
            "completions_file": args.completions_file,
            "skywork_reward_model": args.skywork_reward_model,
            "accuracy": accuracy,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "n": n,
            "per_example": [
                {"inst_id": iid, "correctness": c, "reward": s}
                for iid, c, s in zip(inst_ids, correctness, reward_scores)
            ],
        }, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
