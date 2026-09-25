"""Score already-generated completions (e.g. from run_benchmarks_sampling.py's output JSON, or
the merged alpacaeval2_*.json produced by the launch_dream_benchmarks_* scripts) with any Skywork
reward model, independent of the alpacaeval2 LLM-judge pipeline -- useful for a direct
reward-model-based comparison across checkpoints/backbones without spending judge API calls.

Reuses the exact chat-template + scoring convention from
dllm/dllm/pipelines/rl/grpo/rewards/skywork.py (same reward model family used during
training/guidance), just generalized to read from a completions JSON file instead of being called
inline as a GRPOTrainer reward function.

Usage:
    python score_with_skywork.py \
        --completions_file outputs/alpacaeval2-compare/<run>/alpacaeval2_..._42.json \
        --skywork_reward_model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
"""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_completions(path):
    """Accepts either the merged {"avg_metrics", "args", "results": [...]} shape (alpacaeval2_*.json)
    or a bare list of rows. Each row is expected to have an "output" dict with "prompt" (a chat
    message list, last entry the user turn) and "output" (a list with one completion string) --
    exactly what run_benchmarks_sampling.py / llm_utils.py's canonical_outputs shape produces."""
    with open(path) as f:
        data = json.load(f)
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    if not isinstance(rows, list):
        raise ValueError(f"Unrecognized completions file structure: {path}")

    inst_ids, prompts, completions = [], [], []
    for row in rows:
        out = row.get("output", row)
        prompt = out["prompt"]
        completion = out["output"]
        prompt_text = prompt[-1]["content"] if isinstance(prompt, list) else prompt
        completion_text = completion[0] if isinstance(completion, list) else completion
        inst_ids.append(row.get("inst_id"))
        prompts.append(prompt_text)
        completions.append(completion_text)
    return inst_ids, prompts, completions


def load_reward_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, num_labels=1,
    ).cuda()
    rm.eval()
    return rm, tok


def score(rm, tok, prompts, completions, batch_size=8, max_length=4096):
    scores = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="scoring", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]
        batch_completions = completions[i : i + batch_size]
        convs = [
            [{"role": "user", "content": p}, {"role": "assistant", "content": c}]
            for p, c in zip(batch_prompts, batch_completions)
        ]
        texts = [
            tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in convs
        ]
        with torch.no_grad():
            enc = tok(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length,
            ).to(rm.device)
            batch_scores = rm(**enc).logits.squeeze(-1)
        # squeeze(-1) collapses a batch_size==1 tail dim too far (0-d tensor) -- reshape(-1) first.
        scores.extend(float(s) for s in batch_scores.reshape(-1).tolist())
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--completions_file", type=str, required=True, nargs="+",
        help="One or more completions JSON files.",
    )
    parser.add_argument(
        "--skywork_reward_model", type=str, required=True,
        help="Any Skywork reward model repo id, e.g. Skywork/Skywork-Reward-Llama-3.1-8B-v0.2, "
        "Skywork/Skywork-Reward-V2-Qwen3-0.6B, Skywork/Skywork-Reward-V2-Qwen3-8B.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--output_suffix", type=str, default=".skywork_score.json",
        help="Write <completions_file><output_suffix> with the summary + per-example scores. "
        "Pass an empty string to skip writing.",
    )
    args = parser.parse_args()

    print(f"Loading reward model {args.skywork_reward_model} ...")
    rm, tok = load_reward_model(args.skywork_reward_model)

    for path in args.completions_file:
        inst_ids, prompts, completions = load_completions(path)
        print(f"[{path}] scoring {len(prompts)} completions ...")
        scores = score(
            rm, tok, prompts, completions,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        print(f"[{path}] {args.skywork_reward_model}: mean={mean:.4f} std={std:.4f} n={len(scores)}")

        if args.output_suffix:
            out_path = path + args.output_suffix
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "skywork_reward_model": args.skywork_reward_model,
                        "mean": mean,
                        "std": std,
                        "n": len(scores),
                        "per_example": [
                            {"inst_id": iid, "score": s}
                            for iid, s in zip(inst_ids, scores)
                        ],
                    },
                    f,
                    indent=2,
                )
            print(f"[{path}] wrote {out_path}")


if __name__ == "__main__":
    main()
