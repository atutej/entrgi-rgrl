"""Generate completions on a sample of WildChat prompts, then score them with a Skywork reward
model -- a generalization-style check, as opposed to score_with_skywork.py (which scores
completions already generated as part of the alpacaeval2 pipeline, on alpacaeval2's own prompts,
not WildChat).

NOT held-out, by explicit decision: "allenai/tulu-3-wildchat-if-on-policy-8b" (the dataset every
train.py's --dataset wildchat uses, via dllm.pipelines.rl.grpo.datasets.get_wildchat_questions())
has only 10792 raw rows total (confirmed: get_dataset_split_names returns just ["train"], no
test/val split) -- fewer than get_wildchat_questions's own num_prompts=20000 cutoff, so every
training run actually consumed the ENTIRE deduplicated pool (10752 unique prompts), leaving no
genuinely-unseen slice of this same dataset to draw from. An earlier version of this script
assumed rows beyond index 20000 existed and were never touched by training; that assumption was
wrong for this dataset's actual size, confirmed by a live check (0 rows past the cutoff). Getting
a real held-out guarantee would require either a different dataset entirely, or the broader
allenai/WildChat-1M corpus with training's known prompts explicitly excluded by text -- neither
implemented here. Overlap with training data is accepted as fine for now; this script just draws
a plain sample from the SAME pool get_wildchat_questions() itself builds.

Usage:
    python score_wildchat_heldout.py \
        --model /scratch/.../checkpoints/entrgi_rgrl_clean/dream-grpo-wildchat-1019400/checkpoint-500 \
        --skywork_reward_model Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M \
        --num_held_out_prompts 128
"""

import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from dllm.pipelines.rl.grpo.datasets import get_wildchat_questions


def load_held_out_wildchat_prompts(n, sample_seed=1234):
    """Plain random sample from get_wildchat_questions()'s own pool -- the exact same prompts
    training used are eligible to be drawn here too (see module docstring: no held-out guarantee
    right now, overlap accepted). sample_seed is deliberately NOT get_wildchat_questions's own
    seed=42, just to avoid trivially picking the same ordering/subset every time this is run."""
    pool = get_wildchat_questions()
    texts = [row["prompt"][0]["content"] for row in pool]
    rng = random.Random(sample_seed)
    if n > len(texts):
        raise RuntimeError(f"Requested {n} prompts but the wildchat pool only has {len(texts)}.")
    return rng.sample(texts, n)


def load_dream_backend(model_name_or_path):
    import dllm
    from dllm.pipelines import dream
    from dllm.utils.configs import ModelArguments

    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    model = dllm.utils.get_model(model_args=model_args).eval().cuda()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    sampler = dream.DreamSampler(model=model, tokenizer=tokenizer)
    return model, tokenizer, sampler


def generate(sampler, tokenizer, prompts, max_tokens, temperature, top_p, batch_size):
    import dllm
    from dllm.pipelines import dream

    sampler_config = dream.DreamSamplerConfig(
        max_new_tokens=max_tokens,
        steps=max_tokens,
        temperature=temperature,
        top_p=top_p,
        right_shift_logits=True,
        # DreamSamplerConfig's own dataclass default ("origin") isn't handled by this repo's
        # DreamSampler.sample() at all -- every other script here explicitly overrides it for
        # exactly that reason. "entropy" matches llm_utils.py's own eval-harness default.
        alg="entropy",
    )

    completions = []
    batch_starts = list(range(0, len(prompts), batch_size))
    for start in tqdm(batch_starts, desc="generating", unit="batch"):
        batch = prompts[start : start + batch_size]
        convs = [[{"role": "user", "content": p}] for p in batch]
        input_ids = tokenizer.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=True,
        )
        outputs = sampler.sample(input_ids, sampler_config, return_dict=True)
        texts = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), input_ids)
        completions.extend(texts)
    return completions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Checkpoint dir or HF repo id.")
    parser.add_argument("--skywork_reward_model", type=str, required=True)
    parser.add_argument("--num_held_out_prompts", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--gen_batch_size", type=int, default=8)
    parser.add_argument("--reward_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Where to write the per-prompt results + summary. Defaults to "
        "<model_saving_name>_wildchat_heldout_skywork.json in the cwd.",
    )
    args = parser.parse_args()

    print(f"Sampling WildChat prompts (n={args.num_held_out_prompts}) ...")
    prompts = load_held_out_wildchat_prompts(args.num_held_out_prompts)
    print(f"  got {len(prompts)} prompts from get_wildchat_questions()'s own pool -- NOT "
          f"guaranteed disjoint from training data, overlap accepted (see module docstring).")

    print(f"Loading model {args.model} ...")
    model, tokenizer, sampler = load_dream_backend(args.model)

    print(f"Generating {len(prompts)} completions ...")
    completions = generate(
        sampler, tokenizer, prompts,
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
        batch_size=args.gen_batch_size,
    )

    # Free the policy model before loading the reward model -- both are multi-GB, and this script
    # runs them sequentially on one GPU, not concurrently.
    del model, sampler
    torch.cuda.empty_cache()

    from score_with_skywork import load_reward_model, score as score_completions

    print(f"Loading reward model {args.skywork_reward_model} ...")
    rm, rm_tok = load_reward_model(args.skywork_reward_model)

    print("Scoring ...")
    scores = score_completions(
        rm, rm_tok, prompts, completions,
        batch_size=args.reward_batch_size, max_length=args.max_length,
    )

    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    print(f"{args.model} on sampled WildChat prompts: mean={mean:.4f} std={std:.4f} n={len(scores)}")

    output_file = args.output_file
    if output_file is None:
        model_saving_name = os.path.basename(args.model.rstrip("/")) or "model"
        parent = os.path.basename(os.path.dirname(args.model.rstrip("/")))
        if parent:
            model_saving_name = f"{parent}-{model_saving_name}"
        output_file = f"{model_saving_name}_wildchat_heldout_skywork.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "model": args.model,
                "skywork_reward_model": args.skywork_reward_model,
                "held_out": False,  # see module docstring -- overlap with training data possible
                "mean": mean,
                "std": std,
                "n": len(scores),
                "per_example": [
                    {"prompt": p, "completion": c, "score": s}
                    for p, c, s in zip(prompts, completions, scores)
                ],
            },
            f,
            indent=2,
        )
    print(f"wrote {output_file}")


if __name__ == "__main__":
    main()
