#!/usr/bin/env python3
"""
Dream benchmark generator with optional LoRA adapter loading and reward scoring.

This keeps the Dream sampling path from `bon.py`, supports loading a LoRA
adapter on top of Dream, and scores the generated candidates with the reward
model so the outputs remain comparable to `bon.py`.

Examples:
    python bon_infer.py \
        --dataset_path allenai/reward-bench-2 \
        --subset_size 10 \
        --output_file ./results/dream_eval.json

    torchrun --nproc_per_node=4 bon_infer.py \
        --dream_model Dream-org/Dream-v0-Instruct-7B \
        --adapter_model /hdd1/an34232/entrgi_sft_models/dream-entrgi-sft-lora-r32-alllinear/checkpoint-final \
        --dataset_path allenai/reward-bench-2 \
        --output_file ./results/dream_adapter_eval.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from bon import gather_results, generate_dream_batched, is_main_process
from utils import (
    Config,
    TensorJSONEncoder,
    add_common_args,
    compute_discrete_reward,
    get_model_config,
)


def setup_distributed():
    """Initialize distributed inference if available."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        return rank, world_size, device, True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return 0, 1, device, False


def cleanup_distributed(is_distributed: bool):
    if is_distributed:
        dist.destroy_process_group()


def load_prompt_dataset(dataset_path: str, split: str):
    """Load either a Hub dataset or a local json/jsonl dataset."""
    path = Path(dataset_path)
    if path.is_file():
        return load_dataset("json", data_files={split: str(path)}, split=split)
    if path.is_dir():
        split_file = path / f"{split}.jsonl"
        if split_file.exists():
            return load_dataset("json", data_files={split: str(split_file)}, split=split)
    return load_dataset(dataset_path, split=split)


def extract_prompt(row: Dict[str, Any], prompt_field: str) -> str:
    """Extract a prompt from either a plain prompt field or raw messages schema."""
    if prompt_field in row and row[prompt_field] is not None:
        return row[prompt_field]
    if "messages" in row:
        user_parts = [
            msg.get("content", "")
            for msg in row["messages"]
            if msg.get("role") == "user"
        ]
        if user_parts:
            return "\n".join(user_parts).strip()
    raise KeyError(
        f"Could not find prompt field {prompt_field!r} and row has no usable messages field."
    )


def load_dream_for_inference(
    model_name_or_path: str,
    device: str,
    adapter_model: Optional[str] = None,
):
    """
    Load a Dream base model, optionally merge a LoRA adapter, and return
    `(model, tokenizer, resolved_base_model, resolved_adapter_model)`.
    """
    base_model_name = model_name_or_path
    resolved_adapter = adapter_model

    # Allow passing an adapter path directly as --dream_model.
    if resolved_adapter is None:
        try:
            peft_cfg = PeftConfig.from_pretrained(model_name_or_path)
        except Exception:
            peft_cfg = None
        else:
            resolved_adapter = model_name_or_path
            base_model_name = peft_cfg.base_model_name_or_path

    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    if resolved_adapter is not None:
        model = PeftModel.from_pretrained(model, resolved_adapter)
        model = model.merge_and_unload()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    return model, tokenizer, base_model_name, resolved_adapter


def load_reward_model(
    reward_model_name_or_path: str,
    device: str,
):
    """Load the reward model, merging a PEFT adapter if needed."""
    reward_path = Path(reward_model_name_or_path)
    if (reward_path / "adapter_config.json").exists():
        peft_cfg = PeftConfig.from_pretrained(reward_model_name_or_path)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            num_labels=1,
        ).to(device)
        reward_model = PeftModel.from_pretrained(
            reward_model,
            reward_model_name_or_path,
        ).merge_and_unload()
        reward_tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)
        resolved_reward_model = peft_cfg.base_model_name_or_path
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name_or_path,
            torch_dtype=torch.bfloat16,
            num_labels=1,
        ).to(device)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_or_path)
        resolved_reward_model = reward_model_name_or_path

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    return reward_model, reward_tokenizer, resolved_reward_model


def generate_inference(
    dream_model,
    dream_tokenizer,
    reward_model,
    reward_tokenizer,
    prompts: List[str],
    cfg: Config,
    alg: str = "entropy",
    alg_temp: Optional[float] = None,
    do_deprioritize_eos: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate `K` responses per prompt with Dream and rerank with the reward
    model, mirroring the stored fields from `bon.py`.
    """
    device = cfg.device
    model_cfg = get_model_config(cfg.dream_model, dream_tokenizer)
    mask_id = model_cfg["mask_id"]
    eos_id = model_cfg["eos_id"]

    prompt_ids_list = []
    prompt_lens = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = dream_tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_ids_list.append(inputs.input_ids[0].to(device))
        prompt_lens.append(inputs.input_ids.shape[1])

    max_prompt_len = max(prompt_lens)
    trajectories = generate_dream_batched(
        model=dream_model,
        prompt_ids_list=prompt_ids_list,
        prompt_lens=prompt_lens,
        K=cfg.K,
        steps=cfg.T,
        gen_length=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        mask_id=mask_id,
        eos_id=eos_id,
        alg=alg,
        alg_temp=alg_temp,
        do_deprioritize_eos=do_deprioritize_eos,
        device=device,
    )

    all_results = []
    for b, prompt in enumerate(prompts):
        responses = []
        rewards = []
        for k in range(cfg.K):
            response_tokens = trajectories[b, k, max_prompt_len:]
            response = dream_tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response)
            rewards.append(
                compute_discrete_reward(
                    reward_model,
                    reward_tokenizer,
                    prompt,
                    response,
                    device,
                )
            )

        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
        all_results.append(
            {
                "prompt": prompt,
                "best_response": responses[best_idx],
                "top1_reward": rewards[best_idx],
                "all_responses": responses,
                "all_rewards": rewards,
                "avgN_reward": sum(rewards) / len(rewards),
                "selected_index": best_idx,
                "num_candidates": len(responses),
            }
        )

    return all_results


def save_inference_results(
    cfg: Config,
    method_name: str,
    results: List[Dict[str, Any]],
    adapter_model: Optional[str],
    resolved_base_model: str,
    resolved_reward_model: str,
):
    if not cfg.output_file:
        return

    output_path = Path(cfg.output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": vars(cfg),
        "method": method_name,
        "model": {
            "base_model": resolved_base_model,
            "adapter_model": adapter_model,
            "reward_model": resolved_reward_model,
        },
        "metrics": {
            "num_prompts": len(results),
            "num_candidates_per_prompt": cfg.K,
            "mean_top1_reward": (
                sum(r["top1_reward"] for r in results) / len(results) if results else 0.0
            ),
            "mean_avgN_reward": (
                sum(r["avgN_reward"] for r in results) / len(results) if results else 0.0
            ),
        },
        "results": results,
    }

    with open(cfg.output_file, "w") as f:
        json.dump(output_data, f, indent=2, cls=TensorJSONEncoder)
    print(f"\nSaved inference results to: {cfg.output_file}")


def main():
    rank, world_size, device, is_distributed = setup_distributed()

    parser = argparse.ArgumentParser(
        description="Inference-only Dream generator for benchmark evaluation"
    )
    parser = add_common_args(parser)
    parser.set_defaults(reward_model="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    parser.add_argument(
        "--adapter_model",
        type=str,
        default=None,
        help="Optional PEFT/LoRA adapter path or HF repo to merge into Dream.",
    )
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--alg",
        type=str,
        default="entropy",
        choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
    )
    parser.add_argument("--alg_temp", type=float, default=None)
    parser.add_argument("--no_deprioritize_eos", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    cfg = Config(
        dream_model=args.dream_model,
        reward_model=args.reward_model,
        K=args.K,
        T=args.T,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        dataset_path=args.dataset_path,
        split=args.split,
        prompt_field=args.prompt_field,
        subset_size=args.subset_size,
        subset_name=args.subset_name,
        subset_field=args.subset_field,
        output_file=args.output_file,
        device=device,
        seed=args.seed,
    )

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if is_main_process(rank):
        print(f"Loading dataset: {cfg.dataset_path}")

    dataset = load_prompt_dataset(cfg.dataset_path, cfg.split).shuffle(seed=cfg.seed)
    if cfg.subset_size:
        dataset = dataset.select(range(min(cfg.subset_size, len(dataset))))
    if cfg.subset_name:
        assert (
            cfg.subset_field is not None
        ), "subset_field must be specified when using subset_name"
        dataset = dataset.filter(lambda x: x[cfg.subset_field] == cfg.subset_name)

    all_indices = list(range(len(dataset)))
    local_indices = all_indices[rank::world_size]

    if is_main_process(rank):
        print(f"Dataset size: {len(dataset)}, World size: {world_size}")

    dream_model, dream_tokenizer, resolved_base_model, resolved_adapter = (
        load_dream_for_inference(
            cfg.dream_model,
            device=device,
            adapter_model=args.adapter_model,
        )
    )
    reward_model, reward_tokenizer, resolved_reward_model = load_reward_model(
        cfg.reward_model,
        device=device,
    )

    model_cfg = get_model_config(cfg.dream_model, dream_tokenizer)
    if is_main_process(rank):
        print(f"\n{'=' * 60}")
        print(f"Model type: {model_cfg['type']}")
        print("Method: INFERENCE-ONLY")
        print(f"Base model: {resolved_base_model}")
        print(f"Adapter: {resolved_adapter or 'none'}")
        print(f"Reward model: {cfg.reward_model}")
        print(f"K={cfg.K}, T={cfg.T}, temp={cfg.temperature}")
        print(f"Batch size: {args.batch_size}")
        print(f"{'=' * 60}\n")

    if is_distributed:
        dist.barrier()

    local_results = []
    batch_size = args.batch_size
    batch_range = range(0, len(local_indices), batch_size)
    num_batches = (len(local_indices) + batch_size - 1) // batch_size

    if is_main_process(rank):
        batch_range = tqdm(
            list(batch_range),
            desc="Generating responses",
            total=num_batches,
        )

    for batch_start in batch_range:
        batch_indices = local_indices[batch_start : batch_start + batch_size]
        batch_prompts = [extract_prompt(dataset[idx], cfg.prompt_field) for idx in batch_indices]

        torch.manual_seed(cfg.seed)

        batch_results = generate_inference(
            dream_model,
            dream_tokenizer,
            reward_model,
            reward_tokenizer,
            batch_prompts,
            cfg,
            alg=args.alg,
            alg_temp=args.alg_temp,
            do_deprioritize_eos=not args.no_deprioritize_eos,
        )

        for idx, result in zip(batch_indices, batch_results):
            local_results.append(
                {
                    "idx": idx,
                    "prompt": extract_prompt(dataset[idx], cfg.prompt_field),
                    "dataset_all_info": dataset[idx],
                    **result,
                }
            )
            if is_main_process(rank):
                print(f"[{idx}] Top@1={result['top1_reward']:.4f}")
                print(f"    {result['best_response'][:100]}...")

        if is_distributed:
            dist.barrier()

    all_results = gather_results(local_results, world_size, is_distributed)

    if is_main_process(rank):
        save_inference_results(
            cfg,
            method_name=f"bon_infer_{args.alg}",
            results=all_results,
            adapter_model=resolved_adapter,
            resolved_base_model=resolved_base_model,
            resolved_reward_model=resolved_reward_model,
        )

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
