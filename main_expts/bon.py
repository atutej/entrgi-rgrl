#!/usr/bin/env python3
"""
Best-of-N (BoN) baseline that uses standard sampling for discrete diffusion.
Usage:
    # Single GPU
    python bon.py --dataset_path allenai/reward-bench-2 --subset_size 10
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 bon.py --dataset_path allenai/reward-bench-2 --subset_size 100
"""

import argparse
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Any, List
from tqdm import tqdm
from datasets import load_dataset

from utils import (
    Config, load_models, 
    compute_discrete_reward, get_model_config,
    sample_tokens, deprioritize_eos,
    add_common_args, save_results,
)


# ============================================================================
# DDP Setup/Teardown
# ============================================================================

def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        return rank, world_size, device, True
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return 0, 1, device, False


def cleanup_distributed(is_distributed: bool):
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def gather_results(local_results: List[Dict], world_size: int, is_distributed: bool) -> List[Dict]:
    if not is_distributed:
        return local_results
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    all_results = []
    for proc_results in gathered:
        all_results.extend(proc_results)
    all_results.sort(key=lambda x: x["idx"])
    return all_results


# ============================================================================
# Batched Dream Generation (all B*K trajectories in parallel)
# ============================================================================

@torch.no_grad()
def generate_dream_batched(
    model,
    prompt_ids_list: List[torch.Tensor],
    prompt_lens: List[int],
    K: int,
    steps: int,
    gen_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    mask_id: int,
    eos_id: int,
    alg: str,
    alg_temp: float,
    do_deprioritize_eos: bool,
    device: torch.device
) -> torch.Tensor:
    """
    Batched Dream generation - all B*K trajectories processed in parallel.
    
    Returns:
        Tensor of shape (B, K, total_len)
    """
    B = len(prompt_ids_list)
    max_prompt_len = max(prompt_lens)
    total_len = max_prompt_len + gen_length
    
    # =========================================================================
    # Initialize trajectories with left-padding
    # =========================================================================
    trajectories = torch.full((B * K, total_len), mask_id, dtype=torch.long, device=device)
    valid_mask = torch.ones((B * K, total_len), dtype=torch.bool, device=device)
    
    pad_id = eos_id  # Use EOS as pad for Dream
    trajectory_pad_lens = []
    
    for b in range(B):
        plen = prompt_lens[b]
        pad_len = max_prompt_len - plen
        
        for k in range(K):
            p = b * K + k
            if pad_len > 0:
                trajectories[p, :pad_len] = pad_id
                valid_mask[p, :pad_len] = False
            trajectories[p, pad_len:max_prompt_len] = prompt_ids_list[b]
            trajectory_pad_lens.append(pad_len)
    
    trajectory_pad_lens = torch.tensor(trajectory_pad_lens, device=device)
    
    # =========================================================================
    # Create 4D attention mask for Dream
    # =========================================================================
    def create_4d_attention_mask(valid_mask_2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        key_valid = valid_mask_2d.unsqueeze(1).unsqueeze(2)
        return torch.where(
            key_valid,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float('-inf'), device=device, dtype=dtype)
        )
    
    attention_mask_4d = create_4d_attention_mask(valid_mask, torch.bfloat16)
    
    # =========================================================================
    # Generation loop
    # =========================================================================
    eps = 1e-3
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    
    for step in range(steps):
        t = timesteps[step]
        s = timesteps[step + 1]
        
        # Forward pass with attention mask and logit shift
        logits = model(trajectories, attention_mask=attention_mask_4d).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        
        # Process each trajectory
        for p in range(B * K):
            pad_len = trajectory_pad_lens[p].item()
            
            # Get mask indices (excluding padding)
            is_mask = (trajectories[p] == mask_id)
            is_mask[:pad_len] = False
            
            num_mask_token = is_mask.sum().item()
            if num_mask_token == 0:
                continue
            
            mask_logits = logits[p][is_mask]
            
            # Sample tokens based on algorithm
            if alg == "maskgit_plus":
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature,
                    top_p=top_p if top_p < 1.0 else None,
                    top_k=top_k
                )
            elif alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature,
                    top_p=top_p if top_p < 1.0 else None,
                    top_k=top_k,
                    margin_confidence=True
                )
            elif alg == "entropy":
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature,
                    top_p=top_p if top_p < 1.0 else None,
                    top_k=top_k,
                    neg_entropy=True
                )
            elif alg == "origin":
                # Random transfer
                p_transfer = 1 - s.item() / t.item() if step < steps - 1 else 1
                x0_full = torch.zeros(num_mask_token, device=device, dtype=torch.long) + mask_id
                transfer_mask = torch.rand(num_mask_token, device=device) < p_transfer
                
                if transfer_mask.any():
                    _, sampled = sample_tokens(
                        mask_logits[transfer_mask],
                        temperature=temperature,
                        top_p=top_p if top_p < 1.0 else None,
                        top_k=top_k
                    )
                    x0_full[transfer_mask] = sampled
                
                trajectories[p, is_mask] = x0_full
                continue
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
            
            # De-prioritize EOS
            if do_deprioritize_eos and eos_id is not None:
                confidence = deprioritize_eos(confidence, x0, eos_id)
            
            # Number of tokens to unmask
            num_transfer = int(num_mask_token * (1 - s.item() / t.item())) if step < steps - 1 else num_mask_token
            
            if num_transfer > 0:
                full_confidence = torch.full((total_len,), -torch.inf, device=device, dtype=logits.dtype)
                full_confidence[is_mask] = confidence
                
                if alg_temp is None or alg_temp == 0:
                    _, transfer_idx = torch.topk(full_confidence, num_transfer)
                else:
                    selection_probs = F.softmax(full_confidence / alg_temp, dim=-1)
                    transfer_idx = torch.multinomial(selection_probs, num_samples=num_transfer)
                
                x_candidate = torch.full_like(trajectories[p], mask_id)
                x_candidate[is_mask] = x0
                trajectories[p, transfer_idx] = x_candidate[transfer_idx]
    
    return trajectories.view(B, K, total_len)


# ============================================================================
# Generation
# ============================================================================

def generate(
    dream_model, dream_tokenizer, reward_model, reward_tokenizer,
    prompts: List[str], cfg: Config, alg: str = "entropy",
    alg_temp: float = None, do_deprioritize_eos: bool = True
) -> List[Dict[str, Any]]:
    """
    Batched Best-of-N baseline for Dream.
    All B*K trajectories are processed in parallel.
    """
    device = cfg.device
    model_cfg = get_model_config(cfg.dream_model, dream_tokenizer)
    mask_id = model_cfg["mask_id"]
    eos_id = model_cfg["eos_id"]
    
    B = len(prompts)
    K = cfg.K
    
    # =========================================================================
    # 1. Tokenize all prompts
    # =========================================================================
    prompt_ids_list = []
    prompt_lens = []
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = dream_tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        prompt_ids_list.append(inputs.input_ids[0].to(device))
        prompt_lens.append(inputs.input_ids.shape[1])
    
    max_prompt_len = max(prompt_lens)
    

    trajectories = generate_dream_batched(
        model=dream_model,
        prompt_ids_list=prompt_ids_list,
        prompt_lens=prompt_lens,
        K=K,
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
        device=device
    )
    
    # trajectories shape: (B, K, total_len)
    
    # =========================================================================
    # 3. Decode and compute rewards
    # =========================================================================
    all_results = []
    
    for b in range(B):
        responses = []
        rewards = []
        
        for k in range(K):
            # Extract response tokens (after max_prompt_len)
            response_tokens = trajectories[b, k, max_prompt_len:]
            resp = dream_tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(resp)
            
            # Compute reward
            r = compute_discrete_reward(
                reward_model, reward_tokenizer,
                prompts[b], resp, device
            )
            rewards.append(r)
        
        best_idx = max(range(K), key=lambda i: rewards[i])
        
        all_results.append({
            "best_response": responses[best_idx],
            "top1_reward": rewards[best_idx],
            "all_responses": responses,
            "all_rewards": rewards,
            "avgN_reward": sum(rewards) / len(rewards),
        })
    
    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    rank, world_size, device, is_distributed = setup_distributed()
    
    parser = argparse.ArgumentParser(description="Best-of-N (BoN) baseline for Dream")
    parser = add_common_args(parser)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", type=str, default="entropy",
                        choices=["origin", "maskgit_plus", "topk_margin", "entropy"])
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
    
    # Load dataset
    if is_main_process(rank):
        print(f"Loading dataset: {cfg.dataset_path}")
    
    dataset = load_dataset(cfg.dataset_path, split=cfg.split).shuffle(seed=cfg.seed)
    if cfg.subset_size:
        dataset = dataset.select(range(min(cfg.subset_size, len(dataset))))
    if cfg.subset_name:
        assert cfg.subset_field is not None, "subset_field must be specified when using subset_name"
        dataset = dataset.filter(lambda x: x[cfg.subset_field] == cfg.subset_name)

    all_indices = list(range(len(dataset)))
    local_indices = all_indices[rank::world_size]
    
    if is_main_process(rank):
        print(f"Dataset size: {len(dataset)}, World size: {world_size}")
    
    # Load models
    dream_model, dream_tokenizer, reward_model, reward_tokenizer, _, _ = load_models(cfg)
    
    model_cfg = get_model_config(cfg.dream_model, dream_tokenizer)
    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"Model type: {model_cfg['type']}")
        print(f"Method: BEST-OF-N")
        print(f"K={cfg.K}, T={cfg.T}, temp={cfg.temperature}")
        print(f"Batch size: {args.batch_size}")
        print(f"{'='*60}\n")
    
    if is_distributed:
        dist.barrier()
    
    local_results = []
    batch_size = args.batch_size
    
    num_batches = (len(local_indices) + batch_size - 1) // batch_size
    batch_range = range(0, len(local_indices), batch_size)
    
    if is_main_process(rank):
        batch_range = tqdm(list(batch_range), desc="Evaluating BoN", total=num_batches)
    
    for batch_start in batch_range:
        batch_indices = local_indices[batch_start:batch_start + batch_size]
        batch_prompts = [dataset[idx][cfg.prompt_field] for idx in batch_indices]
        
        torch.manual_seed(cfg.seed)
        
        batch_results = generate(
            dream_model, dream_tokenizer,
            reward_model, reward_tokenizer,
            batch_prompts, cfg,
            alg=args.alg,
            alg_temp=args.alg_temp,
            do_deprioritize_eos=not args.no_deprioritize_eos
        )
        
        for idx, result in zip(batch_indices, batch_results):
            local_results.append({
                "idx": idx,
                "prompt": dataset[idx][cfg.prompt_field],
                "dataset_all_info": dataset[idx],
                **result
            })
            
            if is_main_process(rank):
                print(f"[{idx}] Top@1={result['top1_reward']:.4f}")
                print(f"    {result['best_response'][:100]}...")
    
        if is_distributed:
            dist.barrier()
    
    all_results = gather_results(local_results, world_size, is_distributed)
    
    if is_main_process(rank):
        top1_rewards = [r["top1_reward"] for r in all_results]
        avgN_rewards = [r["avgN_reward"] for r in all_results]
        
        print(f"\n{'='*60}")
        print(f"RESULTS: BEST-OF-N")
        print(f"{'='*60}")
        print(f"Top@1 Reward:  {sum(top1_rewards)/len(top1_rewards):.4f}")
        print(f"Avg@N Reward:  {sum(avgN_rewards)/len(avgN_rewards):.4f}")
        
        save_results(cfg, f"bon_{args.alg}", top1_rewards, avgN_rewards, all_results)
    
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()