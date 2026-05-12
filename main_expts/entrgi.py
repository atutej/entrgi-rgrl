#!/usr/bin/env python3
"""
Usage:
    # Single GPU
    python entrgi.py --dataset_path allenai/reward-bench-2 --subset_size 10
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 entrgi.py --dataset_path allenai/reward-bench-2 --subset_size 100
"""

import argparse
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset

from utils import (
    Config, load_models, get_model_config,
    RewardCache, build_reward_cache, compute_discrete_reward,
    top_p_filter, top_k_filter, get_confidence_for_alg, deprioritize_eos,
    add_common_args, save_results
)

def setup_distributed():
    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.cuda.current_device()

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )

        return rank, world_size, device, True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, False


def cleanup_distributed(is_distributed: bool):
    """Clean up distributed process group."""
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (for logging/saving)."""
    return rank == 0


def gather_results(local_results: List[Dict], world_size: int, is_distributed: bool) -> List[Dict]:
    """Gather results from all processes to rank 0."""
    if not is_distributed:
        return local_results
    
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    
    all_results = []
    for proc_results in gathered:
        all_results.extend(proc_results)
    all_results.sort(key=lambda x: x["idx"])
    
    return all_results


def generate(dream_model, dream_tokenizer, reward_model, reward_tokenizer,
                                    token_mapping, mapped_embeds, prompts: List[str], 
                                    cfg: Config, model_cfg: Dict) -> List[Dict[str, Any]]:
    """
    Follows original Dream implementation with 4D attention mask and logit shift.

    Args:
        dream_model: The generative model (Dream).
        dream_tokenizer: Tokenizer for the generative model.
        reward_model: The reward model.
        reward_tokenizer: Tokenizer for the reward model.
        token_mapping: Tensor mapping token IDs to embedding indices.
        mapped_embeds: Precomputed token embeddings for the reward model.
        prompts: List of input prompt strings.
        cfg: Configuration object with generation parameters.
        model_cfg: Model-specific configuration dictionary.
    Returns:
        List of dictionaries containing generation results for each prompt.
    """

    device = cfg.device
    mask_id = model_cfg["mask_id"]
    eos_id = model_cfg["eos_id"]
    pad_id = model_cfg["pad_id"]
    reward_embed_layer = reward_model.get_input_embeddings()
    
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
    total_len = max_prompt_len + cfg.max_new_tokens
    
    # =========================================================================
    # 2. Create trajectories with padding
    # =========================================================================
    trajectories = torch.full((B * K, total_len), mask_id, dtype=torch.long, device=device)
    valid_token_mask = torch.ones((B * K, total_len), dtype=torch.bool, device=device)
    
    trajectory_pad_lens = []
    
    for b in range(B):
        plen = prompt_lens[b]
        pad_len = max_prompt_len - plen
        
        for k in range(K):
            p = b * K + k
            if pad_len > 0:
                trajectories[p, :pad_len] = pad_id
                valid_token_mask[p, :pad_len] = False
            trajectories[p, pad_len:max_prompt_len] = prompt_ids_list[b]
            trajectory_pad_lens.append(pad_len)
    
    trajectory_pad_lens = torch.tensor(trajectory_pad_lens, device=device)
    
    # =========================================================================
    # 3. Create 4D attention mask for Dream
    # =========================================================================
    def create_bidirectional_attention_mask(valid_mask_2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        key_valid = valid_mask_2d.unsqueeze(1).unsqueeze(2)
        return torch.where(
            key_valid,
            torch.tensor(0.0, device=valid_mask_2d.device, dtype=dtype),
            torch.tensor(float('-inf'), device=valid_mask_2d.device, dtype=dtype)
        )
    
    attention_mask_4d = create_bidirectional_attention_mask(valid_token_mask, torch.bfloat16)
    
    # =========================================================================
    # 4. Build reward caches
    # =========================================================================
    caches = []
    for b in range(B):
        prompt_ids = prompt_ids_list[b].unsqueeze(0)
        cache = build_reward_cache(reward_model, reward_tokenizer, dream_tokenizer, prompt_ids, device)
        caches.append(cache)
    
    # =========================================================================
    # 5. Generation loop
    # =========================================================================
    eps = 1e-3
    timesteps = torch.linspace(1, eps, cfg.T + 1, device=device)
    
    for step in range(cfg.T):
        t, s = timesteps[step], timesteps[step + 1]
        
        # Forward pass with 4D attention mask and logit shift (Dream-specific)
        with torch.no_grad():
            logits = dream_model(trajectories, attention_mask=attention_mask_4d).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        
        # Get mask indices
        mask_indices = []
        for p in range(B * K):
            is_mask = (trajectories[p] == mask_id)
            pad_len = trajectory_pad_lens[p].item()
            is_mask[:pad_len] = False
            mask_indices.append(is_mask)
        
        mask_counts = [m.sum().item() for m in mask_indices]
        
        if sum(mask_counts) == 0:
            break

        # Optimize logits
        phi_opt, phi_init = optimize_logits(
            logits, mask_indices, trajectories, 
            max_prompt_len,
            trajectory_pad_lens,
            B, K,
            reward_model, caches, token_mapping, mapped_embeds, reward_embed_layer,
            eos_id, pad_id, cfg
        )
        
        # Sample and update
        phi_idx = 0
        for p in range(B * K):
            n_masks = mask_counts[p]
            if n_masks == 0:
                continue
            
            mask_pos = torch.where(mask_indices[p])[0]
            
            if phi_opt is not None and phi_idx + n_masks <= phi_opt.shape[0]:
                opt_logits = phi_opt[phi_idx:phi_idx + n_masks]
                init_logits = phi_init[phi_idx:phi_idx + n_masks]
                phi_idx += n_masks
            else:
                opt_logits = logits[p][mask_indices[p]]
                init_logits = opt_logits
            
            sample_logits = opt_logits / cfg.temperature
            if cfg.top_p < 1.0:
                sample_logits = top_p_filter(sample_logits, cfg.top_p)
            if cfg.top_k is not None:
                sample_logits = top_k_filter(sample_logits, cfg.top_k)
        
            probs = F.softmax(sample_logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            
            if cfg.alg == "origin":
                p_transfer = 1 - s.item() / t.item() if step < cfg.T - 1 else 1
                transfer_mask = torch.rand(n_masks, device=device) < p_transfer
                if transfer_mask.any():
                    trajectories[p, mask_pos[transfer_mask]] = sampled[transfer_mask]
            else:
                old_probs = F.softmax(init_logits / cfg.temperature, dim=-1) if cfg.alg == "entropy" else None
                confidence = get_confidence_for_alg(probs, sampled, cfg.alg, logits=opt_logits, old_probs=old_probs)
                
                if cfg.deprioritize_eos:
                    confidence = deprioritize_eos(confidence, sampled, eos_id)
                
                num_unmask = int(n_masks * (1 - s.item() / t.item())) if step < cfg.T - 1 else n_masks
                
                if num_unmask > 0:
                    if cfg.alg_temp is None or cfg.alg_temp == 0:
                        _, selected = torch.topk(confidence, min(num_unmask, n_masks))
                    else:
                        selection_probs = F.softmax(confidence / cfg.alg_temp, dim=-1)
                        selected = torch.multinomial(selection_probs, min(num_unmask, n_masks), replacement=False)
                    trajectories[p, mask_pos[selected]] = sampled[selected]
    
    # =========================================================================
    # 6. Collect results
    # =========================================================================
    trajectories = trajectories.view(B, K, total_len)
    
    all_results = []
    for b in range(B):
        responses, rewards = [], []
        
        for k in range(K):
            response_tokens = trajectories[b, k, max_prompt_len:]
            resp = dream_tokenizer.decode(response_tokens, skip_special_tokens=True)
            r = compute_discrete_reward(
                reward_model, reward_tokenizer,
                caches[b].user_content, resp, device
            )
            responses.append(resp)
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


def optimize_logits(
    base_logits: torch.Tensor, 
    mask_indices: List[torch.Tensor],
    trajectories: torch.Tensor, 
    max_prompt_len: int,
    trajectory_pad_lens: torch.Tensor,
    B: int, K: int,
    reward_model, caches: List[RewardCache], 
    token_mapping: torch.Tensor,
    mapped_embeds: torch.Tensor, 
    reward_embed_layer,
    eos_token_id: int, 
    pad_token_id: int,
    cfg: Config
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Original Dream logit optimization (unchanged from original).
    """
    total_trajectories = B * K
    device = trajectories.device
    response_len = cfg.max_new_tokens
    embed_dim = mapped_embeds.shape[-1]
    
    all_mask_logits = []
    mask_counts = []
    for p in range(total_trajectories):
        n_masks = mask_indices[p].sum().item()
        mask_counts.append(n_masks)
        if n_masks > 0:
            all_mask_logits.append(base_logits[p][mask_indices[p]])
    
    if not all_mask_logits:
        return None, None
    
    phi = torch.cat(all_mask_logits, dim=0).detach().clone().requires_grad_(True)
    phi_init = phi.detach().clone()
    
    optimizer = torch.optim.Adam([phi], lr=cfg.eta)
    
    max_prefix_len = max(cache.prefix_embeds.shape[1] for cache in caches)
    max_suffix_len = max(cache.suffix_embeds.shape[1] for cache in caches)
    
    batched_prefix = torch.zeros(B, max_prefix_len, embed_dim, device=device, dtype=mapped_embeds.dtype)
    batched_suffix = torch.zeros(B, max_suffix_len, embed_dim, device=device, dtype=mapped_embeds.dtype)
    prefix_lens = []
    suffix_lens = []
    
    for b, cache in enumerate(caches):
        plen = cache.prefix_embeds.shape[1]
        slen = cache.suffix_embeds.shape[1]
        batched_prefix[b, max_prefix_len - plen:] = cache.prefix_embeds[0]
        batched_suffix[b, :slen] = cache.suffix_embeds[0]
        prefix_lens.append(plen)
        suffix_lens.append(slen)
    
    batched_prefix = batched_prefix.repeat_interleave(K, dim=0)
    batched_suffix = batched_suffix.repeat_interleave(K, dim=0)
    
    for _ in range(cfg.M):
        optimizer.zero_grad()
        
        all_response_embeds = torch.zeros(total_trajectories, response_len, embed_dim, 
                                          device=device, dtype=mapped_embeds.dtype)
        all_response_token_ids = trajectories[:, max_prompt_len:].clone()
            
        
        phi_idx = 0
        for p in range(total_trajectories):
            response_mask = mask_indices[p][max_prompt_len:]
            mask_pos = torch.where(response_mask)[0]
            n_masks = len(mask_pos)
            
            if n_masks > 0:
                if cfg.use_entrgi:
                    entropy_probs = F.softmax(phi[phi_idx:phi_idx + n_masks], dim=-1)
                    entropy = -torch.sum(entropy_probs * torch.log(entropy_probs + 1e-10), dim=-1)
                    max_entropy = torch.log(torch.tensor(entropy_probs.shape[-1], device=device, dtype=entropy_probs.dtype))
                    entropy_weight = (entropy / max_entropy).detach()

                    sample_logits = phi[phi_idx:phi_idx + n_masks] / cfg.temperature
                    probs = F.softmax(sample_logits, dim=-1)
                    soft_embeds = torch.matmul(probs, mapped_embeds)
                    if cfg.top_p < 1.0:
                        sample_logits = top_p_filter(sample_logits, cfg.top_p)
                    if cfg.top_k is not None:
                        sample_logits = top_k_filter(sample_logits, cfg.top_k)
                    sample_probs = F.softmax(sample_logits, dim=-1)
                    sampled_tokens = torch.multinomial(sample_probs, 1, replacement=True)
                    hard_embeds = mapped_embeds[sampled_tokens].mean(dim=1)
                    soft_embeds = soft_embeds + (entropy_weight).unsqueeze(-1) * (hard_embeds - soft_embeds).detach()
                elif cfg.use_aps:
                    sample_logits = phi[phi_idx:phi_idx + n_masks] / cfg.temperature
                    probs = F.softmax(sample_logits, dim=-1)
                    soft_embeds = torch.matmul(probs, mapped_embeds)
                    if cfg.top_p < 1.0:
                        sample_logits = top_p_filter(sample_logits, cfg.top_p)
                    if cfg.top_k is not None:
                        sample_logits = top_k_filter(sample_logits, cfg.top_k)
                    sample_probs = F.softmax(sample_logits, dim=-1)
                    sampled_tokens = torch.multinomial(sample_probs, 1, replacement=True)
                    hard_embeds = mapped_embeds[sampled_tokens].mean(dim=1)
                    soft_embeds = soft_embeds + (hard_embeds - soft_embeds).detach()
                else:
                    soft_probs = F.softmax(phi[phi_idx:phi_idx + n_masks] / cfg.temperature, dim=-1)
                    soft_embeds = torch.matmul(soft_probs, mapped_embeds)

                all_response_embeds[p, mask_pos] = soft_embeds
                
                argmax_tokens = phi[phi_idx:phi_idx + n_masks].argmax(dim=-1)
                all_response_token_ids[p, mask_pos] = argmax_tokens
                phi_idx += n_masks
            
            unmasked = ~response_mask
            if unmasked.any():
                unmasked_toks = trajectories[p, max_prompt_len:][unmasked]
                all_response_embeds[p, unmasked] = reward_embed_layer(token_mapping[unmasked_toks]).detach()
        
        full_embeds = torch.cat([
            batched_prefix,
            all_response_embeds,
            batched_suffix
        ], dim=1)
        
        total_len = max_prefix_len + response_len + max_suffix_len
        
        attn_mask = torch.ones(total_trajectories, total_len, device=device, dtype=torch.long)
        
        for p in range(total_trajectories):
            b = p // K
            
            prefix_pad = max_prefix_len - prefix_lens[b]
            if prefix_pad > 0:
                attn_mask[p, :prefix_pad] = 0
            
            suffix_pad = max_suffix_len - suffix_lens[b]
            if suffix_pad > 0:
                attn_mask[p, max_prefix_len + response_len + suffix_lens[b]:] = 0
            
            eos_found = False
            for i in range(response_len):
                tid = all_response_token_ids[p, i].item()
                if eos_found or (pad_token_id is not None and tid == pad_token_id):
                    attn_mask[p, max_prefix_len + i] = 0
                if tid == eos_token_id:
                    eos_found = True

        all_rewards = reward_model(inputs_embeds=full_embeds, attention_mask=attn_mask).logits[:, 0]
        
        loss = -all_rewards.sum()
        loss.backward()
        optimizer.step()

    return phi.detach(), phi_init


# ============================================================================
# Main
# ============================================================================

def main():
    rank, world_size, device, is_distributed = setup_distributed()

    if is_main_process(rank):
        print(f"[Rank {rank}] device={device}")
    
    parser = argparse.ArgumentParser(description="EntRGi/APS implementation for gradient-based reward guidance for discrete diffusion language models")
    parser = add_common_args(parser)
    
    parser.add_argument("--alg", type=str, default="entropy",
                        choices=["origin", "maskgit_plus", "topk_margin", "entropy", "anchor"],)
    parser.add_argument("--alg_temp", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--no_deprioritize_eos", action="store_true")
    parser.add_argument("--M", type=int, default=3, help="Gradient optimization steps")
    parser.add_argument("--eta", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of prompts per GPU")
    parser.add_argument("--use_aps", action="store_true", help="Use APS from Rout et al. 2025")
    parser.add_argument("--use_entrgi", action="store_true", help="Use EntRGi (Ours)")
    
    args = parser.parse_args()
    if args.use_entrgi and (args.use_aps):
        raise ValueError("Only one of --use_entrgi or --use_aps can be set.")
    
    cfg = Config(
        dream_model=args.dream_model,
        reward_model=args.reward_model,
        K=args.K, T=args.T, M=args.M,
        eta=args.eta,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
        deprioritize_eos=not args.no_deprioritize_eos,
        dataset_path=args.dataset_path,
        split=args.split,
        prompt_field=args.prompt_field,
        subset_size=args.subset_size,
        subset_name=args.subset_name,
        subset_field=args.subset_field,
        output_file=args.output_file,
        device=device,
        seed=args.seed,
        use_aps=args.use_aps,
        use_entrgi=args.use_entrgi,
    )
    
    batch_size = args.batch_size
    
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
    
     # Split dataset among processes
    all_indices = list(range(len(dataset)))
    local_indices = all_indices[rank::world_size]
    
    if is_main_process(rank):
        print(f"Total dataset size: {len(dataset)}")
        print(f"World size: {world_size}")
        print(f"Batch size: {batch_size}")
    
    # Load models
    if is_main_process(rank):
        print("Loading models...")
    dream_model, dream_tokenizer, reward_model, reward_tokenizer, token_mapping, mapped_embeds = load_models(cfg)
    
    model_cfg = get_model_config(cfg.dream_model, dream_tokenizer)
    
    if is_distributed:
        dist.barrier()
    
    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"Model type: {model_cfg['type']}")
        print(f"Method: EntRGi/APS")
        print(f"K={cfg.K}, T={cfg.T}, M={cfg.M}, eta={cfg.eta}")
        print(f"World size: {world_size}, Batch size: {batch_size}")
        print(f"{'='*60}\n")
    
    local_results = []
    
    num_batches = (len(local_indices) + batch_size - 1) // batch_size
    batch_range = range(0, len(local_indices), batch_size)
    
    if is_main_process(rank):
        batch_range = tqdm(list(batch_range), desc="Evaluating EntRGi/APS", total=num_batches)
    
    for batch_start in batch_range:
        batch_indices = local_indices[batch_start:batch_start + batch_size]
        batch_prompts = [dataset[idx][cfg.prompt_field] for idx in batch_indices]
        
        torch.manual_seed(cfg.seed)
        
        batch_results = generate(
            dream_model, dream_tokenizer, reward_model, reward_tokenizer,
            token_mapping, mapped_embeds, batch_prompts, cfg, model_cfg
        )
        
        if is_distributed:
            dist.barrier()

        for idx, result in zip(batch_indices, batch_results):
            local_results.append({
                "idx": idx,
                "prompt": dataset[idx][cfg.prompt_field],
                "dataset_all_info": dataset[idx],
                **result
            })
            
            if is_main_process(rank):
                print(f"[{idx}] Top@1={result['top1_reward']:.4f}, Avg@N={result['avgN_reward']:.4f}")
    
    all_results = gather_results(local_results, world_size, is_distributed)
    
    if is_main_process(rank):
        cfg.device = str(cfg.device)
        top1_rewards = [r["top1_reward"] for r in all_results]
        avgN_rewards = [r["avgN_reward"] for r in all_results]
        
        print(f"\n{'='*60}")
        print(f"RESULTS: EntRGi/APS")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_results)}")
        print(f"Top@1 Reward:  {sum(top1_rewards)/len(top1_rewards):.4f}")
        print(f"Avg@N Reward:  {sum(avgN_rewards)/len(avgN_rewards):.4f}")
        
        save_results(cfg, "entrgi_aps", top1_rewards, avgN_rewards, all_results)
    
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()