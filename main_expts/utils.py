#!/usr/bin/env python3
"""
Shared utilities.

Contains:
- Configuration dataclass
- Model loading
- Reward computation
- Common sampling functions
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

class TensorJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles PyTorch tensors and numpy arrays."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for all generation methods."""
    # Model paths
    dream_model: str = "Dream-org/Dream-v0-Instruct-7B"
    reward_model: str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    
    K: int = 8                    # Number of trajectories
    T: int = 64                   # Diffusion steps
    
    # Logit optimization
    M: int = 3                    # Gradient optimization steps
    eta: float = 1.0              # Learning rate for logit optimization (η in Algorithm 1)
    beta: float = 1.0             # Reward temperature
    use_aps: bool = False         # Use APS method Rout et al. 2025
    use_entrgi: bool = False      # Use EntRGi method (Ours)
    
    # Sampling parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = None
    alg: str = "entropy"              # origin, maskgit_plus, topk_margin, entropy, anchor
    alg_temp: Optional[float] = None  # Temperature for stochastic position selection from Dream
    deprioritize_eos: bool = True
    
    # Dataset
    dataset_path: str = ""
    split: str = "test"
    prompt_field: str = "prompt"
    subset_name: Optional[str] = None
    subset_size: Optional[int] = None
    subset_field: Optional[str] = None
    
    # Output
    output_file: Optional[str] = None
    device: str = "cuda:0"
    seed: int = 42


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumsum > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    remove_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, mask)
    return logits.masked_fill(remove_mask, torch.finfo(logits.dtype).min)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits."""
    k = min(k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def sample_tokens(logits: torch.Tensor, temperature: float = 0.7, 
                  top_p: float = None, top_k: int = None,
                  margin_confidence: bool = False,
                  neg_entropy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tokens and compute confidence scores.
    
    Following diffusion_generate() exactly:
    - margin_confidence: confidence = top1_prob - top2_prob
    - neg_entropy: confidence = sum(p * log(p)) (negative entropy)
    - default: confidence = sampled token probability
    
    Returns:
        confidence: Confidence scores for each position
        x0: Sampled token ids
    """
    import torch.distributions as dists
    
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_filter(logits, top_p)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    
    probs = torch.softmax(logits, dim=-1)
    
    # Sample tokens
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    # Margin confidence: top1 - top2
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        confidence = top1_probs - top2_probs
    
    # Negative entropy: sum(p * log(p))
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


def get_confidence_for_alg(probs: torch.Tensor, sampled: torch.Tensor, 
                           alg: str, logits: torch.Tensor = None, old_probs: torch.Tensor = None) -> torch.Tensor:
    """
    Compute confidence scores based on algorithm.
    
    Args:
        probs: Probability distribution [N, vocab_size]
        sampled: Sampled token indices [N]
        alg: Algorithm name (maskgit_plus, topk_margin, entropy, anchor)
        logits: Original logits [N, vocab_size] - required for anchor
    
    Returns:
        confidence: Confidence scores [N]
    """
    if alg == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[..., 0] - sorted_probs[..., 1]
    elif alg == "entropy":
        if old_probs is not None:
            probs = old_probs
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        return torch.sum(probs * log_probs, dim=-1)
    elif alg == "anchor":
        # Use logits of sampled tokens as confidence (before softmax normalization)
        if logits is None:
            raise ValueError("anchor algorithm requires logits")
        return logits.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    else:  # maskgit_plus or default
        return probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


def deprioritize_eos(confidence: torch.Tensor, sampled: torch.Tensor, 
                     eos_id: int) -> torch.Tensor:
    """
    De-prioritize EOS tokens by setting their confidence to -inf.
    
    Args:
        confidence: Confidence scores [N]
        sampled: Sampled token indices [N]
        eos_id: EOS token id
    
    Returns:
        Updated confidence tensor
    """
    eos_mask = (sampled == eos_id)
    if eos_mask.any():
        confidence = confidence.clone()
        confidence[eos_mask] = float('-inf')
    return confidence


# =============================================================================
# Model Loading
# =============================================================================

def get_vocab_size(model, model_name: str) -> int:
    """Get vocab size from model."""
    # Dream uses lm_head
    return model.lm_head.out_features

def get_model_config(model_name: str, tokenizer):
    """Returns model-specific configuration for Dream."""
    return {
        "type": "dream",
        "mask_id": tokenizer.mask_token_id,
        "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "shift_logits": True,
        "use_4d_attention_mask": True,
    }


def load_models(cfg: Config):
    """Load Dream and reward models."""
    # Dream model - use .to(device) explicitly
    dream_model = AutoModel.from_pretrained(
        cfg.dream_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(cfg.device).eval()
    dream_tokenizer = AutoTokenizer.from_pretrained(cfg.dream_model, trust_remote_code=True)
    
    for p in dream_model.parameters():
        p.requires_grad = False
    
    # Reward model (with LoRA support)
    # Extract device index for device_map, or use .to() instead
    reward_path = Path(cfg.reward_model)
    if (reward_path / "adapter_config.json").exists():
        from peft import PeftModel, PeftConfig
        peft_cfg = PeftConfig.from_pretrained(cfg.reward_model)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path, torch_dtype=torch.bfloat16,
            num_labels=1
        ).to(cfg.device)  # Changed: use .to() instead of device_map
        reward_model = PeftModel.from_pretrained(reward_model, cfg.reward_model).merge_and_unload()
        reward_tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.reward_model, torch_dtype=torch.bfloat16, num_labels=1
        ).to(cfg.device)  # Changed: use .to() instead of device_map
        reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model)
    
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False
    
    # Token mapping for soft embeddings
    dream_vocab = dream_tokenizer.get_vocab()
    reward_vocab = reward_tokenizer.get_vocab()
    vocab_size = get_vocab_size(dream_model, cfg.dream_model)
    unk_id = reward_tokenizer.unk_token_id or reward_tokenizer.eos_token_id
    
    token_mapping = torch.full((vocab_size,), unk_id, dtype=torch.long, device=cfg.device)
    for tok, did in dream_vocab.items():
        if did < vocab_size and tok in reward_vocab:
            token_mapping[did] = reward_vocab[tok]

    # print overlap statistics
    num_mapped = (token_mapping != unk_id).sum().item()
    print(f"Mapped {num_mapped}/{vocab_size} tokens ({100.0 * num_mapped / vocab_size:.2f}%) from Model to Reward Model.")
    
    reward_embeds = reward_model.get_input_embeddings()
    mapped_embeds = reward_embeds.weight[token_mapping].detach()
        
        
    return dream_model, dream_tokenizer, reward_model, reward_tokenizer, token_mapping, mapped_embeds

# =============================================================================
# Reward Computation
# =============================================================================

@dataclass
class RewardCache:
    """Cached prefix/suffix embeddings for fast reward computation."""
    prefix_embeds: torch.Tensor
    suffix_embeds: torch.Tensor
    user_content: str


def build_reward_cache(reward_model, reward_tokenizer, dream_tokenizer, 
                       prompt_ids: torch.Tensor, device: str) -> RewardCache:
    """Precompute reward model prefix/suffix embeddings."""
    prompt_text = dream_tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
    user_content = prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0] \
                   if "<|im_start|>user\n" in prompt_text else prompt_text
    
    conversation = [{"role": "user", "content": user_content}, 
                    {"role": "assistant", "content": "<<PLACEHOLDER>>"}]
    template = reward_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    prefix_text, suffix_text = template.split("<<PLACEHOLDER>>")
    
    embed_layer = reward_model.get_input_embeddings()
    with torch.no_grad():
        prefix_ids = reward_tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)
        suffix_ids = reward_tokenizer(suffix_text, return_tensors="pt").input_ids.to(device)
        prefix_embeds = embed_layer(prefix_ids)
        suffix_embeds = embed_layer(suffix_ids)
    
    return RewardCache(prefix_embeds, suffix_embeds, user_content)


def compute_discrete_reward(reward_model, reward_tokenizer, prompt: str, 
                            response: str, device: str) -> float:
    """Compute reward from discrete tokens (for final evaluation)."""
    conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    text = reward_tokenizer.apply_chat_template(conversation, tokenize=False)
    if reward_tokenizer.bos_token and text.startswith(reward_tokenizer.bos_token):
        text = text[len(reward_tokenizer.bos_token):]
    inputs = reward_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        return reward_model(**inputs).logits[0, 0].item()


def extract_user_content(dream_tokenizer, prompt_ids: torch.Tensor) -> str:
    """Extract user content from tokenized prompt."""
    prompt_text = dream_tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
    if "<|im_start|>user\n" in prompt_text:
        return prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    return prompt_text


# =============================================================================
# Argument Parsing
# =============================================================================

def add_common_args(parser):
    """Add common arguments shared by all methods."""
    # Model paths
    parser.add_argument("--dream_model", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--reward_model", type=str, default="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    
    # Core parameters
    parser.add_argument("--K", type=int, default=8, help="Number of trajectories")
    parser.add_argument("--T", type=int, default=64, help="Diffusion steps")
    
    # Sampling
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_field", type=str, default="prompt")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--subset_name", type=str, default=None)
    parser.add_argument("--subset_field", type=str, default=None)
    
    # Output
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser


# =============================================================================
# Result Saving
# =============================================================================


def save_results(cfg: Config, method_name: str, top1_rewards: List[float],
                 avgN_rewards: List[float], results: List[Dict]):
    """Save results to JSON file, with tensors saved separately in binary format."""
    if cfg.output_file:
        #create output directory if needed
        output_path = Path(cfg.output_file).parent
        output_path.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": vars(cfg),
            "method": method_name,
            "metrics": {
                "mean_top1_reward": sum(top1_rewards) / len(top1_rewards),
                "mean_avgN_reward": sum(avgN_rewards) / len(avgN_rewards),
                "all_top1_rewards": top1_rewards,
                "all_avgN_rewards": avgN_rewards,
            },
            "results": results,
        }

        # Save JSON metadata
        with open(cfg.output_file, "w") as f:
            json.dump(output_data, f, indent=2, cls=TensorJSONEncoder)
        print(f"\nSaved metadata to: {cfg.output_file}")