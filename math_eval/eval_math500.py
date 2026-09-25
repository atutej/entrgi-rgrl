"""
MATH-500 evaluation for Dream-org/Dream-v0-Instruct-7B (or any dllm-loadable Dream checkpoint).

One process = one shard = one GPU (pin with CUDA_VISIBLE_DEVICES before launching; this script
itself never touches CUDA_VISIBLE_DEVICES). `run_math500_eval.sh` launches --num_shards copies of
this script, each with a different --shard_index, on 4 GPUs, then calls this same script again
with --merge to combine the per-shard result files into one final accuracy number.

Decoding loop seam:
    build_sampler() and build_generation_hooks() below are the ONLY two places that touch how a
    token is produced. With --use_rgrl_guidance, build_sampler() constructs
    dllm.pipelines.rl.rgrl.sampler.RgrlDreamSampler instead of the plain DreamSampler -- its
    _optimize_logits runs entrgi (entropy-aware) guidance at every denoising step, taking a few
    Adam gradient steps on the masked-position logits to increase a Skywork reward model's score
    on the (prompt + in-progress completion) before those logits are used to pick which tokens to
    unmask. build_generation_hooks()'s (tokens_hook, logits_hook) pair is the OTHER way to
    intercept logits (called once per step as (step, x, logits) on the plain DreamSampler path);
    RgrlDreamSampler.sample() ignores them (guidance is inlined into its own loop instead), so
    they're only live when --use_rgrl_guidance is off. Every other function in this file (data
    loading, scoring, sharding, merging) is unrelated to decoding and doesn't change either way.
"""

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# reason_benchmarks/math_500.py is only used for its data loading + scoring (compute_score) --
# it has no decoding loop of its own. It resolves its test_file path relative to eval/, so we add
# eval/ to sys.path and chdir into it rather than duplicating the benchmark definition here.
# ORIG_CWD is captured BEFORE the chdir so relative --output_dir/--model paths the user passes on
# the command line resolve against wherever they invoked this script from, not against eval/.
ORIG_CWD = os.getcwd()
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))
os.chdir(EVAL_DIR)

from reason_benchmarks.math_500 import MATH500  # noqa: E402

import dllm  # noqa: E402
from dllm.pipelines import dream  # noqa: E402
from dllm.utils.configs import ModelArguments  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Dream-org/Dream-v0-Instruct-7B",
                    help="HF repo id or local checkpoint dir, loaded via dllm.utils.get_model.")
    p.add_argument("--seed", type=int, default=42)

    # sharding: round-robin split of the 500 MATH-500 problems across concurrent single-GPU
    # processes. Dataset-agnostic slicing (data[shard_index::num_shards]), same convention as
    # eval/run_sharded.py.
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)

    # decoding args, forwarded to dream.DreamSamplerConfig
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--steps", type=int, default=None, help="Defaults to --max_new_tokens.")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--alg", type=str, default="entropy",
                    choices=["maskgit_plus", "topk_margin", "entropy"])
    p.add_argument("--alg_temp", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=8, help="Prompts per forward-pass batch.")

    p.add_argument("--test_sample_size", type=int, default=None,
                    help="Subsample this many MATH-500 problems (before sharding) for a quick run.")

    # entrgi/RGRL guidance during denoising -- see dllm.pipelines.rl.rgrl.sampler.RgrlDreamSampler.
    # Defaults mirror this repo's own production RGRL training config
    # (vista_sbatch/dream_rgrl_entrgi.sbatch). guidance_kl_beta's frozen-base logits come from
    # model.disable_adapter() (PEFT/LoRA-only) -- this eval script's checkpoint has no trained
    # LoRA adapter, so when guidance_kl_beta != 0.0, run_generation() wraps the model in a
    # freshly-initialized (untrained) LoRA adapter purely so disable_adapter() is a valid context
    # manager. PEFT's default LoRA init sets the B matrix to all zeros, so the adapter's delta is
    # exactly 0 either way -- "frozen base" and "current policy" are thus the SAME distribution
    # here (there's no training loop adapting the LoRA weights in eval), so this trust region
    # anchors guidance to the model's own unguided logits at each masked position, not to some
    # separately-trained reference model.
    p.add_argument("--use_rgrl_guidance", action="store_true", default=False,
                    help="Guide denoising with entrgi/aps reward-model gradient steps (RgrlDreamSampler) "
                    "instead of plain unguided DreamSampler decoding.")
    p.add_argument("--guidance_reward_model", type=str, default="Skywork/Skywork-Reward-V2-Qwen3-8B")
    p.add_argument("--guidance_type", type=str, default="entrgi", choices=["entrgi", "aps"])
    p.add_argument("--guidance_M", type=int, default=3, help="Adam gradient steps per guidance call.")
    p.add_argument("--guidance_eta", type=float, default=0.5, help="Adam learning rate for phi.")
    p.add_argument("--guidance_kl_beta", type=float, default=0.0,
                    help="Trust-region coefficient anchoring guidance to the unguided model's own "
                    "logits at masked positions (see comment above). 0.0 = unconstrained.")
    p.add_argument("--guidance_no_temp_scale_embed", action="store_true", default=False,
                    help="Skip dividing by --temperature when building the soft/hard embedding "
                    "mixture inside _optimize_logits (see _make_no_embed_temp_scale_sampler_cls). "
                    "Does not affect the separate, final per-step token-selection sampling.")
    p.add_argument("--guidance_deprioritize_eos", action="store_true", default=False,
                    help="RgrlDreamSamplerConfig's own default is True: set confidence=-inf at "
                    "every masked position currently predicting EOS, to avoid premature "
                    "termination. CONFIRMED BUGGY at low --temperature here: when many masked "
                    "positions predict EOS at once, torch.topk's tie-breaking over a mostly -inf "
                    "confidence row can select already-RESOLVED (non-mask) positions, and the "
                    "commit step then overwrites them back to mask_token_id -- corrupting "
                    "already-generated content (reproduced: same batch went from coherent output "
                    "at deprioritize_eos=False to an empty string / a leaked prompt fragment / "
                    "literal unresolved <|mask|> tokens at the default True). Defaults to False "
                    "here specifically because of that; pass this flag to opt back into the "
                    "buggy default for comparison.")
    p.add_argument("--zero_unmatched_embeddings", action="store_true", default=False,
                    help="Zero the reward-embedding-space vector for policy tokens absent from the "
                    "reward model's vocab, instead of mapping them to its <unk>/eos token.")

    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--force_overwrite", action="store_true", default=False)

    # combine all --num_shards result files (produced by prior invocations of this script) into
    # one final accuracy number, instead of running generation.
    p.add_argument("--merge", action="store_true", default=False)

    args = p.parse_args()

    # Resolve against the caller's original cwd, not eval/ (see ORIG_CWD comment above) --
    # otherwise a relative --output_dir silently lands under eval/outputs instead of wherever the
    # user ran this from.
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(ORIG_CWD, args.output_dir)
    local_model_path = os.path.join(ORIG_CWD, args.model)
    if not os.path.isabs(args.model) and os.path.isdir(local_model_path):
        args.model = local_model_path

    return args


def model_saving_name(model_path: str) -> str:
    return model_path.rstrip("/").replace("/", "-")


def shard_output_path(args) -> str:
    name = model_saving_name(args.model)
    steps = args.steps or args.max_new_tokens
    tag = f"math500_{name}_max{args.max_new_tokens}steps{steps}t{args.temperature}p{args.top_p}_{args.seed}"
    if getattr(args, "use_rgrl_guidance", False):
        tag += f"_rgrl-{args.guidance_type}-M{args.guidance_M}"
        if args.guidance_kl_beta != 0.0:
            tag += f"-kl{args.guidance_kl_beta}"
        if getattr(args, "guidance_no_temp_scale_embed", False):
            tag += "-notemp"
        if getattr(args, "guidance_deprioritize_eos", False):
            tag += "-deprioreos"
    suffix = f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    return os.path.join(args.output_dir, f"{tag}{suffix}.json")


def merged_output_path(args) -> str:
    merged_args = argparse.Namespace(**vars(args))
    merged_args.num_shards = 1
    merged_args.shard_index = 0
    return shard_output_path(merged_args)


def load_shard(args):
    data = MATH500.load()
    if args.test_sample_size is not None:
        rng = random.Random(args.seed)
        data = rng.sample(data, min(args.test_sample_size, len(data)))
    return data[args.shard_index::args.num_shards]


def load_guidance_model(policy_model, policy_tokenizer, model_name, device, zero_unmatched=False):
    """Reward model + a [policy_vocab_size, reward_embed_dim] embedding table mapping each policy
    token id into the reward model's embedding space -- exactly what RgrlDreamSampler._optimize_logits
    needs to score soft/hard token mixtures at masked positions without re-tokenizing through the
    reward model's own vocab. Ported from dllm.pipelines.rl.rgrl.trainer.RGRLTrainer._load_guidance_model
    (that version reads off self.model/self.processing_class/self.accelerator.device; this is the
    same logic with those passed in explicitly, since there's no trainer here)."""
    reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, dtype=torch.bfloat16, num_labels=1,
    ).to(device)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    policy_vocab = policy_tokenizer.get_vocab()
    reward_vocab = reward_tokenizer.get_vocab()
    try:
        vocab_size = policy_model.lm_head.out_features
    except AttributeError:
        vocab_size = policy_model.config.vocab_size
    reward_embeds = reward_model.get_input_embeddings()

    if zero_unmatched:
        embed_dim = reward_embeds.weight.shape[1]
        mapped_embeds = torch.zeros(
            vocab_size, embed_dim, dtype=reward_embeds.weight.dtype, device=device
        )
        token_mapping = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        for tok, did in policy_vocab.items():
            if did < vocab_size and tok in reward_vocab:
                token_mapping[did] = reward_vocab[tok]
        matched = token_mapping >= 0
        mapped_embeds[matched] = reward_embeds.weight[token_mapping[matched]].detach()
    else:
        unk_id = reward_tokenizer.unk_token_id or reward_tokenizer.eos_token_id
        token_mapping = torch.full((vocab_size,), unk_id, dtype=torch.long, device=device)
        for tok, did in policy_vocab.items():
            if did < vocab_size and tok in reward_vocab:
                token_mapping[did] = reward_vocab[tok]
        mapped_embeds = reward_embeds.weight[token_mapping].detach()

    return reward_model, reward_tokenizer, token_mapping, mapped_embeds


def attach_noop_disable_adapter(model):
    """RgrlDreamSampler.sample()'s guidance_kl_beta trust region calls self.model.disable_adapter()
    (PEFT-only) to get a second 'frozen base' forward pass to build the KL reference distribution
    from. This eval script never trains/adapts the model -- weights are frozen throughout, only
    _optimize_logits's ephemeral `phi` tensor gets gradient steps -- so there's no real adapter to
    disable, and a genuinely no-op one would reproduce that SAME forward pass's output exactly
    (same x, same attention_mask, same position_ids both times). Rather than doing a wasted
    identical second forward pass (or adding real, pointless PEFT/LoRA modules just to get a valid
    context manager), monkeypatch model.forward to cache the real call's output and have
    disable_adapter()'s block replay that cached output -- frozen_logits ends up built from the
    exact same tensor phi_init already is, at zero extra compute."""
    cache = {}
    real_forward = model.forward

    def caching_forward(*args, **kwargs):
        out = real_forward(*args, **kwargs)
        cache["out"] = out
        return out

    @contextlib.contextmanager
    def disable_adapter():
        model.forward = lambda *a, **kw: cache["out"]
        try:
            yield
        finally:
            model.forward = caching_forward

    model.forward = caching_forward
    model.disable_adapter = disable_adapter


def _make_no_embed_temp_scale_sampler_cls():
    """RgrlDreamSampler subclass whose _optimize_logits is a VERBATIM copy of
    dllm.pipelines.rl.rgrl.sampler.RgrlDreamSampler._optimize_logits (as of this writing) with
    exactly one change: `sample_logits = cur_phi / config.temperature` -> `sample_logits = cur_phi`.
    That line feeds BOTH the soft combination (probs -> soft_embeds) and the hard-token sampling
    (sample_probs -> sampled_tokens -> hard_embeds) used to score the embedding mixture the reward
    model sees -- at temperature=0.1 (this eval's near-greedy recipe) dividing by it sharpens phi's
    softmax ~10x before any Adam step even runs, which is suspected to destabilize guidance. This
    does NOT touch the separate, unrelated temperature scaling in sample_tokens() (sample()'s own
    final per-step token-selection call) -- only the internal embedding-mixture computation inside
    the guidance optimization itself.

    A subclass copy (not a monkeypatch/edit) is used deliberately so the actual production
    RgrlDreamSampler used by dllm.pipelines.rl.rgrl.trainer.RGRLTrainer is completely untouched --
    this is a math_eval-only experiment variant, not a change to the training path."""
    from dllm.pipelines.rl.rgrl.sampler import RgrlDreamSampler
    import torch.nn.functional as F
    from dllm.pipelines.dream.models.generation_utils import top_k_logits, top_p_logits

    class RgrlDreamSamplerNoEmbedTempScale(RgrlDreamSampler):
        def _optimize_logits(
            self, base_logits, mask_index, x, max_prompt_len, caches, K, config, device,
            frozen_base_logits=None,
        ):
            B_total = x.size(0)
            response_len = x.size(1) - max_prompt_len
            embed_dim = self.mapped_embeds.shape[-1]

            all_mask_logits, frozen_mask_logits, mask_counts = [], [], []
            for p in range(B_total):
                n = mask_index[p].sum().item()
                mask_counts.append(n)
                if n > 0:
                    all_mask_logits.append(base_logits[p][mask_index[p]])
                    if frozen_base_logits is not None:
                        frozen_mask_logits.append(frozen_base_logits[p][mask_index[p]])

            if not all_mask_logits:
                return None, None, None

            phi = torch.cat(all_mask_logits, dim=0).detach().clone().requires_grad_(True)
            phi_init = phi.detach().clone()

            if config.guidance_kl_beta != 0.0 and frozen_mask_logits:
                ref_logp = F.log_softmax(torch.cat(frozen_mask_logits, dim=0), dim=-1)
            else:
                ref_logp = None

            optimizer = torch.optim.Adam([phi], lr=config.eta)

            B_unique = len(caches)
            max_prefix_len = max(c[0].shape[1] for c in caches)
            max_suffix_len = max(c[1].shape[1] for c in caches)

            batched_prefix = torch.zeros(
                B_unique, max_prefix_len, embed_dim, device=device, dtype=self.mapped_embeds.dtype
            )
            batched_suffix = torch.zeros(
                B_unique, max_suffix_len, embed_dim, device=device, dtype=self.mapped_embeds.dtype
            )
            prefix_lens, suffix_lens = [], []

            for b, (pre, suf, _) in enumerate(caches):
                plen, slen = pre.shape[1], suf.shape[1]
                batched_prefix[b, max_prefix_len - plen:] = pre[0]
                batched_suffix[b, :slen] = suf[0]
                prefix_lens.append(plen)
                suffix_lens.append(slen)

            batched_prefix = batched_prefix.repeat_interleave(K, dim=0)
            batched_suffix = batched_suffix.repeat_interleave(K, dim=0)

            eos_id = self.tokenizer.eos_token_id
            pad_id = getattr(self.tokenizer, "pad_token_id", None) or eos_id
            embed_dtype = self.mapped_embeds.dtype
            ew_accum = []

            for _ in range(config.M):
                optimizer.zero_grad()

                all_response_token_ids = x[:, max_prompt_len:].clone()

                seq_embed_list = []
                phi_idx = 0
                for p in range(B_total):
                    response_mask = mask_index[p][max_prompt_len:]
                    mask_pos = torch.where(response_mask)[0]
                    n_masks = len(mask_pos)
                    unmasked = ~response_mask

                    seq_embed = torch.zeros(
                        response_len, embed_dim, device=device, dtype=embed_dtype
                    )
                    if unmasked.any():
                        unmasked_toks = x[p, max_prompt_len:][unmasked]
                        unmasked_pos = torch.where(unmasked)[0]
                        seq_embed = seq_embed.index_put(
                            (unmasked_pos,),
                            self.mapped_embeds[unmasked_toks],
                        )

                    if n_masks > 0:
                        cur_phi = phi[phi_idx: phi_idx + n_masks]

                        entropy_probs = F.softmax(cur_phi, dim=-1)
                        entropy = -torch.sum(
                            entropy_probs * torch.log(entropy_probs + 1e-10), dim=-1
                        )
                        max_entropy = torch.log(
                            torch.tensor(
                                entropy_probs.shape[-1], device=device, dtype=entropy_probs.dtype,
                            )
                        )
                        if config.guidance_type == "aps":
                            entropy_weight = torch.ones_like(entropy)
                        else:
                            entropy_weight = (entropy / max_entropy).detach()
                        ew_accum.append(entropy_weight.mean().item())

                        # --- ONLY CHANGE vs the original: no `/ config.temperature` here ---
                        sample_logits = cur_phi
                        probs = F.softmax(sample_logits, dim=-1)
                        soft_embeds = torch.matmul(probs.to(embed_dtype), self.mapped_embeds)

                        if config.top_p is not None and config.top_p < 1.0:
                            sample_logits = top_p_logits(sample_logits, config.top_p)
                        if config.top_k is not None:
                            sample_logits = top_k_logits(sample_logits, config.top_k)
                        sample_probs = F.softmax(sample_logits.float(), dim=-1)
                        sampled_tokens = torch.multinomial(sample_probs, 1, replacement=True)
                        hard_embeds = self.mapped_embeds[sampled_tokens].mean(dim=1)
                        soft_embeds = (
                            soft_embeds
                            + entropy_weight.to(embed_dtype).unsqueeze(-1)
                            * (hard_embeds - soft_embeds).detach()
                        )

                        seq_embed = seq_embed.index_put((mask_pos,), soft_embeds)
                        all_response_token_ids[p, mask_pos] = cur_phi.argmax(dim=-1)
                        phi_idx += n_masks

                    seq_embed_list.append(seq_embed)

                all_response_embeds = torch.stack(seq_embed_list, dim=0)

                total_len = max_prefix_len + response_len + max_suffix_len
                full_embeds = torch.cat(
                    [batched_prefix, all_response_embeds, batched_suffix], dim=1
                )

                attn_mask = torch.ones(B_total, total_len, device=device, dtype=torch.long)
                for p in range(B_total):
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
                        if eos_found or (pad_id is not None and tid == pad_id):
                            attn_mask[p, max_prefix_len + i] = 0
                        if tid == eos_id:
                            eos_found = True

                rewards = self.reward_model(
                    inputs_embeds=full_embeds, attention_mask=attn_mask.bool()
                ).logits[:, 0]
                loss = -rewards.sum()

                if ref_logp is not None:
                    cur_logp = F.log_softmax(phi, dim=-1)
                    cur_probs = cur_logp.exp()
                    kl = (cur_probs * (cur_logp - ref_logp)).sum(-1)
                    loss = loss + config.guidance_kl_beta * kl.sum()

                loss.backward()
                optimizer.step()

            ew_mean = sum(ew_accum) / len(ew_accum) if ew_accum else 1.0
            return phi.detach(), phi_init, ew_mean

    return RgrlDreamSamplerNoEmbedTempScale


def build_sampler(model, tokenizer, args):
    """The sampler object whose .sample() runs the denoising loop. Plain DreamSampler by default;
    RgrlDreamSampler (entrgi/aps-guided) when --use_rgrl_guidance is set."""
    if not args.use_rgrl_guidance:
        return dream.DreamSampler(model=model, tokenizer=tokenizer)

    if args.guidance_kl_beta != 0.0:
        attach_noop_disable_adapter(model)

    reward_model, reward_tokenizer, token_mapping, mapped_embeds = load_guidance_model(
        model, tokenizer, args.guidance_reward_model, model.device,
        zero_unmatched=args.zero_unmatched_embeddings,
    )

    if args.guidance_no_temp_scale_embed:
        sampler_cls = _make_no_embed_temp_scale_sampler_cls()
    else:
        from dllm.pipelines.rl.rgrl.sampler import RgrlDreamSampler
        sampler_cls = RgrlDreamSampler

    return sampler_cls(
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        token_mapping=token_mapping,
        mapped_embeds=mapped_embeds,
    )


def build_generation_hooks(args):
    """(tokens_hook, logits_hook), each called once per denoising step as (step, x, logits).
    Identity by default -- this is where RGRL's optimize_logits would intercept `logits` before
    the sampler uses it to pick the next tokens to unmask."""
    tokens_hook = lambda step, x, logits: x
    logits_hook = lambda step, x, logits: logits
    return tokens_hook, logits_hook


def run_generation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = shard_output_path(args)
    if os.path.exists(out_path) and not args.force_overwrite:
        print(f"{out_path} already exists, skipping (use --force_overwrite to redo).")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    shard = load_shard(args)
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(shard)} problems, model={args.model}")

    model_args = ModelArguments(model_name_or_path=args.model, dtype="bfloat16")
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    sampler = build_sampler(model, tokenizer, args)
    tokens_hook, logits_hook = build_generation_hooks(args)

    common_config_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps or args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
    )
    if args.use_rgrl_guidance:
        from dllm.pipelines.rl.rgrl.sampler import RgrlDreamSamplerConfig
        sampler_config = RgrlDreamSamplerConfig(
            **common_config_kwargs,
            M=args.guidance_M,
            eta=args.guidance_eta,
            num_generations=1,  # one completion per (distinct) MATH-500 problem, no GRPO groups
            guidance_type=args.guidance_type,
            guidance_kl_beta=args.guidance_kl_beta,
            deprioritize_eos=args.guidance_deprioritize_eos,
        )
    else:
        sampler_config = dream.DreamSamplerConfig(**common_config_kwargs)

    results = []
    for start in tqdm(range(0, len(shard), args.batch_size), desc=f"shard {args.shard_index}"):
        batch = shard[start:start + args.batch_size]
        chats = [d["input_prompt"] for d in batch]
        input_ids = tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=True)

        outputs = sampler.sample(
            input_ids,
            sampler_config,
            return_dict=True,
            generation_tokens_hook_func=tokens_hook,
            generation_logits_hook_func=logits_hook,
        )
        texts = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), input_ids)

        for inst, text in zip(batch, texts):
            metrics, aux = MATH500.evaluate(inst, {"output": [text]}, aggregation="average")
            results.append({
                "inst_id": inst["inst_id"],
                "problem": inst["input_prompt"],  # chat list; kept so score_math500_skywork.py
                                                   # can rebuild the [user, assistant] turn.
                "ground_truth": inst["ground_truth"],
                "candidate": text,
                "score": metrics["score"],
            })

    accuracy = float(np.mean([r["score"] for r in results])) if results else 0.0
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "accuracy": accuracy, "results": results}, f, indent=2)
    print(f"Shard {args.shard_index}/{args.num_shards} accuracy: {accuracy:.4f} -> {out_path}")


def run_merge(args):
    all_results = []
    for shard_index in range(args.num_shards):
        shard_args = argparse.Namespace(**vars(args))
        shard_args.shard_index = shard_index
        with open(shard_output_path(shard_args)) as f:
            all_results.extend(json.load(f)["results"])

    accuracy = float(np.mean([r["score"] for r in all_results]))
    out_path = merged_output_path(args)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "accuracy": accuracy, "test_sample_size": len(all_results),
                    "results": all_results}, f, indent=2)
    print(f"Merged {args.num_shards} shards ({len(all_results)} problems) -> {out_path}")
    print(f"MATH-500 accuracy: {accuracy:.4f}")


def main():
    args = parse_args()
    if args.merge:
        run_merge(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
