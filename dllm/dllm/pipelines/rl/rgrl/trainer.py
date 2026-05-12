from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from accelerate.utils import gather_object, set_seed
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer, nanstd, split_tensor_dict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from dllm.pipelines.rl.grpo.trainer import DiffuGRPOConfig, DreamGRPOTrainer
from dllm.pipelines.rl.rgrl.sampler import RgrlDreamSampler, RgrlDreamSamplerConfig

if is_peft_available():
    from peft import PeftConfig

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


@dataclass
class RGRLConfig(DiffuGRPOConfig):
    # Must be False for LoRA + grad-checkpoint + DDP.
    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": "Must be False for LoRA + grad-checkpoint + DDP."})

    M: int = field(default=1, metadata={"help": "Adam gradient steps per guidance call."})
    eta: float = field(default=1.0, metadata={"help": "Adam lr for phi."})
    guidance_reward_model: str = field(
        default="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        metadata={"help": "Reward model for gradient guidance during generation."},
    )
    guidance_type: str = field(
        default="entrgi",
        metadata={"help": "'entrgi' (entropy-aware interpolation) or 'aps' (full STE, w=1)."},
    )
    zero_unmatched_embeddings: bool = field(
        default=False,
        metadata={"help": "Zero embedding for policy tokens absent from reward model vocab. Recommended for LLaDA."},
    )


class RGRLTrainer(DreamGRPOTrainer):
    """DreamGRPOTrainer with RG-RL-guided generation."""

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[RGRLConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        sampler_config: Optional[RgrlDreamSamplerConfig] = None,
    ):
        guidance_type = args.guidance_type if args else "entrgi"
        assert guidance_type in {"entrgi", "aps"}, \
            f"guidance_type must be 'entrgi' or 'aps', got {guidance_type!r}."

        if sampler_config is None:
            sampler_config = RgrlDreamSamplerConfig(
                steps=args.steps if args else 64,
                max_new_tokens=args.max_completion_length if args else 256,
                temperature=args.temperature or 1.0 if args else 1.0,
                cfg_scale=args.cfg_scale if args else 0.0,
                alg=getattr(args, "dream_alg", "entropy"),
                top_p=getattr(args, "dream_top_p", 0.95),
                top_k=getattr(args, "dream_top_k", 50),
                right_shift_logits=True,
                M=args.M if args else 1,
                eta=args.eta if args else 1.0,
                num_generations=args.num_generations if args else 4,
                guidance_type=guidance_type,
            )

        from dllm.pipelines.dream.sampler import DreamSampler
        placeholder_sampler = DreamSampler(model=model, tokenizer=processing_class)

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            sampler=placeholder_sampler,
            sampler_config=sampler_config,
        )

        guidance_model_name = args.guidance_reward_model if args else "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
        device = self.accelerator.device
        zero_unmatched = args.zero_unmatched_embeddings if args else False
        self._load_guidance_model(guidance_model_name, device, zero_unmatched=zero_unmatched)

        self.sampler = RgrlDreamSampler(
            model=self.model,
            tokenizer=self.processing_class,
            reward_model=self._guidance_reward_model,
            reward_tokenizer=self._guidance_reward_tokenizer,
            token_mapping=self._guidance_token_mapping,
            mapped_embeds=self._guidance_mapped_embeds,
        )

    def _load_guidance_model(
        self, model_name: str, device: torch.device, zero_unmatched: bool = False
    ):
        reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, num_labels=1
        ).to(device)
        reward_model.eval()
        for p in reward_model.parameters():
            p.requires_grad = False

        policy_vocab = self.processing_class.get_vocab()
        reward_vocab = reward_tokenizer.get_vocab()
        try:
            vocab_size = self.model.lm_head.out_features
        except AttributeError:
            vocab_size = self.model.config.vocab_size
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

        self._guidance_reward_model = reward_model
        self._guidance_reward_tokenizer = reward_tokenizer
        self._guidance_token_mapping = token_mapping
        self._guidance_mapped_embeds = mapped_embeds

    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        generation_batch_size = self.args.generation_batch_size or prompt_ids.size(0)

        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                self.sampler.model = unwrapped_model
                prompt_completion_ids_all = []
                ew_means = []
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    batch = list(prompt_ids[i : i + generation_batch_size])
                    out = self.sampler.sample(batch, self.sampler_config)
                    prompt_completion_ids_all.append(out)
                    if self.sampler._last_entropy_weight_mean is not None:
                        ew_means.append(self.sampler._last_entropy_weight_mean)
                    torch.cuda.empty_cache()

        prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
        if ew_means:
            self._metrics[mode]["guidance/entropy_weight"].append(
                sum(ew_means) / len(ew_means)
            )

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
        completion_lengths = completion_mask.sum(1)

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Mask seeds ensure the same forward-process masks are used across
        # gradient accumulation steps within one generation round.
        self._mask_seeds = [
            torch.randint(0, 2**31 - 1, (1,)).item() for _ in range(self.num_iterations)
        ]
        self._diffu_iter_idx = 0
        self._current_mask_seed = self._mask_seeds[0]

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        rewards_grouped = rewards.view(-1, self.num_generations)  # [B, K]

        mean_grouped_rewards = rewards_grouped.mean(dim=1)
        std_grouped_rewards = rewards_grouped.std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        acc = self.accelerator
        m = self._metrics[mode]

        def log_stats(prefix, t):
            t = t.float()
            m[f"{prefix}/mean"].append(t.mean().item())
            m[f"{prefix}/min"].append(t.min().item())
            m[f"{prefix}/max"].append(t.max().item())

        if mode == "train":
            self.state.num_input_tokens_seen += (
                acc.gather(attention_mask.sum()).sum().item()
            )
        m["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = acc.gather(completion_lengths)
        log_stats("completions/length", agg_completion_lengths)

        agg_terminated_with_eos = acc.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        m["completions/clipped_ratio"].append(
            1 - len(term_completion_lengths) / len(agg_completion_lengths)
        )
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        log_stats("completions/terminated_length", term_completion_lengths)

        for i, name in enumerate(self.reward_func_names):
            m[f"rewards/{name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
            m[f"rewards/{name}/std"].append(nanstd(rewards_per_func[:, i]).item())
        m["reward"].append(mean_grouped_rewards.mean().item())
        m["reward_std"].append(std_grouped_rewards.mean().item())
        m["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        logits_to_keep = completion_ids.size(1)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": completion_mask.sum(),
            "logits_to_keep": torch.tensor(logits_to_keep, device=device),
        }

    def _compute_loss(self, model, inputs):
        if self._mask_seeds:
            idx = self._diffu_iter_idx % max(self.num_iterations, 1)
            inputs = dict(inputs)  # shallow copy
            self._current_mask_seed = self._mask_seeds[idx]
            self._diffu_iter_idx += 1

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        lengths = completion_mask.sum(1).clamp(min=1).float()

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        per_seq_logps = (per_token_logps * completion_mask).sum(1) / lengths

        loss = -per_seq_logps.sum() / self.num_generations

        mode = "train" if model.training else "eval"
        self._metrics[mode]["loss"].append(loss.item())

        return loss

    @profiling_decorator
    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs
