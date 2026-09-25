"""
Ports GDPO's own log-likelihood estimator (gdpo/likelihood_estimators.py's
`NumericalIntegrationEstimator` -- Gauss-Legendre quadrature over the completion-masking rate,
default "gauss-3" in gdpo/slurm_scripts/gsm8k_vista.sbatch's real production runs -- plus its
`MCEstimator`) into dllm's own `DiffuGRPOTrainer`/`DreamGRPOTrainer`, reusing dllm's existing
right-shift-logits/position-ids/attention-mask machinery (`dllm/pipelines/rl/grpo/trainer.py`'s
`_get_per_token_logps`) instead of reimplementing it.

Ported verbatim from the GDPO repo's own `gdpo/dllm_reference/gdpo_estimator_extensions.py`
(same estimator + the GDPO-faithful `_compute_loss`, validated there against the true,
non-dllm `gdpo/gdpo_trainer.py` reference via a controlled synthetic-gradient toy test under real
DeepSpeed) -- this repo's own `DiffuGRPOTrainer`/`DreamGRPOTrainer`
(dllm/dllm/pipelines/rl/grpo/trainer.py) share the exact same method names/attributes/return-dict
shape as the dllm snapshot that file was built against (`_get_per_token_logps`,
`_get_per_token_logps_and_entropies`, `self._mask_seeds`/`_diffu_iter_idx`/`_current_mask_seed`,
`self.sampler_config`, `self.epsilon_low`/`epsilon_high`, `diffu_old_logps_all`/
`diffu_ref_logps_all` in the generation batch dict) -- no adaptation needed beyond this file's
own location.

Not ported: `D1Estimator`. Its scheme (mask completion fully, mask prompt at `p_mask_prompt`,
reweight by 1/p_mask) is already exactly what dllm's own default `_forward_process` +
`_get_per_token_logps` do -- no separate port needed, it's already the fallback ("full_mask") path.

GDPO's own quadrature/MC estimators never mask the prompt at all (only the completion) -- ported
faithfully as such, not matching dllm's own default `_forward_process`'s prompt-masking behavior.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed

from dllm.pipelines.rl import DiffuGRPOConfig, DiffuGRPOTrainer, DreamGRPOTrainer, RGRLConfig, RGRLTrainer

# Gauss-Legendre quadrature points/weights on [-1, 1], ported verbatim from
# gdpo/likelihood_estimators.py::NumericalIntegrationEstimator.
GAUSS_QUADRATURES = {
    "gauss-1": {"points": [0.0], "weights": [2.0]},
    "gauss-2": {"points": [-(1 / 3**0.5), (1 / 3**0.5)], "weights": [1.0, 1.0]},
    "gauss-3": {"points": [-((3 / 5) ** 0.5), 0.0, (3 / 5) ** 0.5], "weights": [5 / 9, 8 / 9, 5 / 9]},
    "gauss-4": {
        "points": [
            -((3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            +((3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            -((3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            +((3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
        ],
        "weights": [
            (18 + 30**0.5) / 36,
            (18 + 30**0.5) / 36,
            (18 - 30**0.5) / 36,
            (18 - 30**0.5) / 36,
        ],
    },
    "gauss-5": {
        "points": [
            0.0,
            -(1 / 3) * (5 - 2 * (10 / 7) ** 0.5) ** 0.5,
            +(1 / 3) * (5 - 2 * (10 / 7) ** 0.5) ** 0.5,
            -(1 / 3) * (5 + 2 * (10 / 7) ** 0.5) ** 0.5,
            +(1 / 3) * (5 + 2 * (10 / 7) ** 0.5) ** 0.5,
        ],
        "weights": [
            128 / 225,
            (322 + 13 * (70**0.5)) / 900,
            (322 + 13 * (70**0.5)) / 900,
            (322 - 13 * (70**0.5)) / 900,
            (322 - 13 * (70**0.5)) / 900,
        ],
    },
}


@dataclass
class GDPOEstimatorConfig(DiffuGRPOConfig):
    importance_sampling_level: str = field(
        default="sequence",
        metadata={
            "help": "Overrides TRL GRPOConfig's own default ('token'). REQUIRED to be 'sequence' "
            "for log_prob_mode in {'gauss-*', 'mc'}: those estimators' per-token values are only "
            "meaningful as a SEQUENCE-LEVEL SUM (each masked position's cross-entropy is "
            "reweighted by 1/rate to make the *sum* an unbiased full-sequence log-likelihood "
            "estimate -- matching gdpo/gdpo_trainer.py's own compute_loss, which computes "
            "`coef_1 = exp((logps - old_logps) / completion_length)` on the SUMMED sequence logp, "
            "never per-token). Under 'token'-level importance sampling, individual masked "
            "positions get amplified by up to ~9x (1/rate, gauss-3's smallest quadrature point), "
            "producing occasional huge per-token ratios -- confirmed empirically: loss ~4000/"
            "grad_norm ~14000 with 'token' (TRL's default) vs ~286/~186 with 'sequence', on an "
            "otherwise-identical first step."
        },
    )
    log_prob_mode: str = field(
        default="full_mask",
        metadata={
            "help": "'full_mask' (dllm's own default: mask completion fully, mask prompt at "
            "p_mask_prompt -- functionally identical to gdpo/likelihood_estimators.py's "
            "D1Estimator) or one of GAUSS_QUADRATURES's keys ('gauss-1'..'gauss-5', GDPO's own "
            "NumericalIntegrationEstimator) or 'mc' (GDPO's own MCEstimator)."
        },
    )
    mc_num_batches: int = field(
        default=128,
        metadata={
            "help": "Number of Monte Carlo draws to average for log_prob_mode='mc', matching "
            "gdpo/likelihood_estimators.py::MCEstimator's num_batches (its own docstring: 128 is "
            "adequate for most benchmarks; 1 suffices for single-token-answer benchmarks)."
        },
    )


class GDPOEstimatorTrainerMixin:
    """Adds GDPO's quadrature/MC log-likelihood estimator AND GDPO's own faithful loss formula
    to `DiffuGRPOTrainer`/`DreamGRPOTrainer`.

    Overrides `_get_per_token_logps` (the estimator itself) and `_compute_loss` (GDPO's own
    plain-mean, no-shared-normalizer, no-grad-accum-division formula, replacing TRL's PPO-clip
    loss_type switch entirely) -- everything else (generation via `MDLMSampler`/`DreamSampler`,
    block-wise chunking) is untouched, inherited as-is.

    Operates on the whole batch in one `model(...)` call per quadrature point/MC draw (no internal
    `batch_size` micro-batching loop, unlike the inherited `_get_per_token_logps`) -- matches
    gdpo/likelihood_estimators.py's own reference behavior; add chunking later if memory requires
    it, it's an orthogonal concern, not part of the estimator itself.
    """

    def _forward_process_at_rate(self, batch, prompt_index, mask_id, rate, seed=None):
        """Masks exactly round(rate * completion_length) completion positions (uniformly random,
        independently per row) -- ported from
        gdpo/likelihood_estimators.py::NumericalIntegrationEstimator.forward_process. The prompt
        is never masked (unlike dllm's own default `_forward_process`, which randomly masks it at
        `p_mask_prompt`) -- GDPO's own quadrature/MC estimators don't mask the prompt at all."""
        if seed is not None:
            set_seed(seed)
        b, length = batch.shape
        target_len = int((~prompt_index).sum().item())
        num_mask = int(round(rate * target_len))
        is_mask_completion = torch.zeros(b, target_len, dtype=torch.bool, device=batch.device)
        for i in range(b):
            perm = torch.randperm(target_len, device=batch.device)[:num_mask]
            is_mask_completion[i, perm] = True
        prompt_len = length - target_len
        is_mask = torch.cat(
            [torch.zeros(b, prompt_len, dtype=torch.bool, device=batch.device), is_mask_completion],
            dim=1,
        )
        noised = torch.where(is_mask, mask_id, batch)
        return noised, is_mask

    def _forward_process_mc(self, batch, prompt_index, mask_id, seed=None):
        """Ported from gdpo/likelihood_estimators.py::MCEstimator.forward_process: draws one
        random completion-masking count `k` for the whole call, then spreads a
        linspace-plus-modulo sequence of per-row counts around `k` (not every row masked at
        exactly the same rate, but the batch's mean rate is anchored at k/target_len) -- reduces
        variance relative to giving every row the identical rate, per the original reference's own
        design. Returns each row's own realized rate alongside the mask, for 1/rate reweighting."""
        if seed is not None:
            set_seed(seed)
        b, length = batch.shape
        target_len = int((~prompt_index).sum().item())
        k = torch.randint(1, target_len + 1, (1,), device=batch.device).item()
        x = torch.round(
            torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)
        ).long()
        x = ((x - 1) % target_len) + 1
        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask_completion = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask_completion[i] = is_mask_completion[i][torch.randperm(target_len, device=batch.device)]
        prompt_len = length - target_len
        is_mask = torch.cat(
            [torch.zeros(b, prompt_len, dtype=torch.bool, device=batch.device), is_mask_completion],
            dim=1,
        )
        noised = torch.where(is_mask, mask_id, batch)
        rate = x.to(torch.float32) / target_len  # [b], each row's own realized rate
        return noised, is_mask, rate

    def _score_noised(self, model, noised_input_ids, input_ids, attention_mask, logits_to_keep):
        """Forward pass + right-shift + cross-entropy, factored out of dllm's own
        `_get_per_token_logps` so both the rate-based (quadrature) and MC front-ends share it.
        Returns raw per-token cross-entropy loss (positive, NOT yet reweighted/masked-out at
        unmasked positions), shape [B, logits_to_keep]."""
        if noised_input_ids.size(0) == 0:
            return torch.zeros((0, logits_to_keep), device=noised_input_ids.device, dtype=torch.float32)
        use_pos_ids = getattr(self.sampler_config, "use_position_ids", True)
        if use_pos_ids:
            pos_id = attention_mask.long().cumsum(-1) - 1
            pos_id = pos_id.masked_fill(attention_mask == 0, 1)
        else:
            pos_id = None

        # Attention-mask SHAPE/dtype convention is family-specific, not just a config value --
        # Dream's own model always expects a 4D boolean mask [B,1,1,T]; LLaDA's own model does
        # NOT accept that shape at all -- grpo/trainer.py's own `_get_per_token_logps` passes a
        # plain 2D mask [B,T] for LLaDA. `right_shift_logits` is already the flag distinguishing
        # these two families everywhere else in this file/module, so key off it here too rather
        # than adding a third, redundant config flag.
        if getattr(self.sampler_config, "right_shift_logits", True):
            fwd_kwargs = {"attention_mask": attention_mask[:, None, None, :].bool()}
        else:
            fwd_kwargs = {"attention_mask": attention_mask}
        if pos_id is not None:
            fwd_kwargs["position_ids"] = pos_id
        logits = model(noised_input_ids, **fwd_kwargs).logits

        if getattr(self.sampler_config, "right_shift_logits", True):
            # Dream's AR convention: raw logits[:, i] predicts token i+1. After shift, logits[:, i]
            # predicts token i, matching completion_targets. LLaDA's MDLMSamplerConfig sets this
            # False explicitly, so this branch is a no-op there.
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        completion_logits = logits[:, -logits_to_keep:, :]
        completion_targets = input_ids[:, -logits_to_keep:]
        per_token_loss = F.cross_entropy(
            completion_logits.reshape(-1, completion_logits.size(-1)),
            completion_targets.reshape(-1),
            reduction="none",
        ).view(input_ids.size(0), logits_to_keep)
        return per_token_loss

    def _get_per_token_logps_at_rate(self, model, input_ids, attention_mask, logits_to_keep, rate, seed=None):
        mask_id = self.processing_class.mask_token_id
        seq_len = input_ids.size(1)
        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.arange(seq_len, device=input_ids.device) < prompt_length

        noised_input_ids, is_mask = self._forward_process_at_rate(input_ids, prompt_index, mask_id, rate, seed=seed)
        per_token_loss = self._score_noised(model, noised_input_ids, input_ids, attention_mask, logits_to_keep)
        completion_is_mask = is_mask[:, -logits_to_keep:]

        # Unbias via 1/rate reweighting at masked positions only (unmasked completion positions
        # under this quadrature point contribute exactly 0 -- matches
        # gdpo/likelihood_estimators.py::NumericalIntegrationEstimator.get_loss's
        # `loss[not_mask_index] = 0`). Sign flipped to match dllm's own convention (`-loss` = logp).
        return torch.where(
            completion_is_mask,
            -per_token_loss / max(rate, 1e-6),
            torch.zeros_like(per_token_loss),
        )

    def _get_per_token_logps_gauss(self, model, input_ids, attention_mask, logits_to_keep, mode, seed=None):
        quad = GAUSS_QUADRATURES[mode]
        total = None
        for point, weight in zip(quad["points"], quad["weights"]):
            rate = point * 0.5 + 0.5  # change of variable, [-1,1] -> [0,1]
            logp_i = self._get_per_token_logps_at_rate(model, input_ids, attention_mask, logits_to_keep, rate, seed=seed)
            total = weight * logp_i if total is None else total + weight * logp_i
        return 0.5 * total  # Jacobian of the [-1,1] -> [0,1] change of variable

    def _get_per_token_logps_mc_single(self, model, input_ids, attention_mask, logits_to_keep, seed=None):
        mask_id = self.processing_class.mask_token_id
        seq_len = input_ids.size(1)
        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.arange(seq_len, device=input_ids.device) < prompt_length

        noised_input_ids, is_mask, rate = self._forward_process_mc(input_ids, prompt_index, mask_id, seed=seed)
        per_token_loss = self._score_noised(model, noised_input_ids, input_ids, attention_mask, logits_to_keep)
        completion_is_mask = is_mask[:, -logits_to_keep:]
        rate = rate.unsqueeze(1).clamp(min=1e-6)  # [B, 1], each row's own realized rate

        return torch.where(completion_is_mask, -per_token_loss / rate, torch.zeros_like(per_token_loss))

    def _get_per_token_logps_mc(self, model, input_ids, attention_mask, logits_to_keep, num_batches, seed=None):
        total = None
        for i in range(num_batches):
            this_seed = None if seed is None else seed + i
            logp_i = self._get_per_token_logps_mc_single(model, input_ids, attention_mask, logits_to_keep, seed=this_seed)
            total = logp_i if total is None else total + logp_i
        return total / num_batches

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, seed=None):
        """
        Overrides the LOWER-level `_get_per_token_logps` (not `_get_per_token_logps_and_entropies`)
        deliberately: `_generate_and_score_completions` precomputes `old_per_token_logps`
        (and `ref_per_token_logps` when `beta!=0`) by calling `_get_per_token_logps` DIRECTLY
        (bypassing `_get_per_token_logps_and_entropies` entirely) -- overriding only the latter
        would leave that precompute pass silently using dllm's default full-mask scheme while
        `_compute_loss`'s later call used this class's quadrature/MC scheme, comparing
        `per_token_logps - old_per_token_logps` across two completely different masking schemes
        and producing an enormous, meaningless ratio. Overriding this one method fixes both call
        sites at once, since `_get_per_token_logps_and_entropies`'s own (inherited, unmodified)
        implementation already just calls `self._get_per_token_logps(...)`.
        """
        mode = getattr(self.args, "log_prob_mode", "full_mask")
        if mode in (None, "full_mask"):
            return super()._get_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size, seed=seed
            )

        if seed is None:
            seed = getattr(self, "_current_mask_seed", None)

        if mode == "mc":
            result = self._get_per_token_logps_mc(
                model, input_ids, attention_mask, logits_to_keep,
                num_batches=getattr(self.args, "mc_num_batches", 128), seed=seed,
            )
        elif mode in GAUSS_QUADRATURES:
            result = self._get_per_token_logps_gauss(model, input_ids, attention_mask, logits_to_keep, mode, seed=seed)
        else:
            raise ValueError(f"Unknown log_prob_mode {mode!r} for GDPOEstimatorTrainerMixin")

        return result

    def _compute_loss(self, model, inputs):
        """
        Replaces TRL's own PPO-clip loss formula (`GRPOTrainer._compute_loss`'s `loss_type`
        switch -- grpo/bnpo/dr_grpo/dapo) with GDPO's OWN actual algorithm, ported line-for-line
        from gdpo/gdpo_trainer.py::GDPOTrainer.compute_loss. This mixin sits before
        DiffuGRPOTrainer in the MRO, so overriding `_compute_loss` here fully replaces (rather
        than delegates to) both DiffuGRPOTrainer's own `_compute_loss` (mask-seed injection) and
        TRL's `GRPOTrainer._compute_loss` beneath it -- the mask-seed injection logic is
        reimplemented inline below (copied from DiffuGRPOTrainer._compute_loss) since we can no
        longer reach it via `super()`.

        Real, faithfully-ported differences from TRL's own formula, not oversights:
          - `logps`/`old_logps`/`ref_logps` are PER-ROW scalars (whole-completion
            log-likelihoods), not per-token tensors -- `_get_per_token_logps_and_entropies`
            already returns this mixin's own quadrature/MC per-token estimate; reduced via
            `(per_token_logps * completion_mask).sum(-1)` to match what gdpo_trainer.py's
            `self.logp_estimator.get_log_likelihood` returns directly.
          - `coef_1 = exp((logps - old_logps) / completion_ids.shape[-1])`: divides by the
            PADDED batch-wide completion length, not each row's own true length -- ported exactly
            as-is even though this only equals a true per-row-length normalization when every row
            in the batch happens to share the same true length; faithfulness to the reference
            takes priority over "fixing" this.
          - Plain `.mean()` reduction: no `loss_type`/dapo-style shared batch-wide normalizer, and
            deliberately NO `/ self.current_gradient_accumulation_steps` division anywhere.
            Confirmed via a controlled toy-model test under real DeepSpeed (identical synthetic
            gradient every micro-batch, so the ratio is exact and noise-free) that
            gdpo_trainer.py's own real production runs get NEITHER HF's automatic python-level
            division (skipped: TRL's own GRPOTrainer.__init__ sets `self.compute_loss_func` to a
            non-None sentinel specifically to disable it, inherited unmodified since GDPOTrainer
            subclasses GRPOTrainer directly) NOR DeepSpeed's own automatic accumulation-step
            scaling (unconditionally disabled by HF's own `training_step` whenever
            `accelerator.distributed_type == DEEPSPEED`, regardless of loss_type) -- so an
            un-divided `.mean()` here is what actually reproduces the reference's real gradient
            scale, not a bug to correct.
          - Metric key names ("kl", "clip_ratio") and mode detection
            (`self.control.should_evaluate`) match gdpo_trainer.py's own convention exactly,
            not TRL's richer low/high-clip breakdown -- that breakdown is TRL's own value-add on
            top of the base algorithm, not part of GDPO's actual algorithm.

        One deliberate DEVIATION from the reference, not a faithful port: the `kl` term below
        normalizes `(ref_logps - logps)` by `completion_ids.shape[-1]` before `exp()`, matching
        `coef_1`'s own normalization. `gdpo_trainer.py::GDPOTrainer.compute_loss` does NOT do
        this (exponentiates the raw summed-sequence gap) -- harmless there only because every
        real GDPO run uses `beta=0.0` (see `gdpo/slurm_scripts/train.yaml` in the original repo),
        making the KL branch dead code. The same un-normalized formula was already found to
        blow up (KL into the millions) when `beta != 0` was actually exercised for RGRL in an
        earlier port of this codebase, and RGRLTrainer._compute_loss in this repo normalizes for
        exactly this reason -- mirrored here rather than reproducing the same latent bug.
        """
        inputs = self._inject_iteration_logps(inputs)
        return self._compute_loss_body(model, inputs)

    def _inject_iteration_logps(self, inputs):
        """Selects this iteration's precomputed old/ref logps slice (from the stacked
        diffu_old_logps_all/diffu_ref_logps_all) and advances _diffu_iter_idx/_current_mask_seed
        by exactly one. Factored out of _compute_loss so RGRLTrainerWithGDPOEstimator's own
        sft-mix _compute_loss (see below) can call this exactly once per step regardless of
        which loss body it then runs, without double-injecting."""
        if self._mask_seeds and (
            inputs.get("diffu_old_logps_all") is not None or inputs.get("diffu_ref_logps_all") is not None
        ):
            idx = self._diffu_iter_idx % self.num_iterations
            inputs = dict(inputs)  # shallow copy -- don't mutate TRL's buffered dict
            if inputs.get("diffu_old_logps_all") is not None:
                inputs["old_per_token_logps"] = inputs["diffu_old_logps_all"][:, idx, :]
            if inputs.get("diffu_ref_logps_all") is not None:
                inputs["ref_per_token_logps"] = inputs["diffu_ref_logps_all"][:, idx, :]
            self._current_mask_seed = self._mask_seeds[idx]
            self._diffu_iter_idx += 1
        return inputs

    def _compute_loss_body(self, model, inputs):
        """The actual GDPO ratio/clip/KL formula, assuming _inject_iteration_logps has already
        run on `inputs` (old_per_token_logps/ref_per_token_logps already selected for this
        iteration if applicable)."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )
        logps = (per_token_logps * completion_mask).sum(-1)

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_logps = (
            (old_per_token_logps * completion_mask).sum(-1)
            if old_per_token_logps is not None
            else logps.detach()
        )

        coef_1 = torch.exp((logps - old_logps) / completion_ids.shape[-1])
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        loss_1 = coef_1 * advantages
        loss_2 = coef_2 * advantages
        loss = -torch.min(loss_1, loss_2)

        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            ref_logps = (ref_per_token_logps * completion_mask).sum(-1)
            # Normalize by the same (padded batch-wide) completion length coef_1 uses before
            # exponentiating -- the raw summed-sequence gap blows up under exp() for anything but
            # very short completions. GDPO's own reference `gdpo_trainer.py` has this exact bug
            # (unnormalized), but it's dead code there since its real runs always use beta=0.0;
            # RGRLTrainer._compute_loss in this repo already normalizes (per-token, before
            # summing) for the same reason -- mirror that here so beta!=0 is actually usable.
            kl = torch.exp((ref_logps - logps) / completion_ids.shape[-1]) - (ref_logps - logps) / completion_ids.shape[-1] - 1
            loss = loss + self.beta * kl
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(kl).mean().item())

        is_clipped = (loss_1 < loss_2).float()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(is_clipped).mean().item()
        )

        return loss.mean()


class DiffuGRPOTrainerWithGDPOEstimator(GDPOEstimatorTrainerMixin, DiffuGRPOTrainer):
    """DiffuGRPOTrainer (LLaDA/MDLM) + GDPO's quadrature/MC likelihood estimator."""


class DreamGRPOTrainerWithGDPOEstimator(GDPOEstimatorTrainerMixin, DreamGRPOTrainer):
    """DreamGRPOTrainer (Dream) + GDPO's quadrature/MC likelihood estimator."""


@dataclass
class RGRLGDPOConfig(GDPOEstimatorConfig, RGRLConfig):
    """Union of RGRLConfig's own fields (M/eta/guidance_*/block_size/awr_exp_weighting/awr_beta/
    sft_mix_prob) and GDPOEstimatorConfig's (log_prob_mode/mc_num_batches) -- lets a single
    train.py/CLI surface expose both, so `loss_backend` (defined in each RGRL train.py, not here)
    can switch trainer classes without switching the argument schema underneath it."""


class RGRLTrainerWithGDPOEstimator(GDPOEstimatorTrainerMixin, RGRLTrainer):
    """RGRLTrainer (guided generation + self-distillation) with its log-prob estimator replaced
    by GDPO's own faithful quadrature/MC estimator, and its loss formula either GDPO's own
    sequence-level ratio/KL loss OR plain unweighted SFT, selected by which GENERATION MODE
    RGRLConfig.sft_mix_prob chose for the current round (see RGRLTrainer._generate_and_score_
    completions): a round generated with M=0 (plain/unguided rollouts, same as GDPO/GRPO's own
    normal sampling) uses GDPO's loss; a round generated WITH guidance uses plain SFT on those
    guided completions instead. The two are never mixed within a round -- generation mode and
    loss formula are chosen together, once per round, not independently.

    This override exists specifically for that arm selection (GDPOEstimatorTrainerMixin's own
    `_compute_loss` alone would always pick its own loss, regardless of generation mode) --
    everything else (the estimator swap, generation itself) is the same mixin pattern as
    DreamGRPOTrainerWithGDPOEstimator/DiffuGRPOTrainerWithGDPOEstimator above.
    """

    def _compute_loss(self, model, inputs):
        inputs = self._inject_iteration_logps(inputs)
        if getattr(self, "_current_round_is_sft_arm", False):
            return self._compute_plain_sft_loss(model, inputs)
        return self._compute_loss_body(model, inputs)

    def _compute_plain_sft_loss(self, model, inputs):
        """Unweighted imitation of every guided rollout: weight=1 for all samples, no advantage
        reweighting, no ratio/clip, no KL-to-ref term. Uses the same estimator
        (_get_per_token_logps_and_entropies -> _get_per_token_logps, log_prob_mode-selected) as
        the GDPO-loss arm, so both arms score logps identically -- only the weighting differs.
        This is the original pre-AWR-refactor RGRL loss, reintroduced ONLY as the loss for
        sft_mix_prob's guided-generation rounds -- see that field's own docstring on RGRLConfig
        for why (plain SFT alone previously caused refusal collapse; this mixture is an
        experiment in whether occasional exposure, diluted by mostly-normal-mode rounds, is
        safe)."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )
        completion_lengths = completion_mask.sum(-1)
        # A length-1 completion is just the EOS token with nothing generated before it (completion_mask
        # is "1 up to and including the first EOS, 0 after" -- grpo/trainer.py's own construction).
        # Confirmed empirically (wandb run 4txv9848): completions/length/min==1.0 recurs alongside
        # is_sft_arm==1.0, with the batch's whole length distribution collapsing narrower as training
        # progresses. Excluding these rows entirely (loss forced to exactly 0, no gradient).
        is_real_completion = (completion_lengths > 1).float()
        # Normalize by completion_ids.shape[-1] -- the PADDED, batch-wide completion width, same
        # constant for every row -- not by each row's own true length (completion_lengths). This
        # matches _compute_loss_body's own coef_1 normalization exactly (see that method's
        # docstring) and is what keeps a short real completion from getting an undiluted,
        # disproportionately large per-token weight relative to full-length rows: the numerator
        # sum is still masked to zero out padding beyond each row's own true completion
        # (`per_token_logps * completion_mask`), only the denominator is now the shared constant.
        per_row_loss = -(
            (per_token_logps * completion_mask).sum(-1) / completion_ids.shape[-1]
        ) * is_real_completion
        num_real = is_real_completion.sum().clamp(min=1.0)
        return per_row_loss.sum() / num_real


class RGRLTrainerWithGDPOEstimatorOnly(GDPOEstimatorTrainerMixin, RGRLTrainer):
    """RGRLTrainer with ONLY its log-prob estimator replaced by GDPO's own faithful quadrature/MC
    estimator (log_prob_mode='gauss-3' etc.) -- RGRL's own native loss formula
    (RGRLTrainer._compute_loss: plain SFT, -per_seq_logps.sum()/num_generations, no ratio/clip)
    is kept as-is, unlike RGRLTrainerWithGDPOEstimator above which also replaces the loss with
    GDPO's own ratio/clip/KL formula.

    The single-random-masking-draw estimator RGRLTrainer inherits by default (DreamGRPOTrainer's
    own `_get_per_token_logps` -> `_forward_process`, one Bernoulli mask draw per call) is a
    noisier, higher-variance estimate of the true sequence log-likelihood than GDPO's own
    Gauss-Legendre quadrature over the masking rate (log_prob_mode='gauss-N') or Monte Carlo
    average ('mc') -- this class lets RGRL use the lower-variance estimator without adopting
    GDPO's loss formula too.

    MRO note: `GDPOEstimatorTrainerMixin` is listed first, so its own `_get_per_token_logps`
    override wins over `DreamGRPOTrainer`'s (inherited via RGRLTrainer) -- but it ALSO defines
    its own `_compute_loss` (the full ratio/clip/KL formula), which would otherwise win by the
    same MRO rule. The override below exists solely to route back to RGRLTrainer's own
    `_compute_loss` instead, which in turn calls `self._get_per_token_logps(...)` polymorphically
    -- so the estimator swap still takes effect there, just without GDPO's loss riding along.
    """

    def _compute_loss(self, model, inputs):
        return RGRLTrainer._compute_loss(self, model, inputs)


class RGRLTrainerWithPSFTLoss(GDPOEstimatorTrainerMixin, RGRLTrainer):
    """RGRLTrainer with its loss replaced by Proximal SFT (PSFT, Zhu et al. 2025,
    arxiv.org/abs/2508.17784): the PPO-clipped surrogate `min(r, clip(r,1-eps,1+eps))`, with
    the "advantage" fixed to the constant 1 (the paper's own framing: SFT is policy gradient with
    a fixed positive advantage; substituting into PPO's clip gives PSFT, not a new formula).

    Implemented as `_compute_loss_body` (GDPOEstimatorTrainerMixin's own faithful ratio/clip/KL
    port) called with `advantages` forced to a ones tensor -- every other mechanical detail
    (sequence-level ratio normalized by completion_ids.shape[-1], the log_prob_mode-selected
    estimator underneath, epsilon_low/epsilon_high clip bounds) is IDENTICAL to GDPO's own loss,
    deliberately: the only thing PSFT changes relative to GDPO is what multiplies the ratio.

    Caveat, not faithful to the paper: `_compute_loss_body`'s optional `self.beta`-weighted
    KL-to-ref term is inherited unchanged. The paper's own PSFT has NO separate KL term at all --
    the clip alone is the entire trust-region mechanism. Set `--beta 0` when using this backend
    to match the paper; a nonzero beta here adds a real extra regularizer PSFT itself doesn't have.

    Needs `old_per_token_logps` from `inputs` (via `diffu_old_logps_all`, precomputed in
    RGRLTrainer._generate_and_score_completions the same way DiffuGRPOTrainer's own version does)
    to form a real ratio. Without it (e.g. num_iterations=1 with no other reuse, so it's never
    precomputed), `_compute_loss_body` falls back to `old_logps=logps.detach()` -- mathematically
    identical to a same-instant snapshot anyway (no gradient step separates them), which makes
    `coef_1==1` exactly and the clip a no-op, so the gradient reduces exactly to plain SFT's for
    that step. This is a real, understood degeneracy (see PSFT's own gradient analysis, Eq. 7:
    at r_t=1, gradient = 1 * I_trust(1) * grad_logp = grad_logp, identical to plain SFT), not a
    bug -- PSFT only differs from plain SFT once num_iterations>1 lets the ratio drift from 1
    across reused-batch gradient steps.
    """

    def _compute_loss(self, model, inputs):
        inputs = self._inject_iteration_logps(inputs)
        return self._compute_psft_loss(model, inputs)

    def _compute_psft_loss(self, model, inputs):
        inputs = dict(inputs)
        inputs["advantages"] = torch.ones(
            inputs["completion_ids"].size(0), device=inputs["completion_ids"].device
        )
        return self._compute_loss_body(model, inputs)
