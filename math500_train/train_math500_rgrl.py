"""
RGRL (entrgi-guided online SFT) training for Dream-v0-Instruct-7B on MATH-500 *prompts only*,
rewarded with a Skywork reward model instead of ground-truth correctness.

Same dataset/reward setup as train_math500_grpo.py in this folder -- see that script's own
docstring for the full rationale (ground truth never enters the loss; a second, logging-only
math500_correctness_reward_func with reward_weight=0.0 tracks real accuracy every step purely as
a sanity check). The only thing that differs here is the TRAINER: RGRLTrainer (or one of its GDPO-
estimator/PSFT-loss variants) instead of plain DreamGRPOTrainer -- RGRL's own entrgi guidance
(_optimize_logits, see dllm.pipelines.rl.rgrl.sampler.RgrlDreamSampler) steers denoising toward
higher predicted Skywork reward at every step, rather than letting GRPO's policy-gradient updates
be the only lever.

Structurally a close copy of dllm/examples/rl/rgrl/dream/train.py (this repo's own production RGRL
script), with --dataset removed (this script only ever does the MATH-500-prompts setup) and the
dataset/reward wiring replaced.

Usage: see math500_train/dream_rgrl_math500.sbatch for the full production invocation.
"""

import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

from datasets import Dataset
from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.pipelines.rl import RGRLTrainer
from dllm.pipelines.rl.gdpo import RGRLGDPOConfig, RGRLTrainerWithGDPOEstimatorOnly, RGRLTrainerWithPSFTLoss
from dllm.pipelines.rl.grpo.rewards.skywork import make_skywork_reward_func

logger = dllm.utils.get_default_logger(__name__)

ORIG_CWD = os.getcwd()
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def get_math500_questions_only() -> Dataset:
    """Same as train_math500_grpo.py's own -- see that docstring. Ground truth is carried along
    ONLY for the logging-only math500_correctness_reward_func below."""
    sys.path.insert(0, str(EVAL_DIR))
    cwd_before = os.getcwd()
    os.chdir(EVAL_DIR)
    try:
        from reason_benchmarks.math_500 import MATH500
        problems = MATH500.load()
    finally:
        os.chdir(cwd_before)

    rows = [
        {"prompt": d["input_prompt"], "ground_truth": d["ground_truth"]}
        for d in problems
    ]
    return Dataset.from_list(rows)


def math500_correctness_reward_func(
    prompts, completions, ground_truth, verbose=False, **kwargs
) -> list[float]:
    """Logging-only -- see train_math500_grpo.py's own copy of this function for the full
    docstring. Registered with reward_weight=0.0 in train() below."""
    sys.path.insert(0, str(EVAL_DIR))
    from reason_benchmarks.math_500 import compute_score

    responses = [completion[0]["content"] for completion in completions]
    scores = [100.0 * compute_score(r, gt) for r, gt in zip(responses, ground_truth)]
    if verbose:
        print(f"[math500_correctness] mean={sum(scores) / len(scores):.1f} "
              f"n={len(scores)} sample_gt={ground_truth[0]!r} sample_response={responses[0][:200]!r}")
    return scores


@dataclass
class TrainingArguments(RGRLGDPOConfig):
    output_dir: str = ".models/Dream-v0-Instruct-7B/rgrl-math500"
    loss_backend: str = field(
        default="grpo",
        metadata={
            "help": "'grpo' (RGRLTrainer's native plain-SFT loss -- this repo's own production "
            "RGRL default, see vista_sbatch/dream_rgrl_entrgi.sbatch), 'gdpo_estimator', or 'psft'."
        },
    )
    log_prob_mode: str = field(
        default="gauss-3",
        metadata={"help": "Only used when loss_backend='gdpo_estimator'."},
    )
    verbose_reward: bool = field(default=False, metadata={"help": "Enable verbose reward printing."})
    dream_alg: str = field(default="entropy", metadata={"help": "entropy, maskgit_plus, topk_margin."})
    dream_top_p: float = field(default=0.95, metadata={"help": "top-p for Dream sampling."})
    dream_top_k: int = field(default=50, metadata={"help": "top-k for Dream sampling."})
    deprioritize_eos: bool = field(
        default=False,
        metadata={"help": "Set confidence=-inf at EOS-sampled positions during denoising. "
                  "False here matches this repo's own production RGRL example's default."},
    )
    reward_model: str = field(
        default="Skywork/Skywork-Reward-V2-Qwen3-8B",
        metadata={"help": "Reward model scoring completions (the actual optimized reward)."},
    )


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "Dream-org/Dream-v0-Instruct-7B"

    # ---- Dataset & reward --------------------------------------------------------
    dataset = get_math500_questions_only()
    reward_functions = [
        make_skywork_reward_func(training_args.reward_model),
        math500_correctness_reward_func,
    ]
    training_args.reward_weights = [1.0, 0.0]
    if training_args.verbose_reward:
        reward_functions = [partial(fn, verbose=True) for fn in reward_functions]

    train_set = dataset.shuffle(seed=training_args.seed)

    # ---- Model & Tokenizer --------------------------------------------------------
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=model_config.model_name_or_path,
        load_in_4bit=(
            model_config.load_in_4bit if hasattr(model_config, "load_in_4bit") else False
        ),
    )
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model.config.use_cache = False

    # ---- LoRA -----------------------------------------------------------------------
    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj",
            ],
            lora_dropout=model_config.lora_dropout,
        )

    # ---- RGRL sampler config (RGRLTrainer builds its own RgrlDreamSampler internally,
    # loading the guidance reward model itself from guidance_reward_model) -----------------
    from dllm.pipelines.rl.rgrl.sampler import RgrlDreamSamplerConfig

    sampler_config = RgrlDreamSamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        temperature=training_args.temperature or 1.0,
        cfg_scale=training_args.cfg_scale,
        alg=training_args.dream_alg,
        top_p=training_args.dream_top_p,
        top_k=training_args.dream_top_k,
        right_shift_logits=True,
        M=training_args.M,
        eta=training_args.eta,
        num_generations=training_args.num_generations,
        guidance_type=training_args.guidance_type,
        guidance_kl_beta=training_args.guidance_kl_beta,
        deprioritize_eos=training_args.deprioritize_eos,
    )

    trainer_cls = {
        "gdpo_estimator": RGRLTrainerWithGDPOEstimatorOnly,
        "psft": RGRLTrainerWithPSFTLoss,
    }.get(training_args.loss_backend, RGRLTrainer)

    logger.info(f"Start RGRL training (Dream, MATH-500 prompts, Skywork reward+guidance), "
                f"loss_backend={training_args.loss_backend}...")
    trainer = trainer_cls(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_set,
        processing_class=tokenizer,
        peft_config=peft_config,
        sampler_config=sampler_config,
    )

    if training_args.save_steps % training_args.num_iterations != 0:
        import warnings
        warnings.warn(
            f"save_steps ({training_args.save_steps}) is not divisible by "
            f"num_iterations ({training_args.num_iterations}). If resuming from a checkpoint, "
            f"you may need to manually pick a checkpoint where the step is divisible by "
            f"{training_args.num_iterations}."
        )

    trainer.train()


if __name__ == "__main__":
    train()
