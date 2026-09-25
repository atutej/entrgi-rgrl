"""
GDPO training for Dream-v0-Instruct-7B with diffusion denoising.

Same as `examples/rl/grpo/dream/train.py`, except the loss/log-likelihood is GDPO's own
faithful port (`dllm.pipelines.rl.gdpo.DreamGRPOTrainerWithGDPOEstimator`) instead of TRL's
own PPO-clip `loss_type` switch -- see dllm/dllm/pipelines/rl/gdpo/estimator.py's own docstring
for exactly what "faithful" means here (plain-mean reduction, no shared batch-wide normalizer,
no gradient-accumulation-steps division, sequence-level -- not per-token -- importance ratio).

Supported datasets: gsm8k, countdown, sudoku, math, code, wildchat, magpie, lmsys

Local users
-----------
- 1 GPU, quick sanity check on countdown (no LoRA):
    accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/rl/gdpo/dream/train.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --load_in_4bit True \\
        --dataset countdown --max_steps 50 \\
        --output_dir .models/Dream-v0-Instruct-7B/gdpo-countdown

- Multi-GPU, WildChat + Skywork reward:
    accelerate launch \\
        --config_file scripts/accelerate_configs/zero2.yaml \\
        examples/rl/gdpo/dream/train.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \\
        --dataset wildchat \\
        --max_steps 500 --learning_rate 5e-6 \\
        --num_generations 4 --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 2 --num_iterations 5 \\
        --steps 128 \\
        --beta 0.0 --epsilon 0.5 \\
        --log_prob_mode gauss-3 \\
        --output_dir .models/Dream-v0-Instruct-7B/gdpo-wildchat
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.pipelines.dream import DreamSampler, DreamSamplerConfig
from dllm.pipelines.rl import get_dataset_and_rewards
from dllm.pipelines.rl.gdpo import GDPOEstimatorConfig, DreamGRPOTrainerWithGDPOEstimator

logger = dllm.utils.get_default_logger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingArguments(GDPOEstimatorConfig):
    output_dir: str = ".models/Dream-v0-Instruct-7B/gdpo"
    dataset: Optional[str] = field(
        default="countdown",
        metadata={
            "help": (
                "Dataset to train on: gsm8k, countdown, sudoku, math, code, wildchat, magpie, "
                "lmsys. Use gsm8k/countdown/sudoku/math/code to sanity-check the Dream logit "
                "handling on verifiable tasks before running the open-ended chat datasets."
            )
        },
    )
    verbose_reward: bool = field(
        default=False,
        metadata={"help": "Enable verbose printing in reward functions."},
    )
    # Dream sampler knobs
    dream_alg: str = field(
        default="entropy",
        metadata={"help": "Confidence algorithm for Dream demasking: entropy, maskgit_plus, topk_margin."},
    )
    dream_top_p: float = field(default=0.95, metadata={"help": "top-p for Dream token sampling."})
    dream_top_k: int = field(default=50, metadata={"help": "top-k for Dream token sampling."})
    reward_model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Reward model for scoring completions. "
                "Only applies to open-ended datasets (wildchat, magpie, lmsys). "
                "Defaults to Skywork/Skywork-Reward-V2-Qwen3-1.7B when unset."
            )
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "Dream-org/Dream-v0-Instruct-7B"

    # ---- Dataset & rewards ------------------------------------------------------
    dataset, reward_functions = get_dataset_and_rewards(
        training_args.dataset,
        reward_model=training_args.reward_model,
    )

    if training_args.verbose_reward:
        reward_functions = [partial(fn, verbose=True) for fn in reward_functions]

    # No held-out eval split -- matches RGRL's and plain GRPO's own train.py exactly (train on
    # the full shuffled dataset, no eval_dataset at all), so all three methods' sbatches are
    # identical apart from the training algorithm itself.
    train_set = dataset.shuffle(seed=training_args.seed)

    # ---- Model & Tokenizer ------------------------------------------------------
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=model_config.model_name_or_path,
        load_in_4bit=(
            model_config.load_in_4bit
            if hasattr(model_config, "load_in_4bit")
            else False
        ),
    )
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model.config.use_cache = False

    # ---- LoRA -------------------------------------------------------------------
    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            lora_dropout=model_config.lora_dropout,
        )

    # ---- Dream sampler ----------------------------------------------------------
    sampler = DreamSampler(model=model, tokenizer=tokenizer)
    sampler_config = DreamSamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        temperature=training_args.temperature or 1.0,
        cfg_scale=training_args.cfg_scale,
        alg=training_args.dream_alg,
        top_p=training_args.dream_top_p,
        top_k=training_args.dream_top_k,
        right_shift_logits=True,
    )

    # ---- Trainer ----------------------------------------------------------------
    logger.info("Start GDPO training (Dream)...")
    trainer = DreamGRPOTrainerWithGDPOEstimator(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_set,
        processing_class=tokenizer,
        peft_config=peft_config,
        sampler=sampler,
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
