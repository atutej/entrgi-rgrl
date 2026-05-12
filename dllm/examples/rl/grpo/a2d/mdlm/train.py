"""
GRPO training for A2D (MDLM) models with diffusion denoising.

Supported datasets: gsm8k, countdown, sudoku, math, code

Local users
-----------
- 1 GPU, quick test (no LoRA):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/rl/grpo/a2d/mdlm/train.py \
        --model_name_or_path dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
        --dataset gsm8k --max_steps 50 \
        --output_dir .models/a2d/Qwen3-0.6B-diffusion-mdlm-v0.1/grpo

- 8 GPUs, DeepSpeed ZeRO-2:
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/rl/grpo/a2d/mdlm/train.py \
        --model_name_or_path dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
        --lora_r 128 --lora_alpha 64 --lora_dropout 0.05 \
        --dataset gsm8k \
        --max_steps 15000 --learning_rate 3e-6 \
        --num_generations 6 --per_device_train_batch_size 6 \
        --gradient_accumulation_steps 2 --num_iterations 12 \
        --block_size 32 --steps 128 \
        --p_mask_prompt 0.15 --beta 0.04 --epsilon 0.5 \
        --sync_ref_model True --ref_model_sync_steps 64 \
        --output_dir .models/a2d/Qwen3-0.6B-diffusion-mdlm-v0.1/grpo

Slurm users
-----------
- 1 Node, 8 GPUs:
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/rl/grpo/a2d/mdlm/train.py" \
        -- --dataset gsm8k --output_dir .models/a2d/Qwen3-0.6B-diffusion-mdlm-v0.1/grpo
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.pipelines.rl import DiffuGRPOConfig, DiffuGRPOTrainer, get_dataset_and_rewards

logger = dllm.utils.get_default_logger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingArguments(DiffuGRPOConfig):
    output_dir: str = ".models/a2d/Qwen3-0.6B-diffusion-mdlm-v0.1/grpo"
    dataset: Optional[str] = field(
        default="gsm8k",
        metadata={"help": "Dataset to train on: gsm8k, countdown, sudoku, math, code."},
    )
    verbose_reward: bool = field(
        default=False,
        metadata={"help": "Enable verbose printing in reward functions."},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"

    # ---- Dataset & rewards ------------------------------------------------------
    dataset, reward_functions = get_dataset_and_rewards(training_args.dataset)

    if training_args.verbose_reward:
        reward_functions = [partial(fn, verbose=True) for fn in reward_functions]

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
    # LoRA is NOT applied inside get_model (i.e. don't pass lora=True to ModelArguments).
    # GRPOTrainer requires peft_config separately so it can manage the reference model
    # and control adapter enable/disable during old-logp computation internally.
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
            task_type="CAUSAL_LM",
            lora_dropout=model_config.lora_dropout,
        )

    # ---- Sampler ----------------------------------------------------------------
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = MDLMSamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        block_size=training_args.block_size,
        temperature=training_args.temperature or 0.0,
        cfg_scale=training_args.cfg_scale,
        remasking=training_args.remasking,
    )

    # ---- Trainer ----------------------------------------------------------------
    logger.info("Start GRPO training...")
    trainer = DiffuGRPOTrainer(
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
