"""
DiffuGRPO training for Dream-v0-Instruct-7B on MATH-500 *prompts only*, rewarded with a Skywork
reward model instead of ground-truth correctness.

The OPTIMIZED reward comes entirely from make_skywork_reward_func() (the same open-ended,
correctness-agnostic reward path this repo's wildchat/magpie/lmsys GRPO runs already use), not
from any math verifier -- ground truth never influences the loss. The intent (per the person
running this) is: train here without labels, then separately re-evaluate the resulting checkpoint
on MATH-500 WITH ground-truth labels (e.g. via ../math_eval/eval_math500.py) to see what a pure
Skywork-reward RL run does to actual correctness.

A second reward function, math500_correctness_reward_func, DOES read ground_truth, but purely as
a sanity-check metric: it's registered with reward_weight=0.0 (see TrainingArguments.reward_weights
below), so TRL still logs its own rewards/math500_correctness/mean every step (the same per-function
metric mechanism already used for rewards/skywork_.../mean) without it ever entering the actual
PPO-clip loss or advantage computation. Useful to eyeball early-training accuracy isn't degenerate
(e.g. all-empty completions) without waiting for a full post-hoc eval_math500.py pass.

Structurally a close copy of dllm/examples/rl/grpo/dream/train.py (this repo's own wildchat+Skywork
GRPO script) -- same TrainingArguments shape, same DreamGRPOTrainer/DreamSampler wiring -- with the
dataset swapped for MATH-500 questions and the --dataset choice removed (this script only ever
does the one setup). Not the "gdpo"/"rgrl" trainers used elsewhere in this repo -- this is plain
GRPO's own PPO-clip loss (TRL's native loss_type), the same variant dream_grpo.sbatch's wildchat
run uses, so a math500-prompts run is comparable to that wildchat run.

Usage (1 GPU quick check, no LoRA):
    accelerate launch --config_file ../dllm/scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        train_math500_grpo.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --max_steps 20 --output_dir /tmp/math500-grpo-smoke

Multi-GPU (matches vista_sbatch/dream_grpo.sbatch's wildchat hyperparameters):
    accelerate launch --config_file ../dllm/scripts/accelerate_configs/zero2.yaml \\
        train_math500_grpo.py \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \\
        --max_steps 500 --learning_rate 5e-6 \\
        --num_generations 4 --per_device_train_batch_size 8 \\
        --gradient_accumulation_steps 1 --num_iterations 5 \\
        --steps 128 --temperature 0.9 --block_size 128 \\
        --beta 0.0 --epsilon 0.2 \\
        --reward_model Skywork/Skywork-Reward-V2-Qwen3-8B \\
        --output_dir /scratch/.../checkpoints/.../math500-grpo
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
from dllm.pipelines.dream import DreamSampler, DreamSamplerConfig
from dllm.pipelines.rl import DiffuGRPOConfig, DreamGRPOTrainer
from dllm.pipelines.rl.grpo.rewards.skywork import make_skywork_reward_func

logger = dllm.utils.get_default_logger(__name__)

# eval/reason_benchmarks/math_500.py is only used here for its data loading (the 500 problem
# strings) -- never its scoring/ground-truth. Same sys.path/chdir convention as
# math_eval/eval_math500.py, so this script can be launched from anywhere.
ORIG_CWD = os.getcwd()
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def get_math500_questions_only() -> Dataset:
    """MATH-500 problems as prompts, plus ground_truth carried along ONLY for the logging-only
    math500_correctness_reward_func below (reward_weight=0.0 -- never affects the loss). Uses the
    exact same prompt template reason_benchmarks/math_500.py's own eval path uses ("Enclose your
    final answer in \\boxed{}"), so there's no train/eval prompt distribution shift when this
    checkpoint later gets re-evaluated with math_eval/eval_math500.py."""
    sys.path.insert(0, str(EVAL_DIR))
    cwd_before = os.getcwd()
    os.chdir(EVAL_DIR)
    try:
        from reason_benchmarks.math_500 import MATH500
        problems = MATH500.load()
    finally:
        os.chdir(cwd_before)

    # MATH500.load() already returns processed instances -- "input_prompt" is the exact chat
    # list eval_math500.py feeds the model ("<problem>\n\nEnclose your final answer in
    # \\boxed{}."), reused as-is here.
    rows = [
        {"prompt": d["input_prompt"], "ground_truth": d["ground_truth"]}
        for d in problems
    ]
    return Dataset.from_list(rows)


def math500_correctness_reward_func(
    prompts, completions, ground_truth, verbose=False, **kwargs
) -> list[float]:
    """Logging-only: MATH-500 boxed-answer correctness (0 or 100 per completion, same
    last_boxed_only_string/is_equiv scoring eval_math500.py uses), scaled to sit in the same
    0-100 range GRPOTrainer logs metrics in. Registered with reward_weight=0.0 in train() --
    TRL still computes and logs rewards/math500_correctness/mean every step, but this value is
    multiplied by 0 before being added to the actual reward used for advantages/loss."""
    sys.path.insert(0, str(EVAL_DIR))
    from reason_benchmarks.math_500 import compute_score

    responses = [completion[0]["content"] for completion in completions]
    scores = [100.0 * compute_score(r, gt) for r, gt in zip(responses, ground_truth)]
    if verbose:
        print(f"[math500_correctness] mean={sum(scores) / len(scores):.1f} "
              f"n={len(scores)} sample_gt={ground_truth[0]!r} sample_response={responses[0][:200]!r}")
    return scores


@dataclass
class TrainingArguments(DiffuGRPOConfig):
    output_dir: str = ".models/Dream-v0-Instruct-7B/grpo-math500"
    verbose_reward: bool = field(
        default=False, metadata={"help": "Enable verbose printing in reward functions."}
    )
    dream_alg: str = field(
        default="entropy",
        metadata={"help": "Confidence algorithm for Dream demasking: entropy, maskgit_plus, topk_margin."},
    )
    dream_top_p: float = field(default=0.95, metadata={"help": "top-p for Dream token sampling."})
    dream_top_k: int = field(default=50, metadata={"help": "top-k for Dream token sampling."})
    reward_model: str = field(
        default="Skywork/Skywork-Reward-V2-Qwen3-8B",
        metadata={"help": "Reward model scoring completions -- MATH-500's ground-truth answers "
                  "are never used as a training signal here, only this reward model."},
    )


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "Dream-org/Dream-v0-Instruct-7B"

    # ---- Dataset & reward --------------------------------------------------------
    dataset = get_math500_questions_only()
    # reward_weights=[1.0, 0.0]: only the Skywork reward enters the actual loss.
    # math500_correctness_reward_func is logging-only (see its own docstring).
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

    # ---- Dream sampler ----------------------------------------------------------------
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

    # ---- Trainer ----------------------------------------------------------------------
    logger.info("Start GRPO training (Dream, MATH-500 prompts, Skywork reward)...")
    trainer = DreamGRPOTrainer(
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

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    train()
