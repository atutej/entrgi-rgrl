from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.pipelines.rl import RGRLTrainer, get_dataset_and_rewards
from dllm.pipelines.rl.rgrl import RgrlLLaDASampler, RgrlLLaDASamplerConfig
from dllm.pipelines.rl.gdpo import (
    RGRLGDPOConfig,
    RGRLTrainerWithGDPOEstimatorOnly,
    RGRLTrainerWithPSFTLoss,
)

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class TrainingArguments(RGRLGDPOConfig):
    output_dir: str = ".models/LLaDA-8B-Instruct/rgrl"
    loss_backend: str = field(
        default="gdpo_estimator",
        metadata={
            "help": "'gdpo_estimator' (default: swaps ONLY the log-prob estimator for GDPO's "
            "own lower-variance quadrature/MC estimator -- see log_prob_mode -- keeping RGRL's "
            "own loss formula unchanged; see RGRLTrainerWithGDPOEstimatorOnly's own docstring), "
            "'psft' (Proximal SFT -- GDPO's own ratio/clip formula with advantages fixed to 1; "
            "see RGRLTrainerWithPSFTLoss's own docstring, including the --beta caveat), or "
            "'grpo' (RGRLTrainer's original native plain-SFT loss, with its inherited "
            "single-random-masking-draw log-prob estimator)."
        },
    )
    log_prob_mode: str = field(
        default="gauss-3",
        metadata={
            "help": "Only used when loss_backend='gdpo_estimator'. 'full_mask' or one of "
            "GAUSS_QUADRATURES's keys ('gauss-1'..'gauss-5') or 'mc' -- gauss-3 matches GDPO's "
            "own real production default."
        },
    )
    dataset: Optional[str] = field(
        default="wildchat",
        metadata={"help": "Dataset: gsm8k, countdown, sudoku, math, code, wildchat, magpie, lmsys."},
    )
    verbose_reward: bool = field(
        default=False,
        metadata={"help": "Enable verbose printing in rule-based reward functions."},
    )
    llada_alg: str = field(
        default="entropy",
        metadata={"help": "Confidence algorithm: entropy, maskgit_plus, topk_margin."},
    )
    llada_top_p: float = field(default=0.95, metadata={"help": "top-p for sampling."})
    llada_top_k: int = field(default=50, metadata={"help": "top-k for sampling."})
    deprioritize_eos: bool = field(
        default=False,
        metadata={"help": "Set confidence=-inf at EOS-sampled positions during denoising."},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Reward model for scoring completions. Only applies to wildchat."},
    )
    zero_unmatched_embeddings: bool = field(
        default=True,
        metadata={"help": "Zero embedding for policy tokens absent from reward model vocab."},
    )


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "GSAI-ML/LLaDA-8B-Instruct"

    dataset, reward_functions = get_dataset_and_rewards(
        training_args.dataset,
        reward_model=training_args.reward_model,
    )

    if training_args.verbose_reward:
        reward_functions = [partial(fn, verbose=True) for fn in reward_functions]

    train_set = dataset.shuffle(seed=training_args.seed)

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

    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=model_config.lora_dropout,
        )

    sampler_config = RgrlLLaDASamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        temperature=training_args.temperature or 1.0,
        cfg_scale=training_args.cfg_scale,
        alg=training_args.llada_alg,
        top_p=training_args.llada_top_p,
        top_k=training_args.llada_top_k,
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
    logger.info(f"Starting RG-RL online SFT (LLaDA), loss_backend={training_args.loss_backend}...")
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
            f"num_iterations ({training_args.num_iterations}). If resuming from "
            f"a checkpoint, you may need to manually pick a compatible checkpoint."
        )

    trainer.train()


if __name__ == "__main__":
    train()
