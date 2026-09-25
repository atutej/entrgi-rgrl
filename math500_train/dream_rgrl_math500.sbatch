#!/bin/bash
#SBATCH --job-name=dream-rgrl-math500
#SBATCH --output=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean/math500_train/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# RGRL (entrgi-guided online SFT, dllm.pipelines.rl.rgrl.RGRLTrainer) on Dream-v0-Instruct-7B,
# trained on MATH-500's 500 problem statements as prompts ONLY (no ground-truth answers -- see
# train_math500_rgrl.py's own docstring) with a Skywork reward model as the sole optimized
# reward. Companion to dream_grpo_math500.sbatch in this same folder -- same dataset, same
# decoding budget/temperature, same training-step count, so the two are directly comparable;
# only the trainer (RGRL's entrgi guidance vs plain GRPO) and its own extra hyperparameters
# (M/eta/guidance_type/guidance_kl_beta/loss_backend) differ.
#
# --max_completion_length/--steps 512, --temperature 0.3: deliberately NOT this repo's own
# production RGRL defaults (128/128/0.9, see dream_rgrl_entrgi.sbatch) -- matches
# dream_grpo_math500.sbatch's own CURRENT decoding budget/temperature exactly instead, per this
# session's own finding that the 128-token/temperature=0.9 rollout config causes completions to
# collapse to bare "\boxed{guess}" answers within ~150-200 steps on MATH-500 prompts specifically.
# No --dream_top_p flag here either, matching dream_grpo_math500.sbatch's own choice to leave it
# unset -- both scripts' TrainingArguments default dream_top_p to 0.95, so this stays in sync with
# GRPO automatically rather than hardcoding a value that could silently drift out of sync again.
# Everything else (M, eta, guidance_type, guidance_kl_beta, loss_backend, log_prob_mode,
# guidance_reward_model, reward_model, LoRA config, learning rate schedule, batch/generation
# counts) is copied verbatim from dream_rgrl_entrgi.sbatch's own production hyperparameters.
#
# --output above is an ABSOLUTE path deliberately -- a relative one resolves against wherever
# `sbatch` is invoked from, not this script's own location.

set -euo pipefail

REPO_ROOT=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean
mkdir -p "$REPO_ROOT/math500_train/logs"
cd "$REPO_ROOT"

CONDA_ROOT=/work/09749/atutej/vista/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dllm_entrgi_gb100

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/opt/apps/cuda/12.4
export CC=/usr/bin/gcc CXX=/usr/bin/g++

export HF_HOME=/scratch/09749/atutej/cache
export TRANSFORMERS_CACHE=/scratch/09749/atutej/cache

export WANDB_ENTITY=atutej
export WANDB_PROJECT=rgrl

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$SLURM_NNODES"

RUN_NAME=dream-rgrl-math500-${SLURM_JOB_ID}
LOGDIR=$SCRATCH/checkpoints/entrgi_rgrl_clean/dream_rgrl_math500/slurm/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

srun --output "${LOGDIR}/%N.out" \
    bash -c "accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_machines \$SLURM_NNODES \
        --num_processes 4 \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        math500_train/train_math500_rgrl.py \
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \
        --max_steps 1500 --max_completion_length 512 --save_steps 100 \
        --learning_rate 5e-6 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --num_generations 4 --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 --num_iterations 5 \
        --steps 512 --temperature 0.3 \
        --dream_alg entropy \
        --M 3 --eta 0.5 \
        --loss_backend grpo --log_prob_mode gauss-3 \
        --beta 0 \
        --guidance_type entrgi \
        --guidance_kl_beta 0.01 \
        --gradient_checkpointing False \
        --logging_steps 5 \
        --guidance_reward_model Skywork/Skywork-Reward-V2-Qwen3-8B \
        --reward_model Skywork/Skywork-Reward-V2-Qwen3-8B \
        --log_completions True \
        --run_name $RUN_NAME \
        --output_dir $SCRATCH/checkpoints/entrgi_rgrl_clean/$RUN_NAME"
