#!/bin/bash
#SBATCH --job-name=dream-grpo-math500
#SBATCH --output=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean/math500_train/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# Plain GRPO (dllm.pipelines.rl.grpo.DreamGRPOTrainer -- TRL's native PPO-clip loss), Dream-v0-
# Instruct-7B, trained on MATH-500's 500 problem statements as prompts ONLY (no ground-truth
# answers -- see train_math500_grpo.py's own docstring) with a Skywork reward model as the sole
# reward signal. Intent: after this finishes, re-evaluate the resulting checkpoint on MATH-500
# WITH ground-truth labels (../math_eval/eval_math500.py) to see what optimizing a pure
# open-ended reward model does to actual correctness.
#
# Hyperparameters copied from this repo's own dream_grpo.sbatch (wildchat + Skywork), the closest
# existing apples-to-apples comparison -- same model/reward-model/batch-size/learning-rate/steps/
# temperature/block_size/beta/epsilon/num_iterations. Only --dataset is gone (this script only
# ever does the MATH-500-prompts setup, no --dataset flag exists), and this uses all 4 GPUs on
# the gb node (--ntasks-per-node=4/--num_processes 4) rather than dream_grpo.sbatch's own 2 --
# confirmed working end-to-end with this exact accelerate/zero2 setup via a live 4-GPU smoke test
# before this sbatch was finalized.
#
# --output above is an ABSOLUTE path deliberately -- a relative one resolves against wherever
# `sbatch` is invoked from, not this script's own location.
#
# To resume a crashed/timed-out run (qgb's MaxWall is 12h), override BOTH RUN_NAME (so this run
# reuses the SAME output_dir the crashed job was writing to, not a fresh one keyed on the new
# SLURM_JOB_ID) and RESUME_FROM_CHECKPOINT (a checkpoint-N dir under that output_dir, or "true" to
# auto-pick the latest one there):
#   sbatch --export=RUN_NAME=dream-grpo-math500-<old_job_id>,RESUME_FROM_CHECKPOINT=true \
#       math500_train/dream_grpo_math500.sbatch

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

RUN_NAME="${RUN_NAME:-dream-grpo-math500-${SLURM_JOB_ID}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LOGDIR=$SCRATCH/checkpoints/entrgi_rgrl_clean/dream_grpo_math500/slurm/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

RESUME_ARGS=""
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    RESUME_ARGS="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
fi

srun --output "${LOGDIR}/%N.out" \
    bash -c "accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_machines \$SLURM_NNODES \
        --num_processes 4 \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        math500_train/train_math500_grpo.py \
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \
        --max_steps 1500 --max_completion_length 512 --save_steps 100 \
        --learning_rate 5e-6 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --num_generations 4 --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 --num_iterations 5 \
        --steps 512 --temperature 0.3 \
        --block_size 512 \
        --dream_alg entropy \
        --beta 0.0 --epsilon 0.2 \
        --logging_steps 5 \
        --reward_model Skywork/Skywork-Reward-V2-Qwen3-8B \
        --log_completions True \
        --run_name $RUN_NAME \
        --output_dir $SCRATCH/checkpoints/entrgi_rgrl_clean/$RUN_NAME \
        $RESUME_ARGS"
