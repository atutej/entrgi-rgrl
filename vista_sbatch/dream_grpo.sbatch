#!/bin/bash
#SBATCH --job-name=dream-grpo-wildchat
#SBATCH --output=vista_sbatch/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# Plain GRPO baseline (dllm.pipelines.rl.grpo.DreamGRPOTrainer -- TRL's own native PPO-clip
# loss_type, dllm's default single-random-masking-draw log-prob estimator, NOT GDPO's own
# ratio/KL formula or quadrature/MC estimator), Dream-v0-Instruct-7B, wildchat.
#
# Deliberately the most faithful plain-GRPO counterpart to this same vista_sbatch/'s own
# dream_gdpo.sbatch -- same model/dataset/reward-model/batch-size/learning-rate/steps/temperature/
# block_size/beta/epsilon/num_iterations, so the two are comparable apples-to-apples. Everything
# GDPO-specific is deliberately DROPPED rather than carried over, since dllm/examples/rl/grpo/
# dream/train.py's own TrainingArguments doesn't define these fields at all:
#   - no --log_prob_mode/--mc_num_batches (GDPOEstimatorConfig-only fields)
#   - no --gradient_checkpointing False (that override existed specifically to dodge a DeepSpeed
#     ZeRO2 bug tripped by GDPO's gauss-3 estimator's 3 forward passes per loss call; plain GRPO
#     does exactly one forward pass per _compute_loss, so it never hits that bug)
#   - no --deprioritize_eos/--eos_penalty (not wired into this train.py's TrainingArguments;
#     porting them would mean pulling in more args than this file's own current CLI surface has)
#   - no --eval_strategy/--eval_steps (this train.py builds no eval_dataset/held-out split at
#     all -- unlike the RGRL/GDPO train.py's, it passes only train_dataset to the trainer, so
#     eval flags would be dead/misleading here)

set -euo pipefail

REPO_ROOT=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean
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

RUN_NAME=dream-grpo-wildchat-${SLURM_JOB_ID}
LOGDIR=$SCRATCH/checkpoints/entrgi_rgrl_clean/dream_grpo/slurm/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

srun --output "${LOGDIR}/%N.out" \
    bash -c "accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_machines \$SLURM_NNODES \
        --num_processes 2 \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        dllm/examples/rl/grpo/dream/train.py \
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \
        --dataset wildchat \
        --max_steps 500 --max_completion_length 128 --save_steps 100 \
        --learning_rate 5e-6 \
        --lr_scheduler_type cosine \
	--warmup_ratio 0.03 \
        --num_generations 4 --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 --num_iterations 5 \
        --steps 128 --temperature 0.9 \
        --block_size 128 \
        --dream_alg entropy \
        --beta 0.0 --epsilon 0.2 \
        --logging_steps 5 \
        --reward_model Skywork/Skywork-Reward-V2-Qwen3-8B \
        --log_completions True \
        --run_name $RUN_NAME \
        --output_dir $SCRATCH/checkpoints/entrgi_rgrl_clean/$RUN_NAME"
