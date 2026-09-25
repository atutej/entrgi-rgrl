#!/bin/bash
#SBATCH --job-name=dream-rgrl-entrgi-wildchat
#SBATCH --output=vista_sbatch/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# RGRL post-training, Dream-v0-Instruct-7B, EntRGi guidance -- started from the "Dream -- EntRGi"
# command in this repo's own README.md ("RGRL: Post-Training" section), wrapped for Vista SLURM
# (accelerate launch via srun, 2 ranks matching the README's own CUDA_VISIBLE_DEVICES=0,1
# --num_processes 2), then extended with this session's own additions on top: guidance_kl_beta
# (frozen-base-model trust region during guidance), and --loss_backend psft (Proximal SFT --
# GDPOEstimatorTrainerMixin's own ratio/clip formula with advantages fixed to 1; see
# RGRLTrainerWithPSFTLoss's docstring). --num_iterations 5 (up from the README's 1) actually
# exercises PSFT's batch-reuse-safety property -- at num_iterations=1 the ratio never has a
# chance to drift from 1, so PSFT's gradient is indistinguishable from plain SFT's. --beta 0
# matches the paper's own PSFT exactly (no separate KL-to-ref term -- the clip alone is the
# trust region; a nonzero beta here would add a real extra regularizer PSFT doesn't have).

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

RUN_NAME=dream-rgrl-entrgi-wildchat-${SLURM_JOB_ID}
LOGDIR=$SCRATCH/checkpoints/entrgi_rgrl_clean/dream_rgrl_entrgi/slurm/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

srun --output "${LOGDIR}/%N.out" \
    bash -c "accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_machines \$SLURM_NNODES \
        --num_processes 2 \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        dllm/examples/rl/rgrl/dream/train.py \
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
        --run_name $RUN_NAME \
        --output_dir $SCRATCH/checkpoints/entrgi_rgrl_clean/$RUN_NAME"
