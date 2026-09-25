#!/bin/bash
#SBATCH --job-name=dream-rgrl-aps-wildchat
#SBATCH --output=vista_sbatch/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# RGRL post-training, Dream-v0-Instruct-7B, APS guidance -- exactly the "Dream -- APS" command
# from this repo's own README.md ("RGRL: Post-Training" section). Identical to
# dream_rgrl_entrgi.sbatch except --guidance_type aps (straight-through estimator, w=1) instead
# of entrgi (entropy-weighted interpolation) -- see that file's own header for the rest of the
# rationale (no extra flags beyond the README's own command).

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

RUN_NAME=dream-rgrl-aps-wildchat-${SLURM_JOB_ID}
LOGDIR=$SCRATCH/checkpoints/entrgi_rgrl_clean/dream_rgrl_aps/slurm/${SLURM_JOB_ID}
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
        --num_generations 4 --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 --num_iterations 1 \
        --steps 128 --temperature 0.9 \
        --dream_alg entropy \
        --M 1 --eta 0.5 \
        --loss_backend gdpo_estimator --log_prob_mode gauss-3 \
        --guidance_type aps \
        --guidance_kl_beta 0.01 \
        --gradient_checkpointing False \
        --logging_steps 5 \
        --guidance_reward_model Skywork/Skywork-Reward-V2-Qwen3-0.6B \
        --reward_model Skywork/Skywork-Reward-V2-Qwen3-0.6B \
        --run_name $RUN_NAME \
        --output_dir $SCRATCH/checkpoints/entrgi_rgrl_clean/$RUN_NAME"
