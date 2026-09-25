#!/bin/bash
#SBATCH --job-name=dream-gdpo-wildchat
#SBATCH --output=vista_sbatch/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# GDPO baseline (dllm.pipelines.rl.gdpo.DreamGRPOTrainerWithGDPOEstimator -- GDPO's own faithful
# loss/log-likelihood port, replacing TRL's PPO-clip loss_type switch entirely; see
# dllm/dllm/pipelines/rl/gdpo/estimator.py's own docstring), Dream-v0-Instruct-7B, wildchat.
# --log_prob_mode gauss-3 is GDPO's own real production default (its own quadrature likelihood
# estimator over the completion-masking rate).
#
# --max_steps/--per_device_train_batch_size/--num_iterations/--block_size/--beta match this
# vista_sbatch/'s own RGRL sbatches' common arguments exactly (500/8/1/128/0.01) for an
# apples-to-apples comparison, rather than the original GDPO repo's own real config
# (kevinrojas1499/GDPO's gdpo/slurm_scripts/train.yaml: beta=0.00, epsilon=0.5, num_iterations=12).
#
# --gradient_checkpointing False is REQUIRED here specifically: gradient checkpointing's
# recompute-during-backward, combined with GDPO's gauss-3 estimator doing 3 separate forward
# passes summed into one loss before a single backward() call, trips DeepSpeed ZeRO2's
# "parameter already reduced" assertion.
#
# No --deprioritize_eos/--eos_penalty here -- that field was removed from this train.py entirely
# (it referenced a DreamSamplerConfig kwarg that doesn't exist in this clean clone and would have
# crashed at startup; the feature was never ported here in the first place).
#
# No --eval_strategy/--eval_steps and no held-out eval split in train.py -- matches RGRL's and
# plain GRPO's own train-only setup exactly, so all three methods' sbatches are identical apart
# from the training algorithm itself.

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

RUN_NAME=dream-gdpo-wildchat-${SLURM_JOB_ID}
LOGDIR=$SCRATCH/checkpoints/entrgi_rgrl_clean/dream_gdpo/slurm/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

srun --output "${LOGDIR}/%N.out" \
    bash -c "accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_machines \$SLURM_NNODES \
        --num_processes 2 \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        dllm/examples/rl/gdpo/dream/train.py \
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
        --beta 0.0 --epsilon 0.5 \
        --logging_steps 5 \
        --log_prob_mode gauss-3 \
        --reward_model Skywork/Skywork-Reward-V2-Qwen3-8B \
        --log_completions True \
        --gradient_checkpointing False \
        --run_name $RUN_NAME \
        --output_dir $SCRATCH/checkpoints/entrgi_rgrl_clean/$RUN_NAME"
