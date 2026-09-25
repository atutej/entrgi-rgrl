#!/bin/bash
#SBATCH --job-name=llada-grpo-wildchat
#SBATCH --output=vista_sbatch/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gb
#SBATCH -A ASC26088

# Plain GRPO baseline (dllm.pipelines.rl.grpo.DiffuGRPOTrainer -- TRL's own native PPO-clip
# loss_type, dllm's default single-random-masking-draw log-prob estimator), LLaDA-8B-Instruct,
# wildchat. Most faithful plain-GRPO counterpart to this same vista_sbatch/'s own
# llada_gdpo.sbatch -- see dream_grpo.sbatch's header for the full rationale on what's
# deliberately dropped (log_prob_mode/mc_num_batches, gradient_checkpointing override,
# deprioritize_eos/eos_penalty, eval_strategy/eval_steps) and why: none of those exist on
# dllm/examples/rl/grpo/llada/train.py's own TrainingArguments, and this train.py builds no
# eval_dataset at all.
#
# --remasking low_confidence matches --dream_alg low_confidence's role for Dream (LLaDA/MDLM's
# own confidence-algorithm field name) -- already DiffuGRPOConfig's own default, set explicitly
# here for clarity/consistency with the other sbatches' style.

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

RUN_NAME=llada-grpo-wildchat-${SLURM_JOB_ID}
LOGDIR=$SCRATCH/checkpoints/entrgi_rgrl_clean/llada_grpo/slurm/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

srun --output "${LOGDIR}/%N.out" \
    bash -c "accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_machines \$SLURM_NNODES \
        --num_processes 2 \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        dllm/examples/rl/grpo/llada/train.py \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --load_in_4bit True --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \
        --dataset wildchat \
        --max_steps 500 --max_completion_length 128 --save_steps 100 \
        --learning_rate 5e-6 \
        --num_generations 4 --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 --num_iterations 1 \
        --steps 128 --temperature 0.9 \
        --block_size 128 \
        --remasking low_confidence \
        --beta 0.01 --epsilon 0.2 \
        --logging_steps 5 \
        --reward_model Skywork/Skywork-Reward-V2-Qwen3-0.6B \
        --log_completions True \
        --run_name $RUN_NAME \
        --output_dir $SCRATCH/checkpoints/entrgi_rgrl_clean/$RUN_NAME"
