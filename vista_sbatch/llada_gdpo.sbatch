#!/bin/bash
#SBATCH --job-name=llada-gdpo-wildchat
#SBATCH --output=vista_sbatch/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gb
#SBATCH -A ASC26088

# GDPO baseline (dllm.pipelines.rl.gdpo.DiffuGRPOTrainerWithGDPOEstimator -- GDPO's own faithful
# loss/log-likelihood port, replacing TRL's PPO-clip loss_type switch entirely; see
# dllm/dllm/pipelines/rl/gdpo/estimator.py's own docstring), LLaDA-8B-Instruct, wildchat.
# --log_prob_mode gauss-3 is GDPO's own real production default (its own quadrature likelihood
# estimator over the completion-masking rate). --gradient_checkpointing False required for the
# same DeepSpeed ZeRO2 reason as dream_gdpo.sbatch.
#
# --max_steps/--per_device_train_batch_size/--num_iterations/--block_size/--beta: see
# dream_gdpo.sbatch's header -- matched to this vista_sbatch/'s own RGRL sbatches' common
# arguments (500/8/1/128/0.01) for an apples-to-apples comparison. No --deprioritize_eos/
# --eos_penalty here -- that
# mechanism was only ported to DreamSampler, not MDLMSampler (LLaDA's sampler), so it doesn't
# exist for this backbone in either the working copy or this clean clone.

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

RUN_NAME=llada-gdpo-wildchat-${SLURM_JOB_ID}
LOGDIR=$SCRATCH/checkpoints/entrgi_rgrl_clean/llada_gdpo/slurm/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

srun --output "${LOGDIR}/%N.out" \
    bash -c "accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_machines \$SLURM_NNODES \
        --num_processes 2 \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        dllm/examples/rl/gdpo/llada/train.py \
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
        --beta 0.0 --epsilon 0.5 \
        --log_prob_mode gauss-3 \
        --reward_model Skywork/Skywork-Reward-V2-Qwen3-0.6B \
        --log_completions True \
        --gradient_checkpointing False \
        --logging_steps 5 \
        --run_name $RUN_NAME \
        --output_dir $SCRATCH/checkpoints/entrgi_rgrl_clean/$RUN_NAME"
