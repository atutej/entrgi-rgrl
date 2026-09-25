#!/bin/bash
#SBATCH --job-name=score-wildchat-heldout
#SBATCH --output=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean/vista_sbatch/logs/%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# Generates completions on a sample of WildChat prompts and scores them with a Skywork reward
# model -- see eval/score_wildchat_heldout.py's own docstring for why this is NOT a held-out
# guarantee (the wildchat dataset training used has no leftover unseen slice; overlap accepted).
# Single GPU, single process (loads the policy model to generate, frees it, then loads the
# reward model to score -- sequential, not concurrent, on one GPU).
#
# --model / --skywork_reward_model below are just EXAMPLES -- override via:
#   sbatch --export=MODEL="...",SKYWORK_REWARD_MODEL="..." vista_sbatch/score_wildchat_heldout.sbatch
# --output above is an ABSOLUTE path deliberately (not "vista_sbatch/logs/%x-%j.out") -- a
# relative --output path resolves against wherever `sbatch` is invoked FROM, not this script's
# own location, which already caused a doubled/wrong log path once (score_with_skywork.sbatch,
# job 1019821, submitted from inside vista_sbatch/ itself).

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

MODEL="${MODEL:-/scratch/09749/atutej/checkpoints/entrgi_rgrl_clean/dream-rgrl-entrgi-wildchat-1018716/checkpoint-500}"
SKYWORK_REWARD_MODEL="${SKYWORK_REWARD_MODEL:-Skywork/Skywork-Reward-V2-Qwen3-8B}"

cd eval
python score_wildchat_heldout.py \
    --model "$MODEL" \
    --skywork_reward_model "$SKYWORK_REWARD_MODEL" \
    --num_held_out_prompts 128 \
    --max_tokens 128 \
    --temperature 0.9 \
    --top_p 0.9 \
    --gen_batch_size 8 \
    --reward_batch_size 8
