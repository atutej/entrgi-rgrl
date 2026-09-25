#!/bin/bash
#SBATCH --job-name=score-with-skywork
#SBATCH --output=vista_sbatch/logs/%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# Scores already-generated alpacaeval2 completions (from launch_dream_benchmarks_inference.sh's
# output) with a Skywork reward model, independent of the LLM-judge pipeline -- see
# eval/score_with_skywork.py's own docstring. Single GPU, single process (no accelerate launch --
# this is a plain script, not a distributed one).
#
# --skywork_reward_model below is just an EXAMPLE (Skywork-Reward-Llama-3.1-8B-v0.2) -- pass
# whichever Skywork model you actually want via SKYWORK_REWARD_MODEL, e.g.:
#   sbatch --export=SKYWORK_REWARD_MODEL="Skywork/Skywork-Reward-V2-Qwen3-8B" vista_sbatch/score_with_skywork.sbatch

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

SKYWORK_REWARD_MODEL="${SKYWORK_REWARD_MODEL:-Skywork/Skywork-Reward-V2-Qwen3-8B}"

cd eval
python score_with_skywork.py \
    --completions_file \
        "outputs/alpacaeval2-compare/dream-dream-grpo-wildchat-1019400_checkpoint-500-test128/alpacaeval2_max1024t0.1p0.9n1_42.json" \
        "outputs/alpacaeval2-compare/dream-dream-rgrl-entrgi-wildchat-1018716_checkpoint-500-test128/alpacaeval2_max1024t0.1p0.9n1_42.json" \
    --skywork_reward_model "$SKYWORK_REWARD_MODEL" \
    --batch_size 8
