#!/bin/bash
#SBATCH --job-name=math500-eval
#SBATCH --output=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean/math_eval/logs/%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# MATH-500 eval of Dream-org/Dream-v0-Instruct-7B, sharded across the 4 GPUs of a single GB200
# node (c672-* nodes each have 4 GPUs; gb partition nodes aren't requested via --gres here, same
# as this repo's other gb sbatches -- --nodes=1 already gives all 4).
#
# After generation+merge finish, also scores every (problem, candidate) pair with a Skywork
# reward model (single GPU, sequential -- see score_math500_skywork.py) unless
# SKYWORK_REWARD_MODEL is set to an empty string.
#
# Override MODEL / MAX_NEW_TOKENS / etc. via:
#   sbatch --export=MODEL="...",MAX_NEW_TOKENS=1024 math_eval/math500_eval.sbatch
# --output above is an ABSOLUTE path deliberately -- a relative one resolves against wherever
# `sbatch` is invoked from, not this script's own location.

set -euo pipefail

REPO_ROOT=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean
mkdir -p "$REPO_ROOT/math_eval/logs"

MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/math_eval/outputs}"
SKYWORK_REWARD_MODEL="${SKYWORK_REWARD_MODEL:-Skywork/Skywork-Reward-V2-Qwen3-8B}"

cd "$REPO_ROOT/math_eval"
EXTRA_ARGS=()
if [[ -n "$SKYWORK_REWARD_MODEL" ]]; then
    EXTRA_ARGS+=(--skywork_reward_model "$SKYWORK_REWARD_MODEL")
fi

./run_math500_eval.sh \
    --num_gpus 4 \
    --model "$MODEL" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --force_overwrite \
    "${EXTRA_ARGS[@]}"
