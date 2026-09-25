#!/bin/bash
#SBATCH --job-name=lcb-eval
#SBATCH --output=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean/lcb_eval/logs/%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb

# LiveCodeBench (LeetCode/functional subset) eval of Dream-org/Dream-v0-Instruct-7B, sharded
# across the 4 GPUs of a single GB200 node -- same conventions as math_eval/math500_eval.sbatch.
# Scoring is delegated to the official lcb_runner evaluator (see score_lcb.py); no custom
# execution sandbox lives in this repo.
#
# Override MODEL / RELEASE_VERSION / etc. via:
#   sbatch --export=MODEL="...",RELEASE_VERSION="release_v2" lcb_eval/lcb_eval.sbatch
# --output above is an ABSOLUTE path deliberately -- a relative one resolves against wherever
# `sbatch` is invoked from, not this script's own location.

set -euo pipefail

REPO_ROOT=/scratch/09749/atutej/entrgi-rgrl-work/entrgi-rgrl-clean
mkdir -p "$REPO_ROOT/lcb_eval/logs"

MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_SAMPLE_SIZE="${TEST_SAMPLE_SIZE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/lcb_eval/outputs}"

cd "$REPO_ROOT/lcb_eval"
EXTRA_ARGS=()
if [[ -n "$TEST_SAMPLE_SIZE" ]]; then
    EXTRA_ARGS+=(--test_sample_size "$TEST_SAMPLE_SIZE")
fi

./run_lcb_eval.sh \
    --num_gpus 4 \
    --model "$MODEL" \
    --release_version "$RELEASE_VERSION" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --force_overwrite \
    "${EXTRA_ARGS[@]}"
