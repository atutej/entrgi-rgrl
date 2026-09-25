#!/bin/bash
# Sharded LiveCodeBench (LeetCode/functional subset) eval: splits the target problems round-robin
# across --num_gpus GPU-pinned subprocesses of eval_lcb.py, waits for all of them, merges, then
# runs score_lcb.py (the official lcb_runner evaluator) to get pass@1.
#
# Usage: ./run_lcb_eval.sh --num_gpus 4 --model Dream-org/Dream-v0-Instruct-7B \
#            --release_version release_v1 --max_new_tokens 512 --temperature 0.1 --top_p 0.9
# Every flag other than --num_gpus/--gpu_ids is forwarded to eval_lcb.py.
set -euo pipefail

CONDA_ROOT=/work/09749/atutej/vista/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dllm_entrgi_gb100

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/opt/apps/cuda/12.4
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export HF_HOME=/scratch/09749/atutej/cache
export TRANSFORMERS_CACHE=/scratch/09749/atutej/cache
export PYTHONPATH=".:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS=4
GPU_IDS=""
FORWARDED=()
RELEASE_VERSION="release_v1"
START_DATE=""
END_DATE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --release_version) RELEASE_VERSION="$2"; FORWARDED+=("$1" "$2"); shift 2 ;;
        --start_date) START_DATE="$2"; FORWARDED+=("$1" "$2"); shift 2 ;;
        --end_date) END_DATE="$2"; FORWARDED+=("$1" "$2"); shift 2 ;;
        *) FORWARDED+=("$1"); shift ;;
    esac
done

if [[ -z "$GPU_IDS" ]]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
IFS=',' read -ra GPU_ID_ARR <<< "$GPU_IDS"
NUM_SHARDS=${#GPU_ID_ARR[@]}

pids=()
for shard_index in "${!GPU_ID_ARR[@]}"; do
    gpu_id="${GPU_ID_ARR[$shard_index]}"
    echo "[shard $shard_index] GPU $gpu_id: python eval_lcb.py ${FORWARDED[*]} --num_shards $NUM_SHARDS --shard_index $shard_index"
    CUDA_VISIBLE_DEVICES="$gpu_id" python eval_lcb.py "${FORWARDED[@]}" \
        --num_shards "$NUM_SHARDS" --shard_index "$shard_index" &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [[ "$fail" -ne 0 ]]; then
    echo "One or more shards failed" >&2
    exit 1
fi

MERGE_OUT=$(python eval_lcb.py "${FORWARDED[@]}" --num_shards "$NUM_SHARDS" --shard_index 0 --merge)
echo "$MERGE_OUT"

LCB_SUBMISSION=$(echo "$MERGE_OUT" | grep -oE '\-> .*_lcb_submission\.json$' | sed 's/^-> //')
if [[ -z "$LCB_SUBMISSION" ]]; then
    echo "Could not parse lcb_submission path from eval_lcb.py's own output; skipping scoring" >&2
    exit 1
fi

SCORE_ARGS=(--custom_output_file "$LCB_SUBMISSION" --release_version "$RELEASE_VERSION")
if [[ -n "$START_DATE" ]]; then SCORE_ARGS+=(--start_date "$START_DATE"); fi
if [[ -n "$END_DATE" ]]; then SCORE_ARGS+=(--end_date "$END_DATE"); fi

python score_lcb.py "${SCORE_ARGS[@]}"
