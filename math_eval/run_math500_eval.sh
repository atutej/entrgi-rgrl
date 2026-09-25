#!/bin/bash
# Sharded MATH-500 eval: splits the 500 problems round-robin across --num_gpus GPU-pinned
# subprocesses of eval_math500.py, waits for all of them, then merges the per-shard result
# files into one final accuracy number. Same conda env / library setup as
# eval/run_eval_sharded.sh, just pointed at this folder's own worker script.
#
# Optional reward-model scoring stage: pass --skywork_reward_model to also score every
# (problem, candidate) pair with a Skywork reward model AFTER generation finishes (runs
# sequentially, single GPU, once all 4 shard processes have exited and freed their GPU memory --
# see score_math500_skywork.py).
#
# Usage: ./run_math500_eval.sh --num_gpus 4 --model Dream-org/Dream-v0-Instruct-7B \
#            --max_new_tokens 512 --temperature 0.1 --top_p 0.9 --output_dir outputs \
#            --skywork_reward_model Skywork/Skywork-Reward-V2-Qwen3-8B
# Every flag other than --num_gpus/--gpu_ids/--skywork_reward_model/--skywork_batch_size is
# forwarded to eval_math500.py.
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
SKYWORK_MODEL=""
SKYWORK_BATCH_SIZE=8
FORWARDED=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --skywork_reward_model) SKYWORK_MODEL="$2"; shift 2 ;;
        --skywork_batch_size) SKYWORK_BATCH_SIZE="$2"; shift 2 ;;
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
    echo "[shard $shard_index] GPU $gpu_id: python eval_math500.py ${FORWARDED[*]} --num_shards $NUM_SHARDS --shard_index $shard_index"
    CUDA_VISIBLE_DEVICES="$gpu_id" python eval_math500.py "${FORWARDED[@]}" \
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

MERGE_OUT=$(python eval_math500.py "${FORWARDED[@]}" --num_shards "$NUM_SHARDS" --shard_index 0 --merge)
echo "$MERGE_OUT"

if [[ -n "$SKYWORK_MODEL" ]]; then
    MERGED_PATH=$(echo "$MERGE_OUT" | grep -oE '\-> .*\.json$' | sed 's/^-> //')
    if [[ -z "$MERGED_PATH" ]]; then
        echo "Could not parse merged output path from eval_math500.py's own output; skipping Skywork scoring" >&2
        exit 1
    fi
    python score_math500_skywork.py \
        --completions_file "$MERGED_PATH" \
        --skywork_reward_model "$SKYWORK_MODEL" \
        --batch_size "$SKYWORK_BATCH_SIZE"
fi
