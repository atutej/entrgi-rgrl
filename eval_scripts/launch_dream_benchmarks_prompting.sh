#!/bin/bash -l
# Dream counterpart of eval/scripts/launch_general_benchmarks_prompting.sh.
#
# NOTE: unlike upstream RLMT's own version of this script, this one DOES go through run_sharded.py
# (matching launch_dream_benchmarks_inference.sh's NUM_SHARDS) -- not an oversight, a fix: the
# inference launcher generates via run_eval_sharded.sh, which writes N per-shard sqlite cache files
# (e.g. ..._shard0of4.sqlite), so an unsharded run_benchmarks_sampling.py call here would look for a
# single unsharded cache file that never existed, miss every entry, and silently fall back to (GPU)
# regeneration -- defeating the point of a GPU-free judging pass. Sharding to match means every
# prompt hits the cache and dllm's model is never loaded; run_sharded.py's CUDA_VISIBLE_DEVICES
# assignment is harmless on a GPU-less node too, since cache hits never touch CUDA at all. Needs an
# OPENAI_API_KEY (or whichever judge each benchmark's evaluate() calls) in the environment.
set -euo pipefail

# Judge-model API keys (OpenAI, Anthropic, ...), if kept outside the shell env -- see
# ~/.config/openai/api_key.env convention. No-op if the file doesn't exist.
[ -f ~/.config/openai/api_key.env ] && source ~/.config/openai/api_key.env

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../eval" && pwd)"
cd "$EVAL_DIR"

MODELS=(
    /scratch/09749/atutej/checkpoints/entrgi_rgrl_clean/dream-rgrl-entrgi-wildchat-1018716/checkpoint-500
    /scratch/09749/atutej/checkpoints/entrgi_rgrl_clean/dream-grpo-wildchat-1019400/checkpoint-500
)
BENCHMARKS=(
    #"creativewritingv3"
    "alpacaeval2"
    #"wildbench"
    # "arena_hard_v2"
    # "wildbench_newref"
)

# Must match launch_dream_benchmarks_inference.sh's MAX_TOKENS_FOR / TEMPERATURE / TOP_P / NUM_GPUS
# / TEST_SAMPLE_SIZE exactly, or the cache key (or the sampled subset, for TEST_SAMPLE_SIZE) won't
# match and this will silently fall back to (GPU) regeneration.
declare -A MAX_TOKENS_FOR=(
    [creativewritingv3]=1024 [alpacaeval2]=1024 [wildbench]=1024 [arena_hard_v2]=1024 [wildbench_newref]=1024
)
TEMPERATURE=0.1
TOP_P=0.9
NUM_SHARDS=4
# Quick-signal runs: set to a row count (e.g. 128) to match an inference run that used the same
# --test_sample_size; leave empty ("") for a full run.
TEST_SAMPLE_SIZE=128

# Must match launch_dream_benchmarks_inference.sh's model_saving_name() logic exactly, or the
# OUTPUT_DIR computed here won't match where that script actually wrote its output.
model_saving_name() {
    local leaf; leaf="$(basename "$1")"
    if [[ "$leaf" == checkpoint-* ]]; then
        echo "$(basename "$(dirname "$1")")_${leaf}"
    else
        echo "$leaf"
    fi
}

for BENCHMARK in "${BENCHMARKS[@]}"; do
for MODEL in "${MODELS[@]}"; do

MODEL_SAVING_NAME="dream-$(model_saving_name "$MODEL")"
OUTPUT_DIR="outputs/${BENCHMARK}-compare/${MODEL_SAVING_NAME}"
if [ -n "$TEST_SAMPLE_SIZE" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-test${TEST_SAMPLE_SIZE}"
fi

if ls "${OUTPUT_DIR}"/*${BENCHMARK}*.json > /dev/null 2>&1; then
    if ls "${OUTPUT_DIR}"/*${BENCHMARK}*.json.score > /dev/null 2>&1; then
        continue
    fi
else
    echo "!!! Could not find ${OUTPUT_DIR}/*${BENCHMARK}*.json -- run launch_dream_benchmarks_inference.sh first"
    continue
fi

MAX_TOKENS="${MAX_TOKENS_FOR[$BENCHMARK]:-1024}"
EXTRA_ARGS="--parallel_eval"
if [ -n "$TEST_SAMPLE_SIZE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --test_sample_size ${TEST_SAMPLE_SIZE}"
fi

CMD="${EVAL_DIR}/run_eval_sharded.sh \
    --num_gpus ${NUM_SHARDS} \
    --benchmark ${BENCHMARK} \
    --model dream/${MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    ${EXTRA_ARGS}"

echo "$CMD"
eval "$CMD"

done
done
