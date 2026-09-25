#!/bin/bash -l
# Dream counterpart of eval/scripts/launch_general_benchmarks_inference.sh.
#
# Unlike the upstream RLMT script (one SBATCH job per model x benchmark, --gres=gpu:1, vllm
# tensor-parallel), each job here claims one full dedicated Vista gb node (4x GB200) and fans out
# across all 4 GPUs via run_eval_sharded.sh -- see eval/run_sharded.py.
#
# Rule-based benchmarks (ifeval/math_500/zebra_logic/mmlu_redux_cot/popqa/ifbench) are scored
# immediately. LLM-judge benchmarks (creativewritingv3/alpacaeval2/wildbench/arena_hard_v2) are run
# with --skip_eval here (sampling only, judge-independent) -- score them afterward with
# launch_dream_benchmarks_prompting.sh once these jobs finish and populate the cache.
set -euo pipefail

# Judge-model API keys (OpenAI, Anthropic, ...), if kept outside the shell env -- see
# ~/.config/openai/api_key.env convention. No-op if the file doesn't exist.
[ -f ~/.config/openai/api_key.env ] && source ~/.config/openai/api_key.env

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../eval" && pwd)"

MODELS=(
    # HF repo id or local checkpoint path -- passed to run_benchmarks_sampling.py as dream/$MODEL
    #"Dream-org/Dream-v0-Instruct-7B"
    /scratch/09749/atutej/checkpoints/entrgi_rgrl_clean/dream-rgrl-entrgi-wildchat-1018716/checkpoint-500/
    /scratch/09749/atutej/checkpoints/entrgi_rgrl_clean/dream-grpo-wildchat-1019400/checkpoint-500/
)
BENCHMARKS=(
    #"ifeval"
    #"math_500"
    #"zebra_logic"
    #"mmlu_redux_cot"
    #"popqa"
    #"ifbench"          # now available: spacy/emoji/syllapy + en_core_web_sm installed in dllm_entrgi_gb100
    #"creativewritingv3"
    "alpacaeval2"
    #"wildbench"
    # "arena_hard_v2"
    # "wildbench_newref"
)

# Diffusion sampling cost scales with max_new_tokens/steps directly (unlike AR decoding), so these
# budgets follow dllm/examples/dream/eval.sh's own per-task instruct-mode defaults where it covers
# the same task (ifeval/math_500-via-minerva_math/mmlu_redux_cot-via-mmlu_generative_dream), and a
# judgment-call default elsewhere (popqa/ifbench/the open-ended judge benchmarks, none of which
# dllm's own harness tests) -- deliberately NOT RLMT's flat 4096/8192 AR convention, which would be
# far too slow for diffusion sampling here. Override per-benchmark as needed.
declare -A MAX_TOKENS_FOR=(
    [ifeval]=1280 [math_500]=512 [zebra_logic]=512 [mmlu_redux_cot]=128 [popqa]=128 [ifbench]=1280
    [creativewritingv3]=1024 [alpacaeval2]=1024 [wildbench]=1024 [arena_hard_v2]=1024 [wildbench_newref]=1024
)

ACCOUNT=ASC26088
PARTITION=gb
NUM_GPUS=4
TIME=4:00:00
TEMPERATURE=0.1
TOP_P=0.9
# Measured on a GB200: bs=8 -> 0.59 prompts/s using ~33/189GB, bs=32 -> 0.72 prompts/s (+22%),
# bs=64 -> 0.76 prompts/s (+29%, but far more memory for a shrinking marginal gain). 32 is a good
# default; raise it if a benchmark's prompts are short enough to leave more memory headroom.
DIFFUSION_BATCH_SIZE=16
# Quick-signal runs: set to a row count (e.g. 128) to finish in a fraction of the full-dataset time.
# Uses a distinct OUTPUT_DIR suffix so it never collides with (or gets skipped-as-already-done by) a
# full run's output, and launch_dream_benchmarks_prompting.sh's own TEST_SAMPLE_SIZE must match this
# exactly -- same seed => same --test_sample_size => same sampled subset, so the judging pass's cache
# lookups line up with what this run actually generated.
TEST_SAMPLE_SIZE=128

# For a full checkpoint path, basename alone collides across runs whose leaf dir is a generic
# "checkpoint-N" (GDPO/GRPO/RGRL can all save "checkpoint-500") -- include the immediate parent
# (the run name) too in that case. Plain HF repo ids ("Dream-org/Dream-v0-Instruct-7B") keep the
# old, cleaner basename-only form.
model_saving_name() {
    local leaf; leaf="$(basename "$1")"
    if [[ "$leaf" == checkpoint-* ]]; then
        echo "$(basename "$(dirname "$1")")_${leaf}"
    else
        echo "$leaf"
    fi
}

mkdir -p "${EVAL_DIR}/joblog"

for BENCHMARK in "${BENCHMARKS[@]}"; do
for MODEL in "${MODELS[@]}"; do

MODEL_SAVING_NAME="dream-$(model_saving_name "$MODEL")"
OUTPUT_DIR="outputs/${BENCHMARK}-compare/${MODEL_SAVING_NAME}"
if [ -n "$TEST_SAMPLE_SIZE" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-test${TEST_SAMPLE_SIZE}"
fi

if ls "${EVAL_DIR}/${OUTPUT_DIR}"/*${BENCHMARK}*.json > /dev/null 2>&1; then
    # echo "Skipping ${MODEL} - already has a *${BENCHMARK}*.json file"
    continue
fi
mkdir -p "${EVAL_DIR}/${OUTPUT_DIR}"

MAX_TOKENS="${MAX_TOKENS_FOR[$BENCHMARK]:-512}"

EXTRA_ARGS="--force_overwrite"
if [ -n "$TEST_SAMPLE_SIZE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --test_sample_size ${TEST_SAMPLE_SIZE}"
fi
case "$BENCHMARK" in
    ifeval|math_500|zebra_logic|mmlu_redux_cot|popqa|ifbench) ;; # rule-based, evaluate immediately
    *) EXTRA_ARGS="$EXTRA_ARGS --skip_eval" ;;                    # needs a judge, see *_prompting.sh
esac
case "$BENCHMARK" in
    wildbench|wildbench_newref|arena_hard_v2|creativewritingv3|alpacaeval2)
        EXTRA_ARGS="$EXTRA_ARGS --parallel_eval" ;;
esac

JOB_NAME="dream-${BENCHMARK}-${MODEL_SAVING_NAME}"
#if squeue -h --me -n "${JOB_NAME}" | grep -q .; then
#    echo "!!! skipping ${JOB_NAME} - already running"
#    continue
#fi
echo "!!! submitting ${JOB_NAME}"

sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${EVAL_DIR}/joblog/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI_gb


${EVAL_DIR}/run_eval_sharded.sh \
    --num_gpus ${NUM_GPUS} \
    --benchmark ${BENCHMARK} \
    --model dream/${MODEL} \
    --max_tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --diffusion_batch_size ${DIFFUSION_BATCH_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    ${EXTRA_ARGS}
EOT

done
done
