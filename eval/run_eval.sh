#!/bin/bash
# Entrypoint for benchmarking Dream/LLaDA checkpoints on the RLMT reason_benchmarks suite
# (vendored from https://github.com/princeton-pli/RLMT/tree/main/eval), sampling via dllm's own
# DreamSampler / MDLMSampler instead of vllm. Pass model as "dream/<hf_repo_or_path>" or
# "llada/<hf_repo_or_path>" -- see llm_utils.py's _batch_dream_query/_batch_llada_query.
#
# Usage: ./run_eval.sh --benchmark math_500 --model dream/Dream-org/Dream-v0-Instruct-7B \
#            --max_tokens 512 --temperature 0.0
# Any extra args are forwarded verbatim to run_benchmarks_sampling.py.
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
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p caches

python run_benchmarks_sampling.py "$@"
