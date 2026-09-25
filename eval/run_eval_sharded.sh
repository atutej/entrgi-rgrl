#!/bin/bash
# Multi-GPU dataset-agnostic wrapper: same env setup as run_eval.sh, but delegates to
# run_sharded.py, which round-robin-splits whatever --benchmark's data across --num_gpus
# GPU-pinned subprocesses of run_benchmarks_sampling.py and merges the results.
#
# Usage: ./run_eval_sharded.sh --num_gpus 4 --benchmark ifeval \
#            --model dream/Dream-org/Dream-v0-Instruct-7B \
#            --max_tokens 1280 --temperature 0.1 --top_p 0.9 --force_overwrite
# Every flag other than --num_gpus/--gpu_ids is forwarded to run_benchmarks_sampling.py.
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

python run_sharded.py "$@"
