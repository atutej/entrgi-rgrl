#!/bin/bash
# EntRGi: Entropy Aware Reward Guidance for Diffusion Language Models
# This script runs all methods (EntRGi, APS, Expectation, BoN) and evaluates with LMUnit

set -e  # Exit on error

# =====================
# Configuration
# =====================

# Output directories
RESULTS_DIR="./results"
LMUNIT_DIR="${RESULTS_DIR}/lmunit_results"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LMUNIT_DIR"

# Experiment settings
SEEDS=(1 2 3)
K_VALUES=(4)
TEMPERATURES=(0.7)

# Datasets configuration: (path, split, prompt_field, prefix)
DATASETS=(
    "THU-KEG/RM-Bench:train:prompt:rm-bench"
    "ScalerLab/JudgeBench:gpt:question:judgebench"
    "allenai/reward-bench-2:test:prompt:reward-bench-2"
)

# Reward models configuration: (model_path, short_name)
REWARD_MODELS=(
    "Skywork/Skywork-Reward-V2-Qwen3-1.7B:skywork-1.7b"
)

# Model settings
T=128                # Diffusion steps
M=3                  # Gradient optimization steps
ETA=0.5              # Learning rate
MAX_NEW_TOKENS=128   # Max generation length
SUBSET_SIZE=64      # Number of prompts per dataset
BATCH_SIZE=4         # Batch size per GPU
NUM_GPUS=4           # Number of GPUs

# =====================
# Main experiment loop
# =====================

for TEMP in "${TEMPERATURES[@]}"; do
    echo "========================================"
    echo "Running experiments with temperature=${TEMP}"
    echo "========================================"

    for K in "${K_VALUES[@]}"; do
        echo "Running with K=${K}..."

        for DATASET_CONFIG in "${DATASETS[@]}"; do
            # Parse dataset config
            IFS=':' read -r DATASET SPLIT PROMPT_FIELD PREFIX <<< "$DATASET_CONFIG"
            
            echo "Dataset: ${DATASET} (${PREFIX})"

            for RM_CONFIG in "${REWARD_MODELS[@]}"; do
                # Parse reward model config
                RM_PATH="${RM_CONFIG%%:*}"
                RM_NAME="${RM_CONFIG##*:}"
                
                echo "  Reward model: ${RM_NAME}"

                for SEED in "${SEEDS[@]}"; do
                    echo "    Seed: ${SEED}"

                    # =====================
                    # Define output file paths
                    # =====================
                    
                    # Gradient-based methods (entrgi.py)
                    FILE_ENTRGI="${RESULTS_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_entrgi_seed${SEED}.json"
                    FILE_APS="${RESULTS_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_aps_seed${SEED}.json"
                    FILE_EXPECTATION="${RESULTS_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_expectation_seed${SEED}.json"
                    
                    # Best-of-N baseline (bon.py)
                    FILE_BON="${RESULTS_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_bon_seed${SEED}.json"

                    # LMUnit evaluation outputs
                    EVAL_ENTRGI="${LMUNIT_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_entrgi_seed${SEED}_eval.json"
                    EVAL_APS="${LMUNIT_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_aps_seed${SEED}_eval.json"
                    EVAL_EXPECTATION="${LMUNIT_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_M${M}_expectation_seed${SEED}_eval.json"
                    EVAL_BON="${LMUNIT_DIR}/${PREFIX}_${RM_NAME}_k${K}_temp${TEMP}_T${T}_bon_seed${SEED}_eval.json"

                    # =====================
                    # Run EntRGi (Ours)
                    # =====================
                    
                    if [ ! -f "$FILE_ENTRGI" ]; then
                        echo "      Running EntRGi..."
                        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 entrgi.py \
                            --K ${K} \
                            --M ${M} \
                            --eta ${ETA} \
                            --T ${T} \
                            --max_new_tokens ${MAX_NEW_TOKENS} \
                            --dataset_path "$DATASET" \
                            --split "$SPLIT" \
                            --prompt_field "$PROMPT_FIELD" \
                            --subset_size ${SUBSET_SIZE} \
                            --batch_size ${BATCH_SIZE} \
                            --alg entropy \
                            --seed "$SEED" \
                            --temperature "$TEMP" \
                            --use_entrgi \
                            --reward_model "$RM_PATH" \
                            --output_file "$FILE_ENTRGI"
                    fi

                    # =====================
                    # Run APS (Rout et al. 2025)
                    # =====================
                    
                    if [ ! -f "$FILE_APS" ]; then
                        echo "      Running APS..."
                        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29501 entrgi.py \
                            --K ${K} \
                            --M ${M} \
                            --eta ${ETA} \
                            --T ${T} \
                            --max_new_tokens ${MAX_NEW_TOKENS} \
                            --dataset_path "$DATASET" \
                            --split "$SPLIT" \
                            --prompt_field "$PROMPT_FIELD" \
                            --subset_size ${SUBSET_SIZE} \
                            --batch_size ${BATCH_SIZE} \
                            --alg entropy \
                            --seed "$SEED" \
                            --temperature "$TEMP" \
                            --use_aps \
                            --reward_model "$RM_PATH" \
                            --output_file "$FILE_APS"
                    fi

                    # =====================
                    # Run Expectation (Continuous Relaxation)
                    # =====================
                    
                    if [ ! -f "$FILE_EXPECTATION" ]; then
                        echo "      Running Expectation..."
                        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29502 entrgi.py \
                            --K ${K} \
                            --M ${M} \
                            --eta ${ETA} \
                            --T ${T} \
                            --max_new_tokens ${MAX_NEW_TOKENS} \
                            --dataset_path "$DATASET" \
                            --split "$SPLIT" \
                            --prompt_field "$PROMPT_FIELD" \
                            --subset_size ${SUBSET_SIZE} \
                            --batch_size ${BATCH_SIZE} \
                            --alg entropy \
                            --seed "$SEED" \
                            --temperature "$TEMP" \
                            --reward_model "$RM_PATH" \
                            --output_file "$FILE_EXPECTATION"
                    fi

                    # =====================
                    # Run Best-of-N (BoN) Baseline
                    # =====================
                    
                    if [ ! -f "$FILE_BON" ]; then
                        echo "      Running BoN..."
                        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NUM_GPUS} --master_port=29503 bon.py \
                            --K ${K} \
                            --T ${T} \
                            --max_new_tokens ${MAX_NEW_TOKENS} \
                            --dataset_path "$DATASET" \
                            --split "$SPLIT" \
                            --prompt_field "$PROMPT_FIELD" \
                            --subset_size ${SUBSET_SIZE} \
                            --batch_size ${BATCH_SIZE} \
                            --alg entropy \
                            --seed "$SEED" \
                            --temperature "$TEMP" \
                            --reward_model "$RM_PATH" \
                            --output_file "$FILE_BON"
                    fi

                    # =====================
                    # LMUnit Evaluation
                    # =====================

                    # Evaluate EntRGi
                    if [ -f "$FILE_ENTRGI" ] && [ ! -f "$EVAL_ENTRGI" ]; then
                        echo "      Evaluating EntRGi with LMUnit..."
                        python lmunit_eval.py \
                            --file "$FILE_ENTRGI" \
                            --output "$EVAL_ENTRGI" \
                            --model ContextualAI/LMUnit-qwen2.5-72b \
                            --tp_size ${NUM_GPUS}
                    fi

                    # Evaluate APS
                    if [ -f "$FILE_APS" ] && [ ! -f "$EVAL_APS" ]; then
                        echo "      Evaluating APS with LMUnit..."
                        python lmunit_eval.py \
                            --file "$FILE_APS" \
                            --output "$EVAL_APS" \
                            --model ContextualAI/LMUnit-qwen2.5-72b \
                            --tp_size ${NUM_GPUS}
                    fi

                    # Evaluate Expectation
                    if [ -f "$FILE_EXPECTATION" ] && [ ! -f "$EVAL_EXPECTATION" ]; then
                        echo "      Evaluating Expectation with LMUnit..."
                        python lmunit_eval.py \
                            --file "$FILE_EXPECTATION" \
                            --output "$EVAL_EXPECTATION" \
                            --model ContextualAI/LMUnit-qwen2.5-72b \
                            --tp_size ${NUM_GPUS}
                    fi

                    # Evaluate BoN
                    if [ -f "$FILE_BON" ] && [ ! -f "$EVAL_BON" ]; then
                        echo "      Evaluating BoN with LMUnit..."
                        python lmunit_eval.py \
                            --file "$FILE_BON" \
                            --output "$EVAL_BON" \
                            --model ContextualAI/LMUnit-qwen2.5-72b \
                            --tp_size ${NUM_GPUS}
                    fi

                done  # SEED
            done  # RM_CONFIG
        done  # DATASET_CONFIG
    done  # K
done  # TEMP

echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "LMUnit evaluations saved to: ${LMUNIT_DIR}"
echo ""
echo "To aggregate results, run:"
echo "  python aggregate_results.py ${RESULTS_DIR}"
