#!/bin/bash
# Experiment: 10-shot similarity-based classification across multiple folds
# Models: qwen3:32b, deepseek-r1:32b, gemma3:27b
#
# This script runs few-shot classification using 10 exemplars from the training set
# with similarity-based retrieval on the validation set for each fold.

set -e  # Exit on error

# Cleanup function for vLLM server
# Source vLLM utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/vllm_utils.sh"

# Graceful handling of keyboard interrupt and cleanup
trap 'echo ""; echo "Experiment interrupted by user. Exiting..."; cleanup_vllm; exit 130' INT EXIT

# Define paths
PROJECT="/home/hauke-licht/Dropbox/papers/galtan_group_appeals"

# Define folds to process
FOLDS=("fold01" "fold02" "fold03" "fold04" "fold05")

# Set PYTHONPATH to include the src directory
export PYTHONPATH="${PROJECT}/src:${PYTHONPATH}"

# Check if the correct conda environment is active
REQUIRED_ENV="icl_group_attribute_classification"
CURRENT_ENV=$(conda info --envs | grep '\*' | awk '{print $1}')

if [ "$CURRENT_ENV" != "$REQUIRED_ENV" ]; then
    echo "ERROR: Conda environment '${REQUIRED_ENV}' is not active."
    echo "Current environment: ${CURRENT_ENV}"
    echo ""
    echo "Please activate the correct environment with:"
    echo "  conda activate ${REQUIRED_ENV}"
    exit 1
fi

echo "✓ Using conda environment: ${CURRENT_ENV}"
echo ""


# Define models to test
# MODELS=(
#     "qwen3:32b"
#     "deepseek-r1:32b"
#     "gemma3:27b"
# )
# MODELS=(
#     "qwen3-next:80b"
#     "magistral:24b"
#     # "gpt-oss:20b"
# )
# MODELS=(
#     "cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"
# )
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  ✅
    # "google/gemma-3-27b-it" ✅
    # "unsloth/gpt-oss-120b-unsloth-bnb-4bit" ❌
    # "cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit" ✅
MODELS=(
    "cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "google/gemma-3-27b-it"
)


# Configuration
DATA_PATH="${PROJECT}/data/annotations/group_mention_categorization/splits/model_selection"
N_EXEMPLARS=10
SAMPLING_STRATEGY="similarity"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
TEMPERATURE=0.0
SEED=42

echo "=============================================="
echo "10-Shot Classification Experiment - Multi-Fold"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Folds: ${FOLDS[@]}"
echo "  N exemplars: ${N_EXEMPLARS}"
echo "  Sampling strategy: ${SAMPLING_STRATEGY}"
echo "  Embedding model: ${EMBEDDING_MODEL}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Seed: ${SEED}"
echo ""
echo "Models to run: ${MODELS[@]}"
echo ""
echo "=============================================="
echo ""

# vLLM server configuration
VLLM_HOST="localhost"
VLLM_PORT=8000
VLLM_API_URL="http://${VLLM_HOST}:${VLLM_PORT}"

OVERWRITE=true # Set to true to overwrite existing outputs without prompt

# Run classification for each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "================================================"
    echo "Running experiment with model: ${MODEL}"
    echo "================================================"
    echo ""
    
    # Create safe filename from model name (replace : and / with _)
    SAFE_MODEL_NAME=$(echo "${MODEL}" | tr '/' '__')
    
    # Start vLLM server once per model
    VLLM_LOG="logs/vllm.log"
    start_vllm "${MODEL}" "${VLLM_HOST}" "${VLLM_PORT}" "${VLLM_LOG}"
    
    # Wait for server to be ready
    if ! wait_for_vllm "${VLLM_API_URL}"; then
        echo "ERROR: Failed to start vLLM server"
        cleanup_vllm
        exit 1
    fi
    echo ""
    
    # Process each fold with this model
    for FOLD in "${FOLDS[@]}"; do
        echo ""
        echo "  ----------------------------------------"
        echo "  Processing ${FOLD} with ${MODEL}"
        echo "  ----------------------------------------"
        echo ""
        
        # Set fold-specific paths
        DATA_DIR="${DATA_PATH}/${FOLD}"
        TRAIN_DATA="${DATA_DIR}/train.pkl"
        VAL_DATA="${DATA_DIR}/val.pkl"
        OUTPUT_DIR="${PROJECT}/experiments/results/10shot/${FOLD}"
        
        # Create output directory if it doesn't exist
        mkdir -p "${OUTPUT_DIR}"
        
        OUTPUT_FILE="${OUTPUT_DIR}/predictions_${SAFE_MODEL_NAME}.pkl"
        
        # Check if output already exists
        if [ -f "${OUTPUT_FILE}" ]; then
            if [ "$OVERWRITE" != true ]; then
                echo "⚠️  Output file already exists: ${OUTPUT_FILE}"
                read -p "Do you want to overwrite it? (y/n): " REPLY
            else
                REPLY="y"
            fi
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Skipping ${FOLD} for ${MODEL}..."
                continue
            fi
        fi
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run classifier
    python -m icl.classifier \
        --input "${VAL_DATA}" \
        --output "${OUTPUT_FILE}" \
        --overwrite \
        --max-errors 0.2 \
        --llm-backend vllm \
        --model-name "${MODEL}" \
        --n-exemplars ${N_EXEMPLARS} \
        --exemplars-file "${TRAIN_DATA}" \
        --exemplars-sampling-strategy ${SAMPLING_STRATEGY} \
        --exemplars-embedding-model ${EMBEDDING_MODEL} \
        --temperature ${TEMPERATURE} \
        --seed ${SEED} \
        --show-progress
        # --test 10 \
    
    # Record end time and calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
        echo ""
        echo "  ✓ Completed ${FOLD} in ${MINUTES}m ${SECONDS}s"
        echo "    Results saved to: ${OUTPUT_FILE}"
        echo ""
    done
    
    # Stop vLLM server after processing all folds for this model
    cleanup_vllm
    VLLM_PID=""
    
    echo ""
    echo "✓ Completed all folds for ${MODEL}"
    echo ""
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results saved to: ${PROJECT}/experiments/results/10shot/"
