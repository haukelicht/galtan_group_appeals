#!/bin/bash
# vLLM Server Utilities
# Provides helper functions for managing vLLM server lifecycle

# Global variable to store vLLM process ID
VLLM_PID=""

# Cleanup function for vLLM server
cleanup_vllm() {
    if [ ! -z "${VLLM_PID}" ]; then
        echo ""
        echo "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill ${VLLM_PID} 2>/dev/null || true
        wait ${VLLM_PID} 2>/dev/null || true
        echo "✓ vLLM server stopped"
    fi
}

# Function to wait for vLLM server to be ready
# Args:
#   $1: API URL (default: http://localhost:8000)
#   $2: Max retries (default: 60)
#   $3: Retry interval in seconds (default: 10)
wait_for_vllm() {
    local API_URL="${1:-http://localhost:8000}"
    local MAX_RETRIES="${2:-60}"
    local RETRY_INTERVAL="${3:-10}"
    
    echo "Waiting for vLLM server at ${API_URL} to be ready..."
    local RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s "${API_URL}/models" > /dev/null 2>&1; then
            echo "✓ vLLM server is ready"
            return 0
        fi
        echo -n "."
        sleep ${RETRY_INTERVAL}
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
    
    echo ""
    echo "ERROR: vLLM server failed to start after ${MAX_RETRIES} attempts"
    return 1
}

# Function to start vLLM server
# Args:
#   $1: Model name (required)
#   $2: Host (default: localhost)
#   $3: Port (default: 8000)
#   $4: Log file path (default: vllm.log)
start_vllm() {
    local MODEL="$1"
    local HOST="${2:-localhost}"
    local PORT="${3:-8000}"
    local LOG_FILE="${4:-vllm.log}"
    
    if [ -z "$MODEL" ]; then
        echo "ERROR: Model name is required"
        return 1
    fi
    
    echo "Starting vLLM server for ${MODEL}..."
    
    vllm serve "${MODEL}" \
        --host "${HOST}" \
        --port ${PORT} \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.92 \
        --max_model_len=10000 \
        > "${LOG_FILE}" 2>&1 &
    
    VLLM_PID=$!
    echo "✓ vLLM server started (PID: ${VLLM_PID})"
    echo "  Logs: ${LOG_FILE}"
    
    return 0
}
