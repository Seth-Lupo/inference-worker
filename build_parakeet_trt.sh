#!/bin/bash
# =============================================================================
# Build Parakeet TDT TensorRT Engines
#
# Converts ONNX models to TensorRT for native GPU inference.
#
# Usage:
#   ./build_parakeet_trt.sh
#
# Requires: Triton container running (uses trtexec inside container)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Paths
DEPLOY_DIR="$(get_deploy_dir)"
MODEL_DIR="${DEPLOY_DIR}/model_repository/parakeet_tdt/1"
ENGINE_DIR="${MODEL_DIR}/engines"
CONTAINER_NAME="voice-agent-triton"

# TensorRT settings
TRT_PRECISION="${TRT_PRECISION:-fp16}"
TRT_WORKSPACE="${TRT_WORKSPACE:-4096}"  # MB

echo "=============================================="
echo "Building Parakeet TensorRT Engines"
echo "=============================================="
echo "Model dir: ${MODEL_DIR}"
echo "Precision: ${TRT_PRECISION}"
echo ""

# Check ONNX models exist
for model in encoder decoder joiner; do
    if [[ ! -f "${MODEL_DIR}/${model}.onnx" ]]; then
        log_error "Missing ${model}.onnx in ${MODEL_DIR}"
        exit 1
    fi
done

# Create engine directory
mkdir -p "${ENGINE_DIR}"

# Function to build TRT engine
build_engine() {
    local model_name="$1"
    local onnx_path="/models/parakeet_tdt/1/${model_name}.onnx"
    local engine_path="/models/parakeet_tdt/1/engines/${model_name}.plan"

    log_step "Building ${model_name} TensorRT engine..."

    # Check if engine already exists
    if [[ -f "${MODEL_DIR}/engines/${model_name}.plan" ]]; then
        log_info "${model_name}.plan already exists, skipping"
        return 0
    fi

    # Build with trtexec inside container
    docker exec "${CONTAINER_NAME}" trtexec \
        --onnx="${onnx_path}" \
        --saveEngine="${engine_path}" \
        --${TRT_PRECISION} \
        --workspace="${TRT_WORKSPACE}" \
        --verbose 2>&1 | tee "${MODEL_DIR}/engines/${model_name}_build.log"

    if [[ -f "${MODEL_DIR}/engines/${model_name}.plan" ]]; then
        log_info "${model_name}.plan built successfully"
    else
        log_error "Failed to build ${model_name}.plan"
        return 1
    fi
}

# Check container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_error "Container ${CONTAINER_NAME} not running. Start with: docker compose up -d triton"
    exit 1
fi

# Build each model
build_engine "encoder"
build_engine "decoder"
build_engine "joiner"

echo ""
echo "=============================================="
log_info "TensorRT engines built successfully!"
echo "=============================================="
echo ""
echo "Engines:"
ls -lh "${ENGINE_DIR}"/*.plan 2>/dev/null || echo "  (none found)"
echo ""
echo "Next: Update model.py to use TensorRT engines"
