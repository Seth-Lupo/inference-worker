#!/bin/bash
# =============================================================================
# Build TensorRT Engines Script
# Converts models to TensorRT format for optimized inference
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_SOURCES="${DEPLOY_DIR}/model_sources"
MODEL_REPO="${DEPLOY_DIR}/model_repository"
ENGINE_CACHE="${DEPLOY_DIR}/engine_cache"

echo "=============================================="
echo "Building TensorRT Engines"
echo "=============================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Load environment
if [ -f "${DEPLOY_DIR}/.env" ]; then
    source "${DEPLOY_DIR}/.env"
fi

# Check for GPU
if ! nvidia-smi &> /dev/null; then
    log_error "GPU not detected. TensorRT engine building requires a GPU."
    exit 1
fi

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
log_info "Building engines for: ${GPU_NAME} (${GPU_MEM})"

mkdir -p "${ENGINE_CACHE}"

# -----------------------------------------------------------------------------
# Build Qwen3 8B TensorRT-LLM Engine
# -----------------------------------------------------------------------------
build_qwen3_engine() {
    log_info "Building Qwen3 8B TensorRT-LLM engine..."

    QWEN_SOURCE="${MODEL_SOURCES}/qwen3_8b/Qwen3-8B"
    QWEN_ENGINE="${MODEL_REPO}/qwen3_8b/1/engine"

    if [ ! -d "$QWEN_SOURCE" ]; then
        log_warn "Qwen3 source model not found at ${QWEN_SOURCE}"
        log_warn "Run ./scripts/download_models.sh first"
        return 1
    fi

    # Check if engine already exists
    if [ -f "${QWEN_ENGINE}/rank0.engine" ]; then
        log_info "Qwen3 engine already exists. Skipping build."
        log_info "Delete ${QWEN_ENGINE} to rebuild."
        return 0
    fi

    mkdir -p "${QWEN_ENGINE}"

    # Run in Triton container with TensorRT-LLM
    docker run --rm --gpus all \
        --shm-size=8g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${MODEL_SOURCES}:/model_sources" \
        -v "${QWEN_ENGINE}:/engine_output" \
        -v "${ENGINE_CACHE}:/engine_cache" \
        -e HF_TOKEN="${HF_TOKEN}" \
        nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 \
        bash -c "
            cd /app/tensorrt_llm/examples/qwen

            # Convert checkpoint
            python convert_checkpoint.py \
                --model_dir /model_sources/qwen3_8b/Qwen3-8B \
                --output_dir /engine_cache/qwen3_ckpt \
                --dtype float16 \
                --use_weight_only \
                --weight_only_precision int8

            # Build engine
            trtllm-build \
                --checkpoint_dir /engine_cache/qwen3_ckpt \
                --output_dir /engine_output \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 4096 \
                --max_seq_len 8192 \
                --paged_kv_cache enable \
                --use_fused_mlp enable
        "

    log_info "Qwen3 8B engine built successfully"
}

# -----------------------------------------------------------------------------
# Build CosyVoice 2 TensorRT Engine (Flow Decoder)
# -----------------------------------------------------------------------------
build_cosyvoice_engine() {
    log_info "Building CosyVoice 2 TensorRT engine..."

    COSYVOICE_SOURCE="${MODEL_SOURCES}/cosyvoice2/CosyVoice2-0.5B"
    COSYVOICE_ENGINE="${MODEL_REPO}/cosyvoice2/1"

    if [ ! -d "$COSYVOICE_SOURCE" ]; then
        log_warn "CosyVoice source model not found at ${COSYVOICE_SOURCE}"
        return 1
    fi

    # Check if flow decoder ONNX exists
    FLOW_ONNX="${COSYVOICE_SOURCE}/flow.decoder.estimator.fp32.onnx"
    if [ ! -f "$FLOW_ONNX" ]; then
        log_warn "CosyVoice flow decoder ONNX not found"
        log_info "The model may need to be exported to ONNX first"
        return 1
    fi

    # Build TensorRT engine for flow decoder
    docker run --rm --gpus all \
        -v "${COSYVOICE_SOURCE}:/model" \
        -v "${COSYVOICE_ENGINE}:/output" \
        nvcr.io/nvidia/tensorrt:24.12-py3 \
        trtexec \
            --onnx=/model/flow.decoder.estimator.fp32.onnx \
            --saveEngine=/output/flow.decoder.estimator.plan \
            --fp16 \
            --minShapes=x:1x80x1,mask:1x1x1,mu:1x80x1,t:1,spks:1x80,cond:1x80x1 \
            --optShapes=x:1x80x100,mask:1x1x100,mu:1x80x100,t:1,spks:1x80,cond:1x80x100 \
            --maxShapes=x:1x80x1000,mask:1x1x1000,mu:1x80x1000,t:1,spks:1x80,cond:1x80x1000

    log_info "CosyVoice 2 TensorRT engine built"
}

# -----------------------------------------------------------------------------
# Convert Parakeet to TensorRT (Optional - ONNX is usually sufficient)
# -----------------------------------------------------------------------------
build_parakeet_engine() {
    log_info "Building Parakeet TDT TensorRT engine..."

    PARAKEET_SOURCE="${MODEL_SOURCES}/parakeet_tdt"
    PARAKEET_ENGINE="${MODEL_REPO}/parakeet_tdt/1"

    if [ ! -f "${PARAKEET_SOURCE}/encoder.int8.onnx" ]; then
        log_warn "Parakeet ONNX model not found"
        return 1
    fi

    # For ASR, ONNX runtime is often sufficient
    # TensorRT conversion is optional and may have shape issues with transducers
    log_info "Copying Parakeet ONNX models (TensorRT conversion optional)"

    mkdir -p "${PARAKEET_ENGINE}"
    cp "${PARAKEET_SOURCE}"/*.onnx "${PARAKEET_ENGINE}/" 2>/dev/null || true
    cp "${PARAKEET_SOURCE}/tokens.txt" "${PARAKEET_ENGINE}/" 2>/dev/null || true

    log_info "Parakeet models copied to model repository"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log_info "Starting engine builds..."

    # Parse arguments
    BUILD_ALL=true
    for arg in "$@"; do
        case $arg in
            --qwen)
                BUILD_ALL=false
                build_qwen3_engine
                ;;
            --cosyvoice)
                BUILD_ALL=false
                build_cosyvoice_engine
                ;;
            --parakeet)
                BUILD_ALL=false
                build_parakeet_engine
                ;;
            --help)
                echo "Usage: $0 [--qwen] [--cosyvoice] [--parakeet]"
                echo "  --qwen       Build only Qwen3 8B engine"
                echo "  --cosyvoice  Build only CosyVoice 2 engine"
                echo "  --parakeet   Build only Parakeet TDT engine"
                echo "  (no args)    Build all engines"
                exit 0
                ;;
        esac
    done

    if $BUILD_ALL; then
        build_parakeet_engine || log_warn "Parakeet build failed or skipped"
        build_qwen3_engine || log_warn "Qwen3 build failed or skipped"
        build_cosyvoice_engine || log_warn "CosyVoice build failed or skipped"
    fi

    echo ""
    echo "=============================================="
    echo -e "${GREEN}Engine Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Model repository contents:"
    find "${MODEL_REPO}" -name "*.engine" -o -name "*.plan" -o -name "*.onnx" 2>/dev/null | head -20
    echo ""
    echo "Start services with: docker-compose up -d"
}

main "$@"
