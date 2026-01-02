#!/bin/bash
# =============================================================================
# Build Qwen3 8B TensorRT-LLM Engine
# Based on: https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md
#           https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR="${DEPLOY_DIR}/qwen3_build"
MODEL_REPO="${DEPLOY_DIR}/model_repository"

# Colors
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

# Configuration
MODEL_NAME="Qwen/Qwen3-8B"
QUANTIZATION="${1:-int8_sq}"  # Options: fp8, int8_sq, int4_awq, none

echo "=============================================="
echo "Building Qwen3 8B TensorRT-LLM Engine"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Quantization: ${QUANTIZATION}"
echo ""

mkdir -p "${WORK_DIR}"

# =============================================================================
# Check GPU Capability
# =============================================================================
log_info "Checking GPU capability..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log_info "GPU: ${GPU_NAME}"

# FP8 requires Hopper (H100) or Ada (L4, L40)
if [[ "$QUANTIZATION" == "fp8" ]]; then
    if [[ ! "$GPU_NAME" =~ (H100|H200|L4|L40|RTX\ 40) ]]; then
        log_warn "FP8 quantization works best on Hopper/Ada GPUs"
        log_warn "For older GPUs (A100, T4, V100), consider using int8_sq or int4_awq"
    fi
fi

# =============================================================================
# Download Model
# =============================================================================
download_model() {
    log_info "Downloading Qwen3-8B from HuggingFace..."

    if [ ! -d "${WORK_DIR}/Qwen3-8B" ]; then
        if [ -n "$HF_TOKEN" ]; then
            huggingface-cli download ${MODEL_NAME} \
                --local-dir "${WORK_DIR}/Qwen3-8B" \
                --token "$HF_TOKEN"
        else
            huggingface-cli download ${MODEL_NAME} \
                --local-dir "${WORK_DIR}/Qwen3-8B"
        fi
        log_info "Model downloaded to ${WORK_DIR}/Qwen3-8B"
    else
        log_info "Model already exists at ${WORK_DIR}/Qwen3-8B"
    fi
}

# =============================================================================
# Quantize and Build Engine
# =============================================================================
build_engine() {
    log_info "Building TensorRT-LLM engine with ${QUANTIZATION} quantization..."

    CHECKPOINT_DIR="${WORK_DIR}/checkpoint_${QUANTIZATION}"
    ENGINE_DIR="${WORK_DIR}/engine_${QUANTIZATION}"

    # Check if engine already exists
    if [ -d "${ENGINE_DIR}" ] && [ -f "${ENGINE_DIR}/rank0.engine" ]; then
        log_info "Engine already exists at ${ENGINE_DIR}"
        log_info "Delete the directory to rebuild"
        return 0
    fi

    # Run quantization and build inside TensorRT-LLM container
    docker run --rm --gpus all \
        --shm-size=16g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${WORK_DIR}:/workspace/build" \
        -w /workspace \
        -e HF_TOKEN="${HF_TOKEN}" \
        nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 \
        bash -c "
            set -e
            cd /app/tensorrt_llm/examples/quantization

            MODEL_DIR=/workspace/build/Qwen3-8B
            CHECKPOINT_DIR=/workspace/build/checkpoint_${QUANTIZATION}
            ENGINE_DIR=/workspace/build/engine_${QUANTIZATION}

            echo '=== Step 1: Quantizing model ==='

            if [ '${QUANTIZATION}' == 'none' ]; then
                # No quantization - just convert checkpoint
                echo 'Converting without quantization (FP16)...'
                python quantize.py \
                    --model_dir \$MODEL_DIR \
                    --output_dir \$CHECKPOINT_DIR \
                    --dtype float16

            elif [ '${QUANTIZATION}' == 'fp8' ]; then
                # FP8 quantization (best for Hopper/Ada GPUs)
                echo 'Quantizing to FP8...'
                python quantize.py \
                    --model_dir \$MODEL_DIR \
                    --qformat fp8 \
                    --kv_cache_dtype fp8 \
                    --output_dir \$CHECKPOINT_DIR

            elif [ '${QUANTIZATION}' == 'int8_sq' ]; then
                # INT8 SmoothQuant (good balance, works on older GPUs)
                echo 'Quantizing with INT8 SmoothQuant...'
                python quantize.py \
                    --model_dir \$MODEL_DIR \
                    --qformat int8_sq \
                    --kv_cache_dtype int8 \
                    --output_dir \$CHECKPOINT_DIR

            elif [ '${QUANTIZATION}' == 'int4_awq' ]; then
                # INT4 AWQ (smallest, good for memory-constrained)
                echo 'Quantizing with INT4 AWQ...'
                python quantize.py \
                    --model_dir \$MODEL_DIR \
                    --qformat int4_awq \
                    --awq_block_size 128 \
                    --output_dir \$CHECKPOINT_DIR

            else
                echo 'Unknown quantization: ${QUANTIZATION}'
                exit 1
            fi

            echo '=== Step 2: Building TensorRT engine ==='
            trtllm-build \
                --checkpoint_dir \$CHECKPOINT_DIR \
                --output_dir \$ENGINE_DIR \
                --max_batch_size 8 \
                --max_input_len 4096 \
                --max_seq_len 8192 \
                --gemm_plugin auto \
                --paged_kv_cache enable \
                --use_fused_mlp enable \
                --multiple_profiles enable

            echo '=== Build complete! ==='
            ls -la \$ENGINE_DIR
        "

    log_info "Engine built at ${ENGINE_DIR}"
}

# =============================================================================
# Setup Triton Model Repository
# =============================================================================
setup_triton() {
    log_info "Setting up Triton model repository..."

    ENGINE_DIR="${WORK_DIR}/engine_${QUANTIZATION}"
    TRITON_MODEL="${MODEL_REPO}/qwen3_8b"

    mkdir -p "${TRITON_MODEL}/1"

    # Link or copy engine files
    if [ -d "${ENGINE_DIR}" ]; then
        ln -sf "${ENGINE_DIR}" "${TRITON_MODEL}/1/engine" 2>/dev/null || \
            cp -r "${ENGINE_DIR}" "${TRITON_MODEL}/1/engine"
    fi

    # Create config.pbtxt for TensorRT-LLM backend
    cat > "${TRITON_MODEL}/config.pbtxt" << 'EOF'
name: "qwen3_8b"
backend: "tensorrtllm"
max_batch_size: 8

model_transaction_policy {
  decoupled: true
}

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "end_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "pad_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "streaming"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1, -1 ]
  },
  {
    name: "sequence_length"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

parameters {
  key: "gpt_model_type"
  value: { string_value: "inflight_fused_batching" }
}

parameters {
  key: "gpt_model_path"
  value: { string_value: "${engine_dir}" }
}

parameters {
  key: "batch_scheduler_policy"
  value: { string_value: "max_utilization" }
}

parameters {
  key: "kv_cache_free_gpu_mem_fraction"
  value: { string_value: "0.8" }
}

parameters {
  key: "max_tokens_in_paged_kv_cache"
  value: { string_value: "8192" }
}
EOF

    # Replace placeholder with actual path
    sed -i "s|\${engine_dir}|/models/qwen3_8b/1/engine|g" "${TRITON_MODEL}/config.pbtxt"

    log_info "Triton model configured at ${TRITON_MODEL}"
}

# =============================================================================
# Main
# =============================================================================
main() {
    download_model
    build_engine
    setup_triton

    echo ""
    echo "=============================================="
    echo -e "${GREEN}Qwen3 8B Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Quantization: ${QUANTIZATION}"
    echo "Engine: ${WORK_DIR}/engine_${QUANTIZATION}"
    echo "Triton model: ${MODEL_REPO}/qwen3_8b"
    echo ""
    echo "To test locally:"
    echo "  trtllm-serve ${WORK_DIR}/engine_${QUANTIZATION} --port 8000"
    echo ""
    echo "Or start Triton:"
    echo "  docker-compose up -d triton"
}

main
