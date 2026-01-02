#!/bin/bash
# =============================================================================
# Build Qwen3 8B TensorRT-LLM Engine
# Uses dedicated TensorRT-LLM development container for building
# Based on: https://github.com/NVIDIA/TensorRT-LLM
#           https://github.com/NVIDIA/TensorRT-Model-Optimizer
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

# Container configuration
# IMPORTANT: Use same TRT-LLM version as Triton serving container!
# Triton 24.12 has TRT-LLM 0.16.0, so use matching DEV container for building
# (tritonserver container is serving-only, no build tools)
TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:0.16.0"
TRTLLM_CONTAINER_NAME="trtllm-builder-qwen3"

# Load environment
if [ -f "${DEPLOY_DIR}/.env" ]; then
    source "${DEPLOY_DIR}/.env"
fi

# Configuration
# Note: Qwen3-8B is already instruct-capable (no separate -Instruct variant)
# Qwen3-8B-Base is the pretrained-only version
MODEL_NAME="Qwen/Qwen3-8B"
MODEL_DIR_NAME="Qwen3-8B"
QUANTIZATION="${1:-int8_sq}"  # Options: fp8, int8_sq, int4_awq, none

# =============================================================================
# Handle cleanup command
# =============================================================================
if [ "$1" == "cleanup" ] || [ "$1" == "clean" ]; then
    echo "=============================================="
    echo "Cleaning up Qwen3 build artifacts"
    echo "=============================================="

    # Stop and remove container if running
    if docker ps -a --format '{{.Names}}' | grep -q "^${TRTLLM_CONTAINER_NAME}$"; then
        log_info "Removing container ${TRTLLM_CONTAINER_NAME}..."
        docker rm -f "${TRTLLM_CONTAINER_NAME}" 2>/dev/null || true
    fi

    # --keep-engine: Remove downloads/checkpoints but keep built engine (default)
    if [ "$2" == "--keep-engine" ] || [ "$2" == "-k" ] || [ -z "$2" ]; then
        log_info "Removing downloaded model and checkpoints (keeping engine)..."

        # Remove HuggingFace model download (~16GB)
        if [ -d "${WORK_DIR}/${MODEL_DIR_NAME}" ]; then
            log_info "  Removing ${MODEL_DIR_NAME}/ (~16GB)..."
            rm -rf "${WORK_DIR}/${MODEL_DIR_NAME}"
        fi

        # Remove quantization checkpoints
        for checkpoint_dir in "${WORK_DIR}"/checkpoint_*; do
            if [ -d "$checkpoint_dir" ]; then
                log_info "  Removing $(basename $checkpoint_dir)/..."
                rm -rf "$checkpoint_dir"
            fi
        done

        # Show what's left
        log_info "Kept engines for inference:"
        ls -d "${WORK_DIR}"/engine_* 2>/dev/null || log_info "  (no engines found)"

        log_info ""
        log_info "To also remove Docker image (~20GB): $0 cleanup --image"
        exit 0
    fi

    # --image: Remove the Docker image
    if [ "$2" == "--image" ] || [ "$2" == "-i" ]; then
        log_info "Removing TensorRT-LLM image ${TRTLLM_IMAGE}..."
        docker rmi "${TRTLLM_IMAGE}" 2>/dev/null || true
        log_info "Image removed"
        exit 0
    fi

    # --all: Remove everything
    if [ "$2" == "--all" ] || [ "$2" == "-a" ]; then
        log_warn "Removing ALL build artifacts at ${WORK_DIR}..."
        rm -rf "${WORK_DIR}"
        log_info "Build directory removed"
        log_info ""
        log_info "To also remove Docker image: $0 cleanup --image"
        exit 0
    fi

    # Show help
    echo "Usage: $0 cleanup [OPTION]"
    echo ""
    echo "Options:"
    echo "  (none), --keep-engine, -k  Remove downloads/checkpoints, keep engines (default)"
    echo "  --image, -i                Remove Docker image (~20GB)"
    echo "  --all, -a                  Remove everything including engines"
    echo ""
    exit 0
fi

echo "=============================================="
echo "Building Qwen3 8B TensorRT-LLM Engine"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Quantization: ${QUANTIZATION}"
echo "Build container: ${TRTLLM_IMAGE}"
echo ""
echo "To cleanup after building:"
echo "  $0 cleanup          # Remove container only"
echo "  $0 cleanup --image  # Also remove Docker image (~20GB)"
echo "  $0 cleanup --all    # Remove everything including downloads"
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
# Check if model weights are complete (not just LFS pointers)
# =============================================================================
check_model_complete() {
    local model_dir="$1"

    # Check if any .safetensors file exists and is larger than 1MB (not a pointer)
    if [ -d "$model_dir" ]; then
        for f in "$model_dir"/*.safetensors "$model_dir"/model-*.safetensors; do
            if [ -f "$f" ]; then
                local size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "0")
                if [ "$size" -gt 1000000 ]; then
                    return 0  # Found a real weight file
                fi
            fi
        done
    fi
    return 1  # No valid weight files found
}

# =============================================================================
# Download Model
# =============================================================================
download_model() {
    log_info "Downloading ${MODEL_NAME} from HuggingFace..."

    local MODEL_PATH="${WORK_DIR}/${MODEL_DIR_NAME}"

    # Check if model exists AND has actual weights (not just LFS pointers)
    if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
        if check_model_complete "$MODEL_PATH"; then
            log_info "Model already exists with weights at $MODEL_PATH"
            return 0
        else
            log_warn "Model directory exists but weights are missing (LFS pointers only)"
            log_info "Attempting to pull LFS files..."

            cd "$MODEL_PATH"
            git lfs pull 2>/dev/null && {
                if check_model_complete "$MODEL_PATH"; then
                    log_info "LFS files pulled successfully"
                    cd "${WORK_DIR}"
                    return 0
                fi
            }
            cd "${WORK_DIR}"

            log_warn "LFS pull failed, removing incomplete download..."
            rm -rf "$MODEL_PATH"
        fi
    fi

    # Ensure git-lfs is installed
    if ! command -v git-lfs &> /dev/null; then
        log_info "Installing git-lfs..."
        sudo yum install -y git-lfs 2>/dev/null || sudo apt-get install -y git-lfs 2>/dev/null || {
            log_error "Could not install git-lfs automatically"
            log_info "Please install manually:"
            log_info "  sudo yum install git-lfs  # Amazon Linux"
            log_info "  sudo apt install git-lfs  # Ubuntu"
            return 1
        }
    fi

    git lfs install

    # Clone with authentication if token provided
    if [ -n "$HF_TOKEN" ]; then
        GIT_URL="https://USER:${HF_TOKEN}@huggingface.co/${MODEL_NAME}"
    else
        GIT_URL="https://huggingface.co/${MODEL_NAME}"
    fi

    log_info "Cloning ${MODEL_NAME} (this will download ~16GB)..."

    # Use GIT_LFS_SKIP_SMUDGE=0 to ensure LFS files are downloaded
    GIT_LFS_SKIP_SMUDGE=0 git clone "$GIT_URL" "$MODEL_PATH" || {
        log_error "git clone failed"
        return 1
    }

    # Verify weights were downloaded
    if ! check_model_complete "$MODEL_PATH"; then
        log_warn "Weights not downloaded, trying git lfs pull..."
        cd "$MODEL_PATH"
        git lfs pull
        cd "${WORK_DIR}"

        if ! check_model_complete "$MODEL_PATH"; then
            log_error "Failed to download model weights"
            log_info ""
            log_info "Try manually:"
            log_info "  cd ${MODEL_PATH}"
            log_info "  git lfs pull"
            log_info ""
            return 1
        fi
    fi

    log_info "Model downloaded successfully"
    return 0
}

# =============================================================================
# Pull TensorRT-LLM Development Container
# =============================================================================
pull_container() {
    log_info "Checking for TensorRT-LLM development container..."

    if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${TRTLLM_IMAGE}$"; then
        log_info "Container image already exists"
    else
        log_info "Pulling ${TRTLLM_IMAGE} (this may take a while, ~20GB)..."
        docker pull "${TRTLLM_IMAGE}"
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

    # Ensure we have the container
    pull_container

    # Run quantization and build inside dedicated TensorRT-LLM development container
    # This container has all the quantization tools (ModelOpt) pre-installed
    log_info "Starting build in container ${TRTLLM_IMAGE}..."

    docker run --rm --gpus all \
        --name "${TRTLLM_CONTAINER_NAME}" \
        --shm-size=16g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${WORK_DIR}:/workspace/build" \
        -w /workspace \
        -e HF_TOKEN="${HF_TOKEN}" \
        "${TRTLLM_IMAGE}" \
        bash -c "
            set -e

            MODEL_DIR=/workspace/build/${MODEL_DIR_NAME}
            CHECKPOINT_DIR=/workspace/build/checkpoint_${QUANTIZATION}
            ENGINE_DIR=/workspace/build/engine_${QUANTIZATION}

            echo '=== Step 1: Quantizing model ==='

            # The TensorRT-LLM container has quantize.py in examples/quantization
            cd /app/tensorrt_llm/examples/quantization

            if [ '${QUANTIZATION}' == 'none' ]; then
                # No quantization - just convert checkpoint
                echo 'Converting without quantization (FP16)...'
                python3 quantize.py \
                    --model_dir \$MODEL_DIR \
                    --output_dir \$CHECKPOINT_DIR \
                    --dtype float16

            elif [ '${QUANTIZATION}' == 'fp8' ]; then
                # FP8 quantization (best for Hopper/Ada GPUs)
                echo 'Quantizing to FP8...'
                python3 quantize.py \
                    --model_dir \$MODEL_DIR \
                    --qformat fp8 \
                    --kv_cache_dtype fp8 \
                    --output_dir \$CHECKPOINT_DIR

            elif [ '${QUANTIZATION}' == 'int8_sq' ]; then
                # INT8 SmoothQuant (good balance, works on older GPUs)
                echo 'Quantizing with INT8 SmoothQuant...'
                python3 quantize.py \
                    --model_dir \$MODEL_DIR \
                    --qformat int8_sq \
                    --kv_cache_dtype int8 \
                    --output_dir \$CHECKPOINT_DIR

            elif [ '${QUANTIZATION}' == 'int4_awq' ]; then
                # INT4 AWQ (smallest, good for memory-constrained)
                echo 'Quantizing with INT4 AWQ...'
                python3 quantize.py \
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
    echo ""
    echo "=============================================="
    echo "Cleanup Options"
    echo "=============================================="
    echo ""
    echo "The TensorRT-LLM dev container (~20GB) can be removed after building:"
    echo "  $0 cleanup          # Just remove any stopped containers"
    echo "  $0 cleanup --image  # Remove the Docker image to free ~20GB"
    echo "  $0 cleanup --all    # Remove image + all downloaded models/checkpoints"
    echo ""
}

main
