#!/bin/bash
# =============================================================================
# Build Qwen3 8B TensorRT-LLM Engine (LLM Model)
#
# Downloads model from HuggingFace, quantizes, and builds TensorRT engine.
# Uses the Triton TRT-LLM container for building to ensure compatibility.
#
# Sources:
#   - https://github.com/NVIDIA/TensorRT-LLM
#   - https://huggingface.co/Qwen/Qwen3-8B
# =============================================================================

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-Qwen3-8B}"
QUANTIZATION="${QUANTIZATION:-int4}"

# Build container (must match serving container for engine compatibility)
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3}"
TRTLLM_CONTAINER_NAME="trtllm-builder-qwen3"

# Engine build parameters
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"

# KV cache settings
KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTION:-0.5}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/qwen3_build"
readonly MODEL_REPO="${DEPLOY_DIR}/model_repository/llm/qwen3_8b"

# =============================================================================
# Load environment and parse args
# =============================================================================
load_env

# Override quantization from command line
if [[ -n "${1:-}" && "$1" != "cleanup" && "$1" != "clean" ]]; then
    QUANTIZATION="$1"
fi

# =============================================================================
# Cleanup Handler
# =============================================================================
if [[ "${1:-}" == "cleanup" || "${1:-}" == "clean" ]]; then
    echo "=============================================="
    echo "Cleaning up Qwen3 build artifacts"
    echo "=============================================="

    remove_container "$TRTLLM_CONTAINER_NAME"

    case "${2:-}" in
        --all|-a)
            log_warn "Removing ALL build artifacts..."
            rm -rf "$WORK_DIR"
            log_info "Build directory removed"
            ;;
        --image|-i)
            log_info "Removing Docker image: $TRTLLM_IMAGE"
            docker rmi "$TRTLLM_IMAGE" 2>/dev/null || true
            ;;
        *)
            log_info "Removing model download and checkpoints (keeping engine)..."
            rm -rf "${WORK_DIR}/${MODEL_DIR_NAME}"
            rm -rf "${WORK_DIR}"/checkpoint_*
            log_info "Kept: ${WORK_DIR}/engine_*"
            ;;
    esac
    exit 0
fi

# =============================================================================
# Functions
# =============================================================================

download_model() {
    log_step "Downloading ${MODEL_NAME} from HuggingFace..."

    local model_path="${WORK_DIR}/${MODEL_DIR_NAME}"

    # Check if already downloaded with real weights
    if has_real_weights "$model_path" "*.safetensors"; then
        log_info "Model already exists at $model_path"
        return 0
    fi

    # Try to pull LFS if directory exists but weights are pointers
    if [[ -d "$model_path" ]]; then
        log_warn "Model directory exists but weights missing, trying LFS pull..."
        if lfs_pull "$model_path" && has_real_weights "$model_path" "*.safetensors"; then
            log_info "LFS pull successful"
            return 0
        fi
        log_warn "LFS pull failed, removing incomplete download..."
        rm -rf "$model_path"
    fi

    mkdir -p "$WORK_DIR"

    # Clone from HuggingFace
    hf_clone "$MODEL_NAME" "$model_path" "${HF_TOKEN:-}" || {
        log_error "Failed to clone model"
        return 1
    }

    # Verify weights were downloaded
    if ! has_real_weights "$model_path" "*.safetensors"; then
        log_warn "Weights not downloaded, trying LFS pull..."
        lfs_pull "$model_path"

        if ! has_real_weights "$model_path" "*.safetensors"; then
            log_error "Failed to download model weights"
            log_info "Try: cd ${model_path} && git lfs pull"
            return 1
        fi
    fi

    log_info "Model downloaded successfully"
}

build_engine() {
    log_step "Building TensorRT-LLM engine (${QUANTIZATION})..."

    local checkpoint_dir="${WORK_DIR}/checkpoint_${QUANTIZATION}"
    local engine_dir="${WORK_DIR}/engine_${QUANTIZATION}"

    # Check if engine already exists
    if [[ -f "${engine_dir}/rank0.engine" ]]; then
        log_info "Engine already exists at ${engine_dir}"
        log_info "Delete directory to rebuild: rm -rf ${engine_dir}"
        return 0
    fi

    # Ensure container image exists
    ensure_docker_image "$TRTLLM_IMAGE"

    log_info "Starting build in container..."

    docker run --rm --gpus all \
        --name "$TRTLLM_CONTAINER_NAME" \
        --shm-size=16g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${WORK_DIR}:/workspace/build" \
        -w /workspace \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        "$TRTLLM_IMAGE" \
        bash -c "
            set -e

            MODEL_DIR=/workspace/build/${MODEL_DIR_NAME}
            CHECKPOINT_DIR=/workspace/build/checkpoint_${QUANTIZATION}
            ENGINE_DIR=/workspace/build/engine_${QUANTIZATION}

            echo '=== TensorRT-LLM Version ==='
            TRTLLM_VERSION=\$(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null | tail -1)
            echo \"TRT-LLM: \${TRTLLM_VERSION}\"

            echo '=== Installing dependencies ==='
            pip install -q accelerate sentencepiece 2>/dev/null || true

            echo '=== Cloning TRT-LLM examples ==='
            if [[ ! -d '/workspace/TensorRT-LLM' ]]; then
                git clone --depth 1 --branch \"v\${TRTLLM_VERSION}\" https://github.com/NVIDIA/TensorRT-LLM.git /workspace/TensorRT-LLM 2>/dev/null || \
                git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git /workspace/TensorRT-LLM
            fi

            # Find qwen convert script
            QWEN_DIR=\$(find /workspace/TensorRT-LLM -type d -name 'qwen' | grep -E 'examples.*qwen\$' | head -1)
            if [[ -z \"\$QWEN_DIR\" ]]; then
                echo 'ERROR: Could not find qwen examples directory'
                exit 1
            fi
            echo \"Using: \$QWEN_DIR\"
            cd \"\$QWEN_DIR\"

            echo '=== Converting/Quantizing model ==='
            case '${QUANTIZATION}' in
                none)
                    python3 convert_checkpoint.py \
                        --model_dir \"\$MODEL_DIR\" \
                        --output_dir \"\$CHECKPOINT_DIR\" \
                        --dtype float16
                    ;;
                fp8)
                    python3 convert_checkpoint.py \
                        --model_dir \"\$MODEL_DIR\" \
                        --output_dir \"\$CHECKPOINT_DIR\" \
                        --dtype float16 \
                        --use_fp8
                    ;;
                int8_sq)
                    python3 convert_checkpoint.py \
                        --model_dir \"\$MODEL_DIR\" \
                        --output_dir \"\$CHECKPOINT_DIR\" \
                        --dtype float16 \
                        --smoothquant 0.5 \
                        --per_token \
                        --per_channel \
                        --int8_kv_cache
                    ;;
                int4_awq|int4)
                    python3 convert_checkpoint.py \
                        --model_dir \"\$MODEL_DIR\" \
                        --output_dir \"\$CHECKPOINT_DIR\" \
                        --dtype float16 \
                        --use_weight_only \
                        --weight_only_precision int4 \
                        --per_group \
                        --group_size 128
                    ;;
                *)
                    echo \"Unknown quantization: ${QUANTIZATION}\"
                    exit 1
                    ;;
            esac

            echo '=== Building TensorRT engine ==='
            trtllm-build \
                --checkpoint_dir \"\$CHECKPOINT_DIR\" \
                --output_dir \"\$ENGINE_DIR\" \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --max_input_len ${MAX_INPUT_LEN} \
                --max_seq_len ${MAX_SEQ_LEN} \
                --gemm_plugin auto \
                --paged_kv_cache enable \
                --use_fused_mlp enable \
                --multiple_profiles enable

            echo '=== Build complete ==='
            ls -la \"\$ENGINE_DIR\"
        "

    log_info "Engine built at ${engine_dir}"
}

setup_triton() {
    log_step "Setting up Triton model repository..."

    local engine_dir="${WORK_DIR}/engine_${QUANTIZATION}"

    mkdir -p "${MODEL_REPO}/1"

    # Copy engine directory
    if [[ -d "$engine_dir" ]]; then
        rm -rf "${MODEL_REPO}/1/engine"
        log_info "Copying engine files (this may take a moment)..."
        cp -r "$engine_dir" "${MODEL_REPO}/1/engine"
        log_info "Engine copied"
    fi

    # Copy tokenizer files
    log_info "Copying tokenizer files..."
    mkdir -p "${MODEL_REPO}/1/tokenizer"

    local src="${WORK_DIR}/${MODEL_DIR_NAME}"
    cp "${src}/tokenizer"*.json "${MODEL_REPO}/1/tokenizer/" 2>/dev/null || true
    cp "${src}/vocab"* "${MODEL_REPO}/1/tokenizer/" 2>/dev/null || true
    cp "${src}/merges"* "${MODEL_REPO}/1/tokenizer/" 2>/dev/null || true
    cp "${src}/special_tokens"* "${MODEL_REPO}/1/tokenizer/" 2>/dev/null || true
    cp "${src}/config.json" "${MODEL_REPO}/1/tokenizer/" 2>/dev/null || true
    cp "${src}/generation_config.json" "${MODEL_REPO}/1/tokenizer/" 2>/dev/null || true

    # Create config.pbtxt
    # Note: Using ${var} placeholder syntax for optional TRT-LLM parameters
    # that get auto-detected by the backend
    cat > "${MODEL_REPO}/config.pbtxt" << EOF
# Qwen3 8B - TensorRT-LLM Backend
# Quantization: ${QUANTIZATION}

name: "qwen3_8b"
backend: "tensorrtllm"
max_batch_size: ${MAX_BATCH_SIZE}

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
  },
  {
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "top_k"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "top_p"
    data_type: TYPE_FP32
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
  value: { string_value: "/models/llm/qwen3_8b/1/engine" }
}

parameters {
  key: "tokenizer_dir"
  value: { string_value: "/models/llm/qwen3_8b/1/tokenizer" }
}

parameters {
  key: "batch_scheduler_policy"
  value: { string_value: "max_utilization" }
}

parameters {
  key: "kv_cache_free_gpu_mem_fraction"
  value: { string_value: "${KV_CACHE_FREE_GPU_MEM_FRACTION}" }
}

parameters {
  key: "xgrammar_tokenizer_info_path"
  value: { string_value: "" }
}

parameters {
  key: "guided_decoding_backend"
  value: { string_value: "" }
}

version_policy: { latest: { num_versions: 1 } }
EOF

    log_info "Created: ${MODEL_REPO}/config.pbtxt"
}

show_summary() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}Qwen3 8B Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Quantization: ${QUANTIZATION}"
    echo "Engine:       ${WORK_DIR}/engine_${QUANTIZATION}"
    echo "Triton model: ${MODEL_REPO}"
    echo ""
    echo "To start Triton:"
    echo "  docker compose up -d triton"
    echo ""
    echo "Cleanup options:"
    echo "  $0 cleanup           # Remove downloads, keep engine"
    echo "  $0 cleanup --all     # Remove everything"
    echo "  $0 cleanup --image   # Remove Docker image (~20GB)"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "=============================================="
    echo "Building Qwen3 8B TensorRT-LLM Engine"
    echo "=============================================="
    echo "Model:        ${MODEL_NAME}"
    echo "Quantization: ${QUANTIZATION}"
    echo "Container:    ${TRTLLM_IMAGE}"
    echo ""

    require_gpu

    download_model
    build_engine
    setup_triton
    show_summary
}

main
