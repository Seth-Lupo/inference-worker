#!/bin/bash
# =============================================================================
# Build CosyVoice 2 TensorRT-LLM Engines (TTS Model)
#
# Downloads CosyVoice2-0.5B from HuggingFace/ModelScope, builds TRT engines,
# and creates Triton model repository with 7 models:
#   - cosyvoice2, cosyvoice2_dit (BLS orchestrators)
#   - tensorrt_llm (TRT-LLM engine)
#   - token2wav, token2wav_dit (vocoders)
#   - audio_tokenizer, speaker_embedding (audio processing)
#
# Usage:
#   ./build_cosyvoice.sh [START_STAGE] [STOP_STAGE]
#   ./build_cosyvoice.sh 0 3       # Run all stages
#   ./build_cosyvoice.sh 1 1       # Only build TRT-LLM engines
#   ./build_cosyvoice.sh 3 3       # Only rebuild model repository
#   ./build_cosyvoice.sh cleanup   # Clean up build artifacts
#
# Stages:
#    0: Download models from HuggingFace/ModelScope
#    1: Build TensorRT-LLM engines (LLM)
#    2: Build ONNX→TRT engines (audio_tokenizer, speaker_embedding)
#    3: Create Triton model repository
#
# Environment Variables:
#   HF_TOKEN          - HuggingFace token for gated models
#   TRT_DTYPE         - TensorRT dtype: bfloat16, float16 (default: bfloat16)
#   TRITON_MAX_BATCH_SIZE - Max batch size (default: 16)
#
# Sources:
#   - Local: deploy/cosyvoice_triton/ (model templates and scripts)
#   - Local: src/cosyvoice/ (Python package)
#   - https://huggingface.co/yuekai/cosyvoice2_llm
#   - https://modelscope.cn/models/iic/CosyVoice2-0.5B
# =============================================================================

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================
HF_MODEL="${HF_MODEL:-yuekai/cosyvoice2_llm}"
MODELSCOPE_MODEL="${MODELSCOPE_MODEL:-iic/CosyVoice2-0.5B}"

TRT_DTYPE="${TRT_DTYPE:-bfloat16}"
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3}"
TRTLLM_CONTAINER_NAME="trtllm-builder-cosyvoice"

# Triton settings
TRITON_MAX_BATCH_SIZE="${TRITON_MAX_BATCH_SIZE:-16}"
BLS_INSTANCE_NUM="${BLS_INSTANCE_NUM:-4}"
DECOUPLED_MODE="${DECOUPLED_MODE:-True}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/cosyvoice_build"
readonly MODEL_REPO="${DEPLOY_DIR}/model_repository/tts"

# Local CosyVoice source (no external cloning needed)
readonly COSYVOICE_SRC="${DEPLOY_DIR}/../src/cosyvoice"
readonly COSYVOICE_TEMPLATES="${DEPLOY_DIR}/model_repository/tts"
readonly COSYVOICE_SCRIPTS="${DEPLOY_DIR}/scripts/cosyvoice"

# =============================================================================
# Load environment
# =============================================================================
load_env

# =============================================================================
# Cleanup Handler
# =============================================================================
if [[ "${1:-}" == "cleanup" || "${1:-}" == "clean" ]]; then
    echo "=============================================="
    echo "Cleaning up CosyVoice build artifacts"
    echo "=============================================="

    remove_container "$TRTLLM_CONTAINER_NAME"

    case "${2:-}" in
        --all|-a)
            log_warn "Removing ALL build artifacts..."
            rm -rf "$WORK_DIR"
            rm -rf "$MODEL_REPO"
            log_info "Cleanup complete"
            ;;
        --image|-i)
            log_info "Removing Docker image: $TRTLLM_IMAGE"
            docker rmi "$TRTLLM_IMAGE" 2>/dev/null || true
            ;;
        *)
            log_info "Removing downloads (keeping engines)..."
            rm -rf "${WORK_DIR}/cosyvoice2_llm"
            rm -rf "${WORK_DIR}/CosyVoice2-0.5B"
            rm -rf "${WORK_DIR}"/trt_weights_*
            log_info "Kept: ${WORK_DIR}/trt_engines_*"
            ;;
    esac
    exit 0
fi

# Parse stages
START_STAGE="${1:-0}"
STOP_STAGE="${2:-3}"

echo "=============================================="
echo "Building CosyVoice 2 TensorRT-LLM"
echo "=============================================="
echo "Stages: ${START_STAGE} to ${STOP_STAGE}"
echo "Container: ${TRTLLM_IMAGE}"
echo ""

mkdir -p "$WORK_DIR"

# Verify local source exists
if [[ ! -d "$COSYVOICE_TEMPLATES" ]]; then
    log_error "CosyVoice templates not found at: $COSYVOICE_TEMPLATES"
    log_error "Expected directory: deploy/model_repository/cosyvoice_templates/"
    exit 1
fi

# =============================================================================
# Stage 0: Download Models
# =============================================================================
stage_download_models() {
    log_step "Stage 0: Downloading models..."

    # Download LLM from HuggingFace
    local llm_dir="${WORK_DIR}/cosyvoice2_llm"
    if ! is_real_file "${llm_dir}/model.safetensors"; then
        log_info "Downloading LLM from HuggingFace: ${HF_MODEL}"
        rm -rf "$llm_dir"

        if ! hf_clone "$HF_MODEL" "$llm_dir" "${HF_TOKEN:-}"; then
            log_error "Failed to download LLM"
            return 1
        fi

        # Verify download
        if ! is_real_file "${llm_dir}/model.safetensors"; then
            log_warn "Weights missing, trying LFS pull..."
            lfs_pull "$llm_dir"

            if ! is_real_file "${llm_dir}/model.safetensors"; then
                log_error "Failed to download LLM weights"
                log_info "Try: cd ${llm_dir} && git lfs pull"
                return 1
            fi
        fi
    else
        log_info "LLM already downloaded"
    fi

    # Download full model from ModelScope
    local modelscope_dir="${WORK_DIR}/CosyVoice2-0.5B"
    if ! is_real_file "${modelscope_dir}/flow.pt"; then
        log_info "Downloading from ModelScope: ${MODELSCOPE_MODEL}"

        mkdir -p "$modelscope_dir"

        # Try modelscope CLI first
        if command -v modelscope &>/dev/null; then
            modelscope download --model "$MODELSCOPE_MODEL" --local_dir "$modelscope_dir"
        # Fallback to git clone
        elif command -v git &>/dev/null; then
            log_info "Using git clone from ModelScope..."
            ensure_git_lfs
            GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 "https://www.modelscope.cn/${MODELSCOPE_MODEL}.git" "$modelscope_dir" || {
                log_warn "ModelScope git clone failed"
            }
            # Ensure LFS files are pulled
            if [[ -d "$modelscope_dir/.git" ]]; then
                log_info "Pulling LFS files..."
                (cd "$modelscope_dir" && git lfs pull)
            fi
        fi

        if ! is_real_file "${modelscope_dir}/flow.pt"; then
            log_warn "Could not auto-download from ModelScope"
            log_info ""
            log_info "Please download manually:"
            log_info "  1. Visit: https://modelscope.cn/models/${MODELSCOPE_MODEL}"
            log_info "  2. Download and extract to: ${modelscope_dir}"
            log_info ""
            log_info "Or install modelscope CLI:"
            log_info "  pip install modelscope"
        fi
    else
        log_info "ModelScope model already downloaded"
    fi

    # Download speaker cache
    if [[ -d "$modelscope_dir" && ! -f "${modelscope_dir}/spk2info.pt" ]]; then
        log_info "Downloading speaker info cache..."
        curl -L -f -o "${modelscope_dir}/spk2info.pt" \
            "https://raw.githubusercontent.com/qi-hua/async_cosyvoice/main/CosyVoice2-0.5B/spk2info.pt" \
            2>/dev/null || log_warn "Could not download spk2info.pt"
    fi
}

# =============================================================================
# Stage 1: Build TensorRT Engines
# =============================================================================
stage_build_engines() {
    log_step "Stage 1: Building TensorRT engines..."

    local weights_dir="${WORK_DIR}/trt_weights_${TRT_DTYPE}"
    local engines_dir="${WORK_DIR}/trt_engines_${TRT_DTYPE}"

    # Check if engines already exist
    if [[ -f "${engines_dir}/rank0.engine" ]]; then
        log_info "Engines already exist at ${engines_dir}"
        return 0
    fi

    # Ensure container image exists
    ensure_docker_image "$TRTLLM_IMAGE"

    log_info "Starting build in container..."

    docker run --rm --gpus all \
        --name "$TRTLLM_CONTAINER_NAME" \
        --shm-size=8g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${WORK_DIR}:/workspace/build" \
        -v "${COSYVOICE_SCRIPTS}:/workspace/scripts:ro" \
        -v "${COSYVOICE_SRC}:/workspace/cosyvoice:ro" \
        -w /workspace/build \
        -e PYTHONPATH=/workspace \
        "$TRTLLM_IMAGE" \
        bash -c "
            set -e

            echo '=== TensorRT-LLM Version ==='
            TRTLLM_VERSION=\$(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null | tail -1)
            echo \"TRT-LLM: \${TRTLLM_VERSION}\"

            echo '=== Installing dependencies ==='
            pip install -q transformers sentencepiece accelerate 2>/dev/null || true

            echo '=== Converting checkpoint ==='
            python3 /workspace/scripts/convert_checkpoint.py \
                --model_dir ./cosyvoice2_llm \
                --output_dir ./trt_weights_${TRT_DTYPE} \
                --dtype ${TRT_DTYPE}

            echo '=== Building TensorRT engines ==='
            trtllm-build \
                --checkpoint_dir ./trt_weights_${TRT_DTYPE} \
                --output_dir ./trt_engines_${TRT_DTYPE} \
                --max_batch_size 16 \
                --max_num_tokens 32768 \
                --gemm_plugin ${TRT_DTYPE}

            echo '=== Build complete ==='
            ls -la ./trt_engines_${TRT_DTYPE}
        "

    log_info "Engines built at ${engines_dir}"
}

# =============================================================================
# Stage 2: Build ONNX→TRT Engines for Audio Processing
# =============================================================================
stage_build_onnx_trt() {
    log_step "Stage 2: Building ONNX→TRT engines for audio processing..."

    local modelscope_dir="${WORK_DIR}/CosyVoice2-0.5B"

    if [[ ! -d "$modelscope_dir" ]]; then
        log_error "ModelScope models not found. Run stage 0 first."
        return 1
    fi

    # Build speech_tokenizer TRT (for audio_tokenizer)
    # Input: feats (batch, 128, time) - variable time dimension
    log_info "Building speech_tokenizer engine..."
    build_trt_engine \
        "${modelscope_dir}/speech_tokenizer_v2.onnx" \
        "${modelscope_dir}/speech_tokenizer_v2.engine" \
        --fp16 \
        "--minShapes=feats:1x128x10" \
        "--optShapes=feats:1x128x500" \
        "--maxShapes=feats:1x128x3000"

    # Build campplus TRT (for speaker_embedding)
    # Input: input (batch, time, 80) - variable time dimension
    log_info "Building campplus engine..."
    build_trt_engine \
        "${modelscope_dir}/campplus.onnx" \
        "${modelscope_dir}/campplus.engine" \
        --fp32 \
        "--minShapes=input:1x4x80" \
        "--optShapes=input:1x500x80" \
        "--maxShapes=input:1x3000x80"

    # Build flow decoder estimator TRT (for token2wav vocoder)
    # Inputs: x, mask, mu, cond - all with shape (batch, channels, time)
    # Requires large workspace (~8GB) for optimization
    log_info "Building flow decoder estimator engine (this takes ~5-10 minutes)..."
    build_trt_engine \
        "${modelscope_dir}/flow.decoder.estimator.fp32.onnx" \
        "${modelscope_dir}/flow.decoder.estimator.fp16.engine" \
        --fp16 \
        "--memPoolSize=8192" \
        "--minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4" \
        "--optShapes=x:2x80x500,mask:2x1x500,mu:2x80x500,cond:2x80x500" \
        "--maxShapes=x:2x80x3000,mask:2x1x3000,mu:2x80x3000,cond:2x80x3000"

    log_info "ONNX→TRT engines built successfully"
}

# =============================================================================
# Stage 3: Create Model Repository
# =============================================================================
stage_create_model_repo() {
    log_step "Stage 3: Setting up Triton model repository..."

    if [[ ! -d "$MODEL_REPO" ]]; then
        log_error "Model repository not found at: $MODEL_REPO"
        return 1
    fi

    # Note: Config files already have correct values set.
    # Template filling is skipped - configs point to /models/tts/assets/
    log_info "Configs already configured, skipping template filling..."

    # Copy assets
    log_info "Copying assets..."
    mkdir -p "${MODEL_REPO}/assets"

    cp -r "${WORK_DIR}/CosyVoice2-0.5B"/* "${MODEL_REPO}/assets/" 2>/dev/null || true

    mkdir -p "${MODEL_REPO}/assets/cosyvoice2_llm"
    cp -r "${WORK_DIR}/cosyvoice2_llm"/* "${MODEL_REPO}/assets/cosyvoice2_llm/" 2>/dev/null || true

    # Copy engines
    mkdir -p "${MODEL_REPO}/tensorrt_llm/1/engine"
    cp -r "${WORK_DIR}/trt_engines_${TRT_DTYPE}"/* "${MODEL_REPO}/tensorrt_llm/1/engine/" 2>/dev/null || true

    log_info "Model repository created at ${MODEL_REPO}"
}

# =============================================================================
# Main
# =============================================================================
main() {
    # Stage 0: Download models
    if [[ $START_STAGE -le 0 && $STOP_STAGE -ge 0 ]]; then
        stage_download_models
    fi

    # Stage 1: Build TRT-LLM engines
    if [[ $START_STAGE -le 1 && $STOP_STAGE -ge 1 ]]; then
        stage_build_engines
    fi

    # Stage 2: Build ONNX→TRT engines
    if [[ $START_STAGE -le 2 && $STOP_STAGE -ge 2 ]]; then
        stage_build_onnx_trt
    fi

    # Stage 3: Create model repo
    if [[ $START_STAGE -le 3 && $STOP_STAGE -ge 3 ]]; then
        stage_create_model_repo
    fi

    echo ""
    echo "=============================================="
    echo -e "${GREEN}CosyVoice 2 Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Model repository: ${MODEL_REPO}"
    echo ""
    echo "Models:"
    ls -1 "$MODEL_REPO" 2>/dev/null | grep -v assets || echo "  (run stages 0-2 to build)"
    echo ""
    echo "To start Triton:"
    echo "  docker compose up -d triton"
    echo ""
    echo "Cleanup options:"
    echo "  $0 cleanup           # Remove downloads, keep engines"
    echo "  $0 cleanup --all     # Remove everything"
    echo "  $0 cleanup --image   # Remove Docker image"
}

main
