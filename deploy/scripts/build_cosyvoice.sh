#!/bin/bash
# =============================================================================
# Build CosyVoice 2 TensorRT-LLM Engines
# Based on: https://github.com/FunAudioLLM/CosyVoice/blob/main/runtime/triton_trtllm/run.sh
# =============================================================================
#
# USAGE:
#   ./build_cosyvoice.sh [START_STAGE] [STOP_STAGE]
#   ./build_cosyvoice.sh -1 2      # Run all stages
#   ./build_cosyvoice.sh 1 1       # Only build TRT engines
#   ./build_cosyvoice.sh cleanup   # Clean up build artifacts
#
# =============================================================================
# STAGES EXPLAINED
# =============================================================================
#
# Stage -1: CLONE COSYVOICE REPO
#   - Clones https://github.com/FunAudioLLM/CosyVoice
#   - Contains: conversion scripts, Triton model templates, Python backends
#   - Output: cosyvoice_build/CosyVoice/
#
# Stage 0: DOWNLOAD MODELS
#   - Downloads LLM checkpoint from HuggingFace (yuekai/cosyvoice2_llm)
#     → This is the autoregressive text-to-token model
#   - Downloads full model from ModelScope (iic/CosyVoice2-0.5B)
#     → Contains: flow.pt (vocoder), speech tokenizer, speaker embeddings
#   - Output: cosyvoice_build/cosyvoice2_llm/, cosyvoice_build/CosyVoice2-0.5B/
#
# Stage 1: BUILD TENSORRT-LLM ENGINES
#   - Runs inside TensorRT-LLM dev container (~20GB image)
#   - Step 1: convert_checkpoint.py converts HuggingFace LLM → TRT-LLM weights
#   - Step 2: trtllm-build compiles weights → optimized TensorRT engine
#   - Output: cosyvoice_build/trt_engines_bfloat16/rank0.engine
#
# Stage 2: CREATE TRITON MODEL REPOSITORY
#   - Copies Triton model templates from CosyVoice repo
#   - Fills in config.pbtxt with paths to engines and model files
#   - Copies all assets into self-contained model_repository/cosyvoice2_full/
#   - Output: model_repository/cosyvoice2_full/ with 5 models:
#
#     ┌─────────────────────────────────────────────────────────────────┐
#     │                    COSYVOICE2 TRITON MODELS                     │
#     ├─────────────────────────────────────────────────────────────────┤
#     │  cosyvoice2/        BLS orchestrator (Python backend)           │
#     │    └── Coordinates all models, handles streaming chunking       │
#     │                                                                 │
#     │  tensorrt_llm/      Text → Speech Tokens (TRT-LLM engine)       │
#     │    └── Autoregressive LLM that generates semantic tokens        │
#     │                                                                 │
#     │  token2wav/         Speech Tokens → Audio (Python backend)      │
#     │    └── Flow-matching vocoder, converts tokens to waveform       │
#     │                                                                 │
#     │  audio_tokenizer/   Reference Audio → Tokens (Python backend)   │
#     │    └── For voice cloning: extracts tokens from reference audio  │
#     │                                                                 │
#     │  speaker_embedding/ Reference Audio → Embedding (Python backend)│
#     │    └── For voice cloning: extracts speaker characteristics      │
#     └─────────────────────────────────────────────────────────────────┘
#
# Stage 3: VERIFICATION (placeholder)
#   - Prints instructions for starting Triton
#
# =============================================================================
# INFERENCE FLOW (after build)
# =============================================================================
#
#   Input Text
#       │
#       ▼
#   ┌───────────────┐
#   │  cosyvoice2   │  BLS orchestrator
#   │   (Python)    │  - Tokenizes text
#   └───────┬───────┘  - Manages streaming
#           │
#           ▼
#   ┌───────────────┐
#   │ tensorrt_llm  │  Autoregressive LLM
#   │  (TRT-LLM)    │  - Generates speech tokens
#   └───────┬───────┘  - Streaming: emits tokens incrementally
#           │
#           ▼
#   ┌───────────────┐
#   │   token2wav   │  Flow-matching vocoder
#   │   (Python)    │  - Converts tokens to audio
#   └───────┬───────┘  - Chunked output for streaming
#           │
#           ▼
#     Audio Output
#
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR="${DEPLOY_DIR}/cosyvoice_build"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Container configuration
# IMPORTANT: Use Triton container for building to guarantee version match!
# Triton 24.12-trtllm-python-py3 has TRT-LLM 0.16.0 built-in
# This ensures the engine is compatible with the serving container
TRTLLM_IMAGE="nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3"
TRTLLM_CONTAINER_NAME="trtllm-builder-cosyvoice"

# Load environment
if [ -f "${DEPLOY_DIR}/.env" ]; then
    source "${DEPLOY_DIR}/.env"
fi

# Configuration
COSYVOICE_REPO="https://github.com/FunAudioLLM/CosyVoice.git"
HF_MODEL="yuekai/cosyvoice2_llm"
MODELSCOPE_MODEL="iic/CosyVoice2-0.5B"
TRT_DTYPE="bfloat16"

# =============================================================================
# Handle cleanup command
# =============================================================================
if [ "$1" == "cleanup" ] || [ "$1" == "clean" ]; then
    echo "=============================================="
    echo "Cleaning up CosyVoice build artifacts"
    echo "=============================================="

    # Stop and remove container if running
    if docker ps -a --format '{{.Names}}' | grep -q "^${TRTLLM_CONTAINER_NAME}$"; then
        log_info "Removing container ${TRTLLM_CONTAINER_NAME}..."
        docker rm -f "${TRTLLM_CONTAINER_NAME}" 2>/dev/null || true
    fi

    # --keep-engine: Remove downloads/checkpoints but keep built engine (default)
    if [ "$2" == "--keep-engine" ] || [ "$2" == "-k" ] || [ -z "$2" ]; then
        log_info "Removing downloaded models and checkpoints (keeping engines)..."

        # Remove HuggingFace LLM download
        if [ -d "${WORK_DIR}/cosyvoice2_llm" ]; then
            log_info "  Removing cosyvoice2_llm/..."
            rm -rf "${WORK_DIR}/cosyvoice2_llm"
        fi

        # Remove ModelScope model download
        if [ -d "${WORK_DIR}/CosyVoice2-0.5B" ]; then
            log_info "  Removing CosyVoice2-0.5B/..."
            rm -rf "${WORK_DIR}/CosyVoice2-0.5B"
        fi

        # Remove CosyVoice repo clone (scripts, not needed after build)
        if [ -d "${WORK_DIR}/CosyVoice" ]; then
            log_info "  Removing CosyVoice/ repo..."
            rm -rf "${WORK_DIR}/CosyVoice"
        fi

        # Remove TRT weights (intermediate, not needed after engine build)
        for weights_dir in "${WORK_DIR}"/trt_weights_*; do
            if [ -d "$weights_dir" ]; then
                log_info "  Removing $(basename $weights_dir)/..."
                rm -rf "$weights_dir"
            fi
        done

        # Show what's left
        log_info "Kept engines for inference:"
        ls -d "${WORK_DIR}"/trt_engines_* 2>/dev/null || log_info "  (no engines found)"

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

# Stages to run (default: all)
START_STAGE=${1:-0}
STOP_STAGE=${2:-3}

echo "=============================================="
echo "Building CosyVoice 2 TensorRT-LLM"
echo "=============================================="
echo "Stages: ${START_STAGE} to ${STOP_STAGE}"
echo "Build container: ${TRTLLM_IMAGE}"
echo ""
echo "To cleanup after building:"
echo "  $0 cleanup          # Remove container only"
echo "  $0 cleanup --image  # Also remove Docker image (~20GB)"
echo "  $0 cleanup --all    # Remove everything including downloads"
echo ""

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# =============================================================================
# Stage -1: Clone CosyVoice Repository
# =============================================================================
if [[ $START_STAGE -le -1 ]] && [[ $STOP_STAGE -ge -1 ]]; then
    log_info "Stage -1: Cloning CosyVoice repository..."

    # Check if repo exists AND has content (not empty dir)
    if [ ! -d "CosyVoice/.git" ]; then
        # Remove empty/broken directory if it exists
        rm -rf CosyVoice 2>/dev/null || true

        log_info "Cloning ${COSYVOICE_REPO}..."
        git clone --recursive "${COSYVOICE_REPO}" || {
            log_error "Failed to clone CosyVoice repository"
            exit 1
        }
        cd CosyVoice
        git submodule update --init --recursive
        cd ..
        log_info "CosyVoice cloned successfully"
    else
        log_info "CosyVoice repo already exists"
    fi
fi

# =============================================================================
# Stage 0: Download Models
# =============================================================================
if [[ $START_STAGE -le 0 ]] && [[ $STOP_STAGE -ge 0 ]]; then
    log_info "Stage 0: Downloading CosyVoice2-0.5B models..."

    # Download LLM checkpoint from HuggingFace
    if [ ! -d "cosyvoice2_llm" ] || [ ! -f "cosyvoice2_llm/config.json" ]; then
        log_info "Downloading LLM from HuggingFace: ${HF_MODEL}"

        mkdir -p cosyvoice2_llm

        # Method 1: git clone (most reliable for LFS files)
        if command -v git &> /dev/null; then
            git lfs install 2>/dev/null || true

            if [ -n "$HF_TOKEN" ]; then
                GIT_URL="https://USER:${HF_TOKEN}@huggingface.co/${HF_MODEL}"
            else
                GIT_URL="https://huggingface.co/${HF_MODEL}"
            fi

            git clone --depth 1 "$GIT_URL" ./cosyvoice2_llm && log_info "LLM downloaded via git"
        fi

        # Method 2: huggingface-cli fallback
        if [ ! -f "cosyvoice2_llm/config.json" ] && command -v huggingface-cli &> /dev/null; then
            log_info "Trying huggingface-cli..."
            if [ -n "$HF_TOKEN" ]; then
                huggingface-cli download --local-dir ./cosyvoice2_llm "${HF_MODEL}" --token "$HF_TOKEN"
            else
                huggingface-cli download --local-dir ./cosyvoice2_llm "${HF_MODEL}"
            fi
        fi

        if [ ! -f "cosyvoice2_llm/config.json" ]; then
            log_error "Could not download LLM. Please install git-lfs:"
            log_info "  sudo yum install git-lfs && git lfs install"
            exit 1
        fi
    else
        log_info "LLM model already downloaded"
    fi

    # Download full model from ModelScope
    if [ ! -d "CosyVoice2-0.5B" ] || [ ! -f "CosyVoice2-0.5B/flow.pt" ]; then
        log_info "Downloading from ModelScope: ${MODELSCOPE_MODEL}"

        mkdir -p CosyVoice2-0.5B

        # Method 1: modelscope CLI
        if command -v modelscope &> /dev/null; then
            modelscope download --model "${MODELSCOPE_MODEL}" --local_dir ./CosyVoice2-0.5B
        # Method 2: git clone from ModelScope
        elif command -v git &> /dev/null; then
            log_info "Using git clone from ModelScope..."
            git lfs install 2>/dev/null || true
            git clone --depth 1 "https://www.modelscope.cn/${MODELSCOPE_MODEL}.git" ./CosyVoice2-0.5B || {
                log_warn "ModelScope git clone failed"
            }
        fi

        # Method 3: Manual instructions
        if [ ! -f "CosyVoice2-0.5B/flow.pt" ]; then
            log_warn "Could not auto-download from ModelScope"
            log_info ""
            log_info "Please download manually:"
            log_info "  1. Visit: https://modelscope.cn/models/${MODELSCOPE_MODEL}"
            log_info "  2. Download and extract to: ${WORK_DIR}/CosyVoice2-0.5B/"
            log_info ""
            log_info "Or install modelscope CLI:"
            log_info "  pip install modelscope"
            log_info "  modelscope download --model ${MODELSCOPE_MODEL} --local_dir ./CosyVoice2-0.5B"
        fi
    else
        log_info "ModelScope model already downloaded"
    fi

    # Download speaker cache
    if [ -d "CosyVoice2-0.5B" ] && [ ! -f "CosyVoice2-0.5B/spk2info.pt" ]; then
        log_info "Downloading speaker info cache..."
        curl -L -o ./CosyVoice2-0.5B/spk2info.pt \
            "https://raw.githubusercontent.com/qi-hua/async_cosyvoice/main/CosyVoice2-0.5B/spk2info.pt" \
            2>/dev/null || log_warn "Could not download spk2info.pt"
    fi
fi

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
# Stage 1: Convert to TensorRT-LLM and Build Engines
# =============================================================================
if [[ $START_STAGE -le 1 ]] && [[ $STOP_STAGE -ge 1 ]]; then
    log_info "Stage 1: Converting checkpoint and building TensorRT engines..."

    TRT_WEIGHTS_DIR="./trt_weights_${TRT_DTYPE}"
    TRT_ENGINES_DIR="./trt_engines_${TRT_DTYPE}"

    # Ensure we have the container
    pull_container

    # Run inside TensorRT-LLM development container
    # Uses CosyVoice's custom convert_checkpoint.py + trtllm-build
    log_info "Starting build in container ${TRTLLM_IMAGE}..."

    docker run --rm --gpus all \
        --name "${TRTLLM_CONTAINER_NAME}" \
        --shm-size=8g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${WORK_DIR}:/workspace/build" \
        -v "${WORK_DIR}/CosyVoice:/workspace/CosyVoice" \
        -w /workspace/build \
        -e PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS \
        "${TRTLLM_IMAGE}" \
        bash -c "
            set -e

            echo '=== TensorRT-LLM Version Check ==='
            python3 -c 'import tensorrt_llm; print(f\"TRT-LLM version: {tensorrt_llm.__version__}\")'

            # Convert checkpoint to TensorRT weights using CosyVoice's script
            echo 'Converting checkpoint to TensorRT weights...'
            python3 /workspace/CosyVoice/runtime/triton_trtllm/scripts/convert_checkpoint.py \
                --model_dir ./cosyvoice2_llm \
                --output_dir ${TRT_WEIGHTS_DIR} \
                --dtype ${TRT_DTYPE}

            # Build TensorRT engines
            echo 'Building TensorRT engines...'
            trtllm-build \
                --checkpoint_dir ${TRT_WEIGHTS_DIR} \
                --output_dir ${TRT_ENGINES_DIR} \
                --max_batch_size 16 \
                --max_num_tokens 32768 \
                --gemm_plugin ${TRT_DTYPE}

            echo 'TensorRT engines built successfully!'
        "

    log_info "Stage 1 complete: Engines at ${WORK_DIR}/trt_engines_${TRT_DTYPE}"
fi

# =============================================================================
# Stage 2: Create Model Repository
# =============================================================================
if [[ $START_STAGE -le 2 ]] && [[ $STOP_STAGE -ge 2 ]]; then
    log_info "Stage 2: Creating Triton model repository..."

    MODEL_REPO="${DEPLOY_DIR}/model_repository/cosyvoice2_full"
    rm -rf "${MODEL_REPO}"
    mkdir -p "${MODEL_REPO}"

    # Copy model repository templates from CosyVoice
    COSYVOICE_TRITON="${WORK_DIR}/CosyVoice/runtime/triton_trtllm/model_repo"

    if [ -d "${COSYVOICE_TRITON}" ]; then
        cp -r "${COSYVOICE_TRITON}/cosyvoice2" "${MODEL_REPO}/"
        cp -r "${COSYVOICE_TRITON}/tensorrt_llm" "${MODEL_REPO}/"
        cp -r "${COSYVOICE_TRITON}/token2wav" "${MODEL_REPO}/"
        cp -r "${COSYVOICE_TRITON}/audio_tokenizer" "${MODEL_REPO}/"
        cp -r "${COSYVOICE_TRITON}/speaker_embedding" "${MODEL_REPO}/"
    else
        log_error "CosyVoice model_repo templates not found at ${COSYVOICE_TRITON}"
        log_error "Run with stage -1 first to clone the repository"
        exit 1
    fi

    # Configuration - paths relative to container mount point /models/cosyvoice2_full
    # These get baked into the config.pbtxt files
    ENGINE_PATH="/models/cosyvoice2_full/tensorrt_llm/1/engine"
    MODEL_DIR="/models/cosyvoice2_full/assets"
    LLM_TOKENIZER_DIR="/models/cosyvoice2_full/assets/cosyvoice2_llm"
    MAX_QUEUE_DELAY_MICROSECONDS=0
    BLS_INSTANCE_NUM=4
    TRITON_MAX_BATCH_SIZE=16
    DECOUPLED_MODE=True  # True for streaming, False for offline

    # Fill templates
    log_info "Filling config templates..."
    cd "${WORK_DIR}/CosyVoice/runtime/triton_trtllm"

    python3 scripts/fill_template.py -i "${MODEL_REPO}/token2wav/config.pbtxt" \
        "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}"

    python3 scripts/fill_template.py -i "${MODEL_REPO}/cosyvoice2/config.pbtxt" \
        "model_dir:${MODEL_DIR},bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}"

    python3 scripts/fill_template.py -i "${MODEL_REPO}/tensorrt_llm/config.pbtxt" \
        "triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"

    python3 scripts/fill_template.py -i "${MODEL_REPO}/audio_tokenizer/config.pbtxt" \
        "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}"

    python3 scripts/fill_template.py -i "${MODEL_REPO}/speaker_embedding/config.pbtxt" \
        "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}"

    # Copy assets into model repo for self-contained deployment
    log_info "Copying assets into model repository..."
    mkdir -p "${MODEL_REPO}/assets"

    # Copy CosyVoice model files
    cp -r "${WORK_DIR}/CosyVoice2-0.5B"/* "${MODEL_REPO}/assets/" 2>/dev/null || true

    # Copy LLM tokenizer
    mkdir -p "${MODEL_REPO}/assets/cosyvoice2_llm"
    cp -r "${WORK_DIR}/cosyvoice2_llm"/* "${MODEL_REPO}/assets/cosyvoice2_llm/" 2>/dev/null || true

    # Copy TensorRT engines into the tensorrt_llm model version directory
    mkdir -p "${MODEL_REPO}/tensorrt_llm/1/engine"
    cp -r "${WORK_DIR}/trt_engines_${TRT_DTYPE}"/* "${MODEL_REPO}/tensorrt_llm/1/engine/" 2>/dev/null || true

    log_info "Stage 2 complete: Model repo at ${MODEL_REPO}"
fi

# =============================================================================
# Stage 3: Test the Setup
# =============================================================================
if [[ $START_STAGE -le 3 ]] && [[ $STOP_STAGE -ge 3 ]]; then
    log_info "Stage 3: Model repository ready for Triton"
    log_info ""
    log_info "To start Triton with CosyVoice2:"
    log_info "  tritonserver --model-repository=${DEPLOY_DIR}/model_repository/cosyvoice2_full"
    log_info ""
    log_info "Or update docker-compose.yml to use the cosyvoice2_full model repo"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}CosyVoice 2 Build Complete${NC}"
echo "=============================================="
echo ""
echo "Model components:"
ls -la "${DEPLOY_DIR}/model_repository/cosyvoice2_full/" 2>/dev/null || echo "Run stages 0-2 to build"
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
