#!/bin/bash
# =============================================================================
# Build CosyVoice 2 TensorRT-LLM Engines
# Based on: https://github.com/FunAudioLLM/CosyVoice/blob/main/runtime/triton_trtllm/run.sh
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

# Load environment
if [ -f "${DEPLOY_DIR}/.env" ]; then
    source "${DEPLOY_DIR}/.env"
fi

# Configuration
COSYVOICE_REPO="https://github.com/FunAudioLLM/CosyVoice.git"
HF_MODEL="yuekai/cosyvoice2_llm"
MODELSCOPE_MODEL="iic/CosyVoice2-0.5B"
TRT_DTYPE="bfloat16"

# Stages to run (default: all)
START_STAGE=${1:-0}
STOP_STAGE=${2:-3}

echo "=============================================="
echo "Building CosyVoice 2 TensorRT-LLM"
echo "=============================================="
echo "Stages: ${START_STAGE} to ${STOP_STAGE}"
echo ""

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# =============================================================================
# Stage -1: Clone CosyVoice Repository
# =============================================================================
if [ $START_STAGE -le -1 ] && [ $STOP_STAGE -ge -1 ]; then
    log_info "Stage -1: Cloning CosyVoice repository..."

    if [ ! -d "CosyVoice" ]; then
        git clone --recursive "${COSYVOICE_REPO}"
        cd CosyVoice
        git submodule update --init --recursive
        cd ..
    else
        log_info "CosyVoice repo already exists"
    fi
fi

# =============================================================================
# Stage 0: Download Models
# =============================================================================
if [ $START_STAGE -le 0 ] && [ $STOP_STAGE -ge 0 ]; then
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
# Stage 1: Convert to TensorRT-LLM and Build Engines
# =============================================================================
if [ $START_STAGE -le 1 ] && [ $STOP_STAGE -ge 1 ]; then
    log_info "Stage 1: Converting checkpoint and building TensorRT engines..."

    TRT_WEIGHTS_DIR="./trt_weights_${TRT_DTYPE}"
    TRT_ENGINES_DIR="./trt_engines_${TRT_DTYPE}"

    # This must run inside the Triton container with TensorRT-LLM
    docker run --rm --gpus all \
        --shm-size=8g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${WORK_DIR}:/workspace/build" \
        -v "${WORK_DIR}/CosyVoice:/workspace/CosyVoice" \
        -w /workspace/build \
        -e PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS \
        nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 \
        bash -c "
            set -e

            # Convert checkpoint to TensorRT weights
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
if [ $START_STAGE -le 2 ] && [ $STOP_STAGE -ge 2 ]; then
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

    # Configuration
    ENGINE_PATH="${WORK_DIR}/trt_engines_${TRT_DTYPE}"
    MODEL_DIR="${WORK_DIR}/CosyVoice2-0.5B"
    LLM_TOKENIZER_DIR="${WORK_DIR}/cosyvoice2_llm"
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

    log_info "Stage 2 complete: Model repo at ${MODEL_REPO}"
fi

# =============================================================================
# Stage 3: Test the Setup
# =============================================================================
if [ $START_STAGE -le 3 ] && [ $STOP_STAGE -ge 3 ]; then
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
