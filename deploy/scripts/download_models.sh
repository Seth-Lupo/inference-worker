#!/bin/bash
# =============================================================================
# Download Models Script
# Downloads all required models for the voice agent
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_SOURCES="${DEPLOY_DIR}/model_sources"
MODEL_REPO="${DEPLOY_DIR}/model_repository"

echo "=============================================="
echo "Downloading Voice Agent Models"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Load environment
if [ -f "${DEPLOY_DIR}/.env" ]; then
    source "${DEPLOY_DIR}/.env"
fi

# Check HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    log_warn "HF_TOKEN not set. Some models may require authentication."
    log_warn "Set it in ${DEPLOY_DIR}/.env or export HF_TOKEN=your_token"
fi

mkdir -p "${MODEL_SOURCES}"
cd "${MODEL_SOURCES}"

# -----------------------------------------------------------------------------
# 1. Silero VAD
# -----------------------------------------------------------------------------
log_info "Downloading Silero VAD..."
SILERO_DIR="${MODEL_SOURCES}/silero_vad"
mkdir -p "${SILERO_DIR}"

if [ ! -f "${SILERO_DIR}/silero_vad.onnx" ]; then
    curl -L -o "${SILERO_DIR}/silero_vad.onnx" \
        "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    log_info "Silero VAD downloaded"
else
    log_info "Silero VAD already exists"
fi

# Copy to model repository
cp "${SILERO_DIR}/silero_vad.onnx" "${MODEL_REPO}/silero_vad/1/model.onnx"
log_info "Silero VAD copied to model repository"

# -----------------------------------------------------------------------------
# 2. Parakeet TDT 0.6B V2 (ONNX version)
# -----------------------------------------------------------------------------
log_info "Downloading Parakeet TDT 0.6B V2..."
PARAKEET_DIR="${MODEL_SOURCES}/parakeet_tdt"
mkdir -p "${PARAKEET_DIR}"

# Download from sherpa-onnx releases (pre-converted ONNX)
PARAKEET_VERSION="v1.10.30"
PARAKEET_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/${PARAKEET_VERSION}"

if [ ! -f "${PARAKEET_DIR}/encoder.onnx" ]; then
    log_info "Downloading Parakeet encoder..."
    curl -L -o "${PARAKEET_DIR}/encoder.int8.onnx" \
        "${PARAKEET_URL}/encoder.int8.onnx" 2>/dev/null || \
        log_warn "Could not download encoder. Check sherpa-onnx releases for the correct URL."

    curl -L -o "${PARAKEET_DIR}/decoder.onnx" \
        "${PARAKEET_URL}/decoder.onnx" 2>/dev/null || true

    curl -L -o "${PARAKEET_DIR}/joiner.onnx" \
        "${PARAKEET_URL}/joiner.onnx" 2>/dev/null || true

    curl -L -o "${PARAKEET_DIR}/tokens.txt" \
        "${PARAKEET_URL}/tokens.txt" 2>/dev/null || true
else
    log_info "Parakeet TDT already exists"
fi

# Alternative: Download from HuggingFace ONNX community
log_info "Alternatively, download from HuggingFace:"
echo "  huggingface-cli download onnx-community/parakeet-tdt-0.6b-v2-ONNX --local-dir ${PARAKEET_DIR}"

# -----------------------------------------------------------------------------
# 3. Qwen3 8B (for TensorRT-LLM conversion)
# -----------------------------------------------------------------------------
log_info "Downloading Qwen3 8B..."
QWEN_DIR="${MODEL_SOURCES}/qwen3_8b"
mkdir -p "${QWEN_DIR}"

if [ ! -d "${QWEN_DIR}/Qwen3-8B" ]; then
    if [ -n "$HF_TOKEN" ]; then
        log_info "Downloading from HuggingFace (this may take a while)..."
        # Using huggingface-cli or git-lfs
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download Qwen/Qwen3-8B --local-dir "${QWEN_DIR}/Qwen3-8B" --token "$HF_TOKEN"
        else
            log_warn "huggingface-cli not found. Install with: pip install huggingface_hub"
            echo "Manual download: https://huggingface.co/Qwen/Qwen3-8B"
        fi
    else
        log_warn "HF_TOKEN required to download Qwen3-8B"
        echo "Set HF_TOKEN and run again, or manually download from:"
        echo "  https://huggingface.co/Qwen/Qwen3-8B"
    fi
else
    log_info "Qwen3 8B already exists"
fi

# -----------------------------------------------------------------------------
# 4. CosyVoice 2 0.5B
# -----------------------------------------------------------------------------
log_info "Downloading CosyVoice 2..."
COSYVOICE_DIR="${MODEL_SOURCES}/cosyvoice2"
mkdir -p "${COSYVOICE_DIR}"

if [ ! -d "${COSYVOICE_DIR}/CosyVoice2-0.5B" ]; then
    if [ -n "$HF_TOKEN" ]; then
        log_info "Downloading CosyVoice2 from HuggingFace..."
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download FunAudioLLM/CosyVoice2-0.5B \
                --local-dir "${COSYVOICE_DIR}/CosyVoice2-0.5B" \
                --token "$HF_TOKEN"
        else
            log_warn "huggingface-cli not found"
        fi
    else
        log_warn "HF_TOKEN required for some CosyVoice models"
    fi
else
    log_info "CosyVoice 2 already exists"
fi

# Link to model repository
if [ -d "${COSYVOICE_DIR}/CosyVoice2-0.5B" ]; then
    mkdir -p "${MODEL_REPO}/cosyvoice2/1/pretrained_models"
    ln -sf "${COSYVOICE_DIR}/CosyVoice2-0.5B" "${MODEL_REPO}/cosyvoice2/1/pretrained_models/CosyVoice2-0.5B" 2>/dev/null || \
        cp -r "${COSYVOICE_DIR}/CosyVoice2-0.5B" "${MODEL_REPO}/cosyvoice2/1/pretrained_models/"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}Model Download Complete${NC}"
echo "=============================================="
echo ""
echo "Downloaded models:"
ls -la "${MODEL_SOURCES}"
echo ""
echo "Next steps:"
echo "  1. Build TensorRT engines: ./scripts/build_engines.sh"
echo "  2. Start services: docker-compose up -d"
