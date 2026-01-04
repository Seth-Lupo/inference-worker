#!/bin/bash
# =============================================================================
# Build Chatterbox Turbo TensorRT Engines (TTS Model)
#
# Downloads Chatterbox-Turbo-ONNX from HuggingFace, builds TRT engines,
# and creates Triton model repository with 5 models:
#   - chatterbox (BLS orchestrator - Python backend)
#   - embed_tokens (TensorRT)
#   - language_model (TensorRT with KV cache)
#   - speech_encoder (TensorRT)
#   - conditional_decoder (TensorRT)
#
# Usage:
#   ./build_chatterbox.sh [START_STAGE] [STOP_STAGE]
#   ./build_chatterbox.sh 0 2       # Run all stages
#   ./build_chatterbox.sh 1 1       # Only build TRT engines
#   ./build_chatterbox.sh cleanup   # Clean up build artifacts
#
# Stages:
#    0: Download ONNX models from HuggingFace
#    1: Build ONNX to TensorRT engines
#    2: Create Triton model repository
#
# Environment Variables:
#   HF_TOKEN              - HuggingFace token (optional)
#   CHATTERBOX_PRECISION  - fp16, fp32 (default: fp16)
#
# Sources:
#   - https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================
HF_REPO="${HF_REPO:-ResembleAI/chatterbox-turbo-ONNX}"
PRECISION="${CHATTERBOX_PRECISION:-fp16}"

# TensorRT image for building
TRT_IMAGE="${TRT_IMAGE:-$(cfg_get 'images.tensorrt' 'nvcr.io/nvidia/tensorrt:25.09-py3')}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/chatterbox_build"
readonly MODEL_REPO="${DEPLOY_DIR}/model_repository/tts"

# ONNX model files based on precision
if [[ "$PRECISION" == "fp16" ]]; then
    EMBED_TOKENS_ONNX="embed_tokens_fp16.onnx"
    LANGUAGE_MODEL_ONNX="language_model_fp16.onnx"
    SPEECH_ENCODER_ONNX="speech_encoder_fp16.onnx"
    CONDITIONAL_DECODER_ONNX="conditional_decoder_fp16.onnx"
else
    EMBED_TOKENS_ONNX="embed_tokens.onnx"
    LANGUAGE_MODEL_ONNX="language_model.onnx"
    SPEECH_ENCODER_ONNX="speech_encoder.onnx"
    CONDITIONAL_DECODER_ONNX="conditional_decoder.onnx"
fi

# Dynamic shapes for each model
# embed_tokens: input_ids (batch, seq) -> embeddings (batch, seq, hidden)
EMBED_TOKENS_MIN="input_ids:1x1"
EMBED_TOKENS_OPT="input_ids:4x256"
EMBED_TOKENS_MAX="input_ids:8x1024"

# language_model: autoregressive with KV cache
# Inputs: input_embeds, attention_mask, position_ids, past_key_values (optional)
# For first pass (no cache): full sequence
# For subsequent passes: single token with cache
LM_MIN="inputs_embeds:1x1x896,attention_mask:1x1,position_ids:1x1"
LM_OPT="inputs_embeds:4x256x896,attention_mask:4x256,position_ids:4x256"
LM_MAX="inputs_embeds:8x1024x896,attention_mask:8x1024,position_ids:8x1024"

# speech_encoder: reference audio -> conditioning
# Input: audio features (batch, time, features)
SPEECH_ENC_MIN="input_features:1x1x80"
SPEECH_ENC_OPT="input_features:1x500x80"
SPEECH_ENC_MAX="input_features:1x3000x80"

# conditional_decoder: single-step mel generation
# Input: speech_tokens, conditioning
COND_DEC_MIN="input_ids:1x1"
COND_DEC_OPT="input_ids:1x256"
COND_DEC_MAX="input_ids:1x1024"

# =============================================================================
# Load environment
# =============================================================================
load_env

# =============================================================================
# Cleanup Handler
# =============================================================================
if [[ "${1:-}" == "cleanup" || "${1:-}" == "clean" ]]; then
    echo "=============================================="
    echo "Cleaning up Chatterbox build artifacts"
    echo "=============================================="

    case "${2:-}" in
        --all|-a)
            log_warn "Removing ALL build artifacts..."
            rm -rf "$WORK_DIR"
            rm -rf "${MODEL_REPO}/chatterbox"
            rm -rf "${MODEL_REPO}/chatterbox_assets"
            log_info "Cleanup complete"
            ;;
        *)
            log_info "Removing downloads (keeping engines)..."
            rm -rf "${WORK_DIR}/onnx"
            log_info "Kept: ${WORK_DIR}/engines"
            ;;
    esac
    exit 0
fi

# Parse stages
START_STAGE="${1:-0}"
STOP_STAGE="${2:-2}"

echo "=============================================="
echo "Building Chatterbox Turbo TensorRT"
echo "=============================================="
echo "Stages: ${START_STAGE} to ${STOP_STAGE}"
echo "Precision: ${PRECISION}"
echo "TRT Image: ${TRT_IMAGE}"
echo ""

mkdir -p "$WORK_DIR"

# =============================================================================
# Stage 0: Download ONNX Models
# =============================================================================
stage_download_models() {
    log_step "Stage 0: Downloading ONNX models from HuggingFace..."

    local onnx_dir="${WORK_DIR}/onnx"

    # Check if models already exist
    if [[ -f "${onnx_dir}/${LANGUAGE_MODEL_ONNX}" ]]; then
        log_info "ONNX models already downloaded"
        return 0
    fi

    mkdir -p "$onnx_dir"

    # Download using huggingface-hub
    log_info "Downloading from ${HF_REPO}..."

    # Check if huggingface_hub is available
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 << EOF
from huggingface_hub import hf_hub_download, list_repo_files
import os

repo_id = "${HF_REPO}"
local_dir = "${onnx_dir}"
precision = "${PRECISION}"

# Files to download based on precision
suffix = "_fp16" if precision == "fp16" else ""
files = [
    f"onnx/embed_tokens{suffix}.onnx",
    f"onnx/embed_tokens{suffix}.onnx_data",
    f"onnx/language_model{suffix}.onnx",
    f"onnx/language_model{suffix}.onnx_data",
    f"onnx/speech_encoder{suffix}.onnx",
    f"onnx/speech_encoder{suffix}.onnx_data",
    f"onnx/conditional_decoder{suffix}.onnx",
    f"onnx/conditional_decoder{suffix}.onnx_data",
]

# Also download config files
config_files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

print(f"Downloading {len(files)} ONNX files...")
for f in files:
    print(f"  {f}")
    hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)

print(f"Downloading config files...")
for f in config_files:
    print(f"  {f}")
    hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)

print("Download complete!")
EOF
    else
        log_error "huggingface_hub not installed. Install with: pip install huggingface_hub"
        return 1
    fi

    # Move ONNX files to root of onnx_dir for easier access
    if [[ -d "${onnx_dir}/onnx" ]]; then
        mv "${onnx_dir}/onnx"/* "${onnx_dir}/"
        rmdir "${onnx_dir}/onnx"
    fi

    log_info "ONNX models downloaded to ${onnx_dir}"
}

# =============================================================================
# Stage 1: Build TensorRT Engines
# =============================================================================
stage_build_engines() {
    log_step "Stage 1: Building TensorRT engines..."

    local onnx_dir="${WORK_DIR}/onnx"
    local engine_dir="${WORK_DIR}/engines"

    if [[ ! -d "$onnx_dir" ]]; then
        log_error "ONNX models not found. Run stage 0 first."
        return 1
    fi

    mkdir -p "$engine_dir"

    # Ensure Docker image exists
    ensure_docker_image "$TRT_IMAGE"

    # Build embed_tokens engine
    log_info "Building embed_tokens engine..."
    build_trt_engine \
        "${onnx_dir}/${EMBED_TOKENS_ONNX}" \
        "${engine_dir}/embed_tokens.engine" \
        --fp16 \
        "--minShapes=${EMBED_TOKENS_MIN}" \
        "--optShapes=${EMBED_TOKENS_OPT}" \
        "--maxShapes=${EMBED_TOKENS_MAX}"

    # Build speech_encoder engine
    log_info "Building speech_encoder engine..."
    build_trt_engine \
        "${onnx_dir}/${SPEECH_ENCODER_ONNX}" \
        "${engine_dir}/speech_encoder.engine" \
        --fp16 \
        "--minShapes=${SPEECH_ENC_MIN}" \
        "--optShapes=${SPEECH_ENC_OPT}" \
        "--maxShapes=${SPEECH_ENC_MAX}"

    # Build conditional_decoder engine (single-step, easier)
    log_info "Building conditional_decoder engine..."
    build_trt_engine \
        "${onnx_dir}/${CONDITIONAL_DECODER_ONNX}" \
        "${engine_dir}/conditional_decoder.engine" \
        --fp16 \
        "--minShapes=${COND_DEC_MIN}" \
        "--optShapes=${COND_DEC_OPT}" \
        "--maxShapes=${COND_DEC_MAX}"

    # Build language_model engine (largest, most complex)
    # Note: This model has KV cache I/O which needs special handling
    log_info "Building language_model engine (this may take 5-10 minutes)..."
    build_trt_engine \
        "${onnx_dir}/${LANGUAGE_MODEL_ONNX}" \
        "${engine_dir}/language_model.engine" \
        --fp16 \
        --workspace=8192 \
        "--minShapes=${LM_MIN}" \
        "--optShapes=${LM_OPT}" \
        "--maxShapes=${LM_MAX}"

    log_info "All TensorRT engines built successfully"
    ls -lh "${engine_dir}"/*.engine
}

# =============================================================================
# Stage 2: Create Model Repository
# =============================================================================
stage_create_model_repo() {
    log_step "Stage 2: Setting up Triton model repository..."

    local engine_dir="${WORK_DIR}/engines"
    local onnx_dir="${WORK_DIR}/onnx"
    local assets_dir="${MODEL_REPO}/chatterbox_assets"

    # Create directories
    mkdir -p "${assets_dir}"
    mkdir -p "${MODEL_REPO}/chatterbox/1"
    mkdir -p "${MODEL_REPO}/chatterbox_lm/1"
    mkdir -p "${MODEL_REPO}/chatterbox_encoder/1"
    mkdir -p "${MODEL_REPO}/chatterbox_decoder/1"
    mkdir -p "${MODEL_REPO}/chatterbox_embed/1"

    # Copy engines
    log_info "Copying TensorRT engines..."
    cp "${engine_dir}/language_model.engine" "${MODEL_REPO}/chatterbox_lm/1/" 2>/dev/null || true
    cp "${engine_dir}/speech_encoder.engine" "${MODEL_REPO}/chatterbox_encoder/1/" 2>/dev/null || true
    cp "${engine_dir}/conditional_decoder.engine" "${MODEL_REPO}/chatterbox_decoder/1/" 2>/dev/null || true
    cp "${engine_dir}/embed_tokens.engine" "${MODEL_REPO}/chatterbox_embed/1/" 2>/dev/null || true

    # Copy config files
    log_info "Copying configuration files..."
    cp "${onnx_dir}/config.json" "${assets_dir}/" 2>/dev/null || true
    cp "${onnx_dir}/tokenizer.json" "${assets_dir}/" 2>/dev/null || true
    cp "${onnx_dir}/tokenizer_config.json" "${assets_dir}/" 2>/dev/null || true

    log_info "Model repository created at ${MODEL_REPO}"
    log_info "Models:"
    ls -1 "$MODEL_REPO" | grep chatterbox || echo "  (models not yet created)"
}

# =============================================================================
# Main
# =============================================================================
main() {
    # Stage 0: Download models
    if [[ $START_STAGE -le 0 && $STOP_STAGE -ge 0 ]]; then
        stage_download_models
    fi

    # Stage 1: Build TRT engines
    if [[ $START_STAGE -le 1 && $STOP_STAGE -ge 1 ]]; then
        stage_build_engines
    fi

    # Stage 2: Create model repo
    if [[ $START_STAGE -le 2 && $STOP_STAGE -ge 2 ]]; then
        stage_create_model_repo
    fi

    echo ""
    echo "=============================================="
    echo -e "${GREEN}Chatterbox Turbo Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Model repository: ${MODEL_REPO}"
    echo ""
    echo "Next steps:"
    echo "  1. Ensure Triton model configs are in place"
    echo "  2. Start Triton: docker compose up -d triton"
    echo ""
    echo "Cleanup options:"
    echo "  $0 cleanup           # Remove downloads, keep engines"
    echo "  $0 cleanup --all     # Remove everything"
}

main
