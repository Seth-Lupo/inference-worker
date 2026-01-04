#!/bin/bash
# =============================================================================
# Build Parakeet TDT 0.6B V2 for Triton (ASR Model)
#
# Architecture:
#   parakeet_tdt (BLS Python backend - orchestrator)
#     ├── parakeet_encoder (native tensorrt_plan backend)
#     └── parakeet_decoder (native tensorrt_plan backend)
#
# Downloads ONNX models from HuggingFace, builds TensorRT engines,
# and creates Triton model repository with native TRT backends.
#
# Usage:
#   ./build_parakeet.sh [START_STAGE] [STOP_STAGE]
#   ./build_parakeet.sh 0 2      # Run all stages
#   ./build_parakeet.sh 1 1      # Only build TRT engines
#   ./build_parakeet.sh cleanup  # Clean up build artifacts
#
# Stages:
#   0: Download ONNX models from HuggingFace
#   1: Build TensorRT engines (encoder, decoder_joint)
#   2: Create Triton model repository
#
# Source: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx
# =============================================================================

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration (from config.yaml, env vars override)
# =============================================================================
HF_REPO="${HF_REPO:-$(cfg_get 'parakeet.hf_repo' 'istupakov/parakeet-tdt-0.6b-v2-onnx')}"

# TRT build settings
TRT_PRECISION="${TRT_PRECISION:-$(cfg_get 'parakeet.precision' 'fp16')}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-$(cfg_get 'parakeet.max_batch_size' '8')}"

# Dynamic shapes from config
ENCODER_MIN_SHAPES="$(cfg_get 'parakeet.shapes.encoder.min' 'audio_signal:1x128x10,length:1')"
ENCODER_OPT_SHAPES="$(cfg_get 'parakeet.shapes.encoder.opt' 'audio_signal:1x128x1000,length:1')"
ENCODER_MAX_SHAPES="$(cfg_get 'parakeet.shapes.encoder.max' 'audio_signal:1x128x3000,length:1')"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/parakeet_build"
readonly ONNX_DIR="${WORK_DIR}/onnx"

# Model directories for native TRT backends
readonly ENCODER_DIR="${DEPLOY_DIR}/model_repository/asr/parakeet_encoder"
readonly DECODER_DIR="${DEPLOY_DIR}/model_repository/asr/parakeet_decoder"
readonly BLS_DIR="${DEPLOY_DIR}/model_repository/asr/parakeet_tdt"

# =============================================================================
# Cleanup Handler
# =============================================================================
if [[ "${1:-}" == "cleanup" || "${1:-}" == "clean" ]]; then
    echo "=============================================="
    echo "Cleaning up Parakeet build artifacts"
    echo "=============================================="

    case "${2:-}" in
        --all|-a)
            log_warn "Removing all build artifacts..."
            rm -rf "$WORK_DIR"
            rm -f "${ENCODER_DIR}/1/model.plan"
            rm -f "${DECODER_DIR}/1/model.plan"
            rm -f "${BLS_DIR}/1/vocab.txt"
            log_info "Cleanup complete"
            ;;
        --engines|-e)
            log_info "Removing TRT engines only..."
            rm -f "${ENCODER_DIR}/1/model.plan"
            rm -f "${DECODER_DIR}/1/model.plan"
            log_info "Engines removed"
            ;;
        *)
            log_info "Removing work directory (keeping model & engines)..."
            rm -rf "$WORK_DIR"
            log_info "Cleanup complete"
            log_info "To also remove model: $0 cleanup --all"
            log_info "To rebuild engines:   $0 cleanup --engines"
            ;;
    esac
    exit 0
fi

# Parse stages
START_STAGE="${1:-0}"
STOP_STAGE="${2:-2}"

echo "=============================================="
echo "Building Parakeet TDT 0.6B V2"
echo "=============================================="
echo "Architecture: BLS + Native TensorRT Backends"
echo "Stages: ${START_STAGE} to ${STOP_STAGE}"
echo ""

# =============================================================================
# Stage 0: Download Models from HuggingFace
# =============================================================================

stage_download_models() {
    log_step "Stage 0: Downloading Parakeet from HuggingFace..."

    mkdir -p "$ONNX_DIR"

    # Check if already downloaded (encoder is the large file)
    if [[ -f "${ONNX_DIR}/encoder-model.onnx" ]] && is_real_file "${ONNX_DIR}/encoder-model.onnx.data" 1000000; then
        log_info "Models already downloaded at: $ONNX_DIR"
        return 0
    fi

    # Clone with LFS
    log_info "Cloning ${HF_REPO}..."
    hf_clone "$HF_REPO" "$ONNX_DIR" || {
        log_error "Failed to clone from HuggingFace"
        return 1
    }

    # Verify key files exist
    local required_files=("encoder-model.onnx" "encoder-model.onnx.data" "decoder_joint-model.onnx" "vocab.txt")
    for file in "${required_files[@]}"; do
        if [[ ! -f "${ONNX_DIR}/${file}" ]]; then
            log_error "Missing required file: ${file}"
            ls -la "$ONNX_DIR"
            return 1
        fi
    done

    log_info "Downloaded ONNX models to: $ONNX_DIR"
}

# =============================================================================
# Stage 1: Build TensorRT Engines
# =============================================================================

stage_build_trt_engines() {
    log_step "Stage 1: Building TensorRT engines for native backends..."

    # Verify ONNX models exist
    if [[ ! -f "${ONNX_DIR}/encoder-model.onnx" ]]; then
        log_error "ONNX models not found. Run stage 0 first."
        return 1
    fi

    mkdir -p "${ENCODER_DIR}/1"
    mkdir -p "${DECODER_DIR}/1"

    # Determine precision flag
    local precision_flag="--fp16"
    [[ "$TRT_PRECISION" == "fp32" ]] && precision_flag="--fp32"

    # Build encoder engine -> parakeet_encoder/1/model.plan
    log_info "Building encoder engine (native TRT backend)..."
    if [[ ! -f "${ENCODER_DIR}/1/model.plan" ]]; then
        build_trt_engine \
            "${ONNX_DIR}/encoder-model.onnx" \
            "${ENCODER_DIR}/1/model.plan" \
            "$precision_flag" \
            "--minShapes=${ENCODER_MIN_SHAPES}" \
            "--optShapes=${ENCODER_OPT_SHAPES}" \
            "--maxShapes=${ENCODER_MAX_SHAPES}"
    else
        log_info "Encoder engine already exists"
    fi

    # Build decoder_joint engine -> parakeet_decoder/1/model.plan
    log_info "Building decoder_joint engine (native TRT backend)..."
    if [[ ! -f "${DECODER_DIR}/1/model.plan" ]]; then
        build_trt_engine \
            "${ONNX_DIR}/decoder_joint-model.onnx" \
            "${DECODER_DIR}/1/model.plan" \
            "$precision_flag"
    else
        log_info "Decoder engine already exists"
    fi

    log_info "TRT engines built for native backends"
}

# =============================================================================
# Stage 2: Setup Model Repository
# =============================================================================

stage_setup_model_repo() {
    log_step "Stage 2: Setting up Triton model repository..."

    if [[ ! -d "$ONNX_DIR" ]]; then
        log_error "ONNX models not found. Run stage 0 first."
        return 1
    fi

    # Copy vocab to BLS model
    mkdir -p "${BLS_DIR}/1"
    cp "${ONNX_DIR}/vocab.txt" "${BLS_DIR}/1/"
    log_info "Copied vocab.txt to BLS model"

    # Verify all components
    log_info "Verifying model repository..."

    local errors=0

    if [[ -f "${ENCODER_DIR}/1/model.plan" ]]; then
        log_info "  ✓ parakeet_encoder/1/model.plan"
    else
        log_error "  ✗ parakeet_encoder/1/model.plan MISSING"
        ((errors++))
    fi

    if [[ -f "${DECODER_DIR}/1/model.plan" ]]; then
        log_info "  ✓ parakeet_decoder/1/model.plan"
    else
        log_error "  ✗ parakeet_decoder/1/model.plan MISSING"
        ((errors++))
    fi

    if [[ -f "${BLS_DIR}/1/model.py" ]]; then
        log_info "  ✓ parakeet_tdt/1/model.py (BLS)"
    else
        log_error "  ✗ parakeet_tdt/1/model.py MISSING"
        ((errors++))
    fi

    if [[ -f "${BLS_DIR}/1/vocab.txt" ]]; then
        log_info "  ✓ parakeet_tdt/1/vocab.txt"
    else
        log_error "  ✗ parakeet_tdt/1/vocab.txt MISSING"
        ((errors++))
    fi

    if [[ $errors -gt 0 ]]; then
        log_error "Model repository incomplete!"
        return 1
    fi

    log_info "Model repository ready"
}

show_summary() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}Parakeet TDT Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Architecture: BLS + Native TensorRT Backends"
    echo ""
    echo "Models:"
    echo "  parakeet_encoder (tensorrt_plan):"
    ls -lh "${ENCODER_DIR}/1/model.plan" 2>/dev/null || echo "    (not built)"
    echo ""
    echo "  parakeet_decoder (tensorrt_plan):"
    ls -lh "${DECODER_DIR}/1/model.plan" 2>/dev/null || echo "    (not built)"
    echo ""
    echo "  parakeet_tdt (BLS orchestrator):"
    ls -l "${BLS_DIR}/1/model.py" 2>/dev/null || echo "    (not found)"
    echo ""
    echo "To start Triton (load all 3 models):"
    echo "  docker compose up -d triton"
    echo ""
    echo "Cleanup options:"
    echo "  $0 cleanup            # Remove downloads, keep engines"
    echo "  $0 cleanup --all      # Remove everything"
    echo "  $0 cleanup --engines  # Rebuild TRT engines only"
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
        stage_build_trt_engines
    fi

    # Stage 2: Setup model repo
    if [[ $START_STAGE -le 2 && $STOP_STAGE -ge 2 ]]; then
        stage_setup_model_repo
    fi

    show_summary
}

main
