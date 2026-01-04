#!/bin/bash
# =============================================================================
# Build Parakeet TDT 0.6B V2 for Triton (ASR Model)
#
# Architecture:
#   parakeet_tdt (BLS Python backend - orchestrator)
#     ├── parakeet_encoder (Python + ONNX Runtime GPU)
#     └── parakeet_decoder (Python + ONNX Runtime GPU)
#
# Downloads ONNX models from HuggingFace and copies them to the model repository.
# ONNX Runtime handles GPU execution.
#
# Usage:
#   ./build_parakeet.sh              # Download and setup ONNX models
#   ./build_parakeet.sh cleanup      # Clean up downloaded files
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

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/parakeet_build"
readonly ONNX_DIR="${DEPLOY_DIR}/model_repository/asr/parakeet_onnx"

# Model directories
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
            log_warn "Removing all Parakeet files..."
            rm -rf "$WORK_DIR"
            rm -rf "$ONNX_DIR"
            rm -f "${BLS_DIR}/1/vocab.txt"
            log_info "Cleanup complete"
            ;;
        *)
            log_info "Removing work directory..."
            rm -rf "$WORK_DIR"
            log_info "Cleanup complete"
            log_info "To also remove ONNX models: $0 cleanup --all"
            ;;
    esac
    exit 0
fi

echo "=============================================="
echo "Building Parakeet TDT 0.6B V2 (ONNX Runtime)"
echo "=============================================="
echo "Architecture: BLS + ONNX Runtime GPU Backends"
echo "Source: ${HF_REPO}"
echo ""

# =============================================================================
# Stage 1: Download ONNX Models from HuggingFace
# =============================================================================
download_onnx_models() {
    log_step "Downloading Parakeet ONNX models from HuggingFace..."

    mkdir -p "$WORK_DIR"
    mkdir -p "$ONNX_DIR"

    # Check if already downloaded (encoder data file is the large one ~600MB)
    if [[ -f "${ONNX_DIR}/encoder-model.onnx" ]] && is_real_file "${ONNX_DIR}/encoder-model.onnx.data" 100000000; then
        log_info "ONNX models already present at: $ONNX_DIR"
        return 0
    fi

    # Clone with LFS
    log_info "Cloning ${HF_REPO}..."
    hf_clone "$HF_REPO" "${WORK_DIR}/onnx_repo" || {
        log_error "Failed to clone from HuggingFace"
        return 1
    }

    # Copy ONNX files to model repository
    local required_files=(
        "encoder-model.onnx"
        "encoder-model.onnx.data"
        "decoder_joint-model.onnx"
        "vocab.txt"
    )

    for file in "${required_files[@]}"; do
        if [[ -f "${WORK_DIR}/onnx_repo/${file}" ]]; then
            cp "${WORK_DIR}/onnx_repo/${file}" "${ONNX_DIR}/"
            log_info "  ✓ ${file}"
        else
            log_error "  ✗ ${file} not found in repo"
            return 1
        fi
    done

    log_info "ONNX models downloaded to: $ONNX_DIR"
}

# =============================================================================
# Stage 2: Setup Model Repository
# =============================================================================
setup_model_repo() {
    log_step "Setting up Triton model repository..."

    # Ensure directories exist
    mkdir -p "${ENCODER_DIR}/1"
    mkdir -p "${DECODER_DIR}/1"
    mkdir -p "${BLS_DIR}/1"

    # Copy vocab to BLS model directory
    if [[ -f "${ONNX_DIR}/vocab.txt" ]]; then
        cp "${ONNX_DIR}/vocab.txt" "${BLS_DIR}/1/"
        log_info "Copied vocab.txt to BLS model"
    else
        log_error "vocab.txt not found in ${ONNX_DIR}"
        return 1
    fi

    # Verify all components
    log_info "Verifying model repository..."

    local errors=0

    # Check ONNX models
    if [[ -f "${ONNX_DIR}/encoder-model.onnx" ]]; then
        log_info "  ✓ parakeet_onnx/encoder-model.onnx"
    else
        log_error "  ✗ parakeet_onnx/encoder-model.onnx MISSING"
        ((errors++))
    fi

    if [[ -f "${ONNX_DIR}/decoder_joint-model.onnx" ]]; then
        log_info "  ✓ parakeet_onnx/decoder_joint-model.onnx"
    else
        log_error "  ✗ parakeet_onnx/decoder_joint-model.onnx MISSING"
        ((errors++))
    fi

    # Check Python backends
    if [[ -f "${ENCODER_DIR}/1/model.py" ]]; then
        log_info "  ✓ parakeet_encoder/1/model.py (ONNX Runtime)"
    else
        log_error "  ✗ parakeet_encoder/1/model.py MISSING"
        ((errors++))
    fi

    if [[ -f "${DECODER_DIR}/1/model.py" ]]; then
        log_info "  ✓ parakeet_decoder/1/model.py (ONNX Runtime)"
    else
        log_error "  ✗ parakeet_decoder/1/model.py MISSING"
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

# =============================================================================
# Summary
# =============================================================================
show_summary() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}Parakeet TDT Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Architecture: BLS + ONNX Runtime GPU Backends"
    echo ""
    echo "ONNX Models:"
    ls -lh "${ONNX_DIR}"/*.onnx 2>/dev/null || echo "  (not found)"
    echo ""
    echo "External Weights:"
    ls -lh "${ONNX_DIR}"/*.data 2>/dev/null || echo "  (none)"
    echo ""
    echo "Python Backends:"
    echo "  parakeet_encoder: ${ENCODER_DIR}/1/model.py"
    echo "  parakeet_decoder: ${DECODER_DIR}/1/model.py"
    echo "  parakeet_tdt:     ${BLS_DIR}/1/model.py"
    echo ""
    echo "To start Triton:"
    echo "  docker compose up -d triton"
    echo ""
    echo "Cleanup options:"
    echo "  $0 cleanup            # Remove work directory only"
    echo "  $0 cleanup --all      # Remove everything including ONNX models"
}

# =============================================================================
# Main
# =============================================================================
main() {
    download_onnx_models
    setup_model_repo
    show_summary
}

main
