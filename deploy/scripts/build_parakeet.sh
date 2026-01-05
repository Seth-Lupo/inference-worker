#!/bin/bash
# =============================================================================
# Build Parakeet TDT 0.6B V2 for Triton (ASR Model)
#
# Architecture (Native ONNX Backend):
#   - parakeet_encoder: Native ONNX Runtime (audio → encoded features)
#   - parakeet_decoder: Native ONNX Runtime (encoded → tokens)
#
# Downloads ONNX models from HuggingFace.
# Models are mounted into container via docker-compose.
#
# Usage:
#   ./build_parakeet.sh              # Download ONNX models
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

# Paths - ONNX files go to models/ directory (mounted by docker-compose)
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/parakeet_build"
readonly ONNX_DIR="${DEPLOY_DIR}/models/parakeet_onnx"

# Config directories (for reference)
readonly ENCODER_CONFIG="${DEPLOY_DIR}/model_repository/asr/parakeet_encoder"
readonly DECODER_CONFIG="${DEPLOY_DIR}/model_repository/asr/parakeet_decoder"

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
echo "Building Parakeet TDT 0.6B V2 (Native ONNX)"
echo "=============================================="
echo "Architecture: Native ONNX Runtime Backend"
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
# Stage 2: Verify Downloads
# =============================================================================
verify_downloads() {
    log_step "Verifying ONNX model files..."

    local errors=0

    # Check ONNX models exist and have reasonable sizes
    if is_real_file "${ONNX_DIR}/encoder-model.onnx" 1000000; then
        local size
        size=$(get_file_size "${ONNX_DIR}/encoder-model.onnx" 2>/dev/null || echo "?")
        log_info "  ✓ encoder-model.onnx ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
    else
        log_error "  ✗ encoder-model.onnx MISSING or too small"
        ((errors++))
    fi

    # External data file for encoder (large weights)
    if is_real_file "${ONNX_DIR}/encoder-model.onnx.data" 100000000; then
        local size
        size=$(get_file_size "${ONNX_DIR}/encoder-model.onnx.data" 2>/dev/null || echo "?")
        log_info "  ✓ encoder-model.onnx.data ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
    else
        log_error "  ✗ encoder-model.onnx.data MISSING or too small (should be ~600MB)"
        ((errors++))
    fi

    if is_real_file "${ONNX_DIR}/decoder_joint-model.onnx" 1000000; then
        local size
        size=$(get_file_size "${ONNX_DIR}/decoder_joint-model.onnx" 2>/dev/null || echo "?")
        log_info "  ✓ decoder_joint-model.onnx ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
    else
        log_error "  ✗ decoder_joint-model.onnx MISSING or too small"
        ((errors++))
    fi

    # Vocab is optional (used by worker for decoding, not Triton)
    if [[ -f "${ONNX_DIR}/vocab.txt" ]]; then
        log_info "  ✓ vocab.txt"
    else
        log_warn "  ⚠ vocab.txt not found (optional, used by worker)"
    fi

    if [[ $errors -gt 0 ]]; then
        log_error "Missing ${errors} required ONNX files"
        return 1
    fi

    log_info "All ONNX files ready"
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
    echo "Architecture: Native ONNX Runtime Backend"
    echo ""
    echo "ONNX Models (${ONNX_DIR}):"
    ls -lh "${ONNX_DIR}"/*.onnx 2>/dev/null || echo "  (not found)"
    echo ""
    echo "External Weights:"
    ls -lh "${ONNX_DIR}"/*.data 2>/dev/null || echo "  (none)"
    echo ""
    echo "Triton Models (via docker-compose volume mounts):"
    echo "  parakeet_encoder: encoder-model.onnx → /models/asr/parakeet_encoder/1/model.onnx"
    echo "  parakeet_decoder: decoder_joint-model.onnx → /models/asr/parakeet_decoder/1/model.onnx"
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
    verify_downloads
    show_summary
}

main
