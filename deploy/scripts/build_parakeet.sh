#!/bin/bash
# =============================================================================
# Build Parakeet TDT 0.6B V2 for Triton (ASR Model)
#
# Downloads ONNX models from HuggingFace, builds TensorRT engines for GPU
# inference, and creates Triton model repository.
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
# Configuration
# =============================================================================
readonly MODEL_NAME="parakeet_tdt"
readonly HF_REPO="istupakov/parakeet-tdt-0.6b-v2-onnx"

# TRT build settings
TRT_PRECISION="${TRT_PRECISION:-fp16}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/parakeet_build"
readonly ONNX_DIR="${WORK_DIR}/onnx"
readonly MODEL_DIR="${DEPLOY_DIR}/model_repository/${MODEL_NAME}"
readonly ENGINE_DIR="${MODEL_DIR}/1/engines"

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
            rm -rf "$MODEL_DIR"
            log_info "Cleanup complete"
            ;;
        --engines|-e)
            log_info "Removing TRT engines only..."
            rm -rf "$ENGINE_DIR"
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
    log_step "Stage 1: Building TensorRT engines..."

    # Verify ONNX models exist
    if [[ ! -f "${ONNX_DIR}/encoder-model.onnx" ]]; then
        log_error "ONNX models not found. Run stage 0 first."
        return 1
    fi

    mkdir -p "$ENGINE_DIR"

    # Determine precision flag
    local precision_flag="--fp16"
    [[ "$TRT_PRECISION" == "fp32" ]] && precision_flag="--fp32"

    # Build encoder (audio features input: batch x time x features)
    # Parakeet encoder expects preprocessed mel features
    log_info "Building encoder engine..."
    build_trt_engine \
        "${ONNX_DIR}/encoder-model.onnx" \
        "${ENGINE_DIR}/encoder.engine" \
        "$precision_flag" \
        "--minShapes=audio_signal:1x128x10,length:1" \
        "--optShapes=audio_signal:1x128x1000,length:1" \
        "--maxShapes=audio_signal:1x128x3000,length:1"

    # Build decoder_joint (combined decoder and joiner)
    log_info "Building decoder_joint engine..."
    build_trt_engine \
        "${ONNX_DIR}/decoder_joint-model.onnx" \
        "${ENGINE_DIR}/decoder_joint.engine" \
        "$precision_flag" \
        "--minShapes=encoder_outputs:1x1x1024,targets:1x1,target_length:1" \
        "--optShapes=encoder_outputs:1x500x1024,targets:1x1,target_length:1" \
        "--maxShapes=encoder_outputs:1x3000x1024,targets:1x1,target_length:1"

    log_info "TRT engines built at: $ENGINE_DIR"
}

# =============================================================================
# Stage 2: Create Model Repository
# =============================================================================

stage_create_model_repo() {
    log_step "Stage 2: Creating Triton model repository..."

    if [[ ! -d "$ONNX_DIR" ]]; then
        log_error "ONNX models not found. Run stage 0 first."
        return 1
    fi

    mkdir -p "${MODEL_DIR}/1"

    # Copy ONNX files (as fallback and for reference)
    log_info "Copying ONNX files..."
    cp "${ONNX_DIR}/encoder-model.onnx" "${MODEL_DIR}/1/"
    cp "${ONNX_DIR}/encoder-model.onnx.data" "${MODEL_DIR}/1/"
    cp "${ONNX_DIR}/decoder_joint-model.onnx" "${MODEL_DIR}/1/"

    # Copy vocab and preprocessor
    cp "${ONNX_DIR}/vocab.txt" "${MODEL_DIR}/1/"
    [[ -f "${ONNX_DIR}/nemo128.onnx" ]] && cp "${ONNX_DIR}/nemo128.onnx" "${MODEL_DIR}/1/"

    log_info "Model files copied successfully"
}

create_triton_config() {
    log_info "Creating Triton configuration..."

    cat > "${MODEL_DIR}/config.pbtxt" << 'EOF'
# Parakeet TDT 0.6B V2 - Automatic Speech Recognition
# Uses TensorRT engines for GPU inference with Python backend

name: "parakeet_tdt"
backend: "python"
max_batch_size: 8

input [
  {
    name: "audio"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "audio_length"
    data_type: TYPE_INT64
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "transcription"
    data_type: TYPE_STRING
    dims: [ 1 ]
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
  key: "model_type"
  value: { string_value: "transducer" }
}

version_policy: { latest: { num_versions: 1 } }
EOF

    log_info "Created: ${MODEL_DIR}/config.pbtxt"
}

show_summary() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}Parakeet TDT Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Model location: ${MODEL_DIR}"
    echo ""
    echo "Contents:"
    ls -la "${MODEL_DIR}/1/" 2>/dev/null || echo "  (run all stages to build)"
    echo ""
    if [[ -d "$ENGINE_DIR" ]]; then
        echo "TRT Engines:"
        ls -lh "${ENGINE_DIR}"/*.engine 2>/dev/null || echo "  (none)"
        echo ""
    fi
    echo "To start Triton:"
    echo "  docker compose up -d triton"
    echo ""
    echo "Cleanup options:"
    echo "  $0 cleanup            # Remove downloads, keep model"
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

    # Stage 2: Create model repo
    if [[ $START_STAGE -le 2 && $STOP_STAGE -ge 2 ]]; then
        stage_create_model_repo
        create_triton_config
    fi

    show_summary
}

main
