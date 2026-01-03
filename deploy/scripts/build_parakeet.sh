#!/bin/bash
# =============================================================================
# Build Parakeet TDT 0.6B V2 for Triton (ASR Model)
#
# Downloads pre-converted ONNX models from sherpa-onnx releases,
# builds TensorRT engines for GPU inference, and creates Triton model repository.
#
# Usage:
#   ./build_parakeet.sh [START_STAGE] [STOP_STAGE]
#   ./build_parakeet.sh 0 2      # Run all stages
#   ./build_parakeet.sh 1 1      # Only build TRT engines
#   ./build_parakeet.sh cleanup  # Clean up build artifacts
#
# Stages:
#   0: Download ONNX models from sherpa-onnx
#   1: Build TensorRT engines (encoder, decoder, joiner)
#   2: Create Triton model repository
#
# Sources:
#   - https://github.com/k2-fsa/sherpa-onnx/releases
#   - https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
# =============================================================================

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================
readonly MODEL_NAME="parakeet_tdt"
readonly SHERPA_VERSION="asr-models"
readonly SHERPA_ARCHIVE="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"

# TRT build settings
TRT_PRECISION="${TRT_PRECISION:-fp16}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/parakeet_build"
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
# Stage 0: Download Models
# =============================================================================

stage_download_models() {
    log_step "Stage 0: Downloading Parakeet from sherpa-onnx..."

    local url="https://github.com/k2-fsa/sherpa-onnx/releases/download/${SHERPA_VERSION}/${SHERPA_ARCHIVE}"
    local archive="${WORK_DIR}/${SHERPA_ARCHIVE}"

    mkdir -p "$WORK_DIR"

    # Check if already extracted (files may be encoder.onnx or encoder.int8.onnx)
    local extracted
    extracted=$(find "$WORK_DIR" -maxdepth 1 -type d -name "sherpa-onnx-*parakeet*" 2>/dev/null | head -1)
    if [[ -n "$extracted" ]] && ls "${extracted}"/encoder*.onnx &>/dev/null; then
        log_info "Models already downloaded at: $extracted"
        return 0
    fi

    # Download if not exists
    if [[ ! -f "$archive" ]]; then
        log_info "Downloading ${SHERPA_ARCHIVE}..."
        if ! curl -L -f -o "$archive" "$url"; then
            log_error "Failed to download from: $url"
            return 1
        fi
    else
        log_info "Archive already downloaded"
    fi

    # Verify it's a valid bzip2 file
    if ! file "$archive" | grep -q "bzip2"; then
        log_error "Downloaded file is not a valid bzip2 archive"
        rm -f "$archive"
        return 1
    fi

    # Extract
    log_info "Extracting archive..."
    (cd "$WORK_DIR" && tar -xjf "$SHERPA_ARCHIVE")

    # Verify extraction (files may be encoder.onnx or encoder.int8.onnx)
    extracted=$(find "$WORK_DIR" -maxdepth 1 -type d -name "sherpa-onnx-*parakeet*" | head -1)
    if [[ -z "$extracted" ]] || ! ls "${extracted}"/encoder*.onnx &>/dev/null; then
        log_error "Could not find extracted ONNX models"
        ls -la "$WORK_DIR"
        return 1
    fi

    log_info "Downloaded: $extracted"
}

# =============================================================================
# Stage 1: Build TensorRT Engines
# =============================================================================

stage_build_trt_engines() {
    log_step "Stage 1: Building TensorRT engines..."

    # Find ONNX models
    local onnx_dir
    onnx_dir=$(find "$WORK_DIR" -maxdepth 1 -type d -name "sherpa-onnx-*parakeet*" | head -1)
    if [[ -z "$onnx_dir" ]]; then
        log_error "ONNX models not found. Run stage 0 first."
        return 1
    fi

    # Find ONNX files (may be encoder.onnx or encoder.int8.onnx)
    local encoder_onnx decoder_onnx joiner_onnx
    encoder_onnx=$(ls "${onnx_dir}"/encoder*.onnx 2>/dev/null | head -1)
    decoder_onnx=$(ls "${onnx_dir}"/decoder*.onnx 2>/dev/null | head -1)
    joiner_onnx=$(ls "${onnx_dir}"/joiner*.onnx 2>/dev/null | head -1)

    if [[ -z "$encoder_onnx" || -z "$decoder_onnx" || -z "$joiner_onnx" ]]; then
        log_error "Missing ONNX files in: $onnx_dir"
        ls -la "$onnx_dir"
        return 1
    fi

    log_info "Found ONNX models: $(basename "$encoder_onnx"), $(basename "$decoder_onnx"), $(basename "$joiner_onnx")"

    mkdir -p "$ENGINE_DIR"

    # Determine precision flag
    local precision_flag="--fp16"
    [[ "$TRT_PRECISION" == "fp32" ]] && precision_flag="--fp32"

    # Build encoder (audio input: variable length)
    log_info "Building encoder engine..."
    build_trt_engine \
        "$encoder_onnx" \
        "${ENGINE_DIR}/encoder.engine" \
        "$precision_flag" \
        "--minShapes=x:1x1000" \
        "--optShapes=x:1x160000" \
        "--maxShapes=x:1x480000"

    # Build decoder (single token input)
    log_info "Building decoder engine..."
    build_trt_engine \
        "$decoder_onnx" \
        "${ENGINE_DIR}/decoder.engine" \
        "$precision_flag" \
        "--minShapes=y:1x1" \
        "--optShapes=y:1x1" \
        "--maxShapes=y:1x1"

    # Build joiner (fixed size hidden states)
    log_info "Building joiner engine..."
    build_trt_engine \
        "$joiner_onnx" \
        "${ENGINE_DIR}/joiner.engine" \
        "$precision_flag" \
        "--minShapes=encoder_out:1x1x1024,decoder_out:1x1x512" \
        "--optShapes=encoder_out:1x1x1024,decoder_out:1x1x512" \
        "--maxShapes=encoder_out:1x1x1024,decoder_out:1x1x512"

    log_info "TRT engines built at: $ENGINE_DIR"
}

# =============================================================================
# Stage 2: Create Model Repository
# =============================================================================

stage_create_model_repo() {
    log_step "Stage 2: Creating Triton model repository..."

    # Find ONNX source
    local source_dir
    source_dir=$(find "$WORK_DIR" -maxdepth 1 -type d -name "sherpa-onnx-*parakeet*" | head -1)
    if [[ -z "$source_dir" ]]; then
        log_error "ONNX models not found. Run stage 0 first."
        return 1
    fi

    mkdir -p "${MODEL_DIR}/1"

    # Copy ONNX files (needed as fallback if TRT fails)
    log_info "Copying ONNX files..."
    cp "${source_dir}"/*.onnx "${MODEL_DIR}/1/" 2>/dev/null || true

    # Copy tokens
    cp "${source_dir}/tokens.txt" "${MODEL_DIR}/1/"

    # Rename INT8 files to standard names if needed
    local f
    for f in "${MODEL_DIR}/1/"*.int8.onnx; do
        if [[ -f "$f" ]]; then
            local newname="${f%.int8.onnx}.onnx"
            log_info "Renaming: $(basename "$f") -> $(basename "$newname")"
            mv "$f" "$newname"
        fi
    done

    # Verify required files
    local required_files=("encoder.onnx" "decoder.onnx" "joiner.onnx" "tokens.txt")
    for file in "${required_files[@]}"; do
        if [[ ! -f "${MODEL_DIR}/1/${file}" ]]; then
            log_error "Missing required file: ${file}"
            return 1
        fi
    done

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
