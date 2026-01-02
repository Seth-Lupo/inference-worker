#!/bin/bash
# =============================================================================
# Setup Parakeet TDT 0.6B V2 for Triton
# Uses pre-converted ONNX models from sherpa-onnx or HuggingFace
#
# Sources:
# - https://huggingface.co/onnx-community/parakeet-tdt-0.6b-v2-ONNX
# - https://github.com/k2-fsa/sherpa-onnx/releases
# - https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/9
# =============================================================================
set -e

# =============================================================================
# CONFIGURATION - Edit these values as needed
# =============================================================================

# Use INT8 quantized version (smaller, slightly less accurate)
USE_INT8="${USE_INT8:-false}"

# Sherpa-ONNX release tag for ASR models
SHERPA_VERSION="asr-models"

# Model name in Triton
MODEL_NAME="parakeet_tdt"

# HuggingFace repository for ONNX models
HF_REPO="onnx-community/parakeet-tdt-0.6b-v2-ONNX"

# =============================================================================
# PATHS - Derived from script location
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_REPO="${DEPLOY_DIR}/model_repository"
WORK_DIR="${DEPLOY_DIR}/parakeet_build"

# =============================================================================
# Logging
# =============================================================================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if file is real (not LFS pointer, not empty, not HTML error)
check_file_real() {
    local file="$1"
    local min_size="${2:-1000000}"  # Default 1MB minimum
    [ ! -f "$file" ] && return 1
    local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")
    [ "$size" -gt "$min_size" ]
}

# Override USE_INT8 from command line if provided
[ -n "$1" ] && USE_INT8="$1"

echo "=============================================="
echo "Setting up Parakeet TDT 0.6B V2"
echo "=============================================="
echo "INT8 quantized: ${USE_INT8}"
echo ""

mkdir -p "${WORK_DIR}"
PARAKEET_DIR="${MODEL_REPO}/parakeet_tdt"
mkdir -p "${PARAKEET_DIR}/1"

# =============================================================================
# Option 1: Download from HuggingFace ONNX Community (recommended)
# =============================================================================
download_from_huggingface() {
    log_info "Downloading from HuggingFace ONNX Community..."

    HF_REPO="onnx-community/parakeet-tdt-0.6b-v2-ONNX"
    HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"

    # Clean up any existing directory
    rm -rf "${WORK_DIR}/parakeet_onnx"
    mkdir -p "${WORK_DIR}/parakeet_onnx"

    # Direct download via curl/wget (no CLI needed)
    log_info "Downloading ONNX files directly..."

    # Main model files
    FILES=(
        "model.onnx"
        "model_fp16.onnx"
        "config.json"
        "preprocessor_config.json"
        "tokenizer.json"
        "vocab.json"
    )

    DOWNLOADED=0
    for file in "${FILES[@]}"; do
        if curl -L -f -o "${WORK_DIR}/parakeet_onnx/${file}" "${HF_BASE}/${file}" 2>/dev/null; then
            log_info "  Downloaded: ${file}"
            DOWNLOADED=$((DOWNLOADED + 1))
        fi
    done

    if [ $DOWNLOADED -gt 0 ]; then
        # Validate ONNX files are real (not HTML error pages)
        VALID_ONNX=false
        for f in "${WORK_DIR}/parakeet_onnx/"*.onnx; do
            if check_file_real "$f"; then
                VALID_ONNX=true
                break
            fi
        done

        if [ "$VALID_ONNX" = true ]; then
            # Copy to model repo
            cp "${WORK_DIR}/parakeet_onnx"/*.onnx "${PARAKEET_DIR}/1/" 2>/dev/null || true
            cp "${WORK_DIR}/parakeet_onnx"/*.txt "${PARAKEET_DIR}/1/" 2>/dev/null || true
            cp "${WORK_DIR}/parakeet_onnx"/*.json "${PARAKEET_DIR}/1/" 2>/dev/null || true
            log_info "HuggingFace download successful"
            return 0
        else
            log_warn "HuggingFace download got invalid files (possibly requires auth)"
        fi
    fi

    log_warn "Direct download failed, trying git clone..."

    # Fallback to git clone
    if command -v git &> /dev/null; then
        git lfs install 2>/dev/null || true
        git clone --depth 1 "https://huggingface.co/${HF_REPO}" "${WORK_DIR}/parakeet_onnx" && {
            cp "${WORK_DIR}/parakeet_onnx"/*.onnx "${PARAKEET_DIR}/1/" 2>/dev/null || true
            cp "${WORK_DIR}/parakeet_onnx"/*.txt "${PARAKEET_DIR}/1/" 2>/dev/null || true

            # Validate files were actually copied (not just LFS pointers)
            if ls "${PARAKEET_DIR}/1/"*.onnx 1>/dev/null 2>&1; then
                # Check file size > 1MB (LFS pointers are ~130 bytes)
                for f in "${PARAKEET_DIR}/1/"*.onnx; do
                    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "0")
                    if [ "$size" -gt 1000000 ]; then
                        log_info "Downloaded real ONNX files from HuggingFace"
                        return 0
                    fi
                done
            fi
            log_warn "HuggingFace clone got LFS pointers, not real files"
        }
    fi

    return 1
}

# =============================================================================
# Option 2: Download from Sherpa-ONNX releases
# =============================================================================
download_from_sherpa() {
    log_info "Downloading from Sherpa-ONNX releases..."

    # Find latest release with parakeet
    SHERPA_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"

    if [ "$USE_INT8" == "true" ]; then
        # INT8 quantized version (smaller, slightly less accurate)
        ARCHIVE="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
    else
        # INT8 is the only available quantization from sherpa-onnx
        # (no FP16/FP32 pre-built available)
        ARCHIVE="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
        log_info "Note: Using INT8 version (only quantization available from sherpa-onnx)"
    fi

    DOWNLOAD_URL="${SHERPA_BASE}/${SHERPA_VERSION}/${ARCHIVE}"

    # Always re-download to ensure we have the right file
    rm -f "${WORK_DIR}/${ARCHIVE}"
    log_info "Downloading ${ARCHIVE}..."
    if ! curl -L -f -o "${WORK_DIR}/${ARCHIVE}" "${DOWNLOAD_URL}"; then
        log_warn "Could not download from sherpa-onnx"
        return 1
    fi

    # Verify it's actually a bzip2 file
    if ! file "${WORK_DIR}/${ARCHIVE}" | grep -q "bzip2"; then
        log_warn "Downloaded file is not a valid bzip2 archive"
        rm -f "${WORK_DIR}/${ARCHIVE}"
        return 1
    fi

    # Extract
    log_info "Extracting..."
    cd "${WORK_DIR}"
    tar -xjf "${ARCHIVE}"

    # Find and copy model files (look for sherpa-onnx specific directory)
    EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "sherpa-onnx-*parakeet*" | head -1)
    if [ -z "$EXTRACTED_DIR" ]; then
        log_error "Could not find extracted parakeet directory"
        return 1
    fi

    log_info "Found extracted directory: ${EXTRACTED_DIR}"
    log_info "Copying files to model repository..."

    # Copy ONNX files
    cp -v "${EXTRACTED_DIR}"/*.onnx "${PARAKEET_DIR}/1/" || {
        log_error "Failed to copy ONNX files"
        return 1
    }

    # Copy tokens.txt
    cp -v "${EXTRACTED_DIR}"/tokens.txt "${PARAKEET_DIR}/1/" || {
        log_error "Failed to copy tokens.txt"
        return 1
    }

    # Rename INT8 files to standard names (sherpa-onnx uses .int8.onnx suffix)
    cd "${PARAKEET_DIR}/1/"
    for f in *.int8.onnx; do
        if [ -f "$f" ]; then
            newname="${f%.int8.onnx}.onnx"
            log_info "Renaming: $f -> $newname"
            mv "$f" "$newname"
        fi
    done

    # Verify we have the required files
    if [ -f "${PARAKEET_DIR}/1/encoder.onnx" ] && \
       [ -f "${PARAKEET_DIR}/1/decoder.onnx" ] && \
       [ -f "${PARAKEET_DIR}/1/joiner.onnx" ] && \
       [ -f "${PARAKEET_DIR}/1/tokens.txt" ]; then
        log_info "Sherpa-ONNX download and setup successful"
        cd "${WORK_DIR}"
        return 0
    else
        log_error "Missing required files after extraction"
        ls -la "${PARAKEET_DIR}/1/"
        cd "${WORK_DIR}"
        return 1
    fi
}

# =============================================================================
# Option 3: Export from NeMo (requires NeMo installation)
# =============================================================================
export_from_nemo() {
    log_info "Exporting from NeMo..."
    log_warn "This requires NeMo to be installed"

    docker run --rm --gpus all \
        -v "${WORK_DIR}:/workspace" \
        -v "${PARAKEET_DIR}/1:/output" \
        nvcr.io/nvidia/nemo:24.07 \
        bash -c "
            pip install nemo_toolkit[asr]
            python3 << 'PYTHON'
import nemo.collections.asr as nemo_asr

# Load model
model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')

# Export to ONNX
model.export('/output/parakeet.onnx')
print('Export complete!')
PYTHON
        " || return 1

    return 0
}

# =============================================================================
# Try download methods in order
# =============================================================================
# Check for REAL files (not LFS pointers or empty files)
if check_file_real "${PARAKEET_DIR}/1/encoder.onnx" || check_file_real "${PARAKEET_DIR}/1/model.onnx"; then
    log_info "Parakeet ONNX model already exists and is valid"
else
    # Clean up any invalid/partial files
    rm -f "${PARAKEET_DIR}/1/"*.onnx 2>/dev/null || true

    download_from_huggingface || download_from_sherpa || {
        log_error "Could not download Parakeet model"
        log_info "Manual download options:"
        echo "  1. https://huggingface.co/onnx-community/parakeet-tdt-0.6b-v2-ONNX"
        echo "  2. https://github.com/k2-fsa/sherpa-onnx/releases"
        exit 1
    }
fi

# =============================================================================
# Verify model files exist before creating config
# =============================================================================
log_info "Verifying model files..."
ls -la "${PARAKEET_DIR}/1/"

# =============================================================================
# Create Triton config
# =============================================================================
# Determine which ONNX files we have (check with size validation)
if check_file_real "${PARAKEET_DIR}/1/encoder.onnx"; then
    # Sherpa-ONNX style (separate encoder/decoder/joiner)
    log_info "Detected sherpa-onnx model format (encoder/decoder/joiner)"

    cat > "${PARAKEET_DIR}/config.pbtxt" << 'EOF'
# Parakeet TDT 0.6B V2 - Automatic Speech Recognition
# Sherpa-ONNX format with separate encoder/decoder/joiner
# Note: This requires a Python backend to orchestrate the models

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

    # Create Python backend wrapper
    cat > "${PARAKEET_DIR}/1/model.py" << 'PYTHON'
"""
Parakeet TDT Triton Python Backend
Wraps sherpa-onnx for transducer-based ASR
"""
import json
import numpy as np
import triton_python_backend_utils as pb_utils

# Note: Requires sherpa-onnx to be installed in the container
# pip install sherpa-onnx

class TritonPythonModel:
    def initialize(self, args):
        import sherpa_onnx

        model_dir = "/models/parakeet_tdt/1"

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder.onnx",
            decoder=f"{model_dir}/decoder.onnx",
            joiner=f"{model_dir}/joiner.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=4,
            provider="cuda",
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            audio = pb_utils.get_input_tensor_by_name(request, "audio").as_numpy()

            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, audio.flatten())
            self.recognizer.decode_stream(stream)

            text = stream.result.text

            output = pb_utils.Tensor("transcription", np.array([text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output]))

        return responses

    def finalize(self):
        pass
PYTHON

elif check_file_real "${PARAKEET_DIR}/1/model.onnx"; then
    # Single ONNX file format (from HuggingFace)
    log_info "Detected single ONNX model format"
    log_error "Single ONNX format requires onnxruntime backend which is NOT available in TRT-LLM Triton image"
    log_error "Please use sherpa-onnx format instead (encoder/decoder/joiner)"
    exit 1
else
    log_error "No valid ONNX model files found!"
    log_error "Expected either encoder.onnx (sherpa format) or model.onnx (HuggingFace format)"
    ls -la "${PARAKEET_DIR}/1/"
    exit 1
fi

log_info "Config written: ${PARAKEET_DIR}/config.pbtxt"

echo ""
echo -e "${GREEN}Parakeet TDT setup complete${NC}"
echo "Model directory: ${PARAKEET_DIR}/1/"
ls -la "${PARAKEET_DIR}/1/"
