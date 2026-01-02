#!/bin/bash
# =============================================================================
# Build Parakeet TDT 0.6B V2 for Triton (ASR Model)
#
# Downloads pre-converted ONNX models from sherpa-onnx releases.
# Uses Python backend with sherpa-onnx for transducer decoding.
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

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/parakeet_build"
readonly MODEL_DIR="${DEPLOY_DIR}/model_repository/${MODEL_NAME}"

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
        *)
            log_info "Removing work directory (keeping model)..."
            rm -rf "$WORK_DIR"
            log_info "Cleanup complete"
            log_info "To also remove model: $0 cleanup --all"
            ;;
    esac
    exit 0
fi

# =============================================================================
# Main Functions
# =============================================================================

download_sherpa_model() {
    log_step "Downloading Parakeet from sherpa-onnx..."

    local url="https://github.com/k2-fsa/sherpa-onnx/releases/download/${SHERPA_VERSION}/${SHERPA_ARCHIVE}"
    local archive="${WORK_DIR}/${SHERPA_ARCHIVE}"

    mkdir -p "$WORK_DIR"

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

    # Find extracted directory (sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8)
    local extracted
    extracted=$(find "$WORK_DIR" -maxdepth 1 -type d -name "sherpa-onnx-*parakeet*" | head -1)

    if [[ -z "$extracted" || ! -d "$extracted" ]]; then
        log_error "Could not find extracted directory"
        ls -la "$WORK_DIR"
        return 1
    fi

    log_info "Found: $extracted"
    echo "$extracted"
}

setup_model_repository() {
    local source_dir="$1"

    log_step "Setting up Triton model repository..."

    mkdir -p "${MODEL_DIR}/1"

    # Copy ONNX files
    log_info "Copying ONNX files..."
    cp -v "${source_dir}"/*.onnx "${MODEL_DIR}/1/"

    # Copy tokens
    cp -v "${source_dir}/tokens.txt" "${MODEL_DIR}/1/"

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
    log_step "Creating Triton configuration..."

    cat > "${MODEL_DIR}/config.pbtxt" << 'EOF'
# Parakeet TDT 0.6B V2 - Automatic Speech Recognition
# Uses sherpa-onnx Python backend for transducer decoding

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

create_python_backend() {
    log_step "Creating Python backend model..."

    cat > "${MODEL_DIR}/1/model.py" << 'PYTHON'
"""
Parakeet TDT 0.6B V2 - Triton Python Backend
Wraps sherpa-onnx for transducer-based ASR.
"""
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python backend for Parakeet TDT ASR."""

    def initialize(self, args):
        """Load the sherpa-onnx recognizer."""
        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.dirname(os.path.realpath(__file__))

        try:
            import sherpa_onnx
            self._init_sherpa(model_dir, sherpa_onnx)
        except ImportError as e:
            pb_utils.Logger.log_error(f"sherpa_onnx not installed: {e}")
            raise

        pb_utils.Logger.log_info("Parakeet TDT initialized successfully")

    def _init_sherpa(self, model_dir, sherpa_onnx):
        """Initialize sherpa-onnx recognizer."""
        encoder = os.path.join(model_dir, "encoder.onnx")
        decoder = os.path.join(model_dir, "decoder.onnx")
        joiner = os.path.join(model_dir, "joiner.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        # Verify files exist
        for f in [encoder, decoder, joiner, tokens]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing: {f}")

        pb_utils.Logger.log_info("Initializing sherpa-onnx with CUDA provider...")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            provider="cuda",
        )

    def execute(self, requests):
        """Run inference on batch of audio inputs."""
        responses = []

        for request in requests:
            audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio")
            audio = audio_tensor.as_numpy().flatten().astype(np.float32)

            # Normalize audio to [-1, 1] range if needed
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / 32768.0

            # Run recognition
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()

            # Create response
            output = pb_utils.Tensor("transcription", np.array([text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output]))

        return responses

    def finalize(self):
        """Cleanup."""
        pass
PYTHON

    log_info "Created: ${MODEL_DIR}/1/model.py"
}

verify_setup() {
    log_step "Verifying setup..."

    echo ""
    echo "Model directory contents:"
    ls -la "${MODEL_DIR}/1/"

    echo ""
    log_info "Parakeet TDT setup complete!"
    echo ""
    echo "Model location: ${MODEL_DIR}"
    echo ""
    echo "To load in Triton, add to docker-compose.yml command:"
    echo "  --load-model=parakeet_tdt"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "=============================================="
    echo "Building Parakeet TDT 0.6B V2"
    echo "=============================================="
    echo ""

    # Check if model already exists with valid files
    if is_real_file "${MODEL_DIR}/1/encoder.onnx"; then
        log_info "Parakeet model already exists at ${MODEL_DIR}"
        log_info "Run '$0 cleanup --all' to rebuild"
        exit 0
    fi

    # Download and extract
    local extracted_dir
    extracted_dir=$(download_sherpa_model)

    # Setup model repository
    setup_model_repository "$extracted_dir"

    # Create configs
    create_triton_config
    create_python_backend

    # Verify
    verify_setup
}

main "$@"
