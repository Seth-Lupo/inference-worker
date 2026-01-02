#!/bin/bash
# =============================================================================
# Setup Silero VAD for Triton
# Source: https://github.com/snakers4/silero-vad
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_REPO="${DEPLOY_DIR}/model_repository"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "=============================================="
echo "Setting up Silero VAD"
echo "=============================================="

# Create model directory
SILERO_DIR="${MODEL_REPO}/silero_vad"
mkdir -p "${SILERO_DIR}/1"

# Download ONNX model
if [ ! -f "${SILERO_DIR}/1/model.onnx" ]; then
    log_info "Downloading Silero VAD ONNX model..."
    curl -L -o "${SILERO_DIR}/1/model.onnx" \
        "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    log_info "Downloaded: ${SILERO_DIR}/1/model.onnx"
else
    log_info "Silero VAD already exists"
fi

# Create Triton config
cat > "${SILERO_DIR}/config.pbtxt" << 'EOF'
# Silero VAD - Voice Activity Detection
# ONNX model, runs on CPU (lightweight, <1ms per chunk)

name: "silero_vad"
backend: "onnxruntime"
max_batch_size: 0  # Silero VAD doesn't support batching well

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "sr"
    data_type: TYPE_INT64
    dims: [ 1 ]
  },
  {
    name: "h"
    data_type: TYPE_FP32
    dims: [ 2, 1, 64 ]
  },
  {
    name: "c"
    data_type: TYPE_FP32
    dims: [ 2, 1, 64 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1 ]
  },
  {
    name: "hn"
    data_type: TYPE_FP32
    dims: [ 2, 1, 64 ]
  },
  {
    name: "cn"
    data_type: TYPE_FP32
    dims: [ 2, 1, 64 ]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]

version_policy: { latest: { num_versions: 1 } }
EOF

log_info "Config written: ${SILERO_DIR}/config.pbtxt"

echo ""
echo -e "${GREEN}Silero VAD setup complete${NC}"
echo "Model: ${SILERO_DIR}/1/model.onnx"
echo "Size: $(du -h ${SILERO_DIR}/1/model.onnx | cut -f1)"
