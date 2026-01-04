#!/bin/bash
# =============================================================================
# Build Qwen3 LLM for vLLM Backend
#
# Downloads the Qwen3 model weights from HuggingFace for use with Triton's
# native vLLM backend.
#
# Usage:
#   ./build_qwen.sh              # Download model
#   ./build_qwen.sh cleanup      # Remove downloaded files
#
# The model is downloaded to the HuggingFace cache and will be loaded by
# vLLM at runtime. Pre-downloading avoids cold-start delays.
#
# Container: nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3
# =============================================================================

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration (from config.yaml, env vars override)
# =============================================================================
readonly MODEL_NAME="qwen3"

# Load from config.yaml (env vars take precedence)
HF_REPO="${HF_REPO:-$(cfg_get 'vllm.model' 'Qwen/Qwen3-4B-Instruct-AWQ')}"
QUANTIZATION="${QUANTIZATION:-$(cfg_get 'vllm.quantization' 'awq')}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(cfg_get 'vllm.max_model_len' '8192')}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-$(cfg_get 'vllm.gpu_memory_utilization' '0.85')}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-$(cfg_get 'vllm.tensor_parallel_size' '1')}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-$(cfg_get 'vllm.enable_prefix_caching' 'true')}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly MODEL_REPO="${DEPLOY_DIR}/model_repository/llm"
readonly MODEL_DIR="${MODEL_REPO}/${MODEL_NAME}"
readonly WEIGHTS_DIR="${DEPLOY_DIR}/models/qwen3_weights"

# Container path (mounted in docker-compose.yml)
readonly CONTAINER_WEIGHTS_PATH="/models/qwen3_weights"

# =============================================================================
# Cleanup Handler
# =============================================================================
if [[ "${1:-}" == "cleanup" || "${1:-}" == "clean" ]]; then
    echo "=============================================="
    echo "Cleaning up Qwen3 build artifacts"
    echo "=============================================="

    case "${2:-}" in
        --all|-a)
            log_warn "Removing downloaded weights..."
            rm -rf "$WEIGHTS_DIR"
            log_info "Weights removed: ${WEIGHTS_DIR}"
            ;;
        *)
            echo "Usage: $0 cleanup --all"
            echo "  --all    Remove local weights directory"
            ;;
    esac
    exit 0
fi

# =============================================================================
# Main
# =============================================================================
log_step "Building Qwen3 LLM for vLLM backend"
echo ""
log_info "Model:        ${HF_REPO}"
log_info "Quantization: ${QUANTIZATION}"
log_info "Max Length:   ${MAX_MODEL_LEN}"
log_info "GPU Memory:   ${GPU_MEMORY_UTIL}"
log_info "Weights:      ${WEIGHTS_DIR}"
echo ""

# Ensure model directory exists
mkdir -p "$MODEL_DIR/1"
mkdir -p "$WEIGHTS_DIR"

# =============================================================================
# Stage 1: Download Model Weights
# =============================================================================
log_step "Downloading Qwen3 model weights..."

# Check if already downloaded
if has_real_weights "$WEIGHTS_DIR" "*.safetensors"; then
    log_info "Model already downloaded: ${WEIGHTS_DIR}"
else
    # Clone with LFS
    hf_clone "$HF_REPO" "$WEIGHTS_DIR" || {
        log_error "Failed to clone ${HF_REPO}"
        log_error "If gated, set HF_TOKEN environment variable"
        exit 1
    }

    # Verify download
    if ! has_real_weights "$WEIGHTS_DIR" "*.safetensors"; then
        log_error "Download incomplete - no safetensors files found"
        exit 1
    fi

    log_info "Download complete"
fi

# =============================================================================
# Stage 2: Create vLLM model.json config
# =============================================================================
log_step "Creating vLLM configuration..."

# Create model.json for vLLM backend
# Use container path where weights are mounted (not HF repo name)
cat > "${MODEL_DIR}/1/model.json" << EOF
{
    "model": "${CONTAINER_WEIGHTS_PATH}",
    "disable_log_requests": true,
    "gpu_memory_utilization": ${GPU_MEMORY_UTIL},
    "max_model_len": ${MAX_MODEL_LEN},
    "tensor_parallel_size": ${TENSOR_PARALLEL},
    "dtype": "auto",
    "quantization": "${QUANTIZATION}",
    "enable_prefix_caching": ${ENABLE_PREFIX_CACHING},
    "trust_remote_code": true
}
EOF

log_info "Created: ${MODEL_DIR}/1/model.json"

# =============================================================================
# Stage 3: Verify config.pbtxt
# =============================================================================
if [[ ! -f "${MODEL_DIR}/config.pbtxt" ]]; then
    log_step "Creating Triton config.pbtxt..."

    cat > "${MODEL_DIR}/config.pbtxt" << 'EOF'
# Qwen3 LLM via vLLM Backend
name: "qwen3"
backend: "vllm"

instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]

parameters: {
  key: "REPORT_CUSTOM_METRICS"
  value: {
    string_value: "true"
  }
}
EOF
    log_info "Created: ${MODEL_DIR}/config.pbtxt"
else
    log_info "config.pbtxt already exists"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
log_info "Qwen3 build complete!"
echo "=============================================="
echo ""
echo "Model:   ${HF_REPO}"
echo "Weights: ${WEIGHTS_DIR}"
echo "Config:  ${MODEL_DIR}/1/model.json"
echo "Mount:   ${CONTAINER_WEIGHTS_PATH} (in container)"
echo ""
echo "To start Triton with Qwen3:"
echo "  docker compose up -d triton"
echo ""
echo "To test:"
echo "  curl -X POST http://localhost:8000/v2/models/qwen3/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text_input\": \"Hello, how are you?\", \"max_tokens\": 50}'"
echo ""
