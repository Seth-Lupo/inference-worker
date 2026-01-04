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
# Container: nvcr.io/nvidia/tritonserver:24.12-vllm-python-py3
# =============================================================================

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================
readonly MODEL_NAME="qwen3"

# Model repository - Qwen3 4B AWQ quantized
HF_REPO="${HF_REPO:-Qwen/Qwen3-4B-Instruct-AWQ}"

# Alternative models (can override via environment):
# - Qwen/Qwen3-4B-Instruct (fp16, larger)
# - Qwen/Qwen3-1.7B-Instruct-AWQ (smaller, faster)
# - Qwen/Qwen3-8B-Instruct-AWQ (larger, smarter)

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly MODEL_REPO="${DEPLOY_DIR}/model_repository/llm"
readonly MODEL_DIR="${MODEL_REPO}/${MODEL_NAME}"
readonly WEIGHTS_DIR="${DEPLOY_DIR}/models/qwen3_weights"

# HuggingFace cache (used by vLLM at runtime)
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

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
            log_info "To clear HF cache: rm -rf ${HF_CACHE}/hub/models--${HF_REPO//\//__}"
            ;;
        --cache|-c)
            log_warn "Clearing HuggingFace cache for ${HF_REPO}..."
            rm -rf "${HF_CACHE}/hub/models--${HF_REPO//\//__}"
            ;;
        *)
            echo "Usage: $0 cleanup [--all|-a | --cache|-c]"
            echo "  --all    Remove local weights directory"
            echo "  --cache  Clear HuggingFace cache for this model"
            ;;
    esac
    exit 0
fi

# =============================================================================
# Main
# =============================================================================
log_step "Building Qwen3 LLM for vLLM backend"
echo ""
log_info "Model: ${HF_REPO}"
log_info "Cache: ${HF_CACHE}"
echo ""

# Ensure model directory exists
mkdir -p "$MODEL_DIR/1"
mkdir -p "$WEIGHTS_DIR"

# =============================================================================
# Stage 1: Download Model Weights
# =============================================================================
log_step "Downloading Qwen3 model weights..."

# Check if already downloaded (look for config.json as indicator)
CACHE_MODEL_DIR="${HF_CACHE}/hub/models--${HF_REPO//\//__}"
if [[ -d "$CACHE_MODEL_DIR" ]] && [[ -f "$CACHE_MODEL_DIR/snapshots/"*"/config.json" ]]; then
    log_info "Model already cached: ${HF_REPO}"
else
    ensure_hf_cli || {
        log_error "huggingface-cli required. Install with: pip install huggingface-hub"
        exit 1
    }

    log_info "Downloading ${HF_REPO}..."

    # Build args
    HF_ARGS=()
    [[ -n "${HF_TOKEN:-}" ]] && HF_ARGS+=("--token" "$HF_TOKEN")

    # Download the full model
    huggingface-cli download "$HF_REPO" "${HF_ARGS[@]}" || {
        log_error "Failed to download ${HF_REPO}"
        log_error "If gated, set HF_TOKEN environment variable"
        exit 1
    }

    log_info "Download complete"
fi

# =============================================================================
# Stage 2: Create vLLM model.json config
# =============================================================================
log_step "Creating vLLM configuration..."

# Create model.json for vLLM backend
cat > "${MODEL_DIR}/1/model.json" << EOF
{
    "model": "${HF_REPO}",
    "disable_log_requests": true,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
    "tensor_parallel_size": 1,
    "dtype": "auto",
    "quantization": "awq",
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
echo "Model: ${HF_REPO}"
echo "Config: ${MODEL_DIR}/1/model.json"
echo ""
echo "To start Triton with Qwen3:"
echo "  docker compose up -d triton"
echo ""
echo "To test:"
echo "  curl -X POST http://localhost:8000/v2/models/qwen3/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text_input\": \"Hello, how are you?\", \"max_tokens\": 50}'"
echo ""
