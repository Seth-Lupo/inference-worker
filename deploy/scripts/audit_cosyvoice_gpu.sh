#!/bin/bash
# =============================================================================
# Audit CosyVoice Models for GPU Usage
#
# Checks if CosyVoice submodels are using GPU and shows how to fix them.
#
# Usage:
#   ./audit_cosyvoice_gpu.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DEPLOY_DIR="$(get_deploy_dir)"
COSYVOICE_REPO="${DEPLOY_DIR}/model_repository/cosyvoice2_full"

echo "=============================================="
echo "Auditing CosyVoice Models for GPU Usage"
echo "=============================================="
echo ""

# Check if cosyvoice2_full exists
if [[ ! -d "$COSYVOICE_REPO" ]]; then
    log_error "CosyVoice model repository not found at: $COSYVOICE_REPO"
    log_info "Run build_cosyvoice.sh first"
    exit 1
fi

# List models
echo "Models in ${COSYVOICE_REPO}:"
ls -la "$COSYVOICE_REPO" | grep "^d" | awk '{print "  " $NF}'
echo ""

# Check each Python model for GPU usage
for model in audio_tokenizer speaker_embedding token2wav cosyvoice2; do
    model_py="${COSYVOICE_REPO}/${model}/1/model.py"

    if [[ -f "$model_py" ]]; then
        echo "=== ${model} ==="

        # Check for CUDA/GPU usage
        if grep -q "\.cuda()\|\.to.*cuda\|device.*cuda\|torch\.device" "$model_py"; then
            log_info "GPU code found"
            grep -n "cuda\|device" "$model_py" | head -5
        else
            log_warn "No GPU code found - likely running on CPU!"
        fi

        # Check for TensorRT
        if grep -q "tensorrt\|trt\|\.plan" "$model_py"; then
            log_info "TensorRT code found"
        fi

        echo ""
    else
        log_warn "${model}/1/model.py not found"
    fi
done

# Check config.pbtxt for instance_group settings
echo "=== Config Instance Groups ==="
for model in audio_tokenizer speaker_embedding token2wav cosyvoice2; do
    config="${COSYVOICE_REPO}/${model}/config.pbtxt"

    if [[ -f "$config" ]]; then
        echo "${model}:"
        if grep -q "KIND_GPU" "$config"; then
            log_info "  KIND_GPU configured"
        elif grep -q "KIND_CPU" "$config"; then
            log_warn "  KIND_CPU configured - should be KIND_GPU!"
        else
            log_warn "  No instance_group found"
        fi

        # Show instance_group
        grep -A3 "instance_group" "$config" 2>/dev/null | head -5 || true
    fi
done

echo ""
echo "=============================================="
echo "Recommendations:"
echo "=============================================="
echo ""
echo "1. For Python models, ensure model.py uses:"
echo "   self.device = torch.device('cuda')"
echo "   model = model.to(self.device)"
echo ""
echo "2. For config.pbtxt, set instance_group:"
echo "   instance_group [{ kind: KIND_GPU }]"
echo ""
echo "3. Consider converting to TensorRT for best performance"
