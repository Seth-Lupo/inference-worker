#!/bin/bash
# =============================================================================
# Fix CosyVoice Models to Use GPU
#
# Patches config.pbtxt files to use KIND_GPU and ensures model.py uses CUDA.
#
# Usage:
#   ./fix_cosyvoice_gpu.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DEPLOY_DIR="$(get_deploy_dir)"
COSYVOICE_REPO="${DEPLOY_DIR}/model_repository/cosyvoice2_full"

echo "=============================================="
echo "Fixing CosyVoice Models for GPU"
echo "=============================================="
echo ""

# Check if cosyvoice2_full exists
if [[ ! -d "$COSYVOICE_REPO" ]]; then
    log_error "CosyVoice model repository not found at: $COSYVOICE_REPO"
    exit 1
fi

# Fix config.pbtxt files to use KIND_GPU
fix_config() {
    local model="$1"
    local config="${COSYVOICE_REPO}/${model}/config.pbtxt"

    if [[ ! -f "$config" ]]; then
        log_warn "Config not found: $config"
        return
    fi

    log_step "Fixing ${model}/config.pbtxt..."

    # Check if already has KIND_GPU
    if grep -q "KIND_GPU" "$config"; then
        log_info "Already configured for GPU"
        return
    fi

    # Replace KIND_CPU with KIND_GPU, or add instance_group if missing
    if grep -q "KIND_CPU" "$config"; then
        sed -i 's/KIND_CPU/KIND_GPU/g' "$config"
        log_info "Changed KIND_CPU -> KIND_GPU"
    elif grep -q "instance_group" "$config"; then
        # Has instance_group but no kind specified - add KIND_GPU
        sed -i 's/instance_group \[{/instance_group [{ kind: KIND_GPU,/g' "$config"
        log_info "Added KIND_GPU to instance_group"
    else
        # No instance_group at all - append one
        echo "" >> "$config"
        echo "instance_group [{ kind: KIND_GPU }]" >> "$config"
        log_info "Added instance_group with KIND_GPU"
    fi
}

# Fix each model's config
for model in audio_tokenizer speaker_embedding token2wav; do
    fix_config "$model"
done

echo ""
log_info "Config files updated!"
echo ""

# Check model.py files for CUDA usage
echo "=============================================="
echo "Checking model.py GPU usage"
echo "=============================================="

for model in audio_tokenizer speaker_embedding token2wav; do
    model_py="${COSYVOICE_REPO}/${model}/1/model.py"

    if [[ ! -f "$model_py" ]]; then
        continue
    fi

    echo ""
    echo "=== ${model}/model.py ==="

    # Look for device initialization
    if grep -q "torch\.device.*cuda" "$model_py"; then
        log_info "Uses torch.device('cuda')"
    elif grep -q "\.cuda()" "$model_py"; then
        log_info "Uses .cuda() method"
    elif grep -q "\.to.*cuda" "$model_py"; then
        log_info "Uses .to('cuda')"
    else
        log_warn "No CUDA device code found!"
        echo ""
        echo "Suggested fix - add to initialize():"
        echo "  self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
        echo "  self.model = self.model.to(self.device)"
    fi
done

echo ""
echo "=============================================="
echo "Next Steps"
echo "=============================================="
echo ""
echo "1. Restart Triton to apply config changes:"
echo "   docker compose restart triton"
echo ""
echo "2. Check logs for GPU usage:"
echo "   docker compose logs triton | grep -E '(audio_tokenizer|speaker_embedding|token2wav)'"
echo ""
echo "3. If models still use CPU, manually edit model.py files"
echo "   to add CUDA device handling"
