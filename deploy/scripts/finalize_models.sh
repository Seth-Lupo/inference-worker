#!/bin/bash
# =============================================================================
# Finalize Model Repository for Triton
#
# This script prepares all models for Triton by:
# 1. Setting correct engine paths in configs
# 2. Copying CosyVoice templates to deployment location
# 3. Verifying all required files are in place
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_REPO="${DEPLOY_DIR}/model_repository"
COSY_BUILD="${DEPLOY_DIR}/cosyvoice_build"
QWEN_BUILD="${DEPLOY_DIR}/qwen3_build"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

echo "=============================================="
echo "  Finalizing Model Repository for Triton"
echo "=============================================="

# =============================================================================
# 1. Qwen3 8B - Update engine path in config
# =============================================================================
echo ""
log_info "=== Qwen3 8B ==="

QWEN_ENGINE=$(find "${QWEN_BUILD}" -maxdepth 1 -type d -name "engine_*" 2>/dev/null | head -1)
if [[ -n "$QWEN_ENGINE" ]]; then
    ENGINE_NAME=$(basename "$QWEN_ENGINE")
    # In container, qwen3_build is mounted at /qwen3_build
    CONTAINER_PATH="/qwen3_build/${ENGINE_NAME}"

    log_info "Updating Qwen3 config to use: ${CONTAINER_PATH}"

    # Update config.pbtxt
    if [[ -f "${MODEL_REPO}/qwen3_8b/config.pbtxt" ]]; then
        sed -i.bak "s|gpt_model_path.*|gpt_model_path\"\n  value: { string_value: \"${CONTAINER_PATH}\" }|" \
            "${MODEL_REPO}/qwen3_8b/config.pbtxt" 2>/dev/null || \
        sed -i '' "s|string_value: \"/models/qwen3_8b/1/engine\"|string_value: \"${CONTAINER_PATH}\"|" \
            "${MODEL_REPO}/qwen3_8b/config.pbtxt"
        log_info "Updated config.pbtxt"
    fi
else
    log_warn "Qwen engine not found in ${QWEN_BUILD}"
fi

# =============================================================================
# 2. CosyVoice - Copy templates and set paths
# =============================================================================
echo ""
log_info "=== CosyVoice ==="

COSY_TEMPLATES="${MODEL_REPO}/cosyvoice_templates"
COSY_DEPLOY="${MODEL_REPO}/cosyvoice2_full"

# CosyVoice model directory (where engines are)
COSY_MODEL_DIR="${COSY_BUILD}/CosyVoice2-0.5B"
COSY_ENGINE_DIR=$(find "${COSY_BUILD}" -maxdepth 1 -type d \( -name "trtllm_engine" -o -name "trt_engines*" \) 2>/dev/null | head -1)

if [[ -d "$COSY_TEMPLATES" ]]; then
    log_info "Creating deployment directory: ${COSY_DEPLOY}"
    rm -rf "$COSY_DEPLOY"
    mkdir -p "$COSY_DEPLOY"

    # Copy each model and substitute paths
    for model in audio_tokenizer speaker_embedding cosyvoice2 token2wav token2wav_dit cosyvoice2_dit tensorrt_llm; do
        src="${COSY_TEMPLATES}/${model}"
        dst="${COSY_DEPLOY}/${model}"

        if [[ -d "$src" ]]; then
            log_info "Copying ${model}..."
            cp -r "$src" "$dst"

            # Update config.pbtxt with real paths
            # In container: cosyvoice_build is at /models/../cosyvoice_build (relative)
            # or we can use absolute container paths
            CONTAINER_MODEL_DIR="/models/../cosyvoice_build/CosyVoice2-0.5B"
            CONTAINER_ENGINE_DIR="/models/../cosyvoice_build/$(basename "${COSY_ENGINE_DIR:-trt_engines_bfloat16}")"

            if [[ -f "${dst}/config.pbtxt" ]]; then
                # Replace ${model_dir} placeholder
                sed -i.bak "s|\${model_dir}|${CONTAINER_MODEL_DIR}|g" "${dst}/config.pbtxt" 2>/dev/null || \
                sed -i '' "s|\${model_dir}|${CONTAINER_MODEL_DIR}|g" "${dst}/config.pbtxt"

                # Replace ${engine_dir} placeholder
                sed -i.bak "s|\${engine_dir}|${CONTAINER_ENGINE_DIR}|g" "${dst}/config.pbtxt" 2>/dev/null || \
                sed -i '' "s|\${engine_dir}|${CONTAINER_ENGINE_DIR}|g" "${dst}/config.pbtxt"

                rm -f "${dst}/config.pbtxt.bak"
            fi
        fi
    done

    log_info "CosyVoice models deployed to ${COSY_DEPLOY}"
else
    log_warn "CosyVoice templates not found at ${COSY_TEMPLATES}"
fi

# =============================================================================
# 3. Parakeet - Already in model_repo, just verify
# =============================================================================
echo ""
log_info "=== Parakeet TDT ==="

if [[ -d "${MODEL_REPO}/parakeet_tdt/1/engines" ]]; then
    log_info "Parakeet engines found"
    ls -la "${MODEL_REPO}/parakeet_tdt/1/engines/"
else
    log_warn "Parakeet engines directory not found"
fi

# =============================================================================
# 4. Update docker-compose volume mounts
# =============================================================================
echo ""
log_info "=== Checking docker-compose.yml ==="

COMPOSE_FILE="${DEPLOY_DIR}/docker-compose.yml"
if [[ -f "$COMPOSE_FILE" ]]; then
    # Check if cosyvoice_build is mounted
    if grep -q "cosyvoice_build" "$COMPOSE_FILE"; then
        log_info "cosyvoice_build volume mount found"
    else
        log_warn "Adding cosyvoice_build volume mount..."
        # Add volume mount after qwen3_build
        sed -i.bak '/qwen3_build/a\      - ./cosyvoice_build:/models/../cosyvoice_build' "$COMPOSE_FILE" 2>/dev/null || \
        log_warn "Could not auto-add volume mount. Please add manually:"
        echo "      - ./cosyvoice_build:/cosyvoice_build"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo -e "  ${GREEN}Finalization Complete${NC}"
echo "=============================================="
echo ""
echo "Model repository structure:"
echo "  ${MODEL_REPO}/"
echo "  ├── parakeet_tdt/      (ASR)"
echo "  ├── qwen3_8b/          (LLM)"
echo "  └── cosyvoice2_full/   (TTS - 7 submodels)"
echo ""
echo "Next steps:"
echo "  1. Run: ./scripts/build_check.sh"
echo "  2. Start Triton: docker compose up -d triton"
echo "  3. Check logs: docker logs -f voice-agent-triton"
