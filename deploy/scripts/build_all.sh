#!/bin/bash
# =============================================================================
# Build All Models
# Orchestrates building all voice agent models
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=============================================="
echo "Building All Voice Agent Models"
echo "=============================================="
echo ""
echo "This will build:"
echo "  1. Silero VAD (ONNX, ~2MB)"
echo "  2. Parakeet TDT 0.6B (ONNX, ~600MB)"
echo "  3. Qwen3 8B (TensorRT-LLM, ~10GB)"
echo "  4. CosyVoice 2 (TensorRT-LLM, ~2GB)"
echo ""
echo "Total disk space needed: ~15-20GB"
echo "Build time: 30-60 minutes (depends on GPU)"
echo ""

read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Track failures
FAILED=()

# 1. Silero VAD
echo ""
echo "========== 1/4: Silero VAD =========="
if bash "${SCRIPT_DIR}/build_silero_vad.sh"; then
    echo -e "${GREEN}Silero VAD: SUCCESS${NC}"
else
    echo -e "${RED}Silero VAD: FAILED${NC}"
    FAILED+=("silero_vad")
fi

# 2. Parakeet TDT
echo ""
echo "========== 2/4: Parakeet TDT =========="
if bash "${SCRIPT_DIR}/build_parakeet.sh"; then
    echo -e "${GREEN}Parakeet TDT: SUCCESS${NC}"
else
    echo -e "${RED}Parakeet TDT: FAILED${NC}"
    FAILED+=("parakeet_tdt")
fi

# 3. Qwen3 8B
echo ""
echo "========== 3/4: Qwen3 8B =========="
if bash "${SCRIPT_DIR}/build_qwen3.sh"; then
    echo -e "${GREEN}Qwen3 8B: SUCCESS${NC}"
else
    echo -e "${RED}Qwen3 8B: FAILED${NC}"
    FAILED+=("qwen3_8b")
fi

# 4. CosyVoice 2
echo ""
echo "========== 4/4: CosyVoice 2 =========="
if bash "${SCRIPT_DIR}/build_cosyvoice.sh" 0 2; then
    echo -e "${GREEN}CosyVoice 2: SUCCESS${NC}"
else
    echo -e "${RED}CosyVoice 2: FAILED${NC}"
    FAILED+=("cosyvoice2")
fi

# Summary
echo ""
echo "=============================================="
echo "Build Summary"
echo "=============================================="

if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}All models built successfully!${NC}"
else
    echo -e "${RED}Failed models: ${FAILED[*]}${NC}"
    echo ""
    echo "To retry individual models:"
    for model in "${FAILED[@]}"; do
        echo "  ./scripts/build_${model}.sh"
    done
fi

echo ""
echo "Model repository:"
ls -la "$(dirname "$SCRIPT_DIR")/model_repository/"
