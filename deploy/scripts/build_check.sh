#!/bin/bash
# Verify all model build artifacts and Triton deployment paths

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_REPO="${DEPLOY_DIR}/model_repository"
COSY_BUILD="${DEPLOY_DIR}/cosyvoice_build"
QWEN_BUILD="${DEPLOY_DIR}/qwen3_build"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass=0 fail=0

check() {
    if [[ -f "$1" ]] && [[ $(stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null) -gt ${2:-0} ]]; then
        echo -e "${GREEN}✓${NC} $3"
        ((pass++))
    else
        echo -e "${RED}✗${NC} $3"
        ((fail++))
    fi
}

check_dir() {
    if [[ -d "$1" ]]; then
        echo -e "${GREEN}✓${NC} $2"
        ((pass++))
    else
        echo -e "${RED}✗${NC} $2"
        ((fail++))
    fi
}

echo "=============================================="
echo "  Build Artifacts (what was built)"
echo "=============================================="

echo ""
echo "=== Parakeet TDT (ASR) - in model_repo ==="
check "${MODEL_REPO}/parakeet_tdt/1/engines/encoder.engine" 1000000 "encoder.engine"
check "${MODEL_REPO}/parakeet_tdt/1/engines/decoder_joint.engine" 1000000 "decoder_joint.engine"
check "${MODEL_REPO}/parakeet_tdt/1/vocab.txt" 0 "vocab.txt"

echo ""
echo "=== Qwen3 8B (LLM) - in qwen3_build ==="
QWEN_ENGINE=$(find "${QWEN_BUILD}" -maxdepth 1 -type d -name "engine_*" 2>/dev/null | head -1)
if [[ -n "$QWEN_ENGINE" ]]; then
    check "${QWEN_ENGINE}/rank0.engine" 1000000 "$(basename "$QWEN_ENGINE")/rank0.engine"
    check "${QWEN_ENGINE}/config.json" 0 "$(basename "$QWEN_ENGINE")/config.json"
else
    echo -e "${RED}✗${NC} engine_* directory not found"
    ((fail++))
fi

echo ""
echo "=== CosyVoice (TTS) - in cosyvoice_build ==="
check "${COSY_BUILD}/CosyVoice2-0.5B/speech_tokenizer_v2.engine" 1000000 "speech_tokenizer_v2.engine"
check "${COSY_BUILD}/CosyVoice2-0.5B/campplus.engine" 1000000 "campplus.engine"
COSY_ENGINE=$(find "${COSY_BUILD}" -maxdepth 1 -type d \( -name "trtllm_engine" -o -name "trt_engines*" \) 2>/dev/null | head -1)
if [[ -n "$COSY_ENGINE" ]]; then
    check "${COSY_ENGINE}/rank0.engine" 1000000 "$(basename "$COSY_ENGINE")/rank0.engine"
else
    echo -e "${RED}✗${NC} trt_engines* directory not found"
    ((fail++))
fi

echo ""
echo "=============================================="
echo "  Triton Deployment (what Triton serves)"
echo "=============================================="

echo ""
echo "=== model_repository/ ==="
for model in parakeet_tdt qwen3_8b; do
    check_dir "${MODEL_REPO}/${model}" "${model}/"
    check "${MODEL_REPO}/${model}/config.pbtxt" 0 "  config.pbtxt"
done

echo ""
echo "=== CosyVoice Models (from templates) ==="
COSY_TEMPLATES="${MODEL_REPO}/cosyvoice_templates"
for model in audio_tokenizer speaker_embedding cosyvoice2 token2wav; do
    check_dir "${COSY_TEMPLATES}/${model}" "${model}/"
    check "${COSY_TEMPLATES}/${model}/config.pbtxt" 0 "  config.pbtxt"
    check "${COSY_TEMPLATES}/${model}/1/model.py" 0 "  model.py"
done

echo ""
echo "=============================================="
echo "  Engine Path References"
echo "=============================================="
echo -e "${YELLOW}Verify these paths match config.pbtxt parameters:${NC}"
echo "  Qwen3:    ${QWEN_ENGINE:-NOT FOUND}"
echo "  CosyVoice LLM: ${COSY_ENGINE:-NOT FOUND}"
echo "  CosyVoice TRT: ${COSY_BUILD}/CosyVoice2-0.5B/"

echo ""
echo "=============================================="
echo -e "  ${GREEN}Passed: ${pass}${NC}  ${RED}Failed: ${fail}${NC}"
echo "=============================================="
[[ $fail -eq 0 ]] && echo -e "${GREEN}All checks passed!${NC}" || echo -e "${RED}Some checks failed - run builds or check paths${NC}"
exit $fail
