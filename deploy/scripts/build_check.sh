#!/bin/bash
# Verify all model build artifacts are in place

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_REPO="${DEPLOY_DIR}/model_repository"
COSY_BUILD="${DEPLOY_DIR}/cosyvoice_build"
QWEN_BUILD="${DEPLOY_DIR}/qwen3_build"

RED='\033[0;31m'
GREEN='\033[0;32m'
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

echo "=== Parakeet TDT (ASR) ==="
check "${MODEL_REPO}/parakeet_tdt/config.pbtxt" 0 "config.pbtxt"
check "${MODEL_REPO}/parakeet_tdt/1/model.py" 0 "model.py"
check "${MODEL_REPO}/parakeet_tdt/1/vocab.txt" 0 "vocab.txt"
check "${MODEL_REPO}/parakeet_tdt/1/engines/encoder.engine" 1000000 "encoder.engine (TRT)"
check "${MODEL_REPO}/parakeet_tdt/1/engines/decoder_joint.engine" 1000000 "decoder_joint.engine (TRT)"

echo ""
echo "=== Qwen3 8B (LLM) ==="
check "${MODEL_REPO}/qwen3_8b/config.pbtxt" 0 "config.pbtxt"
check_dir "${MODEL_REPO}/qwen3_8b/1" "model version 1/"
# Check for TRT-LLM engine (could be fp16, int8, or int4)
QWEN_ENGINE=$(find "${QWEN_BUILD}" -maxdepth 1 -type d -name "engine_*" 2>/dev/null | head -1)
if [[ -n "$QWEN_ENGINE" ]]; then
    check_dir "$QWEN_ENGINE" "trtllm engine: $(basename "$QWEN_ENGINE")"
    check "${QWEN_ENGINE}/rank0.engine" 1000000 "  rank0.engine"
    check "${QWEN_ENGINE}/config.json" 0 "  config.json"
else
    echo -e "${RED}✗${NC} trtllm engine directory"
    ((fail++))
fi

echo ""
echo "=== CosyVoice (TTS) ==="
check "${COSY_BUILD}/CosyVoice2-0.5B/speech_tokenizer_v2.engine" 1000000 "speech_tokenizer_v2.engine (TRT)"
check "${COSY_BUILD}/CosyVoice2-0.5B/campplus.engine" 1000000 "campplus.engine (TRT)"
# Check for TRT-LLM engine (could be trtllm_engine or trt_engines_*)
COSY_ENGINE=$(find "${COSY_BUILD}" -maxdepth 1 -type d \( -name "trtllm_engine" -o -name "trt_engines*" \) 2>/dev/null | head -1)
if [[ -n "$COSY_ENGINE" ]]; then
    check_dir "$COSY_ENGINE" "trtllm engine: $(basename "$COSY_ENGINE")"
    check "${COSY_ENGINE}/rank0.engine" 1000000 "  rank0.engine"
else
    echo -e "${RED}✗${NC} trtllm engine directory"
    ((fail++))
fi

echo ""
echo "=== CosyVoice Templates ==="
COSY_TEMPLATES="${MODEL_REPO}/cosyvoice_templates"
for model in audio_tokenizer speaker_embedding cosyvoice2 token2wav; do
    check_dir "${COSY_TEMPLATES}/${model}" "${model}/"
    check "${COSY_TEMPLATES}/${model}/config.pbtxt" 0 "  config.pbtxt"
    check "${COSY_TEMPLATES}/${model}/1/model.py" 0 "  model.py"
done

echo ""
echo "=== Summary ==="
echo -e "${GREEN}Passed: ${pass}${NC}  ${RED}Failed: ${fail}${NC}"
[[ $fail -eq 0 ]] && echo -e "${GREEN}All checks passed!${NC}" || echo -e "${RED}Some checks failed${NC}"
exit $fail
