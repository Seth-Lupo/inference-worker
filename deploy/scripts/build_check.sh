#!/bin/bash
# Verify all model build artifacts are in place

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_REPO="${DEPLOY_DIR}/model_repository"
COSY_BUILD="${DEPLOY_DIR}/cosyvoice_build"

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

echo "=== Parakeet TDT ==="
check "${MODEL_REPO}/parakeet_tdt/config.pbtxt" 0 "config.pbtxt"
check "${MODEL_REPO}/parakeet_tdt/1/model.py" 0 "model.py"
check "${MODEL_REPO}/parakeet_tdt/1/vocab.txt" 0 "vocab.txt"
check "${MODEL_REPO}/parakeet_tdt/1/engines/encoder.engine" 1000000 "encoder.engine (TRT)"
check "${MODEL_REPO}/parakeet_tdt/1/engines/decoder_joint.engine" 1000000 "decoder_joint.engine (TRT)"

echo ""
echo "=== CosyVoice ==="
check "${COSY_BUILD}/CosyVoice2-0.5B/speech_tokenizer_v2.engine" 1000000 "speech_tokenizer_v2.engine (TRT)"
check "${COSY_BUILD}/CosyVoice2-0.5B/campplus.engine" 1000000 "campplus.engine (TRT)"
check_dir "${COSY_BUILD}/trtllm_engine" "trtllm_engine directory"

echo ""
echo "=== CosyVoice Model Repo ==="
for model in audio_tokenizer speaker_embedding cosyvoice2 token2wav; do
    check_dir "${MODEL_REPO}/${model}" "${model}/"
    check "${MODEL_REPO}/${model}/config.pbtxt" 0 "  config.pbtxt"
    check "${MODEL_REPO}/${model}/1/model.py" 0 "  model.py"
done

echo ""
echo "=== Summary ==="
echo -e "${GREEN}Passed: ${pass}${NC}  ${RED}Failed: ${fail}${NC}"
[[ $fail -eq 0 ]] && echo -e "${GREEN}All checks passed!${NC}" || echo -e "${RED}Some checks failed${NC}"
exit $fail
