#!/bin/bash
# =============================================================================
# Clean Built Models and Caches
# Removes downloaded models, built engines, and caches
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "=============================================="
echo "Clean Voice Agent Models"
echo "=============================================="
echo ""

# What to clean
CLEAN_ALL=false
CLEAN_ENGINES=false
CLEAN_DOWNLOADS=false
CLEAN_CACHE=false
CLEAN_MODEL=""

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all           Clean everything"
    echo "  --engines       Clean built TensorRT engines only"
    echo "  --downloads     Clean downloaded model files only"
    echo "  --cache         Clean engine cache only"
    echo "  --model NAME    Clean specific model (silero_vad, parakeet_tdt, qwen3_8b, cosyvoice2)"
    echo "  --dry-run       Show what would be deleted without deleting"
    echo "  --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Clean everything"
    echo "  $0 --model qwen3_8b         # Clean only Qwen3"
    echo "  $0 --engines --dry-run      # Preview engine cleanup"
}

DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CLEAN_ALL=true
            shift
            ;;
        --engines)
            CLEAN_ENGINES=true
            shift
            ;;
        --downloads)
            CLEAN_DOWNLOADS=true
            shift
            ;;
        --cache)
            CLEAN_CACHE=true
            shift
            ;;
        --model)
            CLEAN_MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Default to showing help if no options
if [ "$CLEAN_ALL" = false ] && [ "$CLEAN_ENGINES" = false ] && \
   [ "$CLEAN_DOWNLOADS" = false ] && [ "$CLEAN_CACHE" = false ] && \
   [ -z "$CLEAN_MODEL" ]; then
    show_help
    exit 0
fi

do_clean() {
    local path="$1"
    local desc="$2"

    if [ -e "$path" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY-RUN]${NC} Would delete: $path ($desc)"
            if [ -d "$path" ]; then
                du -sh "$path" 2>/dev/null | cut -f1 | xargs -I {} echo "         Size: {}"
            fi
        else
            echo -e "${RED}Deleting:${NC} $path ($desc)"
            rm -rf "$path"
        fi
    fi
}

# =============================================================================
# Clean specific model
# =============================================================================
if [ -n "$CLEAN_MODEL" ]; then
    echo "Cleaning model: $CLEAN_MODEL"
    echo ""

    case $CLEAN_MODEL in
        silero_vad)
            do_clean "${DEPLOY_DIR}/model_repository/silero_vad/1/model.onnx" "Silero VAD ONNX"
            ;;
        parakeet_tdt|parakeet)
            do_clean "${DEPLOY_DIR}/model_repository/parakeet_tdt/1" "Parakeet model files"
            do_clean "${DEPLOY_DIR}/parakeet_build" "Parakeet build directory"
            ;;
        qwen3_8b|qwen3|qwen)
            do_clean "${DEPLOY_DIR}/model_repository/qwen3_8b/1/engine" "Qwen3 TensorRT engine"
            do_clean "${DEPLOY_DIR}/qwen3_build" "Qwen3 build directory"
            ;;
        cosyvoice2|cosyvoice)
            do_clean "${DEPLOY_DIR}/model_repository/cosyvoice2_full" "CosyVoice2 model repo"
            do_clean "${DEPLOY_DIR}/cosyvoice_build" "CosyVoice2 build directory"
            ;;
        *)
            echo -e "${RED}Unknown model: $CLEAN_MODEL${NC}"
            echo "Valid models: silero_vad, parakeet_tdt, qwen3_8b, cosyvoice2"
            exit 1
            ;;
    esac
fi

# =============================================================================
# Clean engines
# =============================================================================
if [ "$CLEAN_ALL" = true ] || [ "$CLEAN_ENGINES" = true ]; then
    echo "Cleaning TensorRT engines..."
    echo ""

    # Find all engine files
    find "${DEPLOY_DIR}/model_repository" -name "*.engine" -o -name "*.plan" 2>/dev/null | while read f; do
        do_clean "$f" "TensorRT engine"
    done

    do_clean "${DEPLOY_DIR}/model_repository/qwen3_8b/1/engine" "Qwen3 engine dir"
    do_clean "${DEPLOY_DIR}/qwen3_build/engine_"* "Qwen3 built engines"
    do_clean "${DEPLOY_DIR}/cosyvoice_build/trt_engines_"* "CosyVoice engines"
fi

# =============================================================================
# Clean downloads
# =============================================================================
if [ "$CLEAN_ALL" = true ] || [ "$CLEAN_DOWNLOADS" = true ]; then
    echo "Cleaning downloaded models..."
    echo ""

    do_clean "${DEPLOY_DIR}/qwen3_build/Qwen3-8B" "Qwen3 HuggingFace download"
    do_clean "${DEPLOY_DIR}/cosyvoice_build/cosyvoice2_llm" "CosyVoice LLM download"
    do_clean "${DEPLOY_DIR}/cosyvoice_build/CosyVoice2-0.5B" "CosyVoice ModelScope download"
    do_clean "${DEPLOY_DIR}/cosyvoice_build/CosyVoice" "CosyVoice git repo"
    do_clean "${DEPLOY_DIR}/parakeet_build" "Parakeet downloads"
    do_clean "${DEPLOY_DIR}/model_sources" "Legacy model sources"
fi

# =============================================================================
# Clean cache
# =============================================================================
if [ "$CLEAN_ALL" = true ] || [ "$CLEAN_CACHE" = true ]; then
    echo "Cleaning caches..."
    echo ""

    do_clean "${DEPLOY_DIR}/engine_cache" "Engine cache"
    do_clean "${DEPLOY_DIR}/qwen3_build/checkpoint_"* "Qwen3 checkpoints"
    do_clean "${DEPLOY_DIR}/cosyvoice_build/trt_weights_"* "CosyVoice TRT weights"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run complete. No files were deleted.${NC}"
    echo "Run without --dry-run to actually delete files."
else
    echo -e "${GREEN}Cleanup complete.${NC}"
fi

# Show remaining disk usage
echo ""
echo "Current disk usage:"
du -sh "${DEPLOY_DIR}/model_repository" 2>/dev/null || echo "  model_repository: (empty)"
du -sh "${DEPLOY_DIR}"/*_build 2>/dev/null || echo "  build dirs: (empty)"
