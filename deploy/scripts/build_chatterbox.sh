#!/bin/bash
# =============================================================================
# Build Chatterbox Turbo TensorRT Engines
#
# Downloads ONNX models from HuggingFace and builds TensorRT engines for
# all non-autoregressive models. The LM can optionally use vLLM.
#
# Architecture:
#   - embed_tokens: TensorRT (token embedding)
#   - speech_encoder: TensorRT (reference audio -> speaker embedding)
#   - conditional_decoder: ONNX Runtime (speech tokens -> audio waveform)
#   - language_model: vLLM or ONNX Runtime (autoregressive generation)
#
# Note: conditional_decoder has multiple inputs (speech_tokens, speaker_embeddings,
# speaker_features) making TensorRT compilation complex. Using ONNX Runtime.
#
# Usage:
#   ./build_chatterbox.sh [START_STAGE] [STOP_STAGE]
#   ./build_chatterbox.sh 0 3       # Run all stages
#   ./build_chatterbox.sh 1 1       # Only build TRT engines
#   ./build_chatterbox.sh 3 3       # Only build T3 for vLLM
#   ./build_chatterbox.sh cleanup   # Clean up
#
# Stages:
#    0: Download ONNX models from HuggingFace
#    1: Build TensorRT engines
#    2: Create Triton model repository
#    3: Build T3 model for vLLM (optional LLM serving)
#
# TensorRT Version:
#   Run inside nvcr.io/nvidia/tensorrt:24.12-py3
#   Runtime: nvcr.io/nvidia/tritonserver:24.12-py3
# =============================================================================

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================
readonly MODEL_NAME="chatterbox"
HF_REPO="${HF_REPO:-$(cfg_get 'chatterbox.hf_repo_onnx' 'ResembleAI/chatterbox-turbo-ONNX')}"
HF_REPO_T3="${HF_REPO_T3:-$(cfg_get 'chatterbox.hf_repo' 'ResembleAI/chatterbox-turbo')}"
PRECISION="${CHATTERBOX_PRECISION:-$(cfg_get 'chatterbox.precision' 'fp16')}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/chatterbox_build"
readonly CLONE_DIR="${WORK_DIR}/chatterbox-turbo-ONNX"
readonly ONNX_DIR="${WORK_DIR}/onnx"
readonly MODEL_REPO="${DEPLOY_DIR}/model_repository/tts"
readonly MODEL_DIR="${MODEL_REPO}/${MODEL_NAME}"
readonly ASSETS_DIR="${MODEL_REPO}/${MODEL_NAME}_assets"
readonly VLLM_MODELS="${DEPLOY_DIR}/vllm_models"
readonly T3_MODEL_DIR="${VLLM_MODELS}/t3_turbo"
readonly T3_TEMPLATE_DIR="${DEPLOY_DIR}/models/t3_turbo"

# ONNX model files
if [[ "$PRECISION" == "fp16" ]]; then
    EMBED_ONNX="embed_tokens_fp16.onnx"
    LM_ONNX="language_model_fp16.onnx"
    ENCODER_ONNX="speech_encoder_fp16.onnx"
    DECODER_ONNX="conditional_decoder_fp16.onnx"
else
    EMBED_ONNX="embed_tokens.onnx"
    LM_ONNX="language_model.onnx"
    ENCODER_ONNX="speech_encoder.onnx"
    DECODER_ONNX="conditional_decoder.onnx"
fi

# TensorRT dynamic shapes
# Format: min,opt,max for each input
declare -A TRT_SHAPES=(
    ["embed_tokens"]="input_ids:1x1,input_ids:4x256,input_ids:8x1024"
    ["speech_encoder"]="input_features:1x80x100,input_features:1x80x500,input_features:1x80x3000"
)
# Note: conditional_decoder uses ONNX Runtime (multi-input vocoder)

# =============================================================================
# Cleanup Handler
# =============================================================================
if [[ "${1:-}" == "cleanup" || "${1:-}" == "clean" ]]; then
    echo "=============================================="
    echo "Cleaning up Chatterbox build artifacts"
    echo "=============================================="
    case "${2:-}" in
        --all|-a)
            log_warn "Removing ALL build artifacts..."
            rm -rf "$WORK_DIR"
            rm -rf "$MODEL_DIR"
            rm -rf "$ASSETS_DIR"
            rm -rf "$T3_MODEL_DIR"
            log_info "Cleanup complete"
            ;;
        --t3)
            log_info "Removing T3 model..."
            rm -rf "$T3_MODEL_DIR"
            log_info "T3 model removed"
            ;;
        --engines|-e)
            log_info "Removing TRT engines only..."
            rm -rf "${WORK_DIR}/engines"
            rm -rf "${MODEL_DIR}/1/engines"
            log_info "Engines removed"
            ;;
        *)
            log_info "Removing downloads (keeping engines)..."
            rm -rf "$CLONE_DIR"
            log_info "Cleanup complete"
            log_info "To also remove model: $0 cleanup --all"
            log_info "To rebuild engines:   $0 cleanup --engines"
            ;;
    esac
    exit 0
fi

# Parse stages
START_STAGE="${1:-0}"
STOP_STAGE="${2:-3}"

echo "=============================================="
echo "Building Chatterbox Turbo TensorRT Engines"
echo "=============================================="
echo "Stages: ${START_STAGE} to ${STOP_STAGE}"
echo "Precision: ${PRECISION}"
echo ""

mkdir -p "$WORK_DIR"

# =============================================================================
# Stage 0: Download ONNX Models
# =============================================================================
stage_download() {
    log_step "Stage 0: Downloading ONNX models from HuggingFace..."

    mkdir -p "$ONNX_DIR"

    # Check if already downloaded (LM is the largest file)
    if is_real_file "${ONNX_DIR}/${LM_ONNX}" 100000000; then
        log_info "ONNX models already downloaded at: $ONNX_DIR"
        return 0
    fi

    # Clone repo (shallow, no LFS yet)
    hf_clone_shallow "$HF_REPO" "$CLONE_DIR" || {
        log_error "Failed to clone from HuggingFace"
        return 1
    }

    # Pull specific LFS files based on precision
    if [[ "$PRECISION" == "fp16" ]]; then
        hf_lfs_pull "$CLONE_DIR" \
            --include "onnx/*_fp16.onnx" \
            --include "onnx/*_fp16.onnx_data" \
            --include "*.json"
    else
        hf_lfs_pull "$CLONE_DIR" \
            --include "onnx/*.onnx" \
            --include "onnx/*.onnx_data" \
            --exclude "onnx/*_fp16*" \
            --include "*.json"
    fi

    # Copy to onnx_dir
    cp "${CLONE_DIR}/onnx/${EMBED_ONNX}"* "$ONNX_DIR/" 2>/dev/null || true
    cp "${CLONE_DIR}/onnx/${LM_ONNX}"* "$ONNX_DIR/" 2>/dev/null || true
    cp "${CLONE_DIR}/onnx/${ENCODER_ONNX}"* "$ONNX_DIR/" 2>/dev/null || true
    cp "${CLONE_DIR}/onnx/${DECODER_ONNX}"* "$ONNX_DIR/" 2>/dev/null || true
    cp "${CLONE_DIR}"/*.json "$ONNX_DIR/" 2>/dev/null || true

    # Verify key files
    local required_files=("$EMBED_ONNX" "$LM_ONNX" "$ENCODER_ONNX" "$DECODER_ONNX")
    for file in "${required_files[@]}"; do
        if [[ ! -f "${ONNX_DIR}/${file}" ]]; then
            log_warn "Missing ONNX file: ${file}"
        fi
    done

    log_info "ONNX models ready at: $ONNX_DIR"
    ls -lh "$ONNX_DIR"/*.onnx 2>/dev/null || true
}

# =============================================================================
# Stage 1: Build TensorRT Engines
# =============================================================================
build_engine() {
    local name="$1"
    local onnx_file="$2"
    local engine_file="$3"
    local shapes="$4"

    if [[ -f "$engine_file" ]]; then
        log_info "  $name: Engine exists, skipping"
        return 0
    fi

    if [[ ! -f "$onnx_file" ]]; then
        log_warn "  $name: ONNX not found, skipping"
        return 1
    fi

    log_info "  $name: Building TensorRT engine..."

    # Parse shapes (format: "name:min,name:opt,name:max")
    local min_shape opt_shape max_shape
    IFS=',' read -r min_shape opt_shape max_shape <<< "$shapes"

    local cmd="trtexec --onnx=$onnx_file --saveEngine=$engine_file"
    cmd+=" --minShapes=$min_shape --optShapes=$opt_shape --maxShapes=$max_shape"
    [[ "$PRECISION" == "fp16" ]] && cmd+=" --fp16"
    cmd+=" --workspace=4096"

    # Try to build
    if $cmd > "${engine_file%.engine}.log" 2>&1; then
        local size=$(get_file_size "$engine_file")
        log_info "  $name: Built successfully ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
        return 0
    else
        log_warn "  $name: TensorRT build failed, will use ONNX Runtime"
        cat "${engine_file%.engine}.log" | tail -20
        return 1
    fi
}

stage_build_engines() {
    log_step "Stage 1: Building TensorRT engines..."

    local engine_dir="${WORK_DIR}/engines"
    mkdir -p "$engine_dir"

    # Verify ONNX models exist
    if [[ ! -f "${ONNX_DIR}/${EMBED_ONNX}" ]]; then
        log_error "ONNX models not found. Run stage 0 first."
        return 1
    fi

    if ! command -v trtexec &>/dev/null; then
        log_error "trtexec not found. Run inside TensorRT container:"
        log_error "  docker compose --profile build run model-builder"
        return 1
    fi

    # Build each non-autoregressive model
    local success=0
    local failed=0

    # embed_tokens
    if build_engine "embed_tokens" \
        "${ONNX_DIR}/${EMBED_ONNX}" \
        "${engine_dir}/embed_tokens.engine" \
        "${TRT_SHAPES[embed_tokens]}"; then
        ((success++))
    else
        ((failed++))
    fi

    # speech_encoder (mel features input, no STFT needed in TRT)
    if build_engine "speech_encoder" \
        "${ONNX_DIR}/${ENCODER_ONNX}" \
        "${engine_dir}/speech_encoder.engine" \
        "${TRT_SHAPES[speech_encoder]}"; then
        ((success++))
    else
        ((failed++))
    fi

    # conditional_decoder - skip TensorRT (multiple inputs, use ONNX Runtime)
    log_info "  conditional_decoder: Using ONNX Runtime (multi-input vocoder)"

    echo ""
    log_info "TensorRT build complete: $success succeeded, $failed failed"

    # Language model stays as ONNX (for vLLM or ONNX Runtime)
    log_info "language_model: Kept as ONNX (use vLLM or ONNX Runtime)"
}

# =============================================================================
# Stage 2: Create Model Repository
# =============================================================================
stage_create_repo() {
    log_step "Stage 2: Creating Triton model repository..."

    local engine_dir="${WORK_DIR}/engines"

    mkdir -p "${MODEL_DIR}/1/engines"
    mkdir -p "${ASSETS_DIR}"

    # Copy TensorRT engines (or ONNX fallbacks)
    for model in embed_tokens speech_encoder conditional_decoder; do
        if [[ -f "${engine_dir}/${model}.engine" ]]; then
            cp "${engine_dir}/${model}.engine" "${MODEL_DIR}/1/engines/"
            log_info "  $model: TensorRT engine"
        else
            # Fallback to ONNX
            local onnx_name
            case $model in
                embed_tokens) onnx_name="$EMBED_ONNX" ;;
                speech_encoder) onnx_name="$ENCODER_ONNX" ;;
                conditional_decoder) onnx_name="$DECODER_ONNX" ;;
            esac
            cp "${ONNX_DIR}/${onnx_name}"* "${MODEL_DIR}/1/engines/" 2>/dev/null || true
            log_info "  $model: ONNX fallback"
        fi
    done

    # Copy language model ONNX (for the autoregressive loop)
    cp "${ONNX_DIR}/${LM_ONNX}"* "${MODEL_DIR}/1/engines/" 2>/dev/null || true
    log_info "  language_model: ONNX"

    # Copy config files to assets
    cp "${ONNX_DIR}"/*.json "${ASSETS_DIR}/" 2>/dev/null || true

    log_info "Model repository created: ${MODEL_DIR}"
}

# =============================================================================
# Stage 3: Build T3 Model for vLLM
# =============================================================================
stage_build_t3() {
    log_step "Stage 3: Building T3 model for vLLM..."

    mkdir -p "$T3_MODEL_DIR"

    # T3 model files to download
    local t3_files=(
        "t3_turbo_v1.safetensors"
        "t3_turbo_v1.yaml"
        "tokenizer_config.json"
        "vocab.json"
        "merges.txt"
    )

    # Check if model already exists
    if is_real_file "${T3_MODEL_DIR}/model.safetensors" 100000000; then
        local size
        size=$(get_file_size "${T3_MODEL_DIR}/model.safetensors")
        log_info "T3 model already downloaded ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
    else
        # Download using common function
        hf_download "$HF_REPO_T3" "$T3_MODEL_DIR" "${t3_files[@]}" || {
            log_error "Failed to download T3 model"
            return 1
        }

        # Rename to standard HuggingFace format
        if [[ -f "${T3_MODEL_DIR}/t3_turbo_v1.safetensors" ]]; then
            mv "${T3_MODEL_DIR}/t3_turbo_v1.safetensors" "${T3_MODEL_DIR}/model.safetensors"
            log_info "Renamed to model.safetensors"
        fi
    fi

    # Copy HuggingFace-compatible config files from template
    log_info "Setting up HuggingFace-compatible configuration..."
    if [[ -d "$T3_TEMPLATE_DIR" ]]; then
        cp "${T3_TEMPLATE_DIR}/config.json" "${T3_MODEL_DIR}/" 2>/dev/null || true
        cp "${T3_TEMPLATE_DIR}/configuration_t3.py" "${T3_MODEL_DIR}/" 2>/dev/null || true
        cp "${T3_TEMPLATE_DIR}/modeling_t3.py" "${T3_MODEL_DIR}/" 2>/dev/null || true
    else
        log_warn "T3 template directory not found: $T3_TEMPLATE_DIR"
        log_warn "You may need to create config.json and modeling_t3.py manually"
    fi

    # Verify
    if [[ -f "${T3_MODEL_DIR}/model.safetensors" ]]; then
        local size
        size=$(get_file_size "${T3_MODEL_DIR}/model.safetensors")
        log_info "T3 model ready: $(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")"
    else
        log_error "T3 model download failed"
        return 1
    fi

    log_info "T3 model directory: ${T3_MODEL_DIR}"
}

# =============================================================================
# Summary
# =============================================================================
show_summary() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}Chatterbox Build Complete${NC}"
    echo "=============================================="
    echo ""
    echo "Models built:"
    echo "  - Triton TTS: ${MODEL_DIR}"
    [[ $STOP_STAGE -ge 3 ]] && echo "  - vLLM T3:    ${T3_MODEL_DIR}"
    echo ""

    if [[ -d "${MODEL_DIR}/1/engines" ]]; then
        echo "Engines:"
        ls -lh "${MODEL_DIR}/1/engines/"*.engine 2>/dev/null || echo "  (ONNX fallback)"
        echo ""
    fi

    echo "Next steps:"
    echo "  1. Start Triton: docker compose up -d triton"
    [[ $STOP_STAGE -ge 3 ]] && echo "  2. Start vLLM T3: docker compose --profile t3 up -d"
    echo "  3. Start worker: docker compose up -d worker"
    echo ""
    echo "Cleanup options:"
    echo "  $0 cleanup            # Remove downloads, keep model"
    echo "  $0 cleanup --engines  # Rebuild TRT engines"
    echo "  $0 cleanup --t3       # Remove T3 model"
    echo "  $0 cleanup --all      # Remove everything"
}

# =============================================================================
# Main
# =============================================================================
main() {
    # Stage 0: Download ONNX models
    if [[ $START_STAGE -le 0 && $STOP_STAGE -ge 0 ]]; then
        stage_download
    fi

    # Stage 1: Build TensorRT engines
    if [[ $START_STAGE -le 1 && $STOP_STAGE -ge 1 ]]; then
        stage_build_engines
    fi

    # Stage 2: Create Triton model repository
    if [[ $START_STAGE -le 2 && $STOP_STAGE -ge 2 ]]; then
        stage_create_repo
    fi

    # Stage 3: Build T3 for vLLM
    if [[ $START_STAGE -le 3 && $STOP_STAGE -ge 3 ]]; then
        stage_build_t3
    fi

    show_summary
}

main
