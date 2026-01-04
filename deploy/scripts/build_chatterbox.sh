#!/bin/bash
# =============================================================================
# Build Chatterbox with T3 vLLM Backend
#
# Downloads model weights from HuggingFace and sets up Chatterbox TTS with
# T3 running as a native Triton vLLM backend for high-performance inference.
#
# Architecture:
#   - T3: Triton vLLM backend (text -> speech tokens via Llama-based model)
#   - S3Gen: Python backend (speech tokens -> audio via flow matching + HiFiGAN)
#   - VoiceEncoder: Python backend (reference audio -> speaker embedding)
#   - Progressive streaming: chunks 4,8,16,32,32 with diffusion steps 1,2,5,7,10
#
# T3 is served via Triton's native vLLM backend with trust_remote_code=true.
# Chatterbox Python backend calls T3 via gRPC and runs S3Gen locally.
#
# Usage:
#   ./build_chatterbox.sh [START_STAGE] [STOP_STAGE]
#   ./build_chatterbox.sh 0 3       # Run all stages
#   ./build_chatterbox.sh 3 3       # Only download T3 and assets
#   ./build_chatterbox.sh cleanup   # Clean up
#
# Stages:
#    0: Download ONNX models from HuggingFace (legacy)
#    1: Build TensorRT engines (legacy)
#    2: Create Triton model repository (legacy)
#    3: Download T3 weights and assets for vLLM backend
#
# Container: nvcr.io/nvidia/tritonserver:24.12-vllm-python-py3
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
# T3 weights for vLLM backend (mounted at /models/t3_weights in container)
readonly T3_WEIGHTS_DIR="${DEPLOY_DIR}/models/t3_weights"

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
# For multiple inputs, separate with space
declare -A TRT_SHAPES=(
    ["embed_tokens"]="input_ids:1x1,input_ids:4x256,input_ids:8x1024"
    ["speech_encoder"]="input_features:1x80x100,input_features:1x80x500,input_features:1x80x3000"
    ["conditional_decoder"]="speech_tokens:1x4,speech_tokens:1x32,speech_tokens:1x64"
)
# Note: conditional_decoder uses progressive chunks 4,8,16,32,32 so min=4, opt=32, max=64

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
            # Keep model code files but remove weights
            rm -f "${T3_WEIGHTS_DIR}/t3_cfg.safetensors"
            rm -f "${T3_WEIGHTS_DIR}/model.safetensors"
            rm -f "${T3_WEIGHTS_DIR}/tokenizer.json"
            rm -f "${T3_WEIGHTS_DIR}/conditioning.pt"
            log_info "Cleanup complete"
            ;;
        --t3)
            log_info "Removing T3 weights..."
            rm -f "${T3_WEIGHTS_DIR}/t3_cfg.safetensors"
            rm -f "${T3_WEIGHTS_DIR}/model.safetensors"
            rm -f "${T3_WEIGHTS_DIR}/conditioning.pt"
            log_info "T3 weights removed"
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

    # conditional_decoder (vocoder: speech tokens -> audio)
    if build_engine "conditional_decoder" \
        "${ONNX_DIR}/${DECODER_ONNX}" \
        "${engine_dir}/conditional_decoder.engine" \
        "${TRT_SHAPES[conditional_decoder]}"; then
        ((success++))
    else
        ((failed++))
        log_info "  conditional_decoder: TRT failed, will use ONNX Runtime fallback"
    fi

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
# Stage 3: Download T3 weights and Chatterbox assets for vLLM backend
# =============================================================================
stage_build_t3() {
    log_step "Stage 3: Setting up T3 vLLM backend and Chatterbox assets..."

    mkdir -p "$T3_WEIGHTS_DIR"
    mkdir -p "$ASSETS_DIR"

    # -------------------------------------------------------------------------
    # 3a: Download T3 model weights to vLLM model directory
    # -------------------------------------------------------------------------
    log_info "Downloading T3 Turbo weights for vLLM backend..."

    if is_real_file "${T3_WEIGHTS_DIR}/t3_cfg.safetensors" 100000000; then
        local size
        size=$(get_file_size "${T3_WEIGHTS_DIR}/t3_cfg.safetensors")
        log_info "T3 weights already downloaded ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
    else
        # Download from ResembleAI/chatterbox (main repo has the weights)
        hf_download "ResembleAI/chatterbox" "$T3_WEIGHTS_DIR" "t3_cfg.safetensors" "tokenizer.json" || {
            log_error "Failed to download T3 model"
            return 1
        }
    fi

    # Create symlink for vLLM weight loading
    if [[ -f "${T3_WEIGHTS_DIR}/t3_cfg.safetensors" ]] && [[ ! -f "${T3_WEIGHTS_DIR}/model.safetensors" ]]; then
        ln -sf "t3_cfg.safetensors" "${T3_WEIGHTS_DIR}/model.safetensors"
        log_info "Created symlink: model.safetensors -> t3_cfg.safetensors"
    fi

    # Create tokenizer_config.json for vLLM to use custom EnTokenizer
    cat > "${T3_WEIGHTS_DIR}/tokenizer_config.json" << 'EOF'
{
    "tokenizer_class": "EnTokenizer",
    "auto_map": {
        "AutoTokenizer": "entokenizer.EnTokenizer"
    },
    "model_max_length": 2048,
    "padding_side": "right",
    "truncation_side": "right",
    "clean_up_tokenization_spaces": true
}
EOF
    log_info "Created tokenizer_config.json for EnTokenizer"

    # -------------------------------------------------------------------------
    # 3b: Download VoiceEncoder, S3Gen, and default conditionals
    # -------------------------------------------------------------------------
    log_info "Downloading VoiceEncoder and S3Gen assets..."

    local asset_files=(
        "ve.safetensors"
        "s3gen.safetensors"
        "conds.pt"
    )

    for file in "${asset_files[@]}"; do
        if is_real_file "${ASSETS_DIR}/${file}" 1000000; then
            log_info "  $file: Already exists"
        else
            hf_download "ResembleAI/chatterbox" "$ASSETS_DIR" "$file" || {
                log_warn "Failed to download $file"
            }
        fi
    done

    # Copy tokenizer to assets for Chatterbox Python backend
    if [[ -f "${T3_WEIGHTS_DIR}/tokenizer.json" ]]; then
        cp "${T3_WEIGHTS_DIR}/tokenizer.json" "${ASSETS_DIR}/"
        log_info "Copied tokenizer.json to assets"
    fi

    # Copy T3 weights to assets (for conditioning encoder in Python backend)
    if [[ -f "${T3_WEIGHTS_DIR}/t3_cfg.safetensors" ]]; then
        cp "${T3_WEIGHTS_DIR}/t3_cfg.safetensors" "${ASSETS_DIR}/"
        log_info "Copied t3_cfg.safetensors to assets"
    fi

    # -------------------------------------------------------------------------
    # 3c: Compile voice conditioning files for T3 vLLM backend
    # -------------------------------------------------------------------------
    log_info "Compiling voice conditioning files..."

    local voices_dir="${DEPLOY_DIR}/voices"
    local voices_output_dir="${T3_WEIGHTS_DIR}/voices"

    # Check if we have the required files
    if [[ ! -f "${ASSETS_DIR}/t3_cfg.safetensors" ]]; then
        log_warn "Missing t3_cfg.safetensors - cannot create voice conditioning"
        log_warn "T3 will require runtime conditioning (not supported with vLLM backend)"
    else
        # Find Python
        local conda_python="${CONDA_PREFIX:-/opt/conda}/bin/python"
        local system_python="python3"
        local python_cmd

        if [[ -x "$conda_python" ]]; then
            python_cmd="$conda_python"
        else
            python_cmd="$system_python"
        fi

        # Build command arguments
        local cmd_args=(
            "${SCRIPT_DIR}/create_conditioning.py"
            --assets-dir "${ASSETS_DIR}"
            --output-dir "${voices_output_dir}"
        )

        # Add voices directory if it exists and has audio files
        local has_voices=false
        if [[ -d "$voices_dir" ]]; then
            for ext in wav mp3 flac ogg m4a; do
                if compgen -G "${voices_dir}/*.${ext}" > /dev/null 2>&1; then
                    has_voices=true
                    break
                fi
            done
        fi

        if [[ "$has_voices" == "true" ]]; then
            local voice_count
            voice_count=$(find "$voices_dir" -maxdepth 1 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) | wc -l | tr -d ' ')
            log_info "Found ${voice_count} voice file(s) in ${voices_dir}"
            cmd_args+=(--voices-dir "${voices_dir}")
        else
            log_info "No custom voices in ${voices_dir} - using default voice only"
        fi

        log_info "Running create_conditioning.py..."
        if $python_cmd "${cmd_args[@]}"; then
            log_info "Voice conditioning compiled successfully"

            # Show compiled voices
            if [[ -f "${voices_output_dir}/voices.json" ]]; then
                log_info "Available voices:"
                cat "${voices_output_dir}/voices.json"
            fi
        else
            log_warn "Failed to compile voice conditioning - T3 may not work correctly"
            log_warn "You can create it manually with: python scripts/create_conditioning.py --voices-dir voices/"
        fi
    fi

    # -------------------------------------------------------------------------
    # 3d: Verify model files
    # -------------------------------------------------------------------------
    log_info "Verifying T3 vLLM model files..."
    local t3_missing=0
    local t3_required=(
        "config.json"
        "configuration_t3.py"
        "modeling_t3.py"
        "entokenizer.py"
        "tokenizer.json"
        "tokenizer_config.json"
        "t3_cfg.safetensors"
    )
    for file in "${t3_required[@]}"; do
        if [[ -f "${T3_WEIGHTS_DIR}/${file}" ]]; then
            local size
            size=$(get_file_size "${T3_WEIGHTS_DIR}/${file}")
            log_info "  $file: $(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")"
        else
            log_warn "  $file: MISSING"
            ((t3_missing++))
        fi
    done

    # Check voices
    log_info "Verifying compiled voices..."
    if [[ -d "${T3_WEIGHTS_DIR}/voices" ]]; then
        local voice_count
        voice_count=$(find "${T3_WEIGHTS_DIR}/voices" -name "*.pt" | wc -l | tr -d ' ')
        log_info "  Found ${voice_count} compiled voice(s)"
        if [[ -f "${T3_WEIGHTS_DIR}/voices/voices.json" ]]; then
            log_info "  Voice manifest: voices.json"
        fi
    else
        log_warn "  No voices directory found"
        ((t3_missing++))
    fi

    log_info "Verifying Chatterbox assets..."
    local asset_missing=0
    for file in ve.safetensors s3gen.safetensors conds.pt tokenizer.json t3_cfg.safetensors; do
        if [[ -f "${ASSETS_DIR}/${file}" ]]; then
            local size
            size=$(get_file_size "${ASSETS_DIR}/${file}")
            log_info "  $file: $(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")"
        else
            log_warn "  $file: MISSING"
            ((asset_missing++))
        fi
    done

    if [[ $t3_missing -gt 0 ]]; then
        log_error "$t3_missing T3 model files missing - vLLM backend will not work"
        return 1
    fi

    if [[ $asset_missing -gt 0 ]]; then
        log_warn "$asset_missing asset files missing - TTS may not work correctly"
    fi

    log_info "T3 vLLM model: ${T3_WEIGHTS_DIR}"
    log_info "Chatterbox assets: ${ASSETS_DIR}"
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
    [[ $STOP_STAGE -ge 3 ]] && echo "  - T3 vLLM:    ${T3_WEIGHTS_DIR}"
    [[ $STOP_STAGE -ge 3 ]] && echo "  - Assets:     ${ASSETS_DIR}"
    echo ""

    if [[ -d "${MODEL_DIR}/1/engines" ]]; then
        echo "Engines:"
        ls -lh "${MODEL_DIR}/1/engines/"*.engine 2>/dev/null || echo "  (ONNX fallback)"
        echo ""
    fi

    echo "Architecture:"
    echo "  - T3 runs as native Triton vLLM backend (high-performance token generation)"
    echo "  - Chatterbox Python backend calls T3 via gRPC, runs S3Gen locally"
    echo "  - Progressive streaming: chunks 4,8,16,32,32 with diffusion steps 1,2,5,7,10"
    echo ""
    echo "Next steps:"
    echo "  1. Start Triton: docker compose up -d triton"
    echo "  2. Start worker: docker compose up -d worker"
    echo ""
    echo "Cleanup options:"
    echo "  $0 cleanup            # Remove downloads, keep model"
    echo "  $0 cleanup --engines  # Rebuild TRT engines"
    echo "  $0 cleanup --t3       # Remove T3 weights"
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
