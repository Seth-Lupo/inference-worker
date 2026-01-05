#!/bin/bash
# =============================================================================
# Build Chatterbox TTS for Triton Inference Server
#
# Architecture (Native ONNX + vLLM):
#   - T3: Triton vLLM backend (text + speaker conditioning → speech tokens)
#   - speech_encoder: Native ONNX (reference audio → conditioning)
#   - conditional_decoder: Native ONNX (speech tokens → audio waveform)
#
# ONNX models from: ResembleAI/chatterbox-turbo-ONNX
#
# Usage:
#   ./build_chatterbox.sh              # Build everything
#   ./build_chatterbox.sh cleanup      # Clean up
# =============================================================================

set -euo pipefail

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# HuggingFace repos
HF_REPO_MAIN="${HF_REPO_MAIN:-ResembleAI/chatterbox}"
HF_REPO_ONNX="${HF_REPO_ONNX:-ResembleAI/chatterbox-turbo-ONNX}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/chatterbox_build"

# T3 vLLM model location (mounted at /models/t3_weights in container)
readonly T3_WEIGHTS_DIR="${DEPLOY_DIR}/models/t3_weights"
readonly T3_MODEL_DIR="${DEPLOY_DIR}/model_repository/llm/t3"

# ONNX models for native Triton backend
readonly ONNX_DIR="${DEPLOY_DIR}/models/chatterbox_onnx"
readonly SPEECH_ENCODER_DIR="${DEPLOY_DIR}/model_repository/tts/speech_encoder"
readonly CONDITIONAL_DECODER_DIR="${DEPLOY_DIR}/model_repository/tts/conditional_decoder"

# ONNX precision (fp16 recommended for GPU)
ONNX_DTYPE="${ONNX_DTYPE:-fp16}"

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
            rm -rf "$T3_WEIGHTS_DIR"
            rm -rf "$ONNX_DIR"
            log_info "Cleanup complete"
            ;;
        --weights|-w)
            log_info "Removing downloaded weights..."
            rm -rf "$T3_WEIGHTS_DIR"
            rm -rf "$ONNX_DIR"
            log_info "Weights removed"
            ;;
        *)
            log_info "Removing work directory only..."
            rm -rf "$WORK_DIR"
            log_info "Cleanup complete"
            echo ""
            echo "Other cleanup options:"
            echo "  $0 cleanup --weights  Remove downloaded weights"
            echo "  $0 cleanup --all      Remove everything"
            ;;
    esac
    exit 0
fi

echo "=============================================="
echo "Building Chatterbox TTS (Native ONNX + vLLM)"
echo "=============================================="
echo ""
echo "T3 vLLM model:      ${T3_WEIGHTS_DIR}"
echo "ONNX models:        ${ONNX_DIR}"
echo "ONNX precision:     ${ONNX_DTYPE}"
echo "Backend:            vLLM (T3) + ONNX Runtime (encoder/decoder)"
echo ""

mkdir -p "$WORK_DIR"

# =============================================================================
# Stage 1: Download T3 Model for vLLM Backend
# =============================================================================
download_t3() {
    log_step "Downloading T3 model for vLLM backend..."

    # Check if already downloaded
    if is_real_file "${T3_WEIGHTS_DIR}/t3_cfg.safetensors" 100000000; then
        log_info "T3 weights already downloaded"
    else
        # Clone main chatterbox repo (has weights + code)
        log_info "Cloning ${HF_REPO_MAIN} with LFS..."
        hf_clone "$HF_REPO_MAIN" "${WORK_DIR}/chatterbox" || {
            log_error "Failed to clone from HuggingFace"
            return 1
        }

        # Copy T3 files to weights directory
        mkdir -p "$T3_WEIGHTS_DIR"
        cp "${WORK_DIR}/chatterbox/t3_cfg.safetensors" "$T3_WEIGHTS_DIR/" 2>/dev/null || true
        cp "${WORK_DIR}/chatterbox/tokenizer.json" "$T3_WEIGHTS_DIR/" 2>/dev/null || true

        # Copy model code files if present
        for file in config.json modeling_t3.py configuration_t3.py entokenizer.py; do
            cp "${WORK_DIR}/chatterbox/${file}" "$T3_WEIGHTS_DIR/" 2>/dev/null || true
        done
    fi

    # Create model.safetensors symlink for vLLM
    if [[ -f "${T3_WEIGHTS_DIR}/t3_cfg.safetensors" ]] && [[ ! -e "${T3_WEIGHTS_DIR}/model.safetensors" ]]; then
        ln -sf "t3_cfg.safetensors" "${T3_WEIGHTS_DIR}/model.safetensors"
        log_info "Created symlink: model.safetensors -> t3_cfg.safetensors"
    fi

    # If model code files missing, try chatterbox-turbo repo
    local code_files=("config.json" "modeling_t3.py" "configuration_t3.py" "entokenizer.py")
    local missing_code=false

    for file in "${code_files[@]}"; do
        if [[ ! -f "${T3_WEIGHTS_DIR}/${file}" ]]; then
            missing_code=true
            break
        fi
    done

    if [[ "$missing_code" == "true" ]]; then
        log_info "Downloading T3 model code from chatterbox-turbo..."
        hf_clone "ResembleAI/chatterbox-turbo" "${WORK_DIR}/chatterbox-turbo" 2>/dev/null || true

        for file in "${code_files[@]}"; do
            if [[ ! -f "${T3_WEIGHTS_DIR}/${file}" ]] && [[ -f "${WORK_DIR}/chatterbox-turbo/${file}" ]]; then
                cp "${WORK_DIR}/chatterbox-turbo/${file}" "$T3_WEIGHTS_DIR/"
            fi
        done
    fi

    # Create tokenizer_config.json for vLLM
    if [[ ! -f "${T3_WEIGHTS_DIR}/tokenizer_config.json" ]]; then
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
        log_info "Created tokenizer_config.json"
    fi

    # Verify required files
    log_info "Verifying T3 model files..."
    local required=("t3_cfg.safetensors" "tokenizer.json" "config.json" "modeling_t3.py" "configuration_t3.py" "entokenizer.py")
    local missing=0
    for file in "${required[@]}"; do
        if [[ -f "${T3_WEIGHTS_DIR}/${file}" ]]; then
            log_info "  ✓ ${file}"
        else
            log_warn "  ✗ ${file} MISSING"
            ((missing++))
        fi
    done

    if [[ $missing -gt 0 ]]; then
        log_error "Missing ${missing} required T3 files"
        log_error "T3 model code files may need to be copied manually"
        return 1
    fi

    log_info "T3 model ready at: ${T3_WEIGHTS_DIR}"
}

# =============================================================================
# Stage 2: Download ONNX Models from chatterbox-turbo-ONNX
# =============================================================================
download_onnx() {
    log_step "Downloading Chatterbox ONNX models..."

    mkdir -p "$ONNX_DIR"

    # ONNX models to download (speech_encoder and conditional_decoder)
    # fp16 is recommended for GPU, but fp32 and quantized versions available
    local dtype_suffix=""
    case "$ONNX_DTYPE" in
        fp32) dtype_suffix="" ;;
        fp16) dtype_suffix="_fp16" ;;
        q8)   dtype_suffix="_quantized" ;;
        q4)   dtype_suffix="_q4" ;;
        q4f16) dtype_suffix="_q4f16" ;;
        *) log_error "Unknown ONNX dtype: $ONNX_DTYPE"; return 1 ;;
    esac

    local models=("speech_encoder" "conditional_decoder")
    local all_exist=true

    # Check if models already downloaded
    for model in "${models[@]}"; do
        local onnx_file="${ONNX_DIR}/${model}${dtype_suffix}.onnx"
        if ! is_real_file "$onnx_file" 1000000; then
            all_exist=false
            break
        fi
    done

    if [[ "$all_exist" == "true" ]]; then
        log_info "ONNX models already downloaded"
    else
        # Clone the ONNX repo (shallow, then pull specific files)
        log_info "Cloning ${HF_REPO_ONNX} (shallow)..."
        hf_clone_shallow "$HF_REPO_ONNX" "${WORK_DIR}/chatterbox-onnx" || {
            log_error "Failed to clone ONNX repo"
            return 1
        }

        # Pull only the ONNX files we need
        local lfs_patterns=()
        for model in "${models[@]}"; do
            lfs_patterns+=("onnx/${model}${dtype_suffix}.onnx")
            lfs_patterns+=("onnx/${model}${dtype_suffix}.onnx_data")
        done

        log_info "Pulling ONNX files: ${lfs_patterns[*]}"
        (
            cd "${WORK_DIR}/chatterbox-onnx"
            for pattern in "${lfs_patterns[@]}"; do
                git lfs pull --include "$pattern" 2>/dev/null || true
            done
        )

        # Copy ONNX files to output directory
        for model in "${models[@]}"; do
            local src="${WORK_DIR}/chatterbox-onnx/onnx/${model}${dtype_suffix}.onnx"
            local src_data="${WORK_DIR}/chatterbox-onnx/onnx/${model}${dtype_suffix}.onnx_data"
            local dst="${ONNX_DIR}/${model}${dtype_suffix}.onnx"
            local dst_data="${ONNX_DIR}/${model}${dtype_suffix}.onnx_data"

            if [[ -f "$src" ]]; then
                cp "$src" "$dst"
                log_info "  ✓ ${model}${dtype_suffix}.onnx"
            else
                log_warn "  ✗ ${model}${dtype_suffix}.onnx not found"
            fi

            # Copy external data file if exists
            if [[ -f "$src_data" ]]; then
                cp "$src_data" "$dst_data"
                log_info "  ✓ ${model}${dtype_suffix}.onnx_data"
            fi
        done
    fi

    # Verify downloaded files
    log_info "Verifying ONNX models..."
    local missing=0
    for model in "${models[@]}"; do
        local onnx_file="${ONNX_DIR}/${model}${dtype_suffix}.onnx"
        if is_real_file "$onnx_file" 1000000; then
            local size
            size=$(get_file_size "$onnx_file")
            log_info "  ✓ ${model}${dtype_suffix}.onnx ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
        else
            log_warn "  ✗ ${model}${dtype_suffix}.onnx MISSING or too small"
            ((missing++))
        fi
    done

    if [[ $missing -gt 0 ]]; then
        log_error "Missing ${missing} ONNX model files"
        return 1
    fi

    # Create symlinks to model.onnx in Triton model directories
    log_info "Creating Triton model symlinks..."

    mkdir -p "${SPEECH_ENCODER_DIR}/1"
    mkdir -p "${CONDITIONAL_DECODER_DIR}/1"

    # Symlink for speech_encoder
    local se_src="${ONNX_DIR}/speech_encoder${dtype_suffix}.onnx"
    local se_dst="${SPEECH_ENCODER_DIR}/1/model.onnx"
    if [[ -f "$se_src" ]]; then
        ln -sf "$se_src" "$se_dst"
        log_info "  ✓ speech_encoder/1/model.onnx -> ${se_src}"
    fi
    # Also link external data if exists
    if [[ -f "${se_src}_data" ]]; then
        ln -sf "${se_src}_data" "${SPEECH_ENCODER_DIR}/1/model.onnx_data"
    fi

    # Symlink for conditional_decoder
    local cd_src="${ONNX_DIR}/conditional_decoder${dtype_suffix}.onnx"
    local cd_dst="${CONDITIONAL_DECODER_DIR}/1/model.onnx"
    if [[ -f "$cd_src" ]]; then
        ln -sf "$cd_src" "$cd_dst"
        log_info "  ✓ conditional_decoder/1/model.onnx -> ${cd_src}"
    fi
    # Also link external data if exists
    if [[ -f "${cd_src}_data" ]]; then
        ln -sf "${cd_src}_data" "${CONDITIONAL_DECODER_DIR}/1/model.onnx_data"
    fi

    log_info "ONNX models ready at: ${ONNX_DIR}"
}

# =============================================================================
# Stage 3: Voice Conditioning (Optional - for pre-computed voices)
# =============================================================================
compile_voices() {
    log_step "Checking voice conditioning..."

    local voices_dir="${DEPLOY_DIR}/voices"
    local output_dir="${T3_WEIGHTS_DIR}/voices"

    mkdir -p "$output_dir"

    # Check for voice files
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
        local count
        count=$(find "$voices_dir" -maxdepth 1 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | wc -l | tr -d ' ')
        log_info "Found ${count} voice file(s) in ${voices_dir}"
        log_info "Voice conditioning will be computed at runtime via speech_encoder ONNX model"
    else
        log_info "No custom voices - will use reference audio at runtime"
    fi

    log_info "Note: Voice conditioning is now done via Triton speech_encoder model"
}

# =============================================================================
# Stage 4: Create Triton Configs
# =============================================================================
create_triton_configs() {
    log_step "Creating Triton model configs..."

    # -------------------------------------------------------------------------
    # T3 vLLM model config
    # -------------------------------------------------------------------------
    mkdir -p "${T3_MODEL_DIR}/1"

    # model.json for vLLM backend
    cat > "${T3_MODEL_DIR}/1/model.json" << EOF
{
    "model": "/models/t3_weights",
    "disable_log_requests": true,
    "gpu_memory_utilization": 0.3,
    "max_model_len": 2048,
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "trust_remote_code": true,
    "enforce_eager": true
}
EOF
    log_info "Created: ${T3_MODEL_DIR}/1/model.json"

    # config.pbtxt for Triton
    cat > "${T3_MODEL_DIR}/config.pbtxt" << 'EOF'
# T3 Speech Token Generator via vLLM Backend
name: "t3"
backend: "vllm"

instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]

parameters: {
  key: "REPORT_CUSTOM_METRICS"
  value: { string_value: "true" }
}
EOF
    log_info "Created: ${T3_MODEL_DIR}/config.pbtxt"

    # -------------------------------------------------------------------------
    # Verify ONNX model configs exist
    # -------------------------------------------------------------------------
    if [[ -f "${SPEECH_ENCODER_DIR}/config.pbtxt" ]]; then
        log_info "speech_encoder config.pbtxt exists"
    else
        log_warn "speech_encoder config.pbtxt missing - create it manually"
    fi

    if [[ -f "${CONDITIONAL_DECODER_DIR}/config.pbtxt" ]]; then
        log_info "conditional_decoder config.pbtxt exists"
    else
        log_warn "conditional_decoder config.pbtxt missing - create it manually"
    fi
}

# =============================================================================
# Summary
# =============================================================================
show_summary() {
    echo ""
    echo "=============================================="
    log_info "Chatterbox Build Complete!"
    echo "=============================================="
    echo ""
    echo "T3 vLLM Model:"
    echo "  Location: ${T3_WEIGHTS_DIR}"
    echo "  Triton:   ${T3_MODEL_DIR}"
    ls -lh "${T3_WEIGHTS_DIR}"/*.safetensors 2>/dev/null | head -3
    echo ""
    echo "ONNX Models (${ONNX_DTYPE}):"
    echo "  Location: ${ONNX_DIR}"
    ls -lh "${ONNX_DIR}"/*.onnx 2>/dev/null | head -5
    echo ""
    echo "Triton Models:"
    echo "  - t3 (vLLM): Speech token generation"
    echo "  - speech_encoder (ONNX): Reference audio → conditioning"
    echo "  - conditional_decoder (ONNX): Speech tokens → audio"
    echo ""
    echo "Next steps:"
    echo "  1. Build Docker: docker compose build triton"
    echo "  2. Start Triton: docker compose up -d triton"
    echo "  3. Start worker: docker compose up -d worker"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    download_t3
    download_onnx
    compile_voices
    create_triton_configs
    show_summary
}

main
