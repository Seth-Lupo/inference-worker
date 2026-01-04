#!/bin/bash
# =============================================================================
# Build Chatterbox TTS for Triton Inference Server
#
# Architecture (Pure PyTorch with torch.compile):
#   - T3: Triton vLLM backend (text + speaker conditioning → speech tokens)
#   - S3Gen: PyTorch with torch.compile (speech tokens → audio)
#   - VoiceEncoder: PyTorch (reference audio → speaker embedding)
#
# Usage:
#   ./build_chatterbox.sh              # Build everything
#   ./build_chatterbox.sh cleanup      # Clean up
#
# Container: nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3
# =============================================================================

set -euo pipefail

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# HuggingFace repo (weights only)
HF_REPO_MAIN="${HF_REPO_MAIN:-ResembleAI/chatterbox}"

# Paths
readonly DEPLOY_DIR="$(get_deploy_dir)"
readonly WORK_DIR="${DEPLOY_DIR}/chatterbox_build"

# T3 vLLM model location (mounted at /models/t3_weights in container)
readonly T3_WEIGHTS_DIR="${DEPLOY_DIR}/models/t3_weights"
readonly T3_MODEL_DIR="${DEPLOY_DIR}/model_repository/llm/t3"

# Chatterbox Python backend
readonly CHATTERBOX_DIR="${DEPLOY_DIR}/model_repository/tts/chatterbox"
readonly ASSETS_DIR="${DEPLOY_DIR}/model_repository/tts/chatterbox_assets"

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
            rm -rf "$ASSETS_DIR"
            log_info "Cleanup complete"
            ;;
        --weights|-w)
            log_info "Removing downloaded weights..."
            rm -rf "$T3_WEIGHTS_DIR"
            rm -rf "$ASSETS_DIR"
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
echo "Building Chatterbox TTS (Pure PyTorch)"
echo "=============================================="
echo ""
echo "T3 vLLM model:  ${T3_WEIGHTS_DIR}"
echo "Assets:         ${ASSETS_DIR}"
echo "Backend:        PyTorch + torch.compile()"
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
# Stage 2: Download S3Gen and VoiceEncoder (PyTorch)
# =============================================================================
download_assets() {
    log_step "Downloading S3Gen and VoiceEncoder assets..."

    mkdir -p "$ASSETS_DIR"

    # Required files with minimum size thresholds (bytes)
    # s3gen.safetensors: ~900MB, ve.safetensors: ~8MB
    declare -A required_files=(
        ["s3gen.safetensors"]=100000000   # 100MB min
        ["ve.safetensors"]=1000000        # 1MB min
    )
    # Optional files (nice to have but not critical)
    local optional_files=("conds.pt")

    # Check if already downloaded
    local all_exist=true
    for file in "${!required_files[@]}"; do
        if ! is_real_file "${ASSETS_DIR}/${file}" "${required_files[$file]}"; then
            all_exist=false
            break
        fi
    done

    if [[ "$all_exist" == "true" ]]; then
        log_info "Assets already downloaded"
    else
        # Clone repo if not already cloned by download_t3
        if [[ ! -d "${WORK_DIR}/chatterbox/.git" ]]; then
            log_info "Cloning ${HF_REPO_MAIN} with LFS..."
            hf_clone "$HF_REPO_MAIN" "${WORK_DIR}/chatterbox" || {
                log_error "Failed to clone from HuggingFace"
                return 1
            }
        fi

        # Copy required files
        for file in "${!required_files[@]}"; do
            if [[ -f "${WORK_DIR}/chatterbox/${file}" ]]; then
                cp "${WORK_DIR}/chatterbox/${file}" "$ASSETS_DIR/"
                log_info "  ✓ ${file}"
            else
                log_warn "  ✗ ${file} not found in repo"
            fi
        done

        # Copy optional files
        for file in "${optional_files[@]}"; do
            if [[ -f "${WORK_DIR}/chatterbox/${file}" ]]; then
                cp "${WORK_DIR}/chatterbox/${file}" "$ASSETS_DIR/"
                log_info "  ✓ ${file} (optional)"
            fi
        done
    fi

    # Copy tokenizer to assets (Python backend needs it too)
    if [[ -f "${T3_WEIGHTS_DIR}/tokenizer.json" ]] && [[ ! -f "${ASSETS_DIR}/tokenizer.json" ]]; then
        cp "${T3_WEIGHTS_DIR}/tokenizer.json" "${ASSETS_DIR}/"
        log_info "Copied tokenizer.json to assets"
    fi

    # Verify required files
    local missing=0
    for file in "${!required_files[@]}"; do
        if is_real_file "${ASSETS_DIR}/${file}" "${required_files[$file]}"; then
            log_info "  ✓ ${file}"
        else
            log_warn "  ✗ ${file} MISSING or too small"
            ((missing++))
        fi
    done

    # Check optional files (just info, not failure)
    for file in "${optional_files[@]}"; do
        if [[ -f "${ASSETS_DIR}/${file}" ]]; then
            log_info "  ✓ ${file} (optional)"
        else
            log_info "  - ${file} not present (optional)"
        fi
    done

    if [[ $missing -gt 0 ]]; then
        log_error "Missing ${missing} required asset files"
        return 1
    fi

    log_info "Assets ready at: ${ASSETS_DIR}"
}

# =============================================================================
# Stage 3: Compile Voice Conditioning
# =============================================================================
compile_voices() {
    log_step "Compiling voice conditioning..."

    local voices_dir="${DEPLOY_DIR}/voices"
    local output_dir="${T3_WEIGHTS_DIR}/voices"

    mkdir -p "$output_dir"

    # Check required assets
    if [[ ! -f "${ASSETS_DIR}/ve.safetensors" ]]; then
        log_warn "VoiceEncoder not found - skipping voice compilation"
        return 0
    fi

    # Use python3.12
    local python_cmd="python3.12"

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

    # Build command
    local cmd_args=(
        "${SCRIPT_DIR}/create_conditioning.py"
        --assets-dir "${ASSETS_DIR}"
        --output-dir "${output_dir}"
    )

    if [[ "$has_voices" == "true" ]]; then
        local count
        count=$(find "$voices_dir" -maxdepth 1 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | wc -l | tr -d ' ')
        log_info "Found ${count} voice file(s) in ${voices_dir}"
        cmd_args+=(--voices-dir "${voices_dir}")
    else
        log_info "No custom voices - using default only"
    fi

    # Run compilation
    if $python_cmd "${cmd_args[@]}" 2>/dev/null; then
        log_info "Voice conditioning compiled"
        [[ -f "${output_dir}/voices.json" ]] && cat "${output_dir}/voices.json"
    else
        log_warn "Voice compilation failed - T3 may need runtime conditioning"
    fi
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
    # Chatterbox Python backend config (if not exists)
    # -------------------------------------------------------------------------
    if [[ ! -f "${CHATTERBOX_DIR}/config.pbtxt" ]]; then
        mkdir -p "${CHATTERBOX_DIR}/1"

        cat > "${CHATTERBOX_DIR}/config.pbtxt" << 'EOF'
# Chatterbox TTS Python Backend
# Calls T3 via gRPC for token generation, runs S3Gen locally for audio synthesis
name: "chatterbox"
backend: "python"
max_batch_size: 1

input [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "speaker_id"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  },
  {
    name: "reference_audio"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
  }
]

output [
  {
    name: "audio"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "sample_rate"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

# Enable streaming for progressive audio output
model_transaction_policy {
  decoupled: true
}

parameters {
  key: "ASSETS_DIR"
  value: { string_value: "/models/tts/chatterbox_assets" }
}

parameters {
  key: "T3_MODEL_NAME"
  value: { string_value: "t3" }
}

parameters {
  key: "TRITON_GRPC_URL"
  value: { string_value: "localhost:8001" }
}
EOF
        log_info "Created: ${CHATTERBOX_DIR}/config.pbtxt"
    else
        log_info "Chatterbox config.pbtxt already exists"
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
    echo "S3Gen/VE Assets:"
    echo "  Location: ${ASSETS_DIR}"
    ls -lh "${ASSETS_DIR}"/*.safetensors 2>/dev/null | head -3
    echo ""
    echo "Backend: PyTorch + torch.compile()"
    echo ""
    echo "Voices:"
    if [[ -f "${T3_WEIGHTS_DIR}/voices/voices.json" ]]; then
        cat "${T3_WEIGHTS_DIR}/voices/voices.json"
    else
        echo "  (none compiled)"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Start Triton: docker compose up -d triton"
    echo "  2. Start worker: docker compose up -d worker"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    download_t3
    download_assets
    compile_voices
    create_triton_configs
    show_summary
}

main
