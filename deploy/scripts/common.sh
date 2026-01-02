#!/bin/bash
# =============================================================================
# Shared Utilities for Build Scripts
# Source this file: source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# =============================================================================

# Strict mode
set -euo pipefail

# =============================================================================
# Colors and Logging
# =============================================================================
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $*"; }

# =============================================================================
# Path Resolution
# =============================================================================
get_script_dir() {
    cd "$(dirname "${BASH_SOURCE[1]}")" && pwd
}

get_deploy_dir() {
    dirname "$(get_script_dir)"
}

# =============================================================================
# File Utilities
# =============================================================================

# Get file size (cross-platform: works on macOS and Linux)
get_file_size() {
    local file="$1"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        stat -f%z "$file" 2>/dev/null || echo "0"
    else
        stat -c%s "$file" 2>/dev/null || echo "0"
    fi
}

# Check if file exists and is larger than minimum size (not an LFS pointer)
# Usage: is_real_file "path/to/file" [min_size_bytes]
is_real_file() {
    local file="$1"
    local min_size="${2:-1000000}"  # Default 1MB

    [[ ! -f "$file" ]] && return 1

    local size
    size=$(get_file_size "$file")
    [[ "$size" -gt "$min_size" ]]
}

# Check if directory contains real model weights (not LFS pointers)
# Usage: has_real_weights "model_dir" "*.safetensors"
has_real_weights() {
    local dir="$1"
    local pattern="${2:-*.safetensors}"

    [[ ! -d "$dir" ]] && return 1

    local f
    for f in "$dir"/$pattern; do
        [[ -f "$f" ]] && is_real_file "$f" && return 0
    done
    return 1
}

# =============================================================================
# Git LFS Utilities
# =============================================================================

# Ensure git-lfs is installed
ensure_git_lfs() {
    if command -v git-lfs &>/dev/null; then
        git lfs install --skip-smudge 2>/dev/null || true
        return 0
    fi

    log_info "Installing git-lfs..."
    if command -v yum &>/dev/null; then
        sudo yum install -y git-lfs
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y git-lfs
    elif command -v brew &>/dev/null; then
        brew install git-lfs
    else
        log_error "Cannot install git-lfs automatically. Please install manually."
        return 1
    fi
    git lfs install
}

# Clone a HuggingFace repo with LFS support
# Usage: hf_clone "org/model" "output_dir" [hf_token]
hf_clone() {
    local repo="$1"
    local output_dir="$2"
    local token="${3:-${HF_TOKEN:-}}"

    ensure_git_lfs || return 1

    local git_url
    if [[ -n "$token" ]]; then
        git_url="https://USER:${token}@huggingface.co/${repo}"
    else
        git_url="https://huggingface.co/${repo}"
    fi

    log_info "Cloning ${repo}..."
    GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 "$git_url" "$output_dir"
}

# Pull LFS files in a directory
lfs_pull() {
    local dir="$1"
    (cd "$dir" && git lfs pull)
}

# =============================================================================
# Docker Utilities
# =============================================================================

# Check if a Docker image exists locally
docker_image_exists() {
    local image="$1"
    docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${image}$"
}

# Pull Docker image if not present
ensure_docker_image() {
    local image="$1"
    if docker_image_exists "$image"; then
        log_info "Docker image already exists: $image"
    else
        log_info "Pulling Docker image: $image"
        docker pull "$image"
    fi
}

# Remove Docker container if it exists
remove_container() {
    local name="$1"
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        log_info "Removing container: $name"
        docker rm -f "$name" 2>/dev/null || true
    fi
}

# =============================================================================
# Environment
# =============================================================================

# Load .env file if it exists
load_env() {
    local env_file="${1:-$(get_deploy_dir)/.env}"
    if [[ -f "$env_file" ]]; then
        # shellcheck source=/dev/null
        source "$env_file"
    fi
}

# =============================================================================
# Cleanup Handler
# =============================================================================

# Generic cleanup command handler
# Usage: handle_cleanup "container_name" "work_dir" "$@"
handle_cleanup() {
    local container_name="$1"
    local work_dir="$2"
    local cmd="${3:-}"
    local opt="${4:-}"

    [[ "$cmd" != "cleanup" && "$cmd" != "clean" ]] && return 1

    echo "=============================================="
    echo "Cleaning up build artifacts"
    echo "=============================================="

    remove_container "$container_name"

    case "$opt" in
        --all|-a)
            log_warn "Removing ALL build artifacts at ${work_dir}..."
            rm -rf "$work_dir"
            log_info "Build directory removed"
            ;;
        --image|-i)
            log_info "Note: Use 'docker rmi <image>' to remove specific images"
            ;;
        --keep-engine|-k|"")
            log_info "Removing downloaded models (keeping engines)..."
            # Script-specific cleanup should be done by caller
            ;;
        *)
            echo "Usage: cleanup [--keep-engine|-k | --all|-a | --image|-i]"
            ;;
    esac

    exit 0
}

# =============================================================================
# Validation
# =============================================================================

# Check required commands exist
require_commands() {
    local missing=()
    for cmd in "$@"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required commands: ${missing[*]}"
        return 1
    fi
}

# Check GPU is available
require_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        log_error "nvidia-smi not found. GPU drivers not installed?"
        return 1
    fi

    if ! nvidia-smi &>/dev/null; then
        log_error "nvidia-smi failed. GPU not available?"
        return 1
    fi

    log_info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
}
