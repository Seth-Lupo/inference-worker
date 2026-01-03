#!/bin/bash
# =============================================================================
# EC2 Instance Setup Script
# Run this once on a fresh EC2 instance (Amazon Linux 2 / Ubuntu with NVIDIA GPU)
# =============================================================================
set -e

echo "=============================================="
echo "Voice Agent Inference Worker - EC2 Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }


# -----------------------------------------------------------------------------
# 2. Verify GPU Access
# -----------------------------------------------------------------------------
log_info "Verifying GPU access..."
if nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo ""
    log_info "GPU detected and accessible"
else
    log_error "GPU not detected! Ensure NVIDIA drivers are installed."
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Install Docker Compose (as CLI plugin)
# -----------------------------------------------------------------------------
log_info "Installing Docker Compose..."
if ! docker compose version &> /dev/null; then
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    log_info "Docker Compose installed: $(docker compose version)"
else
    log_info "Docker Compose already installed: $(docker compose version)"
fi

# -----------------------------------------------------------------------------
# 3. Install Git LFS (required for downloading models from HuggingFace)
# -----------------------------------------------------------------------------
log_info "Installing git-lfs..."
if ! command -v git-lfs &> /dev/null; then
    sudo yum install -y git-lfs 2>/dev/null || sudo apt-get install -y git-lfs
    git lfs install
    log_info "git-lfs installed"
else
    log_info "git-lfs already installed"
fi


# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}EC2 Setup Complete!${NC}"
echo "=============================================="

