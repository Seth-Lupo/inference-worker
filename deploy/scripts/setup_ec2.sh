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
# 1. System Updates
# -----------------------------------------------------------------------------
log_info "Updating system packages..."
sudo yum update -y 2>/dev/null || sudo apt-get update -y

# -----------------------------------------------------------------------------
# 2. Install Docker
# -----------------------------------------------------------------------------
log_info "Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Amazon Linux 2
    if [ -f /etc/amazon-linux-release ]; then
        sudo amazon-linux-extras install docker -y
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ec2-user
    # Ubuntu
    elif [ -f /etc/lsb-release ]; then
        sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io
        sudo usermod -aG docker ubuntu
    fi
    log_info "Docker installed successfully"
else
    log_info "Docker already installed"
fi

# -----------------------------------------------------------------------------
# 3. Install Docker Compose
# -----------------------------------------------------------------------------
log_info "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION="v2.24.0"
    sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    log_info "Docker Compose installed: $(docker-compose --version)"
else
    log_info "Docker Compose already installed"
fi

# -----------------------------------------------------------------------------
# 4. Install NVIDIA Container Toolkit
# -----------------------------------------------------------------------------
log_info "Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-container-toolkit &> /dev/null; then
    # Add NVIDIA repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - 2>/dev/null || \
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
        sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

    # Install toolkit
    sudo yum install -y nvidia-container-toolkit 2>/dev/null || \
    sudo apt-get install -y nvidia-container-toolkit

    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    log_info "NVIDIA Container Toolkit installed"
else
    log_info "NVIDIA Container Toolkit already installed"
fi

# -----------------------------------------------------------------------------
# 5. Verify GPU Access
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

# Test Docker GPU access
log_info "Testing Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 && \
    log_info "Docker can access GPU" || \
    log_error "Docker cannot access GPU"

# -----------------------------------------------------------------------------
# 6. Create Directory Structure
# -----------------------------------------------------------------------------
DEPLOY_DIR="/home/ec2-user/inference-worker"
log_info "Creating directory structure at ${DEPLOY_DIR}..."

mkdir -p ${DEPLOY_DIR}/deploy/{model_repository,engine_cache,logs,scripts,model_sources}
mkdir -p ${DEPLOY_DIR}/deploy/logs/{triton,worker}

# -----------------------------------------------------------------------------
# 7. Configure NGC Authentication (Optional)
# -----------------------------------------------------------------------------
log_info "Setting up NGC authentication..."
echo ""
echo "To pull images from NGC, you need an API key."
echo "Get one from: https://ngc.nvidia.com/setup/api-key"
echo ""
read -p "Enter NGC API Key (or press Enter to skip): " NGC_API_KEY

if [ -n "$NGC_API_KEY" ]; then
    docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
    log_info "NGC authentication configured"
else
    log_warn "NGC authentication skipped. You may need to authenticate later."
fi

# -----------------------------------------------------------------------------
# 8. Pull Container Images
# -----------------------------------------------------------------------------
log_info "Pulling Triton Inference Server image..."
docker pull nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

# -----------------------------------------------------------------------------
# 9. Create Environment File
# -----------------------------------------------------------------------------
log_info "Creating environment file..."
cat > ${DEPLOY_DIR}/deploy/.env << 'EOF'
# HuggingFace Token (for downloading models)
HF_TOKEN=

# NGC API Key (for pulling containers)
NGC_API_KEY=

# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Triton Configuration
TRITON_LOG_LEVEL=INFO

# Worker Configuration
LOG_LEVEL=INFO
EOF

log_info "Environment file created at ${DEPLOY_DIR}/deploy/.env"
log_warn "Edit the .env file to add your HuggingFace token"

# -----------------------------------------------------------------------------
# 10. Set Permissions
# -----------------------------------------------------------------------------
log_info "Setting permissions..."
sudo chown -R $(whoami):$(whoami) ${DEPLOY_DIR}
chmod +x ${DEPLOY_DIR}/deploy/scripts/*.sh 2>/dev/null || true

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}EC2 Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Edit ${DEPLOY_DIR}/deploy/.env with your tokens"
echo "  2. Copy your code to ${DEPLOY_DIR}"
echo "  3. Download models: ./scripts/download_models.sh"
echo "  4. Build TensorRT engines: ./scripts/build_engines.sh"
echo "  5. Start services: docker-compose up -d"
echo ""
echo "Note: You may need to log out and back in for Docker group changes."
