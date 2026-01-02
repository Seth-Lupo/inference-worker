#!/bin/bash
# =============================================================================
# Deploy Voice Agent to EC2
# Run from local machine to deploy code to EC2 instance
# =============================================================================
set -e

# Configuration
EC2_HOST="${EC2_HOST:-54.210.151.130}"
EC2_USER="${EC2_USER:-ec2-user}"
EC2_KEY="${EC2_KEY:-./pair.pem}"
REMOTE_DIR="/home/ec2-user/inference-worker"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=============================================="
echo "Deploying Voice Agent to EC2"
echo "=============================================="
echo "Host: ${EC2_USER}@${EC2_HOST}"
echo "Remote: ${REMOTE_DIR}"
echo ""

# Check SSH key
if [ ! -f "$EC2_KEY" ]; then
    log_warn "SSH key not found at ${EC2_KEY}"
    log_warn "Set EC2_KEY environment variable or place key at ./pair.pem"
    exit 1
fi

chmod 600 "$EC2_KEY"

# Sync code to EC2 (excluding large files)
log_info "Syncing code to EC2..."
rsync -avz --progress \
    -e "ssh -i ${EC2_KEY} -o StrictHostKeyChecking=no" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'engine_cache/*' \
    --exclude 'model_sources/*' \
    --exclude 'logs/*' \
    --exclude '*.engine' \
    --exclude '*.plan' \
    --exclude '*.onnx' \
    "${PROJECT_DIR}/" \
    "${EC2_USER}@${EC2_HOST}:${REMOTE_DIR}/"

log_info "Code synced successfully"

# Make scripts executable
log_info "Setting permissions..."
ssh -i "$EC2_KEY" "${EC2_USER}@${EC2_HOST}" "chmod +x ${REMOTE_DIR}/deploy/scripts/*.sh"

# Restart services
log_info "Restarting services..."
ssh -i "$EC2_KEY" "${EC2_USER}@${EC2_HOST}" "cd ${REMOTE_DIR}/deploy && docker-compose up -d --build worker"

echo ""
echo "=============================================="
echo -e "${GREEN}Deployment Complete${NC}"
echo "=============================================="
echo ""
echo "SSH to instance:"
echo "  ssh -i ${EC2_KEY} ${EC2_USER}@${EC2_HOST}"
echo ""
echo "View logs:"
echo "  ssh -i ${EC2_KEY} ${EC2_USER}@${EC2_HOST} 'cd ${REMOTE_DIR}/deploy && docker-compose logs -f'"
