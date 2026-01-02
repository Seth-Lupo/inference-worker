#!/bin/bash
# =============================================================================
# Start Voice Agent Services
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

cd "${DEPLOY_DIR}"

echo "Starting Voice Agent services..."

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if Triton image exists
if ! docker images | grep -q "tritonserver.*24.12-trtllm"; then
    echo "Pulling Triton image..."
    docker pull nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3
fi

# Start services
docker-compose up -d

echo ""
echo "Services starting..."
echo ""
echo "Monitor logs:"
echo "  docker-compose logs -f"
echo ""
echo "Check status:"
echo "  docker-compose ps"
echo ""
echo "Endpoints:"
echo "  Voice Agent: ws://localhost:80"
echo "  Triton HTTP: http://localhost:8000"
echo "  Triton gRPC: localhost:8001"
echo "  Metrics:     http://localhost:8002/metrics"
