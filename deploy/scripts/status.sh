#!/bin/bash
# =============================================================================
# Check Voice Agent Status
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

cd "${DEPLOY_DIR}"

echo "=============================================="
echo "Voice Agent Status"
echo "=============================================="
echo ""

# Container status
echo "Containers:"
docker-compose ps
echo ""

# GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# Triton health
echo "Triton Health:"
curl -s http://localhost:8000/v2/health/ready && echo " - Ready" || echo " - Not ready"
curl -s http://localhost:8000/v2/health/live && echo " - Live" || echo " - Not live"
echo ""

# Loaded models
echo "Loaded Models:"
curl -s http://localhost:8000/v2/models | python3 -m json.tool 2>/dev/null || echo "Could not fetch models"
echo ""

# Worker health
echo "Worker Health:"
if curl -s --max-time 2 http://localhost:80 > /dev/null 2>&1; then
    echo "  WebSocket server: Running"
else
    echo "  WebSocket server: Not responding"
fi
