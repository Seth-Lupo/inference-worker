#!/bin/bash
# =============================================================================
# Stop Voice Agent Services
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

cd "${DEPLOY_DIR}"

echo "Stopping Voice Agent services..."
docker-compose down

echo "Services stopped."
