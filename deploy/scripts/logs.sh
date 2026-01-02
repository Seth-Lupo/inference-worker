#!/bin/bash
# =============================================================================
# View Voice Agent Logs
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

cd "${DEPLOY_DIR}"

# Default: follow all logs
SERVICE="${1:-}"

if [ -n "$SERVICE" ]; then
    docker-compose logs -f "$SERVICE"
else
    docker-compose logs -f
fi
