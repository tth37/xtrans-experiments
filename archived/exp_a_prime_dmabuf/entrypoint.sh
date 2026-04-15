#!/bin/bash
set -euo pipefail

echo "[entrypoint] Role: ${ROLE:-unset}, IPC: ${IPC_PATH:-unset}, GPU: ${GPU_ID:-0}"

# Exporter cleans up stale socket before starting
if [ "${ROLE:-}" = "exporter" ]; then
    rm -f "${UDS_SOCKET:-/shared_uds/dmabuf.sock}"
fi

exec "$@"
