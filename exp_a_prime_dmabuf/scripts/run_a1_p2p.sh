#!/usr/bin/env bash
# run_a1_p2p.sh — Exp A', Task A'1: cuMem FD P2P bandwidth test
#
# Tests explicit cuMem VMM FD export/import across isolated containers.
#
# Usage:
#   bash scripts/run_a1_p2p.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"

echo "=============================================="
echo "Exp A' — Task A'1: cuMem FD P2P Bandwidth"
echo "=============================================="

# Build
echo "--- Building Docker image ---"
cd "$PROJECT_ROOT"
docker build -t tth37/xtrans-experiments:expa-prime-dmabuf \
    -f exp_a_prime_dmabuf/Dockerfile .
cd "$EXP_DIR"

mkdir -p results

# Run cuMem FD test
echo ""
echo "=== Running cuMem FD P2P test ==="
docker compose -f compose.dmabuf.yml down --remove-orphans 2>/dev/null || true
IPC_PATH=cumem_fd \
    docker compose -f compose.dmabuf.yml up \
    --abort-on-container-exit --force-recreate 2>&1 | \
    tee results/cumem_fd_compose.log
docker compose -f compose.dmabuf.yml down --remove-orphans 2>/dev/null || true

echo ""
echo "=============================================="
if [ -f results/cumem_fd_bw.json ]; then
    echo "Results: results/cumem_fd_bw.json"
    python3 -c "
import json
with open('results/cumem_fd_bw.json') as f:
    d = json.load(f)
print('IPC path:', d.get('ipc_path'))
for size, r in d.get('results', {}).items():
    print(f'  {size}: {r[\"bandwidth_gbps\"]:.1f} GB/s ({r[\"avg_latency_us\"]:.1f} us)')
" 2>/dev/null || echo "(install python3 for result summary)"
else
    echo "ERROR: results/cumem_fd_bw.json not found"
fi
echo "=============================================="
