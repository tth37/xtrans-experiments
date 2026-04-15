#!/usr/bin/env bash
# run_a2_compare.sh — Exp A', Task A'2: Three-path IPC comparison
#
# Compares: cuMem FD, DMA-BUF FD (if supported), legacy cudaIPC
#
# Usage:
#   bash scripts/run_a2_compare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"

echo "=============================================="
echo "Exp A' — Task A'2: IPC Path Comparison"
echo "=============================================="

# Build
echo "--- Building Docker image ---"
cd "$PROJECT_ROOT"
docker build -t tth37/xtrans-experiments:expa-prime-dmabuf \
    -f exp_a_prime_dmabuf/Dockerfile .
cd "$EXP_DIR"

mkdir -p results
FAILED=()

# 1. cuMem FD
echo ""
echo "=== Path 1: cuMem FD ==="
docker compose -f compose.dmabuf.yml down --remove-orphans 2>/dev/null || true
if IPC_PATH=cumem_fd docker compose -f compose.dmabuf.yml up \
    --abort-on-container-exit --force-recreate 2>&1 | tee results/cumem_fd_compose.log; then
    echo "  cuMem FD: completed"
else
    FAILED+=(cumem_fd)
fi
docker compose -f compose.dmabuf.yml down --remove-orphans 2>/dev/null || true

# 2. Legacy cudaIPC
echo ""
echo "=== Path 2: Legacy cudaIPC ==="
docker compose -f compose.cudaipc.yml down --remove-orphans 2>/dev/null || true
if docker compose -f compose.cudaipc.yml up \
    --abort-on-container-exit --force-recreate 2>&1 | tee results/cudaipc_compose.log; then
    echo "  cudaIPC: completed"
else
    FAILED+=(cudaipc)
fi
docker compose -f compose.cudaipc.yml down --remove-orphans 2>/dev/null || true

# Compare
echo ""
echo "=== Results Comparison ==="
RESULT_FILES=()
for f in results/cumem_fd_bw.json results/cudaipc_bw.json; do
    [ -f "$f" ] && RESULT_FILES+=("$f")
done

if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    python3 "$SCRIPT_DIR/compare_results.py" "${RESULT_FILES[@]}"
else
    echo "No result files found."
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "WARNING: Failed paths: ${FAILED[*]}"
fi
echo "=============================================="
