#!/usr/bin/env bash
# run_a1_trace.sh — Exp A, Task A1: Syscall tracing to map NCCL gates
#
# Runs the benchmark under strace in three configurations (baseline,
# isolated, shim) and analyzes the traces to identify NCCL's container
# detection gates.
#
# Usage:
#   bash scripts/run_a1_trace.sh          # Full benchmark + trace
#   bash scripts/run_a1_trace.sh --smoke  # Quick smoke test + trace
#
# Output:
#   results/traces/<config>_rank<N>.strace  — Raw strace files
#   results/traces/analysis.json            — Structured analysis

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"

# Parse arguments
SMOKE_TEST="false"
CONFIGS=("baseline" "isolated" "shim")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE_TEST="true"; shift ;;
        --configs) CONFIGS=(); IFS=',' read -ra CONFIGS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "Exp A — Task A1: Syscall Tracing"
echo "=============================================="
echo "Configs: ${CONFIGS[*]}"
echo "Smoke test: $SMOKE_TEST"
echo ""

# Step 1: Build the shim
echo "--- Building shim ---"
make -C "$PROJECT_ROOT/common/shim"
echo ""

# Step 2: Build the Docker image
echo "--- Building Docker image ---"
cd "$PROJECT_ROOT"
docker build -t tth37/xtrans-experiments:expa-benchmark -f exp_a_nccl_gates/Dockerfile .
cd "$EXP_DIR"
echo ""

# Step 3: Create results directory
mkdir -p "$EXP_DIR/results/traces"

# Step 4: Run each config with strace wrapping
for config in "${CONFIGS[@]}"; do
    compose_file="$EXP_DIR/compose.${config}.yml"
    if [[ ! -f "$compose_file" ]]; then
        echo "WARNING: $compose_file not found, skipping"
        continue
    fi

    echo "=== Tracing: $config ==="

    # Clean up any leftover containers from this config
    docker compose -f "$compose_file" down --remove-orphans 2>/dev/null || true

    # Run with strace enabled
    STRACE_WRAP=1 SMOKE_TEST="$SMOKE_TEST" \
        docker compose -f "$compose_file" up \
        --abort-on-container-exit \
        --force-recreate 2>&1 | tee "$EXP_DIR/results/traces/${config}_compose.log"

    # Capture NCCL debug logs
    docker compose -f "$compose_file" logs > "$EXP_DIR/results/traces/${config}_nccl.log" 2>&1 || true

    # Clean up
    docker compose -f "$compose_file" down --remove-orphans 2>/dev/null || true

    echo ""
done

# Step 5: Analyze traces
echo "=== Analyzing traces ==="
python3 "$SCRIPT_DIR/analyze_traces.py" \
    "$EXP_DIR/results/traces/" \
    --output "$EXP_DIR/results/traces/analysis.json"

echo ""
echo "=============================================="
echo "A1 tracing complete."
echo "  Traces:   results/traces/<config>_rank<N>.strace"
echo "  NCCL logs: results/traces/<config>_nccl.log"
echo "  Analysis: results/traces/analysis.json"
echo "=============================================="
