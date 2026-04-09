#!/usr/bin/env bash
# run_a2_benchmark.sh — Exp A, Task A2: Benchmark across 4 configurations
#
# Runs the benchmark in all configurations (baseline, isolated, exp2, shim)
# and produces a comparison table with pass/fail verdict.
#
# Usage:
#   bash scripts/run_a2_benchmark.sh          # Full benchmark
#   bash scripts/run_a2_benchmark.sh --smoke  # Quick smoke test
#
# Output:
#   results/baseline.json    — Shared-namespace benchmark
#   results/isolated.json    — Isolated benchmark (expected broken)
#   results/exp2.json        — exp2 workaround benchmark
#   results/shim.json        — Shim benchmark (the experiment)
#   results/<config>_nccl.log — NCCL debug logs per config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"

# Parse arguments
SMOKE_TEST="false"
CONFIGS=("baseline" "isolated" "exp2" "shim")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE_TEST="true"; shift ;;
        --configs) CONFIGS=(); IFS=',' read -ra CONFIGS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "Exp A — Task A2: Benchmark"
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
mkdir -p "$EXP_DIR/results"

# Step 4: Run each config
FAILED=()
for config in "${CONFIGS[@]}"; do
    compose_file="$EXP_DIR/compose.${config}.yml"
    if [[ ! -f "$compose_file" ]]; then
        echo "WARNING: $compose_file not found, skipping"
        FAILED+=("$config")
        continue
    fi

    echo "=== Running: $config ==="

    # Clean up any leftover containers
    docker compose -f "$compose_file" down --remove-orphans 2>/dev/null || true

    # Run benchmark (no strace wrapping)
    if SMOKE_TEST="$SMOKE_TEST" \
        docker compose -f "$compose_file" up \
        --abort-on-container-exit \
        --force-recreate 2>&1 | tee "$EXP_DIR/results/${config}_compose.log"; then
        echo "  $config: completed"
    else
        echo "  $config: FAILED (exit code $?)"
        FAILED+=("$config")
    fi

    # Capture NCCL debug logs
    docker compose -f "$compose_file" logs > "$EXP_DIR/results/${config}_nccl.log" 2>&1 || true

    # Clean up
    docker compose -f "$compose_file" down --remove-orphans 2>/dev/null || true

    echo ""
done

# Step 5: Compare results
echo "=== Results Comparison ==="
RESULT_FILES=()
for config in "${CONFIGS[@]}"; do
    result_file="$EXP_DIR/results/${config}.json"
    if [[ -f "$result_file" ]]; then
        RESULT_FILES+=("$result_file")
    fi
done

if [[ ${#RESULT_FILES[@]} -gt 0 ]]; then
    python3 "$SCRIPT_DIR/compare_results.py" "${RESULT_FILES[@]}"
else
    echo "No result files found. Check logs in results/<config>_compose.log"
fi

# Step 6: Summary
echo "=============================================="
echo "A2 benchmark complete."
echo ""
echo "Results:"
for config in "${CONFIGS[@]}"; do
    result_file="$EXP_DIR/results/${config}.json"
    if [[ -f "$result_file" ]]; then
        echo "  $config: $result_file"
    else
        echo "  $config: MISSING"
    fi
done
echo ""
echo "NCCL logs:"
for config in "${CONFIGS[@]}"; do
    log_file="$EXP_DIR/results/${config}_nccl.log"
    if [[ -f "$log_file" ]]; then
        echo "  $config: $log_file"
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: The following configs failed: ${FAILED[*]}"
    echo "Check results/<config>_compose.log for details."
fi
echo "=============================================="
