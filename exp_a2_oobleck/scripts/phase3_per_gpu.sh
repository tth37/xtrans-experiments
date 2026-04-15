#!/usr/bin/env bash
# Phase 3: Per-GPU Containers for Exp A2 (Oobleck)
#
# Each GPU in its own container. GPU failure = one container dies.
# Only proceed if Phase 2 showed limitations of multi-GPU containers.
#
# Usage:
#   bash scripts/phase3_per_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"
RESULTS_DIR="$EXP_DIR/results/phase3"

echo "=== Exp A2 Phase 3: Per-GPU Containers ==="
echo ""

mkdir -p "$RESULTS_DIR"

# ---- Step 1: Launch per-GPU containers ----
echo "--- Step 1: Launch Per-GPU Containers ---"
echo "  cd $EXP_DIR"
echo "  docker compose -f compose.phase3.yml up -d"
echo "  docker compose -f compose.phase3.yml ps"
echo ""

# ---- Step 2: Launch pipeline training across containers ----
echo "--- Step 2: Launch Training Across Containers ---"
echo "  Attempt WITHOUT workarounds first (Level 0)."
echo "  Oobleck must coordinate pipeline stages across 4 containers."
echo ""
echo "  Observe:"
echo "  - Does NCCL initialize across container boundaries?"
echo "  - What transport is used (P2P NVLink? SHM? TCP fallback?)?"
echo "  - Does Oobleck's pipeline setup complete?"
echo "  - Training throughput vs Phase 1 and Phase 2?"
echo ""

# ---- Step 3: Kill container C3 (simulate GPU failure) ----
echo "--- Step 3: Kill Container C3 (GPU 3 failure) ---"
echo ""
echo "  docker kill exp-a2-gpu3"
echo "  date -u +%Y-%m-%dT%H:%M:%S.%NZ | tee $RESULTS_DIR/c3_kill_timestamp.txt"
echo ""
echo "  Observe:"
echo "  - Do surviving containers (C0, C1, C2) detect C3's death?"
echo "  - How? (NCCL timeout, TCP connection reset, something else?)"
echo "  - Can Oobleck trigger pipeline template transition?"
echo "  - Can surviving containers reform their NCCL communicator?"
echo "  - Does pipeline stage redistribution succeed across containers?"
echo "  - Training throughput after recovery (3 GPUs)?"
echo ""
echo "  Record NCCL debug logs from all surviving containers:"
echo "  for i in 0 1 2; do"
echo "      docker logs exp-a2-gpu\$i > $RESULTS_DIR/c\${i}_logs_after_c3_kill.txt 2>&1"
echo "  done"
echo ""

# ---- Step 4: Replace C3 with new container ----
echo "--- Step 4: Replace C3 (start C3') ---"
echo ""
echo "  docker compose -f compose.phase3.yml up -d gpu3"
echo ""
echo "  Observe:"
echo "  - Can Oobleck detect and incorporate the new container?"
echo "  - Can NCCL create a communicator spanning old + new containers?"
echo "  - Does training resume with 4 GPUs?"
echo ""

# ---- Step 5: Kill C2 (cascading failure) ----
echo "--- Step 5: Kill Container C2 (cascading: 3 -> 2) ---"
echo ""
echo "  docker kill exp-a2-gpu2"
echo "  date -u +%Y-%m-%dT%H:%M:%S.%NZ | tee $RESULTS_DIR/c2_kill_timestamp.txt"
echo ""
echo "  Observe: cascading recovery behavior across containers."
echo ""

# ---- Step 6: Record observations ----
echo "--- Step 6: Record All Observations ---"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase3 \\"
echo "      --step 'training_across_containers' \\"
echo "      --output $RESULTS_DIR/obs_training.json \\"
echo "      --notes 'Did NCCL init? Transport? Pipeline setup? Throughput?'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase3 \\"
echo "      --step 'c3_failure_and_recovery' \\"
echo "      --output $RESULTS_DIR/obs_c3_failure.json \\"
echo "      --notes 'Detection method, recovery time, comm reform, template transition'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase3 \\"
echo "      --step 'c3_replacement' \\"
echo "      --output $RESULTS_DIR/obs_c3_replacement.json \\"
echo "      --notes 'Could new container join? NCCL reform? Training resume?'"
echo ""

# ---- Cleanup ----
echo "--- Cleanup ---"
echo "  docker compose -f compose.phase3.yml down"
echo ""

echo "=== Phase 3 guide complete. ==="
