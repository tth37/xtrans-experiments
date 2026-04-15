#!/usr/bin/env bash
# Phase 3: Per-GPU Containers for Exp A1 (Tenplex)
#
# Each GPU in its own isolated container. Scaling = container lifecycle.
# Only proceed here if Phase 2 showed that multi-GPU containers trap resources.
#
# Prerequisites:
#   - Docker image built: docker build -t xtrans-exp-a1:latest .
#   - Phase 2 completed and limitations documented
#
# Usage:
#   bash scripts/phase3_per_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"
RESULTS_DIR="$EXP_DIR/results/phase3"

echo "=== Exp A1 Phase 3: Per-GPU Containers ==="
echo ""

mkdir -p "$RESULTS_DIR"

# ---- Step 1: Launch per-GPU containers ----
echo "--- Step 1: Launch Per-GPU Containers ---"
echo "  cd $EXP_DIR"
echo "  docker compose -f compose.phase3.yml up -d"
echo ""
echo "  Verify containers are running:"
echo "  docker compose -f compose.phase3.yml ps"
echo ""
echo "  Each container should see exactly 1 GPU:"
echo "  for i in 0 1 2 3; do"
echo "      echo \"=== Container gpu\$i ===\""
echo "      docker exec exp-a1-gpu\$i nvidia-smi --query-gpu=index,name --format=csv,noheader"
echo "  done"
echo ""

# ---- Step 2: Launch distributed training across containers ----
echo "--- Step 2: Launch Training Across Containers ---"
echo "  This is the key challenge: Tenplex must coordinate across 4 separate"
echo "  containers. Each container has 1 GPU."
echo ""
echo "  Attempt WITHOUT any workarounds first (Level 0):"
echo "  - Set up Tenplex to treat each container as a node with 1 GPU"
echo "  - Use container hostnames for process discovery"
echo "  - Observe: does NCCL initialize? What transport does it use?"
echo ""
echo "  If Level 0 fails, try workarounds progressively (see README.md)."
echo ""
echo "  Record: NCCL debug logs, error messages, container events"
echo "  Save: logs/phase3_level0_nccl.log, logs/phase3_level0_stderr.log"
echo ""

# ---- Step 3: Elastic scale-down (stop containers) ----
echo "--- Step 3: Elastic Scale-Down (stop C2, C3) ---"
echo ""
echo "  # Stop containers for GPU 2 and 3:"
echo "  docker stop exp-a1-gpu2 exp-a1-gpu3"
echo ""
echo "  # Timestamp the stop event:"
echo "  date -u +%Y-%m-%dT%H:%M:%S.%NZ | tee $RESULTS_DIR/scale_down_timestamp.txt"
echo ""
echo "  Observe:"
echo "  - Does Tenplex detect that C2/C3 are gone?"
echo "  - Can it trigger reconfiguration (PTC repartitioning)?"
echo "  - Does the NCCL communicator for C0-C1 reform correctly?"
echo "  - Are GPUs 2/3 visible to the host for other jobs?"
echo ""
echo "  # Check if freed GPUs are available:"
echo "  nvidia-smi"
echo ""
echo "  Record all observations, errors, timings."
echo ""

# ---- Step 4: Elastic scale-up (start new containers) ----
echo "--- Step 4: Elastic Scale-Up (start C2', C3') ---"
echo ""
echo "  # Start new containers (may need to recreate if stopped):"
echo "  docker compose -f compose.phase3.yml up -d gpu2 gpu3"
echo ""
echo "  Observe:"
echo "  - Can Tenplex discover the new containers?"
echo "  - Can it create a new NCCL communicator spanning old + new?"
echo "  - Does PTC state redistribute to the new containers?"
echo "  - Training throughput after scale-up?"
echo ""

# ---- Step 5: Record observations ----
echo "--- Step 5: Record All Observations ---"
echo ""
echo "  For each workaround level attempted:"
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase3 \\"
echo "      --step 'level0_training_attempt' \\"
echo "      --output $RESULTS_DIR/obs_level0_training.json \\"
echo "      --notes 'Describe: did NCCL initialize? transport? errors?'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase3 \\"
echo "      --step 'scale_down_4_to_2' \\"
echo "      --output $RESULTS_DIR/obs_scale_down.json \\"
echo "      --notes 'Describe: did Tenplex detect loss? reconfigure? errors?'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase3 \\"
echo "      --step 'scale_up_2_to_4' \\"
echo "      --output $RESULTS_DIR/obs_scale_up.json \\"
echo "      --notes 'Describe: did new containers join? NCCL reform? errors?'"
echo ""

# ---- Cleanup ----
echo "--- Cleanup ---"
echo "  docker compose -f compose.phase3.yml down"
echo ""

echo "=== Phase 3 guide complete. ==="
