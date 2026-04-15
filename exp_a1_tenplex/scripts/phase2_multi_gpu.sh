#!/usr/bin/env bash
# Phase 2: Multi-GPU Container for Exp A1 (Tenplex)
#
# Runs Tenplex inside a single container with all 4 GPUs.
# Tests whether framework-level elasticity translates to cluster-level benefit.
#
# Prerequisites:
#   - Docker image built: docker build -t xtrans-exp-a1:latest .
#
# Usage:
#   bash scripts/phase2_multi_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"
RESULTS_DIR="$EXP_DIR/results/phase2"

echo "=== Exp A1 Phase 2: Multi-GPU Container ==="
echo ""

mkdir -p "$RESULTS_DIR"

# ---- Step 1: Launch multi-GPU container ----
echo "--- Step 1: Launch Container ---"
echo "  cd $EXP_DIR"
echo "  docker compose -f compose.phase2.yml up -d"
echo "  docker exec -it exp-a1-multi bash"
echo ""

# ---- Step 2: Run training inside container ----
echo "--- Step 2: Run Training (4 GPUs, inside container) ---"
echo "Inside the container, run the same training as Phase 1:"
echo ""
echo "  # Verify GPUs are visible"
echo "  nvidia-smi"
echo "  # Should show all 4 GPUs"
echo ""
echo "  # Run Tenplex training (same as Phase 1)"
echo "  # Record: throughput, NCCL transport"
echo "  # Compare with Phase 1 bare-metal baseline"
echo ""

# ---- Step 3: Elastic scale-down (inside container) ----
echo "--- Step 3: Elastic Scale-Down (4 -> 2 GPUs, inside container) ---"
echo "Trigger Tenplex resize to 2 GPUs inside the container."
echo ""
echo "  KEY OBSERVATION: After Tenplex scales to 2 GPUs internally,"
echo "  check from OUTSIDE the container:"
echo ""
echo "  # From host (outside the container):"
echo "  nvidia-smi  # Do all 4 GPUs still show as allocated?"
echo ""
echo "  # Try to launch ANOTHER container using the 'freed' GPUs:"
echo "  docker run --rm --runtime=nvidia \\"
echo "      -e NVIDIA_VISIBLE_DEVICES=2,3 \\"
echo "      nvcr.io/nvidia/pytorch:24.07-py3 nvidia-smi"
echo "  # Does this work? Or are GPUs 2,3 still locked to the first container?"
echo ""
echo "  Record whether freed GPUs are actually available to other workloads."
echo ""

# ---- Step 4: Elastic scale-up (add new GPUs) ----
echo "--- Step 4: Elastic Scale-Up (add external GPUs) ---"
echo "Can the running container acquire NEW GPUs?"
echo ""
echo "  # Can you add GPU resources to a running container?"
echo "  # Docker does not support GPU hot-add."
echo "  # K8s In-Place Pod Resize does not support GPU resources (as of 1.33)."
echo "  # Record: what happens when you try."
echo ""

# ---- Step 5: Record observations ----
echo "--- Step 5: Record Observations ---"
echo "Key questions to answer:"
echo "  1. Does Tenplex training work inside a container? Performance vs bare metal?"
echo "  2. Does elastic resize work inside the container?"
echo "  3. Are freed GPUs reclaimable by the cluster?"
echo "  4. Can the container acquire new GPUs at runtime?"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase2 \\"
echo "      --step gpu_resource_test \\"
echo "      --output $RESULTS_DIR/obs_gpu_resource_test.json \\"
echo "      --notes 'Describe whether freed GPUs are available to other workloads'"
echo ""
echo "  DECISION POINT: If freed GPUs are NOT reclaimable, proceed to Phase 3."
echo "  If they ARE reclaimable, document this finding and stop."
echo ""

# ---- Cleanup ----
echo "--- Cleanup ---"
echo "  docker compose -f compose.phase2.yml down"
echo ""

echo "=== Phase 2 guide complete. ==="
