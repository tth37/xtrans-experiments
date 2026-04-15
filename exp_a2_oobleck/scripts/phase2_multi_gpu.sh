#!/usr/bin/env bash
# Phase 2: Multi-GPU Container for Exp A2 (Oobleck)
#
# Runs Oobleck inside a single container with all 4 GPUs.
# Tests whether fault tolerance provides its intended benefit in containers.
#
# Usage:
#   bash scripts/phase2_multi_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"
RESULTS_DIR="$EXP_DIR/results/phase2"

echo "=== Exp A2 Phase 2: Multi-GPU Container ==="
echo ""

mkdir -p "$RESULTS_DIR"

# ---- Step 1: Launch multi-GPU container ----
echo "--- Step 1: Launch Container ---"
echo "  cd $EXP_DIR"
echo "  docker compose -f compose.phase2.yml up -d"
echo "  docker exec -it exp-a2-multi bash"
echo ""

# ---- Step 2: Run training inside container ----
echo "--- Step 2: Run Training (4 GPUs, inside container) ---"
echo "  Run same training as Phase 1 inside the container."
echo "  Record: throughput, compare with bare-metal baseline."
echo ""

# ---- Step 3: Kill one process (simulate GPU failure) ----
echo "--- Step 3: Simulate GPU 3 Failure (inside container) ---"
echo ""
echo "  # From inside the container, kill the rank-3 process:"
echo "  bash scripts/simulate_failure.sh --phase multi_gpu --container exp-a2-multi --target-rank 3"
echo ""
echo "  KEY OBSERVATIONS:"
echo ""
echo "  1. Does Oobleck detect the failure and recover?"
echo "     - Check Oobleck logs for template transition"
echo "     - Record detection time and recovery time"
echo ""
echo "  2. Does the CONTAINER stay healthy?"
echo "     - docker inspect exp-a2-multi --format='{{.State.Health.Status}}'"
echo "     - If health check fails, K8s would kill the ENTIRE pod"
echo "     - This would negate Oobleck's recovery"
echo ""
echo "  3. Is the failed GPU replaceable?"
echo "     - nvidia-smi inside container: GPU 3 is still allocated"
echo "     - nvidia-smi on host: GPU 3 is owned by the container"
echo "     - Can another workload use GPU 3? (No — it's in the container)"
echo "     - The container is now degraded: 4 GPUs allocated, only 3 usable"
echo ""

# ---- Step 4: Kill the container entirely ----
echo "--- Step 4: Catastrophic Failure (kill entire container) ---"
echo ""
echo "  # Simulate a real hardware failure that crashes the container:"
echo "  docker kill exp-a2-multi"
echo ""
echo "  # Result: ALL 4 GPUs are lost."
echo "  # Time to resume training = container restart + Oobleck init + checkpoint load"
echo ""
echo "  Record:"
echo "  - Time from kill to training resumption (full restart)"
echo "  - Compare with Phase 1 single-GPU failure recovery"
echo "  - The blast radius: 1 GPU failure -> all 4 GPUs lost"
echo ""

# ---- Step 5: Record observations ----
echo "--- Step 5: Record Observations ---"
echo "Key questions:"
echo "  1. Does Oobleck's internal recovery work in a container?"
echo "  2. Does the container health model conflict with framework recovery?"
echo "  3. Can the cluster replace a failed GPU without restarting the pod?"
echo "  4. What is the blast radius (1 GPU failure -> N GPUs lost)?"
echo ""
echo "  DECISION POINT:"
echo "  - If the failed GPU is trapped AND/OR container health model conflicts"
echo "    with Oobleck's recovery -> proceed to Phase 3."
echo "  - If everything works fine -> document the finding and stop."
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase2 \\"
echo "      --step internal_recovery \\"
echo "      --output $RESULTS_DIR/obs_internal_recovery.json \\"
echo "      --notes 'Does Oobleck recover internally? Health check interaction?'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase2 \\"
echo "      --step gpu_resource_state \\"
echo "      --output $RESULTS_DIR/obs_gpu_resource_state.json \\"
echo "      --notes 'Is failed GPU replaceable? Trapped in container?'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase2 \\"
echo "      --step blast_radius \\"
echo "      --output $RESULTS_DIR/obs_blast_radius.json \\"
echo "      --notes 'Catastrophic failure: all 4 GPUs lost, full restart time'"
echo ""

# ---- Cleanup ----
echo "--- Cleanup ---"
echo "  docker compose -f compose.phase2.yml down"
echo ""

echo "=== Phase 2 guide complete. ==="
