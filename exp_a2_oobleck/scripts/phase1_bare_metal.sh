#!/usr/bin/env bash
# Phase 1: Bare-Metal Baseline for Exp A2 (Oobleck)
#
# Run on node192 WITHOUT containers. Confirms Oobleck's fault tolerance works.
#
# Prerequisites:
#   - node192 with 4x A100 GPUs
#   - CUDA 12.4+, NCCL 2.21.5, PyTorch installed
#   - Oobleck cloned and built
#
# Usage:
#   bash scripts/phase1_bare_metal.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"
RESULTS_DIR="$EXP_DIR/results/phase1"

echo "=== Exp A2 Phase 1: Bare-Metal Baseline ==="
echo "Experiment dir: $EXP_DIR"
echo "Results dir:    $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ---- Step 0: Environment check ----
echo "--- Step 0: Environment Check ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ---- Step 1: Verify Oobleck installation ----
echo "--- Step 1: Verify Oobleck Installation ---"
echo "TODO: Check that Oobleck is built and importable."
echo "  Expected location: /opt/Oobleck"
echo "  Check: python3 -c 'import oobleck; print(oobleck.__file__)'"
echo "  If not installed, follow: https://github.com/SymbioticLab/Oobleck"
echo ""

# ---- Step 2: Configure pipeline templates ----
echo "--- Step 2: Configure Pipeline Templates ---"
echo "Oobleck needs pre-computed pipeline templates for different GPU counts."
echo ""
echo "  Create templates for: 4 GPUs, 3 GPUs, 2 GPUs"
echo "  The templates define pipeline stage assignments for each scenario."
echo ""
echo "  Check Oobleck docs/examples for template configuration format."
echo "  Save template configs to $RESULTS_DIR/templates/"
echo ""

# ---- Step 3: Start pipeline training (4 GPUs) ----
echo "--- Step 3: Start Pipeline Training (4 GPUs) ---"
echo "TODO: Launch Oobleck training with GPT-2 (small model)."
echo ""
echo "  Example (adjust based on Oobleck's actual CLI):"
echo "    python3 -m oobleck.run \\"
echo "        --model gpt2 \\"
echo "        --num-gpus 4 \\"
echo "        --pipeline-templates templates/ \\"
echo "        --fault-tolerance-level 2"
echo ""
echo "  Record: training throughput, pipeline configuration"
echo "  Save: tee $RESULTS_DIR/step3_training_4gpu.log"
echo ""

# ---- Step 4: Simulate GPU 3 failure (4 -> 3 GPUs) ----
echo "--- Step 4: Simulate GPU 3 Failure ---"
echo ""
echo "  # Find the training process on GPU 3:"
echo "  # nvidia-smi | grep python  (or check Oobleck's process list)"
echo ""
echo "  # Kill it:"
echo "  bash scripts/simulate_failure.sh --phase bare_metal --target-rank 3"
echo ""
echo "  # Timestamp the kill event:"
echo "  date -u +%Y-%m-%dT%H:%M:%S.%NZ | tee $RESULTS_DIR/gpu3_kill_timestamp.txt"
echo ""
echo "  Record:"
echo "  - Failure detection time (from kill to Oobleck log entry)"
echo "  - Recovery time (from detection to training resumption)"
echo "  - Which pipeline template was selected (3-GPU template)"
echo "  - Training throughput after recovery"
echo ""

# ---- Step 5: Simulate GPU 2 failure (3 -> 2 GPUs) ----
echo "--- Step 5: Simulate GPU 2 Failure (cascading) ---"
echo ""
echo "  bash scripts/simulate_failure.sh --phase bare_metal --target-rank 2"
echo "  date -u +%Y-%m-%dT%H:%M:%S.%NZ | tee $RESULTS_DIR/gpu2_kill_timestamp.txt"
echo ""
echo "  Record: same metrics as Step 4"
echo ""

# ---- Step 6: Record observations ----
echo "--- Step 6: Record Observations ---"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase1 \\"
echo "      --step training_baseline \\"
echo "      --output $RESULTS_DIR/obs_training_baseline.json \\"
echo "      --notes 'Training throughput and pipeline config with 4 GPUs'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase1 \\"
echo "      --step gpu3_failure_recovery \\"
echo "      --output $RESULTS_DIR/obs_gpu3_failure.json \\"
echo "      --notes 'Detection time, recovery time, template selected'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a2 --phase phase1 \\"
echo "      --step gpu2_failure_cascading \\"
echo "      --output $RESULTS_DIR/obs_gpu2_failure.json \\"
echo "      --notes 'Cascading recovery behavior, 2-GPU throughput'"
echo ""

echo "=== Phase 1 guide complete. ==="
