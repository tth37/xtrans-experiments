#!/usr/bin/env bash
# Phase 1: Bare-Metal Baseline for Exp A1 (Tenplex)
#
# Run on node192 WITHOUT containers. Confirms Tenplex works as documented.
# This script guides the experiment — adjust commands as needed based on
# Tenplex's actual build system and CLI.
#
# Prerequisites:
#   - node192 with 4x A100 GPUs
#   - CUDA 12.4+, NCCL 2.21.5, PyTorch installed
#   - Tenplex and Megatron-LM cloned and built
#
# Usage:
#   bash scripts/phase1_bare_metal.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"
RESULTS_DIR="$EXP_DIR/results/phase1"

echo "=== Exp A1 Phase 1: Bare-Metal Baseline ==="
echo "Experiment dir: $EXP_DIR"
echo "Results dir:    $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ---- Step 0: Environment check ----
echo "--- Step 0: Environment Check ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
echo "CUDA version:"
nvcc --version 2>/dev/null | grep "release" || echo "nvcc not found"
echo ""
echo "NCCL version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'NCCL: {torch.cuda.nccl.version()}')" 2>/dev/null || echo "PyTorch not found"
echo ""

# ---- Step 1: Verify Tenplex installation ----
echo "--- Step 1: Verify Tenplex Installation ---"
echo "TODO: Check that Tenplex is built and available."
echo "  Expected location: /opt/tenplex or ~/tenplex"
echo "  Check: ls /opt/tenplex/bin/ or similar"
echo "  If not installed, follow: https://github.com/kungfu-team/tenplex"
echo ""

# ---- Step 2: Start training (4 GPUs, TP=2, DP=2) ----
echo "--- Step 2: Start Training (4 GPUs) ---"
echo "TODO: Launch Tenplex training with Megatron-LM."
echo ""
echo "  Example (adjust paths and arguments):"
echo "    cd /opt/tenplex"
echo "    # Start Tenplex controller"
echo "    tenplex-run --np 4 --model gpt2-small \\"
echo "        --tp 2 --dp 2 \\"
echo "        megatron-lm-gpt --micro-batch-size 4"
echo ""
echo "  Record: training throughput (tokens/s), NCCL transport"
echo "  Save logs: tee $RESULTS_DIR/step2_training_4gpu.log"
echo ""

# ---- Step 3: Elastic scale-down (4 -> 2 GPUs) ----
echo "--- Step 3: Elastic Scale-Down (4 -> 2 GPUs) ---"
echo "TODO: Trigger Tenplex elastic resize."
echo ""
echo "  Tenplex should support runtime resize via its control plane."
echo "  Check Tenplex docs for the resize command/API."
echo ""
echo "  Record:"
echo "    - Reconfiguration time (wall clock)"
echo "    - New parallelism config (expect TP=2, DP=1)"
echo "    - Training throughput after resize"
echo "    - PTC repartitioning time"
echo ""
echo "  Save: tee $RESULTS_DIR/step3_scale_down.log"
echo ""

# ---- Step 4: Elastic scale-up (2 -> 4 GPUs) ----
echo "--- Step 4: Elastic Scale-Up (2 -> 4 GPUs) ---"
echo "TODO: Trigger Tenplex elastic scale-up."
echo ""
echo "  Record:"
echo "    - Reconfiguration time"
echo "    - State redistribution to new GPUs"
echo "    - Training throughput after scale-up"
echo ""
echo "  Save: tee $RESULTS_DIR/step4_scale_up.log"
echo ""

# ---- Step 5: Record observations ----
echo "--- Step 5: Record Observations ---"
echo "Use the observation recorder to save structured results:"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase1 \\"
echo "      --step training_baseline \\"
echo "      --output $RESULTS_DIR/obs_training_baseline.json \\"
echo "      --notes 'Describe training throughput and NCCL transport'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase1 \\"
echo "      --step scale_down_4_to_2 \\"
echo "      --output $RESULTS_DIR/obs_scale_down.json \\"
echo "      --notes 'Describe reconfiguration time and behavior'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase1 \\"
echo "      --step scale_up_2_to_4 \\"
echo "      --output $RESULTS_DIR/obs_scale_up.json \\"
echo "      --notes 'Describe scale-up behavior'"
echo ""

echo "=== Phase 1 guide complete. Execute steps manually and record results. ==="
