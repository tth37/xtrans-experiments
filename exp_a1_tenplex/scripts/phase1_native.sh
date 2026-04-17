#!/usr/bin/env bash
# Phase 1: Reproduce Tenplex Elastic Scheduling
#
# Runs tenplex-run on the host. It creates Docker containers with
# --network host and --gpus device=X,Y to train with Megatron-LM.
# The scaling schedule triggers parallelism reconfiguration (4→2→4 GPUs).
#
# Observe: full container stop-recreate at each scaling event,
# PTC state transformation overhead, GPU idle time.
#
# Prerequisites:
#   - Tenplex built: tenplex/bin/tenplex-run exists
#   - Docker image available (pulled or built)
#   - SSH to localhost working
#   - mlfsd running (or this script starts it)
#
# Usage:
#   bash scripts/phase1_native.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"
RESULTS_DIR="$EXP_DIR/results/phase1"
TENPLEX_BIN="$EXP_DIR/tenplex/bin"
CONFIGS_DIR="$EXP_DIR/configs"

# Node configuration
HOST_IP="10.0.2.192"
GPUS_PER_HOST=4
GPUS_PER_CONTAINER=4
MLFS_PORT=20010

# Training image
IMAGE="kungfu.azurecr.io/mw-megatron-lm-23.06-update:v0.0.3"

echo "=== Exp A1 Phase 1: Reproduce Tenplex Elastic Scheduling ==="
echo "Experiment dir: $EXP_DIR"
echo "Results dir:    $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ---- Step 0: Prerequisites check ----
echo "--- Step 0: Prerequisites ---"

echo -n "GPUs: "
nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' '; '
echo ""

echo -n "Docker: "
docker --version

echo -n "SSH: "
ssh -o BatchMode=yes localhost 'echo ok' 2>/dev/null || {
    echo "FAILED — fix with: chmod 600 ~/.ssh/authorized_keys"
    exit 1
}

echo -n "tenplex-run: "
if [ -x "$TENPLEX_BIN/tenplex-run" ]; then
    echo "$TENPLEX_BIN/tenplex-run"
else
    echo "NOT FOUND — build with: cd tenplex && PATH=/usr/local/go/bin:\$PATH make binaries"
    exit 1
fi

echo -n "Docker image: "
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -qF "$IMAGE"; then
    echo "$IMAGE (local)"
else
    echo "NOT FOUND — pull or build first"
    echo "  Pull:  docker pull $IMAGE"
    echo "  Build: docker build -t $IMAGE -f tenplex/Dockerfile tenplex/"
    exit 1
fi

echo ""

# ---- Step 1: Start mlfsd if not running ----
echo "--- Step 1: mlfsd Daemon ---"
if curl -sf "http://localhost:$MLFS_PORT/debug" >/dev/null 2>&1; then
    echo "mlfsd already running on port $MLFS_PORT"
else
    echo "Starting mlfsd in background..."
    nohup "$TENPLEX_BIN/mlfsd" --ctrl-port "$MLFS_PORT" \
        > "$RESULTS_DIR/mlfsd.log" 2>&1 &
    MLFSD_PID=$!
    echo "mlfsd started (PID $MLFSD_PID), log: $RESULTS_DIR/mlfsd.log"
    sleep 2
    if ! kill -0 "$MLFSD_PID" 2>/dev/null; then
        echo "ERROR: mlfsd failed to start. Check $RESULTS_DIR/mlfsd.log"
        exit 1
    fi
fi
echo ""

# ---- Step 2: Start docker events logger ----
echo "--- Step 2: Docker Events Logger ---"
echo "Recording container lifecycle events..."
docker events --filter 'name=trainer' \
    --format '{{.Time}} {{.Action}} {{.Actor.Attributes.name}}' \
    > "$RESULTS_DIR/docker_events.log" 2>&1 &
EVENTS_PID=$!
echo "docker events logger started (PID $EVENTS_PID)"
echo ""

# ---- Step 3: Run tenplex-run ----
echo "--- Step 3: Run tenplex-run ---"
echo "Schedule: 4 GPUs (10min) → 2 GPUs (10min) → 4 GPUs (10min) → stop"
echo "Para config: 4 GPUs = TP=2,DP=2 | 2 GPUs = TP=2,DP=1"
echo ""

export PATH="$TENPLEX_BIN:$PATH"

echo "Launching tenplex-run..."
echo "  Logs: $RESULTS_DIR/tenplex-run.log"
echo ""

set -x
tenplex-run \
    -framework megatron-lm \
    -model gpt \
    -model-size medium \
    -batch-size 32 \
    -micro-batch-size 4 \
    -precision fp16 \
    -dataset enwiki \
    -image "$IMAGE" \
    -user "$USER" \
    -hosts "$HOST_IP" \
    -gpu-per-host "$GPUS_PER_HOST" \
    -gpu-per-container "$GPUS_PER_CONTAINER" \
    -mlfs-port "$MLFS_PORT" \
    -tenplex-prefix "$HOME/.tenplex" \
    -jobid exp-a1-phase1 \
    -time-based \
    -network-interface enp3s0f0np0 \
    -logdir "$RESULTS_DIR/logs" \
    -para-config "$CONFIGS_DIR/para-config.json" \
    -schedule-file "$CONFIGS_DIR/schedule.json" \
    -no-pull-image \
    2>&1 | tee "$RESULTS_DIR/tenplex-run.log"
set +x

echo ""
echo "tenplex-run finished."

# ---- Step 4: Stop background processes ----
echo "--- Step 4: Cleanup ---"
kill "$EVENTS_PID" 2>/dev/null || true
echo "Docker events log: $RESULTS_DIR/docker_events.log"
echo ""

# ---- Step 5: Record observations ----
echo "--- Step 5: Record Observations ---"
echo ""
echo "Review the logs and record findings:"
echo ""
echo "  # Container lifecycle events:"
echo "  cat $RESULTS_DIR/docker_events.log"
echo ""
echo "  # Tenplex orchestration log (scaling events, timings):"
echo "  grep -E 'State transformation|Start training|Finished training|sleep|woke' $RESULTS_DIR/tenplex-run.log"
echo ""
echo "  # Record structured observations:"
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase1 \\"
echo "      --step training_4gpu \\"
echo "      --output $RESULTS_DIR/obs_training_4gpu.json \\"
echo "      --notes 'Training throughput, NCCL transport, container config'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase1 \\"
echo "      --step scale_down_4_to_2 \\"
echo "      --output $RESULTS_DIR/obs_scale_down.json \\"
echo "      --notes 'Reconfiguration time, container stop-recreate, PTC transform'"
echo ""
echo "  python3 $PROJECT_ROOT/common/harness/record_observation.py \\"
echo "      --experiment exp_a1 --phase phase1 \\"
echo "      --step scale_up_2_to_4 \\"
echo "      --output $RESULTS_DIR/obs_scale_up.json \\"
echo "      --notes 'Scale-up reconfiguration, state redistribution'"
echo ""

echo "=== Phase 1 complete. ==="
