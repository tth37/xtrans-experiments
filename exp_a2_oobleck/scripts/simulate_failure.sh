#!/usr/bin/env bash
# simulate_failure.sh — GPU failure simulation for Exp A2
#
# Simulates GPU failures at different containerization levels.
#
# Usage:
#   # Phase 1 (bare metal): kill the training process on rank 3
#   ./simulate_failure.sh --phase bare_metal --target-rank 3
#
#   # Phase 2 (multi-GPU container): kill process inside container
#   ./simulate_failure.sh --phase multi_gpu --container exp-a2-multi --target-rank 3
#
#   # Phase 3 (per-GPU container): kill the entire container
#   ./simulate_failure.sh --phase per_gpu --container exp-a2-gpu3

set -euo pipefail

PHASE=""
CONTAINER=""
TARGET_RANK=""
SIGNAL="KILL"  # Use KILL for immediate death (simulates hardware failure)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)      PHASE="$2"; shift 2 ;;
        --container)  CONTAINER="$2"; shift 2 ;;
        --target-rank) TARGET_RANK="$2"; shift 2 ;;
        --signal)     SIGNAL="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$PHASE" ]; then
    echo "Usage: $0 --phase {bare_metal|multi_gpu|per_gpu} [--container NAME] [--target-rank N]"
    exit 1
fi

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)

case "$PHASE" in
    bare_metal)
        if [ -z "$TARGET_RANK" ]; then
            echo "Error: --target-rank required for bare_metal phase"
            exit 1
        fi
        echo "[$TIMESTAMP] Simulating GPU $TARGET_RANK failure (bare metal)"
        echo "Looking for training process on GPU $TARGET_RANK..."

        # Find the Python training process using GPU $TARGET_RANK
        # This heuristic may need adjustment based on Oobleck's process model
        PID=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
              | awk -F', ' -v gpu="$TARGET_RANK" 'NR==(gpu+1) {print $1}')

        if [ -z "$PID" ]; then
            echo "WARNING: Could not find process on GPU $TARGET_RANK via nvidia-smi."
            echo "Try manually: ps aux | grep python, then kill -$SIGNAL <pid>"
            exit 1
        fi

        echo "Killing PID $PID (GPU $TARGET_RANK) with signal $SIGNAL"
        kill -"$SIGNAL" "$PID"
        echo "[$TIMESTAMP] Process $PID killed."
        ;;

    multi_gpu)
        if [ -z "$CONTAINER" ] || [ -z "$TARGET_RANK" ]; then
            echo "Error: --container and --target-rank required for multi_gpu phase"
            exit 1
        fi
        echo "[$TIMESTAMP] Simulating GPU $TARGET_RANK failure inside container $CONTAINER"

        # Find and kill the training process for the target rank inside the container
        echo "Looking for rank $TARGET_RANK process inside $CONTAINER..."
        # Oobleck/PyTorch typically set LOCAL_RANK env var
        PID=$(docker exec "$CONTAINER" bash -c \
            "ps aux | grep '[p]ython.*rank.*$TARGET_RANK' | head -1 | awk '{print \$2}'" \
            2>/dev/null || true)

        if [ -z "$PID" ]; then
            echo "WARNING: Could not auto-detect rank $TARGET_RANK process."
            echo "Try manually inside container:"
            echo "  docker exec $CONTAINER ps aux | grep python"
            echo "  docker exec $CONTAINER kill -$SIGNAL <pid>"
            exit 1
        fi

        echo "Killing PID $PID inside $CONTAINER with signal $SIGNAL"
        docker exec "$CONTAINER" kill -"$SIGNAL" "$PID"
        echo "[$TIMESTAMP] Process killed inside container."
        ;;

    per_gpu)
        if [ -z "$CONTAINER" ]; then
            echo "Error: --container required for per_gpu phase"
            exit 1
        fi
        echo "[$TIMESTAMP] Killing container $CONTAINER (simulates GPU failure)"
        docker kill "$CONTAINER"
        echo "[$TIMESTAMP] Container $CONTAINER killed."
        echo ""
        echo "Remaining containers:"
        docker ps --filter "name=exp-a2-gpu" --format "table {{.Names}}\t{{.Status}}"
        ;;

    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: bare_metal, multi_gpu, per_gpu"
        exit 1
        ;;
esac

echo ""
echo "Failure simulated at: $TIMESTAMP"
echo "Monitor Oobleck logs for detection and recovery."
