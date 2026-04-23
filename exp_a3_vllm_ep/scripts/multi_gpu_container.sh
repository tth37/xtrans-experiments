#!/usr/bin/env bash
# Exp A3 Multi-GPU container regime: vLLM Elastic EP inside a single
# multi-GPU container.
#
# Container is launched with --gpus '"device=0,1,2,3"', mounts the HF cache
# read-only, and starts vllm serve at DP=2. Bench benchmarks use the same
# `vllm bench serve` command as the native regime, run from the host
# against port ${VLLM_PORT}. The benchmark produces comparable numbers to
# native.
#
# Key multi-GPU-container observations captured by `state`:
#     container's HostConfig.DeviceRequests.DeviceIDs  (what Docker claims)
#     host-side nvidia-smi                              (what's actually in use)
# The delta between these is the "orchestrator trap" finding.
#
# Usage:
#   ./scripts/multi_gpu_container.sh start         # run container + vllm serve
#   ./scripts/multi_gpu_container.sh bench LABEL   # vllm bench from host
#   ./scripts/multi_gpu_container.sh scale NEW_DP  # /scale_elastic_ep
#   ./scripts/multi_gpu_container.sh state [TAG]   # GPU + docker inspect snapshot
#   ./scripts/multi_gpu_container.sh cycle         # full 2->4->2 cycle
#   ./scripts/multi_gpu_container.sh stop

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

CONTAINER_NAME="xtrans-exp-a3-multi-gpu"
MODEL_MOUNT_IN_CTN="/models/qwen3-30b-a3b"  # parent dir for HF blob+snapshot layout
RESULTS_DIR="$PROJECT_ROOT/exp_a3_vllm_ep/results/multi_gpu_container"

# ─── Liveness & diagnostics ──────────────────────────────────────────
multi_gpu_container_liveness() {
    # Container still running (not Exited)
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" = "true" ]
}

multi_gpu_container_diag() {
    log "docker inspect State:"
    docker inspect -f '    Status={{.State.Status}} ExitCode={{.State.ExitCode}} Error={{.State.Error}}' \
        "$CONTAINER_NAME" 2>&1 >&2 || true
    log "Last 40 lines of container log:"
    docker logs --tail 40 "$CONTAINER_NAME" 2>&1 | sed 's/^/    /' >&2 || true
}

# ─── Server lifecycle ─────────────────────────────────────────────────
start() {
    mkdir -p "$RESULTS_DIR"
    docker_ensure_image "$VLLM_IMAGE"

    require_gpus_free || return 1

    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    log "Launching container $CONTAINER_NAME on GPUs 0-3"
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus '"device=0,1,2,3"' \
        --ipc=host \
        --shm-size=16g \
        -v "$MODEL_HOST:$MODEL_MOUNT_IN_CTN:ro" \
        -p "${VLLM_PORT}:8000" \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e VLLM_LOGGING_LEVEL=INFO \
        --entrypoint /bin/bash \
        "$VLLM_IMAGE" \
        -c "vllm serve $MODEL_PATH_IN_CTN \
             --served-model-name $SERVED_MODEL_NAME \
             --host 0.0.0.0 --port 8000 \
             --tensor-parallel-size 1 \
             --data-parallel-size 2 \
             --data-parallel-backend ray \
             --enable-expert-parallel \
             --enable-elastic-ep \
             --enable-eplb \
             --all2all-backend allgather_reducescatter \
             --max-model-len 2048 \
             --max-num-seqs 16 \
             --gpu-memory-utilization 0.90 \
             --enforce-eager \
             --trust-remote-code" \
        > /dev/null

    # Container-side logs persist; we also dump to results/ at stop.
    wait_for_ready "http://localhost:${VLLM_PORT}/health" 600 \
        multi_gpu_container_liveness multi_gpu_container_diag
}

stop() {
    if docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Names}}' \
            | grep -q "$CONTAINER_NAME"; then
        log "Saving container logs to $RESULTS_DIR/container.log"
        docker logs "$CONTAINER_NAME" > "$RESULTS_DIR/container.log" 2>&1 || true
    fi
    log "Removing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    sleep 3
    gpu_snapshot >&2
}

# ─── Subcommand: bench ────────────────────────────────────────────────
bench() {
    local label=${1:-unknown} num_prompts=${2:-32} concurrency=${3:-16}
    vllm_bench "$label" "localhost:$VLLM_PORT" "$num_prompts" "$concurrency" "$RESULTS_DIR"
}

# ─── Subcommand: bench_steady ─────────────────────────────────────────
bench_steady() {
    local label=${1:-unknown} num_prompts=${2:-32} concurrency=${3:-16}
    local input_len=${4:-128} output_len=${5:-128}
    bench_to_steady "$label" "localhost:$VLLM_PORT" "$RESULTS_DIR" \
        "$num_prompts" "$concurrency" "$input_len" "$output_len"
}

# ─── Subcommand: scale ────────────────────────────────────────────────
scale() {
    local new_dp=${1:?target DP size required}
    trigger_scale "http://localhost:$VLLM_PORT" "$new_dp" \
        | tee "$RESULTS_DIR/scale_to_dp${new_dp}_$(date +%H%M%S).log"
}

# ─── Subcommand: state ────────────────────────────────────────────────
# Snapshots both what vLLM sees and what Docker still claims -- this is
# the multi-GPU-container signature finding (DeviceIDs don't change).
state() {
    local tag=${1:-snapshot}
    {
        echo "=== Multi-GPU container state ($tag) at $(date) ==="
        echo ""
        echo "## Container ##"
        docker ps --filter "name=$CONTAINER_NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        echo ""
        echo "## Docker DeviceIDs (what Docker still claims) ##"
        container_deviceids "$CONTAINER_NAME"
        echo ""
        echo "## Host nvidia-smi (what's actually in use) ##"
        gpu_snapshot
        echo ""
        echo "## GPU processes on host ##"
        gpu_processes
        echo ""
        echo "## Scaling flag ##"
        curl -sS -X POST "http://localhost:$VLLM_PORT/is_scaling_elastic_ep" 2>&1 || echo "(unreachable)"
    } | tee "$RESULTS_DIR/state_${tag}.txt"
}

# ─── Subcommand: cycle ────────────────────────────────────────────────
cycle() {
    state "pre_cycle"
    bench dp2_initial 16 8
    scale 4
    sleep 3
    state "post_scale_up"
    bench dp4_post_up 32 16
    scale 2
    sleep 3
    state "post_scale_down"
    bench dp2_post_down 16 8
    state "post_cycle"
    log "Cycle complete. Results in $RESULTS_DIR/"
}

# ─── Subcommand: cycle_steady ─────────────────────────────────────────
# Same 2->4->2 structure but each bench is plateau-seeking. Recommended
# protocol for cross-regime throughput comparison (see Exp C report).
cycle_steady() {
    state "pre_cycle"
    bench_steady dp2_initial 16 8 || log "WARN: dp2_initial did not converge"
    scale 4
    sleep 3
    state "post_scale_up"
    bench_steady dp4_post_up 32 16 || log "WARN: dp4_post_up did not converge"
    scale 2
    sleep 3
    state "post_scale_down"
    bench_steady dp2_post_down 16 8 || log "WARN: dp2_post_down did not converge"
    state "post_cycle"
    log "cycle_steady complete. Plateau summaries in $RESULTS_DIR/bench_steady_*.json"
}

# ─── Dispatch ─────────────────────────────────────────────────────────
cmd=${1:-}; shift || true
case "$cmd" in
    start)          start ;;
    stop)           stop ;;
    bench)          bench "$@" ;;
    bench_steady)   bench_steady "$@" ;;
    scale)          scale "$@" ;;
    state)          state "$@" ;;
    cycle)          cycle ;;
    cycle_steady)   cycle_steady ;;
    *)
        cat <<'EOF' >&2
usage: multi_gpu_container.sh {start|stop|bench LABEL [N] [C]|bench_steady LABEL [N] [C] [IN] [OUT]|scale TARGET_DP|state [TAG]|cycle|cycle_steady}

subcommands:
    start                               launch container + start vllm serve inside
    stop                                save container log, remove container
    bench LABEL [N] [C]                 run `vllm bench serve` single-shot
    bench_steady LABEL [N] [C] [IN] [OUT]
                                        loop `vllm bench serve` until TPOT plateaus
    scale TARGET_DP                     POST /scale_elastic_ep
    state [TAG]                         docker inspect DeviceIDs + host nvidia-smi snapshot
                                        -- the "orchestrator trap" observation
    cycle                               full DP=2 → 4 → 2 cycle with single-shot benches
    cycle_steady                        full DP=2 → 4 → 2 cycle with plateau benches
                                        (recommended for cross-regime comparison)

prerequisites:
    * image `xtrans-vllm-ep:v0.19.0` built (see Dockerfile.base)
    * Qwen3-30B-A3B downloaded to /data/models--Qwen--Qwen3-30B-A3B/
    * all 4 GPUs free on the host
EOF
        exit 1
        ;;
esac
