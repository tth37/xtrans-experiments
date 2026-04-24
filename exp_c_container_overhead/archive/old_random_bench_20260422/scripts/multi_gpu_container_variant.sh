#!/usr/bin/env bash
# Exp C variant of multi_gpu_container.sh — parameterised shm-size + ipc flags.
#
# Same container config as exp_a3_vllm_ep/scripts/multi_gpu_container.sh,
# except --shm-size and --ipc come from env vars so H3 (shm) and H4 (ipc)
# tests can vary them without editing the reference script.
#
# Env vars (all have defaults except VARIANT_TAG):
#   VARIANT_TAG         identifier appended to container name + results dir
#                       (e.g. "shm64m_ipchost", "shm16g_ipcprivate")
#   VARIANT_SHM_SIZE    --shm-size value (default "16g", matches MGC baseline)
#   VARIANT_IPC         --ipc value (default "host", matches MGC baseline)
#   VARIANT_DP          initial data-parallel size (default 2)
#
# Subcommands match multi_gpu_container.sh: start | stop | bench | scale.
# Results land in exp_c_container_overhead/results/variants/<VARIANT_TAG>/.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../exp_a3_vllm_ep/scripts/common.sh
source "$SCRIPT_DIR/../../exp_a3_vllm_ep/scripts/common.sh"

: "${VARIANT_TAG:?VARIANT_TAG is required (e.g. shm64m_ipchost)}"
: "${VARIANT_SHM_SIZE:=16g}"
: "${VARIANT_IPC:=host}"
: "${VARIANT_DP:=2}"

CONTAINER_NAME="xtrans-exp-c-${VARIANT_TAG}"
RESULTS_DIR="$PROJECT_ROOT/exp_c_container_overhead/results/variants/${VARIANT_TAG}"
MODEL_MOUNT_IN_CTN="/models/qwen3-30b-a3b"

variant_liveness() {
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" = "true" ]
}

variant_diag() {
    log "docker inspect State:"
    docker inspect -f '    Status={{.State.Status}} ExitCode={{.State.ExitCode}} Error={{.State.Error}}' \
        "$CONTAINER_NAME" 2>&1 >&2 || true
    log "Last 40 lines of container log:"
    docker logs --tail 40 "$CONTAINER_NAME" 2>&1 | sed 's/^/    /' >&2 || true
}

start() {
    mkdir -p "$RESULTS_DIR"
    docker_ensure_image "$VLLM_IMAGE"
    require_gpus_free || return 1
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    log "Launching $CONTAINER_NAME [shm=$VARIANT_SHM_SIZE ipc=$VARIANT_IPC dp=$VARIANT_DP]"
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus '"device=0,1,2,3"' \
        --ipc="$VARIANT_IPC" \
        --shm-size="$VARIANT_SHM_SIZE" \
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
             --data-parallel-size $VARIANT_DP \
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

    wait_for_ready "http://localhost:${VLLM_PORT}/health" 600 \
        variant_liveness variant_diag
}

stop() {
    if docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Names}}' \
            | grep -q "$CONTAINER_NAME"; then
        log "Saving container logs to $RESULTS_DIR/container.log"
        docker logs "$CONTAINER_NAME" > "$RESULTS_DIR/container.log" 2>&1 || true
    fi
    log "Removing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    sleep 2
    gpu_snapshot >&2
}

bench() {
    local label=${1:-unknown} num_prompts=${2:-32} concurrency=${3:-16}
    vllm_bench "$label" "localhost:$VLLM_PORT" "$num_prompts" "$concurrency" "$RESULTS_DIR"
}

scale() {
    local new_dp=${1:?target DP size required}
    trigger_scale "http://localhost:$VLLM_PORT" "$new_dp" \
        | tee "$RESULTS_DIR/scale_to_dp${new_dp}_$(date +%H%M%S).log"
}

cmd=${1:-}; shift || true
case "$cmd" in
    start) start ;;
    stop)  stop ;;
    bench) bench "$@" ;;
    scale) scale "$@" ;;
    *)
        cat <<'EOF' >&2
usage: multi_gpu_container_variant.sh {start|stop|bench LABEL [N] [C]|scale TARGET_DP}

env vars:
    VARIANT_TAG         required (e.g. shm64m_ipchost)
    VARIANT_SHM_SIZE    default "16g"
    VARIANT_IPC         default "host" (try "private" for H4)
    VARIANT_DP          default 2
EOF
        exit 1
        ;;
esac
