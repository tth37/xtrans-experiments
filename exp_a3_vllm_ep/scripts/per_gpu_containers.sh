#!/usr/bin/env bash
# Exp A3 Per-GPU containers regime: one vLLM Elastic EP rank per container.
#
# Four containers (ep-rank-0 ... ep-rank-3) on a Docker bridge network
# ${NETWORK}. Each container owns exactly one GPU via --gpus device=N.
# ep-rank-0 runs a Ray head + vllm serve; ep-rank-1..3 run Ray workers.
# `vllm bench serve` runs from the host against ep-rank-0's exposed port.
#
# Key per-GPU-container observations:
#     each container's DeviceIDs is ['N'] (not ['0','1','2','3']) -- the
#     multi-GPU-container trap is gone, but NCCL falls back to
#     NET/Socket/0 because all three same-node gates (hostname, /dev/shm,
#     abstract UDS) fail.
#
# Usage:
#   ./scripts/per_gpu_containers.sh up            # full launch: network + all 4
#                                                 # containers + vllm serve
#   ./scripts/per_gpu_containers.sh bench LABEL   # vllm bench from host
#   ./scripts/per_gpu_containers.sh nccl-grep     # extract NCCL transport
#                                                 # selection from vllm log
#   ./scripts/per_gpu_containers.sh state [TAG]   # per-container DeviceIDs +
#                                                 # host nvidia-smi
#   ./scripts/per_gpu_containers.sh down          # teardown

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

NETWORK="xtrans-per-gpu"
MODEL_MOUNT_IN_CTN="/models/qwen3-30b-a3b"
RESULTS_DIR="$PROJECT_ROOT/exp_a3_vllm_ep/results/per_gpu_containers"
SERVE_LOG_IN_CTN="/tmp/vllm-serve.log"

# ─── Internal helpers ─────────────────────────────────────────────────
ensure_network() {
    if ! docker network inspect "$NETWORK" > /dev/null 2>&1; then
        log "Creating docker network $NETWORK"
        docker network create "$NETWORK" > /dev/null
    fi
}

head_ip() {
    # IP of ep-rank-0 on the $NETWORK bridge
    docker inspect ep-rank-0 \
        --format "{{(index .NetworkSettings.Networks \"$NETWORK\").IPAddress}}"
}

launch_rank_container() {
    local rank=$1
    local name="ep-rank-$rank"
    local start_cmd=$2       # ray start command to run (head or worker variant)
    docker rm -f "$name" 2>/dev/null || true

    local extra_ports=()
    if [ "$rank" = "0" ]; then
        extra_ports=(-p "${VLLM_PORT}:${VLLM_PORT}" -p "${RAY_PORT}:${RAY_PORT}")
    fi

    log "Starting $name on GPU $rank"
    docker run -d \
        --name "$name" \
        --hostname "$name" \
        --network "$NETWORK" \
        --gpus "\"device=$rank\"" \
        --ipc=host \
        --shm-size=16g \
        -v "$MODEL_HOST:$MODEL_MOUNT_IN_CTN:ro" \
        "${extra_ports[@]}" \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e VLLM_LOGGING_LEVEL=INFO \
        -e NCCL_DEBUG=INFO \
        --entrypoint /bin/bash \
        "$VLLM_IMAGE" \
        -c "$start_cmd && tail -f /dev/null" \
        > /dev/null
}

# ─── Liveness & diagnostics ──────────────────────────────────────────
# Check that all 4 ep-rank-N containers are still Running AND that the
# vllm-serve process inside ep-rank-0 is still alive (crashes inside the
# backgrounded docker exec are silent otherwise).
per_gpu_containers_liveness() {
    for i in 0 1 2 3; do
        local state
        state=$(docker inspect -f '{{.State.Running}}' "ep-rank-$i" 2>/dev/null)
        [ "$state" = "true" ] || return 1
    done
    # vllm serve PID tracked by pgrep inside ep-rank-0
    docker exec ep-rank-0 pgrep -f 'vllm serve' > /dev/null 2>&1 || return 1
    return 0
}

per_gpu_containers_diag() {
    log "Container states:"
    for i in 0 1 2 3; do
        echo "    ep-rank-$i: $(docker inspect -f 'Status={{.State.Status}} ExitCode={{.State.ExitCode}}' ep-rank-$i 2>/dev/null || echo 'missing')" >&2
    done
    log "Last 40 lines of vllm-serve log (inside ep-rank-0):"
    docker exec ep-rank-0 tail -40 "$SERVE_LOG_IN_CTN" 2>&1 | sed 's/^/    /' >&2 || true
}

# ─── Ray cluster + vLLM launch ────────────────────────────────────────
up() {
    mkdir -p "$RESULTS_DIR"

    # Per-GPU containers use the patched vLLM image by default (fixes the
    # EPLB scale-down + placement-group bugs that otherwise brick per-GPU
    # container scaling; see patches/ and analysis_report.html).
    # Operator can override by setting VLLM_IMAGE explicitly.
    if [ -z "${VLLM_IMAGE:-}" ] || [ "${VLLM_IMAGE}" = "xtrans-vllm-ep:v0.19.0" ]; then
        VLLM_IMAGE="$VLLM_IMAGE_PATCHED"
        export VLLM_IMAGE
        ensure_patched_image || return 1
    else
        docker_ensure_image "$VLLM_IMAGE" || return 1
    fi

    require_gpus_free || return 1

    ensure_network

    # Head
    launch_rank_container 0 \
        "ray start --head --port ${RAY_PORT} --num-gpus 1 \
             --node-ip-address ep-rank-0 \
             --min-worker-port 30000 --max-worker-port 39999 \
             --disable-usage-stats"

    log "Waiting for ray head to be ready..."
    for i in $(seq 1 20); do
        sleep 2
        if docker exec ep-rank-0 ray status > /dev/null 2>&1; then
            log "Ray head ready after $((i*2))s"
            break
        fi
    done

    # Workers
    local hip
    hip=$(head_ip)
    log "Ray head bridge IP: $hip"
    for i in 1 2 3; do
        launch_rank_container "$i" \
            "ray start --address ${hip}:${RAY_PORT} --num-gpus 1 \
                 --node-ip-address ep-rank-$i \
                 --min-worker-port 30000 --max-worker-port 39999 \
                 --disable-usage-stats"
    done

    log "Waiting for all Ray workers to register..."
    for i in $(seq 1 15); do
        sleep 2
        local ngpus
        ngpus=$(docker exec ep-rank-0 ray status 2>/dev/null \
                | grep -E '^\s*[0-9.]+/4.0 GPU$' | awk '{print $1}' | cut -d/ -f2 || true)
        if [ "${ngpus:-0}" = "4.0" ]; then
            log "4-GPU Ray cluster formed after $((i*2))s"
            break
        fi
    done
    docker exec ep-rank-0 ray status > "$RESULTS_DIR/ray_status.txt" 2>&1 || true

    # vllm serve inside ep-rank-0 -- background via bash -c so the exec
    # returns immediately; log file lives in the container at $SERVE_LOG_IN_CTN.
    # Default starts at DP=2 (matching the other regimes' 2->4->2 cycle);
    # override with PER_GPU_DP=<N> (e.g. 2, 4) to cold-start at a different
    # size. Extra serve flags (e.g. --gpu-memory-utilization 0.95) can be
    # passed via EXTRA_SERVE_ARGS.
    local dp_size=${PER_GPU_DP:-2}
    local extra=${EXTRA_SERVE_ARGS:-}
    log "Launching vllm serve in ep-rank-0 at DP=${dp_size} (extra=${extra:-none})"
    docker exec -d ep-rank-0 /bin/bash -c "
        export RAY_ADDRESS=ep-rank-0:${RAY_PORT}
        export VLLM_LOGGING_LEVEL=INFO
        vllm serve ${MODEL_PATH_IN_CTN} \
            --served-model-name ${SERVED_MODEL_NAME} \
            --host 0.0.0.0 --port ${VLLM_PORT} \
            --tensor-parallel-size 1 \
            --data-parallel-size ${dp_size} \
            --data-parallel-size-local 1 \
            --data-parallel-backend ray \
            --data-parallel-address ep-rank-0 \
            --enable-expert-parallel \
            --enable-elastic-ep \
            --enable-eplb \
            --all2all-backend allgather_reducescatter \
            --max-model-len 2048 \
            --max-num-seqs 16 \
            --gpu-memory-utilization 0.90 \
            --enforce-eager \
            --trust-remote-code \
            ${extra} \
            > $SERVE_LOG_IN_CTN 2>&1
    "

    # Give vllm serve ~15s to actually start before we check for its PID --
    # otherwise pgrep fires before the process exists and we falsely abort.
    sleep 15
    wait_for_ready "http://localhost:${VLLM_PORT}/health" 600 \
        per_gpu_containers_liveness per_gpu_containers_diag
}

down() {
    log "Saving per-container logs to $RESULTS_DIR/"
    for i in 0 1 2 3; do
        if docker ps -a --filter "name=ep-rank-$i" --format '{{.Names}}' \
                | grep -q "ep-rank-$i"; then
            docker logs "ep-rank-$i" > "$RESULTS_DIR/ep-rank-$i.log" 2>&1 || true
        fi
    done
    if docker exec ep-rank-0 test -f "$SERVE_LOG_IN_CTN" 2>/dev/null; then
        docker exec ep-rank-0 cat "$SERVE_LOG_IN_CTN" > "$RESULTS_DIR/vllm-serve.log" 2>/dev/null || true
    fi

    log "Stopping containers"
    for i in 0 1 2 3; do
        docker rm -f "ep-rank-$i" 2>/dev/null || true
    done
    log "Network $NETWORK left in place; remove with: docker network rm $NETWORK"
    sleep 3
    gpu_snapshot >&2
}

# ─── Subcommand: bench ────────────────────────────────────────────────
bench() {
    local label=${1:-unknown} num_prompts=${2:-32} concurrency=${3:-16}
    vllm_bench "$label" "localhost:$VLLM_PORT" "$num_prompts" "$concurrency" "$RESULTS_DIR"
}

# ─── Subcommand: scale ────────────────────────────────────────────────
# Use with the default 2->4->2 cycle: scale up from DP=2 first, which
# grows per-GPU expert count via EPLB (building redundancy), then scale
# down consumes that redundancy cleanly. Calling `scale 2` from a cold
# DP=N>2 start will hit vLLM's EPLB `num_redundant>=0` assertion because
# no redundancy has been pre-built; don't do that in this harness.
scale() {
    local new_dp=${1:?target DP size required}
    trigger_scale "http://localhost:$VLLM_PORT" "$new_dp" \
        | tee "$RESULTS_DIR/scale_to_dp${new_dp}_$(date +%H%M%S).log"
}

# ─── Subcommand: state ────────────────────────────────────────────────
state() {
    local tag=${1:-snapshot}
    {
        echo "=== Per-GPU containers state ($tag) at $(date) ==="
        echo ""
        echo "## Containers ##"
        docker ps --filter "name=ep-rank-" --format 'table {{.Names}}\t{{.Status}}'
        echo ""
        echo "## Per-container DeviceIDs (each owns one GPU!) ##"
        for i in 0 1 2 3; do
            echo "  ep-rank-$i: $(container_deviceids ep-rank-$i)"
        done
        echo ""
        echo "## Bridge network IPs ##"
        docker network inspect "$NETWORK" \
            --format '{{range .Containers}}{{.Name}} {{.IPv4Address}}
{{end}}' 2>/dev/null
        echo ""
        echo "## Host nvidia-smi ##"
        gpu_snapshot
        echo ""
        echo "## Scaling flag ##"
        curl -sS -X POST "http://localhost:$VLLM_PORT/is_scaling_elastic_ep" 2>&1 || echo "(unreachable)"
    } | tee "$RESULTS_DIR/state_${tag}.txt"
}

# ─── Subcommand: nccl-grep ────────────────────────────────────────────
# Pull the NCCL transport-selection lines out of the vllm serve log. The
# per-GPU-container headline finding is "NET/Socket/0" on every channel.
nccl_grep() {
    if ! docker exec ep-rank-0 test -f "$SERVE_LOG_IN_CTN" 2>/dev/null; then
        log "vllm-serve log not found in ep-rank-0"
        return 1
    fi
    docker exec ep-rank-0 grep -E \
        'NCCL INFO (Assigned NET plugin|Channel [0-9]+/[0-9]+ : |Check P2P Type|Connected all rings)' \
        "$SERVE_LOG_IN_CTN" \
        | tee "$RESULTS_DIR/nccl_transport.log" \
        | head -40
    echo ""
    echo "Full capture saved to $RESULTS_DIR/nccl_transport.log"
}

# ─── Subcommand: cycle ────────────────────────────────────────────────
# Full reference cycle matching native.sh / multi_gpu_container.sh:
# DP=2 baseline -> scale 4 -> bench -> scale 2 -> bench. Assumes `up`
# was called with PER_GPU_DP=2 (default) and R=0 -- see analysis_report
# §5.7 for why the 2->4->2 pattern works at R=0 (scale-up builds
# redundancy that scale-down consumes).
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
    nccl_grep > /dev/null 2>&1 || true  # capture NCCL transport evidence
    log "Cycle complete. Results in $RESULTS_DIR/"
}

# ─── Dispatch ─────────────────────────────────────────────────────────
cmd=${1:-}; shift || true
case "$cmd" in
    up)         up ;;
    down)       down ;;
    bench)      bench "$@" ;;
    scale)      scale "$@" ;;
    state)      state "$@" ;;
    cycle)      cycle ;;
    nccl-grep)  nccl_grep ;;
    *)
        cat <<'EOF' >&2
usage: per_gpu_containers.sh {up|down|bench LABEL [N] [C]|scale TARGET_DP|state [TAG]|cycle|nccl-grep}

subcommands:
    up                      bridge network + 4 per-GPU containers +
                            Ray cluster + vllm serve at DP=2
                            (override with PER_GPU_DP=<N>). Uses
                            the patched vLLM image by default;
                            auto-builds from Dockerfile.per_gpu_containers
                            on first run.
    down                    save per-container logs, stop containers
    bench LABEL [N] [C]     vllm bench from host
    scale TARGET_DP         POST /scale_elastic_ep. Use with the
                            default 2->4->2 pattern (scale up from
                            DP=2 first, then back down); cold-DP=N>2
                            scale-down hits vLLM's EPLB invariant.
    state [TAG]             per-container DeviceIDs + host nvidia-smi
    cycle                   full DP=2 → 4 → 2 reference cycle with benchmarks
                            (matches native.sh / multi_gpu_container.sh
                            cycle semantics). Assumes PER_GPU_DP=2 default.
    nccl-grep               extract NCCL transport selection from
                            vllm-serve log (the NET/Socket/0 headline)

prerequisites:
    * base image `xtrans-vllm-ep:v0.19.0` built (from Dockerfile.base)
    * Qwen3-30B-A3B at /data/models--Qwen--Qwen3-30B-A3B/
    * all 4 GPUs free on the host
EOF
        exit 1
        ;;
esac
