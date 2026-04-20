#!/usr/bin/env bash
# Phase 3: One EP worker per container.
#
# Goal: 4 Docker containers, each --gpus device=N (N=0..3), joined as
# a single Ray cluster, serving one vLLM Elastic EP cluster at DP=4.
# Scaling = container lifecycle instead of --scale_elastic_ep.
#
# EXPECTED OUTCOME: friction. Problems we expect to hit, any of which
# counts as a research finding:
#   - Ray cluster formation failures (network, port, GPU discovery)
#   - vLLM spawning actors across nodes (model-path visibility, rank
#     assignment when every container sees GPU "0")
#   - NCCL transport selection (hostHash collision or miss, shm gate,
#     IPC socket in namespaced abstract UDS)
#   - Performance collapse to TCP/SHM-over-network
#
# Usage:
#   ./scripts/phase3_per_gpu.sh network      # create bridge network
#   ./scripts/phase3_per_gpu.sh ray-head     # container 0: ray head
#   ./scripts/phase3_per_gpu.sh ray-workers  # containers 1-3: ray workers
#   ./scripts/phase3_per_gpu.sh ray-status   # inspect ray cluster
#   ./scripts/phase3_per_gpu.sh serve        # exec vllm serve in head
#   ./scripts/phase3_per_gpu.sh logs N       # container N's logs
#   ./scripts/phase3_per_gpu.sh exec N CMD   # exec cmd in container N
#   ./scripts/phase3_per_gpu.sh host-view    # snapshot host state
#   ./scripts/phase3_per_gpu.sh teardown     # stop everything

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="xtrans-vllm-ep:v0.19.0"
NETWORK="xtrans-phase3"
MODEL_HOST="/data/models--Qwen--Qwen3-30B-A3B"
MODEL_IN_CTN_MOUNT="/models/qwen3-30b-a3b"
MODEL_PATH_IN_CTN="/models/qwen3-30b-a3b/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
RAY_PORT=26379
VLLM_PORT=8000
RESULTS_DIR="results/phase3"
mkdir -p "$RESULTS_DIR"

ensure_network() {
    if ! docker network inspect "$NETWORK" > /dev/null 2>&1; then
        echo "Creating docker network $NETWORK"
        docker network create "$NETWORK"
    else
        echo "Network $NETWORK exists"
    fi
}

ray_head() {
    ensure_network
    docker rm -f "ep-rank-0" 2>/dev/null || true

    echo "Starting ep-rank-0 (Ray head + vLLM API server) on GPU 0"
    docker run -d \
        --name "ep-rank-0" \
        --hostname "ep-rank-0" \
        --network "$NETWORK" \
        --gpus '"device=0"' \
        --ipc=host \
        --shm-size=16g \
        -p "${VLLM_PORT}:${VLLM_PORT}" \
        -p "${RAY_PORT}:${RAY_PORT}" \
        -v "$MODEL_HOST:$MODEL_IN_CTN_MOUNT:ro" \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e VLLM_LOGGING_LEVEL=INFO \
        -e NCCL_DEBUG=INFO \
        --entrypoint /bin/bash \
        "$IMAGE" \
        -c "ray start --head --port ${RAY_PORT} --num-gpus 1 \
                --node-ip-address ep-rank-0 \
                --min-worker-port 30000 --max-worker-port 39999 \
                --disable-usage-stats && \
            # Keep container alive so we can exec vllm serve into it
            tail -f /dev/null"

    echo "Waiting for ray head to be ready..."
    for i in $(seq 1 20); do
        sleep 2
        if docker exec ep-rank-0 ray status > /dev/null 2>&1; then
            echo "Ray head ready (after $((i*2))s)"
            return 0
        fi
    done
    echo "Ray head not ready after 40s"
    docker logs --tail 30 ep-rank-0
    return 1
}

ray_workers() {
    local head_ip
    head_ip=$(docker inspect ep-rank-0 --format "{{(index .NetworkSettings.Networks \"$NETWORK\").IPAddress}}")
    echo "Ray head IP on $NETWORK: $head_ip"

    for i in 1 2 3; do
        docker rm -f "ep-rank-$i" 2>/dev/null || true
        echo "Starting ep-rank-$i (Ray worker) on GPU $i"
        docker run -d \
            --name "ep-rank-$i" \
            --hostname "ep-rank-$i" \
            --network "$NETWORK" \
            --gpus "\"device=$i\"" \
            --ipc=host \
            --shm-size=16g \
            -v "$MODEL_HOST:$MODEL_IN_CTN_MOUNT:ro" \
            -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
            -e NCCL_DEBUG=INFO \
            --entrypoint /bin/bash \
            "$IMAGE" \
            -c "ray start --address ${head_ip}:${RAY_PORT} --num-gpus 1 \
                    --node-ip-address ep-rank-$i \
                    --min-worker-port 30000 --max-worker-port 39999 \
                    --disable-usage-stats && \
                tail -f /dev/null"
    done

    echo "Waiting for all workers to join..."
    sleep 5
    docker exec ep-rank-0 ray status 2>&1 | head -30
}

serve() {
    # Launch vllm serve inside the head container
    echo "Starting vllm serve inside ep-rank-0 at DP=4..."
    docker exec -d ep-rank-0 /bin/bash -c "
        export RAY_ADDRESS=ep-rank-0:${RAY_PORT}
        export VLLM_LOGGING_LEVEL=INFO
        vllm serve ${MODEL_PATH_IN_CTN} \
            --host 0.0.0.0 \
            --port ${VLLM_PORT} \
            --tensor-parallel-size 1 \
            --data-parallel-size 4 \
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
            > /tmp/vllm-serve.log 2>&1
    "
    echo "vllm serve launched (backgrounded in container). Monitor via:"
    echo "    docker exec ep-rank-0 tail -f /tmp/vllm-serve.log"
}

wait_ready() {
    local timeout=${1:-600}
    local start=$(date +%s)
    echo "Polling http://localhost:${VLLM_PORT}/health (timeout ${timeout}s)..."
    while true; do
        if curl -s --max-time 2 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            local end=$(date +%s)
            echo "READY after $((end - start))s"
            return 0
        fi
        local now=$(date +%s)
        if (( now - start >= timeout )); then
            echo "TIMEOUT after ${timeout}s"
            return 1
        fi
        sleep 5
    done
}

ray_status() {
    docker exec ep-rank-0 ray status 2>&1
}

host_view() {
    local tag="${1:-snapshot}"
    {
        echo "=== Host view ($tag) at $(date) ==="
        echo ""
        echo "## Containers ##"
        docker ps --filter "name=ep-rank-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        echo "## Docker network ##"
        docker network inspect "$NETWORK" --format '{{range .Containers}}{{.Name}} {{.IPv4Address}}
{{end}}' 2>/dev/null
        echo ""
        echo "## nvidia-smi (host) ##"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        echo ""
        echo "## GPU processes ##"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -20
    }
}

teardown() {
    echo "=== Saving container logs ==="
    for i in 0 1 2 3; do
        if docker ps -a --filter "name=ep-rank-$i" --format "{{.Names}}" | grep -q "ep-rank-$i"; then
            docker logs "ep-rank-$i" > "$RESULTS_DIR/ep-rank-$i.log" 2>&1 || true
            if [ "$i" = "0" ] && docker exec ep-rank-0 test -f /tmp/vllm-serve.log 2>/dev/null; then
                docker exec ep-rank-0 cat /tmp/vllm-serve.log > "$RESULTS_DIR/vllm-serve.log" 2>/dev/null || true
            fi
        fi
    done
    echo "=== Stopping containers ==="
    for i in 0 1 2 3; do
        docker rm -f "ep-rank-$i" 2>/dev/null || true
    done
    echo "(Keeping docker network $NETWORK — remove with: docker network rm $NETWORK)"
}

case "${1:-}" in
    network)    ensure_network ;;
    ray-head)   ray_head ;;
    ray-workers) ray_workers ;;
    ray-status) ray_status ;;
    serve)      serve ;;
    wait-ready) wait_ready "${2:-600}" ;;
    logs)       docker logs --tail "${3:-80}" "ep-rank-${2:-0}" ;;
    exec)       shift; i="$1"; shift; docker exec "ep-rank-$i" "$@" ;;
    host-view)  host_view "${2:-snapshot}" ;;
    teardown)   teardown ;;
    *)
        echo "Usage: $0 {network|ray-head|ray-workers|ray-status|serve|wait-ready [T]|logs N [LINES]|exec N CMD|host-view|teardown}"
        exit 1
        ;;
esac
