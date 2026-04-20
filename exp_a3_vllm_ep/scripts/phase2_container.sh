#!/usr/bin/env bash
# Phase 2: vLLM Elastic EP inside a Multi-GPU Container
#
# Run vLLM Elastic EP in a single Docker container with --gpus all (visible to
# GPUs 0-3). Run the same 2->4->2 cycle as Phase 1, but from a containerized
# context. Key measurements:
#   1. Does elastic EP work identically inside a container?
#   2. After scale-down 4->2, does the container still claim all 4 GPUs?
#   3. Can a second workload (from the host) claim GPUs 2-3?
#
# Usage:
#   ./scripts/phase2_container.sh start      # launch container + serving
#   ./scripts/phase2_container.sh bench N    # benchmark at DP=N
#   ./scripts/phase2_container.sh scale-up   # 2 -> 4
#   ./scripts/phase2_container.sh scale-down # 4 -> 2
#   ./scripts/phase2_container.sh host-view  # snapshot Docker + host state
#   ./scripts/phase2_container.sh stop       # clean up

set -euo pipefail
cd "$(dirname "$0")/.."

# ─── Config ───────────────────────────────────────────────────────────
CONTAINER_NAME="xtrans-exp-a3-phase2"
IMAGE="vllm/vllm-openai:v0.19.0"
MODEL_HOST="/data/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
MODEL_IN_CTN="/models/qwen3-30b-a3b"
PORT=8000
INITIAL_DP=2
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=16
GPU_UTIL=0.90
RESULTS_DIR="results/phase2"
mkdir -p "$RESULTS_DIR"

# ─── Subcommands ──────────────────────────────────────────────────────

start() {
    echo "=== Phase 2: Launching container ==="
    # Verify image is available
    if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
        echo "ERROR: image $IMAGE not found. Pull it first."
        exit 1
    fi

    # Verify model exists
    if [ ! -d "$MODEL_HOST" ]; then
        echo "ERROR: model dir $MODEL_HOST not found."
        exit 1
    fi

    # Clean up any previous container
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    # Record the host's view before we start
    host_view "before-container" > /dev/null

    # Launch container with all 4 GPUs, model mounted, port exposed
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus '"device=0,1,2,3"' \
        --ipc=host \
        --shm-size=16g \
        -v "$MODEL_HOST:$MODEL_IN_CTN:ro" \
        -p "$PORT:8000" \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e VLLM_LOGGING_LEVEL=INFO \
        --entrypoint /bin/bash \
        "$IMAGE" \
        -c "vllm serve $MODEL_IN_CTN \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 1 \
            --data-parallel-size $INITIAL_DP \
            --data-parallel-backend ray \
            --enable-expert-parallel \
            --enable-elastic-ep \
            --enable-eplb \
            --all2all-backend allgather_reducescatter \
            --max-model-len $MAX_MODEL_LEN \
            --max-num-seqs $MAX_NUM_SEQS \
            --gpu-memory-utilization $GPU_UTIL \
            --enforce-eager \
            --trust-remote-code"

    echo "Container $CONTAINER_NAME started. Waiting for vLLM to be ready..."
    local start_ts=$(date +%s)
    for i in $(seq 1 60); do
        sleep 10
        if curl -s --max-time 2 "http://localhost:$PORT/health" > /dev/null 2>&1; then
            local end_ts=$(date +%s)
            echo "READY after $((end_ts - start_ts))s"
            host_view "after-ready" > "$RESULTS_DIR/host_view_after_ready.txt"
            return 0
        fi
        echo "  loading... ($((i*10))s)"
    done
    echo "TIMEOUT waiting for vLLM ready; container logs:"
    docker logs --tail 40 "$CONTAINER_NAME"
    exit 1
}

bench() {
    local label="${1:-unknown}"
    local n_req="${2:-16}"
    local workers="${3:-8}"
    echo "=== Bench: $label (n=$n_req) ==="

    python3 - <<PYEOF 2>&1 | tee "$RESULTS_DIR/bench_${label}.log"
import requests, time, concurrent.futures, json

URL = 'http://localhost:$PORT/v1/completions'
MODEL = '$MODEL_IN_CTN'

def send(i):
    start = time.time()
    try:
        r = requests.post(URL, json={
            'model': MODEL,
            'prompt': f'Request {i}: Please write a paragraph about parallel computing. ',
            'max_tokens': 128,
            'temperature': 0.7,
        }, timeout=180)
        elapsed = time.time() - start
        if r.status_code != 200:
            return {'id': i, 'status': r.status_code, 'elapsed': round(elapsed, 3), 'error': r.text[:100]}
        tok = r.json().get('usage', {}).get('completion_tokens', 0)
        return {'id': i, 'status': 200, 'elapsed': round(elapsed, 3), 'tokens': tok}
    except Exception as e:
        return {'id': i, 'status': -1, 'elapsed': round(time.time()-start, 3), 'error': str(e)[:100]}

# Warmup
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(send, range(1000, 1002)))

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=$workers) as ex:
    results = list(ex.map(send, range($n_req)))
total = time.time() - start
ok = [r for r in results if r['status'] == 200]
tok = sum(r['tokens'] for r in ok)
out = {
    'label': '$label',
    'n_requests': $n_req, 'ok': len(ok), 'errors': $n_req - len(ok),
    'total_time_s': round(total, 3),
    'total_tokens': tok,
    'throughput_tok_s': round(tok/total, 1) if total > 0 else 0,
    'avg_latency_s': round(sum(r['elapsed'] for r in ok)/max(len(ok),1), 3),
}
print(f"Throughput: {out['throughput_tok_s']} tok/s, lat: {out['avg_latency_s']}s, OK: {len(ok)}/$n_req")
with open('$RESULTS_DIR/bench_${label}.json', 'w') as f:
    json.dump(out, f, indent=2)
PYEOF
}

scale() {
    local target="$1"
    local label="${2:-scale-to-$target}"
    echo "=== Scale to DP=$target ($label) ==="
    python3 - <<PYEOF 2>&1 | tee "$RESULTS_DIR/${label}.log"
import requests, time, json, subprocess

def gpus():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip().split('\n')
    return [g.strip() for g in out]

pre = gpus()
print(f"Pre-scale GPU memory (host): {pre}")
start = time.time()
r = requests.post('http://localhost:$PORT/scale_elastic_ep',
    json={'new_data_parallel_size': $target, 'drain_timeout': 60}, timeout=900)
elapsed = time.time() - start
mid = gpus()
print(f"HTTP {r.status_code} in {elapsed:.2f}s")
print(f"Post-scale GPU memory (host, immediate): {mid}")
time.sleep(12)
late = gpus()
print(f"Post-scale GPU memory (host, +12s): {late}")
if r.status_code != 200:
    print(f"Body: {r.text[:300]}")
json.dump({
    'operation': '$label', 'target_dp': $target, 'http_code': r.status_code,
    'elapsed_s': round(elapsed, 2),
    'gpu_mem_pre': pre, 'gpu_mem_immediate': mid, 'gpu_mem_plus12s': late,
}, open('$RESULTS_DIR/${label}.json', 'w'), indent=2)
PYEOF
}

host_view() {
    local tag="${1:-snapshot}"
    echo "=== Host view ($tag) at $(date) ==="
    echo ""
    echo "## docker ps ##"
    docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "## docker inspect (HostConfig.DeviceRequests) ##"
    docker inspect "$CONTAINER_NAME" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Container not running."
    echo ""
    echo "## nvidia-smi (host) ##"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    echo ""
    echo "## Processes on GPUs (host view) ##"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader 2>/dev/null | head -20
    echo ""
}

stop() {
    echo "=== Stopping container ==="
    docker logs "$CONTAINER_NAME" > "$RESULTS_DIR/container.log" 2>&1 || true
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    echo "Stopped."
}

logs() {
    docker logs --tail "${1:-50}" "$CONTAINER_NAME"
}

case "${1:-}" in
    start) start ;;
    bench) bench "${2:-unknown}" "${3:-16}" "${4:-8}" ;;
    scale) scale "${2:?target DP}" "${3:-scale-to-$2}" ;;
    scale-up) scale 4 "scale_up_2_to_4" ;;
    scale-down) scale 2 "scale_down_4_to_2" ;;
    host-view) host_view "${2:-snapshot}" ;;
    logs) logs "${2:-50}" ;;
    stop) stop ;;
    *)
        echo "Usage: $0 {start|bench LABEL [N] [W]|scale TARGET [LABEL]|scale-up|scale-down|host-view [TAG]|logs [N]|stop}"
        exit 1
        ;;
esac
