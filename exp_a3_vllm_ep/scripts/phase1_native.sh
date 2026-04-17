#!/usr/bin/env bash
# Phase 1: vLLM Elastic EP — Native Baseline
#
# Start vLLM with Elastic EP on 4 GPUs using Qwen3-30B-A3B (MoE, 128 experts).
# EP=4 (TP=1, DP=4), so each GPU gets 32 experts.
#
# Requires: vLLM 0.17+, Ray, Qwen3-30B-A3B downloaded to /data
#
# Usage:
#   ./scripts/phase1_native.sh              # start serving
#   ./scripts/phase1_native.sh scale-down   # 4→2 DP workers
#   ./scripts/phase1_native.sh scale-up     # 2→4 DP workers
#   ./scripts/phase1_native.sh bench        # run benchmark

set -euo pipefail
cd "$(dirname "$0")/.."

# ─── Config ───────────────────────────────────────────────────────────
MODEL_PATH="/data/models--Qwen--Qwen3-30B-A3B/snapshots"
MODEL_ID="Qwen/Qwen3-30B-A3B"
PORT=8000
HOST="0.0.0.0"
INITIAL_DP=4
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=32
GPU_UTIL=0.85

# Resolve snapshot path (huggingface cache layout)
if [ -d "$MODEL_PATH" ]; then
    SNAPSHOT=$(ls -1 "$MODEL_PATH" | head -1)
    MODEL="$MODEL_PATH/$SNAPSHOT"
else
    # Try using model ID directly (if symlinked or in HF cache)
    MODEL="$MODEL_ID"
fi

VENV="/home/thd/repositories/xtrans-experiments/.venv"
RESULTS_DIR="results/phase1"
mkdir -p "$RESULTS_DIR"

activate_venv() {
    source "$VENV/bin/activate"
}

# ─── Subcommands ──────────────────────────────────────────────────────

start_serving() {
    echo "=== Phase 1: Starting vLLM Elastic EP ==="
    echo "Model: $MODEL"
    echo "DP=$INITIAL_DP, TP=1, EP=$INITIAL_DP (auto)"
    echo "Port: $PORT"
    echo ""

    activate_venv

    # Mixed GPU types: use PCI_BUS_ID ordering for consistency
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    vllm serve "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --data-parallel-size "$INITIAL_DP" \
        --data-parallel-backend ray \
        --enable-expert-parallel \
        --enable-elastic-ep \
        --enable-eplb \
        --eplb-config '{"num_redundant_experts": 0}' \
        --all2all-backend allgather_reducescatter \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --enforce-eager \
        --trust-remote-code \
        2>&1 | tee "$RESULTS_DIR/serve.log"
}

scale_down() {
    local new_dp=${1:-2}
    echo "=== Triggering scale-down: DP=$INITIAL_DP → DP=$new_dp ==="
    activate_venv

    local start_time=$(date +%s%N)
    response=$(curl -s -w "\n%{http_code}" -X POST \
        "http://localhost:$PORT/scale_elastic_ep" \
        -H "Content-Type: application/json" \
        -d "{\"new_data_parallel_size\": $new_dp, \"drain_timeout\": 120}" \
        --max-time 300)

    local end_time=$(date +%s%N)
    local http_code=$(echo "$response" | tail -1)
    local body=$(echo "$response" | head -n -1)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    echo "Response ($http_code): $body"
    echo "Rescaling time: ${elapsed_ms}ms"

    # Record result
    echo "{\"operation\": \"scale_down\", \"from_dp\": $INITIAL_DP, \"to_dp\": $new_dp, \"http_code\": $http_code, \"elapsed_ms\": $elapsed_ms, \"timestamp\": \"$(date -Iseconds)\"}" \
        >> "$RESULTS_DIR/scaling_events.jsonl"
}

scale_up() {
    local new_dp=${1:-4}
    echo "=== Triggering scale-up: → DP=$new_dp ==="
    activate_venv

    local start_time=$(date +%s%N)
    response=$(curl -s -w "\n%{http_code}" -X POST \
        "http://localhost:$PORT/scale_elastic_ep" \
        -H "Content-Type: application/json" \
        -d "{\"new_data_parallel_size\": $new_dp, \"drain_timeout\": 120}" \
        --max-time 300)

    local end_time=$(date +%s%N)
    local http_code=$(echo "$response" | tail -1)
    local body=$(echo "$response" | head -n -1)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    echo "Response ($http_code): $body"
    echo "Rescaling time: ${elapsed_ms}ms"

    echo "{\"operation\": \"scale_up\", \"to_dp\": $new_dp, \"http_code\": $http_code, \"elapsed_ms\": $elapsed_ms, \"timestamp\": \"$(date -Iseconds)\"}" \
        >> "$RESULTS_DIR/scaling_events.jsonl"
}

bench() {
    echo "=== Running benchmark ==="
    activate_venv

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local dp_size=${1:-unknown}

    # Simple throughput test: send N concurrent requests
    python3 -c "
import requests
import json
import time
import concurrent.futures

URL = 'http://localhost:$PORT/v1/completions'
HEADERS = {'Content-Type': 'application/json'}
N_REQUESTS = 20
MAX_TOKENS = 128

def send_request(i):
    payload = {
        'model': '$MODEL_ID',
        'prompt': f'Write a short paragraph about topic {i}: the future of cloud computing.',
        'max_tokens': MAX_TOKENS,
        'temperature': 0.7,
    }
    start = time.time()
    resp = requests.post(URL, json=payload, headers=HEADERS, timeout=120)
    elapsed = time.time() - start
    data = resp.json()
    tokens = data.get('usage', {}).get('completion_tokens', 0)
    return {'request_id': i, 'status': resp.status_code, 'elapsed_s': elapsed, 'tokens': tokens}

print(f'Sending {N_REQUESTS} requests to {URL}...')
overall_start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    results = list(pool.map(send_request, range(N_REQUESTS)))

overall_elapsed = time.time() - overall_start
total_tokens = sum(r['tokens'] for r in results)
avg_latency = sum(r['elapsed_s'] for r in results) / len(results)

print(f'Total time: {overall_elapsed:.2f}s')
print(f'Total tokens: {total_tokens}')
print(f'Throughput: {total_tokens / overall_elapsed:.1f} tokens/s')
print(f'Avg latency: {avg_latency:.2f}s')

output = {
    'timestamp': '$timestamp',
    'dp_size': '$dp_size',
    'n_requests': N_REQUESTS,
    'total_time_s': round(overall_elapsed, 3),
    'total_tokens': total_tokens,
    'throughput_tok_s': round(total_tokens / overall_elapsed, 1),
    'avg_latency_s': round(avg_latency, 3),
    'results': results,
}
with open('$RESULTS_DIR/bench_dp${dp_size}_${timestamp}.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'Results saved to $RESULTS_DIR/bench_dp${dp_size}_${timestamp}.json')
" 2>&1 | tee -a "$RESULTS_DIR/bench.log"
}

check_scaling_status() {
    curl -s "http://localhost:$PORT/is_scaling_elastic_ep" | python3 -m json.tool
}

# ─── Main ─────────────────────────────────────────────────────────────

case "${1:-serve}" in
    serve|start)
        start_serving
        ;;
    scale-down|down)
        scale_down "${2:-2}"
        ;;
    scale-up|up)
        scale_up "${2:-4}"
        ;;
    bench|benchmark)
        bench "${2:-unknown}"
        ;;
    status)
        check_scaling_status
        ;;
    *)
        echo "Usage: $0 {serve|scale-down [N]|scale-up [N]|bench [dp_size]|status}"
        exit 1
        ;;
esac
