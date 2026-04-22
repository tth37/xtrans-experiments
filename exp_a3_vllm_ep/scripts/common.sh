# shellcheck shell=bash
# Shared helpers for Exp A3 phase scripts (phase1_native.sh, phase2_container.sh,
# phase3_per_gpu.sh). Source this file; do not execute it.
#
# Exported vars (may be overridden by caller before sourcing):
#   PROJECT_ROOT        absolute path to the xtrans-experiments checkout
#   VENV_DIR            project venv (.venv/ at project root)
#   MODEL_HOST          HF cache dir on host (for mounting)
#   MODEL_SNAPSHOT      resolved model path (host-side)
#   MODEL_PATH_IN_CTN   model path as seen inside container (phase 2/3 only)
#   VLLM_IMAGE          docker image tag with vllm + ray
#   RAY_PORT            port used for our dedicated Ray head
#   VLLM_PORT           serving port
#
# Exposed functions:
#   log MSG                      timestamped stderr log
#   ensure_venv                  activate .venv
#   wait_for_ready URL [T]       poll URL until HTTP 200, timeout T seconds
#   vllm_bench LABEL HOST:PORT NUM_PROMPTS CONCURRENCY OUT_DIR
#                                run `vllm bench serve` with --dataset-name random
#   trigger_scale URL NEW_DP     POST /scale_elastic_ep, print elapsed
#   gpu_snapshot                 nvidia-smi brief table
#   gpu_processes                nvidia-smi compute apps
#   container_deviceids NAME     docker inspect HostConfig.DeviceRequests.DeviceIDs
#   tmux_oneshot NAME CMD        start CMD in detached tmux session NAME
#   tmux_kill NAME               kill tmux session if it exists
#   docker_ensure_image IMAGE    assert image is present locally
#   all_gpus_free                returns 0 if all 4 GPUs under 100 MiB used
#
# Everything is idempotent and safe to source repeatedly.

# ─── Defaults (override by setting before sourcing) ────────────────────
: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${VENV_DIR:=$PROJECT_ROOT/.venv}"
: "${MODEL_HOST:=/data/models--Qwen--Qwen3-30B-A3B}"
: "${MODEL_SNAPSHOT:=$MODEL_HOST/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
: "${MODEL_PATH_IN_CTN:=/models/qwen3-30b-a3b/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
# Registered server name -- must be identical across phases so `vllm bench
# serve --model $SERVED_MODEL_NAME` hits the right endpoint regardless of
# whether the server is native (uses host paths) or containerised (uses
# in-container paths).
: "${SERVED_MODEL_NAME:=qwen3-30b-a3b}"
: "${VLLM_IMAGE:=xtrans-vllm-ep:v0.19.0}"
# Phase 3 uses the patched image by default (see Dockerfile.phase3);
# auto-built on first use by ensure_patched_image().
: "${VLLM_IMAGE_PATCHED:=xtrans-vllm-ep-patched:v0.19.0}"
: "${RAY_PORT:=26379}"
: "${VLLM_PORT:=8000}"

# ─── Logging ──────────────────────────────────────────────────────────
log() {
    printf '[%(%H:%M:%S)T] %s\n' -1 "$*" >&2
}

# ─── Environment setup ────────────────────────────────────────────────
ensure_venv() {
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
}

# ─── Serving readiness ────────────────────────────────────────────────
# Polls URL until HTTP 200 OR the liveness callback fails OR timeout.
#
# Usage: wait_for_ready URL [TIMEOUT] [LIVENESS_CHECK] [DIAG_ON_FAIL]
#
#   URL             health endpoint (GET, expects HTTP 200)
#   TIMEOUT         seconds before giving up (default 600)
#   LIVENESS_CHECK  optional function name. Called every poll. If it exits
#                   non-zero, wait_for_ready aborts early -- the underlying
#                   server process is already dead, waiting longer is useless.
#                   Should be cheap (< 1s).
#   DIAG_ON_FAIL    optional function name. Called once on failure (timeout
#                   or dead liveness) to dump helpful diagnostics (logs,
#                   container state, etc). Runs AFTER the failure is logged.
#
# Progress is printed every 30s so the operator sees that we're still
# polling and haven't silently hung.
wait_for_ready() {
    local url=$1
    local timeout=${2:-600}
    local liveness_check=${3:-}
    local diag_on_fail=${4:-}
    local start now elapsed last_progress=0
    start=$(date +%s)
    log "Polling $url (timeout ${timeout}s)"
    while :; do
        if curl -fsS --max-time 2 "$url" > /dev/null 2>&1; then
            now=$(date +%s); elapsed=$((now - start))
            log "READY after ${elapsed}s"
            return 0
        fi
        # Liveness check -- is the backing server actually alive?
        if [ -n "$liveness_check" ] && ! "$liveness_check"; then
            now=$(date +%s); elapsed=$((now - start))
            log "ABORT: server process dead after ${elapsed}s (liveness check failed)"
            [ -n "$diag_on_fail" ] && "$diag_on_fail"
            return 2
        fi
        now=$(date +%s); elapsed=$((now - start))
        if (( elapsed >= timeout )); then
            log "TIMEOUT after ${timeout}s"
            [ -n "$diag_on_fail" ] && "$diag_on_fail"
            return 1
        fi
        # Progress ping every 30s
        if (( elapsed - last_progress >= 30 )); then
            log "  ...still waiting (${elapsed}s elapsed)"
            last_progress=$elapsed
        fi
        sleep 5
    done
}

# ─── Benchmarking ─────────────────────────────────────────────────────
# Uses `vllm bench serve --dataset-name random` — produces JSON with TTFT/TPOT/
# throughput percentiles in OUT_DIR.
#
# Usage: vllm_bench LABEL HOST:PORT NUM_PROMPTS MAX_CONCURRENCY OUT_DIR
vllm_bench() {
    local label=$1 hostport=$2 num_prompts=$3 concurrency=$4 out_dir=$5
    local host port
    host=${hostport%:*}
    port=${hostport#*:}
    mkdir -p "$out_dir"

    ensure_venv
    log "Bench[$label]: n=$num_prompts, concurrency=$concurrency"
    # --random-input-len / --random-output-len give reproducible shape.
    # --save-result writes the full JSON into OUT_DIR.
    # --model is the name the bench client uses in the request payload; must
    # match what the server registered. --tokenizer loads locally from the
    # host filesystem for tokenisation during request generation.
    vllm bench serve \
        --backend vllm \
        --model "$SERVED_MODEL_NAME" \
        --tokenizer "$MODEL_SNAPSHOT" \
        --host "$host" \
        --port "$port" \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len 128 \
        --random-output-len 128 \
        --num-prompts "$num_prompts" \
        --max-concurrency "$concurrency" \
        --seed 0 \
        --save-result \
        --result-dir "$out_dir" \
        --result-filename "bench_${label}.json" \
        2>&1 | tee "$out_dir/bench_${label}.log"
}

# ─── /scale_elastic_ep driver ─────────────────────────────────────────
# Usage: trigger_scale http://localhost:8000 4 [drain_timeout]
trigger_scale() {
    local base_url=$1 new_dp=$2 drain_timeout=${3:-60}
    local start end elapsed http_code body
    start=$(date +%s%N)
    log "Scale to DP=$new_dp (drain_timeout=${drain_timeout}s)"
    # --max-time 900 allows up to 15min; the scale itself is usually < 60s.
    local resp
    resp=$(curl -sS -w '\nHTTP_CODE=%{http_code}\n' \
        --max-time 900 \
        -X POST "$base_url/scale_elastic_ep" \
        -H 'Content-Type: application/json' \
        -d "{\"new_data_parallel_size\": $new_dp, \"drain_timeout\": $drain_timeout}")
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))  # ms
    http_code=$(echo "$resp" | awk -F= '/^HTTP_CODE=/{print $2}')
    body=$(echo "$resp" | sed '/^HTTP_CODE=/d')
    log "Scale response: HTTP ${http_code:-?} in ${elapsed}ms"
    printf '%s\n' "$body"
    echo "ELAPSED_MS=$elapsed"
    echo "HTTP_CODE=${http_code:-unknown}"
    [ "$http_code" = "200" ]
}

# ─── Host-side observations ───────────────────────────────────────────
gpu_snapshot() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader 2>/dev/null
}

gpu_processes() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid \
        --format=csv,noheader 2>/dev/null | head -20
}

# Usage: container_deviceids ep-rank-0
container_deviceids() {
    docker inspect "$1" \
        --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null \
        | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    if d:
        print(d[0].get("DeviceIDs") or [])
    else:
        print([])
except Exception:
    print("(unparseable)")
' 2>/dev/null || echo "(inspect failed)"
}

# ─── tmux helpers ─────────────────────────────────────────────────────
# Use tmux to background long-running processes with persistent output
# capture. CMD is run inside the project root.
# Usage: tmux_oneshot name "CMD..."
tmux_oneshot() {
    local name=$1; shift
    tmux_kill "$name"
    tmux new-session -d -s "$name" -c "$PROJECT_ROOT" "$@"
}

tmux_kill() {
    tmux kill-session -t "$1" 2>/dev/null || true
}

# ─── Docker helpers ───────────────────────────────────────────────────
docker_ensure_image() {
    local image=$1
    if ! docker image inspect "$image" > /dev/null 2>&1; then
        log "ERROR: image $image not found locally"
        return 1
    fi
}

# Ensure VLLM_IMAGE_PATCHED exists locally; build it if not.
# Used by phase3_per_gpu.sh to auto-build the patched image on first run.
# Requires xtrans-vllm-ep:v0.19.0 (the base, built from Dockerfile.phase2)
# to already exist.
ensure_patched_image() {
    if docker image inspect "$VLLM_IMAGE_PATCHED" > /dev/null 2>&1; then
        return 0
    fi
    if ! docker image inspect "xtrans-vllm-ep:v0.19.0" > /dev/null 2>&1; then
        log "ERROR: base image xtrans-vllm-ep:v0.19.0 not found; build it first:"
        log "    docker build -t xtrans-vllm-ep:v0.19.0 \\"
        log "        -f $PROJECT_ROOT/exp_a3_vllm_ep/Dockerfile.phase2 \\"
        log "        $PROJECT_ROOT/exp_a3_vllm_ep"
        return 1
    fi
    log "Patched image $VLLM_IMAGE_PATCHED not found; building from Dockerfile.phase3..."
    docker build \
        -t "$VLLM_IMAGE_PATCHED" \
        -f "$PROJECT_ROOT/exp_a3_vllm_ep/Dockerfile.phase3" \
        "$PROJECT_ROOT/exp_a3_vllm_ep" \
        || { log "ERROR: build failed"; return 1; }
    log "Built $VLLM_IMAGE_PATCHED"
}

# Returns 0 (success) if all 4 GPUs look idle (memory used < 100 MiB each).
all_gpus_free() {
    local used_list
    used_list=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -n "$used_list" ] || return 1
    while IFS= read -r used; do
        if [ "$used" -ge 100 ]; then
            return 1
        fi
    done <<< "$used_list"
    return 0
}

# Abort unless all 4 GPUs are free. Set ALLOW_BUSY_GPUS=1 to override.
# Prints a clear message either way so the operator knows why we aborted.
require_gpus_free() {
    if all_gpus_free; then
        return 0
    fi
    log "Some GPUs are busy:"
    gpu_snapshot | sed 's/^/    /' >&2
    if [ "${ALLOW_BUSY_GPUS:-0}" = "1" ]; then
        log "ALLOW_BUSY_GPUS=1 set, continuing anyway"
        return 0
    fi
    log "Aborting. Set ALLOW_BUSY_GPUS=1 to override (at your own risk)."
    return 1
}
