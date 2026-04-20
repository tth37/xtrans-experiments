#!/usr/bin/env bash
# Exp A3 Phase 1: vLLM Elastic EP running natively on bare metal.
#
# Native reference for all subsequent phases. Uses a dedicated Ray head on
# port ${RAY_PORT} to avoid conflicts with other Ray instances on the host.
#
# Starts vLLM at DP=2 (required to make scale-down work under the
# num_redundant_experts=0 EPLB invariant -- scaling up first builds the
# redundancy that scale-down later consumes).
#
# Usage:
#   ./scripts/phase1_native.sh start          # bring up Ray + vllm serve
#   ./scripts/phase1_native.sh bench LABEL NP C
#                                             # run vllm bench serve
#   ./scripts/phase1_native.sh scale NEW_DP   # trigger /scale_elastic_ep
#   ./scripts/phase1_native.sh state          # snapshot GPU + serving state
#   ./scripts/phase1_native.sh cycle          # full 2->4->2 reference cycle
#   ./scripts/phase1_native.sh stop           # shut everything down

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

RESULTS_DIR="$PROJECT_ROOT/exp_a3_vllm_ep/results/phase1"
SERVE_SESSION="a3-phase1-serve"
SERVE_LOG="$RESULTS_DIR/serve.log"

# ─── Server lifecycle ─────────────────────────────────────────────────
start() {
    ensure_venv
    mkdir -p "$RESULTS_DIR"

    if ! all_gpus_free; then
        log "WARNING: some GPUs already have memory in use; continuing anyway"
        gpu_snapshot >&2
    fi

    # Dedicated Ray head. Explicit address+ports prevent collision with any
    # unrelated Ray instance the host might run (we've seen this bite us).
    log "Starting dedicated Ray head at 127.0.0.1:${RAY_PORT}"
    RAY_USAGE_STATS_ENABLED=0 ray start \
        --head \
        --port "$RAY_PORT" \
        --dashboard-port $((RAY_PORT - 114)) \
        --node-ip-address 127.0.0.1 \
        --num-gpus 4 \
        --min-worker-port 30000 --max-worker-port 39999 \
        --disable-usage-stats \
        > "$RESULTS_DIR/ray_head.log" 2>&1

    log "Launching vllm serve in tmux session '$SERVE_SESSION'"
    tmux_oneshot "$SERVE_SESSION" \
        "source $VENV_DIR/bin/activate && \
         export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
         export RAY_ADDRESS=127.0.0.1:$RAY_PORT && \
         vllm serve $MODEL_SNAPSHOT \
             --host 0.0.0.0 --port $VLLM_PORT \
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
             --trust-remote-code \
             2>&1 | tee $SERVE_LOG"

    wait_for_ready "http://localhost:${VLLM_PORT}/health" 300
}

stop() {
    log "Stopping vllm serve tmux session"
    tmux_kill "$SERVE_SESSION"
    # Give vllm processes a moment to exit cleanly, then ensure they're gone
    sleep 3
    pkill -9 -f 'DPMoE|RayWorkerWrapper' 2>/dev/null || true
    log "Stopping Ray"
    ray stop --force > /dev/null 2>&1 || true
    sleep 2
    gpu_snapshot >&2
}

# ─── Subcommand: bench ────────────────────────────────────────────────
bench() {
    local label=${1:-unknown} num_prompts=${2:-32} concurrency=${3:-16}
    vllm_bench "$label" "localhost:$VLLM_PORT" "$num_prompts" "$concurrency" "$RESULTS_DIR"
}

# ─── Subcommand: scale ────────────────────────────────────────────────
scale() {
    local new_dp=${1:?target DP size required}
    trigger_scale "http://localhost:$VLLM_PORT" "$new_dp" \
        | tee "$RESULTS_DIR/scale_to_dp${new_dp}_$(date +%H%M%S).log"
}

# ─── Subcommand: state ────────────────────────────────────────────────
state() {
    local tag=${1:-snapshot}
    {
        echo "=== Phase 1 state ($tag) at $(date) ==="
        echo ""
        echo "## GPU memory + utilisation ##"
        gpu_snapshot
        echo ""
        echo "## GPU processes ##"
        gpu_processes
        echo ""
        echo "## Scaling flag ##"
        curl -sS -X POST "http://localhost:$VLLM_PORT/is_scaling_elastic_ep" 2>&1 || echo "(unreachable)"
    } | tee "$RESULTS_DIR/state_${tag}.txt"
}

# ─── Subcommand: cycle ────────────────────────────────────────────────
# Full reference cycle: DP=2 baseline -> scale up to 4 -> bench -> scale down to 2
# -> bench. This is what we call the "baseline" throughput numbers from.
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

# ─── Dispatch ─────────────────────────────────────────────────────────
cmd=${1:-}; shift || true
case "$cmd" in
    start)  start ;;
    stop)   stop ;;
    bench)  bench "$@" ;;
    scale)  scale "$@" ;;
    state)  state "$@" ;;
    cycle)  cycle ;;
    *)
        cat <<'EOF' >&2
usage: phase1_native.sh {start|stop|bench LABEL [N] [C]|scale TARGET_DP|state [TAG]|cycle}

subcommands:
    start                   bring up Ray head + vllm serve at DP=2
    stop                    stop serve and Ray
    bench LABEL [N] [C]     run `vllm bench serve --dataset-name random`
                            (defaults: N=32 prompts, C=16 concurrency)
    scale TARGET_DP         POST /scale_elastic_ep
    state [TAG]             snapshot GPU + scaling state
    cycle                   full DP=2 → 4 → 2 reference cycle with benchmarks

prerequisites:
    * `uv pip install vllm "ray[default]"` into PROJECT_ROOT/.venv
    * Qwen3-30B-A3B downloaded to /data/models--Qwen--Qwen3-30B-A3B/
    * all 4 GPUs free on the host
EOF
        exit 1
        ;;
esac
