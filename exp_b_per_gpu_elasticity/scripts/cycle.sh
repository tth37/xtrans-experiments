#!/usr/bin/env bash
# Attempt an elastic cycle in the per-GPU container cluster.
#
# Default sequence: DP=4 (already up) -> bench -> scale to 2 -> bench -> scale
# back to 4 -> bench. Each step records timing, bench output, and per-container
# DeviceIDs. This is THE test that tells us whether the current patch set has
# unblocked per-GPU elastic scaling.
#
# Usage:
#   ./scripts/cycle.sh                        # default 4→2→4 cycle
#   CYCLE_STEPS="2 4 2" ./scripts/cycle.sh    # custom DP sequence
#                                             # (each element is the target
#                                             #  DP after /scale_elastic_ep)
#
# Prerequisite: `./scripts/up.sh` has been run and vLLM is serving at DP=4
# (or whatever initial DP the patch set supports).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPB_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd "$EXPB_DIR/.." && pwd)
PHASE3_SCRIPT="$PROJECT_ROOT/exp_a3_vllm_ep/scripts/phase3_per_gpu.sh"

# Resolve the image tag this run is using, for result filing
if [ -f "$EXPB_DIR/results/LAST_BUILD" ]; then
    TAG=$(sed 's|^.*:||' < "$EXPB_DIR/results/LAST_BUILD")
else
    TAG=$(date +%Y%m%d-%H%M)
fi
RESULTS_DIR="$EXPB_DIR/results/${TAG}"
mkdir -p "$RESULTS_DIR"

# Verify vLLM is serving
if ! curl -fsS --max-time 2 http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM not responding on http://localhost:8000/health." >&2
    echo "Run ./scripts/up.sh first." >&2
    exit 1
fi

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$RESULTS_DIR/cycle.log"; }

log "===== cycle.sh starting (image tag: $TAG) ====="
log ""

# Baseline bench + state
log "--- Baseline (current DP, pre-cycle) ---"
"$PHASE3_SCRIPT" state "pre_cycle" > "$RESULTS_DIR/state_pre_cycle.txt" 2>&1
"$PHASE3_SCRIPT" bench "pre_cycle" 32 16 2>&1 | tee -a "$RESULTS_DIR/cycle.log" >/dev/null
log ""

# Execute the user-specified scale sequence
: "${CYCLE_STEPS:=2 4}"
step_i=0
for target in $CYCLE_STEPS; do
    step_i=$((step_i + 1))
    tag="step${step_i}_dp${target}"
    log "--- Step $step_i: scale to DP=$target ---"
    "$PHASE3_SCRIPT" scale "$target" 2>&1 | tee -a "$RESULTS_DIR/cycle.log" >/dev/null
    sleep 3
    "$PHASE3_SCRIPT" state "$tag" > "$RESULTS_DIR/state_${tag}.txt" 2>&1
    # If the scale succeeded, bench at the new DP
    if curl -fsS --max-time 2 http://localhost:8000/health > /dev/null 2>&1; then
        log "    bench at DP=$target"
        "$PHASE3_SCRIPT" bench "$tag" 16 8 2>&1 | tee -a "$RESULTS_DIR/cycle.log" >/dev/null
    else
        log "    SERVER UNREACHABLE after scale to DP=$target -- aborting cycle"
        log "    (this is probably the EPLB num_redundant bug or a placement"
        log "     group mismatch; check docker logs on ep-rank-0)"
        break
    fi
    log ""
done

# NCCL transport evidence (one capture per run is enough)
log "--- Capturing NCCL transport selection ---"
"$PHASE3_SCRIPT" nccl-grep > "$RESULTS_DIR/nccl_transport.log" 2>&1 || true

log ""
log "===== cycle.sh finished ====="
log "Results directory: $RESULTS_DIR"

# Copy in the per-step bench JSONs from phase3 into our results dir for
# self-contained analysis.
cp -a "$PROJECT_ROOT/exp_a3_vllm_ep/results/phase3/bench_"*.json "$RESULTS_DIR/" 2>/dev/null || true
