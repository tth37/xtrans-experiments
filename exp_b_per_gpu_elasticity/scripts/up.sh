#!/usr/bin/env bash
# Bring up the per-GPU container cluster using the most recently built
# patched image. Wraps ../../exp_a3_vllm_ep/scripts/phase3_per_gpu.sh up
# with VLLM_IMAGE pointing to the patched tag.
#
# Usage:
#   ./scripts/up.sh                                    # use results/LAST_BUILD
#   VLLM_IMAGE=xtrans-vllm-ep-patched:TAG ./scripts/up.sh
#   PHASE3_DP=2 ./scripts/up.sh                        # override initial DP
#                                                      # (after placement-group
#                                                      #  patch is validated)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPB_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd "$EXPB_DIR/.." && pwd)

# Pick image: explicit env var wins, else read LAST_BUILD
if [ -z "${VLLM_IMAGE:-}" ]; then
    if [ -f "$EXPB_DIR/results/LAST_BUILD" ]; then
        VLLM_IMAGE=$(cat "$EXPB_DIR/results/LAST_BUILD")
    else
        echo "ERROR: no patched image selected." >&2
        echo "Either build one with ./scripts/build.sh, or export VLLM_IMAGE=..." >&2
        exit 1
    fi
fi
export VLLM_IMAGE

if ! docker image inspect "$VLLM_IMAGE" > /dev/null 2>&1; then
    echo "ERROR: image $VLLM_IMAGE not found locally." >&2
    exit 1
fi

echo "Using patched image: $VLLM_IMAGE"
echo "Delegating to exp_a3_vllm_ep/scripts/phase3_per_gpu.sh up..."
echo ""

# Phase 3's harness picks up VLLM_IMAGE from the environment via common.sh
exec "$PROJECT_ROOT/exp_a3_vllm_ep/scripts/phase3_per_gpu.sh" up
