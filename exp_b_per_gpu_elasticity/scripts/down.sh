#!/usr/bin/env bash
# Teardown wrapper. Delegates to Exp A3 Phase 3's `down` subcommand,
# which stops the 4 per-GPU containers and saves their logs into
# ../exp_a3_vllm_ep/results/phase3/ep-rank-*.log.
#
# To copy those logs into this experiment's per-tag results dir, do:
#     LAST=$(cat ../exp_b_per_gpu_elasticity/results/LAST_BUILD | sed 's|^.*:||')
#     cp ../exp_a3_vllm_ep/results/phase3/ep-rank-*.log \
#        ../exp_a3_vllm_ep/results/phase3/vllm-serve.log \
#        ../exp_b_per_gpu_elasticity/results/$LAST/

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPB_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd "$EXPB_DIR/.." && pwd)

exec "$PROJECT_ROOT/exp_a3_vllm_ep/scripts/phase3_per_gpu.sh" down
