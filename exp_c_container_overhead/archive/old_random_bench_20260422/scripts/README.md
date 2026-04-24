# Exp C scripts

Per-hypothesis diagnostic tooling accumulates here as the
investigation progresses. Start empty; add scripts when a measurement
pattern is repeated enough to deserve one.

## Reusing Exp A3's harness

The existing `exp_a3_vllm_ep/scripts/common.sh` already has most of
the primitives a diagnostic session needs:

- `log "msg"` — timestamped stderr log
- `ensure_venv` — activate project venv
- `wait_for_ready URL [timeout] [liveness_fn] [diag_fn]` — poll with
  fast-fail when the backing process dies
- `vllm_bench LABEL HOST:PORT N C OUT_DIR` — standard bench invocation
  with locked-down random dataset shape
- `trigger_scale URL NEW_DP [drain]` — POST /scale_elastic_ep, print
  elapsed
- `gpu_snapshot`, `gpu_processes` — nvidia-smi helpers
- `require_gpus_free` — abort fast if any GPU is busy (respects
  `ALLOW_BUSY_GPUS=1`)
- `docker_ensure_image`, `ensure_patched_image`
- `tmux_oneshot NAME CMD`, `tmux_kill NAME`

A diagnostic script in this directory can source it:

```bash
#!/usr/bin/env bash
set -euo pipefail
EXP_A3="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../exp_a3_vllm_ep" && pwd)"
# shellcheck source=../../exp_a3_vllm_ep/scripts/common.sh
source "$EXP_A3/scripts/common.sh"

# ... your measurement ...
```

## Conventions for measurement scripts

- **Idempotent and restartable.** GPU-busy aborts are expected on a
  shared host; scripts should not corrupt state on early exit.
- **Results directories include a config tag** so comparing variants
  is easy. e.g., `results/20260422-1430-shm16g/`,
  `results/20260422-1445-shm64g/`.
- **Tee logs.** `... 2>&1 | tee "$RESULTS_DIR/run.log"` everywhere;
  never discard stderr.
- **Save JSON and raw log together.** Bench JSON for numeric data +
  full serve log for post-hoc grep (NCCL transport, EPLB events, etc.).
