# Exp C: Container Overhead Analysis

**Status:** Investigation open, no hypotheses closed yet.
**Relates to:** Exp A3 (the reference experiment). This is a
diagnostic sandbox that will, if it succeeds, feed findings back
into `exp_a3_vllm_ep/` as updates to the multi-GPU-container regime.

## Problem statement

The 2026-04-22 same-session re-run in `exp_a3_vllm_ep/analysis_report.html`
§5.7 showed a **−24% output-throughput gap at DP=4** between the native
regime (171.23 tok/s) and the multi-GPU container regime (130.62 tok/s)
on node192, using the same Qwen3-30B-A3B model, same vLLM 0.19.0, same
Ray 2.55.0, same `vllm bench serve` flags, and the container configured
with `--gpus '"device=0,1,2,3"' --ipc=host --shm-size=16g` mounting the
HF cache read-only.

This gap is **wider than historically measured** in the same experiment:
an earlier run documented in §5.5.1 had native 166.6 tok/s vs.
multi-GPU container 154.1 tok/s at DP=4 → only −7.5%. Something in the
host environment, the vllm/ray stack, or the bench methodology has
shifted between runs, and the gap is now ~3× larger than before.

### The gap is not uniform across metrics

| Metric at DP=4 | Native | Multi-GPU container | Δ |
|---|---|---|---|
| Output tok/s | 171.23 | 130.62 | **−24%** |
| Mean TPOT | 91.1 ms | 109.4 ms | +20% |
| Mean TTFT | **387 ms** | **1777 ms** | **+359% (4.6×)** |

The TTFT blow-up is disproportionate to the output-throughput gap.
This suggests the in-container overhead is concentrated in
setup / first-token paths (CUDA IPC handshake, collective init,
Ray actor warmup, kernel compilation, CUDA graph capture) rather than
in steady-state token generation. Hypothesis ordering below uses that
observation.

## Candidate hypotheses

Not mutually exclusive. Each gets its own subdirectory under
`hypotheses/` with a `NOTES.md` tracking plan, measurements, status.

| # | Hypothesis | Rough priority |
|---|---|---|
| H1 | TTFT / compile / CUDA-graph cold-path cost is the dominant contributor | High — directly suggested by 4.6× TTFT blow-up |
| H2 | NCCL init overhead (distinct from steady-state NCCL) | High — Elastic EP uses stateless groups which reinit per scale event |
| H3 | `--shm-size` sensitivity (vLLM uses shm for some IPC channels) | Medium — easy to vary, informative regardless |
| H4 | `--ipc=host` sensitivity (currently on; what if off or different flavour) | Medium — impacts CUDA IPC handshake |
| H5 | Bench-run variance / concurrent host load | Must-verify first — calibrate how much drift is baseline noise |

## Approach

1. **Reproduce the gap first.** Before any hypothesis work, confirm the
   gap reproduces by re-running both regimes back-to-back on an
   all-quiet host. If it does not reproduce at ~−24%, H5 (variance) is
   the effective explanation and the investigation narrows.

2. **Per-hypothesis isolation.** Each hypothesis changes one variable
   vs. the reference. Measurements per variant use the same bench
   shape and prompt seed as Exp A3 for direct comparability:

       vllm bench serve --dataset-name random --random-input-len 128
                        --random-output-len 128 --num-prompts 32
                        --max-concurrency 16 --seed 0

3. **Measurement emphasis on the breakdown**, not just tok/s. Use
   `vllm bench serve`'s JSON output for TTFT / TPOT / ITL percentiles;
   instrument with `NCCL_DEBUG=INFO` for per-collective init timing;
   `CUDA_LAUNCH_BLOCKING=1` only if needed to isolate compile-time
   behaviour (it will distort steady-state numbers, use with care).

4. **Findings flow back to exp_a3** if they warrant a script change
   (e.g., different `--shm-size`, different `--ipc` mode) or a
   documentation update in §5.5 / §5.7 explaining the gap.

## Reference baseline (to reproduce first)

```bash
cd /home/thd/repositories/xtrans-experiments
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # all 4 MiB

./exp_a3_vllm_ep/scripts/native.sh start
./exp_a3_vllm_ep/scripts/native.sh cycle
./exp_a3_vllm_ep/scripts/native.sh stop
# capture native numbers

./exp_a3_vllm_ep/scripts/multi_gpu_container.sh start
./exp_a3_vllm_ep/scripts/multi_gpu_container.sh cycle
./exp_a3_vllm_ep/scripts/multi_gpu_container.sh stop
# capture multi-GPU container numbers
```

Numbers land in `exp_a3_vllm_ep/results/native/bench_*.json` and
`exp_a3_vllm_ep/results/multi_gpu_container/bench_*.json` (both
gitignored). Extract metrics with:

```python
import json
for f in ['bench_dp2_initial', 'bench_dp4_post_up', 'bench_dp2_post_down']:
    d = json.load(open(f'exp_a3_vllm_ep/results/<regime>/{f}.json'))
    print(f, d['output_throughput'], d['mean_ttft_ms'], d['mean_tpot_ms'])
```

## In scope / out of scope

**In scope:**
- Measurements of any in-container overhead component (TTFT breakdown,
  NCCL init, shm, ipc, Ray-inside-container init).
- Dockerfile / run-flag variants to isolate each.
- Documentation in hypothesis `NOTES.md` files and, eventually, an
  `analysis_report.html` in this directory.

**Out of scope:**
- Modifying the exp_a3 reference baseline. If a finding warrants a
  change, migrate it cleanly in a separate commit (same pattern Exp B
  used — see git log for reference).
- Touching the per-GPU-containers regime. That's a different question
  with a different root cause (NCCL Socket fallback, already
  documented in exp_a3 §5.6.3 and §6.6).
- New models, new hardware, changing vLLM version. Hold the
  experimental axis at one variable at a time.

## Files

- `README.md` — this file
- `hypotheses/h<N>_<name>/NOTES.md` — per-hypothesis plan + findings
- `scripts/` — diagnostic tooling as it accumulates (currently empty
  except `README.md`; most scripts can source `../../exp_a3_vllm_ep/scripts/common.sh`
  to reuse helpers)
- `results/<timestamp>/` — gitignored; bench JSONs, serve logs,
  NCCL traces, `docker inspect` snapshots
- `analysis_report.html` — write when findings stabilise
