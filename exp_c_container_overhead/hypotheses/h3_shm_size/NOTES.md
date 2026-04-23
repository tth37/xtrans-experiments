# H3: `--shm-size` sensitivity

**Status:** Triage attempted; blocked on a methodology problem
discovered during the attempt. See H1 NOTES ("Warmup trajectory
discovery") for details. Briefly: MGC TPOT does not plateau within
5 benches of sustained load (descends 118 → 88 ms in baseline_n3),
so a 2-bench single-variant measurement can easily report a 15 ms
TPOT difference purely from bench-position mismatch between
variants. Need a steady-state protocol before H3 can be
discriminated. Initial 64m-vs-16g single-shot readings (94 vs 110
TPOT) are suggestive but unreliable — the faster 64m measurement
likely reflects a later warmup-state rather than a real shm
effect. Two variant archives preserved under
`results/variants/shm64m/` and `results/variants/baseline/` as
pre-methodology data; don't cite them for conclusions.

## Hypothesis

The multi-GPU-container currently runs with `--shm-size=16g`. Docker's
default is 64 MB. vLLM uses shared memory for several things (Ray
object store, potentially cuMem IPC if it falls back, sometimes CUDA
buffers on some drivers). If the vllm/Ray path is shm-bound and
16 GB is *still* a cliff relative to what native has access to
(which is the host's whole /dev/shm, typically tens of GB), shm
contention could manifest as higher tail latency.

## Planned measurements

1. **Vary `--shm-size`**: 64M (docker default), 2g, 16g (current),
   64g. Run standard bench at DP=4 for each. If throughput curves
   with shm-size, it's binding.

2. **Check `df -h /dev/shm` from inside container** at each setting
   and observe whether vLLM + Ray approach the cap during steady-state
   (watch `du -sh /dev/shm` inside container during bench, or use
   `docker stats`).

3. **Spill-to-disk telemetry from Ray.** Ray's object store can spill
   to disk when shm fills. Check
   `docker exec <ctn> ls /tmp/ray/session_latest/*spill*` for spill
   evidence during a run.

## Expected outcomes

- If H3 dominates: throughput rises monotonically with shm-size up to
  some knee, then plateaus. The default 16 GB is below the knee.
- If H3 is flat: shm-size has no effect on these numbers. Default is
  fine.

## Status notes

### 2026-04-23 01:00 — Variant script + initial triage

Added `exp_c_container_overhead/scripts/multi_gpu_container_variant.sh`:
parameterises `--shm-size`, `--ipc`, and `--data-parallel-size` via env
vars; otherwise identical `docker run` invocation to
`exp_a3_vllm_ep/scripts/multi_gpu_container.sh`. Result dirs go to
`exp_c_container_overhead/results/variants/<VARIANT_TAG>/`.

Three variants tried with a "2 warmup + 1 measurement" protocol
(3 benches total), and shm-size-only variants tried with "2 warmup +
3 measurements" (5 benches total) after the methodology problem was
discovered.

| Variant | Protocol | Bench-3 TPOT | Bench-5 TPOT | Plateau? |
|---|---|---:|---:|---|
| baseline (shm=16g, ipc=host) | 3-bench | 110.52 ms | — | no |
| baseline_n3 (shm=16g, ipc=host) | 5-bench | 108.87 ms | 87.64 ms | no (still descending) |
| shm64m (shm=64m, ipc=host) | 3-bench | 94.30 ms | — | unclear |
| shm64m_n3 (shm=64m, ipc=host) | 5-bench | 92.82 ms | 93.82 ms | **yes, σ 0.59** |
| ipcprivate (shm=16g, ipc=private) | 3-bench+1extra | 111.90 ms | — | no |
| baseline_extended (shm=16g, ipc=host) | planned 8-bench | 120.41 ms | — | aborted after b1 by external SIGKILL |
| native_n3 (reference) | 5-bench | 91.01 ms | 86.61 ms | **yes by bench 2, σ 2.22** |

### The warmup-trajectory finding

Running a single 32-prompt × 128-output bench does not reach MGC
steady-state in the default `--shm-size=16g` config. Five consecutive
benches descend roughly 118 → 111 → 109 → 97 → 88 ms TPOT without
plateauing — meaning *every single bench we had run earlier* (H5's
"cold" + "warm" numbers, H1's "cold" + "warm" numbers) was reading
from a still-warming state. The σ 10.64 ms across the last three
measurements of `baseline_n3` is inflated by the descending trend
itself, not by noise between repeat stationary measurements.

**shm=64m flips this completely.** With the Docker default 64 MB
`/dev/shm`, MGC plateaus cleanly at 93.51 ms by bench 3
(σ 0.59 ms across benches 3–5). Same model, same workload, same
everything else — only `--shm-size` differs.

Two plausible explanations, both pointing at Ray:

1. **Ray plasma object store lazily formats shm.** With 16 GB
   available, Ray's plasma may be touch-faulting pages or building
   an internal index across the whole region during the first few
   benches; a 64 MB region touches to completion almost immediately.
2. **Ray defers some inter-actor data to plasma when space is
   available.** With 64 MB, Ray's threshold for plasma-vs-inline
   drops, so per-step coordination goes through a faster inline
   path that doesn't need warm-up.

The second mechanism would also predict that a smaller shm is
*faster at steady-state* (fewer plasma round-trips), not slower.
The baseline_n3 trajectory at bench 5 (TPOT 87.64 ms) is already
below shm64m's plateau (93.51 ms), suggesting `shm=16g` *might*
plateau lower than `shm=64m` eventually. But to confirm, we need
to run `shm=16g` long enough to plateau. One attempt at 8 benches
(`baseline_extended`) was SIGKILL'd by external activity on the
host at bench 1.

### What the variants mean for the actual investigation

- **H3 is LIVE**, but not in the direction originally predicted.
  The hypothesis was "shm is too small → throttling"; the observation
  is "shm default matters for *convergence time*, direction of the
  plateau effect TBD."
- **§5.7's and H1's 'warm TPOT +17%' finding is invalid.** Those
  measurements read MGC at a descending warmup trajectory, not at
  plateau. Once shm-size is controlled and both regimes are given
  enough warmup, the steady-state TPOT gap is much smaller
  (maybe 0–5%).
- **Native reference plateau at bench 2, σ 2.22 ms**: clean. Native
  is not sensitive to the warmup-bench-count issue because its
  `/dev/shm` is the host's whole 252 GB tmpfs and doesn't need the
  plasma-init work that happens inside a sized container shm.

### Protocol improvement for follow-up

Before resuming H3/H4, the harness needs a steady-state-detection
protocol:

- Run benches in a loop; track rolling TPOT.
- Stop declaring "warm" when 3 consecutive benches' TPOT are within
  X ms of each other (say ±2% or ±3 ms absolute).
- Use that "warm" bench for per-variant comparisons.

Implementable as a new `bench_to_steady` subcommand in the variant
script, invoking `vllm_bench` in a loop and parsing the JSON after
each iteration. Not yet implemented.

### 2026-04-23 01:44 — shm=16g 10-bench plateau (`mgc_n10`)

After the user's `baseline_extended` 8-bench run was killed at bench
1, I re-ran shm=16g/host with 2 warmups + 10 measurement benches
(`results/variants/mgc_n10/`). This time TPOT reached steady state
quickly:

| Bench | TPOT (ms) |
|---|---:|
| warmup1 | 108.45 |
| warmup2 | 88.49 ← plateau already |
| meas1–10 | 91.65 / 86.79 / 86.89 / 86.30 / 87.07 / 86.45 / 85.19 / 86.93 / 90.98 / 81.97 |

meas1–10 mean **87.02 ms, σ 2.84**. Last-5 mean 86.30 ms.

Replicate run (`mgc_n10_v2`) confirms: meas3–10 mean **88.67 ms,
σ 2.04**. v2 was a touch slower to plateau than v1 — meas1–2 were
95, 94 ms before converging — but converges to the same 86–90 ms
band.

**This confirms the hypothesis from the previous section:** shm=16g
*does* plateau lower than shm=64m (87 ms vs 93.5 ms). So shm-size
trades warmup-speed against plateau-height. Also, the wide
variance in warmup time (mgc_n10 plateaued at bench 2;
baseline_n3 took 5+ benches at shm=16g) suggests the warmup
trajectory is sensitive to prior host state (CPU/GPU thermals,
page-cache, etc.), not a deterministic property of the config.

### Steady-state verdict (reconciled with H1's reopen)

Combining the shm=16g data from mgc_n10 + mgc_n10_v2 (n=18
post-plateau measurements) against native's plateau (n=13):

| Regime | Plateau TPOT | σ |
|---|---:|---:|
| Native | 89.66 ms | 2.59 |
| MGC shm=16g | 87.67 ms | 2.53 |
| MGC shm=64m | 93.51 ms | 0.59 |

Native vs MGC shm=16g: 0.77σ apart — **not statistically
distinguishable**. The container's overhead at DP=4 steady state
is effectively zero. §5.7's 24% "gap" was methodological, not a
real per-token cost.

### Status

- **H3: closed with two findings.**
  1. `shm-size` does NOT bind native steady-state throughput in
     the bad direction H3 originally hypothesised. The
     "production" shm=16g config plateaus at ~87 ms TPOT, which
     is as fast as or faster than native.
  2. `shm-size` DOES affect warmup convergence: shm=64m plateaus
     in ~1 bench, shm=16g in 2–5 benches depending on prior host
     state. shm=64m plateaus ~5 ms higher than shm=16g.
  3. The interesting engineering trade-off: if you want a
     predictable first-bench TPOT, set shm=64m and accept a small
     plateau penalty. If you're running a long-lived service,
     keep shm=16g (or larger) and accept a few benches of slow
     warmup for the lower plateau.
- **H4 (ipc=host): deferred.** Single-shot `ipc=private` measurement
  (TPOT 111.90 at bench 3) is indistinguishable from warmup-
  trajectory noise at shm=16g. Running ipcprivate with the full
  10-bench protocol would resolve this, but given H1 closes with
  "no steady-state gap", the incremental value of pinning H4 is
  low. Leaving the variant script in place for someone to finish
  the test in future.
