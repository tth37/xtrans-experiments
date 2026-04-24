# H0: Benchmark variance / host contention

Goal: determine whether the Native→MGC gap under DP=4 ShareGPT saturation is
repeatable or just run-to-run/host-load noise.

Protocol:
- Use `run_overhead_study.py run-pairs`.
- Run Native then MGC back-to-back after waiting for idle GPUs before each regime.
- Default Exp C window is 3 ShareGPT benches per regime (`EXP_C_MAX_BENCHES=3`,
  `EXP_C_EXTRA_SAMPLES=0`) because full 5-bench windows make the initial H0 pass
  too slow while preserving the same workload shape.
- Capture pre/post `nvidia-smi`, process snapshots, harness logs, serve logs, and
  per-bench JSON.

Decision rule:
- Gap is real only if repeated paired runs show MGC slower than Native by more
  than the combined stable-window σ.
- If pair results disagree or contamination appears, rerun the contaminated pair.

Initial observation:
- A full native ShareGPT run was stopped after two samples because the 5-bench
  campaign was too slow for 3 pairs. It already showed warmup: bench1 TPOT
  111.47 ms / TTFT 1155 ms, bench2 TPOT 93.47 ms / TTFT 239 ms.

## First diagnostic result (2026-04-24)

The expected Native > MGC steady-state gap did not reproduce in the completed
ShareGPT stable summaries. Instead:

| Regime | Throughput | TTFT | TPOT |
|---|---:|---:|---:|
| Native stable | 206.68 ± 3.79 | 273.8 ± 29.1 | 97.11 ± 6.83 |
| MGC stable | 225.23 ± 1.74 | 247.5 ± 42.7 | 88.42 ± 0.48 |

MGC is +9.0% throughput and -8.9% TPOT versus Native. The cold single-bench
native attempts were highly variable: p1 bench1 TPOT 111.47 / TTFT 1155 ms,
p1 bench2 TPOT 93.47 / TTFT 239 ms, p2 bench1 TPOT 112.89 / TTFT 1461 ms.

Interpretation: the notable gap is not currently an MGC steady-state overhead.
The misleading gap is more likely caused by cold/warm effects and native
Ray/vLLM startup variance. Future H0 should compare warm-only benches with the
same server kept alive.

## Benchmark-method root cause update

The inconsistency is now attributed primarily to benchmark method:

1. Cold first samples are large outliers for all regimes.
2. Native warm samples are noisier than MGC under ShareGPT.
3. The old fallback summary could still represent an unstable warmup curve when
   convergence failed.

A3 now adds `A3_BENCH_DISCARD_FIRST=1` so cold samples are never included in
convergence or summaries.
