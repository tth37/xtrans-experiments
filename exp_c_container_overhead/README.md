# Exp C: Container Overhead Redo

**Status:** active analysis complete for the benchmark-inconsistency question.
The old random-benchmark study remains archived under
`archive/old_random_bench_20260422/`.

## Conclusion

The apparent Native/MGC container-overhead signal was a benchmark-method
artifact. ShareGPT DP=4 samples have a large cold first bench, and native
Ray/vLLM startup variance is especially visible in TTFT and TPOT. After Exp A3
explicitly discards the first bench and summarizes only the measured warm
window, Native and MGC are statistically tied. PGC remains slower than MGC,
which is consistent with PGC's cross-container NCCL fallback to `NET/Socket/0`.

Fresh robust DP=4 ShareGPT c32 result:

| Regime | Throughput tok/s | TTFT ms | TPOT ms | Measured window |
|---|---:|---:|---:|---|
| Native | 212.30 ± 7.44 | 280.1 ± 54.5 | 94.99 ± 5.00 | benches 2–4 |
| MGC | 212.42 ± 17.90 | 291.0 ± 43.4 | 91.98 ± 4.79 | benches 2–4 |
| PGC | 199.20 ± 5.57 | 581.3 ± 195.5 | 102.46 ± 8.44 | benches 2–4 |

## Harness

`run_overhead_study.py` is retained for paired Native/MGC follow-up probes. It
waits for idle GPUs, records pre/post snapshots, runs the Exp A3 DP=4 `bench`
subcommand, copies JSON/log artifacts, and writes an aggregate summary under
`results/sharegpt_dp4/`.

Default shape:

```bash
export A3_SINGLE_LABEL=dp4_direct_sharegpt_c32
export A3_SINGLE_NUM_PROMPTS=96
export A3_SINGLE_CONCURRENCY=32
export A3_BENCH_DATASET=sharegpt
export A3_BENCH_DATASET_PATH=exp_a3_vllm_ep/data/ShareGPT_V3_unfiltered_cleaned_split.json
export A3_BENCH_NUM_PROMPTS=96
export A3_BENCH_MAX_CONCURRENCY=32
export A3_BENCH_DISCARD_FIRST=1
export A3_BENCH_MAX_BENCHES=4
export A3_BENCH_EXTRA_SAMPLES=1
export A3_MAX_NUM_SEQS=32
```

Run paired probes in tmux if more confirmation is needed:

```bash
tmux new-session -d -s exp-c-sharegpt-dp4 \
  'python3 exp_c_container_overhead/run_overhead_study.py run-pairs --pairs 3 \
   2>&1 | tee /tmp/exp-c-sharegpt-dp4.log'
```

Analyze existing copied results:

```bash
python3 exp_c_container_overhead/run_overhead_study.py analyze
```

## Hypothesis Summary

| ID | Hypothesis | Decision |
|---|---|---|
| H0 | Benchmark variance / host contention | Supported: cold first benches and native startup variance explain the Native/MGC flip. |
| H1 | Container serving config mismatch | Not primary: robust reruns used matching model, DP, ShareGPT shape, Ray backend, eager mode, and `max_num_seqs=32`. |
| H2 | CPU/Ray/API-server overhead inside Docker | Not supported as a measurable MGC penalty after warmup filtering. |
| H3 | Filesystem/cache/path overhead | Deferred; no steady MGC penalty remains to explain. |
| H4 | Container runtime flags | Deferred until a repeatable MGC penalty appears. |
