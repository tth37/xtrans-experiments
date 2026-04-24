# Exp C: Container Overhead Redo

**Status:** Redesign scaffolded; measurements pending.

## Why this experiment was reset

The original Exp C study is archived under
`archive/old_random_bench_20260422/`. It was based on the older random
128/128 benchmark and cycle-oriented methodology. Exp A3 now uses a Python
stable-bench harness, supports DP=4-only `bench`, and shows that ShareGPT
saturation can reveal performance effects hidden by the baseline random
shape. The old Exp C conclusions should therefore be treated as historical
context, not as the active answer.

## Primary question

Why does the multi-GPU-container (MGC) regime underperform native under the
updated robust benchmark, if single-container overhead should theoretically be
near zero?

## Baseline protocol

Use Exp A3's DP=4-only `bench` subcommand for native and MGC, not the
2→4→2 cycle.

Default shape:

```bash
export A3_SINGLE_LABEL=dp4_direct_sharegpt_c32
export A3_SINGLE_NUM_PROMPTS=96
export A3_SINGLE_CONCURRENCY=32
export A3_BENCH_DATASET=sharegpt
export A3_BENCH_DATASET_PATH=exp_a3_vllm_ep/data/ShareGPT_V3_unfiltered_cleaned_split.json
export A3_BENCH_NUM_PROMPTS=96
export A3_BENCH_MAX_CONCURRENCY=32
export A3_BENCH_MAX_BENCHES=5
export A3_BENCH_EXTRA_SAMPLES=1
export A3_MAX_NUM_SEQS=32
```

Run paired measurements, waiting for idle GPUs before each pair:

```bash
python3 exp_a3_vllm_ep/1_native.py bench
python3 exp_a3_vllm_ep/2_multi_gpu_container.py bench
```

## Metrics

Collect throughput, TTFT, TPOT, completed request count, convergence status,
stable-window standard deviation, per-bench warmup trajectory, serve logs,
`nvidia-smi` snapshots, and process ownership snapshots.

Treat a Native-vs-MGC gap as real only if it repeats across at least three
clean paired runs and exceeds the combined stable-window σ.

## Hypotheses

| ID | Hypothesis | First evidence to collect | Status |
|---|---|---|---|
| H0 | Benchmark variance / host contention | Three clean paired native/MGC runs with GPU/process snapshots | Pending |
| H1 | Container serving config mismatch | Diff effective vLLM args, CUDA/Ray env, model path, cache path, `max_num_seqs` | Pending |
| H2 | CPU/Ray/API-server overhead inside Docker | Compare serve logs, request scheduling, Ray actor placement, CPU/API latency | Pending |
| H3 | Filesystem/cache/path overhead | Compare cache/model/tokenizer paths and cold-cache behavior | Pending |
| H4 | Container runtime flags | Test one Docker flag at a time after H0 reproduces the gap | Pending |

## Reporting

The active `analysis_report.html` will be written after the first clean paired
run set. It should report the paired-run table first, then update the
hypothesis table with evidence and decisions. Exp A3 should only link back to
Exp C once the redo has stable conclusions.
