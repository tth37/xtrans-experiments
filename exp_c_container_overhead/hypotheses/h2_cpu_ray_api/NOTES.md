# H2: CPU/Ray/API-server overhead inside Docker

Goal: determine whether the MGC gap is dominated by request scheduling, API
server, CPU, or Ray actor overhead rather than GPU decode kernels.

Evidence to collect after H0 pairs:
- Compare TTFT gap vs TPOT gap. TTFT-only gap suggests API/Ray/scheduling;
  TPOT gap suggests decode-path or communication/kernel overhead.
- Serve-log engine throughput lines: prompt throughput, generation throughput,
  running/waiting requests, KV cache usage.
- Ray actor startup and placement timing from logs.
- Optional follow-up: add CPU sampling around `bench` windows if H0 confirms a
  repeatable gap.

## First diagnostic result (2026-04-24)

Native cold startup repeatedly took ~100 s to `/health`. One serve log shows:

- 06:35:47 DP coordinator starts / Ray backend begins.
- 06:35:50 placement groups are created.
- 06:36:43 first weight-load log appears.
- 06:37:03 vLLM server starts.

This points to a large native Ray/vLLM startup and first-request warmup cost,
not a steady-state MGC container overhead. First-bench native TTFT was
1.1–1.5 s, while the warm second bench dropped to ~239 ms.
