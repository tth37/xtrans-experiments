# H1: TTFT / compile / CUDA-graph cold-path cost dominates the gap

**Status:** Closed with a refined picture. The cold TTFT blow-up the
hypothesis targeted (4.6× at DP=4 in §5.7) is real but is entirely
**one-time setup cost** — warm TTFT inside the container (390 ms) is
within 4% of warm TTFT on native (376 ms). However, the original H1
narrative ("the TTFT blow-up is the gap") is only partly right:
warming up closes the TTFT gap but does **not** close the overall
throughput gap. A −14% warm gap remains at DP=4 out=128, driven
**entirely by TPOT** (+17% per-token cost inside the container).
The long-output test (out=1024) rules out TTFT-amortization as the
explanation for the throughput gap. So the §5.7 cold-bench headline
conflates two distinct container overheads: (a) a one-time TTFT
blow-up and (b) a persistent warm TPOT gap. Next hypotheses to
investigate: H3 (shm-size) and H4 (ipc=host), both candidates for
the per-forward TPOT overhead.

## Hypothesis

The 4.6× TTFT blow-up (387 ms native → 1777 ms multi-GPU container at
DP=4) is disproportionate to the 24% output-throughput drop, suggesting
the extra wall-clock inside the container is concentrated in the
first-token path: kernel compilation, CUDA graph capture, expert
weight paging on first use, MoE config JSON loading with a
"default MoE config" warning in-container that isn't needed on native.

If most of the gap is TTFT-side, a bench with very long output
sequences (e.g. `--random-output-len 1024`) should show a much smaller
gap than the current 128-output bench.

## Planned measurements

1. **Isolate TTFT from output throughput.** Bench both regimes at
   `--random-input-len 128 --random-output-len 1024 --num-prompts 8`.
   If output tok/s gap shrinks from −24% toward the TPOT gap (+20%),
   that confirms H1.

2. **First-request vs. steady-state.** Fire a single warmup request
   before the bench to pay compile/TTFT costs once. Then run the
   standard bench. If container bench gap collapses after warmup,
   the gap is one-time setup cost.

3. **Check the "default MoE config" warning.** The container
   startup log mentions
   `Config file not found at .../fused_moe/configs/E=32,N=768,device_name=NVIDIA_A100-*.json`.
   Native log: does it emit the same warning? If the config IS found
   on native but not in-container (mount issue, path mismatch), that
   alone could explain kernel-selection differences and TTFT cost.

## Expected outcomes

- If H1 is dominant: long-output bench gap shrinks to ~−8% (matching
  TPOT gap), warmup-first bench gap collapses further, and the
  "solution" is either pre-warmup in the harness or making the MoE
  config discoverable in-container.
- If H1 is minor: long-output bench still shows ~−24% gap. Move to
  H2 / H3 / H4.

## Status notes

### 2026-04-22 15:23 — Measurement #3 (MoE config diff) completed

Read of native and MGC serve logs from the session-2 H5 archives
(`native_run1/serve.log`, `multi_gpu_container_run2/container.log`)
showed identical `fused_moe` config warnings in both regimes:

```
WARNING fused_moe.py:1090 Using default MoE config. Performance might
be sub-optimal! Config file not found at <prefix>/vllm/model_executor/
layers/fused_moe/configs/E=64,N=768,device_name=NVIDIA_A100-SXM4-40GB.json
```

The prefix differs (`/home/thd/...venv/lib/...` vs
`/usr/local/lib/...`) because of where vllm is installed, but the
relative config-dir path is the same and the file is missing in
both. Both regimes fall back to the default MoE config.

A second warning fires during scale-up 2→4 for the PCIE GPU variant
(`device_name=NVIDIA_A100-PCIE-40GB.json`) — again identical across
regimes. So neither the missing SXM4 config file nor the missing
PCIE config file is a regime differentiator.

Note: the NOTES.md draft originally referenced `E=32,N=768`; the
actual warning shows `E=64,N=768`. At DP=2 cold start,
`num_physical_experts / ep_size = 128/2 = 64`, which gives
E=64. The draft was written against an incorrect/older recollection.

**Also ruled out by the compilation_config dump in both logs:**
`'mode': <CompilationMode.NONE: 0>`, `'cudagraph_mode':
<CUDAGraphMode.NONE: 0>`, `'max_cudagraph_capture_size': 0`. The
scripts pass `--enforce-eager`, which disables both torch.compile
and CUDA-graph capture. So the original H1 draft's "kernel
compilation, CUDA graph capture" candidates are both out.

**Narrowed H1 candidate list for TTFT blow-up:**

1. **Triton JIT kernel cache state.** vLLM's MoE dispatch uses
   Triton kernels compiled at first invocation. If the container's
   Triton cache dir (`/root/.triton/cache` or similar) is empty
   while the native host has cached kernels from prior runs, the
   first request in-container pays compilation cost that native
   skips. Testable: inspect cache contents / set
   `TRITON_CACHE_DIR` to a common location and re-bench.
2. **First-forward weight paging** / page-cache state of
   bind-mounted model files.
3. **Ray actor / cross-container first-request latency** (for
   multi-GPU container, Ray workers still live in the same network
   namespace as the API server; this is less likely).

Measurements #1 (output-length sweep at DP=4) and #2
(warmup-vs-cold) will discriminate between cases 1/2 and rule out
whether the entire overhead is one-time setup cost.

### 2026-04-22 15:24 — Paused on host contention

Attempted to start H1 measurements #1/#2 but GPU snapshot showed
two external processes (PID 303074 on GPU 0, 15.8 GB; PID 299039
on GPU 3, 27.4 GB). Will resume when host is idle again.

### 2026-04-23 00:35 — Measurements #1 + #2 completed (n=1 per regime)

One session each for native and MGC, same shape per regime:
`cycle` (gives cold DP=4 out=128) → `scale 4` → warm DP=4 out=128 bench
→ warm DP=4 out=1024 bench → `stop`. Archives:
`native_s1/` and `mgc_s1/`. Both sessions ran on clean hosts, with
pre/post `nvidia-smi --query-compute-apps` snapshots in each
`run_meta.txt` confirming no external GPU occupants.

**Per-regime raw numbers (DP=4):**

| Bench | Native tok/s | Native TTFT | Native TPOT | MGC tok/s | MGC TTFT | MGC TPOT |
|---|---:|---:|---:|---:|---:|---:|
| Cold out=128 (cycle) | 139.43 | 447.27 | 112.06 | 134.30 | 1399.94 | 108.99 |
| Warm out=128 | 155.12 | 376.30 | 100.93 | 133.43 | 389.75 | 117.71 |
| Warm out=1024 | 87.06 | 207.11 | 91.76 | 71.98 | 254.99 | 110.94 |

**Per-request wall-time decomposition (DP=4 out=128):**

```
native cold:  447 ms + 128 × 112.06 = 14810 ms/req   (TTFT is 3.0% of total)
native warm:  376 ms + 128 × 100.93 = 13295 ms/req   (TTFT is 2.8% of total)
mgc cold:    1400 ms + 128 × 108.99 = 15351 ms/req   (TTFT is 9.1% of total)
mgc warm:     390 ms + 128 × 117.71 = 15457 ms/req   (TTFT is 2.5% of total)
```

**Gap evolution across conditions (lower = container slower):**

| Metric | Cold out=128 | Warm out=128 | Warm out=1024 |
|---|---:|---:|---:|
| Output tok/s gap | −3.7% | **−14.0%** | **−17.3%** |
| TTFT ratio (MGC/native) | 3.13× | 1.04× | 1.23× |
| TPOT ratio (MGC/native) | 0.97× | **1.166×** | **1.209×** |

### Four signals that reframe H1

**Signal 1 — the TTFT blow-up is one-time setup cost.**
Cold MGC TTFT: 1400 ms. Warm MGC TTFT: 390 ms. That's a 72% drop
(≈1000 ms saved) after one warmup bench. Native warms up too, but
far less: 447 → 376 (16% drop, 71 ms saved). Once both are warm,
TTFT is essentially identical between regimes (390 vs 376, +3.7%).
Conclusion: the container-specific TTFT blow-up is bounded at
the session boundary; after the first bench the "container-pays-an-extra-second-on-TTFT"
phenomenon is gone.

**Signal 2 — warmup grows the throughput gap, does not close it.**
Cold gap at out=128 is only −3.7%. Warm gap is −14.0%. This is
backwards from the naive H1 prediction. Reason: native benefits
from warmup (TPOT 112 → 101 ms, −10%; TTFT −16%) while MGC does
not (TPOT 109 → 118 ms, +8%). The cold-to-cold comparison *masks*
the true steady-state gap because native's cold state is
unexpectedly slow.

**Signal 3 — long output does not amortize the gap.**
If TTFT were the dominant container overhead, extending output
length should shrink the gap. It doesn't: warm out=128 gap is
−14.0%, warm out=1024 gap is −17.3%. The gap persists (and
slightly grows, likely noise). This rules out H1's "the gap is
TTFT divided over few output tokens" framing.

**Signal 4 — the warm gap is 100% TPOT.**
Warm TTFT gap is +3.7% (within noise). Warm TPOT gap is +16.6%.
The 14% throughput penalty maps 1:1 to the TPOT penalty:
`(1 - 100.93/117.71) = -14.25%`. No other component contributes.

### What the §5.7 numbers were actually measuring

§5.7 reported 171.23 (native) vs 130.62 (MGC) at DP=4 — both from
cold benches at the tail of a `cycle` command. In light of the
warm/cold split we now have:

- §5.7's native 171.23 was high because §5.7's native cold TPOT was
  also at 91 ms (close to session-2's **warm** TPOT of 101 ms).
  That's unusually-low-for-cold in today's distribution.
- §5.7's MGC 130.62 was a cold bench; but MGC cold vs warm TPOT are
  about the same (109 vs 118), so that number probably does
  reflect MGC steady-state.
- Result: §5.7's 24% gap = (native-warm-lucky-cold) vs
  (MGC-true-steady-state). That's why the headline is bigger than
  the cleaner measurements below.

Better anchor: warm-to-warm at DP=4 out=128.
- Gap: −14.0% throughput.
- TPOT-driven: +16.6% per-token cost inside container.
- TTFT: negligible contribution to the warm gap.

### Implications for the investigation

H1 is **closed with a partial confirm and a reframe**:

- H1's TTFT hypothesis is correct for cold benches: yes, MGC pays a
  ~1 s TTFT penalty on the first request after scale-up. This is
  visible in §5.7's numbers.
- H1's claim that TTFT explains the *throughput gap* is not
  supported. The gap that survives warmup is TPOT-driven.

The remaining investigation should therefore target **warm per-token
cost**:

- **H3 (shm-size cap)** — the vllm container starts with default
  64 MB `/dev/shm`. Ray actors within the container use /dev/shm
  for Gloo collectives and any shm-backed IPC; if the cap throttles
  a per-forward communication path, that would show up as a warm
  TPOT hit. Raise `--shm-size` and re-bench; if warm TPOT drops,
  H3 is confirmed.
- **H4 (ipc=host)** — the container isolates IPC namespace. Some
  NCCL FD-exchange paths (abstract UDS) route through
  `/dev/shm`-backed semaphores and shared-memory regions. With
  `--ipc=host`, the container shares the host's IPC namespace;
  if warm TPOT drops, H4 is confirmed.

Both tests can coexist with the existing MGC bench harness by
adding a second multi_gpu_container variant script in
`exp_c_container_overhead/scripts/`. H3 and H4 should be run
**warm** (add the warm-bench step the harness already supports via
`scale 4 → bench`) because that's the signal of interest.

### One sub-finding worth flagging

The cycle's cold native DP=4 number (139.43) is substantially lower
than the warm number (155.12) — a 10% warmup benefit on native.
Session-2's H5 "native variance" was essentially sampling from the
*cold* distribution, which is noisier than warm. The H5 variance
estimate (CV 6%) is an upper bound on steady-state variance — warm
variance is probably meaningfully smaller. This further explains
why §5.7's 171.23 looked high: it was a native-cold measurement
that happened to land closer to warm than typical.

If we ever re-do the H5 measurement to support the report's §5.7,
warm benches would give tighter numbers. But that's a methodology
improvement for Exp A3, not part of Exp C's mandate.

### Status

- **H1 closed.** Cold TTFT blow-up confirmed, but is one-time
  setup cost — does not explain the throughput gap. Warm gap is
  TPOT-driven, at −14% throughput / +17% per-token.
- **Next:** H3 (shm-size) and H4 (ipc=host) are the candidates for
  the warm TPOT overhead. H2 (NCCL init) is likely part of the
  cold TTFT blow-up but doesn't affect the warm gap, so I'll
  triage H2 briefly and move to H3/H4 as the primary remaining
  investigation.
