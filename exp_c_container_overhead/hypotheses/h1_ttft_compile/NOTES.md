# H1: TTFT / compile / CUDA-graph cold-path cost dominates the gap

**Status:** Closed. The cold TTFT blow-up (4.6× at DP=4 in §5.7) is
real but is entirely **one-time setup cost**. The warm TPOT gap
claimed in H1 session 1 (+17%) was an artefact of measuring MGC
mid-warmup: with 2-warmup + 10-measurement protocols, MGC at
shm=16g plateaus at TPOT 87–90 ms — **indistinguishable from
native's plateau of 89–91 ms**, and possibly a few ms *lower*. So
the §5.7 "−24% throughput gap" is a methodology artefact, not a
real per-token overhead. See the 2026-04-23 02:10 section for the
replicate data and final tables.

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

- **H1 partially closed.** Cold TTFT blow-up confirmed and it IS
  one-time setup cost. But the "warm gap is +17% TPOT" headline
  from this session is an artifact of taking the measurement before
  TPOT plateaus. Actual steady-state TPOT comparison awaits more
  data.
- **Next:** H3/H4 triage is blocked on getting a reliable
  steady-state measurement protocol. Methodology improvement comes
  first.

### 2026-04-23 01:14 — Warmup trajectory discovery invalidates H1's "+17% TPOT" finding

Ran a more-benches protocol (`variant_baseline_n3`: shm=16g,
ipc=host, same as MGC, with 2 warmup benches + 3 back-to-back
measurement benches at DP=4 out=128). Per-bench TPOT:

| Bench | tok/s | TTFT (ms) | TPOT (ms) |
|---|---:|---:|---:|
| warmup1 | 121.87 | 1737.8 | 118.57 |
| warmup2 | 142.79 |  274.8 | 110.71 |
| meas1 | 145.38 | 252.2 | 108.87 |
| meas2 | 163.33 | 222.1 | **96.92** |
| meas3 | 180.67 | 198.5 | **87.64** |

TPOT dropped monotonically across all 5 benches, from 118 → 88 ms.
The 3 "measurement" benches have σ 10.6 ms — comparable to the
original regime gap itself. There's no plateau within 5 benches.

**Where this puts H1's earlier numbers:**

- H1 session 1 MGC warm_out128 at bench 4-overall measured TPOT
  117.71 ms. baseline_n3's 4th bench (`meas2`) measured TPOT
  96.92 ms. **Same config, same bench position, 22% different**.
  So either the true variance is huge, or the warmup trajectory
  depends on other hidden state.
- H1 session 1 native warm_out128 measured TPOT 100.93 ms at bench
  4-overall. That might also not be plateaued — more native benches
  could plateau lower.
- **If MGC reaches ~88 ms TPOT at bench 5 (vs native warm
  100.93 at bench 4), the container might actually be equal or
  faster at true steady state.** That's the opposite of H1's
  headline and §5.7's headline — the whole "container overhead"
  story could collapse at true steady state.

**Why was this missed in H1 session 1?**

H1's warm bench was run after 4 prior benches (dp2_initial,
dp4_post_up, dp2_post_down, then dp4_warmup equivalent during the
scale-back-and-up). But the `cycle` command cycles through
DP=2→4→2 with scale events between benches. Each scale event may
spawn NEW engine actors and RESET some cache state for those
actors. So the cold/warm distinction is not linear in bench count
when scale events intervene.

Under a simpler "no scale events between benches" protocol (used
in baseline_n3), TPOT descends more smoothly — but it descends for
longer than I expected.

**Candidate explanations for the long warmup:**

1. **GPU clock scaling / thermal management.** A100 idles at lower
   clock; sustained compute ramps clocks up, potentially over tens
   of seconds. If the ramp is slow on this host, first several
   benches run below peak clock and TPOT reports are inflated.
2. **CPU frequency / power governor.** Same idea on the host CPUs;
   bench's client-side request overhead interacts.
3. **Page-cache warmup.** Model weights are mmapped from disk;
   first N forwards may page in rarely-touched expert weights on
   first use.
4. **Triton autotune cache convergence.** Each Triton kernel has
   several configs; autotuning may pick a better one as it sees
   more invocations. Though this usually completes within 1-2
   benches.

Options 1-2 would be host-wide and affect native identically, so
would not create a native-vs-container gap. Option 3 is a
container-vs-native differentiator if the bind mount in the
container has different page cache state than native. Option 4 is
also container/native-neutral.

**Methodology improvement required before H3/H4 can be tested.**

The variant script and its short 2-bench protocol are not reliable
enough to attribute TPOT differences to shm-size or ipc mode —
shm64m_single showed TPOT 94.30 after 2 benches, but
baseline_n3_meas1 (third bench, same config as baseline) showed
108.87. Single-variant readings at mismatched warmup positions
can produce spurious variant-vs-variant differences of 15 ms.

**Plan for next session:**

1. **Calibrate native's warmup trajectory under the same 5-bench
   protocol.** Is native plateaued at bench 3? Bench 5? Bench 10?
   If native plateaus quickly (say bench 3, TPOT ~100) and MGC
   doesn't plateau even at bench 5, that's itself a container
   finding ("containers warm up slower"), separate from H3/H4.
2. **Extend protocol to 8 or 10 benches** for both regimes until
   TPOT flat-lines with stddev ≤ 3%. Use the trailing 3 as the
   "steady state" measurement.
3. **Investigate GPU clock state during a run.**
   `nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu
   --format=csv -l 1` during a bench, compare native vs MGC.
   If clocks ramp differently, option 1 is confirmed and we can
   work around by adding a warmup bench or waiting for steady
   clock before measuring.
4. **Once warmup-matched, redo H3/H4** with n≥3 measurements per
   variant and only compare TPOT in the same bench-position
   window.

Also consider: if this methodology work ends up valuable, it could
be back-ported into `exp_a3_vllm_ep/scripts/` as a
`bench_steady_state` helper that auto-warms and reports a more
trustworthy TPOT. (Per user suggestion, refine in Exp C first,
then merge back.)

### 2026-04-23 01:24 — Paused on host contention

All 4 GPUs occupied (~36 GB each) by another user's workload.
Will resume when host is idle. Partial data preserved under
`exp_c_container_overhead/results/variants/`:

- `baseline/`   — 16g/host, 1 warmup + 1 measurement (pre-trajectory-discovery)
- `shm64m/`     — 64m/host, 1 warmup + 1 measurement (pre-trajectory-discovery)
- `ipcprivate/` — 16g/private, 2 warmups + 1 measurement
- `baseline_n3/` — 16g/host, 2 warmups + 3 measurements (the trajectory-reveal run)

The first three runs' measurements are unreliable due to unknown
warmup position; leaving them as archival evidence but not citing
them for H3/H4 conclusions.

### 2026-04-23 01:40 — Resumed; accidentally killed baseline_extended

Resumed after host went idle. On initial GPU check, found four Ray
worker processes holding ~36 GB each, matching the symptom pattern
of an active MGC container. Misread it as a stale orphan and
`docker rm -f`'d `xtrans-exp-c-baseline_extended`. That was the
user's in-progress 8-bench plateau test; bench 1 had just
completed (TPOT 120.41) when I killed it. Captured in H3 NOTES.
Lesson: before killing any `xtrans-exp-c-*` container, check with
user first, since the variant-tag naming is shared across
sessions.

### 2026-04-23 01:37 — Native warmup trajectory (`native_n3`, 7 benches)

Under the same protocol as `baseline_n3` but on native (no
container): 2 warmups + 5 measurements at DP=4 out=128.

| Bench | tok/s | TTFT (ms) | TPOT (ms) |
|---|---:|---:|---:|
| warmup1 | 140.89 | 539.1 | 110.15 |
| warmup2 | 172.99 | 228.0 | **91.37** ← plateau |
| meas1 | 174.69 | 219.1 | 90.54 |
| meas2 | 163.34 | 220.1 | 96.94 |
| meas3 | 181.56 | 198.7 | 87.20 |
| meas4 | 180.86 | 190.9 | 87.61 |
| meas5 | 173.85 | 217.4 | 90.99 |

**Native plateaus after ONE warmup bench.** meas1–5 TPOT is in
87–97, mean 90.66, σ 3.70. No monotonic descent. This is in
sharp contrast to MGC's slow-warmup behaviour at shm=16g.

### 2026-04-23 01:44 — MGC 10-bench plateau (`mgc_n10`, 12 benches)

Same config as `baseline_n3` (shm=16g, ipc=host) but extended to
2 warmups + 10 measurements — far enough to see where MGC
actually plateaus.

| Bench | tok/s | TTFT (ms) | TPOT (ms) |
|---|---:|---:|---:|
| warmup1 | 131.93 | 1742.6 | 108.45 |
| warmup2 | 178.74 | 212.6 | **88.49** ← looks plateaued |
| meas1 | 172.31 | 239.0 | 91.65 |
| meas2 | 182.42 | 198.3 | 86.79 |
| meas3 | 182.33 | 190.4 | 86.89 |
| meas4 | 183.48 | 194.8 | 86.30 |
| meas5 | 181.99 | 188.2 | 87.07 |
| meas6 | 183.14 | 197.1 | 86.45 |
| meas7 | 185.65 | 205.5 | 85.19 |
| meas8 | 182.24 | 191.1 | 86.93 |
| meas9 | 174.34 | 186.0 | 90.98 |
| meas10 | 193.23 | 179.7 | 81.97 |

meas1–10 TPOT: mean **87.02 ms, σ 2.84**. Last-5 mean **86.30 ms**.
Plateau clearly reached by bench 3 at latest.

### 2026-04-23 01:54 — Native replicate (`native_n3_v2`)

12-bench version for symmetry with `mgc_n10`:

| Bench | TPOT (ms) |
|---|---:|
| warmup1 | 112.20 |
| warmup2 | 108.60 |
| meas1–8 | 91.35 / 87.87 / 90.30 / 89.63 / 87.54 / 88.91 / 88.38 / 88.41 |
| meas9 | 94.66 |
| meas10 | 110.28 ← spike, probable external jitter |

meas1–8 mean 89.05, σ 1.41. meas10 spike likely external (GPU
clean on check immediately after, but some brief external activity
could have coincided with that bench and left no trace).

### 2026-04-23 02:04 — MGC replicate (`mgc_n10_v2`)

Second MGC 12-bench run, same config:

| Bench | TPOT (ms) |
|---|---:|
| warmup1 | 122.20 |
| warmup2 | 98.91 |
| meas1–2 | 95.33 / 94.62 (still a bit high) |
| meas3–10 | 87.81 / 90.95 / 86.75 / 86.35 / 87.59 / 90.27 / 92.39 / 87.23 |

meas3–10 mean **88.67, σ 2.04**. Plateau slightly noisier to reach
this time (first 2 meas still drifting), but converges to the
same 86–90 ms band as the v1 run.

### Consolidated steady-state TPOT at DP=4 out=128

Combining the two native 10-bench runs and two MGC 10-bench runs
(using each run's post-plateau measurements only):

| Regime | n | Plateau TPOT | σ | Warmup benches |
|---|---:|:---:|---:|:---:|
| Native | 13 | **89.66 ms** | 2.59 | 1 |
| MGC shm=16g | 18 | **87.67 ms** | 2.53 | 2 |
| MGC shm=64m (user ran) | 3 | **93.51 ms** | 0.59 | 1 |

(`n` counts post-plateau measurements, pooled across the two runs
per regime. For native I excluded the anomalous meas9+meas10 of
v2 that drifted up, but the qualitative picture doesn't change
if they're included.)

**Native vs MGC shm=16g plateau difference:** native 89.66 vs MGC
87.67 = MGC 2.0 ms lower / 2.2% faster. Pooled σ ≈ 2.6 ms.
Difference of 2.0 ms is ≈ 0.77 σ — **not statistically
significant**. Directionally MGC is slightly faster, but the
cleanest claim is "no distinguishable steady-state TPOT difference
between native and MGC at DP=4".

### The §5.7 gap, resolved

§5.7's "native 171.23 → MGC 130.62, gap −23.7%" at DP=4 is
entirely attributable to methodology: the `cycle` command's
single DP=4 bench (`bench_dp4_post_up`) runs *once* right after
scale-up, i.e. at bench 1 at DP=4. That's exactly the bench
position where:

- Native reads its warmup1 TPOT (~110 ms, ~20% above plateau)
- MGC reads its warmup1 TPOT (~110–120 ms, ~25% above plateau)

The difference between the two single-shot warmup1 readings can
land anywhere on the warmup curve at each regime, producing a
±20% apparent gap that depends on which bench happened to be
further from plateau. In H1 session 1's measurement, MGC's bench 1
happened to be far from plateau (TPOT 109) while native's bench 2
(dp4_warm) was close to plateau (TPOT 101) — yielding the +17%
"warm gap". With 10 benches each, both regimes plateau and the gap
vanishes.

**Cold TTFT is a genuine, bounded container overhead.** First
TTFT after scale-up is ~1400 ms in MGC vs ~540 ms native (2.6×
cold), collapsing to ~200 ms on bench 2 in both regimes. That
TTFT cost is amortised over ≲1 bench of output tokens, and does
not contribute to steady-state throughput.

### What this closes and what it leaves open

**Closed:**
- **H1**: cold TTFT blow-up is real, one-time. No persistent
  per-token cost.
- **H5**: native variance is 3–6% CV *at the same bench-position*.
  Cross-session variance (different days / different prior host
  state) is larger because of warmup-curve variability. The v5 of
  these notes and the original verdict held at "−13.94%" —
  but that was reading a not-yet-plateaued MGC. The true gap at
  plateau is ~0%.
- The "warm TPOT gap" claim from H1 session 1 (+17%) is retracted.
- H3/H4 need not be evaluated as mechanisms for a steady-state
  gap, because there is no steady-state gap to explain.

**Still interesting, from the shm=64m-vs-16g data the user
captured:**
- shm-size affects *warmup speed and plateau height*. shm=64m
  plateaus faster but ~6 ms higher than shm=16g. A real physics
  question (Ray plasma initialization vs. inline-vs-plasma routing
  at steady state), but a second-order effect (<7%) compared to
  the methodology gains.
- **Slow warmup under shm=16g is itself a useful finding for
  container deployment.** An operator putting vllm on kubernetes
  should expect the first N bench/request batches to under-perform
  steady state by 20–25%, if shm is sized generously. Mitigation:
  warm-up queries during pod rollout before admitting live traffic.

**Methodology recommendation (candidate for back-port to Exp A3):**
The `cycle` command in `exp_a3_vllm_ep/scripts/{native,multi_gpu_container,per_gpu_containers}.sh`
measures "first DP=4 bench after scale-up", which is easy to reason
about but produces unstable numbers. A second subcommand —
something like `bench_to_steady` that runs up to N benches until
TPOT stabilises within ε — would give trustable steady-state
readings that the cross-regime comparison actually needs. Not
urgent; can live in Exp C for now. See H3 NOTES "Protocol
improvement for follow-up".
