# H5: Bench variance / host-load interference (baseline noise calibration)

**Status:** Closed. The §5.7 headline "−24% gap" is ~60% signal, ~40%
native-side baseline drift. Session-2 apples-to-apples mean-to-mean gap
at DP=4 is **−13.94%** (native 148.91 → MGC 128.14 tok/s), at 2.33σ
(native's σ) separation — outside variance, genuinely caused by
containerisation. Investigation continues to H1 (TTFT blow-up, still
4.13× in session 2 data), which is the primary mechanism underlying
this 14% gap.

## Hypothesis

node192 is a shared host. Other users (confirmed during the 2026-04-22
sessions) spin up loads on various GPUs at unpredictable times. The
bench numbers might be partially (or mostly) explained by host-load
variance rather than in-container overhead. If the gap varies by ±15%
run-to-run on the native regime alone, the observed −24% gap between
regimes is within noise.

## Planned measurements

1. **Native regime, 3 consecutive runs, same session, no other
   containers or users active.** Record DP=4 output tok/s each time.
   Compute mean and stddev.

2. **Multi-GPU container regime, 3 consecutive runs, same conditions.**
   Same stats.

3. **Criterion:** if native stddev is comparable to the inter-regime
   gap, H5 explains most of the observation and the other hypotheses
   are investigating noise.

## Expected outcomes

- Typical case: native stddev is small (few percent), container stddev
  similar. Gap is real. H5 reports "baseline variance is ~Nx%, gap is
  real within that noise floor" and moves on to H1-H4.
- Surprising case: native itself is noisy. H5 becomes the primary
  explanation and the rest of the investigation tightens its
  methodology (more runs, longer benches, off-hours testing, etc.)
  before drawing any gap-causation conclusions.

## Session 1 (2026-04-22) — blocked by concurrent user activity

Before any H5 result could be measured, the session ran into an
external confound: **another user was actively benchmarking on GPUs
0 and 1 in short iterations throughout the window**, with the GPUs
briefly returning to 4 MiB between iterations — long enough to pass
`require_gpus_free` but not long enough to complete a vLLM startup
without interference. This defeated variance measurement before it
began.

### What was attempted

| Attempt | Time (local) | Outcome | DP=4 tok/s | Note |
|---|---|---|---|---|
| Native run 1 (original) | 12:14 | **Failed** | n/a | NCCL sanity-check `all_reduce` raised `RuntimeError: NCCL error: unhandled cuda error` during `PyNcclCommunicator.__init__`. Happened *after* both Ray engine actors had allocated their model weights. |
| Native run 1 (retry) | 12:18 | Completed | 167.54 | Successful full 2→4→2 cycle. DP=4 = 167.54 tok/s, TTFT 402 ms, TPOT 93.0 ms. But see "provenance" below: the host was not verifiably quiet, just between other-user sweep iterations. |
| Native run 2 | 12:23 | **Failed** | n/a | `ValueError: Free memory on device cuda:0 (21.99/39.49 GiB) on startup is less than desired GPU memory utilization (0.9, 35.54 GiB).` vLLM's `request_memory` aborted in `v1/worker/utils.py:413` before any NCCL init. The other user's process PID 4002704 was holding 17 GB on GPU 0 by the time vLLM tried to allocate. |

### Evidence that the host was not quiet

Two independent GPU holders observed during this session, both owned
by another user:

```
# During run 1 failure at ~12:14:
compute-apps: PID 3945510 python, 19852 MiB on GPU-6e5259ec (GPU 0)
                         same PID, 17044 MiB on GPU-da37de5d (GPU 1)
```

Command line of PID 4002704 (captured during the run 2 failure
diagnosis, same process family as the 12:14 PID):

```
python benchmarks/bench_overlap_vs_prefetch.py --phase e2e \
    --device cuda:0 --remote-device cuda:1 --phase2-decode 80 \
    --phase2-warmup 15 --phase2-trials 1 \
    --prompt-offsets 200000,500000,800000,1200000,1600000,2000000,1800000 \
    --single-cell 4 32768 0.10 \
    --output-dir benchmarks/results/sweep_20260422-12
```

Later in the same session (12:25), a *different* binary from the same
user appeared:

```
PID 4029453 ./examples/14_ampere_tf32_tensorop_gemm_multigpu_new/... \
    800 MiB on GPU 0, 425 MiB on GPU 1
```

So the host is being actively used for multi-iteration GPU sweeps
across this window, with per-iteration durations short enough to leak
through `require_gpus_free`'s instantaneous check.

### Why the "successful" run 1 retry doesn't count as H5 data

Run 1 retry hit 167.54 tok/s at DP=4, which is close to §5.7's
reference 171.23. But we cannot claim that window was quiet — we can
only claim that *during our bench windows* the other user's sweep
happened not to trip us:

- GPU idle-checks pass between sweep iterations; vLLM may start during
  a gap, then bench during a later gap, while other iterations happen
  during steady-state we never observe.
- The bench_dp4_post_up JSON shows all 32 requests succeeded with
  mean TTFT 402 ms (vs §5.7's 387 ms) — suggestive of mild contention
  during first-token but not conclusive.
- A ~2% shortfall vs the §5.7 reference is within the plausible
  measurement-variance band we're trying to establish. Can't tell
  signal from interference with n=1.

Archive preserved under `native_run1_unverified_sweep_active/` in case
the next clean run's numbers match closely and we can use it as
supporting evidence.

Archive `native_run2_failed_memory_taken/` preserved as evidence of
the contention failure mode (contains the `bench_dp2_initial.json`
with 0 successful requests, a 0-byte `scale_to_dp4_*.log`, and the
truncated `serve.log` with the ValueError).

Archive `native_run1_failed/` (from the original 12:14 NCCL failure)
preserved similarly.

### What the failure modes tell us

Two distinct in-startup failure modes came from the same root cause
(concurrent GPU occupancy):

1. **Pre-NCCL `request_memory` abort** (run 2): cleanest failure mode.
   Triggered when ≥ ~3.5 GB of GPU 0 is already occupied — vLLM's 0.9
   utilization target of 35.54 GiB cannot be reserved.
2. **Mid-init NCCL `unhandled cuda error`** (run 1 original): murkier.
   Occurred after vLLM had already allocated 19.8 GB on GPU 0 and
   17.0 GB on GPU 1, during `PyNcclCommunicator.__init__`'s internal
   sanity-check `all_reduce`. Probably either the CUDA driver's peer-
   access setup tripped against the other user's allocation, or
   memory pressure from the concurrent allocator reached NCCL at a
   critical moment.

**Implication for the investigation going forward:** `require_gpus_free`'s
instantaneous check is insufficient protection against a neighbour running
short sweep iterations. See "Methodology improvements" below.

### Methodology improvements to apply before the clean re-run

- **Two-sample idle check** ~10 s apart before proceeding. Both must
  show < 100 MiB on all 4 GPUs. Mitigates but does not eliminate the
  between-iteration trap.
- **Snapshot `nvidia-smi --query-compute-apps` at run start and end**
  into each archive. Makes post-hoc contention detection trivial:
  if either snapshot shows a non-vllm PID on any GPU, the run is
  tainted by definition.
- **Abort immediately on bench completed < num_prompts**; preserves
  current behavior of leaving the archive but gives a clear signal.
- **Prefer off-hours testing** if possible — this user appears to be
  running a 20260422-12 sweep during mid-day hours.

These apply only if we decide to harden the harness; they're not yet
implemented.

## Session 2 (2026-04-22 14:40–15:01) — native side closed, MGC side partial

After the user signalled the host was quiet, three native cycles ran
cleanly back-to-back, with pre/post GPU snapshots confirming no
external process on any GPU during each cycle. Archives:
`native_run{1,2,3}/`, each with `run_meta.txt` containing pre/post
`nvidia-smi --query-compute-apps` snapshots.

One incidental setup finding during session 2 ramp-up: the very first
attempted run (14:40) aborted because a stale Ray head from session 1
was still bound to port 26379 — `native.sh stop`'s `ray stop --force`
doesn't reliably kill Ray heads from a prior shell session. A manual
`ray stop --force` cleared it. Not part of H5's mandate, but a real
operational gotcha worth capturing for when Exp C considers a harness
hardening PR: after a failed run, `ray stop` should be retried until
ports are actually free, or `native.sh start` should fail-fast with a
port-in-use message rather than silently letting Ray die.

### Native variance (n = 3 clean runs)

| Metric | run 1 | run 2 | run 3 | Mean | σ | CV% |
|---|---:|---:|---:|---:|---:|---:|
| DP=2 initial tok/s | 72.89 | 72.26 | 78.02 | 74.39 | 3.16 | **4.25** |
| DP=2 initial TTFT (ms) | 363.2 | 468.3 | 362.1 | 397.9 | 61.0 | 15.3 |
| DP=2 initial TPOT (ms) | 107.6 | 107.8 | 100.4 | 105.3 | 4.24 | 4.0 |
| **DP=4 post-up tok/s** | **158.72** | **146.67** | **141.33** | **148.91** | **8.91** | **5.98** |
| DP=4 post-up TTFT (ms) | 366.5 | 400.8 | 414.5 | 394.0 | 24.7 | 6.3 |
| DP=4 post-up TPOT (ms) | 98.7 | 106.7 | 110.8 | 105.4 | 6.2 | 5.8 |
| DP=2 post-down tok/s | 73.08 | 73.94 | 74.16 | 73.73 | 0.57 | **0.77** |
| DP=2 post-down TTFT (ms) | 240.7 | 229.3 | 242.7 | 237.5 | 7.23 | 3.04 |
| DP=2 post-down TPOT (ms) | 108.4 | 107.2 | 106.8 | 107.4 | 0.83 | 0.77 |

Three clear patterns:

1. **Native DP=4 is noticeably noisier than native DP=2**: CV 5.98%
   vs 0.77–4.25%. The DP=4 range (141.33 → 158.72) is 12.3% of the
   maximum. This says the noise floor is concentrated where the
   sensitive measurement sits.

2. **Post-scale-down DP=2 is the most stable configuration**
   (CV 0.77% on tok/s). Makes sense: by then the system is warm,
   caches primed, EPLB has rebalanced once, and the Ray cluster has
   settled. First-call (DP=2 initial) and scale-up (DP=4 post-up)
   both carry cold-/warming-state variance.

3. **The session-2 native DP=4 mean (148.91) is ~13% below the §5.7
   reference (171.23)**. §5.7's 171.23 sits at +2.5σ above session-2's
   distribution — a ≤1% probability sample if session-2 is
   representative. Something drifted on the host between §5.7 and
   this session. TPOT is the strongest systematic shift: 105.4 ms
   mean now vs 91.1 ms in §5.7 (+15.7%), consistent across all 3
   session-2 runs (low intra-session σ 6.2 ms).

### Multi-GPU container variance

MGC run 1 (14:58) aborted at scale-up 2→4 — `train_real_hf.py`
(external user PID 226389) allocated ~25 GB on GPU 3 mid-run, and
vllm's scale-up emitted `ValueError: Free memory on device cuda:0
(13.95/39.49 GiB) on startup is less than desired GPU memory
utilization (0.9, 35.54 GiB)`. DP=2 initial bench completed cleanly
at 69.94 tok/s before the scale event; that partial data point is
retained and consistent with the clean runs below. Archive:
`multi_gpu_container_run1/`.

User signalled the host was idle again at ~15:10. MGC runs 2 and 3
completed full cycles cleanly.

| Metric | r1 | r2 | r3 | Mean | σ | CV% |
|---|---:|---:|---:|---:|---:|---:|
| DP=2 initial tok/s | 69.94 | 66.06 | 69.10 | 68.37 | 2.04 | 2.98 |
| DP=2 initial TTFT (ms) | 1435.2 | 1430.1 | 1210.6 | 1358.6 | 128.2 | 9.44 |
| DP=2 initial TPOT (ms) | 103.9 | 110.7 | 107.0 | 107.2 | 3.40 | 3.17 |
| **DP=4 post-up tok/s** | — | **136.79** | **119.49** | **128.14** | **12.23** | **9.55** |
| DP=4 post-up TTFT (ms) | — | 1446.6 | 1807.3 | 1626.9 | 255.1 | 15.68 |
| DP=4 post-up TPOT (ms) | — | 106.4 | 120.7 | 113.55 | 10.06 | 8.86 |
| DP=2 post-down tok/s | — | 70.71 | 65.29 | 68.00 | 3.83 | 5.64 |
| DP=2 post-down TTFT (ms) | — | 597.5 | 1092.2 | 844.8 | 349.8 | 41.41 |
| DP=2 post-down TPOT (ms) | — | 109.3 | 114.8 | 112.0 | 3.93 | 3.51 |

Three observations:

1. **MGC DP=4 throughput is noisier than native's** — CV 9.55%
   (n=2) vs native 5.98% (n=3). n=2 is too few to trust the MGC σ
   tightly, but directionally MGC is less reproducible.
2. **MGC DP=4 TTFT is very noisy** — σ 255 ms on a 1627 ms mean
   (CV 15.7%). Two data points swing from 1447 ms → 1807 ms. The
   high noise at TTFT in MGC is itself a clue, and overlaps with
   what H1 is set up to investigate.
3. **MGC DP=2 post-down TTFT has the wildest variance** (CV 41%
   on n=2: 597 vs 1092 ms). Likely because post-scale-down is
   sensitive to whether EPLB rebalance/re-warmup landed cleanly;
   DP=2 post-down TTFT is already much higher than DP=2 initial's
   on native side too (§5.7 reported native DP=2 post-down as the
   warm state with low TTFT, which matches session-2 native's
   237 ms mean; MGC's 845 ms mean indicates the warm state is
   much worse inside the container).

### Session 2 final verdict

**At DP=4, session-2 mean-to-mean:**
- Native: 148.91 tok/s (n=3, σ 8.91)
- MGC:    128.14 tok/s (n=2, σ 12.23)
- **Gap: −13.94%** (separation 2.33σ on native's σ, 1.37σ on pooled σ)

Compared to §5.7's −23.7% headline, the apples-to-apples gap today is
~10 percentage points smaller. Where did those 10 pp go?

- **Native TPOT drifted +15.7% since §5.7** (91.1 → 105.38 ms, σ only
  6.2 ms across runs — not noise, a real systematic shift). That
  single shift explains essentially all of the native-side throughput
  drop (DP=4 throughput is TPOT-bound in this workload).
- **Native TTFT is essentially unchanged** (387 → 394 ms, +1.8%).
- **MGC TPOT barely drifted** (109.4 → 113.55 ms, +3.8%).
- **MGC TTFT dropped a bit** (1777 → 1627 ms, −8.4%).

So the 10 pp of "phantom gap" in §5.7's headline traces to
native TPOT slowing down ~14 ms/token between §5.7 and session 2.
That's a host-level drift (OS/driver/scheduler state?), orthogonal
to the container question. It isn't an H5 mandate to chase that
down, but it is a notable artefact to flag.

**H1 remains the real mechanism of the remaining ~14% gap.**
TTFT ratio (MGC / native) at DP=4:
- §5.7:     1777 / 387 = 4.59×
- Session 2: 1627 / 394 = 4.13×

TTFT ratio is roughly preserved — 4× TTFT inflation inside the
container. TPOT ratio (MGC / native) at DP=4:
- §5.7:      109.4 / 91.1 = 1.20× (+20%)
- Session 2: 113.5 / 105.4 = 1.08× (+8%)

The TPOT ratio narrowed between §5.7 and session 2, purely because
native TPOT rose while MGC TPOT was stable. MGC-specific TPOT
overhead is therefore smaller than §5.7 suggested (+8% rather than
+20%), while TTFT inflation is the same class of anomaly as before
(~4×). **The gap that remains to be explained is almost entirely
TTFT.** That's H1's domain.

### Decision rule outcomes

The notes above originally proposed:

> If native stddev is ≲ 3% and the regime gap reproduces at ≳ 20%,
> gap is real → start H1 next session.
> If native stddev is large (say ≥ 8%), H5 reframes the investigation.

Observed: native CV is 6.0% (intermediate), and the regime gap
reproduces at −13.94% (not 20%+). This sits in between the rule's
branches, so neither triggers cleanly. The actual conclusion is
subtler: the headline shrank on re-measurement, but the remaining
gap is 2.3σ separated — real, just smaller. **Go to H1 with the
revised target of ~14%, not 24%**, and note that TTFT dominates it.

### Implications for §5.7 of the Exp A3 report

The session-2 numbers should be cross-referenced into Exp A3's §5.7
(the one section Exp C is allowed to amend). At minimum, a note
that the §5.7 single-measurement snapshot undercounts variance
(±6% native, ±10% MGC) would save future readers from reading the
point-estimates as high-precision. A concrete addendum paragraph
and/or a session-2 overlay table can go in when the H1 conclusion
is ready, so the update is one atomic edit.

### Methodology improvements actually applied in session 2

Per the items noted earlier in these notes:

- **Two-sample idle check** — applied informally before the
  all-clear (10 s apart, both clean). Not embedded in the scripts.
- **Pre/post `nvidia-smi --query-compute-apps` snapshots** into each
  archive's `run_meta.txt` — applied to all session-2 runs.
  Confirms all 3 native runs were genuinely contention-free and
  pinpoints which concurrent PID tripped the one partial MGC run.
- **Abort on bench completed < num_prompts** — not yet automatic;
  the post-archive summary script surfaces it but nothing blocks
  the run.
- **Fail-fast on stale ray-head port-in-use** — new item surfaced
  during session 2 ramp. Candidate for a small fix to
  `common.sh:start()` in a follow-up commit.

## Status notes

- **2026-04-22 12:14–12:28** — session 1 attempts. Two failures, one
  unverifiable "success". Paused after user direction.
- **2026-04-22 14:40–15:01** — session 2 attempts. 3 native cycles
  cleanly collected. 1 partial MGC cycle (DP=2 succeeded,
  scale-up 2→4 blocked by external user on GPU 3).
- **2026-04-22 15:10–15:21** — session 2 continuation after host
  went idle again. 2 full MGC cycles collected cleanly.
- **Outcome**: H5 closed with verdict "−13.94% apples-to-apples gap
  at DP=4, dominated by TTFT blow-up, native side drifted since
  §5.7." Pivoting to H1.
