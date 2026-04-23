# H2: NCCL init overhead in-container

**Status:** Closed as minor. NCCL-group init duration during the
scale-up 2→4 event is identical in both regimes (19 s each, measured
from the 2026-04-23 native_n3 and mgc_n10 logs). H2's "container
NCCL init is measurably longer" prediction is not supported. The
14 s container overhead on scale-up wall-clock is in Ray's
placement-group / actor-coordination / broadcast phase, not in
NCCL communicator init proper.

This doesn't rule out per-request first-forward NCCL costs
contributing to the cold TTFT blow-up (~1400 ms MGC vs ~540 ms
native on bench 1), but that can only be resolved with
NCCL_DEBUG=INFO instrumentation, and the overall bounded nature
of cold TTFT (bench 2 is within 4% of native) makes the
incremental value of that measurement low.

## Hypothesis

Inside a container, NCCL's same-node detection (hostHash, /dev/shm
st_dev, abstract UDS for cuMem IPC) runs through a full
discovery/probe sequence at each communicator init. Even though the
gates *pass* in the single multi-GPU-container setup (so steady-state
uses NVLink, not Socket), the **probing cost** itself may be larger
than on native where the checks either short-circuit or use cached
host-side state. Elastic EP creates stateless groups that reinit NCCL
per scale event, amplifying any init-phase cost.

## Planned measurements

1. **NCCL_DEBUG=INFO timing.** Re-run both regimes with
   `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT`. Grep the init-phase
   timestamps in the NCCL log: from `NCCL INFO Bootstrap`
   to `NCCL INFO ncclCommInitRank complete` is the init window.
   Compare native vs container.

2. **Repeat init cost.** In Elastic EP, each scale event creates a
   standby group. Capture the
   `[Elastic EP] Created standby communication groups` timestamp
   and subsequent `Switched to new setup` — that delta is dominated
   by NCCL re-init. Compare between regimes.

3. **Counter-check: steady-state NCCL.** Steady-state at DP=4 should
   use NVLink / SHM in both regimes (§5.7 confirms the container
   picks same-node transport). If H2 is correct, TPOT gap should be
   small (+20% observed is not huge). If TPOT is much larger than
   expected after controlling for other hypotheses, steady-state NCCL
   differs too, expanding H2's scope.

## Expected outcomes

- If H2 dominates: container's NCCL init-phase wall-clock is
  measurably longer than native's (probably several hundred ms per
  init). Elastic EP's per-scale reinits amplify that.
- If H2 is minor: init times are comparable between regimes; move on.

## Status notes

### 2026-04-23 02:10 — Triage from existing logs

Extracted timings from `results/variants/native_n3/serve.log` and
`results/variants/mgc_n10/container.log`. Both regimes ran the same
2→4 scale-up during their 12-bench protocols. Phase breakdown:

| Phase | Native | MGC | Δ |
|---|---:|---:|---:|
| Reconfig request → "Created standby communication groups" (NCCL init for new workers) | 19 s | 19 s | **0** |
| Standby → "Transferred weights to new workers" | 2 s | 3 s | +1 |
| Weights → "Switched to new setup" (broadcast, placement, actor sync) | 6 s | 19 s | **+13** |
| **Total scale-up** | 27 s | 41 s | +14 s |

So the NCCL-communicator init itself is unchanged between regimes —
the 19-second "Created standby communication groups" wall-clock is
fixed. The 14-second container penalty on scale-up is entirely in
the post-NCCL phase, where Ray does its placement-group + actor +
broadcast work. That's a Ray / container-boundary phenomenon, not
an NCCL one.

H2's H-specific claim about NCCL init is therefore falsified. The
larger container scale-up time is real (consistent with §5.5.2's
"+51%" Ray actor spawn finding in Exp A3), but it's mislabelled if
called "NCCL init overhead." A more accurate label would be "Ray
placement-group + actor-spawn overhead inside a container."

H2 closed.
