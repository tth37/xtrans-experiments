# H2: NCCL init overhead in-container

**Status:** Open, not yet measured.

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

(empty)
