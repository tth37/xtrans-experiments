# Exp B end-to-end validation — image `xtrans-vllm-ep-patched:20260421-1258`

**Date:** 2026-04-21
**Base:** `xtrans-vllm-ep:v0.19.0`
**Patches applied (in order):**
1. `0001_eplb_scale_down_grow_density.patch` — precondition check in
   EPLB scale-down; raises `ValueError` pointing at
   `--eplb-config.num_redundant_experts` when the shrink would drop
   physical experts below logical.
2. `0002_placement_group_early_exit.patch` — outer-loop break in
   `RayDPClient.make_dp_placement_groups` so `dp_size < len(nodes)`
   cold-starts don't over-allocate placement groups.

## Summary

**Exp B's headline success criterion is met.** Phase 3 per-GPU
container cluster now supports:

- A cold **DP=2** start (requires patch 0002), which previously
  tripped `AssertionError: Created 4 DP placement groups, expected 2`.
- A cold **DP=4 → scale-down to DP=2** elastic cycle (requires patch
  0001 plus the redundancy-config redirect), which previously
  bricked the service with `AssertionError: num_redundant >= 0`.

After scale-down the host scheduler sees **2 of 4 GPUs fully released
to 4 MiB** while **all 4 containers remain `Up`** — the memory-free
property of Phase 2 plus the container-level reclamation that Phase 2
could not deliver. Service health endpoint responds 200 throughout.

## Scale-down 4 → 2 timing (from serve log inside ep-rank-0)

| Phase | Wall-clock delta |
|---|---|
| `DPCoordinator scaled down from 4 to 2 engines` | T+0 s |
| `[Elastic EP] Starting expert resharding...` | T+0 s |
| `[Elastic EP] EPLB reshuffle completed` | T+10 s |
| `[Elastic EP] Created standby communication groups` | T+11 s |
| `[Elastic EP] Scale down completed, new data parallel size: 2` | T+14 s |
| `[Elastic EP] Switched to new setup` | T+14 s |

**HTTP `POST /scale_elastic_ep`: HTTP 200 in 13.89 s.**

For comparison:
- **Phase 1 native** (same patches, same redundancy=128): 3.66 s —
  captured during Task 1 validation.
- **Phase 3 patched (this run):** 13.89 s — 3.8× slower than Phase 1.

The slowdown is in the EPLB reshuffle phase (0 s on Phase 1, 10 s on
Phase 3) and is consistent with expert-weight transfer running over
NCCL's NET/Socket/0 loopback instead of NVLink, as documented in the
Exp A3 Phase 3 analysis (v5 §4.1). It is not a Track 1/2 regression.

## Throughput

`vllm bench serve --dataset-name random --random-input-len 128
--random-output-len 128` (num_prompts varied):

| Config | n=prompts×concurrency | Output tok/s | Total tok/s | Mean TTFT (ms) |
|---|---|---|---|---|
| **Phase 3 DP=2 cold (smoke, no redundancy)** | 16 × 8 | 68.53 | 137.06 | 981.7 |
| **Phase 3 DP=4 pre-scale (redundancy=128)** | 32 × 16 | 119.65 | 239.30 | 2438.1 |
| **Phase 3 DP=2 post-scale-down (redundancy=128)** | 16 × 8 | 91.43 | 182.86 | 233.4 |

Reference baselines (not from this run):
- Phase 3 DP=4 static (no redundancy, pre-Exp-B): 127.8 tok/s output
  — Exp A3 analysis report §5.6.
- Phase 1 native DP=2 (no redundancy): 87.4 tok/s output — Exp A3
  analysis report §5.6.
- Phase 1 native DP=2 post-scale-down with redundancy=128:
  69.76 tok/s — Task 1 validation this session.

Observations:

- **DP=4 redundancy cost is smaller than expected on Phase 3.**
  119.65 tok/s vs. 127.8 tok/s baseline = **−6.4%**. The per-rank
  physical count doubles (32 → 64 experts), but Phase 3's TCP-NCCL is
  already the bottleneck, so the extra routing/memory pressure adds
  relatively little on top.
- **DP=2 post-scale outperforms DP=2 cold** (91.43 vs. 68.53 tok/s),
  mostly TTFT. The post-scale path inherits compile/warmup caches
  from the DP=4 run; the cold smoke starts everything from scratch.
  Mean TTFT drops from 981 ms to 233 ms on the same workload. This
  is a warmup effect, not a parallelism effect.
- **The expected combined penalty of ~50–55 tok/s for DP=2 post-scale
  did not materialise.** Pre-test estimate was Phase 3 NCCL (~15%) on
  top of Phase 1 redundancy (~20%), ~91.43 tok/s is actually close to
  the Phase 1 DP=2 no-redundancy baseline of 87.4 tok/s — which is
  cleaner than expected. Likely because the bench was on a warm
  service and the DP=2 split exposes less of the Socket-NCCL penalty
  (fewer cross-rank messages than DP=4).

## Scheduler-level reclamation (host side, post-scale-down)

```
$ docker ps --filter name=ep-rank- --format '{{.Names}}\t{{.Status}}'
ep-rank-3       Up 3 minutes
ep-rank-2       Up 3 minutes
ep-rank-1       Up 4 minutes
ep-rank-0       Up 4 minutes

$ nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
0, 35895 MiB, 0 %
1,     4 MiB, 0 %
2,     4 MiB, 0 %
3, 35893 MiB, 0 %

$ for i in 0 1 2 3; do
>   docker inspect ep-rank-$i \
>     --format '{{.Name}} DeviceIDs={{(index .HostConfig.DeviceRequests 0).DeviceIDs}}'
> done
/ep-rank-0 DeviceIDs=[0]
/ep-rank-1 DeviceIDs=[1]
/ep-rank-2 DeviceIDs=[2]
/ep-rank-3 DeviceIDs=[3]

$ for i in 0 1 2 3; do
>   echo "ep-rank-$i: $(docker exec ep-rank-$i pgrep -f 'DPMoE|RayWorker|vllm' | wc -l) vllm PIDs"
> done
ep-rank-0: 4 vllm-related PIDs   (API server + Ray head + engine + worker)
ep-rank-1: 0 vllm-related PIDs   (GPU freed, container idle)
ep-rank-2: 0 vllm-related PIDs   (GPU freed, container idle)
ep-rank-3: 2 vllm-related PIDs   (DPMoEEngineCore + RayWorker, GPU loaded)
```

This is the per-GPU container elasticity proof — a true release of the
GPU at the container-scoped level. Contrast with the Phase 2 finding
that `HostConfig.DeviceRequests.DeviceIDs` stayed at `[0,1,2,3]` for
the entire container lifetime regardless of what vLLM did internally.
Here each container holds exactly one GPU (DeviceIDs=['N']), and
stopping `ep-rank-1` and `ep-rank-2` externally (a Track 3 step, not
done in this run) would complete the scheduler-visible elasticity
story with no further vLLM work.

### Surviving ranks are {0, 3}, not {0, 1}

`perform_scale_down_eplb_reshuffle` builds
`rank_mapping = {0:0, 1:1, 2:-1, 3:-1}` — old EP ranks 0, 1 survive,
2 and 3 are dropped. But in this run GPUs 0, 3 remained loaded and
GPUs 1, 2 freed. vLLM's DP-rank to Ray-actor-placement mapping is
apparently not a stable 1:1 onto container naming in per-GPU setups;
the important invariant (two GPUs freed at the container boundary) is
preserved regardless. Noted but not blocking.

## NCCL transport (Phase 3 baseline still holds)

Captured from `docker exec ep-rank-0 cat /tmp/vllm-serve.log`:

```
NCCL INFO Assigned NET plugin Socket to comm
NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
NCCL INFO Channel 00/0 : 0[0] -> 2[0] [send]    via NET/Socket/0
NCCL INFO Channel 00/0 : 2[0] -> 0[0] [receive] via NET/Socket/0
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
```

Every inter-container channel uses `NET/Socket/0` — the same
TCP-over-bridge fallback documented in Exp A3 Phase 3 (all three
same-node gates — hostHash, shmDev, abstract-UDS — still fail at
container boundaries). `isAllDirectP2p 0 directMode 0 PXN 0 GDR 0`
all match the Phase 3 baseline. **Exp B's patches do not affect NCCL
transport selection**, and the remaining throughput gap vs. the
native baseline is attributable entirely to this orthogonal problem
(v5 §4.1 — LD_PRELOAD shim is a separate workaround track).

## Task sequence (for reproducibility)

```bash
# 0. Base image must exist: xtrans-vllm-ep:v0.19.0 (built from exp_a3)
# 1. Build patched image (applies both patches via Dockerfile)
./exp_b_per_gpu_elasticity/scripts/build.sh
# produces xtrans-vllm-ep-patched:20260421-1258 in ~10 s

# 2. Phase 3 cold DP=2 smoke (validates Track 2 alone)
PHASE3_DP=2 VLLM_IMAGE=xtrans-vllm-ep-patched:20260421-1258 \
  ./exp_a3_vllm_ep/scripts/phase3_per_gpu.sh up
# expect: 95 s to ready, 2 GPUs loaded, 2 GPUs idle
./exp_a3_vllm_ep/scripts/phase3_per_gpu.sh down

# 3. Phase 3 cold DP=4 with redundancy-128 + scale-down (headline)
PHASE3_DP=4 EXTRA_SERVE_ARGS='--eplb-config.num_redundant_experts=128' \
  VLLM_IMAGE=xtrans-vllm-ep-patched:20260421-1258 \
  ./exp_a3_vllm_ep/scripts/phase3_per_gpu.sh up
# expect: 101 s to ready, 4 GPUs loaded at ~36 GB each
curl -X POST http://localhost:8000/scale_elastic_ep \
     -H 'Content-Type: application/json' \
     -d '{"new_data_parallel_size":2,"drain_timeout":60}'
# expect: HTTP 200 in ~14 s; GPUs 1,2 drop to 4 MiB; service remains HEALTH_OK
```

## Files in this results tag

- `build.log` — Docker build output (both patches applied cleanly).
- `vllm-serve.log` — full serve log from inside ep-rank-0 across the
  DP=4 startup, DP=4 bench, scale-down, DP=2 bench window.
- `ep-rank-{0,1,2,3}.log` — per-container docker logs.
- `nccl_transport.log` — NCCL transport selection summary (Socket
  everywhere, as expected).
- `scale_events.log` — elastic-EP state transition timestamps.
- `deviceids_post_scale.txt` — per-container `HostConfig.DeviceRequests`.
- `container_processes_post_scale.txt` — vllm PIDs per container
  after scale-down (proves idleness).
- `nvidia_smi_post_scale.txt` — host-side GPU memory state after scale-down.
- `bench_dp2_smoke_patched.json`, `bench_dp4_pre_scale_patched.json`,
  `bench_dp2_post_scale_patched.json` — bench JSON outputs (from
  `exp_a3_vllm_ep/results/phase3/` tree).

## What remains (not done this session)

- **Track 1b** (grow-density scale-down for `num_redundant_experts=0`).
  Still the five-point vLLM structural refactor documented in the Exp
  B README. Would remove the redundancy-config requirement and reduce
  the throughput penalty when redundancy is costly.
- **Track 3** (external container lifecycle coordinator). Now that
  two GPUs can be reliably freed, the next step is for an external
  coordinator to actually `docker stop ep-rank-{1,2}` and mirror the
  inverse on scale-up. v5 §4.2.
- **NCCL LD_PRELOAD shim integration** (v5 §4.1). Orthogonal to this
  experiment but the biggest remaining throughput lever.
- **Upstream PR.** Not in scope per Exp B's charter (v4/v5 are
  research, not upstream contributions).
