# Exp B: Enabling Per-GPU Container Elastic Scaling

**Status:** Workaround sandbox for research plan v5 §4.2, §4.3, §4.4.
**Research plan:** `../docs/research_plan_v5.html`
**Prerequisite experiment:** `../exp_a3_vllm_ep/` (Phases 1–3 complete;
this experiment builds directly on Phase 3's per-GPU container setup).

## Goal (One Sentence)

Make vLLM Elastic EP's `/scale_elastic_ep` API work **correctly and
reproducibly** in a per-GPU container cluster, by patching the vLLM
install inside a derived Docker image — preserving functional parity
with Exp A3 Phase 1 (native bare metal) while gaining the cluster-level
elasticity that Phase 3 demonstrated architecturally but couldn't execute.

## Why This Is The First v5 Experiment

From Exp A3 results:

- **Phase 1 (native):** elastic scaling works perfectly. 2→4→2 cycle,
  1.91× linear scaling, 3.5 s 503 window.
- **Phase 2 (multi-GPU container):** elastic scaling works *inside*,
  but Docker's `HostConfig.DeviceRequests.DeviceIDs` is immutable at
  `[0,1,2,3]` for the container's lifetime. The orchestrator cannot
  reclaim freed GPUs. This is the "memory-free but container-trapped"
  finding.
- **Phase 3 (per-GPU containers):** each container owns exactly 1 GPU,
  so the trap is eliminated architecturally — stopping a container
  genuinely frees a GPU at the scheduler level. But the scaling API
  itself is unusable:
  1. vLLM's Ray-placement logic at startup requires
     `--data-parallel-size-local 1` to not crash
     (`ValueError: Not enough resources to allocate N DP ranks`).
  2. Starting at DP=2 in a 4-Ray-node cluster hits
     `AssertionError: Created 4 DP placement groups, expected 2`.
  3. Starting at DP=4 works, but `POST /scale_elastic_ep
     {"new_data_parallel_size": 2}` fails with
     `AssertionError: num_redundant >= 0` in EPLB, leaving the
     4-container cluster in an unrecoverable half-torn-down state.

Exp B is the workaround sandbox to unblock (1)–(3).

## Scope

**In scope:**
- Patch the vLLM install inside a derived Docker image.
- Reuse Exp A3 Phase 3's per-GPU container harness (bridge network,
  Ray cluster, `ep-rank-N` containers) for testing.
- Demonstrate a clean elastic cycle — scale-up and scale-down — in the
  per-GPU container topology.
- Record scale-event timing and throughput at each DP, comparing
  against Phase 1 and Phase 3 baselines.
- Cover the relevant v5 exploration fields (§4.2 container lifecycle,
  §4.3 hidden assumptions, §4.4 EPLB robustness) opportunistically as
  workarounds require.

**Out of scope (explicit non-goals):**
- **No upstream PR to vLLM.** Patches are local research workarounds.
  Do not follow vLLM's `AGENTS.md` contributor workflow; treat
  `3rdparty/vllm/` as a read-only reference.
- **No K8s operator.** Scaling is driven manually via shell scripts or
  `curl`. Orchestration integration comes later.
- **No NCCL shim integration yet.** The `NET/Socket/0` fallback from
  Phase 3 (v5 §4.1) is a separate workaround track — v5 §4.2 (scaling
  API) should be addressable without recovering NCCL bandwidth first.
  If they interact, document it.
- **No new model or new vLLM version.** Qwen3-30B-A3B on vLLM 0.19.0
  throughout, so results compare directly against Exp A3.

## Starting Point: What's Already Known to Fail

Three concrete assertion/error sites (file:line refs against
`3rdparty/vllm/` submodule, vLLM 0.19.0):

| # | Location | Assertion / error | Triggered when |
|---|----------|-------------------|----------------|
| 1 | [`vllm/v1/engine/utils.py:623`](../3rdparty/vllm/vllm/v1/engine/utils.py#L623) | `Created N DP placement groups, expected M` where N > M | vLLM requests fewer DP ranks than available Ray GPU nodes — e.g. `--data-parallel-size 2 --data-parallel-size-local 1` with a 4-GPU Ray cluster. Outer loop doesn't break early when `len(placement_groups) == dp_size`. |
| 2 | [`vllm/distributed/eplb/policy/default.py:93`](../3rdparty/vllm/vllm/distributed/eplb/policy/default.py#L93) | `assert num_redundant >= 0` | During `replicate_experts`, when `num_phy < num_log`. Triggered by the scale-down reduction in EPLB state. |
| 3 | [`vllm/distributed/eplb/eplb_state.py:780-782`](../3rdparty/vllm/vllm/distributed/eplb/eplb_state.py#L780) | Formula `num_replicas = num_replicas // ep_group.size() * num_gpus` halves physical experts below logical count | Scale-down with `num_redundant_experts=0`. Root cause of #2. |

All three bugs reproduce cleanly in Exp A3 Phase 3. Exp B's job is to
understand them well enough to patch them minimally.

## The Approach: Patch-Based Derived Image

The vLLM install inside `vllm/vllm-openai:v0.19.0` lives at
`/usr/local/lib/python3.12/dist-packages/vllm/`. Patches are applied
at Docker build time with `patch -p1` from `exp_b_per_gpu_elasticity/patches/`.

```
exp_b_per_gpu_elasticity/
├── README.md                # This file
├── Dockerfile               # FROM xtrans-vllm-ep:v0.19.0 + apply patches
├── scripts/
│   ├── build.sh             # Apply all patches → build xtrans-vllm-ep-patched:DATE
│   ├── up.sh                # Bring up per-GPU cluster using the patched image
│   ├── cycle.sh             # Attempt the full elastic cycle + capture data
│   └── down.sh              # Teardown + save logs
├── patches/                 # Each .patch is a numbered, commented unified-diff
│   ├── 0001_description.patch
│   ├── 0002_description.patch
│   └── ...
└── results/                 # gitignored; one subdir per patched-image tag
    └── <tag>/
        ├── build.log
        ├── cycle.log
        ├── bench_*.json
        ├── nccl_transport.log
        └── state_*.txt
```

The base image `xtrans-vllm-ep:v0.19.0` (from `exp_a3_vllm_ep/Dockerfile.phase2`,
`vllm/vllm-openai:v0.19.0 + ray[default]`) is the starting point; we do
**not** edit `3rdparty/vllm/` directly — that submodule stays at upstream
HEAD for reference reading.

## Suggested Investigation Sequence

The three sites above are the prime targets. A pragmatic order:

### Track 1 — Fix EPLB redundancy under scale-down (sites #2 and #3)

**Easiest to isolate** because it reproduces in Phase 1 (native) too —
not only in the per-GPU topology. Start there, validate in Phase 1,
then check it behaves in Phase 3.

Reproduction (in Phase 1):
```bash
cd /home/thd/repositories/xtrans-experiments
./exp_a3_vllm_ep/scripts/phase1_native.sh start
# NB start is at DP=2 (by design — cycle goes 2→4→2); the bug only
# triggers from a *cold* DP=N start with num_redundant_experts=0. To
# reproduce directly:
#   (1) hack the phase1 script to start at DP=4, OR
#   (2) bring up DP=4 manually with vllm serve and try scale-down
```

Minimal patch hypothesis (write as `patches/0001_eplb_scale_down_redundancy.patch`):
the formula at `eplb_state.py:780-782` reduces `num_replicas` below
`num_logical` when there's no prior redundancy. A candidate fix is

```python
num_replicas = max(
    model.num_logical_experts,
    num_replicas // ep_group.size() * num_gpus,
)
```

but this changes behavior and needs empirical verification. Alternative:
use `--eplb-config.num_redundant_experts N` (for N>0) as a runtime
workaround first — does scale-down work? If yes, the patch just needs
to handle the edge case; if no, deeper issue.

**Deliverable:** scale-down 4→2 succeeds in Phase 1 (or, Phase 1 with
the patched image in Phase 2 mode). Measure 503 window and timing;
compare against the Phase 1 baseline of 4.1 s.

### Track 2 — Fix placement-group count mismatch (site #1)

Required for the per-GPU topology; addresses DP < Ray-cluster-size.

Location: `vllm/v1/engine/utils.py:612-625`. The outer loop over Ray
nodes doesn't break once `len(placement_groups) == dp_size` is reached
(the `break` on line 613 only exits the inner `for i in range(...)`
loop, not the outer node loop).

Minimal patch hypothesis: add an outer-loop early-exit.

```python
for node_id, node_resources in ...:
    ...
    if len(placement_groups) >= dp_size:
        break
```

**Deliverable:** `--data-parallel-size 2 --data-parallel-size-local 1`
starts successfully in a 4-GPU Ray cluster (like Phase 3's setup),
using only 2 of the 4 GPUs. Scale-up 2→4 should also work —
verify by calling `/scale_elastic_ep` and watching Ray's placement.

### Track 3 — Coordinate container lifecycle with scaling (v5 §4.2)

**Deferred but not out of scope.** If Tracks 1–2 succeed, the per-GPU
cluster can elastically rescale *within vLLM's Ray cluster*, but
stopping an `ep-rank-N` container when vLLM scales down (so the GPU is
genuinely free at the scheduler level) is a separate coordination
problem. Approach after the first two tracks stabilise.

Sketch: after `POST /scale_elastic_ep {new_dp: 2}` succeeds, the
coordinator (us, externally) runs `docker stop ep-rank-3 ep-rank-2`.
Ray nodes disappear; vLLM should already be ignoring them. Verify:
`nvidia-smi` shows GPUs 2,3 truly idle, `docker ps` shows only 2
containers. For scale-up, the inverse: start `ep-rank-2` and
`ep-rank-3` via `docker run ...`, wait for Ray node registration,
then `POST /scale_elastic_ep {new_dp: 4}`.

## Harness

All scripts assume `cd /home/thd/repositories/xtrans-experiments`.

### Build

```bash
./exp_b_per_gpu_elasticity/scripts/build.sh
```

Tags the image as `xtrans-vllm-ep-patched:YYYYMMDD-HHMM`. The script
prints the exact tag used and writes it to `results/LAST_BUILD`.

### Bring up the per-GPU cluster with the patched image

```bash
./exp_b_per_gpu_elasticity/scripts/up.sh
```

This wraps Exp A3's `phase3_per_gpu.sh up` with `VLLM_IMAGE` set to the
most recently built patched tag. Use `VLLM_IMAGE=xtrans-vllm-ep-patched:TAG ./up.sh`
to pin a specific build. Default DP is still 4; override with
`PHASE3_DP=2` if testing placement-group patches.

### Attempt an elastic cycle

```bash
./exp_b_per_gpu_elasticity/scripts/cycle.sh
```

Runs a programmable scale sequence (default 4 → 2 → 4), capturing
per-step timing, bench results, `nvidia-smi` snapshots, per-container
DeviceIDs, and NCCL transport grep. All output lands in
`results/<tag>/cycle.log`.

### Teardown

```bash
./exp_b_per_gpu_elasticity/scripts/down.sh
```

## Success Criteria

**Minimum viable result:** a documented, repeatable **elastic cycle in a
per-GPU container cluster**. Specifically:

1. Start 4 per-GPU containers + Ray cluster + `vllm serve` at DP=4.
2. `POST /scale_elastic_ep {new_dp: 2}` succeeds (HTTP 200), vLLM
   continues to serve requests after a bounded 503 window, 2 containers
   are no longer holding CUDA contexts (host `nvidia-smi` confirms).
3. `POST /scale_elastic_ep {new_dp: 4}` succeeds, vLLM resumes serving
   on all 4 GPUs, throughput returns to within 10% of the pre-scale
   DP=4 number.
4. Entire cycle repeatable without restart.

**Stretch results:**
- Coordinator-driven container lifecycle (Track 3): scale-down also
  stops 2 containers; scale-up starts 2 new containers. `docker ps`
  reflects current DP.
- Throughput at each DP measured with `vllm bench serve` and compared
  to Phase 1 native and Phase 3 static-DP=4 baselines.

**Non-goal for this experiment (again):** NCCL bandwidth recovery.
Phase 3 will likely still show NET/Socket/0 in NCCL logs; the 23%
throughput drop vs Phase 1 is acceptable here because it's orthogonal
(v5 §4.1 handles that via the LD_PRELOAD shim in a separate track).

## Known Traps / Dead Ends Discovered So Far

These are already observed in Exp A3 Phase 3; don't rediscover them:

- **`ray stop --force` is host-global.** It only kills Ray processes
  of the invoking user, but can interfere with concurrent Ray work.
  Phase 3's harness already uses a dedicated Ray head per container;
  don't call `ray stop` from the host during Exp B runs.
- **`--data-parallel-size-local 1` is mandatory** in a per-GPU cluster
  to avoid the "Not enough resources on DP master node" error. Don't
  remove it.
- **`vllm bench serve --model $SERVED_MODEL_NAME --tokenizer $MODEL_SNAPSHOT`**
  — the bench tool needs the model name to match the server's registered
  name, and a local tokenizer path (can't resolve a short alias to HF Hub).
  Already handled in `common.sh:vllm_bench`.
- **Prior Ray/vLLM cleanup is fragile.** `/tmp/ray/session_*` can be
  owned by root (from other containers) and we can't delete it. Harness
  works around this; don't worry about stale dirs unless things truly
  break.
- **Shared host contention** happens: other users may grab GPUs
  between your `nvidia-smi` check and `vllm serve` launch. The harness
  detects this via `require_gpus_free` and `wait_for_ready` liveness
  checks — don't remove those.

## Files You'll Probably Touch

In this experiment:
- `patches/*.patch` — your actual vLLM modifications.
- `results/<tag>/analysis.md` — jot findings per build.

Elsewhere (with care):
- `3rdparty/vllm/` — **read-only**. Use this for investigating source
  (`grep`, `read`), then author the patch against the installed path
  inside the Docker image.
- `exp_a3_vllm_ep/scripts/common.sh` — shared helpers (`log`,
  `wait_for_ready`, `vllm_bench`, `gpu_snapshot`, `container_deviceids`).
  Mostly stable; extend in place if you need a new helper.
- `exp_a3_vllm_ep/scripts/phase3_per_gpu.sh` — the reference harness
  for per-GPU containers. Exp B's harness wraps this, doesn't replace.

## First Steps (for the future session)

1. Read this README end-to-end.
2. Read `../docs/research_plan_v5.html` §4.1, §4.2, §4.4 for the
   framing, and `../exp_a3_vllm_ep/analysis_report.html` §5.6 for the
   Phase 3 failure details.
3. Verify baseline: `cd ..; ./exp_a3_vllm_ep/scripts/phase3_per_gpu.sh up`
   should still come up cleanly at DP=4 (unpatched) and bench should
   give ~108 tok/s. If not, the environment has drifted and you need
   to fix that first.
4. Pick Track 1 (EPLB redundancy). Write a patch. Run
   `./scripts/build.sh`, then `./scripts/up.sh`, then `./scripts/cycle.sh`
   for a small DP=4→DP=2 scale-down test.
5. Iterate. Record each build's result in `results/<tag>/analysis.md`.
6. When a track's goal is met, commit the patch with a clear message
   referencing the file:line in `3rdparty/vllm/` that was broken.

## References

- Research plan v5: [`../docs/research_plan_v5.html`](../docs/research_plan_v5.html)
- Exp A3 final report: [`../exp_a3_vllm_ep/analysis_report.html`](../exp_a3_vllm_ep/analysis_report.html)
- Exp A3 Phase 3 README section: [`../exp_a3_vllm_ep/README.md`](../exp_a3_vllm_ep/README.md)
- vLLM source (read-only): `../3rdparty/vllm/`
- Phase 3 harness reused here: `../exp_a3_vllm_ep/scripts/phase3_per_gpu.sh`
- Shared script helpers: `../exp_a3_vllm_ep/scripts/common.sh`
- vLLM Elastic EP RFC (upstream): https://github.com/vllm-project/vllm/issues/20323
