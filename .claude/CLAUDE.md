# XTrans Experiments — Project Context for Claude Code

## What This Project Is

Research toward **true elasticity of LLM serving on shared GPU clusters** — the
serving process survives GPU allocation changes, the cluster scheduler actually
sees the scaling, and serving quality is preserved. The active direction is
**per-GPU containers for in-place elastic MoE serving**, built on vLLM Elastic
EP.

## Research Plan (v5 — current)

**Document:** `docs/research_plan_v5.html`

v5 supersedes v4 for the active direction. It narrows the investigation after
the results from Exp A1 (Tenplex, archived) and Exp A3 (vLLM Elastic EP) Phases
1–3 pinpointed the specific obstacles worth solving.

**Core argument:**
1. Production GPU workloads run in containers (K8s, EKS/AKS/GKE).
2. Multi-GPU containers cannot dynamically adjust their GPU count (cgroup
   device allowlist is frozen at container creation; Docker has no hot-modify
   API; K8s In-Place Pod Resize excludes `nvidia.com/gpu`).
3. Stop-restart systems (Tenplex, ByteCheckpoint) are compatible with
   multi-GPU containers — the process terminates between reconfigurations.
4. In-place systems (vLLM Elastic EP, NTP) are **not** — the process stays
   running, so the container never gets recreated, so the orchestrator never
   sees GPU allocation changes.
5. Per-GPU containers are therefore the only path to cluster-visible elasticity
   for in-place systems, but they introduce communication-layer friction
   (NCCL falls back to TCP when its three same-node gates all fail).

See v5 Section 4 for current exploration fields and workaround plans.

## Active Experiment

### Exp A3 (`exp_a3_vllm_ep/`) — In-Place Elastic Expert Parallelism
- **System:** [vLLM](https://github.com/vllm-project/vllm) Elastic EP (v0.19)
- **Model:** Qwen3-30B-A3B (30.5B total, 128 experts, 8 activated/token)
- **Status:** Phases 1, 2, 3 complete. Serves as the running baseline for v5
  exploration.
- **Phase 1 (native):** 166.6 tok/s at DP=4 (1.91× linear scaling from DP=2).
- **Phase 2 (4-GPU container):** 154.1 tok/s (−7.5%) but GPUs are
  container-trapped.
- **Phase 3 (per-GPU containers):** 127.8 tok/s (−23%) — NCCL drops to TCP
  sockets because all three same-node gates fail at container boundaries.
- **vLLM source as reference:** `3rdparty/vllm/` (git submodule).

## Archived Experiments (`archived/`)

- `exp_a1_tenplex/` — v4 Exp A1 (Tenplex, stop-restart elasticity). Completed.
  Confirmed stop-restart works in multi-GPU containers — does not motivate
  per-GPU containers.
- `exp_a_nccl_gates/` — v1–v3 NCCL gate taxonomy + LD_PRELOAD shim. The shim
  (`common/shim/`) is what v5 Section 4.1 plans to apply to per-GPU
  containers.
- `exp_a_prime_dmabuf/` — DMA-BUF feasibility spike (completed).
- `exp_a_double_prime_topo/` — Topology virtualization scaffold.
- `exp_b_rccl/` — AMD RCCL analysis (not started).

## Technical Background

### NCCL's Three Container Gates
NCCL checks three things to decide if two GPUs are on the same node:
1. **hostHash**: `hash(gethostname() + /proc/sys/kernel/random/boot_id)`
2. **shmDev**: `stat("/dev/shm").st_dev`
3. **IPC socket**: abstract UDS (network-namespace-scoped) for cuMem FD passing

If any gate fails → falls back to TCP/socket (2–10× slower than NVLink).
Phase 3 of Exp A3 confirmed this experimentally for vLLM.

### The LD_PRELOAD Shim (`common/shim/`)
A C99 shim that intercepts `gethostname()`, `stat()`, `bind()`, and
`sendmsg()` to make NCCL's gates pass across container boundaries. Validated
at 100% NVLink bandwidth for NCCL benchmarks (156.8 GB/s AllReduce,
262.4 GB/s raw P2P). Applying it to per-GPU vLLM deployments is the
top-priority workaround in v5 Section 4.1.

## Hardware: node192

- 3× NVIDIA A100-SXM4-40GB + 1× NVIDIA A100-PCIE-40GB
- NVLink 3.0 NV12 between GPU0↔GPU1; PIX between GPU2↔GPU3; SYS across
- CUDA 13.1 (12.4+ available), driver 590.44.01
- Docker 27.3.1 with NVIDIA Container Toolkit

## Coding Conventions

- Shim code: C99, minimal dependencies (only libc + libdl)
- Scripts: Python 3.12 via project `.venv/` (`uv pip install`), or Bash
- Container configs: Dockerfile per experiment where needed
- Results: JSON for structured data; `**/results/` is gitignored
- Docker image tags: `xtrans-<exp>:<version>` (e.g. `xtrans-vllm-ep:v0.19.0`)

## Experiment Reports

Each experiment produces an `analysis_report.html` following a standardized
format.

- **Template**: `docs/report_template.html`
- **Guide**: `docs/report_guide.md`

Report structure: Objective, Background, Setup, Configurations, Results,
Analysis, Conclusions, Appendices.

## File Locations

- Current research plan: `docs/research_plan_v5.html`
- Prior plans: `docs/research_plan.html`, `research_plan_v2.html`,
  `research_plan_v3.html`, `research_plan_v4.html`
- Report template: `docs/report_template.html`
- Report guide: `docs/report_guide.md`
- LD_PRELOAD shim: `common/shim/`
- Container helpers: `common/containers/`
- NCCL benchmark: `common/benchmarks/`
- Observation recorder: `common/harness/record_observation.py`
- Phase comparison: `common/harness/compare_phases.py`
- Active experiment (Exp A3, vLLM Elastic EP): `exp_a3_vllm_ep/`
- vLLM source (read-only reference): `3rdparty/vllm/`
- Archived experiments: `archived/`
