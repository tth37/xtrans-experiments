# XTrans Experiments — Project Context for Claude Code

## What This Project Is

An experiment framework for studying what happens when GPU scheduling techniques
— designed and validated on bare metal — meet containerized production
environments. The current research plan (v4) selects open-source scheduling
systems (Tenplex, Oobleck) and progressively containerizes them to discover
challenges at each stage.

## Research Plan (v4)

**Document:** `docs/research_plan_v4.html`

**Research question:** What happens when GPU scheduling techniques — designed
for bare metal — are deployed in containerized production environments?

**Motivation (three acts):**
1. GPU scheduling evolved toward per-GPU granularity (Tenplex, NTP)
2. Production demands containers (K8s, NVL72, multi-tenancy)
3. Nobody has studied what happens when these two collide

**Methodology:** Progressive containerization — for each scheduling technique:
- Phase 1: Bare metal (baseline)
- Phase 2: Multi-GPU container (production approach) — discover limitations
- Phase 3: Per-GPU containers (only if motivated by Phase 2) — discover new challenges

Challenges are discovered empirically, not imagined a priori.

## Current Experiments

### Exp A1 (`exp_a1_tenplex/`) — Elastic GPU Scaling
- **System:** [Tenplex](https://github.com/kungfu-team/tenplex) (SOSP '24)
- **Operation:** Add/remove GPUs from a running training job at runtime
- **Key question:** Does framework-level elasticity translate to cluster-level
  elasticity in containers? Or are freed GPUs trapped?

### Exp A2 (`exp_a2_oobleck/`) — Fault-Tolerant Training
- **System:** [Oobleck](https://github.com/SymbioticLab/Oobleck) (SOSP '23)
- **Operation:** Continue training after GPU failure via pipeline templates
- **Key question:** Does framework-level recovery work when the failure unit
  is a container, not a process? What is the blast radius?

### Exp A3 (`exp_a3_vllm_ep/`) — In-Place Elastic Expert Parallelism
- **System:** [vLLM](https://github.com/vllm-project/vllm) Elastic EP (v0.17+)
- **Operation:** Live rescaling of expert parallel workers for MoE inference
- **Key question:** Does per-GPU-container deployment enable cluster-level
  in-place elasticity that multi-GPU containers fundamentally cannot?
- **Why:** Exp A1 showed stop-restart systems don't need per-GPU containers.
  In-place systems (like vLLM Elastic EP) do — Docker can't change GPU
  assignment of a running container.

### Archived (v1-v3 experiments in `archived/`)
- `exp_a_nccl_gates/` — NCCL gate taxonomy + LD_PRELOAD shim (completed)
- `exp_a_prime_dmabuf/` — DMA-BUF feasibility spike (completed)
- `exp_a_double_prime_topo/` — Topology virtualization (scaffold only)
- `exp_b_rccl/` — AMD RCCL analysis (not started)

These validated that cross-container NCCL communication CAN be recovered
(100% NVLink bandwidth via LD_PRELOAD shim). The v4 plan pivots from
"can we recover communication?" to "what happens when real scheduling
techniques hit the container wall?"

## Technical Background

### NCCL's Three Container Gates
NCCL checks three things to decide if two GPUs are on the same node:
1. **hostHash**: `hash(gethostname() + /proc/sys/kernel/random/boot_id)`
2. **shmDev**: `stat("/dev/shm").st_dev`
3. **IPC socket**: abstract UDS (network-namespace-scoped) for cuMem FD passing

If any gate fails → falls back to TCP/socket (2-10x slower than NVLink).

### The LD_PRELOAD Shim (`common/shim/`)
A 261-line C99 shim that intercepts `gethostname()`, `stat()`, `bind()`, and
`sendmsg()` to make NCCL's gates pass across container boundaries. Validated
at 100% NVLink bandwidth (156.8 GB/s AllReduce, 262.4 GB/s raw P2P). This
is infrastructure for Phase 3 of v4 experiments, not the research itself.

## Hardware: node192

- 4x NVIDIA A100-SXM4-40GB with NVLink 3.0 (600 GB/s bidirectional)
- CUDA 12.4+, NCCL 2.21.5
- Docker 27.3.1 with NVIDIA Container Toolkit

## Coding Conventions

- Shim code: C99, minimal dependencies (only libc + libdl)
- Scripts: Python 3.8+ or Bash
- Container configs: Docker Compose where possible
- Results: JSON for structured data, CSV for time series
- All paths in scripts should be relative to project root
- Docker image tags: `tth37/xtrans-experiments:<experiment>`

## Experiment Reports

Each experiment produces an `analysis_report.html` following a standardized format.

- **Template**: `docs/report_template.html` — copy into experiment dir to start
- **Guide**: `docs/report_guide.md` — section structure, style rules, checklist

Report structure: Objective, Background, Setup, Configurations, Results,
Analysis, Conclusions, Appendices.

## File Locations

- Research plan (current): `docs/research_plan_v4.html`
- Prior plans: `docs/research_plan.html`, `docs/research_plan_v2.html`, `docs/research_plan_v3.html`
- Report template: `docs/report_template.html`
- Report guide: `docs/report_guide.md`
- LD_PRELOAD shim: `common/shim/`
- Container helpers: `common/containers/`
- NCCL benchmark: `common/benchmarks/`
- Observation recorder: `common/harness/record_observation.py`
- Phase comparison: `common/harness/compare_phases.py`
- Exp A1 (Tenplex): `exp_a1_tenplex/`
- Exp A2 (Oobleck): `exp_a2_oobleck/`
- Exp A3 (vLLM EP): `exp_a3_vllm_ep/`
- Archived experiments: `archived/`
