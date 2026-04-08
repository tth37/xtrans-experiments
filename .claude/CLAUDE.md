# XTrans Experiments — Project Context for Claude Code

## What This Project Is

An experiment framework for the XTrans research project. NOT the XTrans system
itself. This contains exploration code, experiment scripts, and shared utilities
for investigating cross-container GPU communication across vendors.

## Research Context

- **exp1/exp2** (in vllm-source) proved NCCL cuMem VMM recovers NVLink bandwidth
  in per-GPU containers on NVIDIA A100. The exp2 workaround uses env vars
  (NCCL_HOSTID, NCCL_CUMEM_ENABLE) + shared /dev/shm + host networking.
- **This project** extends that finding: build a principled LD_PRELOAD shim
  (not env var hacks), test across NCCL versions, then replicate on AMD/Intel.
- **Full research plan**: `docs/research_plan.html` — read Section 2 (Feasibility)
  and Section 6 (Phased Experiments) for experiment details.

## Key Technical Background

### NCCL's Three Container Gates (from exp2)

NCCL checks three things to decide if two GPUs are on the same node:
1. **hostHash**: `hash(gethostname() + /proc/sys/kernel/random/boot_id)` — must match
2. **shmDev**: `stat("/dev/shm").st_dev` — must match (same tmpfs mount)
3. **IPC socket**: abstract UDS (network-namespace-scoped) for cuMem FD passing

If any gate fails → falls back to TCP/socket (2-10x slower than NVLink).

### The LD_PRELOAD Shim Approach

Intercept `gethostname()` and `stat()` via LD_PRELOAD to make NCCL's gates pass
in per-GPU containers. Also use bind mounts for /proc/sys/kernel/random/boot_id.
This is the core mechanism being developed in `common/shim/`.

- NCCL/RCCL/oneCCL are always shared libraries with dynamic libc linkage
- The intercepted functions are POSIX syscalls called once at init (zero data-path overhead)
- Published precedent: Trickle (ATC'05), HetCCL (2026), PhoenixOS (SOSP'24)

### Vendor IPC Divergence (key research finding)

| Vendor | IPC Mechanism | Handle Type | Cross-Container Path |
|--------|--------------|-------------|---------------------|
| NVIDIA | cuMem VMM | POSIX FD | UDS SCM_RIGHTS (proven) |
| NVIDIA legacy | cudaIpcMemHandle | Opaque blob | Requires shared IPC NS (blocked) |
| AMD (RCCL uses) | hsa_amd_ipc_memory_create | Opaque 32B blob | Via /dev/kfd (untested) |
| AMD (HIP VMM) | hipMemExportToShareableHandle | POSIX FD | Beta API, RCCL disables it |
| AMD (DMA-BUF) | hsa_amd_portable_export_dmabuf | DMA-BUF FD | Kernel-mature, untested for P2P |
| Intel L0 | zeMemGetIpcHandle | DMA-BUF FD | oneCCL sockets mode works |

### DMA-BUF Hypothesis

DMA-BUF file descriptors may be a universal cross-container GPU IPC primitive
(available on NVIDIA, AMD, Intel). Validating this is the Phase 1.5 experiment.

## Experiment Phases

### Exp A (exp_a_nccl_gates/) — NCCL Gate Taxonomy + Shim
- Syscall tracing under strace to map all gates
- Build minimal LD_PRELOAD shim to recover NVLink P2P
- Test across NCCL versions (2.18-2.28)
- Hardware: node192 (A100)
- Success: ≥99% NVLink bandwidth without NCCL_* env vars

### Exp A' (exp_a_prime_dmabuf/) — DMA-BUF Feasibility Spike
- Test DMA-BUF export → FD passing → import → GPU P2P on NVIDIA
- Compare: native cuMem IPC vs DMA-BUF vs legacy cudaIPC
- Hardware: node192 (A100)
- Success: DMA-BUF achieves ≥95% of native cuMem bandwidth

### Exp B (exp_b_rccl/) — AMD RCCL Analysis
- RCCL source analysis (same three gates as NCCL, cuMem disabled)
- Test three AMD IPC paths: opaque HSA, DMA-BUF, HIP VMM FD
- Hardware: AMD MI250X/MI300X (cloud)

## Hardware: node192

- 4x NVIDIA A100 80GB with NVLink 3.0 (600 GB/s bidirectional)
- Available now for Exp A and Exp A'
- Docker + NVIDIA Container Toolkit installed
- NCCL 2.21.5 in current container images (can install others)

## Coding Conventions

- Shim code: C99, minimal dependencies (only libc + libdl)
- Scripts: Python 3.8+ or Bash
- Container configs: Docker Compose where possible
- Results: JSON for structured data, CSV for time series
- All paths in scripts should be relative to project root

## Experiment Reports

Each experiment produces an `analysis_report.html` following a standardized format.

- **Template**: `docs/report_template.html` — copy into experiment dir to start a report
- **Guide**: `docs/report_guide.md` — section structure, style rules, checklist
- **Examples**: exp1 and exp2 reports in vllm-source (the template is derived from these)

Report structure (required sections):
1. **Objective** — hypothesis + green highlight-box with one-sentence result
2. **Background** — (optional) context for the experiment
3. **Experimental Setup** — hardware table + software table with exact versions
4. **Configurations Under Test** — table of all configs being compared
5. **Results** — data tables with "vs Baseline" column, figures in `analysis_assets/`
6. **Analysis** — (optional) deeper interpretation, limitations
7. **Conclusions** — numbered findings, each one sentence; blue info-box with next steps
8. **Appendices** — full data tables, separated by `<hr class="appendix">`

Three callout box types:
- `highlight-box` (green): key results
- `warning-box` (orange): limitations, risks
- `info-box` (blue): context, next steps

## File Locations

- Research plan: `docs/research_plan.html`
- Report template: `docs/report_template.html`
- Report guide: `docs/report_guide.md`
- Shim source: `common/shim/`
- Container helpers: `common/containers/`
- Benchmark harness: `common/benchmarks/`
- Per-experiment code: `exp_a_*/`, `exp_b_*/`
