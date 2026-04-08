# XTrans Experiments

Experiment framework for the XTrans research project: understanding and recovering
GPU collective communication performance across container isolation boundaries.

This is **not** the XTrans system itself — it's the exploration and experiment
framework that will inform and validate the XTrans design. The actual XTrans
implementation (daemon + shim library) will be built later based on findings here.

## Project Context

Our exp1–2 in vllm-source proved that NCCL's cuMem VMM path recovers 100% NVLink
bandwidth in per-GPU containers on NVIDIA. This project extends that finding across
vendors (AMD, Intel) and builds toward a general cross-container GPU communication
abstraction.

See `docs/research_plan.html` for the full research plan.

## Directory Structure

```
xtrans-experiments/
├── docs/                      # Research plan, references, notes
│   └── research_plan.html     # Full research plan (Section 1-11)
├── common/                    # Shared utilities across experiments
│   ├── shim/                  # LD_PRELOAD shim framework (C)
│   │   ├── Makefile
│   │   ├── xtrans_shim.h      # Shim interface and config
│   │   └── xtrans_shim.c      # Core interception logic
│   ├── containers/            # Docker/container orchestration
│   │   ├── Dockerfile.nccl    # NCCL test container
│   │   └── docker_helpers.sh  # Container launch helpers
│   └── benchmarks/            # Benchmark harness
│       ├── nccl_bench.py      # NCCL allreduce benchmark wrapper
│       └── parse_results.py   # Result parsing and comparison
├── exp_a_nccl_gates/          # Phase 1: NCCL gate taxonomy + shim
│   └── README.md              # Experiment guide
├── exp_a_prime_dmabuf/        # Phase 1.5: DMA-BUF feasibility spike
│   └── README.md
├── exp_b_rccl/                # Phase 2: AMD RCCL analysis
│   └── README.md
└── .claude/
    └── CLAUDE.md              # Project context for Claude Code sessions
```

## Hardware Requirements

- **node192**: 4x NVIDIA A100 (NVLink 3.0) — available for Exp A, A'
- **AMD MI250X/MI300X**: needed for Exp B — cloud access pending
- **Intel GPU**: stretch goal — needed for oneCCL validation

## Quick Start

```bash
# Build the shim library
cd common/shim && make

# Run NCCL baseline benchmark (no shim, shared container)
cd common/benchmarks && python nccl_bench.py --config baseline

# Run with per-GPU isolation (expect degraded perf)
python nccl_bench.py --config isolated

# Run with shim (expect recovered perf)
python nccl_bench.py --config shim
```

## Experiment Workflow

Each experiment directory (`exp_a_*`, `exp_b_*`, etc.) contains:
- `README.md` — goal, method, success criteria, how to run
- Scripts and configs specific to that experiment
- Results go into `results/` subdirectory (gitignored)

The `common/` directory contains shared infrastructure:
- `shim/` — the LD_PRELOAD library that intercepts NCCL/RCCL gate syscalls
- `containers/` — Docker tooling for per-GPU container setups
- `benchmarks/` — standardized benchmark harness for consistent measurement
