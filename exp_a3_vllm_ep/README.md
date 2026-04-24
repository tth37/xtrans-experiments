# Exp A3: In-Place Elastic Expert Parallelism with vLLM

## Overview

- **System under test:** [vLLM](https://github.com/vllm-project/vllm) Elastic EP
  (v0.17+, using v0.19.0).
- **Scheduling operation:** in-place live rescaling of expert-parallel workers
  for MoE models — the serving process survives GPU allocation changes.
- **Research plan:** `docs/research_plan_v5.html` (v5 supersedes v4 for the
  active direction).
- **vLLM source:** `3rdparty/vllm/` (git submodule).

Findings: see `analysis_report.html` (cross-regime results) and
`mechanism_deep_dive.html` (EP internals, code-level walk-throughs).

## Why this experiment

Exp A1 (Tenplex) showed stop-restart elastic systems don't motivate per-GPU
containers — multi-GPU containers handle them fine. Per-GPU containers are
motivated by **in-place** elasticity, where the serving process stays running
while GPUs are added or removed. Docker cannot change the GPU assignment of a
running container (the cgroup device allowlist is frozen at container
creation), so in-place elasticity is blocked by the multi-GPU-container model.
Per-GPU containers resolve this at the orchestrator level: each worker lives
in its own container, scaling is expressed as container lifecycle.

## Model

**Qwen3-30B-A3B** (`Qwen/Qwen3-30B-A3B`). 30.5B total / 3.3B active, 128
experts (8 activated per token), 48 layers. BF16 weights ≈ 61 GB → ~15.25
GB/GPU at EP=4, leaving room for KV cache on 4× A100-40GB.

Download into `/data/models--Qwen--Qwen3-30B-A3B/` in tmux via HF mirror:

```bash
sudo mkdir -p /data/models--Qwen--Qwen3-30B-A3B && sudo chmod 777 /data/models--Qwen--Qwen3-30B-A3B
tmux new-session -d -s model-download \
    'source .venv/bin/activate && \
     HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
        Qwen/Qwen3-30B-A3B --cache-dir /data \
        > /tmp/qwen3-moe-download.log 2>&1; echo "EXIT_CODE=$?" >> /tmp/qwen3-moe-download.log'
```

## Three regimes (unified Python harness)

All three regimes share `common.py` (venv activation, `vllm bench serve`
stable-bench orchestration, `/scale_elastic_ep` driver, `nvidia-smi` /
container-state snapshots). Each has the same reduced subcommand surface;
results land in `results/<regime>/` (gitignored).

| Regime | Entry point | Process topology | GPU visibility |
|---|---|---|---|
| Native | `1_native.py` | Bare-metal Python + Ray on host | All 4 GPUs visible to every actor |
| Multi-GPU container (MGC) | `2_multi_gpu_container.py` | Single Docker container | `--gpus '"device=0,1,2,3"'` — cgroup-pinned for the container's lifetime |
| Per-GPU containers (PGC) | `3_per_gpu_containers.py` | 4 containers (`ep-rank-0…3`) on Docker bridge network; Ray head in `ep-rank-0` | One GPU per container (`--gpus '"device=N"'`) |

### Docker images

- **Base** `xtrans-vllm-ep:v0.19.0` — `Dockerfile.base` = `vllm/vllm-openai:v0.19.0`
  + `ray[default]==2.55.0` (the upstream image doesn't include Ray; elastic EP
  requires it). Used by MGC directly.
- **Patched** `xtrans-vllm-ep-patched:v0.19.0` — `Dockerfile.per_gpu_containers`
  layers one local vLLM patch on top of the base image. Used by per-GPU
  containers. Auto-built on first `3_per_gpu_containers.py up` invocation.
  - `patches/0001_placement_group_early_exit.patch` — two-line outer-loop
    break in `RayDPClient.make_dp_placement_groups`. Load-bearing for cold
    DP=2 in a 4-Ray-node cluster; mirrors the pattern already present in
    the scale-up sibling `add_dp_placement_groups`. See
    `analysis_report.html` §5.6.3.

### Subcommands

Same subcommand surface across regimes (`start`/`up` and `stop`/`down` are
aliases). `cycle` runs the full DP=2 → 4 → 2 stable-bench cycle. `bench`
runs a DP=4-only stable bench for saturation investigations.

```bash
# Native
python 1_native.py start                 # Ray head + vllm serve at DP=2
python 1_native.py cycle                 # stable bench DP=2 → 4 → 2
python 1_native.py bench                 # DP=4-only stable bench
python 1_native.py stop

# Multi-GPU container
python 2_multi_gpu_container.py start
python 2_multi_gpu_container.py cycle
python 2_multi_gpu_container.py bench
python 2_multi_gpu_container.py stop

# Per-GPU containers
python 3_per_gpu_containers.py up        # bridge net + 4 containers + Ray cluster + vllm serve
python 3_per_gpu_containers.py cycle
python 3_per_gpu_containers.py bench
python 3_per_gpu_containers.py nccl-grep # extract NET/Socket/0 evidence
python 3_per_gpu_containers.py down
```

The harness treats the first bench sample as warmup by default
(`A3_BENCH_DISCARD_FIRST=1`) and excludes it from convergence and summary
statistics. After warmup, it repeats `vllm bench serve` until the last 3 TPOTs
are within 5% of each other, then records 1 extra sample for the final summary.
Output per bench point is `results/<regime>/bench_<label>.json`.
Saturation experiments can be selected without code edits via `A3_BENCH_*`
environment variables, including `A3_BENCH_DATASET`,
`A3_BENCH_DATASET_PATH`, `A3_BENCH_RANDOM_OUTPUT_LEN`,
`A3_BENCH_MAX_CONCURRENCY`, `A3_BENCH_REQUEST_RATE`,
`A3_BENCH_DISCARD_FIRST`, and `A3_BENCH_EXTRA_ARGS`.

The `wait_for_ready` helper aborts fast if the backing process dies and
dumps diagnostics instead of polling to timeout. Hard prerequisite: all 4
GPUs free on the host; override with `ALLOW_BUSY_GPUS=1` at your own risk.

### Triggering scale events manually

```bash
curl -X POST http://localhost:8000/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d '{"new_data_parallel_size": 4, "drain_timeout": 60}'

curl http://localhost:8000/is_scaling_elastic_ep
```

## Results at a glance

Stable-bench TPOT at DP=4 post-scale-up (ms, lower is better; full table in
`analysis_report.html` §5.1):

| Regime | Stable TPOT |
|---|---|
| Native | 93.26 ± 6.32 ms |
| Multi-GPU container | **86.92 ± 0.94 ms** |
| Per-GPU containers | **87.64 ± 0.86 ms** |

MGC and PGC are within ~1% at the baseline random-128/128 DP=4 stable bench,
but a DP=4 ShareGPT saturation bench exposes a visible PGC penalty: about
+8.7% TPOT and −7.1% throughput versus MGC. The main *structural* differences
between the regimes are the orchestrator GPU-allocation view (MGC's
`DeviceIDs` immutable at `[0,1,2,3]` for the container's lifetime vs.
per-GPU containers cleanly releasing GPUs on container stop) and the
NCCL transport (same-node NVLink/IPC/SHM in native and MGC vs.
`NET/Socket/0` in PGC because all three same-node gates fail across
container boundaries).

## Hardware

- **Node:** node192
- **GPUs:** 3× NVIDIA A100-SXM4-40GB + 1× NVIDIA A100-PCIE-40GB
- **Driver:** 590.44.01, CUDA 13.1

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV12    SYS     SYS     ← NUMA 0
GPU1    NV12     X      SYS     SYS     ← NUMA 0
GPU2    SYS     SYS      X      PIX     ← NUMA 1
GPU3    SYS     SYS     PIX      X      ← NUMA 1
```

GPU0 ↔ GPU1 via NVLink 3.0 (12 links); GPU2 ↔ GPU3 via a PCIe bridge (PIX);
cross-group via QPI (SYS). The heterogeneous interconnect applies identically
to all three regimes.

## Dependencies

| Dependency | Notes |
|---|---|
| vLLM 0.19.0 | `.venv/` (native) or base image (containerised) |
| Ray 2.55.0 | `.venv/` (native) or base image (containerised) |
| Qwen3-30B-A3B | `/data/models--Qwen--Qwen3-30B-A3B/` (~60 GB BF16) |
| Docker | 27.3.1 with NVIDIA Container Toolkit (containerised regimes) |
| NCCL shim | `common/shim/` (planned application: v5 §4.1, not yet integrated) |

## References

- vLLM Elastic EP RFC: https://github.com/vllm-project/vllm/issues/20323
- vLLM source: https://github.com/vllm-project/vllm
- Expert-as-a-Service: https://arxiv.org/abs/2509.17863
- ElasticMoE: https://arxiv.org/abs/2510.02613
- Lazarus (elastic MoE training): https://arxiv.org/abs/2407.04656
- Current research plan: `docs/research_plan_v5.html`
