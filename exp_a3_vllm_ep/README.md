# Exp A3: In-Place Elastic Expert Parallelism with vLLM

## Overview

**System under test:** [vLLM](https://github.com/vllm-project/vllm) Elastic EP (v0.17+, using v0.19.0)
**Scheduling operation:** In-place live rescaling of expert parallelism for MoE models
**Research plan:** `docs/research_plan_v4.html` Section 8
**vLLM source:** `vllm/` (git submodule)

vLLM's Elastic EP enables live rescaling of expert parallel workers — adding or removing
GPUs for MoE inference without restarting the serving process. This is **in-place elasticity**:
the process survives GPU allocation changes.

## Why This Experiment

Exp A1 (Tenplex) showed that **stop-restart elastic systems do not motivate per-GPU containers**
— multi-GPU containers handle them fine. Per-GPU containers are motivated by **in-place
elasticity**, where the serving process stays running while GPUs are added/removed. Docker
cannot change GPU assignment of a running container, so in-place elasticity is fundamentally
blocked by multi-GPU containers. Per-GPU containers resolve this: each expert worker lives
in its own container, and scaling is expressed as container lifecycle.

## Model Selection

**Chosen model: Qwen3-30B-A3B** — a modern MoE model with 128 experts, ideal for EP.

| Model | Total Params | Active | Experts | EP=4 per GPU | Fits 4x A100-40GB? |
|-------|-------------|--------|---------|--------------|---------------------|
| **Qwen3-30B-A3B** | 30.5B | 3.3B | 128 (8 active) | 32 experts/GPU, ~15GB | **Yes (selected)** |
| Qwen3.5-35B-A3B | ~35B | ~3.5B | TBD | TBD | Yes but tighter |
| DeepSeek-V2-Lite | 15.7B | 2.4B | 64 (6 active) | 16 experts/GPU | Yes (too small for EP) |
| Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 60 | 15 experts/GPU | Yes (too small for EP) |

**Why Qwen3-30B-A3B:**
- 128 experts makes EP=4 meaningful (32 experts/GPU vs. just 15-16 for smaller models)
- BF16 weights ≈ 61GB → ~15.25GB/GPU with EP=4, leaving ~25GB for KV cache
- Well-supported by vLLM (explicitly listed in docs for EPLB)
- Qwen3 is current-generation (2025), not Qwen1.5 (2024)

**Download:** `Qwen/Qwen3-30B-A3B` via HF mirror → `/data/models--Qwen--Qwen3-30B-A3B/`

## Setup

### 1. vLLM Installation (Done)

```bash
cd /home/thd/repositories/xtrans-experiments
source .venv/bin/activate
uv pip install vllm         # installed v0.19.0
uv pip install "ray[default]"  # installed v2.55.0 (required for elastic EP)

python -c "import vllm; print(vllm.__version__)"  # 0.19.0
```

### 2. Model Download

```bash
# Create cache dir (requires sudo on node192)
sudo mkdir -p /data/models--Qwen--Qwen3-30B-A3B
sudo chmod 777 /data/models--Qwen--Qwen3-30B-A3B

# Download in tmux (16 safetensor shards, ~60GB total)
tmux new-session -d -s model-download \
    'source .venv/bin/activate && \
     HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
        Qwen/Qwen3-30B-A3B --cache-dir /data \
        > /tmp/qwen3-moe-download.log 2>&1; \
     echo "EXIT_CODE=$?" >> /tmp/qwen3-moe-download.log'

# Check progress:
tail -f /tmp/qwen3-moe-download.log
```

### 3. Start Serving with Elastic EP

```bash
# Resolve model path from HF cache
MODEL=/data/models--Qwen--Qwen3-30B-A3B/snapshots/<hash>

# Start with DP=4, TP=1, EP=4 (auto), elastic EP enabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # mixed SXM4 + PCIe GPUs
vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --data-parallel-backend ray \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --enable-eplb \
    --all2all-backend allgather_reducescatter \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --trust-remote-code \
    --port 8000

# Or use the script:
./scripts/phase1_native.sh serve
```

### 4. Trigger Live Rescaling

```bash
# Scale down: 4→2 DP workers
curl -X POST http://localhost:8000/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d '{"new_data_parallel_size": 2, "drain_timeout": 120}'

# Scale up: 2→4 DP workers
curl -X POST http://localhost:8000/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d '{"new_data_parallel_size": 4, "drain_timeout": 120}'

# Check if currently scaling
curl http://localhost:8000/is_scaling_elastic_ep
```

## Experiment Phases

### Phase 1: vLLM Elastic EP Native (Baseline) — COMPLETE

Ran 2026-04-19 on node192 with Qwen3-30B-A3B, vLLM 0.19.0, Ray 2.55.0.

**Revised procedure** (must start at DP=2 and scale up first — scale-down
from a cold DP=4 start hits an EPLB assertion with `num_redundant_experts=0`):

1. Start Ray head at `10.0.2.192:26379` (avoids conflict with unrelated
   Ray at `172.17.0.2:6379` inside Docker on the host).
2. `vllm serve ... --data-parallel-size 2 --enable-elastic-ep ...`
3. Benchmark at DP=2 (baseline).
4. `POST /scale_elastic_ep {"new_data_parallel_size": 4}`.
5. Benchmark at DP=4.
6. `POST /scale_elastic_ep {"new_data_parallel_size": 2}`.
7. Benchmark at DP=2 post-scale-down.
8. Repeat (round 2) + 503 probing.

**Results:**

| Config | Throughput | vs DP=2 initial |
|--------|-----------|-----------------|
| DP=2 initial | 87.4 tok/s | 100% |
| DP=4 post-scale-up | **166.6 tok/s** | **190.6%** (1.91×) |
| DP=2 post-scale-down | 91.6 tok/s | 104.8% |

| Operation | Duration | Notes |
|-----------|---------|-------|
| Scale-up 2→4 (round 1) | 25.81 s | Ray actor spawn + weight transfer + EPLB |
| Scale-up 2→4 (round 2) | 22.97 s | Slight speedup (cached) |
| Scale-down 4→2 (round 1) | 4.05 s + 10 s async teardown | HTTP 200 before GPU release |
| Scale-down 4→2 (round 2) | 9.64 s | Longer due to drain-wait |
| **503 window during scale-down** | **3.53 s** | Middleware flag clears before teardown |

**Key findings:**
- Near-linear scaling efficiency (95%)
- Freed GPUs on bare metal: memory drops to 4 MiB within ~10s of HTTP 200
- No request migration — 503 is instant reject, client must retry
- Round-trip (2→4→2) is repeatable

**Script:** `scripts/phase1_native.sh`
**Results:** `results/phase1/` (including `phase1_results.json`, `round2.json`, `serve.log`)
**Analysis:** `analysis_report.html` sections 5.0–5.4

### Phase 2: Multi-GPU Container (The In-Place Barrier)

Run vLLM inside a single container with all 4 GPUs.
- Scale-down works internally, but GPUs are **trapped** (Docker can't release them)
- Scale-up impossible (Docker can't hot-add GPUs)
- This is the **fundamental limitation** for in-place systems (unlike Tenplex where it was just engineering)

**Script:** `scripts/phase2_container.sh`

### Phase 3: Per-GPU Containers (The Solution)

Each EP worker in its own container. Scaling = container lifecycle.
- Scale-down: stop containers → GPUs **immediately free**
- Scale-up: start new containers → new workers join
- Key challenges: cross-container NCCL All-to-All, dynamic communicator creation

**Script:** `scripts/phase3_per_gpu.sh`

## Smoke Test Findings (2026-04-17, on GPUs 0-1 only)

While waiting for GPUs 2-3 to free up (occupied by sglang), we ran a smoke
test with DP=2 on GPUs 0-1 to validate the toolchain end-to-end.

### What worked
- vLLM v0.19.0 starts cleanly with `--enable-elastic-ep --enable-eplb --enable-expert-parallel --data-parallel-backend ray`
- Model loads in ~60s (weights), full startup ~75s to first token
- Expert assignment is linear as expected: Rank 0 → experts 0-63, Rank 1 → experts 64-127 (64 experts/GPU at EP=2)
- All-to-All backend: `AgRsAll2AllManager` (as configured)
- NCCL version: 2.27.5
- Endpoints `POST /scale_elastic_ep` and `POST /is_scaling_elastic_ep` both register correctly
- Inference works (16 concurrent requests × 64 tokens each = 1024 tokens in 12.1s → **84.7 tok/s**)

### Key findings (bugs / limitations discovered)

**1. Scale-down to DP=1 is broken.** Calling
`POST /scale_elastic_ep {"new_data_parallel_size": 1}` crashes with:
```
File ".../vllm/distributed/eplb/policy/default.py", line 93, in replicate_experts
    assert num_redundant >= 0
AssertionError
```
The EPLB policy assumes multi-rank EP; with `num_redundant_experts=0` and EP=1,
the invariant `num_phy >= num_log` doesn't hold. After the error, the server
is stuck in "scaling" state (middleware returns 503 indefinitely) and must be
killed. **Scale targets must be ≥2 for the default EPLB config.** vLLM's own
test suite (`tests/distributed/test_elastic_ep.py::test_elastic_ep_scaling`)
exercises 2→4 and 4→2 with `num_redundant_experts=0` and passes, so our
Phase 1 plan (4→2→4) should work.

**2. Memory is tight with EP=2 on A100-40GB.** Qwen3-30B-A3B is ~61GB in BF16
→ ~30.5GB/GPU at EP=2. With `--gpu-memory-utilization 0.85` (34GB budget),
only ~4GB remains for KV cache per GPU → max concurrency ~2x for 4K-token
requests. With EP=4, each GPU has ~15.25GB weights + ~19GB KV cache → much
more headroom. **Phase 1 at DP=4 should be comfortable.**

### Smoke-test artifacts
- Log: `results/smoke/serve.log` (successful DP=2 startup and inference)
- Benchmark: 84.7 tok/s at DP=2 (16 concurrent, 64-token completions)

## Elastic EP Investigation Findings

### Architecture (from vLLM v0.19.0 source)

**How Elastic EP works:**
1. **Endpoint:** `POST /scale_elastic_ep` accepts `{"new_data_parallel_size": N, "drain_timeout": 120}`
2. **Middleware:** `ScalingMiddleware` returns 503 to all requests during rescaling
3. **Scale-down flow:** Engines at ranks >= N get `SHUTDOWN_CURRENT_RANK`, remaining engines reinitialize NCCL groups
4. **Scale-up flow:** Existing engines get `ReconfigureDistributedRequest`, new Ray actors spawned, expert weights transferred P2P, EPLB reshuffles expert placement
5. **Stateless NCCL groups:** Key innovation — uses `StatelessGroupCoordinator` instead of global process groups, enabling dynamic communicator creation/destruction

**Key constraints:**
- Ray backend mandatory (`--data-parallel-backend ray`)
- EPLB required (`--enable-eplb`)
- No pipeline parallelism
- Single API server (`--api-server-count 1`)
- EP size = TP × DP (no explicit `--expert-parallel-size` flag)

**Key source files (in `vllm/` submodule):**
- `vllm/entrypoints/serve/elastic_ep/api_router.py` — HTTP endpoint
- `vllm/entrypoints/serve/elastic_ep/middleware.py` — 503 during scaling
- `vllm/v1/engine/core_client.py` — `AsyncMPClient._scale_{up,down}_elastic_ep()`
- `vllm/distributed/elastic_ep/elastic_state.py` — state machine for scaling
- `vllm/distributed/elastic_ep/elastic_execute.py` — weight transfer logic
- `vllm/config/parallel.py` — `enable_elastic_ep` flag definition
- `examples/online_serving/elastic_ep/scale.py` — example client

### Scale-Up State Machine
```
WAIT_NEW_CORE_ENGINES_INIT → CREATE_STANDBY_GROUPS → TRANSFER_EXPERT_MAPPING
→ WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT → TRANSFER_WEIGHTS → SYNC_KV_CACHE_MEMORY_SIZE
→ SWITCH_AND_PREPARE → EPLB_RESHUFFLE → COMPLETE
```

## Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| vLLM v0.19.0 | Installed | `.venv/`, `uv pip install vllm` |
| Ray v2.55.0 | Installed | `.venv/`, `uv pip install "ray[default]"` |
| Qwen3-30B-A3B | Downloading | `/data/models--Qwen--Qwen3-30B-A3B/`, ~60GB |
| NCCL shim | Available | `common/shim/` (may be needed for Phase 3) |
| Docker | Available | 27.3.1 with NVIDIA Container Toolkit |

## Hardware

- **Machine:** node192
- **GPUs:** 3x NVIDIA A100-SXM4-40GB + 1x NVIDIA A100-PCIE-40GB
- **Driver:** 590.44.01, CUDA 12.4–13.1 available
- **Network:** enp3s0f0np0 (10.0.2.192)

**GPU Topology** (heterogeneous):
```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV12    SYS     SYS     ← NUMA 0
GPU1    NV12     X      SYS     SYS     ← NUMA 0
GPU2    SYS     SYS      X      PIX     ← NUMA 1
GPU3    SYS     SYS     PIX      X      ← NUMA 1
```
- GPU0 ↔ GPU1: NVLink 3.0 (12 links, full bandwidth)
- GPU2 ↔ GPU3: PIX (single PCIe bridge)
- Cross-group: SYS (cross-NUMA via QPI, slowest)

**Impact on EP:** All-to-All communication for expert dispatch will have mixed
bandwidth. GPU0-1 experts can communicate fast (NVLink), but cross-group
communication (expert on GPU0 routing to GPU3) traverses QPI. This is the
baseline topology constraint — same for all phases.

## References

- vLLM Elastic EP RFC: https://github.com/vllm-project/vllm/issues/20323
- vLLM source: https://github.com/vllm-project/vllm
- Expert-as-a-Service: https://arxiv.org/abs/2509.17863
- ElasticMoE: https://arxiv.org/abs/2510.02613
- Lazarus (elastic MoE training): https://arxiv.org/abs/2407.04656
- Research plan: `docs/research_plan_v4.html` Section 8
