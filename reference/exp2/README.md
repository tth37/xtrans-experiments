# Reference: exp2 Benchmark Infrastructure

These files are copied from `vllm-source/examples/docker_executor/experiments/exp2_nccl_isolation/`
to provide the working cross-container NCCL benchmark infrastructure for Exp A to build on.

## How exp2 Runs NCCL Across Containers

exp2 does **NOT** use MPI or nccl-tests. It uses **PyTorch torch.distributed** with
`init_method='env://'`, coordinated by Docker Compose:

1. **Docker Compose** launches N containers (one per GPU) as separate services
2. Each container runs `benchmark_docker.py` (single process, no multiprocessing.spawn)
3. Environment variables injected by compose: `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`
4. `torch.distributed.init_process_group(backend='nccl', init_method='env://')` handles NCCL init
5. The `CommBenchmark` class (in `benchmark.py`) runs all_reduce, all_gather, etc. via PyTorch ops
6. NCCL is used under the hood — transport selection, gate checks, P2P setup all happen inside NCCL

## Key Files

| File | Purpose |
|------|---------|
| `benchmark.py` | Core benchmark class — all_reduce, all_gather, reduce_scatter, etc. via torch.distributed |
| `benchmark_docker.py` | Container entry point — inits torch.distributed from env vars, runs benchmarks, saves JSON |
| `compose.baseline.yml` | Shared-namespace config (no isolation) — the performance baseline |
| `compose.cumem_isolation.yml` | **The working config** — per-GPU isolation with cuMem recovery (host network + shared /dev/shm) |
| `compose.shm_isolation.yml` | SHM-only fallback config (for comparison) |
| `Dockerfile` | Container image (PyTorch NGC base + benchmark scripts) |
| `diagnose_nccl.py` | NCCL diagnostic tool — prints topology, transport selection, env vars |

## How to Use for Exp A

For Exp A (NCCL gate taxonomy + shim), adapt these files:

```bash
# 1. Build the container image
cd reference/exp2
docker build -t gpu-comm-benchmark:latest .

# 2. Run baseline (shared namespace, full bandwidth)
docker compose -f compose.baseline.yml up

# 3. Run cumem isolation (exp2 workaround, should recover bandwidth)
docker compose -f compose.cumem_isolation.yml up

# 4. Run with xtrans shim (NEW — this is what Exp A tests)
#    Create a new compose file that:
#    - Uses per-GPU isolation (like compose.cumem_isolation.yml)
#    - Removes NCCL_CUMEM_ENABLE, NCCL_HOSTID, shared /dev/shm, host network
#    - Adds LD_PRELOAD=libxtrans_shim.so instead
#    - Bind-mounts the shim .so and a UDS socket directory
docker compose -f compose.shim.yml up
```

## The Critical Docker Compose Pattern

From `compose.cumem_isolation.yml` — this is the pattern to adapt:

```yaml
services:
  rank0:
    image: gpu-comm-benchmark:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all     # NVML topology discovery
      - CUDA_VISIBLE_DEVICES=0         # Per-GPU compute isolation
      - NCCL_CUMEM_ENABLE=1            # <-- exp2 workaround (shim should replace this)
      - WORLD_SIZE=2
      - RANK=0
      - MASTER_ADDR=127.0.0.1
      - MASTER_PORT=29500
    volumes:
      - shared-shm:/dev/shm            # <-- exp2 workaround (shim should replace this)
    network_mode: host                  # <-- exp2 workaround (shim should replace this)
```

The shim's job is to make NCCL's gates pass WITHOUT the three workarounds marked above
(NCCL_CUMEM_ENABLE, shared /dev/shm, host network).
