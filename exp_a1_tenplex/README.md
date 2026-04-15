# Exp A1: Elastic GPU Scaling with Tenplex in Containers

## Overview

**System under test:** [Tenplex](https://github.com/kungfu-team/tenplex) (SOSP '24)
**Scheduling operation:** Elastic GPU add/remove — change TP+PP+DP at runtime
**Research plan:** `docs/research_plan_v4.html` Section 6

Tenplex enables elastic GPU scaling for distributed training via Parallelizable
Tensor Collections (PTCs). It can add/remove GPUs from a running training job
and reconfigure parallelism dimensions (TP, PP, DP) without restart.

**Tenplex assumes bare metal.** All GPUs share OS namespaces, NCCL communicators
are created freely, processes discover each other via shared-memory or localhost.
This experiment progressively containerizes Tenplex to discover what breaks.

## Three-Phase Structure

### Phase 1: Bare Metal (Baseline)

Confirm Tenplex works on node192 (4x A100) without containers.

**Steps:**
1. Clone and build Tenplex + Megatron-LM
2. Train GPT-2 (small) with TP=2, DP=2 on 4 GPUs
3. Trigger elastic scale-down: 4 -> 2 GPUs
4. Trigger elastic scale-up: 2 -> 4 GPUs
5. Record baseline metrics

**Key recordings:**
- Training throughput (tokens/s)
- Reconfiguration time for scale-down and scale-up
- NCCL transport used (NVLink P2P vs SHM vs TCP)
- Tenplex PTC repartitioning I/O time

### Phase 2: Multi-GPU Container (Production Approach)

Put all 4 GPUs in a single container. This is the standard production deployment.

**Steps:**
1. Build Docker image with Tenplex + Megatron-LM (use `Dockerfile`)
2. Start single container with all 4 GPUs (`compose.phase2.yml`)
3. Run same training as Phase 1
4. Trigger elastic scale-down: 4 -> 2 GPUs *inside the container*
5. **From outside:** try to use the "freed" GPUs (launch another container)
6. Trigger elastic scale-up: try to add *new* GPUs to the running container

**Key questions to answer:**
- Does Tenplex reconfigure successfully inside a container? Any difference vs bare metal?
- After scale-down, are the freed GPUs available to other jobs? Or are they trapped?
- Can K8s/Docker add new GPUs to a running container? Or must you recreate it?
- What does `nvidia-smi` show inside vs outside the container after scale-down?

**This phase determines whether we proceed to Phase 3.** If freed GPUs are
reclaimable at the cluster level, multi-GPU containers may be sufficient and
Phase 3 is not motivated.

### Phase 3: Per-GPU Containers

Each GPU in its own isolated container. Scaling = container lifecycle.

**Steps:**
1. Start 4 per-GPU containers (`compose.phase3.yml`)
2. Launch Tenplex training across all 4 containers
3. Scale-down: `docker stop` containers C2 and C3
4. Observe: does Tenplex detect the loss? Can it reconfigure?
5. Scale-up: start new containers C2' and C3'
6. Observe: can Tenplex incorporate the new containers?
7. Record all observations

**Only proceed here if Phase 2 reveals limitations.**

## Workaround Levels (Phase 3)

If Phase 3 encounters problems, try these progressively:

| Level | Workaround | Description |
|-------|-----------|-------------|
| 0 | Default | No special measures — see how far things work |
| 1 | Configuration | Shared volumes for state, env vars for discovery, network config |
| 2 | Framework patch | Modify Tenplex source: process discovery, NCCL comm lifecycle |
| 3 | System-level | LD_PRELOAD shim (from `common/shim/`) to make NCCL's namespace checks pass |
| 4 | Relaxed isolation | Shared /dev/shm, shared network NS, shared IPC NS |

## Measurements

Record at EVERY phase, for both scale-down and scale-up:

| Metric | How to Measure |
|--------|---------------|
| Training throughput | Megatron-LM logs (tokens/s or samples/s) |
| Reconfiguration time | Wall clock from trigger to training resumption |
| NCCL transport | `NCCL_DEBUG=INFO` logs — look for P2P/SHM/NET transport |
| GPU resource state | `nvidia-smi` inside and outside container |
| PTC state I/O time | Tenplex logs |
| Errors and failures | Full stderr, NCCL debug logs, container events |

Save all results to `results/<phase>_<step>.json` using the observation recorder:
```bash
python ../common/harness/record_observation.py \
    --experiment exp_a1 \
    --phase phase1 \
    --step "scale_down_4_to_2" \
    --output results/phase1_scale_down.json \
    --notes "describe what happened"
```

## File Organization

```
exp_a1_tenplex/
  README.md               # This file
  Dockerfile              # Container image with Tenplex + deps
  compose.phase2.yml      # Phase 2: single multi-GPU container
  compose.phase3.yml      # Phase 3: per-GPU containers
  scripts/
    phase1_bare_metal.sh  # Phase 1 execution guide
    phase2_multi_gpu.sh   # Phase 2 execution guide
    phase3_per_gpu.sh     # Phase 3 execution guide
  results/                # Raw data (gitignored)
  analysis_report.html    # Final report (copy from docs/report_template.html)
  analysis_assets/        # Figures for report
```

## Hardware

- **Machine:** node192
- **GPUs:** 4x NVIDIA A100-SXM4-40GB
- **Interconnect:** NVLink 3.0 (600 GB/s bidirectional)
- **CUDA:** 12.4+, NCCL 2.21.5
- **Docker:** 27.3.1 with NVIDIA Container Toolkit

## References

- Tenplex paper: https://dl.acm.org/doi/10.1145/3694715.3695964
- Tenplex code: https://github.com/kungfu-team/tenplex
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Research plan: `docs/research_plan_v4.html` Section 6
- Report template: `docs/report_template.html`
- Report guide: `docs/report_guide.md`
