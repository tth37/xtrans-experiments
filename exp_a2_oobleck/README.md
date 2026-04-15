# Exp A2: Fault-Tolerant Training with Oobleck in Containers

## Overview

**System under test:** [Oobleck](https://github.com/SymbioticLab/Oobleck) (SOSP '23)
**Scheduling operation:** Fault-tolerant pipeline training — recover from GPU failure
**Research plan:** `docs/research_plan_v4.html` Section 7

Oobleck enables resilient distributed training using pipeline templates. It
pre-computes pipeline configurations for different GPU counts. When a GPU fails,
Oobleck selects the appropriate template for the surviving GPUs and redistributes
pipeline stages — continuing training without a full checkpoint-and-restart.

**Oobleck assumes bare metal.** It detects failures via process exit / NCCL
timeout. All GPUs share namespaces. Communicator reformation is straightforward
because all survivors can see each other. This experiment progressively
containerizes Oobleck to discover what breaks.

## Three-Phase Structure

### Phase 1: Bare Metal (Baseline)

Confirm Oobleck works on node192 (4x A100) without containers.

**Steps:**
1. Clone and build Oobleck
2. Configure pipeline templates for 4, 3, and 2 GPUs
3. Start pipeline training (GPT-2 or similar small model)
4. Kill one process (simulate GPU 3 failure: 4 -> 3 GPUs)
5. Observe recovery: template transition, pipeline redistribution
6. Kill another process (GPU 2 failure: 3 -> 2 GPUs)
7. Observe cascading recovery
8. Record baseline metrics

**Key recordings:**
- Failure detection time (from kill to Oobleck awareness)
- Recovery time (from detection to training resumption)
- Training throughput before, during, and after each failure
- Pipeline template transitions (which templates were selected)

### Phase 2: Multi-GPU Container (Production Approach)

Put all 4 GPUs in a single container.

**Steps:**
1. Build Docker image with Oobleck (`Dockerfile`)
2. Start single container with all 4 GPUs (`compose.phase2.yml`)
3. Run same pipeline training as Phase 1
4. Kill one training process inside the container
5. Observe: does Oobleck recover? Does the container stay healthy?
6. Check: is the failed GPU still allocated? Can K8s replace it?
7. Kill the container entirely (simulate catastrophic GPU failure)
8. Observe: blast radius (all 4 GPUs lost), full restart time

**Key questions to answer:**
- Does Oobleck's internal failure recovery work inside a container?
- After recovering to 3 GPUs, does the container's health probe pass? Or does
  K8s think the pod is unhealthy and kill it (negating Oobleck's recovery)?
- Is the failed GPU (still allocated to the container) usable by anyone? Or is
  it wasted?
- For a real GPU hardware error, does the container crash entirely (all 4 GPUs
  lost) or just the process on that GPU?

**This phase determines whether we proceed to Phase 3.** If failures are
isolated within the container and GPUs can be replaced without pod restart,
Phase 3 is not motivated.

### Phase 3: Per-GPU Containers

Each GPU in its own container. Failure = one container dies, others survive.

**Steps:**
1. Start 4 per-GPU containers (`compose.phase3.yml`)
2. Launch Oobleck pipeline training across all 4
3. Kill container C3 (`docker kill exp-a2-gpu3`)
4. Observe: do C0-C2 detect C3's death? Can they reform NCCL communicator?
   Does Oobleck's template transition work across container boundaries?
5. Start replacement container C3' (new container on same or different GPU)
6. Observe: can Oobleck incorporate C3'?
7. Kill C2 (cascading: 3 -> 2 containers)
8. Observe cascading recovery across containers
9. Record all observations

**Only proceed here if Phase 2 reveals limitations.**

## Workaround Levels (Phase 3)

| Level | Workaround | Description |
|-------|-----------|-------------|
| 0 | Default | No special measures |
| 1 | Configuration | Docker event monitoring for failure detection, shared coordination volume, health checks |
| 2 | Framework patch | Modify Oobleck's failure detection for container events, adjust communicator recreation |
| 3 | System-level | LD_PRELOAD shim (`common/shim/`) for cross-container NCCL |
| 4 | Relaxed isolation | Progressively weaker namespace separation |

## Measurements

Record at EVERY phase:

| Metric | How to Measure |
|--------|---------------|
| Failure detection time | Time from kill to Oobleck log entry |
| Recovery time (total) | Detection + template selection + comm reform + stage redistribution |
| Training throughput | Before failure, during recovery, after recovery |
| Failure blast radius | Phase 2: does 1 GPU failure kill all 4? Phase 3: only 1? |
| GPU replaceability | Can the cluster replace the failed GPU without killing others? |
| NCCL communicator behavior | Does comm hang, crash, or fail gracefully on peer death? |
| Container health model | Does K8s health check conflict with framework recovery? |
| Pipeline template transitions | Which templates selected, timing of each step |

Save results using the observation recorder:
```bash
python ../common/harness/record_observation.py \
    --experiment exp_a2 \
    --phase phase1 \
    --step "gpu3_failure_recovery" \
    --output results/phase1_gpu3_failure.json \
    --notes "describe what happened"
```

## Simulating GPU Failures

Use the failure simulation script:
```bash
# Phase 1 (bare metal): kill a training process
./scripts/simulate_failure.sh --phase bare_metal --target-rank 3

# Phase 2 (multi-GPU container): kill a process inside the container
./scripts/simulate_failure.sh --phase multi_gpu --container exp-a2-multi --target-rank 3

# Phase 3 (per-GPU container): kill an entire container
./scripts/simulate_failure.sh --phase per_gpu --container exp-a2-gpu3
```

## File Organization

```
exp_a2_oobleck/
  README.md               # This file
  Dockerfile              # Container image with Oobleck + deps
  compose.phase2.yml      # Phase 2: single multi-GPU container
  compose.phase3.yml      # Phase 3: per-GPU containers
  scripts/
    phase1_bare_metal.sh  # Phase 1 execution guide
    phase2_multi_gpu.sh   # Phase 2 execution guide
    phase3_per_gpu.sh     # Phase 3 execution guide
    simulate_failure.sh   # GPU failure simulation helper
  results/                # Raw data (gitignored)
  analysis_report.html    # Final report
  analysis_assets/        # Figures for report
```

## Hardware

- **Machine:** node192
- **GPUs:** 4x NVIDIA A100-SXM4-40GB
- **Interconnect:** NVLink 3.0 (600 GB/s bidirectional)
- **CUDA:** 12.4+, NCCL 2.21.5
- **Docker:** 27.3.1 with NVIDIA Container Toolkit

## References

- Oobleck paper: https://dl.acm.org/doi/10.1145/3600006.3613152
- Oobleck code: https://github.com/SymbioticLab/Oobleck
- Bamboo (comparison): https://github.com/uclasystem/bamboo
- Research plan: `docs/research_plan_v4.html` Section 7
- Report template: `docs/report_template.html`
- Report guide: `docs/report_guide.md`
