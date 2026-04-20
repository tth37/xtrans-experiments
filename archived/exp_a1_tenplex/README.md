# Exp A1: Elastic GPU Scaling with Tenplex

## Overview

**System under test:** [Tenplex](https://github.com/kungfu-team/tenplex) (SOSP '24)
**Scheduling operation:** Elastic GPU add/remove — change TP+PP+DP at runtime
**Research plan:** `docs/research_plan_v4.html` Section 6

Tenplex enables elastic GPU scaling for distributed training via Parallelizable
Tensor Collections (PTCs). It can add/remove GPUs from a running training job
and reconfigure parallelism dimensions (TP, PP, DP) without restart.

## Tenplex Architecture

**Tenplex uses multi-GPU Docker containers internally.** The `tenplex-run` CLI
is a Go orchestrator that SSHes to hosts and runs `docker run --network host`
to create training containers. Training runs inside these containers via
`torchrun` + Megatron-LM.

### Execution Flow

`tenplex-run` orchestration (see `tenplex-run/runop/setup.go:Main()`):

1. **CleanMachines** — stops existing `trainer-*` containers, clears `~/.tenplex/training/`
2. **PrepareVMs** — clones `transformer-checkpoint` repo to `~/.tenplex/`
3. **PullImages** — `docker pull` the training image on all hosts
4. **ScalingTraining** — runs the training loop, executing the scaling schedule

### Container Lifecycle During Scaling

**Every parallelism reconfiguration requires full container stop and recreate.**
When scaling from 4 GPUs to 2 GPUs (see `tenplex-run/runop/runop.go:ScalingTraining()`):

1. Training receives stop signal (HTTP `/stop` endpoint)
2. Training process exits → container auto-removes (`docker run --rm`)
3. `tenplex-state-transformer` runs on the host to repartition PTC state
4. New container starts with `docker run --gpus device=0,1` (new GPU assignment)
5. New container loads transformed state and resumes training

There is no mechanism to change a running container's GPU assignment — this is
a Docker limitation. The `--gpus` flag is fixed at container creation time.

### Docker Configuration

| Setting | Value |
|---------|-------|
| Network | `--network host` (shared host network namespace) |
| GPU assignment | `--gpus device=0,1,2,3` (fixed at creation) |
| Shared memory | `--shm-size 1g` |
| Container naming | `trainer-{jobid}-{rank}` |
| Auto-cleanup | `--rm` (container removed on exit) |
| Training command | `torchrun --nproc_per_node N pretrain_gpt.py ...` |

### Required Docker Image

`kungfu.azurecr.io/mw-megatron-lm-23.06-update:v0.0.3`

This is a private Azure Container Registry image. If unavailable, build from
NGC base `nvcr.io/nvidia/pytorch:24.07-py3` + Megatron-LM + Tenplex (see
`Dockerfile`).

## Experiment Goal

Reproduce Tenplex's elastic scheduling (parallelism reconfiguration) on our
hardware. Observe the multi-GPU container lifecycle firsthand and identify
overheads and limitations. Findings will inform subsequent experiment phases.

### Phase 1: Reproduce Elastic Scheduling

Run `tenplex-run` on the host with a scaling schedule (4→2→4 GPUs). Observe:

**Container lifecycle:**
- Are containers fully stopped and recreated at each scaling event?
- How long is each phase (state save → transform → container recreate → state load)?
- What is the total GPU idle time during reconfiguration?

**Training behavior:**
- Training throughput at each parallelism config (tokens/s)
- NCCL transport used (NVLink P2P vs SHM vs TCP)
- Does training converge correctly across reconfiguration events?

**Multi-GPU container observations:**
- After scale-down (4→2), are freed GPUs visible to other Docker containers?
- What does `docker ps` / `nvidia-smi` show during and after reconfiguration?
- How does Tenplex coordinate the stop-transform-restart cycle?

**Script:** `scripts/phase1_native.sh`

### Future Phases

TBD based on Phase 1 findings. Potential directions:
- Production container wrapping (K8s-like deployment, GPU trapping)
- Per-GPU containers (scaling = container lifecycle)
- Cross-container NCCL communication (shim from `common/shim/`)

## Measurements

| Metric | How to Measure |
|--------|---------------|
| Training throughput | Megatron-LM logs (tokens/s or samples/s) |
| Reconfiguration time | Wall clock from scaling trigger to training resumption |
| Container lifecycle | `docker events` + `docker ps` timestamps |
| NCCL transport | `NCCL_DEBUG=INFO` logs — look for P2P/SHM/NET transport |
| GPU idle time | `nvidia-smi -l 1` during reconfiguration |
| PTC state I/O time | `tenplex-state-transformer` logs |
| Errors and failures | Full stderr, NCCL debug logs, container events |

Save results using the observation recorder:
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
  tenplex/                # Tenplex source (git submodule)
  Dockerfile              # Docker image with Tenplex + Megatron-LM
  compose.phase2.yml      # (reserved for future Phase 2)
  configs/
    para-config.json      # Parallelism configs for 4 and 2 GPUs
    schedule.json         # Scaling schedule (4→2→4 GPUs, time-based)
    hosts.txt             # Single host (10.0.2.192)
  scripts/
    phase1_native.sh      # Phase 1: run tenplex-run on host
  results/                # Raw data (gitignored)
  analysis_report.html    # Final report (copy from docs/report_template.html)
  analysis_assets/        # Figures for report
```

## Dependencies

| Dependency | Status | Location |
|-----------|--------|----------|
| Go 1.26.2 | Installed | `/usr/local/go/bin/go` |
| tenplex-run | Built | `tenplex/bin/tenplex-run` |
| tenplex-state-transformer | Built | `tenplex/bin/tenplex-state-transformer` |
| mlfsd (tensor store) | Built | `tenplex/bin/mlfsd` |
| Docker image | Pull in progress | `kungfu.azurecr.io/mw-megatron-lm-23.06-update:v0.0.3` |
| mlfs daemon | Not running | Start: `tenplex/bin/mlfsd --ctrl-port 20010` |
| Dataset (enwiki) | Not checked | Expected at `/data/megatron-lm/gpt-2/enwiki/` |
| SSH to localhost | Working | Fixed: `chmod 600 ~/.ssh/authorized_keys` |

## Hardware

- **Machine:** node192
- **GPUs:** 3x NVIDIA A100-SXM4-40GB + 1x NVIDIA A100-PCIE-40GB
- **Interconnect:** NVLink 3.0 (between SXM4 GPUs)
- **Network:** `enp3s0f0np0` (10.0.2.192), InfiniBand available (`/dev/infiniband/rdma_cm`)
- **CUDA:** 12.4+, NCCL 2.21.5
- **Docker:** 27.3.1 with NVIDIA Container Toolkit

## mlfs Daemon

The mlfs daemon (`mlfsd`) is Tenplex's tensor store, used for:
- Mounting datasets into training containers
- Tracking training iteration state (for time-based scheduling)
- Serving checkpoint state during repartitioning

Start manually:
```bash
tenplex/bin/mlfsd --ctrl-port 20010
```

Or the benchmark scripts expect a systemd service: `systemctl restart mlfs`.

## References

- Tenplex paper: https://dl.acm.org/doi/10.1145/3694715.3695964
- Tenplex code: https://github.com/kungfu-team/tenplex
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Research plan: `docs/research_plan_v4.html` Section 6
- Report template: `docs/report_template.html`
- Report guide: `docs/report_guide.md`
