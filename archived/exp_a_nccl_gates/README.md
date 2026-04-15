# Exp A: NCCL Gate Taxonomy and Minimal Interception

**Phase 1 of the XTrans research plan (Section 6.1)**

## Goal

Transform the ad-hoc exp2 workaround (env vars + shared mounts) into a principled,
minimal interception layer. Understand exactly which system calls NCCL makes for
each gate, and build the thinnest possible LD_PRELOAD shim that satisfies them.

## Success Criteria

- The shim (`libxtrans_shim.so`) recovers ≥99% of NVLink bandwidth on A100 (node192)
- Without any NCCL-specific env vars (no `NCCL_CUMEM_ENABLE`, `NCCL_HOSTID`, etc.)
- Without shared `/dev/shm` between containers
- Without `--network=host`
- The only requirements: the shim library + a bind-mounted UDS socket directory

## Hardware

- node192: 4x NVIDIA A100 80GB, NVLink 3.0 (600 GB/s bidirectional)
- NCCL 2.21.5 (primary), test against 2.18-2.28 in task A3

## Tasks

### A1: Syscall Tracing

Map every system call NCCL makes during communicator init that differs between
working (shared namespace) and broken (isolated container) configurations.

```bash
# 1. Run NCCL in shared container with strace
strace -f -e trace=network,ipc,file -o traces/shared.strace \
    all_reduce_perf -b 8 -e 128M -g 1

# 2. Run NCCL in isolated per-GPU container with strace
strace -f -e trace=network,ipc,file -o traces/isolated.strace \
    all_reduce_perf -b 8 -e 128M -g 1

# 3. Diff the traces
diff traces/shared.strace traces/isolated.strace > traces/diff.txt
```

**Expected findings** (based on source analysis):
- `gethostname()` — returns different values → hostHash mismatch
- `stat("/dev/shm")` — returns different st_dev → shmDev mismatch
- `socket(AF_UNIX, ...)` + `connect()` — abstract UDS fails across network namespaces
- Possibly: `open("/proc/sys/kernel/random/boot_id")` if NCCL_HOSTID not set

**Deliverable**: Complete syscall-level map of all NCCL gates. Document any
gates beyond the three we know about (hostHash, shmDev, IPC socket).

### A2: Minimal LD_PRELOAD Shim

Build and test the shim in `common/shim/`.

```bash
# Build
cd common/shim && make

# Test in isolated per-GPU containers
# Container 0:
LD_PRELOAD=/workspace/xtrans/common/shim/libxtrans_shim.so \
XTRANS_HOSTNAME=node192 \
XTRANS_SHMDEV=0x1 \
XTRANS_VERBOSE=1 \
    all_reduce_perf -b 1M -e 1G -g 1

# Compare bandwidth vs:
#   a) Shared container baseline (no shim, all GPUs visible)
#   b) Isolated container without shim (expect NET fallback)
#   c) exp2 env var workaround (NCCL_CUMEM_ENABLE + NCCL_HOSTID)
```

**Key question**: Does the shim alone recover P2P, or do we also need to handle
the IPC socket (abstract UDS for cuMem FD passing)? If NCCL needs cuMem FD
exchange over UDS and the containers have separate network namespaces, we need
either:
- A shared filesystem UDS socket (bind-mounted directory)
- Or: `NCCL_IPC_USE_ABSTRACT_SOCKET=0` + shared filesystem socket path

The shim currently handles gethostname and stat. Based on A1 results, we may
need to add `connect()` interception for the IPC socket path.

**Deliverable**: `libxtrans_shim.so` (~200 lines C) that recovers NVLink P2P
without NCCL_* env vars.

### A3: Version Stability Test

Test the shim against multiple NCCL versions to build a compatibility matrix.

Target versions: 2.18, 2.19, 2.20, 2.21 (current), 2.25, 2.27, 2.28

For each version:
1. Install in container (or use different NGC container tags)
2. Run with shim → measure bandwidth
3. Run strace → verify same syscalls are intercepted
4. Note any new gates or changed behavior

**Deliverable**: Compatibility matrix showing which NCCL versions the shim works
with and any version-specific adaptations needed.

### A4: Security Analysis

Analyze the threat model of the shim approach.

Questions:
- Can a malicious container use the shim to access another tenant's GPU memory?
- What's the minimal privilege needed?
- How does XTrans's controlled P2P compare to `--ipc=host` (uncontrolled)?

**Deliverable**: Threat model document.

## Results Directory

Store all outputs in `results/` (gitignored):
```
results/
├── traces/           # strace output files
├── baseline.json     # Shared-container bandwidth
├── isolated.json     # Isolated-container bandwidth (broken)
├── shim.json         # Shim-recovered bandwidth
├── exp2_workaround.json  # env-var workaround (for comparison)
└── version_matrix.md # NCCL version compatibility
```

## Relationship to exp2

This experiment builds directly on exp2 (in vllm-source). Key differences:
- exp2 used env vars (NCCL_CUMEM_ENABLE, NCCL_HOSTID, shared /dev/shm, host network)
- This experiment replaces those with a principled LD_PRELOAD shim
- The shim approach is more portable and doesn't require NVIDIA-specific env vars
- Results should show identical bandwidth recovery (both achieve ≥99% NVLink)
