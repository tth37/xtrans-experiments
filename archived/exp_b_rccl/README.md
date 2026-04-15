# Exp B: AMD RCCL Cross-Container Recovery

**Phase 2 of the XTrans research plan (Section 6.3)**

## Goal

Replicate the exp2/Exp A methodology on AMD hardware. Discover RCCL's container
isolation gates and test three IPC paths for cross-container P2P recovery.

## Key Pre-Research Findings

- RCCL is a hard fork of NCCL — uses **same three gates** (hostHash, shmDev, pidHash)
- RCCL **explicitly disables cuMem VMM on AMD**: `ncclIsCuMemSupported()` returns 0
  on `__HIP_PLATFORM_AMD__` (in `src/misc/rocmwrap.cc`)
- Three potential cross-container IPC paths (in priority order):

| Path | Mechanism | Handle Type | Status |
|------|-----------|-------------|--------|
| **B2a** | Opaque HSA IPC via `/dev/kfd` | 32-byte blob | May already work (kernel-resolved) |
| **B2b** | DMA-BUF export (`hsa_amd_portable_export_dmabuf`) | FD | Kernel-mature, untested for P2P |
| **B2c** | HIP VMM FD (`hipMemHandleTypePosixFileDescriptor`) | FD | Beta API in HIP 7.x |

## Success Criteria

- At least one AMD IPC path recovers ≥90% of xGMI bandwidth across containers
- Complete RCCL gate taxonomy (equivalent of our NCCL deep-dive)
- Bandwidth recovery curve analogous to NCCL exp2 results

## Hardware Requirements

- AMD MI250X or MI300X (cloud: Azure ND-MI300X-v5, AMD Accelerator Cloud)
- NOT available in our lab — cloud access required

## Tasks

### B1: RCCL Source Analysis (can do without hardware)

Analyze RCCL source code to document:
- `getHostHash()` implementation (should match NCCL's)
- `shmCanConnect()` / `shmDev` check
- Transport selection logic (P2P > SHM > NET)
- Where cuMem is disabled and why
- HSA IPC handle creation/exchange flow

RCCL repo: https://github.com/ROCm/rccl (may have moved to rocm-systems)

### B2: Three-Path IPC Test (requires hardware)

**B2a: Opaque HSA IPC handles across containers**

The cheapest experiment. `hsa_amd_ipc_memory_create` produces opaque 32-byte
handles resolved kernel-side via `/dev/kfd`. If both containers share `/dev/kfd`
and can see the same GPU, this might just work.

```c
// Container A: export
hsa_amd_ipc_memory_t handle;
hsa_amd_ipc_memory_create(ptr, size, &handle);
// Send 32 bytes to Container B via shared file or socket

// Container B: import
void *mapped;
hsa_amd_ipc_memory_attach(&handle, size, 1, &agent, &mapped);
```

**B2b: DMA-BUF export/import for GPU P2P**

Test whether `hsa_amd_portable_export_dmabuf` can be used for P2P IPC (not just
RDMA NIC registration, which is how RCCL currently uses it).

```c
// Export
int dmabuf_fd;
uint64_t offset;
hsa_amd_portable_export_dmabuf(ptr, size, &dmabuf_fd, &offset);
// Pass dmabuf_fd via UDS SCM_RIGHTS

// Import (unclear API — may need KFD ioctl)
// AMDKFD_IOC_IMPORT_DMABUF ?
```

**B2c: HIP VMM FD path**

```c
hipMemGenericAllocationHandle_t handle;
hipMemCreate(&handle, size, &prop, 0);
int fd;
hipMemExportToShareableHandle(&fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
// Pass fd via UDS SCM_RIGHTS, import on peer
```

### B3: Baseline Measurement

Measure RCCL allreduce bandwidth in three configs:
1. Shared container (all GPUs visible) — baseline
2. Per-GPU containers, default config — broken
3. Per-GPU containers with working IPC path from B2

### B4: RCCL Shim

Build `libxtrans_rccl.so` using the same architecture as the NCCL shim.
Same interceptions (gethostname, stat) since RCCL uses the same gates.

## Notes for Cloud Setup

When provisioning AMD GPU cloud instances:
- Ensure Docker + ROCm container toolkit are installed
- Test that `/dev/kfd` and `/dev/dri/renderD*` are accessible
- ROCm container images: `rocm/pytorch:latest` or `rocm/rocm-terminal:latest`
- RCCL tests: `rccl-tests` (same interface as nccl-tests)
