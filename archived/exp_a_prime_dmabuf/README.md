# Exp A': DMA-BUF Feasibility Spike

**Phase 1.5 of the XTrans research plan (Section 6.2)**

## Goal

Test whether Linux DMA-BUF file descriptors can serve as a **vendor-agnostic**
cross-container GPU memory sharing mechanism. This is the single most important
de-risking experiment for the cross-vendor XTrans story.

## Why This Matters

Pre-research discovered that DMA-BUF FDs are available on all three GPU vendors:
- **NVIDIA**: cuMem VMM handles are internally DMA-BUF FDs on Linux
- **AMD**: `hsa_amd_portable_export_dmabuf(ptr, size, &fd, &offset)`
- **Intel**: Level Zero `ze_ipc_mem_handle_t` contains a DMA-BUF FD

If DMA-BUF export → FD passing via UDS → DMA-BUF import → GPU P2P mapping works
on NVIDIA (where we have hardware), that's strong evidence for a universal mechanism.

## Success Criteria

- DMA-BUF FD passing achieves ≥95% of native cuMem IPC bandwidth on A100
- Across containers with separate PID/IPC/network namespaces
- The only shared resource: a bind-mounted UDS socket directory

## Hardware

- node192: 4x NVIDIA A100 80GB, NVLink 3.0

## Tasks

### A'1: NVIDIA DMA-BUF P2P Test

Write a minimal C/CUDA program that:
1. Process A (container 0, GPU 0): Allocate GPU memory via `cuMemCreate`
2. Export as shareable handle: `cuMemExportToShareableHandle` with `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`
3. Pass FD to Process B via UDS `SCM_RIGHTS` (through bind-mounted socket dir)
4. Process B (container 1, GPU 1): Import via `cuMemImportFromShareableHandle`
5. Map imported memory: `cuMemMap` + `cuMemSetAccess`
6. Run a simple P2P bandwidth test (GPU 1 reads from GPU 0's memory over NVLink)

This tests the exact mechanism NCCL uses internally for cuMem P2P, but exercised
explicitly across containers.

### A'2: DMA-BUF vs cuMem Overhead

Compare three IPC paths head-to-head:
1. **Native cuMem FD** (what NCCL uses with NCCL_CUMEM_ENABLE=1)
2. **DMA-BUF export/import** (if different from path 1)
3. **Legacy cudaIpcMemHandle** (requires shared IPC namespace)

Measure:
- Init latency (handle creation + exchange + mapping)
- Steady-state P2P bandwidth (1MB–1GB transfers)
- Latency for small messages (1B–1KB)

### A'3: Kernel Requirements

Test on different kernel versions:
- 5.15 (our node192)
- 6.x (if available)

Document:
- Required kernel modules (nvidia-uvm, etc.)
- Required capabilities (CAP_SYS_ADMIN? CAP_DAC_OVERRIDE?)
- DMA-BUF export availability (which CUDA versions support it)

## Implementation Notes

The cuMem VMM API (CUDA driver API, not runtime API):
```c
// Allocate
CUmemGenericAllocationHandle handle;
cuMemCreate(&handle, size, &prop, 0);
cuMemAddressReserve(&ptr, size, 0, 0, 0);
cuMemMap(ptr, size, 0, handle, 0);
cuMemSetAccess(ptr, size, &accessDesc, 1);

// Export as FD
int fd;
cuMemExportToShareableHandle(&fd, handle,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);

// Send FD via UDS SCM_RIGHTS...

// Import on peer
CUmemGenericAllocationHandle imported;
cuMemImportFromShareableHandle(&imported, (void*)(intptr_t)fd,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
cuMemMap(peer_ptr, size, 0, imported, 0);
cuMemSetAccess(peer_ptr, size, &accessDesc, 1);
```

UDS FD passing (standard Unix technique):
```c
// Send FD
struct msghdr msg = {0};
struct cmsghdr *cmsg;
char buf[CMSG_SPACE(sizeof(int))];
msg.msg_control = buf;
msg.msg_controllen = sizeof(buf);
cmsg = CMSG_FIRSTHDR(&msg);
cmsg->cmsg_level = SOL_SOCKET;
cmsg->cmsg_type = SCM_RIGHTS;
cmsg->cmsg_len = CMSG_LEN(sizeof(int));
*(int *)CMSG_DATA(cmsg) = fd;
sendmsg(sock, &msg, 0);
```

## Results Directory

```
results/
├── dmabuf_p2p_bw.json     # DMA-BUF P2P bandwidth results
├── cumem_native_bw.json   # Native cuMem IPC bandwidth (comparison)
├── cudaipc_bw.json        # Legacy cudaIPC bandwidth (comparison)
└── kernel_compat.md       # Kernel version compatibility notes
```
