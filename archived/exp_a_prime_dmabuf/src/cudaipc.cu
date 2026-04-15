/*
 * cudaipc.cu — Legacy cudaIPC handle export/import.
 */

#include "cudaipc.h"
#include <cstdio>
#include <cstring>

#define CHECK_CUDA(call, msg) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[cudaipc] %s: %s\n", msg, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

int cudaipc_alloc_and_export(void **dptr, cudaIpcMemHandle_t *handle,
                             size_t size, int gpu_id) {
    CHECK_CUDA(cudaSetDevice(gpu_id), "cudaSetDevice");
    CHECK_CUDA(cudaMalloc(dptr, size), "cudaMalloc");
    CHECK_CUDA(cudaIpcGetMemHandle(handle, *dptr), "cudaIpcGetMemHandle");
    fprintf(stderr, "[cudaipc] allocated %zu bytes on GPU %d\n", size, gpu_id);
    return 0;
}

int cudaipc_import(void **dptr, const cudaIpcMemHandle_t *handle, int remote_gpu_id) {
    /* Enable peer access to the exporter's GPU */
    int can_access = 0;
    int local_gpu;
    cudaGetDevice(&local_gpu);
    cudaDeviceCanAccessPeer(&can_access, local_gpu, remote_gpu_id);
    if (can_access) {
        cudaError_t err = cudaDeviceEnablePeerAccess(remote_gpu_id, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr, "[cudaipc] cudaDeviceEnablePeerAccess(%d): %s\n",
                    remote_gpu_id, cudaGetErrorString(err));
        } else {
            fprintf(stderr, "[cudaipc] peer access GPU %d -> GPU %d enabled\n",
                    local_gpu, remote_gpu_id);
        }
    } else {
        fprintf(stderr, "[cudaipc] WARNING: GPU %d cannot access GPU %d\n",
                local_gpu, remote_gpu_id);
    }

    CHECK_CUDA(cudaIpcOpenMemHandle(dptr, *handle,
               cudaIpcMemLazyEnablePeerAccess),
               "cudaIpcOpenMemHandle");
    fprintf(stderr, "[cudaipc] imported handle -> %p\n", *dptr);
    return 0;
}

void cudaipc_free(void *dptr) {
    cudaFree(dptr);
}

void cudaipc_import_close(void *dptr) {
    cudaIpcCloseMemHandle(dptr);
}
