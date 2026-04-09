/*
 * cudaipc.h — Legacy cudaIPC handle export/import.
 * Requires --ipc=host (shared IPC namespace).
 */

#ifndef CUDAIPC_H
#define CUDAIPC_H

#include <cuda_runtime.h>
#include <stddef.h>

/* Allocate GPU memory with cudaMalloc and get IPC handle */
int cudaipc_alloc_and_export(void **dptr, cudaIpcMemHandle_t *handle,
                             size_t size, int gpu_id);

/* Import IPC handle and get device pointer */
int cudaipc_import(void **dptr, const cudaIpcMemHandle_t *handle, int remote_gpu_id);

/* Free exported allocation */
void cudaipc_free(void *dptr);

/* Close imported handle */
void cudaipc_import_close(void *dptr);

#endif /* CUDAIPC_H */
