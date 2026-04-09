/*
 * cumem_ipc.h — cuMem VMM allocation, export, and import.
 *
 * Uses CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR for cross-container
 * GPU memory sharing via FD passing (SCM_RIGHTS).
 */

#ifndef CUMEM_IPC_H
#define CUMEM_IPC_H

#include <cuda.h>
#include <stddef.h>

/* Holds a cuMem VMM allocation */
typedef struct {
    CUmemGenericAllocationHandle handle;
    CUdeviceptr dptr;
    size_t size;         /* actual allocation size (rounded to granularity) */
    size_t req_size;     /* requested size */
    int gpu_id;
} cumem_alloc_t;

/* Holds an imported cuMem mapping */
typedef struct {
    CUmemGenericAllocationHandle handle;
    CUdeviceptr dptr;
    size_t size;
} cumem_import_t;

/*
 * Query the minimum allocation granularity for the given GPU.
 */
size_t cumem_get_granularity(int gpu_id);

/*
 * Allocate GPU memory with cuMem VMM, requesting a shareable FD handle.
 * Size is rounded up to the allocation granularity.
 */
int cumem_alloc(cumem_alloc_t *out, size_t req_size, int gpu_id);

/*
 * Export allocation as a POSIX file descriptor.
 * Caller must close the FD when done.
 */
int cumem_export_fd(const cumem_alloc_t *alloc, int *fd_out);

/*
 * Import a cuMem handle from a received FD and map it.
 * Sets P2P read/write access for local_gpu_id.
 */
int cumem_import_and_map(cumem_import_t *out, int fd, size_t size,
                         int local_gpu_id, int remote_gpu_id);

/*
 * Free a cuMem allocation.
 */
void cumem_free(cumem_alloc_t *alloc);

/*
 * Unmap and release an imported cuMem handle.
 */
void cumem_import_free(cumem_import_t *imp);

#endif /* CUMEM_IPC_H */
