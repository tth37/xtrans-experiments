/*
 * cumem_ipc.cu — cuMem VMM allocation, export, and import.
 */

#include "cumem_ipc.h"
#include <cstdio>
#include <cstring>

#define CHECK_CU(call, msg) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *errstr; cuGetErrorString(err, &errstr); \
        fprintf(stderr, "[cumem] %s: %s (%d)\n", msg, errstr, err); \
        return -1; \
    } \
} while(0)

size_t cumem_get_granularity(int gpu_id) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = gpu_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t gran = 0;
    cuMemGetAllocationGranularity(&gran, &prop,
                                  CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    return gran;
}

int cumem_alloc(cumem_alloc_t *out, size_t req_size, int gpu_id) {
    memset(out, 0, sizeof(*out));
    out->gpu_id = gpu_id;
    out->req_size = req_size;

    /* Round up to granularity */
    size_t gran = cumem_get_granularity(gpu_id);
    if (gran == 0) gran = 2 * 1024 * 1024; /* default 2MB */
    size_t alloc_size = ((req_size + gran - 1) / gran) * gran;
    out->size = alloc_size;

    /* Set allocation properties */
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = gpu_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    /* Create allocation */
    CHECK_CU(cuMemCreate(&out->handle, alloc_size, &prop, 0),
             "cuMemCreate");

    /* Reserve virtual address range */
    CHECK_CU(cuMemAddressReserve(&out->dptr, alloc_size, gran, 0, 0),
             "cuMemAddressReserve");

    /* Map */
    CHECK_CU(cuMemMap(out->dptr, alloc_size, 0, out->handle, 0),
             "cuMemMap");

    /* Set access for the local GPU */
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = gpu_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CU(cuMemSetAccess(out->dptr, alloc_size, &access, 1),
             "cuMemSetAccess");

    fprintf(stderr, "[cumem] allocated %zu bytes on GPU %d (dptr=0x%llx, gran=%zu)\n",
            alloc_size, gpu_id, (unsigned long long)out->dptr, gran);
    return 0;
}

int cumem_export_fd(const cumem_alloc_t *alloc, int *fd_out) {
    CHECK_CU(cuMemExportToShareableHandle(fd_out, alloc->handle,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
             "cuMemExportToShareableHandle");

    fprintf(stderr, "[cumem] exported fd=%d for %zu bytes\n",
            *fd_out, alloc->size);
    return 0;
}

int cumem_import_and_map(cumem_import_t *out, int fd, size_t size,
                         int local_gpu_id, int remote_gpu_id) {
    memset(out, 0, sizeof(*out));
    out->size = size;

    /* Import from FD */
    CHECK_CU(cuMemImportFromShareableHandle(&out->handle,
             (void *)(intptr_t)fd,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
             "cuMemImportFromShareableHandle");

    /* Reserve VA range */
    size_t gran = cumem_get_granularity(local_gpu_id);
    if (gran == 0) gran = 2 * 1024 * 1024;
    CHECK_CU(cuMemAddressReserve(&out->dptr, size, gran, 0, 0),
             "cuMemAddressReserve(import)");

    /* Map */
    CHECK_CU(cuMemMap(out->dptr, size, 0, out->handle, 0),
             "cuMemMap(import)");

    /* Set access for BOTH local and remote GPUs */
    CUmemAccessDesc access[2] = {};
    access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[0].location.id = local_gpu_id;
    access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[1].location.id = remote_gpu_id;
    access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CU(cuMemSetAccess(out->dptr, size, access, 2),
             "cuMemSetAccess(import)");

    fprintf(stderr, "[cumem] imported fd=%d -> dptr=0x%llx (%zu bytes)\n",
            fd, (unsigned long long)out->dptr, size);
    return 0;
}

void cumem_free(cumem_alloc_t *alloc) {
    if (alloc->dptr) {
        cuMemUnmap(alloc->dptr, alloc->size);
        cuMemAddressFree(alloc->dptr, alloc->size);
    }
    if (alloc->handle) cuMemRelease(alloc->handle);
    memset(alloc, 0, sizeof(*alloc));
}

void cumem_import_free(cumem_import_t *imp) {
    if (imp->dptr) {
        cuMemUnmap(imp->dptr, imp->size);
        cuMemAddressFree(imp->dptr, imp->size);
    }
    if (imp->handle) cuMemRelease(imp->handle);
    memset(imp, 0, sizeof(*imp));
}
