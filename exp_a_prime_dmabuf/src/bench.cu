/*
 * bench.cu — P2P bandwidth measurement using CUDA driver API.
 */

#include "bench.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void bench_trimmed_stats(const float *times_ms, int n,
                         double *avg_ms, double *std_ms) {
    std::vector<float> sorted(times_ms, times_ms + n);
    std::sort(sorted.begin(), sorted.end());

    int trim = (int)(n * 0.05);
    int start = trim;
    int end = n - trim;
    if (end <= start) { start = 0; end = n; }

    double sum = 0;
    for (int i = start; i < end; i++) sum += sorted[i];
    double mean = sum / (end - start);

    double var = 0;
    for (int i = start; i < end; i++) {
        double d = sorted[i] - mean;
        var += d * d;
    }
    *avg_ms = mean;
    *std_ms = (end - start > 1) ? sqrt(var / (end - start - 1)) : 0;
}

int bench_p2p_bandwidth(CUdeviceptr src_dptr, size_t copy_size,
                        int iterations, int warmup,
                        bench_result_t *result) {
    CUresult err;

    /* Allocate local destination buffer */
    CUdeviceptr dst;
    err = cuMemAlloc(&dst, copy_size);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[bench] cuMemAlloc failed: %d\n", err);
        return -1;
    }

    /* Create CUDA events for timing */
    CUevent start_ev, end_ev;
    cuEventCreate(&start_ev, CU_EVENT_DEFAULT);
    cuEventCreate(&end_ev, CU_EVENT_DEFAULT);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        cuMemcpyDtoD(dst, src_dptr, copy_size);
    }
    cuCtxSynchronize();

    /* Timed iterations */
    std::vector<float> times(iterations);
    for (int i = 0; i < iterations; i++) {
        cuEventRecord(start_ev, 0);
        cuMemcpyDtoD(dst, src_dptr, copy_size);
        cuEventRecord(end_ev, 0);
        cuEventSynchronize(end_ev);
        cuEventElapsedTime(&times[i], start_ev, end_ev);
    }

    /* Compute stats */
    double avg_ms, std_ms;
    bench_trimmed_stats(times.data(), iterations, &avg_ms, &std_ms);

    double size_gb = (double)copy_size / (1024.0 * 1024.0 * 1024.0);
    result->bw_gbps = size_gb / (avg_ms / 1000.0);
    result->avg_latency_us = avg_ms * 1000.0;
    result->std_latency_us = std_ms * 1000.0;
    result->size_bytes = copy_size;

    /* Cleanup */
    cuEventDestroy(start_ev);
    cuEventDestroy(end_ev);
    cuMemFree(dst);

    return 0;
}
