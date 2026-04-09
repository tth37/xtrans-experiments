/*
 * bench.h — P2P bandwidth measurement utilities.
 */

#ifndef BENCH_H
#define BENCH_H

#include <cuda.h>
#include "uds.h"  /* for bench_result_t */

/*
 * Measure GPU-to-GPU P2P read bandwidth using cuMemcpy.
 * src_dptr: remote GPU memory (mapped via cuMem import)
 * copy_size: bytes to copy per iteration
 * iterations: number of timed iterations
 * warmup: warmup iterations (not timed)
 * result: output bandwidth/latency stats
 */
int bench_p2p_bandwidth(CUdeviceptr src_dptr, size_t copy_size,
                        int iterations, int warmup,
                        bench_result_t *result);

/*
 * Compute trimmed mean and std (drop top/bottom 5%).
 */
void bench_trimmed_stats(const float *times_ms, int n,
                         double *avg_ms, double *std_ms);

#endif /* BENCH_H */
