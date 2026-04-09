/*
 * dmabuf_bench.cu — DMA-BUF / cuMem IPC P2P benchmark.
 *
 * Single binary with --role exporter|importer and --ipc-path cumem_fd|cudaipc.
 * Measures cross-container GPU P2P bandwidth via explicit cuMem VMM or
 * legacy cudaIPC FD/handle passing over Unix domain sockets.
 *
 * Usage:
 *   dmabuf_bench --role exporter --ipc-path cumem_fd --gpu 0 \
 *                --socket /shared_uds/dmabuf.sock --output /results/cumem_fd.json
 *
 * Or via environment variables: ROLE, IPC_PATH, GPU_ID, UDS_SOCKET, OUTPUT_FILE
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

extern "C" {
#include "uds.h"
}
#include "cumem_ipc.h"
#include "cudaipc.h"
#include "bench.h"

/* Default configuration */
#define DEFAULT_SOCKET  "/shared_uds/dmabuf.sock"
#define DEFAULT_OUTPUT  "/results/dmabuf_bw.json"
#define DEFAULT_ITERS   100
#define DEFAULT_WARMUP  10
#define FILL_PATTERN    0xA5

/* --- configuration --- */

typedef struct {
    const char *role;       /* "exporter" or "importer" */
    const char *ipc_path;   /* "cumem_fd" or "cudaipc" */
    int gpu_id;
    const char *socket_path;
    const char *output_file;
    int iterations;
    int warmup;
    int verbose;
    /* parsed sizes in bytes */
    size_t sizes[32];
    int num_sizes;
} config_t;

static const char *env_or(const char *env, const char *def) {
    const char *v = getenv(env);
    return (v && v[0]) ? v : def;
}

static int env_int(const char *env, int def) {
    const char *v = getenv(env);
    return v ? atoi(v) : def;
}

static size_t parse_size(const char *s) {
    char *end;
    size_t val = strtoul(s, &end, 10);
    if (*end == 'K' || *end == 'k') val *= 1024;
    else if (*end == 'M' || *end == 'm') val *= 1024 * 1024;
    else if (*end == 'G' || *end == 'g') val *= 1024 * 1024 * 1024;
    return val;
}

static void parse_sizes(config_t *cfg, const char *sizes_str) {
    cfg->num_sizes = 0;
    char buf[256];
    strncpy(buf, sizes_str, sizeof(buf) - 1);
    char *tok = strtok(buf, ",");
    while (tok && cfg->num_sizes < 32) {
        cfg->sizes[cfg->num_sizes++] = parse_size(tok);
        tok = strtok(NULL, ",");
    }
}

static void parse_config(config_t *cfg, int argc, char **argv) {
    cfg->role = env_or("ROLE", "exporter");
    cfg->ipc_path = env_or("IPC_PATH", "cumem_fd");
    cfg->gpu_id = env_int("GPU_ID", 0);
    cfg->socket_path = env_or("UDS_SOCKET", DEFAULT_SOCKET);
    cfg->output_file = env_or("OUTPUT_FILE", DEFAULT_OUTPUT);
    cfg->iterations = env_int("ITERATIONS", DEFAULT_ITERS);
    cfg->warmup = env_int("WARMUP", DEFAULT_WARMUP);
    cfg->verbose = env_int("VERBOSE", 1);

    parse_sizes(cfg, env_or("SIZES", "1M,4M,16M,64M,256M,1G"));

    /* CLI overrides */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--role") && i + 1 < argc) cfg->role = argv[++i];
        else if (!strcmp(argv[i], "--ipc-path") && i + 1 < argc) cfg->ipc_path = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) cfg->gpu_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--socket") && i + 1 < argc) cfg->socket_path = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) cfg->output_file = argv[++i];
        else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) cfg->iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) cfg->warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc) parse_sizes(cfg, argv[++i]);
    }
}

/* --- fill kernel --- */
__global__ void fill_kernel(char *ptr, size_t n, char val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}

__global__ void verify_kernel(const char *ptr, size_t n, char expected, int *errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && ptr[idx] != expected) atomicAdd(errors, 1);
}

/* --- JSON output --- */
static void write_json(const config_t *cfg, bench_result_t *results, int n) {
    FILE *f = fopen(cfg->output_file, "w");
    if (!f) { perror("fopen(output)"); return; }

    /* Get system info */
    int driver_ver = 0, runtime_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    cudaRuntimeGetVersion(&runtime_ver);
    char gpu_name[256] = "unknown";
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        strncpy(gpu_name, prop.name, sizeof(gpu_name) - 1);

    time_t now = time(NULL);
    char ts[64];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", localtime(&now));

    fprintf(f, "{\n");
    fprintf(f, "  \"timestamp\": \"%s\",\n", ts);
    fprintf(f, "  \"experiment\": \"exp_a_prime\",\n");
    fprintf(f, "  \"ipc_path\": \"%s\",\n", cfg->ipc_path);
    fprintf(f, "  \"exporter_gpu\": %d,\n", cfg->gpu_id);
    fprintf(f, "  \"gpu_name\": \"%s\",\n", gpu_name);
    fprintf(f, "  \"cuda_driver\": %d,\n", driver_ver);
    fprintf(f, "  \"cuda_runtime\": %d,\n", runtime_ver);
    fprintf(f, "  \"iterations\": %d,\n", cfg->iterations);
    fprintf(f, "  \"warmup\": %d,\n", cfg->warmup);
    fprintf(f, "  \"results\": {\n");
    for (int i = 0; i < n; i++) {
        double size_mb = (double)results[i].size_bytes / (1024.0 * 1024.0);
        fprintf(f, "    \"%.2fMB\": {\n", size_mb);
        fprintf(f, "      \"size_bytes\": %lu,\n", (unsigned long)results[i].size_bytes);
        fprintf(f, "      \"bandwidth_gbps\": %.3f,\n", results[i].bw_gbps);
        fprintf(f, "      \"avg_latency_us\": %.3f,\n", results[i].avg_latency_us);
        fprintf(f, "      \"std_latency_us\": %.3f\n", results[i].std_latency_us);
        fprintf(f, "    }%s\n", (i < n - 1) ? "," : "");
    }
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);
    fprintf(stderr, "[main] results written to %s\n", cfg->output_file);
}

/* --- exporter --- */

static int run_exporter_cumem(const config_t *cfg) {
    int server_fd = uds_listen(cfg->socket_path);
    if (server_fd < 0) return -1;

    int client_fd = uds_accept(server_fd);
    if (client_fd < 0) { uds_cleanup(server_fd, cfg->socket_path); return -1; }

    bench_result_t results[32];
    int result_count = 0;

    for (int si = 0; si < cfg->num_sizes; si++) {
        size_t req_size = cfg->sizes[si];
        fprintf(stderr, "\n[exporter] === Size: %zu bytes ===\n", req_size);

        /* Allocate */
        cumem_alloc_t alloc;
        if (cumem_alloc(&alloc, req_size, cfg->gpu_id) < 0) return -1;

        /* Fill with pattern */
        int blocks = (alloc.size + 255) / 256;
        fill_kernel<<<blocks, 256>>>((char *)alloc.dptr, alloc.size, FILL_PATTERN);
        cuCtxSynchronize();

        /* Export FD */
        int fd;
        if (cumem_export_fd(&alloc, &fd) < 0) return -1;

        /* Send metadata + FD to importer */
        ipc_meta_t meta = { alloc.size, cfg->gpu_id, 0 };
        if (uds_send_meta(client_fd, &meta) < 0) return -1;
        if (uds_send_fd(client_fd, fd) < 0) return -1;
        close(fd);

        /* Wait for importer to finish benchmark */
        bench_result_t result;
        if (uds_recv_result(client_fd, &result) < 0) return -1;
        result.size_bytes = req_size;
        results[result_count++] = result;

        fprintf(stderr, "[exporter] %zu bytes: %.2f GB/s (%.1f us)\n",
                req_size, result.bw_gbps, result.avg_latency_us);

        /* Wait for importer to unmap, then free */
        uds_recv_ack(client_fd);
        cumem_free(&alloc);
    }

    /* Write JSON results */
    write_json(cfg, results, result_count);

    close(client_fd);
    uds_cleanup(server_fd, cfg->socket_path);
    return 0;
}

static int run_exporter_cudaipc(const config_t *cfg) {
    int server_fd = uds_listen(cfg->socket_path);
    if (server_fd < 0) return -1;

    int client_fd = uds_accept(server_fd);
    if (client_fd < 0) { uds_cleanup(server_fd, cfg->socket_path); return -1; }

    cudaSetDevice(cfg->gpu_id);
    bench_result_t results[32];
    int result_count = 0;

    for (int si = 0; si < cfg->num_sizes; si++) {
        size_t req_size = cfg->sizes[si];
        fprintf(stderr, "\n[exporter] === Size: %zu bytes (cudaipc) ===\n", req_size);

        /* Allocate + export */
        void *dptr;
        cudaIpcMemHandle_t handle;
        if (cudaipc_alloc_and_export(&dptr, &handle, req_size, cfg->gpu_id) < 0)
            return -1;

        /* Fill with pattern */
        int blocks = (req_size + 255) / 256;
        fill_kernel<<<blocks, 256>>>((char *)dptr, req_size, FILL_PATTERN);
        cudaDeviceSynchronize();

        /* Send metadata + handle blob */
        ipc_meta_t meta = { req_size, cfg->gpu_id, 0 };
        if (uds_send_meta(client_fd, &meta) < 0) return -1;
        if (uds_send_blob(client_fd, &handle, sizeof(handle)) < 0) return -1;

        /* Wait for benchmark result */
        bench_result_t result;
        if (uds_recv_result(client_fd, &result) < 0) return -1;
        result.size_bytes = req_size;
        results[result_count++] = result;

        fprintf(stderr, "[exporter] %zu bytes: %.2f GB/s (%.1f us)\n",
                req_size, result.bw_gbps, result.avg_latency_us);

        /* Wait for importer to close, then free */
        uds_recv_ack(client_fd);
        cudaipc_free(dptr);
    }

    write_json(cfg, results, result_count);
    close(client_fd);
    uds_cleanup(server_fd, cfg->socket_path);
    return 0;
}

/* --- importer --- */

static int run_importer_cumem(const config_t *cfg) {
    int sock = uds_connect(cfg->socket_path, 30, 500);
    if (sock < 0) return -1;

    /* Get peer GPU ID from env (exporter's GPU) */
    int remote_gpu_id = env_int("REMOTE_GPU_ID", 0);

    for (int si = 0; si < cfg->num_sizes; si++) {
        /* Recv metadata + FD */
        ipc_meta_t meta;
        if (uds_recv_meta(sock, &meta) < 0) return -1;
        int fd = uds_recv_fd(sock);
        if (fd < 0) return -1;

        remote_gpu_id = meta.gpu_id;
        fprintf(stderr, "\n[importer] === Size: %lu bytes from GPU %d ===\n",
                (unsigned long)meta.size, meta.gpu_id);

        /* Import and map */
        cumem_import_t imp;
        if (cumem_import_and_map(&imp, fd, meta.size,
                                 cfg->gpu_id, remote_gpu_id) < 0)
            return -1;
        close(fd);

        /* Verify pattern */
        int *d_errors;
        cuMemAlloc((CUdeviceptr *)&d_errors, sizeof(int));
        cuMemsetD32((CUdeviceptr)d_errors, 0, 1);
        int blocks = (meta.size + 255) / 256;
        verify_kernel<<<blocks, 256>>>((const char *)imp.dptr, meta.size,
                                       FILL_PATTERN, d_errors);
        cuCtxSynchronize();
        int h_errors = 0;
        cuMemcpyDtoH(&h_errors, (CUdeviceptr)d_errors, sizeof(int));
        cuMemFree((CUdeviceptr)d_errors);
        if (h_errors > 0)
            fprintf(stderr, "[importer] WARNING: %d verification errors!\n", h_errors);
        else
            fprintf(stderr, "[importer] pattern verified OK\n");

        /* Benchmark P2P read */
        size_t copy_size = cfg->sizes[si]; /* use requested size, not rounded */
        bench_result_t result;
        if (bench_p2p_bandwidth(imp.dptr, copy_size,
                                cfg->iterations, cfg->warmup, &result) < 0)
            return -1;

        fprintf(stderr, "[importer] %.2f GB/s (%.1f +/- %.1f us)\n",
                result.bw_gbps, result.avg_latency_us, result.std_latency_us);

        /* Send result back to exporter */
        if (uds_send_result(sock, &result) < 0) return -1;

        /* Unmap and notify */
        cumem_import_free(&imp);
        uds_send_ack(sock);
    }

    close(sock);
    return 0;
}

static int run_importer_cudaipc(const config_t *cfg) {
    int sock = uds_connect(cfg->socket_path, 30, 500);
    if (sock < 0) return -1;

    cudaSetDevice(cfg->gpu_id);

    for (int si = 0; si < cfg->num_sizes; si++) {
        /* Recv metadata + handle blob */
        ipc_meta_t meta;
        if (uds_recv_meta(sock, &meta) < 0) return -1;
        cudaIpcMemHandle_t handle;
        if (uds_recv_blob(sock, &handle, sizeof(handle)) < 0) return -1;

        fprintf(stderr, "\n[importer] === Size: %lu bytes (cudaipc) ===\n",
                (unsigned long)meta.size);

        /* Import */
        void *dptr;
        if (cudaipc_import(&dptr, &handle, meta.gpu_id) < 0) return -1;

        /* Benchmark P2P read */
        size_t copy_size = cfg->sizes[si];
        bench_result_t result;
        if (bench_p2p_bandwidth((CUdeviceptr)dptr, copy_size,
                                cfg->iterations, cfg->warmup, &result) < 0)
            return -1;

        fprintf(stderr, "[importer] %.2f GB/s (%.1f +/- %.1f us)\n",
                result.bw_gbps, result.avg_latency_us, result.std_latency_us);

        /* Send result + cleanup */
        if (uds_send_result(sock, &result) < 0) return -1;
        cudaipc_import_close(dptr);
        uds_send_ack(sock);
    }

    close(sock);
    return 0;
}

/* --- main --- */

int main(int argc, char **argv) {
    config_t cfg;
    parse_config(&cfg, argc, argv);

    fprintf(stderr, "[main] role=%s ipc_path=%s gpu=%d socket=%s\n",
            cfg.role, cfg.ipc_path, cfg.gpu_id, cfg.socket_path);

    /* Initialize CUDA driver API */
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[main] cuInit failed: %d\n", err);
        return 1;
    }

    int ret;
    int use_driver_api = (strcmp(cfg.ipc_path, "cudaipc") != 0);

    CUdevice dev;
    CUcontext ctx = NULL;
    if (use_driver_api) {
        /* cuMem VMM path uses driver API context */
        cuDeviceGet(&dev, cfg.gpu_id);
        cuCtxCreate(&ctx, 0, dev);
    } else {
        /* Legacy cudaIPC uses runtime API only */
        cudaSetDevice(cfg.gpu_id);
    }

    if (strcmp(cfg.role, "exporter") == 0) {
        if (strcmp(cfg.ipc_path, "cudaipc") == 0)
            ret = run_exporter_cudaipc(&cfg);
        else
            ret = run_exporter_cumem(&cfg);
    } else {
        if (strcmp(cfg.ipc_path, "cudaipc") == 0)
            ret = run_importer_cudaipc(&cfg);
        else
            ret = run_importer_cumem(&cfg);
    }

    if (ctx) cuCtxDestroy(ctx);
    return ret < 0 ? 1 : 0;
}
