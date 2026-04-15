/*
 * uds.h — Unix domain socket helpers for IPC FD passing.
 *
 * Provides UDS server/client, SCM_RIGHTS FD passing, and a simple
 * metadata protocol for the DMA-BUF benchmark.
 */

#ifndef UDS_H
#define UDS_H

#include <stdint.h>

/* Message types for the wire protocol */
#define UDS_MSG_META    1
#define UDS_MSG_BLOB    2
#define UDS_MSG_ACK     3
#define UDS_MSG_RESULT  4

/* Metadata exchanged between exporter and importer */
typedef struct {
    uint64_t size;
    int32_t  gpu_id;
    uint64_t va_offset;
} ipc_meta_t;

/* Benchmark result for one size */
typedef struct {
    double bw_gbps;
    double avg_latency_us;
    double std_latency_us;
    uint64_t size_bytes;
} bench_result_t;

/* Server: create + listen on a filesystem UDS */
int uds_listen(const char *socket_path);

/* Server: accept one connection */
int uds_accept(int server_fd);

/* Client: connect with retries (for startup race) */
int uds_connect(const char *socket_path, int max_retries, int retry_ms);

/* Send/recv a file descriptor via SCM_RIGHTS */
int uds_send_fd(int sock, int fd_to_send);
int uds_recv_fd(int sock);

/* Send/recv metadata struct */
int uds_send_meta(int sock, const ipc_meta_t *meta);
int uds_recv_meta(int sock, ipc_meta_t *meta);

/* Send/recv acknowledgment */
int uds_send_ack(int sock);
int uds_recv_ack(int sock);

/* Send/recv opaque blob (for legacy cudaIPC handle) */
int uds_send_blob(int sock, const void *data, uint32_t len);
int uds_recv_blob(int sock, void *data, uint32_t max_len);

/* Send/recv benchmark result */
int uds_send_result(int sock, const bench_result_t *result);
int uds_recv_result(int sock, bench_result_t *result);

/* Cleanup: close + unlink */
void uds_cleanup(int server_fd, const char *socket_path);

#endif /* UDS_H */
