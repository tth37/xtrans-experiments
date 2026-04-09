/*
 * uds.c — Unix domain socket helpers for IPC FD passing.
 */

#include "uds.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>

#define LOG(fmt, ...) fprintf(stderr, "[uds] " fmt "\n", ##__VA_ARGS__)

/* --- helpers --- */

static int send_all(int sock, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = send(sock, p, len, 0);
        if (n <= 0) return -1;
        p += n;
        len -= n;
    }
    return 0;
}

static int recv_all(int sock, void *buf, size_t len) {
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t n = recv(sock, p, len, 0);
        if (n <= 0) return -1;
        p += n;
        len -= n;
    }
    return 0;
}

static void msleep(int ms) {
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

/* --- server/client --- */

int uds_listen(const char *socket_path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    unlink(socket_path);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(fd); return -1;
    }
    if (listen(fd, 1) < 0) {
        perror("listen"); close(fd); return -1;
    }
    LOG("listening on %s", socket_path);
    return fd;
}

int uds_accept(int server_fd) {
    int fd = accept(server_fd, NULL, NULL);
    if (fd < 0) { perror("accept"); return -1; }
    LOG("accepted connection");
    return fd;
}

int uds_connect(const char *socket_path, int max_retries, int retry_ms) {
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    for (int i = 0; i < max_retries; i++) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) { perror("socket"); return -1; }

        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            LOG("connected to %s", socket_path);
            return fd;
        }
        close(fd);
        if (i < max_retries - 1) {
            LOG("connect retry %d/%d...", i + 1, max_retries);
            msleep(retry_ms);
        }
    }
    LOG("connect failed after %d retries", max_retries);
    return -1;
}

/* --- SCM_RIGHTS FD passing --- */

int uds_send_fd(int sock, int fd_to_send) {
    char dummy = 'F';
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };

    char buf[CMSG_SPACE(sizeof(int))];
    memset(buf, 0, sizeof(buf));

    struct msghdr msg = {0};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));

    if (sendmsg(sock, &msg, 0) < 0) {
        perror("sendmsg(SCM_RIGHTS)");
        return -1;
    }
    return 0;
}

int uds_recv_fd(int sock) {
    char dummy;
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };

    char buf[CMSG_SPACE(sizeof(int))];
    memset(buf, 0, sizeof(buf));

    struct msghdr msg = {0};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    if (recvmsg(sock, &msg, 0) < 0) {
        perror("recvmsg(SCM_RIGHTS)");
        return -1;
    }

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET ||
        cmsg->cmsg_type != SCM_RIGHTS) {
        LOG("no SCM_RIGHTS in message");
        return -1;
    }

    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

/* --- structured messages --- */

int uds_send_meta(int sock, const ipc_meta_t *meta) {
    uint32_t type = UDS_MSG_META;
    if (send_all(sock, &type, sizeof(type)) < 0) return -1;
    return send_all(sock, meta, sizeof(*meta));
}

int uds_recv_meta(int sock, ipc_meta_t *meta) {
    uint32_t type;
    if (recv_all(sock, &type, sizeof(type)) < 0) return -1;
    if (type != UDS_MSG_META) { LOG("expected META, got %u", type); return -1; }
    return recv_all(sock, meta, sizeof(*meta));
}

int uds_send_ack(int sock) {
    uint32_t type = UDS_MSG_ACK;
    return send_all(sock, &type, sizeof(type));
}

int uds_recv_ack(int sock) {
    uint32_t type;
    if (recv_all(sock, &type, sizeof(type)) < 0) return -1;
    if (type != UDS_MSG_ACK) { LOG("expected ACK, got %u", type); return -1; }
    return 0;
}

int uds_send_blob(int sock, const void *data, uint32_t len) {
    uint32_t type = UDS_MSG_BLOB;
    if (send_all(sock, &type, sizeof(type)) < 0) return -1;
    if (send_all(sock, &len, sizeof(len)) < 0) return -1;
    return send_all(sock, data, len);
}

int uds_recv_blob(int sock, void *data, uint32_t max_len) {
    uint32_t type, len;
    if (recv_all(sock, &type, sizeof(type)) < 0) return -1;
    if (type != UDS_MSG_BLOB) { LOG("expected BLOB, got %u", type); return -1; }
    if (recv_all(sock, &len, sizeof(len)) < 0) return -1;
    if (len > max_len) { LOG("blob too large: %u > %u", len, max_len); return -1; }
    return recv_all(sock, data, len);
}

int uds_send_result(int sock, const bench_result_t *result) {
    uint32_t type = UDS_MSG_RESULT;
    if (send_all(sock, &type, sizeof(type)) < 0) return -1;
    return send_all(sock, result, sizeof(*result));
}

int uds_recv_result(int sock, bench_result_t *result) {
    uint32_t type;
    if (recv_all(sock, &type, sizeof(type)) < 0) return -1;
    if (type != UDS_MSG_RESULT) { LOG("expected RESULT, got %u", type); return -1; }
    return recv_all(sock, result, sizeof(*result));
}

void uds_cleanup(int server_fd, const char *socket_path) {
    close(server_fd);
    unlink(socket_path);
}
