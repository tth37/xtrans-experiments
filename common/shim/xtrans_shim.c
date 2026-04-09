/*
 * xtrans_shim.c — XTrans LD_PRELOAD Shim
 *
 * Intercepts gethostname() and stat()/__xstat() to satisfy NCCL/RCCL's
 * container isolation gates. See xtrans_shim.h for documentation.
 *
 * Phase 2: Also intercepts bind() and sendmsg() to redirect NCCL's abstract
 * Unix domain sockets to filesystem paths in a shared directory, enabling
 * cuMem FD passing across containers with separate network namespaces.
 *
 * A1 strace findings showed NCCL uses:
 *   - bind(@"tmp/nccl-socket-<rank>-<hash>") for each proxy socket
 *   - sendmsg to @"tmp/nccl-socket-<rank>-<hash>" with SCM_RIGHTS for FD passing
 * These are abstract UDS (network-namespace-scoped). The shim redirects them
 * to filesystem paths in XTRANS_UDS_DIR so they work across network namespaces.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "xtrans_shim.h"

/* ---------- internal state (initialized once) ---------- */

static int   g_initialized = 0;
static int   g_verbose = 0;
static char  g_hostname[256] = XTRANS_DEFAULT_HOSTNAME;
static dev_t g_shmdev = XTRANS_DEFAULT_SHMDEV;
static char  g_uds_dir[256] = "";  /* empty = UDS redirect disabled */

/* original libc function pointers */
static int (*real_gethostname)(char *, size_t) = NULL;
static int (*real_stat)(const char *, struct stat *) = NULL;
static int (*real___xstat)(int, const char *, struct stat *) = NULL;
static int (*real_bind)(int, const struct sockaddr *, socklen_t) = NULL;
static ssize_t (*real_sendmsg)(int, const struct msghdr *, int) = NULL;

#define LOG(fmt, ...) \
    do { if (g_verbose) fprintf(stderr, "[xtrans-shim] " fmt "\n", ##__VA_ARGS__); } while(0)

static void shim_init(void) {
    if (g_initialized) return;
    g_initialized = 1;

    /* resolve real libc functions */
    real_gethostname = dlsym(RTLD_NEXT, "gethostname");
    real_stat        = dlsym(RTLD_NEXT, "stat");
    real___xstat     = dlsym(RTLD_NEXT, "__xstat");
    real_bind        = dlsym(RTLD_NEXT, "bind");
    real_sendmsg     = dlsym(RTLD_NEXT, "sendmsg");

    /* read config from environment */
    const char *env;

    env = getenv("XTRANS_VERBOSE");
    if (env && env[0] == '1') g_verbose = 1;

    env = getenv("XTRANS_HOSTNAME");
    if (env && env[0] != '\0') {
        strncpy(g_hostname, env, sizeof(g_hostname) - 1);
        g_hostname[sizeof(g_hostname) - 1] = '\0';
    }

    env = getenv("XTRANS_SHMDEV");
    if (env && env[0] != '\0') {
        g_shmdev = (dev_t)strtoul(env, NULL, 0);
    }

    env = getenv("XTRANS_UDS_DIR");
    if (env && env[0] != '\0') {
        strncpy(g_uds_dir, env, sizeof(g_uds_dir) - 1);
        g_uds_dir[sizeof(g_uds_dir) - 1] = '\0';
    }

    LOG("initialized: hostname=%s shmdev=0x%lx uds_dir=%s",
        g_hostname, (unsigned long)g_shmdev,
        g_uds_dir[0] ? g_uds_dir : "(disabled)");
}

/* ---------- UDS redirect helpers ---------- */

/*
 * Check if a sockaddr_un uses an abstract NCCL socket path.
 * Abstract sockets have sun_path[0] == '\0', and NCCL uses
 * "tmp/nccl-socket-<rank>-<hash>" starting at sun_path[1].
 * Returns pointer to the abstract name (after null byte) or NULL.
 */
static const char *is_nccl_abstract_socket(const struct sockaddr *addr, socklen_t addrlen) {
    if (!addr || addrlen < sizeof(sa_family_t) + 2)
        return NULL;
    if (addr->sa_family != AF_UNIX)
        return NULL;

    const struct sockaddr_un *un = (const struct sockaddr_un *)addr;
    /* abstract socket: sun_path[0] == '\0' */
    if (un->sun_path[0] != '\0')
        return NULL;

    const char *name = &un->sun_path[1];
    if (strncmp(name, XTRANS_NCCL_SOCKET_PREFIX,
                strlen(XTRANS_NCCL_SOCKET_PREFIX)) == 0)
        return name;

    return NULL;
}

/*
 * Build a filesystem sockaddr_un from an abstract NCCL socket name.
 * Maps @"tmp/nccl-socket-0-<hash>" → "<uds_dir>/nccl-socket-0-<hash>".
 * Returns the new addrlen, or 0 on failure.
 */
static socklen_t build_fs_sockaddr(struct sockaddr_un *out,
                                   const char *abstract_name) {
    /* Skip "tmp/" prefix from the abstract name */
    const char *short_name = abstract_name;
    if (strncmp(short_name, "tmp/", 4) == 0)
        short_name += 4;

    int n = snprintf(out->sun_path, sizeof(out->sun_path),
                     "%s/%s", g_uds_dir, short_name);
    if (n < 0 || (size_t)n >= sizeof(out->sun_path))
        return 0;

    out->sun_family = AF_UNIX;
    return offsetof(struct sockaddr_un, sun_path) + n + 1;
}

/* ---------- intercepted functions ---------- */

/*
 * gethostname() — NCCL calls this in getHostHash() (src/misc/utils.cc:73).
 * Return a common hostname so hostHash matches across containers.
 */
int gethostname(char *name, size_t len) {
    shim_init();

    size_t hlen = strlen(g_hostname);
    if (hlen >= len) {
        errno = ENAMETOOLONG;
        return -1;
    }

    memcpy(name, g_hostname, hlen + 1);
    LOG("gethostname() -> \"%s\"", g_hostname);
    return 0;
}

/*
 * stat() / __xstat() — NCCL calls stat("/dev/shm") in fillInfo() (src/init.cc:678).
 * For /dev/shm, return a faked st_dev so shmDev matches across containers.
 * For all other paths, pass through to real stat.
 */
int stat(const char *pathname, struct stat *statbuf) {
    shim_init();

    int ret;
    if (real_stat) {
        ret = real_stat(pathname, statbuf);
    } else {
        /* fallback: use __xstat with _STAT_VER */
        ret = real___xstat(1, pathname, statbuf);
    }

    if (ret == 0 && strcmp(pathname, XTRANS_SHM_PATH) == 0) {
        dev_t orig = statbuf->st_dev;
        statbuf->st_dev = g_shmdev;
        LOG("stat(\"%s\"): st_dev 0x%lx -> 0x%lx",
            pathname, (unsigned long)orig, (unsigned long)g_shmdev);
    }

    return ret;
}

/*
 * __xstat() — Some glibc versions route stat() through __xstat().
 * Same logic as stat() above.
 */
int __xstat(int ver, const char *pathname, struct stat *statbuf) {
    shim_init();

    if (!real___xstat) {
        errno = ENOSYS;
        return -1;
    }

    int ret = real___xstat(ver, pathname, statbuf);

    if (ret == 0 && strcmp(pathname, XTRANS_SHM_PATH) == 0) {
        dev_t orig = statbuf->st_dev;
        statbuf->st_dev = g_shmdev;
        LOG("__xstat(\"%s\"): st_dev 0x%lx -> 0x%lx",
            pathname, (unsigned long)orig, (unsigned long)g_shmdev);
    }

    return ret;
}

/*
 * bind() — NCCL proxy threads bind abstract UDS for cuMem IPC.
 * Pattern: @"tmp/nccl-socket-<rank>-<hash>"
 * Redirect to filesystem path in XTRANS_UDS_DIR if configured.
 */
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    shim_init();

    if (g_uds_dir[0]) {
        const char *name = is_nccl_abstract_socket(addr, addrlen);
        if (name) {
            struct sockaddr_un fs_addr;
            socklen_t fs_len = build_fs_sockaddr(&fs_addr, name);
            if (fs_len > 0) {
                LOG("bind(%d): redirect @\"%s\" -> \"%s\"",
                    sockfd, name, fs_addr.sun_path);
                /* Remove stale socket file if it exists */
                unlink(fs_addr.sun_path);
                return real_bind(sockfd, (struct sockaddr *)&fs_addr, fs_len);
            }
        }
    }

    return real_bind(sockfd, addr, addrlen);
}

/*
 * sendmsg() — NCCL sends cuMem FDs via sendmsg with SCM_RIGHTS.
 * The msg_name contains the destination abstract UDS address.
 * Redirect to filesystem path in XTRANS_UDS_DIR if configured.
 */
ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags) {
    shim_init();

    if (g_uds_dir[0] && msg && msg->msg_name) {
        const char *name = is_nccl_abstract_socket(
            (struct sockaddr *)msg->msg_name, msg->msg_namelen);
        if (name) {
            struct sockaddr_un fs_addr;
            socklen_t fs_len = build_fs_sockaddr(&fs_addr, name);
            if (fs_len > 0) {
                LOG("sendmsg(%d): redirect @\"%s\" -> \"%s\"",
                    sockfd, name, fs_addr.sun_path);
                /* Build modified msghdr with filesystem address */
                struct msghdr new_msg = *msg;
                new_msg.msg_name = &fs_addr;
                new_msg.msg_namelen = fs_len;
                return real_sendmsg(sockfd, &new_msg, flags);
            }
        }
    }

    return real_sendmsg(sockfd, msg, flags);
}
