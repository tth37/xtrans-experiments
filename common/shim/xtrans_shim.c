/*
 * xtrans_shim.c — XTrans LD_PRELOAD Shim
 *
 * Intercepts gethostname() and stat()/__xstat() to satisfy NCCL/RCCL's
 * container isolation gates. See xtrans_shim.h for documentation.
 *
 * This is the core shim that Exp A will test and refine. It starts minimal
 * and will be extended based on syscall tracing results (Exp A1).
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "xtrans_shim.h"

/* ---------- internal state (initialized once) ---------- */

static int   g_initialized = 0;
static int   g_verbose = 0;
static char  g_hostname[256] = XTRANS_DEFAULT_HOSTNAME;
static dev_t g_shmdev = XTRANS_DEFAULT_SHMDEV;

/* original libc function pointers */
static int (*real_gethostname)(char *, size_t) = NULL;
static int (*real_stat)(const char *, struct stat *) = NULL;
static int (*real___xstat)(int, const char *, struct stat *) = NULL;

#define LOG(fmt, ...) \
    do { if (g_verbose) fprintf(stderr, "[xtrans-shim] " fmt "\n", ##__VA_ARGS__); } while(0)

static void shim_init(void) {
    if (g_initialized) return;
    g_initialized = 1;

    /* resolve real libc functions */
    real_gethostname = dlsym(RTLD_NEXT, "gethostname");
    real_stat        = dlsym(RTLD_NEXT, "stat");
    real___xstat     = dlsym(RTLD_NEXT, "__xstat");

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

    LOG("initialized: hostname=%s shmdev=0x%lx", g_hostname, (unsigned long)g_shmdev);
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
