/*
 * xtrans_shim.h — XTrans LD_PRELOAD Shim Interface
 *
 * This shim intercepts libc calls that NCCL/RCCL use for container detection
 * ("gates"), returning values that make the CCL believe all containers are on
 * the same host with shared namespaces.
 *
 * Intercepted functions:
 *   gethostname()  — returns a common hostname across containers
 *   stat()/__xstat() — fakes /dev/shm st_dev to match across containers
 *
 * Configuration via environment variables:
 *   XTRANS_HOSTNAME    — hostname to return (default: "xtrans-node")
 *   XTRANS_SHMDEV      — st_dev value to return for /dev/shm (default: 0x1)
 *   XTRANS_VERBOSE      — set to "1" for debug logging to stderr
 *
 * Build:
 *   make              — builds libxtrans_shim.so
 *
 * Usage:
 *   LD_PRELOAD=./libxtrans_shim.so <nccl-application>
 */

#ifndef XTRANS_SHIM_H
#define XTRANS_SHIM_H

#define XTRANS_DEFAULT_HOSTNAME "xtrans-node"
#define XTRANS_DEFAULT_SHMDEV   0x1
#define XTRANS_SHM_PATH         "/dev/shm"

#endif /* XTRANS_SHIM_H */
