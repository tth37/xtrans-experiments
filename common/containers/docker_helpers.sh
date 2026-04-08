#!/usr/bin/env bash
# docker_helpers.sh — Container launch helpers for XTrans experiments
#
# Source this file in experiment scripts:
#   source "$(dirname "$0")/../common/containers/docker_helpers.sh"
#
# Provides functions for launching per-GPU containers with various
# isolation configurations.

set -euo pipefail

# --- Configuration ---
# Override these in your experiment script or environment
XTRANS_IMAGE="${XTRANS_IMAGE:-nvcr.io/nvidia/pytorch:24.07-py3}"
XTRANS_SHIM_PATH="${XTRANS_SHIM_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../shim" && pwd)/libxtrans_shim.so}"
XTRANS_PROJECT_ROOT="${XTRANS_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
XTRANS_NETWORK="${XTRANS_NETWORK:-xtrans-net}"

# --- Helper Functions ---

# Ensure the shared bridge network exists
ensure_network() {
    if ! docker network inspect "$XTRANS_NETWORK" &>/dev/null; then
        echo "[docker_helpers] Creating bridge network: $XTRANS_NETWORK"
        docker network create "$XTRANS_NETWORK"
    fi
}

# Launch a per-GPU container
#   $1 — GPU index (0, 1, 2, ...)
#   $2 — container name suffix (e.g., "baseline", "shim")
#   $3... — additional docker args
launch_gpu_container() {
    local gpu_idx="$1"
    local suffix="$2"
    shift 2
    local name="xtrans-gpu${gpu_idx}-${suffix}"

    echo "[docker_helpers] Launching $name (GPU $gpu_idx)"
    docker run -d \
        --name "$name" \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES="$gpu_idx" \
        --network "$XTRANS_NETWORK" \
        --shm-size=1g \
        -v "$XTRANS_PROJECT_ROOT:/workspace/xtrans:ro" \
        "$@" \
        "$XTRANS_IMAGE" \
        sleep infinity
}

# Launch a per-GPU container with the xtrans shim loaded
#   $1 — GPU index
#   $2 — container name suffix
#   $3... — additional docker args
launch_gpu_container_with_shim() {
    local gpu_idx="$1"
    local suffix="$2"
    shift 2

    launch_gpu_container "$gpu_idx" "$suffix" \
        -e LD_PRELOAD="/workspace/xtrans/common/shim/libxtrans_shim.so" \
        -e XTRANS_VERBOSE=1 \
        "$@"
}

# Launch per-GPU containers sharing /dev/shm and host network
# (the exp2-style workaround, for comparison baseline)
launch_gpu_container_shared_ns() {
    local gpu_idx="$1"
    local suffix="$2"
    shift 2
    local name="xtrans-gpu${gpu_idx}-${suffix}"

    echo "[docker_helpers] Launching $name (GPU $gpu_idx, shared NS)"
    docker run -d \
        --name "$name" \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES="$gpu_idx" \
        --network host \
        --ipc=host \
        -v "$XTRANS_PROJECT_ROOT:/workspace/xtrans:ro" \
        "$@" \
        "$XTRANS_IMAGE" \
        sleep infinity
}

# Stop and remove containers matching a pattern
cleanup_containers() {
    local pattern="${1:-xtrans-gpu}"
    local containers
    containers=$(docker ps -a --filter "name=$pattern" -q 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "[docker_helpers] Cleaning up containers matching '$pattern'"
        docker rm -f $containers 2>/dev/null || true
    fi
}

# Execute a command in a running container
container_exec() {
    local name="$1"
    shift
    docker exec "$name" "$@"
}

# Run nccl-tests allreduce inside a set of containers
# $1 — comma-separated container names (e.g., "xtrans-gpu0-test,xtrans-gpu1-test")
# $2 — total number of GPUs
# $3... — extra nccl-tests args
run_nccl_allreduce() {
    local containers="$1"
    local ngpus="$2"
    shift 2

    local first_container
    first_container=$(echo "$containers" | cut -d, -f1)

    echo "[docker_helpers] Running allreduce on $ngpus GPUs across containers: $containers"
    echo "[docker_helpers] Note: multi-container nccl-tests requires torchrun or mpirun"
    echo "[docker_helpers] This is a placeholder — implement based on your container setup"
}

echo "[docker_helpers] Loaded. Project root: $XTRANS_PROJECT_ROOT"
