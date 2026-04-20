# syntax=docker/dockerfile:1
# Dockerfile for Exp A1: Tenplex elastic GPU scaling
#
# Build:
#   docker build -t xtrans-exp-a1:latest .
#
# Tenplex requires: Go 1.21+, Python 3.8+, CUDA, NCCL, Megatron-LM
# Base image provides CUDA, NCCL, PyTorch, and build tools.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
FROM ${BASE_IMAGE}

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        strace \
        golang-go \
        git \
        wget \
        htop \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Ensure Go >= 1.21 (NGC container may have older version)
# If the system Go is too old, install a newer one:
RUN go version | grep -qE 'go1\.(2[1-9]|[3-9])' || \
    (wget -q https://go.dev/dl/go1.22.4.linux-amd64.tar.gz && \
     rm -rf /usr/local/go && tar -C /usr/local -xzf go1.22.4.linux-amd64.tar.gz && \
     rm go1.22.4.linux-amd64.tar.gz)
ENV PATH="/usr/local/go/bin:${PATH}"

# --- Install nccl-tests (for communication validation) ---
RUN cd /opt && \
    git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
ENV PATH="/opt/nccl-tests/build:${PATH}"

# --- Clone Tenplex ---
# Pin to a known commit for reproducibility. Update as needed.
RUN cd /opt && \
    git clone https://github.com/kungfu-team/tenplex.git
    # To pin: && cd tenplex && git checkout <commit-hash>

# --- Clone Megatron-LM (Tenplex integrates with it) ---
# Tenplex was tested with specific Megatron-LM versions.
# Check tenplex/README.md for compatible version.
RUN cd /opt && \
    git clone https://github.com/NVIDIA/Megatron-LM.git
    # To pin: && cd Megatron-LM && git checkout <tag>

# --- Build Tenplex ---
# NOTE: The build steps below may need adjustment based on Tenplex's
# current build system. Check /opt/tenplex/README.md after cloning.
WORKDIR /opt/tenplex
# RUN make  # or: go build ./... && pip install -e .

# --- Workspace ---
WORKDIR /workspace
COPY scripts/ /workspace/scripts/

# Default: keep container alive for interactive use
CMD ["sleep", "infinity"]
