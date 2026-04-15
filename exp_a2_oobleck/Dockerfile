# syntax=docker/dockerfile:1
# Dockerfile for Exp A2: Oobleck fault-tolerant training
#
# Build:
#   docker build -t xtrans-exp-a2:latest .
#
# Oobleck requires: Python 3.8+, Rust (for some components), CUDA, NCCL,
# PyTorch, Colossal-AI.
# Base image provides CUDA, NCCL, PyTorch, and build tools.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
FROM ${BASE_IMAGE}

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        strace \
        git \
        wget \
        htop \
        iproute2 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Install Rust (Oobleck has Rust components) ---
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# --- Install nccl-tests (for communication validation) ---
RUN cd /opt && \
    git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
ENV PATH="/opt/nccl-tests/build:${PATH}"

# --- Clone Oobleck ---
# Pin to a known commit for reproducibility. Update as needed.
RUN cd /opt && \
    git clone https://github.com/SymbioticLab/Oobleck.git
    # To pin: && cd Oobleck && git checkout <commit-hash>

# --- Install Oobleck dependencies ---
# NOTE: The install steps below may need adjustment based on Oobleck's
# current requirements. Check /opt/Oobleck/README.md after cloning.
WORKDIR /opt/Oobleck
# RUN pip install -e .
# RUN pip install colossalai  # or specific version

# --- Workspace ---
WORKDIR /workspace
COPY scripts/ /workspace/scripts/

# Default: keep container alive for interactive use
CMD ["sleep", "infinity"]
