# Dockerfile — Exp A container image
#
# Based on reference/exp2/Dockerfile with strace added for A1 tracing.
# Same image serves both A1 (STRACE_WRAP=1) and A2 (default) runs.
#
# Build (from project root):
#   docker build -t tth37/xtrans-experiments:expa-benchmark -f exp_a_nccl_gates/Dockerfile .

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev strace gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.5.1 numpy \
    --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace

# Build shim from source inside the container (avoids glibc mismatch)
COPY common/shim/ /workspace/shim/
RUN cd /workspace/shim && make clean && make

# Copy benchmark scripts
COPY exp_a_nccl_gates/benchmark.py /workspace/
COPY exp_a_nccl_gates/benchmark_docker.py /workspace/
COPY exp_a_nccl_gates/entrypoint.sh /workspace/

RUN chmod +x /workspace/entrypoint.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["bash", "/workspace/entrypoint.sh"]
CMD ["python3", "benchmark_docker.py"]
