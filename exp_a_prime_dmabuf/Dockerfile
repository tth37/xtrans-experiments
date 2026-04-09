# Dockerfile — Exp A' container image
#
# CUDA devel image with nvcc for compiling the cuMem VMM benchmark.
#
# Build (from project root):
#   docker build -t tth37/xtrans-experiments:expa-prime-dmabuf \
#     -f exp_a_prime_dmabuf/Dockerfile .

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY exp_a_prime_dmabuf/Makefile /workspace/
COPY exp_a_prime_dmabuf/src/ /workspace/src/
RUN make

COPY exp_a_prime_dmabuf/entrypoint.sh /workspace/
RUN chmod +x /workspace/entrypoint.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["bash", "/workspace/entrypoint.sh"]
CMD ["./dmabuf_bench"]
