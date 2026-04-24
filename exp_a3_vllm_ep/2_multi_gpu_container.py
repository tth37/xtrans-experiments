#!/usr/bin/env python3
"""Exp A3 multi-GPU-container regime Python harness."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import common as a3

CONTAINER_NAME = "xtrans-exp-a3-multi-gpu"
MODEL_MOUNT_IN_CTN = "/models/qwen3-30b-a3b"
RESULTS_DIR = a3.EXP_DIR / "results" / "multi_gpu_container"


def liveness() -> bool:
    state = a3.output(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME])
    return state == "true"


def diag() -> None:
    a3.log("Container state:")
    print("    " + (a3.output(["docker", "inspect", "-f", "Status={{.State.Status}} ExitCode={{.State.ExitCode}}", CONTAINER_NAME]) or "missing"), file=sys.stderr)
    a3.log("Last 40 lines of container log:")
    for line in a3.output(["docker", "logs", "--tail", "40", CONTAINER_NAME]).splitlines():
        print(f"    {line}", file=sys.stderr)


def start() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    a3.docker_ensure_image(a3.VLLM_IMAGE)
    a3.require_gpus_free()
    a3.run(["docker", "rm", "-f", CONTAINER_NAME], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    extra = os.environ.get("EXTRA_SERVE_ARGS", "")
    serve_cmd = f"""vllm serve {a3.MODEL_PATH_IN_CTN} \
 --served-model-name {a3.SERVED_MODEL_NAME} \
 --host 0.0.0.0 --port 8000 \
 --tensor-parallel-size 1 \
 --data-parallel-size {os.environ.get("A3_START_DP", "2")} \
 --data-parallel-backend ray \
 --enable-expert-parallel \
 --enable-elastic-ep \
 --enable-eplb \
 --all2all-backend allgather_reducescatter \
 --max-model-len 2048 \
 --max-num-seqs {os.environ.get("A3_MAX_NUM_SEQS", "16")} \
 --gpu-memory-utilization 0.90 \
 --enforce-eager \
 --trust-remote-code \
 {extra}"""
    a3.log(f"Launching container {CONTAINER_NAME}")
    a3.run([
        "docker", "run", "-d", "--name", CONTAINER_NAME,
        "--gpus", '"device=0,1,2,3"',
        "--ipc=host",
        "--shm-size", "16g",
        "-v", f"{a3.MODEL_HOST}:{MODEL_MOUNT_IN_CTN}:ro",
        "-p", f"{a3.VLLM_PORT}:8000",
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", "VLLM_LOGGING_LEVEL=INFO",
        "--entrypoint", "/bin/bash",
        a3.VLLM_IMAGE,
        "-lc", serve_cmd,
    ], stdout=subprocess.DEVNULL)
    a3.wait_for_ready(f"http://localhost:{a3.VLLM_PORT}/health", 600, liveness, diag)


def stop() -> None:
    if a3.output(["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"]):
        a3.log(f"Saving container logs to {RESULTS_DIR / 'container.log'}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "container.log").write_text(a3.output(["docker", "logs", CONTAINER_NAME]) + "\n")
    a3.log(f"Removing container {CONTAINER_NAME}")
    a3.run(["docker", "rm", "-f", CONTAINER_NAME], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    print(a3.gpu_snapshot(), file=sys.stderr)


def scale(new_dp: int) -> None:
    a3.trigger_scale(f"http://localhost:{a3.VLLM_PORT}", new_dp, RESULTS_DIR)


def state(tag: str = "snapshot") -> None:
    a3.write_state(RESULTS_DIR / f"state_{tag}.txt", [
        f"=== Multi-GPU container state ({tag}) at {time.ctime()} ===", "",
        "## Container ##", a3.output(["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"]), "",
        "## Docker DeviceIDs (what Docker still claims) ##", a3.container_deviceids(CONTAINER_NAME), "",
        "## Host nvidia-smi (what's actually in use) ##", a3.gpu_snapshot(), "",
        "## GPU processes on host ##", a3.gpu_processes(), "",
        "## Scaling flag ##", a3.post_text(f"http://localhost:{a3.VLLM_PORT}"),
    ])


def cycle() -> None:
    started_here = False
    if not a3.http_get_ok(f"http://localhost:{a3.VLLM_PORT}/health"):
        start()
        started_here = True
    try:
        a3.run_bench_cycle("multi_gpu_container", RESULTS_DIR, scale, state)
    finally:
        if started_here:
            stop()


def bench_dp4() -> None:
    started_here = False
    if not a3.http_get_ok(f"http://localhost:{a3.VLLM_PORT}/health"):
        os.environ["A3_START_DP"] = "4"
        start()
        started_here = True
    try:
        a3.run_single_bench(os.environ.get("A3_SINGLE_LABEL", "dp4_direct"), RESULTS_DIR, int(os.environ.get("A3_SINGLE_NUM_PROMPTS", "128")), int(os.environ.get("A3_SINGLE_CONCURRENCY", "32")))
    finally:
        if started_here:
            stop()


def usage() -> None:
    print("usage: python 2_multi_gpu_container.py {start|up|stop|down|scale TARGET_DP|state [TAG]|cycle|bench}", file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str]) -> None:
    cmd, *args = argv
    if cmd in {"start", "up"}:
        start()
    elif cmd in {"stop", "down"}:
        stop()
    elif cmd == "scale" and len(args) == 1:
        scale(int(args[0]))
    elif cmd == "state":
        state(args[0] if args else "snapshot")
    elif cmd == "cycle" and not args:
        cycle()
    elif cmd == "bench" and not args:
        bench_dp4()
    else:
        usage()


if __name__ == "__main__":
    main(sys.argv[1:]) if len(sys.argv) > 1 else usage()
