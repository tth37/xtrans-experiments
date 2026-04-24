#!/usr/bin/env python3
"""Exp A3 per-GPU-container regime Python harness."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import common as a3

NETWORK = "xtrans-per-gpu"
MODEL_MOUNT_IN_CTN = "/models/qwen3-30b-a3b"
RESULTS_DIR = a3.EXP_DIR / "results" / "per_gpu_containers"
SERVE_LOG_IN_CTN = "/tmp/vllm-serve.log"


def ensure_network() -> None:
    if a3.run(["docker", "network", "inspect", NETWORK], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        a3.log(f"Creating docker network {NETWORK}")
        a3.run(["docker", "network", "create", NETWORK], stdout=subprocess.DEVNULL)


def head_ip() -> str:
    import json
    raw = a3.output(["docker", "inspect", "ep-rank-0"], check=True)
    data = json.loads(raw)[0]
    return data["NetworkSettings"]["Networks"][NETWORK]["IPAddress"]


def launch_rank_container(rank: int, start_cmd: str, image: str) -> None:
    name = f"ep-rank-{rank}"
    a3.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ports: list[str] = []
    if rank == 0:
        ports = ["-p", f"{a3.VLLM_PORT}:{a3.VLLM_PORT}", "-p", f"{a3.RAY_PORT}:{a3.RAY_PORT}"]
    a3.run([
        "docker", "run", "-d", "--name", name,
        "--hostname", name,
        "--network", NETWORK,
        *ports,
        "--gpus", f"device={rank}",
        "--ipc=host",
        "--shm-size", "16g",
        "-v", f"{a3.MODEL_HOST}:{MODEL_MOUNT_IN_CTN}:ro",
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", "VLLM_LOGGING_LEVEL=INFO",
        "-e", "NCCL_DEBUG=INFO",
        "--entrypoint", "/bin/bash",
        image,
        "-c", f"{start_cmd} && tail -f /dev/null",
    ], stdout=subprocess.DEVNULL)


def liveness() -> bool:
    for rank in range(4):
        state = a3.output(["docker", "inspect", "-f", "{{.State.Running}}", f"ep-rank-{rank}"])
        if state != "true":
            return False
    return a3.run(["docker", "exec", "ep-rank-0", "pgrep", "-f", "vllm serve"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def diag() -> None:
    a3.log("Container states:")
    for rank in range(4):
        print("    " + f"ep-rank-{rank}: " + (a3.output(["docker", "inspect", "-f", "Status={{.State.Status}} ExitCode={{.State.ExitCode}}", f"ep-rank-{rank}"]) or "missing"), file=sys.stderr)
    a3.log("Last 40 lines of vllm-serve log (inside ep-rank-0):")
    for line in a3.output(["docker", "exec", "ep-rank-0", "tail", "-40", SERVE_LOG_IN_CTN]).splitlines():
        print(f"    {line}", file=sys.stderr)


def up() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    image = os.environ.get("VLLM_IMAGE")
    if not image or image == "xtrans-vllm-ep:v0.19.0":
        image = a3.VLLM_IMAGE_PATCHED
        a3.ensure_patched_image()
    else:
        a3.docker_ensure_image(image)
    a3.require_gpus_free()
    ensure_network()
    launch_rank_container(0, f"ray start --head --port {a3.RAY_PORT} --num-gpus 1 --node-ip-address ep-rank-0 --min-worker-port 30000 --max-worker-port 39999 --disable-usage-stats", image)
    a3.log("Waiting for ray head to be ready...")
    for i in range(1, 21):
        time.sleep(2)
        if a3.run(["docker", "exec", "ep-rank-0", "ray", "status"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            a3.log(f"Ray head ready after {i * 2}s")
            break
    hip = head_ip()
    a3.log(f"Ray head bridge IP: {hip}")
    for rank in range(1, 4):
        launch_rank_container(rank, f"ray start --address {hip}:{a3.RAY_PORT} --num-gpus 1 --node-ip-address ep-rank-{rank} --min-worker-port 30000 --max-worker-port 39999 --disable-usage-stats", image)
    a3.log("Waiting for all Ray workers to register...")
    for i in range(1, 16):
        time.sleep(2)
        status = a3.output(["docker", "exec", "ep-rank-0", "ray", "status"])
        if "4.0/4.0 GPU" in status or "0.0/4.0 GPU" in status:
            a3.log(f"4-GPU Ray cluster formed after {i * 2}s")
            break
    (RESULTS_DIR / "ray_status.txt").write_text(a3.output(["docker", "exec", "ep-rank-0", "ray", "status"]) + "\n")
    dp_size = int(os.environ.get("PER_GPU_DP", "2"))
    extra = os.environ.get("EXTRA_SERVE_ARGS", "")
    a3.log(f"Launching vllm serve in ep-rank-0 at DP={dp_size} (extra={extra or 'none'})")
    serve = f"""
export RAY_ADDRESS=ep-rank-0:{a3.RAY_PORT}
export VLLM_LOGGING_LEVEL=INFO
vllm serve {a3.MODEL_PATH_IN_CTN} \
  --served-model-name {a3.SERVED_MODEL_NAME} \
  --host 0.0.0.0 --port {a3.VLLM_PORT} \
  --tensor-parallel-size 1 \
  --data-parallel-size {dp_size} \
  --data-parallel-size-local 1 \
  --data-parallel-backend ray \
  --data-parallel-address ep-rank-0 \
  --enable-expert-parallel \
  --enable-elastic-ep \
  --enable-eplb \
  --all2all-backend allgather_reducescatter \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --trust-remote-code \
  {extra} \
  > {SERVE_LOG_IN_CTN} 2>&1
"""
    a3.run(["docker", "exec", "-d", "ep-rank-0", "/bin/bash", "-c", serve])
    time.sleep(15)
    a3.wait_for_ready(f"http://localhost:{a3.VLLM_PORT}/health", 600, liveness, diag)


def down() -> None:
    a3.log(f"Saving per-container logs to {RESULTS_DIR}/")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for rank in range(4):
        name = f"ep-rank-{rank}"
        if a3.output(["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"]):
            (RESULTS_DIR / f"{name}.log").write_text(a3.output(["docker", "logs", name]) + "\n")
    if a3.run(["docker", "exec", "ep-rank-0", "test", "-f", SERVE_LOG_IN_CTN], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        (RESULTS_DIR / "vllm-serve.log").write_text(a3.output(["docker", "exec", "ep-rank-0", "cat", SERVE_LOG_IN_CTN]) + "\n")
    a3.log("Stopping containers")
    for rank in range(4):
        a3.run(["docker", "rm", "-f", f"ep-rank-{rank}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a3.log(f"Network {NETWORK} left in place; remove with: docker network rm {NETWORK}")
    time.sleep(3)
    print(a3.gpu_snapshot(), file=sys.stderr)


def scale(new_dp: int) -> None:
    a3.trigger_scale(f"http://localhost:{a3.VLLM_PORT}", new_dp, RESULTS_DIR)


def state(tag: str = "snapshot") -> None:
    ip_lines = a3.output(["docker", "network", "inspect", NETWORK, "--format", "{{range .Containers}}{{.Name}} {{.IPv4Address}}{{println}}{{end}}"])
    device_lines = [f"  ep-rank-{rank}: {a3.container_deviceids(f'ep-rank-{rank}')}" for rank in range(4)]
    a3.write_state(RESULTS_DIR / f"state_{tag}.txt", [
        f"=== Per-GPU containers state ({tag}) at {time.ctime()} ===", "",
        "## Containers ##", a3.output(["docker", "ps", "--filter", "name=ep-rank-", "--format", "table {{.Names}}\t{{.Status}}"]), "",
        "## Per-container DeviceIDs (each owns one GPU!) ##", *device_lines, "",
        "## Bridge network IPs ##", ip_lines, "",
        "## Host nvidia-smi ##", a3.gpu_snapshot(), "",
        "## Scaling flag ##", a3.post_text(f"http://localhost:{a3.VLLM_PORT}"),
    ])


def nccl_grep() -> None:
    if a3.run(["docker", "exec", "ep-rank-0", "test", "-f", SERVE_LOG_IN_CTN], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise SystemExit("vllm-serve log not found in ep-rank-0")
    pattern = r"NCCL INFO (Assigned NET plugin|Channel [0-9]+/[0-9]+ : |Check P2P Type|Connected all rings)"
    text = a3.output(["docker", "exec", "ep-rank-0", "grep", "-E", pattern, SERVE_LOG_IN_CTN])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "nccl_transport.log").write_text(text + "\n")
    print("\n".join(text.splitlines()[:40]))
    print(f"\nFull capture saved to {RESULTS_DIR / 'nccl_transport.log'}")


def cycle() -> None:
    started_here = False
    if not a3.http_get_ok(f"http://localhost:{a3.VLLM_PORT}/health"):
        up()
        started_here = True
    def after() -> None:
        try:
            nccl_grep()
        except SystemExit:
            pass
    try:
        a3.run_plateau_cycle("per_gpu_containers", RESULTS_DIR, scale, state, after)
    finally:
        if started_here:
            down()


def usage() -> None:
    print("usage: python 3_per_gpu_containers.py {start|up|stop|down|scale TARGET_DP|state [TAG]|cycle|nccl-grep}", file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str]) -> None:
    cmd, *args = argv
    if cmd in {"start", "up"}:
        up()
    elif cmd in {"stop", "down"}:
        down()
    elif cmd == "scale" and len(args) == 1:
        scale(int(args[0]))
    elif cmd == "state":
        state(args[0] if args else "snapshot")
    elif cmd == "cycle" and not args:
        cycle()
    elif cmd == "nccl-grep" and not args:
        nccl_grep()
    else:
        usage()


if __name__ == "__main__":
    main(sys.argv[1:]) if len(sys.argv) > 1 else usage()
