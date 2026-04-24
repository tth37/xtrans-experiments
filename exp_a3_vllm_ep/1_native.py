#!/usr/bin/env python3
"""Exp A3 native regime Python harness."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import common as a3

RESULTS_DIR = a3.EXP_DIR / "results" / "native"
SERVE_SESSION = "a3-native-serve"
SERVE_LOG = RESULTS_DIR / "serve.log"


def native_liveness() -> bool:
    return a3.run(["tmux", "has-session", "-t", SERVE_SESSION], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def native_diag() -> None:
    a3.log("Last 30 lines of vllm serve log:")
    if SERVE_LOG.exists():
        for line in SERVE_LOG.read_text(errors="replace").splitlines()[-30:]:
            print(f"    {line}", file=sys.stderr)


def start() -> None:
    a3.ensure_venv()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    a3.require_gpus_free()
    a3.log(f"Starting dedicated Ray head at 127.0.0.1:{a3.RAY_PORT}")
    with (RESULTS_DIR / "ray_head.log").open("w") as logf:
        a3.run([
            "ray", "start", "--head", "--port", str(a3.RAY_PORT),
            "--dashboard-port", str(a3.RAY_PORT - 114),
            "--node-ip-address", "127.0.0.1", "--num-gpus", "4",
            "--min-worker-port", "30000", "--max-worker-port", "39999",
            "--disable-usage-stats",
        ], env={**os.environ, "RAY_USAGE_STATS_ENABLED": "0"}, stdout=logf, stderr=subprocess.STDOUT)
    SERVE_LOG.unlink(missing_ok=True)
    a3.log(f"Launching vllm serve in tmux session '{SERVE_SESSION}'")
    cmd = f"""source {a3.VENV_DIR}/bin/activate && \
export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
export RAY_ADDRESS=127.0.0.1:{a3.RAY_PORT} && \
vllm serve {a3.MODEL_SNAPSHOT} \
  --served-model-name {a3.SERVED_MODEL_NAME} \
  --host 0.0.0.0 --port {a3.VLLM_PORT} \
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
  {os.environ.get('EXTRA_SERVE_ARGS', '')} \
  2>&1 | tee {SERVE_LOG}"""
    a3.tmux_oneshot(SERVE_SESSION, cmd)
    a3.wait_for_ready(f"http://localhost:{a3.VLLM_PORT}/health", 300, native_liveness, native_diag)


def stop() -> None:
    a3.log("Stopping vllm serve tmux session")
    a3.tmux_kill(SERVE_SESSION)
    time.sleep(3)
    a3.run("pkill -9 -f 'DPMoE|RayWorkerWrapper'", check=False, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a3.log("Stopping Ray")
    a3.run(["ray", "stop", "--force"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    print(a3.gpu_snapshot(), file=sys.stderr)


def scale(new_dp: int) -> None:
    a3.trigger_scale(f"http://localhost:{a3.VLLM_PORT}", new_dp, RESULTS_DIR)


def state(tag: str = "snapshot") -> None:
    a3.write_state(RESULTS_DIR / f"state_{tag}.txt", [
        f"=== Native state ({tag}) at {time.ctime()} ===", "",
        "## GPU memory + utilisation ##", a3.gpu_snapshot(), "",
        "## GPU processes ##", a3.gpu_processes(), "",
        "## Scaling flag ##", a3.post_text(f"http://localhost:{a3.VLLM_PORT}"),
    ])


def cycle() -> None:
    started_here = False
    if not a3.http_get_ok(f"http://localhost:{a3.VLLM_PORT}/health"):
        start()
        started_here = True
    try:
        a3.run_bench_cycle("native", RESULTS_DIR, scale, state)
    finally:
        if started_here:
            stop()


def bench() -> None:
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
    print("usage: python 1_native.py {start|up|stop|down|scale TARGET_DP|state [TAG]|cycle|bench}", file=sys.stderr)
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
        bench()
    else:
        usage()


if __name__ == "__main__":
    main(sys.argv[1:]) if len(sys.argv) > 1 else usage()
