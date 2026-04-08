#!/usr/bin/env python3
"""
nccl_bench.py — Standardized NCCL benchmark harness for XTrans experiments.

Wraps nccl-tests (all_reduce_perf) with consistent configuration, output
parsing, and result storage. Used across all experiments for comparable
measurements.

Usage:
    # Single-container baseline (all GPUs visible)
    python nccl_bench.py --mode shared --gpus 0,1 --output results/baseline.json

    # Per-GPU containers (isolated, expect degraded)
    python nccl_bench.py --mode isolated --gpus 0,1 --output results/isolated.json

    # Per-GPU containers with xtrans shim
    python nccl_bench.py --mode shim --gpus 0,1 --output results/shim.json

    # Compare results
    python parse_results.py results/baseline.json results/isolated.json results/shim.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Default nccl-tests parameters
DEFAULTS = {
    "binary": "all_reduce_perf",
    "min_bytes": "1M",
    "max_bytes": "1G",
    "step_factor": "2",
    "iters": "50",
    "warmup_iters": "20",
    "op": "sum",
    "datatype": "float",
}


def find_nccl_tests_binary(name="all_reduce_perf"):
    """Locate nccl-tests binary in common paths."""
    search_paths = [
        "/opt/nccl-tests/build",
        "/usr/local/bin",
        os.path.expanduser("~/nccl-tests/build"),
    ]
    for p in search_paths:
        full = os.path.join(p, name)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return full
    return name  # hope it's in PATH


def build_nccl_test_cmd(args):
    """Build the nccl-tests command line."""
    binary = find_nccl_tests_binary(args.binary)
    cmd = [
        binary,
        "-b", args.min_bytes,
        "-e", args.max_bytes,
        "-f", args.step_factor,
        "-n", str(args.iters),
        "-w", str(args.warmup_iters),
        "-o", args.op,
        "-d", args.datatype,
        "-g", "1",  # 1 GPU per process (always, for per-container model)
    ]
    return cmd


def parse_nccl_output(stdout):
    """Parse nccl-tests stdout into structured results.

    Extracts the table rows with size/bandwidth/latency data.
    Returns list of dicts with keys: size, count, type, redop, root,
    time_us, algbw_gbps, busbw_gbps.
    """
    results = []
    in_table = False
    for line in stdout.split("\n"):
        line = line.strip()
        # Detect table start
        if line.startswith("#") and "size" in line.lower() and "time" in line.lower():
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("#") or not line:
            continue
        # Parse data row
        parts = line.split()
        if len(parts) >= 8:
            try:
                results.append({
                    "size_bytes": int(parts[0]),
                    "count": int(parts[1]),
                    "type": parts[2],
                    "redop": parts[3],
                    "root": int(parts[4]),
                    "time_us": float(parts[5]),
                    "algbw_gbps": float(parts[6]),
                    "busbw_gbps": float(parts[7]),
                })
            except (ValueError, IndexError):
                continue
    return results


def run_benchmark(args):
    """Run the benchmark and return structured results."""
    cmd = build_nccl_test_cmd(args)

    env = os.environ.copy()

    # Apply NCCL env vars based on mode
    if args.mode == "shim":
        shim_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "shim", "libxtrans_shim.so"
        )
        if os.path.exists(shim_path):
            env["LD_PRELOAD"] = shim_path
            env["XTRANS_VERBOSE"] = "1"
        else:
            print(f"WARNING: Shim not found at {shim_path}, run `make` in common/shim/ first",
                  file=sys.stderr)

    if args.nccl_cumem:
        env["NCCL_CUMEM_ENABLE"] = "1"

    if args.nccl_debug:
        env["NCCL_DEBUG"] = "INFO"

    # Extra env vars
    for kv in (args.env or []):
        k, v = kv.split("=", 1)
        env[k] = v

    print(f"[nccl_bench] Mode: {args.mode}")
    print(f"[nccl_bench] GPUs: {args.gpus}")
    print(f"[nccl_bench] Command: {' '.join(cmd)}")
    print(f"[nccl_bench] Running...")

    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=600
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[nccl_bench] FAILED (exit {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None

    print(f"[nccl_bench] Completed in {elapsed:.1f}s")

    # Parse results
    data_points = parse_nccl_output(result.stdout)

    return {
        "metadata": {
            "mode": args.mode,
            "gpus": args.gpus,
            "binary": args.binary,
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(elapsed, 2),
            "nccl_cumem": args.nccl_cumem,
            "shim_loaded": args.mode == "shim",
            "cmd": " ".join(cmd),
        },
        "data": data_points,
        "raw_stdout": result.stdout,
        "raw_stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description="XTrans NCCL Benchmark Harness")
    parser.add_argument("--mode", choices=["shared", "isolated", "shim", "custom"],
                        default="shared", help="Container isolation mode")
    parser.add_argument("--gpus", default="0,1",
                        help="GPU indices (comma-separated)")
    parser.add_argument("--binary", default=DEFAULTS["binary"],
                        help="nccl-tests binary name")
    parser.add_argument("--min-bytes", default=DEFAULTS["min_bytes"], dest="min_bytes")
    parser.add_argument("--max-bytes", default=DEFAULTS["max_bytes"], dest="max_bytes")
    parser.add_argument("--step-factor", default=DEFAULTS["step_factor"], dest="step_factor")
    parser.add_argument("--iters", type=int, default=int(DEFAULTS["iters"]))
    parser.add_argument("--warmup-iters", type=int, default=int(DEFAULTS["warmup_iters"]),
                        dest="warmup_iters")
    parser.add_argument("--op", default=DEFAULTS["op"])
    parser.add_argument("--datatype", default=DEFAULTS["datatype"])
    parser.add_argument("--nccl-cumem", action="store_true", dest="nccl_cumem",
                        help="Set NCCL_CUMEM_ENABLE=1")
    parser.add_argument("--nccl-debug", action="store_true", dest="nccl_debug",
                        help="Set NCCL_DEBUG=INFO")
    parser.add_argument("--env", action="append",
                        help="Extra env vars (KEY=VALUE), can repeat")
    parser.add_argument("--output", "-o", help="Output JSON file path")

    args = parser.parse_args()

    results = run_benchmark(args)
    if results is None:
        sys.exit(1)

    # Print summary
    if results["data"]:
        peak = max(results["data"], key=lambda d: d["busbw_gbps"])
        print(f"\n[nccl_bench] Peak bus bandwidth: {peak['busbw_gbps']:.2f} GB/s "
              f"at {peak['size_bytes'] / 1e6:.0f} MB")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[nccl_bench] Results saved to {args.output}")
    else:
        print("\n[nccl_bench] Tip: use --output to save results as JSON")


if __name__ == "__main__":
    main()
