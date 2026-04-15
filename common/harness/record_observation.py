#!/usr/bin/env python3
"""
record_observation.py — Structured observation recorder for v4 experiments.

Records experimental observations with timestamps, metadata, and free-form
notes into JSON files. Designed to be called after each experimental step
to build a structured record of what happened.

Usage:
    # Record a single observation
    python record_observation.py \
        --experiment exp_a1 \
        --phase phase1 \
        --step "scale_down_4_to_2" \
        --output results/phase1_scale_down.json \
        --notes "Tenplex reconfigured from TP=2,DP=2 to TP=2,DP=1 in 45s"

    # Record with metrics
    python record_observation.py \
        --experiment exp_a2 \
        --phase phase3 \
        --step "c3_failure_recovery" \
        --output results/phase3_c3_failure.json \
        --notes "NCCL communicator hung for 30s then timed out" \
        --metric "detection_time_s=30.2" \
        --metric "recovery_time_s=0" \
        --metric "recovered=false"

    # Record with a log file attachment
    python record_observation.py \
        --experiment exp_a1 \
        --phase phase3 \
        --step "level0_training_attempt" \
        --output results/phase3_level0.json \
        --notes "NCCL fell back to TCP, 4.2 GB/s" \
        --attach-log logs/nccl_debug.log

    # Append to an existing observation file (adds to "updates" list)
    python record_observation.py \
        --experiment exp_a1 \
        --phase phase3 \
        --step "level0_training_attempt" \
        --output results/phase3_level0.json \
        --notes "After enabling shim: NVLink restored, 156 GB/s" \
        --append
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def get_gpu_info():
    """Collect GPU information from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_container_info():
    """Check if we're running inside a container."""
    # Check for Docker
    if os.path.exists("/.dockerenv"):
        return "docker"
    # Check for cgroup v2
    try:
        with open("/proc/1/cgroup") as f:
            if "docker" in f.read():
                return "docker"
    except (FileNotFoundError, PermissionError):
        pass
    return "bare_metal"


def get_git_info():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def parse_metric(metric_str):
    """Parse a metric string like 'key=value' into (key, parsed_value)."""
    key, _, value = metric_str.partition("=")
    # Try to parse as number or boolean
    if value.lower() == "true":
        return key, True
    if value.lower() == "false":
        return key, False
    try:
        return key, int(value)
    except ValueError:
        try:
            return key, float(value)
        except ValueError:
            return key, value


def read_log_tail(path, max_lines=100):
    """Read the last N lines of a log file."""
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except (FileNotFoundError, PermissionError) as e:
        return f"Error reading {path}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Record structured experimental observations"
    )
    parser.add_argument("--experiment", required=True,
                        help="Experiment ID (e.g., exp_a1, exp_a2)")
    parser.add_argument("--phase", required=True,
                        help="Phase (e.g., phase1, phase2, phase3)")
    parser.add_argument("--step", required=True,
                        help="Step name (e.g., scale_down_4_to_2)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSON file path")
    parser.add_argument("--notes", default="",
                        help="Free-form observation notes")
    parser.add_argument("--metric", action="append", default=[],
                        help="Key=value metric (repeatable)")
    parser.add_argument("--attach-log",
                        help="Attach tail of a log file")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing observation file")
    parser.add_argument("--workaround-level", type=int, default=None,
                        help="Workaround level (0-4) if in Phase 3")

    args = parser.parse_args()

    now = datetime.now(timezone.utc)

    # Parse metrics
    metrics = {}
    for m in args.metric:
        key, value = parse_metric(m)
        metrics[key] = value

    # Build observation record
    observation = {
        "timestamp": now.isoformat(),
        "experiment": args.experiment,
        "phase": args.phase,
        "step": args.step,
        "notes": args.notes,
        "metrics": metrics,
        "environment": {
            "runtime": get_container_info(),
            "gpu_info": get_gpu_info(),
            "git_commit": get_git_info(),
        },
    }

    if args.workaround_level is not None:
        observation["workaround_level"] = args.workaround_level

    if args.attach_log:
        observation["log_tail"] = read_log_tail(args.attach_log)
        observation["log_file"] = args.attach_log

    # Write or append
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.append and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        if "updates" not in existing:
            existing["updates"] = []
        existing["updates"].append(observation)
        data = existing
    else:
        data = observation

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[observation] Recorded: {args.experiment}/{args.phase}/{args.step}")
    print(f"[observation] Saved to: {args.output}")
    if args.notes:
        print(f"[observation] Notes: {args.notes}")
    if metrics:
        print(f"[observation] Metrics: {metrics}")


if __name__ == "__main__":
    main()
