#!/usr/bin/env python3
"""
compare_results.py — Compare benchmark results across Exp A configurations.

Loads PyTorch benchmark JSON results from different container configs
and prints a comparison table with bandwidth ratios vs baseline.

Usage:
    python3 compare_results.py results/baseline.json results/isolated.json \
                               results/exp2.json results/shim.json
"""

import json
import sys
from pathlib import Path


SUCCESS_THRESHOLD = 0.99  # 99% of baseline = pass


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_peak_bandwidth(results, operation="all_reduce"):
    """Get peak bandwidth for an operation from results JSON."""
    op_data = results.get("results", {}).get(operation, {})
    if not op_data:
        return 0.0, "N/A"

    peak_bw = 0.0
    peak_size = "N/A"
    for size_label, metrics in op_data.items():
        bw = metrics.get("bandwidth_gbps", 0)
        if bw > peak_bw:
            peak_bw = bw
            peak_size = size_label
    return peak_bw, peak_size


def get_largest_msg_bandwidth(results, operation="all_reduce"):
    """Get bandwidth at the largest message size for an operation."""
    op_data = results.get("results", {}).get(operation, {})
    if not op_data:
        return 0.0, "N/A"

    # Find largest message size
    largest_size = 0
    largest_label = "N/A"
    largest_bw = 0.0
    for size_label, metrics in op_data.items():
        size_elements = metrics.get("size_elements", 0)
        if size_elements > largest_size:
            largest_size = size_elements
            largest_label = size_label
            largest_bw = metrics.get("bandwidth_gbps", 0)
    return largest_bw, largest_label


def has_nccl_env_vars(results):
    """Check if NCCL-specific env vars were set (excluding debug)."""
    nccl_env = results.get("nccl_env", {})
    for key in nccl_env:
        if key.startswith("NCCL_") and key not in (
            "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_SOCKET_IFNAME"
        ):
            return True
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: compare_results.py <result1.json> [result2.json] ...")
        sys.exit(1)

    operations = ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "p2p"]

    # Load all results
    all_results = []
    for path in sys.argv[1:]:
        try:
            r = load_results(path)
            r["_file"] = Path(path).name
            all_results.append(r)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"WARNING: could not load {path}: {e}", file=sys.stderr)

    if not all_results:
        print("No valid result files loaded.", file=sys.stderr)
        sys.exit(1)

    # Summary header
    print(f"\n{'='*90}")
    print("Exp A — Benchmark Comparison")
    print(f"{'='*90}")

    # Per-operation comparison (focus on all_reduce as primary metric)
    for op in operations:
        print(f"\n--- {op} ---")
        print(f"{'Config':<15} {'Deployment':<12} {'Peak BW':>10} {'Largest':>10} "
              f"{'vs Base':>9} {'NCCL env':>9} {'Shim':>6}")
        print("-" * 75)

        baseline_bw = None
        for r in all_results:
            deployment = r.get("deployment", "?")
            peak_bw, peak_size = get_peak_bandwidth(r, op)
            large_bw, large_size = get_largest_msg_bandwidth(r, op)
            nccl_env = has_nccl_env_vars(r)
            shim = "LD_PRELOAD" in r.get("nccl_env", {})

            if baseline_bw is None:
                baseline_bw = peak_bw

            ratio = (peak_bw / baseline_bw * 100) if baseline_bw > 0 else 0

            print(f"{r['_file']:<15} {deployment:<12} "
                  f"{peak_bw:>7.1f} GB/s {large_bw:>7.1f} GB/s "
                  f"{ratio:>7.1f}% "
                  f"{'yes' if nccl_env else 'no':>9} "
                  f"{'yes' if shim else 'no':>6}")

    # Pass/fail verdict
    print(f"\n{'='*90}")
    print("VERDICT")
    print(f"{'='*90}")

    baseline = all_results[0] if all_results else None
    baseline_peak, _ = get_peak_bandwidth(baseline, "all_reduce") if baseline else (0, "")

    for r in all_results:
        if r.get("deployment") == "shim":
            shim_peak, _ = get_peak_bandwidth(r, "all_reduce")
            if baseline_peak > 0:
                ratio = shim_peak / baseline_peak
                passed = ratio >= SUCCESS_THRESHOLD
                status = "PASS" if passed else "FAIL"
                print(f"  Shim all_reduce: {shim_peak:.1f} / {baseline_peak:.1f} GB/s "
                      f"= {ratio*100:.1f}% (threshold: {SUCCESS_THRESHOLD*100:.0f}%) "
                      f"[{status}]")
                nccl_env = has_nccl_env_vars(r)
                if nccl_env:
                    print(f"  WARNING: NCCL env vars detected in shim config")
            else:
                print("  Cannot compute ratio: baseline bandwidth is 0")
            break
    else:
        print("  No shim results found — run compose.shim.yml first")

    print()


if __name__ == "__main__":
    main()
