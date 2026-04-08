#!/usr/bin/env python3
"""
parse_results.py — Compare benchmark results across configurations.

Usage:
    python parse_results.py results/baseline.json results/isolated.json results/shim.json
"""

import json
import sys
from pathlib import Path


def load_results(path):
    with open(path) as f:
        return json.load(f)


def summarize(results):
    """Extract key metrics from a result set."""
    data = results["data"]
    if not data:
        return {"peak_busbw": 0, "mode": results["metadata"]["mode"]}

    peak = max(data, key=lambda d: d["busbw_gbps"])
    large_msg = [d for d in data if d["size_bytes"] >= 64 * 1024 * 1024]
    avg_large = (sum(d["busbw_gbps"] for d in large_msg) / len(large_msg)
                 if large_msg else 0)

    return {
        "mode": results["metadata"]["mode"],
        "gpus": results["metadata"]["gpus"],
        "peak_busbw_gbps": peak["busbw_gbps"],
        "peak_at_mb": peak["size_bytes"] / 1e6,
        "avg_large_busbw_gbps": round(avg_large, 2),
        "shim_loaded": results["metadata"].get("shim_loaded", False),
        "nccl_cumem": results["metadata"].get("nccl_cumem", False),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_results.py <result1.json> [result2.json] ...")
        sys.exit(1)

    summaries = []
    for path in sys.argv[1:]:
        results = load_results(path)
        s = summarize(results)
        s["file"] = Path(path).name
        summaries.append(s)

    # Print comparison table
    print(f"\n{'File':<30} {'Mode':<12} {'Peak BusBW':>12} {'Avg(≥64M)':>12} {'Shim':>6} {'cuMem':>6}")
    print("-" * 80)

    baseline_bw = None
    for s in summaries:
        if baseline_bw is None:
            baseline_bw = s["peak_busbw_gbps"]

        recovery = (s["peak_busbw_gbps"] / baseline_bw * 100
                    if baseline_bw > 0 else 0)

        print(f"{s['file']:<30} {s['mode']:<12} "
              f"{s['peak_busbw_gbps']:>9.2f} GB/s "
              f"{s['avg_large_busbw_gbps']:>9.2f} GB/s "
              f"{'yes' if s['shim_loaded'] else 'no':>6} "
              f"{'yes' if s['nccl_cumem'] else 'no':>6}")

    # Recovery ratio
    if len(summaries) >= 2 and baseline_bw and baseline_bw > 0:
        print(f"\n--- Recovery vs first file (baseline) ---")
        for s in summaries[1:]:
            ratio = s["peak_busbw_gbps"] / baseline_bw * 100
            print(f"  {s['file']}: {ratio:.1f}% of baseline peak bandwidth")


if __name__ == "__main__":
    main()
