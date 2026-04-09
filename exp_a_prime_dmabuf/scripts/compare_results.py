#!/usr/bin/env python3
"""
compare_results.py — Compare DMA-BUF IPC benchmark results across paths.

Usage:
    python3 compare_results.py results/cumem_fd_bw.json results/cudaipc_bw.json
"""

import json
import sys
from pathlib import Path

SUCCESS_THRESHOLD = 0.95  # 95% of native cuMem = pass


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_peak_bw(results):
    data = results.get("results", {})
    if not data:
        return 0.0, "N/A"
    peak_bw = 0.0
    peak_size = "N/A"
    for size_label, metrics in data.items():
        bw = metrics.get("bandwidth_gbps", 0)
        if bw > peak_bw:
            peak_bw = bw
            peak_size = size_label
    return peak_bw, peak_size


def main():
    if len(sys.argv) < 2:
        print("Usage: compare_results.py <result1.json> [result2.json] ...")
        sys.exit(1)

    entries = []
    for path in sys.argv[1:]:
        try:
            r = load_results(path)
            peak_bw, peak_size = get_peak_bw(r)
            entries.append({
                "file": Path(path).name,
                "ipc_path": r.get("ipc_path", "?"),
                "peak_bw": peak_bw,
                "peak_size": peak_size,
                "results": r.get("results", {}),
            })
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"WARNING: {path}: {e}", file=sys.stderr)

    if not entries:
        print("No valid results.", file=sys.stderr)
        sys.exit(1)

    # Print header
    print(f"\n{'='*75}")
    print("Exp A' — IPC Path Comparison")
    print(f"{'='*75}")

    # Per-size comparison
    sizes = list(entries[0]["results"].keys()) if entries else []
    print(f"\n{'Size':<12}", end="")
    for e in entries:
        print(f" {e['ipc_path']:>15}", end="")
    if len(entries) >= 2:
        print(f" {'vs cuMem':>10}", end="")
    print()
    print("-" * 75)

    baseline_entry = entries[0]
    for size in sizes:
        print(f"{size:<12}", end="")
        baseline_bw = baseline_entry["results"].get(size, {}).get("bandwidth_gbps", 0)
        for e in entries:
            bw = e["results"].get(size, {}).get("bandwidth_gbps", 0)
            print(f" {bw:>12.1f} GB/s", end="")
        if len(entries) >= 2:
            other_bw = entries[-1]["results"].get(size, {}).get("bandwidth_gbps", 0)
            ratio = (other_bw / baseline_bw * 100) if baseline_bw > 0 else 0
            print(f" {ratio:>8.1f}%", end="")
        print()

    # Summary
    print(f"\n{'='*75}")
    print("Peak Bandwidth:")
    baseline_peak = entries[0]["peak_bw"] if entries else 0
    for e in entries:
        ratio = (e["peak_bw"] / baseline_peak * 100) if baseline_peak > 0 else 0
        status = "PASS" if ratio >= SUCCESS_THRESHOLD * 100 else "FAIL"
        print(f"  {e['ipc_path']:<15} {e['peak_bw']:>8.1f} GB/s at {e['peak_size']} "
              f"({ratio:.1f}%) [{status}]")

    # Exp A baseline comparison
    exp_a_p2p_baseline = 218.9  # GB/s from Exp A results
    print(f"\nvs Exp A P2P baseline ({exp_a_p2p_baseline:.1f} GB/s):")
    for e in entries:
        ratio = (e["peak_bw"] / exp_a_p2p_baseline * 100)
        print(f"  {e['ipc_path']:<15} {ratio:.1f}%")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
