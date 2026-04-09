#!/usr/bin/env python3
"""
version_matrix.py — Build NCCL version compatibility matrix from A3 results.

Loads version_*.json result files, extracts NCCL version and bandwidth,
and produces a formatted compatibility matrix.

Usage:
    python3 version_matrix.py results/version_2.18.json results/version_2.21.json ...
"""

import json
import sys
from pathlib import Path


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_peak_bw(results, operation="all_reduce"):
    op_data = results.get("results", {}).get(operation, {})
    if not op_data:
        return 0.0
    return max(m.get("bandwidth_gbps", 0) for m in op_data.values())


def main():
    if len(sys.argv) < 2:
        print("Usage: version_matrix.py <result1.json> [result2.json] ...")
        sys.exit(1)

    # Load all results
    entries = []
    for path in sys.argv[1:]:
        try:
            r = load_results(path)
            nccl_ver = r.get("nccl_version", "?")
            pytorch_ver = r.get("pytorch_version", "?")
            deployment = r.get("deployment", "?")
            entries.append({
                "file": Path(path).name,
                "nccl_version": nccl_ver,
                "pytorch_version": pytorch_ver,
                "deployment": deployment,
                "all_reduce": get_peak_bw(r, "all_reduce"),
                "all_gather": get_peak_bw(r, "all_gather"),
                "p2p": get_peak_bw(r, "p2p"),
                "nccl_env": r.get("nccl_env", {}),
            })
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"WARNING: {path}: {e}", file=sys.stderr)

    if not entries:
        print("No valid results loaded.", file=sys.stderr)
        sys.exit(1)

    # Use the NCCL 2.21 result as baseline if available, otherwise first entry
    baseline_bw = None
    for e in entries:
        if "2.21" in str(e["nccl_version"]) or "2, 21" in str(e["nccl_version"]):
            baseline_bw = e["all_reduce"]
            break
    if baseline_bw is None:
        baseline_bw = entries[0]["all_reduce"]

    # Print matrix
    print(f"\n{'='*85}")
    print("NCCL Version Compatibility Matrix")
    print(f"{'='*85}")
    print(f"{'NCCL':<20} {'PyTorch':<12} {'AllReduce':>10} {'AllGather':>10} "
          f"{'P2P':>10} {'vs 2.21':>9} {'Status':>8}")
    print("-" * 85)

    all_pass = True
    for e in entries:
        ratio = (e["all_reduce"] / baseline_bw * 100) if baseline_bw > 0 else 0
        passed = ratio >= 95  # Allow 5% variance across versions
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"{str(e['nccl_version']):<20} {e['pytorch_version']:<12} "
              f"{e['all_reduce']:>7.1f} GB/s {e['all_gather']:>7.1f} GB/s "
              f"{e['p2p']:>7.1f} GB/s "
              f"{ratio:>7.1f}% "
              f"{status:>8}")

    print("-" * 85)
    print(f"Baseline (NCCL 2.21): {baseline_bw:.1f} GB/s all_reduce")
    if all_pass:
        print("VERDICT: All versions PASS (>=95% of baseline)")
    else:
        print("VERDICT: Some versions FAIL — check logs")
    print(f"{'='*85}\n")

    # Save structured output
    output_path = Path(sys.argv[1]).parent / "version_matrix.json"
    matrix = {
        "baseline_nccl": "2.21",
        "baseline_all_reduce_gbps": baseline_bw,
        "versions": entries,
    }
    with open(output_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"Matrix saved to {output_path}")


if __name__ == "__main__":
    main()
