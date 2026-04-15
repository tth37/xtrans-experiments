#!/usr/bin/env python3
"""
compare_phases.py — Compare observations across experiment phases.

Reads observation JSON files from different phases and produces a
comparison summary. Useful for comparing bare-metal vs multi-GPU
container vs per-GPU container results.

Usage:
    python compare_phases.py \
        results/phase1_training.json \
        results/phase2_training.json \
        results/phase3_training.json
"""

import json
import sys
from pathlib import Path


def load_observation(path):
    """Load an observation JSON file."""
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: compare_phases.py <obs1.json> [obs2.json] ...")
        sys.exit(1)

    observations = []
    for path in sys.argv[1:]:
        obs = load_observation(path)
        obs["_file"] = Path(path).name
        observations.append(obs)

    # Print comparison table
    print("\n=== Phase Comparison ===\n")
    print(f"{'File':<40} {'Phase':<10} {'Step':<30} {'Runtime':<12}")
    print("-" * 95)

    for obs in observations:
        runtime = obs.get("environment", {}).get("runtime", "unknown")
        print(f"{obs['_file']:<40} {obs['phase']:<10} {obs['step']:<30} {runtime:<12}")

    # Print metrics comparison
    all_metric_keys = set()
    for obs in observations:
        all_metric_keys.update(obs.get("metrics", {}).keys())

    if all_metric_keys:
        print(f"\n--- Metrics ---\n")
        header = f"{'Metric':<30}"
        for obs in observations:
            header += f" {obs['phase']:<15}"
        print(header)
        print("-" * (30 + 15 * len(observations)))

        for key in sorted(all_metric_keys):
            row = f"{key:<30}"
            for obs in observations:
                val = obs.get("metrics", {}).get(key, "—")
                row += f" {str(val):<15}"
            print(row)

    # Print notes
    print(f"\n--- Notes ---\n")
    for obs in observations:
        notes = obs.get("notes", "")
        if notes:
            print(f"[{obs['phase']}/{obs['step']}]")
            print(f"  {notes}\n")

    # Print updates (if any observations have appended updates)
    for obs in observations:
        if "updates" in obs:
            print(f"\n--- Updates for {obs['_file']} ---")
            for update in obs["updates"]:
                print(f"  [{update['timestamp']}] {update.get('notes', '')}")


if __name__ == "__main__":
    main()
