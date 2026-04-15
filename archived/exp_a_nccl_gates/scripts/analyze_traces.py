#!/usr/bin/env python3
"""
analyze_traces.py — Parse strace output and extract NCCL gate-related syscalls.

Reads strace files from different container configurations (baseline, isolated,
shim) and identifies syscalls related to NCCL's three container gates:
  1. hostHash — gethostname(), open boot_id
  2. shmDev — stat/xstat on /dev/shm
  3. IPC socket — AF_UNIX socket/bind/connect, sendmsg SCM_RIGHTS

Outputs a summary table and structured JSON for further analysis.

Usage:
    python3 analyze_traces.py results/traces/
    python3 analyze_traces.py --config baseline:traces/baseline_rank0.strace \
                              --config isolated:traces/isolated_rank0.strace
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# Gate categories and their syscall patterns
GATE_PATTERNS = {
    "hostHash": [
        (r'gethostname\(', "gethostname"),
        (r'open(?:at)?\(.*boot_id', "open boot_id"),
    ],
    "shmDev": [
        (r'(?:__x)?stat(?:64)?\(.*"/dev/shm"', "stat /dev/shm"),
        (r'stat(?:x)?\(.*"/dev/shm"', "stat /dev/shm"),
    ],
    "ipc_socket": [
        (r'socket\(AF_UNIX', "socket AF_UNIX"),
        (r'bind\(\d+,\s*\{sa_family=AF_UNIX', "bind AF_UNIX"),
        (r'connect\(\d+,\s*\{sa_family=AF_UNIX', "connect AF_UNIX"),
        (r'sendmsg\(.*SCM_RIGHTS', "sendmsg SCM_RIGHTS"),
    ],
}


def parse_strace_file(filepath):
    """Parse an strace output file and extract gate-related syscalls.

    Returns a dict of {gate_name: [list of matching lines]}.
    """
    gates = defaultdict(list)
    all_matches = []

    try:
        with open(filepath, 'r', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                for gate_name, patterns in GATE_PATTERNS.items():
                    for pattern, label in patterns:
                        if re.search(pattern, line):
                            entry = {
                                "line_num": line_num,
                                "gate": gate_name,
                                "label": label,
                                "raw": line,
                            }
                            gates[gate_name].append(entry)
                            all_matches.append(entry)
                            break  # avoid double-matching same line
    except FileNotFoundError:
        print(f"WARNING: trace file not found: {filepath}", file=sys.stderr)
        return gates, []

    return gates, all_matches


def extract_abstract_sockets(matches):
    """Extract abstract socket paths from AF_UNIX syscalls."""
    abstract_sockets = []
    # Abstract sockets show as sun_path=@... or sun_path="\0..."
    pattern = r'sun_path=[@\\0]([^"}\s]+)'
    for entry in matches:
        m = re.search(pattern, entry["raw"])
        if m:
            abstract_sockets.append(m.group(1))
    return abstract_sockets


def diff_configs(configs):
    """Compare gate-related syscalls across configurations.

    Returns a dict of {gate_name: {config_name: count}}.
    """
    diff = {}
    for gate_name in GATE_PATTERNS:
        diff[gate_name] = {}
        for config_name, (gates, _) in configs.items():
            diff[gate_name][config_name] = len(gates.get(gate_name, []))
    return diff


def print_summary(configs, diff):
    """Print a formatted summary table."""
    config_names = list(configs.keys())

    # Header
    header = f"{'Gate':<20}"
    for name in config_names:
        header += f" {name:>12}"
    print("\n" + "=" * (20 + 13 * len(config_names)))
    print("NCCL Gate Syscall Analysis")
    print("=" * (20 + 13 * len(config_names)))
    print(header)
    print("-" * (20 + 13 * len(config_names)))

    # Gate rows
    for gate_name in GATE_PATTERNS:
        row = f"{gate_name:<20}"
        for name in config_names:
            count = diff[gate_name].get(name, 0)
            row += f" {count:>12}"
        print(row)

    print("-" * (20 + 13 * len(config_names)))

    # Detail: abstract sockets
    print("\nAbstract Unix Sockets Detected:")
    for config_name, (gates, all_matches) in configs.items():
        ipc_matches = gates.get("ipc_socket", [])
        abstract = extract_abstract_sockets(ipc_matches)
        if abstract:
            print(f"  {config_name}: {', '.join(set(abstract))}")
        else:
            print(f"  {config_name}: (none)")

    # Detail: gethostname return values
    print("\ngethostname() Return Values:")
    for config_name, (gates, _) in configs.items():
        hostnames = []
        for entry in gates.get("hostHash", []):
            if "gethostname" in entry["label"]:
                # Extract hostname from strace output like: gethostname("hostname", 256) = 0
                m = re.search(r'gethostname\("([^"]*)"', entry["raw"])
                if m:
                    hostnames.append(m.group(1))
        if hostnames:
            print(f"  {config_name}: {', '.join(set(hostnames))}")
        else:
            print(f"  {config_name}: (not captured or failed)")

    # Detail: stat /dev/shm st_dev values
    print("\nstat(\"/dev/shm\") st_dev Values:")
    for config_name, (gates, _) in configs.items():
        devs = []
        for entry in gates.get("shmDev", []):
            # Extract st_dev from strace: stat("/dev/shm", {st_mode=..., st_dev=makedev(0, ...
            m = re.search(r'st_dev=makedev\(([^)]+)\)', entry["raw"])
            if m:
                devs.append(m.group(1))
        if devs:
            print(f"  {config_name}: {', '.join(set(devs))}")
        else:
            print(f"  {config_name}: (not captured)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze strace output for NCCL gate-related syscalls"
    )
    parser.add_argument(
        "trace_dir", nargs="?",
        help="Directory containing strace files (named <config>_rank<N>.strace)"
    )
    parser.add_argument(
        "--config", action="append", metavar="NAME:PATH",
        help="Explicit config:path pair (can repeat)"
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Which rank's traces to analyze (default: 0)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # Collect trace files
    trace_files = {}

    if args.config:
        for spec in args.config:
            name, path = spec.split(":", 1)
            trace_files[name] = path
    elif args.trace_dir:
        trace_dir = Path(args.trace_dir)
        for f in sorted(trace_dir.glob(f"*_rank{args.rank}.strace")):
            config_name = f.stem.replace(f"_rank{args.rank}", "")
            trace_files[config_name] = str(f)
    else:
        parser.error("Provide either a trace directory or --config arguments")

    if not trace_files:
        print("No trace files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(trace_files)} trace file(s):")
    for name, path in trace_files.items():
        print(f"  {name}: {path}")

    # Parse all trace files
    configs = {}
    for config_name, filepath in trace_files.items():
        gates, all_matches = parse_strace_file(filepath)
        configs[config_name] = (gates, all_matches)

    # Diff and report
    diff = diff_configs(configs)
    print_summary(configs, diff)

    # Identify potential unknown gates
    if len(configs) >= 2 and "baseline" in configs and "isolated" in configs:
        baseline_gates = configs["baseline"][0]
        isolated_gates = configs["isolated"][0]
        print("Gate Comparison (baseline vs isolated):")
        for gate_name in GATE_PATTERNS:
            b_count = len(baseline_gates.get(gate_name, []))
            i_count = len(isolated_gates.get(gate_name, []))
            status = "SAME" if b_count == i_count else "DIFFERS"
            print(f"  {gate_name}: baseline={b_count}, isolated={i_count} [{status}]")
        print()

    # Write structured output
    if args.output:
        output = {
            "configs": {},
            "diff": diff,
        }
        for config_name, (gates, all_matches) in configs.items():
            output["configs"][config_name] = {
                "gate_counts": {g: len(entries) for g, entries in gates.items()},
                "matches": [
                    {"gate": m["gate"], "label": m["label"], "line": m["line_num"]}
                    for m in all_matches
                ],
                "abstract_sockets": extract_abstract_sockets(
                    gates.get("ipc_socket", [])
                ),
            }

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Analysis saved to {args.output}")


if __name__ == "__main__":
    main()
