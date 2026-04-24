#!/usr/bin/env python3
"""Exp C harness for Native-vs-MGC container overhead redo.

Runs paired DP=4 ShareGPT benches through the Exp A3 Python harness, captures
snapshots, and summarizes whether Native-vs-MGC gaps survive warmup filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_C = PROJECT_ROOT / "exp_c_container_overhead"
EXP_A3 = PROJECT_ROOT / "exp_a3_vllm_ep"
RESULTS_ROOT = EXP_C / "results" / "sharegpt_dp4"
SHAREGPT_PATH = EXP_A3 / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json"
LABEL = "dp4_direct_sharegpt_c32"

REGIMES = {
    "native": EXP_A3 / "1_native.py",
    "mgc": EXP_A3 / "2_multi_gpu_container.py",
}
A3_RESULT_DIRS = {
    "native": EXP_A3 / "results" / "native",
    "mgc": EXP_A3 / "results" / "multi_gpu_container",
}
SERVE_LOG_NAMES = {
    "native": ["serve.log", "ray_head.log"],
    "mgc": ["container.log"],
}


@dataclass
class Summary:
    regime: str
    pair: int
    converged: bool
    stable_n: int
    throughput: float
    throughput_std: float
    ttft: float
    ttft_std: float
    tpot: float
    tpot_std: float
    completed_ok: bool
    external_gpu_processes: bool


def log(message: str) -> None:
    print(f"[exp-c] {message}", flush=True)


def run(cmd: list[str], *, cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None,
        stdout=None, stderr=None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, stdout=stdout, stderr=stderr, text=True, check=check)


def output(cmd: list[str], *, cwd: Path = PROJECT_ROOT, check: bool = False) -> str:
    result = run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return result.stdout or ""


def bench_env(pair: int, regime: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "A3_SINGLE_LABEL": f"{LABEL}_p{pair}_{regime}",
        "A3_SINGLE_NUM_PROMPTS": "96",
        "A3_SINGLE_CONCURRENCY": "32",
        "A3_BENCH_DATASET_PATH": str(SHAREGPT_PATH),
        "A3_BENCH_NUM_PROMPTS": "96",
        "A3_BENCH_MAX_CONCURRENCY": "32",
        "A3_BENCH_MAX_BENCHES": os.environ.get("EXP_C_MAX_BENCHES", "4"),
        "A3_BENCH_EXTRA_SAMPLES": os.environ.get("EXP_C_EXTRA_SAMPLES", "1"),
        "A3_BENCH_SEED": os.environ.get("EXP_C_SEED", "0"),
        "A3_BENCH_DISCARD_FIRST": os.environ.get("EXP_C_DISCARD_FIRST", "1"),
        "A3_MAX_NUM_SEQS": "32",
    })
    return env


def gpu_process_snapshot() -> str:
    query = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    return output(query).strip()


def gpu_snapshot() -> str:
    return output(["nvidia-smi"]).strip()


def gpu_process_pids() -> list[int]:
    text = gpu_process_snapshot()
    pids: list[int] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                pids.append(int(parts[1]))
            except ValueError:
                pass
    return pids


def wait_for_idle(timeout_s: int, poll_s: int) -> None:
    start = time.time()
    while True:
        pids = gpu_process_pids()
        if not pids:
            return
        elapsed = int(time.time() - start)
        if elapsed >= timeout_s:
            raise SystemExit(f"GPUs still busy after {timeout_s}s; pids={pids}")
        log(f"GPUs busy with pids={pids}; waiting {poll_s}s")
        time.sleep(poll_s)


def write_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## nvidia-smi ##",
        gpu_snapshot(),
        "",
        "## compute processes ##",
        gpu_process_snapshot() or "(none)",
        "",
        "## docker ps ##",
        output(["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Image}}"]),
        "",
        "## host process sample ##",
        output(["bash", "-lc", "ps -eo pid,user,etime,cmd | rg 'vllm|ray|python3 exp_a3|docker run|nvidia-smi' | rg -v rg || true"]),
    ]
    path.write_text("\n".join(lines) + "\n")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_regime(pair: int, regime: str, idle_timeout_s: int, idle_poll_s: int) -> None:
    script = REGIMES[regime]
    run_dir = RESULTS_ROOT / f"pair_{pair:02d}" / regime
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"pair {pair} {regime}: waiting for idle GPUs")
    wait_for_idle(idle_timeout_s, idle_poll_s)
    write_snapshot(run_dir / "pre_snapshot.txt")

    env = bench_env(pair, regime)
    (run_dir / "env.json").write_text(json.dumps({k: env[k] for k in sorted(env) if k.startswith(("A3_", "EXP_C_"))}, indent=2) + "\n")

    log(f"pair {pair} {regime}: running {script.name} bench")
    start = time.time()
    with (run_dir / "harness.log").open("w") as log_file:
        result = run(["python3", str(script), "bench"], env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - start
    (run_dir / "exit.json").write_text(json.dumps({"returncode": result.returncode, "elapsed_s": elapsed}, indent=2) + "\n")
    write_snapshot(run_dir / "post_snapshot.txt")

    a3_dir = A3_RESULT_DIRS[regime]
    label = env["A3_SINGLE_LABEL"]
    for path in a3_dir.glob(f"bench_{label}*.*"):
        copy_if_exists(path, run_dir / path.name)
    for name in SERVE_LOG_NAMES[regime]:
        copy_if_exists(a3_dir / name, run_dir / name)

    if result.returncode != 0:
        raise SystemExit(f"{regime} pair {pair} failed; see {run_dir / 'harness.log'}")


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def summary_for(pair: int, regime: str) -> Summary:
    run_dir = RESULTS_ROOT / f"pair_{pair:02d}" / regime
    path = run_dir / f"bench_{LABEL}_p{pair}_{regime}.json"
    data = load_summary(path)
    completed_ok = True
    for bench_file in sorted(run_dir.glob(f"bench_{LABEL}_p{pair}_{regime}_b*.json")):
        bench_data = load_summary(bench_file)
        completed_ok = completed_ok and bench_data.get("completed") == bench_data.get("num_prompts")
    pre = (run_dir / "pre_snapshot.txt").read_text(errors="replace") if (run_dir / "pre_snapshot.txt").exists() else ""
    post = (run_dir / "post_snapshot.txt").read_text(errors="replace") if (run_dir / "post_snapshot.txt").exists() else ""
    external = "## compute processes ##\n(none)" not in pre or "## compute processes ##\n(none)" not in post
    return Summary(
        regime=regime,
        pair=pair,
        converged=bool(data.get("converged")),
        stable_n=int(data.get("stable_n", 0)),
        throughput=float(data.get("stable_mean_output_throughput", 0.0)),
        throughput_std=float(data.get("stable_stdev_output_throughput", 0.0)),
        ttft=float(data.get("stable_mean_ttft_ms", 0.0)),
        ttft_std=float(data.get("stable_stdev_ttft_ms", 0.0)),
        tpot=float(data.get("stable_mean_tpot_ms", 0.0)),
        tpot_std=float(data.get("stable_stdev_tpot_ms", 0.0)),
        completed_ok=completed_ok,
        external_gpu_processes=external,
    )


def available_pairs() -> list[int]:
    pairs: list[int] = []
    for pair_dir in sorted(RESULTS_ROOT.glob("pair_*")):
        try:
            pair = int(pair_dir.name.split("_")[1])
        except Exception:
            continue
        if all((pair_dir / regime / f"bench_{LABEL}_p{pair}_{regime}.json").exists() for regime in REGIMES):
            pairs.append(pair)
    return pairs


def render_markdown(summaries: list[Summary]) -> str:
    lines = [
        "| Pair | Regime | Throughput tok/s | TTFT ms | TPOT ms | Converged | Complete | Contamination |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.pair} | {s.regime} | {s.throughput:.2f} ± {s.throughput_std:.2f} | "
            f"{s.ttft:.1f} ± {s.ttft_std:.1f} | {s.tpot:.2f} ± {s.tpot_std:.2f} | "
            f"{s.converged} | {s.completed_ok} | {s.external_gpu_processes} |"
        )
    return "\n".join(lines) + "\n"


def analyze() -> None:
    pairs = available_pairs()
    if not pairs:
        raise SystemExit(f"No complete pairs under {RESULTS_ROOT}")
    summaries = [summary_for(pair, regime) for pair in pairs for regime in REGIMES]
    print(render_markdown(summaries))

    native = [s for s in summaries if s.regime == "native"]
    mgc = [s for s in summaries if s.regime == "mgc"]
    pair_gaps = []
    for pair in pairs:
        n = next(s for s in native if s.pair == pair)
        m = next(s for s in mgc if s.pair == pair)
        pair_gaps.append({
            "pair": pair,
            "throughput_gap_pct": (m.throughput - n.throughput) / n.throughput * 100.0,
            "tpot_gap_pct": (m.tpot - n.tpot) / n.tpot * 100.0,
            "ttft_gap_pct": (m.ttft - n.ttft) / n.ttft * 100.0,
            "combined_tpot_sigma_ms": (n.tpot_std ** 2 + m.tpot_std ** 2) ** 0.5,
            "tpot_gap_ms": m.tpot - n.tpot,
        })
    aggregate = {
        "pairs": pairs,
        "native_mean_throughput": statistics.mean(s.throughput for s in native),
        "mgc_mean_throughput": statistics.mean(s.throughput for s in mgc),
        "native_mean_tpot_ms": statistics.mean(s.tpot for s in native),
        "mgc_mean_tpot_ms": statistics.mean(s.tpot for s in mgc),
        "native_mean_ttft_ms": statistics.mean(s.ttft for s in native),
        "mgc_mean_ttft_ms": statistics.mean(s.ttft for s in mgc),
        "pair_gaps": pair_gaps,
    }
    aggregate["mean_throughput_gap_pct"] = (aggregate["mgc_mean_throughput"] - aggregate["native_mean_throughput"]) / aggregate["native_mean_throughput"] * 100.0
    aggregate["mean_tpot_gap_pct"] = (aggregate["mgc_mean_tpot_ms"] - aggregate["native_mean_tpot_ms"]) / aggregate["native_mean_tpot_ms"] * 100.0
    aggregate["mean_ttft_gap_pct"] = (aggregate["mgc_mean_ttft_ms"] - aggregate["native_mean_ttft_ms"]) / aggregate["native_mean_ttft_ms"] * 100.0
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (RESULTS_ROOT / "summary.md").write_text(render_markdown(summaries) + "\n```json\n" + json.dumps(aggregate, indent=2) + "\n```\n")
    print(json.dumps(aggregate, indent=2))


def run_pairs(args: argparse.Namespace) -> None:
    if not SHAREGPT_PATH.exists():
        raise SystemExit(f"ShareGPT dataset not found at {SHAREGPT_PATH}")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for pair in range(args.start_pair, args.start_pair + args.pairs):
        for regime in ("native", "mgc"):
            run_regime(pair, regime, args.idle_timeout_s, args.idle_poll_s)
    analyze()


def run_fast(args: argparse.Namespace) -> None:
    """Run faster H0 probes: one ShareGPT bench per cold-started regime."""
    if not SHAREGPT_PATH.exists():
        raise SystemExit(f"ShareGPT dataset not found at {SHAREGPT_PATH}")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for pair in range(args.start_pair, args.start_pair + args.pairs):
        for regime in ("native", "mgc"):
            env = os.environ.copy()
            env["EXP_C_MAX_BENCHES"] = "1"
            env["EXP_C_EXTRA_SAMPLES"] = "0"
            old = os.environ.copy()
            os.environ.update(env)
            try:
                run_regime(pair, regime, args.idle_timeout_s, args.idle_poll_s)
            finally:
                os.environ.clear()
                os.environ.update(old)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run-pairs", help="run paired native/MGC DP=4 ShareGPT benches")
    run_p.add_argument("--pairs", type=int, default=3)
    run_p.add_argument("--start-pair", type=int, default=1)
    run_p.add_argument("--idle-timeout-s", type=int, default=21600)
    run_p.add_argument("--idle-poll-s", type=int, default=300)
    fast_p = sub.add_parser("run-fast", help="run paired native/MGC cold single-bench ShareGPT probes")
    fast_p.add_argument("--pairs", type=int, default=3)
    fast_p.add_argument("--start-pair", type=int, default=1)
    fast_p.add_argument("--idle-timeout-s", type=int, default=21600)
    fast_p.add_argument("--idle-poll-s", type=int, default=300)
    sub.add_parser("analyze", help="summarize completed pair results")
    args = parser.parse_args(argv)
    if args.cmd == "run-pairs":
        run_pairs(args)
    elif args.cmd == "run-fast":
        run_fast(args)
    elif args.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main(sys.argv[1:])
