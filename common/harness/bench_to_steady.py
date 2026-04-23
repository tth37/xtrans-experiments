#!/usr/bin/env python3
"""
bench_to_steady.py — Run vllm bench serve in a loop until TPOT plateaus.

Problem this solves:
    The original Exp A3 `cycle` harness runs a SINGLE bench at each DP size.
    That single bench is at bench-position 1 after scale-up (or cold start),
    which is 20–30% above steady-state TPOT on the warm-up curve. The
    offset-to-plateau depends on pre-run host state, so cross-regime
    comparisons from single-shot readings have huge latent variance.
    Exp C (container-overhead analysis) showed that the "−24% MGC vs native"
    gap reported in §5.7 was almost entirely this methodology artefact.

What this does:
    Runs `vllm bench serve` repeatedly at a fixed shape. After each bench,
    checks whether the last WINDOW consecutive TPOT readings are within
    EPS_PCT% of each other (max-min range). When they are, declares
    convergence, runs EXTRA_SAMPLES more benches for statistical power,
    and stops. Reports plateau mean/σ/n plus the full per-bench
    trajectory.

Non-convergence handling:
    If MAX_BENCHES is reached without convergence, reports the last WINDOW
    readings as best estimate and flags `converged: false` in the summary.
    Caller should treat such readings as unreliable.

Output:
    Under OUT_DIR:
        bench_<label>_b1.json, bench_<label>_b1.log
        bench_<label>_b2.json, bench_<label>_b2.log
        ...
        bench_steady_<label>.json   ← summary with plateau metrics

Usage:
    python bench_to_steady.py \\
        --label dp4_post_up \\
        --out-dir /path/to/results/native \\
        --host localhost --port 8000 \\
        --served-model-name qwen3-30b-a3b \\
        --tokenizer-path /data/models--Qwen--Qwen3-30B-A3B/snapshots/abc... \\
        --num-prompts 32 --max-concurrency 16 \\
        --random-input-len 128 --random-output-len 128
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def log(msg):
    print(f"[bench_to_steady] {msg}", file=sys.stderr, flush=True)


def run_one_bench(args, label_i, out_dir):
    """Invoke `vllm bench serve` once. Returns path to result JSON."""
    result_filename = f"bench_{label_i}.json"
    log_filename = f"bench_{label_i}.log"
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", args.served_model_name,
        "--tokenizer", args.tokenizer_path,
        "--host", args.host,
        "--port", str(args.port),
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(args.random_input_len),
        "--random-output-len", str(args.random_output_len),
        "--num-prompts", str(args.num_prompts),
        "--max-concurrency", str(args.max_concurrency),
        "--seed", "0",
        "--save-result",
        "--result-dir", str(out_dir),
        "--result-filename", result_filename,
    ]
    log_path = out_dir / log_filename
    with log_path.open("w") as logf:
        try:
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
            log(f"  !! vllm bench serve exited non-zero ({e.returncode}); tail of {log_filename}:")
            try:
                for line in log_path.read_text().splitlines()[-10:]:
                    log(f"    {line}")
            except Exception:
                pass
            raise
    return out_dir / result_filename


def parse_bench_json(json_path):
    """Return (tok_s, ttft_ms, tpot_ms, completed, num_prompts) from bench JSON."""
    with json_path.open() as f:
        d = json.load(f)
    return (
        float(d.get("output_throughput", 0)),
        float(d.get("mean_ttft_ms", 0)),
        float(d.get("mean_tpot_ms", 0)),
        int(d.get("completed", 0)),
        int(d.get("num_prompts", 0)),
    )


def check_converged(window, eps_pct):
    """Return (converged: bool, range_pct: float) for the window."""
    if not window:
        return False, float("inf")
    mn, mx = min(window), max(window)
    mean = sum(window) / len(window)
    if mean <= 0:
        return False, float("inf")
    range_pct = (mx - mn) / mean * 100.0
    return range_pct <= eps_pct, range_pct


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label", required=True, help="Short identifier (e.g. dp4_post_up)")
    p.add_argument("--out-dir", required=True, type=Path)

    # Server connection
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", required=True)
    p.add_argument("--tokenizer-path", required=True)

    # Bench shape (matches Exp A3 cycle defaults)
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--random-input-len", type=int, default=128)
    p.add_argument("--random-output-len", type=int, default=128)

    # Plateau-seeking parameters
    p.add_argument("--max-benches", type=int, default=15,
                   help="Hard cap on bench count (default 15)")
    p.add_argument("--warmup-min", type=int, default=2,
                   help="Minimum benches before checking convergence (default 2)")
    p.add_argument("--window-size", type=int, default=3,
                   help="How many consecutive benches to check for stability (default 3)")
    p.add_argument("--eps-pct", type=float, default=3.0,
                   help="Max range%% in the window to declare convergence (default 3%%)")
    p.add_argument("--extra-samples", type=int, default=2,
                   help="Extra benches to run after convergence for statistical power (default 2)")

    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tpots, tok_s_list, ttfts = [], [], []
    converged = False
    plateau_idx = None  # 1-indexed bench number where the converging window starts
    benches_post_convergence = 0

    log(f"Starting plateau-seeking bench for label='{args.label}'")
    log(f"  shape: n={args.num_prompts} c={args.max_concurrency} in={args.random_input_len} out={args.random_output_len}")
    log(f"  plateau: warmup_min={args.warmup_min} window={args.window_size} eps={args.eps_pct}% max={args.max_benches} extra={args.extra_samples}")

    total_cap = args.max_benches
    i = 0
    while i < total_cap:
        i += 1
        label_i = f"{args.label}_b{i}"
        json_path = run_one_bench(args, label_i, out_dir)
        tok_s, ttft, tpot, completed, requested = parse_bench_json(json_path)
        tpots.append(tpot)
        tok_s_list.append(tok_s)
        ttfts.append(ttft)

        status_note = ""
        if completed < requested:
            status_note = f"  [WARN: only {completed}/{requested} succeeded]"

        log(f"  bench {i}: TPOT={tpot:7.2f}ms  tok/s={tok_s:7.2f}  TTFT={ttft:7.1f}ms{status_note}")

        # Convergence check
        if not converged:
            if i >= args.warmup_min and len(tpots) >= args.window_size:
                window = tpots[-args.window_size:]
                is_conv, range_pct = check_converged(window, args.eps_pct)
                log(f"    window[{i - args.window_size + 1}..{i}] range={max(window) - min(window):.2f}ms ({range_pct:.2f}% of mean)")
                if is_conv:
                    converged = True
                    plateau_idx = i - args.window_size + 1
                    log(f"    *** CONVERGED at bench {i}; plateau window starts at bench {plateau_idx} ***")
        else:
            benches_post_convergence += 1
            if benches_post_convergence >= args.extra_samples:
                log(f"    Extra samples complete; stopping after bench {i}")
                break

    # Final convergence bookkeeping
    if not converged:
        log(f"!! Did NOT converge within {args.max_benches} benches. Last window as best estimate.")
        # Use the last `window_size + extra_samples` benches as fallback plateau estimate
        fallback_n = min(args.window_size + args.extra_samples, len(tpots))
        plateau_start = len(tpots) - fallback_n + 1
    else:
        plateau_start = plateau_idx

    # Plateau stats over benches [plateau_start .. final]
    plateau_tpots = tpots[plateau_start - 1:]
    plateau_tok_s = tok_s_list[plateau_start - 1:]
    plateau_ttfts = ttfts[plateau_start - 1:]

    def stats(xs):
        if not xs:
            return 0.0, 0.0
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return m, s

    tpot_mean, tpot_std = stats(plateau_tpots)
    tok_s_mean, tok_s_std = stats(plateau_tok_s)
    ttft_mean, ttft_std = stats(plateau_ttfts)

    summary = {
        "label": args.label,
        "bench_shape": {
            "num_prompts": args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
        },
        "plateau_params": {
            "warmup_min": args.warmup_min,
            "window_size": args.window_size,
            "eps_pct": args.eps_pct,
            "max_benches": args.max_benches,
            "extra_samples": args.extra_samples,
        },
        "converged": converged,
        "plateau_start_bench": plateau_start,
        "total_benches": len(tpots),
        "plateau_n": len(plateau_tpots),
        "plateau_mean_tpot_ms": tpot_mean,
        "plateau_stdev_tpot_ms": tpot_std,
        "plateau_mean_output_throughput": tok_s_mean,
        "plateau_stdev_output_throughput": tok_s_std,
        "plateau_mean_ttft_ms": ttft_mean,
        "plateau_stdev_ttft_ms": ttft_std,
        "per_bench_tpot_ms": tpots,
        "per_bench_output_throughput": tok_s_list,
        "per_bench_ttft_ms": ttfts,
    }
    summary_path = out_dir / f"bench_steady_{args.label}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log(f"=== SUMMARY for label='{args.label}' ===")
    log(f"  converged:        {converged}")
    log(f"  total_benches:    {len(tpots)}")
    log(f"  plateau window:   bench {plateau_start}..{len(tpots)} (n={len(plateau_tpots)})")
    log(f"  plateau TPOT:     {tpot_mean:7.2f} ms (σ {tpot_std:.2f})")
    log(f"  plateau tok/s:    {tok_s_mean:7.2f}    (σ {tok_s_std:.2f})")
    log(f"  plateau TTFT:     {ttft_mean:7.1f} ms (σ {ttft_std:.1f})")
    log(f"  summary written:  {summary_path}")

    # Non-zero exit on non-convergence so callers can detect it
    sys.exit(0 if converged else 2)


if __name__ == "__main__":
    main()
