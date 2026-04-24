#!/usr/bin/env python3
"""Shared Python harness for Exp A3 vLLM Elastic EP regimes."""

from __future__ import annotations

import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "exp_a3_vllm_ep"
VENV_DIR = Path(os.environ.get("VENV_DIR", PROJECT_ROOT / ".venv"))

MODEL_HOST = os.environ.get("MODEL_HOST", "/data/models--Qwen--Qwen3-30B-A3B")
MODEL_SNAPSHOT = os.environ.get(
    "MODEL_SNAPSHOT",
    f"{MODEL_HOST}/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
)
MODEL_PATH_IN_CTN = os.environ.get(
    "MODEL_PATH_IN_CTN",
    "/models/qwen3-30b-a3b/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
)
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen3-30b-a3b")
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "xtrans-vllm-ep:v0.19.0")
VLLM_IMAGE_PATCHED = os.environ.get("VLLM_IMAGE_PATCHED", "xtrans-vllm-ep-patched:v0.19.0")
RAY_PORT = int(os.environ.get("RAY_PORT", "26379"))
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))

LivenessCheck = Callable[[], bool]
DiagCallback = Callable[[], None]


@dataclass(frozen=True)
class BenchConfig:
    dataset_name: str
    dataset_path: str | None
    num_prompts: int
    max_concurrency: int | None
    random_input_len: int
    random_output_len: int
    request_rate: str | None
    seed: int
    max_benches: int
    warmup_min: int
    discard_first: int
    window_size: int
    eps_pct: float
    extra_samples: int

    @classmethod
    def for_cycle(cls, num_prompts: int, max_concurrency: int) -> "BenchConfig":
        request_rate = os.environ.get("A3_BENCH_REQUEST_RATE")
        concurrency_env = os.environ.get("A3_BENCH_MAX_CONCURRENCY")
        return cls(
            dataset_name=os.environ.get("A3_BENCH_DATASET", "random"),
            dataset_path=os.environ.get("A3_BENCH_DATASET_PATH") or None,
            num_prompts=int(os.environ.get("A3_BENCH_NUM_PROMPTS", str(num_prompts))),
            max_concurrency=(
                int(concurrency_env) if concurrency_env else max_concurrency
            ),
            random_input_len=int(os.environ.get("A3_BENCH_RANDOM_INPUT_LEN", "128")),
            random_output_len=int(os.environ.get("A3_BENCH_RANDOM_OUTPUT_LEN", "128")),
            request_rate=request_rate,
            seed=int(os.environ.get("A3_BENCH_SEED", "0")),
            max_benches=int(os.environ.get("A3_BENCH_MAX_BENCHES", "8")),
            warmup_min=int(os.environ.get("A3_BENCH_WARMUP_MIN", "2")),
            discard_first=int(os.environ.get("A3_BENCH_DISCARD_FIRST", "1")),
            window_size=int(os.environ.get("A3_BENCH_WINDOW_SIZE", "3")),
            eps_pct=float(os.environ.get("A3_BENCH_EPS_PCT", "5")),
            extra_samples=int(os.environ.get("A3_BENCH_EXTRA_SAMPLES", "1")),
        )


def log(message: str) -> None:
    print(f"[a3] {message}", file=sys.stderr, flush=True)


def run(
    cmd: list[str] | str,
    *,
    check: bool = True,
    capture: bool = False,
    text: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout=None,
    stderr=None,
    shell: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=text,
        cwd=str(cwd or PROJECT_ROOT),
        env=env,
        stdout=stdout,
        stderr=stderr,
        shell=shell,
        timeout=timeout,
    )


def output(cmd: list[str] | str, *, check: bool = False, shell: bool = False) -> str:
    result = run(cmd, check=check, capture=True, shell=shell)
    return (result.stdout or "").strip()


def ensure_venv() -> None:
    activate = VENV_DIR / "bin" / "activate"
    if not activate.exists():
        raise SystemExit(f"venv not found at {VENV_DIR}; create it first")
    os.environ["PATH"] = f"{VENV_DIR / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ.setdefault("VIRTUAL_ENV", str(VENV_DIR))


def http_get_ok(url: str, timeout: int = 2) -> bool:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def http_post_json(base_url: str, path: str, payload: dict, timeout: int = 900) -> tuple[int | None, str]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return None, str(exc)


def post_text(base_url: str, path: str = "/is_scaling_elastic_ep") -> str:
    code, body = http_post_json(base_url, path, {}, timeout=10)
    if code is None:
        return f"(unreachable: {body})"
    return body or f"HTTP {code}"


def wait_for_ready(
    url: str,
    timeout_s: int = 600,
    liveness_check: LivenessCheck | None = None,
    diag_on_fail: DiagCallback | None = None,
) -> None:
    start = time.time()
    last_progress = 0.0
    log(f"Polling {url} (timeout {timeout_s}s)")
    while True:
        if http_get_ok(url):
            elapsed = int(time.time() - start)
            log(f"READY after {elapsed}s")
            return
        if liveness_check is not None and not liveness_check():
            elapsed = int(time.time() - start)
            log(f"ABORT: server process dead after {elapsed}s (liveness check failed)")
            if diag_on_fail is not None:
                diag_on_fail()
            raise SystemExit(2)
        elapsed = time.time() - start
        if elapsed >= timeout_s:
            log(f"TIMEOUT after {timeout_s}s")
            if diag_on_fail is not None:
                diag_on_fail()
            raise SystemExit(1)
        if elapsed - last_progress >= 30:
            log(f"  ...still waiting ({int(elapsed)}s elapsed)")
            last_progress = elapsed
        time.sleep(5)


def gpu_snapshot() -> str:
    return output([
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ])


def gpu_processes() -> str:
    return "\n".join(output([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader",
    ]).splitlines()[:20])


def all_gpus_free() -> bool:
    used = output([
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ])
    if not used:
        return False
    try:
        return all(int(line.strip()) < 100 for line in used.splitlines() if line.strip())
    except ValueError:
        return False


def require_gpus_free() -> None:
    if all_gpus_free():
        return
    log("Some GPUs are busy:")
    for line in gpu_snapshot().splitlines():
        print(f"    {line}", file=sys.stderr)
    if os.environ.get("ALLOW_BUSY_GPUS", "0") == "1":
        log("ALLOW_BUSY_GPUS=1 set, continuing anyway")
        return
    raise SystemExit("Aborting. Set ALLOW_BUSY_GPUS=1 to override (at your own risk).")


def tmux_kill(name: str) -> None:
    run(["tmux", "kill-session", "-t", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tmux_oneshot(name: str, command: str) -> None:
    tmux_kill(name)
    run(["tmux", "new-session", "-d", "-s", name, "-c", str(PROJECT_ROOT), "bash", "-lc", command])


def docker_ensure_image(image: str) -> None:
    if run(["docker", "image", "inspect", image], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise SystemExit(f"ERROR: image {image} not found locally")


def ensure_patched_image() -> None:
    if run(["docker", "image", "inspect", VLLM_IMAGE_PATCHED], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    base = "xtrans-vllm-ep:v0.19.0"
    if run(["docker", "image", "inspect", base], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise SystemExit(
            "ERROR: base image xtrans-vllm-ep:v0.19.0 not found; build it first:\n"
            f"    docker build -t {base} -f {EXP_DIR / 'Dockerfile.base'} {EXP_DIR}"
        )
    log(f"Patched image {VLLM_IMAGE_PATCHED} not found; building from Dockerfile.per_gpu_containers...")
    run([
        "docker",
        "build",
        "-t",
        VLLM_IMAGE_PATCHED,
        "-f",
        str(EXP_DIR / "Dockerfile.per_gpu_containers"),
        str(EXP_DIR),
    ])
    log(f"Built {VLLM_IMAGE_PATCHED}")


def container_deviceids(name: str) -> str:
    raw = output(["docker", "inspect", name, "--format", "{{json .HostConfig.DeviceRequests}}"])
    if not raw:
        return "(inspect failed)"
    try:
        data = json.loads(raw)
        if data:
            return str(data[0].get("DeviceIDs") or [])
        return "[]"
    except Exception:
        return "(unparseable)"


def write_state(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text, end="")


def trigger_scale(base_url: str, new_dp: int, results_dir: Path, drain_timeout: int = 60) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    start = time.time_ns()
    log(f"Scale to DP={new_dp} (drain_timeout={drain_timeout}s)")
    code, body = http_post_json(
        base_url,
        "/scale_elastic_ep",
        {"new_data_parallel_size": new_dp, "drain_timeout": drain_timeout},
        timeout=900,
    )
    elapsed_ms = (time.time_ns() - start) // 1_000_000
    log(f"Scale response: HTTP {code if code is not None else '?'} in {elapsed_ms}ms")
    log_path = results_dir / f"scale_to_dp{new_dp}_{time.strftime('%H%M%S')}.log"
    log_path.write_text(f"{body}\nELAPSED_MS={elapsed_ms}\nHTTP_CODE={code or 'unknown'}\n")
    print(body)
    print(f"ELAPSED_MS={elapsed_ms}")
    print(f"HTTP_CODE={code or 'unknown'}")
    if code != 200:
        raise SystemExit(1)


def _bench_cmd(label_i: str, out_dir: Path, host: str, port: int, cfg: BenchConfig) -> list[str]:
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        SERVED_MODEL_NAME,
        "--tokenizer",
        MODEL_SNAPSHOT,
        "--host",
        host,
        "--port",
        str(port),
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        cfg.dataset_name,
        "--num-prompts",
        str(cfg.num_prompts),
        "--seed",
        str(cfg.seed),
        "--save-result",
        "--result-dir",
        str(out_dir),
        "--result-filename",
        f"bench_{label_i}.json",
    ]
    if cfg.dataset_path:
        cmd += ["--dataset-path", cfg.dataset_path]
    if cfg.dataset_name == "random":
        cmd += [
            "--random-input-len",
            str(cfg.random_input_len),
            "--random-output-len",
            str(cfg.random_output_len),
        ]
    elif "A3_BENCH_OUTPUT_LEN" in os.environ:
        cmd += ["--output-len", os.environ["A3_BENCH_OUTPUT_LEN"]]
    if cfg.request_rate:
        cmd += ["--request-rate", cfg.request_rate]
    if cfg.max_concurrency is not None:
        cmd += ["--max-concurrency", str(cfg.max_concurrency)]
    extra = os.environ.get("A3_BENCH_EXTRA_ARGS")
    if extra:
        cmd += shlex.split(extra)
    return cmd


def run_one_bench(label_i: str, out_dir: Path, host: str, port: int, cfg: BenchConfig) -> Path:
    ensure_venv()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"bench_{label_i}.log"
    cmd = _bench_cmd(label_i, out_dir, host, port, cfg)
    with log_path.open("w") as log_file:
        try:
            run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            log(f"  !! vllm bench serve exited non-zero ({exc.returncode}); tail of {log_path.name}:")
            for line in log_path.read_text(errors="replace").splitlines()[-10:]:
                log(f"    {line}")
            raise
    return out_dir / f"bench_{label_i}.json"


def parse_bench_json(json_path: Path) -> tuple[float, float, float, int, int]:
    data = json.loads(json_path.read_text())
    return (
        float(data.get("output_throughput", 0)),
        float(data.get("mean_ttft_ms", 0)),
        float(data.get("mean_tpot_ms", 0)),
        int(data.get("completed", 0)),
        int(data.get("num_prompts", 0)),
    )


def check_converged(window: list[float], eps_pct: float) -> tuple[bool, float]:
    if not window:
        return False, float("inf")
    mean = sum(window) / len(window)
    if mean <= 0:
        return False, float("inf")
    range_pct = (max(window) - min(window)) / mean * 100.0
    return range_pct <= eps_pct, range_pct


def bench(label: str, host: str, port: int, out_dir: Path, cfg: BenchConfig) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"bench_{label}.log"
    old_stderr = sys.stderr
    with log_file.open("w") as lf:
        class Tee:
            def write(self, text: str) -> int:
                old_stderr.write(text)
                lf.write(text)
                lf.flush()
                return len(text)
            def flush(self) -> None:
                old_stderr.flush()
                lf.flush()
        sys.stderr = Tee()  # type: ignore[assignment]
        try:
            return _bench(label, host, port, out_dir, cfg)
        finally:
            sys.stderr = old_stderr


def _bench(label: str, host: str, port: int, out_dir: Path, cfg: BenchConfig) -> bool:
    tpots: list[float] = []
    tok_s_list: list[float] = []
    ttfts: list[float] = []
    converged = False
    stable_idx = 1
    benches_post_convergence = 0

    log(f"bench[{label}]: dataset={cfg.dataset_name} n={cfg.num_prompts} c={cfg.max_concurrency} in={cfg.random_input_len} out={cfg.random_output_len} max={cfg.max_benches} discard_first={cfg.discard_first} eps={cfg.eps_pct}%")
    for i in range(1, cfg.max_benches + 1):
        label_i = f"{label}_b{i}"
        json_path = run_one_bench(label_i, out_dir, host, port, cfg)
        tok_s, ttft, tpot, completed, requested = parse_bench_json(json_path)
        tpots.append(tpot)
        tok_s_list.append(tok_s)
        ttfts.append(ttft)
        status_note = f"  [WARN: only {completed}/{requested} succeeded]" if completed < requested else ""
        log(f"  bench {i}: TPOT={tpot:7.2f}ms  tok/s={tok_s:7.2f}  TTFT={ttft:7.1f}ms{status_note}")
        measured_tpots = tpots[cfg.discard_first :]
        if i <= cfg.discard_first:
            log(f"    warmup/discard bench {i}; excluded from convergence and summary")
        elif not converged:
            if i >= cfg.warmup_min and len(measured_tpots) >= cfg.window_size:
                window = measured_tpots[-cfg.window_size :]
                is_conv, range_pct = check_converged(window, cfg.eps_pct)
                log(f"    window[{i - cfg.window_size + 1}..{i}] range={max(window) - min(window):.2f}ms ({range_pct:.2f}% of mean)")
                if is_conv:
                    converged = True
                    stable_idx = i - cfg.window_size + 1
                    log(f"    *** CONVERGED at bench {i}; stable window starts at bench {stable_idx} ***")
        else:
            benches_post_convergence += 1
            if benches_post_convergence >= cfg.extra_samples:
                log(f"    Extra samples complete; stopping after bench {i}")
                break

    if not converged:
        log(f"!! Did NOT converge within {cfg.max_benches} benches. Last window as best estimate.")
        measured_n = max(0, len(tpots) - cfg.discard_first)
        fallback_n = min(cfg.window_size + cfg.extra_samples, measured_n)
        stable_start = max(cfg.discard_first + 1, len(tpots) - fallback_n + 1)
    else:
        stable_start = stable_idx

    stable_tpots = tpots[stable_start - 1 :]
    stable_tok_s = tok_s_list[stable_start - 1 :]
    stable_ttfts = ttfts[stable_start - 1 :]

    def stats(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0

    tpot_mean, tpot_std = stats(stable_tpots)
    tok_s_mean, tok_s_std = stats(stable_tok_s)
    ttft_mean, ttft_std = stats(stable_ttfts)
    summary = {
        "label": label,
        "bench_shape": {
            "dataset_name": cfg.dataset_name,
            "dataset_path": cfg.dataset_path,
            "num_prompts": cfg.num_prompts,
            "max_concurrency": cfg.max_concurrency,
            "request_rate": cfg.request_rate,
            "random_input_len": cfg.random_input_len,
            "random_output_len": cfg.random_output_len,
        },
        "stable_params": {
            "warmup_min": cfg.warmup_min,
            "discard_first": cfg.discard_first,
            "window_size": cfg.window_size,
            "eps_pct": cfg.eps_pct,
            "max_benches": cfg.max_benches,
            "extra_samples": cfg.extra_samples,
        },
        "converged": converged,
        "stable_start_bench": stable_start,
        "total_benches": len(tpots),
        "stable_n": len(stable_tpots),
        "stable_mean_tpot_ms": tpot_mean,
        "stable_stdev_tpot_ms": tpot_std,
        "stable_mean_output_throughput": tok_s_mean,
        "stable_stdev_output_throughput": tok_s_std,
        "stable_mean_ttft_ms": ttft_mean,
        "stable_stdev_ttft_ms": ttft_std,
        "per_bench_tpot_ms": tpots,
        "per_bench_output_throughput": tok_s_list,
        "per_bench_ttft_ms": ttfts,
    }
    summary_path = out_dir / f"bench_{label}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    log("")
    log(f"=== SUMMARY for label='{label}' ===")
    log(f"  converged:        {converged}")
    log(f"  total_benches:    {len(tpots)}")
    log(f"  stable window:   bench {stable_start}..{len(tpots)} (n={len(stable_tpots)})")
    log(f"  stable TPOT:     {tpot_mean:7.2f} ms (σ {tpot_std:.2f})")
    log(f"  stable tok/s:    {tok_s_mean:7.2f}    (σ {tok_s_std:.2f})")
    log(f"  stable TTFT:     {ttft_mean:7.1f} ms (σ {ttft_std:.1f})")
    log(f"  summary written:  {summary_path}")
    return converged



def run_single_bench(label: str, results_dir: Path, num_prompts: int, concurrency: int) -> None:
    if not bench(label, "localhost", VLLM_PORT, results_dir, BenchConfig.for_cycle(num_prompts, concurrency)):
        log(f"WARN: {label} did not converge")

def run_bench_cycle(regime_name: str, results_dir: Path, scale_fn: Callable[[int], None], state_fn: Callable[[str], None], after_cycle: Callable[[], None] | None = None) -> None:
    label_suffix = os.environ.get("A3_BENCH_LABEL_SUFFIX", "")

    def label(base: str) -> str:
        return f"{base}_{label_suffix}" if label_suffix else base

    state_fn(label("pre_cycle"))
    dp2_initial = label("dp2_initial")
    if not bench(dp2_initial, "localhost", VLLM_PORT, results_dir, BenchConfig.for_cycle(16, 8)):
        log(f"WARN: {dp2_initial} did not converge")
    scale_fn(4)
    time.sleep(3)
    state_fn(label("post_scale_up"))
    dp4_post_up = label("dp4_post_up")
    if not bench(dp4_post_up, "localhost", VLLM_PORT, results_dir, BenchConfig.for_cycle(32, 16)):
        log(f"WARN: {dp4_post_up} did not converge")
    scale_fn(2)
    time.sleep(3)
    state_fn(label("post_scale_down"))
    dp2_post_down = label("dp2_post_down")
    if not bench(dp2_post_down, "localhost", VLLM_PORT, results_dir, BenchConfig.for_cycle(16, 8)):
        log(f"WARN: {dp2_post_down} did not converge")
    state_fn(label("post_cycle"))
    if after_cycle is not None:
        after_cycle()
    log(f"cycle complete for {regime_name}. Stable summaries in {results_dir}/bench_*.json")
