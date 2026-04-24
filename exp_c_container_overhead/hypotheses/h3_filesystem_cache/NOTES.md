# H3: Filesystem/cache/path overhead

Goal: determine whether MGC pays repeated model/tokenizer/config/cache overhead
because the container sees different paths or cache state.

Evidence to collect:
- Compare native and MGC model/tokenizer paths and cache directories.
- Compare first-bench vs warm-bench TTFT and TPOT trajectories.
- Check whether fused-MoE config warnings differ between native and MGC.

Only test variants if H0 shows a repeatable gap and H1/H2 do not explain it.
