# H1: TTFT / compile / CUDA-graph cold-path cost dominates the gap

**Status:** Open, not yet measured.

## Hypothesis

The 4.6× TTFT blow-up (387 ms native → 1777 ms multi-GPU container at
DP=4) is disproportionate to the 24% output-throughput drop, suggesting
the extra wall-clock inside the container is concentrated in the
first-token path: kernel compilation, CUDA graph capture, expert
weight paging on first use, MoE config JSON loading with a
"default MoE config" warning in-container that isn't needed on native.

If most of the gap is TTFT-side, a bench with very long output
sequences (e.g. `--random-output-len 1024`) should show a much smaller
gap than the current 128-output bench.

## Planned measurements

1. **Isolate TTFT from output throughput.** Bench both regimes at
   `--random-input-len 128 --random-output-len 1024 --num-prompts 8`.
   If output tok/s gap shrinks from −24% toward the TPOT gap (+20%),
   that confirms H1.

2. **First-request vs. steady-state.** Fire a single warmup request
   before the bench to pay compile/TTFT costs once. Then run the
   standard bench. If container bench gap collapses after warmup,
   the gap is one-time setup cost.

3. **Check the "default MoE config" warning.** The container
   startup log mentions
   `Config file not found at .../fused_moe/configs/E=32,N=768,device_name=NVIDIA_A100-*.json`.
   Native log: does it emit the same warning? If the config IS found
   on native but not in-container (mount issue, path mismatch), that
   alone could explain kernel-selection differences and TTFT cost.

## Expected outcomes

- If H1 is dominant: long-output bench gap shrinks to ~−8% (matching
  TPOT gap), warmup-first bench gap collapses further, and the
  "solution" is either pre-warmup in the harness or making the MoE
  config discoverable in-container.
- If H1 is minor: long-output bench still shows ~−24% gap. Move to
  H2 / H3 / H4.

## Status notes

(empty — session hasn't started)
