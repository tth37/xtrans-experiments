# H5: Bench variance / host-load interference (baseline noise calibration)

**Status:** Open, MUST-verify first.

## Hypothesis

node192 is a shared host. Other users (confirmed during the 2026-04-22
sessions) spin up loads on various GPUs at unpredictable times. The
bench numbers might be partially (or mostly) explained by host-load
variance rather than in-container overhead. If the gap varies by ±15%
run-to-run on the native regime alone, the observed −24% gap between
regimes is within noise.

## Planned measurements

1. **Native regime, 3 consecutive runs, same session, no other
   containers or users active.** Record DP=4 output tok/s each time.
   Compute mean and stddev.

2. **Multi-GPU container regime, 3 consecutive runs, same conditions.**
   Same stats.

3. **Criterion:** if native stddev is comparable to the inter-regime
   gap, H5 explains most of the observation and the other hypotheses
   are investigating noise.

## Expected outcomes

- Typical case: native stddev is small (few percent), container stddev
  similar. Gap is real. H5 reports "baseline variance is ~Nx%, gap is
  real within that noise floor" and moves on to H1-H4.
- Surprising case: native itself is noisy. H5 becomes the primary
  explanation and the rest of the investigation tightens its
  methodology (more runs, longer benches, off-hours testing, etc.)
  before drawing any gap-causation conclusions.

## Status notes

(empty — run before H1-H4 to avoid chasing noise)
