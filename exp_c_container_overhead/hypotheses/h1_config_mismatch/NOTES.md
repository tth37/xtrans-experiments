# H1: Container serving config mismatch

Goal: determine whether Native and MGC are accidentally running different vLLM,
Ray, CUDA, model, tokenizer, cache, or server settings.

Evidence to collect after H0 pairs:
- Effective vLLM `non-default args` lines from native `serve.log` and MGC
  `container.log`.
- `A3_*` environment snapshot from each Exp C pair.
- Model/tokenizer paths, served model name, all2all backend, eager mode,
  `--max-num-seqs`, `--gpu-memory-utilization`, Ray address/backend, and vLLM
  version banners.
- MoE config warnings and whether native/container resolve the same fused-MoE
  config path.

Decision rule:
- If a config/env mismatch aligns with the performance gap, fix the harness or
  make that mismatch the primary finding before testing lower-level hypotheses.
