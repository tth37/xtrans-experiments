# H3: `--shm-size` sensitivity

**Status:** Open, not yet measured.

## Hypothesis

The multi-GPU-container currently runs with `--shm-size=16g`. Docker's
default is 64 MB. vLLM uses shared memory for several things (Ray
object store, potentially cuMem IPC if it falls back, sometimes CUDA
buffers on some drivers). If the vllm/Ray path is shm-bound and
16 GB is *still* a cliff relative to what native has access to
(which is the host's whole /dev/shm, typically tens of GB), shm
contention could manifest as higher tail latency.

## Planned measurements

1. **Vary `--shm-size`**: 64M (docker default), 2g, 16g (current),
   64g. Run standard bench at DP=4 for each. If throughput curves
   with shm-size, it's binding.

2. **Check `df -h /dev/shm` from inside container** at each setting
   and observe whether vLLM + Ray approach the cap during steady-state
   (watch `du -sh /dev/shm` inside container during bench, or use
   `docker stats`).

3. **Spill-to-disk telemetry from Ray.** Ray's object store can spill
   to disk when shm fills. Check
   `docker exec <ctn> ls /tmp/ray/session_latest/*spill*` for spill
   evidence during a run.

## Expected outcomes

- If H3 dominates: throughput rises monotonically with shm-size up to
  some knee, then plateaus. The default 16 GB is below the knee.
- If H3 is flat: shm-size has no effect on these numbers. Default is
  fine.

## Status notes

(empty)
