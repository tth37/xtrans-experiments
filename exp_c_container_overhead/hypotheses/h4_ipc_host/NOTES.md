# H4: `--ipc=host` vs alternatives

**Status:** Open, not yet measured.

## Hypothesis

The container currently runs with `--ipc=host`, which shares the
host's IPC namespace (System V IPC, POSIX shm). This is commonly
recommended for CUDA apps because CUDA IPC uses shared-memory
primitives. But it's a mixed-bag: sharing the host's IPC can
accidentally help OR hurt depending on what else is on the host
(other users' CUDA processes leak into the same namespace).

## Planned measurements

1. **Run without `--ipc=host`.** Drop the flag; container gets its
   own IPC namespace. Measure DP=4 bench. If significantly worse,
   confirms ipc=host is contributing positively (but doesn't pin
   down how much of the gap is attributable to it).

2. **Run with `--ipc=private` explicitly.** Same as above but
   documented explicitly in Dockerfile.

3. **Run with `--ipc=container:<other>`** — not useful here, skip.

4. **Cross with H3 shm-size.** shm-size and ipc=host both affect
   shared memory paths; confirm their effects are independent or
   interacting.

## Expected outcomes

- Drop ipc=host → likely throughput regression, confirms current flag
  is load-bearing.
- If even with ipc=host we see a ~24% gap, the flag helps but
  doesn't close the gap. Other hypotheses carry the remainder.

## Status notes

(empty)
