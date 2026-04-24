# H4: Container runtime flags

Goal: determine whether Docker runtime flags cause the MGC gap.

Variants, only after H0 reproduces a real gap:
- `--ipc=host` vs private IPC.
- `--shm-size` variants.
- Docker network mode variants.
- Cache/model mount variants.

Change one flag at a time and compare against the same Native baseline window.
