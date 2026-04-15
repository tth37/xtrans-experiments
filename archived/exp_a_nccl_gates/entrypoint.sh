#!/bin/bash
# entrypoint.sh — Container entrypoint with optional strace wrapping.
#
# When STRACE_WRAP=1, wraps the command in strace for syscall tracing (A1).
# Otherwise, exec's the command directly (A2 benchmarks).
#
# Trace output goes to /results/traces/${DEPLOYMENT}_rank${RANK}.strace

set -euo pipefail

if [ "${STRACE_WRAP:-0}" = "1" ]; then
    trace_dir="/results/traces"
    mkdir -p "$trace_dir"
    trace_file="${trace_dir}/${DEPLOYMENT:-unknown}_rank${RANK:-0}.strace"
    echo "[entrypoint] strace enabled, writing to $trace_file"
    exec strace -f -e trace=network,ipc,file \
        -o "$trace_file" \
        "$@"
else
    exec "$@"
fi
