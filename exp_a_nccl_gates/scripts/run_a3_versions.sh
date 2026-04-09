#!/usr/bin/env bash
# run_a3_versions.sh — Exp A, Task A3: NCCL Version Stability Test
#
# Builds images for each NCCL version and runs the shim benchmark.
#
# Usage:
#   bash scripts/run_a3_versions.sh          # Full benchmark, all versions
#   bash scripts/run_a3_versions.sh --smoke  # Quick smoke test, all versions
#   bash scripts/run_a3_versions.sh --only 2.21  # Single version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXP_DIR/.." && pwd)"
REPO="tth37/xtrans-experiments"

# --- Version matrix ---
# Format: "NCCL_LABEL:DOCKERFILE:BUILD_ARGS"
VERSIONS=(
    "2.18:Dockerfile.ngc:--build-arg NGC_TAG=23.06-py3"
    "2.19:Dockerfile.ngc:--build-arg NGC_TAG=23.10-py3"
    "2.20:Dockerfile.version:--build-arg PYTORCH_VERSION=2.4.0"
    "2.21:Dockerfile::"
)

# --- Parse arguments ---
SMOKE_TEST="false"
SKIP_BUILD="false"
ONLY_VERSION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE_TEST="true"; shift ;;
        --skip-build) SKIP_BUILD="true"; shift ;;
        --only) ONLY_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "Exp A — Task A3: NCCL Version Sweep"
echo "=============================================="
echo "Smoke test: $SMOKE_TEST"
echo "Skip build: $SKIP_BUILD"
[[ -n "$ONLY_VERSION" ]] && echo "Only version: $ONLY_VERSION"
echo ""

mkdir -p "$EXP_DIR/results"

# --- Build and run each version ---
FAILED=()
for entry in "${VERSIONS[@]}"; do
    IFS=':' read -r label dockerfile build_args <<< "$entry"

    # Skip if --only specified and doesn't match
    if [[ -n "$ONLY_VERSION" && "$label" != "$ONLY_VERSION" ]]; then
        continue
    fi

    tag="${REPO}:expa-nccl${label}"
    # The existing image for 2.21
    if [[ "$dockerfile" == "Dockerfile" && -z "$build_args" ]]; then
        tag="${REPO}:expa-benchmark"
    fi

    echo "=== NCCL $label (image: $tag) ==="

    # Build image
    if [[ "$SKIP_BUILD" != "true" ]]; then
        echo "  Building from $dockerfile..."
        cd "$PROJECT_ROOT"
        if ! docker build -t "$tag" \
            $build_args \
            -f "exp_a_nccl_gates/$dockerfile" . 2>&1 | tail -3; then
            echo "  BUILD FAILED for NCCL $label"
            FAILED+=("$label")
            continue
        fi
        cd "$EXP_DIR"
    fi

    # Run benchmark
    echo "  Running benchmark..."
    docker compose -f compose.version.yml down --remove-orphans 2>/dev/null || true

    if EXPA_IMAGE="$tag" NCCL_LABEL="$label" SMOKE_TEST="$SMOKE_TEST" \
        docker compose -f compose.version.yml up \
        --abort-on-container-exit --force-recreate 2>&1 | tail -3; then
        echo "  NCCL $label: completed"
    else
        echo "  NCCL $label: FAILED"
        FAILED+=("$label")
    fi

    # Save NCCL logs
    EXPA_IMAGE="$tag" NCCL_LABEL="$label" \
        docker compose -f compose.version.yml logs \
        > "results/version_${label}_nccl.log" 2>&1 || true

    docker compose -f compose.version.yml down --remove-orphans 2>/dev/null || true
    echo ""
done

# --- Results ---
echo "=== Version Matrix ==="
RESULT_FILES=()
for entry in "${VERSIONS[@]}"; do
    IFS=':' read -r label _ _ <<< "$entry"
    f="$EXP_DIR/results/version_${label}.json"
    if [[ -f "$f" ]]; then
        RESULT_FILES+=("$f")
    fi
done

if [[ ${#RESULT_FILES[@]} -gt 0 ]]; then
    python3 "$SCRIPT_DIR/version_matrix.py" "${RESULT_FILES[@]}"
else
    echo "No result files found."
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: Failed versions: ${FAILED[*]}"
fi
echo "=============================================="
