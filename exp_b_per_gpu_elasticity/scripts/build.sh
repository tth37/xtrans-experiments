#!/usr/bin/env bash
# Build a patched vLLM image.
#
# Applies every .patch in ../patches/ to the base xtrans-vllm-ep:v0.19.0,
# tags the result with a timestamp, and writes the tag to results/LAST_BUILD.
# Build log lands in results/<tag>/build.log.
#
# Usage:
#   ./scripts/build.sh                # normal build, timestamped tag
#   ./scripts/build.sh --tag mytag    # explicit tag
#   ./scripts/build.sh --no-cache     # docker build --no-cache

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPB_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd "$EXPB_DIR/.." && pwd)

TAG=""
BUILD_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG=$2; shift 2 ;;
        --no-cache) BUILD_ARGS+=("--no-cache"); shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$TAG" ]; then
    TAG=$(date +%Y%m%d-%H%M)
fi
IMAGE="xtrans-vllm-ep-patched:${TAG}"
RESULTS_DIR="$EXPB_DIR/results/${TAG}"
mkdir -p "$RESULTS_DIR"

# Sanity check: base image must exist
if ! docker image inspect xtrans-vllm-ep:v0.19.0 > /dev/null 2>&1; then
    echo "ERROR: base image xtrans-vllm-ep:v0.19.0 not found. Build it first:" >&2
    echo "    cd $PROJECT_ROOT/exp_a3_vllm_ep && \\" >&2
    echo "        docker build -t xtrans-vllm-ep:v0.19.0 -f Dockerfile.phase2 ." >&2
    exit 1
fi

# List patches being applied, for reproducibility
patch_count=$(find "$EXPB_DIR/patches" -maxdepth 1 -name '*.patch' 2>/dev/null | wc -l)
echo "Building $IMAGE with $patch_count patch(es):"
for p in "$EXPB_DIR/patches"/*.patch; do
    [ -e "$p" ] && echo "  - $(basename "$p")"
done
if [ "$patch_count" = "0" ]; then
    echo "  (none -- this will build an image identical to the base)"
fi
echo ""

docker build "${BUILD_ARGS[@]}" \
    -t "$IMAGE" \
    -f "$EXPB_DIR/Dockerfile" \
    "$EXPB_DIR" \
    2>&1 | tee "$RESULTS_DIR/build.log"

echo "$IMAGE" > "$EXPB_DIR/results/LAST_BUILD"
echo ""
echo "Built: $IMAGE"
echo "Tag recorded: $EXPB_DIR/results/LAST_BUILD"
echo "Build log:    $RESULTS_DIR/build.log"
