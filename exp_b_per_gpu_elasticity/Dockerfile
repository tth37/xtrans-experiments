# Exp B patched image for per-GPU container elastic scaling.
#
# Layers on top of exp_a3's xtrans-vllm-ep:v0.19.0 (= vllm/vllm-openai:v0.19.0
# + ray[default]==2.55.0) and applies every .patch in ./patches/ to the
# vllm install inside the image.
#
# All .patch files MUST be unified diffs generated with paths rooted at
# the vllm Python package, e.g.:
#     diff -u a/vllm/distributed/eplb/policy/default.py \
#             b/vllm/distributed/eplb/policy/default.py
# so that `patch -p1` from the /usr/local/lib/python3.12/dist-packages dir
# applies cleanly.
#
# Recommended patch file naming (numbered, descriptive):
#     patches/0001_eplb_scale_down_redundancy.patch
#     patches/0002_placement_group_early_exit.patch
#     patches/0003_...

FROM xtrans-vllm-ep:v0.19.0

WORKDIR /usr/local/lib/python3.12/dist-packages

COPY patches/ /tmp/exp-b-patches/

# Apply each patch in numeric order. Fail-fast if any patch fails to
# apply -- silent skips produce confusing runtime failures later.
RUN set -eu; \
    patches=$(find /tmp/exp-b-patches -maxdepth 1 -name '*.patch' 2>/dev/null | sort); \
    if [ -z "$patches" ]; then \
        echo "No patches to apply -- image will be identical to base."; \
    else \
        for p in $patches; do \
            echo "=== Applying $p ==="; \
            patch -p1 < "$p"; \
        done; \
        echo "All patches applied."; \
    fi

# Quick sanity-check the vLLM install still imports after patching.
RUN python3 -c "import vllm; print('vllm', vllm.__version__, 'imports OK')"
