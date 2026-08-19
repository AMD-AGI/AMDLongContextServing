# syntax=docker/dockerfile:1.6
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Custom vLLM-ROCm image with the public long-context-serving stack, rebased
# onto current upstream: vLLM v0.25.1 + AITER v0.1.19.post2.
# It bundles the benchmark/report code and the locked prompt substrate corpus
# used by the campaign.
#
# The long-context MLA decode and prefill kernels ship natively in the pinned
# AITER release, and vLLM v0.25.1 pads MLA query heads to 16 itself
# (AiterMLAHelper.get_mla_padded_q / _unpadded_o), so Kimi-Linear's nhead=4
# (after TP=8) works with a stock AITER. This image is vanilla vLLM v0.25.1 plus
# upstream AITER, with the long-context-serving app and substrate baked in.
#
# Published campaign numbers were measured on an earlier AITER build of the
# decode kernel, so treat a sweep on a different pin as needing revalidation
# rather than assuming parity.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai-rocm:v0.25.1
ARG GIT_IMAGE=alpine/git:latest
ARG AITER_REPO=https://github.com/ROCm/aiter.git
# AITER pin: the release tag validated for this stack, published 2026-08-05.
# Earlier tags are not supported, so do not fall back to them. The resolved SHA
# is recorded at /etc/long_context_serving.aiter_head_sha in the image. Bump
# deliberately; stage 2 re-verifies the kernel.
ARG AITER_REF=v0.1.19.post2
# AITER derives its package version from git via setuptools_scm, which cannot
# describe the --depth=1 detached checkout: fetching by tag name still writes no
# local refs/tags entry, so `git describe` finds no names even on a tag pin. Feed
# it a concrete version instead (SETUPTOOLS_SCM_PRETEND_VERSION); the real commit
# SHA is still recorded at /etc/long_context_serving.aiter_head_sha.
ARG AITER_PRETEND_VERSION=0.1.19.post2
# NOTE: no flydsl find-links pin. pip resolves flydsl from its default
# indexes. If the build fails with "Could not find a version that satisfies
# the requirement flydsl==...", pass a PIP_FIND_LINKS build arg pointing at
# AMD's nightlies wheel index, e.g.
# https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/flydsl/

# ---------------------------------------------------------------------------
# Stage 1: fetch AITER at the pinned ref and record the resolved commit SHA.
# Done in a separate stage so the final image doesn't carry the .git history.
# Uses init+fetch rather than `clone --branch` so AITER_REF can be a tag, a
# branch, or a raw commit SHA interchangeably (`--branch` rejects raw SHAs).
FROM ${GIT_IMAGE} AS aiter_src
ARG AITER_REPO
ARG AITER_REF
WORKDIR /src
RUN apk add --no-cache git \
    && mkdir -p aiter \
    && cd aiter \
    && git init -q \
    && git remote add origin "${AITER_REPO}" \
    && git fetch -q --depth=1 origin "${AITER_REF}" \
    && git checkout -q FETCH_HEAD \
    && git submodule update --init --recursive --depth=1 \
    && git rev-parse HEAD > /src/AITER_HEAD_SHA.txt \
    && echo "[stage1] AITER ${AITER_REF} = $(cat /src/AITER_HEAD_SHA.txt)"

# ---------------------------------------------------------------------------
# Stage 2: install AITER over the vLLM-ROCm v0.25.1 base, verify the fixed
# decode kernel, and bake the long-context-serving app + substrate corpus.
FROM ${VLLM_BASE_IMAGE}

ARG AITER_REPO
ARG AITER_REF
ARG AITER_PRETEND_VERSION

ENV VLLM_ROCM_USE_AITER=1 \
    VLLM_ROCM_USE_AITER_MLA=1 \
    LONG_CONTEXT_SERVING_ROOT=/workspace/long-context-serving \
    PYTHONPATH=/workspace/long-context-serving/app

# Reinstall AITER from the pinned release tag over the base image's copy.
COPY --from=aiter_src /src/aiter /tmp/aiter-src
COPY --from=aiter_src /src/AITER_HEAD_SHA.txt /etc/long_context_serving.aiter_head_sha
# The post-install `import aiter` smoke check is intentionally non-fatal:
# `import aiter` can eagerly JIT-compile native modules that call `rocminfo`,
# which returns exit 1 inside `docker build` (no GPU access; --device flags
# are docker-run flags, not docker-build flags). The pip install itself does
# not need a GPU; JIT succeeds at first `import aiter` inside `docker run`
# where /dev/kfd is mapped through.
RUN set -eux; \
    export SETUPTOOLS_SCM_PRETEND_VERSION="${AITER_PRETEND_VERSION}"; \
    (python3 -m pip uninstall -y aiter amd-aiter || true); \
    python3 -m pip install --no-cache-dir /tmp/aiter-src; \
    echo "[install] AITER ref=${AITER_REF} commit=$(cat /etc/long_context_serving.aiter_head_sha)"; \
    if python3 -c "import aiter; print('[install] AITER at', aiter.__file__)" 2>/dev/null; then \
        echo "[install] AITER import verified at build time"; \
    else \
        echo "[install] AITER import skipped at build time (no GPU visible to docker build; JIT will run at first container start)"; \
    fi; \
    rm -rf /tmp/aiter-src

# ---------------------------------------------------------------------------
# Verify the persistent fp8 MLA decode kernel.
#
# The pinned AITER tag ships the validated build, so nothing is copied in here.
# This step only asserts the installed kernel is the known-good artifact, which
# turns a silent wrong-answer regression into a build failure. The expected md5
# tracks AITER_REF and must be updated whenever the pin moves.
ARG FIXED_PS_CO_MD5=c1b06569a97f6a4af37e6426caa2074b
RUN set -eux; \
    MLA_DIR="$(python3 -c 'import aiter_meta, os; print(os.path.join(os.path.dirname(aiter_meta.__file__), "hsa/gfx950/mla"))' 2>/dev/null \
        || echo /usr/local/lib/python3.12/dist-packages/aiter_meta/hsa/gfx950/mla)"; \
    TARGET="${MLA_DIR}/mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co"; \
    test -f "${TARGET}" || { echo "[kernel-verify] kernel not found: ${TARGET}" >&2; exit 1; }; \
    got="$(md5sum "${TARGET}" | awk '{print $1}')"; \
    echo "[kernel-verify] ${TARGET} md5=${got}"; \
    test "${got}" = "${FIXED_PS_CO_MD5}" || { \
        echo "[kernel-verify] FAIL: expected ${FIXED_PS_CO_MD5} (validated upstream build)." >&2; \
        echo "[kernel-verify] AITER_REF=${AITER_REF} does not ship the expected kernel." >&2; \
        exit 1; }; \
    echo "[kernel-verify] OK: decode kernel is the expected upstream build"

# Install the host-side tools needed by `make run`, plus matplotlib for report
# plotting.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates git make; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --no-cache-dir matplotlib

# Copy the slim release tree and clone the locked prompt substrate into the
# image. The Docker build context is the repo root; .dockerignore keeps local
# model/cache/run outputs out of the image.
WORKDIR /workspace/long-context-serving
COPY Makefile pyproject.toml README.md LICENSE NOTICE ./
COPY app ./app
COPY docker ./docker
COPY experiments/hf_long_context ./experiments/hf_long_context
COPY data/metadata/substrate_repos_manifest.json ./data/metadata/substrate_repos_manifest.json

RUN set -eux; \
    python3 -c "import long_context_serving; print('[bake] long_context_serving import OK')"; \
    python3 app/scripts/clone_substrate_repos.py; \
    python3 -c 'import json; from pathlib import Path; root=Path("/workspace/long-context-serving"); manifest=json.loads((root/"data/metadata/substrate_repos_manifest.json").read_text(encoding="utf-8")); repos=manifest.get("repos") or []; missing=[row["path"] for row in repos if not (root/row["path"]).is_dir()]; assert repos, "substrate manifest has no repos"; assert not missing, f"missing substrate repos: {missing}"; print(f"[image] substrate repos ready: {len(repos)}")'

LABEL org.long-context-serving.source="long-context-serving" \
      org.long-context-serving.base="vllm-openai-rocm:v0.25.1" \
      org.long-context-serving.aiter.ref="${AITER_REF}" \
      org.long-context-serving.aiter.tag="v0.1.19.post2" \
      org.long-context-serving.aiter.commit="see /etc/long_context_serving.aiter_head_sha" \
      org.long-context-serving.runtime_hotfixes="none (kernel md5 verified at build)"
