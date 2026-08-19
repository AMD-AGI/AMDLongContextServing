#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Build the public long-context-serving image (vLLM v0.25.1 + upstream AITER
# pinned at v0.1.19.post2), release source, and the locked substrate corpus
# needed for the MI355X long-context campaign. The Dockerfile lives alongside
# this script; arbitrary docker-build flags (e.g. --no-cache, --progress=plain,
# --platform) can be passed through as positional arguments.
#
# Examples:
#   ./build.sh
#   ./build.sh --no-cache --progress=plain
#   IMAGE_TAG=my-vllm:dev ./build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"
[ -f "${DOCKERFILE}" ] || { echo "Dockerfile not found at ${DOCKERFILE}" >&2; exit 1; }

IMAGE_TAG="${IMAGE_TAG:-long-context-serving:v0.25.1-longctx}"
VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-vllm/vllm-openai-rocm:v0.25.1}"
GIT_IMAGE="${GIT_IMAGE:-alpine/git:latest}"
AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
# AITER release tag validated for this stack; earlier tags are not supported.
AITER_REF="${AITER_REF:-v0.1.19.post2}"
# setuptools_scm can't describe a shallow detached commit; feed it a version.
AITER_PRETEND_VERSION="${AITER_PRETEND_VERSION:-0.1.19.post2}"

echo "Building ${IMAGE_TAG}"
echo "  base:           ${VLLM_BASE_IMAGE}"
echo "  AITER:          ${AITER_REPO} @ ${AITER_REF} (pinned tag)"
echo "  flydsl index:   (not pinned)"
echo "  build context:  ${REPO_ROOT}"
echo "  dockerfile:     ${DOCKERFILE}"

DOCKER_BUILD_CMD=(
    docker build
    "$@"
    --build-arg "VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE}"
    --build-arg "GIT_IMAGE=${GIT_IMAGE}"
    --build-arg "AITER_REPO=${AITER_REPO}"
    --build-arg "AITER_REF=${AITER_REF}"
    --build-arg "AITER_PRETEND_VERSION=${AITER_PRETEND_VERSION}"
    -t "${IMAGE_TAG}"
    -f "${DOCKERFILE}"
    "${REPO_ROOT}"
)

exec "${DOCKER_BUILD_CMD[@]}"
