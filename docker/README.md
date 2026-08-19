# `long-context-serving:v0.25.1-longctx`

Docker support for the public long-context-serving reproduction image. The
image is the full public artifact: the vLLM/AITER runtime, repo code, and the
locked prompt substrate corpus. Model weights and run artifacts stay outside
the image.

Starting from `vllm/vllm-openai-rocm:v0.25.1`, the image:

1. Reinstalls AITER from the pinned upstream release tag
   [`v0.1.19.post2`](https://github.com/ROCm/aiter/releases/tag/v0.1.19.post2),
   which ships the long-context MLA decode/prefill kernels natively.
2. Copies the repo to `/workspace/long-context-serving`, exposes `app/` via
   `PYTHONPATH`, and clones the locked substrate from
   `data/metadata/substrate_repos_manifest.json`.

Kimi-Linear's post-TP nhead=4 is served by vLLM v0.25.1's own MLA head padding
(`AiterMLAHelper.get_mla_padded_q` / `_unpadded_o`).

## Build

From the repo root:

```bash
./docker/build.sh
```

or through the public Make wrapper:

```bash
make image
```

Pass arbitrary Docker build flags through:

```bash
./docker/build.sh --no-cache --progress=plain
```

Environment overrides:

| Variable | Default |
|---|---|
| `IMAGE_TAG` | `long-context-serving:v0.25.1-longctx` |
| `VLLM_BASE_IMAGE` | `vllm/vllm-openai-rocm:v0.25.1` |
| `AITER_REPO` | `https://github.com/ROCm/aiter.git` |
| `AITER_REF` | `v0.1.19.post2` (upstream release tag) |
| `AITER_PRETEND_VERSION` | `0.1.19.post2` |
| `GIT_IMAGE` | `alpine/git:latest` |

### About `AITER_PRETEND_VERSION`

AITER derives its package version from git via `setuptools_scm`, which cannot
`git describe` the shallow detached checkout the build creates. Fetching by tag
name still writes no local `refs/tags` entry, so `git describe` finds no names
even on a tag pin and the build would otherwise fail to determine a version. The
build feeds `setuptools_scm` a concrete version
(`SETUPTOOLS_SCM_PRETEND_VERSION`) instead; the real resolved commit SHA is
recorded at `/etc/long_context_serving.aiter_head_sha`.

### About flydsl

AITER declares a `flydsl` dependency that is not pinned in this image; pip
resolves it from its default indexes. If the build fails with
`Could not find a version that satisfies the requirement flydsl==...`, pass a
`PIP_FIND_LINKS` build arg pointing at AMD's nightlies wheel index, e.g.
`https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/flydsl/`.

## Run

Use the repo-level wrapper:

```bash
make run
```

`make run` mounts the Hugging Face cache at `/hf`, mounts the host run output
directory at `/outputs/hf_long_context/runs`, and runs the in-image Make target
from `/workspace/long-context-serving`.
