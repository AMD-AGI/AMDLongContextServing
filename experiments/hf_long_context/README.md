# HF Long-Context FP8 Sweep

This directory contains the Kimi-Linear-48B-A3B-Instruct long-context FP8
decode sweep on AMD MI355X. The public reproduction path is the repo-level
command:

```bash
make run
```

`make run` builds `long-context-serving:v0.25.1-longctx` when needed, starts the
image, verifies the locked substrate is present, runs the requested sweep, and
writes reports/charts to the host `experiments/hf_long_context/runs/` tree.

## Hardware and Runtime Assumptions

- Reference hardware: one 8-GPU AMD Instinct MI355X node (`gfx950`), ROCm
  7.0.0, Ubuntu 22.04.2 LTS, AMD EPYC host CPUs.
- The headline 64Mi point requires TP=8 and the full 8-GPU HBM budget. A
  single MI355X can run shorter smoke sweeps when `HFLC_FP8_TP=1` and the
  sweep list is capped accordingly.
- Host requirements for the public path are Docker, a working ROCm GPU driver,
  and access to the Hugging Face model/token. `git`, `make`, Python tooling,
  and the substrate corpus live inside the image.

## Image Contents

[`../../docker/Dockerfile`](../../docker/Dockerfile) starts from pinned
`vllm/vllm-openai-rocm:v0.25.1`, then:

1. Reinstalls AITER from the pinned upstream release tag `v0.1.19.post2`, which
   ships the long-context MLA decode/prefill kernels natively. nhead<16 is
   handled by vLLM v0.25.1's own MLA head padding.
2. Copies this repo to `/workspace/long-context-serving` and exposes `app/` via
   `PYTHONPATH`.
3. Clones every source substrate repo pinned by
   [`../../data/metadata/substrate_repos_manifest.json`](../../data/metadata/substrate_repos_manifest.json).

The image does not include model weights or run artifacts. `make run` mounts
the Hugging Face cache at `/hf` and the host run directory at
`/outputs/hf_long_context/runs`.

## Running the Campaign

```bash
make run
make run FROM=1Ki TO=16Ki
make run FROM=4Mi TO=8Mi REPEATS=3
```

`FROM`/`TO` are inclusive context-length bounds expanded to powers of two.
Default behavior is the powers-of-two sweep from 1Ki to 64Mi tokens, the
`ROCM_AITER_MLA` attention backend, FP8 KV (`fp8_e4m3`), and cudagraph mode
`FULL_AND_PIECEWISE`.

Each sweep point starts a fresh `vllm serve`, runs the benchmark, tears the
server down, and appends to the campaign report. Small points use the repeat
defaults in [`config/defaults.mk`](config/defaults.mk); large points keep the
configured cold-start pass behavior.

Useful host-side overrides:

```bash
IMAGE_TAG=my-image:tag make run
RUNS_DIR=/data/long-context-runs make run
HF_CACHE=/data/hf-cache HF_TOKEN="$HF_TOKEN" make run
```

Inside the image, `make hflc-config-help` prints every `HFLC_*` knob with its
default and docstring.

## Artifacts

After every sweep point, files under each `phase1/` directory include:

- `run_meta.json`, `launch_env.json`, `server_cmd.txt`: exact invocation,
  environment, and prompt token IDs.
- `vllm_server.log`, `client.log`: server and client logs.
- `results.jsonl`, `summary.json`, `sweep_summary.json`: per-request and
  per-length metrics.
- `prefill_progress.jsonl`: present when prefill-progress polling is enabled.
- `kv_calibration.json`: calibration gate output when enabled.

Keep `report.md` in the campaign directory: it is the primary campaign report.
The same directory also contains `consolidated_results.jsonl` and the generated
timing/decode charts.

Prefill progress polling is off by default
(`HFLC_FP8_EMIT_PREFILL_PROGRESS=0`): it polls a vLLM debug endpoint that is
not built into the public image, so enabling it only logs a harmless 404 and
has no effect on the measurements. To watch a long prefill, tail
`phase1/vllm_server.log` directly.

## Diagnostics and Resume

From an interactive shell inside the image:

```bash
make hf-long-context-fp8-gpu
make hf-long-context-fp8-queue-tail
```

Resume an interrupted campaign in place by preserving the original campaign
timestamp and passing only unfinished lengths:

```bash
make run FROM=32Mi TO=64Mi STAMP=20260515T035829Z
```

Rebuild a report from existing artifacts without re-running a sweep:

```bash
docker run --rm -it \
  -v "$PWD/experiments/hf_long_context/runs:/outputs/hf_long_context/runs" \
  --workdir /workspace/long-context-serving \
  --entrypoint /bin/bash \
  long-context-serving:v0.25.1-longctx \
  -lc 'make hf-long-context-fp8-report \
        HFLC_FP8_RUN_DIR=/outputs/hf_long_context/runs/<single_L_run_dir> \
        HFLC_APPEND_TO_CAMPAIGN=1'
```

## Prompt Shape

The default public sweep uses `repo_grounded_en_v2`, which builds prompts from
the included source-code substrate. The runner also accepts `benchmark_v1` and
`custom_file` for internal experiments:

```bash
python3 app/scripts/benchmark_vllm_long_context.py \
  --model-id moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --sweep-prompt-tokens 1024 \
  --prompt-shape repo_grounded_en_v2
```

Custom prompt-shape files use this schema:

```json
{
  "schema_version": 1,
  "name": "custom_name",
  "messages": [
    {"role": "system", "content": "{{system_prompt}}"},
    {"role": "user", "content": "{{prompt_text}}"}
  ]
}
```

Allowed placeholders in `messages[*].content`: `{{prompt_text}}`,
`{{system_prompt}}`, `{{max_new_tokens}}`, `{{prompt_tokens}}`, `{{model_id}}`.

## Files

| Path | Contents |
|---|---|
| [`config/defaults.mk`](config/defaults.mk) | `HFLC_*` defaults and docs consumed by `hflc-config-help`. |
| [`driver_vllm_two_phase.sh`](driver_vllm_two_phase.sh) | vLLM serve + benchmark client driver invoked by Make. |
| [`../../docker/`](../../docker/) | Dockerfile and build wrapper for the benchmark image. |
