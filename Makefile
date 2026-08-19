# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

SHELL := /bin/bash

# Resolve repo root from this Makefile path so callers do not need to set env vars.
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
REPO_ROOT := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))

LOCKED ?= 1
SUBSTRATE_ROOT := $(REPO_ROOT)/data/substrate

# One-command wrapper knobs (see the `run` target). FROM/TO are inclusive
# context-length bounds expanded to consecutive powers of two; they accept
# Ki/Mi suffixes (1Ki=1024, 2Mi=2097152) or raw token counts. REPEATS sets
# measured runs per point (empty = profile default). STAMP resumes an existing
# campaign timestamp in place.
FROM ?= 1Ki
TO ?= 64Mi
REPEATS ?=
STAMP ?=

# Docker image knobs (see the `image` and `run` targets). IMAGE_TAG is the
# vLLM-ROCm image (vLLM v0.25.1 + AITER v0.1.19.post2) that the sweep runs
# inside. HF_CACHE is the host HuggingFace cache, bind-mounted to /hf so model
# weights persist across runs; it defers to an existing HF_HOME /
# HUGGINGFACE_HUB_CACHE before falling back to the standard per-user cache.
# RUNS_DIR is the host output directory mounted into the image for reports and
# charts.
IMAGE_TAG ?= long-context-serving:v0.25.1-longctx
HF_CACHE ?= $(or $(HF_HOME),$(HUGGINGFACE_HUB_CACHE),$(HOME)/.cache/huggingface)
RUNS_DIR ?= $(REPO_ROOT)/experiments/hf_long_context/runs

# HuggingFace access token for gated model downloads. Resolution order:
#   1. HF_TOKEN in the environment (wins if set)
#   2. the file named by HF_TOKEN_FILE (default $(HOME)/hf_token) if it exists
# Leave both unset/absent for ungated models. The token is exported into the
# recipe environment and inherited by `docker run` via a value-less `-e
# HF_TOKEN`, so the secret never appears on a command line (where `ps` could
# expose it on a shared host). It is never written to disk or echoed.
HF_TOKEN_FILE ?= $(HOME)/hf_token
HF_TOKEN ?= $(strip $(if $(wildcard $(HF_TOKEN_FILE)),$(file < $(HF_TOKEN_FILE)),))
export HF_TOKEN

include $(REPO_ROOT)/experiments/hf_long_context/config/defaults.mk

# ============================================================================
# Public API
# ============================================================================
# Campaign entrypoints (the only targets a typical user invokes):
#   make run                                  # all-in-one: build image (if needed) +
#                                             #   enter image + sweep
#       make run FROM=1Ki TO=2Mi              #   quick test over a sub-range
#       make run FROM=4Mi TO=8Mi REPEATS=3    #   skips build/hydration if already present
#       make run FROM=32Mi TO=64Mi STAMP=...  #   resume an existing campaign
#   make image                                # build the benchmark image (host-side)
#   make hf-long-context-fp8-sweep            # 1K..64M long-context sweep (range via HFLC_CAMPAIGN_SWEEP_LIST)
#   make hflc-config-help                     # list of HFLC_* env vars
#
# `make run` is host/container aware: invoked on the host it builds the image
# (if missing) and re-enters itself inside the image; invoked inside the
# image it verifies the substrate and runs the sweep directly.
#
# Live-diagnostic helpers:
#   make hf-long-context-fp8-gpu
#   make hf-long-context-fp8-queue-tail
#
# Post-run regeneration without re-running the sweep:
#   make hf-long-context-fp8-report HFLC_FP8_RUN_DIR=... HFLC_APPEND_TO_CAMPAIGN=1
#
# Everything else (run, run-list, assert,
# hflc-config-validate, substrate-prune-stale) is invoked transitively from
# the public entrypoints and is not meant to be called directly.
# ============================================================================

.PHONY: \
	run \
	run-in-container \
	image \
	substrate-ensure \
	hf-long-context-fp8-sweep \
	hf-long-context-fp8-run \
	hf-long-context-fp8-run-list \
	hf-long-context-fp8-assert \
	hf-long-context-fp8-report \
	hf-long-context-fp8-gpu \
	hf-long-context-fp8-queue-tail \
	hflc-config-help \
	hflc-config-validate \
	substrate-clone \
	substrate-prune-stale

# ----------------------------------------------------------------------------
# One-command entrypoint: build image (if needed) + enter container + run the sweep
# ----------------------------------------------------------------------------

# Build the benchmark image: the upstream vLLM-ROCm base plus a pinned upstream
# AITER build, both taken as published (see NOTICE). Idempotent: Docker
# layer-caches, so a repeat call with no Dockerfile changes is fast. Host-side
# only.
image:
	@set -euo pipefail; \
	echo "==> image: building $(IMAGE_TAG) via docker/build.sh"; \
	IMAGE_TAG="$(IMAGE_TAG)" $(REPO_ROOT)/docker/build.sh

# Host/container-aware entrypoint.
#   - Inside a container (we got here via our own docker run, or the user
#     manually shelled into the image): hydrate substrate (if needed) and run
#     the sweep directly.
#   - On the host: build the image if missing, then docker-run this same
#     `make run` inside the container with the HF cache and run output mounted.
# Container detection keys off HFLC_IN_CONTAINER (set by our docker run) or
# /.dockerenv (present in any Docker container) -- NOT off `import aiter`,
# which can succeed on a bare-metal ROCm host and wrongly bypass the image.
run:
	@set -euo pipefail; \
	if [ -n "$${HFLC_IN_CONTAINER:-}" ] || [ -f /.dockerenv ]; then \
		$(MAKE) --no-print-directory run-in-container \
			FROM="$(FROM)" TO="$(TO)" REPEATS="$(REPEATS)" STAMP="$(STAMP)"; \
	else \
		echo "==> run: on host -> ensuring image '$(IMAGE_TAG)' exists"; \
		if ! docker image inspect "$(IMAGE_TAG)" >/dev/null 2>&1; then \
			echo "==> run: image not found -> building"; \
			$(MAKE) --no-print-directory image; \
		else \
			echo "==> run: image '$(IMAGE_TAG)' already present (skipping build)"; \
		fi; \
		echo "==> run: entering container to run FROM=$(FROM) TO=$(TO) REPEATS=$(REPEATS) PASSES=$(HFLC_CAMPAIGN_PASS_START)..$(HFLC_CAMPAIGN_NUM_PASSES)"; \
		mkdir -p "$(RUNS_DIR)"; \
		mkdir -p "$(HF_CACHE)" 2>/dev/null || echo "==> run: cannot mkdir HF_CACHE=$(HF_CACHE) as $$(id -un); leaving it for dockerd (root) to create/mount"; \
		TTY_FLAGS="-i"; if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS="-it"; fi; \
		docker run --rm $$TTY_FLAGS \
			--entrypoint /bin/bash \
			--network=host --ipc=host --privileged \
			--device=/dev/kfd --device=/dev/dri \
			--user root --group-add video \
			--cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
			--security-opt seccomp=unconfined \
			--shm-size=16g \
			-e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf/transformers \
			-e HF_TOKEN \
			-e HFLC_IN_CONTAINER=1 \
			-v "$(HF_CACHE)":/hf \
			-v "$(RUNS_DIR)":/outputs/hf_long_context/runs \
			--workdir /workspace/long-context-serving \
			"$(IMAGE_TAG)" \
			-lc 'make run FROM=$(FROM) TO=$(TO) REPEATS=$(REPEATS) STAMP=$(STAMP) HFLC_RUNS_DIR=/outputs/hf_long_context/runs HFLC_CAMPAIGN_NUM_PASSES=$(HFLC_CAMPAIGN_NUM_PASSES) HFLC_CAMPAIGN_PASS_START=$(HFLC_CAMPAIGN_PASS_START) HFLC_CAMPAIGN_RUN_ID=$(HFLC_CAMPAIGN_RUN_ID) HFLC_FP8_CRASH_CAPTURE=$(HFLC_FP8_CRASH_CAPTURE)'; \
	fi

# In-container worker: expand FROM..TO into the campaign sweep list, ensure the
# substrate is present, then run the sweep. Not meant to be called directly.
run-in-container: substrate-ensure
	@set -euo pipefail; \
	SWEEP_LIST="$$(python3 $(REPO_ROOT)/app/scripts/expand_sweep_range.py "$(FROM)" "$(TO)")"; \
	echo "==> run: FROM=$(FROM) TO=$(TO) -> sweep_list=$$SWEEP_LIST"; \
	if [ -n "$(REPEATS)" ]; then \
		echo "    repeats (HFLC_FP8_NUM_RUNS)=$(REPEATS)"; \
		$(MAKE) --no-print-directory hf-long-context-fp8-sweep \
			HFLC_CAMPAIGN_SWEEP_LIST="$$SWEEP_LIST" \
			HFLC_CAMPAIGN_RUN_ID_STAMP_OVERRIDE="$(STAMP)" \
			HFLC_FP8_NUM_RUNS="$(REPEATS)"; \
	else \
		$(MAKE) --no-print-directory hf-long-context-fp8-sweep \
			HFLC_CAMPAIGN_RUN_ID_STAMP_OVERRIDE="$(STAMP)" \
			HFLC_CAMPAIGN_SWEEP_LIST="$$SWEEP_LIST"; \
	fi

# Hydrate the substrate corpus only if it is not already present. Presence is
# keyed off the repo paths recorded in the substrate manifest: if every one of
# them exists under data/substrate/, hydration is skipped.
substrate-ensure:
	@set -euo pipefail; \
	if python3 -c 'import json,sys,os; \
m=json.load(open("$(REPO_ROOT)/data/metadata/substrate_repos_manifest.json")); \
paths=[os.path.join("$(REPO_ROOT)", r["path"]) for r in m["repos"]]; \
sys.exit(0 if paths and all(os.path.isdir(p) and os.listdir(p) for p in paths) else 1)'; then \
		echo "==> substrate: already hydrated under $(SUBSTRATE_ROOT) (skipping clone)"; \
	else \
		echo "==> substrate: missing or incomplete -> hydrating"; \
		$(MAKE) --no-print-directory substrate-clone; \
	fi

# ----------------------------------------------------------------------------
# Substrate hydration
# ----------------------------------------------------------------------------

substrate-clone:
	@$(MAKE) substrate-prune-stale
	@if [ "$(LOCKED)" = "0" ]; then \
		echo "INFO: LOCKED=0 -> syncing substrate repos to upstream HEAD and overwriting data/metadata/substrate_repos_manifest.json"; \
		python3 $(REPO_ROOT)/app/scripts/clone_substrate_repos.py --unlocked-head; \
	else \
		python3 $(REPO_ROOT)/app/scripts/clone_substrate_repos.py; \
	fi

substrate-prune-stale:
	@STALE="$(SUBSTRATE_ROOT)/benford-mini"; \
	if [ -d "$$STALE" ]; then \
		echo "Pruning stale substrate path: $$STALE"; \
		rm -rf "$$STALE"; \
	else \
		echo "No stale substrate path found at $$STALE"; \
	fi

# ----------------------------------------------------------------------------
# Config discoverability + validation
# ----------------------------------------------------------------------------

hflc-config-help:
	@python3 $(REPO_ROOT)/app/scripts/hflc_config_help.py \
		--defaults "$(REPO_ROOT)/experiments/hf_long_context/config/defaults.mk"

hflc-config-validate:
	@REPO_ROOT="$(REPO_ROOT)" \
	HFLC_PROFILE="$(HFLC_PROFILE)" \
	HFLC_FP8_TP="$(HFLC_FP8_TP)" \
	HFLC_FP8_DP="$(HFLC_FP8_DP)" \
	HFLC_FP8_PP="$(HFLC_FP8_PP)" \
	HFLC_FP8_MAX_NUM_SEQS="$(HFLC_FP8_MAX_NUM_SEQS)" \
	HFLC_FP8_SWEEP="$(HFLC_FP8_SWEEP)" \
	HFLC_FP8_SWEEP_LIST="$(HFLC_FP8_SWEEP_LIST)" \
	HFLC_FP8_TOKENIZER_ID="$(HFLC_FP8_TOKENIZER_ID)" \
	HFLC_FP8_PROMPT_SHAPE="$(HFLC_FP8_PROMPT_SHAPE)" \
	HFLC_FP8_PROMPT_SHAPE_FILE="$(HFLC_FP8_PROMPT_SHAPE_FILE)" \
	HFLC_FP8_PROMPT_TOKEN_IDS_FILE="$(HFLC_FP8_PROMPT_TOKEN_IDS_FILE)" \
	HFLC_FP8_QUERY_API_MODE="$(HFLC_FP8_QUERY_API_MODE)" \
	HFLC_FP8_ENGLISH_ONLY_MASK_MODE="$(HFLC_FP8_ENGLISH_ONLY_MASK_MODE)" \
	HFLC_FP8_ENGLISH_ONLY_MASK_BIAS="$(HFLC_FP8_ENGLISH_ONLY_MASK_BIAS)" \
	HFLC_FP8_DUMP_REQUEST_PAYLOAD="$(HFLC_FP8_DUMP_REQUEST_PAYLOAD)" \
	HFLC_FP8_DUMP_QUERY_MESSAGES="$(HFLC_FP8_DUMP_QUERY_MESSAGES)" \
	HFLC_FP8_DUMP_PROMPT_TOKEN_IDS="$(HFLC_FP8_DUMP_PROMPT_TOKEN_IDS)" \
	HFLC_FP8_NO_STREAM="$(HFLC_FP8_NO_STREAM)" \
	HFLC_FP8_REQUIRE_LAUNCH_CONTRACT="$(HFLC_FP8_REQUIRE_LAUNCH_CONTRACT)" \
	HFLC_FP8_ENFORCE_EAGER="$(HFLC_FP8_ENFORCE_EAGER)" \
	HFLC_FP8_DISABLE_CUSTOM_ALL_REDUCE="$(HFLC_FP8_DISABLE_CUSTOM_ALL_REDUCE)" \
	HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB="$(HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB)" \
	HFLC_FP8_VLLM_COMPILATION_CONFIG="$(HFLC_FP8_VLLM_COMPILATION_CONFIG)" \
	HFLC_FP8_REQUIRE_KV_CALIBRATION="$(HFLC_FP8_REQUIRE_KV_CALIBRATION)" \
	HFLC_FP8_KV_CALIBRATION_GATE="$(HFLC_FP8_KV_CALIBRATION_GATE)" \
	HFLC_FP8_CRASH_CAPTURE="$(HFLC_FP8_CRASH_CAPTURE)" \
	HFLC_FP8_CRASH_CAPTURE_SET_ULIMIT="$(HFLC_FP8_CRASH_CAPTURE_SET_ULIMIT)" \
	HFLC_FP8_CRASH_CAPTURE_ENABLE_HSA_DEBUG="$(HFLC_FP8_CRASH_CAPTURE_ENABLE_HSA_DEBUG)" \
	HFLC_FP8_CAPTURE_MODE="$(HFLC_FP8_CAPTURE_MODE)" \
	HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILE="$(HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILE)" \
	HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILES="$(HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILES)" \
	HFLC_FP8_CAPTURE_SUFFIX_QUERY_DIR="$(HFLC_FP8_CAPTURE_SUFFIX_QUERY_DIR)" \
	HFLC_FP8_CAMPAIGN_DIR="$(HFLC_FP8_CAMPAIGN_DIR)" \
	HFLC_FP8_CURRENT_LABEL="$(HFLC_FP8_CURRENT_LABEL)" \
	python3 $(REPO_ROOT)/app/scripts/hflc_config_validate.py

# ----------------------------------------------------------------------------
# Campaign sweep flow (vLLM v0.25.1 + AITER v0.1.19.post2)
# ----------------------------------------------------------------------------
#
# The sweep targets run inside the $(IMAGE_TAG) image (built from
# docker/Dockerfile): stock vLLM v0.25.1 with AITER installed from the pinned
# upstream release tag v0.1.19.post2. The long-context MLA decode/prefill
# kernels and the nhead<16 head-padding are all native to that stack, so the
# image needs no guard and there is nothing bespoke in it to verify.

# 1K..64M powers-of-2 sweep. Override the list by passing
# HFLC_CAMPAIGN_SWEEP_LIST=...
HFLC_CAMPAIGN_SWEEP_LIST ?= 1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864
HFLC_CAMPAIGN_RUN_ID     ?= fp8_aiter_v0.25.1_longctx

HFLC_CAMPAIGN_CUDAGRAPH_MODE ?= FULL_AND_PIECEWISE
# Cold-start passes per point for the campaign sweep; the report medians across
# them per prompt. Override to 1
# for a quick single-pass sweep.
HFLC_CAMPAIGN_NUM_PASSES ?= 3
# First pass index to run, so an interrupted campaign can be continued without
# recomputing the passes it already has. Pass indices appear verbatim in each
# run directory name (`..._pass<N>_<stamp>`), so resuming at 2 with
# HFLC_CAMPAIGN_RUN_ID_STAMP_OVERRIDE set writes pass 2 alongside the existing
# pass 1 rather than colliding with it. Set together with NUM_PASSES to run a
# single pass: START=2 NUM_PASSES=2 runs only pass 2.
HFLC_CAMPAIGN_PASS_START ?= 1
# Per-file char cap. Inherits from the global HFLC_FP8_MAX_CHARS_PER_FILE
# default (64 KiB), which defends against an accidentally-quadratic
# `_truncate_block_with_footer` re-tokenize loop in benchmark_common.py
# when the substrate cumulative budget lands on a many-line multi-MB file
# (root cause of the 2026-04-30 8h+ hang at L=8M; reproduced again on
# 2026-05-14 in an 8M sweep when this cap was missing). Most source files are under 64 KiB
# so this is a no-op for them. Override only after _truncate_block_with_footer
# has been made non-quadratic.
HFLC_CAMPAIGN_MAX_CHARS_PER_FILE ?= $(HFLC_FP8_MAX_CHARS_PER_FILE)

# Optional opt-in: continue an interrupted campaign by reusing its timestamp
# stamp instead of forking a new sibling. Pass the original campaign's UTC
# timestamp (e.g. HFLC_CAMPAIGN_RUN_ID_STAMP_OVERRIDE=20260515T035829Z) and the
# original sweep list trimmed to the unfinished points; the wrapper will
# write into the same _<RUN_ID>_<STAMP>_campaign directory.
HFLC_CAMPAIGN_RUN_ID_STAMP_OVERRIDE ?=

hf-long-context-fp8-sweep:
	@set -euo pipefail; \
	HFLC_WRAPPER_ENTRYPOINT="$${HFLC_WRAPPER_ENTRYPOINT:-$@}"; \
	HFLC_WRAPPER_CHAIN="$${HFLC_WRAPPER_CHAIN:-}$@|"; \
	export HFLC_WRAPPER_ENTRYPOINT HFLC_WRAPPER_CHAIN; \
	stamp_override="$(HFLC_CAMPAIGN_RUN_ID_STAMP_OVERRIDE)"; \
	if [ -n "$$stamp_override" ]; then \
		run_id_stamp="$(HFLC_CAMPAIGN_RUN_ID)_$${stamp_override}"; \
		echo "==> sweep: continuing existing campaign (stamp_override=$$stamp_override)"; \
	else \
		run_id_stamp="$(HFLC_CAMPAIGN_RUN_ID)_$$(date -u +%Y%m%dT%H%M%SZ)"; \
	fi; \
	cudagraph_mode="$(HFLC_CAMPAIGN_CUDAGRAPH_MODE)"; \
	current_label="MI355X AITER MLA FP8 (vLLM v0.25.1 + AITER v0.1.19.post2, cudagraph=$${cudagraph_mode})"; \
	num_passes="$(HFLC_CAMPAIGN_NUM_PASSES)"; \
	if [ -z "$$num_passes" ] || [ "$$num_passes" -lt 1 ]; then num_passes=1; fi; \
	pass_start="$(HFLC_CAMPAIGN_PASS_START)"; \
	if [ -z "$$pass_start" ] || [ "$$pass_start" -lt 1 ]; then pass_start=1; fi; \
	if [ "$$pass_start" -gt "$$num_passes" ]; then \
		echo "==> sweep: pass_start=$$pass_start exceeds num_passes=$$num_passes; nothing to do" >&2; \
		exit 1; \
	fi; \
	echo "==> sweep: list=$(HFLC_CAMPAIGN_SWEEP_LIST) run_id=$$run_id_stamp passes=$$pass_start..$$num_passes"; \
	echo "    label=$$current_label"; \
	for pass_n in $$(seq $$pass_start $$num_passes); do \
	echo "==> sweep: pass $$pass_n/$$num_passes (full point list)"; \
	ALLOW_CUDAGRAPH=1 \
	$(MAKE) --no-print-directory hf-long-context-fp8-run-list \
	    HFLC_FP8_SWEEP_LIST="$(HFLC_CAMPAIGN_SWEEP_LIST)" \
	    HFLC_FP8_RUN_ID="$$run_id_stamp" \
	    HFLC_FP8_PASS_INDEX="$$pass_n" \
	    HFLC_FP8_PROMPT_SHAPE=repo_grounded_en_v2 \
	    HFLC_FP8_COHERENCE_GUARDRAILS=0 \
	    HFLC_FP8_POST_RUN_CACHE_CLEANUP=all \
	    HFLC_FP8_MAX_CHARS_PER_FILE="$(HFLC_CAMPAIGN_MAX_CHARS_PER_FILE)" \
	    HFLC_FP8_VLLM_COMPILATION_CONFIG="{\"cudagraph_mode\":\"$$cudagraph_mode\"}" \
	    HFLC_FP8_BASELINE_REPORT_MD="$(HFLC_CAMPAIGN_TRITON_BASELINE_REPORT_MD)" \
	    HFLC_FP8_BASELINE_LABEL="$(HFLC_CAMPAIGN_TRITON_BASELINE_LABEL)" \
	    HFLC_FP8_CURRENT_LABEL="$$current_label"; \
	done

# Optional lower-bound baseline for the cross-comparison plots. Unset by
# default: the release ships no bundled baseline campaign, so the report is
# generated without an overlay. Point REPORT_MD at a prior campaign's
# report.md (and set a LABEL) to draw the comparison curve.
HFLC_CAMPAIGN_TRITON_BASELINE_REPORT_MD ?=
HFLC_CAMPAIGN_TRITON_BASELINE_LABEL     ?=

# ----------------------------------------------------------------------------
# Core single-point + sweep + post-run targets
# ----------------------------------------------------------------------------

hf-long-context-fp8-run: hflc-config-validate
	@set -euo pipefail; \
	HFLC_WRAPPER_ENTRYPOINT="$${HFLC_WRAPPER_ENTRYPOINT:-$@}"; \
	HFLC_WRAPPER_CHAIN="$${HFLC_WRAPPER_CHAIN:-}$@|"; \
	export HFLC_WRAPPER_ENTRYPOINT HFLC_WRAPPER_CHAIN; \
	MAX_PROMPT="$$(python3 -c 'import sys; vals=[int(x.strip()) for x in sys.argv[1].split(",") if x.strip()]; print(max(vals))' "$(HFLC_FP8_SWEEP)")"; \
	MAX_LEN="$(HFLC_FP8_MAX_MODEL_LEN)"; \
	if [ -z "$$MAX_LEN" ]; then \
		MAX_LEN="$$((MAX_PROMPT + 16384))"; \
	fi; \
	TARGET_PROMPT="$(HFLC_FP8_TARGET_PROMPT_TOKENS)"; \
	if [ -z "$$TARGET_PROMPT" ]; then \
		TARGET_PROMPT="$$MAX_PROMPT"; \
	fi; \
	PHASE_DIR="$(HFLC_FP8_RUN_DIR)/phase1"; \
		echo "Running FP8 long-context benchmark"; \
		echo "  RUN_ID=$(HFLC_FP8_RUN_ID)"; \
		echo "  RUN_DIR=$(HFLC_FP8_RUN_DIR)"; \
		echo "  SWEEP=$(HFLC_FP8_SWEEP)"; \
		echo "  MAX_MODEL_LEN=$$MAX_LEN"; \
		echo "  TARGET_PROMPT_TOKENS=$$TARGET_PROMPT"; \
		echo "  PREFIX_SESSION_MODE=$(HFLC_FP8_PREFIX_SESSION_MODE)"; \
		echo "  SUFFIX_QUERY_FILE=$(HFLC_FP8_SUFFIX_QUERY_FILE)"; \
		echo "  SUFFIX_QUERY_FILES=$(HFLC_FP8_SUFFIX_QUERY_FILES)"; \
		echo "  SUFFIX_QUERY_DIR=$(HFLC_FP8_SUFFIX_QUERY_DIR)"; \
		echo "  PREFIX_QUERY_MAX_COUNT=$(HFLC_FP8_PREFIX_QUERY_MAX_COUNT)"; \
		echo "  PREFIX_RELOAD_BETWEEN_QUERIES=$(HFLC_FP8_PREFIX_RELOAD_BETWEEN_QUERIES)"; \
		echo "  GPU_MEMORY_UTILIZATION=$(HFLC_FP8_GPU_MEMORY_UTILIZATION)"; \
		echo "  VLLM_COMPILATION_CONFIG=$(HFLC_FP8_VLLM_COMPILATION_CONFIG)"; \
		echo "  DISABLE_CUSTOM_ALL_REDUCE=$(HFLC_FP8_DISABLE_CUSTOM_ALL_REDUCE)"; \
		echo "  CUSTOM_ALL_REDUCE_MAX_SIZE_MB=$(HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB)"; \
		mkdir -p "$$PHASE_DIR"; \
		ROOT="$(REPO_ROOT)" \
		REPO_ROOT="$(REPO_ROOT)" \
		SUBSTRATE_ROOT="$(HFLC_SUBSTRATE_ROOT)" \
		RUN_ID="$(HFLC_FP8_RUN_ID)" \
		RUN_DIR="$(HFLC_FP8_RUN_DIR)" \
	PHASE_MODE=phase1 \
	PORT="$(HFLC_FP8_PORT)" VLLM_DP_MASTER_PORT="$(HFLC_FP8_DP_MASTER_PORT)" \
	TARGET_PROMPT_TOKENS="$$TARGET_PROMPT" \
		TP="$(HFLC_FP8_TP)" DATA_PARALLEL_SIZE="$(HFLC_FP8_DP)" PP="$(HFLC_FP8_PP)" \
			HIP_VISIBLE_DEVICES="$(HFLC_FP8_GPU_LIST)" \
			GPU_MEMORY_UTILIZATION="$(HFLC_FP8_GPU_MEMORY_UTILIZATION)" \
			MODEL_ID="$(HFLC_FP8_MODEL_ID)" \
			VLLM_DTYPE="$(HFLC_FP8_VLLM_DTYPE)" \
			KV_CACHE_DTYPE="$(HFLC_FP8_KV_CACHE_DTYPE)" \
		VLLM_ATTENTION_BACKEND=ROCM_AITER_MLA \
		VLLM_ENFORCE_EAGER="$(HFLC_FP8_ENFORCE_EAGER)" \
		VLLM_DISABLE_ROCM_SKINNY_GEMM="$(HFLC_FP8_DISABLE_ROCM_SKINNY_GEMM)" \
		VLLM_DISABLE_CUSTOM_ALL_REDUCE="$(HFLC_FP8_DISABLE_CUSTOM_ALL_REDUCE)" \
		VLLM_CUSTOM_ALL_REDUCE_MAX_SIZE_MB="$(HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB)" \
		VLLM_COMPILATION_CONFIG='$(HFLC_FP8_VLLM_COMPILATION_CONFIG)' \
		VLLM_MAX_NUM_SEQS="$(HFLC_FP8_MAX_NUM_SEQS)" \
	VLLM_KV_CACHE_MEMORY_BYTES="$(HFLC_FP8_VLLM_KV_CACHE_MEMORY_BYTES)" \
	VLLM_NUM_GPU_BLOCKS_OVERRIDE="$(HFLC_FP8_VLLM_NUM_GPU_BLOCKS_OVERRIDE)" \
	KV_CALIBRATION_GATE="$(HFLC_FP8_KV_CALIBRATION_GATE)" \
	KV_CALIBRATION_ONLY="$(HFLC_FP8_KV_CALIBRATION_ONLY)" \
	KV_CALIBRATION_HEADROOM_RATIO="$(HFLC_FP8_KV_CALIBRATION_HEADROOM_RATIO)" \
	KV_CALIBRATION_HEADROOM_MIN_BLOCKS="$(HFLC_FP8_KV_CALIBRATION_HEADROOM_MIN_BLOCKS)" \
	KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS="$(HFLC_FP8_KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS)" \
	KV_CALIBRATION_METRICS_TIMEOUT_SEC="$(HFLC_FP8_KV_CALIBRATION_METRICS_TIMEOUT_SEC)" \
	CRASH_CAPTURE="$(HFLC_FP8_CRASH_CAPTURE)" \
	CRASH_CAPTURE_SET_ULIMIT="$(HFLC_FP8_CRASH_CAPTURE_SET_ULIMIT)" \
	CRASH_CAPTURE_ENABLE_HSA_DEBUG="$(HFLC_FP8_CRASH_CAPTURE_ENABLE_HSA_DEBUG)" \
		FP8_COHERENCE_GUARDRAILS="$(HFLC_FP8_COHERENCE_GUARDRAILS)" \
	MAX_NEW_TOKENS="$(HFLC_FP8_MAX_NEW_TOKENS)" \
	POST_RUN_CACHE_CLEANUP="$(HFLC_FP8_POST_RUN_CACHE_CLEANUP)" \
	TEMPERATURE="$(HFLC_FP8_TEMPERATURE)" \
	TOP_P="$(HFLC_FP8_TOP_P)" \
	TOP_K="$(HFLC_FP8_TOP_K)" \
	REPETITION_PENALTY="$(HFLC_FP8_REPETITION_PENALTY)" \
	PRESENCE_PENALTY="$(HFLC_FP8_PRESENCE_PENALTY)" \
	FREQUENCY_PENALTY="$(HFLC_FP8_FREQUENCY_PENALTY)" \
	NUM_RUNS="$(HFLC_FP8_NUM_RUNS)" \
	WARMUP_DISCARD_REQUESTS="$(HFLC_FP8_WARMUP_DISCARD_REQUESTS)" \
	EMIT_PREFILL_PROGRESS="$(HFLC_FP8_EMIT_PREFILL_PROGRESS)" \
	PREFILL_PROGRESS_INTERVAL_SEC="$(HFLC_FP8_PREFILL_PROGRESS_INTERVAL_SEC)" \
	PREFILL_PROGRESS_TIMEOUT_SEC="$(HFLC_FP8_PREFILL_PROGRESS_TIMEOUT_SEC)" \
	ADAPTIVE_REQUEST_TIMEOUT=0 REQUEST_TIMEOUT_SEC="$(HFLC_FP8_REQUEST_TIMEOUT_SEC)" \
	TIMEOUT_SCLK_GUARD=1 TIMEOUT_SCLK_MIN_MHZ=1500 \
	TIMEOUT_SCLK_SAMPLE_COUNT=6 TIMEOUT_SCLK_SAMPLE_INTERVAL_SEC=5 \
	TIMEOUT_SCLK_MAX_EXTENSIONS=24 TIMEOUT_SCLK_EXTENSION_SEC=300 \
	SUBSTRATE_MAX_FILES=0 \
	SUBSTRATE_MAX_CHARS_PER_FILE="$(HFLC_FP8_MAX_CHARS_PER_FILE)" \
	PROMPT_SHAPE="$(HFLC_FP8_PROMPT_SHAPE)" \
	PROMPT_SHAPE_FILE="$(HFLC_FP8_PROMPT_SHAPE_FILE)" \
	PREFIX_SESSION_MODE="$(HFLC_FP8_PREFIX_SESSION_MODE)" \
	PREFIX_SESSION_ID="$(HFLC_FP8_PREFIX_SESSION_ID)" \
	PREFIX_SESSION_DIR="$(HFLC_FP8_PREFIX_SESSION_DIR)" \
	PREFIX_COMPAT_STRICT="$(HFLC_FP8_PREFIX_COMPAT_STRICT)" \
	SUFFIX_QUERY_FILE="$(HFLC_FP8_SUFFIX_QUERY_FILE)" \
	SUFFIX_QUERY_FILES="$(HFLC_FP8_SUFFIX_QUERY_FILES)" \
	SUFFIX_QUERY_DIR="$(HFLC_FP8_SUFFIX_QUERY_DIR)" \
	PREFIX_QUERY_MAX_COUNT="$(HFLC_FP8_PREFIX_QUERY_MAX_COUNT)" \
	PREFIX_RELOAD_BETWEEN_QUERIES="$(HFLC_FP8_PREFIX_RELOAD_BETWEEN_QUERIES)" \
	PREFIX_MAX_TOKENS="$(HFLC_FP8_PREFIX_MAX_TOKENS)" \
	PREFIX_SAVE_SHARDS="$(HFLC_FP8_PREFIX_SAVE_SHARDS)" \
		VLLM_ENABLE_PREFIX_CACHING="$(HFLC_FP8_VLLM_ENABLE_PREFIX_CACHING)" \
		VLLM_ROCM_AITER_MLA_USE_FLASHINFER_DECODE="$(VLLM_ROCM_AITER_MLA_USE_FLASHINFER_DECODE)" \
		FLASHINFER_HIP_MLA_USE_AITER_FASTPATH="$(FLASHINFER_HIP_MLA_USE_AITER_FASTPATH)" \
		FLASHINFER_HIP_MLA_USE_REPO_AITER_SNAPSHOT="$(FLASHINFER_HIP_MLA_USE_REPO_AITER_SNAPSHOT)" \
		FLASHINFER_HIP_MLA_GRAPH_KV_HEADROOM="$(FLASHINFER_HIP_MLA_GRAPH_KV_HEADROOM)" \
		PREFIX_TARGET_TP="$(HFLC_FP8_PREFIX_TARGET_TP)" \
		PREFIX_STATE_CHUNK_BYTES="$(HFLC_FP8_PREFIX_STATE_CHUNK_BYTES)" \
		PREFIX_STATE_CALIBRATION_STRICT_GATE="$(HFLC_FP8_PREFIX_STATE_CALIBRATION_STRICT_GATE)" \
	PREFIX_STATE_CALIBRATION_SAFETY_RATIO="$(HFLC_FP8_PREFIX_STATE_CALIBRATION_SAFETY_RATIO)" \
	PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES="$(HFLC_FP8_PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES)" \
	PHASE1_SWEEP="$(HFLC_FP8_SWEEP)" \
	PHASE1_MAX_MODEL_LEN="$$MAX_LEN" \
	PHASE1_ALLOW_LONG=1 \
	bash $(REPO_ROOT)/experiments/hf_long_context/driver_vllm_two_phase.sh; \
	$(MAKE) --no-print-directory hf-long-context-fp8-assert HFLC_FP8_RUN_DIR="$(HFLC_FP8_RUN_DIR)"

# Run multiple FP8 points in arbitrary user-specified order.
# Example:
#   make hf-long-context-fp8-run-list \
#     HFLC_FP8_SWEEP_LIST=104857600,67108864,33554432
#
# Prefix checkpointing with the campaign/run-list flow:
# - Use HFLC_FP8_PREFIX_SESSION_* here, not HFLC_FP8_CAPTURE_*.
# - HFLC_FP8_CAPTURE_* only applies to hf-long-context-fp8-capture.
# - This is the preferred way to build a single-length prefix session while
#   keeping the normal campaign logging/progress/report plumbing.
# Example:
#   make hf-long-context-fp8-run-list \
#     HFLC_FP8_SWEEP_LIST=33554432 \
#     HFLC_FP8_PREFIX_SESSION_MODE=build \
#     HFLC_FP8_PREFIX_SESSION_ID=prefix32m \
#     HFLC_FP8_PREFIX_SESSION_DIR=/app/long-context-serving/experiments/hf_long_context/prefix_sessions
hf-long-context-fp8-run-list: hflc-config-validate
	@set -euo pipefail; \
	HFLC_WRAPPER_ENTRYPOINT="$${HFLC_WRAPPER_ENTRYPOINT:-$@}"; \
	HFLC_WRAPPER_CHAIN="$${HFLC_WRAPPER_CHAIN:-}$@|"; \
	export HFLC_WRAPPER_ENTRYPOINT HFLC_WRAPPER_CHAIN; \
	LIST="$(HFLC_FP8_SWEEP_LIST)"; \
	if [ -z "$$LIST" ]; then \
		echo "Usage: make hf-long-context-fp8-run-list HFLC_FP8_SWEEP_LIST=104857600,67108864,..."; \
		exit 1; \
	fi; \
	idx=0; \
	IFS=',' read -r -a PTS <<< "$$LIST"; \
	for raw in "$${PTS[@]}"; do \
		pt="$$(echo "$$raw" | tr -d '[:space:]')"; \
		if ! [[ "$$pt" =~ ^[0-9]+$$ ]]; then \
			echo "Invalid token length: '$$raw'"; \
			exit 2; \
		fi; \
		port="$$(($(HFLC_FP8_PORT_BASE) + idx))"; \
		dp_port="$$(($(HFLC_FP8_DP_MASTER_PORT_BASE) + idx))"; \
		max_len="$(HFLC_FP8_MAX_MODEL_LEN)"; \
		if [ -z "$$max_len" ]; then \
			max_len="$$((pt + 16384))"; \
		fi; \
		if [ "$$pt" -le "$(HFLC_FP8_SMALL_POINT_THRESHOLD)" ]; then \
			point_num_runs="$(HFLC_FP8_SMALL_POINT_NUM_RUNS)"; \
		else \
			point_num_runs="$(HFLC_FP8_NUM_RUNS)"; \
		fi; \
		if [ "$$pt" -le "$(HFLC_FP8_WARMUP_DISCARD_THRESHOLD)" ]; then \
			point_warmup_discard="$(HFLC_FP8_WARMUP_DISCARD_REQUESTS)"; \
		else \
			point_warmup_discard=0; \
		fi; \
		pass_n="$(HFLC_FP8_PASS_INDEX)"; \
		if [ -z "$$pass_n" ] || [ "$$pass_n" -lt 1 ]; then pass_n=1; fi; \
		run_id="$(HFLC_FP8_RUN_ID)_$${pt}_pass$${pass_n}_$$(date -u +%Y%m%dT%H%M%SZ)"; \
		run_dir="$(HFLC_RUNS_DIR)/$$run_id"; \
		echo "==> point=$$pt pass=$$pass_n run_id=$$run_id port=$$port dp_port=$$dp_port max_model_len=$$max_len num_runs=$$point_num_runs warmup_discard=$$point_warmup_discard"; \
		$(MAKE) --no-print-directory hf-long-context-fp8-run \
			HFLC_FP8_RUN_ID="$$run_id" \
			HFLC_FP8_SWEEP="$$pt" \
			HFLC_FP8_MAX_MODEL_LEN="$$max_len" \
			HFLC_FP8_PORT="$$port" \
			HFLC_FP8_DP_MASTER_PORT="$$dp_port" \
			HFLC_FP8_MODEL_ID="$(HFLC_FP8_MODEL_ID)" \
			HFLC_FP8_MAX_NEW_TOKENS="$(HFLC_FP8_MAX_NEW_TOKENS)" \
			HFLC_FP8_TEMPERATURE="$(HFLC_FP8_TEMPERATURE)" \
			HFLC_FP8_TOP_P="$(HFLC_FP8_TOP_P)" \
			HFLC_FP8_TOP_K="$(HFLC_FP8_TOP_K)" \
			HFLC_FP8_REPETITION_PENALTY="$(HFLC_FP8_REPETITION_PENALTY)" \
			HFLC_FP8_PRESENCE_PENALTY="$(HFLC_FP8_PRESENCE_PENALTY)" \
			HFLC_FP8_FREQUENCY_PENALTY="$(HFLC_FP8_FREQUENCY_PENALTY)" \
			HFLC_FP8_POST_RUN_CACHE_CLEANUP="$(HFLC_FP8_POST_RUN_CACHE_CLEANUP)" \
			HFLC_FP8_NUM_RUNS="$$point_num_runs" \
			HFLC_FP8_WARMUP_DISCARD_REQUESTS="$$point_warmup_discard" \
			HFLC_FP8_REQUEST_TIMEOUT_SEC="$(HFLC_FP8_REQUEST_TIMEOUT_SEC)" \
				HFLC_FP8_EMIT_PREFILL_PROGRESS="$(HFLC_FP8_EMIT_PREFILL_PROGRESS)" \
					HFLC_FP8_PROMPT_SHAPE="$(HFLC_FP8_PROMPT_SHAPE)" \
					HFLC_FP8_PROMPT_SHAPE_FILE="$(HFLC_FP8_PROMPT_SHAPE_FILE)" \
					HFLC_FP8_SUFFIX_QUERY_FILE="$(HFLC_FP8_SUFFIX_QUERY_FILE)" \
					HFLC_FP8_SUFFIX_QUERY_FILES="$(HFLC_FP8_SUFFIX_QUERY_FILES)" \
					HFLC_FP8_SUFFIX_QUERY_DIR="$(HFLC_FP8_SUFFIX_QUERY_DIR)" \
					HFLC_FP8_PREFIX_QUERY_MAX_COUNT="$(HFLC_FP8_PREFIX_QUERY_MAX_COUNT)" \
					HFLC_FP8_PREFIX_RELOAD_BETWEEN_QUERIES="$(HFLC_FP8_PREFIX_RELOAD_BETWEEN_QUERIES)" \
					HFLC_FP8_PREFILL_PROGRESS_INTERVAL_SEC="$(HFLC_FP8_PREFILL_PROGRESS_INTERVAL_SEC)" \
					HFLC_FP8_PREFILL_PROGRESS_TIMEOUT_SEC="$(HFLC_FP8_PREFILL_PROGRESS_TIMEOUT_SEC)" \
					HFLC_FP8_GPU_LIST="$(HFLC_FP8_GPU_LIST)" \
					HFLC_FP8_GPU_MEMORY_UTILIZATION="$(HFLC_FP8_GPU_MEMORY_UTILIZATION)" \
					HFLC_FP8_VLLM_DTYPE="$(HFLC_FP8_VLLM_DTYPE)" \
					HFLC_FP8_KV_CACHE_DTYPE="$(HFLC_FP8_KV_CACHE_DTYPE)" \
					HFLC_FP8_MAX_CHARS_PER_FILE="$(HFLC_FP8_MAX_CHARS_PER_FILE)" \
					HFLC_FP8_TP="$(HFLC_FP8_TP)" \
					HFLC_FP8_DP="$(HFLC_FP8_DP)" \
					HFLC_FP8_PP="$(HFLC_FP8_PP)" \
			HFLC_FP8_ENFORCE_EAGER="$(HFLC_FP8_ENFORCE_EAGER)" \
			HFLC_FP8_DISABLE_ROCM_SKINNY_GEMM="$(HFLC_FP8_DISABLE_ROCM_SKINNY_GEMM)" \
				HFLC_FP8_VLLM_COMPILATION_CONFIG='$(HFLC_FP8_VLLM_COMPILATION_CONFIG)' \
				HFLC_FP8_COHERENCE_GUARDRAILS="$(HFLC_FP8_COHERENCE_GUARDRAILS)" \
				HFLC_FP8_REQUIRE_LAUNCH_CONTRACT="$(HFLC_FP8_REQUIRE_LAUNCH_CONTRACT)" \
				HFLC_FP8_CRASH_CAPTURE="$(HFLC_FP8_CRASH_CAPTURE)" \
				HFLC_FP8_CRASH_CAPTURE_SET_ULIMIT="$(HFLC_FP8_CRASH_CAPTURE_SET_ULIMIT)" \
				HFLC_FP8_CRASH_CAPTURE_ENABLE_HSA_DEBUG="$(HFLC_FP8_CRASH_CAPTURE_ENABLE_HSA_DEBUG)" \
				HFLC_SUBSTRATE_ROOT="$(HFLC_SUBSTRATE_ROOT)"; \
		$(MAKE) --no-print-directory hf-long-context-fp8-report \
			HFLC_FP8_RUN_DIR="$$run_dir" \
			HFLC_FP8_CAMPAIGN_DIR="$(HFLC_FP8_CAMPAIGN_DIR)" \
			HFLC_APPEND_TO_CAMPAIGN=1 \
			HFLC_FIT_CUTOFF_TOKENS="$(HFLC_FIT_CUTOFF_TOKENS)"; \
			idx="$$((idx + 1))"; \
		done

hf-long-context-fp8-assert:
	@RUN_DIR="$(HFLC_FP8_RUN_DIR)"; \
	PHASE_DIR="$$RUN_DIR/phase1"; \
	LAUNCH_ENV="$$PHASE_DIR/launch_env.json"; \
	RUN_META="$$PHASE_DIR/run_meta.json"; \
	if [ ! -f "$$LAUNCH_ENV" ]; then \
		echo "Missing launch env artifact: $$LAUNCH_ENV"; \
		exit 1; \
	fi; \
	if [ ! -f "$$RUN_META" ]; then \
		echo "Missing run metadata artifact: $$RUN_META"; \
		exit 2; \
	fi; \
	python3 -c 'import json,sys; m=json.load(open(sys.argv[1],"r",encoding="utf-8")); v=m.get("vllm") or {}; bad=[]; backend=str(v.get("attention_backend","")); kv=str(v.get("kv_cache_dtype","")); backend=="ROCM_AITER_MLA" or bad.append(("attention_backend",backend,"ROCM_AITER_MLA")); kv=="fp8_e4m3" or bad.append(("kv_cache_dtype",kv,"fp8_e4m3")); [print(f"ASSERT FAIL {k}: got={g!r} expected={e!r}") for k,g,e in bad]; sys.exit(3 if bad else 0)' "$$RUN_META" || exit $$?; \
	echo "ASSERT OK: served backend and KV dtype match the measured recipe"; \
	SERVER_LOG="$$PHASE_DIR/vllm_server.log"; \
	if [ ! -f "$$SERVER_LOG" ]; then \
		echo "Missing server log artifact: $$SERVER_LOG"; \
		exit 4; \
	fi; \
	if [ "$(HFLC_FP8_REQUIRE_LAUNCH_CONTRACT)" = "1" ]; then \
		python3 app/scripts/hflc_assert_launch_contract.py "$$LAUNCH_ENV" "$$RUN_META" || exit $$?; \
		echo "ASSERT OK: launch contract in launch_env.json"; \
	fi; \
	if [ "$(HFLC_FP8_KV_CALIBRATION_GATE)" = "1" ]; then \
		KV_JSON="$$PHASE_DIR/kv_calibration.json"; \
		if [ ! -f "$$KV_JSON" ]; then \
			echo "ASSERT FAIL missing calibration artifact: $$KV_JSON"; \
			exit 6; \
		fi; \
		python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p,'r',encoding='utf-8')); s=d.get('summary') or {}; ok=bool(s.get('all_pass',False)); print(f\"KV calibration summary: all_pass={ok} fail_count={int(s.get('fail_count',0))} target_count={int(s.get('target_count',0))}\"); sys.exit(0 if ok else 7)" "$$KV_JSON"; \
		echo "ASSERT OK: calibration artifact exists and all targets passed"; \
	fi; \
	if [ "$(HFLC_FP8_EMIT_PREFILL_PROGRESS)" = "1" ]; then \
		PREFILL_JSONL="$$PHASE_DIR/prefill_progress.jsonl"; \
		if [ ! -s "$$PREFILL_JSONL" ]; then \
			echo "ASSERT FAIL missing or empty prefill progress artifact: $$PREFILL_JSONL"; \
			exit 8; \
		fi; \
		echo "ASSERT OK: prefill progress artifact exists"; \
	fi

hf-long-context-fp8-gpu:
	watch -n 10 'rocm-smi --showuse --showmemuse --showclocks | sed -n "1,160p"'

hf-long-context-fp8-report:
	@REPORT_PYTHON="python3"; \
	if [ -n "$(HFLC_VLLM_VENV_NAME)" ] && [ -x "$(REPO_ROOT)/.venvs/$(HFLC_VLLM_VENV_NAME)/bin/python3" ]; then \
		REPORT_PYTHON="$(REPO_ROOT)/.venvs/$(HFLC_VLLM_VENV_NAME)/bin/python3"; \
	elif [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python3" ]; then \
		REPORT_PYTHON="$$VIRTUAL_ENV/bin/python3"; \
	fi; \
	RUN_DIR="$(HFLC_FP8_RUN_DIR)"; \
	PHASE_DIR="$$RUN_DIR/phase1"; \
	if [ ! -d "$$PHASE_DIR" ]; then \
		echo "Missing phase dir: $$PHASE_DIR"; \
		exit 1; \
	fi; \
	if [ -f "$$PHASE_DIR/summary.json" ]; then \
		PYTHONPATH="$(REPO_ROOT)/app${PYTHONPATH:+:$$PYTHONPATH}" "$$REPORT_PYTHON" $(REPO_ROOT)/app/scripts/build_hf_long_context_report.py --run-dir "$$PHASE_DIR"; \
	else \
		echo "Skipping single-run report for in-flight phase (missing $$PHASE_DIR/summary.json)"; \
	fi; \
	if [ "$(HFLC_APPEND_TO_CAMPAIGN)" = "1" ]; then \
		REL_PHASE_DIR="$$(realpath --relative-to="$(REPO_ROOT)" "$$PHASE_DIR")"; \
		PYTHONPATH="$(REPO_ROOT)/app${PYTHONPATH:+:$$PYTHONPATH}" "$$REPORT_PYTHON" $(REPO_ROOT)/app/scripts/build_hf_long_context_consolidated_report.py \
			--campaign-dir "$(HFLC_FP8_CAMPAIGN_DIR)" \
			--root "$(REPO_ROOT)" \
			--append-phase-dir "$$REL_PHASE_DIR" \
			--fit-cutoff-tokens "$(HFLC_FIT_CUTOFF_TOKENS)" \
			--baseline-report-md "$(HFLC_FP8_BASELINE_REPORT_MD)" \
			--baseline-label "$(HFLC_FP8_BASELINE_LABEL)" \
			--current-label "$(HFLC_FP8_CURRENT_LABEL)" \
			$(if $(strip $(HFLC_FP8_AUTO_WRAPPER_TARGET)),--auto-wrapper-target "$(HFLC_FP8_AUTO_WRAPPER_TARGET)" --auto-wrapper-run-id-var "$(HFLC_FP8_AUTO_WRAPPER_RUN_ID_VAR)",); \
	fi

hf-long-context-fp8-queue-tail:
	@if [ ! -d "$(HFLC_FP8_CAMPAIGN_DIR)" ]; then \
		echo "Missing campaign dir: $(HFLC_FP8_CAMPAIGN_DIR)"; \
		exit 1; \
	fi
	tail -n +1 -f "$(HFLC_FP8_CAMPAIGN_DIR)/queue_fp8_20m_32m_64m_100m_guarded.nohup.log" "$(HFLC_FP8_CAMPAIGN_DIR)/campaign.log"
