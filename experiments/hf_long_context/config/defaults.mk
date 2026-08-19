# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# HFLC (HuggingFace Long Context) defaults and profiles.
# Loaded by top-level Makefile after REPO_ROOT is resolved.

# @doc HFLC_PROFILE|Execution profile preset: smoke, long, debug, capture.
HFLC_PROFILE ?= long

# Paths
# @doc HFLC_DIR|HF long-context workspace directory.
HFLC_DIR ?= $(REPO_ROOT)/experiments/hf_long_context
# @doc HFLC_RUNS_DIR|Directory where HFLC run artifacts are written.
HFLC_RUNS_DIR ?= $(HFLC_DIR)/runs
# @doc HFLC_SUBSTRATE_ROOT|Prompt substrate root used to build deterministic context prompts.
HFLC_SUBSTRATE_ROOT ?= $(REPO_ROOT)/data/substrate

# FP8 campaign core
# @doc HFLC_FP8_MODEL_ID|Model id/path passed to vLLM.
# Default to the canonical repo id so Docker/devcontainer flows do not depend on
# a particular local snapshot mount such as /hf/....
HFLC_FP8_MODEL_ID ?= moonshotai/Kimi-Linear-48B-A3B-Instruct
# @doc HFLC_FP8_TOKENIZER_ID|Optional tokenizer id/path override used by the benchmark client.
HFLC_FP8_TOKENIZER_ID ?=
# @doc HFLC_FP8_RUN_ID|Run id prefix for FP8 runs.
HFLC_FP8_RUN_ID ?= kimi_fp8_$(shell date -u +%Y%m%dT%H%M%SZ)
# @doc HFLC_FP8_RUN_DIR|Run directory for the current run id.
HFLC_FP8_RUN_DIR ?= $(HFLC_RUNS_DIR)/$(HFLC_FP8_RUN_ID)
# @doc HFLC_FP8_SWEEP|Comma-separated prompt token lengths for a single fp8 run target.
HFLC_FP8_SWEEP ?= 16777216
# @doc HFLC_FP8_SWEEP_LIST|Comma-separated prompt token lengths for ordered list execution.
HFLC_FP8_SWEEP_LIST ?=
# @doc HFLC_FP8_MAX_MODEL_LEN|Optional explicit max model length; defaults to prompt+16384.
HFLC_FP8_MAX_MODEL_LEN ?=
# @doc HFLC_FP8_PORT|vLLM service port for single-point run.
HFLC_FP8_PORT ?= 18841
# @doc HFLC_FP8_DP_MASTER_PORT|vLLM data-parallel master port.
HFLC_FP8_DP_MASTER_PORT ?= 19841
# @doc HFLC_FP8_PORT_BASE|Base service port for list runs.
HFLC_FP8_PORT_BASE ?= 18860
# @doc HFLC_FP8_DP_MASTER_PORT_BASE|Base DP master port for list runs.
HFLC_FP8_DP_MASTER_PORT_BASE ?= 19860
# @doc HFLC_FP8_MAX_NEW_TOKENS|Max generated tokens per request.
HFLC_FP8_MAX_NEW_TOKENS ?= 4096
# @doc HFLC_FP8_MAX_NUM_SEQS|vLLM max_num_seqs setting.
HFLC_FP8_MAX_NUM_SEQS ?= 1
# @doc HFLC_FP8_TP|Tensor parallel size.
HFLC_FP8_TP ?= 8
# @doc HFLC_FP8_DP|Data parallel size.
HFLC_FP8_DP ?= 1
# @doc HFLC_FP8_PP|Pipeline parallel size.
HFLC_FP8_PP ?= 1
# @doc HFLC_FP8_GPU_LIST|Visible GPU list.
HFLC_FP8_GPU_LIST ?= 0,1,2,3,4,5,6,7
# @doc HFLC_FP8_VLLM_DTYPE|vLLM model compute dtype for FP8 long-context flows.
HFLC_FP8_VLLM_DTYPE ?= bfloat16
# @doc HFLC_FP8_KV_CACHE_DTYPE|vLLM KV cache dtype for FP8 long-context flows.
HFLC_FP8_KV_CACHE_DTYPE ?= fp8_e4m3

# Runtime/compilation behavior
# @doc HFLC_FP8_ENFORCE_EAGER|Set vLLM enforce-eager (0/1).
HFLC_FP8_ENFORCE_EAGER ?= 0
# @doc HFLC_FP8_DISABLE_ROCM_SKINNY_GEMM|Pass --disable-rocm-skinny-gemm to the benchmark launcher (0/1).
HFLC_FP8_DISABLE_ROCM_SKINNY_GEMM ?= 0
# @doc HFLC_FP8_DISABLE_CUSTOM_ALL_REDUCE|Pass --vllm-disable-custom-all-reduce to the benchmark launcher (0/1).
HFLC_FP8_DISABLE_CUSTOM_ALL_REDUCE ?= 0
# @doc HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB|Override ROCm custom-allreduce max tensor size in MiB (0 keeps vLLM default).
HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB ?= 0
# @doc HFLC_FP8_VLLM_COMPILATION_CONFIG|JSON passed to --compilation-config.
HFLC_FP8_VLLM_COMPILATION_CONFIG ?=
# @doc HFLC_FP8_COHERENCE_GUARDRAILS|Pin launch settings that affect the measurement (currently max_num_seqs) to the measured recipe (0/1).
HFLC_FP8_COHERENCE_GUARDRAILS ?= 1
# @doc HFLC_FP8_REQUIRE_LAUNCH_CONTRACT|Assert the recorded launch contract (prompt shape, eager mode) after each run (0/1).
HFLC_FP8_REQUIRE_LAUNCH_CONTRACT ?= 0

# Calibration + KV pool controls
# @doc HFLC_FP8_KV_CALIBRATION_GATE|Enable KV calibration gate before run.
HFLC_FP8_KV_CALIBRATION_GATE ?= 1
# @doc HFLC_FP8_KV_CALIBRATION_ONLY|Run calibration only.
HFLC_FP8_KV_CALIBRATION_ONLY ?= 0
# @doc HFLC_FP8_KV_CALIBRATION_HEADROOM_RATIO|Headroom ratio for calibration gate.
HFLC_FP8_KV_CALIBRATION_HEADROOM_RATIO ?= 0.005
# @doc HFLC_FP8_KV_CALIBRATION_HEADROOM_MIN_BLOCKS|Minimum spare blocks required.
HFLC_FP8_KV_CALIBRATION_HEADROOM_MIN_BLOCKS ?= 2
# @doc HFLC_FP8_KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS|Assumed max_num_seqs during gate (0 means actual setting).
HFLC_FP8_KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS ?= 0
# @doc HFLC_FP8_KV_CALIBRATION_METRICS_TIMEOUT_SEC|Timeout for calibration metric probing.
HFLC_FP8_KV_CALIBRATION_METRICS_TIMEOUT_SEC ?= 180
# @doc HFLC_FP8_REQUIRE_KV_CALIBRATION|Require gate enabled for benchmark runs.
HFLC_FP8_REQUIRE_KV_CALIBRATION ?= 1
# @doc HFLC_FP8_VLLM_KV_CACHE_MEMORY_BYTES|Optional explicit --kv-cache-memory-bytes.
HFLC_FP8_VLLM_KV_CACHE_MEMORY_BYTES ?= 0
# @doc HFLC_FP8_VLLM_NUM_GPU_BLOCKS_OVERRIDE|Optional explicit --num-gpu-blocks-override.
HFLC_FP8_VLLM_NUM_GPU_BLOCKS_OVERRIDE ?= 0

# Request execution controls
# @doc HFLC_FP8_NUM_RUNS|Measured runs per point (the floor applied to every point; large points stay at this value).
HFLC_FP8_NUM_RUNS ?= 1
# @doc HFLC_FP8_SMALL_POINT_NUM_RUNS|Measured runs for points at or below HFLC_FP8_SMALL_POINT_THRESHOLD. Small-prompt TTFT/decode is dominated by sub-100ms request overhead + single-sample jitter, so these cheap points are repeated and the report uses the median. Repeats run in-server (no relaunch), so the added cost is a few seconds. Large points keep HFLC_FP8_NUM_RUNS since their measured request is expensive and they are compute-bound/stable.
HFLC_FP8_SMALL_POINT_NUM_RUNS ?= 5
# @doc HFLC_FP8_SMALL_POINT_THRESHOLD|Prompt-token threshold (inclusive) below/at which HFLC_FP8_SMALL_POINT_NUM_RUNS applies. Default 65536 (64Ki).
HFLC_FP8_SMALL_POINT_THRESHOLD ?= 65536
# @doc HFLC_FP8_PASS_INDEX|Which cold-start pass this run-list invocation represents (1-based). Set by the campaign sweep, which loops the whole point list once per pass so each pass of a given point is separated by a full sweep (lets large points fully release GPU/distributed resources between cold starts). Stamped into the per-run run_id as passN. Default 1.
HFLC_FP8_PASS_INDEX ?= 1
# @doc HFLC_FP8_WARMUP_DISCARD_REQUESTS|Untimed warm-up requests issued (and discarded) before the first measured iteration, applied ONLY to points at/below HFLC_FP8_WARMUP_DISCARD_THRESHOLD. The first request after a server launch pays a one-time CUDA-graph-capture/compile cost for the new prompt-token shape (~tens of seconds, roughly constant); discarding it keeps the measured TTFT at steady state. Set 0 to disable everywhere. Default 1.
HFLC_FP8_WARMUP_DISCARD_REQUESTS ?= 1
# @doc HFLC_FP8_WARMUP_DISCARD_THRESHOLD|Prompt-token threshold (inclusive) below/at which the warm-up discard applies. The discard's cost is one full prefill at that length, while its benefit is removing the ~constant cold-capture overhead — so it is worth it only where steady prefill is small relative to that overhead. Above this threshold the steady prefill dwarfs the cold cost and a discard would nearly double wall-clock (e.g. 64Mi would run its >10h prefill twice per pass), so it is skipped. Default 524288 (512Ki); the steady prefill crosses the ~tens-of-seconds capture cost around 256Ki-512Ki.
HFLC_FP8_WARMUP_DISCARD_THRESHOLD ?= 4194304
# @doc HFLC_FP8_REQUEST_TIMEOUT_SEC|Per-request timeout in seconds.
HFLC_FP8_REQUEST_TIMEOUT_SEC ?= 345600
# @doc HFLC_FP8_GPU_MEMORY_UTILIZATION|vLLM gpu-memory-utilization.
HFLC_FP8_GPU_MEMORY_UTILIZATION ?= 0.952

# Progress/monitoring
# @doc HFLC_FP8_EMIT_PREFILL_PROGRESS|Poll the vLLM prefill-progress debug endpoint and emit prefill_progress.jsonl. Off by default: that debug endpoint is not built into the public image, so polling it only logs a harmless 404. Has no effect on the TTFT/decode measurements. Set 1 only against an image that bakes the endpoint.
HFLC_FP8_EMIT_PREFILL_PROGRESS ?= 0
# @doc HFLC_FP8_PREFILL_PROGRESS_INTERVAL_SEC|Prefill polling interval.
HFLC_FP8_PREFILL_PROGRESS_INTERVAL_SEC ?= 20
# @doc HFLC_FP8_PREFILL_PROGRESS_TIMEOUT_SEC|Prefill polling timeout (0 uses request timeout).
HFLC_FP8_PREFILL_PROGRESS_TIMEOUT_SEC ?= 0

# Crash-capture controls (default OFF to avoid large core/gpucore dumps).
# @doc HFLC_FP8_CRASH_CAPTURE|Enable crash capture collection (0/1).
HFLC_FP8_CRASH_CAPTURE ?= 0
# @doc HFLC_FP8_CRASH_CAPTURE_SET_ULIMIT|Set ulimit -c unlimited during preflight (0/1).
HFLC_FP8_CRASH_CAPTURE_SET_ULIMIT ?= 0
# @doc HFLC_FP8_CRASH_CAPTURE_ENABLE_HSA_DEBUG|Enable HSA debug for richer GPU core dumps (0/1).
HFLC_FP8_CRASH_CAPTURE_ENABLE_HSA_DEBUG ?= 0

# Prompt construction
# @doc HFLC_FP8_MAX_CHARS_PER_FILE|Prompt builder per-file char cap in bytes (0 disables cap).
# Default 65536 (64 KiB) defends against an accidentally-quadratic re-tokenize
# loop in `_truncate_block_with_footer` (app/long_context_serving/benchmark_common.py)
# when the substrate cumulative budget lands on a many-line multi-MB file
# (e.g. datashader/examples/data/.data_stubs/nyc_taxi.csv). At
# sweep points >= ~4M tokens, no cap means substrate-build can wedge for
# many hours before vLLM even starts. Most real source files are under
# 64 KiB so this is a no-op for them. Override to 0 only after
# `_truncate_block_with_footer` has been made non-quadratic.
HFLC_FP8_MAX_CHARS_PER_FILE ?= 65536
# @doc HFLC_FP8_PROMPT_TOKEN_IDS_FILE|Optional JSON artifact containing exact prompt token IDs for replay through the native benchmark flow.
HFLC_FP8_PROMPT_TOKEN_IDS_FILE ?=
# @doc HFLC_FP8_PROMPT_SHAPE|Prompt shape id: benchmark_v1, repo_grounded_en_v1, repo_grounded_en_v2, custom_file.
HFLC_FP8_PROMPT_SHAPE ?= benchmark_v1
# @doc HFLC_FP8_PROMPT_SHAPE_FILE|Prompt shape JSON file when shape=custom_file.
HFLC_FP8_PROMPT_SHAPE_FILE ?=
# @doc HFLC_FP8_QUERY_API_MODE|Direct-request API mode: auto, messages, or prompt_ids.
HFLC_FP8_QUERY_API_MODE ?= auto

# Prefix session controls
# @doc HFLC_FP8_PREFIX_SESSION_MODE|Prefix session mode: none/build/restore/query/build_and_query.
HFLC_FP8_PREFIX_SESSION_MODE ?= none
# @doc HFLC_FP8_PREFIX_SESSION_ID|Prefix session id.
HFLC_FP8_PREFIX_SESSION_ID ?=
# @doc HFLC_FP8_PREFIX_SESSION_DIR|Prefix session state directory.
HFLC_FP8_PREFIX_SESSION_DIR ?=
# @doc HFLC_FP8_PREFIX_COMPAT_STRICT|Prefix compatibility strictness (1 strict).
HFLC_FP8_PREFIX_COMPAT_STRICT ?= 1
# @doc HFLC_FP8_SUFFIX_QUERY_FILE|Single suffix query file.
HFLC_FP8_SUFFIX_QUERY_FILE ?=
# @doc HFLC_FP8_SUFFIX_QUERY_FILES|Comma-separated suffix query files.
HFLC_FP8_SUFFIX_QUERY_FILES ?=
# @doc HFLC_FP8_SUFFIX_QUERY_DIR|Suffix query directory.
HFLC_FP8_SUFFIX_QUERY_DIR ?=
# @doc HFLC_FP8_PREFIX_QUERY_MAX_COUNT|Max suffix queries to execute from set.
HFLC_FP8_PREFIX_QUERY_MAX_COUNT ?= 0
# @doc HFLC_FP8_PREFIX_RELOAD_BETWEEN_QUERIES|Reload prefix state between suffix queries (0/1).
HFLC_FP8_PREFIX_RELOAD_BETWEEN_QUERIES ?= 0
# @doc HFLC_FP8_PREFIX_MAX_TOKENS|Explicit prefix max tokens guard.
HFLC_FP8_PREFIX_MAX_TOKENS ?= 0
# @doc HFLC_FP8_PREFIX_SAVE_SHARDS|Persist prefix state shards (0/1).
HFLC_FP8_PREFIX_SAVE_SHARDS ?= 1
# @doc HFLC_FP8_VLLM_ENABLE_PREFIX_CACHING|Pass --enable-prefix-caching to vLLM server (0/1).
HFLC_FP8_VLLM_ENABLE_PREFIX_CACHING ?= 0
# @doc HFLC_FP8_PREFIX_TARGET_TP|Target TP for future reshard path.
HFLC_FP8_PREFIX_TARGET_TP ?= 0
# @doc HFLC_FP8_PREFIX_STATE_CHUNK_BYTES|Prefix state dump chunk bytes.
HFLC_FP8_PREFIX_STATE_CHUNK_BYTES ?= 67108864
# @doc HFLC_FP8_PREFIX_STATE_CALIBRATION_STRICT_GATE|Fail capture on unsafe memory headroom.
HFLC_FP8_PREFIX_STATE_CALIBRATION_STRICT_GATE ?= 1
# @doc HFLC_FP8_PREFIX_STATE_CALIBRATION_SAFETY_RATIO|Safety headroom ratio for capture.
HFLC_FP8_PREFIX_STATE_CALIBRATION_SAFETY_RATIO ?= 0.10
# @doc HFLC_FP8_PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES|Minimum safety bytes for capture.
HFLC_FP8_PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES ?= 2147483648
# @doc HFLC_FP8_TARGET_PROMPT_TOKENS|Optional explicit target prompt token count.
HFLC_FP8_TARGET_PROMPT_TOKENS ?=
# @doc HFLC_FP8_ENGLISH_ONLY_MASK_MODE|Generation-time English-only mask mode: off, bias, or hard.
HFLC_FP8_ENGLISH_ONLY_MASK_MODE ?= hard
# @doc HFLC_FP8_ENGLISH_ONLY_MASK_BIAS|Bias value used when HFLC_FP8_ENGLISH_ONLY_MASK_MODE=bias.
HFLC_FP8_ENGLISH_ONLY_MASK_BIAS ?= 100.0
# @doc HFLC_FP8_DUMP_REQUEST_PAYLOAD|Persist request payload debug artifacts (0/1).
HFLC_FP8_DUMP_REQUEST_PAYLOAD ?= 0
# @doc HFLC_FP8_DUMP_QUERY_MESSAGES|Persist rendered query-message debug artifacts (0/1).
HFLC_FP8_DUMP_QUERY_MESSAGES ?= 0
# @doc HFLC_FP8_DUMP_PROMPT_TOKEN_IDS|Persist prompt-token debug artifacts (0/1).
HFLC_FP8_DUMP_PROMPT_TOKEN_IDS ?= 0
# @doc HFLC_FP8_NO_STREAM|Use non-streaming HTTP completion requests in the native benchmark flow (0/1).
HFLC_FP8_NO_STREAM ?= 0

# Capture workflow defaults
# @doc HFLC_FP8_CAPTURE_MODE|Capture helper mode: build, restore, query, build_and_query, none.
HFLC_FP8_CAPTURE_MODE ?= build
# @doc HFLC_FP8_CAPTURE_SESSION_ID|Capture helper session id.
HFLC_FP8_CAPTURE_SESSION_ID ?=
# @doc HFLC_FP8_CAPTURE_SESSION_DIR|Capture helper session directory.
HFLC_FP8_CAPTURE_SESSION_DIR ?= $(HFLC_DIR)/prefix_sessions
# @doc HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILE|Capture helper single suffix query file.
HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILE ?=
# @doc HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILES|Capture helper suffix query files list.
HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILES ?=
# @doc HFLC_FP8_CAPTURE_SUFFIX_QUERY_DIR|Capture helper suffix query directory.
HFLC_FP8_CAPTURE_SUFFIX_QUERY_DIR ?=
# @doc HFLC_FP8_CAPTURE_QUERY_MAX_COUNT|Capture helper max suffix queries.
HFLC_FP8_CAPTURE_QUERY_MAX_COUNT ?= 0
# @doc HFLC_FP8_CAPTURE_RELOAD_BETWEEN_QUERIES|Capture helper reload behavior.
HFLC_FP8_CAPTURE_RELOAD_BETWEEN_QUERIES ?= 0
# @doc HFLC_FP8_CAPTURE_PREFIX_MAX_TOKENS|Capture helper prefix max token cap.
HFLC_FP8_CAPTURE_PREFIX_MAX_TOKENS ?= 0
# @doc HFLC_FP8_CAPTURE_TARGET_PROMPT_TOKENS|Capture helper target prompt tokens.
HFLC_FP8_CAPTURE_TARGET_PROMPT_TOKENS ?=

# Campaign/report helpers
# @doc HFLC_FP8_CAMPAIGN_DIR|Consolidated campaign directory.
HFLC_FP8_CAMPAIGN_DIR ?= $(HFLC_RUNS_DIR)/$(HFLC_FP8_RUN_ID)_campaign

# Misc reporting controls
# @doc HFLC_FIT_CUTOFF_TOKENS|Cutoff token length used for fit overlays.
HFLC_FIT_CUTOFF_TOKENS ?= 65536
# @doc HFLC_FP8_BASELINE_REPORT_MD|Optional reference report.md for TTFT/decode comparison plots. Empty by default; set it to a prior campaign's report.md to overlay a baseline.
HFLC_FP8_BASELINE_REPORT_MD ?=
# @doc HFLC_FP8_BASELINE_LABEL|Label for the default lower-target reference in comparison plots.
HFLC_FP8_BASELINE_LABEL ?= Triton mfma_head4 (QK-fused, PV-split)
# @doc HFLC_FP8_CURRENT_LABEL|Label for the current campaign in comparison plots.
HFLC_FP8_CURRENT_LABEL ?= Current campaign
# @doc HFLC_APPEND_TO_CAMPAIGN|Append run to campaign consolidated report (0/1).
HFLC_APPEND_TO_CAMPAIGN ?= 0
# @doc HFLC_WRAPPER_ENTRYPOINT|Name of the Make target the user actually invoked at the top of the call chain. Stamped by each wrapper recipe via :- defaulting so the outermost wrapper wins. Propagated through env to driver_vllm_two_phase.sh -> benchmark_vllm_long_context.py and recorded in run_meta.json so consolidated reports can print the true reproducibility command instead of guessing.
HFLC_WRAPPER_ENTRYPOINT ?=
# @doc HFLC_WRAPPER_CHAIN|Pipe-separated chain of wrapper targets traversed during recursive $(MAKE) invocations (outer -> inner). Recorded alongside HFLC_WRAPPER_ENTRYPOINT in run_meta.json for diagnostic visibility.
HFLC_WRAPPER_CHAIN ?=

# Canonical variable list used by help/print/validate targets.
HFLC_CONFIG_VARS := \
  HFLC_PROFILE HFLC_DIR HFLC_RUNS_DIR HFLC_SUBSTRATE_ROOT \
  HFLC_FP8_MODEL_ID HFLC_FP8_TOKENIZER_ID HFLC_FP8_RUN_ID HFLC_FP8_RUN_DIR HFLC_FP8_SWEEP HFLC_FP8_SWEEP_LIST HFLC_FP8_MAX_MODEL_LEN \
  HFLC_FP8_PORT HFLC_FP8_DP_MASTER_PORT HFLC_FP8_PORT_BASE HFLC_FP8_DP_MASTER_PORT_BASE \
  HFLC_FP8_MAX_NEW_TOKENS HFLC_FP8_MAX_NUM_SEQS HFLC_FP8_TP HFLC_FP8_DP HFLC_FP8_PP HFLC_FP8_GPU_LIST HFLC_FP8_VLLM_DTYPE HFLC_FP8_KV_CACHE_DTYPE \
  HFLC_FP8_ENFORCE_EAGER HFLC_FP8_DISABLE_ROCM_SKINNY_GEMM HFLC_FP8_DISABLE_CUSTOM_ALL_REDUCE HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB HFLC_FP8_VLLM_COMPILATION_CONFIG HFLC_FP8_COHERENCE_GUARDRAILS HFLC_FP8_REQUIRE_LAUNCH_CONTRACT \
  HFLC_FP8_KV_CALIBRATION_GATE HFLC_FP8_KV_CALIBRATION_ONLY HFLC_FP8_KV_CALIBRATION_HEADROOM_RATIO HFLC_FP8_KV_CALIBRATION_HEADROOM_MIN_BLOCKS \
  HFLC_FP8_KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS HFLC_FP8_KV_CALIBRATION_METRICS_TIMEOUT_SEC HFLC_FP8_REQUIRE_KV_CALIBRATION \
  HFLC_FP8_VLLM_KV_CACHE_MEMORY_BYTES HFLC_FP8_VLLM_NUM_GPU_BLOCKS_OVERRIDE \
  HFLC_FP8_NUM_RUNS HFLC_FP8_SMALL_POINT_NUM_RUNS HFLC_FP8_SMALL_POINT_THRESHOLD HFLC_FP8_PASS_INDEX HFLC_FP8_WARMUP_DISCARD_REQUESTS HFLC_FP8_WARMUP_DISCARD_THRESHOLD HFLC_FP8_REQUEST_TIMEOUT_SEC HFLC_FP8_GPU_MEMORY_UTILIZATION \
  HFLC_FP8_EMIT_PREFILL_PROGRESS HFLC_FP8_PREFILL_PROGRESS_INTERVAL_SEC HFLC_FP8_PREFILL_PROGRESS_TIMEOUT_SEC \
  HFLC_FP8_CRASH_CAPTURE HFLC_FP8_CRASH_CAPTURE_SET_ULIMIT HFLC_FP8_CRASH_CAPTURE_ENABLE_HSA_DEBUG \
  HFLC_FP8_MAX_CHARS_PER_FILE HFLC_FP8_PROMPT_TOKEN_IDS_FILE HFLC_FP8_PROMPT_SHAPE HFLC_FP8_PROMPT_SHAPE_FILE HFLC_FP8_QUERY_API_MODE \
  HFLC_FP8_PREFIX_SESSION_MODE HFLC_FP8_PREFIX_SESSION_ID HFLC_FP8_PREFIX_SESSION_DIR HFLC_FP8_PREFIX_COMPAT_STRICT \
  HFLC_FP8_SUFFIX_QUERY_FILE HFLC_FP8_SUFFIX_QUERY_FILES HFLC_FP8_SUFFIX_QUERY_DIR HFLC_FP8_PREFIX_QUERY_MAX_COUNT HFLC_FP8_PREFIX_RELOAD_BETWEEN_QUERIES \
  HFLC_FP8_PREFIX_MAX_TOKENS HFLC_FP8_PREFIX_SAVE_SHARDS HFLC_FP8_VLLM_ENABLE_PREFIX_CACHING HFLC_FP8_PREFIX_TARGET_TP \
  HFLC_FP8_PREFIX_STATE_CHUNK_BYTES HFLC_FP8_PREFIX_STATE_CALIBRATION_STRICT_GATE HFLC_FP8_PREFIX_STATE_CALIBRATION_SAFETY_RATIO HFLC_FP8_PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES \
  HFLC_FP8_TARGET_PROMPT_TOKENS HFLC_FP8_ENGLISH_ONLY_MASK_MODE HFLC_FP8_ENGLISH_ONLY_MASK_BIAS HFLC_FP8_DUMP_REQUEST_PAYLOAD HFLC_FP8_DUMP_QUERY_MESSAGES HFLC_FP8_DUMP_PROMPT_TOKEN_IDS HFLC_FP8_NO_STREAM HFLC_FP8_CAPTURE_MODE HFLC_FP8_CAPTURE_SESSION_ID HFLC_FP8_CAPTURE_SESSION_DIR \
  HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILE HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILES HFLC_FP8_CAPTURE_SUFFIX_QUERY_DIR \
  HFLC_FP8_CAPTURE_QUERY_MAX_COUNT HFLC_FP8_CAPTURE_RELOAD_BETWEEN_QUERIES HFLC_FP8_CAPTURE_PREFIX_MAX_TOKENS HFLC_FP8_CAPTURE_TARGET_PROMPT_TOKENS \
  HFLC_FP8_CAMPAIGN_DIR \
  HFLC_FIT_CUTOFF_TOKENS HFLC_APPEND_TO_CAMPAIGN \
  HFLC_FP8_BASELINE_REPORT_MD HFLC_FP8_BASELINE_LABEL HFLC_FP8_CURRENT_LABEL \
  HFLC_WRAPPER_ENTRYPOINT HFLC_WRAPPER_CHAIN

# Profile presets (command-line variable overrides still win).
ifeq ($(HFLC_PROFILE),smoke)
HFLC_FP8_SWEEP_LIST := 1024,2048,4096,8192,16384
HFLC_FP8_SWEEP := 1024
HFLC_FP8_MAX_NEW_TOKENS := 128
HFLC_FP8_NUM_RUNS := 1
HFLC_FP8_REQUEST_TIMEOUT_SEC := 14400
HFLC_FIT_CUTOFF_TOKENS := 8192
endif

ifeq ($(HFLC_PROFILE),long)
HFLC_FP8_MAX_NEW_TOKENS := 4096
HFLC_FP8_NUM_RUNS := 1
HFLC_FP8_REQUEST_TIMEOUT_SEC := 345600
endif

ifeq ($(HFLC_PROFILE),debug)
HFLC_FP8_SWEEP := 1024
HFLC_FP8_MAX_NEW_TOKENS := 64
HFLC_FP8_NUM_RUNS := 1
HFLC_FP8_REQUEST_TIMEOUT_SEC := 7200
HFLC_FIT_CUTOFF_TOKENS := 8192
endif

ifeq ($(HFLC_PROFILE),capture)
HFLC_FP8_CAPTURE_MODE := build_and_query
HFLC_FP8_PREFIX_SESSION_MODE := build_and_query
HFLC_FP8_NUM_RUNS := 1
HFLC_FP8_MAX_NEW_TOKENS := 256
HFLC_FP8_PREFIX_SAVE_SHARDS := 1
HFLC_FP8_PREFIX_COMPAT_STRICT := 1
HFLC_FP8_REQUEST_TIMEOUT_SEC := 43200
endif
