#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

show_help() {
  cat <<'EOF'
driver_vllm_two_phase.sh

Purpose:
  Stable wrapper for app/scripts/benchmark_vllm_long_context.py with
  process/port cleanup and two-phase long-context execution.
  Optional adaptive request timeout uses previous successful TTFT and
  predicts the next target with quadratic scaling:
    TTFT_next ~= TTFT_prev * (tokens_next / tokens_prev)^2

Usage:
  bash experiments/hf_long_context/driver_vllm_two_phase.sh
  bash experiments/hf_long_context/driver_vllm_two_phase.sh --help

Control via environment variables:
  ROOT                         Repo root (default: derived from this script's location, then /app/long-context-serving)
  RUN_ID                       Run id suffix (default: vllm_long_context_<utc timestamp>)
  RUN_DIR                      Run output directory (default: $ROOT/experiments/hf_long_context/runs/$RUN_ID)
  PH1_DIR                      Phase-1 output dir (default: $RUN_DIR/phase1)
  PH2_DIR                      Phase-2 output dir (default: $RUN_DIR/phase2)
  DRIVER_LOG                   Log file path (default: $RUN_DIR/driver.log)
  PHASE_MODE                   phase1 | phase2 | both (default: both)

  MODEL_ID                     Model to benchmark (default: moonshotai/Kimi-Linear-48B-A3B-Instruct)
  VLLM_BIN                     vLLM executable path (default: /usr/local/bin/vllm)
  PORT                         Preferred vLLM port (default: 18192)
  TP                           Tensor parallel size (default: 8)
  DATA_PARALLEL_SIZE           Data parallel size (default: 1)
  PP                           Pipeline parallel size (default: 1)
  HIP_VISIBLE_DEVICES          GPU list (default: 0,1,2,3,4,5,6,7)
  SUBSTRATE_ROOT               Prompt substrate root (default: $ROOT/data/substrate)
  SUBSTRATE_MAX_FILES          Max substrate files for prompt seed (default: 0 = all)
  SUBSTRATE_MAX_CHARS_PER_FILE Max chars per substrate file (default: 200000)
  SUBSTRATE_CHAR_BUDGET_MULTIPLIER
                               Prompt-seed char budget multiplier (default: 12)
  PROMPT_SHAPE                 Prompt shape preset: benchmark_v1 | repo_grounded_en_v1 | repo_grounded_en_v2 | repo_grounded_universal_v1 | custom_file
                               (default: benchmark_v1)
  PROMPT_SHAPE_FILE            Prompt-shape JSON file path; required when PROMPT_SHAPE=custom_file
  PREFIX_SESSION_MODE          Prefix session mode: none|build|restore|query|build_and_query (default: none)
  PREFIX_SESSION_ID            Prefix session id (default: empty -> runner default)
  PREFIX_SESSION_DIR           Prefix session directory (default: empty -> runner default)
  PREFIX_COMPAT_STRICT         Prefix restore strict compatibility check 1/0 (default: 1)
  SUFFIX_QUERY_FILE            Optional suffix query file for query/build_and_query modes (default: empty)
  SUFFIX_QUERY_FILES           Optional comma-separated suffix query files (default: empty)
  SUFFIX_QUERY_DIR             Optional directory; all files become suffix queries in sorted order
                               (default: empty)
  PREFIX_QUERY_MAX_COUNT       Optional cap on suffix queries per run; 0 means all (default: 0)
  PREFIX_RELOAD_BETWEEN_QUERIES
                               1/0 reload prefix state from disk between queries (default: 0)
  PREFIX_MAX_TOKENS            Optional prefix token cap; 0 disables cap (default: 0)
  PREFIX_SAVE_SHARDS           Save prefix state shards in build mode 1/0 (default: 1)
  PREFIX_TARGET_TP             Target TP hint for future reshard flow (default: 0)
  TARGET_PROMPT_TOKENS         Prompt token target used by benchmark runner (default: 32768)
  PREFIX_STATE_CHUNK_BYTES     Prefix state dump/load chunk bytes (default: 67108864)
  PREFIX_STATE_CALIBRATION_STRICT_GATE
                               Prefix dump strict calibration gate 1/0 (default: 1)
  PREFIX_STATE_CALIBRATION_SAFETY_RATIO
                               Prefix dump calibration safety ratio (default: 0.10)
  PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES
                               Prefix dump calibration minimum safety bytes (default: 2147483648)

  MAX_NEW_TOKENS               Generation length per request (default: 512)
  TEMPERATURE                  Sampling temperature (default: 0.0 = greedy)
  TOP_P                        Nucleus sampling top-p (default: 1.0 = no nucleus)
  TOP_K                        Top-k sampling (default: 0 = no top-k)
  REPETITION_PENALTY           Repetition penalty (default: 1.0 = off)
  PRESENCE_PENALTY             Presence penalty (default: 0.0 = off)
  FREQUENCY_PENALTY            Frequency penalty (default: 0.0 = off)
  GPU_MEMORY_UTILIZATION       vLLM --gpu-memory-utilization (default: 0.9)
  KV_CACHE_DTYPE               vLLM --kv-cache-dtype (default: auto)
  VLLM_CALCULATE_KV_SCALES     1/0 add --calculate-kv-scales (default: 0)
  VLLM_MAX_NUM_SEQS            vLLM --max-num-seqs (default: 0 = vLLM default)
  VLLM_MAX_NUM_BATCHED_TOKENS  vLLM --max-num-batched-tokens (default: 0)
  VLLM_ATTENTION_BACKEND       vLLM --attention-backend (default: unset)
  VLLM_KV_CACHE_MEMORY_BYTES   vLLM --kv-cache-memory-bytes (default: 0)
  VLLM_NUM_GPU_BLOCKS_OVERRIDE vLLM --num-gpu-blocks-override (default: 0)
  VLLM_COMPILATION_CONFIG      vLLM --compilation-config JSON string (default: unset)
  VLLM_DISABLE_ROCM_SKINNY_GEMM
                               1/0 add --disable-rocm-skinny-gemm to the benchmark
                               launcher so the server records
                               VLLM_ROCM_USE_SKINNY_GEMM=0 (default: 0)
  HFLC_ALLOW_UNSAFE_LONG_PIECEWISE
                               1/0 bypass the long-prefix PIECEWISE safety downgrade for
                               restored query runs (default: 0)
  HFLC_LONG_PIECEWISE_QUERY_DISABLE_TOKENS
                               Disable PIECEWISE automatically at or above this restored-query
                               prompt length (default: 33554432)
  FP8_COHERENCE_GUARDRAILS     1/0 pin max-num-seqs to the measured recipe for fp8+aiter
                               (default: 1)
  VLLM_BLOCK_SIZE              vLLM --block-size (default: 0)
  KV_CALIBRATION_GATE          1/0 run KV capacity calibration before measured requests (default: 1)
  KV_CALIBRATION_ONLY          1/0 startup-only calibration run; skip benchmark requests (default: 0)
  KV_CALIBRATION_HEADROOM_RATIO
                               Calibration headroom ratio over prompt length (default: 0.005)
  KV_CALIBRATION_HEADROOM_MIN_BLOCKS
                               Calibration minimum headroom in blocks (default: 2)
  KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS
                               Max-num-seqs used by calibration math (default: 0 -> VLLM_MAX_NUM_SEQS -> 1)
  KV_CALIBRATION_METRICS_TIMEOUT_SEC
                               Timeout waiting for startup metrics in server log (default: 180)
  NUM_RUNS                     Measured runs per prompt length (default: 1)
  WARMUP_DISCARD_REQUESTS      Untimed warm-up requests discarded before the first measured
                               iteration to absorb one-time graph-capture cost (default: 1)
  MEASURE_K2FT                 1/0 enable K2FT cold/hot probe per measured point (default: 0)
  K2FT_DELTA_TOKENS            Probe rule delta for L_probe=max(min_probe, L_current-delta) (default: 256)
  K2FT_MIN_PROBE_TOKENS        Probe rule minimum prompt tokens (default: 1024)
  K2FT_RUNS                    Probe requests per point (cold->hot, default: 2)
  EMIT_PREFILL_PROGRESS        1/0 emit prefill_progress.jsonl by polling /debug/prefill_progress (default: 0; endpoint not in the public image)
  PREFILL_PROGRESS_INTERVAL_SEC Poll interval for prefill progress endpoint (default: 20)
  PREFILL_PROGRESS_TIMEOUT_SEC  Poll timeout for endpoint requests; 0 uses request timeout (default: 0)
  SERVER_READY_TIMEOUT_SEC     vLLM health timeout seconds (default: 3600)
  REQUEST_TIMEOUT_SEC          Per-request timeout seconds (default: 14400)
  ADAPTIVE_REQUEST_TIMEOUT     1/0 enable TTFT-based timeout growth (default: 1)
  ADAPTIVE_TIMEOUT_SCALE       Quadratic predictor scale factor (default: 1.25)
  ADAPTIVE_TIMEOUT_EXTRA_SEC   Extra slack seconds added to predicted timeout (default: 30)
  ADAPTIVE_TIMEOUT_CAP_SEC     Hard cap on adaptive timeout; 0 disables cap (default: 0)
  TIMEOUT_SCLK_GUARD           1/0 on timeout, sample sclk before failing (default: 1)
  TIMEOUT_SCLK_MIN_MHZ         Busy threshold per GPU for sclk guard (default: 500)
  TIMEOUT_SCLK_SAMPLE_COUNT    Samples per timeout guard window (default: 6)
  TIMEOUT_SCLK_SAMPLE_INTERVAL_SEC  Seconds between sclk samples (default: 5)
  TIMEOUT_SCLK_MIN_BUSY_GPUS   Busy GPU threshold (0=auto from TP*PP, default: 0)
  TIMEOUT_SCLK_REQUIRED_HIT_RATIO  Fraction of samples requiring busy GPUs (default: 0.5)
  TIMEOUT_SCLK_MAX_EXTENSIONS  Max timeout extensions after busy checks (default: 3)
  TIMEOUT_SCLK_EXTENSION_SEC   Timeout increase per busy guard pass (default: 120)

  PHASE1_MAX_MODEL_LEN         Phase-1 max-model-len (default: 1049088)
  PHASE1_SWEEP                 Phase-1 sweep tokens CSV (default: 1K..1M)
  PHASE1_ALLOW_LONG            1/0 set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 in phase1 (default: 0)
  PHASE2_MAX_MODEL_LEN         Phase-2 max-model-len (default: 8389120)
  PHASE2_SWEEP                 Phase-2 sweep tokens CSV (default: 2M,4M,8M)

  TRUST_REMOTE_CODE            1/0 add --trust-remote-code (default: 1)
  SAVE_GENERATED_TEXT          1/0 add --save-generated-text (default: 1)
  NO_STREAM                    1/0 add --no-stream (default: 0)
  VLLM_ENFORCE_EAGER           1/0 add --vllm-enforce-eager (default: 0)
  VLLM_ENABLE_PREFIX_CACHING   1/0 add --vllm-enable-prefix-caching (default: 0; auto-enabled for build_and_query)

  CRASH_CAPTURE                1/0 collect crash artifacts (default: 1)
  CRASH_CAPTURE_DIR            Crash artifact dir (default: $RUN_DIR/crash_capture)
  CRASH_CAPTURE_SET_ULIMIT     1/0 enforce ulimit -c 0 (default: 1)
  CRASH_CAPTURE_ENABLE_HSA_DEBUG  1/0 export HSA_ENABLE_DEBUG=1 (default: 1)
  CRASH_CAPTURE_SET_CORE_PATTERN  1/0 attempt kernel.core_pattern update (default: 0)
  CRASH_CAPTURE_CORE_PATTERN   core_pattern target (default: /proc/%p/cwd/core.%p)
  CRASH_CAPTURE_MERGE          1/0 run roccoremerge when core/gpucore pair exists (default: 0)
  CRASH_CAPTURE_DMESG_LINES    dmesg tail lines in capture bundle (default: 400)

Phase guidance:
  phase1:
    - Uses configured max-model-len without VLLM_ALLOW_LONG_MAX_MODEL_LEN.
    - Use for baseline/stability checks near model-default context (for example 1K..1M).
  phase2:
    - Sets VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 automatically.
    - Use only when testing beyond model-default context (for example 2M+).
  both:
    - Recommended for full sweeps: baseline first, then extended-context stress.

Examples:
  # Full two-phase run with defaults
  RUN_ID=kimi_tp8_full_$(date -u +%Y%m%dT%H%M%SZ) \
  bash experiments/hf_long_context/driver_vllm_two_phase.sh

  # Phase-1 only for a stability quick check
  PHASE_MODE=phase1 \
  MODEL_ID=moonshotai/Kimi-Linear-48B-A3B-Instruct \
  PHASE1_SWEEP=1024,2048,4096,8192,16384 \
  bash experiments/hf_long_context/driver_vllm_two_phase.sh

  # Phase-2 only for extended context (requires allow-long gate)
  PHASE_MODE=phase2 \
  MODEL_ID=moonshotai/Kimi-Linear-48B-A3B-Instruct \
  PHASE2_SWEEP=2097152,4194304 \
  bash experiments/hf_long_context/driver_vllm_two_phase.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "Unknown argument(s): $*" >&2
  echo "" >&2
  show_help >&2
  exit 2
fi

is_true() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

compilation_config_cudagraph_mode() {
  python3 - "$1" <<'PY'
import json
import re
import sys

text = str(sys.argv[1] or "").strip()
if not text:
    raise SystemExit(0)

try:
    payload = json.loads(text)
except json.JSONDecodeError:
    if not (text.startswith("{") and text.endswith("}")):
        raise
    body = text[1:-1].strip()
    payload = {}
    if body:
        for item in body.split(","):
            piece = item.strip()
            if not piece:
                continue
            if ":" not in piece:
                raise json.JSONDecodeError("missing ':' in object item", text, 0)
            key_raw, value_raw = piece.split(":", 1)
            key = key_raw.strip().strip("\"'")
            value_text = value_raw.strip()
            lowered = value_text.lower()
            if lowered == "true":
                value = True
            elif lowered == "false":
                value = False
            elif lowered == "null":
                value = None
            elif re.fullmatch(r"-?\d+", value_text):
                value = int(value_text)
            elif re.fullmatch(r"-?\d+\.\d+", value_text):
                value = float(value_text)
            else:
                value = value_text.strip("\"'")
            payload[key] = value

if not isinstance(payload, dict):
    raise SystemExit(2)

mode = str(payload.get("cudagraph_mode") or "").strip().upper()
if mode:
    print(mode)
PY
}

compilation_config_with_cudagraph_mode() {
  python3 - "$1" "$2" <<'PY'
import json
import re
import sys

text = str(sys.argv[1] or "").strip()
mode = str(sys.argv[2] or "").strip().upper()
if not mode:
    raise SystemExit(2)

if not text:
    print(json.dumps({"cudagraph_mode": mode}, separators=(",", ":"), sort_keys=True))
    raise SystemExit(0)

try:
    payload = json.loads(text)
except json.JSONDecodeError:
    if not (text.startswith("{") and text.endswith("}")):
        raise
    body = text[1:-1].strip()
    payload = {}
    if body:
        for item in body.split(","):
            piece = item.strip()
            if not piece:
                continue
            if ":" not in piece:
                raise json.JSONDecodeError("missing ':' in object item", text, 0)
            key_raw, value_raw = piece.split(":", 1)
            key = key_raw.strip().strip("\"'")
            value_text = value_raw.strip()
            lowered = value_text.lower()
            if lowered == "true":
                value = True
            elif lowered == "false":
                value = False
            elif lowered == "null":
                value = None
            elif re.fullmatch(r"-?\d+", value_text):
                value = int(value_text)
            elif re.fullmatch(r"-?\d+\.\d+", value_text):
                value = float(value_text)
            else:
                value = value_text.strip("\"'")
            payload[key] = value

if not isinstance(payload, dict):
    raise SystemExit(2)

payload["cudagraph_mode"] = mode
print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
PY
}

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DERIVED_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-${REPO_ROOT:-$DERIVED_ROOT}}"

# Guardrail: some environments export ROOT=/app (container parent) instead of
# the actual repo root at /app/long-context-serving. Auto-correct when detected.
if [[ -d "${ROOT}" && ! -f "${ROOT}/app/scripts/benchmark_vllm_long_context.py" ]]; then
  if [[ -f "${ROOT}/long-context-serving/app/scripts/benchmark_vllm_long_context.py" ]]; then
    ROOT="${ROOT}/long-context-serving"
  fi
fi

if [[ ! -d "${ROOT}" ]]; then
  echo "Missing ROOT directory: ${ROOT}" >&2
  exit 1
fi
cd "${ROOT}"

RUN_ID="${RUN_ID:-vllm_long_context_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/hf_long_context/runs/${RUN_ID}}"
PH1_DIR="${PH1_DIR:-${RUN_DIR}/phase1}"
PH2_DIR="${PH2_DIR:-${RUN_DIR}/phase2}"
mkdir -p "${RUN_DIR}"

DRIVER_LOG="${DRIVER_LOG:-${RUN_DIR}/driver.log}"
touch "${DRIVER_LOG}"

log() {
  echo "[$(utc_now)] $*" | tee -a "${DRIVER_LOG}"
}

MODEL_ID="${MODEL_ID:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
VLLM_BIN="${VLLM_BIN:-/usr/local/bin/vllm}"
PORT="${PORT:-18192}"
TP="${TP:-8}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
PP="${PP:-1}"
GPU_LIST="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SUBSTRATE_ROOT="${SUBSTRATE_ROOT:-${ROOT}/data/substrate}"
SUBSTRATE_MAX_FILES="${SUBSTRATE_MAX_FILES:-0}"
SUBSTRATE_MAX_CHARS_PER_FILE="${SUBSTRATE_MAX_CHARS_PER_FILE:-200000}"
SUBSTRATE_CHAR_BUDGET_MULTIPLIER="${SUBSTRATE_CHAR_BUDGET_MULTIPLIER:-12}"
PROMPT_SHAPE="${PROMPT_SHAPE:-benchmark_v1}"
PROMPT_SHAPE_FILE="${PROMPT_SHAPE_FILE:-}"
PREFIX_SESSION_MODE="${PREFIX_SESSION_MODE:-none}"
PREFIX_SESSION_ID="${PREFIX_SESSION_ID:-}"
PREFIX_SESSION_DIR="${PREFIX_SESSION_DIR:-}"
PREFIX_COMPAT_STRICT="${PREFIX_COMPAT_STRICT:-1}"
SUFFIX_QUERY_FILE="${SUFFIX_QUERY_FILE:-}"
SUFFIX_QUERY_FILES="${SUFFIX_QUERY_FILES:-}"
SUFFIX_QUERY_DIR="${SUFFIX_QUERY_DIR:-}"
PREFIX_QUERY_MAX_COUNT="${PREFIX_QUERY_MAX_COUNT:-0}"
PREFIX_RELOAD_BETWEEN_QUERIES="${PREFIX_RELOAD_BETWEEN_QUERIES:-0}"
PREFIX_MAX_TOKENS="${PREFIX_MAX_TOKENS:-0}"
PREFIX_SAVE_SHARDS="${PREFIX_SAVE_SHARDS:-1}"
PREFIX_TARGET_TP="${PREFIX_TARGET_TP:-0}"
TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-32768}"
PREFIX_STATE_CHUNK_BYTES="${PREFIX_STATE_CHUNK_BYTES:-67108864}"
PREFIX_STATE_CALIBRATION_STRICT_GATE="${PREFIX_STATE_CALIBRATION_STRICT_GATE:-1}"
PREFIX_STATE_CALIBRATION_SAFETY_RATIO="${PREFIX_STATE_CALIBRATION_SAFETY_RATIO:-0.10}"
PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES="${PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES:-2147483648}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
VLLM_CALCULATE_KV_SCALES="${VLLM_CALCULATE_KV_SCALES:-0}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-0}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-0}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-0}"
VLLM_NUM_GPU_BLOCKS_OVERRIDE="${VLLM_NUM_GPU_BLOCKS_OVERRIDE:-0}"
VLLM_COMPILATION_CONFIG="${VLLM_COMPILATION_CONFIG:-}"
VLLM_DISABLE_ROCM_SKINNY_GEMM="${VLLM_DISABLE_ROCM_SKINNY_GEMM:-0}"
HFLC_ALLOW_UNSAFE_LONG_PIECEWISE="${HFLC_ALLOW_UNSAFE_LONG_PIECEWISE:-0}"
HFLC_LONG_PIECEWISE_QUERY_DISABLE_TOKENS="${HFLC_LONG_PIECEWISE_QUERY_DISABLE_TOKENS:-33554432}"
FP8_COHERENCE_GUARDRAILS="${FP8_COHERENCE_GUARDRAILS:-1}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-3600}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-14400}"
ADAPTIVE_REQUEST_TIMEOUT="${ADAPTIVE_REQUEST_TIMEOUT:-1}"
ADAPTIVE_TIMEOUT_SCALE="${ADAPTIVE_TIMEOUT_SCALE:-1.25}"
ADAPTIVE_TIMEOUT_EXTRA_SEC="${ADAPTIVE_TIMEOUT_EXTRA_SEC:-30}"
ADAPTIVE_TIMEOUT_CAP_SEC="${ADAPTIVE_TIMEOUT_CAP_SEC:-0}"
TIMEOUT_SCLK_GUARD="${TIMEOUT_SCLK_GUARD:-1}"
TIMEOUT_SCLK_MIN_MHZ="${TIMEOUT_SCLK_MIN_MHZ:-500}"
TIMEOUT_SCLK_SAMPLE_COUNT="${TIMEOUT_SCLK_SAMPLE_COUNT:-6}"
TIMEOUT_SCLK_SAMPLE_INTERVAL_SEC="${TIMEOUT_SCLK_SAMPLE_INTERVAL_SEC:-5}"
TIMEOUT_SCLK_MIN_BUSY_GPUS="${TIMEOUT_SCLK_MIN_BUSY_GPUS:-0}"
TIMEOUT_SCLK_REQUIRED_HIT_RATIO="${TIMEOUT_SCLK_REQUIRED_HIT_RATIO:-0.5}"
TIMEOUT_SCLK_MAX_EXTENSIONS="${TIMEOUT_SCLK_MAX_EXTENSIONS:-3}"
TIMEOUT_SCLK_EXTENSION_SEC="${TIMEOUT_SCLK_EXTENSION_SEC:-120}"
NUM_RUNS="${NUM_RUNS:-1}"
WARMUP_DISCARD_REQUESTS="${WARMUP_DISCARD_REQUESTS:-1}"
KV_CALIBRATION_GATE="${KV_CALIBRATION_GATE:-1}"
KV_CALIBRATION_ONLY="${KV_CALIBRATION_ONLY:-0}"
KV_CALIBRATION_HEADROOM_RATIO="${KV_CALIBRATION_HEADROOM_RATIO:-0.005}"
KV_CALIBRATION_HEADROOM_MIN_BLOCKS="${KV_CALIBRATION_HEADROOM_MIN_BLOCKS:-2}"
KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS="${KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS:-0}"
KV_CALIBRATION_METRICS_TIMEOUT_SEC="${KV_CALIBRATION_METRICS_TIMEOUT_SEC:-180}"
MEASURE_K2FT="${MEASURE_K2FT:-0}"
K2FT_DELTA_TOKENS="${K2FT_DELTA_TOKENS:-256}"
K2FT_MIN_PROBE_TOKENS="${K2FT_MIN_PROBE_TOKENS:-1024}"
K2FT_RUNS="${K2FT_RUNS:-2}"
EMIT_PREFILL_PROGRESS="${EMIT_PREFILL_PROGRESS:-0}"
PREFILL_PROGRESS_INTERVAL_SEC="${PREFILL_PROGRESS_INTERVAL_SEC:-20}"
PREFILL_PROGRESS_TIMEOUT_SEC="${PREFILL_PROGRESS_TIMEOUT_SEC:-0}"

PHASE1_MAX_MODEL_LEN="${PHASE1_MAX_MODEL_LEN:-1049088}"
PHASE2_MAX_MODEL_LEN="${PHASE2_MAX_MODEL_LEN:-8389120}"
PHASE1_SWEEP="${PHASE1_SWEEP:-1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576}"
PHASE2_SWEEP="${PHASE2_SWEEP:-2097152,4194304,8388608}"
PHASE1_ALLOW_LONG="${PHASE1_ALLOW_LONG:-0}"
PHASE_MODE="${PHASE_MODE:-both}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
SAVE_GENERATED_TEXT="${SAVE_GENERATED_TEXT:-1}"
NO_STREAM="${NO_STREAM:-0}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}"

# Crash-capture controls.
# CRASH_CAPTURE: master switch for preflight + post-phase artifact collection.
# CRASH_CAPTURE_DIR: root directory for phase-local capture outputs.
# CRASH_CAPTURE_SET_ULIMIT: enforce `ulimit -c 0` so graph-mode retries do not dump giant core files.
# CRASH_CAPTURE_ENABLE_HSA_DEBUG: export HSA_ENABLE_DEBUG=1 to improve GPU dump metadata.
# CRASH_CAPTURE_SET_CORE_PATTERN: attempt to set kernel.core_pattern for host cores.
# CRASH_CAPTURE_CORE_PATTERN: target value used when setting kernel.core_pattern.
# CRASH_CAPTURE_MERGE: run roccoremerge when matching core.<pid> and gpucore.<pid> exist.
# CRASH_CAPTURE_DMESG_LINES: number of dmesg lines captured at phase end.
CRASH_CAPTURE="${CRASH_CAPTURE:-1}"
CRASH_CAPTURE_DIR="${CRASH_CAPTURE_DIR:-${RUN_DIR}/crash_capture}"
CRASH_CAPTURE_SET_ULIMIT="${CRASH_CAPTURE_SET_ULIMIT:-1}"
CRASH_CAPTURE_ENABLE_HSA_DEBUG="${CRASH_CAPTURE_ENABLE_HSA_DEBUG:-1}"
CRASH_CAPTURE_SET_CORE_PATTERN="${CRASH_CAPTURE_SET_CORE_PATTERN:-0}"
CRASH_CAPTURE_CORE_PATTERN="${CRASH_CAPTURE_CORE_PATTERN:-/proc/%p/cwd/core.%p}"
CRASH_CAPTURE_MERGE="${CRASH_CAPTURE_MERGE:-0}"
CRASH_CAPTURE_DMESG_LINES="${CRASH_CAPTURE_DMESG_LINES:-400}"

# Never allow this driver to inherit or create host/GPU core dumps.
# Crash-capture still preserves text artifacts, but binary core files were too
# large and too common during graph-mode bring-up to leave enabled by default.
ulimit -c 0 2>/dev/null || true

export HIP_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONUNBUFFERED=1

# FP8 coherence guardrails for Kimi TP=8/AITER campaign: keep launch settings
# aligned with the known-good coherent recipe unless explicitly disabled.
if [[ "${KV_CACHE_DTYPE}" == "fp8_e4m3" && "${VLLM_ATTENTION_BACKEND}" == "ROCM_AITER_MLA" ]]; then
  if is_true "${FP8_COHERENCE_GUARDRAILS}"; then
    if [[ "${VLLM_MAX_NUM_SEQS}" == "0" ]]; then
      VLLM_MAX_NUM_SEQS="1"
    fi
    if [[ "${VLLM_MAX_NUM_SEQS}" != "1" && "${VLLM_MAX_NUM_SEQS}" != "16" ]]; then
      log "ERROR FP8_COHERENCE_GUARDRAILS=1 requires VLLM_MAX_NUM_SEQS in {1,16} (got ${VLLM_MAX_NUM_SEQS})"
      exit 1
    fi
  fi
fi

# Non-eager FP8 MLA decode runs with CUDA graphs. Callers that pass no explicit
# compilation config get graph-off, so an unconfigured run never silently picks
# up graph capture; the narrow downgrade below covers long restored prefixes.
if [[ "${KV_CACHE_DTYPE}" == "fp8_e4m3" && "${VLLM_ATTENTION_BACKEND}" == "ROCM_AITER_MLA" ]] \
  && ! is_true "${VLLM_ENFORCE_EAGER}"; then
  if [[ -z "${VLLM_COMPILATION_CONFIG}" ]]; then
    VLLM_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
  fi
  if [[ "${PREFIX_SESSION_MODE}" == "query" ]] \
    && ! is_true "${HFLC_ALLOW_UNSAFE_LONG_PIECEWISE}" \
    && [[ "${HFLC_LONG_PIECEWISE_QUERY_DISABLE_TOKENS}" =~ ^[0-9]+$ ]] \
    && [[ "${TARGET_PROMPT_TOKENS}" =~ ^[0-9]+$ ]] \
    && (( HFLC_LONG_PIECEWISE_QUERY_DISABLE_TOKENS > 0 )) \
    && (( TARGET_PROMPT_TOKENS >= HFLC_LONG_PIECEWISE_QUERY_DISABLE_TOKENS )); then
    # Prefix-reuse query mode can request PIECEWISE graphs, but very long
    # restored prefixes were observed to be less stable than graph-off. This
    # downgrade is intentionally narrow: only restored-prefix query mode, and
    # only at/above the configured token gate.
    requested_graph_mode="$(compilation_config_cudagraph_mode "${VLLM_COMPILATION_CONFIG}" || true)"
    if [[ "${requested_graph_mode}" == "PIECEWISE" ]]; then
      updated_compilation_config="$(compilation_config_with_cudagraph_mode "${VLLM_COMPILATION_CONFIG}" "NONE" || true)"
      if [[ -n "${updated_compilation_config}" ]]; then
        VLLM_COMPILATION_CONFIG="${updated_compilation_config}"
        log "guardrail applied: downgraded cudagraph_mode=PIECEWISE -> NONE for long restored-prefix query at target_prompt_tokens=${TARGET_PROMPT_TOKENS}; set HFLC_ALLOW_UNSAFE_LONG_PIECEWISE=1 to bypass"
      fi
    fi
  fi
fi
export VLLM_COMPILATION_CONFIG

if is_true "${VLLM_ROCM_AITER_MLA_USE_FLASHINFER_DECODE}"; then
  FLASHINFER_ROCM_REPO_ROOT="${ROOT}/experiments/hf_long_context/external/flashinfer_rocm"
  if [[ ! -d "${FLASHINFER_ROCM_REPO_ROOT}/flashinfer" ]]; then
    log "ERROR VLLM_ROCM_AITER_MLA_USE_FLASHINFER_DECODE=1 but missing local ROCm FlashInfer repo at ${FLASHINFER_ROCM_REPO_ROOT}"
    exit 1
  fi
  export PYTHONPATH="${FLASHINFER_ROCM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
  log "ERROR missing executable VLLM_BIN=${VLLM_BIN}"
  exit 1
fi
if [[ ! -f "${ROOT}/app/scripts/benchmark_vllm_long_context.py" ]]; then
  log "ERROR missing runner app/scripts/benchmark_vllm_long_context.py"
  exit 1
fi
if [[ ! -d "${SUBSTRATE_ROOT}" ]]; then
  log "ERROR missing SUBSTRATE_ROOT=${SUBSTRATE_ROOT}"
  exit 1
fi
case "${PROMPT_SHAPE}" in
  benchmark_v1|repo_grounded_en_v1|repo_grounded_en_v2|repo_grounded_universal_v1|custom_file) ;;
  *)
    log "ERROR invalid PROMPT_SHAPE=${PROMPT_SHAPE} (expected: benchmark_v1|repo_grounded_en_v1|repo_grounded_en_v2|repo_grounded_universal_v1|custom_file)"
    exit 1
    ;;
esac
if [[ "${PROMPT_SHAPE}" == "custom_file" && -z "${PROMPT_SHAPE_FILE}" ]]; then
  log "ERROR PROMPT_SHAPE=custom_file requires PROMPT_SHAPE_FILE"
  exit 1
fi
if [[ -n "${PROMPT_SHAPE_FILE}" && ! -f "${PROMPT_SHAPE_FILE}" ]]; then
  log "ERROR PROMPT_SHAPE_FILE not found: ${PROMPT_SHAPE_FILE}"
  exit 1
fi
case "${PHASE_MODE}" in
  both|phase1|phase2) ;;
  *)
    log "ERROR invalid PHASE_MODE=${PHASE_MODE} (expected: both|phase1|phase2)"
    exit 1
    ;;
esac

port_pids() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null \
      | awk -v p=":${port}" '$4 ~ p"$" {print}' \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | sort -u
    return 0
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u
    return 0
  fi

  if command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null \
      | awk -v p=":${port}" '$4 ~ p"$" {print $7}' \
      | sed -n 's#\([0-9][0-9]*\)/.*#\1#p' \
      | sort -u
    return 0
  fi
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk -v p=":${port}" '$4 ~ p"$" {found=1} END{exit(found?0:1)}'
    return $?
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -P -n >/dev/null 2>&1
    return $?
  fi

  if command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk -v p=":${port}" '$4 ~ p"$" {found=1} END{exit(found?0:1)}'
    return $?
  fi

  # Dependency-free fallback: if bind succeeds, port was free.
  python3 - "${port}" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(0)  # in use
finally:
    s.close()
sys.exit(1)  # free
PY
}

pick_free_port() {
  local base="$1"
  local limit="${2:-50}"
  local candidate
  for ((candidate=base; candidate<base+limit; candidate++)); do
    if ! port_in_use "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

inspect_and_stop_pid() {
  local pid="$1"
  if ! ps -p "${pid}" >/dev/null 2>&1; then
    return 0
  fi
  local stat ppid pgid
  stat="$(ps -o stat= -p "${pid}" | awk '{print $1}')"
  ppid="$(ps -o ppid= -p "${pid}" | tr -d ' ')"
  pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
  log "Found PID=${pid} STAT=${stat:-?} PPID=${ppid:-?} PGID=${pgid:-?}"

  case "${stat}" in
    Z*)
      if [[ -n "${ppid}" ]]; then
        log "Zombie PID ${pid}; sending TERM to parent ${ppid}"
        kill -TERM "${ppid}" 2>/dev/null || true
      fi
      ;;
    *)
      if [[ -n "${pgid}" ]]; then
        log "Stopping process group ${pgid} for PID ${pid}"
        kill -TERM "-${pgid}" 2>/dev/null || true
        sleep 5
        kill -KILL "-${pgid}" 2>/dev/null || true
      fi
      ;;
  esac
}

cleanup_port() {
  local port="$1"
  local pids
  pids="$(port_pids "${port}" || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    inspect_and_stop_pid "${pid}"
  done <<< "${pids}"
  sleep 2
}

snapshot_core_basenames() {
  find "${ROOT}" -maxdepth 1 -type f \( -name 'core.*' -o -name 'gpucore.*' \) -printf '%f\n' \
    | LC_ALL=C sort -u
}

capture_cmd_to_file() {
  local out_file="$1"
  shift
  {
    echo "[$(utc_now)] cmd: $*"
    set +e
    "$@"
    local rc=$?
    set -e
    echo "[$(utc_now)] rc=${rc}"
    echo
  } >> "${out_file}" 2>&1
}

crash_capture_preflight() {
  local phase_name="$1"
  local phase_capture_dir="$2"
  local preflight_path="${phase_capture_dir}/preflight.txt"
  mkdir -p "${phase_capture_dir}" "${phase_capture_dir}/cores"

  : > "${preflight_path}"
  {
    echo "timestamp_utc=$(utc_now)"
    echo "phase=${phase_name}"
    echo "root=${ROOT}"
    echo "run_id=${RUN_ID}"
    echo "cwd=$(pwd)"
    echo "driver_log=${DRIVER_LOG}"
    echo "core_pattern_before=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo unavailable)"
    echo "ulimit_core_before=$(ulimit -c 2>/dev/null || echo unavailable)"
  } >> "${preflight_path}"

  if is_true "${CRASH_CAPTURE_SET_ULIMIT}"; then
    if ulimit -c 0 2>/dev/null; then
      echo "ulimit_core_after=$(ulimit -c 2>/dev/null || echo unavailable)" >> "${preflight_path}"
    else
      echo "ulimit_set_zero_failed=1" >> "${preflight_path}"
      log "WARN crash_capture phase=${phase_name} failed to set ulimit -c 0"
    fi
  fi

  if is_true "${CRASH_CAPTURE_ENABLE_HSA_DEBUG}"; then
    export HSA_ENABLE_DEBUG=1
  fi
  echo "HSA_ENABLE_DEBUG=${HSA_ENABLE_DEBUG:-unset}" >> "${preflight_path}"

  if is_true "${CRASH_CAPTURE_SET_CORE_PATTERN}"; then
    if [[ -w /proc/sys/kernel/core_pattern ]]; then
      printf '%s\n' "${CRASH_CAPTURE_CORE_PATTERN}" > /proc/sys/kernel/core_pattern || true
    elif command -v sysctl >/dev/null 2>&1; then
      sysctl -w "kernel.core_pattern=${CRASH_CAPTURE_CORE_PATTERN}" >/dev/null 2>&1 || true
    fi
  fi

  {
    echo "core_pattern_after=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo unavailable)"
  } >> "${preflight_path}"

  env \
    | rg '^(HSA|ROCR|HIP|VLLM|NCCL|TORCH|PYTORCH|LD_LIBRARY_PATH|PATH)=' \
    > "${phase_capture_dir}/env_snapshot.txt" || true
  {
    echo "timestamp_utc=$(utc_now)"
    echo "uname=$(uname -a 2>/dev/null || true)"
    echo "python_version=$(python3 --version 2>/dev/null || true)"
    echo "vllm_version_cli=$(${VLLM_BIN} --version 2>/dev/null || true)"
    echo "pip_show_vllm_begin"
    python3 -m pip show vllm 2>/dev/null || true
    echo "pip_show_vllm_end"
  } > "${phase_capture_dir}/tool_versions.txt"
  if command -v rocm-smi >/dev/null 2>&1; then
    capture_cmd_to_file "${phase_capture_dir}/rocm_smi_pre.txt" \
      rocm-smi --showid --showproductname --showdriverversion --showpids --showpidgpus --showuse --showmemuse --showclocks
  fi
}

crash_capture_collect() {
  local phase_name="$1"
  local phase_capture_dir="$2"
  local before_list_path="$3"
  local phase_rc="$4"
  local after_list_path="${phase_capture_dir}/core_files_after.txt"
  local new_list_path="${phase_capture_dir}/core_files_new.txt"
  local details_path="${phase_capture_dir}/core_file_details.txt"
  local summary_path="${phase_capture_dir}/summary.txt"
  local merge_log_path="${phase_capture_dir}/roccoremerge.log"
  local new_count

  snapshot_core_basenames > "${after_list_path}"
  if [[ -f "${before_list_path}" ]]; then
    comm -13 "${before_list_path}" "${after_list_path}" > "${new_list_path}"
  else
    cp "${after_list_path}" "${new_list_path}"
  fi
  new_count="$(rg -c '.*' "${new_list_path}" || true)"

  : > "${details_path}"
  while IFS= read -r core_name; do
    [[ -n "${core_name}" ]] || continue
    local core_path="${ROOT}/${core_name}"
    [[ -f "${core_path}" ]] || continue
    ln -sfn "${core_path}" "${phase_capture_dir}/cores/${core_name}" || true
    {
      echo "=== ${core_name} ==="
      ls -lh "${core_path}"
      file "${core_path}"
      readelf -h "${core_path}" 2>/dev/null | sed -n '1,40p'
      if [[ "${core_name}" == gpucore.* ]] && command -v /opt/rocm/bin/rocgdb >/dev/null 2>&1; then
        /opt/rocm/bin/rocgdb --batch -q \
          -ex "set pagination off" \
          -ex "core-file ${core_path}" \
          -ex "info agents" \
          -ex "info queues" \
          -ex "info dispatches" 2>&1 \
          | sed '/Couldn.t find general-purpose registers/d;/#0  <unavailable>/d;/^$/d'
      fi
      echo
    } >> "${details_path}" 2>&1
  done < "${new_list_path}"

  : > "${merge_log_path}"
  if is_true "${CRASH_CAPTURE_MERGE}"; then
    if command -v roccoremerge >/dev/null 2>&1; then
      declare -A merge_pids=()
      while IFS= read -r core_name; do
        [[ -n "${core_name}" ]] || continue
        if [[ "${core_name}" =~ ^gpucore\.([0-9]+)$ ]]; then
          merge_pids["${BASH_REMATCH[1]}"]=1
        elif [[ "${core_name}" =~ ^core\.([0-9]+)$ ]]; then
          merge_pids["${BASH_REMATCH[1]}"]=1
        fi
      done < "${new_list_path}"
      for pid in "${!merge_pids[@]}"; do
        local cpu_core="${ROOT}/core.${pid}"
        local gpu_core="${ROOT}/gpucore.${pid}"
        local merged_core="${phase_capture_dir}/combined.${pid}"
        if [[ ! -f "${cpu_core}" || ! -f "${gpu_core}" ]]; then
          echo "skip pid=${pid} reason=missing_pair cpu_core=${cpu_core} gpu_core=${gpu_core}" >> "${merge_log_path}"
          continue
        fi
        echo "merge pid=${pid} out=${merged_core}" >> "${merge_log_path}"
        set +e
        roccoremerge -f "${merged_core}" "${cpu_core}" "${gpu_core}" >> "${merge_log_path}" 2>&1
        local merge_rc=$?
        set -e
        echo "merge pid=${pid} rc=${merge_rc}" >> "${merge_log_path}"
      done
    else
      echo "roccoremerge_unavailable=1" >> "${merge_log_path}"
    fi
  fi

  if command -v rocm-smi >/dev/null 2>&1; then
    capture_cmd_to_file "${phase_capture_dir}/rocm_smi_post.txt" \
      rocm-smi --showid --showproductname --showdriverversion --showpids --showpidgpus --showuse --showmemuse --showclocks
  fi
  if command -v dmesg >/dev/null 2>&1; then
    set +e
    dmesg -T | tail -n "${CRASH_CAPTURE_DMESG_LINES}" > "${phase_capture_dir}/dmesg_tail.txt" 2>&1
    local dmesg_rc=$?
    set -e
    if [[ "${dmesg_rc}" -ne 0 ]]; then
      echo "dmesg_unavailable_or_permission_denied rc=${dmesg_rc}" >> "${phase_capture_dir}/dmesg_tail.txt"
    fi
  fi
  ps -eo pid,ppid,pgid,stat,etime,cmd \
    | rg -i 'vllm|python|roc|amd|kfd|renderD' \
    > "${phase_capture_dir}/process_snapshot.txt" || true

  : > "${summary_path}"
  {
    echo "timestamp_utc=$(utc_now)"
    echo "phase=${phase_name}"
    echo "phase_exit_code=${phase_rc}"
    echo "core_pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo unavailable)"
    echo "new_core_file_count=${new_count}"
    echo "new_core_files_path=${new_list_path}"
    echo "core_file_details_path=${details_path}"
    echo "rocm_smi_pre_path=${phase_capture_dir}/rocm_smi_pre.txt"
    echo "rocm_smi_post_path=${phase_capture_dir}/rocm_smi_post.txt"
    echo "dmesg_tail_path=${phase_capture_dir}/dmesg_tail.txt"
    echo "process_snapshot_path=${phase_capture_dir}/process_snapshot.txt"
    echo "merge_log_path=${merge_log_path}"
  } >> "${summary_path}"

  log "crash_capture phase=${phase_name} exit_code=${phase_rc} new_core_files=${new_count} summary=${summary_path}"
}

run_phase() {
  local phase_name="$1"
  local out_dir="$2"
  local max_model_len="$3"
  local sweep_tokens="$4"
  local allow_long="$5"
  local phase_capture_dir="${CRASH_CAPTURE_DIR}/${phase_name}"
  local before_list_path="${phase_capture_dir}/core_files_before.txt"
  local phase_rc
  local phase_prefill_path="${out_dir}/prefill_progress.jsonl"

  # Each phase creates its own output directory here, when it actually runs.
  # Creating both up front left an empty phase2/ beside phase1/ in every run
  # directory, which reads as a phase that failed or was skipped.
  mkdir -p "${out_dir}"

  export HFLC_RUN_DIR="${RUN_DIR}"
  export HFLC_PHASE_DIR="${out_dir}"
  export HFLC_PHASE_NAME="${phase_name}"

  if is_true "${allow_long}"; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  else
    unset VLLM_ALLOW_LONG_MAX_MODEL_LEN || true
  fi

  # The live control-plane probes use vLLM's dev-only RPC/cache routes. Direct
  # runs do not automatically enable these routes, so force them on whenever
  # live-layout probing or a prefix-session workflow is active.
  if is_true "${HFLC_QUERY_LIVE_LAYOUT_ENABLE:-0}" || [[ "${PREFIX_SESSION_MODE}" != "none" ]]; then
    export VLLM_SERVER_DEV_MODE=1
  else
    unset VLLM_SERVER_DEV_MODE || true
  fi

  if is_true "${CRASH_CAPTURE}"; then
    mkdir -p "${phase_capture_dir}"
    crash_capture_preflight "${phase_name}" "${phase_capture_dir}"
    snapshot_core_basenames > "${before_list_path}"
  fi

  cleanup_port "${PORT}"
  if port_in_use "${PORT}"; then
    local old_port="${PORT}"
    PORT="$(pick_free_port "$((PORT + 1))")"
    log "Port ${old_port} is still busy; switching to PORT=${PORT}"
  fi

  local cmd=(
    python3 app/scripts/benchmark_vllm_long_context.py
    --run-id "${RUN_ID}_${phase_name}"
    --out-dir "${out_dir}"
    --model-id "${MODEL_ID}"
    --vllm-bin "${VLLM_BIN}"
    --vllm-port "${PORT}"
    --tensor-parallel-size "${TP}"
    --vllm-data-parallel-size "${DATA_PARALLEL_SIZE}"
    --pipeline-parallel-size "${PP}"
    --vllm-max-model-len "${max_model_len}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --vllm-dtype "${VLLM_DTYPE}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    --vllm-kv-cache-memory-bytes "${VLLM_KV_CACHE_MEMORY_BYTES}"
    --vllm-num-gpu-blocks-override "${VLLM_NUM_GPU_BLOCKS_OVERRIDE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --repetition-penalty "${REPETITION_PENALTY}"
    --presence-penalty "${PRESENCE_PENALTY}"
    --frequency-penalty "${FREQUENCY_PENALTY}"
    --sweep-prompt-tokens "${sweep_tokens}"
    --substrate-root "${SUBSTRATE_ROOT}"
    --substrate-max-files "${SUBSTRATE_MAX_FILES}"
    --substrate-max-chars-per-file "${SUBSTRATE_MAX_CHARS_PER_FILE}"
    --substrate-char-budget-multiplier "${SUBSTRATE_CHAR_BUDGET_MULTIPLIER}"
    --prompt-shape "${PROMPT_SHAPE}"
    --prefix-session-mode "${PREFIX_SESSION_MODE}"
    --prefix-session-id "${PREFIX_SESSION_ID}"
    --prefix-session-dir "${PREFIX_SESSION_DIR}"
    --prefix-compat-strict "${PREFIX_COMPAT_STRICT}"
    --suffix-query-file "${SUFFIX_QUERY_FILE}"
    --suffix-query-files "${SUFFIX_QUERY_FILES}"
    --suffix-query-dir "${SUFFIX_QUERY_DIR}"
    --prefix-query-max-count "${PREFIX_QUERY_MAX_COUNT}"
    --prefix-reload-between-queries "${PREFIX_RELOAD_BETWEEN_QUERIES}"
    --prefix-max-tokens "${PREFIX_MAX_TOKENS}"
    --prefix-save-shards "${PREFIX_SAVE_SHARDS}"
    --prefix-target-tp "${PREFIX_TARGET_TP}"
    --target-prompt-tokens "${TARGET_PROMPT_TOKENS}"
    --prefix-state-chunk-bytes "${PREFIX_STATE_CHUNK_BYTES}"
    --prefix-state-calibration-strict-gate "${PREFIX_STATE_CALIBRATION_STRICT_GATE}"
    --prefix-state-calibration-safety-ratio "${PREFIX_STATE_CALIBRATION_SAFETY_RATIO}"
    --prefix-state-calibration-safety-min-bytes "${PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES}"
    --server-ready-timeout-sec "${SERVER_READY_TIMEOUT_SEC}"
    --request-timeout-sec "${REQUEST_TIMEOUT_SEC}"
    --adaptive-timeout-scale "${ADAPTIVE_TIMEOUT_SCALE}"
    --adaptive-timeout-extra-sec "${ADAPTIVE_TIMEOUT_EXTRA_SEC}"
    --adaptive-timeout-cap-sec "${ADAPTIVE_TIMEOUT_CAP_SEC}"
    --timeout-sclk-min-mhz "${TIMEOUT_SCLK_MIN_MHZ}"
    --timeout-sclk-sample-count "${TIMEOUT_SCLK_SAMPLE_COUNT}"
    --timeout-sclk-sample-interval-sec "${TIMEOUT_SCLK_SAMPLE_INTERVAL_SEC}"
    --timeout-sclk-min-busy-gpus "${TIMEOUT_SCLK_MIN_BUSY_GPUS}"
    --timeout-sclk-required-hit-ratio "${TIMEOUT_SCLK_REQUIRED_HIT_RATIO}"
    --timeout-sclk-max-extensions "${TIMEOUT_SCLK_MAX_EXTENSIONS}"
    --timeout-sclk-extension-sec "${TIMEOUT_SCLK_EXTENSION_SEC}"
    --num-runs "${NUM_RUNS}"
    --warmup-discard-requests "${WARMUP_DISCARD_REQUESTS}"
  )
  if is_true "${KV_CALIBRATION_GATE}"; then
    cmd+=(
      --calibrate-kv-capacity
      --calibration-strict-gate
      --calibration-headroom-ratio "${KV_CALIBRATION_HEADROOM_RATIO}"
      --calibration-headroom-min-blocks "${KV_CALIBRATION_HEADROOM_MIN_BLOCKS}"
      --calibration-assumed-max-num-seqs "${KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS}"
      --calibration-metrics-timeout-sec "${KV_CALIBRATION_METRICS_TIMEOUT_SEC}"
    )
  fi
  if is_true "${KV_CALIBRATION_ONLY}"; then
    cmd+=(--calibration-only)
  fi
  if is_true "${MEASURE_K2FT}"; then
    cmd+=(
      --measure-k2ft
      --k2ft-delta-tokens "${K2FT_DELTA_TOKENS}"
      --k2ft-min-probe-tokens "${K2FT_MIN_PROBE_TOKENS}"
      --k2ft-runs "${K2FT_RUNS}"
    )
  fi
  if is_true "${EMIT_PREFILL_PROGRESS}"; then
    cmd+=(
      --emit-prefill-progress
      --prefill-progress-interval-sec "${PREFILL_PROGRESS_INTERVAL_SEC}"
      --prefill-progress-timeout-sec "${PREFILL_PROGRESS_TIMEOUT_SEC}"
    )
  fi
  if is_true "${ADAPTIVE_REQUEST_TIMEOUT}"; then
    cmd+=(--adaptive-request-timeout)
  fi
  if is_true "${VLLM_CALCULATE_KV_SCALES}"; then
    cmd+=(--calculate-kv-scales)
  fi
  if is_true "${TIMEOUT_SCLK_GUARD}"; then
    cmd+=(--timeout-sclk-guard)
  fi
  if [[ -n "${PROMPT_SHAPE_FILE}" ]]; then
    cmd+=(--prompt-shape-file "${PROMPT_SHAPE_FILE}")
  fi

  if is_true "${TRUST_REMOTE_CODE}"; then
    cmd+=(--trust-remote-code)
  fi
  if is_true "${SAVE_GENERATED_TEXT}"; then
    cmd+=(--save-generated-text)
  fi
  if is_true "${NO_STREAM}"; then
    cmd+=(--no-stream)
  fi
  if is_true "${VLLM_ENFORCE_EAGER}"; then
    cmd+=(--vllm-enforce-eager)
  fi
  if is_true "${VLLM_DISABLE_ROCM_SKINNY_GEMM}"; then
    cmd+=(--disable-rocm-skinny-gemm)
  fi
  if is_true "${VLLM_ENABLE_PREFIX_CACHING}"; then
    cmd+=(--vllm-enable-prefix-caching)
  fi
  if [[ -n "${VLLM_ATTENTION_BACKEND}" ]]; then
    cmd+=(--vllm-attention-backend "${VLLM_ATTENTION_BACKEND}")
  fi
  if [[ -n "${VLLM_COMPILATION_CONFIG}" ]]; then
    cmd+=(--vllm-compilation-config "${VLLM_COMPILATION_CONFIG}")
  fi
  if [[ "${VLLM_BLOCK_SIZE}" -gt 0 ]]; then
    cmd+=(--vllm-block-size "${VLLM_BLOCK_SIZE}")
  fi
  if is_true "${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"; then
    cmd+=(--vllm-disable-custom-all-reduce)
  fi
  log "phase=${phase_name} start model=${MODEL_ID} port=${PORT} max_model_len=${max_model_len} sweep=${sweep_tokens}"
  printf '[%s] phase=%s cmd=%q\n' "$(utc_now)" "${phase_name}" "${cmd[*]}" >> "${DRIVER_LOG}"
  set +e
  "${cmd[@]}" 2>&1 | tee -a "${DRIVER_LOG}"
  phase_rc="${PIPESTATUS[0]}"
  set -e

  if is_true "${EMIT_PREFILL_PROGRESS}" && [[ ! -s "${phase_prefill_path}" ]]; then
    log "phase=${phase_name} artifact_missing prefill_progress_jsonl=${phase_prefill_path}"
    if [[ "${phase_rc}" -eq 0 ]]; then
      phase_rc=98
    fi
  fi

  if is_true "${CRASH_CAPTURE}"; then
    crash_capture_collect "${phase_name}" "${phase_capture_dir}" "${before_list_path}" "${phase_rc}"
  fi

  if [[ "${phase_rc}" -ne 0 ]]; then
    log "phase=${phase_name} failed rc=${phase_rc}"
    return "${phase_rc}"
  fi
  log "phase=${phase_name} done"
}

log "driver_start run_id=${RUN_ID} run_dir=${RUN_DIR}"
log "config model=${MODEL_ID} tp=${TP} dp=${DATA_PARALLEL_SIZE} pp=${PP} hip_visible_devices=${HIP_VISIBLE_DEVICES} substrate_root=${SUBSTRATE_ROOT} substrate_max_files=${SUBSTRATE_MAX_FILES} substrate_max_chars_per_file=${SUBSTRATE_MAX_CHARS_PER_FILE} substrate_char_budget_multiplier=${SUBSTRATE_CHAR_BUDGET_MULTIPLIER} prompt_shape=${PROMPT_SHAPE} prompt_shape_file=${PROMPT_SHAPE_FILE:-unset} prefix_session_mode=${PREFIX_SESSION_MODE} prefix_session_id=${PREFIX_SESSION_ID:-unset} prefix_session_dir=${PREFIX_SESSION_DIR:-unset} prefix_compat_strict=${PREFIX_COMPAT_STRICT} suffix_query_file=${SUFFIX_QUERY_FILE:-unset} suffix_query_files=${SUFFIX_QUERY_FILES:-unset} suffix_query_dir=${SUFFIX_QUERY_DIR:-unset} prefix_query_max_count=${PREFIX_QUERY_MAX_COUNT} prefix_reload_between_queries=${PREFIX_RELOAD_BETWEEN_QUERIES} prefix_max_tokens=${PREFIX_MAX_TOKENS} prefix_save_shards=${PREFIX_SAVE_SHARDS} prefix_target_tp=${PREFIX_TARGET_TP} target_prompt_tokens=${TARGET_PROMPT_TOKENS} prefix_state_chunk_bytes=${PREFIX_STATE_CHUNK_BYTES} prefix_state_calibration_strict_gate=${PREFIX_STATE_CALIBRATION_STRICT_GATE} prefix_state_calibration_safety_ratio=${PREFIX_STATE_CALIBRATION_SAFETY_RATIO} prefix_state_calibration_safety_min_bytes=${PREFIX_STATE_CALIBRATION_SAFETY_MIN_BYTES} hflc_allow_unsafe_long_piecewise=${HFLC_ALLOW_UNSAFE_LONG_PIECEWISE} hflc_long_piecewise_query_disable_tokens=${HFLC_LONG_PIECEWISE_QUERY_DISABLE_TOKENS} phase_mode=${PHASE_MODE} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} vllm_dtype=${VLLM_DTYPE} kv_cache_dtype=${KV_CACHE_DTYPE} calculate_kv_scales=${VLLM_CALCULATE_KV_SCALES} vllm_attention_backend=${VLLM_ATTENTION_BACKEND:-unset} vllm_compilation_config=${VLLM_COMPILATION_CONFIG:-unset} fp8_coherence_guardrails=${FP8_COHERENCE_GUARDRAILS} vllm_block_size=${VLLM_BLOCK_SIZE} vllm_kv_cache_memory_bytes=${VLLM_KV_CACHE_MEMORY_BYTES} vllm_num_gpu_blocks_override=${VLLM_NUM_GPU_BLOCKS_OVERRIDE} vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS} vllm_max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} vllm_disable_custom_all_reduce=${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0} vllm_custom_all_reduce_max_size_mb=${VLLM_CUSTOM_ALL_REDUCE_MAX_SIZE_MB:-0} kv_calibration_gate=${KV_CALIBRATION_GATE} kv_calibration_only=${KV_CALIBRATION_ONLY} kv_calibration_headroom_ratio=${KV_CALIBRATION_HEADROOM_RATIO} kv_calibration_headroom_min_blocks=${KV_CALIBRATION_HEADROOM_MIN_BLOCKS} kv_calibration_assumed_max_num_seqs=${KV_CALIBRATION_ASSUMED_MAX_NUM_SEQS} kv_calibration_metrics_timeout_sec=${KV_CALIBRATION_METRICS_TIMEOUT_SEC} measure_k2ft=${MEASURE_K2FT} k2ft_delta_tokens=${K2FT_DELTA_TOKENS} k2ft_min_probe_tokens=${K2FT_MIN_PROBE_TOKENS} k2ft_runs=${K2FT_RUNS} emit_prefill_progress=${EMIT_PREFILL_PROGRESS} prefill_progress_interval_sec=${PREFILL_PROGRESS_INTERVAL_SEC} prefill_progress_timeout_sec=${PREFILL_PROGRESS_TIMEOUT_SEC} adaptive_timeout=${ADAPTIVE_REQUEST_TIMEOUT} scale=${ADAPTIVE_TIMEOUT_SCALE} extra_sec=${ADAPTIVE_TIMEOUT_EXTRA_SEC} cap_sec=${ADAPTIVE_TIMEOUT_CAP_SEC} sclk_guard=${TIMEOUT_SCLK_GUARD} sclk_min_mhz=${TIMEOUT_SCLK_MIN_MHZ} sclk_samples=${TIMEOUT_SCLK_SAMPLE_COUNT} sclk_interval_sec=${TIMEOUT_SCLK_SAMPLE_INTERVAL_SEC} sclk_min_busy_gpus=${TIMEOUT_SCLK_MIN_BUSY_GPUS} sclk_hit_ratio=${TIMEOUT_SCLK_REQUIRED_HIT_RATIO} sclk_max_extensions=${TIMEOUT_SCLK_MAX_EXTENSIONS} sclk_extension_sec=${TIMEOUT_SCLK_EXTENSION_SEC}"
log "crash_capture enabled=${CRASH_CAPTURE} dir=${CRASH_CAPTURE_DIR} set_ulimit=${CRASH_CAPTURE_SET_ULIMIT} set_core_pattern=${CRASH_CAPTURE_SET_CORE_PATTERN} core_pattern=${CRASH_CAPTURE_CORE_PATTERN} enable_hsa_debug=${CRASH_CAPTURE_ENABLE_HSA_DEBUG} merge=${CRASH_CAPTURE_MERGE} dmesg_lines=${CRASH_CAPTURE_DMESG_LINES}"

if [[ "${PHASE_MODE}" == "both" || "${PHASE_MODE}" == "phase1" ]]; then
  run_phase "phase1" "${PH1_DIR}" "${PHASE1_MAX_MODEL_LEN}" "${PHASE1_SWEEP}" "${PHASE1_ALLOW_LONG}"
fi
if [[ "${PHASE_MODE}" == "both" || "${PHASE_MODE}" == "phase2" ]]; then
  run_phase "phase2" "${PH2_DIR}" "${PHASE2_MAX_MODEL_LEN}" "${PHASE2_SWEEP}" "1"
fi

# Post-run cache cleanup. Honors POST_RUN_CACHE_CLEANUP from the env (the
# Makefile maps HFLC_FP8_POST_RUN_CACHE_CLEANUP through to here). Triton
# inductor + torch_compile caches under phase{1,2}/server_compile_cache/ are
# typically 3+ GB per L; without this, multi-L campaigns leak ~50 GB+ per
# sweep. The standalone target hf-long-context-clean-server-compile-cache
# does the same job offline.
case "${POST_RUN_CACHE_CLEANUP:-none}" in
  triton)
    for ph in phase1 phase2; do
      d="${RUN_DIR}/${ph}/server_compile_cache/triton_cache"
      [[ -d "${d}" ]] && { log "post_run_cleanup remove ${d}"; rm -rf "${d}"; } || true
    done
    ;;
  all)
    for ph in phase1 phase2; do
      d="${RUN_DIR}/${ph}/server_compile_cache"
      [[ -d "${d}" ]] && { log "post_run_cleanup remove ${d}"; rm -rf "${d}"; } || true
    done
    ;;
  none|"")
    ;;
  *)
    log "post_run_cleanup unknown POST_RUN_CACHE_CLEANUP=${POST_RUN_CACHE_CLEANUP}; expected triton|all|none"
    ;;
esac

log "all_done"
