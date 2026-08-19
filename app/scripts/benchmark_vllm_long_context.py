#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark long-context TTFT/decode throughput through vLLM OpenAI API."""
# pylint: disable=line-too-long,too-many-lines,too-many-arguments
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-nested-blocks,too-many-return-statements,too-many-statements
# pylint: disable=broad-exception-caught,chained-comparison,simplifiable-condition
# pylint: disable=no-member

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib import error as url_error
from urllib import request as url_request

import torch
from transformers import AutoConfig, AutoTokenizer

try:
    from vllm.platforms import current_platform
except Exception:  # noqa: BLE001
    class _FallbackCurrentPlatform:
        """Best-effort fallback used when vLLM cannot import in unit tests."""

        @staticmethod
        def fp8_dtype() -> Any:
            for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
                dtype = getattr(torch, name, None)
                if dtype is not None:
                    return dtype
            raise RuntimeError("vLLM current_platform is unavailable and no FP8 dtype fallback exists")

    current_platform = _FallbackCurrentPlatform()

from _paths import repo_path
from long_context_serving.benchmark_common import (
    add_common_sweep_args,
    append_jsonl_row as _append_jsonl_row,
    as_float as _as_float,
    build_basic_sweep_summary_rows,
    build_run_summary_payload,
    build_summary_base_kwargs,
    build_substrate_prompt_seed_from_args,
    build_timing_metrics_row,
    format_k_tokens as _format_k_tokens,
    load_jsonl_dicts as _load_jsonl,
    make_iteration_measurement_row,
    now_iso as _now_iso,
    parse_sweep_targets as _parse_sweep_targets,
    parse_token_list as _parse_token_list,
    resolve_include_exts as _resolve_include_exts,
    timestamp_run_id as _timestamp_run_id,
    update_measure_progress,
    write_sweep_summary_artifacts,
)
from long_context_serving.host_env import collect_host_env, collect_mla_kernel_fingerprint
from long_context_serving.io_utils import write_records
from long_context_serving.kv_capacity_calibration import (
    build_kv_capacity_calibration,
    render_kv_calibration_markdown,
    wait_for_kv_startup_metrics,
)
from long_context_serving import prefix_session as _prefix_session_module
from long_context_serving.prompt_shape import (
    PromptShapeError,
    load_prompt_shape_spec,
    render_messages,
)
from long_context_serving.text_utils import extract_response_text, stream_chat_completions
from long_context_serving.text_utils import stream_text_completions

try:
    from tqdm.auto import tqdm as TQDM_PROGRESS
except Exception:  # noqa: BLE001
    TQDM_PROGRESS = None

_prefix_session_lib: Any = _prefix_session_module


_CJK_CHAR_RE = re.compile(
    "["
    "\u3400-\u4DBF"
    "\u4E00-\u9FFF"
    "\uF900-\uFAFF"
    "\U00020000-\U0002A6DF"
    "\U0002A700-\U0002B73F"
    "\U0002B740-\U0002B81F"
    "\U0002B820-\U0002CEAF"
    "\U0002CEB0-\U0002EBEF"
    "\U00030000-\U0003134F"
    "]"
)

_KIMI_MLA_HF_OVERRIDE_KEYS = (
    "kimi_mla_sliding_window_tokens",
    "kimi_mla_sink_keep_tokens",
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _sha256_hex_bytes(data: bytes) -> str:
    """Returns a SHA256 hex digest for raw bytes."""

    return hashlib.sha256(data).hexdigest()


def _build_fp8_checkpoint_contract(kv_cache_dtype: str) -> Dict[str, Any]:
    """Builds the exact FP8 decode contract used by this runtime.

    Args:
        kv_cache_dtype: User-facing KV cache dtype token.

    Returns:
        Manifest-ready dictionary describing the FP8 storage format. Returns an
        empty dictionary when the KV cache is not FP8.
    """

    token = str(kv_cache_dtype or "").strip().lower()
    if not token.startswith("fp8"):
        return {}
    try:
        fp8_dtype = current_platform.fp8_dtype()
    except Exception:  # noqa: BLE001
        return {
            "fp8_storage_format": token,
            "fp8_storage_format_source": "fallback_kv_cache_dtype_token",
            "fp8_decode_lut_f32_bits": [],
            "fp8_decode_lut_sha256": "",
        }
    raw = torch.arange(256, dtype=torch.uint8)
    decoded = raw.view(fp8_dtype).to(torch.float32).contiguous()
    bit_words = [
        int(word) & 0xFFFFFFFF
        for word in decoded.view(torch.int32).tolist()
    ]
    lut_bytes = struct.pack("<256I", *bit_words)
    return {
        "fp8_storage_format": str(fp8_dtype),
        "fp8_storage_format_source": "current_platform.fp8_dtype",
        "fp8_decode_lut_f32_bits": bit_words,
        "fp8_decode_lut_sha256": _sha256_hex_bytes(lut_bytes),
    }


def _annotate_tensor_files_with_fp8_contract(
    tensor_files: List[Dict[str, Any]],
    fp8_contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Annotates uint8 prefix-state tensor entries with FP8 contract metadata.

    Args:
        tensor_files: Existing manifest tensor rows.
        fp8_contract: Manifest-level FP8 contract metadata.

    Returns:
        Annotated tensor rows.
    """

    if not fp8_contract:
        return tensor_files
    out: List[Dict[str, Any]] = []
    for row in tensor_files:
        entry = dict(row)
        if str(entry.get("dtype") or "") == "torch.uint8":
            entry["fp8_storage_format"] = str(
                fp8_contract.get("fp8_storage_format") or ""
            )
            entry["fp8_decode_lut_sha256"] = str(
                fp8_contract.get("fp8_decode_lut_sha256") or ""
            )
        out.append(entry)
    return out


def _contains_cjk(text: str) -> bool:
    """Returns whether decoded token text contains any CJK character."""

    return bool(_CJK_CHAR_RE.search(str(text or "")))


def _build_english_only_output_mask(
    tokenizer: Any,
    *,
    mode: str,
    bias_value: float,
    vocab_size_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build generation-time token controls that suppress CJK output."""

    return _prefix_session_lib.build_english_only_output_mask(
        tokenizer,
        mode=mode,
        bias_value=bias_value,
        vocab_size_limit=vocab_size_limit,
    )


def _apply_generation_output_mask(
    payload: Dict[str, Any],
    output_mask_payload: Optional[Mapping[str, Any]],
) -> None:
    """Merges generation-time output masking into an OpenAI request payload."""

    if not output_mask_payload:
        return
    for key, value in output_mask_payload.items():
        payload[key] = value


def _maybe_upgrade_legacy_fp8_manifest(
    session_dir: Path,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Backfills missing FP8 decode metadata for legacy manifests in-place.

    Args:
        session_dir: Prefix session directory containing the manifest.
        manifest: Parsed manifest dictionary.

    Returns:
        Possibly upgraded manifest dictionary.
    """

    kv_cache_dtype = str(manifest.get("kv_cache_dtype") or "")
    if not str(kv_cache_dtype).lower().startswith("fp8"):
        return manifest
    if str(manifest.get("fp8_storage_format") or "").strip():
        return manifest
    fp8_contract = _build_fp8_checkpoint_contract(kv_cache_dtype)
    if not fp8_contract:
        return manifest
    updated = dict(manifest)
    updated.update(fp8_contract)
    updated["fp8_storage_format_source"] = "legacy_inferred_current_platform"
    updated["tensor_files"] = _annotate_tensor_files_with_fp8_contract(
        [dict(row) for row in list(updated.get("tensor_files") or [])],
        fp8_contract,
    )
    (session_dir / "prefix_session_manifest.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return updated


def _fetch_prefill_progress(base_url: str, timeout_sec: float) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/debug/prefill_progress"
    try:
        with url_request.urlopen(url, timeout=max(1.0, float(timeout_sec))) as resp:
            body = resp.read()
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "requests": [],
            "error": str(exc)[:1000],
            "source": "vllm_prefill_progress_endpoint",
        }
    if not isinstance(payload, dict):
        return {
            "requests": [],
            "error": f"invalid_payload_type={type(payload).__name__}",
            "source": "vllm_prefill_progress_endpoint",
        }
    payload.setdefault("requests", [])
    payload.setdefault("source", "vllm_prefill_progress_endpoint")
    return payload


def _resolve_active_internal_request_id(
    base_url: str,
    external_request_id: str,
    timeout_sec: int,
    retry_total_sec: float = 0.0,
    poll_interval_sec: float = 0.05,
) -> str:
    """Resolve an external OpenAI request ID to the active internal engine ID.

    Args:
        base_url: vLLM server base URL.
        external_request_id: OpenAI-visible request ID such as ``chatcmpl-*``.
        timeout_sec: Endpoint timeout in seconds.
        retry_total_sec: Optional extra wall-clock time to keep polling for a
            live internal request mapping. This helps short streaming capture
            requests whose external stream ID becomes visible slightly before
            vLLM publishes the active internal request mapping.
        poll_interval_sec: Delay between retry polls when ``retry_total_sec``
            is positive.

    Returns:
        Internal request ID when one is currently active, otherwise an empty
        string.
    """

    external = str(external_request_id or "").strip()
    if not external:
        return ""
    deadline = time.perf_counter() + max(0.0, float(retry_total_sec))
    interval = max(0.01, float(poll_interval_sec))
    while True:
        req = url_request.Request(
            f"{base_url.rstrip('/')}/resolve_internal_request_ids",
            method="POST",
            data=json.dumps({"external_request_id": external}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with url_request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            payload = {}
        internal_ids = payload.get("internal_request_ids")
        if isinstance(internal_ids, list):
            for request_id in internal_ids:
                internal = str(request_id or "").strip()
                if internal:
                    return internal
        progress_payload = _fetch_prefill_progress(
            base_url=base_url,
            timeout_sec=float(timeout_sec),
        )
        requests = list(progress_payload.get("requests") or [])
        if len(requests) == 1:
            internal = str((requests[0] or {}).get("request_id") or "").strip()
            if internal:
                return internal
        if time.perf_counter() >= deadline:
            break
        time.sleep(interval)
    return ""


def _describe_request_prefix_apc_rows(
    *,
    base_url: str,
    request_id: str,
    prefix_token_count: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    """Fetch the scheduler APC row layout for one active internal request.

    Args:
        base_url: vLLM server base URL.
        request_id: Active internal engine request ID.
        prefix_token_count: Number of reusable prefix tokens.
        timeout_sec: Endpoint timeout in seconds.

    Returns:
        Parsed JSON response from the dev-only APC row description endpoint.
    """

    req = url_request.Request(
        f"{base_url.rstrip('/')}/describe_request_prefix_apc_rows",
        method="POST",
        data=json.dumps(
            {
                "request_id": str(request_id or ""),
                "prefix_token_count": int(prefix_token_count),
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with url_request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _scheduler_source_group_rows(payload: Mapping[str, Any]) -> Dict[str, List[int]]:
    """Extract live KV-cache row IDs from a scheduler APC-row payload.

    Args:
        payload: JSON payload returned by ``/describe_request_prefix_apc_rows``.

    Returns:
        A JSON-serializable mapping from KV-cache group ID to positive live row
        IDs suitable for direct prefix-state serialization.
    """

    def _row_positive_ids(row: Mapping[str, Any]) -> List[int]:
        live_ids = row.get("prefix_live_block_ids")
        if not isinstance(live_ids, list):
            live_ids = row.get("live_block_ids")
        if not isinstance(live_ids, list):
            live_ids = row.get("positive_row_ids")
        if not isinstance(live_ids, list):
            live_ids = row.get("allocated_block_ids")
        if not isinstance(live_ids, list):
            live_ids = row.get("cached_block_ids")
        positive_ids = sorted({int(value) for value in list(live_ids or []) if int(value) > 0})
        if positive_ids:
            return [int(value) for value in positive_ids]

        # Streaming compact layouts expose sparse tail rows at offset
        # ``num_full_blocks`` via ``tail_probe_block_ids`` even when the main
        # ``live_block_ids`` prefix slice is empty for linear groups.
        try:
            num_full_blocks = int(row.get("num_full_blocks", -1) or -1)
        except Exception:
            num_full_blocks = -1
        tail_probe = row.get("tail_probe_block_ids")
        if isinstance(tail_probe, Mapping) and num_full_blocks >= 0:
            tail_value = tail_probe.get(str(num_full_blocks))
            if tail_value is not None and int(tail_value) > 0:
                return [int(tail_value)]
        return []

    rows: Dict[str, List[int]] = {}
    manager_rows = list(payload.get("manager_rows") or [])
    linear_target_last_rows = {
        str(key): int(value)
        for key, value in dict(payload.get("linear_target_last_rows") or {}).items()
        if int(value) > 0
    }
    linear_manifest_keys = ["-1", "1", "2"]
    linear_key_index = 0
    for row in manager_rows:
        if not isinstance(row, Mapping):
            continue
        group_index = row.get("group_index")
        kv_cache_group_id = row.get("kv_cache_group_id", -1)
        group_id = int(
            group_index if group_index is not None else kv_cache_group_id
        )
        if group_id < 0:
            continue
        positive_ids = _row_positive_ids(row)
        try:
            num_full_blocks = int(row.get("num_full_blocks", 0) or 0)
        except Exception:
            num_full_blocks = 0
        dense_like = num_full_blocks > 0 and len(positive_ids) >= num_full_blocks
        if dense_like:
            if positive_ids:
                rows[str(group_id)] = [int(value) for value in positive_ids]
            continue

        aligned_row = 0
        if linear_key_index < len(linear_manifest_keys):
            aligned_row = int(
                linear_target_last_rows.get(linear_manifest_keys[linear_key_index]) or 0
            )
            linear_key_index += 1
        if aligned_row > 0:
            # Keep streaming sparse groups in the compact aligned layout.
            # Appending live tail rows here silently reintroduces transient
            # continuation state like [1, 20] when the save contract should be
            # just [1].
            rows[str(group_id)] = [int(aligned_row)]
        elif positive_ids:
            rows[str(group_id)] = [int(value) for value in positive_ids]
    return rows


def _scheduler_source_group_rows_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return compact sparse rows plus live manager-row lineage for capture RPC."""
    return {
        "payload_version": 2,
        "group_rows": _scheduler_source_group_rows(payload),
        "manager_rows": [
            dict(row)
            for row in list(payload.get("manager_rows") or [])
            if isinstance(row, Mapping)
        ],
        "accepted_tokens": 1,
    }


def _streaming_layout_from_scheduler_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize scheduler APC rows into compact StreamingLLM session layout."""

    manager_rows_in = list(payload.get("manager_rows") or [])
    linear_target_last_rows = {
        str(key): int(value)
        for key, value in dict(payload.get("linear_target_last_rows") or {}).items()
        if int(value) > 0
    }
    global_target_rows = [
        int(value) for value in list(payload.get("global_target_rows") or []) if int(value) > 0
    ]
    try:
        global_target_group_id = int(payload.get("global_target_group_id") or -1)
    except Exception:
        global_target_group_id = -1

    normalized_rows: List[Dict[str, Any]] = []
    for row in manager_rows_in:
        if not isinstance(row, Mapping):
            continue
        live_ids = row.get("prefix_live_block_ids")
        if not isinstance(live_ids, list):
            live_ids = row.get("live_block_ids")
        if not isinstance(live_ids, list):
            live_ids = row.get("positive_row_ids")
        if not isinstance(live_ids, list):
            live_ids = row.get("allocated_block_ids")
        if not isinstance(live_ids, list):
            live_ids = []
        cached_pairs = [
            (int(index), int(value))
            for index, value in enumerate(live_ids)
            if int(value) > 0
        ]
        if not cached_pairs:
            try:
                num_full_blocks = int(row.get("num_full_blocks", -1) or -1)
            except Exception:
                num_full_blocks = -1
            tail_probe = row.get("tail_probe_block_ids")
            if isinstance(tail_probe, Mapping) and num_full_blocks >= 0:
                tail_value = tail_probe.get(str(num_full_blocks))
                if tail_value is not None and int(tail_value) > 0:
                    cached_pairs = [(int(num_full_blocks), int(tail_value))]
        normalized_rows.append(
            {
                "group_index": int(row.get("group_index") or 0),
                "kv_cache_group_id": int(row.get("kv_cache_group_id") or 0),
                "block_size": int(row.get("block_size") or 0),
                "num_full_blocks": int(row.get("num_full_blocks") or 0),
                "cached_block_offsets": [int(index) for index, _ in cached_pairs],
                "cached_block_ids": [int(value) for _, value in cached_pairs],
            }
        )

    if not global_target_rows and normalized_rows:
        dense_row = max(
            normalized_rows,
            key=lambda row: len(list(row.get("cached_block_ids") or [])),
        )
        global_target_rows = [
            int(value)
            for value in list(dense_row.get("cached_block_ids") or [])
            if int(value) > 0
        ]
        global_target_group_id = int(dense_row.get("kv_cache_group_id") or -1)

    if not linear_target_last_rows and normalized_rows:
        linear_manifest_keys = ["-1", "1", "2"]
        linear_candidates = [
            row
            for row in sorted(
                normalized_rows,
                key=lambda row: len(list(row.get("cached_block_ids") or [])),
            )
            if len(list(row.get("cached_block_ids") or [])) == 1
        ]
        for key, row in zip(linear_manifest_keys, linear_candidates):
            cached_ids = [
                int(value)
                for value in list(row.get("cached_block_ids") or [])
                if int(value) > 0
            ]
            if cached_ids:
                linear_target_last_rows[str(key)] = int(cached_ids[-1])

    return {
        "full_block_hash_count": int(payload.get("full_block_hash_count") or 0),
        "linear_target_last_rows": linear_target_last_rows,
        "global_target_rows": global_target_rows,
        "source_group_rows": _scheduler_source_group_rows(payload),
        "manager_rows": normalized_rows,
        "global_target_group_id": int(global_target_group_id),
    }


class _PrefillProgressRecorder:
    """Poll vLLM prefill pointer endpoint during one measured request."""

    def __init__(
        self,
        *,
        base_url: str,
        out_path: Path,
        target_index: int,
        target_prompt_tokens_requested: int,
        target_prompt_tokens_effective: int,
        iteration: int,
        interval_sec: float,
        poll_timeout_sec: float,
        progress_cb: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self.base_url = str(base_url)
        self.out_path = out_path
        self.target_index = int(target_index)
        self.target_prompt_tokens_requested = int(target_prompt_tokens_requested)
        self.target_prompt_tokens_effective = int(target_prompt_tokens_effective)
        self.iteration = int(iteration)
        self.interval_sec = max(0.005, float(interval_sec))
        self.poll_timeout_sec = max(1.0, float(poll_timeout_sec))
        self._progress_cb = progress_cb
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._had_not_available = False
        self._last_point: Dict[str, Any] = {}
        self._started = False
        self._stopped = False

    def _build_row(
        self,
        *,
        status: str,
        request_id: str = "",
        done_tokens: int = 0,
        total_tokens: int = 0,
        prefill_pct: float = 0.0,
        engine_step: int = 0,
        coarse_state: str = "",
        request_status: str = "",
        note: str = "",
    ) -> Dict[str, Any]:
        return {
            "timestamp_utc": _now_iso(),
            "target_index": int(self.target_index),
            "target_prompt_tokens_requested": int(self.target_prompt_tokens_requested),
            "target_prompt_tokens_effective": int(self.target_prompt_tokens_effective),
            "iteration": int(self.iteration),
            "request_id": str(request_id),
            "prefill_done_tokens": int(max(0, done_tokens)),
            "prefill_total_tokens": int(max(0, total_tokens)),
            "prefill_pct": round(float(max(0.0, prefill_pct)), 6),
            "engine_step": int(max(0, engine_step)),
            "coarse_state": str(coarse_state),
            "request_status": str(request_status),
            "source": "vllm_prefill_progress_endpoint",
            "status": str(status),
            "note": str(note),
        }

    def _next_interval_sec(self, row: Mapping[str, Any]) -> float:
        """Use fine-grained polling near the prefill/decode boundary."""

        total = int(row.get("prefill_total_tokens") or 0)
        done = int(row.get("prefill_done_tokens") or 0)
        remaining = max(0, total - done)
        coarse_state = str(row.get("coarse_state") or "")
        if coarse_state == "decode":
            return 0.005
        if total > 0 and (
            remaining <= 131072
            or (done / float(total)) >= 0.995
        ):
            return min(float(self.interval_sec), 0.01)
        return float(self.interval_sec)

    def _pick_request(self, requests: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(requests, list):
            return None
        rows = [r for r in requests if isinstance(r, dict)]
        if not rows:
            return None
        # Prefer the largest active prefill to track long single-request runs.
        rows.sort(
            key=lambda r: (
                int(r.get("prefill_total_tokens") or 0),
                int(r.get("prefill_done_tokens") or 0),
            ),
            reverse=True,
        )
        return rows[0]

    def _run(self) -> None:
        while not self._stop_event.is_set():
            next_sleep_sec = float(self.interval_sec)
            payload = _fetch_prefill_progress(self.base_url, self.poll_timeout_sec)
            err = str(payload.get("error") or "")
            if err:
                if not self._had_not_available:
                    self._had_not_available = True
                    _append_jsonl_row(
                        self.out_path,
                        self._build_row(
                            status="not_available",
                            note=f"endpoint_error={err[:500]}",
                        ),
                    )
            req = self._pick_request(payload.get("requests"))
            if req is not None:
                done = int(req.get("prefill_done_tokens") or 0)
                total = int(req.get("prefill_total_tokens") or 0)
                pct = float(req.get("prefill_pct") or 0.0)
                row = self._build_row(
                    status="running",
                    request_id=str(req.get("request_id") or ""),
                    done_tokens=done,
                    total_tokens=total,
                    prefill_pct=pct,
                    engine_step=int(req.get("engine_step") or 0),
                    coarse_state=str(req.get("coarse_state") or ""),
                    request_status=str(req.get("request_status") or ""),
                )
                self._last_point = dict(row)
                _append_jsonl_row(self.out_path, row)
                next_sleep_sec = self._next_interval_sec(row)
                if callable(self._progress_cb):
                    try:
                        self._progress_cb(dict(row))
                    except Exception as exc:  # noqa: BLE001
                        _append_jsonl_row(
                            self.out_path,
                            self._build_row(
                                status="callback_error",
                                request_id=str(row.get("request_id") or ""),
                                done_tokens=int(row.get("prefill_done_tokens") or 0),
                                total_tokens=int(row.get("prefill_total_tokens") or 0),
                                prefill_pct=float(row.get("prefill_pct") or 0.0),
                                engine_step=int(row.get("engine_step") or 0),
                                coarse_state=str(row.get("coarse_state") or ""),
                                request_status=str(row.get("request_status") or ""),
                                note=f"progress_cb_error={str(exc)[:500]}",
                            ),
                        )
            elif (
                callable(self._progress_cb)
                and str(self._last_point.get("status") or "") == "running"
                and str(self._last_point.get("request_id") or "").strip()
            ):
                missing_row = dict(self._last_point)
                missing_row["status"] = "running_disappeared"
                missing_row["note"] = "request_missing_after_running"
                self._last_point = dict(missing_row)
                try:
                    self._progress_cb(dict(missing_row))
                except Exception as exc:  # noqa: BLE001
                    _append_jsonl_row(
                        self.out_path,
                        self._build_row(
                            status="callback_error",
                            request_id=str(missing_row.get("request_id") or ""),
                            done_tokens=int(
                                missing_row.get("prefill_done_tokens") or 0
                            ),
                            total_tokens=int(
                                missing_row.get("prefill_total_tokens") or 0
                            ),
                            prefill_pct=float(
                                missing_row.get("prefill_pct") or 0.0
                            ),
                            engine_step=int(missing_row.get("engine_step") or 0),
                            coarse_state=str(
                                missing_row.get("coarse_state") or ""
                            ),
                            request_status=str(
                                missing_row.get("request_status") or ""
                            ),
                            note=f"progress_cb_error={str(exc)[:500]}",
                        ),
                    )
                next_sleep_sec = min(next_sleep_sec, 0.005)
            self._stop_event.wait(next_sleep_sec)

    def start(self) -> None:
        """Start the background prefill-progress recorder thread."""

        if self._started:
            return
        self._started = True
        bootstrap = self._build_row(
            status="initialized",
            done_tokens=0,
            total_tokens=int(self.target_prompt_tokens_effective),
            prefill_pct=0.0,
            note="recorder_started",
        )
        self._last_point = dict(bootstrap)
        _append_jsonl_row(self.out_path, bootstrap)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, *, final_status: str, note: str = "") -> None:
        """Stop the recorder thread and emit one final terminal record."""

        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        last = dict(self._last_point) if self._last_point else {}
        final_row = self._build_row(
            status=str(final_status),
            request_id=str(last.get("request_id") or ""),
            done_tokens=int(last.get("prefill_done_tokens") or 0),
            total_tokens=int(last.get("prefill_total_tokens") or 0),
            prefill_pct=float(last.get("prefill_pct") or 0.0),
            engine_step=int(last.get("engine_step") or 0),
            coarse_state=str(last.get("coarse_state") or ""),
            request_status=str(last.get("request_status") or ""),
            note=note,
        )
        _append_jsonl_row(self.out_path, final_row)


def _extract_cached_prompt_tokens(usage: Mapping[str, Any]) -> int:
    """Best-effort extraction of cached prompt-token count from usage payload."""
    if not isinstance(usage, Mapping):
        return 0
    direct = usage.get("cached_tokens")
    if isinstance(direct, (int, float)):
        return max(0, int(direct))
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        cached = details.get("cached_tokens")
        if isinstance(cached, (int, float)):
            return max(0, int(cached))
    return 0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(str(payload).encode("utf-8"))


def _chat_template_hash(tokenizer: Any) -> str:
    """Hash the tokenizer chat template for prefix-session compatibility."""

    return _prefix_session_lib.chat_template_hash(tokenizer)


def _parse_nonnegative_cli_int(raw: str) -> int:
    token = str(raw or "").strip()
    try:
        value = int(token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected non-negative integer, got {raw!r}"
        ) from exc
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"expected non-negative integer, got {value}"
        )
    return value


def _resolve_kimi_mla_hf_overrides(
    args: argparse.Namespace,
) -> Dict[str, int]:
    # StreamingLLM sliding-window / sink-keep trimming was an internal test
    # feature and is not part of the public release: no hf_overrides are sent.
    return {}


def _expand_prompt_seed_ids_to_target(
    prompt_seed_ids: List[int],
    *,
    target_tokens: int,
    repeat_to_target: bool = True,
) -> Tuple[List[int], Dict[str, Any]]:
    """Repeat a prompt seed until it reaches the requested token length.

    This is used for synthetic ultra-long context sweeps once the corpus has
    been fully exhausted. It keeps the launch path honest about the requested
    length while avoiding a hard stop at the source corpus size.
    """

    target = max(1, int(target_tokens))
    base = [int(x) for x in prompt_seed_ids]
    base_len = len(base)
    if base_len <= 0 or target <= base_len:
        return base[:target], {
            "prompt_seed_extension_mode": "none",
            "prompt_seed_source_tokens": int(base_len),
            "prompt_seed_expanded_tokens": int(min(target, base_len)),
            "prompt_seed_repeat_count": 1 if base_len > 0 else 0,
        }

    if not bool(repeat_to_target):
        return base, {
            "prompt_seed_extension_mode": "preserve_shortfall",
            "prompt_seed_source_tokens": int(base_len),
            "prompt_seed_expanded_tokens": int(base_len),
            "prompt_seed_repeat_count": 1,
        }

    expanded: List[int] = list(base)
    repeat_count = 1
    while len(expanded) < target:
        remaining = target - len(expanded)
        expanded.extend(base[:remaining])
        repeat_count += 1
    return expanded, {
        "prompt_seed_extension_mode": "repeat_to_target",
        "prompt_seed_source_tokens": int(base_len),
        "prompt_seed_expanded_tokens": int(len(expanded)),
        "prompt_seed_repeat_count": int(repeat_count),
    }


def _load_prompt_token_ids_file(path_str: str) -> Tuple[List[int], Dict[str, Any]]:
    """Load exact prompt token IDs from a JSON artifact.

    Args:
        path_str: Path to a JSON file containing either a bare list of token IDs
            or an object with a ``prompt_token_ids`` field.

    Returns:
        Tuple[List[int], Dict[str, Any]]: Loaded prompt token IDs plus metadata
        describing the source artifact.

    Raises:
        ValueError: If the artifact is missing token IDs or has an unsupported
            structure.
    """
    path = Path(path_str).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    token_ids_raw: Any
    source_label = "json_list"
    if isinstance(payload, Mapping):
        if "prompt_token_ids" not in payload:
            raise ValueError(
                "prompt token artifact object must contain a 'prompt_token_ids' field"
            )
        token_ids_raw = payload.get("prompt_token_ids")
        source_label = str(payload.get("prompt_token_ids_source") or "prompt_token_ids_field")
    elif isinstance(payload, list):
        token_ids_raw = payload
    else:
        raise ValueError(
            "prompt token artifact must be a JSON list or object containing 'prompt_token_ids'"
        )
    if not isinstance(token_ids_raw, list):
        raise ValueError("'prompt_token_ids' must be a JSON list of integers")
    token_ids = [int(token_id) for token_id in token_ids_raw]
    if not token_ids:
        raise ValueError("prompt token artifact is empty")
    return token_ids, {
        "source": "prompt_token_ids_file",
        "path": str(path),
        "count": int(len(token_ids)),
        "sha256": _sha256_text(json.dumps(token_ids, separators=(",", ":"))),
        "artifact_format": str(source_label),
    }


def _serialize_hf_overrides(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


_STRICT_ENV_KEYS = [
    "HIP_VISIBLE_DEVICES",
    "PYTHONPATH",
    "VLLM_BATCH_INVARIANT",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_FP4BMM",
    "VLLM_ROCM_USE_AITER_MLA",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_ATTENTION_BACKEND",
]


_LEGACY_AITER_SERVER_ENV_DROP_KEYS = (
    "VLLM_AITER_HEAD4_DECODE_MODE",
    "VLLM_AITER_HEAD4_FALLBACK_MODE",
    "VLLM_AITER_SUB16_MODE",
    "VLLM_AITER_SUB16_FALLBACK",
    "VLLM_AITER_SUB16_COMPUTE_MODE",
    "VLLM_AITER_SUB16_APPLY",
    "VLLM_AITER_SUB16_BF16_KV_SCALE_MODE",
    "VLLM_AITER_SUB16_SHADOW_CHECK",
    "VLLM_AITER_SUB16_SHADOW_MAX_ABS_ERR",
    "VLLM_AITER_SUB16_SHADOW_MAX_REL_ERR",
    "VLLM_AITER_SUB16_BF16_WORKSPACE_MAX_BYTES",
    "VLLM_AITER_SUB16_BF16_WORKSPACE_CHUNK_ROWS",
    "VLLM_AITER_SUB16_FAST_MAX_KV_TOKENS",
    "VLLM_AITER_MLA_NUM_KV_SPLITS_OVERRIDE",
    "VLLM_AITER_MLA_STAGE1_REF_QH16_Q1",
    "VLLM_AITER_MLA_STAGE1_REF_QH16_Q1_STRICT_FAIL",
    "VLLM_AITER_MLA_STAGE1_MODE",
    "VLLM_AITER_MLA_PROFILE",
    "VLLM_AITER_MLA_PROFILE_MAX_STEPS",
    "VLLM_AITER_MLA_STAGE1_MFMA_MIN_TOKENS",
    "VLLM_AITER_MLA_STAGE1_MFMA_TARGET_TOKENS",
    "VLLM_AITER_MLA_STAGE1_MFMA_IMPL",
    "VLLM_AITER_MLA_HIP_LDS_STAGE12_TREE",
    "VLLM_AITER_MLA_HIP_LDS_VARIANT",
    "VLLM_AITER_MLA_STAGE1_WEIGHTED_V_MODE",
    "VLLM_AITER_MLA_STAGE1_MFMA_BLOCK_N",
    "VLLM_AITER_MLA_TARGET_TOKENS_PER_SPLIT",
    "VLLM_AITER_MLA_STAGE2_MODE",
    "VLLM_MLA_FP8_DECODE_BYPASS_QUANT",
)


_NATIVE_SERVER_ENV_DROP_KEYS = (
    "VLLM_ALL2ALL_BACKEND",
    "VLLM_BASE_URL",
    "VLLM_BIN",
    "VLLM_BATCH_INVARIANT",
    "VLLM_COMPILATION_CONFIG",
    "VLLM_CUSTOM_ALL_REDUCE_MAX_SIZE_MB",
    "VLLM_DISABLE_CUSTOM_ALL_REDUCE",
    "VLLM_DISABLE_ROCM_SKINNY_GEMM",
    "VLLM_DTYPE",
    "VLLM_DP_SIZE",
    "VLLM_ENABLE_EP",
    "VLLM_ENABLE_PREFIX_CACHING",
    "VLLM_ENFORCE_EAGER",
    "VLLM_KV_CACHE_MEMORY_BYTES",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_NUM_GPU_BLOCKS_OVERRIDE",
    "VLLM_PORT",
    "VLLM_PP_SIZE",
    "VLLM_READY_TIMEOUT_SEC",
    "VLLM_TP_SIZE",
)


def _sanitize_server_env(
    *,
    server_env: Dict[str, str],
    args: argparse.Namespace,
) -> Dict[str, str]:
    """Normalize server environment for native bundled `v0.19` runs.

    Args:
        server_env: Environment that will be passed to `vllm serve`.
        args: Parsed benchmark arguments.

    Returns:
        Sanitized environment mapping.

    The native HF long-context flow passes server behavior explicitly on the
    CLI. This sanitizer removes compatibility-only ambient `VLLM_*` variables
    that come from outer service wrappers and older driver flows so `vllm
    serve` does not see duplicate or unsupported configuration channels.
    """

    runtime_mode = str(getattr(args, "vllm_mla_runtime_mode", "aiter") or "aiter")
    attention_backend = str(getattr(args, "vllm_attention_backend", "") or "")
    batch_invariant = bool(int(getattr(args, "vllm_batch_invariant", 0) or 0))
    preserve_legacy_aiter_env = bool(
        int(getattr(args, "vllm_preserve_legacy_aiter_env", 0) or 0)
    )

    for key in _NATIVE_SERVER_ENV_DROP_KEYS:
        server_env.pop(key, None)

    if runtime_mode == "aiter" and not preserve_legacy_aiter_env:
        # The current native v0.19 AITER lanes use the installed package code,
        # not the older repo-local head4/sub16 env-driven snapshots. Those
        # legacy tuning vars now only add misleading launch metadata and
        # "Unknown vLLM environment variable" warnings at server startup.
        for key in _LEGACY_AITER_SERVER_ENV_DROP_KEYS:
            server_env.pop(key, None)

    if attention_backend == "ROCM_AITER_MLA":
        # Make the bundled flow deterministic instead of inheriting whatever
        # the container happened to export before the benchmark started.
        server_env["VLLM_ROCM_USE_AITER"] = "1"
        server_env["VLLM_ROCM_USE_AITER_FP4BMM"] = "0"
        server_env["VLLM_ROCM_USE_AITER_MLA"] = "1"
        server_env["VLLM_ROCM_USE_AITER_MOE"] = "1"

    if batch_invariant:
        server_env["VLLM_BATCH_INVARIANT"] = "1"

    return server_env


def _strict_env_payload(
    *, vllm_hf_overrides: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "env": {key: str(os.environ.get(key, "")) for key in _STRICT_ENV_KEYS},
        "vllm_hf_overrides": {
            key: int((vllm_hf_overrides or {}).get(key) or 0)
            for key in _KIMI_MLA_HF_OVERRIDE_KEYS
        },
    }


def _strict_env_hash(
    *, vllm_hf_overrides: Optional[Mapping[str, Any]] = None
) -> str:
    payload = _strict_env_payload(vllm_hf_overrides=vllm_hf_overrides)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical)


def _render_server_cmd_artifact(
    cmd: List[str],
    server_env: Mapping[str, str],
    *,
    vllm_hf_overrides: Optional[Mapping[str, Any]] = None,
) -> str:
    lines: List[str] = []
    if vllm_hf_overrides is not None:
        lines.append(
            "# hf_overrides "
            + json.dumps(dict(vllm_hf_overrides), sort_keys=True)
        )
    if str(server_env.get("VLLM_SERVER_DEV_MODE", "")).strip():
        lines.append(
            "# env VLLM_SERVER_DEV_MODE="
            + str(server_env.get("VLLM_SERVER_DEV_MODE") or "")
        )
    if "VLLM_ROCM_USE_SKINNY_GEMM" in server_env:
        lines.append(
            "# env VLLM_ROCM_USE_SKINNY_GEMM="
            + str(server_env.get("VLLM_ROCM_USE_SKINNY_GEMM") or "")
        )
    lines.append(" ".join(shlex.quote(token) for token in cmd))
    return "\n".join(lines) + "\n"


def _build_rocprof_prefix(
    args: argparse.Namespace,
    *,
    out_dir: Path,
) -> Tuple[List[str], Dict[str, Any]]:
    """Builds an optional rocprof wrapper around the vLLM server command."""

    if not bool(getattr(args, "rocprof_enable", False)):
        return [], {}

    rocprof_dir = out_dir / "rocprof"
    rocprof_dir.mkdir(parents=True, exist_ok=True)
    data_dir = rocprof_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_basename = str(getattr(args, "rocprof_output_basename", "") or "").strip()
    if not output_basename:
        output_basename = "server_rocprof"
    mode = str(getattr(args, "rocprof_mode", "") or "").strip().lower()
    if not mode or mode == "auto":
        mode = "sys"

    tool_version = int(getattr(args, "rocprof_tool_version", 0) or 0)
    rocprof_input = str(getattr(args, "rocprof_input", "") or "").strip()

    if mode == "none":
        return [], {}
    if mode not in {"sys", "hip", "hsa"}:
        raise SystemExit(f"Unsupported rocprof mode: {mode}")

    rocprofv3_bin = shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3"
    rocprofv2_bin = shutil.which("rocprofv2") or "/opt/rocm/bin/rocprofv2"
    rocprof_bin = shutil.which("rocprof") or "/opt/rocm/bin/rocprof"

    selected_tool = ""
    selected_bin = ""
    if tool_version == 3:
        if Path(rocprofv3_bin).exists():
            selected_tool = "rocprofv3"
            selected_bin = str(rocprofv3_bin)
    elif tool_version == 2:
        if Path(rocprofv2_bin).exists():
            selected_tool = "rocprofv2"
            selected_bin = str(rocprofv2_bin)
    elif tool_version == 1:
        if Path(rocprof_bin).exists():
            selected_tool = "rocprof"
            selected_bin = str(rocprof_bin)
    else:
        # Auto: prefer the newest profiler that preserves JSON-valued CLI args.
        if Path(rocprofv3_bin).exists():
            selected_tool = "rocprofv3"
            selected_bin = str(rocprofv3_bin)
        elif Path(rocprofv2_bin).exists():
            selected_tool = "rocprofv2"
            selected_bin = str(rocprofv2_bin)
        elif Path(rocprof_bin).exists():
            selected_tool = "rocprof"
            selected_bin = str(rocprof_bin)

    if not selected_bin:
        raise SystemExit(
            "rocprof_requested_but_missing: expected rocprofv3, rocprofv2, or rocprof under /opt/rocm/bin or PATH"
        )

    prefix: List[str] = [selected_bin]
    output_prefix: Path
    summary_output: Optional[Path] = None
    if selected_tool == "rocprofv3":
        output_stem = output_basename
        if output_stem.lower().endswith(".csv"):
            output_stem = output_stem[:-4]
        output_stem = f"{output_stem}_%pid%"
        output_prefix = data_dir / output_stem
        if mode == "sys":
            prefix.append("--sys-trace")
        elif mode == "hip":
            prefix.append("--hip-trace")
        elif mode == "hsa":
            prefix.append("--hsa-trace")
        if bool(int(getattr(args, "rocprof_stats", 0) or 0)):
            prefix.append("--stats")
            prefix.append("--summary")
        prefix.extend(["--output-format", "csv", "json"])
        if rocprof_input:
            prefix.extend(["-i", rocprof_input])
        prefix.extend(["-d", str(data_dir), "-o", str(output_stem), "--"])
    elif selected_tool == "rocprofv2":
        output_stem = output_basename
        if output_stem.lower().endswith(".csv"):
            output_stem = output_stem[:-4]
        output_prefix = rocprof_dir / output_stem
        if mode == "sys":
            prefix.append("--sys-trace")
        elif mode == "hip":
            prefix.append("--hip-trace")
        elif mode == "hsa":
            prefix.append("--hsa-trace")
        if rocprof_input:
            prefix.extend(["-i", rocprof_input])
        prefix.extend(["-d", str(data_dir), "-o", output_stem])
    else:
        output_prefix = rocprof_dir / output_basename
        if tool_version in {1, 2}:
            prefix.extend(["--tool-version", str(tool_version)])
        if bool(int(getattr(args, "rocprof_stats", 0) or 0)):
            prefix.append("--stats")
        if bool(int(getattr(args, "rocprof_timestamp", 0) or 0)):
            prefix.extend(["--timestamp", "on"])
        if mode == "sys":
            prefix.append("--sys-trace")
        elif mode == "hip":
            prefix.append("--hip-trace")
        elif mode == "hsa":
            prefix.append("--hsa-trace")
        if rocprof_input:
            prefix.extend(["-i", rocprof_input])
        prefix.extend(["-d", str(data_dir), "-o", str(output_prefix)])

    return prefix, {
        "enabled": True,
        "bin": str(selected_bin),
        "tool_name": selected_tool,
        "mode": mode,
        "stats": bool(int(getattr(args, "rocprof_stats", 0) or 0)),
        "timestamp": bool(int(getattr(args, "rocprof_timestamp", 0) or 0)),
        "tool_version": int(tool_version),
        "input": rocprof_input,
        "artifact_dir": str(rocprof_dir),
        "output_prefix": str(output_prefix),
        "summary_output": (str(summary_output) if summary_output is not None else ""),
    }


def _sync_launch_env_artifacts(
    *,
    out_dir: Path,
    run_meta: Dict[str, Any],
    launch_env: Mapping[str, Any],
    server_env: Mapping[str, str],
) -> None:
    """Persist launch metadata using the actual server subprocess environment."""

    effective_launch_env = dict(launch_env)
    mirrored_server_keys = (
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_FP4BMM",
        "VLLM_ROCM_USE_AITER_MLA",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_ROCM_AITER_MLA_USE_FLASHINFER_DECODE",
        "FLASHINFER_HIP_MLA_USE_AITER_FASTPATH",
        "FLASHINFER_HIP_MLA_USE_REPO_AITER_SNAPSHOT",
        "FLASHINFER_HIP_MLA_GRAPH_KV_HEADROOM",
    )
    for key in mirrored_server_keys:
        effective_launch_env[key] = str(server_env.get(key, "") or "")
    run_meta["launch_env"] = dict(effective_launch_env)
    run_meta["host_env"] = collect_host_env()
    run_meta["mla_kernel_fingerprint"] = collect_mla_kernel_fingerprint()
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "launch_env.json").write_text(
        json.dumps(effective_launch_env, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _token_ids_to_bytes(token_ids: List[int]) -> bytes:
    if not token_ids:
        return b""
    return b"".join(struct.pack("<I", int(tok)) for tok in token_ids)


def _write_token_ids_bin(path: Path, token_ids: List[int]) -> str:
    """Write uint32 token IDs to disk and return the content digest."""

    return _prefix_session_lib.write_token_ids_bin(path, token_ids)


def _read_token_ids_bin(path: Path) -> Tuple[List[int], str]:
    """Read uint32 token IDs from disk and return ``(token_ids, sha256)``."""

    return _prefix_session_lib.read_token_ids_bin(path)


def _tokenize_text(tokenizer: Any, text: str) -> List[int]:
    """Tokenizes raw text without adding special tokens.

    Args:
        tokenizer: Hugging Face-compatible tokenizer.
        text: Raw text to tokenize.

    Returns:
        List[int]: Token IDs for ``text``.
    """
    return [int(x) for x in tokenizer(str(text), add_special_tokens=False)["input_ids"]]


def _coerce_chat_template_ids(rendered: Any) -> Optional[List[int]]:
    """Normalize ``apply_chat_template(tokenize=True)`` output to a flat id list.

    Depending on the transformers/tokenizer version this can return a plain
    ``List[int]``, a batched list-of-lists, a tensor, or a ``BatchEncoding``/
    dict keyed by ``input_ids``. Returns ``None`` when the value cannot be
    interpreted as a token-id sequence.
    """
    if rendered is None:
        return None
    if hasattr(rendered, "keys"):
        try:
            if "input_ids" in rendered.keys():
                rendered = rendered["input_ids"]
        except Exception:  # noqa: BLE001
            return None
    tolist = getattr(rendered, "tolist", None)
    if callable(tolist):
        try:
            rendered = tolist()
        except Exception:  # noqa: BLE001
            pass
    try:
        seq = list(rendered)
    except Exception:  # noqa: BLE001
        return None
    if seq and isinstance(seq[0], (list, tuple)):
        seq = list(seq[0])
    try:
        return [int(x) for x in seq]
    except (TypeError, ValueError):
        return None


def _tokenize_messages(tokenizer: Any, messages: List[Dict[str, str]]) -> List[int]:
    """Tokenizes chat messages using the tokenizer chat template when available.

    Args:
        tokenizer: Hugging Face-compatible tokenizer.
        messages: Rendered chat messages.

    Returns:
        List[int]: Token IDs representing the serialized chat prompt.
    """
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            out = apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            coerced = _coerce_chat_template_ids(out)
            if coerced is not None:
                return coerced
        except Exception:
            pass
    flat = "\n".join([f"{m.get('role','')}:\n{m.get('content','')}" for m in messages])
    return _tokenize_text(tokenizer, flat)


def _tokenize_messages_with_generation_prompt(
    tokenizer: Any, messages: List[Dict[str, str]]
) -> List[int]:
    """Tokenize chat messages with the assistant generation prompt appended."""

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            out = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            coerced = _coerce_chat_template_ids(out)
            if coerced is not None:
                return coerced
        except Exception:
            pass
        try:
            rendered = apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str):
                return _tokenize_text(tokenizer, rendered)
        except Exception:
            pass
    flat = "\n".join([f"{m.get('role','')}:\n{m.get('content','')}" for m in messages])
    return _tokenize_text(tokenizer, flat + "\nassistant:")


def _request_debug_flag(debug_payload: Optional[Mapping[str, Any]], key: str) -> bool:
    """Return whether one optional request-debug artifact is enabled."""

    return bool(debug_payload and bool(debug_payload.get(key)))


def _prompt_token_preview(tokenizer: Any, token_ids: Sequence[int], limit: int = 256) -> str:
    """Decode a short prompt-token preview for human inspection."""

    preview_ids = [int(token_id) for token_id in list(token_ids)[: max(0, int(limit))]]
    if not preview_ids:
        return ""
    try:
        return str(
            tokenizer.decode(
                preview_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except Exception:  # noqa: BLE001
        return ""


def _extract_snapshot_sha_from_path(path_text: str) -> str:
    """Best-effort extraction of a Hugging Face snapshot SHA from a path."""

    if not str(path_text or "").strip():
        return ""
    parts = Path(str(path_text)).parts
    for idx, part in enumerate(parts):
        if part == "snapshots" and idx + 1 < len(parts):
            return str(parts[idx + 1])
    tail = Path(str(path_text)).name
    if re.fullmatch(r"[0-9a-f]{40}", tail):
        return str(tail)
    return ""


def _candidate_hf_hub_roots() -> List[Path]:
    """Return candidate local Hugging Face cache roots for snapshot lookup."""

    roots: List[Path] = []
    seen: set[str] = set()
    raw_candidates = [
        str(os.environ.get("HUGGINGFACE_HUB_CACHE", "") or "").strip(),
        str(os.environ.get("HF_HUB_CACHE", "") or "").strip(),
        str(os.environ.get("HF_HOME", "") or "").strip(),
        "/hf/hub",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    ]
    for raw in raw_candidates:
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.name != "hub":
            candidate = candidate / "hub"
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        roots.append(candidate)
    return roots


def _resolve_local_hf_snapshot_path(repo_id: str, commit_hash: str) -> str:
    """Locate one cached local snapshot path for ``repo_id`` and ``commit_hash``."""

    repo = str(repo_id or "").strip()
    sha = str(commit_hash or "").strip()
    if not repo or not sha or Path(repo).exists() or "/" not in repo:
        return ""
    snapshot_rel = Path(f"models--{repo.replace('/', '--')}") / "snapshots" / sha
    for root in _candidate_hf_hub_roots():
        candidate = root / snapshot_rel
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    """Return a stable tokenizer vocab size when available."""

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return int(vocab_size)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
            if isinstance(vocab, Mapping):
                return int(len(vocab))
        except Exception:  # noqa: BLE001
            pass
    return 0


def _build_tokenizer_resolution_payload(
    *,
    tokenizer_id: str,
    tokenizer: Any,
    model_config: Any,
    chat_template_hash: str,
) -> Dict[str, Any]:
    """Build a stable tokenizer-resolution artifact for debug diffs."""

    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    resolved_name_or_path = str(
        getattr(tokenizer, "name_or_path", "") or init_kwargs.get("name_or_path") or ""
    )
    requested_id = str(tokenizer_id or "").strip()
    tokenizer_commit_hash = str(
        init_kwargs.get("_commit_hash")
        or getattr(tokenizer, "_commit_hash", "")
        or getattr(model_config, "_commit_hash", "")
        or ""
    ).strip()
    resolved_revision = str(init_kwargs.get("revision") or "").strip()
    local_snapshot_path = ""
    local_snapshot_sha = ""
    for candidate_text in (resolved_name_or_path, requested_id):
        if not str(candidate_text or "").strip():
            continue
        candidate_path = Path(str(candidate_text)).expanduser()
        if candidate_path.exists():
            local_snapshot_path = str(candidate_path.resolve())
            local_snapshot_sha = _extract_snapshot_sha_from_path(local_snapshot_path)
            break
    if not local_snapshot_path:
        local_snapshot_path = _resolve_local_hf_snapshot_path(
            requested_id or resolved_name_or_path,
            tokenizer_commit_hash,
        )
        local_snapshot_sha = _extract_snapshot_sha_from_path(local_snapshot_path)
    if not local_snapshot_sha:
        local_snapshot_sha = tokenizer_commit_hash
    return {
        "requested_tokenizer_id": requested_id,
        "resolved_tokenizer_name_or_path": resolved_name_or_path,
        "resolved_tokenizer_revision": resolved_revision,
        "resolved_tokenizer_commit_hash": tokenizer_commit_hash,
        "resolved_model_name_or_path": str(
            getattr(model_config, "_name_or_path", "") or ""
        ),
        "resolved_model_commit_hash": str(
            getattr(model_config, "_commit_hash", "") or ""
        ),
        "local_snapshot_path": str(local_snapshot_path or ""),
        "local_snapshot_sha": str(local_snapshot_sha or ""),
        "chat_template_hash": str(chat_template_hash or ""),
        "tokenizer_class": str(tokenizer.__class__.__name__),
        "vocab_size": int(_tokenizer_vocab_size(tokenizer)),
        "model_vocab_size": int(getattr(model_config, "vocab_size", 0) or 0),
    }


def _normalize_prefix_capture(prefix_capture: Any) -> Dict[str, Any]:
    """Normalizes optional prefix-capture metadata into a stable dictionary.

    Args:
        prefix_capture: Raw prefix-capture payload from prompt-shape metadata
            or a saved manifest.

    Returns:
        Dict[str, Any]: Normalized metadata when valid, otherwise an empty
        dictionary.
    """
    return _prefix_session_lib.normalize_prefix_capture(prefix_capture)


def _is_repo_grounded_shape(prompt_shape_id: str) -> bool:
    """Return whether the prompt shape uses repo-grounded task wording."""

    return str(prompt_shape_id or "") in {
        "repo_grounded_en_v1",
        "repo_grounded_en_v2",
    }


def _default_suffix_query(prompt_shape_id: str, max_new_tokens: int) -> str:
    shape = str(prompt_shape_id or "benchmark_v1")
    max_tok = max(1, int(max_new_tokens))
    if _is_repo_grounded_shape(shape):
        return (
            "Write a technical report grounded in CONTEXT.\n\n"
            "Requirements:\n"
            f"1. Minimum {max_tok} tokens.\n"
            "2. Cite at least 8 concrete file paths that appear in CONTEXT.\n"
            "3. Cover architecture, build/test workflow, dependencies, bottlenecks, and actionable next steps.\n"
            "4. No exam/problem-solution format.\n"
            "5. No meta commentary about instructions."
        )
    return (
        "[BENCHMARK TASK]\n"
        f"Write a continuous technical response with at least {max_tok} tokens. "
        "Do not end early."
    )


_PROMPT_FILE_MARKER_RE = re.compile(r"^\[FILE (.+?)\]$", flags=re.MULTILINE)


def _build_prompt_context_summary(prompt_text: str) -> str:
    """Summarize prompt coverage from repo/file markers for long-context shapes."""

    text = str(prompt_text or "")
    paths_raw = _PROMPT_FILE_MARKER_RE.findall(text)
    seen: set[str] = set()
    paths: List[str] = []
    for raw in paths_raw:
        path = str(raw or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    if not paths:
        return "Context coverage summary unavailable."

    repo_names = []
    seen_repos: set[str] = set()
    for path in paths:
        repo = path.split("/", 1)[0] if "/" in path else path
        if repo and repo not in seen_repos:
            seen_repos.add(repo)
            repo_names.append(repo)

    head = paths[:4]
    mid_start = max(0, (len(paths) // 2) - 2)
    middle = paths[mid_start : mid_start + 4]
    tail = paths[-4:]
    representative: List[str] = []
    seen_rep: set[str] = set()
    for block in (head, middle, tail):
        for path in block:
            if path not in seen_rep:
                seen_rep.add(path)
                representative.append(path)

    lines = [
        f"Repositories covered: {', '.join(repo_names)}.",
        f"Approximate file count in CONTEXT: {len(paths)}.",
        "Representative files across the beginning, middle, and end of CONTEXT:",
    ]
    lines.extend(f"- {path}" for path in representative[:12])
    lines.append(
        "Use this coverage summary to reason about the repository as a whole, not just the final files in CONTEXT."
    )
    return "\n".join(lines)


def _build_prompt_shape_render_context(
    *,
    prompt_text: str,
    system_prompt: str,
    max_new_tokens: int,
    prompt_tokens: int,
    model_id: str,
    prompt_text_summary_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the placeholder map used to render prompt-shape messages."""

    summary_source = (
        str(prompt_text_summary_source)
        if prompt_text_summary_source is not None
        else str(prompt_text)
    )
    return {
        "prompt_text": str(prompt_text),
        "prompt_context_summary": _build_prompt_context_summary(summary_source),
        "system_prompt": str(system_prompt),
        "max_new_tokens": int(max_new_tokens),
        "prompt_tokens": int(prompt_tokens),
        "model_id": str(model_id),
    }


def _build_prefix_messages_for_session(
    *,
    prompt_shape_id: str,
    system_prompt: str,
    prompt_text: str,
) -> List[Dict[str, str]]:
    """Builds legacy prefix-session messages for prompt shapes without splitting.

    Args:
        prompt_shape_id: Prompt-shape identifier.
        system_prompt: System instruction text.
        prompt_text: Context payload text.

    Returns:
        List[Dict[str, str]]: Prefix messages suitable for message-based
        capture/query flows.
    """
    shape = str(prompt_shape_id or "benchmark_v1")
    if _is_repo_grounded_shape(shape):
        user_content = f"<CONTEXT>\n{prompt_text}\n</CONTEXT>"
    else:
        user_content = str(prompt_text)
    return [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user", "content": user_content},
    ]


def _truncate_messages_for_prefix_capture(
    *,
    messages: List[Dict[str, str]],
    prefix_capture: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Truncates rendered messages at the configured prefix/suffix boundary.

    Args:
        messages: Fully rendered prompt-shape messages.
        prefix_capture: Normalized prefix-capture metadata describing where the
            reusable prefix ends.

    Returns:
        List[Dict[str, str]]: Messages up to and including the split point.

    Raises:
        RuntimeError: If the split metadata is invalid for the rendered
            messages.
    """
    if not messages:
        raise RuntimeError("Cannot truncate empty rendered message list for prefix capture")
    message_index = int(prefix_capture.get("message_index") or 0)
    if message_index < 0 or message_index >= len(messages):
        raise RuntimeError(
            f"prefix_capture.message_index out of range: {message_index} for {len(messages)} messages"
        )
    split_after = str(prefix_capture.get("split_after") or "")
    if not split_after:
        raise RuntimeError("prefix_capture.split_after is empty")
    out = [dict(msg) for msg in messages[: message_index + 1]]
    target = dict(out[message_index])
    content = str(target.get("content") or "")
    split_idx = content.find(split_after)
    if split_idx < 0:
        raise RuntimeError(
            f"prefix_capture.split_after not found in rendered messages[{message_index}]"
        )
    target["content"] = content[: split_idx + len(split_after)]
    out[message_index] = target
    return out


def _build_same_message_prefix_token_ids(
    *,
    tokenizer: Any,
    prefix_messages: List[Dict[str, str]],
    message_index: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Build exact open-message prefix tokens for same-message continuation.

    The saved prefix tokens must reflect the chat template *before* the final
    user message is closed, otherwise they will not be a literal token-prefix
    of the later full query prompt. This helper appends a continuation marker
    to the captured truncated message stub, serializes that open-message form,
    and then splits the serialized text around the marker.

    Args:
        tokenizer: Hugging Face-compatible tokenizer with chat-template
            support.
        prefix_messages: Rendered messages truncated at the same-message prefix
            capture boundary.
        message_index: Optional message index to extend. When omitted, the last
            message is treated as the continued same-message stub.

    Returns:
        Tuple[List[int], List[int]]: ``(prefix_token_ids,
        continuation_tail_token_ids)``.

    Raises:
        RuntimeError: If the tokenizer cannot serialize the chat template or
        if the split markers cannot be found in the serialized prompt.
    """
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError("Tokenizer does not support same-message prefix capture without apply_chat_template")

    continuation_marker = "<<LCS_SUFFIX_SLOT_15CC4E12>>"
    if not prefix_messages:
        raise RuntimeError("same-message prefix capture requires non-empty prefix_messages")
    target_index = (
        len(prefix_messages) - 1 if message_index is None else int(message_index)
    )
    if target_index < 0 or target_index >= len(prefix_messages):
        raise RuntimeError(
            f"same-message message_index out of range: {target_index} for {len(prefix_messages)} messages"
        )
    continuation_messages = [dict(msg) for msg in prefix_messages]
    continuation_messages[target_index] = {
        "role": str(continuation_messages[target_index].get("role") or ""),
        "content": (
            str(continuation_messages[target_index].get("content") or "")
            + continuation_marker
        ),
    }
    try:
        rendered_text = apply_chat_template(
            continuation_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Tokenizer apply_chat_template did not return rendered text for same-message prefix capture"
        ) from exc
    if not isinstance(rendered_text, str):
        raise RuntimeError(
            "Tokenizer apply_chat_template returned non-text output for same-message prefix capture"
        )

    continuation_char_index = rendered_text.find(
        continuation_marker,
    )
    if continuation_char_index < 0:
        raise RuntimeError(
            "Failed to locate continuation marker in same-message prefix prompt text"
        )
    continuation_after_index = continuation_char_index + len(continuation_marker)
    prefix_token_ids = _tokenize_text(
        tokenizer,
        rendered_text[:continuation_char_index],
    )
    continuation_tail_text = rendered_text[continuation_after_index:]
    continuation_tail_token_ids = _tokenize_text(tokenizer, continuation_tail_text)
    return [int(x) for x in prefix_token_ids], [int(x) for x in continuation_tail_token_ids]


def _build_prefix_session_payload(
    *,
    tokenizer: Any,
    prompt_ids: List[int],
    prompt_text: str,
    prompt_shape_spec: Mapping[str, Any],
    prompt_shape_id: str,
    system_prompt: str,
    max_new_tokens: int,
    model_id: str,
) -> Dict[str, Any]:
    """Builds the persisted prefix-session payload for capture or reuse.

    Args:
        tokenizer: Hugging Face-compatible tokenizer.
        prompt_ids: Token IDs for the selected context prefix payload.
        prompt_text: Decoded text for ``prompt_ids``.
        prompt_shape_spec: Prompt-shape specification to render.
        prompt_shape_id: Prompt-shape identifier.
        system_prompt: System instruction text.
        max_new_tokens: Max completion tokens used while rendering template
            placeholders.
        model_id: Model identifier exposed to the prompt-shape renderer.

    Returns:
        Dict[str, Any]: Prefix-session payload containing rendered prefix
        messages, serialized prefix tokens, normalized prefix-capture metadata,
        and any continuation-tail token IDs needed for same-message query
        continuation.
    """
    prefix_capture = _normalize_prefix_capture(prompt_shape_spec.get("prefix_capture"))
    if str(prefix_capture.get("continuation_mode") or "") == "same_message":
        rendered_messages = render_messages(
            prompt_shape_spec,
            _build_prompt_shape_render_context(
                prompt_text=str(prompt_text),
                system_prompt=str(system_prompt),
                max_new_tokens=int(max_new_tokens),
                prompt_tokens=int(len(prompt_ids)),
                model_id=str(model_id),
            ),
        )
        prefix_messages = _truncate_messages_for_prefix_capture(
            messages=rendered_messages,
            prefix_capture=prefix_capture,
        )
        prefix_token_ids, continuation_tail_token_ids = _build_same_message_prefix_token_ids(
            tokenizer=tokenizer,
            prefix_messages=prefix_messages,
            message_index=int(prefix_capture.get("message_index") or len(prefix_messages) - 1),
        )
        return {
            "prefix_messages": prefix_messages,
            "prefix_token_ids": prefix_token_ids,
            "prefix_capture": prefix_capture,
            "continuation_tail_token_ids": continuation_tail_token_ids,
        }

    prefix_messages = _build_prefix_messages_for_session(
        prompt_shape_id=prompt_shape_id,
        system_prompt=system_prompt,
        prompt_text=prompt_text,
    )
    return {
        "prefix_messages": prefix_messages,
        "prefix_token_ids": _tokenize_messages(tokenizer, prefix_messages),
        "prefix_capture": prefix_capture,
        "continuation_tail_token_ids": [],
    }


def _ceil_div_int(num: int, den: int) -> int:
    """Return ``ceil(num / den)`` for positive integers, else ``0``."""

    numerator = int(num)
    denominator = int(den)
    if numerator <= 0 or denominator <= 0:
        return 0
    return (numerator + denominator - 1) // denominator


def _build_prefix_session_attention_metadata(run_meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Build explicit checkpoint attention metadata for new saved sessions."""

    vllm_meta = dict(run_meta.get("vllm") or {})
    block_size_tokens = max(0, int(vllm_meta.get("block_size") or 0))
    sliding_window_tokens = max(0, int(vllm_meta.get("mla_sliding_window_tokens") or 0))
    sink_keep_tokens = max(0, int(vllm_meta.get("mla_sink_keep_tokens") or 0))
    if sliding_window_tokens > 0 or sink_keep_tokens > 0:
        return {
            "schema_version": "0.2",
            "attention_mode": "streaming_mla",
            "attention_config": {
                "type": "streaming_mla",
                "sliding_window_tokens": int(sliding_window_tokens),
                "sink_keep_tokens": int(sink_keep_tokens),
                "effective_live_block_count": 0,
                "effective_sink_block_count": int(
                    _ceil_div_int(sink_keep_tokens, block_size_tokens)
                ),
                "block_size_tokens": int(block_size_tokens),
                "dense_mla_group_id": -1,
            },
        }
    return {
        "schema_version": "0.2",
        "attention_mode": "full_attention",
        "attention_config": {"type": "full_attention"},
    }


def _annotate_streaming_attention_config_from_manifest(
    manifest: Dict[str, Any],
    *,
    default_block_size_tokens: int = 0,
) -> Dict[str, Any]:
    """Refresh derived StreamingLLM checkpoint metadata from saved APC layout."""

    attention_mode = str(manifest.get("attention_mode") or "").strip().lower()
    if attention_mode != "streaming_mla":
        return manifest
    attention_config = dict(manifest.get("attention_config") or {})
    layout = dict(manifest.get("apc_cache_layout") or {})
    manager_rows = [
        dict(row) for row in list(layout.get("manager_rows") or []) if isinstance(row, Mapping)
    ]
    dense_row: Dict[str, Any] = {}
    if manager_rows:
        dense_row = max(
            manager_rows,
            key=lambda row: len(list(row.get("cached_block_ids") or [])),
        )
    global_rows = [int(value) for value in list(layout.get("global_target_rows") or []) if int(value) > 0]
    live_block_count = len(global_rows)
    prefix_state_capture = dict(manifest.get("prefix_state_capture") or {})
    if live_block_count <= 0:
        for rank_row in list(prefix_state_capture.get("ranks") or []):
            rank_manifest_path = Path(str((rank_row or {}).get("rank_manifest_path") or ""))
            if not rank_manifest_path.exists():
                continue
            try:
                rank_manifest = json.loads(rank_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rank_global_rows = [
                int(value)
                for value in list(rank_manifest.get("global_target_rows") or [])
                if int(value) > 0
            ]
            if rank_global_rows:
                global_rows = list(rank_global_rows)
                live_block_count = len(global_rows)
            if not dense_row:
                dense_row = {
                    "kv_cache_group_id": int(rank_manifest.get("global_target_group_id") or -1),
                    "block_size": int(
                        max(
                            [
                                int(row.get("block_size") or 0)
                                for row in list(rank_manifest.get("manager_rows") or [])
                                if isinstance(row, Mapping)
                            ]
                            or [0]
                        )
                    ),
                }
            if live_block_count > 0:
                break
    if live_block_count <= 0 and dense_row:
        live_block_count = len(
            [int(value) for value in list(dense_row.get("cached_block_ids") or []) if int(value) > 0]
        )
    block_size_tokens = max(
        0,
        int(
            attention_config.get("block_size_tokens")
            or manifest.get("block_size")
            or (dense_row.get("block_size") if dense_row else 0)
            or max(
                [
                    int((row or {}).get("block_size") or 0)
                    for row in list(manager_rows or [])
                    if isinstance(row, Mapping)
                ]
                or [0]
            )
            or int(default_block_size_tokens or 0)
            or 0
        ),
    )
    sink_keep_tokens = max(0, int(attention_config.get("sink_keep_tokens") or 0))
    try:
        layout_global_target_group_id = int(layout.get("global_target_group_id", -1))
    except Exception:
        layout_global_target_group_id = -1
    attention_config.update(
        {
            "type": "streaming_mla",
            "effective_live_block_count": int(live_block_count),
            "effective_sink_block_count": int(
                attention_config.get("effective_sink_block_count")
                or _ceil_div_int(sink_keep_tokens, block_size_tokens)
            ),
            "block_size_tokens": int(block_size_tokens),
            "dense_mla_group_id": int(
                dense_row.get("kv_cache_group_id")
                if dense_row
                else layout_global_target_group_id
                if int(layout_global_target_group_id) >= 0
                else attention_config.get("dense_mla_group_id", -1)
            ),
        }
    )
    manifest["attention_config"] = attention_config
    return manifest


def _streaming_apc_layout_from_rpc_results(
    rpc_results: Sequence[Mapping[str, Any]],
    *,
    attention_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    dense_group_id = int((attention_config or {}).get("dense_mla_group_id") or -1)
    effective_live_block_count = max(
        0, int((attention_config or {}).get("effective_live_block_count") or 0)
    )
    effective_sink_block_count = max(
        0, int((attention_config or {}).get("effective_sink_block_count") or 0)
    )
    for row in rpc_results:
        if not isinstance(row, Mapping):
            continue
        global_target_rows = [
            int(value)
            for value in list(row.get("global_target_rows") or [])
            if int(value) > 0
        ]
        linear_target_last_rows = {
            str(key): int(value)
            for key, value in dict(row.get("linear_target_last_rows") or {}).items()
            if int(value) > 0
        }
        if not global_target_rows and not linear_target_last_rows:
            continue
        manager_rows = []
        for manager_row in list(row.get("manager_rows") or []):
            if not isinstance(manager_row, Mapping):
                continue
            cached_block_offsets = [
                int(value)
                for value in list((manager_row or {}).get("cached_block_offsets") or [])
                if int(value) >= 0
            ]
            cached_block_ids = [
                int(value)
                for value in list((manager_row or {}).get("cached_block_ids") or [])
                if int(value) > 0
            ]
            positive_row_ids = [
                int(value)
                for value in list((manager_row or {}).get("positive_row_ids") or [])
                if int(value) > 0
            ]
            # Older debug capture RPCs only returned ``positive_row_ids`` for
            # diagnostics, which loses the physical block offsets needed to
            # reinstall an exact compact streaming layout. Treat those rows as
            # incomplete so the richer scheduler payload can fill them in.
            if (not cached_block_offsets or not cached_block_ids) and positive_row_ids:
                group_index = int((manager_row or {}).get("group_index") or 0)
                num_full_blocks = int((manager_row or {}).get("num_full_blocks") or 0)
                if group_index == dense_group_id and effective_live_block_count > 0:
                    compact_prefix_count = min(
                        len(positive_row_ids), int(effective_live_block_count)
                    )
                    tail_count = max(0, len(positive_row_ids) - compact_prefix_count)
                    sink_count = min(
                        int(effective_sink_block_count), int(compact_prefix_count)
                    )
                    window_count = max(0, int(compact_prefix_count) - int(sink_count))
                    cached_block_offsets = list(range(int(sink_count)))
                    if window_count > 0:
                        cached_block_offsets.extend(
                            range(
                                max(int(sink_count), int(num_full_blocks) - int(window_count)),
                                int(num_full_blocks),
                            )
                        )
                    cached_block_offsets.extend(
                        range(int(num_full_blocks), int(num_full_blocks) + int(tail_count))
                    )
                    cached_block_ids = list(positive_row_ids)
                elif effective_sink_block_count > 0:
                    compact_prefix_count = min(
                        len(positive_row_ids), int(effective_sink_block_count)
                    )
                    tail_count = max(0, len(positive_row_ids) - compact_prefix_count)
                    cached_block_offsets = list(
                        range(
                            max(0, int(num_full_blocks) - int(compact_prefix_count)),
                            int(num_full_blocks),
                        )
                    )
                    cached_block_offsets.extend(
                        range(int(num_full_blocks), int(num_full_blocks) + int(tail_count))
                    )
                    cached_block_ids = list(positive_row_ids)
            if not cached_block_offsets or not cached_block_ids:
                continue
            manager_rows.append(
                {
                    "group_index": int((manager_row or {}).get("group_index") or 0),
                    "kv_cache_group_id": int(
                        (manager_row or {}).get("kv_cache_group_id") or 0
                    ),
                    "block_size": int((manager_row or {}).get("block_size") or 0),
                    "num_full_blocks": int(
                        (manager_row or {}).get("num_full_blocks") or 0
                    ),
                    "cached_block_offsets": cached_block_offsets,
                    "cached_block_ids": cached_block_ids,
                }
            )
        return {
            "full_block_hash_count": int(row.get("full_block_hash_count") or 0),
            "linear_target_last_rows": linear_target_last_rows,
            "global_target_rows": global_target_rows,
            "source_group_rows": {},
            "manager_rows": manager_rows,
            "global_target_group_id": int(row.get("global_target_group_id") or -1),
        }
    return {}


def _session_dir(prefix_session_dir: Path, prefix_session_id: str) -> Path:
    return prefix_session_dir / str(prefix_session_id)


def _write_prefix_session(
    *,
    prefix_session_dir: Path,
    prefix_session_id: str,
    prefix_messages: List[Dict[str, str]],
    prefix_token_ids: List[int],
    continuation_tail_token_ids: List[int],
    run_meta: Mapping[str, Any],
    prompt_shape_meta: Mapping[str, Any],
    strict_env_hash: str,
    strict_env_inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Writes prefix-session artifacts and manifest metadata to disk.

    Args:
        prefix_session_dir: Root directory containing saved prefix sessions.
        prefix_session_id: Stable session identifier.
        prefix_messages: Human-readable rendered prefix messages.
        prefix_token_ids: Serialized reusable prefix token IDs.
        continuation_tail_token_ids: Token IDs that must remain after any
            same-message suffix continuation.
        run_meta: Run metadata used to populate manifest compatibility fields.
        prompt_shape_meta: Prompt-shape metadata used for compatibility checks.
        strict_env_hash: Hash of strict environment settings.

    Returns:
        Dict[str, Any]: Saved session directory and manifest payload.
    """
    session_dir = _session_dir(prefix_session_dir, prefix_session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    token_bin = session_dir / "prefix_token_ids.bin"
    token_hash = _write_token_ids_bin(token_bin, prefix_token_ids)
    continuation_tail_bin = session_dir / "continuation_tail_token_ids.bin"
    continuation_tail_hash = _write_token_ids_bin(
        continuation_tail_bin,
        continuation_tail_token_ids,
    )
    (session_dir / "prefix_token_sha256.txt").write_text(token_hash + "\n", encoding="utf-8")
    (session_dir / "prefix_messages.json").write_text(
        json.dumps(prefix_messages, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    vllm_meta = dict(run_meta.get("vllm") or {})
    fp8_contract = _build_fp8_checkpoint_contract(str(vllm_meta.get("kv_cache_dtype") or ""))
    prefix_capture = _normalize_prefix_capture(prompt_shape_meta.get("prefix_capture"))
    attention_metadata = _build_prefix_session_attention_metadata(run_meta)
    manifest: Dict[str, Any] = {
        "format_version": 1,
        "schema_version": str(attention_metadata.get("schema_version") or "0.2"),
        "attention_mode": str(attention_metadata.get("attention_mode") or "full_attention"),
        "attention_config": dict(attention_metadata.get("attention_config") or {}),
        "session_id": str(prefix_session_id),
        "created_at_utc": _now_iso(),
        "model_id": str(run_meta.get("model_id") or ""),
        "tokenizer_id": str(run_meta.get("tokenizer_id") or ""),
        "chat_template_hash": str(run_meta.get("chat_template_hash") or ""),
        "prompt_shape_id": str(prompt_shape_meta.get("id") or ""),
        "prompt_shape_template_sha256": str(prompt_shape_meta.get("template_sha256") or ""),
        "prefix_capture": dict(prefix_capture),
        "prefix_token_count": int(len(prefix_token_ids)),
        "prefix_token_sha256": str(token_hash),
        "continuation_tail_token_count": int(len(continuation_tail_token_ids)),
        "continuation_tail_sha256": str(continuation_tail_hash),
        "capture_topology": {
            "tp": int(vllm_meta.get("tensor_parallel_size") or 0),
            "dp": int(vllm_meta.get("data_parallel_size") or 0),
            "pp": int(vllm_meta.get("pipeline_parallel_size") or 0),
        },
        "kv_cache_dtype": str(vllm_meta.get("kv_cache_dtype") or ""),
        "attention_backend": str(vllm_meta.get("attention_backend") or ""),
        "block_size": int(vllm_meta.get("block_size") or 0),
        "max_model_len": int(vllm_meta.get("max_model_len") or 0),
        "layer_map": [],
        "tensor_files": [
            {
                "path": "prefix_token_ids.bin",
                "shape": [int(len(prefix_token_ids))],
                "dtype": "uint32",
                "checksum_sha256": str(token_hash),
            },
            {
                "path": "continuation_tail_token_ids.bin",
                "shape": [int(len(continuation_tail_token_ids))],
                "dtype": "uint32",
                "checksum_sha256": str(continuation_tail_hash),
                "kind": "continuation_tail_token_ids",
            },
        ],
        "fp8_scales": [],
        "strict_env_hash": str(strict_env_hash),
        "strict_env_inputs": dict(strict_env_inputs or {}),
    }
    manifest.update(fp8_contract)
    (session_dir / "prefix_session_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"session_dir": str(session_dir), "manifest": manifest}


def _load_prefix_session(
    *,
    prefix_session_dir: Path,
    prefix_session_id: str,
) -> Dict[str, Any]:
    """Loads a previously saved prefix session from disk.

    Args:
        prefix_session_dir: Root directory containing saved prefix sessions.
        prefix_session_id: Session identifier to load.

    Returns:
        Dict[str, Any]: Loaded manifest, prefix messages, prefix token IDs, and
        any continuation-tail token IDs.

    Raises:
        RuntimeError: If required session artifacts are missing or token hashes
            do not match the manifest.
    """
    return _prefix_session_lib.load_prefix_session(
        prefix_session_dir=prefix_session_dir,
        prefix_session_id=prefix_session_id,
    )


def _validate_prefix_session_compat(
    *,
    manifest: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    prompt_shape_meta: Mapping[str, Any],
    strict_env_hash: str,
    strict: bool,
) -> Tuple[bool, List[str]]:
    """Checks whether a saved prefix session is reusable for the current run.

    Args:
        manifest: Saved prefix-session manifest.
        run_meta: Current run metadata.
        prompt_shape_meta: Current prompt-shape metadata.
        strict_env_hash: Hash of strict environment settings for this run.
        strict: Whether to enforce strict environment matching.

    Returns:
        Tuple[bool, List[str]]: ``(is_compatible, mismatch_messages)``.
    """
    return _prefix_session_lib.validate_prefix_session_compat(
        manifest=manifest,
        run_meta=run_meta,
        prompt_shape_meta=prompt_shape_meta,
        strict_env_hash=strict_env_hash,
        strict=strict,
    )


def _run_json_cmd(argv: List[str], timeout_sec: int = 10) -> Optional[Dict[str, Any]]:
    try:
        proc = subprocess.run(  # noqa: S603
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except Exception:
        return None
    raw = str(proc.stdout or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _http_json_post(url: str, payload: Mapping[str, Any], timeout_sec: int) -> Dict[str, Any]:
    req = url_request.Request(
        str(url),
        method="POST",
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with url_request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
        body = resp.read()
    out = json.loads(body.decode("utf-8"))
    if not isinstance(out, dict):
        raise RuntimeError(f"Invalid JSON response type from {url}: {type(out).__name__}")
    return out


def _debug_prefix_cache_state(
    *,
    base_url: str,
    prompt_token_ids: Sequence[int],
    timeout_sec: int,
    max_block_rows: int = 64,
) -> Dict[str, Any]:
    """Fetch the dev-only APC cache snapshot for a specific prompt.

    Args:
        base_url: Base URL of the running vLLM server.
        prompt_token_ids: Prompt token IDs whose APC mapping should be probed.
        timeout_sec: HTTP timeout in seconds.
        max_block_rows: Maximum number of block rows to include in the response.

    Returns:
        Dict[str, Any]: Parsed JSON response from ``/debug_prefix_cache_state``.
    """

    return _http_json_post(
        f"{base_url.rstrip('/')}/debug_prefix_cache_state",
        {
            "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
            "max_block_rows": int(max_block_rows),
        },
        timeout_sec=max(1, int(timeout_sec)),
    )


def _extract_apc_cache_row_targets(debug_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the APC-backed block rows that should be persisted to disk.

    The native APC cache stores only the rows that are actually reusable on a
    future prompt replay. For the current Kimi/vLLM topology, that means:

    - sparse/KDA groups where aligned boundary states are cached
    - one dense MLA group where every full-block row is cached

    Args:
        debug_state: JSON payload returned by ``/debug_prefix_cache_state``.

    Returns:
        Dict[str, Any]: Normalized APC cache metadata, including
        ``linear_target_last_rows``, ``global_target_rows``,
        ``source_group_rows``, and compact ``manager_rows`` suitable for
        saving the exact native APC-backed rows.
    """

    def _canonical_sparse_cached_block_id(
        *,
        block_offsets: Sequence[int],
        block_ids: Sequence[int],
        num_full_blocks: int,
    ) -> int:
        """Return the canonical sparse APC row for one streaming group.

        For Kimi streaming capture, active requests often expose two positive
        sparse rows:
        - the aligned full-prefix boundary row that should be persisted
        - a transient continuation row allocated for the in-flight request

        The persisted checkpoint must keep the row at the aligned boundary,
        which is the highest cached block offset that still lies inside the
        dense full-block prefix. Falling back to the earliest positive row keeps
        behavior sensible on smaller or partially described debug payloads.

        Args:
            block_offsets: Cached block offsets reported by APC debug probes.
            block_ids: Positive cached block IDs aligned with ``block_offsets``.
            num_full_blocks: Number of full prompt blocks in the reusable prefix.

        Returns:
            int: Canonical sparse APC row ID, or ``0`` when no positive row is
            available.
        """

        pairs = [
            (int(offset), int(block_id))
            for offset, block_id in zip(block_offsets, block_ids, strict=False)
            if int(block_id) > 0
        ]
        if not pairs:
            return 0
        aligned_pairs = [
            (offset, block_id) for offset, block_id in pairs if offset < int(num_full_blocks)
        ]
        if aligned_pairs:
            return int(max(aligned_pairs, key=lambda item: item[0])[1])
        return int(pairs[0][1])

    prompt_probe = dict(debug_state.get("prompt_probe") or {})
    manager_rows = list(prompt_probe.get("manager_rows") or [])
    linear_manifest_keys = ["-1", "1", "2"]
    linear_key_index = 0
    linear_target_last_rows: Dict[str, int] = {}
    global_target_rows: List[int] = []
    source_group_rows: Dict[str, List[int]] = {}
    normalized_rows: List[Dict[str, Any]] = []

    for row in manager_rows:
        match_rows = list(row.get("match_rows") or [])
        normalized_matches: List[Dict[str, Any]] = []
        cached_block_offsets: List[int] = []
        cached_block_ids: List[int] = []
        for match in match_rows:
            matched_block_ids = [
                int(block_id)
                for block_id in list(match.get("matched_block_ids") or [])
                if int(block_id) > 0
            ]
            block_offset = int(match.get("block_offset") or 0)
            normalized_matches.append(
                {
                    "block_offset": block_offset,
                    "matched_block_ids": matched_block_ids,
                    "lookup_key_hex": str(match.get("lookup_key_hex") or ""),
                    "block_hash_hex": str(match.get("block_hash_hex") or ""),
                }
            )
            if len(matched_block_ids) == 1:
                cached_block_offsets.append(block_offset)
                cached_block_ids.append(int(matched_block_ids[0]))

        group_index = int(row.get("group_index") or 0)
        kv_cache_group_id = int(row.get("kv_cache_group_id") or 0)
        block_size = int(row.get("block_size") or 0)
        num_full_blocks = int(row.get("num_full_blocks") or 0)
        source_group_rows[str(group_index)] = [int(value) for value in cached_block_ids]

        normalized_rows.append(
            {
                "group_index": group_index,
                "kv_cache_group_id": kv_cache_group_id,
                "block_size": block_size,
                "num_full_blocks": num_full_blocks,
                "cached_block_offsets": cached_block_offsets,
                "cached_block_ids": cached_block_ids,
                "match_rows": normalized_matches,
            }
        )

        if not cached_block_ids:
            continue

        dense_full_prefix = (
            num_full_blocks > 0
            and len(cached_block_offsets) == int(num_full_blocks)
            and cached_block_offsets == list(range(int(num_full_blocks)))
        )

        if dense_full_prefix and not global_target_rows:
            global_target_rows = [int(ids) for ids in cached_block_ids]
        elif linear_key_index < len(linear_manifest_keys):
            canonical_sparse_row = _canonical_sparse_cached_block_id(
                block_offsets=cached_block_offsets,
                block_ids=cached_block_ids,
                num_full_blocks=num_full_blocks,
            )
            if canonical_sparse_row > 0:
                linear_target_last_rows[linear_manifest_keys[linear_key_index]] = int(
                    canonical_sparse_row
                )
                linear_key_index += 1

    return {
        "prompt_probe": prompt_probe,
        "manager_rows": normalized_rows,
        "linear_target_last_rows": linear_target_last_rows,
        "global_target_rows": global_target_rows,
        "source_group_rows": source_group_rows,
        "full_block_hash_count": int(prompt_probe.get("full_block_hash_count") or 0),
    }


def _call_collective_rpc(
    *,
    base_url: str,
    method: str,
    args: List[Any],
    kwargs: Optional[Mapping[str, Any]] = None,
    timeout_sec: int,
) -> List[Dict[str, Any]]:
    """Call the dev-mode collective RPC API and normalize per-rank rows."""
    deadline = time.monotonic() + min(max(5.0, float(timeout_sec)), 30.0)
    last_404: Optional[url_error.HTTPError] = None
    while True:
        try:
            return _prefix_session_lib.call_collective_rpc(
                base_url=base_url,
                method=method,
                args=args,
                kwargs=kwargs,
                timeout_sec=timeout_sec,
            )
        except url_error.HTTPError as exc:
            if int(getattr(exc, "code", 0) or 0) != 404:
                raise
            last_404 = exc
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
    if last_404 is not None:
        raise last_404


def _evaluate_prefix_state_dump_calibration(
    rpc_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in rpc_results:
        free_b = int(row.get("free_bytes") or 0)
        req_b = int(row.get("required_free_bytes") or 0)
        rows.append(
            {
                "rank": int(row.get("rank") or row.get("rank_index") or 0),
                "tensor_count": int(row.get("tensor_count") or 0),
                "total_state_bytes": int(row.get("total_state_bytes") or 0),
                "max_tensor_bytes": int(row.get("max_tensor_bytes") or 0),
                "max_chunk_workspace_bytes": int(row.get("max_chunk_workspace_bytes") or 0),
                "free_bytes": int(free_b),
                "required_free_bytes": int(req_b),
                "pass": bool(free_b >= req_b),
            }
        )
    fail_rows = [r for r in rows if not bool(r.get("pass"))]
    return {
        "rows": rows,
        "summary": {
            "rank_count": int(len(rows)),
            "fail_count": int(len(fail_rows)),
            "all_pass": bool(len(fail_rows) == 0),
            "min_free_bytes": int(min((r["free_bytes"] for r in rows), default=0)),
            "max_required_free_bytes": int(
                max((r["required_free_bytes"] for r in rows), default=0)
            ),
        },
    }


def _require_collective_rpc_ok(
    *,
    method: str,
    results: List[Dict[str, Any]],
) -> None:
    """Raise if any rank reported a failed collective RPC."""

    _prefix_session_lib.require_collective_rpc_ok(method=method, results=results)


def _env_int(name: str, default: int) -> int:
    """Parse one integer environment override."""

    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:  # noqa: BLE001
        return int(default)


def _query_live_layout_enabled() -> bool:
    """Return whether to capture non-invasive live request layout metadata."""

    return bool(_env_flag("HFLC_QUERY_LIVE_LAYOUT_ENABLE"))


def _query_row_hash_config() -> Dict[str, Any]:
    """Return live prefix-row hash capture settings from environment overrides."""

    return {
        "enabled": bool(_env_flag("HFLC_QUERY_ROW_HASH_ENABLE")),
        "row_base_shift": str(
            os.environ.get("HFLC_QUERY_ROW_HASH_BASE_SHIFT", "auto") or "auto"
        ),
    }


def _fetch_query_prefix_row_hashes(
    *,
    base_url: str,
    timeout_sec: int,
    session_dir: str,
    session_id: str,
    linear_target_last_rows: Mapping[str, Any],
    global_target_rows: List[int],
    row_base_shift: str,
) -> Dict[str, Any]:
    """Fetch live prefix-row hashes for the current session's target rows."""

    session_root = Path(str(session_dir or "")).expanduser().resolve()
    session_id_str = str(session_id or "").strip()
    if (
        session_id_str
        and session_root.name == session_id_str
        and (session_root / "prefix_session_manifest.json").exists()
    ):
        # `debug_prefix_state_row_hashes` expects the parent directory that
        # contains `<session_id>/`, not the full session path itself.
        session_root = session_root.parent

    results = _call_collective_rpc(
        base_url=base_url,
        method="debug_prefix_state_row_hashes",
        args=[],
        kwargs={
            "session_dir": str(session_root),
            "session_id": str(session_id_str),
            "linear_target_last_rows_json": json.dumps(
                {str(key): int(value) for key, value in dict(linear_target_last_rows).items()}
            ),
            "global_target_rows_json": json.dumps([int(value) for value in global_target_rows]),
            "row_base_shift": str(row_base_shift or "auto"),
        },
        timeout_sec=max(120, int(timeout_sec)),
    )
    _require_collective_rpc_ok(method="debug_prefix_state_row_hashes", results=results)
    return {"results": results}


def _normalize_prefix_session_debug_root(*, session_dir: str, session_id: str) -> str:
    """Return the session-root directory expected by debug_prefix_state_row_hashes."""

    root = Path(str(session_dir or "")).expanduser()
    session_token = str(session_id or "").strip()
    if session_token and root.name == session_token:
        root = root.parent
    return str(root)


def _update_prefix_manifest_with_state(
    *,
    session_dir: Path,
    rpc_results: List[Dict[str, Any]],
    chunk_bytes: int,
    fallback_scheduler_payload: Optional[Mapping[str, Any]] = None,
    default_block_size_tokens: int = 0,
) -> Dict[str, Any]:
    manifest_path = session_dir / "prefix_session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fp8_contract = {}
    kv_cache_dtype = str(manifest.get("kv_cache_dtype") or "")
    if str(kv_cache_dtype).lower().startswith("fp8"):
        fp8_contract = {
            "fp8_storage_format": str(manifest.get("fp8_storage_format") or ""),
            "fp8_decode_lut_sha256": str(manifest.get("fp8_decode_lut_sha256") or ""),
        }
    tensor_files = list(manifest.get("tensor_files") or [])
    rank_rows: List[Dict[str, Any]] = []
    total_bytes = 0
    for row in rpc_results:
        rank = int(row.get("rank") or row.get("rank_index") or 0)
        rank_entry = {
            "rank": int(rank),
            "tensor_count": int(row.get("tensor_count") or 0),
            "total_bytes": int(row.get("total_bytes") or 0),
            "rank_manifest_path": str(row.get("rank_manifest_path") or ""),
        }
        rank_rows.append(rank_entry)
        total_bytes += int(rank_entry["total_bytes"])
        entries = row.get("tensor_entries")
        if not isinstance(entries, list):
            rank_manifest_path = Path(str(rank_entry["rank_manifest_path"] or ""))
            if rank_manifest_path and rank_manifest_path.exists():
                rank_manifest = json.loads(rank_manifest_path.read_text(encoding="utf-8"))
                entries = list(rank_manifest.get("tensor_entries") or [])
            else:
                entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tensor_files.append(
                {
                    "path": str(entry.get("path") or ""),
                    "shape": [int(x) for x in list(entry.get("shape") or [])],
                    "dtype": str(entry.get("dtype") or ""),
                    "nbytes": int(entry.get("nbytes") or 0),
                    "group_id": int(entry.get("group_id") or -1),
                    "selection": str(entry.get("selection") or ""),
                    "saved_row_count": int(entry.get("saved_row_count") or 0),
                    "row_index_runs": [
                        [int(item[0]), int(item[1])]
                        for item in list(entry.get("row_index_runs") or [])
                        if isinstance(item, (list, tuple)) and len(item) >= 2
                    ],
                    "checksum_sha256": str(entry.get("checksum_sha256") or ""),
                    "kind": "prefix_state_tensor",
                    **(
                        {
                            "fp8_storage_format": str(
                                fp8_contract.get("fp8_storage_format") or ""
                            ),
                            "fp8_decode_lut_sha256": str(
                                fp8_contract.get("fp8_decode_lut_sha256") or ""
                            ),
                        }
                        if fp8_contract and str(entry.get("dtype") or "") == "torch.uint8"
                        else {}
                    ),
                }
            )
    manifest["tensor_files"] = tensor_files
    manifest["prefix_state_format"] = {
        "name": "kvstate_bin_v1",
        "endianness": "little",
        "per_rank_shards": True,
        "chunk_bytes": int(max(1, int(chunk_bytes))),
    }
    manifest["prefix_state_capture"] = {
        "rank_count": int(len(rank_rows)),
        "total_bytes": int(total_bytes),
        "ranks": rank_rows,
    }
    if str(manifest.get("attention_mode") or "").strip().lower() == "streaming_mla":
        streaming_layout = _streaming_apc_layout_from_rpc_results(
            rpc_results,
            attention_config=dict(manifest.get("attention_config") or {}),
        )
        fallback_streaming_layout = _streaming_layout_from_scheduler_payload(
            fallback_scheduler_payload or {}
        )
        if fallback_streaming_layout:
            if not streaming_layout:
                streaming_layout = fallback_streaming_layout
            else:
                if not list(streaming_layout.get("manager_rows") or []):
                    streaming_layout["manager_rows"] = list(
                        fallback_streaming_layout.get("manager_rows") or []
                    )
                if not dict(streaming_layout.get("source_group_rows") or {}):
                    streaming_layout["source_group_rows"] = dict(
                        fallback_streaming_layout.get("source_group_rows") or {}
                    )
                if not list(streaming_layout.get("global_target_rows") or []):
                    streaming_layout["global_target_rows"] = list(
                        fallback_streaming_layout.get("global_target_rows") or []
                    )
                if not dict(streaming_layout.get("linear_target_last_rows") or {}):
                    streaming_layout["linear_target_last_rows"] = dict(
                        fallback_streaming_layout.get("linear_target_last_rows") or {}
                    )
                if int(streaming_layout.get("global_target_group_id") or -1) < 0:
                    streaming_layout["global_target_group_id"] = int(
                        fallback_streaming_layout.get("global_target_group_id") or -1
                    )
                if int(streaming_layout.get("full_block_hash_count") or 0) <= 0:
                    streaming_layout["full_block_hash_count"] = int(
                        fallback_streaming_layout.get("full_block_hash_count") or 0
                    )
        if streaming_layout:
            manifest["apc_cache_layout"] = {
                "full_block_hash_count": int(
                    streaming_layout.get("full_block_hash_count") or 0
                ),
                "linear_target_last_rows": {
                    str(key): int(value)
                    for key, value in dict(
                        streaming_layout.get("linear_target_last_rows") or {}
                    ).items()
                    if int(value) > 0
                },
                "global_target_rows": [
                    int(value)
                    for value in list(streaming_layout.get("global_target_rows") or [])
                    if int(value) > 0
                ],
                "source_group_rows": {
                    str(key): [
                        int(value) for value in list(values or []) if int(value) > 0
                    ]
                    for key, values in dict(
                        streaming_layout.get("source_group_rows") or {}
                    ).items()
                    if [int(value) for value in list(values or []) if int(value) > 0]
                },
                "manager_rows": [
                    {
                        "group_index": int(row.get("group_index") or 0),
                        "kv_cache_group_id": int(row.get("kv_cache_group_id") or 0),
                        "block_size": int(row.get("block_size") or 0),
                        "num_full_blocks": int(row.get("num_full_blocks") or 0),
                        "cached_block_offsets": [
                            int(value)
                            for value in list(row.get("cached_block_offsets") or [])
                            if int(value) >= 0
                        ],
                        "cached_block_ids": [
                            int(value)
                            for value in list(row.get("cached_block_ids") or [])
                            if int(value) > 0
                        ],
                    }
                    for row in list(streaming_layout.get("manager_rows") or [])
                    if isinstance(row, Mapping)
                ],
                "global_target_group_id": int(
                    streaming_layout.get("global_target_group_id") or -1
                ),
            }
    manifest = _annotate_streaming_attention_config_from_manifest(
        manifest,
        default_block_size_tokens=int(default_block_size_tokens),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for rank_entry in rank_rows:
        rank_manifest_path = Path(str(rank_entry.get("rank_manifest_path") or ""))
        if not rank_manifest_path.exists():
            continue
        rank_index = int(rank_entry.get("rank") or 0)
        rpc_row = next(
            (
                row
                for row in rpc_results
                if int(row.get("rank") or row.get("rank_index") or 0) == rank_index
            ),
            None,
        )
        if rank_manifest_path.exists():
            try:
                rank_manifest = json.loads(rank_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                rank_manifest = {}
        else:
            rank_manifest = {}
        if isinstance(rpc_row, Mapping):
            rank_manifest.update(dict(rpc_row))
        else:
            rank_manifest.update(dict(rank_entry))
        rank_manifest["schema_version"] = str(manifest.get("schema_version") or "")
        rank_manifest["attention_mode"] = str(manifest.get("attention_mode") or "")
        rank_manifest_path.write_text(
            json.dumps(rank_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def _update_prefix_manifest_with_apc_cache_layout(
    *,
    session_dir: Path,
    apc_cache_layout: Mapping[str, Any],
) -> Dict[str, Any]:
    """Persist compact APC cache layout metadata into the session manifest.

    Args:
        session_dir: Prefix-session directory containing the root manifest.
        apc_cache_layout: Extracted APC cache metadata for the saved prefix.

    Returns:
        The updated manifest payload.
    """

    manifest_path = session_dir / "prefix_session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["apc_cache_layout"] = {
        "full_block_hash_count": int(apc_cache_layout.get("full_block_hash_count") or 0),
        "linear_target_last_rows": {
            str(key): int(value)
            for key, value in dict(apc_cache_layout.get("linear_target_last_rows") or {}).items()
            if int(value) > 0
        },
        "global_target_rows": [
            int(value)
            for value in list(apc_cache_layout.get("global_target_rows") or [])
            if int(value) > 0
        ],
        "source_group_rows": {
            str(key): [int(value) for value in list(values or []) if int(value) > 0]
            for key, values in dict(apc_cache_layout.get("source_group_rows") or {}).items()
            if [int(value) for value in list(values or []) if int(value) > 0]
        },
        "manager_rows": [
            {
                "group_index": int(row.get("group_index") or 0),
                "kv_cache_group_id": int(row.get("kv_cache_group_id") or 0),
                "block_size": int(row.get("block_size") or 0),
                "num_full_blocks": int(row.get("num_full_blocks") or 0),
                "cached_block_offsets": [
                    int(value)
                    for value in list(row.get("cached_block_offsets") or [])
                    if int(value) >= 0
                ],
                "cached_block_ids": [
                    int(value)
                    for value in list(row.get("cached_block_ids") or [])
                    if int(value) > 0
                ],
            }
            for row in list(apc_cache_layout.get("manager_rows") or [])
            if isinstance(row, Mapping)
        ],
        "global_target_group_id": int(apc_cache_layout.get("global_target_group_id") or -1),
    }
    manifest = _annotate_streaming_attention_config_from_manifest(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _first_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value or "")
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        out = float(match.group(1))
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _parse_sclk_rows_from_amd_smi(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    data = ((payload.get("gpu_data") if "gpu_data" in payload else None) or ((payload.get("data") or {}).get("gpu_data") if isinstance(payload.get("data"), dict) else None))
    if not isinstance(data, list):
        return rows
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        clock = item.get("clock") if isinstance(item.get("clock"), dict) else {}
        block = clock.get("gfx_0") if isinstance(clock.get("gfx_0"), dict) else {}
        raw_clk = block.get("clk")
        if isinstance(raw_clk, dict):
            raw_clk = raw_clk.get("value")
        sclk = _first_float(raw_clk)
        if sclk is None:
            continue
        rows.append({"gpu": idx, "sclk_mhz": float(sclk)})
    return rows


def _parse_sclk_rows_from_rocm_smi(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, item in payload.items():
        if not isinstance(item, dict):
            continue
        sclk: Optional[float] = None
        for field, raw in item.items():
            lname = str(field).lower()
            if "sclk" in lname and "speed" in lname:
                sclk = _first_float(raw)
                if sclk is not None:
                    break
        if sclk is None:
            for field, raw in item.items():
                lname = str(field).lower()
                if ("gfx" in lname or "clk" in lname or "clock" in lname) and "level" not in lname:
                    sclk = _first_float(raw)
                    if sclk is not None:
                        break
        if sclk is None:
            continue
        gpu_idx = _first_float(str(key).replace("card", ""))
        rows.append({"gpu": int(gpu_idx) if gpu_idx is not None else len(rows), "sclk_mhz": float(sclk)})
    return rows


def _collect_sclk_rows() -> Tuple[str, List[Dict[str, Any]]]:
    amd = shutil.which("amd-smi")
    if amd:
        for argv in (
            [amd, "metric", "--clock", "--json"],
            [amd, "metric", "--clock", "--perf-level", "--json"],
        ):
            payload = _run_json_cmd(argv, timeout_sec=10)
            if payload:
                rows = _parse_sclk_rows_from_amd_smi(payload)
                if rows:
                    return ("amd-smi", rows)

    for rocm in (shutil.which("rocm-smi"), "/opt/rocm/bin/rocm-smi"):
        if not rocm:
            continue
        payload = _run_json_cmd([rocm, "--showclocks", "--json"], timeout_sec=10)
        if payload:
            rows = _parse_sclk_rows_from_rocm_smi(payload)
            if rows:
                return ("rocm-smi", rows)
    return ("none", [])


def _sample_sclk_busy(
    *,
    expected_gpu_count: int,
    min_sclk_mhz: float,
    sample_count: int,
    sample_interval_sec: float,
    min_busy_gpus: int,
    required_hit_ratio: float,
) -> Dict[str, Any]:
    sample_n = max(1, int(sample_count))
    interval_sec = max(0.0, float(sample_interval_sec))
    threshold_mhz = max(0.0, float(min_sclk_mhz))
    expected = max(1, int(expected_gpu_count))
    busy_gpu_target = int(min_busy_gpus) if int(min_busy_gpus) > 0 else expected
    busy_gpu_target = max(1, busy_gpu_target)
    ratio = min(1.0, max(0.0, float(required_hit_ratio)))
    required_hits = max(1, int(math.ceil(sample_n * ratio)))

    samples: List[Dict[str, Any]] = []
    hit_count = 0
    max_busy = 0
    max_peak = 0.0
    collector = "none"

    for idx in range(sample_n):
        c, rows = _collect_sclk_rows()
        if c != "none":
            collector = c
        sclk_vals = [float(r.get("sclk_mhz") or 0.0) for r in rows]
        busy_count = sum(1 for v in sclk_vals if v >= threshold_mhz)
        peak = max(sclk_vals) if sclk_vals else 0.0
        max_busy = max(max_busy, busy_count)
        max_peak = max(max_peak, peak)
        if busy_count >= busy_gpu_target:
            hit_count += 1
        samples.append(
            {
                "sample_index": idx + 1,
                "gpu_count": len(sclk_vals),
                "busy_gpu_count": busy_count,
                "peak_sclk_mhz": round(float(peak), 6),
            }
        )
        if idx + 1 < sample_n and interval_sec > 0:
            time.sleep(interval_sec)

    return {
        "busy": bool(hit_count >= required_hits),
        "hits": int(hit_count),
        "required_hits": int(required_hits),
        "sample_count": int(sample_n),
        "collector": collector,
        "threshold_mhz": round(float(threshold_mhz), 6),
        "min_busy_gpus": int(busy_gpu_target),
        "max_busy_gpu_count": int(max_busy),
        "max_peak_sclk_mhz": round(float(max_peak), 6),
        "samples": samples,
    }


def _is_timeout_exception(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    if isinstance(exc, url_error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
    lowered = str(exc).lower()
    return "timed out" in lowered or "timeout" in lowered


def _wait_health(base_url: str, timeout_sec: int, proc: Optional[subprocess.Popen[Any]] = None) -> bool:
    """Wait for the server health endpoint to return HTTP 200."""

    return _prefix_session_lib.wait_for_vllm_health(
        base_url=base_url,
        timeout_sec=timeout_sec,
        proc=proc,
    )


_CUDAGRAPH_MODE_KEY_RE = re.compile(
    r'cudagraph_mode"?\s*[:=]\s*"?(?P<mode>[A-Za-z_]+)"?'
)
_CUDAGRAPH_DOWNGRADE_RE = re.compile(r"setting cudagraph_mode=(?P<mode>[A-Z_]+)")
def _parse_jsonish_compilation_config(raw: str) -> object:
    text = str(raw or "").strip()
    if not text:
        raise json.JSONDecodeError("empty compilation config", text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if not (text.startswith("{") and text.endswith("}")):
        raise json.JSONDecodeError("not a JSON object", text, 0)
    body = text[1:-1].strip()
    if not body:
        return {}
    result: Dict[str, Any] = {}
    for item in body.split(","):
        piece = item.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise json.JSONDecodeError("missing ':' in object item", text, 0)
        key_raw, value_raw = piece.split(":", 1)
        key = key_raw.strip().strip("\"'")
        if not key:
            raise json.JSONDecodeError("empty object key", text, 0)
        value_text = value_raw.strip()
        lowered = value_text.lower()
        if lowered == "true":
            value: Any = True
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
        result[key] = value
    return result


def _normalize_compilation_config_json(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    payload = _parse_jsonish_compilation_config(text)
    if not isinstance(payload, dict):
        raise json.JSONDecodeError(
            "compilation config must be a JSON object", text, 0
        )
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _effective_vllm_compilation_config(args: argparse.Namespace) -> str:
    """Return the effective vLLM compilation-config JSON for this launch."""

    return _normalize_compilation_config_json(
        str(args.vllm_compilation_config or "")
    )


def _requested_cudagraph_mode(compilation_config_raw: str) -> str:
    """Resolve the requested cudagraph mode token from the CLI JSON payload."""

    raw = str(compilation_config_raw or "").strip()
    if not raw:
        return ""
    try:
        payload = _parse_jsonish_compilation_config(raw)
    except Exception:  # noqa: BLE001
        payload = None
    if isinstance(payload, dict):
        mode = str(payload.get("cudagraph_mode") or "").strip().upper()
        if mode:
            return mode
    match = _CUDAGRAPH_MODE_KEY_RE.search(raw)
    return str(match.group("mode") if match else "").strip().upper()


def _resolved_vllm_worker_extension_cls(args: argparse.Namespace) -> str:
    """Return the effective worker extension class for the launched server."""

    return str(getattr(args, "vllm_worker_extension_cls", "") or "").strip()


def _inspect_graph_mode_status(
    *,
    log_path: Path,
    compilation_config_raw: str,
) -> Dict[str, Any]:
    """Parse vLLM graph-capture status from the server log."""

    requested_mode = _requested_cudagraph_mode(compilation_config_raw)
    requested_graph_enabled = bool(requested_mode and requested_mode != "NONE")
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        log_text = ""

    downgrade_modes = [str(m).strip().upper() for m in _CUDAGRAPH_DOWNGRADE_RE.findall(log_text)]
    skip_lines = [
        line.strip()
        for line in log_text.splitlines()
        if "Skipping CUDA graph capture" in line
    ]
    finished_lines = [
        line.strip()
        for line in log_text.splitlines()
        if "Graph capturing finished in" in line
    ]
    error_lines = [
        line.strip()
        for line in log_text.splitlines()
        if (
            ("stream-capture" in line.lower())
            or ("stream capture" in line.lower())
            or (
                "graph capture" in line.lower()
                and ("error" in line.lower() or "fail" in line.lower())
            )
        )
    ]

    resolved_mode = requested_mode
    resolved_from_log = False
    if downgrade_modes:
        resolved_mode = downgrade_modes[-1]
        resolved_from_log = True
    elif skip_lines and any("manually set to `NONE`" in line for line in skip_lines):
        resolved_mode = "NONE"
        resolved_from_log = True
    elif not resolved_mode:
        resolved_mode = "NONE"

    capture_finished = bool(finished_lines)
    skip_capture = bool(skip_lines)
    capture_model_ran = bool(capture_finished)
    capture_ok = bool(
        requested_graph_enabled
        and resolved_mode not in {"", "NONE"}
        and not skip_capture
        and capture_model_ran
    )

    return {
        "requested_cudagraph_mode": requested_mode,
        "requested_graph_enabled": bool(requested_graph_enabled),
        "resolved_cudagraph_mode": resolved_mode,
        "resolved_from_log": bool(resolved_from_log),
        "downgrade_modes": downgrade_modes,
        "skip_cuda_graph_capture": bool(skip_capture),
        "skip_cuda_graph_capture_lines": skip_lines[:8],
        "capture_model_ran": bool(capture_model_ran),
        "graph_capture_finished": bool(capture_finished),
        "graph_capture_finished_lines": finished_lines[:8],
        "capture_error_lines": error_lines[:8],
        "capture_ok": bool(capture_ok),
        "log_path": str(log_path),
    }


def _persist_graph_mode_status(
    *,
    out_dir: Path,
    run_meta: Dict[str, Any],
    compilation_config_raw: str,
    log_path: Path,
) -> Dict[str, Any]:
    """Persist parsed graph-capture status and mirror it into run metadata."""

    status = _inspect_graph_mode_status(
        log_path=log_path,
        compilation_config_raw=compilation_config_raw,
    )
    status_path = out_dir / "graph_mode_status.json"
    status["artifact_path"] = str(status_path)
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    graph_meta = {
        "requested_cudagraph_mode": str(status.get("requested_cudagraph_mode") or ""),
        "resolved_cudagraph_mode": str(status.get("resolved_cudagraph_mode") or ""),
        "requested_graph_enabled": bool(status.get("requested_graph_enabled")),
        "skip_cuda_graph_capture": bool(status.get("skip_cuda_graph_capture")),
        "capture_model_ran": bool(status.get("capture_model_ran")),
        "graph_capture_finished": bool(status.get("graph_capture_finished")),
        "capture_ok": bool(status.get("capture_ok")),
        "artifact_path": str(status_path),
    }
    run_meta["graph_mode_status"] = graph_meta
    vllm_meta = dict(run_meta.get("vllm") or {})
    vllm_meta["requested_cudagraph_mode"] = str(
        status.get("requested_cudagraph_mode") or ""
    )
    vllm_meta["resolved_cudagraph_mode"] = str(
        status.get("resolved_cudagraph_mode") or ""
    )
    vllm_meta["graph_capture_finished"] = bool(status.get("graph_capture_finished"))
    vllm_meta["graph_capture_skipped"] = bool(status.get("skip_cuda_graph_capture"))
    run_meta["vllm"] = vllm_meta
    run_meta["host_env"] = collect_host_env()
    run_meta["mla_kernel_fingerprint"] = collect_mla_kernel_fingerprint()
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return status


def _maybe_require_graph_capture(
    *,
    status: Mapping[str, Any],
) -> None:
    """Fail fast when graph mode was explicitly requested but not activated."""

    if not bool(status.get("requested_graph_enabled")):
        return
    resolved_mode = str(status.get("resolved_cudagraph_mode") or "").strip().upper()
    problems: List[str] = []
    if resolved_mode in {"", "NONE"}:
        problems.append(f"resolved_cudagraph_mode={resolved_mode or 'NONE'}")
    if bool(status.get("skip_cuda_graph_capture")):
        problems.append("skip_cuda_graph_capture=true")
    if not bool(status.get("capture_model_ran")):
        problems.append("capture_model_ran=false")
    if not problems:
        return
    raise SystemExit(
        "graph_mode_unavailable: "
        + ", ".join(problems)
        + f". See {str(status.get('artifact_path') or '')}."
    )


def _start_vllm_server(
    args: argparse.Namespace, out_dir: Path
) -> Tuple[subprocess.Popen[Any], Path, List[str], Dict[str, str]]:
    log_path = out_dir / "vllm_server.log"
    compilation_config = _effective_vllm_compilation_config(args)
    vllm_hf_overrides = _resolve_kimi_mla_hf_overrides(args)
    use_hf_overrides = any(
        int(value or 0) != 0 for value in dict(vllm_hf_overrides).values()
    )
    cmd = [
        str(args.vllm_bin),
        "serve",
        str(args.model_id),
        "--host",
        str(args.vllm_host),
        "--port",
        str(args.vllm_port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.vllm_data_parallel_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--max-model-len",
        str(args.vllm_max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        str(args.vllm_dtype),
        "--kv-cache-dtype",
        str(args.kv_cache_dtype),
    ]
    if str(args.vllm_load_format or "").strip():
        cmd.extend(["--load-format", str(args.vllm_load_format).strip()])
    if str(args.vllm_hf_config_path or "").strip():
        cmd.extend(["--hf-config-path", str(args.vllm_hf_config_path).strip()])
    if use_hf_overrides:
        cmd.extend(["--hf-overrides", _serialize_hf_overrides(vllm_hf_overrides)])
    if compilation_config:
        cmd.extend(["--compilation-config", compilation_config])
    if bool(args.vllm_enforce_eager):
        cmd.append("--enforce-eager")
    if bool(getattr(args, "vllm_enable_prefix_caching", False)):
        cmd.append("--enable-prefix-caching")
        # Surface cached prompt-token counts in OpenAI usage payloads so
        # restore-and-hot-reuse runs can verify APC reuse from client artifacts.
        cmd.append("--enable-prompt-tokens-details")
    if bool(args.calculate_kv_scales):
        cmd.append("--calculate-kv-scales")
    worker_extension_cls = _resolved_vllm_worker_extension_cls(args)
    if worker_extension_cls:
        cmd.extend(["--worker-extension-cls", worker_extension_cls])
    if int(args.vllm_max_num_seqs or 0) > 0:
        cmd.extend(["--max-num-seqs", str(int(args.vllm_max_num_seqs))])
    if int(args.vllm_max_num_batched_tokens or 0) > 0:
        cmd.extend(["--max-num-batched-tokens", str(int(args.vllm_max_num_batched_tokens))])
    if str(args.vllm_attention_backend or "").strip():
        cmd.extend(["--attention-backend", str(args.vllm_attention_backend).strip()])
    if int(args.vllm_block_size or 0) > 0:
        cmd.extend(["--block-size", str(int(args.vllm_block_size))])
    if int(args.vllm_kv_cache_memory_bytes or 0) > 0:
        cmd.extend(["--kv-cache-memory-bytes", str(int(args.vllm_kv_cache_memory_bytes))])
    if int(args.vllm_num_gpu_blocks_override or 0) > 0:
        cmd.extend(["--num-gpu-blocks-override", str(int(args.vllm_num_gpu_blocks_override))])
    if bool(args.trust_remote_code):
        cmd.append("--trust-remote-code")
    if bool(getattr(args, "vllm_disable_custom_all_reduce", False)):
        cmd.append("--disable-custom-all-reduce")

    server_env = _sanitize_server_env(server_env=dict(os.environ), args=args)
    # Keep torch.compile/Triton artifacts private to this launched server. The
    # graph-mode debug path reuses the same model/config repeatedly, and stale
    # or partially-written compile artifacts from earlier failed runs can be
    # reloaded later as seemingly unrelated HIP illegal-memory faults.
    server_cache_root = out_dir / "server_compile_cache"
    inductor_cache = server_cache_root / "inductor_cache"
    triton_cache = server_cache_root / "triton_cache"
    for cache_dir in (server_cache_root, inductor_cache, triton_cache):
        cache_dir.mkdir(parents=True, exist_ok=True)
    server_env["VLLM_CACHE_ROOT"] = str(server_cache_root)
    server_env["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    server_env["TRITON_CACHE_DIR"] = str(triton_cache)
    enable_dev_mode = str(getattr(args, "prefix_session_mode", "none") or "none") != "none"
    if not enable_dev_mode:
        enable_dev_mode = _query_live_layout_enabled()
    if enable_dev_mode:
        # Enable dev-only vLLM RPC surface for local prefix-state capture/restore
        # and for live-layout debugging.
        server_env["VLLM_SERVER_DEV_MODE"] = "1"
    if bool(getattr(args, "disable_rocm_skinny_gemm", False)):
        server_env["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"
    rocprof_prefix, rocprof_meta = _build_rocprof_prefix(args, out_dir=out_dir)
    wrapped_cmd = list(rocprof_prefix) + list(cmd)
    if rocprof_meta:
        (out_dir / "rocprof_meta.json").write_text(
            json.dumps(rocprof_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(  # noqa: S603  # pylint: disable=consider-using-with
            wrapped_cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(repo_path()),
            env=server_env,
            start_new_session=True,
        )
    # Record the session/process-group leader so cleanup can still tear down the
    # full vLLM worker tree even if the parent `vllm serve` process exits first.
    setattr(proc, "_hflc_pgid", proc.pid)
    return proc, log_path, wrapped_cmd, server_env


def _stop_process(proc: Optional[subprocess.Popen[Any]], grace_sec: float = 5.0) -> None:
    if proc is None:
        return
    pgid = getattr(proc, "_hflc_pgid", None)
    if pgid is not None:
        try:
            os.killpg(int(pgid), signal.SIGTERM)
        except Exception:
            pass
    elif proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            return
    deadline = time.time() + float(grace_sec)
    while time.time() < deadline:
        if proc.poll() is not None:
            if pgid is None:
                return
        time.sleep(0.2)
    try:
        if pgid is not None:
            os.killpg(int(pgid), signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def _server_compile_cache_paths(out_dir: Path) -> Dict[str, Path]:
    """Build the per-run compile-cache directory map.

    Args:
        out_dir: Phase output directory for the benchmark run.

    Returns:
        Mapping from logical cache labels to absolute directory paths.
    """

    server_cache_root = out_dir / "server_compile_cache"
    return {
        "server_compile_cache": server_cache_root,
        "inductor_cache": server_cache_root / "inductor_cache",
        "triton_cache": server_cache_root / "triton_cache",
    }


def _cleanup_server_compile_cache(
    out_dir: Path,
    cleanup_mode: str,
) -> Dict[str, Any]:
    """Remove selected compile-cache directories after a run finishes.

    Args:
        out_dir: Phase output directory for the benchmark run.
        cleanup_mode: Cleanup scope. Supported values are ``none``,
            ``triton``, and ``all``.

    Returns:
        Serializable cleanup summary with removed, missing, and failed paths.
    """

    mode = str(cleanup_mode or "none").strip().lower()
    cache_paths = _server_compile_cache_paths(out_dir)
    result: Dict[str, Any] = {
        "cleanup_mode": mode,
        "enabled": mode != "none",
        "removed_paths": [],
        "missing_paths": [],
        "errors": [],
    }
    if mode == "none":
        return result
    if mode not in {"triton", "all"}:
        raise ValueError(f"Unsupported compile-cache cleanup mode: {cleanup_mode}")

    targets = (
        [cache_paths["triton_cache"]]
        if mode == "triton"
        else [cache_paths["server_compile_cache"]]
    )
    for target in targets:
        resolved = str(target.resolve())
        if not target.exists():
            result["missing_paths"].append(resolved)
            continue
        try:
            shutil.rmtree(target)
        except OSError as exc:
            result["errors"].append(
                {
                    "path": resolved,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        result["removed_paths"].append(resolved)
    return result


def _write_compile_cache_cleanup_artifact(
    out_dir: Path,
    cleanup_result: Mapping[str, Any],
) -> None:
    """Persist compile-cache cleanup status next to other phase artifacts.

    Args:
        out_dir: Phase output directory for the benchmark run.
        cleanup_result: Serializable cleanup summary payload.
    """

    artifact_path = out_dir / "server_compile_cache_cleanup.json"
    artifact_path.write_text(
        json.dumps(dict(cleanup_result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stop_prefill_recorder(
    recorder: Optional[_PrefillProgressRecorder],
    stopped: bool,
    *,
    final_status: str,
    note: str,
) -> bool:
    if recorder is None or stopped:
        return stopped
    recorder.stop(final_status=final_status, note=note)
    return True


def _find_subsequence(haystack: List[int], needle: List[int], start_index: int = 0) -> int:
    """Finds the first occurrence of a token subsequence.

    Args:
        haystack: Token sequence to search.
        needle: Token subsequence to locate.
        start_index: Inclusive start offset for the search.

    Returns:
        int: Start index of the subsequence, or ``-1`` when not found.
    """
    if not needle:
        return -1
    start = max(0, int(start_index))
    limit = len(haystack) - len(needle) + 1
    for idx in range(start, max(start, limit)):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return -1


def _build_prompt_token_ids_from_shape(
    *,
    tokenizer: Any,
    prompt_ids: List[int],
    prompt_shape_spec: Mapping[str, Any],
    system_prompt: str,
    max_new_tokens: int,
    model_id: str,
) -> Optional[List[int]]:
    """Render prompt-shape to token IDs without decoding large prompt text.

    This uses a unique marker for ``{{prompt_text}}`` and replaces the marker
    token span with ``prompt_ids`` after chat-template tokenization.
    """
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None
    marker = "<<LCS_PROMPT_SLOT_4E9B7E0D>>"
    try:
        summary_source = ""
        try:
            summary_source = tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:  # noqa: BLE001
            summary_source = ""
        messages = render_messages(
            prompt_shape_spec,
            _build_prompt_shape_render_context(
                prompt_text=marker,
                prompt_text_summary_source=summary_source,
                system_prompt=str(system_prompt),
                max_new_tokens=int(max_new_tokens),
                prompt_tokens=int(len(prompt_ids)),
                model_id=str(model_id),
            ),
        )
        rendered_ids = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        return None
    rendered_ids = _coerce_chat_template_ids(rendered_ids)
    if not rendered_ids:
        return None
    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    marker_ids = [int(x) for x in marker_ids]
    start = _find_subsequence(rendered_ids, marker_ids)
    if start < 0:
        return None
    end = start + len(marker_ids)
    return [*rendered_ids[:start], *[int(x) for x in prompt_ids], *rendered_ids[end:]]


def _build_same_message_query_prompt_token_ids(
    *,
    tokenizer: Any,
    prefix_token_ids: List[int],
    suffix_text: str,
    continuation_tail_token_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """Builds the full prompt-token payload for same-message query continuation.

    Args:
        tokenizer: Hugging Face-compatible tokenizer.
        prefix_token_ids: Saved reusable prefix token IDs.
        suffix_text: User-supplied suffix text that continues the split user
            message.
        continuation_tail_token_ids: Serialized tail that must follow the
            suffix within the same message.

    Returns:
        Tuple[List[int], List[int]]: ``(prompt_token_ids, suffix_token_ids)``.
    """
    return _prefix_session_lib.build_same_message_query_prompt_token_ids(
        tokenizer=tokenizer,
        prefix_token_ids=prefix_token_ids,
        suffix_text=suffix_text,
        continuation_tail_token_ids=continuation_tail_token_ids,
    )


def _persist_generated_text(
    *,
    out_dir: Path,
    target_index: int,
    iteration: int,
    generated_text: str,
) -> str:
    """Persist generated completion text for one measurement row.

    Args:
        out_dir: Output directory for the current run.
        target_index: One-based index of the sweep target.
        iteration: One-based run iteration for the target point.
        generated_text: Decoded completion text emitted by the model.

    Returns:
        Absolute path to the saved text file.
    """
    generated_dir = out_dir / "generated_text"
    generated_dir.mkdir(parents=True, exist_ok=True)
    text_path = generated_dir / f"target_{int(target_index):04d}_run_{int(iteration):03d}.txt"
    text_path.write_text(str(generated_text), encoding="utf-8")
    return str(text_path)


def _persist_request_debug_artifacts(
    *,
    out_dir: Path,
    target_index: int,
    iteration: int,
    debug_payload: Mapping[str, Any],
) -> Dict[str, str]:
    """Persist one request's debug artifacts under ``request_debug/``."""

    request_debug_dir = out_dir / "request_debug"
    request_debug_dir.mkdir(parents=True, exist_ok=True)

    prefix_row_hashes_path = ""
    prefix_row_hashes = dict((debug_payload or {}).get("prefix_row_hashes") or {})
    if prefix_row_hashes:
        prefix_row_hashes_out = (
            request_debug_dir
            / f"target_{int(target_index):04d}_run_{int(iteration):03d}_prefix_row_hashes.json"
        )
        prefix_row_hashes_out.write_text(
            json.dumps(prefix_row_hashes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prefix_row_hashes_path = str(prefix_row_hashes_out)

    debug_to_write = dict(debug_payload or {})
    debug_to_write.pop("prefix_row_hashes", None)
    debug_to_write.pop("request_payload", None)
    debug_to_write.pop("query_messages", None)
    debug_to_write.pop("prompt_token_ids", None)
    debug_to_write.pop("same_message_capture", None)
    debug_to_write.pop("suffix_query_text", None)
    if prefix_row_hashes_path:
        debug_to_write["prefix_row_hashes_path"] = str(prefix_row_hashes_path)

    request_payload_path = ""
    if _request_debug_flag(debug_payload, "dump_request_payload") and (
        debug_payload or {}
    ).get("request_payload") is not None:
        request_payload_out = (
            request_debug_dir
            / f"target_{int(target_index):04d}_run_{int(iteration):03d}_request_payload.json"
        )
        request_payload_out.write_text(
            json.dumps(debug_payload.get("request_payload"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        request_payload_path = str(request_payload_out)

    query_messages_path = ""
    if _request_debug_flag(debug_payload, "dump_query_messages") and (
        debug_payload or {}
    ).get("query_messages") is not None:
        query_messages_out = (
            request_debug_dir
            / f"target_{int(target_index):04d}_run_{int(iteration):03d}_query_messages.json"
        )
        query_messages_out.write_text(
            json.dumps(debug_payload.get("query_messages"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        query_messages_path = str(query_messages_out)

    prompt_token_ids_path = ""
    if _request_debug_flag(debug_payload, "dump_prompt_token_ids") and (
        debug_payload or {}
    ).get("prompt_token_ids") is not None:
        prompt_token_ids_out = (
            request_debug_dir
            / f"target_{int(target_index):04d}_run_{int(iteration):03d}_prompt_token_ids.json"
        )
        prompt_token_ids_payload = {
            "prompt_token_count": int(
                len(list(debug_payload.get("prompt_token_ids") or []))
            ),
            "prompt_token_ids": [
                int(token_id)
                for token_id in list(debug_payload.get("prompt_token_ids") or [])
            ],
            "prompt_token_ids_source": str(
                debug_payload.get("prompt_token_ids_source") or ""
            ),
            "prompt_token_preview_text": str(
                debug_payload.get("prompt_token_preview_text") or ""
            ),
        }
        prompt_token_ids_out.write_text(
            json.dumps(prompt_token_ids_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prompt_token_ids_path = str(prompt_token_ids_out)

    same_message_capture_path = ""
    if (debug_payload or {}).get("same_message_capture") is not None:
        same_message_capture_out = (
            request_debug_dir
            / f"target_{int(target_index):04d}_run_{int(iteration):03d}_same_message_capture.json"
        )
        same_message_capture_out.write_text(
            json.dumps(
                debug_payload.get("same_message_capture"), indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        same_message_capture_path = str(same_message_capture_out)

    suffix_query_path = ""
    if (debug_payload or {}).get("suffix_query_text") is not None:
        suffix_query_out = (
            request_debug_dir
            / f"target_{int(target_index):04d}_run_{int(iteration):03d}_suffix_query.txt"
        )
        suffix_query_out.write_text(
            str(debug_payload.get("suffix_query_text") or ""),
            encoding="utf-8",
        )
        suffix_query_path = str(suffix_query_out)
    if request_payload_path:
        debug_to_write["request_payload_path"] = str(request_payload_path)
    if query_messages_path:
        debug_to_write["query_messages_path"] = str(query_messages_path)
    if prompt_token_ids_path:
        debug_to_write["prompt_token_ids_path"] = str(prompt_token_ids_path)
    if same_message_capture_path:
        debug_to_write["same_message_capture_path"] = str(
            same_message_capture_path
        )
    if suffix_query_path:
        debug_to_write["suffix_query_text_path"] = str(suffix_query_path)

    layout_path = (
        request_debug_dir
        / f"target_{int(target_index):04d}_run_{int(iteration):03d}_live_request_layout.json"
    )
    layout_path.write_text(
        json.dumps(debug_to_write, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "live_request_layout_path": str(layout_path),
        "prefix_row_hashes_path": str(prefix_row_hashes_path),
        "request_payload_path": str(request_payload_path),
        "query_messages_path": str(query_messages_path),
        "prompt_token_ids_path": str(prompt_token_ids_path),
        "same_message_capture_path": str(same_message_capture_path),
        "suffix_query_text_path": str(suffix_query_path),
    }


def _run_one_request(
    *,
    base_url: str,
    model_id: str,
    tokenizer: Any,
    prompt_ids: List[int],
    prompt_shape_spec: Mapping[str, Any],
    system_prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    timeout_sec: int,
    ignore_eos: bool,
    use_stream: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    output_mask_payload: Optional[Mapping[str, Any]],
    messages_override: Optional[List[Dict[str, str]]] = None,
    query_api_mode: str = "auto",
    prompt_token_ids_override: Optional[List[int]] = None,
    request_debug_payload: Optional[Dict[str, Any]] = None,
    stream_progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one vLLM request from prompt IDs via prompt-shape rendering."""
    requested_mode = str(query_api_mode or "auto").strip().lower()
    if requested_mode not in {"auto", "messages", "prompt_ids"}:
        raise ValueError(
            f"Unsupported query_api_mode={query_api_mode!r}; expected auto|messages|prompt_ids"
        )
    if request_debug_payload is not None:
        request_debug_payload["query_api_mode_requested"] = str(requested_mode)

    if prompt_token_ids_override is not None:
        if request_debug_payload is not None and not request_debug_payload.get(
            "prompt_token_ids_source"
        ):
            request_debug_payload["prompt_token_ids_source"] = (
                "explicit_prompt_token_ids_override"
            )
        return _run_one_request_completion_prompt_ids(
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            prompt_token_ids=[int(token_id) for token_id in prompt_token_ids_override],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            timeout_sec=timeout_sec,
            ignore_eos=ignore_eos,
            use_stream=use_stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            output_mask_payload=output_mask_payload,
            prompt_token_count_override=len(prompt_token_ids_override),
            request_debug_payload=request_debug_payload,
            stream_progress_cb=stream_progress_cb,
        )

    rendered_messages: Optional[List[Dict[str, str]]] = None

    def _ensure_rendered_messages() -> List[Dict[str, str]]:
        nonlocal rendered_messages
        if rendered_messages is not None:
            return [dict(msg) for msg in rendered_messages]
        if messages_override is not None:
            rendered_messages = [dict(msg) for msg in list(messages_override)]
            return [dict(msg) for msg in rendered_messages]
        prompt_text = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        rendered_messages = render_messages(
            prompt_shape_spec,
            _build_prompt_shape_render_context(
                prompt_text=prompt_text,
                system_prompt=str(system_prompt),
                max_new_tokens=int(max_new_tokens),
                prompt_tokens=int(len(prompt_ids)),
                model_id=str(model_id),
            ),
        )
        return [dict(msg) for msg in rendered_messages]

    if messages_override is not None and requested_mode != "prompt_ids":
        return _run_one_request_messages(
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            messages=_ensure_rendered_messages(),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            timeout_sec=timeout_sec,
            ignore_eos=ignore_eos,
            use_stream=use_stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            output_mask_payload=output_mask_payload,
            request_debug_payload=request_debug_payload,
            stream_progress_cb=stream_progress_cb,
        )

    prompt_token_ids: Optional[List[int]] = None
    if requested_mode != "messages":
        prompt_token_ids = _build_prompt_token_ids_from_shape(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            prompt_shape_spec=prompt_shape_spec,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        )
    if prompt_token_ids is not None and requested_mode in {"auto", "prompt_ids"}:
        if request_debug_payload is not None and not request_debug_payload.get(
            "prompt_token_ids_source"
        ):
            request_debug_payload["prompt_token_ids_source"] = "prompt_shape"
        if _request_debug_flag(request_debug_payload, "dump_query_messages"):
            request_debug_payload["query_messages"] = _ensure_rendered_messages()
        return _run_one_request_completion_prompt_ids(
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            timeout_sec=timeout_sec,
            ignore_eos=ignore_eos,
            use_stream=use_stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            output_mask_payload=output_mask_payload,
            request_debug_payload=request_debug_payload,
            stream_progress_cb=stream_progress_cb,
        )

    messages = _ensure_rendered_messages()
    if requested_mode == "prompt_ids":
        prompt_token_ids = _tokenize_messages_with_generation_prompt(tokenizer, messages)
        if request_debug_payload is not None and not request_debug_payload.get(
            "prompt_token_ids_source"
        ):
            request_debug_payload["prompt_token_ids_source"] = "messages_chat_template"
        if _request_debug_flag(request_debug_payload, "dump_query_messages"):
            request_debug_payload["query_messages"] = [dict(msg) for msg in messages]
        return _run_one_request_completion_prompt_ids(
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            timeout_sec=timeout_sec,
            ignore_eos=ignore_eos,
            use_stream=use_stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            output_mask_payload=output_mask_payload,
            request_debug_payload=request_debug_payload,
            stream_progress_cb=stream_progress_cb,
        )

    return _run_one_request_messages(
        base_url=base_url,
        model_id=model_id,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        timeout_sec=timeout_sec,
        ignore_eos=ignore_eos,
        use_stream=use_stream,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        output_mask_payload=output_mask_payload,
        request_debug_payload=request_debug_payload,
        stream_progress_cb=stream_progress_cb,
    )


def _run_one_request_completion_prompt_ids(
    *,
    base_url: str,
    model_id: str,
    tokenizer: Any,
    prompt_token_ids: List[int],
    max_new_tokens: int,
    min_new_tokens: int,
    timeout_sec: int,
    ignore_eos: bool,
    use_stream: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    output_mask_payload: Optional[Mapping[str, Any]],
    vllm_xargs: Optional[Mapping[str, Any]] = None,
    prompt_token_count_override: Optional[int] = None,
    request_debug_payload: Optional[Dict[str, Any]] = None,
    stream_progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one completions request from explicit prompt token IDs."""

    if _request_debug_flag(request_debug_payload, "dump_prompt_token_ids"):
        request_debug_payload["prompt_token_ids"] = [
            int(token_id) for token_id in list(prompt_token_ids or [])
        ]
        request_debug_payload["prompt_token_preview_text"] = _prompt_token_preview(
            tokenizer,
            prompt_token_ids,
        )
        if not request_debug_payload.get("prompt_token_ids_source"):
            request_debug_payload["prompt_token_ids_source"] = "explicit_prompt_ids"

    return _prefix_session_lib.run_completion_prompt_ids(
        base_url=base_url,
        model_id=model_id,
        tokenizer=tokenizer,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        timeout_sec=timeout_sec,
        ignore_eos=ignore_eos,
        use_stream=use_stream,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        output_mask_payload=output_mask_payload,
        vllm_xargs=vllm_xargs,
        prompt_token_count_override=prompt_token_count_override,
        request_debug_payload=request_debug_payload,
        stream_progress_cb=stream_progress_cb,
        stream_text_completions_fn=stream_text_completions,
    )


def _run_one_request_messages(
    *,
    base_url: str,
    model_id: str,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    min_new_tokens: int,
    timeout_sec: int,
    ignore_eos: bool,
    use_stream: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    output_mask_payload: Optional[Mapping[str, Any]],
    vllm_xargs: Optional[Mapping[str, Any]] = None,
    request_debug_payload: Optional[Dict[str, Any]] = None,
    stream_progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "ignore_eos": bool(ignore_eos),
    }
    if 0.0 < float(top_p) < 1.0:
        payload["top_p"] = float(top_p)
    if int(top_k) > 0:
        payload["top_k"] = int(top_k)
    if float(repetition_penalty) > 0.0 and abs(float(repetition_penalty) - 1.0) > 1e-9:
        payload["repetition_penalty"] = float(repetition_penalty)
    if abs(float(presence_penalty)) > 1e-9:
        payload["presence_penalty"] = float(presence_penalty)
    if abs(float(frequency_penalty)) > 1e-9:
        payload["frequency_penalty"] = float(frequency_penalty)
    if int(min_new_tokens) > 0:
        payload["min_tokens"] = int(min_new_tokens)
    if vllm_xargs:
        payload["vllm_xargs"] = dict(vllm_xargs)
    _apply_generation_output_mask(payload, output_mask_payload)
    if request_debug_payload is not None:
        request_debug_payload["request_endpoint"] = "/v1/chat/completions"
        request_debug_payload["query_api_mode_resolved"] = "messages"
        if _request_debug_flag(request_debug_payload, "dump_request_payload"):
            request_debug_payload["request_payload"] = json.loads(
                json.dumps(payload, sort_keys=True)
            )
        if _request_debug_flag(request_debug_payload, "dump_query_messages"):
            request_debug_payload["query_messages"] = [
                {"role": str(msg.get("role") or ""), "content": str(msg.get("content") or "")}
                for msg in list(messages or [])
            ]
        if _request_debug_flag(request_debug_payload, "dump_prompt_token_ids"):
            prompt_token_ids = _tokenize_messages_with_generation_prompt(
                tokenizer,
                messages,
            )
            request_debug_payload["prompt_token_ids"] = [
                int(token_id) for token_id in prompt_token_ids
            ]
            request_debug_payload["prompt_token_preview_text"] = _prompt_token_preview(
                tokenizer,
                prompt_token_ids,
            )
            if not request_debug_payload.get("prompt_token_ids_source"):
                request_debug_payload["prompt_token_ids_source"] = "messages_chat_template"
    result: Dict[str, Any]
    if use_stream:
        result = dict(
            stream_chat_completions(
                url=f"{base_url.rstrip('/')}/v1/chat/completions",
                payload=payload,
                timeout_sec=int(timeout_sec),
                on_progress=stream_progress_cb,
            )
        )
    else:
        req = url_request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with url_request.urlopen(req, timeout=int(timeout_sec)) as resp:
            body = resp.read()
        total_ms = (time.perf_counter() - start) * 1000.0
        parsed = json.loads(body.decode("utf-8"))
        usage = dict(parsed.get("usage") or {})
        choices = parsed.get("choices") or []
        first = choices[0] if isinstance(choices, list) and choices else {}
        answer_text = str(extract_response_text(parsed) or "")
        finish_reason = str(first.get("finish_reason") or "")
        stop_reason = str(first.get("stop_reason") or "")
        result = {
            "text": answer_text,
            "answer_text": answer_text,
            "usage": usage,
            "ttft_ms": None,
            "total_ms": round(total_ms, 6),
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
        }
    answer_text = str(result.get("answer_text") or result.get("text") or "")
    usage = dict(result.get("usage") or {})
    usage_completion_tokens = int(usage.get("completion_tokens") or 0)
    retokenized_completion_tokens = len(
        tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    )
    completion_tokens = max(usage_completion_tokens, retokenized_completion_tokens)
    if completion_tokens <= 0:
        raise RuntimeError(
            "No completion tokens returned from stream (empty generation). "
            f"finish_reason={result.get('finish_reason')!r} stop_reason={result.get('stop_reason')!r}"
        )
    total_ms = _as_float(result.get("total_ms"))
    ttft_ms = _as_float(result.get("ttft_ms"), default=total_ms)
    decode_ms = max(total_ms - ttft_ms, 0.0)
    gen_tok_s = (completion_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
    total_tok_s = (completion_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0
    decode_ms_per_token = (decode_ms / completion_tokens) if completion_tokens > 0 else 0.0
    cached_prompt_tokens = _extract_cached_prompt_tokens(usage)
    usage_prompt_tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, Mapping) else 0
    if usage_prompt_tokens <= 0:
        usage_prompt_tokens = len(_tokenize_messages(tokenizer, messages))
    out = build_timing_metrics_row(
        prompt_tokens=int(usage_prompt_tokens),
        completion_tokens=int(completion_tokens),
        ttft_ms=ttft_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        decode_ms_per_token=decode_ms_per_token,
        gen_tokens_per_sec=gen_tok_s,
        total_tokens_per_sec=total_tok_s,
    )
    out.update(
        {
            "finish_reason": str(result.get("finish_reason") or ""),
            "stop_reason": str(result.get("stop_reason") or ""),
            "request_id": str(result.get("request_id") or ""),
            "usage_prompt_tokens": int(usage_prompt_tokens),
            "usage_cached_prompt_tokens": int(cached_prompt_tokens),
            "usage_completion_tokens": int(usage_completion_tokens),
            "retokenized_completion_tokens": int(retokenized_completion_tokens),
            "completion_tokens_source": (
                "retokenized_max"
                if retokenized_completion_tokens > usage_completion_tokens
                else "usage"
            ),
            "generated_text": answer_text,
        }
    )
    return out


def _direct_request_prompt_token_count(
    *,
    tokenizer: Any,
    prompt_ids: List[int],
    prompt_shape_spec: Mapping[str, Any],
    system_prompt: str,
    max_new_tokens: int,
    model_id: str,
) -> int:
    """Return the full rendered prompt-token count for one direct request."""

    prompt_token_ids = _build_prompt_token_ids_from_shape(
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        prompt_shape_spec=prompt_shape_spec,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        model_id=model_id,
    )
    if prompt_token_ids is not None:
        return int(len(prompt_token_ids))

    prompt_text = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    messages = render_messages(
        prompt_shape_spec,
        _build_prompt_shape_render_context(
            prompt_text=prompt_text,
            system_prompt=str(system_prompt),
            max_new_tokens=int(max_new_tokens),
            prompt_tokens=int(len(prompt_ids)),
            model_id=str(model_id),
        ),
    )
    return int(len(_tokenize_messages(tokenizer, messages)))


def _build_direct_query_messages(
    *,
    tokenizer: Any,
    prompt_ids: List[int],
    prompt_shape_spec: Mapping[str, Any],
    prompt_shape_id: str,
    system_prompt: str,
    max_new_tokens: int,
    model_id: str,
    suffix_query_text: str,
) -> Tuple[List[Dict[str, str]], int]:
    """Build a direct-path prompt that matches restore/query continuation."""

    prompt_text = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prefix_session_payload = _build_prefix_session_payload(
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        prompt_text=prompt_text,
        prompt_shape_spec=prompt_shape_spec,
        prompt_shape_id=str(prompt_shape_id),
        system_prompt=str(system_prompt),
        max_new_tokens=int(max_new_tokens),
        model_id=str(model_id),
    )
    prefix_capture = _normalize_prefix_capture(
        prefix_session_payload.get("prefix_capture")
    )
    if str(prefix_capture.get("continuation_mode") or "") == "same_message":
        prefix_messages = [
            dict(msg)
            for msg in list(prefix_session_payload.get("prefix_messages") or [])
        ]
        if prefix_messages:
            query_messages = _prefix_session_lib.build_same_message_query_messages(
                prefix_messages=prefix_messages,
                suffix_text=str(suffix_query_text),
                message_index=int(
                    prefix_capture.get("message_index") or len(prefix_messages) - 1
                ),
            )
            return query_messages, int(len(_tokenize_messages(tokenizer, query_messages)))

    messages = render_messages(
        prompt_shape_spec,
        _build_prompt_shape_render_context(
            prompt_text=prompt_text,
            system_prompt=str(system_prompt),
            max_new_tokens=int(max_new_tokens),
            prompt_tokens=int(len(prompt_ids)),
            model_id=str(model_id),
        ),
    )
    messages.append({"role": "user", "content": str(suffix_query_text)})
    return messages, int(len(_tokenize_messages(tokenizer, messages)))


def _init_request_debug_payload(
    *,
    max_new_tokens: int,
    dump_request_payload: bool = False,
    dump_query_messages: bool = False,
    dump_prompt_token_ids: bool = False,
) -> Dict[str, Any]:
    """Create one generic live-request debug payload."""

    debug_payload: Dict[str, Any] = {
        "external_request_id": "",
        "internal_request_id": "",
        "layout": {},
        "first_stream_event": {},
        "sent_vllm_xargs": {},
        "dump_request_payload": bool(dump_request_payload),
        "dump_query_messages": bool(dump_query_messages),
        "dump_prompt_token_ids": bool(dump_prompt_token_ids),
    }
    return debug_payload


def _make_live_request_layout_probe(
    *,
    base_url: str,
    timeout_sec: int,
    use_stream: bool,
    debug_payload: Dict[str, Any],
    prefix_token_count: int,
    sent_vllm_xargs: Optional[Mapping[str, Any]] = None,
    prefix_row_hash_context: Optional[Mapping[str, Any]] = None,
) -> Optional[Any]:
    """Return a first-delta callback that records live request layout."""

    if int(prefix_token_count) <= 0 or not bool(use_stream):
        if sent_vllm_xargs:
            debug_payload["sent_vllm_xargs"] = dict(sent_vllm_xargs)
        return None

    if sent_vllm_xargs:
        debug_payload["sent_vllm_xargs"] = dict(sent_vllm_xargs)

    probe_state = {
        "resolve_attempts": 0,
        "resolve_disabled": False,
    }

    def _stream_progress_cb(progress: Mapping[str, Any]) -> None:
        progress_type = str(progress.get("type") or "")
        progress_request_id = str(progress.get("request_id") or "").strip()
        if progress_request_id and not debug_payload["external_request_id"]:
            debug_payload["external_request_id"] = progress_request_id
        if progress_type != "delta":
            return
        if not debug_payload["first_stream_event"]:
            debug_payload["first_stream_event"] = {
                str(key): value for key, value in progress.items()
            }
        if debug_payload["internal_request_id"]:
            return
        if probe_state["resolve_disabled"]:
            return
        probe_state["resolve_attempts"] += 1
        internal_request_id = _resolve_active_internal_request_id(
            base_url=str(base_url),
            external_request_id=str(
                debug_payload["external_request_id"] or progress_request_id or ""
            ),
            timeout_sec=int(timeout_sec),
            retry_total_sec=2.0 if probe_state["resolve_attempts"] == 1 else 0.0,
            poll_interval_sec=0.05,
        )
        if not internal_request_id:
            if probe_state["resolve_attempts"] >= 2:
                probe_state["resolve_disabled"] = True
                debug_payload["layout"] = {
                    "ok": False,
                    "error": (
                        "live_request_layout_probe_disabled:"
                        "internal_request_id_unavailable"
                    ),
                }
            return
        debug_payload["internal_request_id"] = str(internal_request_id)
        try:
            debug_payload["layout"] = _describe_request_prefix_apc_rows(
                base_url=str(base_url),
                request_id=str(internal_request_id),
                prefix_token_count=int(prefix_token_count),
                timeout_sec=int(timeout_sec),
            )
        except Exception as exc:  # noqa: BLE001
            debug_payload["layout"] = {
                "ok": False,
                "error": f"describe_request_prefix_apc_rows_failed:{exc}",
            }
            return

        row_hash_context = dict(prefix_row_hash_context or {})
        if (
            not bool(debug_payload.get("prefix_row_hashes"))
            and bool(row_hash_context.get("enabled"))
            and bool(dict(debug_payload.get("layout") or {}).get("ok"))
        ):
            layout = dict(debug_payload.get("layout") or {})
            try:
                debug_payload["prefix_row_hashes"] = _fetch_query_prefix_row_hashes(
                    base_url=str(base_url),
                    timeout_sec=int(timeout_sec),
                    session_dir=_normalize_prefix_session_debug_root(
                        session_dir=str(row_hash_context.get("session_dir") or ""),
                        session_id=str(row_hash_context.get("session_id") or ""),
                    ),
                    session_id=str(row_hash_context.get("session_id") or ""),
                    linear_target_last_rows=dict(layout.get("linear_target_last_rows") or {}),
                    global_target_rows=[
                        int(value) for value in list(layout.get("global_target_rows") or [])
                    ],
                    row_base_shift=str(row_hash_context.get("row_base_shift") or "auto"),
                )
            except Exception as exc:  # noqa: BLE001
                debug_payload["prefix_row_hashes"] = {
                    "ok": False,
                    "error": f"debug_prefix_state_row_hashes_failed:{exc}",
                }

    return _stream_progress_cb


def _run_prefix_session_query_request(
    *,
    base_url: str,
    model_id: str,
    tokenizer: Any,
    loaded_session: Mapping[str, Any],
    fallback_prefix_messages: List[Dict[str, str]],
    suffix_query_text: str,
    max_new_tokens: int,
    min_new_tokens: int,
    timeout_sec: int,
    ignore_eos: bool,
    use_stream: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    output_mask_payload: Optional[Mapping[str, Any]],
    live_layout_prefix_token_count: int = 0,
) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    """Executes one prefix-session query using the correct continuation mode.

    Args:
        base_url: vLLM OpenAI-compatible server base URL.
        model_id: Target model identifier.
        tokenizer: Hugging Face-compatible tokenizer.
        loaded_session: Loaded prefix-session payload and manifest.
        fallback_prefix_messages: Prefix messages for legacy message-append
            flows.
        suffix_query_text: Suffix text to prefill before generation.
        max_new_tokens: Max completion tokens.
        min_new_tokens: Min completion tokens.
        timeout_sec: Request timeout in seconds.
        ignore_eos: Whether EOS should be ignored during generation.
        use_stream: Whether to use streaming responses.
        temperature: Sampling temperature.
        top_p: Top-p sampling value.
        top_k: Top-k sampling value.
        repetition_penalty: Repetition penalty value.
        presence_penalty: Presence penalty value.
        frequency_penalty: Frequency penalty value.
        output_mask_payload: Optional generation-time output mask payload.

    Returns:
        Tuple[Dict[str, Any], int, Dict[str, Any]]: Request metrics,
        prompt-token count used for the request, and live control-plane
        artifacts captured during the first streamed delta.
    """
    debug_payload = _init_request_debug_payload(max_new_tokens=int(max_new_tokens))
    row_hash_config = _query_row_hash_config()

    manifest = _prefix_session_lib.normalize_prefix_session_manifest(
        dict(loaded_session.get("manifest") or {})
    )
    attention_mode = str(manifest.get("attention_mode") or "")
    if attention_mode == "streaming_mla":
        prefix_capture = _normalize_prefix_capture(manifest.get("prefix_capture"))
        prefix_messages = list(loaded_session.get("prefix_messages") or [])
        query_messages = _prefix_session_lib.build_same_message_query_messages(
            prefix_messages=[dict(msg) for msg in prefix_messages],
            suffix_text=str(suffix_query_text),
            message_index=int(
                prefix_capture.get("message_index") or len(prefix_messages) - 1
            ),
        )
        prompt_token_count = len(_tokenize_messages(tokenizer, query_messages))
        prefix_token_ids = [
            int(x) for x in list(loaded_session.get("prefix_token_ids") or [])
        ]
        prefix_block_size = _prefix_session_lib.resolve_prefix_cache_block_size(
            manifest=manifest,
            session_dir=Path(str(loaded_session.get("session_dir") or "")),
        )
        reuse_plan = _prefix_session_lib.build_hot_prefix_reuse_plan(
            prefix_token_ids=prefix_token_ids,
            block_size_tokens=int(prefix_block_size or 0),
            prefer_full_prefix_reuse=False,
        )
        reused_prefix_token_count = int(
            reuse_plan.get("reused_prefix_token_count") or 0
        )
        vllm_xargs = {"hot_prefix_token_count": int(reused_prefix_token_count)}
        streaming_manager_rows = [
            dict(row)
            for row in list(loaded_session.get("streaming_prepared_manager_rows") or [])
            if isinstance(row, Mapping)
        ]
        if not streaming_manager_rows:
            prepared_payload = dict(loaded_session.get("streaming_prepare_payload") or {})
            if prepared_payload:
                streaming_manager_rows = (
                    _prefix_session_lib.build_streaming_manager_rows_from_prepared_layout(
                        prepared_payload
                    )
                )
        if not streaming_manager_rows:
            streaming_manager_rows = _prefix_session_lib.build_streaming_hot_prefix_manager_rows(
                manifest,
                reused_prefix_token_count=int(reused_prefix_token_count),
            )
        if streaming_manager_rows:
            vllm_xargs["hot_prefix_manager_rows"] = streaming_manager_rows
        prefix_row_hash_context = {
            "enabled": bool(row_hash_config.get("enabled")),
            "session_dir": str(loaded_session.get("session_dir") or ""),
            "session_id": str(
                loaded_session.get("session_id")
                or manifest.get("session_id")
                or manifest.get("id")
                or ""
            ),
            "row_base_shift": str(row_hash_config.get("row_base_shift") or "auto"),
        }
        stream_progress_cb = _make_live_request_layout_probe(
            base_url=str(base_url),
            timeout_sec=int(timeout_sec),
            use_stream=bool(use_stream),
            debug_payload=debug_payload,
            prefix_token_count=max(
                int(live_layout_prefix_token_count or 0),
                int(reused_prefix_token_count),
            ),
            sent_vllm_xargs=vllm_xargs,
            prefix_row_hash_context=prefix_row_hash_context,
        )
        metrics = _run_one_request_messages(
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            messages=query_messages,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            timeout_sec=timeout_sec,
            ignore_eos=ignore_eos,
            use_stream=use_stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            output_mask_payload=output_mask_payload,
            vllm_xargs=vllm_xargs,
            stream_progress_cb=stream_progress_cb,
        )
        return metrics, int(prompt_token_count), debug_payload
    prefix_capture = _normalize_prefix_capture(manifest.get("prefix_capture"))
    if str(prefix_capture.get("continuation_mode") or "") == "same_message":
        prefix_messages = list(loaded_session.get("prefix_messages") or [])
        if prefix_messages:
            query_messages = _prefix_session_lib.build_same_message_query_messages(
                prefix_messages=[dict(msg) for msg in prefix_messages],
                suffix_text=suffix_query_text,
                message_index=int(prefix_capture.get("message_index") or len(prefix_messages) - 1),
            )
            prompt_tokens = len(_tokenize_messages(tokenizer, query_messages))
            metrics = _run_one_request_messages(
                base_url=base_url,
                model_id=model_id,
                tokenizer=tokenizer,
                messages=query_messages,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                timeout_sec=timeout_sec,
                ignore_eos=ignore_eos,
                use_stream=use_stream,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                output_mask_payload=output_mask_payload,
                stream_progress_cb=_make_live_request_layout_probe(
                    base_url=str(base_url),
                    timeout_sec=int(timeout_sec),
                    use_stream=bool(use_stream),
                    debug_payload=debug_payload,
                    prefix_token_count=int(live_layout_prefix_token_count or 0),
                    sent_vllm_xargs=None,
                    prefix_row_hash_context={
                        "enabled": bool(row_hash_config.get("enabled")),
                        "session_dir": str(loaded_session.get("session_dir") or ""),
                        "session_id": str(
                            loaded_session.get("session_id")
                            or manifest.get("session_id")
                            or manifest.get("id")
                            or ""
                        ),
                        "row_base_shift": str(
                            row_hash_config.get("row_base_shift") or "auto"
                        ),
                    },
                ),
            )
            return metrics, int(prompt_tokens), debug_payload
        prompt_token_ids, _suffix_token_ids = _build_same_message_query_prompt_token_ids(
            tokenizer=tokenizer,
            prefix_token_ids=[int(x) for x in list(loaded_session.get("prefix_token_ids") or [])],
            suffix_text=suffix_query_text,
            continuation_tail_token_ids=[
                int(x) for x in list(loaded_session.get("continuation_tail_token_ids") or [])
            ],
        )
        metrics = _run_one_request_completion_prompt_ids(
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            timeout_sec=timeout_sec,
            ignore_eos=ignore_eos,
            use_stream=use_stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            output_mask_payload=output_mask_payload,
            stream_progress_cb=_make_live_request_layout_probe(
                base_url=str(base_url),
                timeout_sec=int(timeout_sec),
                use_stream=bool(use_stream),
                debug_payload=debug_payload,
                prefix_token_count=int(live_layout_prefix_token_count or 0),
                sent_vllm_xargs=None,
                prefix_row_hash_context={
                    "enabled": bool(row_hash_config.get("enabled")),
                    "session_dir": str(loaded_session.get("session_dir") or ""),
                    "session_id": str(
                        loaded_session.get("session_id")
                        or manifest.get("session_id")
                        or manifest.get("id")
                        or ""
                    ),
                    "row_base_shift": str(row_hash_config.get("row_base_shift") or "auto"),
                },
            ),
        )
        return metrics, int(len(prompt_token_ids)), debug_payload

    messages = [dict(msg) for msg in list(loaded_session.get("prefix_messages") or fallback_prefix_messages)]
    messages.append({"role": "user", "content": str(suffix_query_text)})
    prompt_tokens = len(_tokenize_messages(tokenizer, messages))
    metrics = _run_one_request_messages(
        base_url=base_url,
        model_id=model_id,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        timeout_sec=timeout_sec,
        ignore_eos=ignore_eos,
        use_stream=use_stream,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        output_mask_payload=output_mask_payload,
        stream_progress_cb=_make_live_request_layout_probe(
            base_url=str(base_url),
            timeout_sec=int(timeout_sec),
            use_stream=bool(use_stream),
            debug_payload=debug_payload,
            prefix_token_count=int(live_layout_prefix_token_count or 0),
            sent_vllm_xargs=None,
            prefix_row_hash_context={
                "enabled": bool(row_hash_config.get("enabled")),
                "session_dir": str(loaded_session.get("session_dir") or ""),
                "session_id": str(
                    loaded_session.get("session_id")
                    or manifest.get("session_id")
                    or manifest.get("id")
                    or ""
                ),
                "row_base_shift": str(row_hash_config.get("row_base_shift") or "auto"),
            },
        ),
    )
    return metrics, int(prompt_tokens), debug_payload


def _run_k2ft_probe(
    *,
    prompt_seed_ids: List[int],
    target_effective: int,
    delta_tokens: int,
    min_probe_tokens: int,
    k2ft_runs: int,
    base_url: str,
    model_id: str,
    tokenizer: Any,
    prompt_shape_spec: Mapping[str, Any],
    system_prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    timeout_sec: int,
    ignore_eos: bool,
    use_stream: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    output_mask_payload: Optional[Mapping[str, Any]],
    query_api_mode: str,
    timeout_sclk_guard: bool,
    timeout_sclk_min_mhz: float,
    timeout_sclk_sample_count: int,
    timeout_sclk_sample_interval_sec: float,
    timeout_sclk_min_busy_gpus: int,
    timeout_sclk_required_hit_ratio: float,
    timeout_sclk_max_extensions: int,
    timeout_sclk_extension_sec: int,
    timeout_sclk_expected_gpu_count: int,
) -> Dict[str, Any]:
    """Estimate KV-ready-to-first-token latency via cold/hot probe TTFT."""
    probe_floor = max(1, int(min_probe_tokens))
    probe_delta = max(0, int(delta_tokens))
    probe_tokens = min(len(prompt_seed_ids), max(probe_floor, int(target_effective) - probe_delta))
    formula = f"max({probe_floor}, L_current({int(target_effective)}) - {probe_delta})"
    out: Dict[str, Any] = {
        "k2ft_enabled": True,
        "k2ft_method": "two_stage_cold_hot_ttft_delta",
        "k2ft_probe_formula": formula,
        "k2ft_probe_tokens": int(probe_tokens),
        "k2ft_runs_configured": max(2, int(k2ft_runs)),
        "k2ft_runs_completed": 0,
        "k2ft_cold_ttft_ms": 0.0,
        "k2ft_hot_ttft_ms": 0.0,
        "k2ft_ms": 0.0,
        "prefill_est_ms": 0.0,
        "k2ft_cached_prompt_tokens_cold": 0,
        "k2ft_cached_prompt_tokens_hot": 0,
        "k2ft_cache_hit_confidence": "low",
        "k2ft_status": "error",
        "k2ft_error": "",
    }
    if probe_tokens <= 0:
        out["k2ft_error"] = "k2ft_probe_tokens<=0"
        return out

    probe_ids = prompt_seed_ids[:probe_tokens]
    probe_count = max(2, int(k2ft_runs))
    probe_metrics: List[Dict[str, Any]] = []

    for _ in range(probe_count):
        try:
            metrics, _timeout_guard_info = _run_one_request_with_timeout_guard(
                base_url=base_url,
                model_id=model_id,
                tokenizer=tokenizer,
                prompt_ids=probe_ids,
                prompt_shape_spec=prompt_shape_spec,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                timeout_sec=timeout_sec,
                ignore_eos=ignore_eos,
                use_stream=use_stream,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                output_mask_payload=output_mask_payload,
                query_api_mode=query_api_mode,
                timeout_sclk_guard=timeout_sclk_guard,
                timeout_sclk_min_mhz=timeout_sclk_min_mhz,
                timeout_sclk_sample_count=timeout_sclk_sample_count,
                timeout_sclk_sample_interval_sec=timeout_sclk_sample_interval_sec,
                timeout_sclk_min_busy_gpus=timeout_sclk_min_busy_gpus,
                timeout_sclk_required_hit_ratio=timeout_sclk_required_hit_ratio,
                timeout_sclk_max_extensions=timeout_sclk_max_extensions,
                timeout_sclk_extension_sec=timeout_sclk_extension_sec,
                timeout_sclk_expected_gpu_count=timeout_sclk_expected_gpu_count,
            )
            probe_metrics.append(metrics)
            out["k2ft_runs_completed"] = int(len(probe_metrics))
        except Exception as exc:  # noqa: BLE001
            out["k2ft_error"] = str(exc)[:3000]
            break

    if len(probe_metrics) >= 2:
        cold_ttft = _as_float(probe_metrics[0].get("ttft_ms"))
        hot_ttft = _as_float(probe_metrics[-1].get("ttft_ms"))
        cold_cached = int(probe_metrics[0].get("usage_cached_prompt_tokens") or 0)
        hot_cached = int(probe_metrics[-1].get("usage_cached_prompt_tokens") or 0)
        out["k2ft_cold_ttft_ms"] = round(float(cold_ttft), 6)
        out["k2ft_hot_ttft_ms"] = round(float(hot_ttft), 6)
        out["k2ft_ms"] = round(float(hot_ttft), 6)
        out["prefill_est_ms"] = round(max(0.0, float(cold_ttft) - float(hot_ttft)), 6)
        out["k2ft_cached_prompt_tokens_cold"] = int(cold_cached)
        out["k2ft_cached_prompt_tokens_hot"] = int(hot_cached)
        ratio = (float(hot_ttft) / float(cold_ttft)) if cold_ttft > 0 else 1.0
        if hot_cached > 0 or ratio <= 0.6:
            out["k2ft_cache_hit_confidence"] = "high"
        else:
            out["k2ft_cache_hit_confidence"] = "low"
        out["k2ft_status"] = "ok"
        return out

    if len(probe_metrics) == 1:
        only_ttft = _as_float(probe_metrics[0].get("ttft_ms"))
        out["k2ft_cold_ttft_ms"] = round(float(only_ttft), 6)
        out["k2ft_status"] = "partial_error"
        if not out["k2ft_error"]:
            out["k2ft_error"] = "k2ft_hot_probe_missing"
        return out

    if not out["k2ft_error"]:
        out["k2ft_error"] = "k2ft_probe_failed"
    return out


def _run_one_request_with_timeout_guard(
    *,
    base_url: str,
    model_id: str,
    tokenizer: Any,
    prompt_ids: List[int],
    prompt_shape_spec: Mapping[str, Any],
    system_prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    timeout_sec: int,
    ignore_eos: bool,
    use_stream: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    output_mask_payload: Optional[Mapping[str, Any]],
    timeout_sclk_guard: bool,
    timeout_sclk_min_mhz: float,
    timeout_sclk_sample_count: int,
    timeout_sclk_sample_interval_sec: float,
    timeout_sclk_min_busy_gpus: int,
    timeout_sclk_required_hit_ratio: float,
    timeout_sclk_max_extensions: int,
    timeout_sclk_extension_sec: int,
    timeout_sclk_expected_gpu_count: int,
    messages_override: Optional[List[Dict[str, str]]] = None,
    query_api_mode: str = "auto",
    prompt_token_ids_override: Optional[List[int]] = None,
    request_debug_payload: Optional[Dict[str, Any]] = None,
    stream_progress_cb: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    timeout_now = max(1, int(timeout_sec))
    extension_sec = max(1, int(timeout_sclk_extension_sec))
    max_extensions = max(0, int(timeout_sclk_max_extensions))
    attempts = 0
    extensions = 0
    events: List[Dict[str, Any]] = []

    while True:
        attempts += 1
        try:
            metrics = _run_one_request(
                base_url=base_url,
                model_id=model_id,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                prompt_shape_spec=prompt_shape_spec,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                timeout_sec=timeout_now,
                ignore_eos=ignore_eos,
                use_stream=use_stream,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                output_mask_payload=output_mask_payload,
                messages_override=messages_override,
                query_api_mode=query_api_mode,
                prompt_token_ids_override=prompt_token_ids_override,
                request_debug_payload=request_debug_payload,
                stream_progress_cb=stream_progress_cb,
            )
            info = {
                "attempts": int(attempts),
                "extensions": int(extensions),
                "final_timeout_sec": int(timeout_now),
                "events": events,
            }
            return metrics, info
        except Exception as exc:  # noqa: BLE001
            exc_info = {
                "attempts": int(attempts),
                "extensions": int(extensions),
                "final_timeout_sec": int(timeout_now),
                "events": events,
            }
            if (not timeout_sclk_guard) or (not _is_timeout_exception(exc)):
                setattr(exc, "timeout_guard_info", exc_info)
                raise
            if extensions >= max_extensions:
                setattr(exc, "timeout_guard_info", exc_info)
                raise

            sample = _sample_sclk_busy(
                expected_gpu_count=max(1, int(timeout_sclk_expected_gpu_count)),
                min_sclk_mhz=float(timeout_sclk_min_mhz),
                sample_count=int(timeout_sclk_sample_count),
                sample_interval_sec=float(timeout_sclk_sample_interval_sec),
                min_busy_gpus=int(timeout_sclk_min_busy_gpus),
                required_hit_ratio=float(timeout_sclk_required_hit_ratio),
            )
            event = {
                "attempt": int(attempts),
                "timeout_sec_before": int(timeout_now),
                "sclk_busy": bool(sample.get("busy", False)),
                "sclk_hits": int(sample.get("hits", 0)),
                "sclk_required_hits": int(sample.get("required_hits", 0)),
                "sclk_collector": str(sample.get("collector") or "none"),
                "sclk_min_busy_gpus": int(sample.get("min_busy_gpus", 0)),
                "sclk_max_busy_gpu_count": int(sample.get("max_busy_gpu_count", 0)),
                "sclk_max_peak_mhz": round(_as_float(sample.get("max_peak_sclk_mhz")), 6),
                "sclk_samples": list(sample.get("samples") or []),
                "action": "terminate",
            }
            if bool(sample.get("busy", False)):
                timeout_now += extension_sec
                extensions += 1
                event["action"] = "extend_timeout"
                event["timeout_sec_after"] = int(timeout_now)
                events.append(event)
                continue
            events.append(event)
            setattr(
                exc,
                "timeout_guard_info",
                {
                    "attempts": int(attempts),
                    "extensions": int(extensions),
                    "final_timeout_sec": int(timeout_now),
                    "events": events,
                },
            )
            raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM long-context TTFT/decode throughput")
    parser.add_argument("--model-id", default="moonshotai/Kimi-Linear-48B-A3B-Instruct")
    parser.add_argument("--tokenizer-id", default="")
    parser.add_argument(
        "--query-api-mode",
        choices=("auto", "messages", "prompt_ids"),
        default="auto",
        help=(
            "Force the direct-query request path. 'auto' preserves the current "
            "behavior, which prefers prompt-token replay when available."
        ),
    )
    parser.add_argument("--vllm-bin", default="/tmp/vllm015_rocmsrc/bin/vllm")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=18090)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--vllm-data-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-max-model-len", type=int, default=8389120)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--vllm-load-format", default="")
    parser.add_argument("--vllm-hf-config-path", default="")
    parser.add_argument("--vllm-compilation-config", default="")
    parser.add_argument(
        "--vllm-mla-runtime-mode",
        choices=("aiter",),
        default="aiter",
        help="Kimi MLA decode runtime mode. Only the stock ROCm AITER path is supported.",
    )
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-worker-extension-cls", default="")
    parser.add_argument("--vllm-enable-prefix-caching", action="store_true")
    parser.add_argument("--vllm-disable-custom-all-reduce", action="store_true")
    parser.add_argument(
        "--disable-rocm-skinny-gemm",
        action="store_true",
        help=(
            "Set VLLM_ROCM_USE_SKINNY_GEMM=0 for the launched vLLM server "
            "and record that policy in run artifacts."
        ),
    )
    parser.add_argument("--calculate-kv-scales", action="store_true")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=0)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--vllm-attention-backend", default="")
    parser.add_argument(
        "--vllm-batch-invariant",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Set VLLM_BATCH_INVARIANT for the launched vLLM server. "
            "Use 1 to force batch-shape-invariant FlashAttention behavior."
        ),
    )
    parser.add_argument(
        "--vllm-preserve-legacy-aiter-env",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Preserve the older env-driven AITER head4/sub16/mla stage knobs "
            "for the launched vLLM server. Use 1 only when intentionally "
            "running the historical snapshot-backed AITER head4 path under "
            "native v0.19."
        ),
    )
    parser.add_argument("--vllm-block-size", type=int, default=0)
    parser.add_argument("--vllm-kv-cache-memory-bytes", type=int, default=0)
    parser.add_argument("--vllm-num-gpu-blocks-override", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--rocprof-enable",
        action="store_true",
        default=_env_flag("HFLC_FP8_ROCPROF_ENABLE", False),
        help="Wrap the launched vLLM server process with rocprof.",
    )
    parser.add_argument(
        "--rocprof-mode",
        default=str(os.environ.get("HFLC_FP8_ROCPROF_MODE", "auto") or "auto"),
        choices=["auto", "none", "sys", "hip", "hsa"],
        help="rocprof trace mode for the wrapped server process.",
    )
    parser.add_argument(
        "--rocprof-stats",
        type=int,
        choices=[0, 1],
        default=1 if _env_flag("HFLC_FP8_ROCPROF_STATS", True) else 0,
        help="Emit rocprof stats artifacts in addition to traces.",
    )
    parser.add_argument(
        "--rocprof-timestamp",
        type=int,
        choices=[0, 1],
        default=1 if _env_flag("HFLC_FP8_ROCPROF_TIMESTAMP", True) else 0,
        help="Enable rocprof kernel timestamp capture.",
    )
    parser.add_argument(
        "--rocprof-tool-version",
        type=int,
        choices=[0, 1, 2],
        default=int(str(os.environ.get("HFLC_FP8_ROCPROF_TOOL_VERSION", "0") or "0")),
        help="Optional rocprof tool version override; 0 keeps the default.",
    )
    parser.add_argument(
        "--rocprof-input",
        default=str(os.environ.get("HFLC_FP8_ROCPROF_INPUT", "") or ""),
        help="Optional rocprof input file for counter or trace configuration.",
    )
    parser.add_argument(
        "--rocprof-output-basename",
        default=str(os.environ.get("HFLC_FP8_ROCPROF_OUTPUT_BASENAME", "server_rocprof") or "server_rocprof"),
        help="Base filename for rocprof artifacts inside each run directory.",
    )
    parser.add_argument("--server-ready-timeout-sec", type=int, default=1800)
    parser.add_argument("--request-timeout-sec", type=int, default=14400, help="Base per-request timeout in seconds.")
    parser.add_argument(
        "--adaptive-request-timeout",
        action="store_true",
        help="Estimate next timeout from previous successful TTFT using quadratic token scaling.",
    )
    parser.add_argument(
        "--adaptive-timeout-scale",
        type=float,
        default=1.25,
        help="Multiplier applied to predicted TTFT when adaptive timeout is enabled.",
    )
    parser.add_argument(
        "--adaptive-timeout-extra-sec",
        type=float,
        default=30.0,
        help="Fixed slack seconds added to adaptive timeout prediction.",
    )
    parser.add_argument(
        "--adaptive-timeout-cap-sec",
        type=int,
        default=0,
        help="Optional hard cap for adaptive timeout (0 disables cap).",
    )
    parser.add_argument(
        "--timeout-sclk-guard",
        action="store_true",
        help="On timeout, sample GPU sclk before failing; extend timeout while GPUs stay busy.",
    )
    parser.add_argument(
        "--timeout-sclk-min-mhz",
        type=float,
        default=500.0,
        help="Per-GPU sclk threshold (MHz) that counts as busy for timeout guard.",
    )
    parser.add_argument(
        "--timeout-sclk-sample-count",
        type=int,
        default=6,
        help="Number of sclk samples to collect after timeout.",
    )
    parser.add_argument(
        "--timeout-sclk-sample-interval-sec",
        type=float,
        default=5.0,
        help="Seconds between successive sclk samples.",
    )
    parser.add_argument(
        "--timeout-sclk-min-busy-gpus",
        type=int,
        default=0,
        help="Minimum GPUs above sclk threshold to consider busy (0 uses expected world size).",
    )
    parser.add_argument(
        "--timeout-sclk-required-hit-ratio",
        type=float,
        default=0.5,
        help="Required fraction of samples that must satisfy busy-gpu threshold.",
    )
    parser.add_argument(
        "--timeout-sclk-max-extensions",
        type=int,
        default=3,
        help="Maximum timeout extensions allowed after timeout+sclk-busy checks.",
    )
    parser.add_argument(
        "--timeout-sclk-extension-sec",
        type=int,
        default=120,
        help="Seconds to extend timeout per successful sclk-busy guard check.",
    )
    parser.add_argument("--no-stream", action="store_true")

    parser.add_argument("--prompt-source", choices=["substrate"], default="substrate")
    parser.add_argument("--system-prompt", default="You are a coding assistant.")
    parser.add_argument(
        "--prompt-shape",
        choices=[
            "benchmark_v1",
            "repo_grounded_en_v1",
            "repo_grounded_en_v2",
            "repo_grounded_universal_v1",
            "custom_file",
        ],
        default="benchmark_v1",
    )
    parser.add_argument("--prompt-shape-file", default="")
    parser.add_argument(
        "--prompt-token-ids-file",
        default="",
        help=(
            "Optional JSON artifact containing the exact prompt token IDs to "
            "send to vLLM. This bypasses substrate prompt construction and "
            "lets debug runs hold the prompt fixed while varying scheduler "
            "settings such as max_num_batched_tokens."
        ),
    )
    parser.add_argument(
        "--prefix-session-mode",
        choices=["none", "build", "restore", "query", "build_and_query"],
        default="none",
    )
    parser.add_argument("--prefix-session-id", default="")
    parser.add_argument("--prefix-session-dir", default="")
    parser.add_argument("--prefix-compat-strict", type=int, choices=[0, 1], default=1)
    parser.add_argument("--suffix-query-file", default="")
    parser.add_argument("--suffix-query-files", default="")
    parser.add_argument("--suffix-query-dir", default="")
    parser.add_argument("--prefix-query-max-count", type=int, default=0)
    parser.add_argument(
        "--prefix-reload-between-queries",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--prefix-max-tokens", type=int, default=0)
    parser.add_argument("--prefix-save-shards", type=int, choices=[0, 1], default=1)
    parser.add_argument("--prefix-target-tp", type=int, default=0)
    parser.add_argument("--prefix-state-chunk-bytes", type=int, default=67108864)
    parser.add_argument(
        "--prefix-state-calibration-strict-gate",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--prefix-state-calibration-safety-ratio",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--prefix-state-calibration-safety-min-bytes",
        type=int,
        default=2147483648,
    )
    parser.add_argument("--target-prompt-tokens", type=int, default=32768)
    add_common_sweep_args(parser)
    parser.add_argument("--extra-prompt-tokens", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument(
        "--english-only-mask-mode",
        choices=("off", "bias", "hard"),
        default="hard",
    )
    parser.add_argument("--english-only-mask-bias", type=float, default=100.0)
    parser.add_argument(
        "--dump-request-payload",
        action="store_true",
        help="Persist the exact OpenAI-compatible request payload for each measured query.",
    )
    parser.add_argument(
        "--dump-query-messages",
        action="store_true",
        help="Persist the final rendered query messages for each measured query.",
    )
    parser.add_argument(
        "--dump-prompt-token-ids",
        action="store_true",
        help="Persist prompt token IDs plus a short decoded preview for each measured query.",
    )

    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--warmup-discard-requests",
        type=int,
        default=1,
        help=(
            "Untimed warm-up requests issued and discarded before the first "
            "measured iteration at each point, to absorb the one-time CUDA-graph "
            "capture/compile cost for a new prompt-token shape so the measured "
            "TTFT reflects steady state. 0 disables."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-generated-text", action="store_true")
    parser.add_argument("--calibrate-kv-capacity", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--calibration-strict-gate", action="store_true")
    parser.add_argument("--calibration-headroom-ratio", type=float, default=0.005)
    parser.add_argument("--calibration-headroom-min-blocks", type=int, default=2)
    parser.add_argument("--calibration-assumed-max-num-seqs", type=int, default=0)
    parser.add_argument("--calibration-metrics-timeout-sec", type=int, default=180)
    parser.add_argument("--measure-k2ft", action="store_true")
    parser.add_argument("--k2ft-delta-tokens", type=int, default=256)
    parser.add_argument("--k2ft-min-probe-tokens", type=int, default=1024)
    parser.add_argument("--k2ft-runs", type=int, default=2)
    parser.add_argument(
        "--emit-prefill-progress",
        action="store_true",
        help="Poll /debug/prefill_progress during each measured request and emit prefill_progress.jsonl.",
    )
    parser.add_argument(
        "--prefill-progress-interval-sec",
        type=float,
        default=20.0,
        help="Polling interval for /debug/prefill_progress.",
    )
    parser.add_argument(
        "--prefill-progress-timeout-sec",
        type=int,
        default=0,
        help="Endpoint poll timeout seconds (0 means use request-timeout-sec).",
    )

    parser.add_argument("--substrate-root", default=str(repo_path("data", "substrate")))
    parser.add_argument("--substrate-include-exts", default="")
    parser.add_argument(
        "--substrate-max-files",
        type=int,
        default=0,
        help="Maximum number of substrate files to read (0 means all matching files).",
    )
    parser.add_argument("--substrate-max-chars-per-file", type=int, default=200000)
    parser.add_argument(
        "--substrate-char-budget-multiplier",
        type=float,
        default=12.0,
        help="Character budget factor applied to target prompt tokens when building substrate prompt seed.",
    )

    parser.add_argument("--plot-png", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--post-run-cache-cleanup",
        choices=["none", "triton", "all"],
        default="none",
        help="Remove selected per-run compile-cache directories after the server stops.",
    )
    return parser.parse_args()


def _read_suffix_query_entries(
    *,
    suffix_query_file: str,
    suffix_query_files: str,
    suffix_query_dir: str,
    prefix_query_max_count: int,
    prompt_shape_id: str,
    max_new_tokens: int,
) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    seen_paths: set[str] = set()

    def _append_from_path(path: Path) -> None:
        resolved = str(path.expanduser().resolve())
        if resolved in seen_paths:
            return
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"suffix query file not found: {resolved}")
        text = path.read_text(encoding="utf-8")
        seen_paths.add(resolved)
        entries.append({"text": text, "path": resolved})

    single = str(suffix_query_file or "").strip()
    if single:
        _append_from_path(Path(single))

    files_raw = str(suffix_query_files or "").strip()
    if files_raw:
        for item in [x.strip() for x in files_raw.split(",") if x.strip()]:
            _append_from_path(Path(item))

    query_dir_raw = str(suffix_query_dir or "").strip()
    if query_dir_raw:
        query_dir = Path(query_dir_raw).expanduser().resolve()
        if not query_dir.exists() or not query_dir.is_dir():
            raise RuntimeError(f"suffix query dir not found: {query_dir}")
        for file_path in sorted([p for p in query_dir.iterdir() if p.is_file()]):
            _append_from_path(file_path)

    if not entries:
        entries.append(
            {
                "text": _default_suffix_query(prompt_shape_id, int(max_new_tokens)),
                "path": "",
            }
        )

    max_count = int(prefix_query_max_count or 0)
    if max_count > 0:
        entries = entries[:max_count]
    return entries


def _maybe_run_kv_calibration(
    *,
    args: argparse.Namespace,
    run_id: str,
    out_dir: Path,
    run_meta: Dict[str, Any],
    log_path: Path,
    sweep_targets_requested: List[int],
    prompt_seed_len: int,
    prefill_progress_path: Optional[Path] = None,
) -> Optional[int]:
    calibration_enabled = bool(
        args.calibrate_kv_capacity or args.calibration_only or args.calibration_strict_gate
    )
    if not calibration_enabled:
        return None

    startup_metrics = wait_for_kv_startup_metrics(
        log_path,
        timeout_sec=float(args.calibration_metrics_timeout_sec),
        poll_interval_sec=0.5,
    )
    calibration_prompt_cap = int(prompt_seed_len)
    calibration_targets_effective = sorted(
        {
            min(int(t), int(calibration_prompt_cap))
            for t in sweep_targets_requested
            if int(t) > 0
        }
    )
    assumed_max_num_seqs = int(args.calibration_assumed_max_num_seqs or 0)
    if assumed_max_num_seqs <= 0:
        assumed_max_num_seqs = int(args.vllm_max_num_seqs or 0)
    if assumed_max_num_seqs <= 0:
        # Conservative default for this benchmark workflow: one long request at a time.
        assumed_max_num_seqs = 1

    calibration = build_kv_capacity_calibration(
        target_prompt_tokens=[int(x) for x in calibration_targets_effective],
        max_new_tokens=int(args.max_new_tokens),
        max_num_seqs=int(assumed_max_num_seqs),
        startup_metrics=startup_metrics,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        headroom_ratio=float(args.calibration_headroom_ratio),
        headroom_min_blocks=int(args.calibration_headroom_min_blocks),
    )
    calibration["enabled"] = True
    calibration["strict_gate"] = bool(args.calibration_strict_gate)
    calibration["calibration_only"] = bool(args.calibration_only)
    calibration["source_log"] = str(log_path)
    calibration["target_prompt_tokens_requested"] = [int(x) for x in sweep_targets_requested]
    calibration["target_prompt_tokens_effective"] = [int(x) for x in calibration_targets_effective]
    calibration["max_model_len"] = int(args.vllm_max_model_len)
    calibration["max_new_tokens"] = int(args.max_new_tokens)
    calibration["assumed_max_num_seqs"] = int(assumed_max_num_seqs)

    calibration_json_path = out_dir / "kv_calibration.json"
    calibration_md_path = out_dir / "kv_calibration.md"
    calibration_json_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_md_path.write_text(
        render_kv_calibration_markdown(calibration),
        encoding="utf-8",
    )

    run_meta["kv_calibration"] = {
        "enabled": True,
        "strict_gate": bool(args.calibration_strict_gate),
        "calibration_only": bool(args.calibration_only),
        "json_path": str(calibration_json_path),
        "markdown_path": str(calibration_md_path),
        "assumed_max_num_seqs": int(assumed_max_num_seqs),
        "all_pass": bool((calibration.get("summary") or {}).get("all_pass", False)),
    }
    run_meta["host_env"] = collect_host_env()
    run_meta["mla_kernel_fingerprint"] = collect_mla_kernel_fingerprint()
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fail_count = int((calibration.get("summary") or {}).get("fail_count", 0))
    if bool(args.calibration_only):
        if fail_count > 0:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "run_id": run_id,
                        "out_dir": str(out_dir),
                        "calibration": calibration,
                        "error": "kv_calibration_failed",
                    },
                    sort_keys=True,
                )
            )
            return 2
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_id": run_id,
                    "out_dir": str(out_dir),
                    "calibration": calibration,
                },
                sort_keys=True,
            )
        )
        return 0

    if bool(args.calibration_strict_gate) and fail_count > 0:
        first_fail = next(
            (r for r in (calibration.get("rows") or []) if not bool(r.get("pass"))),
            {},
        )
        if (
            bool(args.emit_prefill_progress)
            and prefill_progress_path is not None
        ):
            fail_rows = [r for r in (calibration.get("rows") or []) if not bool(r.get("pass"))]
            if not fail_rows:
                fail_rows = [first_fail]
            for fail in fail_rows:
                target_prompt = int(fail.get("target_prompt_tokens") or 0)
                _append_jsonl_row(
                    prefill_progress_path,
                    {
                        "timestamp_utc": _now_iso(),
                        "target_index": 0,
                        "target_prompt_tokens_requested": target_prompt,
                        "target_prompt_tokens_effective": target_prompt,
                        "iteration": 0,
                        "request_id": "",
                        "prefill_done_tokens": 0,
                        "prefill_total_tokens": target_prompt,
                        "prefill_pct": 0.0,
                        "engine_step": 0,
                        "source": "vllm_prefill_progress_endpoint",
                        "status": "error",
                        "note": (
                            "calibration_gate_failed_before_measurement:"
                            f" margin_blocks={int(fail.get('margin_blocks') or 0)}"
                            f" recommended_action={str(fail.get('recommended_action') or '')}"
                        )[:500],
                    },
                )

        raise SystemExit(
            "KV calibration gate failed before measurement: "
            f"target_prompt_tokens={int(first_fail.get('target_prompt_tokens') or 0)} "
            f"required_blocks={int(first_fail.get('blocks_required') or 0)} "
            f"available_blocks={int(first_fail.get('blocks_avail') or 0)} "
            f"margin_blocks={int(first_fail.get('margin_blocks') or 0)}. "
            f"Recommended action: {str(first_fail.get('recommended_action') or 'n/a')} "
            f"({str(first_fail.get('recommended_message') or '')}). "
            f"See {calibration_json_path}."
        )
    return None


def _run_prefix_session_mode(
    *,
    args: argparse.Namespace,
    run_id: str,
    out_dir: Path,
    run_meta: Dict[str, Any],
    prompt_shape_spec: Mapping[str, Any],
    prompt_shape_meta: Mapping[str, Any],
    tokenizer: Any,
    prompt_seed_ids: List[int],
    sweep_targets: List[int],
    output_mask_payload: Optional[Mapping[str, Any]],
) -> int:
    """Runs the dedicated prefix-session workflow for build/restore/query modes.

    Args:
        args: Parsed CLI arguments.
        run_id: Stable run identifier.
        out_dir: Output directory for artifacts.
        run_meta: Mutable run metadata payload.
        prompt_shape_spec: Loaded prompt-shape specification.
        prompt_shape_meta: Prompt-shape metadata and hashes.
        tokenizer: Hugging Face-compatible tokenizer.
        prompt_seed_ids: Full context seed token IDs.
        sweep_targets: Requested sweep target token counts.
        output_mask_payload: Optional generation-time output mask payload.

    Returns:
        int: Process exit code.
    """
    mode = str(args.prefix_session_mode or "none")
    session_root = (
        Path(str(args.prefix_session_dir)).expanduser().resolve()
        if str(args.prefix_session_dir or "").strip()
        else (out_dir / "prefix_sessions")
    )
    session_root.mkdir(parents=True, exist_ok=True)
    session_id = str(args.prefix_session_id or f"{run_id}_prefix")
    strict_env_inputs = _strict_env_payload(
        vllm_hf_overrides=dict((run_meta.get("vllm") or {}).get("hf_overrides") or {})
    )
    strict_env_hash = _strict_env_hash(
        vllm_hf_overrides=dict((run_meta.get("vllm") or {}).get("hf_overrides") or {})
    )
    run_meta["strict_env_hash"] = str(strict_env_hash)
    run_meta["strict_env_inputs"] = dict(strict_env_inputs)

    prefix_target = max(1, min(int(args.target_prompt_tokens), len(prompt_seed_ids)))
    prefix_ids = prompt_seed_ids[:prefix_target]
    if int(args.prefix_max_tokens or 0) > 0 and len(prefix_ids) > int(args.prefix_max_tokens):
        raise SystemExit(
            f"prefix_max_tokens exceeded: prefix_tokens={len(prefix_ids)} limit={int(args.prefix_max_tokens)}"
        )
    prefix_text = tokenizer.decode(
        prefix_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prefix_session_payload = _build_prefix_session_payload(
        tokenizer=tokenizer,
        prompt_ids=prefix_ids,
        prompt_text=prefix_text,
        prompt_shape_spec=prompt_shape_spec,
        prompt_shape_id=str(prompt_shape_meta.get("id") or str(args.prompt_shape)),
        system_prompt=str(args.system_prompt),
        max_new_tokens=int(args.max_new_tokens),
        model_id=str(args.model_id),
    )
    prefix_messages = [dict(msg) for msg in list(prefix_session_payload.get("prefix_messages") or [])]
    prefix_token_ids = [int(x) for x in list(prefix_session_payload.get("prefix_token_ids") or [])]
    continuation_tail_token_ids = [
        int(x) for x in list(prefix_session_payload.get("continuation_tail_token_ids") or [])
    ]

    proc: Optional[subprocess.Popen[Any]] = None
    query_rows: List[Dict[str, Any]] = []
    prefill_progress_path = out_dir / "prefill_progress.jsonl"
    if bool(args.emit_prefill_progress) and prefill_progress_path.exists():
        prefill_progress_path.unlink()
    if bool(args.emit_prefill_progress):
        _append_jsonl_row(
            prefill_progress_path,
            {
                "timestamp_utc": _now_iso(),
                "target_index": 0,
                "target_prompt_tokens_requested": 0,
                "target_prompt_tokens_effective": 0,
                "iteration": 0,
                "request_id": "",
                "prefill_done_tokens": 0,
                "prefill_total_tokens": 0,
                "prefill_pct": 0.0,
                "engine_step": 0,
                "source": "vllm_prefill_progress_endpoint",
                "status": "initialized",
                "note": "prefix_session_mode_initialized",
            },
        )

    # Prefix-state shard capture now persists the native APC cache layout, so
    # we must have APC enabled during any build that saves shards.
    if (
        mode in {"build", "build_and_query"}
        and bool(int(args.prefix_save_shards))
        and not bool(getattr(args, "vllm_enable_prefix_caching", False))
    ):
        args.vllm_enable_prefix_caching = True

    # Query-mode restored-prefix reuse also depends on APC being enabled so
    # the restored session can be registered with vLLM's prefix-cache index.
    if (
        mode in {"query", "build_and_query"}
        and not bool(getattr(args, "vllm_enable_prefix_caching", False))
    ):
        args.vllm_enable_prefix_caching = True

    try:
        proc, log_path, cmd, server_env = _start_vllm_server(args, out_dir)
        _sync_launch_env_artifacts(
            out_dir=out_dir,
            run_meta=run_meta,
            launch_env=dict(run_meta.get("launch_env") or {}),
            server_env=server_env,
        )
        (out_dir / "server_cmd.txt").write_text(
            _render_server_cmd_artifact(
                cmd,
                server_env,
                vllm_hf_overrides=dict(
                    ((run_meta.get("vllm") or {}).get("hf_overrides") or {})
                ),
            ),
            encoding="utf-8",
        )
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        if not _wait_health(base_url, int(args.server_ready_timeout_sec), proc=proc):
            _persist_graph_mode_status(
                out_dir=out_dir,
                run_meta=run_meta,
                compilation_config_raw=_effective_vllm_compilation_config(args),
                log_path=log_path,
            )
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            except Exception:
                pass
            proc_rc = proc.poll() if proc is not None else None
            raise SystemExit(
                f"vLLM did not become healthy (process_rc={proc_rc}). Log tail:\n{tail}"
            )
        graph_mode_status = _persist_graph_mode_status(
            out_dir=out_dir,
            run_meta=run_meta,
            compilation_config_raw=_effective_vllm_compilation_config(args),
            log_path=log_path,
        )
        _maybe_require_graph_capture(status=graph_mode_status)

        calibration_exit = _maybe_run_kv_calibration(
            args=args,
            run_id=run_id,
            out_dir=out_dir,
            run_meta=run_meta,
            log_path=log_path,
            sweep_targets_requested=[int(x) for x in sweep_targets],
            prompt_seed_len=int(len(prompt_seed_ids)),
            prefill_progress_path=(out_dir / "prefill_progress.jsonl"),
        )
        if calibration_exit is not None:
            return int(calibration_exit)

        def _start_prefill_recorder(
            *,
            target_index: int,
            target_prompt_tokens_requested: int,
            target_prompt_tokens_effective: int,
            iteration: int,
            progress_cb: Optional[Callable[[Mapping[str, Any]], None]] = None,
            interval_sec_override: Optional[float] = None,
        ) -> Optional[_PrefillProgressRecorder]:
            # Prefix-state capture needs the live prefill-progress stream even
            # when the caller did not request persistent progress artifacts.
            # This is especially important for streaming checkpoints, where the
            # exact save boundary exists while the request is still RUNNING.
            if not (bool(args.emit_prefill_progress) or callable(progress_cb)):
                return None
            recorder = _PrefillProgressRecorder(
                base_url=base_url,
                out_path=prefill_progress_path,
                target_index=int(target_index),
                target_prompt_tokens_requested=int(target_prompt_tokens_requested),
                target_prompt_tokens_effective=int(target_prompt_tokens_effective),
                iteration=int(iteration),
                interval_sec=(
                    float(interval_sec_override)
                    if interval_sec_override is not None
                    else float(args.prefill_progress_interval_sec)
                ),
                poll_timeout_sec=float(
                    int(args.prefill_progress_timeout_sec)
                    if int(args.prefill_progress_timeout_sec) > 0
                    else int(args.request_timeout_sec)
                ),
                progress_cb=progress_cb,
            )
            recorder.start()
            return recorder

        loaded: Dict[str, Any] = {}
        capture_state_results: List[Dict[str, Any]] = []
        restore_state_results: List[Dict[str, Any]] = []
        dump_calibration_payload: Dict[str, Any] = {}
        prefix_state_load_count = 0
        if mode in {"build", "build_and_query"}:
            saved = _write_prefix_session(
                prefix_session_dir=session_root,
                prefix_session_id=session_id,
                prefix_messages=prefix_messages,
                prefix_token_ids=prefix_token_ids,
                continuation_tail_token_ids=continuation_tail_token_ids,
                run_meta=run_meta,
                prompt_shape_meta=prompt_shape_meta,
                strict_env_hash=strict_env_hash,
                strict_env_inputs=strict_env_inputs,
            )
            capture_session_attention_mode = str(
                (saved.get("manifest") or {}).get("attention_mode") or "full_attention"
            ).strip().lower()
            capture_same_message = (
                str(
                    (prefix_session_payload.get("prefix_capture") or {}).get(
                        "continuation_mode"
                    )
                    or ""
                )
                == "same_message"
            )
            prefix_rendered_tokens = int(len(prefix_token_ids))
            capture_block_size = max(1, int(getattr(args, "block_size", 0) or 512))
            capture_prompt_token_ids = [int(token_id) for token_id in prefix_ids]
            capture_prompt_tokens = int(len(capture_prompt_token_ids))
            capture_request_prompt_token_ids = [
                int(token_id) for token_id in capture_prompt_token_ids
            ]
            capture_via_messages = not bool(capture_same_message)
            stream_same_message_capture_api = str(
                os.environ.get(
                    "HFLC_STREAM_SAME_MESSAGE_CAPTURE_API",
                    "prompt_ids",
                )
                or "prompt_ids"
            ).strip().lower()
            if stream_same_message_capture_api not in {"prompt_ids", "messages"}:
                raise RuntimeError(
                    "Unsupported HFLC_STREAM_SAME_MESSAGE_CAPTURE_API="
                    f"{stream_same_message_capture_api!r}; expected 'prompt_ids' or 'messages'"
                )
            stream_same_message_capture_mode = str(
                os.environ.get(
                    "HFLC_STREAM_SAME_MESSAGE_CAPTURE_MODE",
                    "aligned",
                )
                or "aligned"
            ).strip().lower()
            if stream_same_message_capture_mode not in {"full", "aligned"}:
                raise RuntimeError(
                    "Unsupported HFLC_STREAM_SAME_MESSAGE_CAPTURE_MODE="
                    f"{stream_same_message_capture_mode!r}; expected 'full' or 'aligned'"
                )
            if capture_same_message and capture_session_attention_mode == "streaming_mla":
                if stream_same_message_capture_mode == "full":
                    # Streaming same-message checkpoints may need the full
                    # logical prefix boundary when we want to cold-restore the
                    # entire prefix as already-computed context.
                    capture_prompt_tokens = int(len(prefix_token_ids))
                    capture_prompt_token_ids = [
                        int(token_id)
                        for token_id in list(prefix_token_ids)
                    ]
                    capture_request_prompt_token_ids = [
                        int(token_id)
                        for token_id in list(prefix_token_ids)
                    ]
                else:
                    # Aligned same-message streaming checkpoints still need to
                    # capture at the live prompt-continuation boundary, not at
                    # the start of decode for a shorter aligned-only request.
                    # Send the full logical prefix so the engine can observe
                    # the request at ``computed_tokens == aligned_prefix`` and
                    # serialize the correct sparse/KDA source rows there.
                    #
                    # The capture boundary must be derived from the actual
                    # serialized same-message prefix that will be restored
                    # later, not from the raw requested seed length. At long
                    # points the chat-template round trip can change the token
                    # count slightly; using the raw target would ask the engine
                    # to wait for a boundary the live request can never reach.
                    capture_request_prompt_token_ids = [
                        int(token_id)
                        for token_id in list(prefix_token_ids)
                    ]
                    capture_prompt_tokens = (
                        (int(len(prefix_token_ids)) // int(capture_block_size))
                        * int(capture_block_size)
                    )
                    if capture_prompt_tokens <= 0:
                        capture_prompt_tokens = int(len(prefix_token_ids))
                if stream_same_message_capture_api == "messages":
                    capture_via_messages = True
            if capture_same_message and capture_session_attention_mode != "streaming_mla":
                # Non-streaming same-message capture can still use the exact
                # open-message prompt-token prefix directly.
                capture_prompt_token_ids = [int(token_id) for token_id in prefix_token_ids]
                capture_prompt_tokens = int(len(capture_prompt_token_ids))
                capture_request_prompt_token_ids = [
                    int(token_id) for token_id in capture_prompt_token_ids
                ]
            elif capture_via_messages:
                capture_prompt_tokens = int(
                    len(_tokenize_messages(tokenizer, prefix_messages))
                )
            capture_prefill_trigger_tokens = (
                (int(capture_prompt_tokens) // int(capture_block_size))
                * int(capture_block_size)
            )
            if (
                capture_session_attention_mode == "streaming_mla"
                and stream_same_message_capture_mode == "full"
            ):
                # For streaming sessions the capture request already uses the
                # full logical same-message prefix, so capture at that exact
                # prompt boundary.
                capture_prefill_trigger_tokens = int(capture_prompt_tokens)
            if capture_prefill_trigger_tokens <= 0:
                capture_prefill_trigger_tokens = int(capture_prompt_tokens)
            capture_request_id = ""
            capture_external_request_id = ""
            capture_cache_debug_state: Dict[str, Any] = {}
            capture_stream_events: List[Dict[str, Any]] = []

            def _log_capture_debug(
                note: str,
                *,
                request_id: str = "",
                status: str = "capture_note",
            ) -> None:
                _append_jsonl_row(
                    prefill_progress_path,
                    {
                        "timestamp_utc": _now_iso(),
                        "target_index": 0,
                        "target_prompt_tokens_requested": int(capture_prompt_tokens),
                        "target_prompt_tokens_effective": int(capture_prompt_tokens),
                        "iteration": 1,
                        "request_id": str(request_id or ""),
                        "prefill_done_tokens": 0,
                        "prefill_total_tokens": int(capture_prompt_tokens),
                        "prefill_pct": 0.0,
                        "engine_step": 0,
                        "coarse_state": "",
                        "request_status": "",
                        "source": "prefix_capture_debug",
                        "status": str(status),
                        "note": str(note),
                    },
                )

            def _capture_state_for_request(
                req_id: str,
                *,
                linear_target_last_rows: Optional[Mapping[str, Any]] = None,
                global_target_rows: Optional[Sequence[int]] = None,
                scheduler_row_payload: Optional[Mapping[str, Any]] = None,
                source_group_rows: Optional[Mapping[str, Any]] = None,
            ) -> None:
                nonlocal capture_state_results, capture_request_id, capture_cache_debug_state
                req = str(req_id or "").strip()
                source_group_rows_payload: Dict[str, Any]
                if isinstance(source_group_rows, Mapping) and isinstance(
                    source_group_rows.get("group_rows"),
                    Mapping,
                ):
                    source_group_rows_payload = {
                        "payload_version": int(
                            source_group_rows.get("payload_version") or 2
                        ),
                        "group_rows": {
                            str(key): [
                                int(value)
                                for value in list(values or [])
                                if int(value) > 0
                            ]
                            for key, values in dict(
                                source_group_rows.get("group_rows") or {}
                            ).items()
                            if [
                                int(value)
                                for value in list(values or [])
                                if int(value) > 0
                            ]
                        },
                        "manager_rows": [
                            dict(row)
                            for row in list(source_group_rows.get("manager_rows") or [])
                            if isinstance(row, Mapping)
                        ],
                        "accepted_tokens": max(
                            1,
                            int(source_group_rows.get("accepted_tokens") or 1),
                        ),
                    }
                else:
                    source_group_rows_payload = {
                        str(key): [
                            int(value) for value in list(values or []) if int(value) > 0
                        ]
                        for key, values in dict(source_group_rows or {}).items()
                        if [
                            int(value) for value in list(values or []) if int(value) > 0
                        ]
                    }
                    if scheduler_row_payload is not None:
                        source_group_rows_payload = _scheduler_source_group_rows_payload(
                            dict(scheduler_row_payload)
                        )
                if not req and not source_group_rows_payload:
                    return
                if capture_state_results:
                    return
                capture_request_id = req
                linear_targets_payload = {
                    str(key): int(value)
                    for key, value in dict(linear_target_last_rows or {}).items()
                    if int(value) > 0
                }
                global_targets_payload = [
                    int(value) for value in list(global_target_rows or []) if int(value) > 0
                ]
                if scheduler_row_payload is not None:
                    capture_cache_debug_state = dict(scheduler_row_payload)
                capture_state_results = _call_collective_rpc(
                    base_url=base_url,
                    method="save_prefix_state_for_apc_rows",
                    args=[
                        str(session_root),
                        str(session_id),
                        str(req),
                        int(capture_prompt_tokens),
                        json.dumps(linear_targets_payload, sort_keys=True),
                        json.dumps(global_targets_payload),
                        int(args.prefix_state_chunk_bytes),
                        json.dumps(source_group_rows_payload, sort_keys=True),
                    ],
                    timeout_sec=int(args.request_timeout_sec),
                )
                _require_collective_rpc_ok(
                    method="save_prefix_state_for_apc_rows",
                    results=capture_state_results,
                )
                _update_prefix_manifest_with_state(
                    session_dir=Path(str(saved.get("session_dir"))),
                    rpc_results=capture_state_results,
                    chunk_bytes=int(args.prefix_state_chunk_bytes),
                    fallback_scheduler_payload=scheduler_row_payload,
                    default_block_size_tokens=int(capture_block_size),
                )

            def _validate_capture_state_results_active_request() -> None:
                if not capture_state_results:
                    return
                mismatches: List[str] = []
                accepted_sources = {
                    "scheduler_apc_rows:active_request",
                    "scheduler_apc_rows:described_request_rows",
                }
                for rank_index, row in enumerate(capture_state_results):
                    source = str((row or {}).get("selection_source") or "")
                    if source not in accepted_sources:
                        mismatches.append(
                            f"rank={rank_index}:selection_source={source or 'missing'}"
                        )
                if mismatches:
                    raise SystemExit(
                        "prefix_state_capture_failed_non_active_request: "
                        + "; ".join(mismatches)
                    )

            def _capture_state_results_are_active_request() -> bool:
                if not capture_state_results:
                    return False
                return all(
                    str((row or {}).get("selection_source") or "")
                    in {
                        "scheduler_apc_rows:active_request",
                        "scheduler_apc_rows:described_request_rows",
                    }
                    for row in capture_state_results
                )

            def _discard_capture_state_results() -> None:
                nonlocal capture_state_results, capture_request_id, capture_cache_debug_state
                capture_state_results = []
                capture_request_id = ""
                capture_cache_debug_state = {}

            def _prepare_scheduler_capture_inputs(
                payload: Optional[Mapping[str, Any]],
            ) -> Tuple[Dict[str, Any], Dict[str, int], List[int], Dict[str, List[int]]]:
                """Normalize scheduler APC payloads for save-prefix RPC capture.

                Long StreamingLLM requests can expose a compact manager-row
                layout where the scheduler payload is structurally useful but
                still reports ``ok = false`` because it does not match the old
                active-request APC shape. For exact capture we still want the
                compact dense MLA manager rows plus the sparse tail rows, so
                normalize that payload before handing it to the save RPC.
                """

                scheduler_row_payload = (
                    dict(payload) if isinstance(payload, Mapping) else {}
                )
                linear_target_last_rows: Dict[str, int] = {}
                global_target_rows: List[int] = []
                source_group_rows = _scheduler_source_group_rows(scheduler_row_payload)
                if bool(scheduler_row_payload.get("ok")):
                    linear_target_last_rows = {
                        str(key): int(value)
                        for key, value in dict(
                            scheduler_row_payload.get("linear_target_last_rows") or {}
                        ).items()
                        if int(value) > 0
                    }
                    global_target_rows = [
                        int(value)
                        for value in list(
                            scheduler_row_payload.get("global_target_rows") or []
                        )
                        if int(value) > 0
                    ]
                    return (
                        scheduler_row_payload,
                        linear_target_last_rows,
                        global_target_rows,
                        source_group_rows,
                    )

                if (
                    capture_session_attention_mode == "streaming_mla"
                    and source_group_rows
                ):
                    normalized_payload = _streaming_layout_from_scheduler_payload(
                        scheduler_row_payload
                    )
                    normalized_source_group_rows = {
                        str(key): [
                            int(value)
                            for value in list(values or [])
                            if int(value) > 0
                        ]
                        for key, values in dict(
                            normalized_payload.get("source_group_rows") or {}
                        ).items()
                        if [
                            int(value)
                            for value in list(values or [])
                            if int(value) > 0
                        ]
                    }
                    normalized_linear = {
                        str(key): int(value)
                        for key, value in dict(
                            normalized_payload.get("linear_target_last_rows") or {}
                        ).items()
                        if int(value) > 0
                    }
                    normalized_global = [
                        int(value)
                        for value in list(
                            normalized_payload.get("global_target_rows") or []
                        )
                        if int(value) > 0
                    ]
                    if normalized_source_group_rows:
                        scheduler_row_payload = dict(normalized_payload)
                        source_group_rows = normalized_source_group_rows
                    if normalized_linear:
                        linear_target_last_rows = normalized_linear
                    if normalized_global:
                        global_target_rows = normalized_global

                return (
                    scheduler_row_payload,
                    linear_target_last_rows,
                    global_target_rows,
                    source_group_rows,
                )

            def _engine_capture_payload_results_ok(
                payload: Mapping[str, Any],
            ) -> Tuple[bool, List[Dict[str, Any]]]:
                """Return whether an engine post-step capture payload is usable."""

                rows = list(payload.get("prefix_state_rpc_results") or [])
                if not rows:
                    return False, rows
                try:
                    _require_collective_rpc_ok(
                        method="save_prefix_state_for_apc_rows",
                        results=rows,
                    )
                except Exception as exc:  # noqa: BLE001
                    _log_capture_debug(
                        "engine_post_step_capture_rpc_error=" + str(exc)[:500],
                        request_id=str(payload.get("request_id") or ""),
                        status="capture_error",
                    )
                    return False, rows
                return True, rows

            def _capture_streaming_with_scheduler_rows(
                *,
                external_request_id: str,
                internal_request_id: str,
                capture_mode: str,
                wait_note: str,
                non_ok_note: str,
                ok_note: str,
            ) -> bool:
                """Capture streaming APC state using the explicit scheduler row layout.

                The generic active-request APC derivation only retains one sparse
                row per linear group. For StreamingLLM checkpoints that can expose
                both the aligned boundary row and the compact tail row, we prefer
                the richer scheduler description so the saved state and replay
                metadata stay consistent.
                """

                req = str(external_request_id or "").strip()
                internal_req = str(internal_request_id or "").strip()
                effective_req = str(internal_req or req).strip()
                if not effective_req:
                    return False

                scheduler_row_payload = _describe_request_prefix_apc_rows(
                    base_url=base_url,
                    request_id=effective_req,
                    prefix_token_count=int(capture_prompt_tokens),
                    timeout_sec=int(args.request_timeout_sec),
                )
                (
                    scheduler_row_payload,
                    linear_target_last_rows,
                    global_target_rows,
                    source_group_rows,
                ) = _prepare_scheduler_capture_inputs(scheduler_row_payload)
                multi_row_sparse_groups = sum(
                    1 for values in source_group_rows.values() if len(list(values or [])) > 1
                )
                if not source_group_rows or multi_row_sparse_groups <= 0:
                    _log_capture_debug(
                        wait_note,
                        request_id=req,
                    )
                    return False
                if not bool(scheduler_row_payload.get("ok")):
                    _log_capture_debug(
                        non_ok_note,
                        request_id=req,
                    )
                _discard_capture_state_results()
                _capture_state_for_request(
                    effective_req,
                    linear_target_last_rows=linear_target_last_rows,
                    global_target_rows=global_target_rows,
                    scheduler_row_payload=scheduler_row_payload,
                    source_group_rows=source_group_rows,
                )
                if capture_stream_events:
                    capture_stream_events[-1]["capture_mode"] = capture_mode
                _log_capture_debug(
                    ok_note,
                    request_id=req,
                )
                return True

            def _capture_stream_progress_cb(event: Mapping[str, Any]) -> None:
                nonlocal capture_external_request_id
                if capture_state_results:
                    return
                req = str(event.get("request_id") or "").strip()
                if not req:
                    return
                capture_external_request_id = req
                if len(capture_stream_events) < 32:
                    capture_stream_events.append(
                        {
                            "type": str(event.get("type") or ""),
                            "external_request_id": req,
                            "ttft_ms": int(event.get("ttft_ms") or 0)
                            if str(event.get("type") or "") == "ttft"
                            else None,
                        }
                    )
                if capture_session_attention_mode == "streaming_mla":
                    # Streaming exactness depends on the live active-request
                    # APC rows, not the post-request APC cache snapshot. We
                    # still keep the prefill-boundary callback armed, but we
                    # also attempt capture here at TTFT because some short
                    # requests never surface an intermediate prefill-progress
                    # sample once the reusable boundary has been crossed.
                    resolve_retry_total_sec = 5.0
                    internal_req = _resolve_active_internal_request_id(
                        base_url=base_url,
                        external_request_id=req,
                        timeout_sec=int(args.request_timeout_sec),
                        retry_total_sec=resolve_retry_total_sec,
                        poll_interval_sec=0.05,
                    )
                    _log_capture_debug(
                        f"stream_active_capture_candidate_internal={internal_req or ''}",
                        request_id=req,
                    )
                    if internal_req and capture_stream_events:
                        capture_stream_events[-1]["internal_request_id"] = internal_req
                        capture_stream_events[-1]["capture_mode"] = "ttft_active_candidate"
                try:
                    _capture_state_for_request(req)
                    if capture_state_results and _capture_state_results_are_active_request():
                        _log_capture_debug(
                            "stream_direct_capture_ok",
                            request_id=req,
                        )
                        if capture_stream_events:
                            capture_stream_events[-1]["direct_request_id"] = req
                            capture_stream_events[-1]["capture_mode"] = "direct_request_id"
                        return
                    if capture_state_results:
                        _log_capture_debug(
                            "stream_direct_capture_non_active_request",
                            request_id=req,
                        )
                        if capture_stream_events:
                            capture_stream_events[-1][
                                "direct_capture_error"
                            ] = "selection_source_non_active_request"
                        _discard_capture_state_results()
                except Exception as exc:  # noqa: BLE001
                    _log_capture_debug(
                        f"stream_direct_capture_error={str(exc)[:500]}",
                        request_id=req,
                        status="capture_error",
                    )
                    if capture_stream_events:
                        capture_stream_events[-1]["direct_capture_error"] = str(exc)[:500]
                resolve_retry_total_sec = (
                    5.0 if capture_session_attention_mode == "streaming_mla" else 1.0
                )
                internal_req = _resolve_active_internal_request_id(
                    base_url=base_url,
                    external_request_id=req,
                    timeout_sec=int(args.request_timeout_sec),
                    retry_total_sec=resolve_retry_total_sec,
                    poll_interval_sec=0.05,
                )
                _log_capture_debug(
                    f"stream_resolved_internal={internal_req or ''}",
                    request_id=req,
                )
                if internal_req and capture_stream_events:
                    capture_stream_events[-1]["internal_request_id"] = internal_req
                if capture_session_attention_mode == "streaming_mla":
                    if _capture_streaming_with_scheduler_rows(
                        external_request_id=req,
                        internal_request_id=internal_req,
                        capture_mode="described_request_rows",
                        wait_note="stream_capture_waiting_for_source_group_rows",
                        non_ok_note="stream_capture_using_manager_rows_from_non_ok_payload",
                        ok_note="stream_scheduler_capture_ok",
                    ):
                        return
                if internal_req:
                    try:
                        _capture_state_for_request(internal_req)
                        if capture_state_results and _capture_state_results_are_active_request():
                            _log_capture_debug(
                                "stream_internal_active_capture_ok",
                                request_id=req,
                            )
                            if capture_stream_events:
                                capture_stream_events[-1]["capture_mode"] = "active_request"
                            return
                        if capture_state_results:
                            _log_capture_debug(
                                "stream_internal_active_capture_non_active_request",
                                request_id=req,
                            )
                            if capture_stream_events:
                                capture_stream_events[-1][
                                    "direct_capture_error"
                                ] = "selection_source_non_active_request"
                            _discard_capture_state_results()
                    except Exception as exc:  # noqa: BLE001
                        _log_capture_debug(
                            f"stream_internal_active_capture_error={str(exc)[:500]}",
                            request_id=req,
                            status="capture_error",
                        )
                        if capture_stream_events:
                            capture_stream_events[-1]["direct_capture_error"] = str(exc)[:500]
                scheduler_row_payload: Dict[str, Any] = {}
                linear_target_last_rows: Dict[str, int] = {}
                global_target_rows: List[int] = []
                source_group_rows: Dict[str, List[int]] = {}
                if internal_req:
                    scheduler_row_payload = _describe_request_prefix_apc_rows(
                        base_url=base_url,
                        request_id=internal_req,
                        prefix_token_count=int(capture_prompt_tokens),
                        timeout_sec=int(args.request_timeout_sec),
                    )
                    (
                        scheduler_row_payload,
                        linear_target_last_rows,
                        global_target_rows,
                        source_group_rows,
                    ) = _prepare_scheduler_capture_inputs(
                        scheduler_row_payload
                    )
                    if not bool(scheduler_row_payload.get("ok")) and source_group_rows:
                        _log_capture_debug(
                            "stream_capture_using_manager_rows_from_non_ok_payload",
                            request_id=req,
                        )
                if (
                    capture_session_attention_mode == "streaming_mla"
                    and not source_group_rows
                ):
                    _log_capture_debug(
                        "stream_capture_waiting_for_source_group_rows",
                        request_id=req,
                    )
                    return
                _capture_state_for_request(
                    internal_req,
                    linear_target_last_rows=linear_target_last_rows,
                    global_target_rows=global_target_rows,
                    scheduler_row_payload=scheduler_row_payload,
                    source_group_rows=source_group_rows,
                )

            def _capture_prefill_progress_cb(progress: Mapping[str, Any]) -> None:
                if capture_state_results:
                    return
                req = str(progress.get("request_id") or "").strip()
                total = int(progress.get("prefill_total_tokens") or 0)
                done = int(progress.get("prefill_done_tokens") or 0)
                coarse_state = str(progress.get("coarse_state") or "")
                if not req or total <= 0:
                    return
                if done < int(capture_prefill_trigger_tokens):
                    return
                if len(capture_stream_events) < 32:
                    event_row = {
                        "type": "prefill_boundary",
                        "external_request_id": req,
                        "prefill_done_tokens": int(done),
                        "prefill_total_tokens": int(total),
                        "prefix_rendered_tokens": int(prefix_rendered_tokens),
                        "capture_prefill_trigger_tokens": int(
                            capture_prefill_trigger_tokens
                        ),
                        "coarse_state": coarse_state,
                        "request_status": str(progress.get("request_status") or ""),
                    }
                    capture_stream_events.append(event_row)
                try:
                    _capture_state_for_request(req)
                    if capture_state_results and _capture_state_results_are_active_request():
                        _log_capture_debug(
                            "prefill_direct_capture_ok",
                            request_id=req,
                        )
                        if capture_stream_events:
                            capture_stream_events[-1]["direct_request_id"] = req
                            capture_stream_events[-1]["capture_mode"] = "direct_request_id"
                        return
                    if capture_state_results:
                        _log_capture_debug(
                            "prefill_direct_capture_non_active_request",
                            request_id=req,
                        )
                        if capture_stream_events:
                            capture_stream_events[-1][
                                "direct_capture_error"
                            ] = "selection_source_non_active_request"
                        _discard_capture_state_results()
                except Exception as exc:  # noqa: BLE001
                    _log_capture_debug(
                        f"prefill_direct_capture_error={str(exc)[:500]}",
                        request_id=req,
                        status="capture_error",
                    )
                    if capture_stream_events:
                        capture_stream_events[-1]["direct_capture_error"] = str(exc)[:500]
                resolve_retry_total_sec = (
                    5.0 if capture_session_attention_mode == "streaming_mla" else 1.0
                )
                internal_req = _resolve_active_internal_request_id(
                    base_url=base_url,
                    external_request_id=req,
                    timeout_sec=int(args.request_timeout_sec),
                    retry_total_sec=resolve_retry_total_sec,
                    poll_interval_sec=0.02,
                )
                _log_capture_debug(
                    f"prefill_resolved_internal={internal_req or ''}",
                    request_id=req,
                )
                effective_req = str(internal_req or req).strip()
                if capture_stream_events:
                    capture_stream_events[-1]["internal_request_id"] = effective_req
                if capture_session_attention_mode == "streaming_mla":
                    if _capture_streaming_with_scheduler_rows(
                        external_request_id=req,
                        internal_request_id=effective_req,
                        capture_mode="described_request_rows",
                        wait_note="prefill_capture_waiting_for_source_group_rows",
                        non_ok_note="prefill_capture_using_manager_rows_from_non_ok_payload",
                        ok_note="prefill_scheduler_capture_ok",
                    ):
                        return
                if effective_req:
                    try:
                        _capture_state_for_request(effective_req)
                        if capture_state_results and _capture_state_results_are_active_request():
                            _log_capture_debug(
                                "prefill_internal_active_capture_ok",
                                request_id=req,
                            )
                            if capture_stream_events:
                                capture_stream_events[-1]["capture_mode"] = "active_request"
                            return
                        if capture_state_results:
                            _log_capture_debug(
                                "prefill_internal_active_capture_non_active_request",
                                request_id=req,
                            )
                            if capture_stream_events:
                                capture_stream_events[-1][
                                    "direct_capture_error"
                                ] = "selection_source_non_active_request"
                            _discard_capture_state_results()
                    except Exception as exc:  # noqa: BLE001
                        _log_capture_debug(
                            f"prefill_internal_active_capture_error={str(exc)[:500]}",
                            request_id=req,
                            status="capture_error",
                        )
                        if capture_stream_events:
                            capture_stream_events[-1]["direct_capture_error"] = str(exc)[:500]
                scheduler_row_payload = _describe_request_prefix_apc_rows(
                    base_url=base_url,
                    request_id=effective_req,
                    prefix_token_count=int(capture_prompt_tokens),
                    timeout_sec=int(args.request_timeout_sec),
                )
                (
                    scheduler_row_payload,
                    linear_target_last_rows,
                    global_target_rows,
                    source_group_rows,
                ) = _prepare_scheduler_capture_inputs(
                    scheduler_row_payload
                )
                if (
                    capture_session_attention_mode == "streaming_mla"
                    and not bool(scheduler_row_payload.get("ok"))
                    and source_group_rows
                ):
                    _log_capture_debug(
                        "prefill_capture_using_manager_rows_from_non_ok_payload",
                        request_id=req,
                    )
                if (
                    capture_session_attention_mode == "streaming_mla"
                    and not source_group_rows
                ):
                    _log_capture_debug(
                        "prefill_capture_waiting_for_source_group_rows",
                        request_id=req,
                    )
                    return
                _capture_state_for_request(
                    effective_req,
                    linear_target_last_rows=linear_target_last_rows,
                    global_target_rows=global_target_rows,
                    scheduler_row_payload=scheduler_row_payload,
                    source_group_rows=source_group_rows,
                )

            if bool(int(args.prefix_save_shards)):
                calibration_rpc = _call_collective_rpc(
                    base_url=base_url,
                    method="estimate_prefix_state_dump_overhead",
                    args=[
                        "",
                        int(capture_prompt_tokens),
                        int(args.prefix_state_chunk_bytes),
                        float(args.prefix_state_calibration_safety_ratio),
                        int(args.prefix_state_calibration_safety_min_bytes),
                    ],
                    timeout_sec=int(args.request_timeout_sec),
                )
                _require_collective_rpc_ok(
                    method="estimate_prefix_state_dump_overhead",
                    results=calibration_rpc,
                )
                calibration = _evaluate_prefix_state_dump_calibration(calibration_rpc)
                dump_calibration_payload = {
                    "ok": bool((calibration.get("summary") or {}).get("all_pass", False)),
                    "run_id": run_id,
                    "session_id": session_id,
                    "session_dir": str(saved.get("session_dir") or ""),
                    "strict_gate": bool(int(args.prefix_state_calibration_strict_gate)),
                    "request_id": "",
                    "prefix_token_count": int(capture_prompt_tokens),
                    "chunk_bytes": int(args.prefix_state_chunk_bytes),
                    "safety_ratio": float(args.prefix_state_calibration_safety_ratio),
                    "safety_min_bytes": int(args.prefix_state_calibration_safety_min_bytes),
                    "calibration": calibration,
                }
                (out_dir / "prefix_state_calibration.json").write_text(
                    json.dumps(dump_calibration_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if bool(int(args.prefix_state_calibration_strict_gate)) and not bool(
                    (calibration.get("summary") or {}).get("all_pass", False)
                ):
                    fail_rows = [
                        r
                        for r in list(calibration.get("rows") or [])
                        if not bool(r.get("pass"))
                    ]
                    snippets = [
                        f"rank={int(r.get('rank', -1))}: free={int(r.get('free_bytes', 0))} < required={int(r.get('required_free_bytes', 0))}"
                        for r in fail_rows[:3]
                    ]
                    raise SystemExit(
                        "prefix_state_calibration_gate_failed: " + "; ".join(snippets)
                    )

            engine_capture_result_path: Optional[Path] = None
            capture_request_vllm_xargs: Optional[Dict[str, Any]] = None
            raw_stream_capture_phase = str(
                os.environ.get("HFLC_STREAM_DEBUG_CAPTURE_PHASE", "") or ""
            ).strip().lower()
            allow_stream_capture_phase_override = bool(
                int(
                    os.environ.get(
                        "HFLC_STREAM_DEBUG_ALLOW_CAPTURE_PHASE_OVERRIDE", "0"
                    )
                    or "0"
                )
            )
            if (
                allow_stream_capture_phase_override
                and raw_stream_capture_phase in {
                    "pre",
                    "mid",
                    "post",
                    "continuation_pre",
                }
            ):
                stream_capture_phase = raw_stream_capture_phase
            else:
                # Streaming same-message capture needs the start of the first
                # prompt-continuation step: the aligned prefix is already
                # visible as computed context, but no continuation/query
                # tokens have executed yet. Ignore ambient shell overrides
                # unless the caller explicitly opts into debug-only phase
                # experimentation.
                stream_capture_phase = "continuation_pre"
            if (
                bool(int(args.prefix_save_shards))
                and capture_session_attention_mode == "streaming_mla"
            ):
                engine_capture_result_path = (
                    Path(str(saved.get("session_dir") or ""))
                    / "engine_prefill_capture_result.json"
                )
                if engine_capture_result_path.exists():
                    engine_capture_result_path.unlink()
                capture_request_vllm_xargs = {
                    "debug_prefix_capture_enabled": True,
                    "debug_prefix_capture_session_dir": str(session_root),
                    "debug_prefix_capture_session_id": str(session_id),
                    "debug_prefix_capture_result_path": str(engine_capture_result_path),
                    "debug_prefix_capture_prefix_token_count": int(capture_prompt_tokens),
                    "debug_prefix_capture_trigger_token_count": int(
                        capture_prefill_trigger_tokens
                    ),
                    "debug_prefix_capture_chunk_bytes": int(
                        args.prefix_state_chunk_bytes
                    ),
                    # Same-message streaming capture now defaults to the
                    # post-execute/pre-update boundary, while still allowing a
                    # local override for focused runtime debugging.
                    "debug_prefix_capture_phase": stream_capture_phase,
                }

            capture_probe_new_tokens = (
                1
                if (
                    bool(int(args.prefix_save_shards))
                    and capture_session_attention_mode == "streaming_mla"
                    and stream_capture_phase in {"pre", "mid", "continuation_pre"}
                )
                else (
                    2
                    if (
                        bool(int(args.prefix_save_shards))
                        and capture_session_attention_mode == "streaming_mla"
                    )
                    else (
                        1
                        if bool(int(args.prefix_save_shards))
                        else max(
                            1,
                            int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else 1,
                        )
                    )
                )
            )
            capture_probe_min_new_tokens = (
                0
                if (
                    bool(int(args.prefix_save_shards))
                    and capture_session_attention_mode == "streaming_mla"
                )
                else (
                    int(capture_probe_new_tokens)
                    if bool(int(args.prefix_save_shards))
                    else 0
                )
            )
            capture_probe_output_mask_payload = (
                None
                if (
                    bool(int(args.prefix_save_shards))
                    and capture_session_attention_mode == "streaming_mla"
                )
                else output_mask_payload
            )
            use_engine_only_streaming_capture = bool(
                bool(int(args.prefix_save_shards))
                and capture_session_attention_mode == "streaming_mla"
            )
            capture_request_debug_payload: Dict[str, Any] = {}
            capture_debug_paths: Dict[str, str] = {}
            capture_stream_progress_cb: Optional[Any] = None
            capture_progress_cb: Optional[Any] = (
                _capture_stream_progress_cb
                if bool(int(args.prefix_save_shards))
                and not use_engine_only_streaming_capture
                else None
            )
            capture_stream_progress_cb = capture_progress_cb
            capture_recorder = _start_prefill_recorder(
                target_index=0,
                target_prompt_tokens_requested=int(capture_prompt_tokens),
                target_prompt_tokens_effective=int(capture_prompt_tokens),
                iteration=1,
                progress_cb=(
                    _capture_prefill_progress_cb
                    if bool(int(args.prefix_save_shards))
                    and not use_engine_only_streaming_capture
                    else None
                ),
                interval_sec_override=(
                    (
                        0.005
                        if (
                            bool(int(args.prefix_save_shards))
                            and capture_session_attention_mode == "streaming_mla"
                        )
                        else (0.05 if bool(int(args.prefix_save_shards)) else None)
                    )
                ),
            )
            try:
                if capture_same_message:
                    capture_metrics = _run_one_request_completion_prompt_ids(
                        base_url=base_url,
                        model_id=str(args.model_id),
                        tokenizer=tokenizer,
                        prompt_token_ids=capture_request_prompt_token_ids,
                        max_new_tokens=int(capture_probe_new_tokens),
                        min_new_tokens=int(capture_probe_min_new_tokens),
                        timeout_sec=int(args.request_timeout_sec),
                        ignore_eos=(
                            True
                            if bool(int(args.prefix_save_shards))
                            else bool(args.ignore_eos)
                        ),
                        use_stream=bool(not args.no_stream) or bool(int(args.prefix_save_shards)),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                        repetition_penalty=float(args.repetition_penalty),
                        presence_penalty=float(args.presence_penalty),
                        frequency_penalty=float(args.frequency_penalty),
                        output_mask_payload=capture_probe_output_mask_payload,
                        vllm_xargs=capture_request_vllm_xargs,
                        prompt_token_count_override=int(capture_prompt_tokens),
                        stream_progress_cb=capture_stream_progress_cb,
                    )
                else:
                    capture_metrics = _run_one_request_messages(
                        base_url=base_url,
                        model_id=str(args.model_id),
                        tokenizer=tokenizer,
                        messages=prefix_messages,
                        max_new_tokens=int(capture_probe_new_tokens),
                        min_new_tokens=int(capture_probe_min_new_tokens),
                        timeout_sec=int(args.request_timeout_sec),
                        ignore_eos=(
                            True
                            if bool(int(args.prefix_save_shards))
                            else bool(args.ignore_eos)
                        ),
                        use_stream=bool(not args.no_stream) or bool(int(args.prefix_save_shards)),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                        repetition_penalty=float(args.repetition_penalty),
                        presence_penalty=float(args.presence_penalty),
                        frequency_penalty=float(args.frequency_penalty),
                        output_mask_payload=capture_probe_output_mask_payload,
                        vllm_xargs=capture_request_vllm_xargs,
                        stream_progress_cb=capture_stream_progress_cb,
                    )
                if capture_recorder is not None:
                    capture_recorder.stop(
                        final_status="completed",
                        note="prefix_capture_request_done",
                    )
            except Exception as exc:  # noqa: BLE001
                salvaged_capture = False
                if (
                    capture_session_attention_mode == "streaming_mla"
                    and engine_capture_result_path is not None
                    and engine_capture_result_path.exists()
                ):
                    try:
                        engine_capture_payload = json.loads(
                            engine_capture_result_path.read_text(encoding="utf-8")
                        )
                        if bool(engine_capture_payload.get("ok")):
                            engine_capture_ok, engine_capture_rows = (
                                _engine_capture_payload_results_ok(
                                    engine_capture_payload
                                )
                            )
                            if engine_capture_ok:
                                capture_state_results = list(engine_capture_rows)
                                capture_cache_debug_state = dict(
                                    engine_capture_payload.get("scheduler_row_payload") or {}
                                )
                                capture_request_id = str(
                                    engine_capture_payload.get("request_id")
                                    or capture_request_id
                                )
                                if capture_state_results:
                                    _update_prefix_manifest_with_state(
                                        session_dir=Path(str(saved.get("session_dir"))),
                                        rpc_results=capture_state_results,
                                        chunk_bytes=int(args.prefix_state_chunk_bytes),
                                        fallback_scheduler_payload=capture_cache_debug_state,
                                        default_block_size_tokens=int(capture_block_size),
                                    )
                                capture_metrics = {
                                    "request_id": str(capture_request_id or ""),
                                    "prompt_tokens": int(capture_prompt_tokens),
                                    "completion_tokens": 0,
                                    "generated_text": "",
                                    "finish_reason": "",
                                    "stop_reason": "",
                                    "ttft_ms": 0,
                                    "decode_ms": 0,
                                    "decode_ms_per_token": 0.0,
                                    "gen_tokens_per_sec": 0.0,
                                    "total_ms": 0,
                                    "total_tokens_per_sec": 0.0,
                                    "usage_cached_prompt_tokens": 0,
                                }
                                salvaged_capture = True
                                _log_capture_debug(
                                    "engine_post_step_capture_salvaged_after_request_error="
                                    + str(exc)[:200],
                                    request_id=str(capture_request_id or ""),
                                )
                    except Exception as salvage_exc:  # noqa: BLE001
                        _log_capture_debug(
                            "engine_post_step_capture_salvage_error="
                            + str(salvage_exc)[:200],
                            request_id=str(capture_request_id or ""),
                            status="capture_error",
                        )
                if capture_recorder is not None:
                    capture_recorder.stop(
                        final_status=("completed" if salvaged_capture else "error"),
                        note=(
                            "prefix_capture_request_done_salvaged"
                            if salvaged_capture
                            else f"prefix_capture_request_error={str(exc)[:500]}"
                        ),
                    )
                if not salvaged_capture:
                    raise
            if capture_request_debug_payload:
                capture_debug_paths = _persist_request_debug_artifacts(
                    out_dir=out_dir,
                    target_index=0,
                    iteration=0,
                    debug_payload=capture_request_debug_payload,
                )
            if bool(int(args.prefix_save_shards)):
                if (
                    capture_session_attention_mode == "streaming_mla"
                    and engine_capture_result_path is not None
                    and engine_capture_result_path.exists()
                ):
                    try:
                        engine_capture_payload = json.loads(
                            engine_capture_result_path.read_text(encoding="utf-8")
                        )
                        if bool(engine_capture_payload.get("ok")):
                            engine_capture_ok, engine_capture_rows = (
                                _engine_capture_payload_results_ok(
                                    engine_capture_payload
                                )
                            )
                            if engine_capture_ok:
                                if not (
                                    capture_state_results
                                    and _capture_state_results_are_active_request()
                                ):
                                    capture_state_results = list(engine_capture_rows)
                                    capture_cache_debug_state = dict(
                                        engine_capture_payload.get("scheduler_row_payload")
                                        or {}
                                    )
                                    capture_request_id = str(
                                        engine_capture_payload.get("request_id")
                                        or capture_request_id
                                    )
                                    if capture_state_results:
                                        _update_prefix_manifest_with_state(
                                            session_dir=Path(str(saved.get("session_dir"))),
                                            rpc_results=capture_state_results,
                                            chunk_bytes=int(args.prefix_state_chunk_bytes),
                                            fallback_scheduler_payload=capture_cache_debug_state,
                                            default_block_size_tokens=int(capture_block_size),
                                        )
                                else:
                                    _log_capture_debug(
                                        "preserving_live_capture_over_engine_post_step_result",
                                        request_id=str(capture_request_id or ""),
                                    )
                                _log_capture_debug(
                                    "engine_post_step_capture_ok",
                                    request_id=str(capture_request_id or ""),
                                )
                        else:
                            _log_capture_debug(
                                "engine_post_step_capture_error="
                                + str(engine_capture_payload.get("error") or "unknown"),
                                request_id=str(capture_request_id or ""),
                                status="capture_error",
                            )
                    except Exception as exc:  # noqa: BLE001
                        _log_capture_debug(
                            f"engine_post_step_capture_read_error={str(exc)[:500]}",
                            request_id=str(capture_request_id or ""),
                            status="capture_error",
                        )
                cache_probe_rows = max(
                    1,
                    int(capture_prompt_tokens) // max(1, int(capture_block_size)),
                )
                apc_cache_layout: Dict[str, Any] = {}
                try:
                    cache_debug_state = _debug_prefix_cache_state(
                        base_url=base_url,
                        prompt_token_ids=[
                            int(token_id)
                            for token_id in capture_request_prompt_token_ids
                        ],
                        max_block_rows=int(cache_probe_rows),
                        timeout_sec=int(args.request_timeout_sec),
                    )
                    if bool(cache_debug_state.get("ok")):
                        apc_cache_layout = _extract_apc_cache_row_targets(cache_debug_state)
                except Exception as exc:  # noqa: BLE001
                    _log_capture_debug(
                        f"post_request_apc_snapshot_error={str(exc)[:500]}",
                        status="capture_error",
                    )
                if apc_cache_layout.get("source_group_rows"):
                    if capture_session_attention_mode == "streaming_mla" and not capture_state_results:
                        raise SystemExit(
                            "streaming_prefix_capture_failed_live_rows_missing: "
                            "refusing to fall back to __apc_cache_snapshot__"
                        )
                    if not capture_state_results:
                        _capture_state_for_request(
                            "__apc_cache_snapshot__",
                            linear_target_last_rows=dict(
                                apc_cache_layout.get("linear_target_last_rows") or {}
                            ),
                            global_target_rows=list(
                                apc_cache_layout.get("global_target_rows") or []
                            ),
                            scheduler_row_payload=apc_cache_layout,
                            source_group_rows=dict(
                                apc_cache_layout.get("source_group_rows") or {}
                            ),
                        )
                    else:
                        _log_capture_debug(
                            "preserving_active_streaming_capture_over_post_request_snapshot"
                        )
                    if not (
                        capture_session_attention_mode == "streaming_mla"
                        and capture_state_results
                    ):
                        _update_prefix_manifest_with_apc_cache_layout(
                            session_dir=Path(str(saved.get("session_dir"))),
                            apc_cache_layout=apc_cache_layout,
                        )
                    else:
                        _log_capture_debug(
                            "preserving_streaming_active_manifest_over_post_request_snapshot"
                        )
                if not capture_state_results:
                    capture_external_request_id = str(capture_metrics.get("request_id") or "")
                    fallback_internal_request_id = _resolve_active_internal_request_id(
                            base_url=base_url,
                            external_request_id=capture_external_request_id,
                            timeout_sec=int(args.request_timeout_sec),
                            retry_total_sec=1.0,
                            poll_interval_sec=0.05,
                        )
                    fallback_scheduler_row_payload: Dict[str, Any] = {}
                    fallback_linear_target_last_rows: Dict[str, int] = {}
                    fallback_global_target_rows: List[int] = []
                    fallback_source_group_rows: Dict[str, List[int]] = {}
                    if fallback_internal_request_id:
                        fallback_scheduler_row_payload = _describe_request_prefix_apc_rows(
                            base_url=base_url,
                            request_id=fallback_internal_request_id,
                            prefix_token_count=int(capture_prompt_tokens),
                            timeout_sec=int(args.request_timeout_sec),
                        )
                        (
                            fallback_scheduler_row_payload,
                            fallback_linear_target_last_rows,
                            fallback_global_target_rows,
                            fallback_source_group_rows,
                        ) = _prepare_scheduler_capture_inputs(
                            fallback_scheduler_row_payload
                        )
                    _capture_state_for_request(
                        fallback_internal_request_id,
                        linear_target_last_rows=fallback_linear_target_last_rows,
                        global_target_rows=fallback_global_target_rows,
                        scheduler_row_payload=fallback_scheduler_row_payload,
                        source_group_rows=fallback_source_group_rows,
                    )
                if not capture_state_results:
                    raise SystemExit(
                        "prefix_state_capture_failed: request_id unavailable or request not active at capture time"
                    )
                _validate_capture_state_results_active_request()
                if capture_session_attention_mode == "streaming_mla":
                    _update_prefix_manifest_with_state(
                        session_dir=Path(str(saved.get("session_dir"))),
                        rpc_results=capture_state_results,
                        chunk_bytes=int(args.prefix_state_chunk_bytes),
                        fallback_scheduler_payload=capture_cache_debug_state,
                        default_block_size_tokens=int(capture_block_size),
                    )
            capture_payload = {
                "ok": True,
                "run_id": run_id,
                "session_id": session_id,
                "session_dir": str(saved.get("session_dir") or ""),
                "prefix_tokens_requested": int(args.target_prompt_tokens),
                "prefix_tokens_effective": int(len(prefix_ids)),
                "capture_prompt_tokens": int(capture_prompt_tokens),
                "prefix_rendered_tokens": int(len(prefix_token_ids)),
                "prefix_capture": dict(prefix_session_payload.get("prefix_capture") or {}),
                "continuation_tail_token_count": int(len(continuation_tail_token_ids)),
                "capture_probe_query": "",
                "capture_probe_max_new_tokens": int(capture_probe_new_tokens),
                "capture_probe_min_new_tokens": int(capture_probe_min_new_tokens),
                "capture_probe_metrics": capture_metrics,
                "capture_request_id": str(capture_metrics.get("request_id") or capture_request_id or ""),
                "capture_external_request_id": str(capture_external_request_id),
                "capture_internal_request_id": str(capture_request_id),
                "capture_live_request_layout_path": str(
                    capture_debug_paths.get("live_request_layout_path") or ""
                ),
                "capture_live_external_request_id": str(
                    capture_request_debug_payload.get("external_request_id") or ""
                ),
                "capture_live_internal_request_id": str(
                    capture_request_debug_payload.get("internal_request_id") or ""
                ),
                "capture_live_layout_ok": bool(
                    dict(capture_request_debug_payload.get("layout") or {}).get("ok")
                ),
                "prefix_state_saved": bool(int(args.prefix_save_shards)),
                "prefix_state_chunk_bytes": int(args.prefix_state_chunk_bytes),
                "prefix_state_rpc_results": capture_state_results,
                "prefix_cache_debug_state": capture_cache_debug_state,
                "prefix_state_capture_stream_events": capture_stream_events,
                "prefix_state_calibration": dump_calibration_payload,
            }
            (out_dir / "prefix_capture_metrics.json").write_text(
                json.dumps(capture_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded = _load_prefix_session(prefix_session_dir=session_root, prefix_session_id=session_id)

        if mode in {"restore", "query"} and not loaded:
            loaded = _load_prefix_session(prefix_session_dir=session_root, prefix_session_id=session_id)

        if mode in {"restore", "query", "build_and_query"}:
            manifest = dict(loaded.get("manifest") or {})
            normalized_manifest = _prefix_session_lib.normalize_prefix_session_manifest(manifest)
            loaded_attention_mode = str(normalized_manifest.get("attention_mode") or "")
            compat_ok, mismatches = _validate_prefix_session_compat(
                manifest=normalized_manifest,
                run_meta=run_meta,
                prompt_shape_meta=prompt_shape_meta,
                strict_env_hash=strict_env_hash,
                strict=bool(int(args.prefix_compat_strict)),
            )
            restore_payload = {
                "ok": bool(compat_ok),
                "run_id": run_id,
                "session_id": session_id,
                "session_dir": str(loaded.get("session_dir") or ""),
                "compat_strict": bool(int(args.prefix_compat_strict)),
                "mismatches": list(mismatches),
            }
            (out_dir / "prefix_restore_metrics.json").write_text(
                json.dumps(restore_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not compat_ok:
                raise SystemExit(
                    "prefix session compatibility check failed: " + "; ".join(mismatches)
                )
            if bool(int(args.prefix_save_shards)):

                def _persist_restore_state_payload(
                    results: List[Dict[str, Any]],
                    *,
                    prepare_payload: Mapping[str, Any] | None = None,
                    post_load_prefix_row_hashes: Mapping[str, Any] | None = None,
                ) -> None:
                    restore_state_payload = {
                        "ok": True,
                        "run_id": run_id,
                        "session_id": session_id,
                        "session_dir": str(loaded.get("session_dir") or ""),
                        "prefix_state_loaded": True,
                        "prefix_state_chunk_bytes": int(args.prefix_state_chunk_bytes),
                        "prefix_state_reload_between_queries": bool(
                            int(args.prefix_reload_between_queries)
                        ),
                        "prefix_state_load_count": int(prefix_state_load_count),
                        "prefix_state_rpc_results": list(results),
                    }
                    if prepare_payload:
                        restore_state_payload["prefix_state_prepare_payload"] = dict(
                            prepare_payload
                        )
                    if post_load_prefix_row_hashes:
                        restore_state_payload["post_load_prefix_row_hashes"] = dict(
                            post_load_prefix_row_hashes
                        )
                    (out_dir / "prefix_restore_state_rpc.json").write_text(
                        json.dumps(restore_state_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                def _persist_register_state_payload(payload: Mapping[str, Any]) -> None:
                    """Persist APC registration results for a restored prefix session.

                    Args:
                        payload: Parsed response payload from the registration endpoint.
                    """

                    register_state_payload = {
                        "run_id": run_id,
                        "session_id": session_id,
                        "session_dir": str(loaded.get("session_dir") or ""),
                        "prefix_state_load_count": int(prefix_state_load_count),
                    }
                    register_state_payload.update(dict(payload))
                    (out_dir / "prefix_register_loaded_cache.json").write_text(
                        json.dumps(register_state_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                def _load_prefix_state_from_disk() -> None:
                    nonlocal restore_state_results, prefix_state_load_count
                    prepare_payload: Mapping[str, Any] | None = None
                    post_load_prefix_row_hashes: Mapping[str, Any] | None = None
                    row_hash_config = _query_row_hash_config()
                    if mode in {"query", "build_and_query"} and loaded_attention_mode in {
                        "full_attention",
                        "streaming_mla",
                    }:
                        def _prepare_loaded_prefix_cache_with_retry() -> Mapping[str, Any]:
                            """Retry transient prepare failures caused by in-flight cache resets."""

                            last_payload: Mapping[str, Any] | None = None
                            for attempt in range(12):
                                payload = _prefix_session_lib.prepare_loaded_prefix_cache(
                                    base_url=base_url,
                                    prefix_session_dir=str(session_root),
                                    prefix_session_id=str(session_id),
                                    timeout_sec=int(args.request_timeout_sec),
                                )
                                last_payload = payload
                                if bool(payload.get("ok")):
                                    return payload
                                if str(payload.get("error") or "") != "reset_prefix_cache_failed":
                                    return payload
                                time.sleep(min(1.0 + 0.5 * float(attempt), 4.0))
                            return dict(last_payload or {})

                        prepare_payload = _prepare_loaded_prefix_cache_with_retry()
                        if not bool(prepare_payload.get("ok")):
                            raise SystemExit(
                                "prepare_loaded_prefix_cache failed: "
                                + str(dict(prepare_payload))
                            )
                        if loaded_attention_mode == "streaming_mla":
                            prepare_payload = _prefix_session_lib.project_streaming_prepare_payload_to_saved_layout(
                                prepare_payload,
                                dict(loaded.get("manifest") or {}),
                            )
                        plan_id = str(prepare_payload.get("plan_id") or "").strip()
                        if loaded_attention_mode == "full_attention" and not plan_id:
                            raise SystemExit(
                                "prepare_loaded_prefix_cache returned no plan_id: "
                                + str(dict(prepare_payload))
                            )
                        restore_state_results = _call_collective_rpc(
                            base_url=base_url,
                            method="load_prefix_state_into_apc_targets",
                            args=[
                                str(session_root),
                                str(session_id),
                                json.dumps(
                                    dict(prepare_payload.get("linear_target_last_rows") or {}),
                                    sort_keys=True,
                                ),
                                json.dumps(list(prepare_payload.get("global_target_rows") or [])),
                                int(prepare_payload.get("global_target_group_id") or -1),
                                int(args.prefix_compat_strict),
                                int(args.prefix_state_chunk_bytes),
                            ],
                            timeout_sec=int(args.request_timeout_sec),
                        )
                        try:
                            _require_collective_rpc_ok(
                                method="load_prefix_state_into_apc_targets",
                                results=restore_state_results,
                            )
                        except Exception:  # noqa: BLE001
                            if plan_id:
                                _prefix_session_lib.register_loaded_prefix_cache(
                                    base_url=base_url,
                                    prefix_session_dir=str(session_root),
                                    prefix_session_id=str(session_id),
                                    timeout_sec=int(args.request_timeout_sec),
                                    plan_id=plan_id,
                                    abort=True,
                                )
                            raise
                        if loaded_attention_mode == "streaming_mla":
                            loaded["streaming_prepare_payload"] = dict(prepare_payload)
                            loaded["streaming_prepared_manager_rows"] = (
                                _prefix_session_lib.build_streaming_manager_rows_from_prepared_layout(
                                    prepare_payload
                                )
                            )
                    else:
                        restore_state_results = _call_collective_rpc(
                            base_url=base_url,
                            method="load_prefix_state",
                            args=[
                                str(session_root),
                                str(session_id),
                                int(args.prefix_compat_strict),
                                int(args.prefix_state_chunk_bytes),
                            ],
                            timeout_sec=int(args.request_timeout_sec),
                        )
                        _require_collective_rpc_ok(
                            method="load_prefix_state",
                            results=restore_state_results,
                        )
                        loaded["streaming_prepare_payload"] = {}
                        loaded["streaming_prepared_manager_rows"] = []
                    if bool(row_hash_config.get("enabled")):
                        linear_target_last_rows = dict(
                            (prepare_payload or {}).get("linear_target_last_rows")
                            or (
                                restore_state_results[0].get("linear_target_last_rows")
                                if restore_state_results
                                else {}
                            )
                            or {}
                        )
                        global_target_rows = list(
                            (prepare_payload or {}).get("global_target_rows")
                            or (
                                restore_state_results[0].get("global_target_rows")
                                if restore_state_results
                                else []
                            )
                            or []
                        )
                        if linear_target_last_rows or global_target_rows:
                            try:
                                post_load_prefix_row_hashes = _fetch_query_prefix_row_hashes(
                                    base_url=str(base_url),
                                    timeout_sec=int(args.request_timeout_sec),
                                    session_dir=_normalize_prefix_session_debug_root(
                                        session_dir=str(loaded.get("session_dir") or ""),
                                        session_id=str(session_id),
                                    ),
                                    session_id=str(session_id),
                                    linear_target_last_rows=linear_target_last_rows,
                                    global_target_rows=[
                                        int(value) for value in global_target_rows
                                    ],
                                    row_base_shift=str(
                                        row_hash_config.get("row_base_shift") or "auto"
                                    ),
                                )
                            except Exception as exc:  # noqa: BLE001
                                post_load_prefix_row_hashes = {
                                    "ok": False,
                                    "error": f"post_load_debug_prefix_state_row_hashes_failed:{exc}",
                                }
                    prefix_state_load_count += 1
                    _persist_restore_state_payload(
                        restore_state_results,
                        prepare_payload=prepare_payload,
                        post_load_prefix_row_hashes=post_load_prefix_row_hashes,
                    )
                    if (
                        mode in {"query", "build_and_query"}
                        and loaded_attention_mode == "full_attention"
                    ):
                        register_payload = _prefix_session_lib.register_loaded_prefix_cache(
                            base_url=base_url,
                            prefix_session_dir=str(session_root),
                            prefix_session_id=str(session_id),
                            timeout_sec=int(args.request_timeout_sec),
                            plan_id=str(prepare_payload.get("plan_id") or ""),
                        )
                        if not bool(register_payload.get("ok")):
                            raise SystemExit(
                                "register_loaded_prefix_cache failed: "
                                + str(register_payload)
                            )
                        _persist_register_state_payload(register_payload)

                # For exactness/debugging, build_and_query must re-enter the
                # query loop from the same disk-loaded checkpoint boundary as
                # restore/query. Reusing the live hot build state would compare
                # different control-plane starting points.
                if mode in {"restore", "query", "build_and_query"}:
                    _load_prefix_state_from_disk()

        if mode in {"query", "build_and_query"}:
            query_entries = _read_suffix_query_entries(
                suffix_query_file=str(args.suffix_query_file or ""),
                suffix_query_files=str(args.suffix_query_files or ""),
                suffix_query_dir=str(args.suffix_query_dir or ""),
                prefix_query_max_count=int(args.prefix_query_max_count or 0),
                prompt_shape_id=str(prompt_shape_meta.get("id") or ""),
                max_new_tokens=int(args.max_new_tokens),
            )
            reload_between_queries = bool(int(args.prefix_reload_between_queries))
            query_debug_dir = out_dir / "query_debug"
            query_debug_dir.mkdir(parents=True, exist_ok=True)
            for query_index, query_entry in enumerate(query_entries, start=1):
                if (
                    bool(int(args.prefix_save_shards))
                    and reload_between_queries
                    and query_index > 1
                ):
                    _load_prefix_state_from_disk()

                suffix_query_text = str(query_entry.get("text") or "")
                suffix_query_path = str(query_entry.get("path") or "")
                loaded_manifest = _prefix_session_lib.normalize_prefix_session_manifest(
                    dict(loaded.get("manifest") or {})
                )
                loaded_prefix_capture = _normalize_prefix_capture(
                    loaded_manifest.get("prefix_capture")
                )
                if str(loaded_manifest.get("attention_mode") or "") == "streaming_mla":
                    if (
                        str(loaded_prefix_capture.get("continuation_mode") or "")
                        == "same_message"
                    ):
                        query_messages = _prefix_session_lib.build_same_message_query_messages(
                            prefix_messages=[
                                dict(msg)
                                for msg in list(loaded.get("prefix_messages") or [])
                            ],
                            suffix_text=suffix_query_text,
                            message_index=int(
                                loaded_prefix_capture.get("message_index")
                                or max(
                                    0,
                                    len(list(loaded.get("prefix_messages") or [])) - 1,
                                )
                            ),
                        )
                        query_prompt_tokens = int(
                            len(_prefix_session_lib.tokenize_messages(tokenizer, query_messages))
                        )
                    else:
                        query_prompt_tokens = int(
                            len(list(loaded.get("prefix_token_ids") or []))
                            + len(
                                _prefix_session_lib.tokenize_text(
                                    tokenizer, str(suffix_query_text)
                                )
                            )
                            + len(
                                list(loaded.get("continuation_tail_token_ids") or [])
                            )
                        )
                elif str(loaded_prefix_capture.get("continuation_mode") or "") == "same_message":
                    query_messages = _prefix_session_lib.build_same_message_query_messages(
                        prefix_messages=[
                            dict(msg)
                            for msg in list(loaded.get("prefix_messages") or prefix_messages)
                        ],
                        suffix_text=suffix_query_text,
                        message_index=int(
                            loaded_prefix_capture.get("message_index")
                            or max(
                                0,
                                len(list(loaded.get("prefix_messages") or prefix_messages)) - 1,
                            )
                        ),
                    )
                    query_prompt_tokens = int(len(_tokenize_messages(tokenizer, query_messages)))
                else:
                    query_messages = [dict(msg) for msg in list(loaded.get("prefix_messages") or prefix_messages)]
                    query_messages.append({"role": "user", "content": suffix_query_text})
                    query_prompt_tokens = int(len(_tokenize_messages(tokenizer, query_messages)))
                query_recorder = _start_prefill_recorder(
                    target_index=int(query_index),
                    target_prompt_tokens_requested=int(query_prompt_tokens),
                    target_prompt_tokens_effective=int(query_prompt_tokens),
                    iteration=1,
                )
                try:
                    metrics, query_prompt_tokens, query_debug = _run_prefix_session_query_request(
                        base_url=base_url,
                        model_id=str(args.model_id),
                        tokenizer=tokenizer,
                        loaded_session=loaded,
                        fallback_prefix_messages=prefix_messages,
                        suffix_query_text=suffix_query_text,
                        max_new_tokens=int(args.max_new_tokens),
                        min_new_tokens=int(args.min_new_tokens),
                        timeout_sec=int(args.request_timeout_sec),
                        ignore_eos=bool(args.ignore_eos),
                        use_stream=bool(not args.no_stream),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                        repetition_penalty=float(args.repetition_penalty),
                        presence_penalty=float(args.presence_penalty),
                        frequency_penalty=float(args.frequency_penalty),
                        output_mask_payload=output_mask_payload,
                        live_layout_prefix_token_count=int(args.target_prompt_tokens),
                    )
                    if query_recorder is not None:
                        query_recorder.stop(
                            final_status="completed",
                            note=f"prefix_query_{query_index}_done",
                        )
                except Exception as exc:  # noqa: BLE001
                    if query_recorder is not None:
                        query_recorder.stop(
                            final_status="error",
                            note=f"prefix_query_{query_index}_error={str(exc)[:500]}",
                        )
                    raise
                generated_path = _persist_generated_text(
                    out_dir=out_dir,
                    target_index=int(query_index),
                    iteration=1,
                    generated_text=str(metrics.get("generated_text") or ""),
                )
                prefix_row_hashes_path = ""
                prefix_row_hashes = dict((query_debug or {}).get("prefix_row_hashes") or {})
                if prefix_row_hashes:
                    prefix_row_hashes_out = (
                        query_debug_dir
                        / f"query_{int(query_index):04d}_prefix_row_hashes.json"
                    )
                    prefix_row_hashes_out.write_text(
                        json.dumps(prefix_row_hashes, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    prefix_row_hashes_path = str(prefix_row_hashes_out)
                query_debug_to_write = dict(query_debug or {})
                query_debug_to_write.pop("prefix_row_hashes", None)
                if prefix_row_hashes_path:
                    query_debug_to_write["prefix_row_hashes_path"] = str(prefix_row_hashes_path)
                query_debug_path = (
                    query_debug_dir / f"query_{int(query_index):04d}_live_request_layout.json"
                )
                query_debug_path.write_text(
                    json.dumps(query_debug_to_write, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                row = {
                    "run_id": run_id,
                    "session_id": session_id,
                    "query_index": int(query_index),
                    "status": "ok",
                    "timestamp_utc": _now_iso(),
                    "suffix_query_file": suffix_query_path,
                    "suffix_query_text": suffix_query_text,
                    "suffix_query_sha256": _sha256_text(suffix_query_text),
                    "prefix_tokens_requested": int(args.target_prompt_tokens),
                    "prefix_tokens_effective": int(len(prefix_ids)),
                    "prefix_rendered_tokens": int(
                        len(loaded.get("prefix_token_ids") or prefix_token_ids)
                    ),
                    "prefix_state_load_count": int(prefix_state_load_count),
                    "prefix_state_reload_between_queries": bool(
                        int(args.prefix_reload_between_queries)
                    ),
                    "generated_text_path": generated_path,
                    "live_request_layout_path": str(query_debug_path),
                    "prefix_row_hashes_path": str(prefix_row_hashes_path),
                    "live_internal_request_id": str(
                        (query_debug or {}).get("internal_request_id") or ""
                    ),
                    "live_external_request_id": str(
                        (query_debug or {}).get("external_request_id") or ""
                    ),
                    "live_layout_ok": bool(
                        dict((query_debug or {}).get("layout") or {}).get("ok")
                    ),
                }
                row.update(metrics)
                row.pop("generated_text", None)
                query_rows.append(row)
            write_records(out_dir / "prefix_query_metrics.jsonl", query_rows)
            write_records(out_dir / "results.jsonl", query_rows)

        summary = {
            "ok": True,
            "run_id": run_id,
            "mode": mode,
            "session_id": session_id,
            "session_root": str(session_root),
            "query_count": int(len(query_rows)),
            "prefix_state_load_count": int(prefix_state_load_count),
            "artifacts": {
                "run_meta_json": str(out_dir / "run_meta.json"),
                "launch_env_json": str(out_dir / "launch_env.json"),
                "tokenizer_resolution_json": str(out_dir / "tokenizer_resolution.json")
                if (out_dir / "tokenizer_resolution.json").exists()
                else "",
                "server_cmd_txt": str(out_dir / "server_cmd.txt"),
                "graph_mode_status_json": str(out_dir / "graph_mode_status.json")
                if (out_dir / "graph_mode_status.json").exists()
                else "",
                "mla_decode_profile_jsonl": str(out_dir / "mla_decode_profile.jsonl")
                if (out_dir / "mla_decode_profile.jsonl").exists()
                else "",
                "prefix_capture_metrics_json": str(out_dir / "prefix_capture_metrics.json")
                if (out_dir / "prefix_capture_metrics.json").exists()
                else "",
                "prefix_state_calibration_json": str(out_dir / "prefix_state_calibration.json")
                if (out_dir / "prefix_state_calibration.json").exists()
                else "",
                "prefix_restore_state_rpc_json": str(out_dir / "prefix_restore_state_rpc.json")
                if (out_dir / "prefix_restore_state_rpc.json").exists()
                else "",
                "prefix_restore_metrics_json": str(out_dir / "prefix_restore_metrics.json")
                if (out_dir / "prefix_restore_metrics.json").exists()
                else "",
                "prefix_query_metrics_jsonl": str(out_dir / "prefix_query_metrics.jsonl")
                if (out_dir / "prefix_query_metrics.jsonl").exists()
                else "",
                "prefill_progress_jsonl": str(prefill_progress_path)
                if bool(args.emit_prefill_progress) and prefill_progress_path.exists()
                else "",
                "results_jsonl": str(out_dir / "results.jsonl")
                if (out_dir / "results.jsonl").exists()
                else "",
                "generated_text_dir": str(out_dir / "generated_text")
                if (out_dir / "generated_text").exists()
                else "",
            },
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "run_id": run_id, "out_dir": str(out_dir), "summary": summary}, sort_keys=True))
        return 0
    finally:
        _stop_process(proc)
        if proc is not None:
            cleanup_result = _cleanup_server_compile_cache(
                out_dir,
                str(getattr(args, "post_run_cache_cleanup", "none")),
            )
            _write_compile_cache_cleanup_artifact(out_dir, cleanup_result)
            if cleanup_result.get("errors"):
                print(
                    json.dumps(
                        {
                            "warning": "server_compile_cache_cleanup_failed",
                            "details": cleanup_result,
                        },
                        sort_keys=True,
                    )
                )


def main() -> int:
    """Run the benchmark CLI and write phase artifacts to disk."""

    args = _parse_args()
    try:
        prompt_shape_spec, prompt_shape_meta = load_prompt_shape_spec(
            prompt_shape=str(args.prompt_shape),
            prompt_shape_file=str(args.prompt_shape_file),
        )
    except PromptShapeError as exc:
        raise SystemExit(f"Prompt-shape validation failed: {exc}") from exc

    run_id = str(args.run_id or _timestamp_run_id())
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else repo_path(
        "experiments", "hf_long_context", "runs", run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_targets = _parse_sweep_targets(args, default_target=int(args.target_prompt_tokens))
    extra_targets = _parse_token_list(str(getattr(args, "extra_prompt_tokens", "") or ""))
    sweep_targets = sorted(set([*sweep_targets, *extra_targets]))
    base_target_tokens = max(sweep_targets) if sweep_targets else max(1, int(args.target_prompt_tokens))

    tokenizer_id = str(args.tokenizer_id or args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model_config = AutoConfig.from_pretrained(
        str(args.model_id),
        trust_remote_code=bool(args.trust_remote_code),
    )
    chat_template_hash = _chat_template_hash(tokenizer)
    tokenizer_resolution = _build_tokenizer_resolution_payload(
        tokenizer_id=tokenizer_id,
        tokenizer=tokenizer,
        model_config=model_config,
        chat_template_hash=chat_template_hash,
    )
    (out_dir / "tokenizer_resolution.json").write_text(
        json.dumps(tokenizer_resolution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_mask_meta = _build_english_only_output_mask(
        tokenizer,
        mode=str(args.english_only_mask_mode),
        bias_value=float(args.english_only_mask_bias),
        vocab_size_limit=getattr(model_config, "vocab_size", None),
    )
    output_mask_payload = dict(output_mask_meta.get("payload") or {})
    exact_prompt_token_ids_active = bool(str(args.prompt_token_ids_file or "").strip())
    if exact_prompt_token_ids_active:
        prompt_seed_ids, exact_prompt_meta = _load_prompt_token_ids_file(
            str(args.prompt_token_ids_file)
        )
        substrate_prompt_meta = {
            "source": "prompt_token_ids_file",
            "prompt_token_ids_file": str(exact_prompt_meta.get("path") or ""),
            "prompt_seed_extension_mode": "none",
            "prompt_seed_repeat_policy": "exact_prompt_ids_file",
            "prompt_seed_source_tokens": int(len(prompt_seed_ids)),
            "prompt_seed_expanded_tokens": int(len(prompt_seed_ids)),
            "prompt_seed_repeat_count": 1,
            "requested_prompt_tokens": int(base_target_tokens),
            "artifact_format": str(exact_prompt_meta.get("artifact_format") or ""),
            "sha256": str(exact_prompt_meta.get("sha256") or ""),
        }
    else:
        include_exts = _resolve_include_exts(str(args.substrate_include_exts))
        prompt_seed_ids, substrate_prompt_meta = build_substrate_prompt_seed_from_args(
            tokenizer=tokenizer,
            args=args,
            target_tokens=base_target_tokens,
            include_exts=include_exts,
            char_budget_multiplier=float(args.substrate_char_budget_multiplier),
            allow_unlimited_files=True,
            allow_unlimited_chars=True,
            decode_to_text=False,
            context_markup=str(prompt_shape_meta.get("context_markup") or ""),
        )
        if not prompt_seed_ids:
            raise SystemExit("Could not build substrate prompt seed.")
        # All Make-launched sweeps respect the exact request: pad the seed up
        # to ``target_tokens`` even when a truncated_file_path is present.
        # Previously a truncated file disabled repeat_to_target and yielded
        # effective ``prompt_tokens`` slightly below the requested power of 2
        # (e.g. 1024 -> 1018), which made comparison plots' x-ticks fall off
        # round powers of 2.
        had_truncated_file = bool(substrate_prompt_meta.get("truncated_file_path"))
        repeat_prompt_seed = True
        prompt_seed_ids, prompt_seed_expansion_meta = _expand_prompt_seed_ids_to_target(
            list(prompt_seed_ids),
            target_tokens=base_target_tokens,
            repeat_to_target=repeat_prompt_seed,
        )
        substrate_prompt_meta.update(prompt_seed_expansion_meta)
        substrate_prompt_meta["prompt_seed_repeat_policy"] = "repeat_to_target"
        substrate_prompt_meta["prompt_seed_repeat_policy_overrode_truncated_tail"] = (
            had_truncated_file
        )
        substrate_prompt_meta["requested_prompt_tokens"] = int(base_target_tokens)

    resolved_vllm_hf_overrides = _resolve_kimi_mla_hf_overrides(args)
    strict_env_inputs = _strict_env_payload(
        vllm_hf_overrides=resolved_vllm_hf_overrides
    )
    launch_env = {
        "HIP_VISIBLE_DEVICES": str(os.environ.get("HIP_VISIBLE_DEVICES", "")),
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": str(
            os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "")
        ),
        "VLLM_ROCM_USE_SKINNY_GEMM": (
            "0"
            if bool(getattr(args, "disable_rocm_skinny_gemm", False))
            else str(os.environ.get("VLLM_ROCM_USE_SKINNY_GEMM", ""))
        ),
        "VLLM_AITER_HEAD4_DECODE_MODE": str(
            os.environ.get("VLLM_AITER_HEAD4_DECODE_MODE", "")
        ),
        "VLLM_AITER_HEAD4_FALLBACK_MODE": str(
            os.environ.get("VLLM_AITER_HEAD4_FALLBACK_MODE", "")
        ),
        "VLLM_AITER_SUB16_MODE": str(
            os.environ.get("VLLM_AITER_SUB16_MODE", "")
        ),
        "VLLM_AITER_SUB16_FALLBACK": str(
            os.environ.get("VLLM_AITER_SUB16_FALLBACK", "")
        ),
        "VLLM_AITER_SUB16_COMPUTE_MODE": str(
            os.environ.get("VLLM_AITER_SUB16_COMPUTE_MODE", "")
        ),
        "VLLM_AITER_SUB16_APPLY": str(
            os.environ.get("VLLM_AITER_SUB16_APPLY", "")
        ),
        "VLLM_AITER_SUB16_BF16_KV_SCALE_MODE": str(
            os.environ.get("VLLM_AITER_SUB16_BF16_KV_SCALE_MODE", "")
        ),
        "VLLM_AITER_SUB16_SHADOW_CHECK": str(
            os.environ.get("VLLM_AITER_SUB16_SHADOW_CHECK", "")
        ),
        "VLLM_AITER_SUB16_SHADOW_MAX_ABS_ERR": str(
            os.environ.get("VLLM_AITER_SUB16_SHADOW_MAX_ABS_ERR", "")
        ),
        "VLLM_AITER_SUB16_SHADOW_MAX_REL_ERR": str(
            os.environ.get("VLLM_AITER_SUB16_SHADOW_MAX_REL_ERR", "")
        ),
        "VLLM_AITER_SUB16_BF16_WORKSPACE_MAX_BYTES": str(
            os.environ.get("VLLM_AITER_SUB16_BF16_WORKSPACE_MAX_BYTES", "")
        ),
        "VLLM_AITER_SUB16_BF16_WORKSPACE_CHUNK_ROWS": str(
            os.environ.get("VLLM_AITER_SUB16_BF16_WORKSPACE_CHUNK_ROWS", "")
        ),
        "VLLM_AITER_SUB16_FAST_MAX_KV_TOKENS": str(
            os.environ.get("VLLM_AITER_SUB16_FAST_MAX_KV_TOKENS", "")
        ),
        "VLLM_AITER_MLA_NUM_KV_SPLITS_OVERRIDE": str(
            os.environ.get("VLLM_AITER_MLA_NUM_KV_SPLITS_OVERRIDE", "")
        ),
        "VLLM_AITER_MLA_STAGE1_REF_QH16_Q1": str(
            os.environ.get("VLLM_AITER_MLA_STAGE1_REF_QH16_Q1", "")
        ),
        "VLLM_AITER_MLA_STAGE1_REF_QH16_Q1_STRICT_FAIL": str(
            os.environ.get("VLLM_AITER_MLA_STAGE1_REF_QH16_Q1_STRICT_FAIL", "")
        ),
        "VLLM_AITER_MLA_STAGE1_MODE": str(
            os.environ.get("VLLM_AITER_MLA_STAGE1_MODE", "")
        ),
        "VLLM_AITER_MLA_PROFILE": str(
            os.environ.get("VLLM_AITER_MLA_PROFILE", "")
        ),
        "VLLM_AITER_MLA_PROFILE_MAX_STEPS": str(
            os.environ.get("VLLM_AITER_MLA_PROFILE_MAX_STEPS", "")
        ),
        "VLLM_AITER_MLA_STAGE1_MFMA_MIN_TOKENS": str(
            os.environ.get("VLLM_AITER_MLA_STAGE1_MFMA_MIN_TOKENS", "")
        ),
        "VLLM_AITER_MLA_STAGE1_MFMA_TARGET_TOKENS": str(
            os.environ.get("VLLM_AITER_MLA_STAGE1_MFMA_TARGET_TOKENS", "")
        ),
        "VLLM_AITER_MLA_STAGE1_MFMA_IMPL": str(
            os.environ.get("VLLM_AITER_MLA_STAGE1_MFMA_IMPL", "")
        ),
        "VLLM_AITER_MLA_HIP_LDS_STAGE12_TREE": str(
            os.environ.get("VLLM_AITER_MLA_HIP_LDS_STAGE12_TREE", "")
        ),
        "VLLM_AITER_MLA_HIP_LDS_VARIANT": str(
            os.environ.get("VLLM_AITER_MLA_HIP_LDS_VARIANT", "")
        ),
        "VLLM_AITER_MLA_STAGE1_WEIGHTED_V_MODE": str(
            os.environ.get("VLLM_AITER_MLA_STAGE1_WEIGHTED_V_MODE", "")
        ),
        "VLLM_AITER_MLA_STAGE1_MFMA_BLOCK_N": str(
            os.environ.get("VLLM_AITER_MLA_STAGE1_MFMA_BLOCK_N", "")
        ),
        "VLLM_AITER_MLA_TARGET_TOKENS_PER_SPLIT": str(
            os.environ.get("VLLM_AITER_MLA_TARGET_TOKENS_PER_SPLIT", "")
        ),
        "VLLM_AITER_MLA_STAGE2_MODE": str(
            os.environ.get("VLLM_AITER_MLA_STAGE2_MODE", "")
        ),
        "VLLM_ROCM_AITER_MLA_USE_FLASHINFER_DECODE": str(
            os.environ.get("VLLM_ROCM_AITER_MLA_USE_FLASHINFER_DECODE", "")
        ),
        "FLASHINFER_HIP_MLA_USE_AITER_FASTPATH": str(
            os.environ.get("FLASHINFER_HIP_MLA_USE_AITER_FASTPATH", "")
        ),
        "FLASHINFER_HIP_MLA_USE_REPO_AITER_SNAPSHOT": str(
            os.environ.get("FLASHINFER_HIP_MLA_USE_REPO_AITER_SNAPSHOT", "")
        ),
        "FLASHINFER_HIP_MLA_GRAPH_KV_HEADROOM": str(
            os.environ.get("FLASHINFER_HIP_MLA_GRAPH_KV_HEADROOM", "")
        ),
        "HFLC_RUN_DIR": str(os.environ.get("HFLC_RUN_DIR", "")),
        "HFLC_PHASE_DIR": str(os.environ.get("HFLC_PHASE_DIR", "")),
        "HFLC_PHASE_NAME": str(os.environ.get("HFLC_PHASE_NAME", "")),
        "HFLC_WRAPPER_ENTRYPOINT": str(os.environ.get("HFLC_WRAPPER_ENTRYPOINT", "")),
        "HFLC_WRAPPER_CHAIN": str(os.environ.get("HFLC_WRAPPER_CHAIN", "")),
        "VLLM_MLA_FP8_DECODE_BYPASS_QUANT": str(
            os.environ.get("VLLM_MLA_FP8_DECODE_BYPASS_QUANT", "")
        ),
        "VLLM_ATTENTION_BACKEND": str(
            os.environ.get("VLLM_ATTENTION_BACKEND", "")
        ),
        "VLLM_COMPILATION_CONFIG": _normalize_compilation_config_json(
            _effective_vllm_compilation_config(args)
        ),
        "VLLM_SERVER_DEV_MODE": str(os.environ.get("VLLM_SERVER_DEV_MODE", "")),
        "HFLC_FP8_ROCPROF_ENABLE": str(os.environ.get("HFLC_FP8_ROCPROF_ENABLE", "")),
        "HFLC_FP8_ROCPROF_MODE": str(os.environ.get("HFLC_FP8_ROCPROF_MODE", "")),
        "HFLC_FP8_ROCPROF_STATS": str(os.environ.get("HFLC_FP8_ROCPROF_STATS", "")),
        "HFLC_FP8_ROCPROF_TIMESTAMP": str(os.environ.get("HFLC_FP8_ROCPROF_TIMESTAMP", "")),
        "HFLC_FP8_ROCPROF_TOOL_VERSION": str(os.environ.get("HFLC_FP8_ROCPROF_TOOL_VERSION", "")),
        "HFLC_FP8_ROCPROF_INPUT": str(os.environ.get("HFLC_FP8_ROCPROF_INPUT", "")),
        "PYTHONPATH": str(os.environ.get("PYTHONPATH", "")),
    }

    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": _now_iso(),
        "mode": "vllm_openai_api",
        "model_id": str(args.model_id),
        "tokenizer_id": tokenizer_id,
        "chat_template_hash": chat_template_hash,
        "tokenizer_resolution": dict(tokenizer_resolution),
        "prompt_source": str(args.prompt_source),
        "system_prompt": str(args.system_prompt),
        "query_api_mode": str(args.query_api_mode),
        "prompt_shape": {
            "id": str(prompt_shape_meta.get("id") or str(args.prompt_shape)),
            "source": str(prompt_shape_meta.get("source") or "preset"),
            "file": str(prompt_shape_meta.get("file") or ""),
            "template_sha256": str(prompt_shape_meta.get("template_sha256") or ""),
            "context_markup": str(prompt_shape_meta.get("context_markup") or ""),
            "prefix_capture": dict(prompt_shape_meta.get("prefix_capture") or {}),
            "validation": {
                "status": str(
                    (prompt_shape_meta.get("validation") or {}).get("status")
                    or "ok"
                ),
                "message": str(
                    (prompt_shape_meta.get("validation") or {}).get("message")
                    or "validated"
                ),
            },
        },
        "prefix_session": {
            "mode": str(args.prefix_session_mode or "none"),
            "session_id": str(args.prefix_session_id or ""),
            "session_dir": str(args.prefix_session_dir or ""),
            "compat_strict": bool(int(args.prefix_compat_strict)),
            "suffix_query_file": str(args.suffix_query_file or ""),
            "suffix_query_files": str(args.suffix_query_files or ""),
            "suffix_query_dir": str(args.suffix_query_dir or ""),
            "prefix_query_max_count": int(args.prefix_query_max_count or 0),
            "prefix_reload_between_queries": bool(
                int(args.prefix_reload_between_queries or 0)
            ),
            "prefix_max_tokens": int(args.prefix_max_tokens or 0),
            "prefix_save_shards": bool(int(args.prefix_save_shards)),
            "prefix_target_tp": int(args.prefix_target_tp or 0),
            "prefix_state_chunk_bytes": int(args.prefix_state_chunk_bytes or 0),
            "prefix_state_format": "kvstate_bin_v1",
            "prefix_state_calibration_strict_gate": bool(
                int(args.prefix_state_calibration_strict_gate)
            ),
            "prefix_state_calibration_safety_ratio": float(
                args.prefix_state_calibration_safety_ratio
            ),
            "prefix_state_calibration_safety_min_bytes": int(
                args.prefix_state_calibration_safety_min_bytes or 0
            ),
        },
        "sweep_prompt_tokens": [int(x) for x in sweep_targets],
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "ignore_eos": bool(args.ignore_eos),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "repetition_penalty": float(args.repetition_penalty),
        "presence_penalty": float(args.presence_penalty),
        "frequency_penalty": float(args.frequency_penalty),
        "english_only_output_mask": {
            "mode": str(output_mask_meta.get("mode") or "off"),
            "bias_value": output_mask_meta.get("bias_value"),
            "cjk_banned_count": int(output_mask_meta.get("cjk_banned_count") or 0),
            "allowed_token_count": output_mask_meta.get("allowed_token_count"),
            "logit_bias_count": int(output_mask_meta.get("logit_bias_count") or 0),
        },
        "request_debug": {
            "dump_request_payload": bool(args.dump_request_payload),
            "dump_query_messages": bool(args.dump_query_messages),
            "dump_prompt_token_ids": bool(args.dump_prompt_token_ids),
        },
        "request_stream": bool(not args.no_stream),
        "num_runs": int(args.num_runs),
        "save_generated_text": bool(args.save_generated_text),
        "calibrate_kv_capacity": bool(args.calibrate_kv_capacity),
        "calibration_only": bool(args.calibration_only),
        "calibration_strict_gate": bool(args.calibration_strict_gate),
        "calibration_headroom_ratio": float(args.calibration_headroom_ratio),
        "calibration_headroom_min_blocks": int(args.calibration_headroom_min_blocks),
        "calibration_assumed_max_num_seqs": int(args.calibration_assumed_max_num_seqs),
        "calibration_metrics_timeout_sec": int(args.calibration_metrics_timeout_sec),
        "measure_k2ft": bool(args.measure_k2ft),
        "k2ft_method": "two_stage_cold_hot_ttft_delta",
        "k2ft_delta_tokens": int(args.k2ft_delta_tokens),
        "k2ft_min_probe_tokens": int(args.k2ft_min_probe_tokens),
        "k2ft_runs": int(args.k2ft_runs),
        "emit_prefill_progress": bool(args.emit_prefill_progress),
        "prefill_progress_interval_sec": float(args.prefill_progress_interval_sec),
        "prefill_progress_timeout_sec": int(args.prefill_progress_timeout_sec),
        "request_timeout_sec_base": int(args.request_timeout_sec),
        "adaptive_request_timeout": bool(args.adaptive_request_timeout),
        "adaptive_timeout_scale": float(args.adaptive_timeout_scale),
        "adaptive_timeout_extra_sec": float(args.adaptive_timeout_extra_sec),
        "adaptive_timeout_cap_sec": int(args.adaptive_timeout_cap_sec),
        "timeout_sclk_guard": bool(args.timeout_sclk_guard),
        "timeout_sclk_min_mhz": float(args.timeout_sclk_min_mhz),
        "timeout_sclk_sample_count": int(args.timeout_sclk_sample_count),
        "timeout_sclk_sample_interval_sec": float(args.timeout_sclk_sample_interval_sec),
        "timeout_sclk_min_busy_gpus": int(args.timeout_sclk_min_busy_gpus),
        "timeout_sclk_required_hit_ratio": float(args.timeout_sclk_required_hit_ratio),
        "timeout_sclk_max_extensions": int(args.timeout_sclk_max_extensions),
        "timeout_sclk_extension_sec": int(args.timeout_sclk_extension_sec),
        "out_dir": str(out_dir),
        "strict_env_hash": "",
        "strict_env_inputs": dict(strict_env_inputs),
        "vllm": {
            "bin": str(args.vllm_bin),
            "host": str(args.vllm_host),
            "port": int(args.vllm_port),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "data_parallel_size": int(args.vllm_data_parallel_size),
            "pipeline_parallel_size": int(args.pipeline_parallel_size),
            "max_model_len": int(args.vllm_max_model_len),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "dtype": str(args.vllm_dtype),
            "kv_cache_dtype": str(args.kv_cache_dtype),
            "load_format": str(args.vllm_load_format or ""),
            "hf_config_path": str(args.vllm_hf_config_path or ""),
            "hf_overrides": dict(resolved_vllm_hf_overrides),
            "compilation_config": _effective_vllm_compilation_config(args),
            "mla_runtime_mode": str(args.vllm_mla_runtime_mode or "aiter"),
            "worker_extension_cls": _resolved_vllm_worker_extension_cls(args),
            "enforce_eager": bool(args.vllm_enforce_eager),
            "calculate_kv_scales": bool(args.calculate_kv_scales),
            "max_num_seqs": int(args.vllm_max_num_seqs or 0),
            "max_num_batched_tokens": int(args.vllm_max_num_batched_tokens or 0),
            "attention_backend": str(args.vllm_attention_backend or ""),
            "batch_invariant": bool(int(args.vllm_batch_invariant or 0)),
            "preserve_legacy_aiter_env": bool(
                int(args.vllm_preserve_legacy_aiter_env or 0)
            ),
            "block_size": int(args.vllm_block_size or 0),
            "kv_cache_memory_bytes": int(args.vllm_kv_cache_memory_bytes or 0),
            "num_gpu_blocks_override": int(args.vllm_num_gpu_blocks_override or 0),
            "mla_fp8_decode_bypass_quant_env": str(
                os.environ.get("VLLM_MLA_FP8_DECODE_BYPASS_QUANT", "")
            ),
            "trust_remote_code": bool(args.trust_remote_code),
        },
        "substrate_prompt": dict(substrate_prompt_meta),
        "exact_prompt_token_ids": (
            {
                "enabled": bool(exact_prompt_token_ids_active),
                "path": str(args.prompt_token_ids_file or ""),
                "count": int(len(prompt_seed_ids)) if exact_prompt_token_ids_active else 0,
            }
        ),
        "launch_env": dict(launch_env),
    }
    run_meta["host_env"] = collect_host_env()
    run_meta["mla_kernel_fingerprint"] = collect_mla_kernel_fingerprint()
    run_meta["wrapper"] = {
        "entrypoint": str(os.environ.get("HFLC_WRAPPER_ENTRYPOINT", "")),
        "chain": [
            part for part in str(os.environ.get("HFLC_WRAPPER_CHAIN", "")).split("|") if part
        ],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "launch_env.json").write_text(
        json.dumps(launch_env, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if exact_prompt_token_ids_active and str(args.prefix_session_mode or "none") != "none":
        raise SystemExit(
            "--prompt-token-ids-file is not supported with --prefix-session-mode"
        )

    if str(args.prefix_session_mode or "none") != "none":
        return _run_prefix_session_mode(
            args=args,
            run_id=run_id,
            out_dir=out_dir,
            run_meta=run_meta,
            prompt_shape_spec=prompt_shape_spec,
            prompt_shape_meta=prompt_shape_meta,
            tokenizer=tokenizer,
            prompt_seed_ids=prompt_seed_ids,
            sweep_targets=sweep_targets,
            output_mask_payload=output_mask_payload,
        )

    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    prefill_progress_path = out_dir / "prefill_progress.jsonl"
    if bool(args.emit_prefill_progress) and prefill_progress_path.exists():
        prefill_progress_path.unlink()
    if bool(args.emit_prefill_progress):
        _append_jsonl_row(
            prefill_progress_path,
            {
                "timestamp_utc": _now_iso(),
                "target_index": 0,
                "target_prompt_tokens_requested": 0,
                "target_prompt_tokens_effective": 0,
                "iteration": 0,
                "request_id": "",
                "prefill_done_tokens": 0,
                "prefill_total_tokens": 0,
                "prefill_pct": 0.0,
                "engine_step": 0,
                "source": "vllm_prefill_progress_endpoint",
                "status": "initialized",
                "note": "run_initialized_before_requests",
            },
        )

    proc: Optional[subprocess.Popen[Any]] = None
    try:
        proc, log_path, cmd, server_env = _start_vllm_server(args, out_dir)
        _sync_launch_env_artifacts(
            out_dir=out_dir,
            run_meta=run_meta,
            launch_env=dict(run_meta.get("launch_env") or {}),
            server_env=server_env,
        )
        (out_dir / "server_cmd.txt").write_text(
            _render_server_cmd_artifact(
                cmd,
                server_env,
                vllm_hf_overrides=dict(
                    ((run_meta.get("vllm") or {}).get("hf_overrides") or {})
                ),
            ),
            encoding="utf-8",
        )
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        if not _wait_health(base_url, int(args.server_ready_timeout_sec), proc=proc):
            _persist_graph_mode_status(
                out_dir=out_dir,
                run_meta=run_meta,
                compilation_config_raw=_effective_vllm_compilation_config(args),
                log_path=log_path,
            )
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            except Exception:
                pass
            proc_rc = proc.poll() if proc is not None else None
            raise SystemExit(
                f"vLLM did not become healthy (process_rc={proc_rc}). Log tail:\n{tail}"
            )
        graph_mode_status = _persist_graph_mode_status(
            out_dir=out_dir,
            run_meta=run_meta,
            compilation_config_raw=_effective_vllm_compilation_config(args),
            log_path=log_path,
        )
        _maybe_require_graph_capture(status=graph_mode_status)

        calibration_exit = _maybe_run_kv_calibration(
            args=args,
            run_id=run_id,
            out_dir=out_dir,
            run_meta=run_meta,
            log_path=log_path,
            sweep_targets_requested=[int(x) for x in sweep_targets],
            prompt_seed_len=int(len(prompt_seed_ids)),
            prefill_progress_path=(
                prefill_progress_path if bool(args.emit_prefill_progress) else None
            ),
        )
        if calibration_exit is not None:
            return int(calibration_exit)

        direct_query_mode_enabled = bool(
            str(args.suffix_query_file or "").strip()
            or str(args.suffix_query_files or "").strip()
            or str(args.suffix_query_dir or "").strip()
        )
        direct_query_entries: List[Dict[str, str]] = []
        if direct_query_mode_enabled:
            direct_query_entries = _read_suffix_query_entries(
                suffix_query_file=str(args.suffix_query_file or ""),
                suffix_query_files=str(args.suffix_query_files or ""),
                suffix_query_dir=str(args.suffix_query_dir or ""),
                prefix_query_max_count=int(args.prefix_query_max_count or 0),
                prompt_shape_id=str(prompt_shape_meta.get("id") or ""),
                max_new_tokens=int(args.max_new_tokens),
            )
        effective_direct_query_entries = (
            direct_query_entries if direct_query_entries else [{"text": "", "path": ""}]
        )

        # No pre-measurement warm-up: GPU clocks were observed to run at
        # near-max for every measured request, so warm-up was not preventing
        # throttling; small-prompt timing noise is handled instead by repeating
        # cheap points and reporting the median (HFLC_FP8_SMALL_POINT_NUM_RUNS).
        # warmup_summary/warmup_rows_path are retained as empty for run_meta
        # schema stability.
        warmup_rows_path = out_dir / "warmup_rows.jsonl"
        warmup_rows: List[Dict[str, Any]] = []
        warmup_summary: Dict[str, Any] = {"enabled": False}

        measure_runs = max(1, int(args.num_runs))
        total_iters = len(sweep_targets) * len(effective_direct_query_entries) * measure_runs
        use_progress = (not bool(args.no_progress)) and TQDM_PROGRESS is not None
        progress = (
            TQDM_PROGRESS(
                total=total_iters,
                desc=f"vllm-long-context {run_id}",
                dynamic_ncols=True,
            )
            if use_progress
            else None
        )

        timeout_base_sec = max(1, int(args.request_timeout_sec))
        timeout_scale = max(0.0, float(args.adaptive_timeout_scale))
        timeout_extra_sec = max(0.0, float(args.adaptive_timeout_extra_sec))
        timeout_cap_sec = max(0, int(args.adaptive_timeout_cap_sec))
        timeout_sclk_guard = bool(args.timeout_sclk_guard)
        timeout_sclk_min_mhz = max(0.0, float(args.timeout_sclk_min_mhz))
        timeout_sclk_sample_count = max(1, int(args.timeout_sclk_sample_count))
        timeout_sclk_sample_interval_sec = max(0.0, float(args.timeout_sclk_sample_interval_sec))
        timeout_sclk_min_busy_gpus = int(args.timeout_sclk_min_busy_gpus)
        timeout_sclk_required_hit_ratio = min(1.0, max(0.0, float(args.timeout_sclk_required_hit_ratio)))
        timeout_sclk_max_extensions = max(0, int(args.timeout_sclk_max_extensions))
        timeout_sclk_extension_sec = max(1, int(args.timeout_sclk_extension_sec))
        timeout_sclk_expected_gpu_count = max(
            1,
            int(args.tensor_parallel_size) * max(1, int(args.pipeline_parallel_size)),
        )
        prev_success_tokens = 0
        prev_success_ttft_ms = 0.0

        rows: List[Dict[str, Any]] = []
        for target_idx, target_requested in enumerate(sweep_targets):
            target_effective = min(int(target_requested), len(prompt_seed_ids))
            target_truncated = bool(target_effective < int(target_requested))
            prompt_ids = prompt_seed_ids[:target_effective]
            timeout_policy = "fixed"
            timeout_predicted_ttft_ms = 0.0
            request_timeout_sec_effective = int(timeout_base_sec)
            if (
                bool(args.adaptive_request_timeout)
                and prev_success_tokens > 0
                and prev_success_ttft_ms > 0.0
                and target_effective > 0
            ):
                ratio = float(target_effective) / float(prev_success_tokens)
                timeout_predicted_ttft_ms = float(prev_success_ttft_ms) * ratio * ratio
                predicted_wait_sec = (timeout_predicted_ttft_ms / 1000.0) * timeout_scale + timeout_extra_sec
                request_timeout_sec_effective = max(timeout_base_sec, int(math.ceil(predicted_wait_sec)))
                if timeout_cap_sec > 0:
                    request_timeout_sec_effective = min(request_timeout_sec_effective, timeout_cap_sec)
                timeout_policy = "adaptive_quadratic_from_prev_target"

            target_rows: List[Dict[str, Any]] = []
            for query_index, query_entry in enumerate(
                effective_direct_query_entries,
                start=1,
            ):
                suffix_query_text = str(query_entry.get("text") or "")
                suffix_query_path = str(query_entry.get("path") or "")
                artifact_target_index = (
                    int(target_idx) * int(len(effective_direct_query_entries))
                    + int(query_index)
                )
                for iter_idx in range(measure_runs):
                    row = make_iteration_measurement_row(
                        run_id=run_id,
                        target_idx=target_idx,
                        target_requested=int(target_requested),
                        target_effective=int(target_effective),
                        iter_idx=iter_idx,
                        model_id=str(args.model_id),
                    )
                    row.update(
                        {
                            "target_prompt_tokens_truncated": target_truncated,
                            "finish_reason": "",
                            "stop_reason": "",
                            "request_timeout_sec_base": int(timeout_base_sec),
                            "request_timeout_sec_effective": int(
                                request_timeout_sec_effective
                            ),
                            "request_timeout_policy": timeout_policy,
                            "request_timeout_predicted_ttft_ms": round(
                                float(timeout_predicted_ttft_ms), 6
                            ),
                            "request_timeout_attempts": 0,
                            "request_timeout_extensions": 0,
                            "request_timeout_final_timeout_sec": int(
                                request_timeout_sec_effective
                            ),
                            "request_timeout_sclk_guard_enabled": bool(
                                timeout_sclk_guard
                            ),
                            "request_timeout_sclk_guard_event_count": 0,
                            "request_timeout_sclk_guard_last_action": "",
                            "request_timeout_sclk_guard_last_busy": False,
                            "request_timeout_sclk_guard_last_busy_gpus": 0,
                            "request_timeout_sclk_guard_last_peak_sclk_mhz": 0.0,
                            "request_timeout_sclk_guard_events": [],
                            "k2ft_enabled": bool(args.measure_k2ft),
                            "k2ft_method": "two_stage_cold_hot_ttft_delta",
                            "k2ft_probe_formula": "",
                            "k2ft_probe_tokens": 0,
                            "k2ft_runs_configured": max(2, int(args.k2ft_runs)),
                            "k2ft_runs_completed": 0,
                            "k2ft_cold_ttft_ms": 0.0,
                            "k2ft_hot_ttft_ms": 0.0,
                            "k2ft_ms": 0.0,
                            "prefill_est_ms": 0.0,
                            "k2ft_cached_prompt_tokens_cold": 0,
                            "k2ft_cached_prompt_tokens_hot": 0,
                            "k2ft_cache_hit_confidence": "",
                            "k2ft_status": "disabled",
                            "k2ft_error": "",
                            "live_request_layout_path": "",
                            "live_internal_request_id": "",
                            "live_external_request_id": "",
                            "live_layout_ok": False,
                            "suffix_query_file": suffix_query_path,
                            "suffix_query_sha256": _sha256_text(suffix_query_text)
                            if suffix_query_text
                            else "",
                            "suffix_query_index": (
                                int(query_index) if direct_query_entries else 0
                            ),
                        }
                    )
                    if target_effective <= 0:
                        row["status"] = "error"
                        row["error"] = "target_effective<=0"
                    elif target_effective + int(args.max_new_tokens) > int(
                        args.vllm_max_model_len
                    ):
                        row["status"] = "skipped_max_context"
                        row["error"] = (
                            f"prompt_tokens({target_effective})+max_new_tokens({int(args.max_new_tokens)})"
                            f">{int(args.vllm_max_model_len)}"
                        )
                    else:
                        direct_query_messages: Optional[List[Dict[str, str]]] = None
                        direct_prompt_token_count: int
                        if direct_query_entries:
                            (
                                direct_query_messages,
                                direct_prompt_token_count,
                            ) = _build_direct_query_messages(
                                tokenizer=tokenizer,
                                prompt_ids=prompt_ids,
                                prompt_shape_spec=prompt_shape_spec,
                                prompt_shape_id=str(
                                    prompt_shape_meta.get("id") or str(args.prompt_shape)
                                ),
                                system_prompt=str(args.system_prompt),
                                max_new_tokens=int(args.max_new_tokens),
                                model_id=str(args.model_id),
                                suffix_query_text=suffix_query_text,
                            )
                        else:
                            if bool(exact_prompt_token_ids_active):
                                direct_prompt_token_count = int(len(prompt_ids))
                            else:
                                direct_prompt_token_count = _direct_request_prompt_token_count(
                                    tokenizer=tokenizer,
                                    prompt_ids=prompt_ids,
                                    prompt_shape_spec=prompt_shape_spec,
                                    system_prompt=str(args.system_prompt),
                                    max_new_tokens=int(args.max_new_tokens),
                                    model_id=str(args.model_id),
                                )

                        prefill_recorder: Optional[_PrefillProgressRecorder] = None
                        prefill_recorder_stopped = False
                        request_debug_payload: Dict[str, Any] = {}
                        request_live_layout_enabled = _query_live_layout_enabled()
                        request_dump_enabled = bool(args.dump_request_payload) or bool(
                            args.dump_query_messages
                        ) or bool(args.dump_prompt_token_ids)
                        stream_progress_cb: Optional[Any] = None

                        if bool(args.emit_prefill_progress):
                            prefill_recorder = _PrefillProgressRecorder(
                                base_url=base_url,
                                out_path=prefill_progress_path,
                                target_index=int(artifact_target_index),
                                target_prompt_tokens_requested=int(
                                    row["target_prompt_tokens_requested"]
                                ),
                                target_prompt_tokens_effective=int(
                                    row["target_prompt_tokens_effective"]
                                ),
                                iteration=int(row["iteration"]),
                                interval_sec=float(args.prefill_progress_interval_sec),
                                poll_timeout_sec=float(
                                    int(args.prefill_progress_timeout_sec)
                                    if int(args.prefill_progress_timeout_sec) > 0
                                    else int(request_timeout_sec_effective)
                                ),
                            )
                            prefill_recorder.start()
                        if (
                            bool(request_live_layout_enabled)
                            or bool(request_dump_enabled)
                        ):
                            request_debug_payload = _init_request_debug_payload(
                                max_new_tokens=int(args.max_new_tokens),
                                dump_request_payload=bool(args.dump_request_payload),
                                dump_query_messages=bool(args.dump_query_messages),
                                dump_prompt_token_ids=bool(
                                    args.dump_prompt_token_ids
                                ),
                            )
                            request_debug_payload["query_api_mode_requested"] = str(
                                args.query_api_mode
                            )
                            if bool(request_dump_enabled):
                                prefix_capture = _normalize_prefix_capture(
                                    prompt_shape_meta.get("prefix_capture")
                                )
                                request_debug_payload["suffix_query_text"] = str(
                                    suffix_query_text or ""
                                )
                                request_debug_payload["same_message_capture"] = {
                                    "direct_query_mode_enabled": bool(
                                        direct_query_mode_enabled
                                    ),
                                    "messages_override_used": bool(
                                        direct_query_messages is not None
                                    ),
                                    "prompt_shape_id": str(
                                        prompt_shape_meta.get("id")
                                        or str(args.prompt_shape)
                                    ),
                                    "prefix_capture": dict(prefix_capture),
                                    "continuation_mode": str(
                                        prefix_capture.get("continuation_mode") or ""
                                    ),
                                    "split_after": str(
                                        prefix_capture.get("split_after") or ""
                                    ),
                                    "message_index": (
                                        int(prefix_capture.get("message_index"))
                                        if str(
                                            prefix_capture.get("message_index") or ""
                                        ).strip()
                                        else -1
                                    ),
                                    "suffix_query_file": str(suffix_query_path or ""),
                                    "suffix_query_index": (
                                        int(query_index)
                                        if bool(direct_query_entries)
                                        else 0
                                    ),
                                    "suffix_query_sha256": (
                                        _sha256_text(suffix_query_text)
                                        if suffix_query_text
                                        else ""
                                    ),
                                }
                        if bool(request_live_layout_enabled) and request_debug_payload:
                            stream_progress_cb = _make_live_request_layout_probe(
                                base_url=str(base_url),
                                timeout_sec=int(request_timeout_sec_effective),
                                use_stream=bool(not args.no_stream),
                                debug_payload=request_debug_payload,
                                prefix_token_count=int(direct_prompt_token_count),
                                sent_vllm_xargs=None,
                            )
                        # Untimed warm-up: the first request after a server
                        # launch pays a one-time CUDA-graph capture/compile cost
                        # for this prompt-token shape (tens of seconds at long
                        # context). Issue and discard it before the first measured
                        # iteration so the recorded TTFT reflects steady state.
                        warmup_discard = max(0, int(getattr(args, "warmup_discard_requests", 0) or 0))
                        if iter_idx == 0 and warmup_discard > 0:
                            for _wd in range(warmup_discard):
                                try:
                                    _run_one_request_with_timeout_guard(
                                        base_url=base_url,
                                        model_id=str(args.model_id),
                                        tokenizer=tokenizer,
                                        prompt_ids=prompt_ids,
                                        prompt_shape_spec=prompt_shape_spec,
                                        system_prompt=str(args.system_prompt),
                                        max_new_tokens=int(args.max_new_tokens),
                                        min_new_tokens=int(args.min_new_tokens),
                                        timeout_sec=int(request_timeout_sec_effective),
                                        ignore_eos=bool(args.ignore_eos),
                                        use_stream=bool(not args.no_stream),
                                        temperature=float(args.temperature),
                                        top_p=float(args.top_p),
                                        top_k=int(args.top_k),
                                        repetition_penalty=float(args.repetition_penalty),
                                        presence_penalty=float(args.presence_penalty),
                                        frequency_penalty=float(args.frequency_penalty),
                                        output_mask_payload=output_mask_payload,
                                        timeout_sclk_guard=bool(timeout_sclk_guard),
                                        timeout_sclk_min_mhz=float(timeout_sclk_min_mhz),
                                        timeout_sclk_sample_count=int(timeout_sclk_sample_count),
                                        timeout_sclk_sample_interval_sec=float(timeout_sclk_sample_interval_sec),
                                        timeout_sclk_min_busy_gpus=int(timeout_sclk_min_busy_gpus),
                                        timeout_sclk_required_hit_ratio=float(timeout_sclk_required_hit_ratio),
                                        timeout_sclk_max_extensions=int(timeout_sclk_max_extensions),
                                        timeout_sclk_extension_sec=int(timeout_sclk_extension_sec),
                                        timeout_sclk_expected_gpu_count=int(timeout_sclk_expected_gpu_count),
                                        messages_override=direct_query_messages,
                                        query_api_mode=str(args.query_api_mode),
                                        prompt_token_ids_override=(
                                            list(prompt_ids)
                                            if bool(exact_prompt_token_ids_active)
                                            else None
                                        ),
                                    )
                                    print(
                                        f"[warmup] discarded untimed request for "
                                        f"{target_effective}-token prompt",
                                        flush=True,
                                    )
                                except Exception as _wexc:  # noqa: BLE001
                                    print(
                                        f"[warmup] discard request failed "
                                        f"(continuing): {str(_wexc)[:200]}",
                                        flush=True,
                                    )
                        try:
                            metrics, timeout_guard_info = _run_one_request_with_timeout_guard(
                                base_url=base_url,
                                model_id=str(args.model_id),
                                tokenizer=tokenizer,
                                prompt_ids=prompt_ids,
                                prompt_shape_spec=prompt_shape_spec,
                                system_prompt=str(args.system_prompt),
                                max_new_tokens=int(args.max_new_tokens),
                                min_new_tokens=int(args.min_new_tokens),
                                timeout_sec=int(request_timeout_sec_effective),
                                ignore_eos=bool(args.ignore_eos),
                                use_stream=bool(not args.no_stream),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                top_k=int(args.top_k),
                                repetition_penalty=float(args.repetition_penalty),
                                presence_penalty=float(args.presence_penalty),
                                frequency_penalty=float(args.frequency_penalty),
                                output_mask_payload=output_mask_payload,
                                timeout_sclk_guard=bool(timeout_sclk_guard),
                                timeout_sclk_min_mhz=float(timeout_sclk_min_mhz),
                                timeout_sclk_sample_count=int(
                                    timeout_sclk_sample_count
                                ),
                                timeout_sclk_sample_interval_sec=float(
                                    timeout_sclk_sample_interval_sec
                                ),
                                timeout_sclk_min_busy_gpus=int(
                                    timeout_sclk_min_busy_gpus
                                ),
                                timeout_sclk_required_hit_ratio=float(
                                    timeout_sclk_required_hit_ratio
                                ),
                                timeout_sclk_max_extensions=int(
                                    timeout_sclk_max_extensions
                                ),
                                timeout_sclk_extension_sec=int(
                                    timeout_sclk_extension_sec
                                ),
                                timeout_sclk_expected_gpu_count=int(
                                    timeout_sclk_expected_gpu_count
                                ),
                                messages_override=direct_query_messages,
                                query_api_mode=str(args.query_api_mode),
                                prompt_token_ids_override=(
                                    list(prompt_ids)
                                    if bool(exact_prompt_token_ids_active)
                                    else None
                                ),
                                request_debug_payload=request_debug_payload,
                                stream_progress_cb=stream_progress_cb,
                            )
                            prefill_recorder_stopped = _stop_prefill_recorder(
                                prefill_recorder,
                                prefill_recorder_stopped,
                                final_status="completed",
                                note="request_completed",
                            )
                            row.update(metrics)
                            row["request_timeout_attempts"] = int(
                                timeout_guard_info.get("attempts", 0)
                            )
                            row["request_timeout_extensions"] = int(
                                timeout_guard_info.get("extensions", 0)
                            )
                            row["request_timeout_final_timeout_sec"] = int(
                                timeout_guard_info.get(
                                    "final_timeout_sec",
                                    request_timeout_sec_effective,
                                )
                            )
                            guard_events = list(timeout_guard_info.get("events") or [])
                            row["request_timeout_sclk_guard_events"] = guard_events
                            row["request_timeout_sclk_guard_event_count"] = int(
                                len(guard_events)
                            )
                            if guard_events:
                                last = guard_events[-1]
                                row["request_timeout_sclk_guard_last_action"] = str(
                                    last.get("action") or ""
                                )
                                row["request_timeout_sclk_guard_last_busy"] = bool(
                                    last.get("sclk_busy", False)
                                )
                                row["request_timeout_sclk_guard_last_busy_gpus"] = int(
                                    last.get("sclk_max_busy_gpu_count", 0)
                                )
                                row["request_timeout_sclk_guard_last_peak_sclk_mhz"] = round(
                                    _as_float(last.get("sclk_max_peak_mhz")),
                                    6,
                                )
                            if bool(args.save_generated_text):
                                row["generated_text_path"] = _persist_generated_text(
                                    out_dir=out_dir,
                                    target_index=int(artifact_target_index),
                                    iteration=int(row["iteration"]),
                                    generated_text=str(
                                        metrics.get("generated_text") or ""
                                    ),
                                )
                            if bool(args.measure_k2ft):
                                row.update(
                                    _run_k2ft_probe(
                                        prompt_seed_ids=prompt_seed_ids,
                                        target_effective=int(target_effective),
                                        delta_tokens=int(args.k2ft_delta_tokens),
                                        min_probe_tokens=int(
                                            args.k2ft_min_probe_tokens
                                        ),
                                        k2ft_runs=int(args.k2ft_runs),
                                        base_url=base_url,
                                        model_id=str(args.model_id),
                                        tokenizer=tokenizer,
                                        prompt_shape_spec=prompt_shape_spec,
                                        system_prompt=str(args.system_prompt),
                                        max_new_tokens=int(args.max_new_tokens),
                                        min_new_tokens=int(args.min_new_tokens),
                                        timeout_sec=int(
                                            request_timeout_sec_effective
                                        ),
                                        ignore_eos=bool(args.ignore_eos),
                                        use_stream=bool(not args.no_stream),
                                        temperature=float(args.temperature),
                                        top_p=float(args.top_p),
                                        top_k=int(args.top_k),
                                        repetition_penalty=float(
                                            args.repetition_penalty
                                        ),
                                        presence_penalty=float(args.presence_penalty),
                                        frequency_penalty=float(
                                            args.frequency_penalty
                                        ),
                                        output_mask_payload=output_mask_payload,
                                        query_api_mode=str(args.query_api_mode),
                                        timeout_sclk_guard=bool(timeout_sclk_guard),
                                        timeout_sclk_min_mhz=float(
                                            timeout_sclk_min_mhz
                                        ),
                                        timeout_sclk_sample_count=int(
                                            timeout_sclk_sample_count
                                        ),
                                        timeout_sclk_sample_interval_sec=float(
                                            timeout_sclk_sample_interval_sec
                                        ),
                                        timeout_sclk_min_busy_gpus=int(
                                            timeout_sclk_min_busy_gpus
                                        ),
                                        timeout_sclk_required_hit_ratio=float(
                                            timeout_sclk_required_hit_ratio
                                        ),
                                        timeout_sclk_max_extensions=int(
                                            timeout_sclk_max_extensions
                                        ),
                                        timeout_sclk_extension_sec=int(
                                            timeout_sclk_extension_sec
                                        ),
                                        timeout_sclk_expected_gpu_count=int(
                                            timeout_sclk_expected_gpu_count
                                        ),
                                    )
                                )
                        except Exception as exc:  # noqa: BLE001
                            prefill_recorder_stopped = _stop_prefill_recorder(
                                prefill_recorder,
                                prefill_recorder_stopped,
                                final_status="error",
                                note=f"request_failed:{str(exc)[:240]}",
                            )
                            msg = str(exc)
                            if isinstance(exc, url_error.HTTPError):
                                try:
                                    body = exc.read().decode(
                                        "utf-8", errors="replace"
                                    )
                                except Exception:
                                    body = ""
                                if body:
                                    msg = f"{msg} | body={body}"
                            timeout_guard_info = getattr(
                                exc, "timeout_guard_info", None
                            )
                            if isinstance(timeout_guard_info, dict):
                                row["request_timeout_attempts"] = int(
                                    timeout_guard_info.get("attempts", 0)
                                )
                                row["request_timeout_extensions"] = int(
                                    timeout_guard_info.get("extensions", 0)
                                )
                                row["request_timeout_final_timeout_sec"] = int(
                                    timeout_guard_info.get(
                                        "final_timeout_sec",
                                        request_timeout_sec_effective,
                                    )
                                )
                                guard_events = list(
                                    timeout_guard_info.get("events") or []
                                )
                                row["request_timeout_sclk_guard_events"] = (
                                    guard_events
                                )
                                row["request_timeout_sclk_guard_event_count"] = int(
                                    len(guard_events)
                                )
                                if guard_events:
                                    last = guard_events[-1]
                                    row["request_timeout_sclk_guard_last_action"] = str(
                                        last.get("action") or ""
                                    )
                                    row["request_timeout_sclk_guard_last_busy"] = bool(
                                        last.get("sclk_busy", False)
                                    )
                                    row[
                                        "request_timeout_sclk_guard_last_busy_gpus"
                                    ] = int(last.get("sclk_max_busy_gpu_count", 0))
                                    row[
                                        "request_timeout_sclk_guard_last_peak_sclk_mhz"
                                    ] = round(
                                        _as_float(last.get("sclk_max_peak_mhz")),
                                        6,
                                    )
                            row["status"] = "error"
                            row["error"] = msg[:5000]
                        if request_debug_payload:
                            debug_paths = _persist_request_debug_artifacts(
                                out_dir=out_dir,
                                target_index=int(artifact_target_index),
                                iteration=int(row["iteration"]),
                                debug_payload=request_debug_payload,
                            )
                            row.update(debug_paths)
                            row["live_internal_request_id"] = str(
                                request_debug_payload.get("internal_request_id") or ""
                            )
                            row["live_external_request_id"] = str(
                                request_debug_payload.get("external_request_id") or ""
                            )
                            row["live_layout_ok"] = bool(
                                dict(request_debug_payload.get("layout") or {}).get(
                                    "ok"
                                )
                            )
                    row.pop("generated_text", None)

                    rows.append(row)
                    target_rows.append(row)
                    _append_jsonl_row(results_path, row)
                    update_measure_progress(
                        progress=progress,
                        row=row,
                        target_effective=int(target_effective),
                        iteration=iter_idx + 1,
                        total_iterations=measure_runs,
                        extra_ok_fields={
                            "ttft_s": f"{_as_float(row.get('ttft_ms'))/1000.0:.2f}",
                            "timeout_s": int(
                                row.get(
                                    "request_timeout_final_timeout_sec",
                                    request_timeout_sec_effective,
                                )
                            ),
                            "x": int(row.get("request_timeout_extensions", 0)),
                        },
                        extra_error_fields={
                            "timeout_s": int(
                                row.get(
                                    "request_timeout_final_timeout_sec",
                                    request_timeout_sec_effective,
                                )
                            ),
                            "x": int(row.get("request_timeout_extensions", 0)),
                        },
                    )
            ok_target_rows = [r for r in target_rows if str(r.get("status")) == "ok"]
            if ok_target_rows:
                prev_success_tokens = int(target_effective)
                prev_success_ttft_ms = mean([_as_float(r.get("ttft_ms")) for r in ok_target_rows])
        if progress is not None:
            progress.close()

        write_records(out_dir / "results.jsonl", rows)

        ok_rows = [r for r in rows if str(r.get("status", "ok")) == "ok"]
        k2ft_ok_rows = [
            r
            for r in ok_rows
            if bool(r.get("k2ft_enabled"))
            and str(r.get("k2ft_status", "")) == "ok"
            and _as_float(r.get("k2ft_ms")) > 0.0
        ]
        k2ft_vals = [_as_float(r.get("k2ft_ms")) for r in k2ft_ok_rows]
        prefill_vals = [_as_float(r.get("prefill_est_ms")) for r in k2ft_ok_rows]

        sweep_summary_rows = build_basic_sweep_summary_rows(ok_rows)
        by_target: Dict[int, List[Dict[str, Any]]] = {}
        for row in ok_rows:
            tgt = int(row.get("target_prompt_tokens_effective", 0) or 0)
            by_target.setdefault(tgt, []).append(row)
        by_target_summary = {
            int(row.get("target_prompt_tokens_effective", 0)): row for row in sweep_summary_rows
        }
        for tgt, vals in sorted(by_target.items()):
            summary_row = by_target_summary.get(int(tgt))
            if summary_row is None:
                continue
            vals = by_target[tgt]
            k2ft_t = [
                _as_float(r.get("k2ft_ms"))
                for r in vals
                if bool(r.get("k2ft_enabled"))
                and str(r.get("k2ft_status", "")) == "ok"
                and _as_float(r.get("k2ft_ms")) > 0.0
            ]
            prefill_t = [
                _as_float(r.get("prefill_est_ms"))
                for r in vals
                if bool(r.get("k2ft_enabled"))
                and str(r.get("k2ft_status", "")) == "ok"
                and _as_float(r.get("k2ft_ms")) > 0.0
            ]
            high_conf_count = len(
                [
                    r
                    for r in vals
                    if bool(r.get("k2ft_enabled"))
                    and str(r.get("k2ft_status", "")) == "ok"
                    and str(r.get("k2ft_cache_hit_confidence", "")).lower() == "high"
                ]
            )
            summary_row.update(
                {
                    "k2ft_sample_count": int(len(k2ft_t)),
                    "k2ft_ms_mean": round(mean(k2ft_t), 6) if k2ft_t else 0.0,
                    "prefill_est_ms_mean": round(mean(prefill_t), 6) if prefill_t else 0.0,
                    "k2ft_high_conf_count": int(high_conf_count),
                }
            )

        plot_result, plot_png = write_sweep_summary_artifacts(out_dir=out_dir, sweep_summary_rows=sweep_summary_rows, model_id=str(args.model_id), max_new_tokens=int(args.max_new_tokens), title_prefix="vLLM", plot_png_override=str(args.plot_png))

        summary = build_run_summary_payload(
            **build_summary_base_kwargs(
                run_id=run_id,
                model_id=str(args.model_id),
                rows=rows,
                ok_rows=ok_rows,
                sweep_plot=plot_result,
                sweep_rows=sweep_summary_rows,
            ),
            artifacts={
                "run_meta_json": str(out_dir / "run_meta.json"),
                "launch_env_json": str(out_dir / "launch_env.json"),
                "tokenizer_resolution_json": str(out_dir / "tokenizer_resolution.json")
                if (out_dir / "tokenizer_resolution.json").exists()
                else "",
                "results_jsonl": str(out_dir / "results.jsonl"),
                "summary_json": str(out_dir / "summary.json"),
                "sweep_summary_json": str(out_dir / "sweep_summary.json"),
                "kv_calibration_json": str(out_dir / "kv_calibration.json")
                if (out_dir / "kv_calibration.json").exists()
                else "",
                "kv_calibration_markdown": str(out_dir / "kv_calibration.md")
                if (out_dir / "kv_calibration.md").exists()
                else "",
                "graph_mode_status_json": str(out_dir / "graph_mode_status.json")
                if (out_dir / "graph_mode_status.json").exists()
                else "",
                "mla_decode_profile_jsonl": str(out_dir / "mla_decode_profile.jsonl")
                if (out_dir / "mla_decode_profile.jsonl").exists()
                else "",
                "warmup_rows_jsonl": str(warmup_rows_path) if warmup_rows else "",
                "prefill_progress_jsonl": str(prefill_progress_path)
                if bool(args.emit_prefill_progress) and prefill_progress_path.exists()
                else "",
                "prompt_length_scaling_png": str(plot_png) if bool(plot_result.get("ok")) else "",
                "vllm_server_log": str(out_dir / "vllm_server.log"),
                "generated_text_dir": str(out_dir / "generated_text") if bool(args.save_generated_text) else "",
                "request_debug_dir": str(out_dir / "request_debug")
                if (out_dir / "request_debug").exists()
                else "",
            },
            extra_fields={
                "warmup": warmup_summary,
                "k2ft_sample_count": int(len(k2ft_ok_rows)),
                "k2ft_ms_mean": round(mean(k2ft_vals), 6) if k2ft_vals else 0.0,
                "prefill_est_ms_mean": round(mean(prefill_vals), 6) if prefill_vals else 0.0,
                "prefill_progress_capture_enabled": bool(args.emit_prefill_progress),
                "prefill_progress_sample_count": int(
                    len(
                        _load_jsonl(prefill_progress_path)
                        if bool(args.emit_prefill_progress) and prefill_progress_path.exists()
                        else []
                    )
                ),
            },
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "run_id": run_id, "out_dir": str(out_dir), "summary": summary}, sort_keys=True))
    finally:
        _stop_process(proc)
        if proc is not None:
            cleanup_result = _cleanup_server_compile_cache(
                out_dir,
                str(getattr(args, "post_run_cache_cleanup", "none")),
            )
            _write_compile_cache_cleanup_artifact(out_dir, cleanup_result)
            if cleanup_result.get("errors"):
                print(
                    json.dumps(
                        {
                            "warning": "server_compile_cache_cleanup_failed",
                            "details": cleanup_result,
                        },
                        sort_keys=True,
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
