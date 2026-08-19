# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hot prefix-session query helpers for reusing saved HBM state.

This module exposes a small library API for asking multiple independent
questions against a saved prefix session after the prefix state has been loaded
into an already-running vLLM server.

Example:
    ```python
    from long_context_serving.prefix_session import HotPrefixSessionClient, HotPrefixSessionConfig

    client = HotPrefixSessionClient.from_config(
        HotPrefixSessionConfig(
            base_url="http://127.0.0.1:18860",
            prefix_session_dir="/app/long-context-serving/experiments/hf_long_context/prefix_sessions",
            prefix_session_id="current_repo_grounded_en_v2_4m",
        )
    )
    client.restore_prefix_state()
    first = client.ask("Summarize the repository architecture.")
    second = client.ask("What is the build and test workflow?")
    assert second.prefix_state_load_count == first.prefix_state_load_count
    ```
"""

from __future__ import annotations

import hashlib
import os
import json
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib import request as url_request

import torch
from transformers import AutoTokenizer

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

from long_context_serving.http_utils import post_json  # pylint: disable=no-name-in-module
from long_context_serving.text_utils import (
    extract_response_text,
    stream_chat_completions,
    stream_text_completions,
)

__all__ = [
    "build_english_only_output_mask",
    "build_hot_prefix_prompt_token_ids",
    "build_same_message_query_messages",
    "HotPrefixQueryResult",
    "HotPrefixSessionClient",
    "HotPrefixSessionConfig",
    "HotPrefixSessionInfo",
    "normalize_prefix_session_manifest",
]


_CJK_CHAR_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u30FF\uAC00-\uD7AF]"
)
_LEGACY_PREFIX_SESSION_SCHEMA_VERSION = "0.1"
_CURRENT_PREFIX_SESSION_SCHEMA_VERSION = "0.2"
_ATTENTION_MODE_FULL = "full_attention"
_ATTENTION_MODE_STREAMING = "streaming_mla"
_BLOCK_ROW_SELECTIONS = {
    "group_rows",
    "group_rows_global_fallback",
    "global_rows_fallback",
}


def _sha256_hex_bytes(data: bytes) -> str:
    """Return a SHA256 hex digest for raw bytes."""

    return hashlib.sha256(data).hexdigest()


def _sha256_text(payload: str) -> str:
    """Return a SHA256 hex digest for UTF-8 text."""

    return _sha256_hex_bytes(str(payload).encode("utf-8"))


def _safe_int(value: Any, default: int = 0) -> int:
    """Return ``value`` coerced to ``int`` or ``default`` on failure."""

    try:
        return int(value)
    except Exception:
        return int(default)


def _ceil_div(num: int, den: int) -> int:
    """Return ``ceil(num / den)`` for positive integers, else ``0``."""

    numerator = int(num)
    denominator = int(den)
    if numerator <= 0 or denominator <= 0:
        return 0
    return (numerator + denominator - 1) // denominator


def _block_size_from_manager_rows(manager_rows: Any) -> int:
    """Return the maximum positive block size advertised by manager rows."""

    sizes = [
        int((row or {}).get("block_size") or 0)
        for row in list(manager_rows or [])
        if isinstance(row, Mapping) and int((row or {}).get("block_size") or 0) > 0
    ]
    return int(max(sizes, default=0))


def normalize_prefix_session_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize checkpoint metadata across legacy and current schema versions."""

    normalized = dict(manifest)
    schema_version = str(normalized.get("schema_version") or "").strip()
    if not schema_version:
        schema_version = _LEGACY_PREFIX_SESSION_SCHEMA_VERSION
    attention_mode = str(normalized.get("attention_mode") or "").strip().lower()
    if not attention_mode:
        attention_mode = _ATTENTION_MODE_FULL
    if attention_mode not in {_ATTENTION_MODE_FULL, _ATTENTION_MODE_STREAMING}:
        attention_mode = _ATTENTION_MODE_FULL

    raw_attention_config = normalized.get("attention_config")
    attention_config = (
        dict(raw_attention_config)
        if isinstance(raw_attention_config, Mapping)
        else {}
    )
    block_size_tokens = max(
        0,
        _safe_int(
            attention_config.get("block_size_tokens")
            or normalized.get("block_size")
            or (normalized.get("prefix_state_format") or {}).get("block_size")
            or _block_size_from_manager_rows(
                (normalized.get("apc_cache_layout") or {}).get("manager_rows")
            ),
            0,
        ),
    )

    if attention_mode == _ATTENTION_MODE_STREAMING:
        sliding_window_tokens = max(
            0,
            _safe_int(attention_config.get("sliding_window_tokens"), 0),
        )
        sink_keep_tokens = max(
            0,
            _safe_int(attention_config.get("sink_keep_tokens"), 0),
        )
        effective_sink_block_count = max(
            0,
            _safe_int(
                attention_config.get("effective_sink_block_count"),
                _ceil_div(sink_keep_tokens, block_size_tokens),
            ),
        )
        attention_config = {
            "type": _ATTENTION_MODE_STREAMING,
            "sliding_window_tokens": int(sliding_window_tokens),
            "sink_keep_tokens": int(sink_keep_tokens),
            "effective_live_block_count": max(
                0,
                _safe_int(attention_config.get("effective_live_block_count"), 0),
            ),
            "effective_sink_block_count": int(effective_sink_block_count),
            "block_size_tokens": int(block_size_tokens),
            "dense_mla_group_id": _safe_int(
                attention_config.get("dense_mla_group_id"),
                -1,
            ),
        }
    else:
        attention_config = {"type": _ATTENTION_MODE_FULL}

    normalized["schema_version"] = str(schema_version)
    normalized["attention_mode"] = str(attention_mode)
    normalized["attention_config"] = attention_config
    return normalized


def _runtime_attention_metadata_from_run_meta(run_meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive checkpoint attention semantics from benchmark run metadata."""

    vllm_meta = dict(run_meta.get("vllm") or {})
    block_size_tokens = max(0, _safe_int(vllm_meta.get("block_size"), 0))
    sliding_window_tokens = max(
        0,
        _safe_int(
            vllm_meta.get("mla_sliding_window_tokens")
            or vllm_meta.get("kimi_mla_sliding_window_tokens_arg"),
            0,
        ),
    )
    sink_keep_tokens = max(
        0,
        _safe_int(
            vllm_meta.get("mla_sink_keep_tokens")
            or vllm_meta.get("kimi_mla_sink_keep_tokens_arg"),
            0,
        ),
    )
    if sliding_window_tokens > 0 or sink_keep_tokens > 0:
        return {
            "schema_version": _CURRENT_PREFIX_SESSION_SCHEMA_VERSION,
            "attention_mode": _ATTENTION_MODE_STREAMING,
            "attention_config": {
                "type": _ATTENTION_MODE_STREAMING,
                "sliding_window_tokens": int(sliding_window_tokens),
                "sink_keep_tokens": int(sink_keep_tokens),
                "effective_live_block_count": 0,
                "effective_sink_block_count": int(
                    _ceil_div(sink_keep_tokens, block_size_tokens)
                ),
                "block_size_tokens": int(block_size_tokens),
                "dense_mla_group_id": -1,
            },
        }
    return {
        "schema_version": _CURRENT_PREFIX_SESSION_SCHEMA_VERSION,
        "attention_mode": _ATTENTION_MODE_FULL,
        "attention_config": {"type": _ATTENTION_MODE_FULL},
    }


def _build_fp8_checkpoint_contract(kv_cache_dtype: str) -> Dict[str, Any]:
    """Build the exact FP8 decode contract used by this runtime."""

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
    bit_words = [int(word) & 0xFFFFFFFF for word in decoded.view(torch.int32).tolist()]
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
    """Annotate uint8 prefix-state tensors with FP8 contract metadata."""

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


def maybe_upgrade_legacy_fp8_manifest(
    session_dir: Path,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Backfill missing FP8 decode metadata for legacy manifests in-place."""

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


def chat_template_hash(tokenizer: Any) -> str:
    """Hash the tokenizer chat template for compatibility checks."""

    template = str(getattr(tokenizer, "chat_template", "") or "")
    return _sha256_text(template)


def write_token_ids_bin(path: Path, token_ids: List[int]) -> str:
    """Write uint32 token IDs to disk and return the SHA256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = struct.pack(f"<{len(token_ids)}I", *[int(x) for x in token_ids]) if token_ids else b""
    path.write_bytes(raw)
    return _sha256_hex_bytes(raw)


def read_token_ids_bin(path: Path) -> Tuple[List[int], str]:
    """Read uint32 token IDs from disk and return ``(token_ids, sha256)``."""

    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise RuntimeError(f"Invalid token-id bin length (expected multiple of 4): {path}")
    count = len(raw) // 4
    token_ids = list(struct.unpack(f"<{count}I", raw)) if count else []
    return token_ids, _sha256_hex_bytes(raw)


def tokenize_text(tokenizer: Any, text: str) -> List[int]:
    """Tokenize plain text without adding special tokens."""

    out = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if not isinstance(out, list):
        out = list(out)
    return [int(x) for x in out]


def tokenize_messages(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> List[int]:
    """Tokenize rendered chat messages using the tokenizer chat template.

    Args:
        tokenizer: Hugging Face-compatible tokenizer.
        messages: Rendered OpenAI-style chat messages.
        add_generation_prompt: Whether to serialize the template with the
            model-specific assistant-generation suffix.

    Returns:
        Token IDs representing the serialized chat prompt.
    """

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            out = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=bool(add_generation_prompt),
            )
            if isinstance(out, list):
                return [int(x) for x in out]
        except Exception:
            pass
    flat = "\n".join([f"{msg.get('role', '')}:\n{msg.get('content', '')}" for msg in messages])
    return tokenize_text(tokenizer, flat)


def _contains_cjk(text: str) -> bool:
    """Return whether decoded token text contains any CJK character."""

    return bool(_CJK_CHAR_RE.search(str(text or "")))


def _extract_cached_prompt_tokens(usage: Mapping[str, Any]) -> int:
    """Extract cached prompt-token count from an OpenAI-compatible usage block."""

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


def build_english_only_output_mask(
    tokenizer: Any,
    *,
    mode: str,
    bias_value: float,
    vocab_size_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build generation-time token controls that suppress CJK output.

    Args:
        tokenizer: Hugging Face-compatible tokenizer.
        mode: One of ``off``, ``bias``, or ``hard``.
        bias_value: Magnitude for the soft-bias mode. The emitted value is
            always negative.
        vocab_size_limit: Optional upper bound on valid output token IDs.
            Token IDs greater than or equal to this bound are excluded from
            the generated mask.

    Returns:
        Dictionary with payload fields plus summary metadata.
    """

    normalized_mode = str(mode or "hard").strip().lower()
    if normalized_mode not in {"off", "bias", "hard"}:
        raise ValueError(f"unsupported english-only mask mode: {mode}")
    if normalized_mode == "off":
        return {
            "mode": "off",
            "cjk_banned_count": 0,
            "allowed_token_count": None,
            "logit_bias_count": 0,
            "bias_value": None,
            "payload": {},
        }

    vocab = tokenizer.get_vocab()
    vocab_ids = sorted({int(token_id) for token_id in vocab.values()})
    effective_vocab_limit = None
    if vocab_size_limit is not None:
        effective_vocab_limit = max(0, int(vocab_size_limit))
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    banned_ids: List[int] = []
    allowed_ids: List[int] = []
    negative_bias = -abs(float(bias_value))

    for token_id in vocab_ids:
        if effective_vocab_limit is not None and token_id >= effective_vocab_limit:
            continue
        if token_id in special_ids:
            allowed_ids.append(token_id)
            continue
        piece = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if _contains_cjk(piece):
            banned_ids.append(token_id)
        else:
            allowed_ids.append(token_id)

    if normalized_mode == "hard":
        payload: Dict[str, Any] = {"allowed_token_ids": allowed_ids}
    else:
        payload = {
            "logit_bias": {str(int(token_id)): float(negative_bias) for token_id in banned_ids}
        }
    return {
        "mode": normalized_mode,
        "cjk_banned_count": int(len(banned_ids)),
        "allowed_token_count": int(len(allowed_ids)) if normalized_mode == "hard" else None,
        "logit_bias_count": int(len(banned_ids)) if normalized_mode == "bias" else 0,
        "bias_value": float(negative_bias) if normalized_mode == "bias" else None,
        "payload": payload,
    }


def normalize_prefix_capture(prefix_capture: Any) -> Dict[str, Any]:
    """Normalize prompt-shape prefix-capture metadata to a stable mapping."""

    if not isinstance(prefix_capture, Mapping):
        return {}
    message_index = prefix_capture.get("message_index")
    split_after = prefix_capture.get("split_after")
    continuation_mode = prefix_capture.get("continuation_mode")
    if not isinstance(message_index, int) or isinstance(message_index, bool) or message_index < 0:
        return {}
    if not isinstance(split_after, str) or not split_after:
        return {}
    if not isinstance(continuation_mode, str) or not continuation_mode:
        return {}
    return {
        "message_index": int(message_index),
        "split_after": str(split_after),
        "continuation_mode": str(continuation_mode),
    }


def build_same_message_query_prompt_token_ids(
    *,
    tokenizer: Any,
    prefix_token_ids: List[int],
    suffix_text: str,
    continuation_tail_token_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """Build full prompt-token payload for same-message suffix continuation."""

    suffix_token_ids = tokenize_text(tokenizer, suffix_text)
    prompt_token_ids = [
        *[int(x) for x in prefix_token_ids],
        *suffix_token_ids,
        *[int(x) for x in continuation_tail_token_ids],
    ]
    return prompt_token_ids, suffix_token_ids


def build_same_message_query_messages(
    *,
    prefix_messages: List[Dict[str, str]],
    suffix_text: str,
    message_index: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Build same-message query messages from a saved prefix-message stub.

    Args:
        prefix_messages: Saved rendered messages truncated at the prefix
            capture boundary.
        suffix_text: User-supplied suffix text that should continue the same
            captured message.
        message_index: Optional explicit message index to extend. When omitted,
            the last message in ``prefix_messages`` is extended.

    Returns:
        Rendered messages ready for ``/v1/chat/completions``.

    Raises:
        ValueError: If ``prefix_messages`` is empty or the target message index
            is out of range.
    """

    if not prefix_messages:
        raise ValueError("prefix_messages must be non-empty")
    target_index = len(prefix_messages) - 1 if message_index is None else int(message_index)
    if target_index < 0 or target_index >= len(prefix_messages):
        raise ValueError(
            f"message_index out of range for prefix_messages: {target_index} not in [0, {len(prefix_messages)})"
        )
    out = [dict(msg) for msg in prefix_messages]
    updated = dict(out[target_index])
    updated["content"] = str(updated.get("content") or "") + str(suffix_text)
    out[target_index] = updated
    return out


def build_hot_prefix_prompt_token_ids(
    *,
    tokenizer: Any,
    suffix_text: str,
    continuation_tail_token_ids: List[int],
    replay_prefix_tail_token_ids: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Build the compact prompt-token payload for a restored hot prefix.

    This form is used after the prefix rows are already loaded into HBM. Only
    the uncached prefix tail, same-message suffix, and continuation tail are
    sent over HTTP.

    Args:
        tokenizer: Tokenizer used for suffix tokenization.
        suffix_text: User-supplied suffix content that follows ``Task:\n``.
        continuation_tail_token_ids: Chat-template tail token IDs that trigger
            assistant generation for the same user message.
        replay_prefix_tail_token_ids: Optional non-block-aligned prefix tokens
            that must be replayed before the new suffix so the request matches
            the saved prompt exactly.

    Returns:
        Tuple ``(prompt_token_ids, suffix_token_ids)`` where
        ``prompt_token_ids`` contains the replay tail, suffix, and continuation
        tail.
    """

    suffix_token_ids = tokenize_text(tokenizer, suffix_text)
    prompt_token_ids = [
        *[int(x) for x in list(replay_prefix_tail_token_ids or [])],
        *suffix_token_ids,
        *[int(x) for x in continuation_tail_token_ids],
    ]
    return prompt_token_ids, suffix_token_ids


def build_compact_hot_prefix_prompt_token_ids(
    *,
    tokenizer: Any,
    reused_prefix_token_count: int,
    suffix_text: str,
    prefix_messages: Optional[List[Dict[str, str]]] = None,
    message_index: Optional[int] = None,
    continuation_tail_token_ids: Optional[List[int]] = None,
    replay_prefix_tail_token_ids: Optional[List[int]] = None,
) -> Tuple[List[int], int]:
    """Build the exact uncached prompt tail for compact hot-prefix replay.

    Same-message replay must preserve the tokenizer's boundary behavior at the
    split point. For repo-grounded prompts, tokenizing ``suffix_text`` in
    isolation overcounts by a few tokens relative to re-tokenizing the exact
    same-message prompt continuation, so prefer the full message-level rebuild
    when the saved prefix messages are available.
    """

    reused_prefix_count = max(0, int(reused_prefix_token_count or 0))
    if prefix_messages:
        query_messages = build_same_message_query_messages(
            prefix_messages=[dict(msg) for msg in list(prefix_messages or [])],
            suffix_text=str(suffix_text),
            message_index=message_index,
        )
        # Same-message replay must preserve the assistant-generation suffix
        # that the original direct prefill path saw. Without it, the restored
        # request can continue the user prompt instead of starting the answer.
        full_prompt_token_ids = tokenize_messages(
            tokenizer,
            query_messages,
            add_generation_prompt=True,
        )
        if reused_prefix_count <= len(full_prompt_token_ids):
            prompt_token_ids = [
                int(x) for x in full_prompt_token_ids[reused_prefix_count:]
            ]
            return prompt_token_ids, int(len(full_prompt_token_ids))

    prompt_token_ids, _suffix_token_ids = build_hot_prefix_prompt_token_ids(
        tokenizer=tokenizer,
        suffix_text=str(suffix_text),
        continuation_tail_token_ids=[
            int(x) for x in list(continuation_tail_token_ids or [])
        ],
        replay_prefix_tail_token_ids=[
            int(x) for x in list(replay_prefix_tail_token_ids or [])
        ],
    )
    return prompt_token_ids, int(reused_prefix_count + len(prompt_token_ids))


def build_streaming_hot_prefix_manager_rows(
    manifest: Mapping[str, Any],
    *,
    include_sparse_groups: bool = True,
    reused_prefix_token_count: int | None = None,
) -> List[Dict[str, Any]]:
    """Extract compact streaming manager rows for hot-prefix restore.

    Args:
        manifest: Saved prefix-session manifest.
        include_sparse_groups: Whether to include sparse streaming-state
            groups alongside the dense MLA group. Exact restored replay needs
            these sparse groups because they carry the compact sink/window tail
            state that the first continuation attention step reads.
        reused_prefix_token_count: Logical prefix-token count that the request
            will treat as already computed. When replaying an unaligned tail,
            this is smaller than ``manifest["prefix_token_count"]`` and must
            drive the sparse-row offsets so restored Mamba/KDA state lands on
            the same live block indices as the direct prefill path.

    Returns:
        Normalized manager-row descriptors suitable for ``vllm_xargs``. An
        empty list means the session does not expose compact streaming rows.
    """

    layout = dict((manifest.get("apc_cache_layout") or {}))
    rows = list(layout.get("manager_rows") or [])
    global_target_rows = [
        int(value)
        for value in list(layout.get("global_target_rows") or [])
        if int(value) > 0
    ]
    source_group_rows = {
        int(key): [int(value) for value in list(values or []) if int(value) > 0]
        for key, values in dict(layout.get("source_group_rows") or {}).items()
    }
    dense_group_index = int(layout.get("global_target_group_id") or -1)
    if dense_group_index < 0 and global_target_rows:
        for group_index, group_rows in source_group_rows.items():
            if list(group_rows) == list(global_target_rows):
                dense_group_index = int(group_index)
                break
    if dense_group_index < 0 and source_group_rows:
        dense_group_index = max(
            source_group_rows,
            key=lambda group_index: len(source_group_rows.get(int(group_index), [])),
            default=-1,
        )
    block_size = max(
        int((manifest.get("attention_config") or {}).get("block_size_tokens") or 0),
        int((manifest.get("block_size") or 0)),
        int((manifest.get("prefix_state_format") or {}).get("block_size") or 0),
        int((manifest.get("runtime_vllm") or {}).get("block_size") or 0),
        _block_size_from_manager_rows(rows),
    )
    prefix_token_count = int(
        reused_prefix_token_count
        if reused_prefix_token_count is not None
        else (manifest.get("prefix_token_count") or 0)
    )
    dense_num_full_blocks = (
        _ceil_div(prefix_token_count, block_size) if block_size > 0 else 0
    )
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        group_index = int(row.get("group_index") or 0)
        num_full_blocks = int(row.get("num_full_blocks") or 0)
        if int(group_index) == int(dense_group_index) and dense_num_full_blocks > 0:
            num_full_blocks = int(dense_num_full_blocks)
        cached_block_ids = [
            int(x) for x in list(row.get("cached_block_ids") or []) if int(x) > 0
        ]
        cached_block_offsets = [
            int(x)
            for x in list(row.get("cached_block_offsets") or [])
            if int(x) >= 0
        ]
        if int(group_index) == int(dense_group_index) and num_full_blocks > 0:
            filtered_pairs = [
                (int(offset), int(block_id))
                for offset, block_id in zip(cached_block_offsets, cached_block_ids)
                if int(offset) < int(num_full_blocks)
            ]
            cached_block_offsets = [int(offset) for offset, _ in filtered_pairs]
            cached_block_ids = [int(block_id) for _, block_id in filtered_pairs]
        if not cached_block_ids or not cached_block_offsets:
            continue
        if not include_sparse_groups and dense_group_index >= 0:
            if int(group_index) != int(dense_group_index):
                continue
        normalized.append(
            {
                "group_index": int(group_index),
                "kv_cache_group_id": int(row.get("kv_cache_group_id") or 0),
                "num_full_blocks": int(num_full_blocks),
                "cached_block_ids": cached_block_ids,
                "cached_block_offsets": cached_block_offsets,
            }
        )

    linear_target_last_rows = {
        str(key): int(value)
        for key, value in dict(layout.get("linear_target_last_rows") or {}).items()
        if int(value or 0) > 0
    }
    # Newer streaming checkpoints already persist a complete compact manager-row
    # description in ``apc_cache_layout.manager_rows``. Requiring
    # ``source_group_rows`` here accidentally drops that exact sparse layout and
    # falls back to generic dense hot-prefix reuse, which breaks restored
    # same-message continuation exactness at layer 0 attention.
    if normalized and global_target_rows and linear_target_last_rows and dense_group_index >= 0:
        return [
            dict(normalized_row)
            for normalized_row in normalized
        ]
    if not global_target_rows or not linear_target_last_rows or not source_group_rows:
        return []
    if dense_group_index < 0:
        return []
    num_full_blocks = int(dense_num_full_blocks)
    if num_full_blocks <= 0:
        num_full_blocks = max(len(global_target_rows), 1)

    sparse_saved_rows = [
        int(value)
        for key in ["-1", "1", "2"]
        for value in [linear_target_last_rows.get(key)]
        if int(value or 0) > 0
    ]
    sparse_row_iter = iter(sparse_saved_rows)
    reconstructed_by_group = {
        int(row.get("group_index") or 0): dict(row) for row in normalized
    }
    for group_index in sorted(source_group_rows):
        if int(group_index) == int(dense_group_index):
            dense_pairs = list(enumerate(int(value) for value in global_target_rows))
            if num_full_blocks > 0:
                dense_pairs = [
                    (int(offset), int(block_id))
                    for offset, block_id in dense_pairs
                    if int(offset) < int(num_full_blocks)
                ]
            dense_offsets = [int(offset) for offset, _ in dense_pairs]
            dense_block_ids = [int(block_id) for _, block_id in dense_pairs]
            if not dense_block_ids:
                continue
            reconstructed_by_group.setdefault(
                int(group_index),
                {
                    "group_index": int(group_index),
                    "kv_cache_group_id": int(group_index),
                    "num_full_blocks": int(num_full_blocks),
                    "cached_block_ids": dense_block_ids,
                    "cached_block_offsets": dense_offsets,
                },
            )
            continue
        if not include_sparse_groups:
            continue
        if int(group_index) in reconstructed_by_group:
            continue
        group_rows = list(source_group_rows.get(int(group_index), []))
        compact_block_count = max(1, len(group_rows))
        sparse_offset = int(max(0, compact_block_count - 1))
        saved_row = next(sparse_row_iter, None)
        if int(saved_row or 0) <= 0:
            saved_row = int(group_rows[0]) if group_rows else 0
        if int(saved_row or 0) <= 0:
            continue
        reconstructed_by_group[int(group_index)] = {
            "group_index": int(group_index),
            "kv_cache_group_id": int(group_index),
            "num_full_blocks": int(compact_block_count),
            "cached_block_ids": [int(saved_row)],
            "cached_block_offsets": [int(sparse_offset)],
        }
    return [
        reconstructed_by_group[group_index]
        for group_index in sorted(reconstructed_by_group)
    ]


def project_streaming_prepare_payload_to_saved_layout(
    prepared_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project a full prepared layout back onto the saved compact StreamLLM layout.

    ``prepare_loaded_prefix_cache`` allocates a live block layout for the full
    aligned prefix length. That is valid for dense full-prefix restore, but a
    saved StreamLLM checkpoint only persists the compact hot-prefix window. To
    load those compact shards correctly, we must map the saved manager-row
    offsets onto the freshly allocated live block IDs and pass that compact
    subset to the load RPC.

    Args:
        prepared_payload: Live prepared layout returned by the server.
        manifest: Saved prefix-session manifest.

    Returns:
        A prepare-payload-like mapping whose target rows and registered groups
        reference only the saved compact StreamLLM layout. If the manifest does
        not expose compact manager rows, the original payload is returned.
    """

    layout = dict((manifest.get("apc_cache_layout") or {}))
    saved_rows = [
        dict(row)
        for row in list(layout.get("manager_rows") or [])
        if isinstance(row, Mapping)
    ]
    prepared_rows = [
        dict(row)
        for row in list(prepared_payload.get("registered_groups") or [])
        if isinstance(row, Mapping)
    ]
    if not saved_rows or not prepared_rows:
        return dict(prepared_payload)

    prepared_by_group = {
        int(row.get("group_index") or 0): dict(row) for row in prepared_rows
    }
    saved_to_live_block_ids: Dict[int, int] = {}
    compact_rows: List[Dict[str, Any]] = []

    for saved_row in saved_rows:
        group_index = int(saved_row.get("group_index") or 0)
        prepared_row = prepared_by_group.get(int(group_index))
        if not prepared_row:
            continue
        allocated_block_ids = [
            int(block_id)
            for block_id in list(prepared_row.get("allocated_block_ids") or [])
        ]
        saved_block_ids = [
            int(block_id)
            for block_id in list(saved_row.get("cached_block_ids") or [])
            if int(block_id) > 0
        ]
        saved_block_offsets = [
            int(offset)
            for offset in list(saved_row.get("cached_block_offsets") or [])
            if int(offset) >= 0
        ]
        live_pairs: List[Tuple[int, int]] = []
        for saved_offset, saved_block_id in zip(saved_block_offsets, saved_block_ids):
            if saved_offset < 0 or saved_offset >= len(allocated_block_ids):
                continue
            live_block_id = int(allocated_block_ids[saved_offset] or 0)
            if live_block_id <= 0:
                continue
            saved_to_live_block_ids[int(saved_block_id)] = int(live_block_id)
            live_pairs.append((int(saved_offset), int(live_block_id)))
        if not live_pairs:
            continue
        compact_rows.append(
            {
                "group_index": int(group_index),
                "kv_cache_group_id": int(
                    prepared_row.get("kv_cache_group_id")
                    or saved_row.get("kv_cache_group_id")
                    or group_index
                ),
                "num_full_blocks": int(
                    saved_row.get("num_full_blocks")
                    or prepared_row.get("num_full_blocks")
                    or 0
                ),
                "cached_block_offsets": [int(offset) for offset, _ in live_pairs],
                "cached_block_ids": [int(block_id) for _, block_id in live_pairs],
            }
        )

    if not compact_rows:
        return dict(prepared_payload)

    saved_global_target_rows = [
        int(block_id)
        for block_id in list(layout.get("global_target_rows") or [])
        if int(block_id) > 0
    ]
    compact_global_target_rows = [
        int(saved_to_live_block_ids[int(block_id)])
        for block_id in saved_global_target_rows
        if int(block_id) in saved_to_live_block_ids
        and int(saved_to_live_block_ids[int(block_id)]) > 0
    ]
    saved_linear_target_last_rows = {
        str(key): int(value)
        for key, value in dict(layout.get("linear_target_last_rows") or {}).items()
        if int(value or 0) > 0
    }
    compact_linear_target_last_rows = {
        str(key): int(saved_to_live_block_ids[int(value)])
        for key, value in saved_linear_target_last_rows.items()
        if int(value) in saved_to_live_block_ids
        and int(saved_to_live_block_ids[int(value)]) > 0
    }

    compact_payload = dict(prepared_payload)
    compact_payload["global_target_rows"] = list(compact_global_target_rows)
    compact_payload["linear_target_last_rows"] = dict(compact_linear_target_last_rows)
    compact_payload["global_target_group_id"] = int(
        layout.get("global_target_group_id")
        or prepared_payload.get("global_target_group_id")
        or -1
    )
    compact_payload["registered_groups"] = list(compact_rows)
    compact_payload["source_group_rows"] = {
        str(key): [int(value) for value in list(values or []) if int(value) > 0]
        for key, values in dict(layout.get("source_group_rows") or {}).items()
    }
    return compact_payload


def build_streaming_manager_rows_from_prepared_layout(
    prepared_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Convert prepared live APC targets into compact hot-prefix manager rows.

    Args:
        prepared_payload: Payload returned by ``prepare_loaded_prefix_cache``
            for a streaming session.

    Returns:
        Normalized manager rows that reference the freshly allocated live block
        IDs instead of the saved on-disk row IDs.
    """

    rows: List[Dict[str, Any]] = []
    for raw_row in list(prepared_payload.get("registered_groups") or []):
        if not isinstance(raw_row, Mapping):
            continue
        cached_pairs: List[Tuple[int, int]]
        if raw_row.get("cached_block_offsets") is not None or raw_row.get("cached_block_ids") is not None:
            cached_pairs = [
                (int(offset), int(block_id))
                for offset, block_id in zip(
                    list(raw_row.get("cached_block_offsets") or []),
                    list(raw_row.get("cached_block_ids") or []),
                )
                if int(offset) >= 0 and int(block_id) > 0
            ]
        else:
            allocated_block_ids = [
                int(block_id)
                for block_id in list(raw_row.get("allocated_block_ids") or [])
            ]
            cached_pairs = [
                (int(offset), int(block_id))
                for offset, block_id in enumerate(allocated_block_ids)
                if int(block_id) > 0
            ]
        if not cached_pairs:
            continue
        rows.append(
            {
                "group_index": int(raw_row.get("group_index") or 0),
                "kv_cache_group_id": int(raw_row.get("kv_cache_group_id") or 0),
                "num_full_blocks": int(raw_row.get("num_full_blocks") or 0),
                "cached_block_offsets": [int(offset) for offset, _ in cached_pairs],
                "cached_block_ids": [int(block_id) for _, block_id in cached_pairs],
            }
        )
    return rows


def _infer_block_size_from_saved_state(
    *,
    session_dir: Path,
    prefix_token_count: int,
) -> int:
    """Infer the vLLM cache block size from saved rank manifests.

    This fallback exists for older saved prefix sessions that predate an
    explicit block-size field in ``prefix_session_manifest.json``.

    Args:
        session_dir: Saved prefix-session directory.
        prefix_token_count: Total captured prefix-token count.

    Returns:
        Inferred block size in tokens, or ``0`` if no stable inference is
        available.
    """

    if int(prefix_token_count) <= 0:
        return 0
    candidate_counts: Dict[int, int] = {}
    for manifest_path in sorted((session_dir / "kv_state").glob("rank_*/rank_manifest.json")):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = list(raw.get("tensor_entries") or raw.get("tensor_files") or [])
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("selection") or "") != "global_rows_fallback":
                continue
            saved_row_count = int(entry.get("saved_row_count") or 0)
            if saved_row_count <= 0:
                continue
            candidate = (int(prefix_token_count) + saved_row_count - 1) // saved_row_count
            if candidate <= 0:
                continue
            candidate_counts[candidate] = int(candidate_counts.get(candidate) or 0) + 1
    if not candidate_counts:
        return 0
    ranked = sorted(
        candidate_counts.items(),
        key=lambda item: (-int(item[1]), int(item[0])),
    )
    return int(ranked[0][0])


def resolve_prefix_cache_block_size(
    *,
    manifest: Mapping[str, Any],
    session_dir: Path,
) -> int:
    """Resolve the reusable vLLM prefix-cache block size in tokens.

    Args:
        manifest: Saved prefix-session manifest.
        session_dir: Saved prefix-session directory.

    Returns:
        Prefix-cache block size in tokens, or ``0`` if the session does not
        expose enough information to derive it.
    """

    direct_candidates = [
        (manifest.get("attention_config") or {}).get("block_size_tokens"),
        manifest.get("block_size"),
        (manifest.get("prefix_state_format") or {}).get("block_size"),
        (manifest.get("runtime_vllm") or {}).get("block_size"),
        _block_size_from_manager_rows((manifest.get("apc_cache_layout") or {}).get("manager_rows")),
    ]
    for value in direct_candidates:
        block_size = int(value or 0)
        if block_size > 0:
            return block_size
    return _infer_block_size_from_saved_state(
        session_dir=session_dir,
        prefix_token_count=int(manifest.get("prefix_token_count") or 0),
    )


def _streaming_same_message_reuse_mode() -> str:
    """Return the normalized streaming hot-prefix reuse mode."""

    token = str(
        os.environ.get("HFLC_STREAM_SAME_MESSAGE_REUSE_MODE", "aligned") or "aligned"
    ).strip().lower()
    return token if token in {"aligned", "full"} else "aligned"


def build_hot_prefix_reuse_plan(
    *,
    prefix_token_ids: List[int],
    block_size_tokens: int,
    prefer_full_prefix_reuse: bool = False,
) -> Dict[str, Any]:
    """Split a saved prefix into reusable blocks and a replay tail.

    Args:
        prefix_token_ids: Full captured prefix token IDs.
        block_size_tokens: vLLM cache block size in tokens.
        prefer_full_prefix_reuse: When true, reuse the entire saved prefix
            without replaying an unaligned tail. This matches direct-loaded
            prefix-state restore paths where the saved prefix state already
            includes the non-block-aligned suffix of the prefix.

    Returns:
        Mapping with ``reused_prefix_token_count`` and
        ``replay_prefix_tail_token_ids``.
    """

    normalized_block_size = int(block_size_tokens or 0)
    if bool(prefer_full_prefix_reuse) or normalized_block_size <= 0:
        return {
            "reused_prefix_token_count": int(len(prefix_token_ids)),
            "replay_prefix_tail_token_ids": [],
        }
    reused_prefix_token_count = (
        int(len(prefix_token_ids)) // normalized_block_size
    ) * normalized_block_size
    return {
        "reused_prefix_token_count": int(reused_prefix_token_count),
        "replay_prefix_tail_token_ids": [
            int(x) for x in list(prefix_token_ids[reused_prefix_token_count:])
        ],
    }


def load_prefix_session(
    *,
    prefix_session_dir: Path,
    prefix_session_id: str,
) -> Dict[str, Any]:
    """Load a saved prefix session from disk."""

    session_dir = prefix_session_dir / str(prefix_session_id)
    manifest_path = session_dir / "prefix_session_manifest.json"
    token_path = session_dir / "prefix_token_ids.bin"
    continuation_tail_path = session_dir / "continuation_tail_token_ids.bin"
    messages_path = session_dir / "prefix_messages.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing prefix session manifest: {manifest_path}")
    if not token_path.exists():
        raise RuntimeError(f"Missing prefix token file: {token_path}")
    if not messages_path.exists():
        raise RuntimeError(f"Missing prefix messages file: {messages_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = maybe_upgrade_legacy_fp8_manifest(session_dir, manifest)
    manifest = normalize_prefix_session_manifest(manifest)
    messages = json.loads(messages_path.read_text(encoding="utf-8"))
    token_ids, token_hash = read_token_ids_bin(token_path)
    expected_hash = str(manifest.get("prefix_token_sha256") or "")
    if expected_hash and expected_hash != token_hash:
        raise RuntimeError(
            f"Prefix token hash mismatch: expected={expected_hash} actual={token_hash}"
        )
    continuation_tail_token_ids: List[int] = []
    continuation_tail_hash = ""
    expected_tail_hash = str(manifest.get("continuation_tail_sha256") or "")
    if continuation_tail_path.exists():
        continuation_tail_token_ids, continuation_tail_hash = read_token_ids_bin(
            continuation_tail_path
        )
        if expected_tail_hash and expected_tail_hash != continuation_tail_hash:
            raise RuntimeError(
                "Continuation tail token hash mismatch: "
                f"expected={expected_tail_hash} actual={continuation_tail_hash}"
            )
    elif expected_tail_hash:
        raise RuntimeError(f"Missing continuation tail token file: {continuation_tail_path}")
    return {
        "session_dir": str(session_dir),
        "manifest": manifest,
        "prefix_messages": messages,
        "prefix_token_ids": token_ids,
        "prefix_token_sha256": token_hash,
        "continuation_tail_token_ids": continuation_tail_token_ids,
        "continuation_tail_sha256": continuation_tail_hash,
        "prefix_state_row_shift": infer_prefix_state_row_shift(session_dir=session_dir),
    }


def infer_prefix_state_row_shift(*, session_dir: Path) -> int:
    """Infer whether saved prefix-state rows need a +1 load-time shift.

    Older prefix captures persisted live request block rows starting at row 0,
    while the current hot-prefix fast path addresses reusable blocks starting at
    block 1. When that happens, loading the saved shard rows one row later keeps
    the on-device layout aligned with the hot-prefix request contract.

    Args:
        session_dir: Saved prefix-session directory that contains ``kv_state``.

    Returns:
        ``1`` when the saved rank manifests consistently look zero-based and can
        be shifted forward safely, otherwise ``0``.
    """

    kv_state_dir = session_dir / "kv_state"
    if not kv_state_dir.exists():
        return 0

    saw_zero_start = False
    saw_positive_start = False
    shift_safe = True

    for manifest_path in sorted(kv_state_dir.glob("rank_*/rank_manifest.json")):
        try:
            rank_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = list(rank_manifest.get("tensor_entries") or rank_manifest.get("tensor_files") or [])
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("selection") or "") not in _BLOCK_ROW_SELECTIONS:
                continue
            shape = list(entry.get("shape") or [])
            max_rows = int(shape[0]) if shape else 0
            if max_rows <= 0:
                continue
            row_runs = [
                (int(item[0]), int(item[1]))
                for item in list(entry.get("row_index_runs") or [])
                if isinstance(item, (list, tuple))
                and len(item) >= 2
                and int(item[1]) > 0
            ]
            if not row_runs:
                continue
            min_start = min(start for start, _ in row_runs)
            max_end = max(start + length - 1 for start, length in row_runs)
            if min_start == 0:
                saw_zero_start = True
                if max_end + 1 >= max_rows:
                    shift_safe = False
            elif min_start > 0:
                saw_positive_start = True

    if saw_zero_start and not saw_positive_start and shift_safe:
        return 1
    return 0


def validate_prefix_session_compat(
    *,
    manifest: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    prompt_shape_meta: Mapping[str, Any],
    strict_env_hash: str,
    strict: bool,
) -> Tuple[bool, List[str]]:
    """Check whether a saved prefix session is reusable for the current run."""

    mismatches: List[str] = []
    normalized_manifest = normalize_prefix_session_manifest(manifest)
    runtime_vllm_meta = dict(run_meta.get("vllm") or {})
    runtime_attention = _runtime_attention_metadata_from_run_meta(run_meta)
    runtime_fp8_contract = _build_fp8_checkpoint_contract(
        str(runtime_vllm_meta.get("kv_cache_dtype") or "")
    )
    runtime_prefix_capture = normalize_prefix_capture(prompt_shape_meta.get("prefix_capture"))
    manifest_prefix_capture = normalize_prefix_capture(normalized_manifest.get("prefix_capture"))
    checks = [
        (
            "model_id",
            str(run_meta.get("model_id") or ""),
            str(normalized_manifest.get("model_id") or ""),
        ),
        (
            "tokenizer_id",
            str(run_meta.get("tokenizer_id") or ""),
            str(normalized_manifest.get("tokenizer_id") or ""),
        ),
        (
            "chat_template_hash",
            str(run_meta.get("chat_template_hash") or ""),
            str(normalized_manifest.get("chat_template_hash") or ""),
        ),
        (
            "prompt_shape_id",
            str(prompt_shape_meta.get("id") or ""),
            str(normalized_manifest.get("prompt_shape_id") or ""),
        ),
        (
            "prompt_shape_template_sha256",
            str(prompt_shape_meta.get("template_sha256") or ""),
            str(normalized_manifest.get("prompt_shape_template_sha256") or ""),
        ),
        (
            "kv_cache_dtype",
            str(runtime_vllm_meta.get("kv_cache_dtype") or ""),
            str(normalized_manifest.get("kv_cache_dtype") or ""),
        ),
        (
            "attention_backend",
            str(runtime_vllm_meta.get("attention_backend") or ""),
            str(normalized_manifest.get("attention_backend") or ""),
        ),
    ]
    if runtime_fp8_contract:
        checks.extend(
            [
                (
                    "fp8_storage_format",
                    str(runtime_fp8_contract.get("fp8_storage_format") or ""),
                    str(normalized_manifest.get("fp8_storage_format") or ""),
                ),
                (
                    "fp8_decode_lut_sha256",
                    str(runtime_fp8_contract.get("fp8_decode_lut_sha256") or ""),
                    str(normalized_manifest.get("fp8_decode_lut_sha256") or ""),
                ),
            ]
        )
    for key, expected, actual in checks:
        if key in {"model_id", "tokenizer_id"}:
            expected_aliases = _normalize_model_id_aliases(expected)
            actual_aliases = _normalize_model_id_aliases(actual)
            if expected_aliases and actual_aliases and expected_aliases & actual_aliases:
                continue
        if expected != actual:
            mismatches.append(f"{key}: expected={expected!r} actual={actual!r}")
    if runtime_prefix_capture != manifest_prefix_capture:
        mismatches.append(
            "prefix_capture: "
            f"expected={runtime_prefix_capture!r} actual={manifest_prefix_capture!r}"
        )
    manifest_attention_mode = str(normalized_manifest.get("attention_mode") or "")
    runtime_attention_mode = str(runtime_attention.get("attention_mode") or "")
    if runtime_attention_mode != manifest_attention_mode:
        mismatches.append(
            "attention_mode: "
            f"expected={runtime_attention_mode!r} actual={manifest_attention_mode!r}"
        )
    elif runtime_attention_mode == _ATTENTION_MODE_STREAMING:
        runtime_attention_config = dict(runtime_attention.get("attention_config") or {})
        manifest_attention_config = dict(normalized_manifest.get("attention_config") or {})
        for key in ("sliding_window_tokens", "sink_keep_tokens"):
            expected = _safe_int(runtime_attention_config.get(key), 0)
            actual = _safe_int(manifest_attention_config.get(key), 0)
            if expected != actual:
                mismatches.append(
                    f"attention_config.{key}: expected={expected!r} actual={actual!r}"
                )
        expected_block_size = _safe_int(runtime_attention_config.get("block_size_tokens"), 0)
        actual_block_size = _safe_int(manifest_attention_config.get("block_size_tokens"), 0)
        if expected_block_size > 0 and expected_block_size != actual_block_size:
            mismatches.append(
                "attention_config.block_size_tokens: "
                f"expected={expected_block_size!r} actual={actual_block_size!r}"
            )
        expected_sink_blocks = _safe_int(
            runtime_attention_config.get("effective_sink_block_count"),
            0,
        )
        actual_sink_blocks = _safe_int(
            manifest_attention_config.get("effective_sink_block_count"),
            0,
        )
        if expected_sink_blocks > 0 and expected_sink_blocks != actual_sink_blocks:
            mismatches.append(
                "attention_config.effective_sink_block_count: "
                f"expected={expected_sink_blocks!r} actual={actual_sink_blocks!r}"
            )
    legacy_missing_strict_inputs = (
        str(normalized_manifest.get("schema_version") or "")
        == _LEGACY_PREFIX_SESSION_SCHEMA_VERSION
        and str(normalized_manifest.get("attention_mode") or "") == _ATTENTION_MODE_FULL
        and not isinstance(normalized_manifest.get("strict_env_inputs"), Mapping)
    )
    if (
        strict
        and str(strict_env_hash or "") != str(normalized_manifest.get("strict_env_hash") or "")
        and not legacy_missing_strict_inputs
    ):
        mismatches.append(
            "strict_env_hash: "
            "expected="
            f"{str(strict_env_hash)!r} actual={str(normalized_manifest.get('strict_env_hash') or '')!r}"
        )
    return len(mismatches) == 0, mismatches


def require_collective_rpc_ok(
    *,
    method: str,
    results: List[Dict[str, Any]],
) -> None:
    """Raise when any rank reports a collective RPC error."""

    errors: List[str] = []
    for row in results:
        ok = row.get("ok")
        if ok is False:
            rank = row.get("rank")
            if not isinstance(rank, int):
                rank = row.get("rank_index")
            errors.append(f"rank={rank}: {row.get('error') or row.get('errors') or 'rpc_failed'}")
    if errors:
        raise RuntimeError(f"{method} failed: " + "; ".join([str(x) for x in errors]))


def call_collective_rpc(
    *,
    base_url: str,
    method: str,
    args: List[Any],
    kwargs: Optional[Mapping[str, Any]] = None,
    timeout_sec: int,
) -> List[Dict[str, Any]]:
    """Call the vLLM dev-mode collective RPC surface and normalize results."""

    rpc_url = f"{base_url.rstrip('/')}/collective_rpc"
    payload: Dict[str, Any] = {
        "method": str(method),
        "args": [str(x) for x in list(args)],
        "kwargs": {str(k): str(v) for k, v in dict(kwargs or {}).items()},
    }
    if int(timeout_sec) > 0:
        payload["timeout"] = float(timeout_sec)
    raw = post_json(rpc_url, payload, timeout_sec=max(10, int(timeout_sec) + 5))
    results = raw.get("results")
    if not isinstance(results, list):
        raise RuntimeError(
            f"collective_rpc returned invalid results type: {type(results).__name__}"
        )
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(results):
        if isinstance(item, dict):
            row = dict(item)
        else:
            row = {
                "ok": False,
                "rank_index": int(idx),
                "error": f"non_dict_result:{type(item).__name__}",
                "value": item,
            }
        row.setdefault("rank_index", int(idx))
        normalized.append(row)
    return normalized


def register_loaded_prefix_cache(
    *,
    base_url: str,
    prefix_session_dir: str,
    prefix_session_id: str,
    timeout_sec: int,
    plan_id: str = "",
    abort: bool = False,
) -> Dict[str, Any]:
    """Register a restored prefix session with vLLM's prefix-cache hash index.

    Args:
        base_url: vLLM server base URL.
        prefix_session_dir: Root directory containing saved prefix sessions.
        prefix_session_id: Saved prefix-session identifier.
        timeout_sec: Request timeout in seconds.
        plan_id: Optional prepared registration plan identifier.
        abort: Whether to abort a prepared plan instead of finalizing it.

    Returns:
        Parsed JSON response from the registration endpoint.
    """

    payload = {
        "session_dir": str(prefix_session_dir),
        "session_id": str(prefix_session_id),
    }
    if str(plan_id or "").strip():
        payload["plan_id"] = str(plan_id)
    if bool(abort):
        payload["abort"] = True
    return post_json(
        f"{base_url.rstrip('/')}/register_loaded_prefix_cache",
        payload,
        timeout_sec=max(10, int(timeout_sec)),
    )


def prepare_loaded_prefix_cache(
    *,
    base_url: str,
    prefix_session_dir: str,
    prefix_session_id: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    """Prepare final APC target rows for a saved prefix session."""
    payload = {
        "session_dir": str(prefix_session_dir),
        "session_id": str(prefix_session_id),
    }
    return post_json(
        f"{base_url.rstrip('/')}/prepare_loaded_prefix_cache",
        payload,
        timeout_sec=max(10, int(timeout_sec)),
    )


def run_hot_prefix_completion(
    *,
    base_url: str,
    model_id: str,
    tokenizer: Any,
    prefix_session_dir: str,
    prefix_session_id: str,
    suffix_text: str,
    continuation_tail_token_ids: List[int],
    prefix_token_count: int,
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
    request_id: Optional[str] = None,
    stream_progress_cb: Optional[Any] = None,
    stream_text_completions_fn: Any = stream_text_completions,
) -> Dict[str, Any]:
    """Run a server-side hot-prefix completion against a saved session.

    Args:
        base_url: vLLM server base URL.
        model_id: Target model identifier.
        tokenizer: Hugging Face-compatible tokenizer.
        prefix_session_dir: Root directory containing saved prefix sessions.
        prefix_session_id: Saved prefix-session identifier.
        suffix_text: User question appended after the captured prefix boundary.
        continuation_tail_token_ids: Saved same-message continuation tail token IDs.
        prefix_token_count: Saved captured prefix token count.
        max_new_tokens: Maximum completion tokens.
        min_new_tokens: Minimum completion tokens.
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
        request_id: Optional request identifier forwarded to the backend.
        stream_progress_cb: Optional callback for stream deltas.
        stream_text_completions_fn: Streaming helper used for tests.

    Returns:
        Metrics dictionary describing the request and completion.
    """

    payload: Dict[str, Any] = {
        "model": model_id,
        "session_dir": str(prefix_session_dir),
        "session_id": str(prefix_session_id),
        "suffix_text": str(suffix_text),
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
    if str(request_id or "").strip():
        payload["request_id"] = str(request_id).strip()
    _apply_generation_output_mask(payload, output_mask_payload)

    expected_prompt_tokens = (
        int(prefix_token_count)
        + len(tokenize_text(tokenizer, suffix_text))
        + len(list(continuation_tail_token_ids or []))
    )
    endpoint_url = f"{base_url.rstrip('/')}/query_hot_prefix_completion"

    if use_stream:
        result = dict(
            stream_text_completions_fn(
                url=endpoint_url,
                payload=payload,
                timeout_sec=int(timeout_sec),
                on_progress=stream_progress_cb,
            )
        )
        generated_text = str(result.get("answer_text") or result.get("text") or "")
        usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else {}
        prompt_tokens = int(
            usage.get("prompt_tokens")
            if isinstance(usage.get("prompt_tokens"), (int, float))
            else expected_prompt_tokens
        )
        completion_tokens = int(
            usage.get("completion_tokens")
            if isinstance(usage.get("completion_tokens"), (int, float))
            else len(tokenize_text(tokenizer, generated_text))
        )
        total_tokens = int(
            usage.get("total_tokens")
            if isinstance(usage.get("total_tokens"), (int, float))
            else prompt_tokens + completion_tokens
        )
        cached_prompt_tokens = _extract_cached_prompt_tokens(usage)
        ttft_ms = int(result.get("ttft_ms") or 0)
        total_ms = int(result.get("total_ms") or 0)
        decode_ms = max(0, total_ms - ttft_ms)
        decode_msp = (float(decode_ms) / max(1, completion_tokens)) if completion_tokens > 0 else 0.0
        gen_tps = (float(completion_tokens) / (float(decode_ms) / 1000.0)) if decode_ms > 0 else 0.0
        total_tps = (float(total_tokens) / (float(total_ms) / 1000.0)) if total_ms > 0 else 0.0
        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "ttft_ms": int(ttft_ms),
            "decode_ms": int(decode_ms),
            "total_ms": int(total_ms),
            "decode_ms_per_token": round(float(decode_msp), 6),
            "gen_tokens_per_sec": round(float(gen_tps), 6),
            "total_tokens_per_sec": round(float(total_tps), 6),
            "finish_reason": str(result.get("finish_reason") or ""),
            "stop_reason": str(result.get("stop_reason") or ""),
            "generated_text": generated_text,
            "request_id": str(result.get("request_id") or ""),
            "usage_cached_prompt_tokens": int(cached_prompt_tokens),
        }

    result = post_json(endpoint_url, payload, timeout_sec=int(timeout_sec))
    if isinstance(result, Mapping) and isinstance(result.get("error"), Mapping):
        error = result.get("error") or {}
        raise RuntimeError(
            "Hot prefix completion failed: "
            f"{error.get('message') or error.get('code') or result}"
        )
    choices = result.get("choices") if isinstance(result, Mapping) else None
    first = choices[0] if isinstance(choices, list) and choices else {}
    generated_text = str(first.get("text") or "")
    finish_reason = str(first.get("finish_reason") or "")
    stop_reason = str(first.get("stop_reason") or "")
    usage = (
        result.get("usage")
        if isinstance(result, Mapping) and isinstance(result.get("usage"), Mapping)
        else {}
    )
    prompt_tokens = int(
        usage.get("prompt_tokens")
        if isinstance(usage.get("prompt_tokens"), (int, float))
        else expected_prompt_tokens
    )
    completion_tokens = int(
        usage.get("completion_tokens")
        if isinstance(usage.get("completion_tokens"), (int, float))
        else len(tokenize_text(tokenizer, generated_text))
    )
    total_tokens = int(
        usage.get("total_tokens")
        if isinstance(usage.get("total_tokens"), (int, float))
        else prompt_tokens + completion_tokens
    )
    cached_prompt_tokens = _extract_cached_prompt_tokens(usage)
    ttft_ms = int(result.get("ttft_ms") or 0)
    total_ms = int(result.get("total_ms") or 0)
    decode_ms = max(0, total_ms - ttft_ms)
    decode_msp = (float(decode_ms) / max(1, completion_tokens)) if completion_tokens > 0 else 0.0
    gen_tps = (float(completion_tokens) / (float(decode_ms) / 1000.0)) if decode_ms > 0 else 0.0
    total_tps = (float(total_tokens) / (float(total_ms) / 1000.0)) if total_ms > 0 else 0.0
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "ttft_ms": int(ttft_ms),
        "decode_ms": int(decode_ms),
        "total_ms": int(total_ms),
        "decode_ms_per_token": round(float(decode_msp), 6),
        "gen_tokens_per_sec": round(float(gen_tps), 6),
        "total_tokens_per_sec": round(float(total_tps), 6),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "generated_text": generated_text,
        "request_id": str(result.get("id") or ""),
        "usage_cached_prompt_tokens": int(cached_prompt_tokens),
    }


def wait_for_vllm_health(
    base_url: str,
    timeout_sec: int,
    proc: Any = None,
) -> bool:
    """Wait for the vLLM health endpoint to return HTTP 200."""

    deadline = time.time() + float(timeout_sec)
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        if proc is not None and getattr(proc, "poll", None) is not None and proc.poll() is not None:
            return False
        try:
            with url_request.urlopen(health_url, timeout=5) as resp:
                if int(getattr(resp, "status", 0) or 0) == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _apply_generation_output_mask(
    payload: Dict[str, Any],
    output_mask_payload: Optional[Mapping[str, Any]],
) -> None:
    """Merge generation-time output masking into an OpenAI request payload."""

    if not output_mask_payload:
        return
    for key, value in output_mask_payload.items():
        payload[key] = value


def run_completion_prompt_ids(
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
    request_id: Optional[str] = None,
    request_debug_payload: Optional[Dict[str, Any]] = None,
    stream_progress_cb: Optional[Any] = None,
    stream_text_completions_fn: Any = stream_text_completions,
) -> Dict[str, Any]:
    """Run one vLLM text-completions request from prompt token IDs."""

    payload: Dict[str, Any] = {
        "model": model_id,
        "prompt": [int(x) for x in prompt_token_ids],
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
    if str(request_id or "").strip():
        payload["request_id"] = str(request_id).strip()
    _apply_generation_output_mask(payload, output_mask_payload)
    if request_debug_payload is not None:
        request_debug_payload["request_endpoint"] = "/v1/completions"
        request_debug_payload["query_api_mode_resolved"] = "prompt_ids"
        if bool(request_debug_payload.get("dump_request_payload")):
            request_debug_payload["request_payload"] = json.loads(
                json.dumps(payload, sort_keys=True)
            )
        if bool(request_debug_payload.get("dump_prompt_token_ids")):
            request_debug_payload["prompt_token_ids"] = [
                int(token_id) for token_id in list(prompt_token_ids or [])
            ]

    expected_prompt_tokens = max(
        0,
        int(prompt_token_count_override)
        if prompt_token_count_override is not None
        else len(prompt_token_ids),
    )

    if use_stream:
        result = dict(
            stream_text_completions_fn(
                url=f"{base_url.rstrip('/')}/v1/completions",
                payload=payload,
                timeout_sec=int(timeout_sec),
                on_progress=stream_progress_cb,
            )
        )
        generated_text = str(result.get("answer_text") or result.get("text") or "")
        usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else {}
        prompt_tokens = int(
            usage.get("prompt_tokens")
            if isinstance(usage.get("prompt_tokens"), (int, float))
            else expected_prompt_tokens
        )
        prompt_tokens = max(prompt_tokens, expected_prompt_tokens)
        completion_tokens = int(
            usage.get("completion_tokens")
            if isinstance(usage.get("completion_tokens"), (int, float))
            else len(tokenize_text(tokenizer, generated_text))
        )
        total_tokens = int(
            usage.get("total_tokens")
            if isinstance(usage.get("total_tokens"), (int, float))
            else prompt_tokens + completion_tokens
        )
        cached_prompt_tokens = _extract_cached_prompt_tokens(usage)
        ttft_ms = int(result.get("ttft_ms") or 0)
        total_ms = int(result.get("total_ms") or 0)
        decode_ms = max(0, total_ms - ttft_ms)
        decode_msp = (float(decode_ms) / max(1, completion_tokens)) if completion_tokens > 0 else 0.0
        gen_tps = (float(completion_tokens) / (float(decode_ms) / 1000.0)) if decode_ms > 0 else 0.0
        total_tps = (float(total_tokens) / (float(total_ms) / 1000.0)) if total_ms > 0 else 0.0
        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "ttft_ms": int(ttft_ms),
            "decode_ms": int(decode_ms),
            "total_ms": int(total_ms),
            "decode_ms_per_token": round(float(decode_msp), 6),
            "gen_tokens_per_sec": round(float(gen_tps), 6),
            "total_tokens_per_sec": round(float(total_tps), 6),
            "finish_reason": str(result.get("finish_reason") or ""),
            "stop_reason": str(result.get("stop_reason") or ""),
            "generated_text": generated_text,
            "request_id": str(result.get("request_id") or ""),
            "usage_cached_prompt_tokens": int(cached_prompt_tokens),
        }

    result = post_json(
        f"{base_url.rstrip('/')}/v1/completions",
        payload,
        timeout_sec=int(timeout_sec),
    )
    choices = result.get("choices") if isinstance(result, Mapping) else None
    first = choices[0] if isinstance(choices, list) and choices else {}
    generated_text = str(first.get("text") or "")
    finish_reason = str(first.get("finish_reason") or "")
    stop_reason = str(first.get("stop_reason") or "")
    usage = (
        result.get("usage")
        if isinstance(result, Mapping) and isinstance(result.get("usage"), Mapping)
        else {}
    )
    prompt_tokens = int(
        usage.get("prompt_tokens")
        if isinstance(usage.get("prompt_tokens"), (int, float))
        else expected_prompt_tokens
    )
    prompt_tokens = max(prompt_tokens, expected_prompt_tokens)
    completion_tokens = int(
        usage.get("completion_tokens")
        if isinstance(usage.get("completion_tokens"), (int, float))
        else len(tokenize_text(tokenizer, generated_text))
    )
    total_tokens = int(
        usage.get("total_tokens")
        if isinstance(usage.get("total_tokens"), (int, float))
        else prompt_tokens + completion_tokens
    )
    cached_prompt_tokens = _extract_cached_prompt_tokens(usage)
    ttft_ms = int(result.get("ttft_ms") or 0)
    total_ms = int(result.get("total_ms") or 0)
    decode_ms = max(0, total_ms - ttft_ms)
    decode_msp = (float(decode_ms) / max(1, completion_tokens)) if completion_tokens > 0 else 0.0
    gen_tps = (float(completion_tokens) / (float(decode_ms) / 1000.0)) if decode_ms > 0 else 0.0
    total_tps = (float(total_tokens) / (float(total_ms) / 1000.0)) if total_ms > 0 else 0.0
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "ttft_ms": int(ttft_ms),
        "decode_ms": int(decode_ms),
        "total_ms": int(total_ms),
        "decode_ms_per_token": round(float(decode_msp), 6),
        "gen_tokens_per_sec": round(float(gen_tps), 6),
        "total_tokens_per_sec": round(float(total_tps), 6),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "generated_text": generated_text,
        "request_id": str(result.get("id") or ""),
        "usage_cached_prompt_tokens": int(cached_prompt_tokens),
    }


def run_chat_completion_messages(
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
    request_id: Optional[str] = None,
    stream_progress_cb: Optional[Any] = None,
    stream_chat_completions_fn: Any = stream_chat_completions,
) -> Dict[str, Any]:
    """Run one vLLM chat-completions request from rendered messages.

    Args:
        base_url: vLLM server base URL.
        model_id: Target model identifier.
        tokenizer: Hugging Face-compatible tokenizer.
        messages: Rendered OpenAI-style chat messages.
        max_new_tokens: Maximum completion tokens.
        min_new_tokens: Minimum completion tokens.
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
        vllm_xargs: Optional custom request arguments forwarded to vLLM.
        request_id: Optional request identifier forwarded to the backend.
        stream_progress_cb: Optional callback for stream deltas.
        stream_chat_completions_fn: Streaming helper used for tests.

    Returns:
        Metrics dictionary describing the request and completion.
    """

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [dict(msg) for msg in messages],
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
    if str(request_id or "").strip():
        payload["request_id"] = str(request_id).strip()
    _apply_generation_output_mask(payload, output_mask_payload)

    expected_prompt_tokens = max(0, len(tokenize_messages(tokenizer, messages)))

    if use_stream:
        result = dict(
            stream_chat_completions_fn(
                url=f"{base_url.rstrip('/')}/v1/chat/completions",
                payload=payload,
                timeout_sec=int(timeout_sec),
                on_progress=stream_progress_cb,
            )
        )
        generated_text = str(result.get("answer_text") or result.get("text") or "")
        usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else {}
        prompt_tokens = int(
            usage.get("prompt_tokens")
            if isinstance(usage.get("prompt_tokens"), (int, float))
            else expected_prompt_tokens
        )
        completion_tokens = int(
            usage.get("completion_tokens")
            if isinstance(usage.get("completion_tokens"), (int, float))
            else len(tokenize_text(tokenizer, generated_text))
        )
        total_tokens = int(
            usage.get("total_tokens")
            if isinstance(usage.get("total_tokens"), (int, float))
            else prompt_tokens + completion_tokens
        )
        cached_prompt_tokens = _extract_cached_prompt_tokens(usage)
        ttft_ms = int(result.get("ttft_ms") or 0)
        total_ms = int(result.get("total_ms") or 0)
        decode_ms = max(0, total_ms - ttft_ms)
        decode_msp = (float(decode_ms) / max(1, completion_tokens)) if completion_tokens > 0 else 0.0
        gen_tps = (float(completion_tokens) / (float(decode_ms) / 1000.0)) if decode_ms > 0 else 0.0
        total_tps = (float(total_tokens) / (float(total_ms) / 1000.0)) if total_ms > 0 else 0.0
        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "ttft_ms": int(ttft_ms),
            "decode_ms": int(decode_ms),
            "total_ms": int(total_ms),
            "decode_ms_per_token": round(float(decode_msp), 6),
            "gen_tokens_per_sec": round(float(gen_tps), 6),
            "total_tokens_per_sec": round(float(total_tps), 6),
            "finish_reason": str(result.get("finish_reason") or ""),
            "stop_reason": str(result.get("stop_reason") or ""),
            "generated_text": generated_text,
            "request_id": str(result.get("request_id") or ""),
            "usage_cached_prompt_tokens": int(cached_prompt_tokens),
        }

    result = post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        payload,
        timeout_sec=int(timeout_sec),
    )
    choices = result.get("choices") if isinstance(result, Mapping) else None
    first = choices[0] if isinstance(choices, list) and choices else {}
    generated_text = str(extract_response_text(result) or "")
    finish_reason = str(first.get("finish_reason") or "")
    stop_reason = str(first.get("stop_reason") or "")
    usage = (
        result.get("usage")
        if isinstance(result, Mapping) and isinstance(result.get("usage"), Mapping)
        else {}
    )
    prompt_tokens = int(
        usage.get("prompt_tokens")
        if isinstance(usage.get("prompt_tokens"), (int, float))
        else expected_prompt_tokens
    )
    completion_tokens = int(
        usage.get("completion_tokens")
        if isinstance(usage.get("completion_tokens"), (int, float))
        else len(tokenize_text(tokenizer, generated_text))
    )
    total_tokens = int(
        usage.get("total_tokens")
        if isinstance(usage.get("total_tokens"), (int, float))
        else prompt_tokens + completion_tokens
    )
    cached_prompt_tokens = _extract_cached_prompt_tokens(usage)
    ttft_ms = int(result.get("ttft_ms") or 0)
    total_ms = int(result.get("total_ms") or 0)
    decode_ms = max(0, total_ms - ttft_ms)
    decode_msp = (float(decode_ms) / max(1, completion_tokens)) if completion_tokens > 0 else 0.0
    gen_tps = (float(completion_tokens) / (float(decode_ms) / 1000.0)) if decode_ms > 0 else 0.0
    total_tps = (float(total_tokens) / (float(total_ms) / 1000.0)) if total_ms > 0 else 0.0
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "ttft_ms": int(ttft_ms),
        "decode_ms": int(decode_ms),
        "total_ms": int(total_ms),
        "decode_ms_per_token": round(float(decode_msp), 6),
        "gen_tokens_per_sec": round(float(gen_tps), 6),
        "total_tokens_per_sec": round(float(total_tps), 6),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "generated_text": generated_text,
        "request_id": str(result.get("id") or ""),
        "usage_cached_prompt_tokens": int(cached_prompt_tokens),
    }


def _fetch_server_model_ids(base_url: str, timeout_sec: int) -> List[str]:
    """Fetch model IDs from the OpenAI-compatible models endpoint."""

    try:
        with url_request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=max(1, int(timeout_sec))) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    model_ids: List[str] = []
    for item in data:
        if isinstance(item, Mapping):
            model_id = str(item.get("id") or "").strip()
            if model_id:
                model_ids.append(model_id)
    return model_ids


def _normalize_model_id_aliases(model_id: str) -> set[str]:
    """Return equivalent aliases for a saved/server model identifier.

    This accepts both the OpenAI-compatible served model alias
    (for example ``moonshotai/Kimi-Linear-48B-A3B-Instruct``) and the
    fully-resolved local HF snapshot path used in saved manifests
    (for example ``/hf/hub/models--moonshotai--Kimi.../snapshots/<sha>``).
    """

    raw = str(model_id or "").strip()
    if not raw:
        return set()
    aliases = {raw}
    trimmed = raw.rstrip("/")
    aliases.add(trimmed)
    aliases.add(Path(trimmed).name)
    marker = "/models--"
    if marker in trimmed and "/snapshots/" in trimmed:
        try:
            repo_segment = trimmed.split(marker, 1)[1].split("/snapshots/", 1)[0]
        except Exception:
            repo_segment = ""
        if repo_segment:
            repo_parts = [part for part in repo_segment.split("--") if part]
            if len(repo_parts) >= 2:
                repo_id = "/".join(repo_parts[:2])
                aliases.add(repo_id)
            aliases.add(repo_segment)
    return {alias for alias in aliases if alias}


def _model_ids_are_compatible(expected_model_id: str, server_model_ids: Sequence[str]) -> bool:
    """Return whether one saved model identifier matches any live server alias."""

    expected_aliases = _normalize_model_id_aliases(expected_model_id)
    if not expected_aliases:
        return True
    actual_aliases: set[str] = set()
    for candidate in list(server_model_ids or []):
        actual_aliases.update(_normalize_model_id_aliases(str(candidate)))
    return bool(expected_aliases & actual_aliases)


def _resolve_compatible_server_model_id(
    expected_model_id: str,
    server_model_ids: Sequence[str],
) -> str:
    """Return the best live server model id to use for OpenAI requests."""

    expected_aliases = _normalize_model_id_aliases(expected_model_id)
    for candidate in list(server_model_ids or []):
        if expected_aliases & _normalize_model_id_aliases(str(candidate)):
            return str(candidate).strip()
    return str(expected_model_id or "").strip()


@dataclass(slots=True)
class HotPrefixSessionConfig:
    """Configuration for a reusable hot-prefix question-answering session."""

    base_url: str
    prefix_session_dir: str | Path
    prefix_session_id: str
    timeout_sec: int = 600
    health_timeout_sec: int = 240
    compat_strict: bool = True
    trust_remote_code: bool = False
    auto_restore_on_first_query: bool = True
    request_strategy: str = "prefix_cache_replay"
    max_new_tokens: int = 256
    min_new_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    use_stream: bool = True
    output_mask_payload: Optional[Mapping[str, Any]] = None


@dataclass(slots=True)
class HotPrefixSessionInfo:
    """Static metadata describing one saved prefix session."""

    session_id: str
    session_dir: str
    model_id: str
    tokenizer_id: str
    prompt_shape_id: str
    prefix_token_count: int
    continuation_tail_token_count: int
    schema_version: str
    attention_mode: str
    attention_config: Dict[str, Any]
    capture_topology: Dict[str, Any]
    prefix_capture: Dict[str, Any]
    prefix_state_format: Dict[str, Any]


@dataclass(slots=True)
class HotPrefixQueryResult:
    """User-facing result for one hot-prefix query."""

    text: str
    session_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: int
    decode_ms: int
    total_ms: int
    request_id: str
    finish_reason: str
    stop_reason: str
    prefix_state_load_count: int
    prefix_restored_this_call: bool
    cached_prompt_tokens: int = 0


class HotPrefixSessionClient:
    """Client for repeatedly querying a saved prefix session kept hot in HBM."""

    def __init__(self, config: HotPrefixSessionConfig, tokenizer: Any | None = None) -> None:
        self.config = config
        self._session_root = Path(config.prefix_session_dir).expanduser().resolve()
        self._loaded_session = load_prefix_session(
            prefix_session_dir=self._session_root,
            prefix_session_id=str(config.prefix_session_id),
        )
        self._manifest = dict(self._loaded_session.get("manifest") or {})
        self._prefix_capture = normalize_prefix_capture(self._manifest.get("prefix_capture"))
        if str(self._prefix_capture.get("continuation_mode") or "") != "same_message":
            raise ValueError(
                "HotPrefixSessionClient currently supports only same_message prefix_capture sessions"
            )
        tokenizer_id = str(
            self._manifest.get("tokenizer_id") or self._manifest.get("model_id") or ""
        )
        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=bool(config.trust_remote_code),
        )
        self._validate_local_session_compat()
        self._prefix_cache_block_size_tokens = resolve_prefix_cache_block_size(
            manifest=self._manifest,
            session_dir=Path(str(self._loaded_session.get("session_dir") or "")),
        )
        self._hot_prefix_reuse_plan = build_hot_prefix_reuse_plan(
            prefix_token_ids=[
                int(x) for x in list(self._loaded_session.get("prefix_token_ids") or [])
            ],
            block_size_tokens=int(self._prefix_cache_block_size_tokens),
            prefer_full_prefix_reuse=(
                str(self._manifest.get("attention_mode") or "")
                == _ATTENTION_MODE_STREAMING
                and _streaming_same_message_reuse_mode() == "full"
            ),
        )
        self._streaming_prepared_manager_rows: List[Dict[str, Any]] = []
        self._streaming_prepared_layout: Dict[str, Any] = {}
        self._prefix_state_loaded = False
        self._prefix_cache_registered = False
        self._prefix_state_load_count = 0
        self._resolved_server_model_id: str = ""
        self._info = HotPrefixSessionInfo(
            session_id=str(self._manifest.get("session_id") or config.prefix_session_id),
            session_dir=str(self._loaded_session.get("session_dir") or ""),
            model_id=str(self._manifest.get("model_id") or ""),
            tokenizer_id=str(self._manifest.get("tokenizer_id") or ""),
            prompt_shape_id=str(self._manifest.get("prompt_shape_id") or ""),
            prefix_token_count=int(self._manifest.get("prefix_token_count") or 0),
            continuation_tail_token_count=int(
                self._manifest.get("continuation_tail_token_count") or 0
            ),
            schema_version=str(self._manifest.get("schema_version") or ""),
            attention_mode=str(self._manifest.get("attention_mode") or ""),
            attention_config=dict(self._manifest.get("attention_config") or {}),
            capture_topology=dict(self._manifest.get("capture_topology") or {}),
            prefix_capture=dict(self._prefix_capture),
            prefix_state_format=dict(self._manifest.get("prefix_state_format") or {}),
        )

    @classmethod
    def from_config(
        cls,
        config: HotPrefixSessionConfig,
        tokenizer: Any | None = None,
    ) -> "HotPrefixSessionClient":
        """Construct a client from configuration."""

        return cls(config=config, tokenizer=tokenizer)

    @property
    def prefix_state_load_count(self) -> int:
        """Return how many times prefix state has been loaded into HBM."""

        return int(self._prefix_state_load_count)

    def session_info(self) -> HotPrefixSessionInfo:
        """Return static metadata describing the saved prefix session."""

        return HotPrefixSessionInfo(
            session_id=self._info.session_id,
            session_dir=self._info.session_dir,
            model_id=self._info.model_id,
            tokenizer_id=self._info.tokenizer_id,
            prompt_shape_id=self._info.prompt_shape_id,
            prefix_token_count=int(self._info.prefix_token_count),
            continuation_tail_token_count=int(self._info.continuation_tail_token_count),
            schema_version=str(self._info.schema_version),
            attention_mode=str(self._info.attention_mode),
            attention_config=dict(self._info.attention_config),
            capture_topology=dict(self._info.capture_topology),
            prefix_capture=dict(self._info.prefix_capture),
            prefix_state_format=dict(self._info.prefix_state_format),
        )

    def _validate_local_session_compat(self) -> None:
        expected_hash = str(self._manifest.get("chat_template_hash") or "")
        actual_hash = chat_template_hash(self._tokenizer)
        if bool(self.config.compat_strict) and expected_hash and expected_hash != actual_hash:
            raise RuntimeError(
                "Prefix session chat template mismatch: "
                f"expected={expected_hash} actual={actual_hash}"
            )

    def ensure_server_healthy(self) -> None:
        """Verify that the configured vLLM server is reachable and compatible."""

        ok = wait_for_vllm_health(
            str(self.config.base_url),
            int(self.config.health_timeout_sec),
        )
        if not ok:
            raise RuntimeError(
                f"vLLM server did not become healthy: {self.config.base_url!r}"
            )
        if not bool(self.config.compat_strict):
            return
        model_ids = _fetch_server_model_ids(
            str(self.config.base_url),
            int(self.config.timeout_sec),
        )
        expected_model_id = str(self._manifest.get("model_id") or "")
        self._resolved_server_model_id = _resolve_compatible_server_model_id(
            expected_model_id,
            model_ids,
        )
        if model_ids and expected_model_id and not _model_ids_are_compatible(
            expected_model_id,
            model_ids,
        ):
            raise RuntimeError(
                "Running server model does not match saved prefix session: "
                f"expected={expected_model_id!r} actual={model_ids!r}"
            )

    def _request_model_id(self) -> str:
        """Return the live server model id to send with OpenAI-compatible requests."""

        resolved = str(self._resolved_server_model_id or "").strip()
        if resolved:
            return resolved
        return str(self._manifest.get("model_id") or "").strip()

    def restore_prefix_state(self, force: bool = False) -> bool:
        """Load saved prefix state into the running vLLM server HBM cache."""

        self.ensure_server_healthy()
        if self._prefix_state_loaded and not bool(force):
            return False
        self._prefix_cache_registered = False
        self._streaming_prepared_manager_rows = []
        self._streaming_prepared_layout = {}
        chunk_bytes = int(
            ((self._manifest.get("prefix_state_format") or {}).get("chunk_bytes") or 0)
        )
        if chunk_bytes <= 0:
            raise RuntimeError("Saved prefix session is missing prefix_state_format.chunk_bytes")

        def _prepare_loaded_prefix_cache_with_retry() -> Mapping[str, Any]:
            """Retry transient prepare failures caused by in-flight cache resets."""

            last_payload: Mapping[str, Any] | None = None
            for attempt in range(12):
                payload = prepare_loaded_prefix_cache(
                    base_url=str(self.config.base_url),
                    prefix_session_dir=str(self._session_root),
                    prefix_session_id=str(self.config.prefix_session_id),
                    timeout_sec=int(self.config.timeout_sec),
                )
                last_payload = payload
                if bool(payload.get("ok")):
                    return payload
                if str(payload.get("error") or "") != "reset_prefix_cache_failed":
                    return payload
                time.sleep(min(1.0 + 0.5 * float(attempt), 4.0))
            return dict(last_payload or {})

        if self._request_strategy() == "prefix_cache_replay":
            prepare_payload = _prepare_loaded_prefix_cache_with_retry()
            if not bool(prepare_payload.get("ok")):
                raise RuntimeError(f"prepare_loaded_prefix_cache failed: {prepare_payload}")
            plan_id = str(prepare_payload.get("plan_id") or "").strip()
            results = call_collective_rpc(
                base_url=str(self.config.base_url),
                method="load_prefix_state_into_apc_targets",
                args=[
                    str(self._session_root),
                    str(self.config.prefix_session_id),
                    json.dumps(
                        dict(prepare_payload.get("linear_target_last_rows") or {}),
                        sort_keys=True,
                    ),
                    json.dumps(list(prepare_payload.get("global_target_rows") or [])),
                    int(prepare_payload.get("global_target_group_id") or -1),
                    int(bool(self.config.compat_strict)),
                    int(chunk_bytes),
                ],
                timeout_sec=int(self.config.timeout_sec),
            )
            try:
                require_collective_rpc_ok(
                    method="load_prefix_state_into_apc_targets",
                    results=results,
                )
            except Exception:  # noqa: BLE001
                if plan_id:
                    register_loaded_prefix_cache(
                        base_url=str(self.config.base_url),
                        prefix_session_dir=str(self._session_root),
                        prefix_session_id=str(self.config.prefix_session_id),
                        timeout_sec=int(self.config.timeout_sec),
                        plan_id=plan_id,
                        abort=True,
                    )
                raise
            register_payload = register_loaded_prefix_cache(
                base_url=str(self.config.base_url),
                prefix_session_dir=str(self._session_root),
                prefix_session_id=str(self.config.prefix_session_id),
                timeout_sec=int(self.config.timeout_sec),
                plan_id=str(prepare_payload.get("plan_id") or ""),
            )
            if not bool(register_payload.get("ok")):
                raise RuntimeError(f"register_loaded_prefix_cache failed: {register_payload}")
            self._prefix_cache_registered = True
        elif (
            self._request_strategy() == "compact_hot_prefix"
            and str(self._manifest.get("attention_mode") or "") == _ATTENTION_MODE_STREAMING
        ):
            prepare_payload = _prepare_loaded_prefix_cache_with_retry()
            if not bool(prepare_payload.get("ok")):
                raise RuntimeError(f"prepare_loaded_prefix_cache failed: {prepare_payload}")
            prepare_payload = project_streaming_prepare_payload_to_saved_layout(
                prepare_payload,
                self._manifest,
            )
            plan_id = str(prepare_payload.get("plan_id") or "").strip()
            if (
                str(self._manifest.get("attention_mode") or "") != _ATTENTION_MODE_STREAMING
                and not plan_id
            ):
                raise RuntimeError(
                    f"prepare_loaded_prefix_cache returned no plan_id: {prepare_payload}"
                )
            results = call_collective_rpc(
                base_url=str(self.config.base_url),
                method="load_prefix_state_into_apc_targets",
                args=[
                    str(self._session_root),
                    str(self.config.prefix_session_id),
                    json.dumps(
                        dict(prepare_payload.get("linear_target_last_rows") or {}),
                        sort_keys=True,
                    ),
                    json.dumps(list(prepare_payload.get("global_target_rows") or [])),
                    int(prepare_payload.get("global_target_group_id") or -1),
                    int(bool(self.config.compat_strict)),
                    int(chunk_bytes),
                ],
                timeout_sec=int(self.config.timeout_sec),
            )
            try:
                require_collective_rpc_ok(
                    method="load_prefix_state_into_apc_targets",
                    results=results,
                )
            except Exception:  # noqa: BLE001
                register_loaded_prefix_cache(
                    base_url=str(self.config.base_url),
                    prefix_session_dir=str(self._session_root),
                    prefix_session_id=str(self.config.prefix_session_id),
                    timeout_sec=int(self.config.timeout_sec),
                    plan_id=plan_id,
                    abort=True,
                )
                raise
            self._streaming_prepared_manager_rows = (
                build_streaming_manager_rows_from_prepared_layout(prepare_payload)
            )
            self._streaming_prepared_layout = dict(prepare_payload)
        else:
            results = call_collective_rpc(
                base_url=str(self.config.base_url),
                method="load_prefix_state",
                args=[
                    str(self._session_root),
                    str(self.config.prefix_session_id),
                    int(bool(self.config.compat_strict)),
                    int(chunk_bytes),
                    int(self._loaded_session.get("prefix_state_row_shift") or 0),
                ],
                timeout_sec=int(self.config.timeout_sec),
            )
            require_collective_rpc_ok(method="load_prefix_state", results=results)
        self._prefix_state_loaded = True
        self._prefix_state_load_count += 1
        return True

    def _resolve_generation_options(self, overrides: Mapping[str, Any]) -> Dict[str, Any]:
        allowed = {
            "max_new_tokens",
            "min_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "ignore_eos",
            "use_stream",
            "output_mask_payload",
        }
        unknown = sorted(set(overrides.keys()) - allowed)
        if unknown:
            raise TypeError(f"Unsupported ask() override(s): {', '.join(unknown)}")
        options = {
            "max_new_tokens": int(self.config.max_new_tokens),
            "min_new_tokens": int(self.config.min_new_tokens),
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "top_k": int(self.config.top_k),
            "repetition_penalty": float(self.config.repetition_penalty),
            "presence_penalty": float(self.config.presence_penalty),
            "frequency_penalty": float(self.config.frequency_penalty),
            "ignore_eos": bool(self.config.ignore_eos),
            "use_stream": bool(self.config.use_stream),
            "output_mask_payload": self.config.output_mask_payload,
        }
        for key, value in overrides.items():
            options[key] = value
        options["max_new_tokens"] = int(options["max_new_tokens"])
        options["min_new_tokens"] = int(options["min_new_tokens"])
        options["temperature"] = float(options["temperature"])
        options["top_p"] = float(options["top_p"])
        options["top_k"] = int(options["top_k"])
        options["repetition_penalty"] = float(options["repetition_penalty"])
        options["presence_penalty"] = float(options["presence_penalty"])
        options["frequency_penalty"] = float(options["frequency_penalty"])
        options["ignore_eos"] = bool(options["ignore_eos"])
        options["use_stream"] = bool(options["use_stream"])
        return options

    def _request_strategy(self) -> str:
        """Return the normalized query transport strategy for this client."""

        strategy = str(self.config.request_strategy or "prefix_cache_replay").strip().lower()
        if (
            strategy == "prefix_cache_replay"
            and str(self._manifest.get("attention_mode") or "") == _ATTENTION_MODE_STREAMING
        ):
            # Streaming sessions save the exact same-message prompt boundary,
            # so we must retokenize the full prompt and slice at the reusable
            # prefix boundary. The legacy server-side suffix-only route
            # tokenizes ``suffix_text`` in isolation and can append extra chat
            # template tokens after the boundary.
            strategy = "compact_hot_prefix"
        if strategy not in {
            "prefix_cache_replay",
            "prefix_token_replay",
            "compact_hot_prefix",
            "server_hot_prefix",
        }:
            raise ValueError(
                "Unsupported HotPrefixSessionConfig.request_strategy: "
                f"{self.config.request_strategy!r}"
            )
        return strategy

    def _ask_one(
        self,
        question: str,
        *,
        force_restore_before_query: bool,
        generation_overrides: Mapping[str, Any],
    ) -> HotPrefixQueryResult:
        if not str(question or "").strip():
            raise ValueError("question must be non-empty")
        prefix_restored_this_call = False
        if force_restore_before_query:
            prefix_restored_this_call = self.restore_prefix_state(force=True)
        elif not self._prefix_state_loaded:
            if not bool(self.config.auto_restore_on_first_query):
                raise RuntimeError(
                    "Prefix state is not loaded. Call restore_prefix_state() first "
                    "or enable auto_restore_on_first_query."
                )
            prefix_restored_this_call = self.restore_prefix_state(force=False)
        else:
            self.ensure_server_healthy()

        options = self._resolve_generation_options(generation_overrides)
        request_strategy = self._request_strategy()
        request_model_id = self._request_model_id()
        if request_strategy == "prefix_cache_replay" and not self._prefix_cache_registered:
            register_loaded_prefix_cache(
                base_url=str(self.config.base_url),
                prefix_session_dir=str(self._session_root),
                prefix_session_id=str(self.config.prefix_session_id),
                timeout_sec=int(self.config.timeout_sec),
            )
            self._prefix_cache_registered = True
        continuation_tail_token_ids = [
            int(x) for x in list(self._loaded_session.get("continuation_tail_token_ids") or [])
        ]
        prompt_token_ids: List[int]
        virtual_prompt_tokens: Optional[int]
        vllm_xargs: Optional[Mapping[str, Any]]
        if request_strategy == "compact_hot_prefix":
            manifest = normalize_prefix_session_manifest(
                dict(self._loaded_session.get("manifest") or {})
            )
            if str(manifest.get("attention_mode") or "") == _ATTENTION_MODE_STREAMING:
                reused_prefix_token_count = int(
                    self._hot_prefix_reuse_plan.get("reused_prefix_token_count") or 0
                )
                prefix_messages = [
                    dict(msg)
                    for msg in list(self._loaded_session.get("prefix_messages") or [])
                ]
                query_messages = build_same_message_query_messages(
                    prefix_messages=prefix_messages,
                    suffix_text=str(question),
                    message_index=int(
                        self._prefix_capture.get("message_index")
                        or max(0, len(prefix_messages) - 1)
                    ),
                )
                vllm_xargs = {
                    "hot_prefix_token_count": int(reused_prefix_token_count)
                }
                streaming_manager_rows = [
                    dict(row) for row in self._streaming_prepared_manager_rows
                ] or build_streaming_hot_prefix_manager_rows(
                    manifest,
                    include_sparse_groups=True,
                    reused_prefix_token_count=int(reused_prefix_token_count),
                )
                if streaming_manager_rows:
                    vllm_xargs["hot_prefix_manager_rows"] = streaming_manager_rows
                metrics = run_chat_completion_messages(
                    base_url=str(self.config.base_url),
                    model_id=str(request_model_id),
                    tokenizer=self._tokenizer,
                    messages=query_messages,
                    max_new_tokens=int(options["max_new_tokens"]),
                    min_new_tokens=int(options["min_new_tokens"]),
                    timeout_sec=int(self.config.timeout_sec),
                    ignore_eos=bool(options["ignore_eos"]),
                    use_stream=bool(options["use_stream"]),
                    temperature=float(options["temperature"]),
                    top_p=float(options["top_p"]),
                    top_k=int(options["top_k"]),
                    repetition_penalty=float(options["repetition_penalty"]),
                    presence_penalty=float(options["presence_penalty"]),
                    frequency_penalty=float(options["frequency_penalty"]),
                    output_mask_payload=options.get("output_mask_payload"),
                    vllm_xargs=vllm_xargs,
                )
            else:
                reused_prefix_token_count = int(
                    self._hot_prefix_reuse_plan.get("reused_prefix_token_count") or 0
                )
                streaming_manager_rows = build_streaming_hot_prefix_manager_rows(
                    manifest,
                    include_sparse_groups=True,
                    reused_prefix_token_count=int(reused_prefix_token_count),
                )
                replay_prefix_tail_token_ids = [
                    int(x)
                    for x in list(
                        self._hot_prefix_reuse_plan.get("replay_prefix_tail_token_ids") or []
                    )
                ]
                prompt_token_ids, virtual_prompt_tokens = build_compact_hot_prefix_prompt_token_ids(
                    tokenizer=self._tokenizer,
                    reused_prefix_token_count=int(reused_prefix_token_count),
                    suffix_text=str(question),
                    prefix_messages=[
                        dict(msg)
                        for msg in list(self._loaded_session.get("prefix_messages") or [])
                    ]
                    if str(self._prefix_capture.get("continuation_mode") or "") == "same_message"
                    else None,
                    message_index=int(
                        self._prefix_capture.get("message_index")
                        or max(
                            0,
                            len(list(self._loaded_session.get("prefix_messages") or [])) - 1,
                        )
                    )
                    if str(self._prefix_capture.get("continuation_mode") or "") == "same_message"
                    else None,
                    continuation_tail_token_ids=continuation_tail_token_ids,
                    replay_prefix_tail_token_ids=replay_prefix_tail_token_ids,
                )
                vllm_xargs = {
                    "hot_prefix_token_count": int(reused_prefix_token_count)
                }
                if streaming_manager_rows:
                    vllm_xargs["hot_prefix_manager_rows"] = streaming_manager_rows
                metrics = run_completion_prompt_ids(
                    base_url=str(self.config.base_url),
                    model_id=str(request_model_id),
                    tokenizer=self._tokenizer,
                    prompt_token_ids=prompt_token_ids,
                    max_new_tokens=int(options["max_new_tokens"]),
                    min_new_tokens=int(options["min_new_tokens"]),
                    timeout_sec=int(self.config.timeout_sec),
                    ignore_eos=bool(options["ignore_eos"]),
                    use_stream=bool(options["use_stream"]),
                    temperature=float(options["temperature"]),
                    top_p=float(options["top_p"]),
                    top_k=int(options["top_k"]),
                    repetition_penalty=float(options["repetition_penalty"]),
                    presence_penalty=float(options["presence_penalty"]),
                    frequency_penalty=float(options["frequency_penalty"]),
                    output_mask_payload=options.get("output_mask_payload"),
                    vllm_xargs=vllm_xargs,
                    prompt_token_count_override=virtual_prompt_tokens,
                )
        elif request_strategy == "prefix_token_replay":
            prefix_messages = list(self._loaded_session.get("prefix_messages") or [])
            if prefix_messages:
                query_messages = build_same_message_query_messages(
                    prefix_messages=[dict(msg) for msg in prefix_messages],
                    suffix_text=str(question),
                    message_index=int(
                        self._prefix_capture.get("message_index") or len(prefix_messages) - 1
                    ),
                )
                prompt_token_ids = []
                virtual_prompt_tokens = None
                vllm_xargs = None
                metrics = run_chat_completion_messages(
                    base_url=str(self.config.base_url),
                    model_id=str(request_model_id),
                    tokenizer=self._tokenizer,
                    messages=query_messages,
                    max_new_tokens=int(options["max_new_tokens"]),
                    min_new_tokens=int(options["min_new_tokens"]),
                    timeout_sec=int(self.config.timeout_sec),
                    ignore_eos=bool(options["ignore_eos"]),
                    use_stream=bool(options["use_stream"]),
                    temperature=float(options["temperature"]),
                    top_p=float(options["top_p"]),
                    top_k=int(options["top_k"]),
                    repetition_penalty=float(options["repetition_penalty"]),
                    presence_penalty=float(options["presence_penalty"]),
                    frequency_penalty=float(options["frequency_penalty"]),
                    output_mask_payload=options.get("output_mask_payload"),
                    vllm_xargs=vllm_xargs,
                )
            else:
                prompt_token_ids, _suffix_token_ids = build_same_message_query_prompt_token_ids(
                    tokenizer=self._tokenizer,
                    prefix_token_ids=[
                        int(x) for x in list(self._loaded_session.get("prefix_token_ids") or [])
                    ],
                    suffix_text=str(question),
                    continuation_tail_token_ids=continuation_tail_token_ids,
                )
                virtual_prompt_tokens = None
                vllm_xargs = None
                metrics = run_completion_prompt_ids(
                    base_url=str(self.config.base_url),
                    model_id=str(request_model_id),
                    tokenizer=self._tokenizer,
                    prompt_token_ids=prompt_token_ids,
                    max_new_tokens=int(options["max_new_tokens"]),
                    min_new_tokens=int(options["min_new_tokens"]),
                    timeout_sec=int(self.config.timeout_sec),
                    ignore_eos=bool(options["ignore_eos"]),
                    use_stream=bool(options["use_stream"]),
                    temperature=float(options["temperature"]),
                    top_p=float(options["top_p"]),
                    top_k=int(options["top_k"]),
                    repetition_penalty=float(options["repetition_penalty"]),
                    presence_penalty=float(options["presence_penalty"]),
                    frequency_penalty=float(options["frequency_penalty"]),
                    output_mask_payload=options.get("output_mask_payload"),
                    vllm_xargs=vllm_xargs,
                    prompt_token_count_override=virtual_prompt_tokens,
                )
        elif request_strategy in {"prefix_cache_replay", "server_hot_prefix"}:
            prompt_token_ids = []
            virtual_prompt_tokens = None
            vllm_xargs = None
            metrics = run_hot_prefix_completion(
                base_url=str(self.config.base_url),
                model_id=str(request_model_id),
                tokenizer=self._tokenizer,
                prefix_session_dir=str(self._session_root),
                prefix_session_id=str(self.config.prefix_session_id),
                suffix_text=str(question),
                continuation_tail_token_ids=continuation_tail_token_ids,
                prefix_token_count=int(self._info.prefix_token_count),
                max_new_tokens=int(options["max_new_tokens"]),
                min_new_tokens=int(options["min_new_tokens"]),
                timeout_sec=int(self.config.timeout_sec),
                ignore_eos=bool(options["ignore_eos"]),
                use_stream=bool(options["use_stream"]),
                temperature=float(options["temperature"]),
                top_p=float(options["top_p"]),
                top_k=int(options["top_k"]),
                repetition_penalty=float(options["repetition_penalty"]),
                presence_penalty=float(options["presence_penalty"]),
                frequency_penalty=float(options["frequency_penalty"]),
                output_mask_payload=options.get("output_mask_payload"),
            )
        else:
            raise RuntimeError(f"Unhandled request strategy: {request_strategy}")
        return HotPrefixQueryResult(
            text=str(metrics.get("generated_text") or ""),
            session_id=str(self._info.session_id),
            prompt_tokens=int(
                metrics.get("prompt_tokens")
                or virtual_prompt_tokens
                or len(prompt_token_ids)
            ),
            completion_tokens=int(metrics.get("completion_tokens") or 0),
            ttft_ms=int(metrics.get("ttft_ms") or 0),
            decode_ms=int(metrics.get("decode_ms") or 0),
            total_ms=int(metrics.get("total_ms") or 0),
            request_id=str(metrics.get("request_id") or ""),
            finish_reason=str(metrics.get("finish_reason") or ""),
            stop_reason=str(metrics.get("stop_reason") or ""),
            prefix_state_load_count=int(self._prefix_state_load_count),
            prefix_restored_this_call=bool(prefix_restored_this_call),
            cached_prompt_tokens=int(metrics.get("usage_cached_prompt_tokens") or 0),
        )

    def ask(self, question: str, **generation_overrides: Any) -> HotPrefixQueryResult:
        """Ask one question against the saved prefix currently loaded in HBM."""

        return self._ask_one(
            question,
            force_restore_before_query=False,
            generation_overrides=generation_overrides,
        )

    def ask_many(
        self,
        questions: Sequence[str],
        reload_between_queries: bool = False,
    ) -> List[HotPrefixQueryResult]:
        """Ask multiple questions while optionally reloading prefix state between them."""

        out: List[HotPrefixQueryResult] = []
        for idx, question in enumerate(list(questions)):
            out.append(
                self._ask_one(
                    str(question),
                    force_restore_before_query=bool(reload_between_queries and idx > 0),
                    generation_overrides={},
                )
            )
        return out
