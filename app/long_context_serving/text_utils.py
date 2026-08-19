# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared text helpers for response handling."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping, Optional
from urllib import request


def clean_response_text(text: Optional[str]) -> Optional[str]:
    """Clean model meta-reasoning phrases from output text.

    Args:
        text: Raw model output text.

    Returns:
        Cleaned response text, or original text when no safe cleanup is found.
    """
    if not text:
        return text
    phrases = (
        "we need to",
        "the user asks",
        "so we need",
        "we should",
        "the context",
        "likely referring",
        "we have",
    )
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    filtered = [s for s in sentences if not any(p in s.lower() for p in phrases)]
    if not filtered:
        return text.strip()
    cleaned = ". ".join(filtered).strip()
    if not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def extract_message_text(message: Mapping[str, Any] | None) -> Optional[str]:
    """Extract assistant text from one chat message dictionary.

    Args:
        message: OpenAI-style assistant message.

    Returns:
        Preferred assistant text, or ``None`` when unavailable.
    """
    if not isinstance(message, Mapping):
        return None
    value = message.get("content") or message.get("reasoning_content") or message.get("reasoning")
    if value is None:
        return None
    return str(value)


def extract_response_text(payload: Mapping[str, Any] | None) -> Optional[str]:
    """Extract assistant text from a chat completion payload.

    Args:
        payload: OpenAI chat-completions response payload.

    Returns:
        Preferred assistant text, or ``None``.
    """
    if not isinstance(payload, Mapping):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0] if isinstance(choices[0], Mapping) else {}
    message = first.get("message") if isinstance(first, Mapping) else None
    return extract_message_text(message if isinstance(message, Mapping) else None)


def stream_chat_completions(
    *,
    url: str,
    payload: Mapping[str, Any],
    timeout_sec: int = 120,
    on_progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Mapping[str, Any]:
    """Stream a chat completion and return timing + assembled text.

    Args:
        url: Full chat-completions endpoint URL.
        payload: Request payload (will be copied and sent with ``stream=True``).
        timeout_sec: Request timeout in seconds.
        on_progress: Optional callback invoked with streaming progress events.

    Returns:
        Dict with separate ``answer_text`` and ``thinking_text`` channels, plus
        ``text`` (alias of ``answer_text``), ``usage``, ``ttft_ms``,
        ``total_ms``, ``finish_reason``, and ``stop_reason``.
    """
    req_payload = dict(payload)
    req_payload["stream"] = True
    req_payload["stream_options"] = {"include_usage": True}
    req = request.Request(
        url,
        method="POST",
        data=json.dumps(req_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft_ms: Optional[int] = None
    answer_text_parts: list[str] = []
    thinking_text_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = ""
    stop_reason = ""
    request_id = ""

    saw_terminal_chunk = False
    saw_usage = False
    with request.urlopen(req, timeout=timeout_sec) as resp:
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            payload_bytes = line[5:].strip()
            if payload_bytes == b"[DONE]":
                break
            try:
                chunk = json.loads(payload_bytes.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict) and chunk.get("id"):
                request_id = str(chunk.get("id") or request_id)
            if isinstance(chunk, dict) and "usage" in chunk and chunk["usage"]:
                usage = dict(chunk["usage"])
                saw_usage = True
            choice = (chunk.get("choices") or [{}])[0] if isinstance(chunk, dict) else {}
            delta = choice.get("delta") or {}
            answer_content = delta.get("content") or ""
            thinking_content = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if answer_content or thinking_content:
                if ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - start) * 1000)
                    if on_progress is not None:
                        on_progress(
                            {
                                "type": "ttft",
                                "ttft_ms": int(ttft_ms),
                                "request_id": str(request_id),
                            }
                        )
            if answer_content:
                answer_text_parts.append(str(answer_content))
            if thinking_content:
                thinking_text_parts.append(str(thinking_content))
            if on_progress is not None and (answer_content or thinking_content):
                on_progress(
                    {
                        "type": "delta",
                        "answer_delta": str(answer_content or ""),
                        "thinking_delta": str(thinking_content or ""),
                        "request_id": str(request_id),
                    }
                )
            # OpenAI-style finish and stop reasons, e.g. "stop", "length",
            # "tool_calls", "content_filter", etc.
            finish_reason = str(choice.get("finish_reason") or finish_reason)
            # Additional stop reason field.
            # This is non-standard but can be used by custom backends to provide
            # more info about why the stream stopped. vLLM may include this when
            # the stream is cut short due to max_tokens or other limits.
            stop_reason = str(choice.get("stop_reason") or stop_reason)
            if str(choice.get("finish_reason") or ""):
                saw_terminal_chunk = True
            if on_progress is not None and usage:
                on_progress(
                    {
                        "type": "usage",
                        "usage": usage,
                        "request_id": str(request_id),
                    }
                )
            # Some local vLLM builds do not send a final ``data: [DONE]`` line
            # and may also omit the optional usage chunk entirely. The terminal
            # chunk is the authoritative end-of-generation signal, so return as
            # soon as it is observed instead of waiting for the socket to close.
            if saw_terminal_chunk:
                break

    total_ms = int((time.perf_counter() - start) * 1000)
    answer_text = "".join(answer_text_parts)
    thinking_text = "".join(thinking_text_parts)
    return {
        "text": answer_text,
        "answer_text": answer_text,
        "thinking_text": thinking_text,
        "usage": usage,
        "ttft_ms": int(ttft_ms) if ttft_ms is not None else None,
        "total_ms": total_ms,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "request_id": str(request_id),
    }


def stream_text_completions(
    *,
    url: str,
    payload: Mapping[str, Any],
    timeout_sec: int = 120,
    on_progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Mapping[str, Any]:
    """Stream a text completion and return timing + assembled text."""
    req_payload = dict(payload)
    req_payload["stream"] = True
    req_payload["stream_options"] = {"include_usage": True}
    req = request.Request(
        url,
        method="POST",
        data=json.dumps(req_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft_ms: Optional[int] = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = ""
    stop_reason = ""
    request_id = ""

    saw_terminal_chunk = False
    saw_usage = False
    with request.urlopen(req, timeout=timeout_sec) as resp:
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload_bytes = line[5:].strip()
            if payload_bytes == b"[DONE]":
                break
            try:
                chunk = json.loads(payload_bytes.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict) and chunk.get("id"):
                request_id = str(chunk.get("id") or request_id)
            if isinstance(chunk, dict) and "usage" in chunk and chunk["usage"]:
                usage = dict(chunk["usage"])
                saw_usage = True
            choice = (chunk.get("choices") or [{}])[0] if isinstance(chunk, dict) else {}
            delta_text = ""
            if isinstance(choice, Mapping):
                delta_text = str(choice.get("text") or "")
                finish_reason = str(choice.get("finish_reason") or finish_reason)
                stop_reason = str(choice.get("stop_reason") or stop_reason)
                if str(choice.get("finish_reason") or ""):
                    saw_terminal_chunk = True
            if delta_text:
                if ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - start) * 1000)
                    if on_progress is not None:
                        on_progress(
                            {
                                "type": "ttft",
                                "ttft_ms": int(ttft_ms),
                                "request_id": str(request_id),
                            }
                        )
                text_parts.append(delta_text)
                if on_progress is not None:
                    on_progress(
                        {
                            "type": "delta",
                            "answer_delta": str(delta_text),
                            "thinking_delta": "",
                            "request_id": str(request_id),
                        }
                    )
            if on_progress is not None and usage:
                on_progress(
                    {
                        "type": "usage",
                        "usage": usage,
                        "request_id": str(request_id),
                    }
                )
            # See the chat-completions variant above: usage is best-effort
            # metadata, not a required part of the termination contract.
            if saw_terminal_chunk:
                break

    total_ms = int((time.perf_counter() - start) * 1000)
    text = "".join(text_parts)
    return {
        "text": text,
        "answer_text": text,
        "thinking_text": "",
        "usage": usage,
        "ttft_ms": int(ttft_ms) if ttft_ms is not None else None,
        "total_ms": total_ms,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "request_id": str(request_id),
    }
