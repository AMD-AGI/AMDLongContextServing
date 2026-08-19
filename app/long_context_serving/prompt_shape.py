# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prompt-shape loading, validation, and rendering for long-context benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_PLACEHOLDERS = {
    "prompt_text",
    "prompt_context_summary",
    "system_prompt",
    "max_new_tokens",
    "prompt_tokens",
    "model_id",
}
ALLOWED_PREFIX_CAPTURE_CONTINUATION_MODES = {"same_message"}
ALLOWED_CONTEXT_MARKUP = {"file_only_v1", "repo_file_markers_v1"}

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class PromptShapeError(ValueError):
    """Raised when prompt-shape loading/validation/rendering fails."""


def _builtin_benchmark_v1() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "benchmark_v1",
        "messages": [
            {"role": "system", "content": "{{system_prompt}}"},
            {
                "role": "user",
                "content": (
                    "{{prompt_text}}\n\n"
                    "[BENCHMARK TASK]\n"
                    "Write a continuous technical response with at least {{max_new_tokens}} tokens. "
                    "Do not end early.\n"
                ),
            },
        ],
    }


def _builtin_repo_grounded_en_v1() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "repo_grounded_en_v1",
        "context_markup": "file_only_v1",
        "prefix_capture": {
            "message_index": 1,
            "split_after": "Task:\n",
            "continuation_mode": "same_message",
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a repository analysis assistant. Respond in English only. "
                    "Use only evidence from CONTEXT. Do not mention benchmarking, prompts, "
                    "or \"the user asks\". Do not produce coding-challenge/HumanEval-style "
                    "answers. If evidence is insufficient, output INSUFFICIENT_EVIDENCE."
                ),
            },
            {
                "role": "user",
                "content": (
                    "<CONTEXT>\n"
                    "{{prompt_text}}\n"
                    "</CONTEXT>\n\n"
                    "Task:\n"
                    "Write a technical report grounded in CONTEXT.\n\n"
                    "Requirements:\n"
                    "1. Minimum {{max_new_tokens}} tokens.\n"
                    "2. Cite at least 8 concrete file paths that appear in CONTEXT.\n"
                    "3. Cover architecture, build/test workflow, dependencies, "
                    "bottlenecks, and actionable next steps.\n"
                    "4. No exam/problem-solution format.\n"
                    "5. No meta commentary about instructions."
                ),
            },
        ],
    }


def _builtin_repo_grounded_en_v2() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "repo_grounded_en_v2",
        "context_markup": "repo_file_markers_v1",
        "prefix_capture": {
            "message_index": 1,
            "split_after": "Task:\n",
            "continuation_mode": "same_message",
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a repository analysis assistant. Respond in English only. "
                    "CONTEXT uses [REPO name] to mark repository boundaries and "
                    "[FILE repo/path] to mark file contents. Treat the nearest "
                    "[REPO name] tag as the repo for the following files. "
                    "Use only evidence from CONTEXT. Do not mention benchmarking, prompts, "
                    "or \"the user asks\". Do not produce coding-challenge/HumanEval-style "
                    "answers. If evidence is insufficient, output INSUFFICIENT_EVIDENCE."
                ),
            },
            {
                "role": "user",
                "content": (
                    "<CONTEXT>\n"
                    "{{prompt_text}}\n"
                    "</CONTEXT>\n\n"
                    "Task:\n"
                    "Write a technical report grounded in CONTEXT.\n\n"
                    "Requirements:\n"
                    "1. Minimum {{max_new_tokens}} tokens.\n"
                    "2. Cite at least 8 concrete file paths that appear in CONTEXT.\n"
                    "3. Cover architecture, build/test workflow, dependencies, "
                    "bottlenecks, and actionable next steps.\n"
                    "4. No exam/problem-solution format.\n"
                    "5. No meta commentary about instructions."
                ),
            },
        ],
    }


def _builtin_repo_grounded_universal_v1() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "repo_grounded_universal_v1",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a repository analysis assistant. Respond in English only. "
                    "Use only evidence from CONTEXT. If evidence is insufficient, "
                    "output INSUFFICIENT_EVIDENCE."
                ),
            },
            {
                "role": "user",
                "content": (
                    "<CONTEXT>\n"
                    "{{prompt_text}}\n"
                    "</CONTEXT>\n\n"
                    "Task:\n"
                    "Analyze the repository context.\n\n"
                    "Rules:\n"
                    "1. Detect whether CONTEXT contains one repo or multiple "
                    "repos from file paths/content boundaries.\n"
                    "2. If single repo: produce one repo report.\n"
                    "3. If multi repo: produce per-repo sections plus a "
                    "cross-repo comparison section.\n"
                    "4. Cite concrete file paths from CONTEXT for each claim.\n"
                    "5. Minimum {{max_new_tokens}} tokens.\n"
                    "6. No benchmark/prompt meta commentary."
                ),
            },
        ],
    }


def _builtin_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "benchmark_v1": _builtin_benchmark_v1(),
        "repo_grounded_en_v1": _builtin_repo_grounded_en_v1(),
        "repo_grounded_en_v2": _builtin_repo_grounded_en_v2(),
        "repo_grounded_universal_v1": _builtin_repo_grounded_universal_v1(),
    }


def _canonical_sha256(spec: Mapping[str, Any]) -> str:
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_placeholders(text: str) -> Iterable[str]:
    for match in _PLACEHOLDER_RE.finditer(text or ""):
        yield str(match.group(1))


def validate_prompt_shape_spec(spec: Mapping[str, Any]) -> Tuple[str, str]:
    """Validate prompt-shape schema and placeholder usage.

    Returns:
        Tuple of ("ok", "message") when valid.

    Raises:
        PromptShapeError: On any schema/content validation issue.
    """
    errors: List[str] = []

    schema_version = spec.get("schema_version")
    if schema_version != 1:
        errors.append(f"schema_version must be 1 (got {schema_version!r})")

    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")

    context_markup = spec.get("context_markup")
    if context_markup is not None and context_markup not in ALLOWED_CONTEXT_MARKUP:
        errors.append(
            "context_markup must be one of "
            f"{sorted(ALLOWED_CONTEXT_MARKUP)} when present"
        )

    messages = spec.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty array")
    else:
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Mapping):
                errors.append(f"messages[{idx}] must be an object")
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role not in ALLOWED_ROLES:
                errors.append(
                    f"messages[{idx}].role must be one of {sorted(ALLOWED_ROLES)} (got {role!r})"
                )
            if not isinstance(content, str):
                errors.append(f"messages[{idx}].content must be a string")
                continue
            used = set(_extract_placeholders(content))
            invalid = sorted(p for p in used if p not in ALLOWED_PLACEHOLDERS)
            if invalid:
                errors.append(
                    f"messages[{idx}].content uses unsupported placeholders: {', '.join(invalid)}"
                )

    prefix_capture = spec.get("prefix_capture")
    if prefix_capture is not None:
        if not isinstance(prefix_capture, Mapping):
            errors.append("prefix_capture must be an object when present")
        else:
            message_index = prefix_capture.get("message_index")
            split_after = prefix_capture.get("split_after")
            continuation_mode = prefix_capture.get("continuation_mode")
            if not isinstance(message_index, int) or isinstance(message_index, bool) or message_index < 0:
                errors.append("prefix_capture.message_index must be a non-negative integer")
            if not isinstance(split_after, str) or not split_after:
                errors.append("prefix_capture.split_after must be a non-empty string")
            if continuation_mode not in ALLOWED_PREFIX_CAPTURE_CONTINUATION_MODES:
                errors.append(
                    "prefix_capture.continuation_mode must be 'same_message'"
                )
            if (
                isinstance(messages, list)
                and messages
                and isinstance(message_index, int)
                and not isinstance(message_index, bool)
            ):
                if message_index >= len(messages):
                    errors.append(
                        "prefix_capture.message_index must reference an "
                        "existing message "
                        f"(got {message_index}, len={len(messages)})"
                    )
                elif isinstance(split_after, str) and split_after:
                    target = messages[message_index]
                    if isinstance(target, Mapping):
                        content = target.get("content")
                        if isinstance(content, str) and split_after not in content:
                            errors.append(
                                f"prefix_capture.split_after must appear in messages[{message_index}].content"
                            )

    if errors:
        raise PromptShapeError("; ".join(errors))
    return "ok", "validated"


def load_prompt_shape_spec(
    *,
    prompt_shape: str,
    prompt_shape_file: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and validate prompt-shape spec from preset or file.

    Returns:
        (spec, metadata)
        metadata keys: id, source, file, template_sha256, validation
    """
    shape_id = str(prompt_shape or "benchmark_v1").strip() or "benchmark_v1"
    shape_file = str(prompt_shape_file or "").strip()

    spec: Dict[str, Any]
    source: str
    file_path: str = ""

    if shape_id == "custom_file":
        if not shape_file:
            raise PromptShapeError(
                "--prompt-shape-file is required when --prompt-shape custom_file"
            )
        path = Path(shape_file).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise PromptShapeError(f"prompt-shape file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise PromptShapeError(f"failed to parse prompt-shape JSON: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PromptShapeError("custom prompt-shape file must contain a JSON object")
        spec = dict(payload)
        source = "file"
        file_path = str(path)
    else:
        builtins = _builtin_specs()
        if shape_id not in builtins:
            raise PromptShapeError(
                "unknown --prompt-shape "
                f"{shape_id!r}; expected one of benchmark_v1, "
                "repo_grounded_en_v1, repo_grounded_en_v2, "
                "repo_grounded_universal_v1, custom_file"
            )
        spec = dict(builtins[shape_id])
        source = "preset"

    status, message = validate_prompt_shape_spec(spec)
    meta = {
        "id": str(spec.get("name") or shape_id),
        "source": source,
        "file": file_path,
        "template_sha256": _canonical_sha256(spec),
        "context_markup": str(spec.get("context_markup") or ""),
        "prefix_capture": dict(spec.get("prefix_capture") or {}),
        "validation": {"status": status, "message": message},
    }
    return spec, meta


def render_messages(spec: Mapping[str, Any], context: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Renders chat messages from a validated prompt-shape spec.

    Args:
        spec: Validated prompt-shape specification.
        context: Placeholder values used to render message content.

    Returns:
        List[Dict[str, str]]: Rendered chat messages in prompt order.

    Raises:
        PromptShapeError: If the spec is malformed or a placeholder value is
            missing from ``context``.
    """

    def _replace(content: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            key = str(match.group(1))
            if key not in context:
                raise PromptShapeError(f"missing placeholder value: {key}")
            return str(context[key])

        return _PLACEHOLDER_RE.sub(_sub, str(content))

    messages = spec.get("messages")
    if not isinstance(messages, list):
        raise PromptShapeError("invalid prompt-shape spec: messages must be an array")

    rendered: List[Dict[str, str]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise PromptShapeError(f"invalid prompt-shape spec: messages[{idx}] must be an object")
        role = str(msg.get("role") or "").strip()
        if role not in ALLOWED_ROLES:
            raise PromptShapeError(
                f"invalid prompt-shape spec: messages[{idx}].role must be one of {sorted(ALLOWED_ROLES)}"
            )
        content = msg.get("content")
        if not isinstance(content, str):
            raise PromptShapeError(f"invalid prompt-shape spec: messages[{idx}].content must be a string")
        rendered.append({"role": role, "content": _replace(content)})
    return rendered
