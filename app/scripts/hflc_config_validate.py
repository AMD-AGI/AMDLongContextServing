#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate effective HFLC make configuration exported as environment variables."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


KEYS = (
    "REPO_ROOT",
    "HFLC_PROFILE",
    "HFLC_FP8_TP",
    "HFLC_FP8_DP",
    "HFLC_FP8_PP",
    "HFLC_FP8_MAX_NUM_SEQS",
    "HFLC_FP8_TOKENIZER_ID",
    "HFLC_FP8_SWEEP",
    "HFLC_FP8_SWEEP_LIST",
    "HFLC_FP8_PROMPT_SHAPE",
    "HFLC_FP8_PROMPT_SHAPE_FILE",
    "HFLC_FP8_QUERY_API_MODE",
    "HFLC_FP8_ENGLISH_ONLY_MASK_MODE",
    "HFLC_FP8_ENGLISH_ONLY_MASK_BIAS",
    "HFLC_FP8_DUMP_REQUEST_PAYLOAD",
    "HFLC_FP8_DUMP_QUERY_MESSAGES",
    "HFLC_FP8_DUMP_PROMPT_TOKEN_IDS",
    "HFLC_FP8_REQUIRE_LAUNCH_CONTRACT",
    "HFLC_FP8_ENFORCE_EAGER",
    "HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB",
    "HFLC_FP8_VLLM_COMPILATION_CONFIG",
    "HFLC_FP8_REQUIRE_KV_CALIBRATION",
    "HFLC_FP8_KV_CALIBRATION_GATE",
    "HFLC_FP8_CAPTURE_MODE",
    "HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILE",
    "HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILES",
    "HFLC_FP8_CAPTURE_SUFFIX_QUERY_DIR",
)


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
    result: dict[str, object] = {}
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
            value: object = True
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


def _as_int(cfg: dict[str, str], errors: list[str], name: str) -> int:
    raw = cfg.get(name, "")
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{name} must be an integer, got {raw!r}")
        return 0
    if value <= 0:
        errors.append(f"{name} must be > 0, got {value}")
    return value


def _validate_token_list(cfg: dict[str, str], errors: list[str], name: str) -> None:
    text = cfg.get(name, "").strip()
    if not text:
        return
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts or any(not p.isdigit() for p in parts):
        errors.append(f"{name} must be comma-separated positive integers, got {cfg.get(name)!r}")


def _validate_nonnegative_int(
    cfg: dict[str, str], errors: list[str], name: str
) -> None:
    raw = cfg.get(name, "")
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{name} must be a non-negative integer, got {raw!r}")
        return
    if value < 0:
        errors.append(f"{name} must be >= 0, got {value}")


def main() -> int:
    cfg = {k: os.environ.get(k, "") for k in KEYS}
    errors: list[str] = []

    root = Path(cfg["REPO_ROOT"])
    if not root.is_dir():
        errors.append(f"REPO_ROOT does not exist: {root}")
    elif not (root / "Makefile").is_file():
        errors.append(f"REPO_ROOT is missing Makefile: {root}")

    if cfg["HFLC_PROFILE"] not in {"smoke", "long", "debug", "capture"}:
        errors.append(
            f"HFLC_PROFILE must be smoke|long|debug|capture, got {cfg['HFLC_PROFILE']!r}"
        )

    for key in ("HFLC_FP8_TP", "HFLC_FP8_DP", "HFLC_FP8_PP", "HFLC_FP8_MAX_NUM_SEQS"):
        _as_int(cfg, errors, key)

    if cfg["HFLC_FP8_QUERY_API_MODE"] not in {"auto", "messages", "prompt_ids"}:
        errors.append(
            "HFLC_FP8_QUERY_API_MODE must be auto|messages|prompt_ids, "
            f"got {cfg['HFLC_FP8_QUERY_API_MODE']!r}"
        )

    if cfg["HFLC_FP8_ENGLISH_ONLY_MASK_MODE"] not in {"off", "bias", "hard"}:
        errors.append(
            "HFLC_FP8_ENGLISH_ONLY_MASK_MODE must be off|bias|hard, "
            f"got {cfg['HFLC_FP8_ENGLISH_ONLY_MASK_MODE']!r}"
        )

    try:
        float(cfg["HFLC_FP8_ENGLISH_ONLY_MASK_BIAS"] or "0")
    except ValueError:
        errors.append(
            "HFLC_FP8_ENGLISH_ONLY_MASK_BIAS must be a float, "
            f"got {cfg['HFLC_FP8_ENGLISH_ONLY_MASK_BIAS']!r}"
        )

    for key in (
        "HFLC_FP8_DUMP_REQUEST_PAYLOAD",
        "HFLC_FP8_DUMP_QUERY_MESSAGES",
        "HFLC_FP8_DUMP_PROMPT_TOKEN_IDS",
    ):
        raw = cfg.get(key, "")
        if raw not in {"0", "1"}:
            errors.append(f"{key} must be 0 or 1, got {raw!r}")

    for key in (
        "HFLC_FP8_CUSTOM_ALL_REDUCE_MAX_SIZE_MB",
    ):
        _validate_nonnegative_int(cfg, errors, key)

    _validate_token_list(cfg, errors, "HFLC_FP8_SWEEP")
    _validate_token_list(cfg, errors, "HFLC_FP8_SWEEP_LIST")

    shape = cfg["HFLC_FP8_PROMPT_SHAPE"].strip()
    shape_file = cfg["HFLC_FP8_PROMPT_SHAPE_FILE"].strip()
    if shape == "custom_file":
        if not shape_file:
            errors.append("HFLC_FP8_PROMPT_SHAPE=custom_file requires HFLC_FP8_PROMPT_SHAPE_FILE")
        elif not Path(shape_file).is_file():
            errors.append(f"HFLC_FP8_PROMPT_SHAPE_FILE does not exist: {shape_file}")
    elif shape_file and not Path(shape_file).is_file():
        errors.append(f"HFLC_FP8_PROMPT_SHAPE_FILE set but file missing: {shape_file}")

    comp = cfg["HFLC_FP8_VLLM_COMPILATION_CONFIG"].strip()
    if comp:
        try:
            parsed_comp = _parse_jsonish_compilation_config(comp)
            if not isinstance(parsed_comp, dict):
                raise json.JSONDecodeError(
                    "compilation config must be a JSON object", comp, 0
                )
        except json.JSONDecodeError as exc:
            errors.append(f"HFLC_FP8_VLLM_COMPILATION_CONFIG is not valid JSON: {exc}")

    if (
        cfg["HFLC_FP8_REQUIRE_KV_CALIBRATION"] == "1"
        and cfg["HFLC_FP8_KV_CALIBRATION_GATE"] != "1"
    ):
        errors.append(
            "HFLC_FP8_REQUIRE_KV_CALIBRATION=1 requires HFLC_FP8_KV_CALIBRATION_GATE=1"
        )

    if (
        cfg["HFLC_FP8_REQUIRE_LAUNCH_CONTRACT"] == "1"
        and cfg["HFLC_FP8_ENFORCE_EAGER"] != "0"
    ):
        errors.append(
            "HFLC_FP8_REQUIRE_LAUNCH_CONTRACT=1 requires HFLC_FP8_ENFORCE_EAGER=0"
        )

    if cfg["HFLC_FP8_CAPTURE_MODE"] in {"query", "build_and_query"}:
        has_query_source = any(
            cfg[k].strip()
            for k in (
                "HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILE",
                "HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILES",
                "HFLC_FP8_CAPTURE_SUFFIX_QUERY_DIR",
            )
        )
        if not has_query_source:
            errors.append(
                "HFLC_FP8_CAPTURE_MODE query/build_and_query requires one of "
                "HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILE, HFLC_FP8_CAPTURE_SUFFIX_QUERY_FILES, "
                "HFLC_FP8_CAPTURE_SUFFIX_QUERY_DIR"
            )

    if errors:
        print("HFLC config validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 2

    print("HFLC config validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
