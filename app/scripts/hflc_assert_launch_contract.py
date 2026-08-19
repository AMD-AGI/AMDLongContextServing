#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate HF long-context launch metadata for the HIP FP8 flow."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    """Check launch/run metadata consistency for the coherent HIP path."""
    if len(sys.argv) != 3:
        print(
            "usage: hflc_assert_launch_contract.py <launch_env.json> <run_meta.json>",
            file=sys.stderr,
        )
        return 2
    env = _load_json(sys.argv[1])
    meta = _load_json(sys.argv[2])
    prompt_shape = meta.get("prompt_shape") or {}
    bad: list[tuple[str, str, str]] = []

    eager = str(env.get("VLLM_ENFORCE_EAGER", "0"))
    if eager != "0":
        bad.append(("VLLM_ENFORCE_EAGER", eager, "0"))

    expected_shape = str(env.get("PROMPT_SHAPE", "")).strip()
    expected_file = str(env.get("PROMPT_SHAPE_FILE", "")).strip()

    prompt_id = str(prompt_shape.get("id", ""))
    prompt_source = str(prompt_shape.get("source", ""))
    prompt_file = str(prompt_shape.get("file", ""))

    if expected_file and prompt_source != "file":
        bad.append(("prompt_shape.source", prompt_source, "file"))
    if expected_file and not prompt_file.endswith(expected_file):
        bad.append(("prompt_shape.file", prompt_file, expected_file))
    if (
        not expected_file
        and expected_shape
        and expected_shape != "custom_file"
        and prompt_id != expected_shape
    ):
        bad.append(("prompt_shape.id", prompt_id, expected_shape))
    if (
        not expected_file
        and expected_shape
        and expected_shape != "custom_file"
        and prompt_source != "preset"
    ):
        bad.append(("prompt_shape.source", prompt_source, "preset"))
    for key, got, expected in bad:
        print(f"ASSERT FAIL {key}: got={got!r} expected={expected!r}")
    return 6 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
