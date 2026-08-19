#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Build consolidated report across multiple HF long-context run phases."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from statistics import median
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from long_context_serving.benchmark_common import load_jsonl_dicts
from long_context_serving.host_env import render_environment_section as _render_environment_section


TTFT_COMPARE_PNG = "ttft_comparison.png"
DECODE_COMPARE_PNG = "decode_comparison.png"
DECODE_COMPARE_CURRENT_LABEL_PROMPTS = {
    1024,
    32768,
    131072,
    524288,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    58720256,   # 56Mi (off the powers-of-2 grid; added for the post-hoc 56Mi point)
    67108864,
    134217728,
    268435456,
}
TTFT_COMPARE_CURRENT_LABEL_PROMPTS = set(DECODE_COMPARE_CURRENT_LABEL_PROMPTS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fmt_ttft_human(ttft_s: float) -> str:
    if ttft_s < 60.0:
        if ttft_s < 10.0:
            return f"{ttft_s:.2f}s"
        return f"{ttft_s:.1f}s"
    if ttft_s < 3600.0:
        return f"{ttft_s / 60.0:.1f}m"
    if ttft_s < 86400.0:
        return f"{ttft_s / 3600.0:.2f}h"
    return f"{ttft_s / 86400.0:.2f}d"


def _fmt_ttft_plot_label(ttft_s: float) -> str:
    return _fmt_ttft_human(ttft_s)


def _parse_duration_seconds(text: str) -> float:
    """Parse one TTFT cell back into seconds.

    Args:
        text: Markdown-cell text such as ``0.10s`` or ``1.3m``.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the value cannot be parsed.
    """
    cleaned = _strip_md_code(text)
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhd]?)", cleaned)
    if not match:
        raise ValueError(f"Unsupported duration cell: {text!r}")
    value = float(match.group(1))
    suffix = match.group(2)
    if suffix == "m":
        return value * 60.0
    if suffix == "h":
        return value * 3600.0
    if suffix == "d":
        return value * 86400.0
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_driver_kv_pairs(line: str) -> Dict[str, str]:
    return {match.group(1): match.group(2) for match in re.finditer(r"([A-Za-z0-9_]+)=([^ ]+)", line)}


def _load_run_meta_with_driver_fallback(phase_dir: Path) -> Dict[str, Any]:
    run_meta_path = phase_dir / "run_meta.json"
    if run_meta_path.exists():
        return _load_json(run_meta_path)

    driver_log = phase_dir.parent / "driver.log"
    if not driver_log.exists():
        raise FileNotFoundError(f"Missing run metadata for phase: {phase_dir}")

    config_line = ""
    phase_line = ""
    driver_start_line = ""
    for raw in driver_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if " driver_start " in raw:
            driver_start_line = raw
        elif " config " in raw:
            config_line = raw
        elif " phase=phase1 start " in raw:
            phase_line = raw

    config_pairs = _parse_driver_kv_pairs(config_line)
    phase_pairs = _parse_driver_kv_pairs(phase_line)
    start_pairs = _parse_driver_kv_pairs(driver_start_line)
    prompt_tokens = int(config_pairs.get("target_prompt_tokens", phase_pairs.get("sweep", "0")) or 0)
    max_model_len = int(phase_pairs.get("max_model_len", "0") or 0)
    tp = int(config_pairs.get("tp", "0") or 0)
    dp = int(config_pairs.get("dp", "0") or 0)
    pp = int(config_pairs.get("pp", "0") or 0)
    return {
        "run_id": str(start_pairs.get("run_id") or phase_dir.parent.name),
        "generated_at_utc": "",
        "sweep_prompt_tokens": [prompt_tokens] if prompt_tokens > 0 else [],
        "vllm": {
            "attention_backend": str(config_pairs.get("vllm_attention_backend") or ""),
            "kv_cache_dtype": str(config_pairs.get("kv_cache_dtype") or ""),
            "enforce_eager": False,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "pipeline_parallel_size": pp,
        },
        "launch_env": {
            "VLLM_AITER_HEAD4_DECODE_MODE": str(config_pairs.get("vllm_aiter_head4_decode_mode") or ""),
            "VLLM_AITER_MLA_STAGE1_MODE": str(config_pairs.get("vllm_aiter_mla_stage1_mode") or ""),
            "VLLM_AITER_MLA_STAGE2_MODE": str(config_pairs.get("vllm_aiter_mla_stage2_mode") or ""),
        },
    }


def _read_last_jsonl_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last = stripped
    if not last:
        return {}
    try:
        payload = json.loads(last)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_markdown_table(lines: List[str], start_idx: int) -> Tuple[List[str], List[List[str]]]:
    header = [cell.strip() for cell in lines[start_idx].strip().strip("|").split("|")]
    rows: List[List[str]] = []
    idx = start_idx + 2
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
        idx += 1
    return header, rows


def _strip_md_code(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("`") and value.endswith("`") and len(value) >= 2:
        value = value[1:-1]
    return value.strip()


def _load_baseline_metric_rows(report_md: Path) -> List[Dict[str, Any]]:
    text = report_md.read_text(encoding="utf-8")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        normalized = line.strip()
        if not (
            normalized.startswith("| Prompt(req) | Prompt(eff) | Outcome | TTFT (s) | Decode tok/s | Completion |")
            or normalized.startswith("| Prompt(req) | Prompt(eff) | Outcome | TTFT | Decode tok/s | Completion |")
        ):
            continue
        header, rows = _parse_markdown_table(lines, idx)
        if not header or not rows:
            break
        normalized_header = [_strip_md_code(cell) for cell in header]
        try:
            prompt_eff_idx = normalized_header.index("Prompt(eff)")
            outcome_idx = normalized_header.index("Outcome")
            if "TTFT (s)" in normalized_header:
                ttft_idx = normalized_header.index("TTFT (s)")
            else:
                ttft_idx = normalized_header.index("TTFT")
            tok_idx = normalized_header.index("Decode tok/s")
        except ValueError:
            continue
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                prompt_eff = int(_strip_md_code(row[prompt_eff_idx]))
                ttft_s = _parse_duration_seconds(row[ttft_idx])
                tok_s = float(_strip_md_code(row[tok_idx]))
            except Exception:
                continue
            outcome = _strip_md_code(row[outcome_idx])
            if outcome != "completed":
                continue
            out.append(
                {
                    "prompt_tokens": prompt_eff,
                    "tok_s": tok_s,
                    "ttft_s": ttft_s,
                    "prompt_human": _fmt_prompt(prompt_eff),
                }
            )
        return sorted(out, key=lambda item: int(item["prompt_tokens"]))
    return []


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _md_link(
    path: Path,
    base_dir: Path,
    label: Optional[str] = None,
    fragment: Optional[str] = None,
) -> str:
    rel = Path(os.path.relpath(str(path.resolve()), str(base_dir.resolve()))).as_posix()
    if fragment:
        rel = f"{rel}{fragment}"
    txt = label or rel
    return f"[`{txt}`]({rel})"


def _md_link_if_exists(path: Path, base_dir: Path, label: Optional[str] = None) -> str:
    return _md_link(path, base_dir, label=label) if path.exists() else "`-`"


def _rel_display(path: Path, base_dir: Path) -> str:
    """Format one path relative to a display base directory.

    Args:
        path: Path to render.
        base_dir: Base directory used for the relative display form.

    Returns:
        str: Relative POSIX-style path when possible, else the raw string form.
    """
    try:
        return Path(
            os.path.relpath(str(path.resolve()), str(base_dir.resolve()))
        ).as_posix()
    except Exception:
        return str(path)


def _fmt_prompt_list(prompts: Iterable[int]) -> str:
    vals = [int(p) for p in prompts if int(p) > 0]
    return ", ".join(_fmt_prompt(p) for p in sorted(set(vals)))


def _select_host_env(phase_run_metas: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the first non-empty host_env block from the campaign's runs."""
    for rm in phase_run_metas:
        env = rm.get("host_env") if isinstance(rm, dict) else None
        if isinstance(env, dict) and env:
            return env
    return {}


def _read_server_cmd_text(phase_dir: Path) -> str:
    path = phase_dir / "server_cmd.txt"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _read_vllm_log_text(phase_dir: Path) -> str:
    path = phase_dir / "vllm_server.log"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _infer_backend_from_log(log_text: str) -> str:
    patterns = (
        r"Using AttentionBackendEnum\.([A-Z0-9_]+) backend\.",
        r"Using ([A-Z0-9_]+) backend\.",
    )
    for pattern in patterns:
        match = re.search(pattern, log_text)
        if match:
            return str(match.group(1) or "").strip()
    return ""


def _infer_mode_from_log(log_text: str) -> str:
    line_patterns = (
        (r"\[vllm\._aiter_ops\] head4 decode mode=([A-Za-z0-9_]+)", "decode_mode={value}"),
        (r"\[vllm\._aiter_ops\] sub16 decode mode=([A-Za-z0-9_]+)", "sub16_mode={value}"),
        (r"Using TRT-LLM ragged DeepSeek prefill for MLA", "TRT-LLM ragged DeepSeek prefill for MLA"),
    )
    for pattern, template in line_patterns:
        match = re.search(pattern, log_text)
        if not match:
            continue
        if "{value}" in template:
            return template.format(value=str(match.group(1) or "").strip())
        return template
    return ""


def _extract_graph_log_details(phase_dir: Path) -> Dict[str, Any]:
    log_path = phase_dir / "vllm_server.log"
    if not log_path.exists():
        return {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}

    details: Dict[str, Any] = {}
    patterns = (
        ("configured_graph_mode", r"cudagraph_mode': <CUDAGraphMode\.([A-Z0-9_]+):"),
        ("prefill_graph_mode", r"Capturing CUDA graphs \(mixed prefill-decode, ([A-Z0-9_]+)\)"),
        ("decode_graph_mode", r"Capturing CUDA graphs \(decode, ([A-Z0-9_]+)\)"),
    )
    for lineno, line in enumerate(lines, start=1):
        for key, pattern in patterns:
            if key in details:
                continue
            match = re.search(pattern, line)
            if match:
                details[key] = str(match.group(1) or "").strip()
                details[f"{key}_line"] = lineno
    return details


def _extract_allreduce_log_details(phase_dir: Path) -> Dict[str, Any]:
    """Extract all-reduce evidence from one vLLM server log.

    Args:
        phase_dir: Phase directory containing ``vllm_server.log``.

    Returns:
        Mapping with any discovered custom-all-reduce mode, graph registration,
        and decode-phase implementation details plus their line numbers.
    """
    log_path = phase_dir / "vllm_server.log"
    if not log_path.exists():
        return {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}

    details: Dict[str, Any] = {}
    for lineno, line in enumerate(lines, start=1):
        if "custom_all_reduce_mode" not in details:
            mode_match = re.search(r"disable_custom_all_reduce=(True|False)", line)
            if mode_match:
                details["custom_all_reduce_mode"] = (
                    "disabled" if mode_match.group(1) == "True" else "enabled"
                )
                details["custom_all_reduce_mode_line"] = lineno
        if "decode_allreduce_impl" not in details:
            decode_match = re.search(
                r"\[long-context-serving\]\[allreduce-trace\] impl=([A-Za-z0-9_]+)\s+phase_hint=decode\b",
                line,
            )
            if decode_match:
                details["decode_allreduce_impl"] = str(
                    decode_match.group(1) or ""
                ).strip()
                details["decode_allreduce_impl_line"] = lineno
        if "custom_all_reduce_registration" not in details:
            registration_match = re.search(
                r"\[custom_all_reduce\.py:\d+\] Registering (\d+) cuda graph addresses",
                line,
            )
            if registration_match:
                details["custom_all_reduce_registration"] = (
                    f"{int(registration_match.group(1))} addresses"
                )
                details["custom_all_reduce_registration_line"] = lineno
    return details


def _extract_graph_mode(run_meta: Dict[str, Any]) -> str:
    if not isinstance(run_meta, dict):
        return ""
    graph_status = dict(run_meta.get("graph_mode_status") or {})
    vllm_meta = dict(run_meta.get("vllm") or {})
    resolved = str(
        graph_status.get("resolved_cudagraph_mode")
        or vllm_meta.get("resolved_cudagraph_mode")
        or ""
    ).strip()
    requested = str(
        graph_status.get("requested_cudagraph_mode")
        or vllm_meta.get("requested_cudagraph_mode")
        or ""
    ).strip()
    if resolved:
        return resolved
    if requested:
        return requested
    graph_finished = graph_status.get("graph_capture_finished")
    if graph_finished is True:
        return "captured"
    return ""


def _infer_backend_and_decode_mode(
    phase_dir: Path,
    run_meta: Dict[str, Any],
) -> Tuple[str, str, str, Dict[str, Any]]:
    vllm_meta = dict(run_meta.get("vllm") or {}) if isinstance(run_meta, dict) else {}
    launch_env = dict(run_meta.get("launch_env") or {}) if isinstance(run_meta, dict) else {}
    graph_details = _extract_graph_log_details(phase_dir)

    backend = str(vllm_meta.get("attention_backend") or "").strip()
    decode_mode = str(launch_env.get("VLLM_AITER_HEAD4_DECODE_MODE") or "").strip()
    stage1_mode = str(launch_env.get("VLLM_AITER_MLA_STAGE1_MODE") or "").strip()
    display_mode = stage1_mode or decode_mode
    if backend and display_mode:
        return backend, decode_mode, display_mode, graph_details

    server_cmd = _read_server_cmd_text(phase_dir).lower()
    log_text = _read_vllm_log_text(phase_dir)
    requested_graph_mode = str(vllm_meta.get("requested_cudagraph_mode") or "").strip().upper()
    resolved_graph_mode = str(vllm_meta.get("resolved_cudagraph_mode") or "").strip().upper()
    graph_mode = resolved_graph_mode or requested_graph_mode

    if not backend:
        backend = _infer_backend_from_log(log_text)

    if not display_mode and backend == "TRITON_MLA":
        display_mode = "Triton MLA"

    if not display_mode and backend == "ROCM_AITER_MLA":
        display_mode = "AITER MLA"

    if not backend and "--attention-backend rocm_aiter_mla" in server_cmd:
        backend = "ROCM_AITER_MLA"
        display_mode = display_mode or "AITER MLA"
    elif not backend and "--attention-backend triton_mla" in server_cmd:
        backend = "TRITON_MLA"
        display_mode = display_mode or "Triton MLA"

    if not display_mode:
        log_mode = _infer_mode_from_log(log_text)
        if log_mode:
            display_mode = log_mode
        elif graph_mode and graph_mode != "NONE":
            display_mode = f"cudagraph_mode={graph_mode}"

    return backend, decode_mode, display_mode, graph_details


def _load_run_meta_optional(phase_dir: Path) -> Dict[str, Any]:
    try:
        return _load_run_meta_with_driver_fallback(phase_dir)
    except Exception:
        return {}


def _common_nonempty_str(values: Iterable[Any]) -> str:
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return ""
    uniq = sorted(set(cleaned))
    return uniq[0] if len(uniq) == 1 else ""


def _collect_mode_evidence(
    rows: List[Dict[str, Any]],
    value_key: str,
    line_key: str,
) -> List[Tuple[str, Optional[Path], int]]:
    seen: Dict[str, Tuple[Optional[Path], int]] = {}
    for row in rows:
        value = str(row.get(value_key, "")).strip()
        if not value or value in seen:
            continue
        source = str(row.get("source", "")).strip()
        if not source:
            seen[value] = (None, 0)
            continue
        phase_dir = Path(source)
        log_path = phase_dir / "vllm_server.log"
        line_no = int(row.get(line_key, 0) or 0)
        seen[value] = (log_path, line_no)
    return [(value, path, line_no) for value, (path, line_no) in sorted(seen.items())]


def _render_mode_evidence(
    rows: List[Dict[str, Any]],
    root: Path,
    base_dir: Path,
    value_key: str,
    line_key: str,
) -> str:
    parts: List[str] = []
    for value, path, line_no in _collect_mode_evidence(rows, value_key, line_key):
        abs_path = (root / path).resolve() if path else None
        if abs_path and line_no > 0 and abs_path.exists():
            link = _md_link(abs_path, base_dir, label=f"vllm_server.log:L{line_no}", fragment=f"#L{line_no}")
            parts.append(f"`{value}` ({link})")
        elif abs_path and abs_path.exists():
            link = _md_link(abs_path, base_dir, label="vllm_server.log")
            parts.append(f"`{value}` ({link})")
        else:
            parts.append(f"`{value}`")
    return ", ".join(parts)


def _reference_description(label: str, report_md: str) -> str:
    low = str(label or "").lower()
    path_txt = str(report_md or "").lower()
    if "mfma_head4" in low or "triton" in low:
        return (
            "Lower target: historical recommended ROCm Triton `mfma_head4 head_fused` path. Stage-1 fuses QK and PV across the 4 local TP heads (heads padded to 16) in one Triton MFMA tile per KV split; stage-2 is the split-K online-softmax reduction (`STAGE2_MODE=torch_ref` in the snapshotted env)."
        )
    if str(label or "").strip():
        return f"Reference campaign: `{label}`."
    if path_txt:
        return f"Reference campaign sourced from `{Path(report_md).name}`."
    return ""


def _phase_artifact_links(source_rel: str, root: Path, base_dir: Path) -> Dict[str, str]:
    phase_dir = (root / source_rel).resolve()
    return {
        "phase": _md_link_if_exists(phase_dir, base_dir, label=Path(source_rel).name),
        "server_cmd": _md_link_if_exists(phase_dir / "server_cmd.txt", base_dir, label="server_cmd.txt"),
        "run_meta": _md_link_if_exists(phase_dir / "run_meta.json", base_dir, label="run_meta.json"),
        "launch_env": _md_link_if_exists(phase_dir / "launch_env.json", base_dir, label="launch_env.json"),
    }


def _derive_campaign_make_wrapper(
    root: Path,
    campaign_dir: Path,
    latest_rows: List[Dict[str, Any]],
    *,
    auto_wrapper_target: str = "",
    auto_wrapper_run_id_var: str = "",
) -> str:
    prompts = sorted(
        {
            int(row.get("prompt_tokens_requested") or row.get("prompt_tokens") or 0)
            for row in latest_rows
            if int(row.get("prompt_tokens_requested") or row.get("prompt_tokens") or 0) > 0
        }
    )
    if not prompts:
        return ""

    run_id = campaign_dir.name[:-9] if campaign_dir.name.endswith("_campaign") else campaign_dir.name
    backends = {
        str(row.get("backend") or "").strip().lower()
        for row in latest_rows
        if str(row.get("backend") or "").strip()
    }
    if backends == {"fastapi"}:
        example_source = str(latest_rows[0].get("source", "") or "").strip()
        example_meta = (
            _load_run_meta_optional((root / example_source).resolve())
            if example_source
            else {}
        )
        fastapi_meta = dict(example_meta.get("fastapi") or {})
        prompt_shape = dict(example_meta.get("prompt_shape") or {})
        model_id = str(example_meta.get("model_id") or latest_rows[0].get("model_id") or "")
        max_new_tokens = int(example_meta.get("max_new_tokens") or 0)
        base_url = str(fastapi_meta.get("base_url") or "")
        shape_id = str(prompt_shape.get("id") or "")
        parts = [
            "make hf-long-context-fastapi-run \\",
            f"  HFLC_FASTAPI_RUN_ID={run_id} \\",
            f"  HFLC_FASTAPI_SWEEP={','.join(str(p) for p in prompts)}",
        ]
        if model_id:
            parts[-1] += " \\"
            parts.append(f"  HFLC_FASTAPI_MODEL_ID={model_id}")
        if base_url:
            parts[-1] += " \\"
            parts.append(f"  HFLC_FASTAPI_BASE_URL={base_url}")
        if max_new_tokens > 0:
            parts[-1] += " \\"
            parts.append(f"  HFLC_FASTAPI_MAX_NEW_TOKENS={max_new_tokens}")
        if shape_id:
            parts[-1] += " \\"
            parts.append(f"  HFLC_FASTAPI_PROMPT_SHAPE={shape_id}")
        return "\n".join(parts)

    run_id_low = run_id.lower()
    kv_dtypes = sorted(
        {
            str(row.get("kv_cache_dtype") or "").strip()
            for row in latest_rows
            if str(row.get("kv_cache_dtype") or "").strip()
        }
    )

    if auto_wrapper_target and auto_wrapper_run_id_var and kv_dtypes == ["auto"]:
        return "\n".join(
            [
                f"make {auto_wrapper_target} \\",
                f"  {auto_wrapper_run_id_var}={run_id}",
            ]
        )

    metas = [
        _load_run_meta_optional((root / str(row.get("source", ""))).resolve())
        for row in latest_rows
        if str(row.get("source", "")).strip()
    ]
    metas = [m for m in metas if m]
    prompt_shape_id = _common_nonempty_str(
        (m.get("prompt_shape", {}) or {}).get("id", "") for m in metas
    )
    # Prefer the wrapper entrypoint recorded at launch time (see
    # benchmark_vllm_long_context.py run_meta["wrapper"]["entrypoint"], stamped
    # by each Make wrapper recipe via :- defaulting). Fall back to the
    # `launch_env.HFLC_WRAPPER_ENTRYPOINT` mirror, then to a heuristic guess.
    recorded_entrypoint = _common_nonempty_str(
        (m.get("wrapper", {}) or {}).get("entrypoint", "") for m in metas
    )
    if not recorded_entrypoint:
        recorded_entrypoint = _common_nonempty_str(
            (m.get("launch_env", {}) or {}).get("HFLC_WRAPPER_ENTRYPOINT", "") for m in metas
        )
    target_is_guess = False
    if recorded_entrypoint:
        target = recorded_entrypoint
    else:
        target = "hf-long-context-fp8-sweep"
        target_is_guess = True

    parts: List[str] = []
    if target_is_guess:
        parts.append(
            "# NOTE: wrapper target inferred from cudagraph_mode (no recorded "
            "HFLC_WRAPPER_ENTRYPOINT in run_meta.json). Verify against the "
            "actual invocation; pre-2026-05-15 runs do not record this."
        )
    parts.extend([
        f"make {target} \\",
        f"  HFLC_FP8_RUN_ID={run_id} \\",
        f"  HFLC_FP8_SWEEP_LIST={','.join(str(p) for p in prompts)}",
    ])
    if prompt_shape_id:
        parts[-1] += " \\"
        parts.append(f"  HFLC_FP8_PROMPT_SHAPE={prompt_shape_id}")
    return "\n".join(parts)


def _fmt_prompt(n: int) -> str:
    if n <= 0:
        return str(n)
    # Prefer the largest unit that divides exactly. Binary (Ki/Mi/Gi) and
    # decimal (K/M/G) are checked separately so a value like 60_000_000
    # renders as "60M" while 67_108_864 renders as "64Mi".
    for divisor, suffix in (
        (1024 ** 3, "Gi"),
        (1000 ** 3, "G"),
        (1024 ** 2, "Mi"),
        (1000 ** 2, "M"),
        (1024, "Ki"),
        (1000, "K"),
    ):
        if n % divisor == 0:
            return f"{n // divisor}{suffix}"
    return str(n)


def _fmt_sig4(value: float) -> str:
    return format(float(value), ".4g")


def _pad_decimal_cells(values: List[float], suffix: str = "") -> List[str]:
    rendered = [_fmt_sig4(v) for v in values]
    dot_positions = [s.find(".") if "." in s else len(s) for s in rendered]
    max_dot = max(dot_positions, default=0)
    padded: List[str] = []
    for text, dot_pos in zip(rendered, dot_positions):
        left_pad = " " * (max_dot - dot_pos)
        padded.append(f"`{left_pad}{text}{suffix}`")
    return padded


def _fmt_optional_num_cell(value: Optional[float], suffix: str = "", places: Optional[int] = None) -> str:
    if value is None:
        return "`-`"
    num = float(value)
    rendered = f"{num:.{int(places)}f}" if places is not None else _fmt_sig4(num)
    return f"`{rendered}{suffix}`"


def _filter_compare_metric_rows(rows: List[Dict[str, Any]], metric_key: str) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        outcome = str(row.get("outcome", "completed") or "completed")
        if "outcome" in row and outcome != "completed":
            continue
        if int(row.get("prompt_tokens", 0)) <= 0:
            continue
        if float(row.get(metric_key, 0.0)) <= 0.0:
            continue
        filtered.append(row)
    return sorted(filtered, key=lambda r: _compare_x_prompt(r))


def _compare_x_prompt(row: Dict[str, Any]) -> int:
    """Return the prompt length to plot at on cross-campaign comparison axes.

    Snaps to ``prompt_tokens_requested`` (the originally-requested power of 2)
    when present, falling back to the effective post-tokenizer
    ``prompt_tokens`` otherwise. Some campaigns run with
    ``prompt_seed_extension_mode=preserve_shortfall``, which yields effective
    ``prompt_tokens`` slightly less than the request (e.g. 1024 -> 1018 due to
    BPE shortfall when truncating a real-text seed). Comparison plots and
    speed-up tables snap to the requested length so series with different
    extension modes share the same powers-of-2 ticks.
    """

    requested = int(row.get("prompt_tokens_requested", 0) or 0)
    if requested > 0:
        return requested
    return int(row.get("prompt_tokens", 0) or 0)


def _classify_outcome(raw_status: str, error: str) -> str:
    s = (raw_status or "").strip().lower()
    e = (error or "").lower()
    if s in {"pending_retest", "in_flight"}:
        return s
    if s == "ok":
        return "completed"
    if "out of memory" in e or "hip out of memory" in e:
        return "oom"
    if "timeout" in e:
        return "timeout"
    if "hang" in e:
        return "hang"
    return "engine_error"


def _extract_prompt_seed_config(run_meta: Dict[str, Any]) -> Dict[str, Any]:
    substrate_prompt = (
        dict(run_meta.get("substrate_prompt") or {}) if isinstance(run_meta, dict) else {}
    )
    return {
        "prompt_seed_extension_mode": str(
            substrate_prompt.get("prompt_seed_extension_mode") or "none"
        ),
        "prompt_seed_source_tokens": int(
            substrate_prompt.get("prompt_seed_source_tokens") or 0
        ),
        "prompt_seed_expanded_tokens": int(
            substrate_prompt.get("prompt_seed_expanded_tokens") or 0
        ),
        "prompt_seed_repeat_count": int(
            substrate_prompt.get("prompt_seed_repeat_count") or 0
        ),
        "requested_prompt_tokens": int(
            substrate_prompt.get("requested_prompt_tokens") or 0
        ),
    }


def _classify_empty_phase(phase_dir: Path) -> Tuple[str, str]:
    """Classify a launched phase with no results rows yet."""

    progress = _read_last_jsonl_record(phase_dir / "prefill_progress.jsonl")
    if progress:
        status = str(progress.get("status") or "").strip().lower()
        request_status = str(progress.get("request_status") or "").strip().lower()
        note = str(progress.get("note") or "").strip().lower()
        if status == "error" or "failed" in note or "error" in note:
            return "engine_error", note
        if status == "completed" and note == "request_completed":
            return "in_flight", ""
        if status in {"running", "completed"} or request_status in {"running"}:
            return "in_flight", ""
    driver_log = phase_dir.parent / "driver.log"
    if driver_log.exists():
        try:
            driver_lines = driver_log.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            driver_lines = []
        failed_markers = (
            "failed rc=",
            "vllm did not become healthy",
            "phase=phase1 failed",
            "RuntimeError: Engine core initialization failed",
        )
        for line in reversed(driver_lines):
            low = line.lower()
            if any(marker in low for marker in failed_markers):
                return "engine_error", _extract_root_cause(phase_dir, "")
    if (phase_dir / "vllm_server.log").exists():
        return "in_flight", ""
    return "engine_error", _extract_root_cause(phase_dir, "")


def _verify_fp8_launch_settings(
    phase_dir: Path, run_meta: Dict[str, Any], prompt_tokens: int
) -> Dict[str, Any]:
    """Validate required launch settings for FP8 AITER long-point runs.

    Returns a dict with booleans and missing requirements. This is only used
    as a classification guardrail, not as a hard pass/fail for completed rows.
    """
    out = {
        "required": False,
        "server_cmd_ok": False,
        "launch_env_ok": False,
        "launch_env_na": False,
        "log_indicator_ok": False,
        "missing": [],
    }
    vllm_meta = run_meta.get("vllm", {}) if isinstance(run_meta, dict) else {}
    launch_env = run_meta.get("launch_env", {}) if isinstance(run_meta, dict) else {}
    backend = str(vllm_meta.get("attention_backend", ""))
    kv_dtype = str(vllm_meta.get("kv_cache_dtype", ""))
    if backend != "ROCM_AITER_MLA" or kv_dtype != "fp8_e4m3":
        return out
    if prompt_tokens < 20 * 1024 * 1024:
        return out

    out["required"] = True
    required_missing: List[str] = []

    cmd_path = phase_dir / "server_cmd.txt"
    cmd_txt = cmd_path.read_text(encoding="utf-8", errors="replace") if cmd_path.exists() else ""
    cmd_needles = (
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend ROCM_AITER_MLA",
        "--max-num-seqs",
    )
    max_num_seqs_ok = bool(re.search(r"--max-num-seqs\s+(1|16)\b", cmd_txt))
    cmd_ok = all(n in cmd_txt for n in cmd_needles) and max_num_seqs_ok
    out["server_cmd_ok"] = cmd_ok
    if not cmd_ok:
        required_missing.append("server_cmd flags")

    env_decode = str(launch_env.get("VLLM_AITER_HEAD4_DECODE_MODE", ""))
    env_sub16 = str(launch_env.get("VLLM_AITER_SUB16_MODE", ""))
    env_bypass = str(launch_env.get("VLLM_MLA_FP8_DECODE_BYPASS_QUANT", ""))
    env_stage1 = str(launch_env.get("VLLM_AITER_MLA_STAGE1_MODE", ""))
    env_stage2 = str(launch_env.get("VLLM_AITER_MLA_STAGE2_MODE", ""))
    legacy_mfma_mixed_contract = (
        env_bypass == "1"
        and env_decode == "direct_head4"
        and env_sub16 == "off"
        and env_stage1 == "mfma_head4"
        and env_stage2 in {"triton", "hip_fused"}
    )
    legacy_ok = env_bypass == "1" and (
        env_sub16 == "project_to_16" or env_decode == "math_ref_head4"
    )
    mfma_ok = (
        env_bypass == "1"
        and env_decode == "mfma_head4"
        and env_sub16 == "off"
        and env_stage1 == "mfma_head4"
        and env_stage2 in {"triton", "hip_fused", "torch_ref"}
    )
    env_ok = legacy_ok or mfma_ok
    # The env-driven head4/sub16 knobs only exist on the historical
    # snapshot-backed AITER path. Runs on the current stack do not set them, so
    # their absence is "not applicable" rather than a verification gap.
    legacy_env_present = any(
        (env_decode, env_sub16, env_bypass, env_stage1, env_stage2)
    )
    if legacy_mfma_mixed_contract or not legacy_env_present:
        out["launch_env_ok"] = None
        out["launch_env_na"] = True
    else:
        out["launch_env_ok"] = env_ok
    if not env_ok and legacy_env_present and not legacy_mfma_mixed_contract:
        required_missing.append(
            "launch_env contract "
            f"(got decode={env_decode or '<unset>'}, "
            f"sub16={env_sub16 or '<unset>'}, bypass={env_bypass or '<unset>'}, "
            f"stage1={env_stage1 or '<unset>'}, stage2={env_stage2 or '<unset>'})"
        )

    log_path = phase_dir / "vllm_server.log"
    log_txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    # Historical markers emitted only by the snapshot-backed decode path.
    legacy_indicators = (
        "FP8 decode bypass-quant enabled; using tuple decode_q path",
        "[vllm._aiter_ops] head4 decode mode=mfma_head4",
    )
    # Stock-stack equivalent: vLLM echoes the resolved launch arguments, so the
    # backend and KV dtype the run actually served with are recoverable there.
    stock_backend_ok = bool(
        re.search(r"'attention_backend':\s*'ROCM_AITER_MLA'", log_txt)
    ) or _infer_backend_from_log(log_txt) == "ROCM_AITER_MLA"
    stock_kv_ok = bool(re.search(r"'kv_cache_dtype':\s*'fp8_e4m3'", log_txt))
    log_ok = any(ind in log_txt for ind in legacy_indicators) or (
        stock_backend_ok and stock_kv_ok
    )
    out["log_indicator_ok"] = log_ok
    if not log_ok:
        required_missing.append("vllm_server.log AITER MLA + fp8 KV confirmation")

    out["missing"] = required_missing
    return out


def _extract_root_cause(phase_dir: Path, error: str) -> str:
    err = (error or "").strip()
    generic_client_errors = (
        "connection refused",
        "remote end closed connection",
        "failed to establish a new connection",
    )
    if err and not any(s in err.lower() for s in generic_client_errors):
        return err[:320]
    graph_status_path = phase_dir / "graph_mode_status.json"
    graph_capture_ok = False
    graph_mode = ""
    if graph_status_path.exists():
        try:
            graph_status = _load_json(graph_status_path)
        except Exception:
            graph_status = {}
        graph_capture_ok = bool(graph_status.get("graph_capture_finished") or graph_status.get("capture_ok"))
        graph_mode = str(graph_status.get("resolved_cudagraph_mode") or graph_status.get("requested_cudagraph_mode") or "")
    log_path = phase_dir / "vllm_server.log"
    lines: List[str] = []
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            lines = []

    preferred_patterns = (
        r"torch\.OutOfMemoryError:.*",
        r"RuntimeError: Worker failed with error '.*?'",
        r"HIP out of memory\..*",
        r"operation not permitted when stream is capturing.*",
        r"RuntimeError: Engine core initialization failed.*",
    )
    flat_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
    for pattern in preferred_patterns:
        for flat in reversed(flat_lines):
            match = re.search(pattern, flat)
            if not match:
                continue
            cause = match.group(0).strip()
            if graph_capture_ok and "out of memory" in cause.lower():
                prefix = f"graph_capture_finished({graph_mode or 'unknown'}) then "
                if not cause.lower().startswith("graph_capture_finished"):
                    cause = prefix + cause
            return cause[:320]

    noisy_substrings = ("hipmoduleload",)
    # Whole-word matching matters here: as substrings, "fault" hits "default" and
    # "hip" hits "chip"/"bpreshuffle"-style tokens, so routine tuning and startup
    # lines got reported as root causes.
    key_re = re.compile(r"\b(error|exception|out of memory|fault|traceback|hip)\b")
    for line in reversed(lines):
        low = line.lower()
        if any(noisy in low for noisy in noisy_substrings):
            continue
        if key_re.search(low):
            cause = re.sub(r"\s+", " ", line).strip()
            if graph_capture_ok and "out of memory" in low:
                cause = f"graph_capture_finished({graph_mode or 'unknown'}) then {cause}"
            return cause[:320]

    driver_log = phase_dir.parent / "driver.log"
    if driver_log.exists():
        try:
            driver_lines = driver_log.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            driver_lines = []
        driver_patterns = (
            r"RuntimeError: Engine core initialization failed.*",
            r"vLLM did not become healthy.*",
            r"phase=phase1 failed rc=\d+.*",
        )
        for line in reversed(driver_lines):
            flat = re.sub(r"\s+", " ", line).strip()
            for pattern in driver_patterns:
                match = re.search(pattern, flat)
                if match:
                    return match.group(0).strip()[:320]
    return ""


def _norm_source(root: Path, source: str) -> str:
    p = Path(source).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    try:
        return Path(os.path.relpath(str(p), str(root))).as_posix()
    except Exception:
        return str(p)


def _collect_rows(root: Path, source_rel: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    phase_dir = (root / source_rel).resolve()
    run_meta = _load_run_meta_with_driver_fallback(phase_dir)
    rows = load_jsonl_dicts(phase_dir / "results.jsonl")
    out: List[Dict[str, Any]] = []
    backend, decode_mode, display_mode, graph_details = _infer_backend_and_decode_mode(phase_dir, run_meta)
    allreduce_details = _extract_allreduce_log_details(phase_dir)
    kv_dtype = str(run_meta.get("vllm", {}).get("kv_cache_dtype", ""))
    stage1_mode = str(run_meta.get("launch_env", {}).get("VLLM_AITER_MLA_STAGE1_MODE", ""))
    enforce_eager = bool(run_meta.get("vllm", {}).get("enforce_eager", False))
    prompt_seed_cfg = _extract_prompt_seed_config(run_meta)
    graph_mode = _extract_graph_mode(run_meta)
    configured_graph_mode = str(graph_details.get("configured_graph_mode") or "")
    prefill_graph_mode = str(graph_details.get("prefill_graph_mode") or "")
    decode_graph_mode = str(graph_details.get("decode_graph_mode") or "")
    configured_graph_line = int(graph_details.get("configured_graph_mode_line") or 0)
    prefill_graph_line = int(graph_details.get("prefill_graph_mode_line") or 0)
    decode_graph_line = int(graph_details.get("decode_graph_mode_line") or 0)
    custom_all_reduce_mode = str(allreduce_details.get("custom_all_reduce_mode") or "")
    custom_all_reduce_registration = str(
        allreduce_details.get("custom_all_reduce_registration") or ""
    )
    decode_allreduce_impl = str(allreduce_details.get("decode_allreduce_impl") or "")
    custom_all_reduce_mode_line = int(
        allreduce_details.get("custom_all_reduce_mode_line") or 0
    )
    custom_all_reduce_registration_line = int(
        allreduce_details.get("custom_all_reduce_registration_line") or 0
    )
    decode_allreduce_impl_line = int(
        allreduce_details.get("decode_allreduce_impl_line") or 0
    )
    if configured_graph_mode and graph_mode in {"", "NONE"}:
        graph_mode = configured_graph_mode
    if backend == "TRITON_MLA":
        variant = "triton_mla"
    elif backend == "ROCM_AITER_MLA" and display_mode:
        variant = f"aiter_{display_mode}"
    elif backend:
        variant = backend.lower()
    else:
        variant = "unknown"

    # Some failed runs never emit results.jsonl. Surface these as engine_error
    # rows so failure points appear in consolidated tables.
    if not rows:
        prompt_list = run_meta.get("sweep_prompt_tokens") or []
        prompt_req = int(prompt_list[0]) if prompt_list else 0
        outcome, err = _classify_empty_phase(phase_dir)
        root_cause = "" if outcome == "in_flight" else _extract_root_cause(phase_dir, err)
        out.append(
            {
                "variant": variant,
                "backend": backend,
                "kv_cache_dtype": kv_dtype,
                "decode_mode": decode_mode,
                "stage1_mode": stage1_mode,
                "display_mode": display_mode,
                "enforce_eager": enforce_eager,
                "graph_mode": graph_mode,
                "configured_graph_mode": configured_graph_mode,
                "prefill_graph_mode": prefill_graph_mode,
                "decode_graph_mode": decode_graph_mode,
                "configured_graph_line": configured_graph_line,
                "prefill_graph_line": prefill_graph_line,
                "decode_graph_line": decode_graph_line,
                "custom_all_reduce_mode": custom_all_reduce_mode,
                "custom_all_reduce_registration": custom_all_reduce_registration,
                "decode_allreduce_impl": decode_allreduce_impl,
                "custom_all_reduce_mode_line": custom_all_reduce_mode_line,
                "custom_all_reduce_registration_line": custom_all_reduce_registration_line,
                "decode_allreduce_impl_line": decode_allreduce_impl_line,
                "prompt_tokens_requested": prompt_req,
                "prompt_tokens": prompt_req,
                "prompt_human": _fmt_prompt(prompt_req),
                "raw_status": outcome,
                "outcome": outcome,
                "ttft_s": 0.0,
                "tok_s": 0.0,
                "completion_tokens": 0,
                "k2ft_probe_tokens": 0,
                "k2ft_ms": 0.0,
                "prefill_est_ms": 0.0,
                "k2ft_cache_hit_confidence": "",
                "k2ft_status": "",
                "finish_reason": "",
                "generated_text_path": "",
                "error": err,
                "root_cause": root_cause,
                "iteration": 0,
                "run_id": str(run_meta.get("run_id") or phase_dir.parent.name),
                "source": source_rel,
                "timestamp_utc": str(run_meta.get("generated_at_utc") or ""),
                "verify_required": False,
                "verify_server_cmd_ok": False,
                "verify_launch_env_ok": False,
                "verify_launch_env_na": False,
                "verify_log_indicator_ok": False,
                **prompt_seed_cfg,
                "verify_missing": [],
            }
        )
        return out, run_meta

    for row in rows:
        prompt_req = int(row.get("target_prompt_tokens_requested", row.get("prompt_tokens", 0)) or 0)
        prompt_eff = int(row.get("target_prompt_tokens_effective", row.get("prompt_tokens", 0)) or 0)
        raw_status = str(row.get("status") or "")
        err = str(row.get("error") or "")
        outcome = _classify_outcome(raw_status, err)
        verify = _verify_fp8_launch_settings(phase_dir, run_meta, prompt_eff)
        # `launch_env_ok` is None, with `launch_env_na` set, when the legacy env
        # contract does not apply to the stack that produced the run. Absence of
        # a contract that cannot exist is not a failure, and `not None` is True,
        # so this has to consult `launch_env_na` before treating it as one.
        launch_env_failed = not verify.get("launch_env_na") and not verify.get("launch_env_ok")
        if (
            verify.get("required")
            and raw_status == "ok"
            and (
                not verify.get("server_cmd_ok")
                or launch_env_failed
                or not verify.get("log_indicator_ok")
            )
        ):
            outcome = "pending_retest"
            misses = ", ".join(str(x) for x in verify.get("missing", []))
            err = f"invalid_config: {misses or 'launch verification incomplete'}"
        out.append(
            {
                "variant": variant,
                "backend": backend,
                "kv_cache_dtype": kv_dtype,
                "decode_mode": decode_mode,
                "stage1_mode": stage1_mode,
                "display_mode": display_mode,
                "enforce_eager": enforce_eager,
                "graph_mode": graph_mode,
                "configured_graph_mode": configured_graph_mode,
                "prefill_graph_mode": prefill_graph_mode,
                "decode_graph_mode": decode_graph_mode,
                "configured_graph_line": configured_graph_line,
                "prefill_graph_line": prefill_graph_line,
                "decode_graph_line": decode_graph_line,
                "custom_all_reduce_mode": custom_all_reduce_mode,
                "custom_all_reduce_registration": custom_all_reduce_registration,
                "decode_allreduce_impl": decode_allreduce_impl,
                "custom_all_reduce_mode_line": custom_all_reduce_mode_line,
                "custom_all_reduce_registration_line": custom_all_reduce_registration_line,
                "decode_allreduce_impl_line": decode_allreduce_impl_line,
                "prompt_tokens_requested": prompt_req,
                "prompt_tokens": prompt_eff,
                "prompt_human": _fmt_prompt(prompt_eff),
                "raw_status": raw_status,
                "outcome": outcome,
                "ttft_s": round(float(row.get("ttft_ms") or 0.0) / 1000.0, 6),
                "tok_s": round(float(row.get("gen_tokens_per_sec") or 0.0), 6),
                "completion_tokens": int(row.get("completion_tokens") or 0),
                "k2ft_probe_tokens": int(row.get("k2ft_probe_tokens") or 0),
                "k2ft_ms": round(float(row.get("k2ft_ms") or 0.0), 6),
                "prefill_est_ms": round(float(row.get("prefill_est_ms") or 0.0), 6),
                "k2ft_cache_hit_confidence": str(row.get("k2ft_cache_hit_confidence") or ""),
                "k2ft_status": str(row.get("k2ft_status") or ""),
                "finish_reason": str(row.get("finish_reason") or ""),
                "generated_text_path": str(row.get("generated_text_path") or ""),
                "error": err,
                # A row that finished with no error has no root cause to report.
                # Scanning the server log regardless surfaces unrelated warnings
                # as though they explained something.
                "root_cause": _extract_root_cause(phase_dir, err) if (err or outcome != "completed") else "",
                "iteration": int(row.get("iteration", 0) or 0),
                "run_id": str(row.get("run_id") or run_meta.get("run_id") or phase_dir.parent.name),
                "source": source_rel,
                "timestamp_utc": str(row.get("timestamp_utc") or ""),
                "verify_required": bool(verify.get("required", False)),
                "verify_server_cmd_ok": bool(verify.get("server_cmd_ok", False)),
                "verify_launch_env_ok": verify.get("launch_env_ok", False),
                "verify_launch_env_na": bool(verify.get("launch_env_na", False)),
                "verify_log_indicator_ok": bool(verify.get("log_indicator_ok", False)),
                "verify_missing": list(verify.get("missing", [])),
                **prompt_seed_cfg,
            }
        )
    return out, run_meta


def _fit_ttft_piecewise(
    rows: List[Dict[str, Any]], cutoff_tokens: int
) -> Dict[str, Dict[str, float]]:
    done = [
        r
        for r in rows
        if r.get("outcome") == "completed"
        and int(r.get("prompt_tokens", 0)) > 0
        and float(r.get("ttft_s", 0.0)) > 0.0
    ]
    short = sorted([r for r in done if int(r.get("prompt_tokens", 0)) <= cutoff_tokens], key=lambda r: int(r.get("prompt_tokens", 0)))
    long = sorted([r for r in done if int(r.get("prompt_tokens", 0)) >= cutoff_tokens], key=lambda r: int(r.get("prompt_tokens", 0)))

    out: Dict[str, Dict[str, float]] = {
        "short_linear": {"n": 0, "a": 0.0, "b": 0.0, "r2": 0.0},
        "long_quadratic": {"n": 0, "a": 0.0, "b": 0.0, "c": 0.0, "r2": 0.0},
    }
    if len(short) >= 2:
        x = np.array([int(r["prompt_tokens"]) for r in short], dtype=np.float64)
        y = np.array([float(r["ttft_s"]) for r in short], dtype=np.float64)
        b, a = np.polyfit(x, y, 1)
        yhat = a + b * x
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        out["short_linear"] = {"n": int(len(short)), "a": float(a), "b": float(b), "r2": float(r2)}
    if len(long) >= 3:
        x = np.array([int(r["prompt_tokens"]) for r in long], dtype=np.float64)
        y = np.array([float(r["ttft_s"]) for r in long], dtype=np.float64)
        c, b, a = np.polyfit(x, y, 2)
        yhat = a + b * x + c * (x**2)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        out["long_quadratic"] = {"n": int(len(long)), "a": float(a), "b": float(b), "c": float(c), "r2": float(r2)}
    return out


def _make_prompt_plot(rows: List[Dict[str, Any]], png_path: Path) -> None:
    done = [r for r in rows if r.get("outcome") == "completed" and float(r.get("ttft_s", 0.0)) > 0]
    fail = [
        r
        for r in rows
        if r.get("outcome") not in {"completed", "pending_retest", "in_flight"}
        and float(r.get("ttft_s", 0.0)) > 0
    ]
    done = sorted(done, key=lambda r: int(r.get("prompt_tokens", 0)))
    fail = sorted(fail, key=lambda r: int(r.get("prompt_tokens", 0)))

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 7.5))
    fig.patch.set_facecolor("#0f1117")
    ax1.set_facecolor("#0f1117")

    if done:
        x = [int(r["prompt_tokens"]) for r in done]
        y = [float(r["ttft_s"]) for r in done]
        ax1.scatter(x, y, marker="o", s=34, color="#4aa3ff", label="completed TTFT")
        for xi, yi in zip(x, y):
            ax1.annotate(_fmt_ttft_human(yi), (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", color="#c9d1d9", fontsize=16)
    if fail:
        x = [int(r["prompt_tokens"]) for r in fail]
        y = [float(r["ttft_s"]) for r in fail]
        ax1.scatter(x, y, marker="x", color="#ff6b6b", s=70, label="failed points (with TTFT)")
        for xi, yi in zip(x, y):
            ax1.annotate(_fmt_ttft_human(yi), (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", color="#ff9b9b", fontsize=16)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: _fmt_ttft_human(v) if v > 0 else ""))
    ax1.set_ylabel("TTFT (s / m / h)")
    ax1.grid(True, linestyle="--", alpha=0.30, color="#4d5666")
    ax1.legend(loc="best")

    ax1.set_xlabel("Prompt tokens")
    ax1.tick_params(colors="#e6edf3")
    ax1.xaxis.label.set_color("#e6edf3")
    ax1.yaxis.label.set_color("#e6edf3")
    for spine in ax1.spines.values():
        spine.set_color("#8b949e")

    ticks = sorted({int(r.get("prompt_tokens", 0)) for r in rows if int(r.get("prompt_tokens", 0)) > 0})
    if ticks:
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([_fmt_prompt(t) for t in ticks], rotation=45, ha="right")

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def _make_decode_fit_plot(
    rows: List[Dict[str, Any]],
    png_path: Path,
    cutoff_tokens: int = 0,
    enable_inverse_fit: bool = False,
    min_prompt_tokens: int = 1,
) -> Dict[str, Any]:
    fit_rows = [
        r
        for r in rows
        if r.get("outcome") == "completed"
        and int(r.get("prompt_tokens", 0)) >= int(min_prompt_tokens)
        and float(r.get("tok_s", 0.0)) > 0
    ]
    fit_rows = sorted(fit_rows, key=lambda r: int(r.get("prompt_tokens", 0)))
    if len(fit_rows) < 2:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Insufficient completed points for decode fit", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        return {
            "a": 0.0,
            "b": 0.0,
            "r2": 0.0,
            "n": 0,
            "piecewise": {},
            "enabled": bool(enable_inverse_fit),
            "min_prompt_tokens": int(min_prompt_tokens),
        }

    x = np.array([int(r["prompt_tokens"]) for r in fit_rows], dtype=np.float64)
    tok = np.array([float(r["tok_s"]) for r in fit_rows], dtype=np.float64)
    a = 0.0
    b = 0.0
    r2 = 0.0

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.scatter(x, tok, color="#4aa3ff", marker="o", label="measured completed")
    for xi, yi in zip(x, tok):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", color="#c9d1d9", fontsize=16)
    if enable_inverse_fit:
        y = 1.0 / np.maximum(tok, 1e-12)
        b, a = np.polyfit(x, y, 1)  # y = a + b*x
        yhat = a + b * x
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        x_line = np.geomspace(float(np.min(x)), float(np.max(x)), 300)
        tok_fit = 1.0 / np.maximum(a + b * x_line, 1e-12)
        ax.plot(x_line, tok_fit, color="#ffb347", linewidth=2.0, label="inverse-linear fit")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Decode tok/s")
    ax.grid(True, linestyle="--", alpha=0.30, color="#4d5666")
    ax.legend(loc="best")
    if enable_inverse_fit:
        ax.set_title(f"1/tok_s = a + b*n, a={a:.10f}, b={b:.12e}, R^2={r2:.6f}")
    else:
        ax.set_title("Decode throughput trend (empirical points only)")
    ax.tick_params(colors="#e6edf3")
    ax.xaxis.label.set_color("#e6edf3")
    ax.yaxis.label.set_color("#e6edf3")
    ax.title.set_color("#e6edf3")
    ticks = sorted({int(r.get("prompt_tokens", 0)) for r in rows if int(r.get("prompt_tokens", 0)) > 0})
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_prompt(t) for t in ticks], rotation=45, ha="right")
    for spine in ax.spines.values():
        spine.set_color("#8b949e")
    piecewise: Dict[str, Dict[str, float]] = {}
    if cutoff_tokens > 0 and enable_inverse_fit:
        short_rows = [r for r in fit_rows if int(r.get("prompt_tokens", 0)) <= cutoff_tokens]
        long_rows = [r for r in fit_rows if int(r.get("prompt_tokens", 0)) >= cutoff_tokens]
        ax.axvline(float(cutoff_tokens), color="#2ca02c", linestyle="--", alpha=0.8)
        for key, subset, color in [
            ("short_inv_linear", short_rows, "#9467bd"),
            ("long_inv_linear", long_rows, "#8c564b"),
        ]:
            if len(subset) < 2:
                piecewise[key] = {"n": int(len(subset)), "a": 0.0, "b": 0.0, "r2": 0.0}
                continue
            xs = np.array([int(r["prompt_tokens"]) for r in subset], dtype=np.float64)
            ts = np.array([float(r["tok_s"]) for r in subset], dtype=np.float64)
            ys = 1.0 / np.maximum(ts, 1e-12)
            bb, aa = np.polyfit(xs, ys, 1)
            ysh = aa + bb * xs
            ss_res2 = float(np.sum((ys - ysh) ** 2))
            ss_tot2 = float(np.sum((ys - np.mean(ys)) ** 2))
            r22 = float(1.0 - ss_res2 / ss_tot2) if ss_tot2 > 0 else 0.0
            xl = np.geomspace(float(np.min(xs)), float(np.max(xs)), 150)
            tokf = 1.0 / np.maximum(aa + bb * xl, 1e-12)
            ax.plot(xl, tokf, color=color, linestyle="--", linewidth=1.8, label=f"{key} fit")
            piecewise[key] = {"n": int(len(subset)), "a": float(aa), "b": float(bb), "r2": float(r22)}
        ax.legend(loc="best")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return {
        "a": float(a),
        "b": float(b),
        "r2": float(r2),
        "n": int(len(fit_rows)),
        "piecewise": piecewise,
        "enabled": bool(enable_inverse_fit),
        "min_prompt_tokens": int(min_prompt_tokens),
    }


def _make_k2ft_plot(rows: List[Dict[str, Any]], png_path: Path) -> Dict[str, Any]:
    fit_rows = [
        r
        for r in rows
        if r.get("outcome") == "completed"
        and int(r.get("prompt_tokens", 0)) > 0
        and float(r.get("k2ft_ms", 0.0)) > 0.0
    ]
    fit_rows = sorted(fit_rows, key=lambda r: int(r.get("prompt_tokens", 0)))
    if len(fit_rows) < 2:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Insufficient completed points for K2FT plot", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        return {"n": int(len(fit_rows))}

    x = np.array([int(r["prompt_tokens"]) for r in fit_rows], dtype=np.float64)
    y = np.array([float(r["k2ft_ms"]) for r in fit_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y, marker="o", color="#2ca02c", label="K2FT ms")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("K2FT (ms)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    ax.set_title("K2FT latency trend (KV-ready to first token)")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return {"n": int(len(fit_rows))}


def _make_decode_compare_plot(series_specs: List[Dict[str, Any]], png_path: Path) -> Dict[str, Any]:
    colors = ["#ffb347", "#4aa3ff", "#3fb950", "#ff7b72", "#c297ff", "#79c0ff"]
    series_rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(series_specs):
        rows = _filter_compare_metric_rows(list(spec.get("rows") or []), "tok_s")
        series_rows.append(
            {
                "series_index": idx,
                "label": str(spec.get("label", "") or f"series_{idx+1}"),
                "rows": rows,
                "report_md": str(spec.get("report_md", "") or ""),
                "current": bool(spec.get("current")),
                "color": colors[idx % len(colors)],
            }
        )

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    if not any(spec["rows"] for spec in series_rows):
        ax.text(0.5, 0.5, "No decode throughput data available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        return {"series": [], "table_rows": []}

    for spec in series_rows:
        rows = list(spec["rows"])
        if not rows:
            continue
        x = [_compare_x_prompt(r) for r in rows]
        y = [float(r["tok_s"]) for r in rows]
        (line,) = ax.plot(x, y, marker="o", color=str(spec["color"]), linewidth=2.0, label=str(spec["label"]))
        spec["line"] = line

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Decode tok/s")
    ax.set_title("Decode throughput comparison across campaigns")
    ax.grid(True, linestyle="--", alpha=0.30, color="#4d5666")
    ax.margins(y=0.16)

    common_prompts: Optional[set[int]] = None
    for spec in series_rows:
        rows = list(spec["rows"])
        if not rows:
            continue
        prompt_set = {_compare_x_prompt(r) for r in rows if _compare_x_prompt(r) > 0}
        common_prompts = prompt_set if common_prompts is None else (common_prompts & prompt_set)
    legend_specs = [spec for spec in series_rows if spec.get("line") is not None]
    if common_prompts:
        anchor_prompt = max(common_prompts)

        def _legend_score(spec: Dict[str, Any]) -> float:
            for row in list(spec["rows"]):
                if _compare_x_prompt(row) == anchor_prompt:
                    return float(row.get("tok_s", 0.0))
            return float("-inf")

        legend_specs = sorted(
            legend_specs,
            key=lambda spec: (_legend_score(spec), -int(spec.get("series_index", 0))),
            reverse=True,
        )
        legend_title = f"Legend order by decode at common max prompt ({_fmt_prompt(anchor_prompt)})"
    else:
        legend_title = None

    # Legend below the axes (outside the plot) so it doesn't obscure data
    # near the upper-right of the curve at long L. ncol=1 keeps long labels
    # on one line each. bbox_to_anchor y=-0.22 clears the rotated x-tick
    # labels at 45deg.
    ax.legend(
        [spec["line"] for spec in legend_specs],
        [str(spec["label"]) for spec in legend_specs],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        frameon=True,
        title=legend_title,
    )
    ax.tick_params(colors="#e6edf3")
    ax.xaxis.label.set_color("#e6edf3")
    ax.yaxis.label.set_color("#e6edf3")
    ax.title.set_color("#e6edf3")

    all_y_values = [
        float(r["tok_s"])
        for spec in series_rows
        for r in list(spec["rows"])
        if float(r.get("tok_s", 0.0)) > 0.0
    ]
    y_min = min(all_y_values) if all_y_values else 1.0
    y_max = max(all_y_values) if all_y_values else 1.0
    log_span = math.log(y_max) - math.log(y_min) if y_max > y_min and y_min > 0 else 0.0

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    margin_px = 6.0
    placed_bboxes = []
    label_points = []
    label_all_points = len([spec for spec in series_rows if spec["rows"]]) <= 2
    for spec in series_rows:
        rows = list(spec["rows"])
        series_idx = int(spec.get("series_index", 0))
        label_color = str(spec["color"])
        for row in rows:
            xi = _compare_x_prompt(row)
            yi = float(row["tok_s"])
            if (not label_all_points) and xi not in DECODE_COMPARE_CURRENT_LABEL_PROMPTS:
                continue
            label_points.append(
                {
                    "series_idx": series_idx,
                    "x": xi,
                    "y": yi,
                    "color": label_color,
                }
            )

    point_display_positions = [
        ax.transData.transform((float(item["x"]), float(item["y"])))
        for item in label_points
        if float(item["y"]) > 0.0
    ]
    candidate_offsets_mid = (
        (0, 14),
        (14, 10),
        (-14, 10),
        (16, -10),
        (-16, -10),
        (0, -14),
        (18, 0),
        (-18, 0),
        # Wider horizontal spreads for densely-spaced points (e.g. 56Mi at
        # 58_720_256 vs 64Mi at 67_108_864 sit only ~0.2 log2 units apart, so
        # the ±18 px candidates above can't separate their labels).
        (28, 8),
        (-28, 8),
        (28, -8),
        (-28, -8),
        (40, 0),
        (-40, 0),
    )
    candidate_offsets_top = (
        (0, -14),
        (16, -10),
        (-16, -10),
        (18, 0),
        (-18, 0),
        (14, 10),
        (-14, 10),
        (0, 14),
        (28, -8),
        (-28, -8),
        (40, 0),
        (-40, 0),
    )
    candidate_offsets_bottom = (
        (0, 14),
        (14, 10),
        (-14, 10),
        (18, 0),
        (-18, 0),
        (16, -10),
        (-16, -10),
        (0, -14),
        (28, 8),
        (-28, 8),
        (40, 0),
        (-40, 0),
    )

    for item in label_points:
        xi = int(item["x"])
        yi = float(item["y"])
        current_px, current_py = ax.transData.transform((float(xi), float(yi)))
        if log_span > 0.0 and yi > 0.0:
            y_frac = (math.log(yi) - math.log(y_min)) / log_span
        else:
            y_frac = 0.5
        if y_frac >= 0.82:
            candidate_offsets = candidate_offsets_top
        elif y_frac <= 0.18:
            candidate_offsets = candidate_offsets_bottom
        else:
            rot = int(item["series_idx"]) % len(candidate_offsets_mid)
            candidate_offsets = candidate_offsets_mid[rot:] + candidate_offsets_mid[:rot]

        chosen = None
        for xoff, yoff in candidate_offsets:
            va = "bottom" if yoff >= 0 else "top"
            ann = ax.annotate(
                f"{yi:.1f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(xoff, yoff),
                ha="center",
                va=va,
                color=str(item["color"]),
                fontsize=13,
                clip_on=True,
                bbox=dict(boxstyle="round,pad=0.18", fc="#0f1117", ec="none", alpha=0.72),
            )
            bbox = ann.get_window_extent(renderer).padded(2.0)
            inside_axes = (
                bbox.x0 >= axes_bbox.x0 + margin_px
                and bbox.x1 <= axes_bbox.x1 - margin_px
                and bbox.y0 >= axes_bbox.y0 + margin_px
                and bbox.y1 <= axes_bbox.y1 - margin_px
            )
            overlaps_label = any(bbox.overlaps(prev) for prev in placed_bboxes)
            overlaps_point = False
            for px, py in point_display_positions:
                if abs(px - current_px) < 1e-6 and abs(py - current_py) < 1e-6:
                    continue
                if bbox.x0 - 6.0 <= px <= bbox.x1 + 6.0 and bbox.y0 - 6.0 <= py <= bbox.y1 + 6.0:
                    overlaps_point = True
                    break
            if inside_axes and not overlaps_label and not overlaps_point:
                chosen = (ann, bbox)
                break
            ann.remove()

        if chosen is None:
            xoff, yoff = candidate_offsets[0]
            va = "bottom" if yoff >= 0 else "top"
            ann = ax.annotate(
                f"{yi:.1f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(xoff, yoff),
                ha="center",
                va=va,
                color=str(item["color"]),
                fontsize=13,
                clip_on=True,
                bbox=dict(boxstyle="round,pad=0.18", fc="#0f1117", ec="none", alpha=0.72),
            )
            bbox = ann.get_window_extent(renderer).padded(2.0)
            chosen = (ann, bbox)

        placed_bboxes.append(chosen[1])
    ticks = sorted(
        {
            _compare_x_prompt(r)
            for spec in series_rows
            for r in list(spec["rows"])
            if _compare_x_prompt(r) > 0
        }
    )
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_prompt(t) for t in ticks], rotation=45, ha="right")
    for spine in ax.spines.values():
        spine.set_color("#8b949e")

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" expands the figure to include the below-axes legend
    # without clipping the rotated x-tick labels.
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    table_rows: List[Dict[str, Any]] = []
    series_by_prompt = [
        {
            "label": str(spec["label"]),
            "current": bool(spec.get("current")),
            "report_md": str(spec.get("report_md", "")),
            "values": {_compare_x_prompt(r): float(r["tok_s"]) for r in spec["rows"]},
        }
        for spec in series_rows
    ]
    prompt_tokens_union = sorted({prompt for spec in series_by_prompt for prompt in spec["values"].keys()})
    if series_by_prompt:
        baseline_label = str(series_by_prompt[0]["label"])
        for prompt_tokens in prompt_tokens_union:
            value_by_label: Dict[str, Optional[float]] = {}
            speedup_vs_first: Dict[str, Optional[float]] = {}
            speedup_vs_previous: Dict[str, Optional[float]] = {}
            prev_label: Optional[str] = None
            prev_value: Optional[float] = None
            baseline_value = series_by_prompt[0]["values"].get(prompt_tokens)
            for spec in series_by_prompt:
                label = str(spec["label"])
                value = spec["values"].get(prompt_tokens)
                value_by_label[label] = value
                if label != baseline_label:
                    speedup_vs_first[label] = (
                        (value / baseline_value)
                        if value is not None and baseline_value is not None and baseline_value > 0.0
                        else None
                    )
                if prev_label is not None:
                    speedup_vs_previous[label] = (
                        (value / prev_value)
                        if value is not None and prev_value is not None and prev_value > 0.0
                        else None
                    )
                prev_label = label
                prev_value = value
            table_rows.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "prompt_human": _fmt_prompt(prompt_tokens),
                    "value_by_label": value_by_label,
                    "speedup_vs_first": speedup_vs_first,
                    "speedup_vs_previous": speedup_vs_previous,
                }
            )
    return {
        "series": [
            {
                "label": str(spec["label"]),
                "report_md": str(spec.get("report_md", "")),
                "current": bool(spec.get("current")),
                "n": len(spec["rows"]),
            }
            for spec in series_rows
        ],
        "table_rows": table_rows,
    }


def _make_ttft_compare_plot(series_specs: List[Dict[str, Any]], png_path: Path) -> Dict[str, Any]:
    colors = ["#ffb347", "#4aa3ff", "#3fb950", "#ff7b72", "#c297ff", "#79c0ff"]
    series_rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(series_specs):
        rows = _filter_compare_metric_rows(list(spec.get("rows") or []), "ttft_s")
        series_rows.append(
            {
                "label": str(spec.get("label", "") or f"series_{idx+1}"),
                "rows": rows,
                "report_md": str(spec.get("report_md", "") or ""),
                "current": bool(spec.get("current")),
                "color": colors[idx % len(colors)],
            }
        )

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    if not any(spec["rows"] for spec in series_rows):
        ax.text(0.5, 0.5, "No TTFT data available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        return {"series": [], "table_rows": []}

    for spec in series_rows:
        rows = list(spec["rows"])
        if not rows:
            continue
        x = [_compare_x_prompt(r) for r in rows]
        y = [float(r["ttft_s"]) for r in rows]
        ax.plot(x, y, marker="o", color=str(spec["color"]), linewidth=2.0, label=str(spec["label"]))
        if not bool(spec.get("current")):
            continue
        # Label placement: alternate up/down per labeled point, plus shift
        # horizontally when consecutive labeled points are close on log-x
        # (e.g. 56Mi vs 64Mi sit ~0.2 log2 units apart and would overlap with
        # plain up/down alternation).
        labeled_idx = 0
        prev_log2_x: Optional[float] = None
        for row in rows:
            xi = _compare_x_prompt(row)
            yi = float(row["ttft_s"])
            if xi not in TTFT_COMPARE_CURRENT_LABEL_PROMPTS:
                continue
            log2_x = math.log2(xi) if xi > 0 else 0.0
            close_to_prev = (
                prev_log2_x is not None
                and (log2_x - prev_log2_x) < 0.6
            )
            if close_to_prev:
                # Push horizontally and slightly outward vertically.
                xoff = 22 if (labeled_idx % 2 == 0) else -22
                yoff = 8 if (labeled_idx % 2 == 0) else -18
                ha = "center"
            else:
                xoff = 0
                yoff = 8 if (labeled_idx % 2 == 0) else -18
                ha = "center"
            ax.annotate(
                _fmt_ttft_plot_label(yi),
                (xi, yi),
                textcoords="offset points",
                xytext=(xoff, yoff),
                ha=ha,
                color="#c9d1d9",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.18", fc="#0f1117", ec="none", alpha=0.72),
            )
            prev_log2_x = log2_x
            labeled_idx += 1

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("TTFT comparison across campaigns")
    ax.grid(True, linestyle="--", alpha=0.30, color="#4d5666")
    # Legend below the axes (outside the plot); see _make_decode_compare_plot
    # for rationale. ncol=1 stacked; y=-0.22 clears the 45deg rotated ticks.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        frameon=True,
    )
    ax.tick_params(colors="#e6edf3")
    ax.xaxis.label.set_color("#e6edf3")
    ax.yaxis.label.set_color("#e6edf3")
    ax.title.set_color("#e6edf3")
    ticks = sorted(
        {
            _compare_x_prompt(r)
            for spec in series_rows
            for r in list(spec["rows"])
            if _compare_x_prompt(r) > 0
        }
    )
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_prompt(t) for t in ticks], rotation=45, ha="right")
    for spine in ax.spines.values():
        spine.set_color("#8b949e")

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" expands the figure to include the below-axes legend
    # without clipping the rotated x-tick labels.
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    table_rows: List[Dict[str, Any]] = []
    series_by_prompt = [
        {
            "label": str(spec["label"]),
            "current": bool(spec.get("current")),
            "report_md": str(spec.get("report_md", "")),
            "values": {_compare_x_prompt(r): float(r["ttft_s"]) for r in spec["rows"]},
        }
        for spec in series_rows
    ]
    prompt_tokens_union = sorted({prompt for spec in series_by_prompt for prompt in spec["values"].keys()})
    if series_by_prompt:
        baseline_label = str(series_by_prompt[0]["label"])
        for prompt_tokens in prompt_tokens_union:
            value_by_label: Dict[str, Optional[float]] = {}
            ratio_vs_first: Dict[str, Optional[float]] = {}
            ratio_vs_previous: Dict[str, Optional[float]] = {}
            prev_value: Optional[float] = None
            baseline_value = series_by_prompt[0]["values"].get(prompt_tokens)
            for spec in series_by_prompt:
                label = str(spec["label"])
                value = spec["values"].get(prompt_tokens)
                value_by_label[label] = value
                if label != baseline_label:
                    ratio_vs_first[label] = (
                        (value / baseline_value)
                        if value is not None and baseline_value is not None and baseline_value > 0.0
                        else None
                    )
                if prev_value is not None:
                    ratio_vs_previous[label] = (
                        (value / prev_value)
                        if value is not None and prev_value is not None and prev_value > 0.0
                        else None
                    )
                prev_value = value
            table_rows.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "prompt_human": _fmt_prompt(prompt_tokens),
                    "value_by_label": value_by_label,
                    "ratio_vs_first": ratio_vs_first,
                    "ratio_vs_previous": ratio_vs_previous,
                }
            )
    return {
        "series": [
            {
                "label": str(spec["label"]),
                "report_md": str(spec.get("report_md", "")),
                "current": bool(spec.get("current")),
                "n": len(spec["rows"]),
            }
            for spec in series_rows
        ],
        "table_rows": table_rows,
    }


def _row_preference_key(row: Dict[str, Any]) -> Tuple[int, int, int, str, int, str]:
    """Rank duplicate prompt rows so one canonical point survives per prompt.

    Preference order:
    1. completed rows over failed/pending rows
    2. non-trivial completions (>1 token) over degenerate completions
    3. verified launch-contract rows over unverified rows when verification is required
    4. newer timestamps, higher iteration, then lexical source path
    """

    outcome = str(row.get("outcome", ""))
    completed = int(outcome == "completed")
    nontrivial = int(int(row.get("completion_tokens", 0) or 0) > 1)
    verify_required = bool(row.get("verify_required"))
    verify_launch = bool(row.get("verify_launch_env_ok")) or bool(row.get("verify_launch_env_na"))
    verified = int(
        (not verify_required)
        or (
            bool(row.get("verify_server_cmd_ok"))
            and verify_launch
            and bool(row.get("verify_log_indicator_ok"))
        )
    )
    return (
        completed,
        nontrivial,
        verified,
        str(row.get("timestamp_utc", "")),
        int(row.get("iteration", 0) or 0),
        str(row.get("source", "")),
    )


def _coalesce_latest(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select one canonical row per prompt length.

    Despite the historical name, this now prefers the strongest completed row
    first and uses timestamp only as a later tie-breaker. This avoids plots and
    summaries showing both an older non-English 64M point and the later
    English-only verification row.

    When a prompt length has multiple completed iterations from the SAME
    canonical run (e.g. small points run with HFLC_FP8_SMALL_POINT_NUM_RUNS>1 to
    average out the sub-100ms measurement-floor noise), the canonical row's
    `ttft_s` and `tok_s` are replaced with the MEDIAN across those iterations.
    Larger points typically have a single iteration, so this is a no-op there.
    """

    by_prompt: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        prompt_tokens = int(row.get("prompt_tokens", 0))
        old = by_prompt.get(prompt_tokens)
        if old is None or _row_preference_key(row) > _row_preference_key(old):
            by_prompt[prompt_tokens] = row

    canonical = sorted(by_prompt.values(), key=lambda r: int(r.get("prompt_tokens", 0)))

    # Median TTFT/decode across all completed rows that share the same prompt
    # AND the same measurement config (backend/decode_mode/kv dtype). This
    # combines:
    #   - in-server repeats of one run (HFLC_FP8_SMALL_POINT_NUM_RUNS>1), and
    #   - the N cold-start passes of an n-pass campaign (each pass is a distinct
    #     run_id / phase dir, but identical config),
    # while still NOT medianing across a manually-appended baseline of a
    # different config that happens to live in the same campaign sources.
    def _config_key(r: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            str(r.get("backend", "")),
            str(r.get("decode_mode", "")),
            str(r.get("kv_cache_dtype", "")),
        )

    for crow in canonical:
        prompt_tokens = int(crow.get("prompt_tokens", 0))
        ckey = _config_key(crow)
        siblings = [
            r
            for r in rows
            if int(r.get("prompt_tokens", 0)) == prompt_tokens
            and _config_key(r) == ckey
            and str(r.get("outcome", "")) == "completed"
        ]
        if len(siblings) <= 1:
            continue
        ttfts = [float(r.get("ttft_s", 0.0) or 0.0) for r in siblings if float(r.get("ttft_s", 0.0) or 0.0) > 0.0]
        toks = [float(r.get("tok_s", 0.0) or 0.0) for r in siblings if float(r.get("tok_s", 0.0) or 0.0) > 0.0]
        if ttfts:
            crow["ttft_s"] = round(float(median(ttfts)), 6)
        if toks:
            crow["tok_s"] = round(float(median(toks)), 6)
        crow["measured_iterations"] = len(siblings)
    return canonical


def build_report(
    root: Path,
    campaign_dir: Path,
    sources: List[str],
    rows: List[Dict[str, Any]],
    fit: Dict[str, float],
    cutoff_tokens: int = 0,
    ttft_fit: Optional[Dict[str, Dict[str, float]]] = None,
    ttft_compare: Optional[Dict[str, Any]] = None,
    decode_compare: Optional[Dict[str, Any]] = None,
    generated_at_utc: str = "",
    auto_wrapper_target: str = "",
    auto_wrapper_run_id_var: str = "",
) -> None:
    """Render the consolidated markdown campaign report.

    Args:
        root: Repository root used for resolving source paths.
        campaign_dir: Output campaign directory.
        sources: Relative phase directories included in the report.
        rows: Raw consolidated rows before prompt coalescing.
        fit: Decode-fit summary payload.
        cutoff_tokens: Token cutoff used for fit overlays.
        ttft_fit: Optional TTFT fit metadata.
        ttft_compare: Optional TTFT comparison payload.
        decode_compare: Optional decode comparison payload.
        generated_at_utc: Optional fixed generated-at timestamp for
            reproducible rebuilds.
        auto_wrapper_target: Optional wrapper make target used to recreate
            auto-dtype campaigns.
        auto_wrapper_run_id_var: Optional wrapper make variable used to pass
            the run identifier.
    """
    report = campaign_dir / "report.md"
    root = root.resolve()
    latest_rows = _coalesce_latest(rows)
    failures = [r for r in latest_rows if r.get("outcome") not in {"completed", "pending_retest", "in_flight"}]
    pending = [r for r in latest_rows if r.get("outcome") in {"pending_retest", "in_flight"}]

    backends = sorted({str(r.get("backend", "")) for r in latest_rows if str(r.get("backend", ""))})
    dtypes = sorted({str(r.get("kv_cache_dtype", "")) for r in latest_rows if str(r.get("kv_cache_dtype", ""))})
    dec_modes = sorted(
        {str(r.get("display_mode", "")) for r in latest_rows if str(r.get("display_mode", ""))}
    )
    graph_modes = sorted(
        {str(r.get("graph_mode", "")).strip() for r in latest_rows if str(r.get("graph_mode", "")).strip()}
    )
    configured_graph_modes = _render_mode_evidence(
        latest_rows, root, campaign_dir, "configured_graph_mode", "configured_graph_line"
    )
    prefill_graph_modes = _render_mode_evidence(
        latest_rows, root, campaign_dir, "prefill_graph_mode", "prefill_graph_line"
    )
    decode_graph_modes = _render_mode_evidence(
        latest_rows, root, campaign_dir, "decode_graph_mode", "decode_graph_line"
    )
    custom_all_reduce_modes = _render_mode_evidence(
        latest_rows,
        root,
        campaign_dir,
        "custom_all_reduce_mode",
        "custom_all_reduce_mode_line",
    )
    custom_all_reduce_registrations = _render_mode_evidence(
        latest_rows,
        root,
        campaign_dir,
        "custom_all_reduce_registration",
        "custom_all_reduce_registration_line",
    )
    decode_allreduce_impls = _render_mode_evidence(
        latest_rows,
        root,
        campaign_dir,
        "decode_allreduce_impl",
        "decode_allreduce_impl_line",
    )
    example_source = str(latest_rows[0].get("source", "")).strip() if latest_rows else ""
    example_phase_dir = (root / example_source).resolve() if example_source else None
    example_run_meta_md = (
        _md_link(example_phase_dir / "run_meta.json", campaign_dir, label="run_meta.json")
        if example_phase_dir and (example_phase_dir / "run_meta.json").exists()
        else "`run_meta.json`"
    )
    example_server_cmd_md = (
        _md_link(example_phase_dir / "server_cmd.txt", campaign_dir, label="server_cmd.txt")
        if example_phase_dir and (example_phase_dir / "server_cmd.txt").exists()
        else "`server_cmd.txt`"
    )
    # Model id is not hard-coded in the title (the make flow lets users pick any
    # model); surface it as a metadata line instead, read from the first source's
    # run_meta.
    report_model_id = ""
    for row in latest_rows:
        meta = _load_run_meta_optional((root / str(row.get("source", ""))).resolve())
        mid = str(meta.get("model_id") or "")
        if mid:
            report_model_id = mid
            break

    lines: List[str] = []
    lines.append("# Long-Context Consolidated Report")
    lines.append("")
    if report_model_id:
        lines.append(f"- Model: `{report_model_id}`")
    lines.append(f"- Generated (UTC): `{generated_at_utc or _now_iso()}`")
    lines.append(f"- Campaign dir: `{_rel_display(campaign_dir, root)}`")
    lines.append(f"- Sources count: `{len(sources)}`")
    lines.append(f"- Backend(s): `{', '.join(backends)}`")
    lines.append(f"- KV dtype(s): `{', '.join(dtypes)}`")
    lines.append(f"- Decode mode(s): `{', '.join(dec_modes)}`")
    lines.append(f"- Graph mode(s): `{', '.join(graph_modes)}`")
    if configured_graph_modes:
        lines.append(f"- Configured graph mode: {configured_graph_modes}")
    if prefill_graph_modes:
        lines.append(f"- Prefill graph capture mode(s): {prefill_graph_modes}")
    if decode_graph_modes:
        lines.append(f"- Decode graph capture mode(s): {decode_graph_modes}")
    if custom_all_reduce_modes:
        lines.append(f"- Custom all-reduce mode(s): {custom_all_reduce_modes}")
    else:
        lines.append("- Custom all-reduce mode(s): `unavailable from vLLM log`")
    if custom_all_reduce_registrations:
        lines.append(
            f"- Custom all-reduce graph registration(s): {custom_all_reduce_registrations}"
        )
    if decode_allreduce_impls:
        lines.append(
            f"- Decode all-reduce implementation(s): {decode_allreduce_impls}"
        )
    else:
        lines.append("- Decode all-reduce implementation(s): `unavailable from vLLM log`")
    lines.append(
        "- Parallelism/GPU mapping and exact launch commands are recorded per run in each phase directory's "
        f"{example_run_meta_md} and {example_server_cmd_md}. See the per-prompt table below for each run's files."
    )
    lines.append("")

    env_metas = [
        _load_run_meta_optional((root / str(row.get("source", ""))).resolve())
        for row in latest_rows
        if str(row.get("source", "")).strip()
    ]
    lines.extend(_render_environment_section(_select_host_env(env_metas)))

    # Per-Point Measured Table + scaling/decode plots come first after the
    # environment so the headline timing numbers are immediately visible.
    lines.append("## Per-Point Measured Table")
    lines.append("")
    lines.append("| Prompt(req) | Prompt(eff) | Outcome | TTFT | Decode tok/s | Completion | Iters | Source |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---|")
    for r in latest_rows:
        src = (root / str(r.get("source", ""))).resolve()
        src_md = _md_link(src, campaign_dir) if src.exists() else f"`{r.get('source', '')}`"
        outcome = str(r.get("outcome", ""))
        ttft_cell = f"`{_fmt_ttft_human(float(r.get('ttft_s', 0.0) or 0.0))}`"
        toks_cell = f"`{float(r.get('tok_s', 0.0)):8.2f}`"
        iters = int(r.get("measured_iterations", 1) or 1)
        lines.append(
            f"| {int(r.get('prompt_tokens_requested', 0))} | {int(r.get('prompt_tokens', 0))} | "
            f"`{outcome}` | {ttft_cell} | {toks_cell} | "
            f"{int(r.get('completion_tokens', 0))} | {iters} | {src_md} |"
        )
    lines.append("")
    lines.append(
        "> TTFT and decode tok/s are the median across `Iters` measured "
        "iterations (run in-server, no relaunch). TTFT below ~8Ki tokens is "
        "dominated by fixed per-request overhead (sub-100ms) rather than "
        "prefill compute, so small differences there are measurement noise, "
        "not a scaling signal. TTFT becomes a meaningful prefill metric at the "
        "longer context points."
    )
    lines.append("")
    lines.append("![Prompt scaling](prompt_length_scaling.png)")
    lines.append("")
    lines.append("![Decode trend](decode_inverse_linear_fit.png)")
    lines.append("")


    lines.append("## Reproducibility")
    lines.append("")
    wrapper_cmd = _derive_campaign_make_wrapper(
        root,
        campaign_dir,
        latest_rows,
        auto_wrapper_target=str(auto_wrapper_target or ""),
        auto_wrapper_run_id_var=str(auto_wrapper_run_id_var or ""),
    )
    if wrapper_cmd:
        lines.append("- Campaign wrapper entrypoint:")
        lines.append("")
        lines.append("```bash")
        lines.append(wrapper_cmd)
        lines.append("```")
        lines.append("")
    lines.append("- Exact per-phase `vllm serve` invocations are the source of truth in each phase's `server_cmd.txt`.")
    lines.append("")
    lines.append("| Prompt | Phase | vLLM command | Run metadata | Launch env |")
    lines.append("|---|---|---|---|---|")
    for row in latest_rows:
        prompt_label = _fmt_prompt(int(row.get("prompt_tokens", 0) or 0))
        links = _phase_artifact_links(str(row.get("source", "")), root, campaign_dir)
        lines.append(
            f"| {prompt_label} | {links['phase']} | {links['server_cmd']} | {links['run_meta']} | {links['launch_env']} |"
        )
    lines.append("")

    compare_series = []
    if ttft_compare and bool(ttft_compare.get("series")):
        compare_series = list(ttft_compare.get("series") or [])
    elif decode_compare and bool(decode_compare.get("series")):
        compare_series = list(decode_compare.get("series") or [])
    if compare_series:
        lines.append("## Comparison References")
        lines.append("")
        lines.append("| Series | Role | Description | Source |")
        lines.append("|---|---|---|---|")
        for idx, spec in enumerate(compare_series):
            label = str(spec.get("label", "") or "")
            report_md = str(spec.get("report_md", "") or "")
            if bool(spec.get("current")):
                role = "current"
                description = "Current campaign under test."
                source_md = "`this campaign`"
            else:
                role = "lower target" if idx == 0 else "upper target"
                description = _reference_description(label, report_md) or "-"
                source_md = _md_link(Path(report_md), campaign_dir, label=Path(report_md).name) if report_md else "`-`"
            lines.append(f"| `{label}` | `{role}` | {description} | {source_md} |")
        lines.append("")

    if pending:
        lines.append("## Pending Retest / In-Flight")
        lines.append("")
        lines.append("| Prompt | Outcome | Verification gaps | Source |")
        lines.append("|---|---|---|---|")
        for r in pending:
            src = (root / str(r.get("source", ""))).resolve()
            src_md = _md_link(src, campaign_dir) if src.exists() else f"`{r.get('source', '')}`"
            gaps = ", ".join(str(x) for x in (r.get("verify_missing") or []))
            if not gaps:
                gaps = "-"
            lines.append(
                f"| {_fmt_prompt(int(r.get('prompt_tokens', 0)))} | `{r.get('outcome', '')}` | {gaps[:220]} | {src_md} |"
            )
        lines.append("")

    if decode_compare and bool(decode_compare.get("present")):
        compare_series = list(decode_compare.get("series") or [])
        lines.append("## TTFT Comparison")
        lines.append("")
        lines.append("| Series | Source | Points |")
        lines.append("|---|---|---:|")
        for spec in compare_series:
            report_md = str(spec.get("report_md", "") or "")
            if report_md:
                report_path = Path(report_md)
                source_md = _md_link(report_path, campaign_dir, label=report_path.name)
            elif bool(spec.get("current")):
                source_md = "`this campaign`"
            else:
                source_md = "`-`"
            lines.append(
                f"| `{spec.get('label', '')}` | {source_md} | {int(spec.get('n', 0))} |"
            )
        lines.append("")
        lines.append(f"![TTFT comparison]({TTFT_COMPARE_PNG})")
        lines.append("")
        lines.append(
            "> **Footnote (x-axis snap).** Comparison plots and tables snap"
            " each series to its `prompt_tokens_requested` (the"
            " originally-requested power of 2) so series with slightly"
            " different effective `prompt_tokens` share the same ticks. Even"
            " sweeps configured with `prompt_seed_extension_mode=repeat_to_target`"
            " (substrate seed expanded to exactly 2^N tokens) land a few"
            " thousand tokens below 2^N at the request layer because the"
            " chat-template wrapping (system + user role markers + assistant"
            " turn) consumes ~1-3K tokens that displace substrate content."
            " Older campaigns also used `preserve_shortfall` (truncated"
            " rather than wrapped seed), which produced similar offsets via"
            " a different mechanism. The snap is plot/table only; the"
            " underlying JSONL still records the effective `prompt_tokens`."
        )
        lines.append("")
        ttft_table_rows = list((ttft_compare or {}).get("table_rows") or [])
        if ttft_table_rows and compare_series:
            base_label = str(compare_series[0].get("label", "") or "baseline")
            series_headers = [str(spec.get("label", "") or "") for spec in compare_series]
            later_labels = series_headers[1:]
            lines.append("### TTFT Ratio Table")
            lines.append("")
            header_cells = ["Prompt"] + [f"{label} TTFT" for label in series_headers]
            header_cells.extend([f"{label} / {base_label}" for label in later_labels])
            if len(series_headers) >= 3:
                last_label = series_headers[-1]
                prev_label = series_headers[-2]
                header_cells.append(f"{last_label} / {prev_label}")
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(["---"] + ["---:"] * (len(header_cells) - 1)) + "|")
            for row in ttft_table_rows:
                value_by_label = dict(row.get("value_by_label") or {})
                ratio_vs_first = dict(row.get("ratio_vs_first") or {})
                ratio_vs_previous = dict(row.get("ratio_vs_previous") or {})
                rendered_cells = [f"`{row['prompt_human']}`"]
                rendered_cells.extend(
                    (
                        f"`{_fmt_ttft_human(float(value_by_label.get(label)))}`"
                        if value_by_label.get(label) is not None
                        else "`-`"
                    )
                    for label in series_headers
                )
                rendered_cells.extend(
                    _fmt_optional_num_cell(ratio_vs_first.get(label), suffix="x") for label in later_labels
                )
                if len(series_headers) >= 3:
                    rendered_cells.append(
                        _fmt_optional_num_cell(ratio_vs_previous.get(series_headers[-1]), suffix="x")
                    )
                lines.append("| " + " | ".join(rendered_cells) + " |")
            lines.append("")
        lines.append("## Decode Throughput Comparison")
        lines.append("")
        lines.append("| Series | Source | Points |")
        lines.append("|---|---|---:|")
        for spec in compare_series:
            report_md = str(spec.get("report_md", "") or "")
            if report_md:
                report_path = Path(report_md)
                source_md = _md_link(report_path, campaign_dir, label=report_path.name)
            elif bool(spec.get("current")):
                source_md = "`this campaign`"
            else:
                source_md = "`-`"
            lines.append(
                f"| `{spec.get('label', '')}` | {source_md} | {int(spec.get('n', 0))} |"
            )
        lines.append("")
        lines.append(f"![Decode throughput comparison]({DECODE_COMPARE_PNG})")
        lines.append("")
        lines.append(
            "> **Footnote (x-axis snap).** As in the TTFT chart, this plot"
            " and its speed-up table snap each series to"
            " `prompt_tokens_requested`. Effective `prompt_tokens` lands"
            " ~1-3K below 2^N even under `repeat_to_target` because the"
            " chat-template wrapping displaces substrate content; snapping"
            " puts heterogeneous-mode runs on shared ticks. Underlying JSONL"
            " is unchanged."
        )
        lines.append("")
        table_rows = list(decode_compare.get("table_rows") or [])
        if table_rows and compare_series:
            base_label = str(compare_series[0].get("label", "") or "baseline")
            series_headers = [str(spec.get("label", "") or "") for spec in compare_series]
            later_labels = series_headers[1:]
            lines.append("### Decode Speed-up Table")
            lines.append("")
            header_cells = ["Prompt"] + [f"{label} tok/s" for label in series_headers]
            header_cells.extend([f"{label} vs {base_label}" for label in later_labels])
            if len(series_headers) >= 3:
                last_label = series_headers[-1]
                prev_label = series_headers[-2]
                header_cells.append(f"{last_label} vs {prev_label}")
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(["---"] + ["---:"] * (len(header_cells) - 1)) + "|")
            for row in table_rows:
                value_by_label = dict(row.get("value_by_label") or {})
                speedup_vs_first = dict(row.get("speedup_vs_first") or {})
                speedup_vs_previous = dict(row.get("speedup_vs_previous") or {})
                rendered_cells = [f"`{row['prompt_human']}`"]
                rendered_cells.extend(_fmt_optional_num_cell(value_by_label.get(label)) for label in series_headers)
                rendered_cells.extend(
                    _fmt_optional_num_cell(speedup_vs_first.get(label), suffix="x") for label in later_labels
                )
                if len(series_headers) >= 3:
                    rendered_cells.append(
                        _fmt_optional_num_cell(speedup_vs_previous.get(series_headers[-1]), suffix="x")
                    )
                lines.append("| " + " | ".join(rendered_cells) + " |")
            lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the consolidated campaign report builder CLI."""
    parser = argparse.ArgumentParser(description="Build consolidated HF long-context campaign report.")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--root", default="/app/long-context-serving")
    parser.add_argument("--append-phase-dir", action="append", default=[])
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Regenerate report.md only. Read consolidated sources from the existing "
            "campaign directory, recompute the in-memory summaries, and leave plots "
            "plus consolidated JSONL artifacts untouched."
        ),
    )
    parser.add_argument("--fit-cutoff-tokens", type=int, default=0)
    parser.add_argument("--enable-inverse-linear-fit", action="store_true")
    parser.add_argument("--decode-min-prompt-tokens", type=int, default=1)
    parser.add_argument("--baseline-report-md", default="")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--current-label", default="HIP kernel")
    parser.add_argument("--auto-wrapper-target", default="")
    parser.add_argument("--auto-wrapper-run-id-var", default="")
    parser.add_argument(
        "--generated-at-utc",
        default="",
        help="Optional fixed report timestamp used for reproducible rebuilds.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)

    src_path = campaign_dir / "consolidated_sources.txt"
    existing: List[str] = []
    if src_path.exists():
        existing = [ln.strip() for ln in src_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    merged = list(existing)
    for s in args.append_phase_dir:
        rel = _norm_source(root, s)
        if rel not in merged:
            merged.append(rel)
    merged = sorted(set(merged))
    if not bool(args.report_only):
        src_path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")

    all_rows: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    for source_rel in merged:
        phase_dir = root / source_rel
        if not (phase_dir / "run_meta.json").exists() and not (phase_dir.parent / "driver.log").exists():
            unresolved.append(source_rel)
            continue
        rows, _meta = _collect_rows(root, source_rel)
        all_rows.extend(rows)

    if unresolved:
        print(
            f"warning: {len(unresolved)} of {len(merged)} recorded sources did not resolve under {root}",
            file=sys.stderr,
        )
        for rel in unresolved[:5]:
            print(f"warning:   missing {rel}", file=sys.stderr)

    # Sources are recorded relative to whichever tree wrote them, so rebuilding a
    # campaign from a different tree than it was produced in (host vs container)
    # resolves nothing. Continuing would overwrite a good report and JSONL with
    # empty ones and still exit successfully.
    #
    # Whether that deserves a hard failure depends on what is at stake. If the
    # campaign already holds results, an empty rebuild destroys them and the run
    # must stop. If it holds nothing yet, there is nothing to lose, and failing
    # would abort the sweep that is still producing points -- a campaign whose
    # first point dies before writing run_meta.json or driver.log lands here.
    if merged and not all_rows:
        existing_artifacts = [
            p for p in (campaign_dir / "consolidated_results.jsonl", campaign_dir / "report.md") if p.exists()
        ]
        report = {
            "ok": False,
            "error": "no_rows_from_sources",
            "campaign_dir": str(campaign_dir),
            "root": str(root),
            "sources": len(merged),
            "unresolved": len(unresolved),
            "hint": "check that consolidated_sources.txt paths resolve under --root",
        }
        if existing_artifacts:
            report["preserved"] = [p.name for p in existing_artifacts]
            print(json.dumps(report), file=sys.stderr)
            return 2
        report["note"] = "campaign has no artifacts yet; nothing written and nothing lost"
        print(json.dumps(report), file=sys.stderr)
        return 0

    all_rows = sorted(
        all_rows,
        key=lambda r: (
            int(r.get("prompt_tokens", 0)),
            str(r.get("timestamp_utc", "")),
            int(r.get("iteration", 0)),
            str(r.get("source", "")),
        ),
    )
    if not bool(args.report_only):
        _write_jsonl(campaign_dir / "consolidated_results.jsonl", all_rows)

    canonical_rows = _coalesce_latest(all_rows)

    with tempfile.TemporaryDirectory(prefix="hflc-report-only-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        prompt_plot_path = (
            temp_dir / "prompt_length_scaling.png"
            if bool(args.report_only)
            else campaign_dir / "prompt_length_scaling.png"
        )
        decode_plot_path = (
            temp_dir / "decode_inverse_linear_fit.png"
            if bool(args.report_only)
            else campaign_dir / "decode_inverse_linear_fit.png"
        )
        _make_prompt_plot(canonical_rows, prompt_plot_path)
        fit = _make_decode_fit_plot(
            canonical_rows,
            decode_plot_path,
            cutoff_tokens=int(args.fit_cutoff_tokens or 0),
            enable_inverse_fit=bool(args.enable_inverse_linear_fit),
            min_prompt_tokens=int(args.decode_min_prompt_tokens or 1),
        )
        ttft_compare: Optional[Dict[str, Any]] = None
        decode_compare: Optional[Dict[str, Any]] = None
        compare_series: List[Dict[str, Any]] = []
        if str(args.baseline_report_md or "").strip():
            baseline_report_md = Path(str(args.baseline_report_md)).expanduser().resolve()
            baseline_rows = (
                _load_baseline_metric_rows(baseline_report_md)
                if baseline_report_md.exists()
                else []
            )
            if baseline_rows:
                compare_series.append(
                    {
                        "label": str(args.baseline_label),
                        "rows": baseline_rows,
                        "report_md": str(baseline_report_md),
                        "current": False,
                    }
                )
        if compare_series:
            compare_series.append(
                {
                    "label": str(args.current_label),
                    "rows": canonical_rows,
                    "report_md": "",
                    "current": True,
                }
            )
            ttft_compare_path = (
                temp_dir / TTFT_COMPARE_PNG
                if bool(args.report_only)
                else campaign_dir / TTFT_COMPARE_PNG
            )
            decode_compare_path = (
                temp_dir / DECODE_COMPARE_PNG
                if bool(args.report_only)
                else campaign_dir / DECODE_COMPARE_PNG
            )
            ttft_compare = _make_ttft_compare_plot(compare_series, ttft_compare_path)
            compare_stats = _make_decode_compare_plot(compare_series, decode_compare_path)
            decode_compare = {
                "present": True,
                "series": list(compare_stats.get("series") or ttft_compare.get("series") or []),
                "table_rows": list(compare_stats.get("table_rows") or []),
            }
        ttft_fit = (
            _fit_ttft_piecewise(canonical_rows, int(args.fit_cutoff_tokens or 0))
            if int(args.fit_cutoff_tokens or 0) > 0
            else None
        )
        build_report(
            root,
            campaign_dir,
            merged,
            all_rows,
            fit,
            cutoff_tokens=int(args.fit_cutoff_tokens or 0),
            ttft_fit=ttft_fit,
            ttft_compare=ttft_compare,
            decode_compare=decode_compare,
            generated_at_utc=str(args.generated_at_utc or ""),
            auto_wrapper_target=str(args.auto_wrapper_target or ""),
            auto_wrapper_run_id_var=str(args.auto_wrapper_run_id_var or ""),
        )

    print(
        json.dumps(
            {
                "ok": True,
                "campaign_dir": str(campaign_dir),
                "sources": len(merged),
                "rows": len(all_rows),
                "fit": fit,
                "fit_cutoff_tokens": int(args.fit_cutoff_tokens or 0),
                "ttft_fit": ttft_fit or {},
                "ttft_compare": ttft_compare or {},
                "decode_compare": decode_compare or {},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
