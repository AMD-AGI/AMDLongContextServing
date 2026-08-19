# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""KV-capacity calibration helpers for long-context vLLM runs.

This module is intentionally independent of vLLM internals. It consumes startup
log lines and computes a conservative block-level capacity gate for target
prompt lengths.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_RE_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([0-9][0-9,]*)\s*tokens", re.IGNORECASE)
_RE_BLOCK_SIZE = re.compile(
    r"Setting attention block size to\s*([0-9][0-9,]*)\s*tokens",
    re.IGNORECASE,
)
_RE_MAX_CONCURRENCY = re.compile(
    r"Maximum concurrency for\s*([0-9][0-9,]*)\s*tokens per request:\s*([0-9]+(?:\.[0-9]+)?)x",
    re.IGNORECASE,
)


def _to_int(raw: str, default: int = 0) -> int:
    try:
        return int(str(raw).replace(",", "").strip())
    except Exception:  # noqa: BLE001
        return int(default)


def _to_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(str(raw).strip())
    except Exception:  # noqa: BLE001
        return float(default)


def parse_kv_startup_metrics_from_text(text: str) -> Dict[str, Any]:
    """Parse KV-capacity startup metrics from vLLM server log text."""
    metrics: Dict[str, Any] = {
        "tokens_avail": 0,
        "block_size_tokens": 0,
        "max_concurrency_tokens": 0,
        "max_concurrency_x": 0.0,
        "ready": False,
    }
    if not text:
        return metrics

    kv_matches = list(_RE_KV_TOKENS.finditer(text))
    if kv_matches:
        metrics["tokens_avail"] = _to_int(kv_matches[-1].group(1))

    block_matches = list(_RE_BLOCK_SIZE.finditer(text))
    if block_matches:
        metrics["block_size_tokens"] = _to_int(block_matches[-1].group(1))

    conc_matches = list(_RE_MAX_CONCURRENCY.finditer(text))
    if conc_matches:
        metrics["max_concurrency_tokens"] = _to_int(conc_matches[-1].group(1))
        metrics["max_concurrency_x"] = _to_float(conc_matches[-1].group(2))

    metrics["ready"] = bool(metrics["tokens_avail"] > 0 and metrics["block_size_tokens"] > 0)
    return metrics


def wait_for_kv_startup_metrics(
    log_path: Path,
    *,
    timeout_sec: float,
    poll_interval_sec: float = 0.5,
) -> Dict[str, Any]:
    """Wait for startup metrics to appear in log file."""
    start = time.time()
    timeout = max(0.0, float(timeout_sec))
    poll = max(0.1, float(poll_interval_sec))
    last_text = ""
    while True:
        if log_path.exists():
            try:
                last_text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                last_text = ""
        metrics = parse_kv_startup_metrics_from_text(last_text)
        metrics["log_path"] = str(log_path)
        metrics["wait_elapsed_sec"] = round(time.time() - start, 6)
        if bool(metrics.get("ready")):
            return metrics
        if (time.time() - start) >= timeout:
            metrics["error"] = "startup_metrics_timeout"
            return metrics
        time.sleep(poll)


def _headroom_tokens(
    *,
    target_prompt_tokens: int,
    block_size_tokens: int,
    headroom_ratio: float,
    headroom_min_blocks: int,
) -> int:
    ratio_tokens = int(math.ceil(max(0.0, float(headroom_ratio)) * max(0, int(target_prompt_tokens))))
    min_tokens = max(0, int(headroom_min_blocks)) * max(1, int(block_size_tokens))
    return max(ratio_tokens, min_tokens)


def evaluate_kv_capacity_target(
    *,
    target_prompt_tokens: int,
    max_new_tokens: int,
    max_num_seqs: int,
    tokens_avail: int,
    block_size_tokens: int,
    headroom_ratio: float = 0.005,
    headroom_min_blocks: int = 2,
) -> Dict[str, Any]:
    """Evaluate whether available KV blocks are sufficient for one target."""
    prompt_tokens = max(0, int(target_prompt_tokens))
    gen_tokens = max(0, int(max_new_tokens))
    seqs = max(1, int(max_num_seqs))
    block = max(1, int(block_size_tokens))
    avail_tokens = max(0, int(tokens_avail))
    headroom = _headroom_tokens(
        target_prompt_tokens=prompt_tokens,
        block_size_tokens=block,
        headroom_ratio=float(headroom_ratio),
        headroom_min_blocks=int(headroom_min_blocks),
    )
    per_seq_total = prompt_tokens + gen_tokens + headroom
    required_total = seqs * per_seq_total
    blocks_required = int(math.ceil(required_total / float(block)))
    blocks_avail = int(avail_tokens // block)
    margin_blocks = int(blocks_avail - blocks_required)
    margin_tokens = int(margin_blocks * block)
    decision = "PASS" if margin_blocks >= 0 else "FAIL"
    return {
        "target_prompt_tokens": prompt_tokens,
        "max_new_tokens": gen_tokens,
        "max_num_seqs": seqs,
        "tokens_avail": avail_tokens,
        "block_size_tokens": block,
        "headroom_ratio": float(headroom_ratio),
        "headroom_min_blocks": int(headroom_min_blocks),
        "headroom_tokens": int(headroom),
        "required_tokens_per_seq": int(per_seq_total),
        "required_tokens_total": int(required_total),
        "required_tokens_rounded_to_block": int(blocks_required * block),
        "blocks_required": int(blocks_required),
        "blocks_avail": int(blocks_avail),
        "margin_blocks": int(margin_blocks),
        "margin_tokens": int(margin_tokens),
        "decision": decision,
        "pass": bool(margin_blocks >= 0),
    }


def recommend_kv_adjustment(
    row: Mapping[str, Any],
    *,
    gpu_memory_utilization: float,
    util_cap: float = 0.99,
) -> Dict[str, Any]:
    """Return a practical next-action recommendation from one calibration row."""
    margin_blocks = int(row.get("margin_blocks") or 0)
    block = max(1, int(row.get("block_size_tokens") or 1))
    max_num_seqs = max(1, int(row.get("max_num_seqs") or 1))
    util_now = max(0.1, float(gpu_memory_utilization))
    util_cap = max(util_now, float(util_cap))

    if margin_blocks >= 0:
        return {
            "action": "keep_settings",
            "message": "Calibration pass. Keep current settings for this target.",
            "suggested": {},
        }

    deficit_blocks = abs(margin_blocks)
    deficit_tokens = int(deficit_blocks * block)
    if max_num_seqs > 1:
        return {
            "action": "reduce_max_num_seqs",
            "message": (
                "Calibration fail. Reduce max_num_seqs (prefer 1 for single-request long runs) "
                "before increasing memory knobs."
            ),
            "suggested": {"vllm_max_num_seqs": 1},
        }

    tokens_avail = max(1, int(row.get("tokens_avail") or 1))
    required_rounded = max(tokens_avail, int(row.get("required_tokens_rounded_to_block") or tokens_avail))
    util_needed = util_now * (required_rounded / float(tokens_avail))
    util_suggested = min(util_cap, round(util_needed + 0.01, 4))
    projected_tokens = int(tokens_avail * (util_suggested / max(1e-6, util_now)))
    if (
        util_suggested > util_now + 1e-6
        and util_suggested <= util_cap + 1e-9
        and projected_tokens >= required_rounded
    ):
        return {
            "action": "raise_gpu_memory_utilization",
            "message": "Calibration fail. Increase gpu_memory_utilization and recalibrate.",
            "suggested": {"gpu_memory_utilization": util_suggested},
        }

    blocks_override = int(row.get("blocks_required") or 0) + max(16, int(row.get("headroom_min_blocks") or 2))
    return {
        "action": "set_num_gpu_blocks_override",
        "message": (
            "Calibration fail. Configure explicit KV pool size via num_gpu_blocks_override "
            "or kv_cache_memory_bytes."
        ),
        "suggested": {
            "vllm_num_gpu_blocks_override": int(blocks_override),
            "deficit_blocks": int(deficit_blocks),
            "deficit_tokens": int(deficit_tokens),
        },
    }


def build_kv_capacity_calibration(
    *,
    target_prompt_tokens: Iterable[int],
    max_new_tokens: int,
    max_num_seqs: int,
    startup_metrics: Mapping[str, Any],
    gpu_memory_utilization: float,
    headroom_ratio: float = 0.005,
    headroom_min_blocks: int = 2,
) -> Dict[str, Any]:
    """Build calibration report rows across targets."""
    raw_tokens_avail = int(startup_metrics.get("tokens_avail") or 0)
    conc_tokens = int(startup_metrics.get("max_concurrency_tokens") or 0)
    conc_x = float(startup_metrics.get("max_concurrency_x") or 0.0)
    if conc_tokens > 0 and conc_x > 0.0:
        # For hybrid models (e.g., Kimi), this estimate is more reliable than
        # raw "GPU KV cache size" tokens because vLLM already folds model-specific
        # cache topology into the reported concurrency figure.
        tokens_avail = int(math.floor(conc_tokens * conc_x))
        tokens_source = "max_concurrency_estimate"
    else:
        tokens_avail = int(raw_tokens_avail)
        tokens_source = "gpu_kv_cache_size"
    block = int(startup_metrics.get("block_size_tokens") or 0)
    rows: List[Dict[str, Any]] = []
    for target in sorted({max(1, int(t)) for t in target_prompt_tokens}):
        row = evaluate_kv_capacity_target(
            target_prompt_tokens=int(target),
            max_new_tokens=int(max_new_tokens),
            max_num_seqs=int(max_num_seqs),
            tokens_avail=tokens_avail,
            block_size_tokens=block,
            headroom_ratio=float(headroom_ratio),
            headroom_min_blocks=int(headroom_min_blocks),
        )
        rec = recommend_kv_adjustment(
            row,
            gpu_memory_utilization=float(gpu_memory_utilization),
        )
        row["tokens_avail_source"] = str(tokens_source)
        row["tokens_avail_raw"] = int(raw_tokens_avail)
        row["tokens_avail_effective_for_gate"] = int(tokens_avail)
        row["recommended_action"] = str(rec.get("action") or "")
        row["recommended_message"] = str(rec.get("message") or "")
        row["recommended_suggested"] = dict(rec.get("suggested") or {})
        rows.append(row)

    pass_count = len([r for r in rows if bool(r.get("pass"))])
    fail_count = len(rows) - pass_count
    return {
        "ready": bool(startup_metrics.get("ready")),
        "startup_metrics": dict(startup_metrics),
        "inputs": {
            "max_new_tokens": int(max_new_tokens),
            "max_num_seqs": int(max(1, max_num_seqs)),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "headroom_ratio": float(headroom_ratio),
            "headroom_min_blocks": int(headroom_min_blocks),
            "tokens_avail_source": str(tokens_source),
            "tokens_avail_raw": int(raw_tokens_avail),
            "tokens_avail_effective_for_gate": int(tokens_avail),
        },
        "rows": rows,
        "summary": {
            "target_count": len(rows),
            "pass_count": int(pass_count),
            "fail_count": int(fail_count),
            "all_pass": bool(fail_count == 0),
        },
    }


def render_kv_calibration_markdown(calibration: Mapping[str, Any]) -> str:
    """Render a compact markdown summary for calibration artifacts."""
    startup = dict(calibration.get("startup_metrics") or {})
    inputs = dict(calibration.get("inputs") or {})
    rows = list(calibration.get("rows") or [])
    summary = dict(calibration.get("summary") or {})
    out: List[str] = []
    out.append("# KV Capacity Calibration")
    out.append("")
    out.append("- ready: `{}`".format(bool(calibration.get("ready"))))
    out.append("- tokens_avail: `{}`".format(int(startup.get("tokens_avail") or 0)))
    out.append("- tokens_avail_source: `{}`".format(str(inputs.get("tokens_avail_source") or "")))
    out.append("- tokens_avail_effective_for_gate: `{}`".format(int(inputs.get("tokens_avail_effective_for_gate") or 0)))
    out.append("- block_size_tokens: `{}`".format(int(startup.get("block_size_tokens") or 0)))
    out.append("- max_concurrency_tokens: `{}`".format(int(startup.get("max_concurrency_tokens") or 0)))
    out.append("- max_concurrency_x: `{}`".format(float(startup.get("max_concurrency_x") or 0.0)))
    out.append("- pass_count: `{}`".format(int(summary.get("pass_count") or 0)))
    out.append("- fail_count: `{}`".format(int(summary.get("fail_count") or 0)))
    out.append("")
    out.append(
        "| target_prompt_tokens | blocks_req | blocks_avail | margin_blocks | decision | recommended_action |"
    )
    out.append("|---:|---:|---:|---:|---|---|")
    for row in rows:
        out.append(
            "| {target} | {req} | {avail} | {margin} | {decision} | {action} |".format(
                target=int(row.get("target_prompt_tokens") or 0),
                req=int(row.get("blocks_required") or 0),
                avail=int(row.get("blocks_avail") or 0),
                margin=int(row.get("margin_blocks") or 0),
                decision=str(row.get("decision") or ""),
                action=str(row.get("recommended_action") or ""),
            )
        )
    out.append("")
    return "\n".join(out).strip() + "\n"
