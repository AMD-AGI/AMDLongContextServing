#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Build markdown report for HF long-context sweep runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from long_context_serving.benchmark_common import load_jsonl_dicts
from long_context_serving.host_env import render_environment_section


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path, *, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a JSON object when it exists, or return a default mapping."""
    if not path.exists():
        return dict(default or {})
    return _load_json(path)


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:  # noqa: BLE001
        return "-"


def _md_inline(text: str, limit: int = 240) -> str:
    """Normalize text for compact markdown table display."""
    value = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    return value.replace("|", "\\|")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _md_link(target: Path, base_dir: Path, label: str | None = None) -> str:
    """Return markdown link to target path relative to base_dir."""
    try:
        rel = os.path.relpath(str(target.resolve()), str(base_dir.resolve()))
    except Exception:  # noqa: BLE001
        rel = str(target)
    href = Path(rel).as_posix()
    text = label or href
    return f"[`{text}`]({href})"


def _resolve_existing_path(value: Any, run_dir: Path) -> Path | None:
    """Resolve path-like value from summary artifacts to an existing path."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.append(run_dir / p)
    for cand in candidates:
        try:
            if cand.exists():
                return cand.resolve()
        except Exception:  # noqa: BLE001
            continue
    return None


def _log_power_fit(points: List[tuple[float, float]]) -> Dict[str, float]:
    if len(points) < 2:
        return {"alpha": 0.0, "r2": 0.0}
    xs = [math.log(p[0]) for p in points]
    ys = [math.log(p[1]) for p in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"alpha": 0.0, "r2": 0.0}
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    alpha = sxy / sxx
    yhat = [my + alpha * (x - mx) for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"alpha": alpha, "r2": r2}


def _linreg(points: List[tuple[float, float]]) -> Dict[str, float]:
    if len(points) < 2:
        return {"a": 0.0, "b": 0.0, "r2": 0.0}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"a": my, "b": 0.0, "r2": 0.0}
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    yhat = [a + b * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"a": a, "b": b, "r2": r2}


def _fixed_power_r2(points: List[tuple[float, float]], power: float) -> Dict[str, float]:
    if len(points) < 2:
        return {"r2": 0.0, "factor_min": 0.0, "factor_max": 0.0}
    xs = [math.log(p[0]) for p in points]
    ys = [math.log(p[1]) for p in points]
    a = sum(y - power * x for x, y in zip(xs, ys)) / len(xs)
    yhat = [a + power * x for x in xs]
    my = sum(ys) / len(ys)
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    factors = [math.exp(y - yh) for y, yh in zip(ys, yhat)]
    return {"r2": r2, "factor_min": min(factors), "factor_max": max(factors)}


def build_report(run_dir: Path, report_md: Path) -> Path:
    run_meta = _load_json(run_dir / "run_meta.json")
    summary = _load_json(run_dir / "summary.json")
    sweep_payload = _load_json_if_exists(run_dir / "sweep_summary.json", default={"rows": []})
    prefix_capture_metrics = _load_json_if_exists(run_dir / "prefix_capture_metrics.json")
    results_rows = load_jsonl_dicts(run_dir / "results.jsonl")
    sweep_rows: List[Dict[str, Any]] = list(sweep_payload.get("rows") or [])
    skipped_rows = [r for r in results_rows if str(r.get("status", "")) == "skipped_max_context"]
    error_rows = [r for r in results_rows if str(r.get("status", "")) == "error"]
    prefill_progress_rows = load_jsonl_dicts(run_dir / "prefill_progress.jsonl")

    run_id = str(summary.get("run_id") or run_meta.get("run_id") or run_dir.name)
    model_id = str(summary.get("model_id") or run_meta.get("model_id") or "")
    plot_path = str((summary.get("artifacts") or {}).get("prompt_length_scaling_png") or "prompt_length_scaling.png")
    plot_name = Path(plot_path).name if plot_path else "prompt_length_scaling.png"

    lines: List[str] = []
    lines.append("# HF Long-Context Sweep Report")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Created (UTC): `{_now_iso()}`")
    lines.append(f"- Model: `{model_id}`")
    lines.append(f"- Prompt source: `{run_meta.get('prompt_source', '-')}`")
    prompt_shape = run_meta.get("prompt_shape", {}) if isinstance(run_meta, dict) else {}
    if isinstance(prompt_shape, dict) and prompt_shape:
        lines.append(
            f"- Prompt shape: id=`{prompt_shape.get('id', '-')}` source=`{prompt_shape.get('source', '-')}`"
        )
        shape_file = str(prompt_shape.get("file") or "").strip()
        if shape_file:
            lines.append(f"- Prompt shape file: `{shape_file}`")
    lines.append(f"- Sweep points: `{len(sweep_rows)}`")
    lines.append(f"- Fixed max new tokens: `{run_meta.get('max_new_tokens', '-')}`")
    lines.append(f"- Status counts: ok=`{summary.get('sample_count_ok', 0)}`, skipped=`{summary.get('sample_count_skipped', 0)}`, error=`{summary.get('sample_count_error', 0)}`")
    if prefix_capture_metrics:
        lines.append(
            f"- Prefix capture: ok=`{prefix_capture_metrics.get('ok', False)}` "
            f"state_saved=`{prefix_capture_metrics.get('prefix_state_saved', False)}` "
            f"tokens=`{prefix_capture_metrics.get('prefix_tokens_effective', '-')}`"
        )
        lines.append(f"- Prefix session id: `{prefix_capture_metrics.get('session_id', '-')}`")
    lines.append("")
    host_env_block = run_meta.get("host_env") if isinstance(run_meta, dict) else None
    if isinstance(host_env_block, dict):
        lines.extend(render_environment_section(host_env_block))
    lines.append("## Aggregate Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Prompt tokens (first row) | `{summary.get('prompt_tokens', '-')}` |")
    lines.append(f"| TTFT mean (ms) | `{_fmt(summary.get('ttft_ms_mean'), 2)}` |")
    lines.append(f"| TTFT p95 (ms) | `{_fmt(summary.get('ttft_ms_p95'), 2)}` |")
    lines.append(f"| Decode tok/s mean | `{_fmt(summary.get('gen_tokens_per_sec_mean'), 3)}` |")
    lines.append(f"| Decode ms/token mean | `{_fmt(summary.get('decode_ms_per_token_mean'), 4)}` |")
    lines.append(f"| K2FT samples | `{summary.get('k2ft_sample_count', 0)}` |")
    lines.append(f"| K2FT mean (ms) | `{_fmt(summary.get('k2ft_ms_mean'), 2)}` |")
    lines.append(f"| Prefill estimate mean (ms) | `{_fmt(summary.get('prefill_est_ms_mean'), 2)}` |")
    lines.append("")
    lines.append("## Per-Length Results")
    lines.append("")
    lines.append("| Prompt Tokens (req) | Prompt Tokens (eff) | TTFT Mean (ms) | TTFT P95 (ms) | Decode tok/s Mean | Decode ms/token Mean | K2FT Mean (ms) | Prefill Est Mean (ms) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sweep_rows:
        lines.append(
            f"| {row.get('target_prompt_tokens_requested', '-')} "
            f"| {row.get('target_prompt_tokens_effective', '-')} "
            f"| {_fmt(row.get('ttft_ms_mean'), 2)} "
            f"| {_fmt(row.get('ttft_ms_p95'), 2)} "
            f"| {_fmt(row.get('gen_tokens_per_sec_mean'), 3)} "
            f"| {_fmt(row.get('decode_ms_per_token_mean'), 4)} "
            f"| {_fmt(row.get('k2ft_ms_mean'), 2)} "
            f"| {_fmt(row.get('prefill_est_ms_mean'), 2)} |"
        )
    lines.append("")
    k2ft_rows = [
        r
        for r in results_rows
        if str(r.get("status", "")) == "ok" and bool(r.get("k2ft_enabled"))
    ]
    if k2ft_rows:
        lines.append("## K2FT (KV-ready → first token)")
        lines.append("")
        lines.append(
            f"- Method: `{run_meta.get('k2ft_method', 'two_stage_cold_hot_ttft_delta') if isinstance(run_meta, dict) else 'two_stage_cold_hot_ttft_delta'}`"
        )
        lines.append(
            f"- Probe rule: `L_probe = max({int(run_meta.get('k2ft_min_probe_tokens', 1024))}, L_current - {int(run_meta.get('k2ft_delta_tokens', 256))})`"
        )
        lines.append(
            f"- Configured probe runs: `{int(run_meta.get('k2ft_runs', 2))}` (cold then hot)."
        )
        lines.append("")
        lines.append("| Prompt(req) | Prompt(eff) | L_probe | Cold TTFT (ms) | Hot TTFT (ms) | K2FT (ms) | Prefill est (ms) | Confidence | Status | Response File |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
        ordered_k2ft = sorted(k2ft_rows, key=lambda r: int(r.get("target_prompt_tokens_requested") or 0))
        for row in ordered_k2ft:
            response_file = str(row.get("generated_text_path") or "")
            response_path = Path(response_file) if response_file else Path("")
            run_local = run_dir / "generated_text" / response_path.name
            response_target = response_path if response_path.exists() else run_local if run_local.exists() else None
            if response_target:
                response_md = _md_link(response_target, report_md.parent, label=f"generated_text/{response_target.name}")
            else:
                response_md = "`-`"
            lines.append(
                f"| {row.get('target_prompt_tokens_requested', '-')} "
                f"| {row.get('target_prompt_tokens_effective', '-')} "
                f"| {row.get('k2ft_probe_tokens', '-')} "
                f"| {_fmt(row.get('k2ft_cold_ttft_ms'), 2)} "
                f"| {_fmt(row.get('k2ft_hot_ttft_ms'), 2)} "
                f"| {_fmt(row.get('k2ft_ms'), 2)} "
                f"| {_fmt(row.get('prefill_est_ms'), 2)} "
                f"| `{row.get('k2ft_cache_hit_confidence', '-')}` "
                f"| `{row.get('k2ft_status', '-')}` "
                f"| {response_md} |"
            )
        lines.append("")
    lines.append("## Prefill Progress")
    lines.append("")
    if not prefill_progress_rows:
        lines.append("- `prefill progress not captured`")
        lines.append("")
    else:
        agg: Dict[tuple[int, int, int], Dict[str, Any]] = {}
        for prow in prefill_progress_rows:
            req = int(prow.get("target_prompt_tokens_requested") or 0)
            eff = int(prow.get("target_prompt_tokens_effective") or 0)
            it = int(prow.get("iteration") or 0)
            key = (req, eff, it)
            if key not in agg:
                agg[key] = {
                    "requested": req,
                    "effective": eff,
                    "iteration": it,
                    "samples": 0,
                    "first_ts": str(prow.get("timestamp_utc") or ""),
                    "last_ts": str(prow.get("timestamp_utc") or ""),
                    "max_pct": 0.0,
                    "last_done": 0,
                    "last_total": 0,
                    "last_status": str(prow.get("status") or ""),
                    "last_note": str(prow.get("note") or ""),
                }
            row = agg[key]
            row["samples"] = int(row["samples"]) + 1
            row["last_ts"] = str(prow.get("timestamp_utc") or row["last_ts"])
            pct = float(prow.get("prefill_pct") or 0.0)
            row["max_pct"] = max(float(row["max_pct"]), pct)
            row["last_done"] = int(prow.get("prefill_done_tokens") or 0)
            row["last_total"] = int(prow.get("prefill_total_tokens") or 0)
            row["last_status"] = str(prow.get("status") or row["last_status"])
            row["last_note"] = str(prow.get("note") or row["last_note"])

        lines.append("| Prompt(req) | Prompt(eff) | Iter | Samples | Last Pointer | Max % | Last Status | First Seen (UTC) | Last Seen (UTC) | Note |")
        lines.append("|---:|---:|---:|---:|---|---:|---|---|---|---|")
        ordered_prefill = sorted(agg.values(), key=lambda r: (int(r["requested"]), int(r["iteration"])))
        for row in ordered_prefill:
            lines.append(
                f"| {row['requested']} "
                f"| {row['effective']} "
                f"| {row['iteration']} "
                f"| {row['samples']} "
                f"| `{int(row['last_done'])}/{int(row['last_total'])}` "
                f"| {_fmt(row['max_pct'], 6)} "
                f"| `{row['last_status']}` "
                f"| `{row['first_ts']}` "
                f"| `{row['last_ts']}` "
                f"| `{_md_inline(str(row['last_note']), limit=120)}` |"
            )
        lines.append("")
    scaling_points: List[tuple[float, float]] = []
    for row in sweep_rows:
        n = float(row.get("target_prompt_tokens_requested") or 0)
        ttft_s = float(row.get("ttft_ms_mean") or 0.0) / 1000.0
        if n > 0 and ttft_s > 0:
            scaling_points.append((n, ttft_s))
    if scaling_points:
        tail_points = [p for p in scaling_points if p[0] >= 65536]
        all_fit = _log_power_fit(scaling_points)
        all_n2 = _fixed_power_r2(scaling_points, 2.0)
        lines.append("## Scaling Fit")
        lines.append("")
        lines.append("| Region | Points | Best-fit exponent α in TTFT ∝ n^α | R² (best fit) | R² (fixed n²) | n² factor range |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(
            f"| All | {len(scaling_points)} | {_fmt(all_fit.get('alpha'), 3)} | {_fmt(all_fit.get('r2'), 3)} | {_fmt(all_n2.get('r2'), 3)} | "
            f"{_fmt(all_n2.get('factor_min'), 3)}x..{_fmt(all_n2.get('factor_max'), 3)}x |"
        )
        if len(tail_points) >= 2:
            tail_fit = _log_power_fit(tail_points)
            tail_n2 = _fixed_power_r2(tail_points, 2.0)
            lines.append(
                f"| >=64K | {len(tail_points)} | {_fmt(tail_fit.get('alpha'), 3)} | {_fmt(tail_fit.get('r2'), 3)} | {_fmt(tail_n2.get('r2'), 3)} | "
                f"{_fmt(tail_n2.get('factor_min'), 3)}x..{_fmt(tail_n2.get('factor_max'), 3)}x |"
            )
        lines.append("")
        decode_tail = [
            (
                float(row.get("target_prompt_tokens_requested") or 0),
                float(row.get("gen_tokens_per_sec_mean") or 0),
            )
            for row in sweep_rows
            if float(row.get("target_prompt_tokens_requested") or 0) >= 8192
            and float(row.get("gen_tokens_per_sec_mean") or 0) > 0
        ]
        if len(decode_tail) >= 2:
            recip_points = [(n, 1.0 / tok_s) for n, tok_s in decode_tail]
            reg = _linreg(recip_points)
            lines.append("## Decode Decline Fit")
            lines.append("")
            lines.append("- Region: `prompt_tokens >= 8192`")
            lines.append(
                f"- Linearized fit: `1/tok_s = a + b*n`, with `a={_fmt(reg.get('a'), 8)}`, `b={_fmt(reg.get('b'), 12)}`, `R²={_fmt(reg.get('r2'), 6)}`"
            )
            b = float(reg.get("b") or 0.0)
            a = float(reg.get("a") or 0.0)
            if b > 0:
                c = 1.0 / b
                k = a * c
                lines.append(
                    f"- Equivalent form: `tok_s ≈ c / (n + k)` with `c={_fmt(c, 3)}`, `k={_fmt(k, 3)}`"
                )
            lines.append("")
    if skipped_rows:
        lines.append("## Skipped Points")
        lines.append("")
        lines.append("| Prompt Tokens (req) | Reason |")
        lines.append("|---:|---|")
        for row in skipped_rows:
            lines.append(f"| {row.get('target_prompt_tokens_requested', '-')} | `{row.get('error', '-')}` |")
        lines.append("")
    if error_rows:
        lines.append("## Error Points")
        lines.append("")
        lines.append("| Prompt Tokens (req) | Error |")
        lines.append("|---:|---|")
        for row in error_rows:
            lines.append(f"| {row.get('target_prompt_tokens_requested', '-')} | `{row.get('error', '-')}` |")
        lines.append("")
    ok_rows = [r for r in results_rows if str(r.get("status", "")) == "ok"]
    if ok_rows:
        lines.append("## Responses")
        lines.append("")
        lines.append("| Prompt Tokens (req) | Prompt Tokens (eff) | Completion Tokens | Finish Reason | Response File | Response Preview |")
        lines.append("|---:|---:|---:|---|---|---|")
        ordered_ok = sorted(
            ok_rows,
            key=lambda r: int(r.get("target_prompt_tokens_requested") or 0),
        )
        for row in ordered_ok:
            response_text = ""
            response_file = str(row.get("generated_text_path") or "")
            response_path = Path(response_file) if response_file else Path("")
            if response_path and response_path.exists():
                try:
                    response_text = response_path.read_text(encoding="utf-8")
                except Exception:  # noqa: BLE001
                    response_text = ""
            run_local = run_dir / "generated_text" / response_path.name
            if not response_text and run_local.exists():
                try:
                    response_text = run_local.read_text(encoding="utf-8")
                except Exception:  # noqa: BLE001
                    response_text = ""
            response_display = f"generated_text/{response_path.name}" if response_path.name else (response_file or "-")
            response_target = response_path if response_path.exists() else run_local if run_local.exists() else None
            if response_target:
                response_display_md = _md_link(
                    response_target,
                    report_md.parent,
                    label=f"generated_text/{response_target.name}",
                )
            else:
                response_display_md = f"`{response_display}`"
            lines.append(
                f"| {row.get('target_prompt_tokens_requested', '-')} "
                f"| {row.get('target_prompt_tokens_effective', '-')} "
                f"| {row.get('completion_tokens', '-')} "
                f"| `{row.get('finish_reason', '-')}` "
                f"| {response_display_md} "
                f"| {_md_inline(response_text)} |"
            )
        lines.append("")
    lines.append("## Chart")
    lines.append("")
    lines.append(f"![Prompt Length Scaling]({plot_name})")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in sorted((summary.get("artifacts") or {}).items()):
        resolved = _resolve_existing_path(value, run_dir)
        if resolved is not None:
            lines.append(f"- `{key}`: {_md_link(resolved, report_md.parent)}")
        else:
            lines.append(f"- `{key}`: `{value}`")
    if (
        (run_dir / "prefill_progress.jsonl").exists()
        and "prefill_progress_jsonl" not in (summary.get("artifacts") or {})
    ):
        lines.append(
            f"- `prefill_progress_jsonl`: {_md_link(run_dir / 'prefill_progress.jsonl', report_md.parent)}"
        )
    lines.append("")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_md


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HF long-context markdown report")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-md", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    report_md = Path(args.report_md).expanduser().resolve() if args.report_md else run_dir / "report.md"
    out = build_report(run_dir, report_md)
    print(json.dumps({"ok": True, "report_md": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
