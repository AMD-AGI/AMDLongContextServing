# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import json

from long_context_serving.substrate_utils import (
    DEFAULT_CODE_DOC_EXTS,
    DEFAULT_DATA_EXTS,
    load_file_list,
)


CONTEXT_MARKUP_FILE_ONLY_V1 = "file_only_v1"
CONTEXT_MARKUP_REPO_FILE_MARKERS_V1 = "repo_file_markers_v1"

_LOCKFILE_BASENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "cargo.lock",
    "poetry.lock",
    "pipfile.lock",
    "uv.lock",
}

_HIGH_SIGNAL_BASENAMES = {
    "readme",
    "readme.md",
    "readme.rst",
    "contributing.md",
    "changelog.md",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "makefile",
    "cargo.toml",
    "tsconfig.json",
    "dockerfile",
}

_REPO_OVERVIEW_DOC_BASENAMES = {
    "readme",
    "readme.md",
    "readme.rst",
    "contributing.md",
    "changelog.md",
}

_ENTRYPOINT_BASENAMES = {
    "__init__.py",
    "main.py",
    "main.ts",
    "main.js",
    "index.py",
    "index.ts",
    "index.js",
    "app.py",
    "cli.py",
}

_LOW_SIGNAL_DIR_PARTS = {
    "dist",
    "build",
    "out",
    "generated",
    "gen",
    "vendor",
    "third_party",
    "node_modules",
}

_LOW_SIGNAL_CONTENT_DIR_PARTS = {
    "examples",
    "gallery",
    "tutorial",
    "tutorials",
    "user_guide",
    "getting_started",
}

_LOW_SIGNAL_GENERATED_DIR_PARTS = {
    "_sri",
}


def timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def quantile(values: Iterable[float], q: float) -> float:
    nums = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not nums:
        return 0.0
    if len(nums) == 1:
        return nums[0]
    p = max(0.0, min(1.0, float(q)))
    idx = p * (len(nums) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return nums[lo]
    frac = idx - lo
    return nums[lo] * (1.0 - frac) + nums[hi] * frac


def append_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def load_jsonl_dicts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def parse_token_list(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        vals.append(max(1, int(token)))
    return sorted(set(vals))


def parse_pow2_ki_targets(min_ki: int, max_ki: int, million_point: int) -> List[int]:
    lower = max(1, int(min_ki))
    upper = max(lower, int(max_ki))
    vals: List[int] = []
    cur = 1
    while cur < lower:
        cur *= 2
    while cur <= upper:
        vals.append(cur * 1024)
        cur *= 2
    mp = int(million_point or 0)
    if upper >= 1024 and mp > 0:
        vals = [v for v in vals if v != 1024 * 1024]
        vals.append(mp)
    return sorted(set(vals))


def parse_sweep_targets(args: Any, *, default_target: int | None = None) -> List[int]:
    explicit = parse_token_list(str(getattr(args, "sweep_prompt_tokens", "") or ""))
    if explicit:
        return explicit
    if bool(getattr(args, "sweep_pow2_ki", False)):
        return parse_pow2_ki_targets(
            min_ki=int(getattr(args, "sweep_min_ki", 1)),
            max_ki=int(getattr(args, "sweep_max_ki", 1024)),
            million_point=int(getattr(args, "sweep_million_point", 1010000) or 0),
        )
    if default_target is None:
        return []
    return [max(1, int(default_target))]


def format_k_tokens(tokens: int) -> str:
    if tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return str(tokens)


def write_sweep_plot_png(
    rows: List[Dict[str, Any]],
    png_path: Path,
    model_id: str,
    max_new_tokens: int,
    title_prefix: str,
) -> Dict[str, Any]:
    payload = {"ok": False, "png_path": str(png_path), "error": ""}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        payload["error"] = f"matplotlib unavailable: {exc}"
        return payload

    if not rows:
        payload["error"] = "no rows"
        return payload

    xs = [int(r["target_prompt_tokens_effective"]) for r in rows]
    ttft = [as_float(r["ttft_ms_mean"]) / 1000.0 for r in rows]
    tok_s = [as_float(r["gen_tokens_per_sec_mean"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(xs, ttft, marker="o", linewidth=2.0, color="#56b4e9")
    ax1.set_ylabel("TTFT (s)")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(f"{title_prefix} Long-Context Scaling: {model_id} (max_new_tokens={max_new_tokens})")

    ax2.plot(xs, tok_s, marker="o", linewidth=2.0, color="#e69f00")
    ax2.set_ylabel("Decode throughput (tok/s)")
    ax2.set_xlabel("Prompt length (tokens, log2 scale)")
    ax2.grid(True, alpha=0.25)

    if len(xs) > 1:
        ax2.set_xscale("log", base=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([format_k_tokens(x) for x in xs], rotation=0)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)
    payload["ok"] = True
    return payload


def resolve_include_exts(
    raw: str,
    *,
    default_code_exts: Sequence[str] = DEFAULT_CODE_DOC_EXTS,
    default_data_exts: Sequence[str] = DEFAULT_DATA_EXTS,
) -> List[str]:
    parts = [p.strip().lower() for p in str(raw or "").split(",") if p.strip()]
    if parts:
        return sorted({p if p.startswith(".") else f".{p}" for p in parts})
    return sorted({*default_code_exts, *default_data_exts})


def _is_lockfile_path(rel_path: Path) -> bool:
    """Return whether a substrate path is a dependency lockfile."""

    name = str(rel_path.name or "").strip().lower()
    return bool(name) and (name in _LOCKFILE_BASENAMES or name.endswith(".lock"))


def _substrate_path_priority(rel_path: Path) -> Tuple[int, int, int, str]:
    """Assign a stable priority so repo-report prompts end on higher-signal files.

    Lower tuples sort earlier and are therefore consumed first by the prompt
    packer. The ordering intentionally prefers docs, manifests, workflows,
    build scripts, and common entrypoints over noisy generated or internal
    implementation files.
    """

    parts = [str(part).strip().lower() for part in rel_path.parts]
    name = str(rel_path.name or "").strip().lower()
    path_text = rel_path.as_posix().lower()
    is_top_level = len(parts) <= 2
    has_low_signal_dir = any(part in _LOW_SIGNAL_DIR_PARTS for part in parts[:-1])
    has_generated_metadata_dir = any(
        part in _LOW_SIGNAL_GENERATED_DIR_PARTS for part in parts[:-1]
    )
    has_low_signal_content_dir = any(
        part in _LOW_SIGNAL_CONTENT_DIR_PARTS for part in parts[:-1]
    )
    has_docs_dir = "docs" in parts[:-1]
    is_repo_overview_doc = (
        name in _REPO_OVERVIEW_DOC_BASENAMES
        and not has_docs_dir
        and not has_low_signal_content_dir
    )
    is_manifest = name in _HIGH_SIGNAL_BASENAMES
    is_top_manifest = bool(is_manifest and len(parts) <= 3)
    is_nested_manifest = bool(is_manifest and not is_top_manifest)
    is_workflow = ".github" in parts or "workflows" in parts
    is_build = (
        any(part in {"scripts", "make", "cmake", "docker"} for part in parts[:-1])
        or path_text.endswith("/action.yml")
        or path_text.endswith("/action.yaml")
    )
    is_entrypoint = name in _ENTRYPOINT_BASENAMES
    is_test = any(part in {"tests", "test"} for part in parts[:-1])
    is_source = any(part in {"src", "lib"} for part in parts[:-1])

    if has_low_signal_content_dir or has_generated_metadata_dir:
        group = 7
    elif is_repo_overview_doc:
        group = 0
    elif is_top_manifest:
        group = 1
    elif is_workflow or ((is_build and not has_docs_dir) or (is_top_level and is_entrypoint)):
        group = 2
    elif is_top_level:
        group = 3
    elif is_entrypoint:
        group = 4
    elif is_source:
        group = 5
    elif is_test:
        group = 6
    elif has_docs_dir or is_nested_manifest:
        group = 7
    elif has_low_signal_dir:
        group = 7
    else:
        group = 7
    return (int(group), 0 if is_top_level else 1, len(parts), rel_path.as_posix())


def _repo_local_substrate_sort_key(rel_path: Path) -> Tuple[str, int, int, int, str]:
    """Keep prompts within one repo while preferring higher-signal files first."""

    repo_name = str(rel_path.parts[0] or "") if len(rel_path.parts) >= 2 else ""
    priority = _substrate_path_priority(rel_path)
    return (repo_name, *priority)


def _tokenize_text(tokenizer: Any, text: str) -> List[int]:
    """Tokenize text without special tokens and normalize to ints."""

    payload = tokenizer(str(text), add_special_tokens=False)
    raw_ids = payload.get("input_ids") if isinstance(payload, Mapping) else None
    return [int(token_id) for token_id in (raw_ids or [])]


def _truncate_block_with_footer(
    *,
    tokenizer: Any,
    block_text: str,
    rel_posix: str,
    remaining_tokens: int,
) -> Tuple[str, List[int], Dict[str, Any]]:
    """Fit a partial substrate block into the remaining token budget.

    The long-context repo-grounded prompts can otherwise end in the middle of a
    source line, which encourages the model to continue code rather than switch
    into the task/reporting mode that follows ``</CONTEXT>``. When we must clip
    a file block, prefer a line-aligned prefix plus an explicit truncation
    footer.
    """

    budget = max(0, int(remaining_tokens))
    if budget <= 0:
        return "", [], {
            "truncated": True,
            "truncation_mode": "empty",
            "truncation_footer": "",
            "line_aligned": True,
        }

    footer_candidates = [
        f"\n[TRUNCATED FILE {rel_posix}]\n",
        "\n[TRUNCATED FILE]\n",
        "\n[TRUNCATED]\n",
    ]
    chosen_footer = ""
    chosen_footer_ids: List[int] = []
    for footer in footer_candidates:
        footer_ids = _tokenize_text(tokenizer, footer)
        if len(footer_ids) < budget:
            chosen_footer = footer
            chosen_footer_ids = footer_ids
            break

    if not chosen_footer_ids:
        clipped_ids = _tokenize_text(tokenizer, block_text)[:budget]
        return block_text, clipped_ids, {
            "truncated": True,
            "truncation_mode": "token_slice",
            "truncation_footer": "",
            "line_aligned": False,
        }

    prefix_budget = max(0, budget - len(chosen_footer_ids))
    lines = block_text.splitlines(keepends=True)
    prefix_parts: List[str] = []
    prefix_ids: List[int] = []
    for line in lines:
        candidate_text = "".join(prefix_parts) + line
        candidate_ids = _tokenize_text(tokenizer, candidate_text)
        if len(candidate_ids) > prefix_budget:
            break
        prefix_parts.append(line)
        prefix_ids = candidate_ids

    if prefix_parts:
        fitted_text = "".join(prefix_parts) + chosen_footer
        fitted_ids = _tokenize_text(tokenizer, fitted_text)
        while prefix_parts and len(fitted_ids) > budget:
            prefix_parts.pop()
            fitted_text = "".join(prefix_parts) + chosen_footer
            fitted_ids = _tokenize_text(tokenizer, fitted_text)
        return fitted_text, fitted_ids, {
            "truncated": True,
            "truncation_mode": "line_footer",
            "truncation_footer": chosen_footer.strip(),
            "line_aligned": True,
        }

    clipped_ids = _tokenize_text(tokenizer, block_text)[:prefix_budget] + chosen_footer_ids
    return block_text, clipped_ids[:budget], {
        "truncated": True,
        "truncation_mode": "token_footer",
        "truncation_footer": chosen_footer.strip(),
        "line_aligned": False,
    }


def build_substrate_prompt_seed_ids(
    *,
    tokenizer: Any,
    substrate_root: Path,
    target_tokens: int,
    include_exts: List[str],
    max_files: int,
    max_chars_per_file: int,
    char_budget_multiplier: float,
    allow_unlimited_files: bool,
    allow_unlimited_chars: bool,
    context_markup: str = CONTEXT_MARKUP_FILE_ONLY_V1,
) -> Tuple[List[int], Dict[str, Any]]:
    """Build substrate prompt token IDs and metadata.

    Args:
        tokenizer: Tokenizer used to serialize each substrate block.
        substrate_root: Root directory containing ordered substrate files.
        target_tokens: Maximum number of prompt-seed tokens to keep.
        include_exts: File extensions eligible for inclusion.
        max_files: Maximum file count, unless unlimited files are enabled.
        max_chars_per_file: Per-file character limit, unless unlimited chars
            are enabled.
        char_budget_multiplier: Approximate char budget per target token.
        allow_unlimited_files: Whether ``max_files <= 0`` means no file cap.
        allow_unlimited_chars: Whether ``max_chars_per_file <= 0`` means no
            per-file char cap.
        context_markup: Context serialization format. ``file_only_v1`` emits
            only ``[FILE ...]`` tags. ``repo_file_markers_v1`` additionally
            emits ``[REPO ...]`` boundaries when the top-level repo changes.

    Returns:
        Tuple[List[int], Dict[str, Any]]: Extracted token IDs plus prompt
        construction metadata.
    """
    if not substrate_root.exists():
        raise SystemExit(f"Substrate root not found: {substrate_root}")
    context_markup_norm = str(context_markup or CONTEXT_MARKUP_FILE_ONLY_V1).strip()
    if context_markup_norm not in {
        CONTEXT_MARKUP_FILE_ONLY_V1,
        CONTEXT_MARKUP_REPO_FILE_MARKERS_V1,
    }:
        raise SystemExit(f"Unsupported context markup: {context_markup_norm}")
    rel_paths = load_file_list(substrate_root, include_exts=include_exts)
    if not rel_paths:
        raise SystemExit(
            f"No substrate files found under {substrate_root} for extensions: {','.join(include_exts)}"
        )
    eligible_paths = [rel for rel in rel_paths if not _is_lockfile_path(rel)]
    # Keep repo locality so a prompt budget tends to stay within one repository,
    # but prefer higher-signal files inside that repo so long-context sweeps do
    # not end on repetitive implementation tails or generated constants.
    ranked_paths = sorted(eligible_paths, key=_repo_local_substrate_sort_key)
    if not ranked_paths:
        raise SystemExit(
            "Unable to build substrate prompt text (all eligible files were filtered as lockfiles)."
        )

    used_paths: List[str] = []
    used_repos: List[str] = []
    bytes_read = 0
    budget_multiplier = max(1.0, float(char_budget_multiplier))
    char_budget = max(16384, int(max(1, target_tokens) * budget_multiplier))
    budget_hit = False
    token_budget_hit = False
    raw_token_count = 0

    file_limit_raw = int(max_files)
    if allow_unlimited_files and file_limit_raw <= 0:
        selected_paths = ranked_paths
        file_limit = file_limit_raw
    else:
        file_limit = max(1, file_limit_raw)
        selected_paths = ranked_paths[:file_limit]

    raw_max_chars = int(max_chars_per_file)
    per_file_unlimited = bool(allow_unlimited_chars and raw_max_chars <= 0)
    per_file_chars = 0 if per_file_unlimited else max(1024, raw_max_chars)

    extracted_ids: List[int] = []
    truncated_file_path = ""
    partial_file_truncation_mode = "none"
    partial_file_truncation_footer = ""
    partial_file_line_aligned = True
    current_repo = ""
    for idx, rel in enumerate(selected_paths):
        abs_path = substrate_root / rel
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not text:
            continue
        if not per_file_unlimited:
            text = text[:per_file_chars]
        rel_posix = rel.as_posix()
        repo_name = rel.parts[0] if len(rel.parts) >= 2 else ""
        prefix = "" if idx == 0 else "\n\n"
        if (
            context_markup_norm == CONTEXT_MARKUP_REPO_FILE_MARKERS_V1
            and repo_name
            and repo_name != current_repo
        ):
            block = f"{prefix}[REPO {repo_name}]\n[FILE {rel_posix}]\n{text}"
            current_repo = str(repo_name)
            if repo_name not in used_repos:
                used_repos.append(str(repo_name))
        else:
            block = f"{prefix}[FILE {rel_posix}]\n{text}"
            if repo_name and repo_name not in used_repos:
                used_repos.append(str(repo_name))
        block_ids = _tokenize_text(tokenizer, block)
        raw_token_count += int(len(block_ids))
        if target_tokens <= 0:
            extracted_ids.extend(block_ids)
        else:
            remaining = int(target_tokens) - len(extracted_ids)
            if remaining > 0:
                if len(block_ids) <= remaining:
                    extracted_ids.extend(block_ids)
                else:
                    fitted_text, fitted_ids, truncation_meta = _truncate_block_with_footer(
                        tokenizer=tokenizer,
                        block_text=block,
                        rel_posix=rel_posix,
                        remaining_tokens=remaining,
                    )
                    extracted_ids.extend(fitted_ids[:remaining])
                    truncated_file_path = str(rel_posix)
                    partial_file_truncation_mode = str(
                        truncation_meta.get("truncation_mode") or "token_slice"
                    )
                    partial_file_truncation_footer = str(
                        truncation_meta.get("truncation_footer") or ""
                    )
                    partial_file_line_aligned = bool(
                        truncation_meta.get("line_aligned", False)
                    )
                    bytes_read += len(fitted_text)
                    used_paths.append(rel_posix)
                    if bytes_read >= char_budget:
                        budget_hit = True
                    token_budget_hit = True
                    break
        used_paths.append(rel_posix)
        bytes_read += len(block)
        if bytes_read >= char_budget:
            budget_hit = True
            break
        if target_tokens > 0 and len(extracted_ids) >= int(target_tokens):
            token_budget_hit = True
            break

    if not extracted_ids:
        raise SystemExit("Unable to build substrate prompt text (all selected files were unreadable/empty).")
    meta = {
        "substrate_root": str(substrate_root),
        "include_exts": include_exts,
        "available_file_count": len(rel_paths),
        "filtered_lockfile_count": int(len(rel_paths) - len(ranked_paths)),
        "selected_file_count": len(used_paths),
        "selected_repo_count": len(used_repos),
        "selected_repos": list(used_repos),
        "context_markup": context_markup_norm,
        "selected_files_head": used_paths[:16],
        "selected_files_tail": used_paths[-8:] if len(used_paths) > 8 else used_paths,
        "char_budget": char_budget,
        "char_budget_multiplier": budget_multiplier,
        "bytes_read": bytes_read,
        "budget_hit": bool(budget_hit),
        "token_budget_hit": bool(token_budget_hit),
        "raw_corpus_tokens": int(raw_token_count),
        "extracted_prompt_seed_tokens": len(extracted_ids),
        "target_prompt_tokens": int(target_tokens),
        "truncated_file_path": truncated_file_path,
        "partial_file_truncation_mode": partial_file_truncation_mode,
        "partial_file_truncation_footer": partial_file_truncation_footer,
        "partial_file_line_aligned": bool(partial_file_line_aligned),
        "file_limit": file_limit,
        "file_limit_effective": len(selected_paths),
        "max_chars_per_file": 0 if per_file_unlimited else per_file_chars,
        "max_chars_per_file_unlimited": bool(per_file_unlimited),
    }
    return extracted_ids, meta


def build_substrate_prompt_seed_text(
    *,
    tokenizer: Any,
    substrate_root: Path,
    target_tokens: int,
    include_exts: List[str],
    max_files: int,
    max_chars_per_file: int,
    char_budget_multiplier: float,
    allow_unlimited_files: bool,
    allow_unlimited_chars: bool,
    context_markup: str = CONTEXT_MARKUP_FILE_ONLY_V1,
) -> Tuple[str, Dict[str, Any]]:
    """Build prompt seed text and metadata from substrate files."""
    extracted_ids, meta = build_substrate_prompt_seed_ids(
        tokenizer=tokenizer,
        substrate_root=substrate_root,
        target_tokens=target_tokens,
        include_exts=include_exts,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
        char_budget_multiplier=char_budget_multiplier,
        allow_unlimited_files=allow_unlimited_files,
        allow_unlimited_chars=allow_unlimited_chars,
        context_markup=context_markup,
    )
    prompt_seed = tokenizer.decode(
        extracted_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return prompt_seed, meta


def build_substrate_prompt_seed_from_args(
    *,
    tokenizer: Any,
    args: Any,
    target_tokens: int,
    include_exts: List[str],
    char_budget_multiplier: float,
    allow_unlimited_files: bool,
    allow_unlimited_chars: bool,
    decode_to_text: bool,
    context_markup: str = CONTEXT_MARKUP_FILE_ONLY_V1,
) -> Tuple[Any, Dict[str, Any]]:
    """Build substrate prompt seed directly from parsed CLI args."""
    seed_ids, meta = build_substrate_prompt_seed_ids(
        tokenizer=tokenizer,
        substrate_root=Path(str(getattr(args, "substrate_root"))).expanduser().resolve(),
        target_tokens=target_tokens,
        include_exts=include_exts,
        max_files=int(getattr(args, "substrate_max_files")),
        max_chars_per_file=int(getattr(args, "substrate_max_chars_per_file")),
        char_budget_multiplier=float(char_budget_multiplier),
        allow_unlimited_files=bool(allow_unlimited_files),
        allow_unlimited_chars=bool(allow_unlimited_chars),
        context_markup=str(context_markup or CONTEXT_MARKUP_FILE_ONLY_V1),
    )
    if not decode_to_text:
        return seed_ids, meta
    prompt_seed = tokenizer.decode(
        seed_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return prompt_seed, meta


def make_base_measurement_row(
    *,
    run_id: str,
    target_index: int,
    target_prompt_tokens_requested: int,
    target_prompt_tokens_effective: int,
    iteration: int,
    model_id: str,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "target_index": int(target_index),
        "target_prompt_tokens_requested": int(target_prompt_tokens_requested),
        "target_prompt_tokens_effective": int(target_prompt_tokens_effective),
        "iteration": int(iteration),
        "timestamp_utc": now_iso(),
        "model_id": str(model_id),
        "status": "ok",
        "error": "",
        "prompt_tokens": int(target_prompt_tokens_effective),
        "completion_tokens": 0,
        "ttft_ms": 0.0,
        "decode_ms": 0.0,
        "total_ms": 0.0,
        "decode_ms_per_token": 0.0,
        "gen_tokens_per_sec": 0.0,
        "total_tokens_per_sec": 0.0,
    }


def make_iteration_measurement_row(
    *,
    run_id: str,
    target_idx: int,
    target_requested: int,
    target_effective: int,
    iter_idx: int,
    model_id: str,
) -> Dict[str, Any]:
    """Convenience wrapper around ``make_base_measurement_row`` for loops."""
    return make_base_measurement_row(
        run_id=run_id,
        target_index=int(target_idx) + 1,
        target_prompt_tokens_requested=int(target_requested),
        target_prompt_tokens_effective=int(target_effective),
        iteration=int(iter_idx) + 1,
        model_id=str(model_id),
    )


def build_basic_sweep_summary_rows(ok_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_target: Dict[int, List[Dict[str, Any]]] = {}
    for row in ok_rows:
        tgt = int(row.get("target_prompt_tokens_effective", 0) or 0)
        by_target.setdefault(tgt, []).append(row)

    out: List[Dict[str, Any]] = []
    for tgt in sorted(by_target):
        vals = by_target[tgt]
        ttft_t = [as_float(r.get("ttft_ms")) for r in vals]
        tok_t = [as_float(r.get("gen_tokens_per_sec")) for r in vals]
        dec_t = [as_float(r.get("decode_ms_per_token")) for r in vals]
        req = int(vals[0].get("target_prompt_tokens_requested", tgt))
        out.append(
            {
                "target_prompt_tokens_requested": req,
                "target_prompt_tokens_effective": int(tgt),
                "sample_count": len(vals),
                "ttft_ms_mean": round(mean(ttft_t), 6) if ttft_t else 0.0,
                "ttft_ms_median": round(median(ttft_t), 6) if ttft_t else 0.0,
                "ttft_ms_p95": round(quantile(ttft_t, 0.95), 6) if ttft_t else 0.0,
                "gen_tokens_per_sec_mean": round(mean(tok_t), 6) if tok_t else 0.0,
                "gen_tokens_per_sec_median": round(median(tok_t), 6) if tok_t else 0.0,
                "decode_ms_per_token_mean": round(mean(dec_t), 6) if dec_t else 0.0,
            }
        )
    return out


def build_basic_summary(rows: List[Dict[str, Any]], ok_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ttft_vals = [as_float(r.get("ttft_ms")) for r in ok_rows]
    tok_vals = [as_float(r.get("gen_tokens_per_sec")) for r in ok_rows]
    total_tok_vals = [as_float(r.get("total_tokens_per_sec")) for r in ok_rows]
    decode_msp_vals = [as_float(r.get("decode_ms_per_token")) for r in ok_rows]
    completion_vals = [as_float(r.get("completion_tokens")) for r in ok_rows]
    return {
        "sample_count": len(rows),
        "sample_count_ok": len(ok_rows),
        "sample_count_skipped": len([r for r in rows if str(r.get("status")) == "skipped_max_context"]),
        "sample_count_error": len([r for r in rows if str(r.get("status")) == "error"]),
        "prompt_tokens": int(ok_rows[0]["prompt_tokens"]) if ok_rows else 0,
        "completion_tokens_mean": round(mean(completion_vals), 6) if completion_vals else 0.0,
        "ttft_ms_mean": round(mean(ttft_vals), 6) if ttft_vals else 0.0,
        "ttft_ms_median": round(median(ttft_vals), 6) if ttft_vals else 0.0,
        "ttft_ms_p95": round(quantile(ttft_vals, 0.95), 6) if ttft_vals else 0.0,
        "gen_tokens_per_sec_mean": round(mean(tok_vals), 6) if tok_vals else 0.0,
        "gen_tokens_per_sec_median": round(median(tok_vals), 6) if tok_vals else 0.0,
        "gen_tokens_per_sec_p05": round(quantile(tok_vals, 0.05), 6) if tok_vals else 0.0,
        "gen_tokens_per_sec_p95": round(quantile(tok_vals, 0.95), 6) if tok_vals else 0.0,
        "total_tokens_per_sec_mean": round(mean(total_tok_vals), 6) if total_tok_vals else 0.0,
        "decode_ms_per_token_mean": round(mean(decode_msp_vals), 6) if decode_msp_vals else 0.0,
    }


def add_common_sweep_args(parser: argparse.ArgumentParser) -> None:
    """Add shared sweep CLI arguments for prompt-length benchmarks."""
    parser.add_argument("--sweep-prompt-tokens", default="")
    parser.add_argument("--sweep-pow2-ki", action="store_true")
    parser.add_argument("--sweep-min-ki", type=int, default=1)
    parser.add_argument("--sweep-max-ki", type=int, default=1024)
    parser.add_argument("--sweep-million-point", type=int, default=1010000)


def build_timing_metrics_row(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: float,
    decode_ms: float,
    total_ms: float,
    decode_ms_per_token: float,
    gen_tokens_per_sec: float,
    total_tokens_per_sec: float,
) -> Dict[str, Any]:
    """Build a normalized timing-metrics payload for one measurement row."""
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "ttft_ms": round(float(ttft_ms), 6),
        "decode_ms": round(float(decode_ms), 6),
        "total_ms": round(float(total_ms), 6),
        "decode_ms_per_token": round(float(decode_ms_per_token), 6),
        "gen_tokens_per_sec": round(float(gen_tokens_per_sec), 6),
        "total_tokens_per_sec": round(float(total_tokens_per_sec), 6),
    }


def write_sweep_summary_artifacts(
    *,
    out_dir: Path,
    sweep_summary_rows: List[Dict[str, Any]],
    model_id: str,
    max_new_tokens: int,
    title_prefix: str,
    plot_png_override: str = "",
) -> Tuple[Dict[str, Any], Path]:
    """Write sweep summary JSON and optional scaling plot for a run."""
    plot_png = (
        Path(plot_png_override).expanduser().resolve()
        if str(plot_png_override or "").strip()
        else out_dir / "prompt_length_scaling.png"
    )
    plot_result = write_sweep_plot_png(
        rows=sweep_summary_rows,
        png_path=plot_png,
        model_id=str(model_id),
        max_new_tokens=int(max_new_tokens),
        title_prefix=title_prefix,
    )
    sweep_summary_json = out_dir / "sweep_summary.json"
    sweep_summary_json.write_text(
        json.dumps({"rows": sweep_summary_rows, "plot": plot_result}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plot_result, plot_png


def update_measure_progress(
    *,
    progress: Any,
    row: Mapping[str, Any],
    target_effective: int,
    iteration: int,
    total_iterations: int,
    extra_ok_fields: Mapping[str, Any] | None = None,
    extra_error_fields: Mapping[str, Any] | None = None,
) -> None:
    """Update tqdm progress for a measured row using a shared schema."""
    if progress is None:
        return
    progress.update(1)
    if str(row.get("status", "ok")) == "ok":
        payload: Dict[str, Any] = {
            "phase": "measure",
            "len": format_k_tokens(int(target_effective)),
            "run": f"{int(iteration)}/{int(total_iterations)}",
            "tok_s": f"{as_float(row.get('gen_tokens_per_sec')):.2f}",
            "ttft_ms": f"{as_float(row.get('ttft_ms')):.1f}",
        }
        if extra_ok_fields:
            payload.update(dict(extra_ok_fields))
        progress.set_postfix(payload)
        return
    payload = {
        "phase": str(row.get("status") or "error"),
        "len": format_k_tokens(int(target_effective)),
        "run": f"{int(iteration)}/{int(total_iterations)}",
    }
    if extra_error_fields:
        payload.update(dict(extra_error_fields))
    progress.set_postfix(payload)


def build_run_summary_payload(
    *,
    run_id: str,
    model_id: str,
    rows: List[Dict[str, Any]],
    ok_rows: List[Dict[str, Any]],
    artifacts: Dict[str, Any],
    sweep_plot: Dict[str, Any],
    sweep_rows: List[Dict[str, Any]],
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build standard run-level summary payload."""
    summary: Dict[str, Any] = {
        "run_id": str(run_id),
        "generated_at_utc": now_iso(),
        "model_id": str(model_id),
        **build_basic_summary(rows, ok_rows),
        "artifacts": dict(artifacts),
        "sweep_plot": dict(sweep_plot),
        "sweep_rows": list(sweep_rows),
    }
    if extra_fields:
        summary.update(dict(extra_fields))
    return summary


def build_summary_base_kwargs(
    *,
    run_id: str,
    model_id: str,
    rows: List[Dict[str, Any]],
    ok_rows: List[Dict[str, Any]],
    sweep_plot: Dict[str, Any],
    sweep_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build common kwargs for ``build_run_summary_payload``."""
    return {
        "run_id": str(run_id),
        "model_id": str(model_id),
        "rows": rows,
        "ok_rows": ok_rows,
        "sweep_plot": sweep_plot,
        "sweep_rows": sweep_rows,
    }
