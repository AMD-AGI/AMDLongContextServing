# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Record I/O helpers for JSONL and Parquet benchmark artifacts.

This module provides a small format-agnostic interface for reading/writing
record lists used by benchmark scripts. It supports:
- JSONL (`.jsonl`)
- Parquet (`.parquet`) with sidecar metadata for nested JSON columns
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


Scalar = (str, int, float, bool, type(None))


def _jsonl_read(path: Path) -> List[Dict[str, Any]]:
    """Read rows from a JSONL file.

    Args:
        path: Input JSONL path.

    Returns:
        Parsed row list.
    """
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to a JSONL file.

    Args:
        path: Output JSONL path.
        rows: Record rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _parquet_meta_path(path: Path) -> Path:
    """Return metadata sidecar path for a parquet file.

    Args:
        path: Parquet path.

    Returns:
        Sidecar path.
    """
    return path.with_name(path.name + ".meta.json")


def _normalize_rows_for_parquet(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize rows so pandas/pyarrow can store them in parquet.

    Nested structures are encoded as JSON strings and tracked in metadata.

    Args:
        rows: Input record rows.

    Returns:
        Tuple of normalized rows and JSON-encoded column names.
    """
    json_cols: Set[str] = set()
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        out: Dict[str, Any] = {}
        for key, value in dict(row).items():
            if isinstance(value, Scalar):
                out[key] = value
            elif isinstance(value, (dict, list, tuple, set)):
                out[key] = json.dumps(value, sort_keys=True)
                json_cols.add(str(key))
            else:
                out[key] = str(value)
        normalized.append(out)
    return normalized, sorted(json_cols)


def _restore_json_columns(
    rows: Sequence[Mapping[str, Any]],
    json_columns: Iterable[str],
) -> List[Dict[str, Any]]:
    """Restore JSON-encoded columns back to Python objects.

    Args:
        rows: Input record rows.
        json_columns: Column names encoded as JSON text.

    Returns:
        Decoded row list.
    """
    json_cols = set(json_columns)
    restored: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        for key in json_cols:
            value = out.get(key)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text:
                continue
            try:
                out[key] = json.loads(text)
            except Exception:  # noqa: BLE001
                pass
        restored.append(out)
    return restored


def _require_pandas() -> Any:
    """Import and return pandas.

    Returns:
        Imported pandas module.

    Raises:
        RuntimeError: If pandas is unavailable.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Parquet I/O requires pandas + pyarrow/fastparquet.") from exc
    return pd


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load records from JSONL or Parquet.

    Args:
        path: Input file path.

    Returns:
        Record rows.

    Raises:
        ValueError: If file extension is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _jsonl_read(path)
    if suffix == ".parquet":
        pd = _require_pandas()
        frame = pd.read_parquet(path)
        frame = frame.where(pd.notnull(frame), None)
        rows: List[Dict[str, Any]] = frame.to_dict(orient="records")
        meta_path = _parquet_meta_path(path)
        if not meta_path.exists():
            return rows
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        json_columns = meta.get("json_columns") or []
        return _restore_json_columns(rows, json_columns)
    raise ValueError(f"Unsupported record format: {path}")


def write_records(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write records to JSONL or Parquet.

    Args:
        path: Output file path (`.jsonl` or `.parquet`).
        rows: Record rows.

    Raises:
        ValueError: If file extension is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        _jsonl_write(path, rows)
        return
    if suffix == ".parquet":
        pd = _require_pandas()
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized_rows, json_cols = _normalize_rows_for_parquet(rows)
        frame = pd.DataFrame(normalized_rows)
        frame.to_parquet(path, index=False, compression="zstd")
        meta_path = _parquet_meta_path(path)
        meta_path.write_text(
            json.dumps({"json_columns": json_cols}, sort_keys=True),
            encoding="utf-8",
        )
        return
    raise ValueError(f"Unsupported record format: {path}")

