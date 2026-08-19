# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Long-context serving package."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Final


def _version_from_pyproject() -> str | None:
    """Read ``[project].version`` from the repository ``pyproject.toml``.

    This fallback keeps ``__version__`` available when running from a source
    checkout via ``PYTHONPATH=app`` (without an installed wheel).
    """
    try:
        import tomllib
    except Exception:  # pragma: no cover
        return None
    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = ((data.get("project") or {}).get("version") or "").strip()
        return version or None
    except Exception:  # pragma: no cover
        return None


def _resolve_version() -> str:
    """Resolve package version from installed metadata, then source fallback."""
    try:
        return metadata.version("long-context-serving")
    except metadata.PackageNotFoundError:
        return _version_from_pyproject() or "0.0.0"


__version__: Final[str] = _resolve_version()
