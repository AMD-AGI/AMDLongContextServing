# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Path helpers for repo-relative script defaults."""

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

__all__ = ["repo_root", "repo_path"]


def _is_repo_root(path: Path) -> bool:
    """Return whether ``path`` looks like the long-context-serving repo root."""
    has_app = (path / "app" / "long_context_serving").exists()
    has_repo_marker = (path / ".git").exists() or (
        (path / "Makefile").exists() and (path / "pyproject.toml").exists()
    )
    return has_app and has_repo_marker


def repo_root() -> Path:
    """Return the long_context_serving repository root.

    Uses ``ROOT`` when set and valid; otherwise falls back to filesystem
    discovery relative to this script.

    Returns:
        Repository root path.
    """
    env_root = os.getenv("ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _is_repo_root(candidate):
            return candidate
    fallback = Path(__file__).resolve().parents[2]
    if _is_repo_root(fallback):
        return fallback
    raise RuntimeError(
        "Unable to resolve repo root: set ROOT to long-context-serving repository root "
        "(e.g. export ROOT=/path/to/long_context_serving)"
    )


def repo_path(*parts: str) -> Path:
    """Build a path relative to the repository root.

    Args:
        *parts: Path components under the repo root.

    Returns:
        Absolute path under the repo root.
    """
    return repo_root().joinpath(*parts)
