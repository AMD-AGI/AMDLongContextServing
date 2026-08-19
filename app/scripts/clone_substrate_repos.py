#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Clone or refresh substrate repositories with lockfile support.

This script materializes external code substrate repositories under
``data/substrate``. Repositories are not committed to git; they are runtime
assets only.

Default behavior is lockfile mode: read `data/metadata/substrate_repos_manifest.json`
and sync each repository to the exact recorded `head` commit for reproducible
substrate reconstruction.

Use `--unlocked-head` only when intentionally refreshing to the latest upstream
HEADs; this updates the manifest lockfile to the newly observed commits.

Targets:
- bokeh/bokeh
- python-visualization/folium
- python-visualization/branca
- holoviz/holoviews
- holoviz/hvplot
- holoviz/panel
- holoviz/datashader
- pandas-dev/pandas
- numpy/numpy
- scipy/scipy
- matplotlib/matplotlib
- pydata/xarray
- geopandas/geopandas
- shapely/shapely
- pyproj4/pyproj
- rasterio/rasterio
- plotly/plotly.py
- scikit-learn/scikit-learn
- statsmodels/statsmodels
- networkx/networkx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from _paths import repo_path

REPOS: List[Tuple[str, str]] = [
    ("bokeh", "https://github.com/bokeh/bokeh"),
    ("folium", "https://github.com/python-visualization/folium"),
    ("branca", "https://github.com/python-visualization/branca"),
    ("holoviews", "https://github.com/holoviz/holoviews"),
    ("hvplot", "https://github.com/holoviz/hvplot"),
    ("panel", "https://github.com/holoviz/panel"),
    ("datashader", "https://github.com/holoviz/datashader"),
    ("pandas", "https://github.com/pandas-dev/pandas"),
    ("numpy", "https://github.com/numpy/numpy"),
    ("scipy", "https://github.com/scipy/scipy"),
    ("matplotlib", "https://github.com/matplotlib/matplotlib"),
    ("xarray", "https://github.com/pydata/xarray"),
    ("geopandas", "https://github.com/geopandas/geopandas"),
    ("shapely", "https://github.com/shapely/shapely"),
    ("pyproj", "https://github.com/pyproj4/pyproj"),
    ("rasterio", "https://github.com/rasterio/rasterio"),
    ("plotly_py", "https://github.com/plotly/plotly.py"),
    ("scikit_learn", "https://github.com/scikit-learn/scikit-learn"),
    ("statsmodels", "https://github.com/statsmodels/statsmodels"),
    ("networkx", "https://github.com/networkx/networkx"),
]


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    """Run a subprocess command.

    Args:
        cmd: Command vector.
        cwd: Optional working directory.

    Raises:
        subprocess.CalledProcessError: If command exits non-zero.
    """
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _git_head(path: Path) -> str:
    """Return git HEAD SHA for a repository path.

    Args:
        path: Repository directory.

    Returns:
        Commit SHA string.
    """
    out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True)
    return out.strip()


def _git_branch(path: Path) -> str:
    """Return current branch name when available.

    Args:
        path: Repository directory.

    Returns:
        Branch name or ``"HEAD"`` if detached.
    """
    out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"], text=True)
    return out.strip()


def _relative_to_repo(path: Path) -> str:
    """Return repo-relative path for manifest output.

    Args:
        path: Absolute path to convert.

    Returns:
        Repo-relative path string.
    """
    repo = repo_path()
    return str(path.resolve().relative_to(repo))


def _ensure_git_repo(dest: Path, url: str) -> None:
    """Ensure destination is a git repo with origin configured.

    Args:
        dest: Destination directory.
        url: Expected origin URL.
    """
    if (dest / ".git").exists():
        out = subprocess.check_output(
            ["git", "-C", str(dest), "config", "--get", "remote.origin.url"],
            text=True,
        ).strip()
        if out != url:
            _run(["git", "-C", str(dest), "remote", "set-url", "origin", url])
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", str(dest)])
    _run(["git", "-C", str(dest), "remote", "add", "origin", url])


def _sync_to_locked_head(name: str, url: str, dest: Path, commit_sha: str) -> Dict[str, str]:
    """Sync repository to an exact locked commit SHA.

    Args:
        name: Logical repository name.
        url: Clone URL.
        dest: Destination path.
        commit_sha: Locked commit SHA.

    Returns:
        Summary record for manifest output.
    """
    _ensure_git_repo(dest, url)
    try:
        _run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit_sha])
    except subprocess.CalledProcessError:
        # Fallback for servers that do not support shallow fetch by SHA.
        _run(["git", "-C", str(dest), "fetch", "origin", commit_sha])
    _run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"])
    _run(["git", "-C", str(dest), "clean", "-fd"])

    return {
        "name": name,
        "url": url,
        "path": _relative_to_repo(dest),
        "status": "locked_synced",
        "branch": _git_branch(dest),
        "head": _git_head(dest),
    }


def _sync_to_latest_head(name: str, url: str, dest: Path) -> Dict[str, str]:
    """Sync repository to latest upstream HEAD.

    Args:
        name: Logical repository name.
        url: Clone URL.
        dest: Destination path.

    Returns:
        Summary record for manifest output.
    """
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", url, str(dest)])
    else:
        _ensure_git_repo(dest, url)
        _run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", "HEAD"])
        _run(["git", "-C", str(dest), "reset", "--hard", "FETCH_HEAD"])
        _run(["git", "-C", str(dest), "clean", "-fd"])

    return {
        "name": name,
        "url": url,
        "path": _relative_to_repo(dest),
        "status": "head_synced",
        "branch": _git_branch(dest),
        "head": _git_head(dest),
    }


def _load_locked_specs(manifest_path: Path) -> List[Tuple[str, str, str]]:
    """Load locked repo specs from manifest.

    Args:
        manifest_path: Path to lock manifest JSON.

    Returns:
        List of `(name, url, head)` tuples.
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    repos = payload.get("repos")
    if not isinstance(repos, list) or not repos:
        raise ValueError(f"Invalid lock manifest (repos missing/empty): {manifest_path}")
    out: List[Tuple[str, str, str]] = []
    for row in repos:
        name = str((row or {}).get("name", "")).strip()
        url = str((row or {}).get("url", "")).strip()
        head = str((row or {}).get("head", "")).strip()
        if not name or not url or not head:
            raise ValueError(f"Invalid lock manifest row: {row}")
        out.append((name, url, head))
    return out


def main() -> int:
    """Run substrate clone workflow.

    Returns:
        Exit code. ``0`` on success.
    """
    parser = argparse.ArgumentParser(description="Sync substrate repos with locked or head mode")
    parser.add_argument(
        "--substrate-root",
        default=str(repo_path("data", "substrate")),
    )
    parser.add_argument(
        "--manifest-json",
        default=str(repo_path("data", "metadata", "substrate_repos_manifest.json")),
    )
    parser.add_argument(
        "--unlocked-head",
        action="store_true",
        help="Opt out of lockfile mode and refresh each repo to origin HEAD.",
    )
    args = parser.parse_args()

    substrate_root = Path(args.substrate_root)
    manifest_path = Path(args.manifest_json)
    locked_mode = not args.unlocked_head

    rows: List[Dict[str, str]] = []
    if locked_mode:
        if not manifest_path.exists():
            print(
                f"Lock manifest not found: {manifest_path}. "
                "Run with --unlocked-head once to bootstrap/update it.",
                file=sys.stderr,
            )
            return 2
        specs = _load_locked_specs(manifest_path)
        for name, url, head in specs:
            dest = substrate_root / name
            row = _sync_to_locked_head(name, url, dest, head)
            if row["head"] != head:
                print(
                    f"Locked sync mismatch for {name}: expected {head}, got {row['head']}",
                    file=sys.stderr,
                )
                return 2
            rows.append(row)
    else:
        print(
            "INFO: unlocked mode enabled; syncing to upstream HEAD and "
            "rewriting substrate lock manifest.",
            file=sys.stderr,
        )
        for name, url in REPOS:
            dest = substrate_root / name
            rows.append(_sync_to_latest_head(name, url, dest))

    payload = {
        "mode": "unlocked_head" if args.unlocked_head else "locked",
        "substrate_root": _relative_to_repo(substrate_root),
        "repos": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
