#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Render HFLC variable defaults and documentation from defaults.mk."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_defaults(path: Path) -> list[tuple[str, str, str]]:
    docs: dict[str, str] = {}
    rows: list[tuple[str, str, str]] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# @doc "):
            payload = raw[len("# @doc ") :]
            key, _, desc = payload.partition("|")
            docs[key.strip()] = desc.strip()
            continue

        match = re.match(r"^(HFLC_[A-Z0-9_]+)\s*\?=\s*(.*)$", raw)
        if not match:
            continue

        var = match.group(1)
        default = match.group(2).strip()
        rows.append((var, default, docs.get(var, "(no description)")))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", required=True, type=Path)
    args = parser.parse_args()

    rows = parse_defaults(args.defaults)

    print("HFLC profiles: smoke | long | debug | capture")
    print("Variables (default from defaults.mk):")
    print(f"{'Variable':42} {'Default':30} Description")
    print("-" * 120)
    for var, default, desc in rows:
        print(f"{var:42} {default[:30]:30} {desc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
