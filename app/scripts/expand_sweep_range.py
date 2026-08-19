#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Expand an inclusive FROM..TO context-length range into a power-of-two sweep list.

Used by the Makefile `run` target so users can write `FROM=1Ki TO=2Mi` instead of
a full comma-separated token list. Accepts Ki/Mi suffixes (1Ki=1024,
2Mi=2097152) or raw integer token counts.

Emits a comma-separated list of every power of two P with FROM <= P <= TO,
snapped to the power-of-two grid: the low bound rounds up and the high bound
rounds down to the nearest power of two, matching the campaign's 2^N sweep
points.
"""

import math
import sys

_SUFFIXES = {"ki": 1024, "mi": 1024 * 1024, "gi": 1024 * 1024 * 1024}


def parse_tokens(value: str) -> int:
    """Parse a token count that may carry a Ki/Mi/Gi suffix.

    Args:
        value: e.g. "1024", "1Ki", "2Mi".

    Returns:
        Integer token count.

    Raises:
        ValueError: If the value is not a positive integer count.
    """
    raw = str(value).strip()
    lowered = raw.lower()
    suffix = lowered[-2:]
    if suffix in _SUFFIXES:
        magnitude = float(raw[:-2]) * _SUFFIXES[suffix]
        tokens = int(round(magnitude))
    else:
        tokens = int(raw)
    if tokens <= 0:
        raise ValueError(f"token count must be positive: {value!r}")
    return tokens


def expand(from_value: str, to_value: str) -> list:
    """Expand an inclusive range into ascending powers of two.

    Args:
        from_value: Lower bound (inclusive), Ki/Mi/Gi suffix allowed.
        to_value: Upper bound (inclusive), Ki/Mi/Gi suffix allowed.

    Returns:
        List of powers of two within [from, to].

    Raises:
        ValueError: If FROM > TO or no power of two falls in the range.
    """
    lo = parse_tokens(from_value)
    hi = parse_tokens(to_value)
    if lo > hi:
        raise ValueError(f"FROM ({from_value}) must be <= TO ({to_value})")
    lo_exp = int(math.ceil(math.log2(lo)))
    hi_exp = int(math.floor(math.log2(hi)))
    if lo_exp > hi_exp:
        raise ValueError(
            f"no power of two in range [{from_value}, {to_value}]"
        )
    return [1 << e for e in range(lo_exp, hi_exp + 1)]


def main() -> int:
    """Run CLI: print the expanded comma-separated sweep list."""
    if len(sys.argv) != 3:
        print("usage: expand_sweep_range.py FROM TO", file=sys.stderr)
        return 2
    try:
        points = expand(sys.argv[1], sys.argv[2])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(",".join(str(p) for p in points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
