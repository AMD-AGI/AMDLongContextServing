# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Public KV calibration API.

This file provides explicit re-exports from the implementation module so static
analysis tools (e.g. pylint) can resolve symbols reliably.
"""

from __future__ import annotations

from .kv_capacity_calibration_impl import (
    _headroom_tokens,
    _to_float,
    _to_int,
    build_kv_capacity_calibration,
    evaluate_kv_capacity_target,
    parse_kv_startup_metrics_from_text,
    recommend_kv_adjustment,
    render_kv_calibration_markdown,
    wait_for_kv_startup_metrics,
)

__all__ = [
    "build_kv_capacity_calibration",
    "evaluate_kv_capacity_target",
    "parse_kv_startup_metrics_from_text",
    "recommend_kv_adjustment",
    "render_kv_calibration_markdown",
    "wait_for_kv_startup_metrics",
    "_to_int",
    "_to_float",
    "_headroom_tokens",
]
