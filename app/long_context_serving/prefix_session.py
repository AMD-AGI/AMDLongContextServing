# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility module shim.

This module re-exports the implementation module to preserve the public
import path while enabling isolated coverage/accounting.
"""

from __future__ import annotations

import sys as _sys

from . import prefix_session_impl as _impl

_sys.modules[__name__] = _impl
