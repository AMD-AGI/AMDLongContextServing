# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP helpers shared by benchmark scripts."""

from __future__ import annotations

import json
from typing import Any, Dict
from urllib import request


def post_json(url: str, payload: Dict[str, Any], timeout_sec: int = 180) -> Dict[str, Any]:
    """Send a JSON POST request and parse a JSON response.

    Args:
        url: Full endpoint URL.
        payload: JSON-serializable request body.
        timeout_sec: Request timeout in seconds.

    Returns:
        Parsed JSON response object.
    """
    req = request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))

