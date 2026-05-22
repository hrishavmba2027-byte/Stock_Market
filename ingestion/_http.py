"""Shared HTTP helper with retry logic for ingestion modules.

All ``requests.get`` calls in the ingestion path route through
``get_with_retry`` so a single env var controls retry behavior. The retry
count comes from ``REQUEST_MAX_RETRIES`` (default 5).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

DEFAULT_MAX_RETRIES = 5


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def max_retries() -> int:
    raw = os.environ.get("REQUEST_MAX_RETRIES")
    if not raw:
        return DEFAULT_MAX_RETRIES
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_RETRIES
    return max(1, n)


def get_with_retry(
    url: str,
    *,
    label: str = "request",
    backoff_base: float = 1.0,
    **kwargs: Any,
):
    """GET ``url`` with retries (count from ``REQUEST_MAX_RETRIES``, default 5).

    Retries on ``requests.RequestException`` and on HTTP 429 / 5xx with
    exponential backoff (``backoff_base * 2**attempt``). Returns the final
    ``Response`` (which may carry a non-2xx status) or ``None`` when every
    attempt raised a network exception. The caller decides what to do with
    a non-2xx response.
    """
    import requests

    retries = max_retries()
    last_resp = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, **kwargs)
            print(f"[http] {label} attempt {attempt + 1}/{retries} status {resp.status_code}")
        except requests.RequestException as exc:
            _log(f"[http] {label} attempt {attempt + 1}/{retries} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(backoff_base * (2 ** attempt))
            continue

        last_resp = resp
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            _log(
                f"[http] {label} attempt {attempt + 1}/{retries} status {resp.status_code} — backing off"
            )
            if attempt < retries - 1:
                time.sleep(backoff_base * (2 ** attempt))
            continue

        return resp

    return last_resp
