"""Point-in-time slicing of the workbook frames.

Every accessor here enforces the harness's core invariant: **nothing dated after
the decision bar ``C`` is ever returned.** The workbook is loaded once (by the
caller) and sliced per bar through these functions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def truncate_frames(frames: Dict[str, pd.DataFrame], cutoff: Any) -> Dict[str, pd.DataFrame]:
    """Return per-symbol frames with all rows whose ``Date > cutoff`` removed.

    The single PIT slice point for prices/technicals/training. An empty frame is
    dropped so downstream code never sees a symbol with no visible history.
    """
    C = pd.Timestamp(cutoff).normalize()
    out: Dict[str, pd.DataFrame] = {}
    for sym, frame in frames.items():
        if frame is None or frame.empty or "Date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce")
        sliced = frame.loc[dates <= C].copy()
        if not sliced.empty:
            out[sym] = sliced.reset_index(drop=True)
    return out


def payloads_asof(frames: Dict[str, pd.DataFrame], cutoff: Any, metadata: Dict[str, Any]) -> List[Any]:
    """Build one ``SheetPayload`` per symbol from raw OHLCV truncated ≤ ``cutoff``.

    ``prepare_stock_part`` computes indicators internally, so the payload frame is
    the plain OHLCV frame (Date/Open/High/Low/Close/Adj Close/Volume). Symbols
    with too few rows to form one input window are skipped.
    """
    from main import SheetPayload  # local import: main pulls in torch

    seq_len = int(metadata.get("seq_len", 20))
    truncated = truncate_frames(frames, cutoff)
    payloads: List[Any] = []
    for sym, frame in truncated.items():
        if len(frame) < seq_len + 1:
            continue
        payloads.append(
            SheetPayload(
                name=sym,
                frame=frame.copy(),
                headers=list(frame.columns),
                data_row_count=len(frame),
                worksheet=None,
            )
        )
    return payloads


def log_returns_asof(
    frames: Dict[str, pd.DataFrame],
    cutoff: Any,
    lookback: int = 252,
) -> Dict[str, np.ndarray]:
    """Per-symbol trailing daily log-returns (≤ ``cutoff``), last ``lookback`` obs.

    Used to estimate the covariance matrix Σ for the optimiser. Only rows ≤ C are
    used, so the risk model is strictly point-in-time.
    """
    truncated = truncate_frames(frames, cutoff)
    out: Dict[str, np.ndarray] = {}
    for sym, frame in truncated.items():
        close = pd.to_numeric(frame.get("Close"), errors="coerce").dropna().to_numpy(dtype=float)
        if len(close) < 3:
            continue
        rets = np.diff(np.log(np.clip(close, 1e-9, None)))
        rets = rets[np.isfinite(rets)]
        if len(rets) >= 2:
            out[sym] = rets[-lookback:]
    return out


def returns_hist_asof(
    frames: Dict[str, pd.DataFrame],
    cutoff: Any,
    lookback: int = 252,
) -> Dict[str, List[float]]:
    """Same as :func:`log_returns_asof` but as plain lists (engine-friendly)."""
    return {sym: r.tolist() for sym, r in log_returns_asof(frames, cutoff, lookback).items()}


def close_asof(frames: Dict[str, pd.DataFrame], cutoff: Any) -> Dict[str, float]:
    """Latest close ≤ ``cutoff`` per symbol (the mark-to-market price at bar C)."""
    truncated = truncate_frames(frames, cutoff)
    out: Dict[str, float] = {}
    for sym, frame in truncated.items():
        close = pd.to_numeric(frame.get("Close"), errors="coerce").dropna()
        if not close.empty:
            out[sym] = float(close.iloc[-1])
    return out


def open_on(frames: Dict[str, pd.DataFrame], day: Any) -> Dict[str, float]:
    """Open price *exactly on* ``day`` per symbol (the next-bar fill price).

    Returns only symbols that actually traded on ``day`` — orders for names with no
    bar on the fill day simply don't fill (carried as pending by the caller).
    """
    D = pd.Timestamp(day).normalize()
    out: Dict[str, float] = {}
    for sym, frame in frames.items():
        if frame is None or frame.empty or "Date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        row = frame.loc[dates == D]
        if row.empty:
            continue
        val = pd.to_numeric(row.iloc[-1].get("Open"), errors="coerce")
        if pd.notna(val):
            out[sym] = float(val)
    return out


def day_ohlc(frames: Dict[str, pd.DataFrame], day: Any) -> Dict[str, Dict[str, float]]:
    """Open/High/Low/Close *exactly on* ``day`` per symbol (that bar's full range).

    Used by the intraday target/stop monitor: standing at bar ``C`` we already mark
    to ``C``'s close, so reading ``C``'s O/H/L/C to decide whether a resting limit
    (target) or stop order would have executed *that day* is strictly point-in-time
    — no information from any bar after ``C`` is consulted. Symbols with no bar on
    ``day`` are absent (their orders simply don't trigger).
    """
    D = pd.Timestamp(day).normalize()
    out: Dict[str, Dict[str, float]] = {}
    for sym, frame in frames.items():
        if frame is None or frame.empty or "Date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        row = frame.loc[dates == D]
        if row.empty:
            continue
        r = row.iloc[-1]
        bar: Dict[str, float] = {}
        for key, col in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")):
            val = pd.to_numeric(r.get(col), errors="coerce")
            if pd.notna(val):
                bar[key] = float(val)
        # Require at least a high/low to evaluate a touch; fall back to close.
        if "high" not in bar and "close" in bar:
            bar["high"] = bar["close"]
        if "low" not in bar and "close" in bar:
            bar["low"] = bar["close"]
        if "open" not in bar and "close" in bar:
            bar["open"] = bar["close"]
        if bar:
            out[sym.upper()] = bar
    return out


def recent_closes_asof(
    frames: Dict[str, pd.DataFrame],
    cutoff: Any,
    n: int = 5,
) -> Dict[str, List[float]]:
    """Last ``n`` closes ≤ ``cutoff`` per symbol (the LLM's ``recent_closes``)."""
    truncated = truncate_frames(frames, cutoff)
    out: Dict[str, List[float]] = {}
    for sym, frame in truncated.items():
        close = pd.to_numeric(frame.get("Close"), errors="coerce").dropna().to_numpy(dtype=float)
        if len(close):
            out[sym] = [float(x) for x in close[-n:]]
    return out
