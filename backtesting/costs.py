"""Execution-cost model for the walk-forward backtest (pure functions).

Two frictions are charged on every fill:

* :func:`roundtrip_cost_inr` — the all-in NSE transaction cost (brokerage + STT +
  exchange txn charge + SEBI + GST + stamp duty), expressed as a single
  round-trip basis-point figure and split evenly across the buy and sell legs.
* :func:`atr_slippage_inr` — a market-impact/slippage estimate proportional to the
  name's recent ATR(14): ``mult * ATR14`` rupees of adverse move per share.

Both are deterministic and take plain numbers so they are trivially unit-tested
and identical between the slice run and the full run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def roundtrip_cost_inr(notional: float, *, bps: float = 40.0) -> float:
    """Per-leg transaction cost in rupees for a trade of ``notional`` rupees.

    ``bps`` is the *round-trip* cost in basis points (buy + sell); a single fill is
    charged half of it. Example: ₹1,00,000 notional at 40 bps round-trip ⇒ ₹200
    per leg (0.20%).
    """
    notional = abs(float(notional))
    if notional <= 0:
        return 0.0
    return notional * (float(bps) / 10_000.0) / 2.0


def atr_slippage_inr(qty: float, atr14: float, price: float, *, mult: float = 0.10) -> float:
    """Slippage cost in rupees for filling ``qty`` shares.

    ``mult * ATR14`` rupees of adverse price movement per share, floored at zero.
    A missing/degenerate ATR falls back to a small fraction of price so slippage is
    never negative or NaN.
    """
    qty = abs(float(qty))
    if qty <= 0:
        return 0.0
    atr = float(atr14)
    if not np.isfinite(atr) or atr <= 0:
        atr = abs(float(price)) * 0.01 if np.isfinite(price) else 0.0
    return qty * max(0.0, float(mult) * atr)


def atr14_from_frame(frame: pd.DataFrame, *, period: int = 14) -> float:
    """Wilder-style ATR(14) from the tail of an OHLC frame (≤ C rows only).

    Returns ``nan`` when the frame is too short; callers fall back via
    :func:`atr_slippage_inr`.
    """
    if frame is None or frame.empty or len(frame) < period + 1:
        return float("nan")
    cols = {c.lower(): c for c in frame.columns}
    try:
        high = pd.to_numeric(frame[cols["high"]], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(frame[cols["low"]], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(frame[cols["close"]], errors="coerce").to_numpy(dtype=float)
    except KeyError:
        return float("nan")
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    tr = tr[np.isfinite(tr)]
    if len(tr) < period:
        return float("nan")
    return float(np.mean(tr[-period:]))
