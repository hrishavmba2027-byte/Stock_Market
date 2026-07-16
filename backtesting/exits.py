"""Intraday target / stop-loss exit monitor for the walk-forward backtest.

The LLM suggestion for each name carries a profit *target* (``sell_price``) and a
*stoploss*. A real trader rests a limit-sell at the target (and a protective stop)
the moment the suggestion is acted on, so those orders can fire on *any* trading
day between rebalances — not only on the 15-day retrain / 7-day sentiment bars.
This module reproduces that: every bar it checks each held position's OHLC range
against its active target/stop and books the exit the day the level is touched,
keeping the freed cash until the framework issues a fresh suggestion.

Point-in-time fairness (no look-ahead):

* We only ever read bar ``C``'s own O/H/L/C — the same bar we already mark to. A
  resting order placed at (or before) ``C`` would genuinely have executed during
  ``C`` if the day's range touched its level, so filling it here consults no
  future information.
* **Fill conventions never flatter the result.** A target (limit-sell) fills at
  the target, or at the *open* if the day gapped straight through it (a better
  price we legitimately would have received) — never above the day's range. A stop
  fills at the stop, or at the *open* if the day gapped down through it (a *worse*
  price) — the conservative side.
* **Same-bar ambiguity resolves to the stop.** If a single daily bar spans both
  the stop and the target we cannot know from daily data which printed first, so
  we assume the stop hit first (the pessimistic assumption), avoiding any
  optimistic bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExitSignal:
    ticker: str
    reason: str          # "target" | "stop"
    price: float         # execution price (see fill conventions above)
    target: Optional[float] = None
    stop: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0:          # NaN or non-positive → treat as absent
        return None
    return f


def scan_exits(
    positions: Dict[str, Dict[str, float]],
    last_signal: Dict[str, Dict[str, Any]],
    ohlc: Dict[str, Dict[str, float]],
    *,
    honor_stop: bool = True,
) -> List[ExitSignal]:
    """Return the target/stop exits triggered by ``ohlc`` for held positions.

    ``positions``  : ``{ticker: {qty, avg_price}}`` (the live book).
    ``last_signal``: ``{ticker: {sell_price, stoploss, ...}}`` — the standing LLM
                     suggestion whose levels arm the resting orders.
    ``ohlc``       : ``{ticker: {open, high, low, close}}`` for *this* bar only.
    """
    exits: List[ExitSignal] = []
    for ticker, pos in positions.items():
        if float(pos.get("qty", 0.0)) <= 0:
            continue
        tkr = ticker.upper()
        sig = last_signal.get(tkr) or last_signal.get(ticker) or {}
        target = _num(sig.get("sell_price"))
        stop = _num(sig.get("stoploss")) if honor_stop else None
        if target is None and stop is None:
            continue
        bar = ohlc.get(tkr) or ohlc.get(ticker)
        if not bar:
            continue
        o = _num(bar.get("open"))
        hi = _num(bar.get("high"))
        lo = _num(bar.get("low"))
        if hi is None or lo is None:
            continue

        hit_target = target is not None and hi >= target
        hit_stop = stop is not None and lo <= stop

        if hit_stop and hit_target:
            # Ambiguous within one daily bar → assume the stop printed first.
            fill = stop if (o is None or o >= stop) else o     # gap-down fills at open
            exits.append(ExitSignal(tkr, "stop", fill, target, stop, hi, lo, o))
        elif hit_target:
            fill = target if (o is None or o <= target) else o  # gap-up fills at open
            exits.append(ExitSignal(tkr, "target", fill, target, stop, hi, lo, o))
        elif hit_stop:
            fill = stop if (o is None or o >= stop) else o      # gap-down fills at open
            exits.append(ExitSignal(tkr, "stop", fill, target, stop, hi, lo, o))
    return exits


def monitor_rows(
    bar: Any,
    positions: Dict[str, Dict[str, float]],
    last_signal: Dict[str, Dict[str, Any]],
    ohlc: Dict[str, Dict[str, float]],
    exits: List[ExitSignal],
    *,
    honor_stop: bool = True,
) -> List[Dict[str, Any]]:
    """Audit rows for the ``Price_Monitor`` sheet: the daily price check that drives
    profit-booking, one row per held name that carries an active target/stop."""
    from pandas import Timestamp

    try:
        b = Timestamp(bar).strftime("%Y-%m-%d")
    except Exception:
        b = str(bar)
    fired = {e.ticker: e for e in exits}
    rows: List[Dict[str, Any]] = []
    for ticker, pos in positions.items():
        if float(pos.get("qty", 0.0)) <= 0:
            continue
        tkr = ticker.upper()
        sig = last_signal.get(tkr) or last_signal.get(ticker) or {}
        target = _num(sig.get("sell_price"))
        stop = _num(sig.get("stoploss")) if honor_stop else None
        if target is None and stop is None:
            continue
        bar_px = ohlc.get(tkr) or ohlc.get(ticker) or {}
        ex = fired.get(tkr)
        rows.append({
            "bar": b,
            "ticker": tkr,
            "open": bar_px.get("open"),
            "high": bar_px.get("high"),
            "low": bar_px.get("low"),
            "close": bar_px.get("close"),
            "target": target,
            "stoploss": stop,
            "hit": ex.reason if ex else "",
            "exit_price": round(ex.price, 4) if ex else None,
        })
    return rows
