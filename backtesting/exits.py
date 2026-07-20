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


def scan_exits_and_trail(
    positions: Dict[str, Dict[str, float]],
    last_signal: Dict[str, Dict[str, Any]],
    ohlc: Dict[str, Dict[str, float]],
    *,
    honor_stop: bool = True,
    trail_on_target: bool = True,
    trail_atr_mult: float = 2.0,
    atr_map: Optional[Dict[str, float]] = None,
    max_holding_days: int = 0,
    bar: Any = None,
) -> "tuple[List[ExitSignal], Dict[str, Dict[str, Any]]]":
    """Hold-and-trail exit scan (trade-quality findings D/E).

    Differences from :func:`scan_exits` (which keeps the legacy sell-at-target
    behaviour):

    * **Target touch does not sell.** It *arms a trailing stop*: the standing
      stop ratchets to ``max(previous stop, close − trail_atr_mult*ATR14,
      entry price)`` and the position stays open. 62% of the historical
      target-exits kept rising ≥2% within 15 sessions (median further peak
      +3.31%); trailing captured +₹537k the auto-sell gave up.
    * **Armed positions re-ratchet every bar** (stop only ever moves up), so a
      runner is followed until the trail is hit or the LLM verdict changes.
    * **Age exit**: when ``max_holding_days`` > 0, a position older than that
      (calendar days from its entry bar) is closed at the bar close — the
      16–30-day bucket was the only profitable one; beyond it the sample says
      nothing, so age out and let a fresh suggestion re-enter if warranted.
    * Stop exits are unchanged, and the same-bar stop+target ambiguity still
      resolves pessimistically to the stop.

    Returns ``(exits, stop_updates)`` where ``stop_updates`` maps ticker →
    ``{"stoploss": new_stop, "trail_armed": True}`` for the caller to fold into
    its standing signals (the backtest mutates ``state.last_signal``; production
    writes the ratchet back to the suggestions store).
    """
    exits: List[ExitSignal] = []
    updates: Dict[str, Dict[str, Any]] = {}
    atr_map = atr_map or {}
    for ticker, pos in positions.items():
        if float(pos.get("qty", 0.0)) <= 0:
            continue
        tkr = ticker.upper()
        sig = last_signal.get(tkr) or last_signal.get(ticker) or {}
        target = _num(sig.get("sell_price"))
        stop = _num(sig.get("stoploss")) if honor_stop else None
        armed = bool(sig.get("trail_armed"))
        bar_px = ohlc.get(tkr) or ohlc.get(ticker)
        if not bar_px:
            continue
        o = _num(bar_px.get("open"))
        hi = _num(bar_px.get("high"))
        lo = _num(bar_px.get("low"))
        close = _num(bar_px.get("close"))
        if hi is None or lo is None:
            continue

        hit_stop = stop is not None and lo <= stop
        hit_target = (not armed) and target is not None and hi >= target

        if hit_stop:
            # Stop always wins the same-bar ambiguity (pessimistic).
            fill = stop if (o is None or o >= stop) else o
            exits.append(ExitSignal(tkr, "stop", fill, target, stop, hi, lo, o))
            continue

        atr = _num(atr_map.get(tkr) or atr_map.get(ticker))
        entry = _num(pos.get("avg_price"))

        if hit_target and not trail_on_target:
            fill = target if (o is None or o <= target) else o
            exits.append(ExitSignal(tkr, "target", fill, target, stop, hi, lo, o))
            continue

        # Age exit (checked only when nothing else fired this bar).
        if max_holding_days > 0 and bar is not None and pos.get("entry_bar"):
            try:
                from pandas import Timestamp
                age = (Timestamp(bar) - Timestamp(pos["entry_bar"])).days
            except Exception:
                age = -1
            if age > max_holding_days and close is not None:
                exits.append(ExitSignal(tkr, "max_age", close, target, stop, hi, lo, o))
                continue

        # Trail arming / ratchet — stop only ever moves UP.
        if trail_on_target and close is not None and atr is not None:
            if hit_target or armed:
                candidates = [close - trail_atr_mult * atr]
                if stop is not None:
                    candidates.append(stop)
                if hit_target and entry is not None:
                    candidates.append(entry)     # break-even floor on first arm
                new_stop = max(c for c in candidates if c is not None and c > 0)
                if stop is None or new_stop > stop + 1e-9 or (hit_target and not armed):
                    updates[tkr] = {"stoploss": round(new_stop, 4), "trail_armed": True}
    return exits, updates


def monitor_rows(
    bar: Any,
    positions: Dict[str, Dict[str, float]],
    last_signal: Dict[str, Dict[str, Any]],
    ohlc: Dict[str, Dict[str, float]],
    exits: List[ExitSignal],
    *,
    honor_stop: bool = True,
    stop_updates: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Audit rows for the ``Price_Monitor`` sheet: the daily price check that drives
    profit-booking, one row per held name that carries an active target/stop.
    A trailing-stop ratchet this bar shows as ``hit="trail↑"`` with the new stop."""
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
        upd = (stop_updates or {}).get(tkr)
        hit = ex.reason if ex else ("trail↑" if upd else "")
        rows.append({
            "bar": b,
            "ticker": tkr,
            "open": bar_px.get("open"),
            "high": bar_px.get("high"),
            "low": bar_px.get("low"),
            "close": bar_px.get("close"),
            "target": target,
            "stoploss": upd["stoploss"] if upd else stop,
            "hit": hit,
            "exit_price": round(ex.price, 4) if ex else None,
        })
    return rows
