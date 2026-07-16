"""The backtest clock: trading days and the retrain/sentiment cadence schedule.

Pure functions over the workbook's own dates — the backtest advances one *trading
bar* at a time (never a naive calendar day), so cadence counts skip weekends and
holidays exactly as the market does.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


def trading_days(
    frames: Dict[str, pd.DataFrame],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[pd.Timestamp]:
    """Sorted union of every symbol's ``Date`` values, clipped to [start, end].

    This union *is* the backtest clock — every date on which at least one universe
    name traded. Bounds are inclusive; ``None`` means unbounded on that side.
    """
    all_dates: set = set()
    for frame in frames.values():
        if frame is None or frame.empty or "Date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        all_dates.update(pd.Timestamp(d).normalize() for d in dates)
    days = sorted(all_dates)
    if start is not None:
        lo = pd.Timestamp(start).normalize()
        days = [d for d in days if d >= lo]
    if end is not None:
        hi = pd.Timestamp(end).normalize()
        days = [d for d in days if d <= hi]
    return days


def cadence_schedule(
    days: List[pd.Timestamp],
    retrain_every: int = 15,
    sentiment_every: int = 7,
) -> List[Tuple[pd.Timestamp, Dict[str, bool]]]:
    """Tag each trading bar with the events due on it.

    * Bar 0 fires **both** a retrain and a sentiment refresh (cold-start seed).
    * A retrain is due every ``retrain_every`` trading bars; a sentiment refresh
      every ``sentiment_every`` bars.
    * When both coincide, **retrain supersedes** — only ``{'retrain': True}`` is
      emitted so the loop doesn't double-signal on that bar.

    Returns ``[(day, {'retrain'?: True, 'sentiment'?: True})]``. Bars with no event
    still appear (with an empty dict) so mark-to-market/fills run every bar.
    """
    schedule: List[Tuple[pd.Timestamp, Dict[str, bool]]] = []
    for i, day in enumerate(days):
        events: Dict[str, bool] = {}
        is_retrain = (i % retrain_every == 0)
        is_sentiment = (i % sentiment_every == 0)
        if is_retrain:
            events["retrain"] = True
        elif is_sentiment:
            events["sentiment"] = True
        schedule.append((pd.Timestamp(day), events))
    return schedule
