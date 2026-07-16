"""Common backtest data collector — news + fundamentals in one command.

This is the single entrypoint to run **before** the simulation. It collects, for
the configured backtest universe and window, the two point-in-time inputs the
walk-forward harness consumes off local disk:

1. **News → sentiment** (``backtesting.fetch_news``): Wayback-CDX historical
   articles → FinBERT → 7-day sentiment windows, written to the local news
   workbook (``news_workbook_path``: sheets ``News`` / ``Sentiment`` / ``Manifest``).
2. **Fundamentals** (``backtesting.fetch_fundamentals``): screener.in quarterly +
   annual series → the append-only fundamentals PIT parquet, each period stamped
   with its information-availability date (period-end + reporting lag) so the sim
   never sees a result before it was announced.

Both collectors are idempotent/resumable (news via its ``(target, year)``
manifest, fundamentals via PIT dedup), so re-running only fills gaps.

Usage::

    python -m backtesting.collect_data                      # news + fundamentals, full universe/window
    python -m backtesting.collect_data --tickers TCS,RELIANCE
    python -m backtesting.collect_data --only fundamentals  # skip news
    python -m backtesting.collect_data --only news          # skip fundamentals

Note: ``fetch_news`` requires ``BACKTEST_ENABLED=true`` (it no-ops otherwise);
fundamentals collection runs regardless. The simulation reads whatever is present
and degrades any missing news window to "unavailable" (production-consistent).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from app.config.backtest_settings import get_backtest_settings


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Collect news + fundamentals for the backtest window.")
    p.add_argument("--tickers", default=None, help="Comma-separated subset (default: backtest universe).")
    p.add_argument("--years", default=None, help="News years override (default: derived from the backtest window).")
    p.add_argument("--only", choices=["news", "fundamentals"], default=None,
                   help="Run only one collector (default: both).")
    p.add_argument("--no-annual", action="store_true", help="Fundamentals: quarterly only (skip 10y annual series).")
    p.add_argument("--pause", type=float, default=2.5, help="Fundamentals: seconds between company fetches.")
    p.add_argument("--dry-run", action="store_true", help="Fundamentals: parse without writing the PIT store.")
    args = p.parse_args(argv)

    settings = get_backtest_settings()
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else settings.resolved_tickers()
    )
    _log(f"[collect] universe={len(tickers)} tickers, window={settings.start_date}..{settings.resolved_end_date()}")

    results: Dict[str, Any] = {}

    # 1) Fundamentals (screener.in) — no BACKTEST_ENABLED gate.
    if args.only != "news":
        _log("[collect] === fundamentals (screener.in) ===")
        from backtesting import fetch_fundamentals
        results["fundamentals"] = fetch_fundamentals.collect(
            tickers, settings=settings, include_annual=not args.no_annual,
            pause=args.pause, dry_run=args.dry_run,
        )

    # 2) News → sentiment (Wayback CDX + FinBERT). Requires BACKTEST_ENABLED.
    if args.only != "fundamentals":
        _log("[collect] === news + sentiment (Wayback CDX → FinBERT) ===")
        if not settings.enabled:
            _log("[collect] BACKTEST_ENABLED is false — skipping news (set it to collect news).")
            results["news"] = {"status": "disabled"}
        else:
            from backtesting import fetch_news
            news_argv: List[str] = []
            if args.tickers:
                news_argv += ["--tickers", args.tickers]
            if args.years:
                news_argv += ["--years", args.years]
            results["news"] = fetch_news.run(news_argv)

    print(json.dumps({"status": "ok", "collectors": list(results.keys()),
                      "summary": results}, default=str, indent=2))
    return results


if __name__ == "__main__":
    run()
