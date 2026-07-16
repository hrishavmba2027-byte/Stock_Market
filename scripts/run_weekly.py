"""Weekly production run — refresh news+sentiment, then re-decide with the LLM.

Cadence: **every 7 days.** This run does NOT retrain the model or re-forecast; it
reuses the **already-dumped** forecasts (on the operational Google Sheet) and the
**already-stored** fundamentals (Firestore), and only refreshes the fast-moving
signal — news sentiment — before asking the LLM again.

Stages:
  1. News → Firestore + FinBERT sentiment   (ingestion.collect_all, news+sentiment only)
  2. LLM BUY/HOLD/AVOID suggestions          (features.trade_suggestions)
       inputs: existing forecast path (sheet) + fresh sentiment (Firestore)
               + existing fundamentals (Firestore) → trade_suggestions collection

The incremental fingerprint in the suggestion layer means only stocks whose
sentiment (or forecast/fundamentals) actually changed are re-sent to the LLM.

Usage::

    python -m scripts.run_weekly
    python -m scripts.run_weekly --tickers RELIANCE,TCS
    python -m scripts.run_weekly --dry-run
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from scripts._pipeline import Stage, add_common_flags, exit_code, run_pipeline


def build_stages(tickers: Optional[str], force: bool, allocate: bool = True) -> List[Stage]:
    # News + sentiment only: skip OHLCV, index cache, reddit and X.
    collect = ["-m", "ingestion.collect_all",
               "--no-market-data", "--no-cross-sectional", "--no-reddit", "--no-x"]
    suggest = ["-m", "features.trade_suggestions"]
    allocate_cmd = ["-m", "features.portfolio_allocation"]
    if tickers:
        collect += ["--tickers", tickers]
        suggest += ["--tickers", tickers]
        allocate_cmd += ["--tickers", tickers]
    if force:
        suggest += ["--force"]
    stages: List[Stage] = [
        ("news+sentiment", collect),
        ("llm_suggestions", suggest),
    ]
    # Fund allocation (Markowitz) always follows the LLM → writes the final,
    # website-facing suggestions (BUY %/target/stop, AVOID, HOLD) to Firestore.
    if allocate:
        stages.append(("fund_allocation", allocate_cmd))
    return stages


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Weekly run: fresh news+sentiment → LLM suggestions.")
    p.add_argument("--tickers", default=None, help="Comma-separated subset (default: full universe).")
    p.add_argument("--force", action="store_true", help="Re-send every stock to the LLM (ignore fingerprints).")
    p.add_argument("--no-allocate", action="store_true", help="Skip the fund-allocation / final-suggestions step.")
    add_common_flags(p)
    args = p.parse_args(argv)

    summary = run_pipeline(
        build_stages(args.tickers, args.force, allocate=not args.no_allocate),
        continue_on_error=args.continue_on_error, dry_run=args.dry_run, title="weekly",
    )
    return exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
