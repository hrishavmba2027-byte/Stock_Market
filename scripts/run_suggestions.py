"""Read-only suggestion run — everything is already present; just decide.

Runs **no** collection or training. It reads the data already on disk / in the
cloud — the forecast paths (operational Google Sheet), the latest news sentiment
(Firestore ``sentiment_latest``) and the stored fundamentals (Firestore
``fundamentals``) — channels them into the LLM decision layer, and writes the
BUY/HOLD/AVOID suggestions (with target + stoploss) to Firestore
``trade_suggestions``.

Use this to (re)generate suggestions on demand without touching any upstream
pipeline — e.g. after manually refreshing one input, or to re-run the LLM with
``--force``. It is exactly ``features.trade_suggestions`` with a runbook-friendly
name; the optional allocation step sizes the resulting BUY set.

Stages:
  1. LLM BUY/HOLD/AVOID suggestions   (features.trade_suggestions)
  2. Fund allocation (Markowitz)       (features.portfolio_allocation) — writes the
        final website-facing suggestions to Firestore; skip with --no-allocate.

Usage::

    python -m scripts.run_suggestions
    python -m scripts.run_suggestions --force
    python -m scripts.run_suggestions --no-allocate      # LLM only, no allocation
    python -m scripts.run_suggestions --tickers RELIANCE,TCS --dry-run
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from scripts._pipeline import Stage, add_common_flags, exit_code, run_pipeline


def build_stages(tickers: Optional[str], force: bool, allocate: bool) -> List[Stage]:
    suggest = ["-m", "features.trade_suggestions"]
    if tickers:
        suggest += ["--tickers", tickers]
    if force:
        suggest += ["--force"]
    stages: List[Stage] = [("llm_suggestions", suggest)]
    # Allocation always follows the LLM (writes the final website-facing
    # suggestions to Firestore) unless explicitly skipped.
    if allocate:
        allocate_cmd = ["-m", "features.portfolio_allocation"]
        if tickers:
            allocate_cmd += ["--tickers", tickers]
        stages.append(("fund_allocation", allocate_cmd))
    return stages


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Read existing forecasts+sentiment+fundamentals → LLM suggestions.")
    p.add_argument("--tickers", default=None, help="Comma-separated subset (default: full universe).")
    p.add_argument("--force", action="store_true", help="Re-send every stock to the LLM (ignore fingerprints).")
    p.add_argument("--no-allocate", action="store_true", help="Skip the fund-allocation / final-suggestions step.")
    add_common_flags(p)
    args = p.parse_args(argv)

    summary = run_pipeline(
        build_stages(args.tickers, args.force, allocate=not args.no_allocate),
        continue_on_error=args.continue_on_error, dry_run=args.dry_run, title="suggestions",
    )
    return exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
