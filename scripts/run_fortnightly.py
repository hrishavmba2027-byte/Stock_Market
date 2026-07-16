"""Fortnightly production run — the full pipeline, end to end.

Cadence: **every 15 days.** This is the complete refresh: it collects fresh
technicals, engineers features, incrementally retrains + re-forecasts the model,
refreshes news/sentiment and fundamentals, then re-decides with the LLM.

Stages (in order):
  1-3. Technicals → feature engineering → model train + forecast
       (run_full_workflow.py --live: OHLCV append → indicators → ensemble
        inference → push forecasts to the sheet → monthly_finetune ``--if-due``,
        which incrementally retrains when RETRAIN_INTERVAL_DAYS have elapsed)
  4.   News → Firestore                    (ingestion.collect_all, news step)
  5.   FinBERT sentiment                    (same collect_all run, sentiment step)
  6.   Fundamentals refresh                 (ingestion.fundamentals — latest 5 quarters,
                                             rolling window, Firestore)
  7.   LLM BUY/HOLD/AVOID suggestions       (features.trade_suggestions)

Note: production "training" is always the **incremental** ``monthly_finetune``
(warm-start, never a full-dataset retrain), fired by stage 15 of the workflow when
due — which aligns naturally with this 15-day cadence.

Usage::

    python -m scripts.run_fortnightly                 # live end-to-end
    python -m scripts.run_fortnightly --workflow-dry-run   # forecast pipeline dry (no sheet writes)
    python -m scripts.run_fortnightly --tickers RELIANCE,TCS
    python -m scripts.run_fortnightly --dry-run       # print the stage commands only
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from scripts._pipeline import Stage, add_common_flags, exit_code, run_pipeline


def build_stages(tickers: Optional[str], workflow_dry_run: bool, force: bool, allocate: bool = True) -> List[Stage]:
    # 1-3. Full forecast pipeline (OHLCV → FE → train/forecast → finetune-if-due).
    workflow = ["run_full_workflow.py", "--dry-run" if workflow_dry_run else "--live"]
    if tickers:
        workflow += ["--worksheets", tickers]

    # 4-5. News + sentiment (reuse the market data just refreshed above → skip OHLCV here).
    collect = ["-m", "ingestion.collect_all",
               "--no-market-data", "--no-cross-sectional", "--no-reddit", "--no-x"]
    if tickers:
        collect += ["--tickers", tickers]

    # 6. Fundamentals (latest 5 quarters, rolling window → Firestore).
    fundamentals = ["-m", "ingestion.fundamentals"]
    if tickers:
        fundamentals += ["--tickers", tickers]

    # 7. LLM suggestions.
    suggest = ["-m", "features.trade_suggestions"]
    if tickers:
        suggest += ["--tickers", tickers]
    if force:
        suggest += ["--force"]

    # 8. Fund allocation (Markowitz) → final website-facing suggestions to Firestore.
    allocate_cmd = ["-m", "features.portfolio_allocation"]
    if tickers:
        allocate_cmd += ["--tickers", tickers]

    stages: List[Stage] = [
        ("forecast_pipeline", workflow),   # steps 1-3: technicals, FE, train, forecast
        ("news+sentiment", collect),       # steps 4-5
        ("fundamentals", fundamentals),    # step 6
        ("llm_suggestions", suggest),      # step 7
    ]
    if allocate:
        stages.append(("fund_allocation", allocate_cmd))   # step 8
    return stages


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fortnightly full pipeline: technicals→FE→train/forecast→news→sentiment→fundamentals→LLM.")
    p.add_argument("--tickers", default=None, help="Comma-separated subset (default: full universe).")
    p.add_argument("--workflow-dry-run", action="store_true",
                   help="Run the forecast pipeline in dry-run (no Google Sheet writes).")
    p.add_argument("--force", action="store_true", help="Re-send every stock to the LLM (ignore fingerprints).")
    p.add_argument("--no-allocate", action="store_true", help="Skip the fund-allocation / final-suggestions step.")
    add_common_flags(p)
    args = p.parse_args(argv)

    summary = run_pipeline(
        build_stages(args.tickers, args.workflow_dry_run, args.force, allocate=not args.no_allocate),
        continue_on_error=args.continue_on_error, dry_run=args.dry_run, title="fortnightly",
    )
    return exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
