"""Daily data collection orchestrator.

Runs all ingestion pipelines with partial parallelism:
  1. OHLCV market data append (Data_update.py)          — sequential
  2. NIFTY / VIX cross-sectional index cache             — sequential
  3. News / Reddit / X ingestion                         — parallel (ThreadPoolExecutor)
  4. FinBERT sentiment aggregation (features.sentiment)  — sequential (after step 3)

Steps 3a–3c write to independent parquet files and Firestore collections,
so they are safe to run concurrently. Sentiment must wait for all three
to finish because it reads those parquet outputs.

Each step is independently fault-tolerant — a failure is logged but does
not abort the remaining steps. The final JSON summary includes a per-step
status and row count, and the process exits non-zero if any step failed.

Intended to be called by the ``daily-data-collection`` GitHub Actions
workflow, but also runnable locally:

    python -m ingestion.collect_all
    python -m ingestion.collect_all --no-firestore --no-sentiment
    python -m ingestion.collect_all --tickers RELIANCE,TCS
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
NEWS_PARQUET = _REPO_ROOT / "Data" / "archive" / "news.parquet"
REDDIT_PARQUET = _REPO_ROOT / "Data" / "archive" / "reddit_posts.parquet"
X_PARQUET = _REPO_ROOT / "Data" / "archive" / "x_posts.parquet"
SENTIMENT_PARQUET = _REPO_ROOT / "Data" / "archive" / "sentiment_features.parquet"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _ok(name: str, **kwargs: Any) -> Dict[str, Any]:
    return {"step": name, "status": "ok", **kwargs}


def _skip(name: str, reason: str) -> Dict[str, Any]:
    return {"step": name, "status": "skipped", "reason": reason}


def _fail(name: str, exc: BaseException) -> Dict[str, Any]:
    _log(f"[collect_all] {name} FAILED: {exc}")
    _log(traceback.format_exc())
    return {"step": name, "status": "error", "error": str(exc)}


# ── step implementations ──────────────────────────────────────────────────────

def step_market_data(args: argparse.Namespace) -> Dict[str, Any]:
    """Append today's OHLCV bars to the Google Sheet via Data_update."""
    if args.no_market_data:
        return _skip("market_data", "--no-market-data set")
    try:
        import Data_update

        # Build a minimal namespace that Data_update.run() expects.
        update_args = argparse.Namespace(
            sheet_id=args.sheet_id,
            google_credentials=args.google_credentials,
            worksheet=None,
            worksheets=args.tickers,
            start_date="2015-01-01",
            interval="1d",
        )
        result = Data_update.run_update(update_args)
        rows = int(result.get("rows_added", 0)) if isinstance(result, dict) else 0
        return _ok("market_data", rows_added=rows)
    except Exception as exc:
        return _fail("market_data", exc)


def step_cross_sectional(args: argparse.Namespace, write_firestore: bool) -> Dict[str, Any]:
    """Refresh NIFTY 50 and INDIAVIX index cache, then upload to Firestore and remove parquet."""
    if args.no_cross_sectional:
        return _skip("cross_sectional", "--no-cross-sectional set")
    try:
        from features import cross_sectional

        df = cross_sectional.fetch_index_history()
        rows = int(len(df))

        if write_firestore and not df.empty:
            try:
                written = cross_sectional.write_indices_to_firestore(df)
                if written > 0:
                    cross_sectional.DEFAULT_INDEX_CACHE.unlink(missing_ok=True)
                    _log(f"[collect_all] indices.parquet uploaded ({written} rows) and removed")
            except Exception as exc:
                _log(f"[collect_all] indices Firestore upload failed: {exc}; parquet retained for retry")

        return _ok("cross_sectional", index_rows=rows)
    except Exception as exc:
        return _fail("cross_sectional", exc)


def step_news(args: argparse.Namespace, write_firestore: bool) -> Dict[str, Any]:
    """Fetch news headlines and scrape article bodies."""
    if args.no_news:
        return _skip("news", "--no-news set")
    from ingestion._archive import truthy_env
    collect = truthy_env("COLLECT_NEWS", default=True)
    if not collect:
        _log("[collect_all] COLLECT_NEWS=false — drain-only: uploading parquet archive to Firestore, skipping fresh fetch")
    try:
        from ingestion import news_ingest
        from ingestion.aliases import list_tickers

        tickers = _resolve_tickers(args.tickers, list_tickers)
        df = news_ingest.refresh_news(
            tickers,
            output_path=NEWS_PARQUET,
            backfill_days=args.backfill_days,
            sleep_seconds=0.2,
            write_firestore=write_firestore,
            collect=collect,
            scrape_article_bodies=not args.no_article_bodies,
            article_scrape_workers=args.article_scrape_workers,
        )
        return _ok("news", rows=int(len(df)), collect_mode="full" if collect else "drain")
    except Exception as exc:
        return _fail("news", exc)


def step_reddit(args: argparse.Namespace, write_firestore: bool) -> Dict[str, Any]:
    """Scrape Reddit /new feeds for NIFTY-related posts."""
    if args.no_reddit:
        return _skip("reddit", "--no-reddit set")
    from ingestion._archive import truthy_env
    collect = truthy_env("COLLECT_REDDIT", default=True)
    if not collect:
        _log("[collect_all] COLLECT_REDDIT=false — drain-only: uploading parquet archive to Firestore, skipping fresh fetch")
    try:
        from ingestion import reddit_ingest
        from ingestion.reddit_ingest import DEFAULT_SUBREDDITS

        df = reddit_ingest.refresh_reddit(
            output_path=REDDIT_PARQUET,
            subreddits=DEFAULT_SUBREDDITS,
            backfill_days=args.backfill_days,
            write_firestore=write_firestore,
            collect=collect,
        )
        return _ok("reddit", rows=int(len(df)), collect_mode="full" if collect else "drain")
    except Exception as exc:
        return _fail("reddit", exc)


def step_x(args: argparse.Namespace, write_firestore: bool) -> Dict[str, Any]:
    """Scrape X / Twitter posts for NIFTY tickers (best-effort)."""
    if args.no_x:
        return _skip("x", "--no-x set")
    from ingestion._archive import truthy_env
    collect = truthy_env("COLLECT_X", default=True)
    if not collect:
        _log("[collect_all] COLLECT_X=false — drain-only: uploading parquet archive to Firestore, skipping fresh fetch")
    try:
        from ingestion import x_ingest
        from ingestion.aliases import list_tickers

        tickers = _resolve_tickers(args.tickers, list_tickers)
        df = x_ingest.refresh_x(
            tickers,
            output_path=X_PARQUET,
            backfill_days=args.backfill_days,
            sleep_seconds=0.5,
            write_firestore=write_firestore,
            collect=collect,
        )
        return _ok("x", rows=int(len(df)), collect_mode="full" if collect else "drain")
    except Exception as exc:
        return _fail("x", exc)


def step_sentiment(args: argparse.Namespace, write_firestore: bool) -> Dict[str, Any]:
    """Run FinBERT sentiment scoring over all three text channels."""
    if args.no_sentiment:
        return _skip("sentiment", "--no-sentiment set")
    try:
        import pandas as pd
        from features import sentiment as sent_module

        # In drain-only runs (COLLECT_NEWS/REDDIT/X=false) the source parquets
        # are deleted before this step runs, so refresh_sentiment sees no inputs
        # and returns early without uploading the existing archive. Upload it first.
        if write_firestore and SENTIMENT_PARQUET.exists():
            try:
                existing = pd.read_parquet(SENTIMENT_PARQUET)
                if not existing.empty:
                    written = sent_module.write_sentiment_to_firestore(existing)
                    if written > 0:
                        SENTIMENT_PARQUET.unlink(missing_ok=True)
                        _log(f"[collect_all] pre-drained sentiment_features.parquet ({written} docs)")
            except Exception as exc:
                _log(f"[collect_all] sentiment pre-drain failed: {exc}; continuing with refresh")

        df = sent_module.refresh_sentiment(
            output_path=SENTIMENT_PARQUET,
            news_path=NEWS_PARQUET,
            reddit_path=REDDIT_PARQUET,
            x_path=X_PARQUET,
            write_firestore=write_firestore,
        )
        return _ok("sentiment", rows=int(len(df)))
    except Exception as exc:
        return _fail("sentiment", exc)


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_tickers(
    tickers_arg: Optional[str],
    default_fn: Any,
) -> List[str]:
    if tickers_arg:
        return [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    return default_fn()


def _firestore_enabled(args: argparse.Namespace) -> bool:
    if args.no_firestore:
        return False
    return bool(
        os.environ.get("GOOGLE_CREDENTIALS")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )


# ── arg parsing ───────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect live market data, news, Reddit posts, X posts, and "
            "technical indicator features for the NIFTY 50 universe."
        )
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated tickers to process (default: full NIFTY 50 universe).",
    )
    parser.add_argument(
        "--sheet-id",
        default=os.environ.get("SHEET_ID") or os.environ.get("OPERATIONAL_SHEET_ID"),
        help="Google Sheet ID for OHLCV data (overrides SHEET_ID env var).",
    )
    parser.add_argument(
        "--google-credentials",
        default=os.environ.get("GOOGLE_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Path to service-account JSON credentials file.",
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=7,
        help="Lookback window in days for news / Reddit / X (default: 7).",
    )
    parser.add_argument(
        "--no-firestore",
        action="store_true",
        help="Skip all Firestore writes (parquet staging only).",
    )
    parser.add_argument("--no-market-data", action="store_true", help="Skip OHLCV market data step.")
    parser.add_argument("--no-cross-sectional", action="store_true", help="Skip NIFTY/VIX index cache step.")
    parser.add_argument("--no-news", action="store_true", help="Skip news ingestion step.")
    parser.add_argument("--no-reddit", action="store_true", help="Skip Reddit ingestion step.")
    parser.add_argument("--no-x", action="store_true", help="Skip X/Twitter ingestion step.")
    parser.add_argument("--no-sentiment", action="store_true", help="Skip FinBERT sentiment step.")
    parser.add_argument(
        "--no-article-bodies",
        action="store_true",
        help="Skip second-stage article body scraping in the news step.",
    )
    parser.add_argument(
        "--article-scrape-workers",
        type=int,
        default=8,
        help="Thread-pool size for article body scraping (default: 8).",
    )
    return parser.parse_args(argv)


# ── main ──────────────────────────────────────────────────────────────────────

def _run_parallel(
    tasks: List[tuple[str, Callable[[], Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Run named callables in a thread pool and return results in submission order."""
    futures: Dict[Any, str] = {}
    ordered: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for name, fn in tasks:
            futures[pool.submit(fn)] = name
        for future in as_completed(futures):
            name = futures[future]
            try:
                ordered[name] = future.result()
            except Exception as exc:
                ordered[name] = _fail(name, exc)
    return [ordered[name] for name, _ in tasks]


def collect_all(args: argparse.Namespace) -> Dict[str, Any]:
    write_firestore = _firestore_enabled(args)
    _log(f"[collect_all] Firestore writes: {'enabled' if write_firestore else 'disabled'}")

    results: List[Dict[str, Any]] = []

    # Sequential: market data and index cache must finish before text ingestion
    # so their data is available if any downstream step needs it.
    results.append(step_market_data(args))
    results.append(step_cross_sectional(args, write_firestore))

    # Parallel: news / Reddit / X write to independent parquet files and
    # Firestore collections, so they are safe to run concurrently.
    _log("[collect_all] launching news / reddit / x in parallel")
    parallel_results = _run_parallel([
        ("news",   lambda: step_news(args, write_firestore)),
        ("reddit", lambda: step_reddit(args, write_firestore)),
        ("x",      lambda: step_x(args, write_firestore)),
    ])
    results.extend(parallel_results)

    # Sequential: sentiment reads the parquet files written above.
    results.append(step_sentiment(args, write_firestore))

    failed = [r["step"] for r in results if r["status"] == "error"]
    overall = "error" if failed else "ok"
    return {
        "status": overall,
        "failed_steps": failed,
        "steps": results,
        "firestore_enabled": write_firestore,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    summary = collect_all(args)
    print(json.dumps(summary, indent=2, default=str))
    return 1 if summary["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
