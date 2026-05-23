"""P1.3 — Reddit ingestion via anonymous JSON scraping.

Reddit Support rejected our Data API access request, so the PRAW path is no
longer viable. Instead we scrape Reddit's **public JSON endpoint** —
``https://www.reddit.com/r/{subreddit}/new.json?limit=100`` — which works
without any OAuth credentials, just a polite ``User-Agent`` per Reddit's policy.

Posts (no comments) are pulled from the same 5 NIFTY-focused subreddits as
before, matched to NIFTY 50 tickers via the alias map, and persisted to
``Data/archive/reddit_posts.parquet`` with permalink-hash deduplication. Each row is
also dumped to Firestore collection ``reddit_posts`` in the shape:

    {
        "company_name": "Reliance Industries",
        "news":         "<title>\n\n<selftext>",
        "date_of_news": "2026-05-20T08:30:00+00:00",
        ...
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from ingestion._archive import (
    archive_path_for,
    drain_pending_archive,
    merge_stage_to_archive,
    truthy_env,
    upload_and_clear_archive,
)
from ingestion._firestore import init_firestore_client
from ingestion.aliases import find_tickers_in_text, load_aliases

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Data" / "archive" / "reddit_posts.parquet"
DEFAULT_BACKFILL_DAYS = 7
DEFAULT_SUBREDDITS = (
    "IndianStockMarket",
    "IndiaInvestments",
    "StockMarketIndia",
    "DalalStreetTalks",
    "NIFTY50",
)
POST_FETCH_LIMIT = 100  # Reddit's /new.json caps at 100 per call

REDDIT_BASE = "https://www.reddit.com"
DEFAULT_USER_AGENT = "stock-market-automation/0.1 (anonymous scrape)"
REQUEST_TIMEOUT_SECONDS = 15
INTER_SUBREDDIT_SLEEP_SECONDS = 2.0

FIRESTORE_COLLECTION = "reddit_posts"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _hash_permalink(permalink: str) -> str:
    return hashlib.sha256(permalink.encode("utf-8")).hexdigest()[:16]


def _user_agent() -> str:
    return os.environ.get("REDDIT_USER_AGENT", DEFAULT_USER_AGENT)


def _company_name_for(ticker: str) -> str:
    entry = load_aliases().get(ticker.upper(), {})
    return entry.get("name") or ticker


def fetch_subreddit_json(
    subreddit: str,
    limit: int = POST_FETCH_LIMIT,
    user_agent: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """GET https://www.reddit.com/r/{sub}/new.json with retry on 429/5xx.

    Returns the parsed JSON dict, or ``None`` if every retry failed.
    Network errors / non-200 responses are logged but never raised — the
    caller continues with the next subreddit.
    """
    from ingestion._http import get_with_retry

    url = f"{REDDIT_BASE}/r/{subreddit}/new.json"
    params = {"limit": limit, "raw_json": 1}
    headers = {"User-Agent": user_agent or _user_agent()}

    resp = get_with_retry(
        url,
        label=f"reddit r/{subreddit}",
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if resp is None:
        _log(f"[reddit] r/{subreddit} exhausted retries")
        return None

    if resp.status_code != 200:
        _log(f"[reddit] r/{subreddit} status {resp.status_code} — giving up")
        return None

    try:
        return resp.json()
    except ValueError as exc:
        _log(f"[reddit] r/{subreddit} JSON decode failed: {exc}")
        return None


def _row_from_post(post: Dict[str, Any], subreddit: str) -> Optional[Dict[str, Any]]:
    """Build a parquet row from a Reddit JSON post object.

    Returns ``None`` when the post mentions no NIFTY 50 ticker (those rows
    would only inflate noise downstream).
    """
    title = post.get("title") or ""
    body = post.get("selftext") or ""
    text = f"{title}\n{body}"
    tickers = find_tickers_in_text(text)
    if not tickers:
        return None
    created_utc = post.get("created_utc")
    if created_utc is None:
        return None
    try:
        ts = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None
    raw_permalink = post.get("permalink") or ""
    permalink = f"{REDDIT_BASE}{raw_permalink}" if raw_permalink.startswith("/") else raw_permalink
    return {
        "ts": ts,
        "subreddit": post.get("subreddit") or subreddit,
        "kind": "post",
        "title": title,
        "body": body,
        "score": int(post.get("score") or 0),
        "num_comments": int(post.get("num_comments") or 0),
        "permalink": permalink,
        "permalink_hash": _hash_permalink(permalink),
        "tickers": tickers,
    }


def parse_listing(
    listing: Dict[str, Any],
    subreddit: str,
    cutoff: datetime,
) -> List[Dict[str, Any]]:
    """Pull post rows out of a Reddit ``/new.json`` response.

    Reddit's ``/new`` is roughly time-ordered, so we can break early once we
    hit a post older than the cutoff.
    """
    rows: List[Dict[str, Any]] = []
    if not listing:
        return rows
    children = listing.get("data", {}).get("children", [])
    for child in children:
        post = child.get("data") if isinstance(child, dict) else None
        if not isinstance(post, dict):
            continue
        created_utc = post.get("created_utc")
        if created_utc is not None:
            try:
                ts = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
                if ts < cutoff:
                    # /new is time-ordered; everything after is older still.
                    break
            except (TypeError, ValueError, OverflowError):
                continue
        row = _row_from_post(post, subreddit)
        if row is not None:
            rows.append(row)
    return rows


def _explode_tickers(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per (ticker, post) so per-ticker aggregations are trivial."""
    if not rows:
        return pd.DataFrame()
    flat: List[Dict[str, Any]] = []
    for row in rows:
        for ticker in row["tickers"]:
            flat.append({**row, "ticker": ticker})
    df = pd.DataFrame(flat)
    df = df.drop(columns=["tickers"], errors="ignore")
    return df


def refresh_reddit(
    output_path: Path = DEFAULT_OUTPUT,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    backfill_days: int = DEFAULT_BACKFILL_DAYS,
    post_limit: int = POST_FETCH_LIMIT,
    fetcher: Optional[Callable[[str, int], Optional[Dict[str, Any]]]] = None,
    write_firestore: bool = False,
    firestore_client: Optional[Any] = None,
    collect: bool = True,
) -> pd.DataFrame:
    """Scrape each subreddit's /new feed, merge with cache, write outputs.

    ``fetcher`` is injectable so tests don't make real HTTP calls. It must
    accept ``(subreddit_name, limit)`` and return the parsed JSON dict (or
    ``None`` to skip the subreddit).
    """
    archive_path = archive_path_for(output_path)
    upload_fn = lambda df: write_reddit_to_firestore(df, client=firestore_client)

    # ---- Step 1: drain pending archive (runs even if collection is off) -
    if write_firestore:
        drain_pending_archive(archive_path, upload_fn, _log)
    elif archive_path.exists():
        _log("[reddit] pending archive present but --no-firestore set; leaving file alone")

    # ---- Skip-collect short-circuit (COLLECT_REDDIT=false) --------------
    if not collect:
        _log("[reddit] COLLECT_REDDIT=false; skipping fresh collection (drain-only run)")
        return load_reddit(archive_path)

    # ---- Step 2: fresh collection ---------------------------------------
    cutoff = datetime.now(timezone.utc) - timedelta(days=backfill_days)
    fetcher = fetcher or (lambda sub, lim: fetch_subreddit_json(sub, limit=lim))

    all_rows: List[Dict[str, Any]] = []
    for sub in subreddits:
        listing = fetcher(sub, post_limit)
        rows = parse_listing(listing or {}, sub, cutoff)
        _log(f"[reddit] r/{sub}: {len(rows)} matched")
        all_rows.extend(rows)
        time_module.sleep(INTER_SUBREDDIT_SLEEP_SECONDS)

    new_df = _explode_tickers(all_rows)
    fresh_df = _merge_with_cache(new_df, output_path, cutoff=cutoff)
    _log(f"[reddit] prepared {len(fresh_df)} fresh rows for Firestore staging")

    # ---- Steps 3 + 4: merge-stage and upload ----------------------------
    merged_count = merge_stage_to_archive(
        fresh_df, archive_path, dedup_keys=("ticker", "permalink_hash"), log=_log
    )
    if write_firestore:
        if merged_count > 0:
            upload_and_clear_archive(archive_path, upload_fn, _log)
    else:
        _log("[reddit] Firestore write skipped (--no-firestore); staged archive retained")

    return fresh_df


def _merge_with_cache(new_df: pd.DataFrame, output_path: Path, cutoff: datetime) -> pd.DataFrame:
    del output_path
    combined = new_df.copy() if new_df is not None else pd.DataFrame()
    if combined.empty:
        return combined
    combined["ts"] = pd.to_datetime(combined["ts"], utc=True)
    combined = combined[combined["ts"] >= pd.Timestamp(cutoff)]
    combined = combined.drop_duplicates(subset=["ticker", "permalink_hash"], keep="last")
    combined = combined.sort_values(["ticker", "ts"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Firestore writer (matches the news/x payload shape)
# ---------------------------------------------------------------------------

def _date_of_post(ts: Any) -> Optional[str]:
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts) if ts is not None else None


def _post_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    """One nested Reddit post inside a per-ticker Firestore doc."""
    return {
        "title": row.get("title"),
        "body": row.get("body"),
        "subreddit": row.get("subreddit"),
        "score": int(row.get("score") or 0),
        "kind": row.get("kind", "post"),
        "date_of_post": _date_of_post(row.get("ts")),
        "permalink": row.get("permalink"),
    }


def _firestore_payload_for_ticker(
    ticker: str,
    rows: List[Dict[str, Any]],
    scrape_date: str,
) -> Dict[str, Any]:
    """Build the company-level Firestore doc with all posts nested by hash."""
    posts: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        permalink_hash = row.get("permalink_hash")
        if not permalink_hash:
            continue
        posts[str(permalink_hash)] = _post_entry(row)
    return {
        "company_name": _company_name_for(ticker),
        "ticker": ticker,
        "scrape_date": scrape_date,
        "posts": posts,
    }


def _firestore_doc_id(row_or_ticker: Any) -> str:
    """Doc id is the ticker symbol — one doc per company."""
    if isinstance(row_or_ticker, dict):
        return str(row_or_ticker.get("ticker") or "")
    return str(row_or_ticker or "")


def write_reddit_to_firestore(
    df: pd.DataFrame,
    collection: str = FIRESTORE_COLLECTION,
    client: Optional[Any] = None,
) -> int:
    """Write **one document per ticker** with all posts nested by ``permalink_hash``.

    Returns the number of input rows uploaded (not the number of documents).
    """
    if df.empty:
        return 0
    client = client or init_firestore_client()
    scrape_date = datetime.now(timezone.utc).date().isoformat()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in df.to_dict(orient="records"):
        if not row.get("permalink_hash"):
            continue
        ticker = str(row.get("ticker") or "")
        if not ticker:
            continue
        grouped.setdefault(ticker, []).append(row)

    if not grouped:
        return 0

    coll = client.collection(collection)
    batch = client.batch()
    pending = 0
    rows_uploaded = 0
    for ticker in sorted(grouped):
        rows = grouped[ticker]
        payload = _firestore_payload_for_ticker(ticker, rows, scrape_date)
        batch.set(coll.document(_firestore_doc_id(ticker)), payload)
        pending += 1
        rows_uploaded += len(rows)
        if pending >= 200:
            batch.commit()
            batch = client.batch()
            pending = 0
    if pending:
        batch.commit()
    return rows_uploaded


def load_reddit(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Anonymous Reddit scraper for NIFTY 50 tickers.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--subreddits",
        default=",".join(DEFAULT_SUBREDDITS),
        help="Comma-separated subreddit names (no r/ prefix).",
    )
    parser.add_argument("--backfill-days", type=int, default=DEFAULT_BACKFILL_DAYS)
    parser.add_argument("--post-limit", type=int, default=POST_FETCH_LIMIT)
    parser.add_argument(
        "--no-firestore",
        action="store_true",
        help="Skip Firestore writes (parquet only).",
    )
    args = parser.parse_args(argv)

    subs = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    write_firestore = not args.no_firestore and bool(
        os.environ.get("GOOGLE_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )

    df = refresh_reddit(
        output_path=Path(args.output),
        subreddits=subs,
        backfill_days=args.backfill_days,
        post_limit=args.post_limit,
        write_firestore=write_firestore,
        collect=truthy_env("COLLECT_REDDIT", default=True),
    )
    print(json.dumps({"rows": int(len(df))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
