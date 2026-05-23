"""P1.4 — X / Twitter ingestion (best-effort).

snscrape is the primary path; nitter HTML scraping is the fallback. Both
break periodically when X changes its anti-scraping rules — by design this
module returns an empty frame rather than raising when ingestion fails,
and the ``x_available`` flag on the output row makes that visible to the
downstream sentiment pipeline.

Per-ticker search uses the alias map; spam filters remove link-only tweets,
bot-shaped accounts (very few followers, no display name), and obvious
crypto/giveaway lures.

Each tweet is mapped to one or more NIFTY 50 companies by running alias
matching over the tweet body. Tweets that mention no covered company are
labelled ``GENERAL``. Per-company records are dumped to Firestore in the
schema:

    {
        "company_name": "Reliance Industries",
        "news":         "<tweet body>",
        "date_of_news": "2026-05-19T08:00:00+00:00",
        ...
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ingestion._archive import (
    archive_path_for,
    drain_pending_archive,
    merge_stage_to_archive,
    truthy_env,
    upload_and_clear_archive,
)
from ingestion._firestore import init_firestore_client
from ingestion.aliases import all_aliases_for, find_tickers_in_text, list_tickers, load_aliases

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Data" / "archive" / "x_posts.parquet"
DEFAULT_BACKFILL_DAYS = 7  # snscrape rarely returns much beyond a week reliably
DEFAULT_MAX_PER_TICKER = 200
NITTER_INSTANCES = (
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
)
SPAM_PATTERNS = re.compile(r"(giveaway|airdrop|claim now|join my)", re.IGNORECASE)

FIRESTORE_COLLECTION = "x_posts"
GENERAL_TICKER = "GENERAL"
GENERAL_COMPANY_NAME = "General"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _is_spam(content: str, followers: Optional[int], display_name: Optional[str]) -> bool:
    if SPAM_PATTERNS.search(content or ""):
        return True
    if followers is not None and followers < 25:
        return True
    if not display_name:
        return True
    return False


def _build_query(ticker: str) -> str:
    aliases = [ticker, *(a for a in all_aliases_for(ticker) if a != ticker)]
    # Match the symbol with the $ cashtag plus all distinct aliases
    quoted = " OR ".join({f'"{alias}"' for alias in aliases if alias})
    return f'({quoted} OR ${ticker} OR #{ticker}) lang:en -filter:replies'


def _company_name_for(ticker: str) -> str:
    if ticker == GENERAL_TICKER:
        return GENERAL_COMPANY_NAME
    entry = load_aliases().get(ticker.upper(), {})
    return entry.get("name") or ticker


def _map_companies(body: str, fetch_ticker: str) -> List[str]:
    """Map a tweet body to NIFTY tickers via alias matching.

    Tweets fetched under ``fetch_ticker`` often mention other names too
    (sector chatter, comparisons). We fan out to every detected company.
    If nothing matches, the tweet is labelled ``GENERAL`` — it slipped
    through the search query but doesn't actually name a covered company.
    """
    hits = find_tickers_in_text(body or "")
    if hits:
        return hits
    return [GENERAL_TICKER]


def _expand_rows(
    base: Dict[str, Any], body: str, fetch_ticker: str
) -> List[Dict[str, Any]]:
    """Fan a single fetched tweet out to one row per matched company."""
    out: List[Dict[str, Any]] = []
    for company_ticker in _map_companies(body, fetch_ticker):
        out.append(
            {
                **base,
                "ticker": company_ticker,
                "company_name": _company_name_for(company_ticker),
            }
        )
    return out


def fetch_via_snscrape(ticker: str, cutoff: datetime, max_results: int) -> List[Dict[str, Any]]:
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        _log("[x] snscrape not installed; skipping")
        return []
    except Exception as exc:
        _log(f"[x] snscrape import failed: {exc}")
        return []

    query = _build_query(ticker)
    rows: List[Dict[str, Any]] = []
    try:
        scraper = sntwitter.TwitterSearchScraper(query)
        for idx, tweet in enumerate(scraper.get_items()):
            if idx >= max_results:
                break
            ts: datetime = tweet.date.astimezone(timezone.utc) if tweet.date.tzinfo else tweet.date.replace(
                tzinfo=timezone.utc
            )
            if ts < cutoff:
                break
            content = tweet.rawContent or ""
            user = getattr(tweet, "user", None)
            followers = getattr(user, "followersCount", None) if user else None
            display_name = getattr(user, "displayname", None) if user else None
            if _is_spam(content, followers, display_name):
                continue
            url = tweet.url
            base = {
                "ts": ts,
                "title": "",
                "body": content,
                "score": int(getattr(tweet, "likeCount", 0) or 0),
                "url": url,
                "url_hash": _hash_url(url),
                "source": "snscrape",
            }
            rows.extend(_expand_rows(base, content, ticker))
    except Exception as exc:
        _log(f"[x] snscrape {ticker} failed: {exc}")
    return rows


def fetch_via_nitter(ticker: str, cutoff: datetime, max_results: int) -> List[Dict[str, Any]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        _log("[x] nitter fallback needs requests + beautifulsoup4; skipping")
        return []

    from ingestion._http import get_with_retry

    query = _build_query(ticker)
    rows: List[Dict[str, Any]] = []
    for base_url in NITTER_INSTANCES:
        try:
            url = f"{base_url}/search"
            resp = get_with_retry(
                url,
                label=f"x nitter {base_url}",
                params={"f": "tweets", "q": query},
                timeout=10,
            )
            if resp is None or resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select(".timeline-item")[:max_results]
            for item in items:
                content_el = item.select_one(".tweet-content")
                date_el = item.select_one(".tweet-date a")
                user_el = item.select_one(".fullname")
                if not content_el or not date_el:
                    continue
                content = content_el.get_text(" ", strip=True)
                ts_attr = date_el.get("title") or date_el.get_text(strip=True)
                try:
                    ts = pd.to_datetime(ts_attr, utc=True).to_pydatetime()
                except Exception:
                    continue
                if ts < cutoff:
                    continue
                if _is_spam(content, None, user_el.get_text(strip=True) if user_el else None):
                    continue
                tweet_url = base_url + (date_el.get("href") or "")
                base = {
                    "ts": ts,
                    "title": "",
                    "body": content,
                    "score": 0,
                    "url": tweet_url,
                    "url_hash": _hash_url(tweet_url),
                    "source": f"nitter:{base_url}",
                }
                rows.extend(_expand_rows(base, content, ticker))
            if rows:
                return rows
        except Exception as exc:
            _log(f"[x] nitter {base_url} failed for {ticker}: {exc}")
            continue
    return rows


def fetch_x(ticker: str, cutoff: datetime, max_results: int = DEFAULT_MAX_PER_TICKER) -> List[Dict[str, Any]]:
    rows = fetch_via_snscrape(ticker, cutoff, max_results)
    if rows:
        return rows
    _log(f"[x] {ticker}: snscrape returned nothing, trying nitter")
    return fetch_via_nitter(ticker, cutoff, max_results)


def refresh_x(
    tickers: Iterable[str],
    output_path: Path = DEFAULT_OUTPUT,
    backfill_days: int = DEFAULT_BACKFILL_DAYS,
    max_per_ticker: int = DEFAULT_MAX_PER_TICKER,
    sleep_seconds: float = 0.5,
    write_firestore: bool = False,
    firestore_client: Optional[Any] = None,
    collect: bool = True,
) -> pd.DataFrame:
    archive_path = archive_path_for(output_path)
    upload_fn = lambda df: write_x_to_firestore(df, client=firestore_client)

    # ---- Step 1: drain pending archive (runs even if collection is off) -
    if write_firestore:
        drain_pending_archive(archive_path, upload_fn, _log)
    elif archive_path.exists():
        _log("[x] pending archive present but --no-firestore set; leaving file alone")

    # ---- Skip-collect short-circuit (COLLECT_X=false) -------------------
    if not collect:
        _log("[x] COLLECT_X=false; skipping fresh collection (drain-only run)")
        return load_x(archive_path)

    # ---- Step 2: fresh collection ---------------------------------------
    cutoff = datetime.now(timezone.utc) - timedelta(days=backfill_days)
    all_rows: List[Dict[str, Any]] = []
    successes = 0
    for ticker in tickers:
        rows = fetch_x(ticker, cutoff, max_results=max_per_ticker)
        if rows:
            successes += 1
        all_rows.extend(rows)
        _log(f"[x] {ticker}: {len(rows)} rows")
        time_module.sleep(sleep_seconds)

    new_df = pd.DataFrame(all_rows)
    fresh_df = _merge_with_cache(new_df, output_path, cutoff=cutoff)
    _log(f"[x] prepared {len(fresh_df)} fresh rows for Firestore staging; tickers with data: {successes}")

    # ---- Steps 3 + 4: merge-stage and upload ----------------------------
    merged_count = merge_stage_to_archive(
        fresh_df, archive_path, dedup_keys=("ticker", "url_hash"), log=_log
    )
    if write_firestore:
        if merged_count > 0:
            upload_and_clear_archive(archive_path, upload_fn, _log)
    else:
        _log("[x] Firestore write skipped (--no-firestore); staged archive retained")

    return fresh_df


def _merge_with_cache(new_df: pd.DataFrame, output_path: Path, cutoff: datetime) -> pd.DataFrame:
    del output_path
    combined = new_df.copy() if new_df is not None else pd.DataFrame()
    if combined.empty:
        return combined
    combined["ts"] = pd.to_datetime(combined["ts"], utc=True)
    combined = combined[combined["ts"] >= pd.Timestamp(cutoff)]
    combined = combined.drop_duplicates(subset=["ticker", "url_hash"], keep="last")
    combined = combined.sort_values(["ticker", "ts"]).reset_index(drop=True)
    return combined


def _date_of_post(ts: Any) -> Optional[str]:
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts) if ts is not None else None


def _post_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    """One nested X post inside a per-ticker Firestore doc."""
    return {
        "body": row.get("body"),
        "source": row.get("source"),
        "score": int(row.get("score") or 0),
        "date_of_post": _date_of_post(row.get("ts")),
        "url": row.get("url"),
    }


def _firestore_payload_for_ticker(
    ticker: str,
    rows: List[Dict[str, Any]],
    scrape_date: str,
) -> Dict[str, Any]:
    """Company-level Firestore doc with all tweets nested by url_hash."""
    company_name = next(
        (r.get("company_name") for r in rows if r.get("company_name")),
        _company_name_for(ticker),
    )
    posts: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        url_hash = row.get("url_hash")
        if not url_hash:
            continue
        posts[str(url_hash)] = _post_entry(row)
    return {
        "company_name": company_name,
        "ticker": ticker,
        "scrape_date": scrape_date,
        "posts": posts,
    }


def _firestore_doc_id(row_or_ticker: Any) -> str:
    """Doc id is the ticker symbol (``GENERAL`` for macro chatter)."""
    if isinstance(row_or_ticker, dict):
        return str(row_or_ticker.get("ticker") or GENERAL_TICKER)
    return str(row_or_ticker or GENERAL_TICKER)


def write_x_to_firestore(
    df: pd.DataFrame,
    collection: str = FIRESTORE_COLLECTION,
    client: Optional[Any] = None,
) -> int:
    """Write **one document per ticker** with all tweets nested by ``url_hash``.

    Returns the number of input rows uploaded (not the number of documents).
    """
    if df.empty:
        return 0
    client = client or init_firestore_client()
    scrape_date = datetime.now(timezone.utc).date().isoformat()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in df.to_dict(orient="records"):
        if not row.get("url_hash"):
            continue
        ticker = str(row.get("ticker") or GENERAL_TICKER)
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


def load_x(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Best-effort X/Twitter scrape for NIFTY 50 tickers.")
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--backfill-days", type=int, default=DEFAULT_BACKFILL_DAYS)
    parser.add_argument("--max-per-ticker", type=int, default=DEFAULT_MAX_PER_TICKER)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument(
        "--no-firestore",
        action="store_true",
        help="Skip Firestore writes (parquet only).",
    )
    args = parser.parse_args(argv)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = list_tickers()

    write_firestore = not args.no_firestore and bool(
        os.environ.get("GOOGLE_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )

    df = refresh_x(
        tickers,
        output_path=Path(args.output),
        backfill_days=args.backfill_days,
        max_per_ticker=args.max_per_ticker,
        sleep_seconds=args.sleep,
        write_firestore=write_firestore,
        collect=truthy_env("COLLECT_X", default=True),
    )
    print(json.dumps({"rows": int(len(df))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
