"""P1.2 — News ingestion via yfinance ``Ticker(t).news``.

Policy: only headlines from the **last 7 days** are retained. Anything older
than ``now - 7 days`` is dropped at fetch time and pruned from the parquet
cache on the next run. This keeps news features aligned with the short-term
swing-trading horizon — older headlines have already been absorbed into price
and stop adding signal.

yfinance itself returns ~30 days of headlines; we narrow that to 7. Repeated
runs are idempotent thanks to URL-hash deduplication.

Each headline is mapped to one or more NIFTY 50 companies by running alias
matching over the title. Headlines that mention no covered company are
labelled ``GENERAL`` so macro/market news still flows through to the
sentiment pipeline (where it lands in a market-level bucket).

After yfinance returns a headline + URL, we scrape the article URL via
trafilatura to extract the full article body. The body is stored alongside
the headline in the parquet ``content`` column and in Firestore.

Firestore layout: **one document per ticker** (doc id = ``RELIANCE``),
matching the fundamentals shape:

    {
        "company_name": "Reliance Industries",
        "ticker":       "RELIANCE",
        "scrape_date":  "2026-05-22",
        "articles": {
            "<url_hash>": {
                "headline":     "...",
                "content":      "<scraped article body>",
                "date_of_news": "2026-05-21T08:30:00+00:00",
                "source":       "Moneycontrol",
                "url":          "https://..."
            },
            ...
        }
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time as time_module
from concurrent.futures import ThreadPoolExecutor
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
from ingestion.aliases import find_tickers_in_text, list_tickers, load_aliases

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Data" / "archive" / "news.parquet"
DEFAULT_BACKFILL_DAYS = 7  # Hard cap: keep only headlines from the last 7 days.
NSE_SUFFIX = ".NS"

FIRESTORE_COLLECTION = "news"
GENERAL_TICKER = "GENERAL"
GENERAL_COMPANY_NAME = "General"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _company_name_for(ticker: str) -> str:
    if ticker == GENERAL_TICKER:
        return GENERAL_COMPANY_NAME
    entry = load_aliases().get(ticker.upper(), {})
    return entry.get("name") or ticker


def _map_companies(title: str, fetch_ticker: str) -> List[str]:
    """Return the NIFTY tickers a headline is about.

    Strategy: run alias detection on the title. If anything matches, use
    those hits. Otherwise the headline didn't name a covered company → tag
    it as ``GENERAL`` even though yfinance returned it under ``fetch_ticker``
    (yfinance attaches macro headlines to every ticker, which would inflate
    per-company sentiment if we kept the attribution).
    """
    hits = find_tickers_in_text(title or "")
    if hits:
        return hits
    return [GENERAL_TICKER]


def _extract_news_items(symbol: str, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize yfinance news items into our schema.

    Multi-fans-out per matched company: one row per (headline, company).
    """
    out: List[Dict[str, Any]] = []
    for item in raw or []:
        # yfinance shape varies across versions: top-level keys vs. nested under "content"
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = content.get("title") or item.get("title")
        url = (
            content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else None
        ) or content.get("link") or item.get("link")
        provider = (
            content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else None
        ) or content.get("publisher") or item.get("publisher")
        ts_raw = (
            content.get("pubDate")
            or content.get("displayTime")
            or content.get("providerPublishTime")
            or item.get("providerPublishTime")
        )
        ts = _parse_ts(ts_raw)

        if not title or not url or ts is None:
            continue

        title_str = str(title).strip()
        url_str = str(url)
        url_hash = _hash_url(url_str)
        source_str = str(provider).strip() if provider else None

        for company_ticker in _map_companies(title_str, symbol):
            out.append(
                {
                    "ticker": company_ticker,
                    "company_name": _company_name_for(company_ticker),
                    "ts": ts,
                    "title": title_str,
                    "source": source_str,
                    "url": url_str,
                    "url_hash": url_hash,
                }
            )
    return out


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return pd.to_datetime(value, utc=True).to_pydatetime()
        except Exception:
            return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def fetch_news(symbol: str) -> List[Dict[str, Any]]:
    import yfinance as yf

    ticker = yf.Ticker(f"{symbol}{NSE_SUFFIX}")
    try:
        raw = ticker.news or []
    except Exception as exc:
        _log(f"[news] {symbol} failed: {exc}")
        return []
    return _extract_news_items(symbol, raw)


# ---------------------------------------------------------------------------
# Article body scraping (trafilatura)
# ---------------------------------------------------------------------------

_ARTICLE_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def fetch_article_content(url: str, timeout: float = 10.0) -> Optional[str]:
    """Fetch an article URL and return its extracted body text.

    Uses ``requests`` to download HTML (lets us control the User-Agent) and
    ``trafilatura.extract`` to pull the article body out of arbitrary news
    sites. Returns ``None`` on network failure, non-200 response, or when
    trafilatura can't find an article body (paywall, JS-only render, etc.).
    """
    try:
        import trafilatura
    except ImportError as exc:
        _log(f"[news] article scraping disabled (missing dep: {exc})")
        return None

    from ingestion._http import get_with_retry

    headers = {
        "User-Agent": _ARTICLE_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    resp = get_with_retry(
        url,
        label=f"news article {url}",
        headers=headers,
        timeout=timeout,
        allow_redirects=True,
    )
    if resp is None:
        return None
    if resp.status_code != 200:
        _log(f"[news] article {url} returned HTTP {resp.status_code}")
        return None
    try:
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
    except Exception as exc:
        _log(f"[news] article extraction failed for {url}: {exc}")
        return None
    cleaned = (text or "").strip()
    return cleaned or None


def attach_article_content(
    rows: List[Dict[str, Any]],
    max_workers: int = 8,
) -> None:
    """Populate each row's ``content`` field by scraping the article URL.

    URLs are deduped first — a single article fanned out to multiple tickers
    (multi-company headline) is fetched exactly once. Failures leave the
    row's ``content`` as ``None``; the parquet row + Firestore doc still
    carry the headline as a fallback signal.
    """
    if not rows:
        return
    unique_urls = sorted({row["url"] for row in rows if row.get("url")})
    if not unique_urls:
        return
    _log(f"[news] scraping article bodies for {len(unique_urls)} unique URLs")
    contents: Dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for url, body in zip(unique_urls, executor.map(fetch_article_content, unique_urls)):
            contents[url] = body
    with_content = sum(1 for v in contents.values() if v)
    _log(f"[news] extracted body for {with_content}/{len(unique_urls)} URLs")
    for row in rows:
        row["content"] = contents.get(row.get("url"))


def refresh_news(
    tickers: Iterable[str],
    output_path: Path = DEFAULT_OUTPUT,
    backfill_days: int = DEFAULT_BACKFILL_DAYS,
    sleep_seconds: float = 0.2,
    write_firestore: bool = False,
    firestore_client: Optional[Any] = None,
    collect: bool = True,
    scrape_article_bodies: bool = True,
    article_scrape_workers: int = 8,
) -> pd.DataFrame:
    """Archive-aware refresh with fall-through (no data loss on retry).

    Flow (see :mod:`ingestion._archive` for the contract):

    1. **Drain** — try to upload any retained ``Data/archive/news.parquet``
       from a prior failed run. Successful drains delete the file; failures
       retain it.
    2. **Collect** — pulls headlines from yfinance when collection is enabled.
    3. **Merge-stage** — overlays today's just-fetched rows on top of any
       retained archive rows, deduping on ``(ticker, url_hash)``.
    4. **Upload + clear** — pushes the staged archive to Firestore; deletes
       the file on success or retains it for the next run on failure.
    """
    archive_path = archive_path_for(output_path)
    upload_fn = lambda df: write_news_to_firestore(df, client=firestore_client)

    # ---- Step 1: drain any pending archive (runs even if collection is off)
    if write_firestore:
        drain_pending_archive(archive_path, upload_fn, _log)
    elif archive_path.exists():
        _log("[news] pending archive present but --no-firestore set; leaving file alone")

    # ---- Skip-collect short-circuit (COLLECT_NEWS=false) ----------------
    if not collect:
        _log("[news] COLLECT_NEWS=false; skipping fresh collection (drain-only run)")
        return load_news(archive_path)

    # ---- Step 2: fresh collection ---------------------------------------
    cutoff = datetime.now(timezone.utc) - timedelta(days=backfill_days)
    new_rows: List[Dict[str, Any]] = []
    for symbol in tickers:
        items = fetch_news(symbol)
        items = [row for row in items if row["ts"] >= cutoff]
        new_rows.extend(items)
        _log(f"[news] {symbol} fetched {len(items)} items")
        time_module.sleep(sleep_seconds)

    # Step 2b: scrape article body for each unique URL (in-place adds 'content')
    if scrape_article_bodies and new_rows:
        attach_article_content(new_rows, max_workers=article_scrape_workers)
    else:
        for row in new_rows:
            row.setdefault("content", None)

    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        _log("[news] no items fetched in this run")

    fresh_df = _merge_with_cache(new_df, output_path, cutoff=cutoff)
    _log(f"[news] prepared {len(fresh_df)} fresh rows for Firestore staging")

    # ---- Steps 3 + 4: merge-stage and upload ----------------------------
    merged_count = merge_stage_to_archive(
        fresh_df, archive_path, dedup_keys=("ticker", "url_hash"), log=_log
    )
    if write_firestore:
        if merged_count > 0:
            upload_and_clear_archive(archive_path, upload_fn, _log)
    else:
        _log("[news] Firestore write skipped (--no-firestore); staged archive retained")

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


def _date_of_news(ts: Any) -> Optional[str]:
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts) if ts is not None else None


def _article_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    """One nested article inside a per-ticker Firestore doc."""
    return {
        "headline": row.get("title"),
        "content": row.get("content"),
        "date_of_news": _date_of_news(row.get("ts")),
        "source": row.get("source"),
        "url": row.get("url"),
    }


def _firestore_payload_for_ticker(
    ticker: str,
    rows: List[Dict[str, Any]],
    scrape_date: str,
) -> Dict[str, Any]:
    """Build the company-level Firestore document.

    Shape mirrors ``ingestion.fundamentals`` (one doc per ticker, with
    related items nested by hash).
    """
    company_name = next(
        (r.get("company_name") for r in rows if r.get("company_name")),
        _company_name_for(ticker),
    )
    articles: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        url_hash = row.get("url_hash")
        if not url_hash:
            continue
        articles[str(url_hash)] = _article_entry(row)
    return {
        "company_name": company_name,
        "ticker": ticker,
        "scrape_date": scrape_date,
        "articles": articles,
    }


def _firestore_doc_id(row_or_ticker: Any) -> str:
    """Doc id is just the ticker symbol (or ``GENERAL`` for macro headlines)."""
    if isinstance(row_or_ticker, dict):
        return str(row_or_ticker.get("ticker") or GENERAL_TICKER)
    return str(row_or_ticker or GENERAL_TICKER)


def write_news_to_firestore(
    df: pd.DataFrame,
    collection: str = FIRESTORE_COLLECTION,
    client: Optional[Any] = None,
) -> int:
    """Write **one document per ticker** with all articles nested by ``url_hash``.

    Returns the number of input rows uploaded (not the number of documents).
    The archive layer compares this to ``len(df)`` to decide whether to
    delete the staged archive file.
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


def load_news(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh yfinance news headlines.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (defaults to NIFTY 50).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--backfill-days", type=int, default=DEFAULT_BACKFILL_DAYS)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument(
        "--no-firestore",
        action="store_true",
        help="Skip Firestore writes (parquet only).",
    )
    parser.add_argument(
        "--no-article-bodies",
        action="store_true",
        help="Skip the second-stage article scrape (headlines + URLs only).",
    )
    parser.add_argument(
        "--article-scrape-workers",
        type=int,
        default=8,
        help="Thread pool size for the article-body scrape (default: 8).",
    )
    args = parser.parse_args(argv)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = list_tickers()

    # Default to writing Firestore when service-account credentials are available.
    write_firestore = not args.no_firestore and bool(
        os.environ.get("GOOGLE_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )

    df = refresh_news(
        tickers,
        output_path=Path(args.output),
        backfill_days=args.backfill_days,
        sleep_seconds=args.sleep,
        write_firestore=write_firestore,
        collect=truthy_env("COLLECT_NEWS", default=True),
        scrape_article_bodies=not args.no_article_bodies,
        article_scrape_workers=args.article_scrape_workers,
    )
    print(json.dumps({"rows": int(len(df))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
