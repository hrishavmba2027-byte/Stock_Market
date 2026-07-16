"""Historical news scraper for backtesting (Wayback CDX).

Adapted from the FRM_Framework ingestion spec (DATA_SCRAPING_INGESTION.md), with
these project-specific changes:

* **Entities are company + sector** (no separate GENERAL/macro bucket, per the
  backtest decision). Company targets use alias terms; sector targets use the
  curated sector terms from :mod:`ingestion.sectors`.
* **Aggregation is every ``news_interval_days`` (7)**, NOT monthly — matching
  production ``news_lookback_days``.
* Collected news is stored **entirely locally** in a single Excel workbook
  (``news_workbook_path``) with three sheets — ``News`` (raw articles),
  ``Sentiment`` (per-window aggregates the simulation reads) and ``Manifest``
  ((target, year) checkpoints) — and is **never removed**. A re-run skips any
  ``(target, year)`` already in the manifest; new years / new stocks append.
  There is no Firestore or Google-Sheet dependency anywhere in the backtest.

Pipeline (all cache-first + resumable):
  Stage 1  CDX URL discovery per (target, year)  → live fetch → Wayback fallback
  Stage 2  trafilatura extraction (title, body, date)
  Stage 3  FinBERT scoring + 7-day-interval aggregation
  Stage 4  append to the local news workbook; checkpoint (target, year)

Why Wayback CDX: it indexes back to ~2015 with no recency window (unlike GDELT's
~3 months or ``yfinance.Ticker().news``'s ~30 days). Broad domain scans time out,
so discovery uses hand-curated per-domain article-path prefixes — seeded here and
extensible via ``<data_dir>/news_prefixes.json`` (see :func:`prefixes_for_target`).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.config.backtest_settings import BacktestSettings, get_backtest_settings
from ingestion.aliases import all_aliases_for, list_tickers, load_aliases
from ingestion.sectors import SECTOR_TERMS, list_sectors, sector_entity_id

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"

# Indian financial-news domains, each mapped to the specific news **section
# prefixes** we scan on Wayback CDX. The section prefix (e.g. ".../markets") is
# what makes CDX ``matchType=prefix`` fast — it bounds the index range — while a
# per-target regex ``filter=original:`` (built in ``target_url_regex``) selects
# the company/sector within that section. Do NOT put a ``*`` glob in the middle
# of a prefix: CDX ``matchType=prefix`` treats the string literally, so
# ``domain/*slug*`` matches nothing (the old bug that returned 0 candidate URLs).
#
# Sections were chosen empirically for dense historical coverage + fast CDX
# scans (broad prefixes like ".../article" time out). Five curated Indian
# financial desks:
#   Economic Times, Moneycontrol  — deepest archives, highest per-stock volume.
#   Business Standard, LiveMint, Financial Express — added for enrichment /
#   corroboration across independent desks.
NEWS_DOMAIN_SECTIONS = {
    "economictimes.indiatimes.com": ["markets", "industry"],
    "moneycontrol.com":             ["news"],
    "business-standard.com":        ["markets", "companies"],
    "livemint.com":                 ["market", "companies"],
    "financialexpress.com":         ["market", "industry/banking-finance"],
}

# Back-compat / convenience: the bare domain list.
NEWS_DOMAINS = list(NEWS_DOMAIN_SECTIONS.keys())

_LISTING_MARKERS = ("/page-", "/lite/", "/topic/", "/tag/", "/tags/", "/author/", "/videos/")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# trafilatura is imported lazily and cached: on lxml>=5.2 a bare ``import
# trafilatura`` raises ImportError unless the split-out ``lxml_html_clean`` pkg
# is present, so we keep the *actual* exception (not a misleading "not
# installed") to surface an actionable message once, up front.
_TRAFILATURA: Any = None


def _load_trafilatura() -> Any:
    """Return the ``trafilatura`` module, or the Exception raised importing it."""
    global _TRAFILATURA
    if _TRAFILATURA is None:
        try:
            import trafilatura
            _TRAFILATURA = trafilatura
        except Exception as exc:  # noqa: BLE001 — ImportError or broken transitive dep
            _TRAFILATURA = exc
    return _TRAFILATURA


def trafilatura_available() -> bool:
    return not isinstance(_load_trafilatura(), Exception)


_TRAFILATURA_HINT = (
    "install it with:  pip install -U trafilatura lxml_html_clean   "
    "(lxml>=5.2 moved lxml.html.clean into the separate lxml_html_clean package, "
    "which trafilatura imports)"
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no network)
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Strip query/fragment and a trailing ``/amp`` so duplicates collapse."""
    if not url:
        return ""
    u = url.split("#", 1)[0].split("?", 1)[0]
    for suffix in ("/amp", "/amp/", "amp/"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    return u.rstrip("/")


def is_article_url(url: str) -> bool:
    """Reject listing/pagination/section pages; keep real article URLs."""
    if not url:
        return False
    low = url.lower()
    if any(marker in low for marker in _LISTING_MARKERS):
        return False
    # Require enough path depth to be an article (domain + >=2 path segments).
    path = re.sub(r"^https?://[^/]+", "", low).strip("/")
    segments = [s for s in path.split("/") if s]
    return len(segments) >= 2


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def _cdx_ts_to_iso(ts: str) -> Optional[str]:
    """``YYYYMMDDhhmmss`` Wayback timestamp → ``YYYY-MM-DD`` (or None if unusable).

    The CDX capture timestamp is a reliable, **PIT-safe** date fallback when
    trafilatura can't parse a publish date: it is when the article was archived,
    i.e. always *on or after* publication — so bucketing by it never dates an
    article earlier than it existed (no leakage), it only lands it slightly later.
    """
    s = re.sub(r"\D", "", str(ts or ""))
    if len(s) < 8:
        return None
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def target_url_regex(terms: Iterable[str]) -> Optional[str]:
    """Build a CDX ``filter=original:`` regex that matches a target's URLs.

    Every alias/term is slugified (``HDFC Bank`` → ``hdfc-bank``) and OR-joined,
    longest-first so the alternation prefers the most specific slug. The result
    is wrapped in ``.*(...).*`` so it matches the slug anywhere in the article
    URL — which is where these desks put the company/sector token. Slugs shorter
    than 3 chars are dropped (too noisy). Returns ``None`` if nothing usable.

    Note: CDX regex does not handle look-around efficiently (it times out), so we
    keep a plain alternation and rely on the section prefix to bound the scan.
    """
    slugs: List[str] = []
    for term in terms:
        s = _slug(term)
        if len(s) >= 3 and s not in slugs:
            slugs.append(s)
    if not slugs:
        return None
    slugs.sort(key=len, reverse=True)
    return ".*(" + "|".join(slugs) + ").*"


def prefixes_for_target(
    target: Dict[str, Any],
    *,
    override_map: Optional[Dict[str, List[str]]] = None,
) -> List[Tuple[str, Optional[str]]]:
    """``(cdx_prefix, url_regex)`` pairs to query for a target.

    Each pair is a fast **section prefix** (``domain/section``, for CDX
    ``matchType=prefix``) plus a **regex** that selects this target's articles
    within that section (``filter=original:`` on the CDX side). Prefix bounds the
    scan; regex isolates the company/sector — the two together replace the old
    (broken) mid-path ``*slug*`` glob.

    Precedence: an explicit ``override_map`` entry (curated JSON, keyed by
    display name) wins and is used verbatim with **no** regex filter (curated
    prefixes are assumed already target-specific). Curate
    ``<data_dir>/news_prefixes.json`` (``{target: [prefixes]}``) for precision.
    """
    display_name = target.get("display_name") or target.get("id") or ""
    if override_map and display_name in override_map:
        return [(p, None) for p in override_map[display_name]]
    regex = target_url_regex(target.get("terms") or [display_name])
    pairs: List[Tuple[str, Optional[str]]] = []
    for domain, sections in NEWS_DOMAIN_SECTIONS.items():
        for section in sections:
            pairs.append((f"{domain}/{section}", regex))
    return pairs


def interval_windows(start: date, end: date, interval_days: int) -> List[Tuple[date, date]]:
    """Contiguous ``interval_days``-wide [win_start, win_end] windows over the span.

    This is the 7-day analogue of the spec's monthly grid: news is aggregated
    into these windows instead of calendar months.
    """
    if interval_days < 1:
        raise ValueError("interval_days must be >= 1")
    windows: List[Tuple[date, date]] = []
    cur = start
    step = timedelta(days=interval_days)
    while cur <= end:
        win_end = min(cur + timedelta(days=interval_days - 1), end)
        windows.append((cur, win_end))
        cur = cur + step
    return windows


def assign_window(article_date: date, windows: List[Tuple[date, date]]) -> Optional[date]:
    """Return the window-end date an article falls into, or None if outside."""
    for win_start, win_end in windows:
        if win_start <= article_date <= win_end:
            return win_end
    return None


def aggregate_by_window(
    scored_rows: List[Dict[str, Any]],
    windows: List[Tuple[date, date]],
) -> Dict[str, Dict[str, Any]]:
    """Mean sentiment per ``interval_days`` window.

    ``scored_rows`` items need ``article_date`` (date/ISO str) and ``sentiment``
    (float in [-1, 1]). Returns ``{window_end_iso: {mean_sentiment, n, ...}}``.
    """
    buckets: Dict[str, List[float]] = {}
    for row in scored_rows:
        ad = row.get("article_date")
        if ad is None:
            continue
        if not isinstance(ad, date):
            try:
                ad = datetime.fromisoformat(str(ad)[:10]).date()
            except ValueError:
                continue
        win_end = assign_window(ad, windows)
        if win_end is None:
            continue
        score = row.get("sentiment")
        if score is None:
            continue
        buckets.setdefault(win_end.isoformat(), []).append(float(score))
    out: Dict[str, Dict[str, Any]] = {}
    for win_end_iso, scores in buckets.items():
        n = len(scores)
        mean = sum(scores) / n if n else 0.0
        pos = sum(1 for s in scores if s > 0.1)
        neg = sum(1 for s in scores if s < -0.1)
        out[win_end_iso] = {
            "mean_sentiment": mean,
            "n": n,
            "pos_share": pos / n if n else 0.0,
            "neg_share": neg / n if n else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def build_targets(tickers: Iterable[str]) -> List[Dict[str, Any]]:
    """Company targets (from tickers) + sector targets (company + sector only)."""
    aliases = load_aliases()
    targets: List[Dict[str, Any]] = []
    sectors_seen: List[str] = []
    for t in tickers:
        entry = aliases.get(t.upper(), {})
        targets.append({
            "kind": "company",
            "id": t.upper(),
            "display_name": entry.get("name") or t,
            "terms": all_aliases_for(t),
        })
        sec = entry.get("sector")
        if sec and sec != "Unknown" and sec not in sectors_seen:
            sectors_seen.append(sec)
    for sec in sectors_seen:
        targets.append({
            "kind": "sector",
            "id": sector_entity_id(sec),
            "display_name": sec,
            "terms": SECTOR_TERMS.get(sec, [sec]),
        })
    return targets


# ---------------------------------------------------------------------------
# Manifest / checkpoint (local news workbook)
# ---------------------------------------------------------------------------

def load_collected_manifest(workbook_path: Any) -> set:
    """Return the set of ``(target_id, year)`` pairs already in the workbook."""
    done: set = set()
    sheets = _read_workbook(workbook_path)
    man = sheets.get(MANIFEST_SHEET)
    if man is None or man.empty or not {"target_id", "year"}.issubset(man.columns):
        return done
    for _, row in man.iterrows():
        try:
            done.add((str(row["target_id"]), int(row["year"])))
        except (TypeError, ValueError):
            continue
    return done


# ---------------------------------------------------------------------------
# Network stages (thin; structured for real runs, mocked in tests)
# ---------------------------------------------------------------------------

def cdx_discover(
    prefix: str,
    year: int,
    settings: BacktestSettings,
    url_regex: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Query the Wayback CDX API for a section ``prefix`` within a year.

    ``url_regex`` (optional) is applied as a CDX ``filter=original:`` so only
    URLs matching the target (company/sector) are returned — this is what makes
    the broad section prefix specific without a mid-path glob. Returns
    ``[(url, cdx_timestamp), ...]``. Polite: pause between calls, retry, long
    wait on HTTP 429.
    """
    import requests

    filters = ["statuscode:200", "mimetype:text/html"]
    if url_regex:
        filters.append(f"original:{url_regex}")
    params = {
        "url": prefix,
        "output": "json",
        "matchType": "prefix",
        "collapse": "urlkey",
        "filter": filters,
        "from": f"{year}0101",
        "to": f"{year}1231",
        "fl": "original,timestamp",
        "limit": "20000",
    }
    for attempt in range(1, settings.cdx_retries + 2):
        try:
            resp = requests.get(CDX_ENDPOINT, params=params, timeout=60)
            if resp.status_code == 429:
                _log(f"[backtest-news] CDX 429 for {prefix} {year}; waiting {settings.cdx_rate_limit_wait_seconds}s")
                time.sleep(settings.cdx_rate_limit_wait_seconds)
                continue
            resp.raise_for_status()
            rows = resp.json()
            out: List[Tuple[str, str]] = []
            for row in rows[1:] if rows and isinstance(rows[0], list) else []:
                if len(row) >= 2:
                    out.append((row[0], row[1]))
            return out
        except Exception as exc:
            _log(f"[backtest-news] CDX attempt {attempt} failed for {prefix} {year}: {exc}")
            # "Connection refused" / "Max retries" / read-timeout ⇒ Wayback is
            # throttling us; a long cooldown recovers far better than a 1–4s pause.
            msg = str(exc).lower()
            if any(k in msg for k in ("refused", "max retries", "timed out", "timeout", "connection")):
                time.sleep(settings.cdx_rate_limit_wait_seconds)
            else:
                time.sleep(settings.cdx_request_pause_seconds * attempt)
    return []


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
}


def fetch_article(url: str, cdx_timestamp: str, settings: BacktestSettings) -> Optional[Dict[str, Any]]:
    """Live-first, Wayback-fallback fetch + trafilatura extraction."""
    import requests

    trafilatura = _load_trafilatura()
    if isinstance(trafilatura, Exception):  # unavailable — caught up-front in run()
        return None

    def _extract(html: str) -> Optional[Dict[str, Any]]:
        raw = trafilatura.extract(html, output_format="json", favor_recall=True,
                                  include_comments=False, include_tables=False)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return {
            "title": (data.get("title") or "").strip(),
            "content": (data.get("text") or "").strip(),
            "article_date": data.get("date"),
        }

    candidates = [url, f"https://web.archive.org/web/{cdx_timestamp}/{url}"]
    for attempt_url in candidates:
        for attempt in range(1, settings.news_fetch_retries + 1):
            try:
                resp = requests.get(attempt_url, headers=_BROWSER_HEADERS, timeout=20, allow_redirects=True)
                if resp.status_code == 200 and resp.text:
                    extracted = _extract(resp.text)
                    if extracted and (extracted["title"] or extracted["content"]):
                        return extracted
                break  # non-200 → try next candidate
            except Exception:
                time.sleep(1.0)
    return None


# ---------------------------------------------------------------------------
# Local workbook storage (append-only across a single .xlsx)
# ---------------------------------------------------------------------------

NEWS_SHEET = "News"
SENTIMENT_SHEET = "Sentiment"
MANIFEST_SHEET = "Manifest"


def _read_workbook(path: Any) -> Dict[str, Any]:
    """Return ``{sheet_name: DataFrame}`` for an existing workbook, else ``{}``."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return {}
    try:
        xl = pd.ExcelFile(p)
        return {name: xl.parse(name) for name in xl.sheet_names}
    except Exception as exc:  # pragma: no cover - corrupt/locked file
        _log(f"[backtest-news] could not read {p}: {exc}")
        return {}


def _write_workbook(path: Any, sheets: Dict[str, Any]) -> None:
    """Atomically write ``{sheet_name: DataFrame}`` to ``path``."""
    import os

    import pandas as pd

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".xlsx.tmp")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        written = False
        for name, df in sheets.items():
            frame = df if (df is not None and not df.empty) else pd.DataFrame()
            frame.to_excel(writer, sheet_name=str(name)[:31], index=False)
            written = True
        if not written:
            pd.DataFrame().to_excel(writer, sheet_name=NEWS_SHEET, index=False)
    os.replace(tmp, p)


def _news_rows(target: Dict[str, Any], year: int, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for art in articles:
        url = art.get("url")
        if not url:
            continue
        rows.append({
            "entity_id": target["id"],
            "kind": target["kind"],
            "year": int(year),
            "article_date": art.get("article_date"),
            "headline": art.get("title"),
            "url": url,
            "content": (art.get("content") or "")[:2000],
        })
    return rows


def _sentiment_rows(
    target: Dict[str, Any],
    windows_agg: Dict[str, Dict[str, Any]],
    interval_days: int,
) -> List[Dict[str, Any]]:
    """Flatten per-window aggregates into the schema the simulation reads.

    The backtest aggregates at a single ``news_interval_days`` window, so the 3-day
    and 7-day fields carry the same window mean (there is only one horizon).
    """
    rows: List[Dict[str, Any]] = []
    for win_end, agg in windows_agg.items():
        mean = float(agg.get("mean_sentiment", 0.0))
        n = int(agg.get("n", 0))
        rows.append({
            "entity_id": target["id"],
            "kind": target["kind"],
            "as_of_date": win_end,
            "source": "backtest_cdx",
            "sent_mean_3d": mean,
            "sent_mean_7d": mean,
            "sent_pos_share": float(agg.get("pos_share", 0.0)),
            "sent_neg_share": float(agg.get("neg_share", 0.0)),
            "n_3d": n,
            "n_7d": n,
            "window_days": int(interval_days),
        })
    return rows


def store_to_workbook(
    workbook_path: Any,
    target: Dict[str, Any],
    year: int,
    articles: List[Dict[str, Any]],
    windows_agg: Dict[str, Dict[str, Any]],
    *,
    interval_days: int = 7,
) -> None:
    """Append one ``(target, year)``'s news + sentiment + checkpoint to the workbook.

    Read-modify-write of the whole workbook (fine for a batch collection job, never
    on the simulation hot path). De-duplicates: articles by URL, sentiment by
    ``(entity_id, as_of_date)``, manifest by ``(target_id, year)`` — so re-running a
    pair overwrites rather than double-counts.
    """
    import pandas as pd

    sheets = _read_workbook(workbook_path)

    news = pd.concat(
        [sheets.get(NEWS_SHEET, pd.DataFrame()), pd.DataFrame(_news_rows(target, year, articles))],
        ignore_index=True,
    )
    if "url" in news.columns:
        news = news.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)

    sent = pd.concat(
        [sheets.get(SENTIMENT_SHEET, pd.DataFrame()),
         pd.DataFrame(_sentiment_rows(target, windows_agg, interval_days))],
        ignore_index=True,
    )
    if {"entity_id", "as_of_date"}.issubset(sent.columns):
        sent = sent.drop_duplicates(subset=["entity_id", "as_of_date"], keep="last").reset_index(drop=True)

    manifest = pd.concat(
        [sheets.get(MANIFEST_SHEET, pd.DataFrame()),
         pd.DataFrame([{
             "target_id": target["id"],
             "year": int(year),
             "n_articles": len(articles),
             "collected_at": datetime.now(timezone.utc).isoformat(),
         }])],
        ignore_index=True,
    )
    if {"target_id", "year"}.issubset(manifest.columns):
        manifest = manifest.drop_duplicates(subset=["target_id", "year"], keep="last").reset_index(drop=True)

    _write_workbook(workbook_path, {
        NEWS_SHEET: news,
        SENTIMENT_SHEET: sent,
        MANIFEST_SHEET: manifest,
    })


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_override_map(settings: BacktestSettings) -> Dict[str, List[str]]:
    path = settings.data_dir / "news_prefixes.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            _log(f"[backtest-news] could not read {path}: {exc}")
    return {}


def collect_target_year(
    target: Dict[str, Any],
    year: int,
    settings: BacktestSettings,
    scorer: Any,
    override_map: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Discover → fetch → score → aggregate one (target, year). Returns
    ``(articles, per_window_sentiment)``."""
    seen_urls: set = set()
    discovered: List[Tuple[str, str]] = []
    for prefix, url_regex in prefixes_for_target(target, override_map=override_map):
        for url, ts in cdx_discover(prefix, year, settings, url_regex=url_regex):
            norm = normalize_url(url)
            if norm and norm not in seen_urls and is_article_url(norm):
                seen_urls.add(norm)
                discovered.append((norm, ts))
        time.sleep(settings.cdx_request_pause_seconds)
    _log(f"[backtest-news] {target['id']} {year}: {len(discovered)} candidate URLs")

    discovered_ts = {url: ts for url, ts in discovered}
    articles: List[Dict[str, Any]] = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=settings.news_content_workers) as pool:
        futures = {pool.submit(fetch_article, url, ts, settings): url for url, ts in discovered}
        for fut in futures:
            url = futures[fut]
            try:
                extracted = fut.result()
            except Exception:
                extracted = None
            if extracted and (extracted.get("title") or extracted.get("content")):
                extracted["url"] = url
                # Fall back to the CDX capture date when trafilatura can't parse a
                # publish date — else the article is dropped from sentiment
                # aggregation (which buckets strictly by article_date).
                if not extracted.get("article_date"):
                    ts = discovered_ts.get(url)
                    extracted["article_date"] = _cdx_ts_to_iso(ts)
                articles.append(extracted)

    windows = interval_windows(date(year, 1, 1), date(year, 12, 31), settings.news_interval_days)
    scored = _score_articles(articles, scorer)
    windows_agg = aggregate_by_window(scored, windows)
    return articles, windows_agg


def _score_articles(articles: List[Dict[str, Any]], scorer: Any) -> List[Dict[str, Any]]:
    if not articles:
        return []
    texts = [f"{a.get('title') or ''}. {(a.get('content') or '')[:2000]}" for a in articles]
    frame = scorer.score(texts)
    scored: List[Dict[str, Any]] = []
    for art, (_, row) in zip(articles, frame.iterrows()):
        scored.append({
            "article_date": art.get("article_date") or f"{art.get('year', '')}",
            "sentiment": float(row.get("polarity", 0.0)),
        })
    return scored


def run(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Backtest historical news scraper (Wayback CDX) → local workbook.")
    parser.add_argument("--tickers", default=None, help="Comma-separated subset (default: backtest universe).")
    parser.add_argument("--years", default=None, help="Comma-separated years (default: backtest window).")
    parser.add_argument("--no-store", action="store_true", help="Discover/score only; skip workbook writes.")
    parser.add_argument("--workbook", default=None, help="Override the news workbook path.")
    parser.add_argument("--limit-targets", type=int, default=None, help="Cap number of targets (smoke tests).")
    args = parser.parse_args(argv)

    settings = get_backtest_settings()
    settings.ensure_dirs()
    if not settings.enabled:
        _log("[backtest-news] BACKTEST_ENABLED is false — aborting")
        return {"status": "disabled"}

    # Pre-flight: article extraction is useless without trafilatura. Fail loudly
    # NOW (before any CDX calls or checkpoints) rather than discovering URLs,
    # fetching them, failing every extraction, and falsely marking (target, year)
    # as collected with 0 articles.
    traf = _load_trafilatura()
    if isinstance(traf, Exception):
        _log(f"[backtest-news] ABORT: trafilatura unavailable ({traf}); {_TRAFILATURA_HINT}")
        return {"status": "error", "reason": f"trafilatura unavailable: {traf}"}

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else settings.resolved_tickers()
    )
    years = (
        [int(y) for y in args.years.split(",")] if args.years else settings.news_years()
    )
    targets = build_targets(tickers)
    if args.limit_targets is not None:
        targets = targets[: args.limit_targets]

    override_map = _load_override_map(settings)

    write_store = not args.no_store
    workbook_path = Path(args.workbook) if args.workbook else Path(settings.news_workbook_path)
    collected = load_collected_manifest(workbook_path) if write_store else set()

    scorer = None  # lazy — only build FinBERT if we actually discovered articles
    processed, skipped = 0, 0
    for target in targets:
        for year in years:
            if (target["id"], year) in collected:
                skipped += 1
                _log(f"[backtest-news] skip {target['id']} {year} (already collected)")
                continue
            if scorer is None:
                from features.sentiment import FinBertScorer
                scorer = FinBertScorer(model_name=settings.sentiment_model)
            articles, windows_agg = collect_target_year(target, year, settings, scorer, override_map)
            if write_store:
                store_to_workbook(workbook_path, target, year, articles, windows_agg,
                                  interval_days=settings.news_interval_days)
            processed += 1

    summary = {
        "status": "ok",
        "targets": len(targets),
        "years": years,
        "processed_pairs": processed,
        "skipped_pairs": skipped,
        "workbook": str(workbook_path) if write_store else None,
    }
    print(json.dumps(summary, default=str))
    return summary


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result.get("status") in ("ok", "disabled") else 1)
