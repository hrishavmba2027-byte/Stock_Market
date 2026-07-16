from datetime import datetime, timedelta, timezone

import pandas as pd

from ingestion import news_ingest


def test_default_backfill_window_is_one_week():
    """News policy: only retain headlines from the last 7 days."""
    assert news_ingest.DEFAULT_BACKFILL_DAYS == 7


def test_default_backfill_days_reads_config(monkeypatch):
    """The live window comes from NEWS_LOOKBACK_DAYS, not the constant."""
    monkeypatch.setenv("NEWS_LOOKBACK_DAYS", "10")
    assert news_ingest.default_backfill_days() == 10
    monkeypatch.delenv("NEWS_LOOKBACK_DAYS", raising=False)
    assert news_ingest.default_backfill_days() == 7


def test_merge_with_cache_drops_items_older_than_one_week(tmp_path):
    output = tmp_path / "news.parquet"
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=news_ingest.DEFAULT_BACKFILL_DAYS)
    df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "ts": now - timedelta(days=2),  # within window
                "title": "fresh",
                "source": "X",
                "url": "https://a",
                "url_hash": "ha",
            },
            {
                "ticker": "RELIANCE",
                "ts": now - timedelta(days=14),  # outside 7-day window
                "title": "stale",
                "source": "X",
                "url": "https://b",
                "url_hash": "hb",
            },
        ]
    )
    out = news_ingest._merge_with_cache(df, output, cutoff=cutoff)
    assert len(out) == 1
    assert out.iloc[0]["title"] == "fresh"


def test_extract_news_items_handles_new_yfinance_shape():
    raw = [
        {
            "content": {
                "title": "Reliance Q3 results beat estimates",
                "canonicalUrl": {"url": "https://example.com/reliance-q3"},
                "provider": {"displayName": "Moneycontrol"},
                "pubDate": "2026-01-15T08:00:00Z",
            }
        }
    ]
    items = news_ingest._extract_news_items("RELIANCE", raw)
    # Company row + its implied sector row (ENABLE_SECTOR_NEWS default on).
    company_items = [i for i in items if i["ticker"] == "RELIANCE"]
    assert len(company_items) == 1
    item = company_items[0]
    assert item["ticker"] == "RELIANCE"
    assert item["title"].startswith("Reliance Q3")
    assert item["url"] == "https://example.com/reliance-q3"
    assert item["source"] == "Moneycontrol"
    assert item["url_hash"]
    assert any(news_ingest.is_sector_entity(i["ticker"]) for i in items)


def test_extract_news_items_handles_legacy_shape():
    raw = [
        {
            "title": "TCS deals momentum",
            "link": "https://example.com/tcs-deals",
            "publisher": "BQ Prime",
            "providerPublishTime": 1737000000,  # 2025-01-16 UTC
        }
    ]
    items = news_ingest._extract_news_items("TCS", raw)
    company_items = [i for i in items if i["ticker"] == "TCS"]
    assert len(company_items) == 1
    assert company_items[0]["source"] == "BQ Prime"


def test_extract_news_items_skips_incomplete_rows():
    raw = [{"title": "No URL here"}, {"link": "https://e", "providerPublishTime": 0}]
    assert news_ingest._extract_news_items("INFY", raw) == []


def test_extract_news_items_populates_company_name():
    raw = [
        {
            "content": {
                "title": "Reliance Q3 results beat estimates",
                "canonicalUrl": {"url": "https://example.com/r-q3"},
                "provider": {"displayName": "Moneycontrol"},
                "pubDate": "2026-01-15T08:00:00Z",
            }
        }
    ]
    items = news_ingest._extract_news_items("RELIANCE", raw)
    company_items = [i for i in items if i["ticker"] == "RELIANCE"]
    assert len(company_items) == 1
    assert company_items[0]["ticker"] == "RELIANCE"
    assert company_items[0]["company_name"] == "Reliance Industries"


def test_extract_news_items_marks_macro_news_as_general():
    raw = [
        {
            "content": {
                "title": "NIFTY surges to record high amid foreign inflows",
                "canonicalUrl": {"url": "https://example.com/nifty-rally"},
                "provider": {"displayName": "ET"},
                "pubDate": "2026-01-15T08:00:00Z",
            }
        }
    ]
    # Fetched under RELIANCE but title names no covered company → GENERAL.
    items = news_ingest._extract_news_items("RELIANCE", raw)
    assert len(items) == 1
    assert items[0]["ticker"] == news_ingest.GENERAL_TICKER
    assert items[0]["company_name"] == news_ingest.GENERAL_COMPANY_NAME


def test_extract_news_items_fans_out_to_multiple_companies():
    raw = [
        {
            "content": {
                "title": "TCS and Infosys both reported strong Q3 deal wins",
                "canonicalUrl": {"url": "https://example.com/it-q3"},
                "provider": {"displayName": "BQ"},
                "pubDate": "2026-01-15T08:00:00Z",
            }
        }
    ]
    items = news_ingest._extract_news_items("TCS", raw)
    company_tickers = {row["ticker"] for row in items if news_ingest._entity_kind(row["ticker"]) == "company"}
    assert company_tickers == {"TCS", "INFY"}
    # Each fanned-out company row has the matching company name
    for row in items:
        if row["ticker"] == "TCS":
            assert row["company_name"] == "Tata Consultancy Services"
        elif row["ticker"] == "INFY":
            assert row["company_name"] == "Infosys"
    # Both companies share the IT sector → a single IT sector row is present.
    assert any(news_ingest.is_sector_entity(row["ticker"]) for row in items)


def test_article_entry_carries_headline_content_and_metadata():
    row = {
        "title": "Reliance Q3 results beat",
        "content": "Full article body scraped from Moneycontrol...",
        "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
        "source": "Moneycontrol",
        "url": "https://example.com/r-q3",
    }
    entry = news_ingest._article_entry(row)
    assert entry["headline"] == "Reliance Q3 results beat"
    assert entry["content"].startswith("Full article body")
    assert entry["date_of_news"].startswith("2026-01-15")
    assert entry["source"] == "Moneycontrol"
    assert entry["url"] == "https://example.com/r-q3"


def test_firestore_payload_for_ticker_nests_articles_by_url_hash():
    rows = [
        {
            "ticker": "RELIANCE",
            "company_name": "Reliance Industries",
            "title": "RIL beats estimates",
            "content": "Body A",
            "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
            "source": "ET",
            "url": "https://example.com/a",
            "url_hash": "h1",
        },
        {
            "ticker": "RELIANCE",
            "company_name": "Reliance Industries",
            "title": "RIL retail margin story",
            "content": "Body B",
            "ts": pd.Timestamp("2026-01-15T10:00:00", tz="UTC"),
            "source": "Moneycontrol",
            "url": "https://example.com/b",
            "url_hash": "h2",
        },
    ]
    payload = news_ingest._firestore_payload_for_ticker(
        "RELIANCE", rows, scrape_date="2026-01-16"
    )
    assert payload["company_name"] == "Reliance Industries"
    assert payload["ticker"] == "RELIANCE"
    assert payload["scrape_date"] == "2026-01-16"
    assert set(payload["articles"].keys()) == {"h1", "h2"}
    assert payload["articles"]["h1"]["headline"] == "RIL beats estimates"
    assert payload["articles"]["h1"]["content"] == "Body A"
    assert payload["articles"]["h2"]["source"] == "Moneycontrol"


def test_firestore_doc_id_is_just_the_ticker():
    # Accept either a row dict or a bare ticker string.
    assert news_ingest._firestore_doc_id({"ticker": "RELIANCE"}) == "RELIANCE"
    assert news_ingest._firestore_doc_id("TCS") == "TCS"
    # Missing ticker → GENERAL fallback
    assert news_ingest._firestore_doc_id({}) == news_ingest.GENERAL_TICKER


class _FakeSnap:
    def __init__(self, reference):
        self.reference = reference


class _FakeFirestoreDocRef:
    def __init__(self, store, doc_id):
        self._store = store
        self.doc_id = doc_id


class _FakeFirestoreBatch:
    """Deferred-commit batch that mutates the backing store (so wipe works)."""

    def __init__(self, writes):
        self._writes = writes
        self._ops = []

    def set(self, doc_ref, payload):
        self._ops.append(("set", doc_ref, payload))

    def delete(self, doc_ref):
        self._ops.append(("delete", doc_ref, None))

    def commit(self):
        for op, ref, payload in self._ops:
            if op == "set":
                ref._store[ref.doc_id] = payload
                self._writes.append((ref.doc_id, payload))
            else:
                ref._store.pop(ref.doc_id, None)
        self._ops = []


class _FakeFirestoreCollection:
    def __init__(self, store):
        self._store = store

    def document(self, doc_id):
        return _FakeFirestoreDocRef(self._store, doc_id)

    def stream(self):
        return [_FakeSnap(_FakeFirestoreDocRef(self._store, doc_id))
                for doc_id in list(self._store.keys())]


class _FakeFirestoreClient:
    def __init__(self, expected_collection):
        self.writes = []                 # append-only log of committed set()s
        self.expected_collection = expected_collection
        self.store = {}                  # current collection state (doc_id -> payload)

    def collection(self, name):
        assert name == self.expected_collection
        return _FakeFirestoreCollection(self.store)

    def batch(self):
        return _FakeFirestoreBatch(self.writes)


def test_write_news_to_firestore_groups_into_per_ticker_docs():
    """Two articles for RELIANCE + one GENERAL → 2 docs, with nested articles."""
    df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries",
                "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
                "title": "RIL beats estimates",
                "content": "Body A",
                "source": "ET",
                "url": "https://example.com/a",
                "url_hash": "h1",
            },
            {
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries",
                "ts": pd.Timestamp("2026-01-15T10:00:00", tz="UTC"),
                "title": "RIL retail momentum",
                "content": "Body B",
                "source": "Moneycontrol",
                "url": "https://example.com/b",
                "url_hash": "h2",
            },
            {
                "ticker": news_ingest.GENERAL_TICKER,
                "company_name": news_ingest.GENERAL_COMPANY_NAME,
                "ts": pd.Timestamp("2026-01-15T09:00:00", tz="UTC"),
                "title": "NIFTY hits new high",
                "content": "Macro body",
                "source": "BQ",
                "url": "https://example.com/c",
                "url_hash": "h3",
            },
        ]
    )
    client = _FakeFirestoreClient(news_ingest.FIRESTORE_COLLECTION)
    written = news_ingest.write_news_to_firestore(df, client=client)

    # Returns row count, not doc count — 3 rows uploaded across 2 docs.
    assert written == 3
    doc_ids = {entry[0] for entry in client.writes}
    assert doc_ids == {"RELIANCE", news_ingest.GENERAL_TICKER}

    by_id = {doc_id: payload for doc_id, payload in client.writes}
    reliance = by_id["RELIANCE"]
    assert reliance["company_name"] == "Reliance Industries"
    assert set(reliance["articles"].keys()) == {"h1", "h2"}
    assert reliance["articles"]["h1"]["headline"] == "RIL beats estimates"
    assert reliance["articles"]["h2"]["content"] == "Body B"

    general = by_id[news_ingest.GENERAL_TICKER]
    assert general["company_name"] == news_ingest.GENERAL_COMPANY_NAME
    assert set(general["articles"].keys()) == {"h3"}


def test_write_news_wipes_stale_docs_before_writing():
    """Latest-only: a ticker present last run but absent this run is removed."""
    client = _FakeFirestoreClient(news_ingest.FIRESTORE_COLLECTION)
    # Simulate a stale doc from a prior run (ticker no longer has fresh news).
    client.store["OLDCO"] = {"ticker": "OLDCO", "articles": {"old": {}}}
    df = pd.DataFrame([{
        "ticker": "RELIANCE", "company_name": "Reliance Industries",
        "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"), "title": "fresh",
        "content": "b", "source": "ET", "url": "https://x/a", "url_hash": "h1",
    }])
    news_ingest.write_news_to_firestore(df, client=client)
    # OLDCO wiped; only the freshly collected ticker remains.
    assert set(client.store.keys()) == {"RELIANCE"}


def test_write_news_empty_df_does_not_wipe():
    """A zero-collection run must NOT nuke the last good snapshot."""
    client = _FakeFirestoreClient(news_ingest.FIRESTORE_COLLECTION)
    client.store["RELIANCE"] = {"ticker": "RELIANCE", "articles": {"h1": {}}}
    written = news_ingest.write_news_to_firestore(pd.DataFrame(), client=client)
    assert written == 0
    assert set(client.store.keys()) == {"RELIANCE"}   # preserved


def test_attach_article_content_populates_in_place(monkeypatch):
    """The fan-out helper hydrates each row's `content` from fetch_article_content."""
    rows = [
        {"url": "https://a.example/x", "ticker": "RELIANCE", "url_hash": "h1"},
        {"url": "https://a.example/x", "ticker": "TCS", "url_hash": "h1"},  # same URL, different ticker fan-out
        {"url": "https://b.example/y", "ticker": "INFY", "url_hash": "h2"},
    ]
    calls = []

    def fake_fetch(url, timeout=10.0):
        calls.append(url)
        return f"BODY[{url}]"

    monkeypatch.setattr(news_ingest, "fetch_article_content", fake_fetch)
    news_ingest.attach_article_content(rows, max_workers=2)

    # Deduped: same URL fetched once, applied to both rows
    assert sorted(calls) == ["https://a.example/x", "https://b.example/y"]
    assert rows[0]["content"] == "BODY[https://a.example/x]"
    assert rows[1]["content"] == "BODY[https://a.example/x]"
    assert rows[2]["content"] == "BODY[https://b.example/y]"


def test_attach_article_content_handles_fetch_failure_with_none(monkeypatch):
    rows = [{"url": "https://x.example", "ticker": "RELIANCE", "url_hash": "h1"}]
    monkeypatch.setattr(news_ingest, "fetch_article_content", lambda url, timeout=10.0: None)
    news_ingest.attach_article_content(rows)
    assert rows[0]["content"] is None


def test_merge_with_cache_dedupes_by_url(tmp_path):
    output = tmp_path / "news.parquet"
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    first = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
                "title": "A",
                "source": "X",
                "url": "https://a",
                "url_hash": "h1",
            }
        ]
    )
    merged = news_ingest._merge_with_cache(first, output, cutoff=cutoff)
    merged.to_parquet(output, index=False)
    second = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "ts": datetime(2026, 1, 16, tzinfo=timezone.utc),
                "title": "A updated",
                "source": "X",
                "url": "https://a",
                "url_hash": "h1",
            }
        ]
    )
    combined = news_ingest._merge_with_cache(second, output, cutoff=cutoff)
    # Same hash → only one row, and the newer title wins
    assert len(combined) == 1
    assert combined.iloc[0]["title"] == "A updated"
