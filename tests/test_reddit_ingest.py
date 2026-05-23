from datetime import datetime, timedelta, timezone

import pandas as pd

from ingestion import reddit_ingest


# ----------------------------------------------------------------------------
# Row extraction
# ----------------------------------------------------------------------------


def _make_post(
    title,
    body="",
    score=10,
    created=None,
    permalink="/r/IndianStockMarket/comments/abc/post/",
    subreddit="IndianStockMarket",
    num_comments=2,
):
    created = created or datetime(2026, 1, 15, tzinfo=timezone.utc)
    return {
        "title": title,
        "selftext": body,
        "score": score,
        "num_comments": num_comments,
        "created_utc": created.timestamp(),
        "permalink": permalink,
        "subreddit": subreddit,
    }


def test_row_from_post_extracts_tickers():
    post = _make_post(
        title="Reliance Industries quarterly was strong",
        body="Holding $RELIANCE for the swing",
    )
    row = reddit_ingest._row_from_post(post, "IndianStockMarket")
    assert row is not None
    assert "RELIANCE" in row["tickers"]
    assert row["kind"] == "post"
    assert row["permalink_hash"]
    assert row["permalink"].startswith("https://www.reddit.com/r/")


def test_row_from_post_skips_unrelated_text():
    post = _make_post(title="What is cricket score", body="No tickers here")
    assert reddit_ingest._row_from_post(post, "IndianStockMarket") is None


def test_row_from_post_handles_missing_created_utc():
    post = _make_post(title="Reliance up")
    post.pop("created_utc")
    assert reddit_ingest._row_from_post(post, "IndianStockMarket") is None


# ----------------------------------------------------------------------------
# Listing parser
# ----------------------------------------------------------------------------


def _listing(children):
    return {"data": {"children": [{"data": c} for c in children]}}


def test_parse_listing_extracts_post_rows():
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    matched = _make_post(title="Bullish on TCS this quarter")
    unmatched = _make_post(title="What did you have for lunch", permalink="/r/x/p2")
    listing = _listing([matched, unmatched])
    rows = reddit_ingest.parse_listing(listing, "IndianStockMarket", cutoff)
    assert len(rows) == 1
    assert rows[0]["title"] == "Bullish on TCS this quarter"
    assert "TCS" in rows[0]["tickers"]


def test_parse_listing_respects_cutoff():
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh = _make_post(
        title="HDFC bank looks good",
        created=datetime(2026, 1, 10, tzinfo=timezone.utc),
    )
    stale = _make_post(
        title="HDFC bank old chatter",
        created=datetime(2025, 12, 15, tzinfo=timezone.utc),
        permalink="/r/x/p3",
    )
    # /new is time-ordered newest-first; the parser breaks on the first stale.
    listing = _listing([fresh, stale])
    rows = reddit_ingest.parse_listing(listing, "IndianStockMarket", cutoff)
    assert len(rows) == 1
    assert rows[0]["title"] == "HDFC bank looks good"


def test_parse_listing_handles_empty_listing():
    assert reddit_ingest.parse_listing({}, "IndianStockMarket", datetime.now(timezone.utc)) == []
    assert reddit_ingest.parse_listing(
        {"data": {"children": []}}, "IndianStockMarket", datetime.now(timezone.utc)
    ) == []


# ----------------------------------------------------------------------------
# Explode + cache merge
# ----------------------------------------------------------------------------


def test_explode_tickers_one_row_per_ticker_match():
    rows = [
        {
            "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
            "subreddit": "IndianStockMarket",
            "kind": "post",
            "title": "RELIANCE and TCS both up",
            "body": "",
            "score": 10,
            "num_comments": 1,
            "permalink": "https://www.reddit.com/x",
            "permalink_hash": "h",
            "tickers": ["RELIANCE", "TCS"],
        }
    ]
    df = reddit_ingest._explode_tickers(rows)
    assert len(df) == 2
    assert set(df["ticker"]) == {"RELIANCE", "TCS"}


def test_merge_with_cache_drops_old_and_dedupes(tmp_path):
    output = tmp_path / "reddit.parquet"
    cutoff = datetime(2025, 12, 1, tzinfo=timezone.utc)
    df_new = pd.DataFrame(
        [
            {
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
                "subreddit": "IndianStockMarket",
                "kind": "post",
                "title": "fresh",
                "body": "",
                "score": 5,
                "num_comments": 0,
                "permalink": "https://www.reddit.com/a",
                "permalink_hash": "ha",
                "ticker": "RELIANCE",
            },
            {
                "ts": datetime(2025, 10, 1, tzinfo=timezone.utc),  # before cutoff
                "subreddit": "IndianStockMarket",
                "kind": "post",
                "title": "stale",
                "body": "",
                "score": 0,
                "num_comments": 0,
                "permalink": "https://www.reddit.com/b",
                "permalink_hash": "hb",
                "ticker": "RELIANCE",
            },
        ]
    )
    out = reddit_ingest._merge_with_cache(df_new, output, cutoff=cutoff)
    assert len(out) == 1
    assert out.iloc[0]["title"] == "fresh"


# ----------------------------------------------------------------------------
# Firestore payload + writer
# ----------------------------------------------------------------------------


def test_post_entry_carries_title_body_and_metadata():
    row = {
        "title": "Reliance Q3 beats",
        "body": "Retail margins were the story",
        "score": 42,
        "subreddit": "IndianStockMarket",
        "permalink": "https://www.reddit.com/r/IndianStockMarket/comments/abc/",
        "kind": "post",
        "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
    }
    entry = reddit_ingest._post_entry(row)
    assert entry["title"] == "Reliance Q3 beats"
    assert entry["body"] == "Retail margins were the story"
    assert entry["subreddit"] == "IndianStockMarket"
    assert entry["score"] == 42
    assert entry["kind"] == "post"
    assert entry["date_of_post"].startswith("2026-01-15")
    assert entry["permalink"].endswith("/abc/")


def test_firestore_payload_for_ticker_nests_posts_by_permalink_hash():
    rows = [
        {
            "ticker": "RELIANCE",
            "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
            "title": "RIL beats",
            "body": "",
            "score": 5,
            "subreddit": "IndianStockMarket",
            "permalink": "https://www.reddit.com/a",
            "permalink_hash": "h1",
            "kind": "post",
        },
        {
            "ticker": "RELIANCE",
            "ts": pd.Timestamp("2026-01-15T10:00:00", tz="UTC"),
            "title": "RIL retail",
            "body": "margins",
            "score": 12,
            "subreddit": "IndianStockMarket",
            "permalink": "https://www.reddit.com/b",
            "permalink_hash": "h2",
            "kind": "post",
        },
    ]
    payload = reddit_ingest._firestore_payload_for_ticker(
        "RELIANCE", rows, scrape_date="2026-01-16"
    )
    assert payload["company_name"] == "Reliance Industries"
    assert payload["ticker"] == "RELIANCE"
    assert payload["scrape_date"] == "2026-01-16"
    assert set(payload["posts"].keys()) == {"h1", "h2"}
    assert payload["posts"]["h2"]["body"] == "margins"


def test_firestore_doc_id_is_just_the_ticker():
    assert reddit_ingest._firestore_doc_id({"ticker": "TCS"}) == "TCS"
    assert reddit_ingest._firestore_doc_id("RELIANCE") == "RELIANCE"


class _FakeFirestoreBatch:
    def __init__(self, log):
        self._log = log

    def set(self, doc_ref, payload):
        self._log.append((doc_ref.doc_id, payload))

    def commit(self):
        pass


class _FakeFirestoreDocRef:
    def __init__(self, doc_id):
        self.doc_id = doc_id


class _FakeFirestoreCollection:
    def document(self, doc_id):
        return _FakeFirestoreDocRef(doc_id)


class _FakeFirestoreClient:
    def __init__(self):
        self.writes = []

    def collection(self, name):
        assert name == reddit_ingest.FIRESTORE_COLLECTION
        return _FakeFirestoreCollection()

    def batch(self):
        return _FakeFirestoreBatch(self.writes)


def test_write_reddit_to_firestore_groups_into_per_ticker_docs():
    """Two posts for RELIANCE + one for TCS → 2 docs, posts nested by hash."""
    df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
                "title": "RIL beats",
                "body": "",
                "score": 5,
                "subreddit": "IndianStockMarket",
                "permalink": "https://www.reddit.com/a",
                "permalink_hash": "h1",
                "kind": "post",
            },
            {
                "ticker": "RELIANCE",
                "ts": pd.Timestamp("2026-01-15T11:00:00", tz="UTC"),
                "title": "RIL retail",
                "body": "margins",
                "score": 8,
                "subreddit": "IndianStockMarket",
                "permalink": "https://www.reddit.com/b",
                "permalink_hash": "h2",
                "kind": "post",
            },
            {
                "ticker": "TCS",
                "ts": pd.Timestamp("2026-01-15T09:00:00", tz="UTC"),
                "title": "TCS deal momentum",
                "body": "",
                "score": 12,
                "subreddit": "StockMarketIndia",
                "permalink": "https://www.reddit.com/c",
                "permalink_hash": "h3",
                "kind": "post",
            },
        ]
    )
    client = _FakeFirestoreClient()
    written = reddit_ingest.write_reddit_to_firestore(df, client=client)
    # 3 rows uploaded across 2 docs
    assert written == 3
    doc_ids = {entry[0] for entry in client.writes}
    assert doc_ids == {"RELIANCE", "TCS"}

    by_id = {doc_id: payload for doc_id, payload in client.writes}
    assert set(by_id["RELIANCE"]["posts"].keys()) == {"h1", "h2"}
    assert by_id["RELIANCE"]["posts"]["h2"]["body"] == "margins"
    assert set(by_id["TCS"]["posts"].keys()) == {"h3"}
    assert by_id["TCS"]["company_name"] == "Tata Consultancy Services"


# ----------------------------------------------------------------------------
# refresh_reddit end-to-end with an injected fetcher
# ----------------------------------------------------------------------------


def test_refresh_reddit_uses_injected_fetcher(tmp_path, monkeypatch):
    output = tmp_path / "reddit.parquet"
    # Avoid the 2s inter-subreddit sleep when running the suite.
    monkeypatch.setattr(reddit_ingest, "INTER_SUBREDDIT_SLEEP_SECONDS", 0)
    monkeypatch.setattr(reddit_ingest.time_module, "sleep", lambda *_: None)

    now = datetime.now(timezone.utc)
    listings = {
        "IndianStockMarket": _listing(
            [_make_post(title="RELIANCE moving up", created=now - timedelta(hours=1))]
        ),
        "StockMarketIndia": _listing(
            [_make_post(title="No tickers in this one", permalink="/r/x/p9", created=now)]
        ),
    }

    df = reddit_ingest.refresh_reddit(
        output_path=output,
        subreddits=("IndianStockMarket", "StockMarketIndia"),
        fetcher=lambda sub, _: listings.get(sub),
        write_firestore=False,
    )
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "RELIANCE"


# ----------------------------------------------------------------------------
# Archive retry-queue flow (see ingestion/_archive.py)
# ----------------------------------------------------------------------------


def test_refresh_reddit_drains_archive_then_falls_through_to_collect(tmp_path, monkeypatch):
    """Fall-through: archive is drained AND fresh collection runs in the same call."""
    archive = tmp_path / "archive" / "reddit.parquet"
    archive.parent.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries",
                "subreddit": "IndianStockMarket",
                "kind": "post",
                "title": "Pending upload from yesterday",
                "body": "",
                "score": 5,
                "num_comments": 0,
                "permalink": "https://www.reddit.com/a",
                "permalink_hash": "hpending",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
            }
        ]
    ).to_parquet(archive, index=False)

    upload_calls = []

    def fake_upload(df, client=None):
        upload_calls.append(df.copy())
        return len(df)

    monkeypatch.setattr(reddit_ingest, "write_reddit_to_firestore", fake_upload)
    monkeypatch.setattr(reddit_ingest, "INTER_SUBREDDIT_SLEEP_SECONDS", 0)
    monkeypatch.setattr(reddit_ingest.time_module, "sleep", lambda *_: None)

    fetcher_calls = []
    now = datetime.now(timezone.utc)
    listings = {
        "IndianStockMarket": _listing(
            [_make_post(title="INFY hit a new high today", created=now - timedelta(hours=1))]
        ),
    }

    def tracking_fetcher(sub, lim):
        fetcher_calls.append(sub)
        return listings.get(sub)

    result = reddit_ingest.refresh_reddit(
        output_path=archive,
        subreddits=("IndianStockMarket",),
        fetcher=tracking_fetcher,
        write_firestore=True,
        firestore_client="sentinel",
    )

    # Two uploads happened in sequence:
    #   1. archive drain → yesterday's pending row
    #   2. fresh-collect → today's INFY row
    assert len(upload_calls) == 2
    assert upload_calls[0].iloc[0]["ticker"] == "RELIANCE"
    assert upload_calls[1].iloc[0]["ticker"] == "INFY"
    # Fresh collection RAN (fall-through, not skipped)
    assert fetcher_calls == ["IndianStockMarket"]
    # Archive is empty after both uploads cleared it
    assert not archive.exists()
    assert isinstance(result, pd.DataFrame)


def test_refresh_reddit_merges_retained_archive_when_drain_fails(tmp_path, monkeypatch):
    """If drain fails, today's fresh data merges into the retained archive."""
    archive = tmp_path / "archive" / "reddit.parquet"
    archive.parent.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries",
                "subreddit": "IndianStockMarket",
                "kind": "post",
                "title": "Yesterday's retained row",
                "body": "",
                "score": 5,
                "num_comments": 0,
                "permalink": "https://www.reddit.com/a",
                "permalink_hash": "hpending",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
            }
        ]
    ).to_parquet(archive, index=False)

    # Firestore is broken for the whole run
    def failing_upload(df, client=None):
        raise RuntimeError("firestore unreachable")

    monkeypatch.setattr(reddit_ingest, "write_reddit_to_firestore", failing_upload)
    monkeypatch.setattr(reddit_ingest, "INTER_SUBREDDIT_SLEEP_SECONDS", 0)
    monkeypatch.setattr(reddit_ingest.time_module, "sleep", lambda *_: None)

    now = datetime.now(timezone.utc)
    listings = {
        "IndianStockMarket": _listing(
            [_make_post(title="Fresh INFY mention", created=now - timedelta(hours=1))]
        ),
    }

    reddit_ingest.refresh_reddit(
        output_path=archive,
        subreddits=("IndianStockMarket",),
        fetcher=lambda sub, lim: listings.get(sub),
        write_firestore=True,
        firestore_client="sentinel",
    )

    # Archive is retained AND now holds BOTH yesterday's pending row + today's fresh
    assert archive.exists()
    retained = pd.read_parquet(archive)
    tickers = set(retained["ticker"])
    assert "RELIANCE" in tickers  # yesterday's row preserved
    assert "INFY" in tickers       # today's row merged in


def test_refresh_reddit_stages_and_uploads_on_fresh_collect(tmp_path, monkeypatch):
    """Case 2: empty archive → collect, stage to archive, upload, clear archive."""
    output = tmp_path / "reddit.parquet"
    archive = tmp_path / "archive" / "reddit.parquet"
    assert not archive.exists()

    upload_payloads = []

    def fake_upload(df, client=None):
        upload_payloads.append(df.copy())
        return len(df)

    monkeypatch.setattr(reddit_ingest, "write_reddit_to_firestore", fake_upload)
    monkeypatch.setattr(reddit_ingest, "INTER_SUBREDDIT_SLEEP_SECONDS", 0)
    monkeypatch.setattr(reddit_ingest.time_module, "sleep", lambda *_: None)

    now = datetime.now(timezone.utc)
    listings = {
        "IndianStockMarket": _listing(
            [_make_post(title="RELIANCE Q3 beats", created=now - timedelta(hours=1))]
        ),
    }

    result = reddit_ingest.refresh_reddit(
        output_path=output,
        subreddits=("IndianStockMarket",),
        fetcher=lambda sub, lim: listings.get(sub),
        write_firestore=True,
        firestore_client="sentinel",
    )

    assert len(result) == 1
    # Upload was called exactly once with the just-collected data
    assert len(upload_payloads) == 1
    assert upload_payloads[0].iloc[0]["ticker"] == "RELIANCE"
    # Archive was cleared after successful upload
    assert not archive.exists()


def test_refresh_reddit_keeps_archive_on_upload_failure(tmp_path, monkeypatch):
    """Case 2 (failure): collection happens, archive staged, upload fails → archive retained."""
    output = tmp_path / "reddit.parquet"
    archive = tmp_path / "archive" / "reddit.parquet"

    def failing_upload(df, client=None):
        raise RuntimeError("firestore unreachable")

    monkeypatch.setattr(reddit_ingest, "write_reddit_to_firestore", failing_upload)
    monkeypatch.setattr(reddit_ingest, "INTER_SUBREDDIT_SLEEP_SECONDS", 0)
    monkeypatch.setattr(reddit_ingest.time_module, "sleep", lambda *_: None)

    now = datetime.now(timezone.utc)
    listings = {
        "IndianStockMarket": _listing(
            [_make_post(title="RELIANCE bounced", created=now - timedelta(hours=1))]
        ),
    }

    reddit_ingest.refresh_reddit(
        output_path=output,
        subreddits=("IndianStockMarket",),
        fetcher=lambda sub, lim: listings.get(sub),
        write_firestore=True,
        firestore_client="sentinel",
    )

    # Archive retained for next-run retry
    assert archive.exists()
