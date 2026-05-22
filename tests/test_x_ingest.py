from datetime import datetime, timezone

import pandas as pd

from ingestion import x_ingest


def test_is_spam_filters_giveaways():
    assert x_ingest._is_spam("Free crypto airdrop, claim now!", followers=1000, display_name="Bot Name") is True


def test_is_spam_filters_zero_follower_accounts():
    assert x_ingest._is_spam("Buy RELIANCE", followers=2, display_name="Real Person") is True


def test_is_spam_passes_legit_post():
    assert (
        x_ingest._is_spam(
            "RELIANCE Q3 numbers looked solid, holding through expiry",
            followers=2000,
            display_name="Trader",
        )
        is False
    )


def test_build_query_contains_symbol_and_aliases():
    q = x_ingest._build_query("RELIANCE")
    assert "$RELIANCE" in q
    assert "Reliance" in q


def test_expand_rows_maps_tweet_to_named_companies():
    base = {
        "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
        "title": "",
        "body": "TCS Q3 looks strong",
        "score": 10,
        "url": "https://x.com/u/1",
        "url_hash": "h1",
        "source": "snscrape",
    }
    rows = x_ingest._expand_rows(base, body=base["body"], fetch_ticker="TCS")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "TCS"
    assert rows[0]["company_name"] == "Tata Consultancy Services"


def test_expand_rows_falls_back_to_general():
    base = {
        "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
        "title": "",
        "body": "Markets euphoric on RBI minutes",
        "score": 0,
        "url": "https://x.com/u/2",
        "url_hash": "h2",
        "source": "snscrape",
    }
    rows = x_ingest._expand_rows(base, body=base["body"], fetch_ticker="RELIANCE")
    assert len(rows) == 1
    assert rows[0]["ticker"] == x_ingest.GENERAL_TICKER
    assert rows[0]["company_name"] == x_ingest.GENERAL_COMPANY_NAME


def test_expand_rows_fans_out_to_multiple_companies():
    base = {
        "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
        "title": "",
        "body": "Long TCS and INFY into Q3 results",
        "score": 7,
        "url": "https://x.com/u/3",
        "url_hash": "h3",
        "source": "snscrape",
    }
    rows = x_ingest._expand_rows(base, body=base["body"], fetch_ticker="TCS")
    tickers = {row["ticker"] for row in rows}
    assert tickers == {"TCS", "INFY"}


def test_x_post_entry_carries_body_and_metadata():
    row = {
        "body": "RIL crushing it today",
        "source": "snscrape",
        "url": "https://x.com/u/4",
        "score": 42,
        "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
    }
    entry = x_ingest._post_entry(row)
    assert entry["body"] == "RIL crushing it today"
    assert entry["source"] == "snscrape"
    assert entry["score"] == 42
    assert entry["date_of_post"].startswith("2026-01-15")
    assert entry["url"] == "https://x.com/u/4"


def test_x_firestore_payload_for_ticker_nests_posts():
    rows = [
        {
            "ticker": "RELIANCE",
            "company_name": "Reliance Industries",
            "body": "RIL strong",
            "source": "snscrape",
            "url": "https://x.com/a",
            "url_hash": "h1",
            "score": 5,
            "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
        },
    ]
    payload = x_ingest._firestore_payload_for_ticker(
        "RELIANCE", rows, scrape_date="2026-01-16"
    )
    assert payload["company_name"] == "Reliance Industries"
    assert payload["ticker"] == "RELIANCE"
    assert payload["scrape_date"] == "2026-01-16"
    assert set(payload["posts"].keys()) == {"h1"}
    assert payload["posts"]["h1"]["body"] == "RIL strong"


def test_x_firestore_doc_id_is_just_the_ticker():
    assert x_ingest._firestore_doc_id({"ticker": "RELIANCE"}) == "RELIANCE"
    assert x_ingest._firestore_doc_id("TCS") == "TCS"
    # Missing ticker → GENERAL fallback
    assert x_ingest._firestore_doc_id({}) == x_ingest.GENERAL_TICKER


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
        assert name == x_ingest.FIRESTORE_COLLECTION
        return _FakeFirestoreCollection()

    def batch(self):
        return _FakeFirestoreBatch(self.writes)


def test_write_x_to_firestore_groups_into_per_ticker_docs():
    df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries",
                "ts": pd.Timestamp("2026-01-15T08:00:00", tz="UTC"),
                "body": "RIL strong",
                "source": "snscrape",
                "url": "https://x.com/a",
                "url_hash": "h1",
                "score": 5,
            },
            {
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries",
                "ts": pd.Timestamp("2026-01-15T10:00:00", tz="UTC"),
                "body": "RIL second tweet",
                "source": "snscrape",
                "url": "https://x.com/aa",
                "url_hash": "h1b",
                "score": 8,
            },
            {
                "ticker": x_ingest.GENERAL_TICKER,
                "company_name": x_ingest.GENERAL_COMPANY_NAME,
                "ts": pd.Timestamp("2026-01-15T09:00:00", tz="UTC"),
                "body": "Markets rip",
                "source": "snscrape",
                "url": "https://x.com/b",
                "url_hash": "h2",
                "score": 0,
            },
        ]
    )
    client = _FakeFirestoreClient()
    written = x_ingest.write_x_to_firestore(df, client=client)
    # 3 rows uploaded across 2 docs
    assert written == 3
    doc_ids = {entry[0] for entry in client.writes}
    assert doc_ids == {"RELIANCE", x_ingest.GENERAL_TICKER}

    by_id = {doc_id: payload for doc_id, payload in client.writes}
    assert set(by_id["RELIANCE"]["posts"].keys()) == {"h1", "h1b"}
    assert by_id["RELIANCE"]["posts"]["h1b"]["body"] == "RIL second tweet"
    assert set(by_id[x_ingest.GENERAL_TICKER]["posts"].keys()) == {"h2"}


def test_merge_with_cache_dedupes_by_url_hash(tmp_path):
    output = tmp_path / "x.parquet"
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
                "title": "",
                "body": "Strong day",
                "score": 5,
                "url": "https://x.com/r/1",
                "url_hash": "h1",
                "source": "snscrape",
            },
            {
                "ticker": "RELIANCE",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
                "title": "",
                "body": "Strong day (echo)",
                "score": 5,
                "url": "https://x.com/r/1",
                "url_hash": "h1",
                "source": "snscrape",
            },
        ]
    )
    out = x_ingest._merge_with_cache(df, output, cutoff=cutoff)
    assert len(out) == 1


def test_refresh_x_collect_false_drains_archive_without_fetching(tmp_path, monkeypatch):
    archive = tmp_path / "archive" / "x.parquet"
    archive.parent.mkdir()

    cached = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
                "title": "",
                "body": "cached",
                "score": 1,
                "url": "https://x.com/cached",
                "url_hash": "cached",
                "source": "snscrape",
            }
        ]
    )
    pending = cached.assign(body="pending", url_hash="pending")
    pending.to_parquet(archive, index=False)

    upload_calls = []

    def fake_upload(df, client=None):
        upload_calls.append(df.copy())
        return len(df)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("fresh X collection should be skipped")

    monkeypatch.setattr(x_ingest, "write_x_to_firestore", fake_upload)
    monkeypatch.setattr(x_ingest, "fetch_x", fail_fetch)

    result = x_ingest.refresh_x(
        ["RELIANCE"],
        output_path=archive,
        write_firestore=True,
        firestore_client="sentinel",
        collect=False,
    )

    assert len(upload_calls) == 1
    assert upload_calls[0].iloc[0]["url_hash"] == "pending"
    assert not archive.exists()
    assert result.empty
