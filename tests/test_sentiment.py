from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from features import sentiment


def test_aggregate_produces_per_ticker_date_rows():
    df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "ts": datetime(2026, 1, 14, tzinfo=timezone.utc),
                "text": "great",
                "weight": 1.0,
                "polarity": 0.6,
            },
            {
                "ticker": "RELIANCE",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
                "text": "ok",
                "weight": 1.0,
                "polarity": 0.0,
            },
            {
                "ticker": "RELIANCE",
                "ts": datetime(2026, 1, 16, tzinfo=timezone.utc),
                "text": "bad",
                "weight": 1.0,
                "polarity": -0.4,
            },
        ]
    )
    agg = sentiment._aggregate(df, source="news", weight_col="weight")
    assert not agg.empty
    last = agg[agg["date"] == agg["date"].max()].iloc[0]
    assert last["source"] == "news"
    # The 3-day mean must lie between -0.4 and 0.6
    assert -0.4 <= last["sent_mean_3d"] <= 0.6
    assert last["n_7d"] >= 1
    assert {"sent_pos_share", "sent_neg_share", "sent_volume_z"}.issubset(set(agg.columns))


def test_aggregate_empty_input_returns_empty():
    empty = pd.DataFrame(columns=["ticker", "ts", "text", "weight", "polarity"])
    out = sentiment._aggregate(empty, source="news", weight_col="weight")
    assert out.empty


def test_prepare_news_extracts_title_as_text():
    news_df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
                "title": "Reliance beats estimates",
                "source": "Moneycontrol",
                "url": "u",
                "url_hash": "h",
            }
        ]
    )
    prepared = sentiment._prepare_news(news_df)
    assert "text" in prepared.columns
    assert prepared.iloc[0]["text"] == "Reliance beats estimates"
    # News has no engagement signal, so engagement_raw defaults to 0 and the
    # source label is preserved for the importance weight computation.
    assert prepared.iloc[0]["engagement_raw"] == 0.0
    assert prepared.iloc[0]["source_label"] == "Moneycontrol"


# ----------------------------------------------------------------------------
# Importance weight axes
# ----------------------------------------------------------------------------


def test_engagement_weight_returns_one_for_zero_score():
    assert sentiment._engagement_weight(0) == pytest.approx(1.0)


def test_engagement_weight_has_diminishing_returns():
    """log1p compression: 1000-upvote post is < 3x a 10-upvote post."""
    w_low = sentiment._engagement_weight(10)
    w_high = sentiment._engagement_weight(1000)
    assert w_low < w_high
    assert w_high / w_low < 3.0


def test_engagement_weight_clamps_negative_scores():
    assert sentiment._engagement_weight(-5) == pytest.approx(1.0)


def test_source_quality_news_known_publisher_is_high():
    assert sentiment._source_quality_news("Reuters") == 1.5
    assert sentiment._source_quality_news("moneycontrol") == 1.5  # case-insensitive
    assert sentiment._source_quality_news("Yahoo Finance") == 1.0


def test_source_quality_news_unknown_publisher_falls_back():
    assert sentiment._source_quality_news("RandomBlog") == sentiment.DEFAULT_SOURCE_WEIGHT
    assert sentiment._source_quality_news(None) == sentiment.DEFAULT_SOURCE_WEIGHT


def test_source_quality_reddit_lookup():
    assert sentiment._source_quality_reddit("IndianStockMarket") == 1.5
    assert sentiment._source_quality_reddit("DalalStreetTalks") == 1.0
    assert sentiment._source_quality_reddit("wallstreetbets") == sentiment.DEFAULT_SOURCE_WEIGHT


def test_confidence_weight_high_for_decisive_sentiment():
    # FinBERT was confident: neutral probability close to 0 → high weight.
    assert sentiment._confidence_weight(0.05) == pytest.approx(0.95)


def test_confidence_weight_clipped_at_floor():
    # Fully neutral text would weight 0 without the floor; we clip.
    assert sentiment._confidence_weight(1.0) == sentiment.MIN_CONFIDENCE
    assert sentiment._confidence_weight(0.99) == sentiment.MIN_CONFIDENCE


# ----------------------------------------------------------------------------
# Composite importance
# ----------------------------------------------------------------------------


def test_compose_importance_combines_three_axes():
    df = pd.DataFrame(
        [
            {
                "engagement_raw": 100.0,
                "source_label": "Reuters",     # 1.5
                "neutral": 0.1,                # confidence ≈ 0.9
            },
            {
                "engagement_raw": 0.0,
                "source_label": "UnknownBlog", # 0.5
                "neutral": 0.9,                # confidence = 0.1 (floor)
            },
        ]
    )
    weights = sentiment._compose_importance(df, source="news")
    # First row: ~ (log(101)+1) * 1.5 * 0.9  ≈ 5.62 * 1.5 * 0.9 ≈ 7.58
    # Second row: 1.0 * 0.5 * 0.1 = 0.05
    assert weights.iloc[0] > 50 * weights.iloc[1]  # the high-importance row dominates by >50×


def test_compose_importance_uses_reddit_table_for_reddit_source():
    df = pd.DataFrame(
        [
            {"engagement_raw": 10.0, "source_label": "IndianStockMarket", "neutral": 0.2},
            {"engagement_raw": 10.0, "source_label": "DalalStreetTalks", "neutral": 0.2},
        ]
    )
    weights = sentiment._compose_importance(df, source="reddit")
    # Same engagement and confidence, different subreddit credibility (1.5 vs 1.0).
    assert weights.iloc[0] / weights.iloc[1] == pytest.approx(1.5)


def test_compose_importance_x_is_engagement_times_confidence_only():
    df = pd.DataFrame(
        [
            {"engagement_raw": 50.0, "source_label": None, "neutral": 0.2},
        ]
    )
    weights = sentiment._compose_importance(df, source="x")
    # source_quality is flat 1.0 for X, so weight = engagement × confidence
    expected = sentiment._engagement_weight(50.0) * sentiment.X_SOURCE_WEIGHT * sentiment._confidence_weight(0.2)
    assert weights.iloc[0] == pytest.approx(expected)


def test_latest_per_ticker_source_picks_max_date():
    df = pd.DataFrame(
        [
            {"ticker": "RELIANCE", "source": "news", "date": pd.Timestamp("2026-01-10"), "sent_mean_7d": 0.1},
            {"ticker": "RELIANCE", "source": "news", "date": pd.Timestamp("2026-01-15"), "sent_mean_7d": 0.4},
            {"ticker": "RELIANCE", "source": "reddit", "date": pd.Timestamp("2026-01-12"), "sent_mean_7d": -0.2},
            {"ticker": "TCS", "source": "news", "date": pd.Timestamp("2026-01-15"), "sent_mean_7d": 0.3},
        ]
    )
    latest = sentiment._latest_per_ticker_source(df)
    assert len(latest) == 3
    # The 2026-01-15 RELIANCE-news row must win, not the 2026-01-10 one.
    rel_news = latest[(latest["ticker"] == "RELIANCE") & (latest["source"] == "news")].iloc[0]
    assert rel_news["sent_mean_7d"] == pytest.approx(0.4)


def test_latest_per_ticker_source_handles_empty_input():
    empty = pd.DataFrame(columns=["ticker", "source", "date", "sent_mean_7d"])
    out = sentiment._latest_per_ticker_source(empty)
    assert out.empty


def test_firestore_payload_uses_required_fields_and_nan_to_none():
    row = {
        "ticker": "RELIANCE",
        "source": "news",
        "date": pd.Timestamp("2026-01-15"),
        "sent_mean_3d": 0.42,
        "sent_mean_7d": 0.38,
        "sent_pos_share": float("nan"),  # exercise NaN coercion
        "sent_neg_share": 0.15,
        "sent_volume_z": 1.2,
        "n_3d": 8,
        "n_7d": 23,
    }
    payload = sentiment._firestore_payload(row, scrape_date="2026-01-16")
    assert payload["company_name"] == "Reliance Industries"
    assert payload["ticker"] == "RELIANCE"
    assert payload["source"] == "news"
    assert payload["as_of_date"] == "2026-01-15"
    assert payload["sent_mean_7d"] == pytest.approx(0.38)
    assert payload["sent_pos_share"] is None  # NaN → None for Firestore
    assert payload["n_7d"] == 23
    assert payload["scrape_date"] == "2026-01-16"


def test_firestore_doc_id_combines_ticker_and_source():
    row = {"ticker": "RELIANCE", "source": "reddit"}
    assert sentiment._firestore_doc_id(row) == "RELIANCE_reddit"


# ----------------------------------------------------------------------------
# Fake Firestore client for the wipe-then-write test
# ----------------------------------------------------------------------------


class _FakeDocRef:
    def __init__(self, doc_id):
        self.doc_id = doc_id


class _FakeStreamDoc:
    """Mimics a Firestore DocumentSnapshot for the wipe path."""

    def __init__(self, doc_id):
        self.reference = _FakeDocRef(doc_id)


class _FakeFirestoreBatch:
    def __init__(self, store, deletions):
        self._store = store
        self._deletions = deletions
        self._pending_sets = []
        self._pending_deletes = []

    def set(self, doc_ref, payload):
        self._pending_sets.append((doc_ref.doc_id, payload))

    def delete(self, doc_ref):
        self._pending_deletes.append(doc_ref.doc_id)

    def commit(self):
        for doc_id, payload in self._pending_sets:
            self._store[doc_id] = payload
        for doc_id in self._pending_deletes:
            self._store.pop(doc_id, None)
            self._deletions.append(doc_id)
        self._pending_sets.clear()
        self._pending_deletes.clear()


class _FakeFirestoreCollection:
    def __init__(self, store):
        self._store = store

    def document(self, doc_id):
        return _FakeDocRef(doc_id)

    def stream(self):
        # snapshot the keys so we can mutate the dict during iteration
        for doc_id in list(self._store.keys()):
            yield _FakeStreamDoc(doc_id)


class _FakeFirestoreClient:
    def __init__(self, prior_docs=None):
        self.store = dict(prior_docs or {})
        self.deletions = []

    def collection(self, name):
        assert name == sentiment.FIRESTORE_COLLECTION
        return _FakeFirestoreCollection(self.store)

    def batch(self):
        return _FakeFirestoreBatch(self.store, self.deletions)


def test_write_sentiment_to_firestore_wipes_then_writes_latest():
    df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "source": "news",
                "date": pd.Timestamp("2026-01-10"),
                "sent_mean_7d": 0.1,
                "sent_mean_3d": float("nan"),
                "sent_pos_share": 0.3,
                "sent_neg_share": 0.1,
                "sent_volume_z": 0.5,
                "n_3d": 1,
                "n_7d": 4,
            },
            {
                "ticker": "RELIANCE",
                "source": "news",
                "date": pd.Timestamp("2026-01-15"),  # newer → wins
                "sent_mean_7d": 0.4,
                "sent_mean_3d": 0.5,
                "sent_pos_share": 0.6,
                "sent_neg_share": 0.05,
                "sent_volume_z": 1.1,
                "n_3d": 3,
                "n_7d": 9,
            },
            {
                "ticker": "TCS",
                "source": "reddit",
                "date": pd.Timestamp("2026-01-15"),
                "sent_mean_7d": -0.2,
                "sent_mean_3d": -0.3,
                "sent_pos_share": 0.1,
                "sent_neg_share": 0.4,
                "sent_volume_z": 0.0,
                "n_3d": 2,
                "n_7d": 5,
            },
        ]
    )
    # Stale rows from a prior run that must be wiped.
    prior_docs = {
        "INFY_news": {"as_of_date": "1999-01-01"},
        "RELIANCE_x": {"as_of_date": "1999-01-01"},
    }
    client = _FakeFirestoreClient(prior_docs=prior_docs)
    written = sentiment.write_sentiment_to_firestore(df, client=client)

    # Two docs written (one per surviving ticker/source pair after latest-pick)
    assert written == 2
    assert set(client.store.keys()) == {"RELIANCE_news", "TCS_reddit"}

    # Stale docs were deleted, not silently left behind
    assert "INFY_news" in client.deletions
    assert "RELIANCE_x" in client.deletions

    # The newer RELIANCE-news row is the one persisted
    rel = client.store["RELIANCE_news"]
    assert rel["sent_mean_7d"] == pytest.approx(0.4)
    assert rel["as_of_date"] == "2026-01-15"


def test_write_sentiment_to_firestore_empty_frame_is_noop():
    client = _FakeFirestoreClient(prior_docs={"INFY_news": {"as_of": "old"}})
    written = sentiment.write_sentiment_to_firestore(pd.DataFrame(), client=client)
    assert written == 0
    # Empty input must not touch Firestore at all.
    assert "INFY_news" in client.store
    assert client.deletions == []


def test_importance_weighting_in_aggregate_favors_credible_source():
    """End-to-end: two news items, opposite polarity, very different importance.

    The Reuters / high-confidence item should pull the daily mean toward
    *its* polarity, even though the unknown-blog / neutral item has the
    same nominal sample count.
    """
    asof = datetime(2026, 1, 15, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "ts": asof,
                "text": "good",
                "engagement_raw": 0.0,
                "source_label": "Reuters",
                "polarity": 0.8,   # positive
                "neutral": 0.05,   # confident
            },
            {
                "ticker": "RELIANCE",
                "ts": asof,
                "text": "bad",
                "engagement_raw": 0.0,
                "source_label": "RandomBlog",
                "polarity": -0.8,  # negative, equal magnitude
                "neutral": 0.9,    # mushy → confidence clipped
            },
        ]
    )
    df["weight"] = sentiment._compose_importance(df, source="news")
    agg = sentiment._aggregate(df, source="news", weight_col="weight")
    # The credible+confident positive item dominates → mean is positive.
    last = agg.iloc[-1]
    assert last["sent_mean_7d"] > 0.5
