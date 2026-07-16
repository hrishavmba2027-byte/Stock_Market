"""Tests for the backtest CDX news scraper's pure logic + local workbook storage."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backtesting import fetch_news as fn


def test_normalize_url_strips_query_fragment_amp():
    assert fn.normalize_url("https://x.com/news/a-b?utm=1#frag") == "https://x.com/news/a-b"
    assert fn.normalize_url("https://x.com/news/a-b/amp") == "https://x.com/news/a-b"
    assert fn.normalize_url("https://x.com/news/a-b/") == "https://x.com/news/a-b"


def test_is_article_url_rejects_listings():
    assert fn.is_article_url("https://x.com/markets/tcs-q3-results-123")
    assert not fn.is_article_url("https://x.com/topic/tcs")
    assert not fn.is_article_url("https://x.com/markets/page-2")
    assert not fn.is_article_url("https://x.com/onlyonesegment")


def test_interval_windows_are_seven_day_buckets():
    wins = fn.interval_windows(date(2025, 1, 1), date(2025, 1, 20), 7)
    assert wins[0] == (date(2025, 1, 1), date(2025, 1, 7))
    assert wins[1] == (date(2025, 1, 8), date(2025, 1, 14))
    assert wins[-1][1] == date(2025, 1, 20)  # final partial window clamps to end


def test_assign_window_and_aggregate():
    wins = fn.interval_windows(date(2025, 1, 1), date(2025, 1, 14), 7)
    rows = [
        {"article_date": date(2025, 1, 2), "sentiment": 0.5},
        {"article_date": date(2025, 1, 3), "sentiment": 0.1},
        {"article_date": date(2025, 1, 10), "sentiment": -0.4},
        {"article_date": date(2025, 2, 1), "sentiment": 0.9},  # outside → dropped
    ]
    agg = fn.aggregate_by_window(rows, wins)
    assert agg["2025-01-07"]["n"] == 2
    assert abs(agg["2025-01-07"]["mean_sentiment"] - 0.3) < 1e-9
    assert agg["2025-01-14"]["n"] == 1
    assert "2025-02-04" not in agg  # the out-of-range article created no bucket


def test_build_targets_has_company_and_sector_only():
    targets = fn.build_targets(["TCS"])
    kinds = {t["kind"] for t in targets}
    assert kinds == {"company", "sector"}  # no 'general'
    company = next(t for t in targets if t["kind"] == "company")
    assert company["id"] == "TCS"
    sector = next(t for t in targets if t["kind"] == "sector")
    assert sector["id"].startswith("SECTOR__")


# --- local workbook storage (append-only .xlsx, no Firestore) ----------------

def _company(tid="TCS"):
    return {"id": tid, "kind": "company", "display_name": tid}


def test_store_to_workbook_is_append_only_across_years(tmp_path: Path):
    wb = tmp_path / "news.xlsx"
    target = _company()
    fn.store_to_workbook(wb, target, 2020, [{"url": "https://a", "title": "t1", "content": "c1"}],
                         {"2020-01-07": {"mean_sentiment": 0.1, "n": 3, "pos_share": 0.6, "neg_share": 0.1}})
    fn.store_to_workbook(wb, target, 2021, [{"url": "https://b", "title": "t2", "content": "c2"}],
                         {"2021-01-07": {"mean_sentiment": -0.2, "n": 5, "pos_share": 0.2, "neg_share": 0.5}})
    xl = pd.ExcelFile(wb)
    assert set(xl.sheet_names) == {"News", "Sentiment", "Manifest"}
    news = pd.read_excel(wb, sheet_name="News")
    assert len(news) == 2                              # both years retained
    sent = pd.read_excel(wb, sheet_name="Sentiment")
    assert set(sent["as_of_date"].astype(str)) == {"2020-01-07", "2021-01-07"}


def test_store_to_workbook_dedups_repeat_pair(tmp_path: Path):
    wb = tmp_path / "news.xlsx"
    target = _company()
    agg = {"2020-01-07": {"mean_sentiment": 0.1, "n": 3, "pos_share": 0.6, "neg_share": 0.1}}
    fn.store_to_workbook(wb, target, 2020, [{"url": "https://a", "title": "t1"}], agg)
    fn.store_to_workbook(wb, target, 2020, [{"url": "https://a", "title": "t1"}], agg)  # re-run
    news = pd.read_excel(wb, sheet_name="News")
    assert len(news) == 1                              # same URL not double-counted
    manifest = pd.read_excel(wb, sheet_name="Manifest")
    assert len(manifest) == 1                          # (TCS, 2020) once


def test_sentiment_rows_match_reader_schema(tmp_path: Path):
    wb = tmp_path / "news.xlsx"
    fn.store_to_workbook(wb, _company(), 2020, [],
                         {"2020-01-07": {"mean_sentiment": 0.25, "n": 4, "pos_share": 0.5, "neg_share": 0.25}})
    sent = pd.read_excel(wb, sheet_name="Sentiment")
    for col in ("entity_id", "as_of_date", "source", "sent_mean_3d", "sent_mean_7d",
                "sent_pos_share", "sent_neg_share", "n_3d", "n_7d"):
        assert col in sent.columns
    row = sent.iloc[0]
    assert row["sent_mean_7d"] == row["sent_mean_3d"] == 0.25   # single-window aggregate


def test_load_collected_manifest_and_skip(tmp_path: Path):
    wb = tmp_path / "news.xlsx"
    fn.store_to_workbook(wb, _company(), 2020, [{"url": "https://a", "title": "t"}],
                         {"2020-01-07": {"mean_sentiment": 0.0, "n": 1}})
    done = fn.load_collected_manifest(wb)
    assert ("TCS", 2020) in done
    assert fn.load_collected_manifest(tmp_path / "missing.xlsx") == set()
