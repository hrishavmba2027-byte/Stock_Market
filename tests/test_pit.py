"""Tests for the point-in-time (PIT) append-only data spine."""
from __future__ import annotations

import pandas as pd

from ingestion import _pit


def _df(rows):
    return pd.DataFrame(rows)


def test_append_pit_is_append_only_and_dedups_snapshots(tmp_path):
    path = tmp_path / "fundamentals.parquet"
    day1 = _df([
        {"ticker": "TCS", "quarter": "2025Q2", "scrape_date": "2026-01-01", "revenue": 100},
    ])
    n1 = _pit.append_pit(day1, "fundamentals", ("ticker", "quarter", "scrape_date"), path=path)
    assert n1 == 1

    # Same day re-run → idempotent (same dedup key, replaced in place).
    n2 = _pit.append_pit(day1, "fundamentals", ("ticker", "quarter", "scrape_date"), path=path)
    assert n2 == 1

    # A later scrape_date for the same quarter → a NEW snapshot accumulates.
    day2 = _df([
        {"ticker": "TCS", "quarter": "2025Q2", "scrape_date": "2026-01-16", "revenue": 110},
    ])
    n3 = _pit.append_pit(day2, "fundamentals", ("ticker", "quarter", "scrape_date"), path=path)
    assert n3 == 2  # both snapshots retained (nothing removed)

    stored = pd.read_parquet(path)
    assert set(stored["scrape_date"]) == {"2026-01-01", "2026-01-16"}


def test_load_pit_asof_excludes_future_rows(tmp_path):
    path = tmp_path / "news.parquet"
    df = _df([
        {"ticker": "TCS", "url_hash": "a", "ts": "2026-01-05T00:00:00+00:00"},
        {"ticker": "TCS", "url_hash": "b", "ts": "2026-01-20T00:00:00+00:00"},
    ])
    _pit.append_pit(df, "news", ("ticker", "url_hash"), path=path)
    asof = _pit.load_pit_asof("news", "2026-01-10", "ts", path=path)
    assert list(asof["url_hash"]) == ["a"]  # the 2026-01-20 row is in the future


def test_maybe_append_pit_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(_pit, "is_pit_enabled", lambda: False)
    path = tmp_path / "x.parquet"
    n = _pit.maybe_append_pit(_df([{"a": 1}]), "x", ("a",))
    assert n == 0
    assert not path.exists()


def test_append_pit_empty_is_noop(tmp_path):
    path = tmp_path / "e.parquet"
    assert _pit.append_pit(pd.DataFrame(), "e", ("a",), path=path) == 0
    assert not path.exists()
