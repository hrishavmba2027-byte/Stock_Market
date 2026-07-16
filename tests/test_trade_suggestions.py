"""Tests for the fundamentals wiring in features/trade_suggestions.py."""
from __future__ import annotations

from features import trade_suggestions as ts


class _FakeDoc:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = data

    def to_dict(self):
        return self._data


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def stream(self):
        return iter(self._docs)


class _FakeClient:
    def __init__(self, docs):
        self._docs = docs

    def collection(self, name):
        return _FakeCollection(self._docs)


def test_load_fundamentals_keeps_latest_quarters_newest_first():
    docs = [
        _FakeDoc(
            "RELIANCE",
            {
                "ticker": "RELIANCE",
                "scrape_date": "2026-06-20",
                "quarters": {
                    "2025Q2": {"quarter_end_date": "2025-06-30", "financials": {"revenue": 1}},
                    "2025Q4": {"quarter_end_date": "2025-12-31", "financials": {"revenue": 3}},
                    "2025Q3": {"quarter_end_date": "2025-09-30", "financials": {"revenue": 2}},
                    "2025Q1": {"quarter_end_date": "2025-03-31", "financials": {"revenue": 0}},
                    "2024Q4": {"quarter_end_date": "2024-12-31", "financials": {"revenue": -1}},
                    "2024Q3": {"quarter_end_date": "2024-09-30", "financials": {"revenue": -2}},
                },
            },
        ),
    ]
    out = ts.load_fundamentals(_FakeClient(docs))
    assert "RELIANCE" in out
    summary = out["RELIANCE"]
    # Only the latest FUNDAMENTALS_QUARTERS (5) quarters, newest label first; the
    # oldest (2024Q3) is evicted by the rolling window.
    assert summary["latest_quarter"] == "2025Q4"
    assert list(summary["recent_quarters"].keys()) == [
        "2025Q4", "2025Q3", "2025Q2", "2025Q1", "2024Q4",
    ]
    assert summary["scrape_date"] == "2026-06-20"


def test_load_fundamentals_skips_empty_or_malformed_docs():
    docs = [
        _FakeDoc("EMPTY", {"ticker": "EMPTY", "quarters": {}}),
        _FakeDoc("NOQ", {"ticker": "NOQ"}),
    ]
    out = ts.load_fundamentals(_FakeClient(docs))
    assert out == {}


def test_suggestion_schema_version_is_v3():
    # v3 = forecast + fundamentals + split sentiment (company + sector + general).
    assert ts.SUGGESTION_SCHEMA_VERSION == 3


def test_fingerprint_changes_when_fundamentals_change():
    market = {"date": "2026-06-20", "close": 100.0, "forecast": {"T+1": 101.0}}
    base = ts.fingerprint({"market": market, "sentiment": None, "fundamentals": None})
    with_fund = ts.fingerprint(
        {"market": market, "sentiment": None, "fundamentals": {"latest_quarter": "2025Q4"}}
    )
    assert base != with_fund
