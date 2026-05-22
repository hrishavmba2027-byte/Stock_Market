import math
from datetime import date

import pandas as pd
import pytest

from ingestion import fundamentals


def test_safe_float_handles_none_and_nan():
    assert fundamentals._safe_float(None) is None
    assert fundamentals._safe_float(float("nan")) is None
    assert fundamentals._safe_float(float("inf")) is None
    assert fundamentals._safe_float("3.14") == pytest.approx(3.14)


def test_yoy_handles_zero_and_none():
    assert fundamentals._yoy(None, 100) is None
    assert fundamentals._yoy(100, None) is None
    assert fundamentals._yoy(100, 0) is None
    assert fundamentals._yoy(120, 100) == pytest.approx(0.2)


def test_ratio_returns_none_for_zero_denominator():
    assert fundamentals._ratio(10, 0) is None
    assert fundamentals._ratio(None, 10) is None
    assert fundamentals._ratio(10, 5) == pytest.approx(2.0)


def test_row_value_extracts_first_column():
    df = pd.DataFrame(
        {
            "2025-12-31": [100.0, 20.0],
            "2025-09-30": [90.0, 15.0],
        },
        index=["Total Revenue", "Net Income"],
    )
    assert fundamentals._row_value(df, "Total Revenue", 0) == 100.0
    assert fundamentals._row_value(df, "Net Income", 1) == 15.0
    assert fundamentals._row_value(df, "Missing Key", 0) is None


def test_row_value_handles_empty_or_none():
    assert fundamentals._row_value(None, "Total Revenue", 0) is None
    assert fundamentals._row_value(pd.DataFrame(), "Total Revenue", 0) is None


def test_next_earnings_date_returns_earliest_future():
    class FakeTicker:
        @property
        def earnings_dates(self):
            return pd.DataFrame(
                {"EPS Estimate": [1.0, 1.2, 1.3]},
                index=pd.to_datetime(
                    [
                        "2025-06-01",  # past
                        "2026-04-15",  # future
                        "2026-07-10",  # future, later
                    ],
                    utc=True,
                ),
            )

    today = date(2026, 1, 1)
    result = fundamentals._next_earnings_date(FakeTicker(), today)
    assert result == date(2026, 4, 15)


def test_next_earnings_date_returns_none_when_no_future_dates():
    class FakeTicker:
        @property
        def earnings_dates(self):
            return pd.DataFrame(
                {"EPS Estimate": [1.0]},
                index=pd.to_datetime(["2024-06-01"], utc=True),
            )

    assert fundamentals._next_earnings_date(FakeTicker(), date(2026, 1, 1)) is None


# ----------------------------------------------------------------------------
# Quarterly extractor + Firestore payload
# ----------------------------------------------------------------------------


def test_quarter_label_maps_month_to_correct_quarter():
    assert fundamentals._quarter_label("2025-03-31") == "2025Q1"
    assert fundamentals._quarter_label("2025-06-30") == "2025Q2"
    assert fundamentals._quarter_label("2025-09-30") == "2025Q3"
    assert fundamentals._quarter_label("2025-12-31") == "2025Q4"


def test_quarter_label_handles_invalid_input():
    assert fundamentals._quarter_label("not-a-date") is None
    assert fundamentals._quarter_label(None) is None


def _make_quarterly_frame(rows: dict, dates: list) -> pd.DataFrame:
    return pd.DataFrame(rows, index=list(rows.keys())).set_axis(
        pd.to_datetime(dates), axis=1
    )


def test_quarter_columns_returns_most_recent_first():
    income = pd.DataFrame(
        {
            "2025-12-31": [100, 20],
            "2025-09-30": [95, 18],
            "2025-06-30": [90, 17],
            "2025-03-31": [85, 15],
            "2024-12-31": [80, 12],
        },
        index=["Total Revenue", "Net Income"],
    )
    cols = fundamentals._quarter_columns(income)
    timestamps = [ts for _, ts in cols]
    assert timestamps == sorted(timestamps, reverse=True)
    assert len(cols) == 5
    # Column 0 in the original frame is 2025-12-31; that must be the head.
    assert cols[0][1] == pd.Timestamp("2025-12-31")


def test_quarter_columns_falls_back_to_richer_frame():
    income = pd.DataFrame(
        {"2025-12-31": [100]},
        index=["Total Revenue"],
    )
    balance = pd.DataFrame(
        {
            "2025-12-31": [10],
            "2025-09-30": [9],
            "2025-06-30": [8],
            "2025-03-31": [7],
        },
        index=["Total Assets"],
    )
    cols = fundamentals._quarter_columns(income, balance)
    assert len(cols) == 4  # driven by balance, not income


def test_extract_quarter_financials_computes_ratios_and_fcf():
    income = pd.DataFrame(
        {
            "2025-12-31": [200, 50, 30, 100],
        },
        index=["Total Revenue", "Operating Income", "Net Income", "Gross Profit"],
    )
    balance = pd.DataFrame(
        {"2025-12-31": [300, 100, 500]},
        index=["Total Stockholder Equity", "Total Debt", "Total Assets"],
    )
    cashflow = pd.DataFrame(
        {"2025-12-31": [80, -20]},
        index=["Operating Cash Flow", "Capital Expenditures"],
    )
    fin = fundamentals._extract_quarter_financials(income, balance, cashflow, col_idx=0)
    assert fin["revenue"] == 200
    assert fin["operating_margin"] == pytest.approx(0.25)
    assert fin["net_margin"] == pytest.approx(0.15)
    assert fin["debt_to_equity"] == pytest.approx(100 / 300)
    assert fin["roe"] == pytest.approx(30 / 300)
    assert fin["free_cash_flow"] == pytest.approx(60)  # 80 + (-20)


def test_extract_quarter_financials_handles_missing_frames():
    fin = fundamentals._extract_quarter_financials(None, None, None, col_idx=0)
    assert fin["revenue"] is None
    assert fin["free_cash_flow"] is None
    assert fin["operating_margin"] is None


class _FakeTicker:
    """Stand-in for ``yfinance.Ticker`` used by fetch_quarterly_records."""

    def __init__(self):
        self.quarterly_financials = pd.DataFrame(
            {
                "2025-12-31": [400, 60, 40, 200],
                "2025-09-30": [380, 55, 35, 180],
                "2025-06-30": [360, 50, 30, 170],
                "2025-03-31": [340, 45, 25, 160],
                "2024-12-31": [320, 40, 20, 150],
            },
            index=["Total Revenue", "Operating Income", "Net Income", "Gross Profit"],
        )
        self.quarterly_balance_sheet = pd.DataFrame(
            {
                "2025-12-31": [1000, 200, 1500],
                "2025-09-30": [950, 220, 1480],
                "2025-06-30": [900, 230, 1450],
                "2025-03-31": [880, 240, 1430],
                "2024-12-31": [850, 250, 1400],
            },
            index=["Total Stockholder Equity", "Total Debt", "Total Assets"],
        )
        self.quarterly_cashflow = pd.DataFrame(
            {
                "2025-12-31": [70, -15],
                "2025-09-30": [65, -12],
                "2025-06-30": [60, -10],
                "2025-03-31": [55, -8],
                "2024-12-31": [50, -7],
            },
            index=["Operating Cash Flow", "Capital Expenditures"],
        )
        self.info = {"longName": "Fake Company Ltd."}

    @property
    def earnings_dates(self):
        return pd.DataFrame()


def test_fetch_quarterly_records_returns_last_n_quarters(monkeypatch):
    monkeypatch.setattr(fundamentals, "_yf_ticker", lambda symbol: _FakeTicker())
    records = fundamentals.fetch_quarterly_records("RELIANCE", today=date(2026, 1, 15), lookback=4)
    assert len(records) == 4
    quarters = [r["quarter"] for r in records]
    # Most recent first
    assert quarters == ["2025Q4", "2025Q3", "2025Q2", "2025Q1"]
    assert all(r["scrape_date"] == "2026-01-15" for r in records)
    assert all(r["ticker"] == "RELIANCE" for r in records)
    # company_name comes from the alias map (curated) — fallback would be yfinance .info
    assert records[0]["company_name"] in {"Reliance Industries", "Fake Company Ltd."}


def test_fetch_quarterly_records_respects_lookback(monkeypatch):
    monkeypatch.setattr(fundamentals, "_yf_ticker", lambda symbol: _FakeTicker())
    records = fundamentals.fetch_quarterly_records("RELIANCE", lookback=2)
    assert len(records) == 2


def test_firestore_payload_has_required_fields():
    records = [
        {
            "ticker": "RELIANCE",
            "company_name": "Reliance Industries",
            "scrape_date": "2026-01-15",
            "quarter": "2025Q4",
            "quarter_end_date": "2025-12-31",
            "financials": {"revenue": 400, "net_income": 40},
        },
        {
            "ticker": "RELIANCE",
            "company_name": "Reliance Industries",
            "scrape_date": "2026-01-15",
            "quarter": "2025Q3",
            "quarter_end_date": "2025-09-30",
            "financials": {"revenue": 350, "net_income": 35},
        },
    ]
    payload = fundamentals._firestore_payload(records)
    assert payload["company_name"] == "Reliance Industries"
    assert payload["scrape_date"] == "2026-01-15"
    assert payload["ticker"] == "RELIANCE"
    assert payload["quarters"]["2025Q4"]["quarter_end_date"] == "2025-12-31"
    assert payload["quarters"]["2025Q4"]["financials"] == {"revenue": 400, "net_income": 40}
    assert payload["quarters"]["2025Q3"]["financials"] == {"revenue": 350, "net_income": 35}


def test_firestore_doc_id_is_ticker():
    record = {"ticker": "RELIANCE", "quarter": "2025Q4"}
    assert fundamentals._firestore_doc_id(record) == "RELIANCE"


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
        assert name == fundamentals.FIRESTORE_COLLECTION
        return _FakeFirestoreCollection()

    def batch(self):
        return _FakeFirestoreBatch(self.writes)


def test_write_quarterly_to_firestore_writes_one_doc_per_company():
    records = [
        {
            "ticker": "RELIANCE",
            "company_name": "Reliance Industries",
            "scrape_date": "2026-01-15",
            "quarter": q,
            "quarter_end_date": "2025-12-31",
            "financials": {"revenue": 100 + i},
        }
        for i, q in enumerate(["2025Q4", "2025Q3", "2025Q2", "2025Q1"])
    ]
    client = _FakeFirestoreClient()
    written = fundamentals.write_quarterly_to_firestore(records, client=client)
    assert written == 1
    doc_ids = {entry[0] for entry in client.writes}
    assert doc_ids == {"RELIANCE"}
    payload = client.writes[0][1]
    assert set(payload.keys()) >= {"company_name", "scrape_date", "ticker", "quarters"}
    assert set(payload["quarters"]) == {"2025Q4", "2025Q3", "2025Q2", "2025Q1"}
    assert payload["quarters"]["2025Q4"]["financials"] == {"revenue": 100}


def test_write_quarterly_to_firestore_groups_multiple_companies():
    records = [
        {
            "ticker": ticker,
            "company_name": f"{ticker} Ltd",
            "scrape_date": "2026-01-15",
            "quarter": quarter,
            "quarter_end_date": "2025-12-31",
            "financials": {"revenue": 100},
        }
        for ticker, quarter in [
            ("RELIANCE", "2025Q4"),
            ("RELIANCE", "2025Q3"),
            ("TCS", "2025Q4"),
            ("TCS", "2025Q3"),
        ]
    ]
    client = _FakeFirestoreClient()
    written = fundamentals.write_quarterly_to_firestore(records, client=client)
    assert written == 2
    doc_ids = {entry[0] for entry in client.writes}
    assert doc_ids == {"RELIANCE", "TCS"}


def test_write_quarterly_to_firestore_skips_records_missing_keys():
    records = [
        {"ticker": "", "quarter": "2025Q4", "company_name": "x", "scrape_date": "y", "financials": {}},
        {"ticker": "RELIANCE", "quarter": "", "company_name": "x", "scrape_date": "y", "financials": {}},
        {"ticker": "RELIANCE", "quarter": "2025Q4", "company_name": "x", "scrape_date": "y", "financials": {}},
    ]
    client = _FakeFirestoreClient()
    written = fundamentals.write_quarterly_to_firestore(records, client=client)
    assert written == 1


def test_upload_archive_df_reports_rows_after_company_doc_write(monkeypatch):
    records = [
        {
            "ticker": "RELIANCE",
            "company_name": "Reliance Industries",
            "scrape_date": "2026-01-15",
            "quarter": q,
            "quarter_end_date": "2025-12-31",
            "financials": {"revenue": 100 + i},
        }
        for i, q in enumerate(["2025Q4", "2025Q3", "2025Q2", "2025Q1"])
    ]
    df = fundamentals._records_to_archive_df(records)

    def fake_company_write(uploaded_records, client=None):
        assert len(uploaded_records) == 4
        return 1  # one company doc

    monkeypatch.setattr(fundamentals, "write_quarterly_to_firestore", fake_company_write)
    assert fundamentals._upload_archive_df(df, client="sentinel") == 4


def test_refresh_fundamentals_collect_false_drains_archive_without_fetching(tmp_path, monkeypatch):
    archive = tmp_path / "archive" / "fundamentals.parquet"
    archive.parent.mkdir()

    cached_records = [
        {
            "ticker": "RELIANCE",
            "company_name": "Reliance Industries",
            "scrape_date": "2026-01-15",
            "quarter": "2025Q4",
            "quarter_end_date": "2025-12-31",
            "financials": {"revenue": 100},
        }
    ]
    pending_records = [
        {
            "ticker": "TCS",
            "company_name": "Tata Consultancy Services",
            "scrape_date": "2026-01-15",
            "quarter": "2025Q4",
            "quarter_end_date": "2025-12-31",
            "financials": {"revenue": 200},
        }
    ]
    fundamentals._records_to_archive_df(pending_records).to_parquet(archive, index=False)

    uploaded_records = []

    def fake_write(records, collection=fundamentals.FIRESTORE_COLLECTION, client=None):
        uploaded_records.extend(records)
        return len(records)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("fresh fundamentals collection should be skipped")

    monkeypatch.setattr(fundamentals, "write_quarterly_to_firestore", fake_write)
    monkeypatch.setattr(fundamentals, "fetch_fundamentals", fail_fetch)
    monkeypatch.setattr(fundamentals, "fetch_quarterly_records", fail_fetch)

    summary = fundamentals.refresh_fundamentals(
        ["RELIANCE"],
        output_path=archive,
        archive_path=archive,
        write_firestore=True,
        firestore_client="sentinel",
        collect=False,
    )

    assert uploaded_records[0]["ticker"] == "TCS"
    assert not archive.exists()
    assert summary == {
        "flat_rows": 0,
        "quarterly_records": 0,
        "firestore_writes": 0,
        "archive_drained": 1,
    }
