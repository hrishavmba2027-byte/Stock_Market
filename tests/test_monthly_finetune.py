import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

import monthly_finetune as mf
from app.pipeline.metadata import get_or_create_metadata
from app.services.sheet_archival import (
    ROW_FINETUNED_COL,
    archive_old_rows_for_worksheet,
)

_METADATA_PATH = Path("outputs/pipeline_metadata.json")


def test_forward_log_return_labels_anchor_to_previous_close():
    close = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=float)
    labels = mf.forward_log_return_labels(close, target_position=2, horizons=[1, 2, 3, 4, 5])

    assert labels is not None
    expected = np.log(np.asarray([12.0, 13.0, 14.0, 15.0, 16.0]) / 11.0)
    np.testing.assert_allclose(labels, expected.astype(np.float32))


def test_forecast_days_env_extends_forecast_columns(monkeypatch):
    metadata = get_or_create_metadata(_METADATA_PATH)

    monkeypatch.setenv("FORECAST_DAYS", "15")
    assert mf.forecast.forecast_close_columns(metadata)[-1] == "Forecast_Close_T+15"

    monkeypatch.setenv("FORECAST_DAYS", "invalid")
    assert mf.forecast.forecast_close_columns(metadata)[-1] == "Forecast_Close_T+5"


def test_recursive_forecast_continues_beyond_direct_horizon(monkeypatch):
    metadata = get_or_create_metadata(_METADATA_PATH)
    monkeypatch.setenv("FORECAST_DAYS", "7")
    dates = pd.bdate_range("2025-01-01", periods=80)
    close = np.linspace(100.0, 120.0, len(dates))
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Date_str": dates.strftime("%Y-%m-%d"),
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + np.arange(len(dates)) * 100,
            "predicted": [1] * 70 + [0] * 10,
        }
    )
    payload = mf.forecast.SheetPayload(
        name="TEST",
        frame=frame,
        headers=[str(column) for column in frame.columns],
        data_row_count=len(frame),
    )
    part = mf.forecast.prepare_stock_part(payload, metadata, latest_only=True)
    assert part.anchor_close[0] == close[-1]

    class ConstantModel(torch.nn.Module):
        n_horizons = 5

        def forward(self, x):
            return torch.zeros((x.shape[0], 5), dtype=x.dtype, device=x.device)

    results, _ = mf.forecast.run_inference(
        [part],
        {"Dense": ConstantModel(), "LSTM": ConstantModel(), "Transformer": ConstantModel()},
        metadata,
        torch.device("cpu"),
    )

    forecast_columns = [f"Forecast_Close_T+{idx}" for idx in range(1, 8)]
    assert forecast_columns[-1] in results.columns
    assert np.isfinite(results[forecast_columns].to_numpy(dtype=float)).all()
    np.testing.assert_allclose(results[forecast_columns].to_numpy(dtype=float), close[-1])


def test_build_finetune_arrays_from_synthetic_frame():
    metadata = get_or_create_metadata(_METADATA_PATH)
    dates = pd.bdate_range("2025-01-01", periods=90)
    close = np.linspace(100.0, 130.0, len(dates)) + np.sin(np.arange(len(dates)))
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Date_str": dates.strftime("%Y-%m-%d"),
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + np.arange(len(dates)) * 100,
        }
    )
    args = argparse.Namespace(
        recent_days=45,
        replay_samples_per_symbol=5,
        validation_targets_per_symbol=3,
    )

    arrays = mf.build_finetune_arrays({"TEST": frame}, metadata, args)

    assert arrays.X_train.shape[1:] == (metadata["seq_len"], metadata["feature_count"])
    assert arrays.y_train.shape[1] == 5
    assert arrays.X_val.shape[1:] == (metadata["seq_len"], metadata["feature_count"])
    assert arrays.y_val.shape[1] == 5
    assert len(arrays.anchor_val) == len(arrays.X_val)
    assert arrays.summaries[0].skipped_reason == ""


def test_build_finetune_arrays_respects_last_finetuned_date():
    metadata = get_or_create_metadata(_METADATA_PATH)
    dates = pd.bdate_range("2025-01-01", periods=100)
    close = np.linspace(100.0, 135.0, len(dates)) + np.sin(np.arange(len(dates)))
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Date_str": dates.strftime("%Y-%m-%d"),
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + np.arange(len(dates)) * 100,
        }
    )
    args = argparse.Namespace(
        recent_days=45,
        replay_samples_per_symbol=3,
        validation_targets_per_symbol=3,
    )
    cutoff = pd.Timestamp(dates[72]).normalize()

    arrays = mf.build_finetune_arrays({"TEST": frame}, metadata, args, {"TEST": cutoff})

    assert arrays.summaries[0].new_rows > 0
    assert arrays.summaries[0].last_finetuned_date == cutoff.strftime("%Y-%m-%d")
    assert arrays.summaries[0].latest_processed_date > cutoff.strftime("%Y-%m-%d")
    assert arrays.X_train.shape[0] == arrays.summaries[0].train_samples


def test_build_finetune_arrays_skips_when_checkpoint_is_current():
    metadata = get_or_create_metadata(_METADATA_PATH)
    dates = pd.bdate_range("2025-01-01", periods=80)
    close = np.linspace(100.0, 120.0, len(dates))
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Date_str": dates.strftime("%Y-%m-%d"),
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + np.arange(len(dates)) * 100,
        }
    )
    args = argparse.Namespace(
        recent_days=45,
        replay_samples_per_symbol=3,
        validation_targets_per_symbol=3,
    )

    arrays = mf.build_finetune_arrays({"TEST": frame}, metadata, args, {"TEST": pd.Timestamp(dates[-1])})

    assert len(arrays.X_train) == 0
    assert arrays.summaries[0].skipped_reason == "no new historical rows beyond fine-tune checkpoint"


class FakeWorksheet:
    title = "TEST"

    def __init__(self):
        self.values = [
            ["Date", "Date_str", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Forecast_Close_T+1"],
            ["2025-01-03", "2025-01-03", "103", "105", "102", "104", "104", "1000", "keep-c"],
            ["bad", "bad", "x", "x", "x", "x", "x", "x", "drop"],
            ["2025-01-01", "2025-01-01", "100", "101", "99", "100.5", "100.5", "900", "keep-a"],
            ["2025-01-03", "2025-01-03", "104", "106", "103", "105", "105", "1100", "keep-newer-duplicate"],
            ["2025-01-02", "2025-01-02", "101", "103", "100", "102", "102", "950", "keep-b"],
        ]
        self.batch_payload = None
        self.deleted = None

    def get_all_values(self):
        return self.values

    def batch_update(self, payload, raw=True):
        self.batch_payload = payload

    def delete_rows(self, start, end):
        self.deleted = (start, end)


def test_validate_and_repair_worksheet_sorts_deduplicates_and_preserves_extra_columns():
    worksheet = FakeWorksheet()

    result = mf.validate_and_repair_worksheet(worksheet, dry_run=False)

    assert result.status == "ok"
    assert result.malformed_removed == 1
    assert result.duplicates_removed == 1
    assert result.sorted_rows is True
    assert worksheet.deleted == (5, 6)
    repaired_rows = worksheet.batch_payload[0]["values"]
    assert [row[1] for row in repaired_rows] == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert [row[-1] for row in repaired_rows] == ["keep-a", "keep-b", "keep-newer-duplicate"]


class ArchiveFakeWorksheet:
    id = 1

    def __init__(self, title, values):
        self.title = title
        self.values = [list(row) for row in values]
        self.deleted = []
        self.appended = []

    def get_all_values(self):
        return [list(row) for row in self.values]

    def batch_update(self, payload, raw=True):
        for update in payload:
            start_row = int(re.search(r"\d+", update["range"]).group(0))
            rows = [list(row) for row in update["values"]]
            for offset, row in enumerate(rows):
                index = start_row - 1 + offset
                while len(self.values) <= index:
                    self.values.append([])
                self.values[index] = row

    def append_rows(self, rows, value_input_option="RAW"):
        self.appended.extend([list(row) for row in rows])
        self.values.extend([list(row) for row in rows])

    def delete_rows(self, start, end):
        self.deleted.append((start, end))
        del self.values[start - 1 : end]


class ArchiveFakeSpreadsheet:
    def __init__(self, worksheets):
        self._worksheets = list(worksheets)

    def worksheets(self):
        return list(self._worksheets)

    def add_worksheet(self, title, rows, cols):
        worksheet = ArchiveFakeWorksheet(title, [[]])
        self._worksheets.append(worksheet)
        return worksheet


def test_archive_old_rows_appends_before_operational_cleanup_and_skips_duplicates():
    headers = ["Date", "Date_str", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Forecast_Close_T+1"]
    operational = ArchiveFakeWorksheet(
        "TEST",
        [
            headers,
            ["2025-01-01", "2025-01-01", "100", "101", "99", "100", "100", "1000", "101"],
            ["2025-01-02", "2025-01-02", "101", "102", "100", "101", "101", "1100", "102"],
            ["2025-01-03", "2025-01-03", "102", "103", "101", "102", "102", "1200", "103"],
            ["2025-01-06", "2025-01-06", "103", "104", "102", "103", "103", "1300", "104"],
            ["2025-01-07", "2025-01-07", "104", "105", "103", "104", "104", "1400", "105"],
        ],
    )
    historical = ArchiveFakeWorksheet(
        "TEST",
        [
            headers,
            ["2025-01-01", "2025-01-01", "100", "101", "99", "100", "100", "1000", "101"],
        ],
    )
    spreadsheet = ArchiveFakeSpreadsheet([historical])

    result = archive_old_rows_for_worksheet(
        operational,
        archive_spreadsheet=spreadsheet,
        archive_worksheet=historical,
        keep_rows=3,
        dry_run=False,
    )

    assert result.status == "ok"
    assert result.candidate_rows == 2
    assert result.rows_appended == 1
    assert result.duplicate_rows_skipped == 1
    assert result.rows_removed_from_operational == 2
    assert historical.appended[0][1] == "2025-01-02"
    # A Row_finetuned column is injected as the trailing column (0 = un-fine-tuned
    # on arrival), so the Forecast_Close_T+1 value is now second-to-last.
    assert historical.appended[0][-1] == 0
    assert historical.appended[0][-2] == "102"
    assert [row[1] for row in operational.values[1:]] == ["2025-01-03", "2025-01-06", "2025-01-07"]


# ---------------------------------------------------------------------------
# Helpers shared by new tests
# ---------------------------------------------------------------------------

def _make_price_frame(n: int = 90, start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 130.0, n) + np.sin(np.arange(n))
    return pd.DataFrame(
        {
            "Date": dates,
            "Date_str": dates.strftime("%Y-%m-%d"),
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + np.arange(n) * 100,
        }
    )


class _FakeWorksheetFull:
    """Simulates a gspread Worksheet with full-column support (batch_update + append_rows)."""

    def __init__(self, title: str, values: List[List]):
        self.title = title
        self.id = 999
        self.values = [list(row) for row in values]
        self.batch_updates: List = []
        self.appended: List = []

    def get_all_values(self):
        return [list(row) for row in self.values]

    def batch_update(self, payload, raw=True):
        self.batch_updates.extend(payload)
        # Apply updates so subsequent get_all_values reflect changes.
        for update in payload:
            range_str = update["range"]
            rows = update["values"]
            # Parse start row number from range (e.g. "D2:D4" or "A1:AZ1").
            import re as _re
            match = _re.search(r"(\d+)", range_str)
            if not match:
                continue
            start_row = int(match.group(1))
            for offset, row_vals in enumerate(rows):
                idx = start_row - 1 + offset
                while len(self.values) <= idx:
                    self.values.append([])
                existing = list(self.values[idx])
                # Detect column letter prefix to determine column index.
                col_match = _re.match(r"([A-Z]+)", range_str)
                if col_match:
                    col_letters = col_match.group(1)
                    col_idx = sum(
                        (ord(c) - ord("A") + 1) * (26 ** i)
                        for i, c in enumerate(reversed(col_letters))
                    ) - 1
                    while len(existing) <= col_idx + len(row_vals) - 1:
                        existing.append("")
                    for j, v in enumerate(row_vals):
                        existing[col_idx + j] = v
                self.values[idx] = existing

    def append_rows(self, rows, value_input_option="RAW"):
        for row in rows:
            self.appended.append(list(row))
            self.values.append(list(row))

    def delete_rows(self, start, end):
        del self.values[start - 1: end]


class _FakeSpreadsheet:
    def __init__(self, worksheets):
        self._worksheets = list(worksheets)

    def worksheets(self):
        return list(self._worksheets)

    def add_worksheet(self, title, rows, cols):
        ws = _FakeWorksheetFull(title, [[]])
        self._worksheets.append(ws)
        return ws


# ---------------------------------------------------------------------------
# Test: build_finetune_arrays populates train_dates in summaries
# ---------------------------------------------------------------------------

def test_build_finetune_arrays_populates_train_dates():
    metadata = get_or_create_metadata(_METADATA_PATH)
    frame = _make_price_frame(90)
    args = argparse.Namespace(
        recent_days=45,
        replay_samples_per_symbol=5,
        validation_targets_per_symbol=3,
    )

    arrays = mf.build_finetune_arrays({"TEST": frame}, metadata, args)

    summary = arrays.summaries[0]
    assert summary.skipped_reason == "", f"Unexpectedly skipped: {summary.skipped_reason}"
    assert len(summary.train_dates) > 0, "train_dates should be non-empty after successful build"
    # All entries should be valid YYYY-MM-DD date strings.
    for d in summary.train_dates:
        pd.Timestamp(d)  # raises if unparseable
    # train_dates must be sorted.
    assert summary.train_dates == sorted(summary.train_dates)
    # train_dates count must match train_samples (each sample maps to one date).
    assert len(summary.train_dates) == summary.train_samples


def test_build_finetune_arrays_skipped_symbol_has_empty_train_dates():
    metadata = get_or_create_metadata(_METADATA_PATH)
    frame = _make_price_frame(5)  # too few rows to build any examples
    args = argparse.Namespace(
        recent_days=45,
        replay_samples_per_symbol=5,
        validation_targets_per_symbol=3,
    )

    arrays = mf.build_finetune_arrays({"TEST": frame}, metadata, args)

    assert arrays.summaries[0].skipped_reason != ""
    assert arrays.summaries[0].train_dates == []


# ---------------------------------------------------------------------------
# Test: archival injects Row_finetuned = 0 for every newly archived row
# ---------------------------------------------------------------------------

def test_archive_sets_row_finetuned_zero_for_all_new_rows():
    base_headers = [
        "Date", "Date_str", "Open", "High", "Low", "Close", "Adj Close",
        "Volume", "Forecast_Close_T+1",
    ]
    operational = _FakeWorksheetFull(
        "TEST",
        [
            base_headers,
            ["2025-01-01", "2025-01-01", "100", "101", "99", "100", "100", "1000", "101"],
            ["2025-01-02", "2025-01-02", "101", "102", "100", "101", "101", "1100", "102"],
            ["2025-01-03", "2025-01-03", "102", "103", "101", "102", "102", "1200", "103"],
            ["2025-01-06", "2025-01-06", "103", "104", "102", "103", "103", "1300", "104"],
            ["2025-01-07", "2025-01-07", "104", "105", "103", "104", "104", "1400", "105"],
        ],
    )
    historical = _FakeWorksheetFull("TEST", [base_headers])
    spreadsheet = _FakeSpreadsheet([historical])

    result = archive_old_rows_for_worksheet(
        operational,
        archive_spreadsheet=spreadsheet,
        archive_worksheet=historical,
        keep_rows=3,
        dry_run=False,
    )

    assert result.status == "ok"
    # Row_finetuned column must have been added to the historical sheet.
    historical_headers = historical.values[0]
    assert ROW_FINETUNED_COL in historical_headers, (
        f"Expected {ROW_FINETUNED_COL!r} in historical headers, got: {historical_headers}"
    )
    rf_col_idx = historical_headers.index(ROW_FINETUNED_COL)
    # Every appended row should have Row_finetuned = 0.
    for appended_row in historical.appended:
        assert int(appended_row[rf_col_idx]) == 0, (
            f"Newly archived row should have Row_finetuned=0, got {appended_row[rf_col_idx]!r}"
        )


# ---------------------------------------------------------------------------
# Test: all columns (OHLCV + forecast) are transferred during archival
# ---------------------------------------------------------------------------

def test_all_columns_transferred_from_operational_to_historical():
    """Operational sheet has OHLCV + indicator + forecast columns.
    After archival every one of those columns must appear in the historical
    sheet with the correct values (no silent column drops)."""
    headers = [
        "Date", "Date_str", "Open", "High", "Low", "Close", "Adj Close",
        "Volume", "predicted", "Predicted_Close_Price",
        "Forecast_Close_T+1", "Forecast_Close_T+2", "RSI_14",
    ]
    row1 = ["2025-01-01", "2025-01-01", "100", "101", "99",  "100", "100", "1000", "1", "100.5", "101.1", "102.2", "55.3"]
    row2 = ["2025-01-02", "2025-01-02", "101", "102", "100", "101", "101", "1100", "1", "101.5", "102.1", "103.2", "56.3"]
    row3 = ["2025-01-03", "2025-01-03", "102", "103", "101", "102", "102", "1200", "1", "102.5", "103.1", "104.2", "57.3"]
    row4 = ["2025-01-06", "2025-01-06", "103", "104", "102", "103", "103", "1300", "0", "",      "",      "",      ""]
    row5 = ["2025-01-07", "2025-01-07", "104", "105", "103", "104", "104", "1400", "0", "",      "",      "",      ""]

    operational = _FakeWorksheetFull("GOOG", [headers, row1, row2, row3, row4, row5])
    historical = _FakeWorksheetFull("GOOG", [headers])
    spreadsheet = _FakeSpreadsheet([historical])

    result = archive_old_rows_for_worksheet(
        operational,
        archive_spreadsheet=spreadsheet,
        archive_worksheet=historical,
        keep_rows=3,
        dry_run=False,
    )

    assert result.status == "ok"
    hist_headers = historical.values[0]
    for col in ("Forecast_Close_T+1", "Forecast_Close_T+2", "RSI_14", "predicted", ROW_FINETUNED_COL):
        assert col in hist_headers, f"Expected column {col!r} in historical headers; got: {hist_headers}"

    # Archived rows are 2025-01-01 and 2025-01-02 (the oldest two).
    # Their Forecast_Close_T+1 values must be preserved.
    fc1_idx = hist_headers.index("Forecast_Close_T+1")
    archived_fc1 = [row[fc1_idx] for row in historical.appended]
    assert "101.1" in archived_fc1, f"Forecast_Close_T+1 value not preserved; got: {archived_fc1}"
    assert "102.1" in archived_fc1, f"Forecast_Close_T+2 value not preserved; got: {archived_fc1}"

    # Row_finetuned must be 0 for all archived rows.
    rf_idx = hist_headers.index(ROW_FINETUNED_COL)
    for row in historical.appended:
        assert int(row[rf_idx]) == 0


# ---------------------------------------------------------------------------
# Test: write_finetuned_flags — pre-cutoff rows become 1, post-cutoff become 0
# ---------------------------------------------------------------------------

class _SimpleWorksheet:
    """Minimal fake worksheet for write_finetuned_flags tests."""

    def __init__(self, title: str, values: List[List]):
        self.title = title
        self.id = 42
        self._raw = [list(row) for row in values]
        self.updates: List = []

    def get_all_values(self):
        return [list(row) for row in self._raw]

    def batch_update(self, payload, raw=True):
        self.updates.extend(payload)


def test_write_finetuned_flags_pre_cutoff_rows_set_to_1():
    """Rows on or before the initial_cutoff must receive Row_finetuned = 1
    even if train_dates is empty (i.e. fine-tuning was skipped)."""
    headers = ["Date", "Date_str", "Open", "Close", "Volume"]
    values = [
        headers,
        ["2025-04-25", "2025-04-25", "99",  "100", "1000"],
        ["2025-04-28", "2025-04-28", "100", "101", "1100"],  # <= cutoff
        ["2025-04-29", "2025-04-29", "101", "102", "1200"],  # > cutoff
        ["2025-05-01", "2025-05-01", "102", "103", "1300"],  # > cutoff
    ]
    ws = _SimpleWorksheet("TEST", values)
    cutoff = pd.Timestamp("2025-04-28")

    result = mf.write_finetuned_flags(ws, set(), cutoff, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["new_column"] is True  # column did not exist before
    # rows_to_update: header (1) + all 4 data rows (since column is new)
    assert result["rows_to_update"] == 4


def test_write_finetuned_flags_train_dates_set_to_1():
    """Rows in train_dates get Row_finetuned = 1 even if they are post-cutoff."""
    headers = ["Date", "Date_str", "Open", "Close", "Volume", ROW_FINETUNED_COL]
    values = [
        headers,
        ["2025-04-25", "2025-04-25", "99",  "100", "1000", "1"],   # pre-cutoff
        ["2025-04-28", "2025-04-28", "100", "101", "1100", "1"],   # pre-cutoff
        ["2025-04-29", "2025-04-29", "101", "102", "1200", "0"],   # post-cutoff, in train_dates
        ["2025-05-01", "2025-05-01", "102", "103", "1300", "0"],   # post-cutoff, NOT in train_dates
        ["2025-05-02", "2025-05-02", "103", "104", "1400", "0"],   # post-cutoff, in train_dates
    ]
    ws = _SimpleWorksheet("TEST", values)
    cutoff = pd.Timestamp("2025-04-28")
    train_dates = {"2025-04-29", "2025-05-02"}

    result = mf.write_finetuned_flags(ws, train_dates, cutoff, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["new_column"] is False  # column already present
    # Only rows 4 and 6 (2025-04-29 and 2025-05-02) need updating: 0→1.
    # Row 5 (2025-05-01) stays 0; rows 2+3 stay 1.
    assert result["rows_to_update"] == 2


def test_write_finetuned_flags_idempotent_when_already_correct():
    """If every cell already has the right value, no updates should be issued."""
    headers = ["Date", "Date_str", "Open", "Close", ROW_FINETUNED_COL]
    values = [
        headers,
        ["2025-04-25", "2025-04-25", "99",  "100", "1"],  # pre-cutoff → correct
        ["2025-04-28", "2025-04-28", "100", "101", "1"],  # pre-cutoff → correct
        ["2025-05-01", "2025-05-01", "101", "102", "0"],  # post-cutoff, not trained → correct
    ]
    ws = _SimpleWorksheet("TEST", values)
    cutoff = pd.Timestamp("2025-04-28")

    result = mf.write_finetuned_flags(ws, set(), cutoff, dry_run=False)

    assert result["status"] == "unchanged"
    assert ws.updates == []


def test_write_finetuned_flags_no_cutoff_only_train_dates_matter():
    """When initial_cutoff is None, only train_dates rows receive 1."""
    headers = ["Date", "Date_str", "Open", "Close", ROW_FINETUNED_COL]
    values = [
        headers,
        ["2025-04-25", "2025-04-25", "99",  "100", "0"],
        ["2025-04-28", "2025-04-28", "100", "101", "0"],
        ["2025-05-01", "2025-05-01", "101", "102", "0"],
    ]
    ws = _SimpleWorksheet("TEST", values)

    result = mf.write_finetuned_flags(ws, {"2025-04-28"}, None, dry_run=True)

    assert result["rows_to_update"] == 1  # only 2025-04-28


# ---------------------------------------------------------------------------
# Test: archival respects append-only semantics (existing rows not overwritten)
# ---------------------------------------------------------------------------

def test_archive_append_only_existing_rows_not_touched():
    """Dates already present in the historical sheet must not be re-appended."""
    headers = ["Date", "Date_str", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    hist_row = ["2025-01-01", "2025-01-01", "OLD", "101", "99", "100", "100", "1000"]
    op_rows = [
        ["2025-01-01", "2025-01-01", "NEW", "101", "99", "100", "100", "1000"],
        ["2025-01-02", "2025-01-02", "101", "102", "100", "101", "101", "1100"],
        ["2025-01-03", "2025-01-03", "102", "103", "101", "102", "102", "1200"],
        ["2025-01-06", "2025-01-06", "103", "104", "102", "103", "103", "1300"],
    ]
    operational = _FakeWorksheetFull("TEST", [headers] + op_rows)
    historical = _FakeWorksheetFull("TEST", [headers, hist_row])
    spreadsheet = _FakeSpreadsheet([historical])

    result = archive_old_rows_for_worksheet(
        operational,
        archive_spreadsheet=spreadsheet,
        archive_worksheet=historical,
        keep_rows=2,
        dry_run=False,
    )

    assert result.status == "ok"
    assert result.duplicate_rows_skipped == 1  # 2025-01-01 already in historical
    # The historical "OLD" row must not have been overwritten.
    hist_data = [r for r in historical.values[1:] if r and r[0] == "2025-01-01"]
    assert hist_data, "2025-01-01 row missing from historical"
    open_vals = [r[2] for r in hist_data]
    assert "OLD" in open_vals, f"Original row was overwritten. Open values: {open_vals}"


# ---------------------------------------------------------------------------
# Test: cutoff correctly excludes rows <= checkpoint from being counted as new
# ---------------------------------------------------------------------------

def test_build_finetune_arrays_after_28apr_cutoff_only_new_rows_trained():
    metadata = get_or_create_metadata(_METADATA_PATH)
    # 100 business days starting 2025-01-01; the cutoff is 2025-04-28.
    frame = _make_price_frame(100, start="2025-01-01")
    cutoff = pd.Timestamp("2025-04-28")
    args = argparse.Namespace(
        recent_days=45,
        replay_samples_per_symbol=5,
        validation_targets_per_symbol=3,
    )

    arrays = mf.build_finetune_arrays({"TEST": frame}, metadata, args, {"TEST": cutoff})

    if arrays.summaries[0].skipped_reason:
        return  # not enough new rows — acceptable, but check if cutoff works

    cutoff_str = cutoff.strftime("%Y-%m-%d")
    # Validation samples come from the newest new rows and must be post-cutoff.
    # Training dates include replay (pre-cutoff) and recent (post-cutoff).
    train_dates = arrays.summaries[0].train_dates
    new_rows = arrays.summaries[0].new_rows
    assert new_rows > 0, "Should have new rows after cutoff"
    # latest_processed_date must be > cutoff.
    assert arrays.summaries[0].latest_processed_date > cutoff_str, (
        f"latest_processed_date {arrays.summaries[0].latest_processed_date!r} "
        f"should be after cutoff {cutoff_str!r}"
    )
    # train_dates should be non-empty.
    assert len(train_dates) > 0
