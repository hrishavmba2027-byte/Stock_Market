import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import monthly_finetune as mf
from app.services.sheet_archival import archive_old_rows_for_worksheet


def test_forward_log_return_labels_anchor_to_previous_close():
    close = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=float)
    labels = mf.forward_log_return_labels(close, target_position=2, horizons=[1, 2, 3, 4, 5])

    assert labels is not None
    expected = np.log(np.asarray([12.0, 13.0, 14.0, 15.0, 16.0]) / 11.0)
    np.testing.assert_allclose(labels, expected.astype(np.float32))


def test_forecast_days_env_extends_forecast_columns(monkeypatch):
    metadata = json.loads(Path("outputs/pipeline_metadata.json").read_text())

    monkeypatch.setenv("FORECAST_DAYS", "15")
    assert mf.forecast.forecast_close_columns(metadata)[-1] == "Forecast_Close_T+15"

    monkeypatch.setenv("FORECAST_DAYS", "invalid")
    assert mf.forecast.forecast_close_columns(metadata)[-1] == "Forecast_Close_T+5"


def test_recursive_forecast_continues_beyond_direct_horizon(monkeypatch):
    metadata = json.loads(Path("outputs/pipeline_metadata.json").read_text())
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
    metadata = json.loads(Path("outputs/pipeline_metadata.json").read_text())
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

    assert arrays.X_train.shape[1:] == (20, 29)
    assert arrays.y_train.shape[1] == 5
    assert arrays.X_val.shape[1:] == (20, 29)
    assert arrays.y_val.shape[1] == 5
    assert len(arrays.anchor_val) == len(arrays.X_val)
    assert arrays.summaries[0].skipped_reason == ""


def test_build_finetune_arrays_respects_last_finetuned_date():
    metadata = json.loads(Path("outputs/pipeline_metadata.json").read_text())
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
    metadata = json.loads(Path("outputs/pipeline_metadata.json").read_text())
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
    assert historical.appended[0][-1] == "102"
    assert [row[1] for row in operational.values[1:]] == ["2025-01-03", "2025-01-06", "2025-01-07"]
