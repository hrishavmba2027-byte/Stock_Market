from app.services.google_sheet_updates import (
    DECISION_FEATURE_COLUMNS,
    DecisionFeatureSheetSync,
    PredictionSheetUpdateService,
    append_missing_columns,
    get_column_index,
)


class FakeWorksheet:
    def __init__(self, values, title="RELIANCE"):
        self.title = title
        self.id = 123
        self.values = values
        self.updates = []

    def get_all_values(self):
        return self.values

    def batch_update(self, updates, raw=False):
        self.updates.extend(updates)


class FakeSpreadsheet:
    def __init__(self, worksheets):
        self._worksheets = worksheets

    def worksheets(self):
        return self._worksheets


def test_get_column_index_is_dynamic_case_insensitive():
    headers = ["Date", "predicted", "Predicted_Close_Price"]
    assert get_column_index(headers, "PREDICTED") == 1
    assert get_column_index(headers, "predicted_close_price") == 2
    assert get_column_index(headers, "Predicted_Close_Price") == 2


def test_append_missing_columns_adds_required_outputs():
    headers, changed = append_missing_columns(["Date", "Close"], ["predicted", "Predicted_Close_Price"])
    assert changed is True
    assert headers == ["Date", "Close", "predicted", "Predicted_Close_Price"]


def test_update_prediction_status_writes_only_unprocessed_row():
    worksheet = FakeWorksheet(
        [
            ["Date", "Close", "predicted"],
            ["2026-01-01", "100", "0"],
            ["2026-01-02", "101", "1"],
        ]
    )
    written = PredictionSheetUpdateService().update_prediction_status(
        worksheet=worksheet,
        headers=["Date", "Close", "predicted"],
        row_predictions={2: 110.1234567, 3: 111.0},
    )
    assert written == 1
    ranges = [update["range"] for update in worksheet.updates]
    assert "A1:D1" in ranges
    assert "C2" in ranges
    assert "D2" in ranges
    assert "C3" not in ranges
    assert "D3" not in ranges


def test_update_prediction_status_refreshes_existing_forecasts_without_flag_change():
    forecast_headers = [f"Forecast_Close_T+{horizon}" for horizon in range(1, 31)]
    worksheet = FakeWorksheet(
        [
            ["Date", "Close", "predicted", "Predicted_Close_Price", *forecast_headers],
            ["2026-01-01", "100", "1", "101", *["99"] * 30],
        ]
    )
    forecasts = [100.0 + horizon for horizon in range(1, 31)]
    written = PredictionSheetUpdateService().update_prediction_status(
        worksheet=worksheet,
        headers=["Date", "Close", "predicted", "Predicted_Close_Price", *forecast_headers],
        row_predictions={2: 110.0},
        row_forecasts={2: forecasts},
    )
    assert written == 1
    ranges = [update["range"] for update in worksheet.updates]
    assert "C2" not in ranges
    assert "D2" not in ranges
    assert "E2:AH2" in ranges


def test_update_prediction_status_creates_forecast_columns():
    worksheet = FakeWorksheet(
        [
            ["Date", "Close", "predicted"],
            ["2026-01-01", "100", "0"],
        ]
    )
    forecasts = [100.0 + horizon for horizon in range(1, 31)]
    written = PredictionSheetUpdateService().update_prediction_status(
        worksheet=worksheet,
        headers=["Date", "Close", "predicted"],
        row_predictions={2: 110.0},
        row_forecasts={2: forecasts},
    )
    assert written == 1
    ranges = [update["range"] for update in worksheet.updates]
    assert "A1:AH1" in ranges
    assert "C2" in ranges
    assert "D2" in ranges
    assert "E2:AH2" in ranges


def _decision_values(row_count=70):
    values = [["Date", "Close"]]
    for idx in range(row_count):
        values.append([f"2026-01-{(idx % 28) + 1:02d}", str(100 + idx)])
    return values


def test_decision_feature_sync_writes_both_sheets():
    training = FakeWorksheet(_decision_values(), title="RELIANCE")
    operational = FakeWorksheet(_decision_values(35), title="RELIANCE")

    result = DecisionFeatureSheetSync().sync(
        FakeSpreadsheet([operational]),
        FakeSpreadsheet([training]),
    )

    assert result["worksheets_written"] == 2
    for worksheet in (training, operational):
        assert worksheet.updates
        header_update = worksheet.updates[0]
        assert header_update["range"].startswith("A1:")
        for column in DECISION_FEATURE_COLUMNS:
            assert column in header_update["values"][0]


def test_decision_feature_sync_is_idempotent_when_values_match():
    sync = DecisionFeatureSheetSync()
    base_values = _decision_values()
    named = sync._named_frame(base_values)[2]
    features = sync._compute_features(named, None, None, None)
    values = [["Date", "Close", *DECISION_FEATURE_COLUMNS]]
    for row, (_, feature_row) in zip(base_values[1:], features.iterrows()):
        values.append([*row, *[feature_row[column] for column in DECISION_FEATURE_COLUMNS]])

    worksheet = FakeWorksheet(values, title="RELIANCE")
    record = {
        "symbol": "RELIANCE",
        "worksheet": worksheet,
        "values": values,
        "named": sync._named_frame(values)[2],
    }

    result = sync._process_worksheet(record, None, None, None, dry_run=False)

    assert result["status"] == "unchanged"
    assert worksheet.updates == []
