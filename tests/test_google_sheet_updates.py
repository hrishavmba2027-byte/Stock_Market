from app.services.google_sheet_updates import (
    PredictionSheetUpdateService,
    append_missing_columns,
    get_column_index,
)


class FakeWorksheet:
    title = "RELIANCE"
    id = 123

    def __init__(self, values):
        self.values = values
        self.updates = []

    def get_all_values(self):
        return self.values

    def batch_update(self, updates, raw=False):
        self.updates.extend(updates)


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
