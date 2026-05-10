from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.langchain.chain import build_prediction_write_chain
from app.langchain.schemas import PredictionWriteDecision
from app.models.schemas import model_to_dict
from app.utils.logging import get_logger, log_event


PREDICTED_COL = "predicted"
PREDICTED_PRICE_COL = "Predicted_Close_Price"


def column_number_to_letter(number: int) -> str:
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def get_column_index(headers: List[str], column_name: str) -> Optional[int]:
    wanted = column_name.strip().lower()
    for idx, header in enumerate(headers):
        if str(header).strip().lower() == wanted:
            return idx
    return None


def append_missing_columns(headers: List[str], required_columns: List[str]) -> Tuple[List[str], bool]:
    updated = list(headers)
    changed = False
    for column in required_columns:
        if get_column_index(updated, column) is None:
            updated.append(column)
            changed = True
    return updated, changed


class PredictionSheetUpdateService:
    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self.validation_chain = build_prediction_write_chain()

    def update_prediction_status(
        self,
        worksheet: Any,
        headers: List[str],
        row_predictions: Dict[int, float],
    ) -> int:
        if worksheet is None or not row_predictions:
            return 0

        if not self._validate_existing_worksheet(worksheet):
            return 0

        worksheet_title = str(getattr(worksheet, "title", ""))
        worksheet_id = str(getattr(worksheet, "id", getattr(worksheet, "_properties", {}).get("sheetId", "")))
        current_headers, row_values = self._read_current_values(worksheet, headers)
        output_headers, header_changed = append_missing_columns(
            current_headers,
            [PREDICTED_COL, PREDICTED_PRICE_COL],
        )
        predicted_idx = get_column_index(output_headers, PREDICTED_COL)
        price_idx = get_column_index(output_headers, PREDICTED_PRICE_COL)
        if predicted_idx is None or price_idx is None:
            raise RuntimeError("required prediction output columns could not be resolved")

        predicted_col = column_number_to_letter(predicted_idx + 1)
        price_col = column_number_to_letter(price_idx + 1)
        updates: List[Dict[str, Any]] = []
        if header_changed:
            updates.append(
                {
                    "range": f"A1:{column_number_to_letter(len(output_headers))}1",
                    "values": [output_headers],
                }
            )

        validated_count = 0
        skipped_count = 0
        for sheet_row, prediction in sorted(row_predictions.items()):
            current_predicted_value = self._cell_value(row_values, sheet_row, predicted_idx)
            decision = self._validate_prediction_write(
                worksheet_title,
                int(sheet_row),
                prediction,
                current_predicted_value,
            )
            if not decision.should_write:
                skipped_count += 1
                log_event(
                    self.logger,
                    logging.INFO,
                    "prediction_write_skipped",
                    "Prediction write skipped",
                    worksheet=worksheet_title,
                    worksheet_id=worksheet_id,
                    row_number=sheet_row,
                    decision=model_to_dict(decision),
                )
                continue

            validated_count += 1
            updates.append({"range": f"{predicted_col}{sheet_row}", "values": [[1]]})
            updates.append(
                {
                    "range": f"{price_col}{sheet_row}",
                    "values": [[round(float(decision.predicted_value), 6)]],
                }
            )

        if not updates:
            return 0

        worksheet.batch_update(updates, raw=False)
        log_event(
            self.logger,
            logging.INFO,
            "prediction_sheet_updated",
            "Prediction results written to Google Sheet",
            worksheet=worksheet_title,
            worksheet_id=worksheet_id,
            rows_written=validated_count,
            rows_skipped=skipped_count,
            header_created=header_changed,
            updated_ranges=[item["range"] for item in updates],
        )
        return validated_count

    def _validate_existing_worksheet(self, worksheet: Any) -> bool:
        title = str(getattr(worksheet, "title", "")).strip()
        worksheet_id = getattr(worksheet, "id", None) or getattr(worksheet, "_properties", {}).get("sheetId")
        if not title or worksheet_id is None:
            log_event(
                self.logger,
                logging.ERROR,
                "worksheet_validation_failed",
                "Worksheet is missing title or ID; skipping writes",
                worksheet=title,
                worksheet_id=worksheet_id,
            )
            return False
        return True

    def _read_current_values(self, worksheet: Any, fallback_headers: List[str]) -> Tuple[List[str], List[List[Any]]]:
        try:
            values = worksheet.get_all_values()
        except Exception:
            log_event(
                self.logger,
                logging.WARNING,
                "worksheet_read_before_write_failed",
                "Could not refresh worksheet values before writing; using pipeline headers",
                worksheet=getattr(worksheet, "title", ""),
                exc_info=True,
            )
            return list(fallback_headers), []

        if not values:
            return list(fallback_headers), []
        headers = [str(value).strip() for value in values[0]]
        return headers or list(fallback_headers), values[1:]

    @staticmethod
    def _cell_value(row_values: List[List[Any]], sheet_row: int, zero_based_col: int) -> Any:
        data_index = sheet_row - 2
        if data_index < 0 or data_index >= len(row_values):
            return 0
        row = row_values[data_index]
        if zero_based_col >= len(row):
            return 0
        return row[zero_based_col]

    def _validate_prediction_write(
        self,
        worksheet: str,
        row_number: int,
        predicted_value: float,
        current_predicted_value: Any,
    ) -> PredictionWriteDecision:
        decision = self.validation_chain.invoke(
            {
                "worksheet": worksheet,
                "row_number": row_number,
                "predicted_value": predicted_value,
                "current_predicted_value": current_predicted_value,
            }
        )
        if isinstance(decision, PredictionWriteDecision):
            return decision
        if hasattr(PredictionWriteDecision, "model_validate"):
            return PredictionWriteDecision.model_validate(decision)
        return PredictionWriteDecision.parse_obj(decision)

