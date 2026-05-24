from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

from app.langchain.chain import build_prediction_write_chain
from app.langchain.schemas import PredictionWriteDecision
from app.models.schemas import model_to_dict
from app.utils.logging import get_logger, log_event
from Feature_Engineering import (
    DECISION_FEATURE_COLUMNS,
    compute_decision_features,
    compute_decision_rsi_history,
    detect_date_column,
    find_close_column,
    rsi_percentile_thresholds,
)


PREDICTED_COL = "predicted"
PREDICTED_PRICE_COL = "Predicted_Close_Price"
FORECAST_CLOSE_PREFIX = "Forecast_Close_T+"


def forecast_close_columns() -> List[str]:
    return [f"{FORECAST_CLOSE_PREFIX}{horizon}" for horizon in range(1, 31)]


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_OVERWRITE_INTERNAL_COLS = {"__sheet_row_number", "__sort_position", "has_labels"}
_OVERWRITE_FORWARD_LABEL_PREFIX = "y_logret_h"
_OVERWRITE_NULL_VALUES = {"nan", "none", "nat", "inf", "-inf", ""}


def _clean_cell_for_sheets(value: Any) -> Any:
    """Convert a value to something safe for Google Sheets API."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return "" if not np.isfinite(value) else float(value)
    if isinstance(value, bool):
        return int(value)  # 0 / 1 rather than True / False
    text = str(value).strip()
    return "" if text.lower() in _OVERWRITE_NULL_VALUES else value


def overwrite_worksheet_with_engineered_data(
    worksheet: Any,
    engineered_frame: "pd.DataFrame",
    row_predictions: Dict[int, float],
    row_forecasts: Optional[Dict[int, List[float]]],
    *,
    dry_run: bool = False,
    forecast_horizon: int = 5,
) -> int:
    """Full-sheet overwrite: clear the worksheet and rewrite it with the
    complete feature-engineered data plus the latest model predictions.

    Every call:
      1. Merges ``row_predictions`` / ``row_forecasts`` back into the
         engineered DataFrame using ``__sheet_row_number``.
      2. Strips internal tracking columns and forward-label columns.
      3. Clears the worksheet (header + all data).
      4. Writes the header row + all data rows atomically via a single
         ``worksheet.update()`` call.

    Parameters
    ----------
    worksheet:
        A ``gspread.Worksheet`` object.
    engineered_frame:
        Full-row DataFrame produced by ``compute_indicators()`` (forward
        labels already dropped, internal columns still present).
    row_predictions:
        Mapping of original sheet row-number → predicted Close price.
    row_forecasts:
        Optional mapping of sheet row-number → list of horizon forecasts.
    dry_run:
        When True, skip all Sheets API calls and return the row count.
    forecast_horizon:
        Number of ``Forecast_Close_T+N`` columns to write.

    Returns
    -------
    int
        Number of data rows written to the sheet.
    """
    import pandas as pd  # local import — already a project dependency

    logger = get_logger(__name__)
    worksheet_title = str(getattr(worksheet, "title", ""))

    df = engineered_frame.copy().reset_index(drop=True)
    row_forecasts = row_forecasts or {}

    # ── Retrieve original sheet-row numbers before stripping them ───────────
    _SROW = "__sheet_row_number"
    if _SROW in df.columns:
        row_nums: List[int] = [
            int(v) if pd.notna(v) else 0
            for v in df[_SROW].tolist()
        ]
    else:
        row_nums = list(range(2, len(df) + 2))

    # ── Ensure output columns exist ─────────────────────────────────────────
    if "predicted" not in df.columns:
        df["predicted"] = 0
    if "Predicted_Close_Price" not in df.columns:
        df["Predicted_Close_Price"] = ""

    forecast_cols = [f"Forecast_Close_T+{h}" for h in range(1, forecast_horizon + 1)]
    for col in forecast_cols:
        if col not in df.columns:
            df[col] = ""

    # ── Merge predictions by original sheet-row number ───────────────────────
    pred_idx = df.columns.get_loc("predicted")
    price_idx = df.columns.get_loc("Predicted_Close_Price")
    fc_idxs = [df.columns.get_loc(c) for c in forecast_cols]

    rows_with_predictions = 0
    for i, sheet_row in enumerate(row_nums):
        if sheet_row in row_predictions:
            df.iat[i, pred_idx] = 1
            df.iat[i, price_idx] = round(float(row_predictions[sheet_row]), 6)
            rows_with_predictions += 1
            if sheet_row in row_forecasts:
                fvals = row_forecasts[sheet_row]
                for h_idx, fc_idx in enumerate(fc_idxs):
                    if h_idx < len(fvals) and math.isfinite(float(fvals[h_idx])):
                        df.iat[i, fc_idx] = round(float(fvals[h_idx]), 6)

    # ── Drop internal / forward-label columns ───────────────────────────────
    drop_cols = [
        c for c in df.columns
        if c in _OVERWRITE_INTERNAL_COLS
        or str(c).startswith(_OVERWRITE_FORWARD_LABEL_PREFIX)
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # ── Build list-of-lists for the Sheets API ───────────────────────────────
    headers = list(df.columns)
    data_rows: List[List[Any]] = [
        [_clean_cell_for_sheets(v) for v in row]
        for row in df.itertuples(index=False, name=None)
    ]
    all_values = [headers] + data_rows
    total_rows = len(data_rows)

    log_event(
        logger,
        logging.INFO,
        "worksheet_overwrite_prepared",
        "Full-sheet overwrite prepared",
        worksheet=worksheet_title,
        total_rows=total_rows,
        rows_with_predictions=rows_with_predictions,
        columns=len(headers),
        dry_run=dry_run,
    )

    if dry_run:
        return total_rows

    # ── Atomic clear + write ─────────────────────────────────────────────────
    try:
        worksheet.clear()
        worksheet.update(all_values, "A1")
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "worksheet_overwrite_failed",
            "Full-sheet overwrite failed",
            worksheet=worksheet_title,
            error=str(exc),
        )
        raise

    log_event(
        logger,
        logging.INFO,
        "worksheet_overwritten",
        "Full-sheet overwrite complete",
        worksheet=worksheet_title,
        rows_written=total_rows,
        rows_with_predictions=rows_with_predictions,
    )
    return total_rows


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
        row_forecasts: Optional[Dict[int, List[float]]] = None,
    ) -> int:
        if worksheet is None or not row_predictions:
            return 0
        row_forecasts = row_forecasts or {}

        if not self._validate_existing_worksheet(worksheet):
            return 0

        worksheet_title = str(getattr(worksheet, "title", ""))
        worksheet_id = str(getattr(worksheet, "id", getattr(worksheet, "_properties", {}).get("sheetId", "")))
        current_headers, row_values = self._read_current_values(worksheet, headers)
        forecast_columns = forecast_close_columns() if row_forecasts else []
        output_headers, header_changed = append_missing_columns(
            current_headers,
            [PREDICTED_COL, PREDICTED_PRICE_COL, *forecast_columns],
        )
        predicted_idx = get_column_index(output_headers, PREDICTED_COL)
        price_idx = get_column_index(output_headers, PREDICTED_PRICE_COL)
        if predicted_idx is None or price_idx is None:
            raise RuntimeError("required prediction output columns could not be resolved")
        forecast_indices = [get_column_index(output_headers, column) for column in forecast_columns]
        if forecast_columns and any(index is None for index in forecast_indices):
            raise RuntimeError("required forecast output columns could not be resolved")

        predicted_col = column_number_to_letter(predicted_idx + 1)
        price_col = column_number_to_letter(price_idx + 1)
        forecast_col_letters = [
            column_number_to_letter(int(index) + 1)
            for index in forecast_indices
            if index is not None
        ]
        forecast_columns_contiguous = bool(forecast_indices) and forecast_indices == list(
            range(int(forecast_indices[0]), int(forecast_indices[0]) + len(forecast_indices))
        )
        updates: List[Dict[str, Any]] = []
        if header_changed:
            updates.append(
                {
                    "range": f"A1:{column_number_to_letter(len(output_headers))}1",
                    "values": [output_headers],
                }
            )

        forecast_blocks: List[Tuple[int, List[List[float]]]] = []

        def append_forecast_row(sheet_row: int, values: List[float]) -> None:
            if not forecast_columns_contiguous:
                for column_letter, value in zip(forecast_col_letters, values):
                    updates.append({"range": f"{column_letter}{sheet_row}", "values": [[value]]})
                return
            if forecast_blocks and sheet_row == forecast_blocks[-1][0] + len(forecast_blocks[-1][1]):
                forecast_blocks[-1][1].append(values)
            else:
                forecast_blocks.append((sheet_row, [values]))

        validated_count = 0
        skipped_count = 0
        for sheet_row, prediction in sorted(row_predictions.items()):
            current_predicted_value = self._cell_value(row_values, sheet_row, predicted_idx)
            forecast_values = row_forecasts.get(int(sheet_row))
            should_write_forecast = forecast_values is not None
            decision = self._validate_prediction_write(
                worksheet_title,
                int(sheet_row),
                prediction,
                current_predicted_value,
            )
            if not decision.should_write and not should_write_forecast:
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
            if decision.should_write:
                updates.append({"range": f"{predicted_col}{sheet_row}", "values": [[1]]})
                updates.append(
                    {
                        "range": f"{price_col}{sheet_row}",
                        "values": [[round(float(decision.predicted_value), 6)]],
                    }
                )
            if forecast_values is not None:
                rounded_forecasts = [round(float(value), 6) for value in forecast_values]
                if not all(math.isfinite(value) for value in rounded_forecasts):
                    raise ValueError(f"{worksheet_title}: forecast values for row {sheet_row} must be finite")
                if len(rounded_forecasts) != len(forecast_columns):
                    raise ValueError(
                        f"{worksheet_title}: expected {len(forecast_columns)} forecast values "
                        f"for row {sheet_row}, got {len(rounded_forecasts)}"
                    )
                append_forecast_row(int(sheet_row), rounded_forecasts)

        for start_row, values in forecast_blocks:
            end_row = start_row + len(values) - 1
            updates.append(
                {
                    "range": f"{forecast_col_letters[0]}{start_row}:{forecast_col_letters[-1]}{end_row}",
                    "values": values,
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
            forecast_columns_written=len(forecast_columns),
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

    @staticmethod
    def _forecast_values_missing(
        row_values: List[List[Any]],
        sheet_row: int,
        zero_based_cols: List[int],
    ) -> bool:
        data_index = sheet_row - 2
        if data_index < 0 or data_index >= len(row_values):
            return True
        row = row_values[data_index]
        for col in zero_based_cols:
            if col >= len(row):
                return True
            text = str(row[col]).strip()
            if not text:
                return True
            try:
                value = float(text.replace(",", ""))
            except ValueError:
                return True
            if not math.isfinite(value):
                return True
        return False

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


# ===========================================================================
# Decision-feature sheet sync
# ---------------------------------------------------------------------------
# Writes 11 categorical decision columns (DECISION_FEATURE_COLUMNS) into BOTH
# the training sheet and the daily forecasting sheet. These columns are for
# Claude's decision layer ONLY — they are never added to pipeline_metadata.json
# feature_columns, so model training/inference ignores them entirely.
#
# The write is idempotent: existing decision columns are updated in place
# (header match is case-insensitive, so no duplicate headers are created);
# missing ones are appended in canonical order. Existing data and column order
# are preserved, and no rows are deleted.
# ===========================================================================

_NULL_TEXT = {"nan", "none", "nat", "inf", "-inf", ""}
_BATCH_UPDATE_MAX_CELLS = 50000


def _decision_cell(value: Any) -> Any:
    """Normalise a feature value for a Google Sheets cell (blank for null)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return "" if text.lower() in _NULL_TEXT else text


def _decision_cells_equal(existing: Any, desired: Any) -> bool:
    """Return True when a Sheet cell already contains the desired value."""
    desired_cell = _decision_cell(desired)
    existing_text = str(existing).strip()
    if desired_cell == "":
        return existing_text.lower() in _NULL_TEXT
    if isinstance(desired_cell, (int, float)) and not isinstance(desired_cell, bool):
        try:
            existing_number = float(existing_text.replace(",", ""))
        except ValueError:
            return False
        return math.isfinite(existing_number) and math.isclose(
            existing_number, float(desired_cell), rel_tol=0.0, abs_tol=1e-9
        )
    return existing_text == str(desired_cell)


def _dedupe_headers(headers: List[str]) -> List[str]:
    """Make header names unique so they can index a DataFrame safely."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for index, header in enumerate(headers):
        text = str(header).strip() or f"Column_{index + 1}"
        key = text.lower()
        count = seen.get(key, 0)
        seen[key] = count + 1
        out.append(text if count == 0 else f"{text}__dup{count + 1}")
    return out


def _parse_sheet_dates(values: Any) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def _absolute_range_name(sheet_title: str, range_name: str) -> str:
    try:
        from gspread.utils import absolute_range_name

        return absolute_range_name(sheet_title, range_name)
    except Exception:  # noqa: BLE001
        escaped = str(sheet_title).replace("'", "''")
        return f"'{escaped}'!{range_name}"


def _worksheet_read_range(worksheet: Any) -> str:
    return _absolute_range_name(str(getattr(worksheet, "title", "")), "A:ZZ")


def _update_cell_count(update: Dict[str, Any]) -> int:
    return sum(len(row) for row in update.get("values", []) or [])


class DecisionFeatureSheetSync:
    """Compute and write the decision-layer categorical features to both sheets."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    # -- public API -------------------------------------------------------
    def sync(
        self,
        operational_spreadsheet: Any,
        training_spreadsheet: Any,
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Sync decision features across both spreadsheets.

        RSI percentile thresholds are pooled per symbol from the FULL history of
        BOTH sheets. The operational sheet (a rolling 30-row window) is computed
        with the training sheet's older history prepended as look-back context,
        so its recent rows still get correct long-window indicators.
        """
        operational = self._collect(operational_spreadsheet, "operational")
        training = self._collect(training_spreadsheet, "training")

        # Pool RSI history per symbol across both sheets -> dynamic thresholds.
        rsi_history: Dict[str, Dict[int, List[float]]] = {}
        for record in training + operational:
            if record["named"] is None:
                continue
            pooled = rsi_history.setdefault(record["symbol"], {7: [], 14: []})
            history = compute_decision_rsi_history(self._chronological_frame(record["named"]))
            pooled[7].extend(history.get(7, []))
            pooled[14].extend(history.get(14, []))
        thresholds = {
            symbol: (
                rsi_percentile_thresholds(values[7]),
                rsi_percentile_thresholds(values[14]),
            )
            for symbol, values in rsi_history.items()
        }

        # Training-sheet close history is look-back context for the live sheet.
        training_context: Dict[str, Any] = {}
        for record in training:
            if record["named"] is not None and record["symbol"] not in training_context:
                training_context[record["symbol"]] = self._date_close_context(record["named"])

        results: List[Dict[str, Any]] = []
        for tag, records, spreadsheet in (
            ("training", training, training_spreadsheet),
            ("operational", operational, operational_spreadsheet),
        ):
            pending_updates: List[Dict[str, Any]] = []
            batch_mode = bool(
                not dry_run
                and spreadsheet is not None
                and hasattr(spreadsheet, "values_batch_update")
            )
            result_start = len(results)
            for record in records:
                thr7, thr14 = thresholds.get(record["symbol"], (None, None))
                context = training_context.get(record["symbol"]) if tag == "operational" else None
                result = self._process_worksheet(
                    record, thr7, thr14, context, dry_run,
                    batch_updates=pending_updates if batch_mode else None,
                )
                result["sheet"] = tag
                result["symbol"] = record["symbol"]
                results.append(result)
            if batch_mode and pending_updates:
                try:
                    self._flush_batch_updates(spreadsheet, pending_updates)
                except Exception as exc:  # noqa: BLE001
                    for result in results[result_start:]:
                        if result["status"] == "ok":
                            result["status"] = "error"
                            result["error"] = f"write: {exc}"
                    log_event(
                        self.logger, logging.ERROR, "decision_sync_batch_write_failed",
                        "Decision feature batch write failed", sheet=tag, error=str(exc),
                    )

        summary = self._summarise(results, dry_run)
        log_event(
            self.logger,
            logging.INFO,
            "decision_feature_sync_complete",
            "Decision feature sync finished",
            **{k: summary[k] for k in ("worksheets_written", "worksheets_skipped", "summary")},
        )
        return summary

    # -- worksheet collection --------------------------------------------
    def _collect(self, spreadsheet: Any, tag: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if spreadsheet is None:
            return records
        try:
            worksheets = spreadsheet.worksheets()
        except Exception as exc:  # noqa: BLE001
            log_event(
                self.logger, logging.ERROR, "decision_sync_list_failed",
                "Could not list worksheets", sheet=tag, error=str(exc),
            )
            return records
        if hasattr(spreadsheet, "values_batch_get"):
            ranges = [_worksheet_read_range(worksheet) for worksheet in worksheets]
            try:
                response = spreadsheet.values_batch_get(
                    ranges, params={"majorDimension": "ROWS"}
                )
                value_ranges = response.get("valueRanges", [])
                values_by_index = [
                    value_range.get("values", []) for value_range in value_ranges
                ]
                if len(values_by_index) != len(worksheets):
                    raise RuntimeError(
                        f"expected {len(worksheets)} value ranges, got {len(values_by_index)}"
                    )
                return [
                    self._record_from_values(worksheet, values)
                    for worksheet, values in zip(worksheets, values_by_index)
                ]
            except Exception as exc:  # noqa: BLE001
                log_event(
                    self.logger, logging.WARNING, "decision_sync_batch_read_failed",
                    "Batch worksheet read failed; falling back to per-worksheet reads",
                    sheet=tag, error=str(exc),
                )
        for worksheet in worksheets:
            try:
                values = worksheet.get_all_values()
            except Exception as exc:  # noqa: BLE001
                symbol = str(getattr(worksheet, "title", "")).strip().upper()
                records.append({
                    "symbol": symbol, "worksheet": worksheet,
                    "values": None, "named": None, "read_error": str(exc),
                })
                continue
            records.append(self._record_from_values(worksheet, values))
        return records

    # -- helpers ----------------------------------------------------------
    def _record_from_values(self, worksheet: Any, values: List[List[Any]]) -> Dict[str, Any]:
        symbol = str(getattr(worksheet, "title", "")).strip().upper()
        named = None
        if values and len(values) >= 2:
            try:
                named = self._named_frame(values)[2]
            except Exception:  # noqa: BLE001
                named = None
        return {
            "symbol": symbol, "worksheet": worksheet,
            "values": values, "named": named,
        }

    @staticmethod
    def _named_frame(values: List[List[Any]]) -> Tuple[List[str], List[List[Any]], pd.DataFrame]:
        """Build a DataFrame WITHOUT dropping rows.

        Positional row r maps exactly to sheet row r + 2, which keeps the
        row mapping intact even when the sheet contains blank rows.
        """
        headers = [str(header).strip() for header in values[0]]
        width = len(headers)
        rows = values[1:]
        padded = [list(row)[:width] + [""] * (width - len(row)) for row in rows]
        named = pd.DataFrame(padded, columns=_dedupe_headers(headers))
        return headers, rows, named

    @staticmethod
    def _date_close_context(named: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Chronological Date/Close frame used as look-back context."""
        close_col = find_close_column(named)
        if close_col is None:
            return None
        date_col = detect_date_column(named)
        close = pd.to_numeric(named[close_col], errors="coerce")
        if date_col:
            dates = _parse_sheet_dates(named[date_col])
        else:
            dates = pd.Series(pd.NaT, index=named.index)
        context = pd.DataFrame({"__date__": dates, "__close__": close})
        return context.sort_values("__date__", kind="stable")

    @staticmethod
    def _chronological_frame(named: pd.DataFrame) -> pd.DataFrame:
        date_col = detect_date_column(named)
        if not date_col:
            return named
        dates = _parse_sheet_dates(named[date_col])
        return named.loc[dates.sort_values(kind="stable").index]

    def _compute_features(
        self,
        named: pd.DataFrame,
        thr7: Optional[Tuple[float, float, float]],
        thr14: Optional[Tuple[float, float, float]],
        context: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute the 11 features for one worksheet, preserving physical order."""
        date_col = detect_date_column(named)
        if date_col:
            dates = _parse_sheet_dates(named[date_col])
            order = dates.sort_values(kind="stable").index
        else:
            order = named.index
        ordered = named.loc[order]

        close_col = find_close_column(ordered)
        ordered_close = pd.to_numeric(ordered[close_col], errors="coerce")

        # Prepend training history strictly older than this worksheet's window.
        context_close = pd.Series([], dtype="float64")
        if context is not None and len(context):
            usable = context
            if date_col:
                first = _parse_sheet_dates(ordered[date_col]).min()
                if pd.notna(first):
                    usable = context[context["__date__"] < first]
            context_close = pd.to_numeric(usable["__close__"], errors="coerce")

        close_values = (
            ordered_close.reset_index(drop=True)
            if context_close.empty
            else pd.concat([context_close, ordered_close], ignore_index=True)
        )
        comp = pd.DataFrame({"Close": close_values})
        features_all = compute_decision_features(
            comp, rsi7_thresholds=thr7, rsi14_thresholds=thr14
        )
        features = features_all.iloc[len(context_close):].copy()
        features.index = order
        return features.reindex(named.index)

    def _process_worksheet(
        self,
        record: Dict[str, Any],
        thr7: Optional[Tuple[float, float, float]],
        thr14: Optional[Tuple[float, float, float]],
        context: Optional[pd.DataFrame],
        dry_run: bool,
        batch_updates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        worksheet = record["worksheet"]
        title = str(getattr(worksheet, "title", ""))
        values = record["values"]
        if record.get("read_error"):
            return {"worksheet": title, "status": "error", "error": record["read_error"]}
        if not values or len(values) < 2:
            return {"worksheet": title, "status": "skipped", "reason": "empty"}
        named = record["named"]
        if named is None or find_close_column(named) is None:
            return {"worksheet": title, "status": "skipped", "reason": "no_close_column"}

        try:
            features = self._compute_features(named, thr7, thr14, context)
        except Exception as exc:  # noqa: BLE001
            return {"worksheet": title, "status": "error", "error": f"compute: {exc}"}

        headers = [str(header).strip() for header in values[0]]
        row_count = len(values) - 1
        output_headers, header_changed = append_missing_columns(
            headers, list(DECISION_FEATURE_COLUMNS)
        )
        updates: List[Dict[str, Any]] = []
        if header_changed:
            updates.append({
                "range": f"A1:{column_number_to_letter(len(output_headers))}1",
                "values": [output_headers],
            })
        for column in DECISION_FEATURE_COLUMNS:
            col_index = get_column_index(output_headers, column)
            if col_index is None:
                return {"worksheet": title, "status": "error",
                        "error": f"could not resolve column {column}"}
            letter = column_number_to_letter(col_index + 1)
            run_start: Optional[int] = None
            run_values: List[List[Any]] = []
            for offset, value in enumerate(features[column].tolist()):
                sheet_row = offset + 2
                row = values[offset + 1] if offset + 1 < len(values) else []
                existing = row[col_index] if col_index < len(row) else ""
                desired = _decision_cell(value)
                if col_index < len(headers) and _decision_cells_equal(existing, desired):
                    if run_start is not None:
                        end_row = run_start + len(run_values) - 1
                        updates.append({
                            "range": f"{letter}{run_start}:{letter}{end_row}",
                            "values": run_values,
                        })
                        run_start = None
                        run_values = []
                    continue
                if run_start is None:
                    run_start = sheet_row
                run_values.append([desired])
            if run_start is not None:
                end_row = run_start + len(run_values) - 1
                updates.append({
                    "range": f"{letter}{run_start}:{letter}{end_row}",
                    "values": run_values,
                })

        if not updates:
            return {"worksheet": title, "status": "unchanged", "rows": row_count,
                    "header_changed": False, "columns_written": 0}
        if dry_run:
            return {"worksheet": title, "status": "dry_run", "rows": row_count,
                    "header_changed": header_changed, "updates_planned": len(updates)}
        if batch_updates is not None:
            batch_updates.extend(
                {
                    "range": _absolute_range_name(title, update["range"]),
                    "values": update["values"],
                }
                for update in updates
            )
            return {
                "worksheet": title, "status": "ok", "rows": row_count,
                "header_changed": header_changed,
                "columns_written": len(DECISION_FEATURE_COLUMNS),
                "updates_written": len(updates),
            }
        try:
            worksheet.batch_update(updates, raw=False)
        except Exception as exc:  # noqa: BLE001
            log_event(
                self.logger, logging.ERROR, "decision_sync_write_failed",
                "Decision feature write failed", worksheet=title, error=str(exc),
            )
            return {"worksheet": title, "status": "error", "error": f"write: {exc}"}
        return {
            "worksheet": title, "status": "ok", "rows": row_count,
            "header_changed": header_changed,
            "columns_written": len(DECISION_FEATURE_COLUMNS),
            "updates_written": len(updates),
        }

    def _flush_batch_updates(self, spreadsheet: Any, updates: List[Dict[str, Any]]) -> None:
        chunk: List[Dict[str, Any]] = []
        chunk_cells = 0
        for update in updates:
            update_cells = _update_cell_count(update)
            if chunk and chunk_cells + update_cells > _BATCH_UPDATE_MAX_CELLS:
                self._send_batch_update(spreadsheet, chunk)
                chunk = []
                chunk_cells = 0
            chunk.append(update)
            chunk_cells += update_cells
        if chunk:
            self._send_batch_update(spreadsheet, chunk)

    @staticmethod
    def _send_batch_update(spreadsheet: Any, updates: List[Dict[str, Any]]) -> None:
        body = {
            "valueInputOption": "USER_ENTERED",
            "data": updates,
        }
        for attempt in range(1, 4):
            try:
                spreadsheet.values_batch_update(body)
                return
            except Exception as exc:  # noqa: BLE001
                text = str(exc)
                retryable = "429" in text or "Quota exceeded" in text or "Rate Limit" in text
                if attempt >= 3 or not retryable:
                    raise
                time.sleep(20 * attempt)

    @staticmethod
    def _summarise(results: List[Dict[str, Any]], dry_run: bool) -> Dict[str, Any]:
        written = [r for r in results if r["status"] in ("ok", "dry_run")]
        skipped = [r for r in results if r["status"] == "skipped"]
        unchanged = [r for r in results if r["status"] == "unchanged"]
        errors = [r for r in results if r["status"] == "error"]
        return {
            "dry_run": dry_run,
            "new_columns": list(DECISION_FEATURE_COLUMNS),
            "worksheets_processed": len(results),
            "worksheets_written": len(written),
            "worksheets_skipped": len(skipped),
            "worksheets_unchanged": len(unchanged),
            "worksheets_failed": len(errors),
            "errors": [{"worksheet": e.get("worksheet"), "sheet": e.get("sheet"),
                        "error": e.get("error")} for e in errors],
            "details": results,
            "summary": (
                f"{len(written)} written, {len(unchanged)} unchanged, "
                f"{len(skipped)} skipped, {len(errors)} errors"
            ),
        }


def sync_decision_features(
    operational_spreadsheet: Any,
    training_spreadsheet: Any,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper: sync decision features across both spreadsheets."""
    return DecisionFeatureSheetSync().sync(
        operational_spreadsheet, training_spreadsheet, dry_run=dry_run
    )
