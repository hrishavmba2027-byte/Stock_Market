from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import Data_update
import main as forecast
from app.pipeline.metadata import get_or_create_metadata
from app.services.sheet_archival import (
    DEFAULT_HISTORICAL_TRAINING_SHEET_ID,
    DEFAULT_OPERATIONAL_SHEET_ID,
    DEFAULT_ROLLING_OPERATIONAL_ROWS,
    ArchivalWorksheetResult,
    archive_old_rows_for_worksheet,
    find_worksheet_by_title,
    parse_positive_int,
)
from Feature_Engineering import compute_indicators


DEFAULT_TRAINING_SHEET_ID = DEFAULT_HISTORICAL_TRAINING_SHEET_ID
DEFAULT_OUTPUT_DIR = Path("outputs") / "monthly_finetune"
DEFAULT_STATE_FILE = Path("state") / "monthly_finetune_state.json"
DEFAULT_RECENT_DAYS = 45
DEFAULT_REPLAY_SAMPLES_PER_SYMBOL = 12
DEFAULT_VALIDATION_TARGETS_PER_SYMBOL = 5
DEFAULT_FINE_TUNE_EPOCHS = 8
DEFAULT_FINE_TUNE_PATIENCE = 3
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_BATCH_SIZE = 64
DEFAULT_GRAD_CLIP = 1.0
MIN_FINE_TUNE_TRAIN_SAMPLES = 16
DATE_COLUMNS = ("Date_str", "Date", "Date_")
REQUIRED_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
STATE_SCHEMA_VERSION = 1


@dataclass
class SheetValidationResult:
    worksheet: str
    status: str
    rows_before: int = 0
    rows_after: int = 0
    duplicates_removed: int = 0
    malformed_removed: int = 0
    sorted_rows: bool = False
    repaired_cells: int = 0
    reason: str = ""


@dataclass
class SymbolDatasetSummary:
    symbol: str
    rows: int
    new_rows: int
    train_samples: int
    validation_samples: int
    replay_samples: int
    last_finetuned_date: str = ""
    latest_processed_date: str = ""
    skipped_reason: str = ""


@dataclass
class FineTuneArrays:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    anchor_val: np.ndarray
    symbols_val: np.ndarray
    summaries: List[SymbolDatasetSummary] = field(default_factory=list)


def log(message: str) -> None:
    print(message, file=sys.stderr)


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic incremental fine-tuning workflow: archive overflow rows from "
            "the operational Google Sheet into the historical training sheet, then "
            "warm-start fine-tune the existing Dense/LSTM/Transformer checkpoints only "
            "on rows newer than the last successful fine-tune state."
        )
    )
    parser.add_argument(
        "--operational-sheet-id",
        default=os.environ.get("OPERATIONAL_SHEET_ID") or os.environ.get("SHEET_ID") or DEFAULT_OPERATIONAL_SHEET_ID,
        help="Live operational Google Sheet ID that receives rolling appended rows.",
    )
    parser.add_argument(
        "--historical-sheet-id",
        "--sheet-id",
        dest="historical_sheet_id",
        default=(
            os.environ.get("HISTORICAL_TRAINING_SHEET_ID")
            or os.environ.get("TRAINING_SHEET_ID")
            or DEFAULT_TRAINING_SHEET_ID
        ),
        help="Append-only historical training dataset Google Sheet ID.",
    )
    parser.add_argument("--google-credentials", default=None)
    parser.add_argument("--worksheet", default=None, help="Optional single worksheet/symbol to process.")
    parser.add_argument("--worksheets", default=None, help="Comma-separated worksheet/symbol names to process.")
    parser.add_argument("--model-dir", default=str(Path("outputs") / "Saved_Models"))
    parser.add_argument("--metadata", default=str(Path("outputs") / "pipeline_metadata.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--state-file", default=os.environ.get("MONTHLY_FINETUNE_STATE_FILE", str(DEFAULT_STATE_FILE)))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--rolling-operational-rows",
        type=int,
        default=parse_positive_int(
            os.environ.get("ROLLING_OPERATIONAL_ROWS"),
            DEFAULT_ROLLING_OPERATIONAL_ROWS,
            name="ROLLING_OPERATIONAL_ROWS",
        ),
        help="Number of latest valid dated rows to retain in each operational worksheet.",
    )
    parser.add_argument("--recent-days", type=int, default=DEFAULT_RECENT_DAYS)
    parser.add_argument("--replay-samples-per-symbol", type=int, default=DEFAULT_REPLAY_SAMPLES_PER_SYMBOL)
    parser.add_argument("--validation-targets-per-symbol", type=int, default=DEFAULT_VALIDATION_TARGETS_PER_SYMBOL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_FINE_TUNE_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--min-train-samples", type=int, default=MIN_FINE_TUNE_TRAIN_SAMPLES)
    parser.add_argument(
        "--force-finetune",
        action="store_true",
        help="Deprecated compatibility flag; fine-tuning still uses only checkpoint-new historical rows.",
    )
    parser.add_argument(
        "--skip-yfinance-update",
        action="store_true",
        help="Deprecated compatibility flag; monthly_finetune.py no longer fetches yfinance data.",
    )
    parser.add_argument("--skip-archival", action="store_true", help="Skip operational-to-historical archival.")
    parser.add_argument(
        "--fine-tune-batch-only-new-data",
        action=argparse.BooleanOptionalAction,
        default=env_flag("FINE_TUNE_BATCH_ONLY_NEW_DATA", True),
        help="Restrict training targets to historical rows newer than the fine-tune state checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and train in memory without changing sheets or checkpoints.")
    return parser.parse_args()


def print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False))


def parse_worksheet_filters(worksheets: Optional[str], worksheet: Optional[str]) -> Optional[Set[str]]:
    names = Data_update.parse_csv(worksheets)
    names.extend(Data_update.parse_csv(worksheet))
    if not names:
        return None
    return {name.strip().upper() for name in names if name.strip()}


def default_state() -> Dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "archival": {"worksheets": {}},
        "fine_tune": {"worksheets": {}},
    }


def load_finetune_state(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not path.exists():
        return default_state(), {"status": "missing", "message": "state file does not exist"}
    try:
        state = json.loads(path.read_text())
    except Exception as exc:
        return default_state(), {"status": "corrupted", "message": str(exc)}
    if not isinstance(state, dict) or int(state.get("schema_version", 0) or 0) != STATE_SCHEMA_VERSION:
        return default_state(), {
            "status": "corrupted",
            "message": f"unsupported state schema version: {state.get('schema_version') if isinstance(state, dict) else None}",
        }
    state.setdefault("archival", {}).setdefault("worksheets", {})
    state.setdefault("fine_tune", {}).setdefault("worksheets", {})
    return state, {"status": "ok", "message": ""}


def save_finetune_state(path: Path, state: Dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str, allow_nan=False))
    os.replace(tmp_path, path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def state_date(value: Any) -> Optional[pd.Timestamp]:
    parsed = parse_sheet_date(value)
    return parsed.normalize() if parsed is not None else None


def latest_frame_date(frame: pd.DataFrame) -> str:
    if frame.empty or "Date" not in frame.columns:
        return ""
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.dropna().empty:
        return ""
    return pd.Timestamp(dates.max()).normalize().strftime("%Y-%m-%d")


def latest_historical_dates(spreadsheet: Any, requested: Optional[Set[str]]) -> Dict[str, str]:
    latest: Dict[str, str] = {}
    worksheets, _ = Data_update.iter_target_worksheets(spreadsheet, requested)
    for worksheet in worksheets:
        symbol = worksheet.title.strip().upper()
        try:
            frame = worksheet_to_required_frame(worksheet)
        except Exception as exc:
            log(f"{symbol}: could not read baseline historical date: {exc}")
            continue
        latest[symbol] = latest_frame_date(frame)
    return latest


def finetune_cutoffs_from_state(
    state: Dict[str, Any],
    baseline_dates: Dict[str, str],
    symbols: Iterable[str],
    *,
    only_new_data: bool,
) -> Dict[str, Optional[pd.Timestamp]]:
    if not only_new_data:
        return {symbol: None for symbol in symbols}
    worksheets_state = state.get("fine_tune", {}).get("worksheets", {})
    cutoffs: Dict[str, Optional[pd.Timestamp]] = {}
    for symbol in symbols:
        entry = worksheets_state.get(symbol, {}) if isinstance(worksheets_state, dict) else {}
        stored = entry.get("last_finetuned_date") if isinstance(entry, dict) else None
        fallback = baseline_dates.get(symbol)
        cutoffs[symbol] = state_date(stored or fallback)
    return cutoffs


def update_archival_state(
    state: Dict[str, Any],
    results: Sequence[ArchivalWorksheetResult],
    *,
    operational_sheet_id: str,
    historical_sheet_id: str,
) -> None:
    archival_state = state.setdefault("archival", {})
    archival_state["operational_sheet_id"] = operational_sheet_id
    archival_state["historical_sheet_id"] = historical_sheet_id
    archival_state["last_run_utc"] = utc_now_iso()
    worksheets = archival_state.setdefault("worksheets", {})
    for result in results:
        if result.status not in {"ok", "dry_run"}:
            continue
        existing = worksheets.setdefault(result.worksheet.upper(), {})
        existing.update(
            {
                "last_archival_run_utc": archival_state["last_run_utc"],
                "last_archived_date": result.latest_archived_date or existing.get("last_archived_date", ""),
                "rows_appended": int(existing.get("rows_appended", 0) or 0) + int(result.rows_appended),
                "duplicate_rows_skipped": int(existing.get("duplicate_rows_skipped", 0) or 0)
                + int(result.duplicate_rows_skipped),
            }
        )


def update_finetune_state(
    state: Dict[str, Any],
    arrays: FineTuneArrays,
    *,
    historical_sheet_id: str,
    model_dir: Path,
    metadata_path: Path,
) -> None:
    fine_tune_state = state.setdefault("fine_tune", {})
    fine_tune_state["historical_sheet_id"] = historical_sheet_id
    fine_tune_state["model_dir"] = str(model_dir)
    fine_tune_state["metadata"] = str(metadata_path)
    fine_tune_state["last_successful_run_utc"] = utc_now_iso()
    worksheets = fine_tune_state.setdefault("worksheets", {})
    for summary in arrays.summaries:
        if not summary.latest_processed_date or summary.train_samples <= 0:
            continue
        worksheets[summary.symbol.upper()] = {
            "last_finetuned_date": summary.latest_processed_date,
            "last_successful_run_utc": fine_tune_state["last_successful_run_utc"],
            "train_samples": int(summary.train_samples),
            "validation_samples": int(summary.validation_samples),
            "new_rows": int(summary.new_rows),
        }


def column_number_to_letter(number: int) -> str:
    if number <= 0:
        raise ValueError("column number must be positive")
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def parse_sheet_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(None) if hasattr(parsed, "tz_convert") else parsed.tz_localize(None)
    return pd.Timestamp(parsed).normalize()


def find_schema_index(headers: Sequence[str], canonical_name: str, stock: str) -> Optional[int]:
    return Data_update.find_header_index(headers, canonical_name, stock)


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def read_worksheet_values(worksheet: Any) -> List[List[Any]]:
    return Data_update.with_retry(worksheet.get_all_values, f"{worksheet.title}: get_all_values")


def validate_and_repair_worksheet(worksheet: Any, *, dry_run: bool = False) -> SheetValidationResult:
    title = str(getattr(worksheet, "title", "")).strip()
    stock = title.upper()
    values = read_worksheet_values(worksheet)
    if not values:
        return SheetValidationResult(worksheet=title, status="skipped", reason="empty worksheet")

    width = max(len(row) for row in values)
    padded = [list(row) + [""] * (width - len(row)) for row in values]
    headers = [str(value).strip() for value in padded[0]]
    data_rows = padded[1:]
    schema, schema_error = Data_update.resolve_sheet_schema(headers, stock)
    if schema is None:
        return SheetValidationResult(
            worksheet=title,
            status="skipped",
            rows_before=len(data_rows),
            rows_after=len(data_rows),
            reason=schema_error,
        )

    date_index = schema.get("Date_str")
    if date_index is None:
        date_index = schema.get("Date")
    if date_index is None:
        return SheetValidationResult(
            worksheet=title,
            status="skipped",
            rows_before=len(data_rows),
            rows_after=len(data_rows),
            reason="date column missing",
        )

    valid: List[Tuple[pd.Timestamp, int, List[Any], int]] = []
    malformed_removed = 0
    repaired_cells = 0
    for original_idx, row in enumerate(data_rows):
        if not any(str(value).strip() for value in row):
            malformed_removed += 1
            continue

        parsed_date = parse_sheet_date(row[date_index] if date_index < len(row) else "")
        if parsed_date is None and schema.get("Date") is not None:
            parsed_date = parse_sheet_date(row[int(schema["Date"])])
        if parsed_date is None:
            malformed_removed += 1
            continue

        numeric_values: Dict[str, float] = {}
        row_is_valid = True
        for column in REQUIRED_PRICE_COLUMNS:
            idx = schema.get(column)
            value = finite_number(row[idx]) if idx is not None and idx < len(row) else None
            if value is None:
                row_is_valid = False
                break
            numeric_values[column] = value
        if not row_is_valid:
            malformed_removed += 1
            continue
        if numeric_values["High"] < numeric_values["Low"] or numeric_values["Close"] <= 0:
            malformed_removed += 1
            continue

        normalized = list(row)
        canonical_date = parsed_date.strftime("%Y-%m-%d")
        for column, value in (("Date", canonical_date), ("Date_str", canonical_date)):
            idx = schema.get(column)
            if idx is not None and idx < len(normalized) and str(normalized[idx]).strip() != str(value):
                normalized[idx] = value
                repaired_cells += 1
        for column, number in numeric_values.items():
            idx = schema.get(column)
            if idx is not None and idx < len(normalized):
                cleaned = int(number) if column == "Volume" else float(number)
                if str(normalized[idx]).replace(",", "").strip() != str(cleaned):
                    normalized[idx] = cleaned
                    repaired_cells += 1
        adj_idx = schema.get("Adj Close")
        if adj_idx is not None and adj_idx < len(normalized):
            adj_value = finite_number(normalized[adj_idx])
            if adj_value is None:
                normalized[adj_idx] = float(numeric_values["Close"])
                repaired_cells += 1
        valid.append((parsed_date, original_idx, normalized, original_idx))

    if not valid:
        return SheetValidationResult(
            worksheet=title,
            status="error",
            rows_before=len(data_rows),
            rows_after=0,
            malformed_removed=malformed_removed,
            reason="no valid dated OHLCV rows remain",
        )

    by_date: Dict[str, Tuple[pd.Timestamp, int, List[Any], int]] = {}
    for item in valid:
        key = item[0].strftime("%Y-%m-%d")
        previous = by_date.get(key)
        if previous is None or item[1] > previous[1]:
            by_date[key] = item
    unique = list(by_date.values())
    duplicates_removed = len(valid) - len(unique)
    unique.sort(key=lambda item: (item[0], item[1]))
    repaired_rows = [row for _, _, row, _ in unique]
    sorted_original_indices = [original_idx for _, _, _, original_idx in unique]
    sorted_rows = sorted_original_indices != sorted(sorted_original_indices)
    rows_after = len(repaired_rows)
    changed = (
        malformed_removed > 0
        or duplicates_removed > 0
        or sorted_rows
        or repaired_cells > 0
        or rows_after != len(data_rows)
    )

    if changed and not dry_run:
        end_col = column_number_to_letter(width)
        worksheet.batch_update(
            [{"range": f"A2:{end_col}{rows_after + 1}", "values": repaired_rows}],
            raw=True,
        )
        if len(data_rows) > rows_after:
            worksheet.delete_rows(rows_after + 2, len(data_rows) + 1)

    return SheetValidationResult(
        worksheet=title,
        status="ok" if changed else "clean",
        rows_before=len(data_rows),
        rows_after=rows_after,
        duplicates_removed=duplicates_removed,
        malformed_removed=malformed_removed,
        sorted_rows=sorted_rows,
        repaired_cells=repaired_cells,
    )


def worksheet_to_required_frame(worksheet: Any) -> pd.DataFrame:
    frame, headers, is_empty = Data_update.worksheet_to_frame(worksheet)
    if is_empty or frame.empty:
        return pd.DataFrame(columns=Data_update.REQUIRED_COLUMNS)
    schema, schema_error = Data_update.resolve_sheet_schema(headers, worksheet.title.strip().upper())
    if schema is None:
        raise ValueError(schema_error)
    normalized = Data_update.normalize_frame_to_required_columns(frame, headers, schema)
    normalized["Date"] = Data_update.normalize_date_series(normalized["Date"])
    normalized["Date_str"] = normalized["Date"].dt.strftime("%Y-%m-%d")
    for column in Data_update.NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(
            normalized[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    normalized = normalized.dropna(subset=["Date", *Data_update.NUMERIC_COLUMNS])
    normalized = normalized.drop_duplicates(subset=["Date_str"], keep="last")
    return normalized.sort_values("Date").reset_index(drop=True)


def open_spreadsheet(sheet_id: str, credentials_path: Optional[str]) -> Any:
    client = Data_update.authorize_gspread(credentials_path)
    return Data_update.with_retry(lambda: client.open_by_key(sheet_id), "open spreadsheet")


def open_operational_spreadsheet(args: argparse.Namespace) -> Any:
    return open_spreadsheet(args.operational_sheet_id, args.google_credentials)


def open_historical_spreadsheet(args: argparse.Namespace) -> Any:
    return open_spreadsheet(args.historical_sheet_id, args.google_credentials)


def target_worksheets(spreadsheet: Any, args: argparse.Namespace) -> Tuple[List[Any], List[str]]:
    requested = parse_worksheet_filters(args.worksheets, args.worksheet)
    return Data_update.iter_target_worksheets(spreadsheet, requested)


def run_yfinance_update(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "status": "disabled",
        "reason": "monthly_finetune.py is externally triggered and no longer fetches or watches yfinance data",
        "rows_added": 0,
        "stocks_updated": 0,
        "failures": 0,
    }


def run_operational_archival(
    operational_spreadsheet: Any,
    historical_spreadsheet: Any,
    args: argparse.Namespace,
) -> Tuple[List[ArchivalWorksheetResult], List[str]]:
    requested = parse_worksheet_filters(args.worksheets, args.worksheet)
    worksheets, missing = Data_update.iter_target_worksheets(operational_spreadsheet, requested)
    results: List[ArchivalWorksheetResult] = []
    if args.skip_archival:
        for worksheet in worksheets:
            results.append(
                ArchivalWorksheetResult(
                    worksheet=worksheet.title,
                    status="skipped",
                    reason="skip_archival flag set",
                    rolling_limit=int(args.rolling_operational_rows),
                    dry_run=bool(args.dry_run),
                )
            )
        return results, missing

    for worksheet in worksheets:
        try:
            archive_worksheet = find_worksheet_by_title(historical_spreadsheet, worksheet.title)
            results.append(
                archive_old_rows_for_worksheet(
                    worksheet,
                    archive_spreadsheet=historical_spreadsheet,
                    archive_worksheet=archive_worksheet,
                    keep_rows=int(args.rolling_operational_rows),
                    dry_run=bool(args.dry_run),
                )
            )
        except Exception as exc:
            results.append(
                ArchivalWorksheetResult(
                    worksheet=getattr(worksheet, "title", ""),
                    status="error",
                    reason=str(exc),
                    rolling_limit=int(args.rolling_operational_rows),
                    dry_run=bool(args.dry_run),
                )
            )
    return results, missing


def normalize_model_features(frame: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    feature_columns, _, _, _ = forecast.validate_metadata(metadata)
    frame = forecast.normalize_columns(frame)
    leakage_columns = forecast.forward_label_like_columns(frame.columns, metadata)
    frame = frame.drop(columns=leakage_columns, errors="ignore")
    feature_frame = frame.reindex(columns=feature_columns, fill_value=0)
    return forecast.to_numeric_frame(feature_frame)


def forward_log_return_labels(close: np.ndarray, target_position: int, horizons: Sequence[int]) -> Optional[np.ndarray]:
    anchor_idx = target_position - 1
    if anchor_idx < 0 or not np.isfinite(close[anchor_idx]) or close[anchor_idx] <= 0:
        return None
    labels: List[float] = []
    for horizon in horizons:
        future_idx = target_position + int(horizon) - 1
        if future_idx >= len(close):
            return None
        future_close = close[future_idx]
        if not np.isfinite(future_close) or future_close <= 0:
            return None
        labels.append(float(np.log(future_close / close[anchor_idx])))
    return np.asarray(labels, dtype=np.float32)


def deterministic_replay_positions(candidates: Sequence[int], max_count: int) -> List[int]:
    if max_count <= 0 or not candidates:
        return []
    if len(candidates) <= max_count:
        return list(candidates)
    indices = np.linspace(0, len(candidates) - 1, num=max_count, dtype=int)
    return [int(candidates[idx]) for idx in indices]


def build_symbol_examples(
    symbol: str,
    frame: pd.DataFrame,
    metadata: Dict[str, Any],
    args: argparse.Namespace,
    last_finetuned_date: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, SymbolDatasetSummary]:
    feature_columns, seq_len, feature_count, _ = forecast.validate_metadata(metadata)
    horizons = forecast.metadata_horizons(metadata)
    cutoff_text = last_finetuned_date.strftime("%Y-%m-%d") if last_finetuned_date is not None else ""
    if frame.empty or len(frame) < seq_len + max(horizons):
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=0,
            train_samples=0,
            validation_samples=0,
            replay_samples=0,
            last_finetuned_date=cutoff_text,
            skipped_reason="insufficient rows",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)

    source = frame.copy()
    source[forecast.SORT_POSITION_COL] = np.arange(len(source), dtype=np.int64)
    with contextlib.redirect_stdout(sys.stderr):
        engineered = compute_indicators(source)
    engineered = forecast.normalize_columns(engineered)
    if forecast.SORT_POSITION_COL not in engineered.columns or forecast.TARGET_COL not in engineered.columns:
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=0,
            train_samples=0,
            validation_samples=0,
            replay_samples=0,
            last_finetuned_date=cutoff_text,
            skipped_reason="feature engineering lost required columns",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)

    engineered[forecast.TARGET_COL] = forecast.to_numeric_series(engineered[forecast.TARGET_COL])
    engineered = engineered.dropna(subset=[forecast.TARGET_COL, forecast.SORT_POSITION_COL]).copy()
    if len(engineered) < seq_len + max(horizons):
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=0,
            train_samples=0,
            validation_samples=0,
            replay_samples=0,
            last_finetuned_date=cutoff_text,
            skipped_reason="insufficient engineered rows",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)

    engineered = engineered.sort_values(forecast.SORT_POSITION_COL, kind="stable").reset_index(drop=True)
    positions = forecast.to_numeric_series(engineered[forecast.SORT_POSITION_COL]).astype(np.int64).to_numpy()
    close = forecast.to_numeric_series(engineered[forecast.TARGET_COL]).to_numpy(dtype=np.float64)
    dates = pd.to_datetime(engineered["Date"], errors="coerce", format="mixed") if "Date" in engineered.columns else pd.Series(pd.NaT, index=engineered.index)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    latest_date = dates.dropna().max()
    if pd.isna(latest_date):
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=0,
            train_samples=0,
            validation_samples=0,
            replay_samples=0,
            last_finetuned_date=cutoff_text,
            skipped_reason="no parseable dates",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)

    max_horizon = max(horizons)
    eligible_positions: List[int] = []
    for position in positions:
        target_position = int(position)
        if target_position < seq_len:
            continue
        if target_position + max_horizon - 1 >= len(close):
            continue
        if forward_log_return_labels(close, target_position, horizons) is not None:
            eligible_positions.append(target_position)
    if not eligible_positions:
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=0,
            train_samples=0,
            validation_samples=0,
            replay_samples=0,
            last_finetuned_date=cutoff_text,
            skipped_reason="no labeled target windows",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)

    position_to_date = {
        int(position): pd.Timestamp(date_value).normalize()
        for position, date_value in zip(positions, dates)
        if pd.notna(date_value)
    }
    new_positions = [
        position for position in eligible_positions
        if last_finetuned_date is None or position_to_date.get(position, pd.Timestamp.min) > last_finetuned_date
    ]
    if not new_positions:
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=0,
            train_samples=0,
            validation_samples=0,
            replay_samples=0,
            last_finetuned_date=cutoff_text,
            skipped_reason="no new historical rows beyond fine-tune checkpoint",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)

    validation_count = min(max(1, int(args.validation_targets_per_symbol)), max(1, len(new_positions) // 3))
    val_positions = new_positions[-validation_count:]
    train_recent_positions = new_positions[:-validation_count]
    if not train_recent_positions:
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=len(new_positions),
            train_samples=0,
            validation_samples=len(val_positions),
            replay_samples=0,
            last_finetuned_date=cutoff_text,
            skipped_reason="no new train positions before validation",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)
    val_start_position = min(val_positions)
    older_candidates = [position for position in eligible_positions if position < min(train_recent_positions)]
    replay_positions = deterministic_replay_positions(older_candidates, int(args.replay_samples_per_symbol))
    train_positions = sorted(set(train_recent_positions + replay_positions))

    if not train_positions:
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=len(new_positions),
            train_samples=0,
            validation_samples=len(val_positions),
            replay_samples=len(replay_positions),
            last_finetuned_date=cutoff_text,
            skipped_reason="no train positions before validation",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)

    fit_mask = positions < val_start_position
    if fit_mask.sum() < seq_len:
        summary = SymbolDatasetSummary(
            symbol=symbol,
            rows=len(frame),
            new_rows=len(new_positions),
            train_samples=0,
            validation_samples=len(val_positions),
            replay_samples=len(replay_positions),
            last_finetuned_date=cutoff_text,
            skipped_reason="insufficient pre-validation scaler rows",
        )
        return (*empty_symbol_arrays(seq_len, feature_count, len(horizons)), summary)

    feature_frame = normalize_model_features(engineered, metadata)
    scaler = MinMaxScaler()
    scaler.fit(feature_frame.loc[fit_mask])
    scaled_features = scaler.transform(feature_frame).astype(np.float32)
    position_to_engineered_index = {int(position): idx for idx, position in enumerate(positions)}

    def collect(target_positions: Sequence[int]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        anchor_list: List[float] = []
        for target_position in target_positions:
            labels = forward_log_return_labels(close, int(target_position), horizons)
            target_idx = position_to_engineered_index.get(int(target_position))
            if labels is None or target_idx is None:
                continue
            prior_indices = np.flatnonzero(positions < int(target_position))
            if len(prior_indices) < seq_len:
                continue
            window_indices = prior_indices[-seq_len:]
            if not np.all(positions[window_indices] < int(target_position)):
                raise RuntimeError(f"{symbol}: target row leaked into sequence")
            sequence = scaled_features[window_indices]
            if sequence.shape != (seq_len, feature_count) or not np.isfinite(sequence).all():
                continue
            anchor_idx = target_position - 1
            if anchor_idx >= len(close) or not np.isfinite(close[anchor_idx]) or close[anchor_idx] <= 0:
                continue
            X_list.append(sequence.astype(np.float32))
            y_list.append(labels.astype(np.float32))
            anchor_list.append(float(close[anchor_idx]))
        return X_list, y_list, anchor_list

    X_train_list, y_train_list, _ = collect(train_positions)
    X_val_list, y_val_list, anchor_val_list = collect(val_positions)
    summary = SymbolDatasetSummary(
        symbol=symbol,
        rows=len(frame),
        new_rows=len(new_positions),
        train_samples=len(X_train_list),
        validation_samples=len(X_val_list),
        replay_samples=len(replay_positions),
        last_finetuned_date=cutoff_text,
        latest_processed_date=max(position_to_date[position] for position in new_positions).strftime("%Y-%m-%d"),
    )
    return (
        np.asarray(X_train_list, dtype=np.float32).reshape(-1, seq_len, feature_count),
        np.asarray(y_train_list, dtype=np.float32).reshape(-1, len(horizons)),
        np.asarray(X_val_list, dtype=np.float32).reshape(-1, seq_len, feature_count),
        np.asarray(y_val_list, dtype=np.float32).reshape(-1, len(horizons)),
        np.asarray(anchor_val_list, dtype=np.float32).reshape(-1),
        summary,
    )


def empty_symbol_arrays(seq_len: int, feature_count: int, horizon_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, seq_len, feature_count), dtype=np.float32),
        np.empty((0, horizon_count), dtype=np.float32),
        np.empty((0, seq_len, feature_count), dtype=np.float32),
        np.empty((0, horizon_count), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def build_finetune_arrays(
    frames: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    args: argparse.Namespace,
    last_finetuned_dates: Optional[Dict[str, Optional[pd.Timestamp]]] = None,
) -> FineTuneArrays:
    feature_columns, seq_len, feature_count, _ = forecast.validate_metadata(metadata)
    horizon_count = len(forecast.metadata_horizons(metadata))
    train_X: List[np.ndarray] = []
    train_y: List[np.ndarray] = []
    val_X: List[np.ndarray] = []
    val_y: List[np.ndarray] = []
    val_anchor: List[np.ndarray] = []
    val_symbols: List[np.ndarray] = []
    summaries: List[SymbolDatasetSummary] = []
    last_finetuned_dates = last_finetuned_dates or {}
    for symbol, frame in frames.items():
        X_tr, y_tr, X_va, y_va, anchors, summary = build_symbol_examples(
            symbol,
            frame,
            metadata,
            args,
            last_finetuned_dates.get(symbol),
        )
        summaries.append(summary)
        if len(X_tr):
            train_X.append(X_tr)
            train_y.append(y_tr)
        if len(X_va):
            val_X.append(X_va)
            val_y.append(y_va)
            val_anchor.append(anchors)
            val_symbols.append(np.asarray([symbol] * len(X_va), dtype=object))

    if train_X:
        X_train = np.concatenate(train_X, axis=0)
        y_train = np.concatenate(train_y, axis=0)
    else:
        X_train = np.empty((0, seq_len, feature_count), dtype=np.float32)
        y_train = np.empty((0, horizon_count), dtype=np.float32)
    if val_X:
        X_val = np.concatenate(val_X, axis=0)
        y_val = np.concatenate(val_y, axis=0)
        anchor_val = np.concatenate(val_anchor, axis=0)
        symbols_val = np.concatenate(val_symbols, axis=0)
    else:
        X_val = np.empty((0, seq_len, feature_count), dtype=np.float32)
        y_val = np.empty((0, horizon_count), dtype=np.float32)
        anchor_val = np.empty((0,), dtype=np.float32)
        symbols_val = np.empty((0,), dtype=object)

    for name, array in (("X_train", X_train), ("y_train", y_train), ("X_val", X_val), ("y_val", y_val)):
        if len(array) and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    return FineTuneArrays(X_train, y_train, X_val, y_val, anchor_val, symbols_val, summaries)


def tensor_loader(X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def mean_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            output = model(X_batch.to(device))
            target = y_batch.to(device)
            if output.shape != target.shape:
                raise ValueError(f"model output shape {tuple(output.shape)} != target shape {tuple(target.shape)}")
            loss = criterion(output, target)
            if not torch.isfinite(loss):
                return float("inf")
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def fine_tune_one_model(
    name: str,
    model: nn.Module,
    arrays: FineTuneArrays,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    train_loader = tensor_loader(arrays.X_train, arrays.y_train, int(args.batch_size), shuffle=True)
    val_loader = tensor_loader(arrays.X_val, arrays.y_val, int(args.batch_size), shuffle=False)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    model.train()

    for epoch in range(1, int(args.epochs) + 1):
        train_losses: List[float] = []
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(X_batch.to(device))
            target = y_batch.to(device)
            if output.shape != target.shape:
                raise ValueError(f"{name}: output shape {tuple(output.shape)} != target shape {tuple(target.shape)}")
            if not torch.isfinite(output).all() or not torch.isfinite(target).all():
                raise FloatingPointError(f"{name}: non-finite tensor encountered")
            loss = criterion(output, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: non-finite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"{name}: non-finite gradient norm")
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = mean_loss(model, val_loader, device, criterion)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": val_loss})
        log(f"{name}: epoch={epoch} train_loss={train_loss:.6f} validation_loss={val_loss:.6f}")
        if not math.isfinite(val_loss):
            bad_epochs += 1
        elif val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(args.patience):
            break

    if best_state is None or not math.isfinite(best_loss):
        raise RuntimeError(f"{name}: fine-tuning did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return {"best_validation_loss": best_loss, "epochs_ran": len(history), "history": history}


def predict_array(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    predictions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (X_batch,) in loader:
            output = model(X_batch.to(device))
            if not torch.isfinite(output).all():
                raise FloatingPointError("non-finite model output during evaluation")
            predictions.append(output.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(predictions, axis=0) if predictions else np.empty_like(arrays_shape_like_y(X))


def arrays_shape_like_y(X: np.ndarray) -> np.ndarray:
    return np.empty((len(X), forecast.MAX_FORECAST_HORIZON), dtype=np.float32)


def horizon_metrics(pred_returns: np.ndarray, true_returns: np.ndarray, anchors: np.ndarray, horizons: Sequence[int]) -> Dict[str, Dict[str, float]]:
    if pred_returns.shape != true_returns.shape:
        raise ValueError(f"prediction shape {pred_returns.shape} != true shape {true_returns.shape}")
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1)
    pred_prices = anchors[:, None] * np.exp(pred_returns.astype(np.float64))
    true_prices = anchors[:, None] * np.exp(true_returns.astype(np.float64))
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, horizon in enumerate(horizons):
        errors = pred_prices[:, idx] - true_prices[:, idx]
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        direction = float(np.mean(np.sign(pred_returns[:, idx]) == np.sign(true_returns[:, idx])))
        metrics[f"T+{horizon}"] = {"RMSE": rmse, "MAE": mae, "Directional_Accuracy": direction}
    all_errors = pred_prices - true_prices
    metrics["overall"] = {
        "RMSE": float(np.sqrt(np.mean(all_errors ** 2))),
        "MAE": float(np.mean(np.abs(all_errors))),
        "Directional_Accuracy": float(np.mean(np.sign(pred_returns) == np.sign(true_returns))),
    }
    return metrics


def evaluate_models(
    models: Dict[str, nn.Module],
    arrays: FineTuneArrays,
    metadata: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    horizons = forecast.metadata_horizons(metadata)
    per_model: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        pred = predict_array(model, arrays.X_val, device, int(args.batch_size))
        predictions[name] = pred
        per_model[name] = horizon_metrics(pred, arrays.y_val, arrays.anchor_val, horizons)

    ensemble_returns, weights_used = forecast.ensemble_predictions(predictions, metadata.get("ensemble_weights", {}))
    return {
        "per_model": per_model,
        "ensemble": horizon_metrics(ensemble_returns, arrays.y_val, arrays.anchor_val, horizons),
        "weights_used": weights_used,
        "validation_samples": int(len(arrays.X_val)),
    }


def validate_checkpoint_state(name: str, state: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> None:
    factories = {
        "Dense": forecast.infer_dense_model,
        "LSTM": forecast.infer_lstm_model,
        "Transformer": forecast.infer_transformer_model,
    }
    if name not in factories:
        raise ValueError(f"Unsupported checkpoint name: {name}")
    model = factories[name](state, metadata)
    model.load_state_dict(state)


def save_checkpoints_atomically(
    models: Dict[str, nn.Module],
    model_dir: Path,
    metadata: Dict[str, Any],
    *,
    dry_run: bool,
) -> Dict[str, str]:
    status: Dict[str, str] = {}
    if dry_run:
        return {name: "dry_run_not_written" for name in models}
    model_dir.mkdir(parents=True, exist_ok=True)
    temp_paths: List[Path] = []
    try:
        for name, model in models.items():
            path = model_dir / f"{name}.pt"
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            validate_checkpoint_state(name, state, metadata)
            torch.save(state, tmp_path)
            temp_paths.append(tmp_path)
        for name in models:
            path = model_dir / f"{name}.pt"
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            os.replace(tmp_path, path)
            status[name] = "overwritten"
    finally:
        for tmp_path in temp_paths:
            if tmp_path.exists():
                tmp_path.unlink()
    return status


def run_monthly_finetune(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata)
    model_dir = Path(args.model_dir)
    state_path = Path(args.state_file)
    state, state_status = load_finetune_state(state_path)
    metadata = get_or_create_metadata(metadata_path)
    forecast.validate_metadata(metadata)
    device = forecast.choose_device(args.device)
    log(f"Device: {device}")

    yfinance_update = run_yfinance_update(args)
    requested = parse_worksheet_filters(args.worksheets, args.worksheet)
    operational_spreadsheet = open_operational_spreadsheet(args)
    historical_spreadsheet = open_historical_spreadsheet(args)
    baseline_dates = latest_historical_dates(historical_spreadsheet, requested)
    archival_results, missing_operational = run_operational_archival(
        operational_spreadsheet,
        historical_spreadsheet,
        args,
    )
    update_archival_state(
        state,
        archival_results,
        operational_sheet_id=args.operational_sheet_id,
        historical_sheet_id=args.historical_sheet_id,
    )
    save_finetune_state(state_path, state, dry_run=bool(args.dry_run))

    worksheets, missing_historical = target_worksheets(historical_spreadsheet, args)

    validation_results: List[SheetValidationResult] = []
    frames: Dict[str, pd.DataFrame] = {}
    for worksheet in worksheets:
        try:
            result = validate_and_repair_worksheet(worksheet, dry_run=True)
            validation_results.append(result)
            if result.status in {"ok", "clean"}:
                frames[worksheet.title.strip().upper()] = worksheet_to_required_frame(worksheet)
        except Exception as exc:
            validation_results.append(
                SheetValidationResult(
                    worksheet=getattr(worksheet, "title", ""),
                    status="error",
                    reason=str(exc),
                )
            )
    for name in missing_historical:
        validation_results.append(SheetValidationResult(worksheet=name, status="missing", reason="worksheet not found"))

    cutoffs = finetune_cutoffs_from_state(
        state,
        baseline_dates,
        frames.keys(),
        only_new_data=bool(args.fine_tune_batch_only_new_data),
    )
    arrays = build_finetune_arrays(frames, metadata, args, cutoffs)
    new_rows_available = sum(summary.new_rows for summary in arrays.summaries)
    if len(arrays.X_train) < int(args.min_train_samples):
        status = "no_new_historical_data" if new_rows_available <= 0 else "skipped_insufficient_training_data"
        payload = {
            "status": status,
            "message": (
                "No historical rows are newer than the fine-tune checkpoint."
                if status == "no_new_historical_data"
                else f"Only {len(arrays.X_train)} training samples available; required {args.min_train_samples}."
            ),
            "triggering": {
                "mode": "external_cli_or_scheduler_only",
                "self_watching": False,
                "google_sheet_polling": False,
            },
            "state": {
                "path": str(state_path),
                "load_status": state_status,
                "fine_tune_cutoffs": {symbol: value.strftime("%Y-%m-%d") if value is not None else "" for symbol, value in cutoffs.items()},
            },
            "operational_to_historical_archival": {
                "operational_sheet_id": args.operational_sheet_id,
                "historical_sheet_id": args.historical_sheet_id,
                "rolling_operational_rows": int(args.rolling_operational_rows),
                "results": [result.__dict__ for result in archival_results],
                "missing_operational_worksheets": missing_operational,
            },
            "yfinance_update": yfinance_update,
            "validation": [result.__dict__ for result in validation_results],
            "dataset": {
                "train_samples": int(len(arrays.X_train)),
                "validation_samples": int(len(arrays.X_val)),
                "new_rows_available": int(new_rows_available),
                "symbols": [summary.__dict__ for summary in arrays.summaries],
            },
            "checkpoint_overwrite_status": {name: "skipped_insufficient_data" for name in ("Dense", "LSTM", "Transformer")},
            "incremental_finetuning": {
                "only_new_historical_rows": bool(args.fine_tune_batch_only_new_data),
                "replay_buffer_enabled": int(args.replay_samples_per_symbol) > 0,
                "full_dataset_retrain": False,
            },
        }
        (output_dir / "monthly_finetune_metrics.json").write_text(json.dumps(payload, indent=2, default=str, allow_nan=False))
        return payload
    if len(arrays.X_val) == 0:
        raise RuntimeError("No validation samples available; refusing to overwrite checkpoints")

    models = forecast.load_models(model_dir, metadata, device)
    original_states = {
        name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    training: Dict[str, Any] = {}
    try:
        for name, model in models.items():
            training[name] = fine_tune_one_model(name, model, arrays, args, device)
        evaluation = evaluate_models(models, arrays, metadata, args, device)
        overwrite_status = save_checkpoints_atomically(models, model_dir, metadata, dry_run=bool(args.dry_run))
        update_finetune_state(
            state,
            arrays,
            historical_sheet_id=args.historical_sheet_id,
            model_dir=model_dir,
            metadata_path=metadata_path,
        )
        save_finetune_state(state_path, state, dry_run=bool(args.dry_run))
        status = "dry_run_ok" if args.dry_run else "ok"
    except Exception:
        for name, state in original_states.items():
            models[name].load_state_dict(state)
        raise

    payload = {
        "status": status,
        "triggering": {
            "mode": "external_cli_or_scheduler_only",
            "self_watching": False,
            "google_sheet_polling": False,
            "valid_triggers": ["GitHub Actions workflow_dispatch", "GitHub Actions schedule", "manual CLI", "cron"],
        },
        "state": {
            "path": str(state_path),
            "load_status": state_status,
            "fine_tune_cutoffs": {symbol: value.strftime("%Y-%m-%d") if value is not None else "" for symbol, value in cutoffs.items()},
            "updated_after_successful_checkpoint_write": not bool(args.dry_run),
        },
        "operational_to_historical_archival": {
            "operational_sheet_id": args.operational_sheet_id,
            "historical_sheet_id": args.historical_sheet_id,
            "rolling_operational_rows": int(args.rolling_operational_rows),
            "results": [result.__dict__ for result in archival_results],
            "missing_operational_worksheets": missing_operational,
            "cleanup_requires_successful_archive": True,
            "duplicate_dates_skipped": sum(result.duplicate_rows_skipped for result in archival_results),
        },
        "yfinance_update": yfinance_update,
        "validation": [result.__dict__ for result in validation_results],
        "dataset": {
            "train_samples": int(len(arrays.X_train)),
            "validation_samples": int(len(arrays.X_val)),
            "new_rows_available": int(new_rows_available),
            "symbols": [summary.__dict__ for summary in arrays.summaries],
        },
        "training": training,
        "evaluation": evaluation,
        "checkpoint_overwrite_status": overwrite_status,
        "incremental_finetuning": {
            "only_new_historical_rows": bool(args.fine_tune_batch_only_new_data),
            "replay_buffer_enabled": int(args.replay_samples_per_symbol) > 0,
            "full_dataset_retrain": False,
            "state_updated_after_success": not bool(args.dry_run),
        },
        "leakage_safety": {
            "future_labels_excluded_from_features": True,
            "sequence_windows_exclude_target_row": True,
            "scalers_fit_before_validation_start": True,
            "validation_is_future_separated": True,
            "target_semantics": "label row t uses Close[t-1] anchor and predicts Close[t]..Close[t+4]",
        },
        "automation_ready": {
            "cron": True,
            "apscheduler": False,
            "github_actions": True,
            "vps_scheduler": True,
            "manual_cli": True,
            "self_triggering_watchers": False,
            "idempotent_append_only_archival": True,
        },
        "recursive_forecasting_compatibility": {
            "forecast_days_env": os.environ.get("FORECAST_DAYS", ""),
            "handled_by": "main.py",
            "monthly_finetune_does_not_override_forecast_horizon": True,
        },
    }
    metrics_path = output_dir / "monthly_finetune_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False))
    return payload


def main() -> None:
    args = parse_args()
    try:
        result = run_monthly_finetune(args)
    except Exception as exc:
        log(f"fatal: {exc}")
        result = {"status": "error", "error": str(exc)}
    print_json(result)
    raise SystemExit(1 if result.get("status") == "error" else 0)


if __name__ == "__main__":
    main()
