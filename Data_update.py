from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from google.oauth2.service_account import Credentials as _GoogleCredentials


DEFAULT_SHEET_ID = "1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o"
DEFAULT_START_DATE = "2015-01-01"
REQUIRED_COLUMNS = ["Date", "Date_str", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
APPEND_DEFAULT_COLUMNS = {
    "predicted": 0,
    "Predicted_Close_Price": "",
}
MARKET_TIMEZONE = "Asia/Kolkata"
MARKET_DATA_CUTOFF = time(16, 0)
APPEND_BATCH_SIZE = 500
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def log(message: str) -> None:
    print(message, file=sys.stderr)


def parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    seen = set()
    result: List[str] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        key = text.upper()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally append daily NSE stock data to Google Sheets.")
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID)
    parser.add_argument("--google-credentials", default=None)
    parser.add_argument("--worksheet", default=None, help="Optional single worksheet name to process.")
    parser.add_argument("--worksheets", default=None, help="Comma-separated worksheet names to process.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--interval", default="1d")
    return parser.parse_args()


def print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), allow_nan=False))


def with_retry(operation, description: str, attempts: int = 3, backoff: float = 1.0):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            log(f"{description} failed on attempt {attempt}/{attempts}: {exc}")
            time_module.sleep(backoff * attempt)
    assert last_error is not None
    raise last_error


def authorize_gspread(credentials_path: Optional[str]) -> Any:
    try:
        import gspread
    except ImportError as exc:
        raise RuntimeError("Google Sheets support requires gspread") from exc

    credential_file = (
        credentials_path
        or os.environ.get("GOOGLE_CREDENTIALS")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if not credential_file:
        raise RuntimeError("Google service-account credentials path is required")

    print(f"[google_auth] Using credentials : {credential_file}", flush=True)
    credentials = _GoogleCredentials.from_service_account_file(
        credential_file, scopes=GOOGLE_SCOPES
    )
    print("[google_auth] Service Account Loaded Successfully", flush=True)
    return gspread.authorize(credentials)


def worksheet_to_frame(worksheet: Any) -> Tuple[pd.DataFrame, List[str], bool]:
    values = with_retry(worksheet.get_all_values, f"{worksheet.title}: get_all_values")
    if not values:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), [], True

    headers = [str(value).strip() for value in values[0]]
    rows = values[1:]
    if not headers or not any(headers):
        return pd.DataFrame(columns=REQUIRED_COLUMNS), headers, True

    width = len(headers)
    padded_rows = [list(row) + [""] * (width - len(row)) for row in rows]
    frame = pd.DataFrame([row[:width] for row in padded_rows], columns=headers)
    frame = frame.loc[frame.apply(lambda row: any(str(value).strip() for value in row), axis=1)]
    return frame.reset_index(drop=True), headers, False


def header_key(value: Any) -> str:
    return " ".join(str(value).strip().replace("_", " ").split()).lower()


def find_header_index(headers: Sequence[str], canonical_name: str, stock: str) -> Optional[int]:
    wanted = header_key(canonical_name)
    for idx, header in enumerate(headers):
        if header_key(header) == wanted:
            return idx

    symbol_suffixes = (f"_{stock}.NS", f"_{stock}")
    lower_canonical = canonical_name.lower()
    for idx, header in enumerate(headers):
        text = str(header).strip()
        lower = text.lower()
        if any(lower == f"{lower_canonical}{suffix.lower()}" for suffix in symbol_suffixes):
            return idx

    if canonical_name in {"Open", "High", "Low", "Close", "Volume"}:
        prefix = f"{canonical_name.lower()}_"
        matches = [idx for idx, header in enumerate(headers) if str(header).strip().lower().startswith(prefix)]
        if len(matches) == 1:
            return matches[0]

    if canonical_name == "Date":
        for idx, header in enumerate(headers):
            if str(header).strip().lower() in {"date_", "datetime", "time"}:
                return idx
    return None


def resolve_sheet_schema(headers: Sequence[str], stock: str) -> Tuple[Optional[Dict[str, Optional[int]]], str]:
    if not headers:
        return None, "worksheet has no headers"

    mapping: Dict[str, Optional[int]] = {}
    required = ["Date", "Date_str", "Open", "High", "Low", "Close", "Volume"]
    missing: List[str] = []
    for column in required:
        index = find_header_index(headers, column, stock)
        mapping[column] = index
        if index is None:
            missing.append(column)

    mapping["Adj Close"] = find_header_index(headers, "Adj Close", stock)
    if missing:
        return None, f"schema missing required columns after normalization: {missing}; found={list(headers)}"
    return mapping, ""


def normalize_frame_to_required_columns(
    frame: pd.DataFrame,
    headers: Sequence[str],
    schema: Dict[str, Optional[int]],
) -> pd.DataFrame:
    normalized = pd.DataFrame(index=frame.index)
    for column in REQUIRED_COLUMNS:
        source_index = schema.get(column)
        if source_index is None and column == "Adj Close":
            source_index = schema.get("Close")
        if source_index is None:
            normalized[column] = pd.NA
        else:
            normalized[column] = frame[str(headers[source_index])]
    return normalized


def normalize_date_series(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed.dt.normalize()


def normalize_existing_dates(df: pd.DataFrame) -> Tuple[pd.Series, Set[str]]:
    dates = normalize_date_series(df["Date_str"])
    valid_dates = dates.dropna()
    existing = set(valid_dates.dt.strftime("%Y-%m-%d"))
    return dates, existing


def safe_daily_end_date(now: Optional[datetime] = None) -> date:
    current_time = now.astimezone(ZoneInfo(MARKET_TIMEZONE)) if now else datetime.now(ZoneInfo(MARKET_TIMEZONE))
    if current_time.time() < MARKET_DATA_CUTOFF:
        return current_time.date() - timedelta(days=1)
    return current_time.date()


def canonical_yfinance_column(column: Any) -> str:
    name = str(column).strip()
    normalized = " ".join(name.replace("_", " ").split()).lower()
    mapping = {
        "date": "Date",
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjusted close": "Adj Close",
        "volume": "Volume",
    }
    return mapping.get(normalized, name)


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        required = set(REQUIRED_COLUMNS[2:])
        for level in range(normalized.columns.nlevels):
            level_names = [canonical_yfinance_column(value) for value in normalized.columns.get_level_values(level)]
            if required.issubset(set(level_names)):
                normalized.columns = level_names
                break
        else:
            normalized.columns = [
                canonical_yfinance_column(" ".join(str(part).strip() for part in column if str(part).strip()))
                for column in normalized.columns.to_flat_index()
            ]
    else:
        normalized.columns = [canonical_yfinance_column(column) for column in normalized.columns]
    return normalized


def download_stock_data(symbol: str, start: date, end_exclusive: date, interval: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=False,
    )


def format_yfinance_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = flatten_yfinance_columns(raw).reset_index()
    df.columns = [canonical_yfinance_column(column) for column in df.columns]
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns and column != "Date_str"]
    if missing_columns:
        raise ValueError(f"yfinance response missing required columns: {missing_columns}")

    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    except TypeError:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if getattr(df["Date"].dt, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_convert(None)
    df["Date"] = df["Date"].dt.normalize()
    df = df.dropna(subset=["Date"])
    df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=NUMERIC_COLUMNS)
    df = df[REQUIRED_COLUMNS].drop_duplicates(subset=["Date_str"], keep="last")
    return df.sort_values("Date").reset_index(drop=True)


def appendable_rows(
    df_new: pd.DataFrame,
    headers: Optional[Sequence[str]] = None,
    schema: Optional[Dict[str, Optional[int]]] = None,
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for _, row in df_new.iterrows():
        date_value = pd.to_datetime(row["Date"], errors="coerce")
        if pd.isna(date_value):
            raise ValueError("Cannot append row with invalid Date value")
        values = {
            "Date": date_value.strftime("%Y-%m-%d"),
            "Date_str": str(row["Date_str"]),
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Adj Close": float(row["Adj Close"]),
            "Volume": int(float(row["Volume"])) if pd.notna(row["Volume"]) else 0,
        }
        if headers is None or schema is None:
            rows.append([values[column] for column in REQUIRED_COLUMNS])
            continue

        output_row: List[Any] = [""] * len(headers)
        for column, value in values.items():
            index = schema.get(column)
            if index is not None:
                output_row[index] = value
        for column, value in APPEND_DEFAULT_COLUMNS.items():
            index = find_header_index(headers, column, "")
            if index is not None:
                output_row[index] = value
        rows.append(output_row)
    return rows


def append_rows(
    worksheet: Any,
    df_new: pd.DataFrame,
    headers: Optional[Sequence[str]] = None,
    schema: Optional[Dict[str, Optional[int]]] = None,
) -> int:
    if df_new.empty:
        return 0
    rows = appendable_rows(df_new, headers=headers, schema=schema)
    for start in range(0, len(rows), APPEND_BATCH_SIZE):
        batch = rows[start : start + APPEND_BATCH_SIZE]
        with_retry(lambda batch=batch: worksheet.append_rows(batch, value_input_option="RAW"), f"{worksheet.title}: append_rows")
    return len(rows)


def initialize_empty_sheet(worksheet: Any) -> None:
    with_retry(lambda: worksheet.append_row(REQUIRED_COLUMNS, value_input_option="RAW"), f"{worksheet.title}: initialize")


def process_worksheet(worksheet: Any, start_date: date, max_data_date: date, interval: str) -> Tuple[int, str]:
    stock = worksheet.title.strip().upper()
    symbol = f"{stock}.NS"
    frame, headers, is_empty_sheet = worksheet_to_frame(worksheet)

    if is_empty_sheet:
        if not headers or headers != REQUIRED_COLUMNS:
            initialize_empty_sheet(worksheet)
        existing_dates: Set[str] = set()
        fetch_start = start_date
        last_date_text = "empty"
    else:
        schema, schema_error = resolve_sheet_schema(headers, stock)
        if schema is None:
            reason = schema_error
            log(f"[{symbol}] skipped: {reason}")
            return 0, reason
        frame = normalize_frame_to_required_columns(frame, headers, schema)
        if frame.empty:
            existing_dates = set()
            fetch_start = start_date
            last_date_text = "header_only"
        else:
            dates, existing_dates = normalize_existing_dates(frame)
            if dates.dropna().empty:
                reason = "Date_str contains no parseable dates"
                log(f"[{symbol}] skipped: {reason}")
                return 0, reason
            last_date = dates.max().date()
            fetch_start = last_date + timedelta(days=1)
            last_date_text = last_date.isoformat()

    if fetch_start > max_data_date:
        log(f"[{symbol}] last_date={last_date_text} | no_new_data")
        return 0, "no_new_data"

    fetch_end_exclusive = max_data_date + timedelta(days=1)
    try:
        raw = with_retry(
            lambda: download_stock_data(symbol, fetch_start, fetch_end_exclusive, interval),
            f"{symbol}: yfinance download",
            attempts=3,
            backoff=1.5,
        )
    except Exception as exc:
        reason = f"yfinance failure after retries: {exc}"
        log(f"[{symbol}] skipped: {reason}")
        return 0, reason

    fetched_count = 0 if raw is None else int(len(raw))
    if raw is None or raw.empty:
        log(f"[{symbol}] last_date={last_date_text} | fetched=0 | appended=0 | no_new_data")
        return 0, "no_new_data"

    try:
        df_new = format_yfinance_frame(raw)
    except Exception as exc:
        reason = f"malformed yfinance response: {exc}"
        log(f"[{symbol}] skipped: {reason}")
        return 0, reason

    df_new = df_new[~df_new["Date_str"].isin(existing_dates)].copy()
    df_new = df_new.drop_duplicates(subset=["Date_str"], keep="last")
    if df_new.empty:
        log(f"[{symbol}] last_date={last_date_text} | fetched={fetched_count} | appended=0 | duplicate_only")
        return 0, "duplicate_only"

    append_schema = None
    append_headers: Optional[Sequence[str]] = None
    if not is_empty_sheet:
        append_schema, schema_error = resolve_sheet_schema(headers, stock)
        if append_schema is None:
            log(f"[{symbol}] skipped before append: {schema_error}")
            return 0, schema_error
        append_headers = headers

    appended_count = append_rows(worksheet, df_new, headers=append_headers, schema=append_schema)
    log(f"[{symbol}] last_date={last_date_text} | fetched={fetched_count} | appended={appended_count}")
    return appended_count, "updated" if appended_count else "no_new_data"


def selected_worksheet_names(args: argparse.Namespace) -> Optional[Set[str]]:
    names = parse_csv(args.worksheets)
    names.extend(parse_csv(args.worksheet))
    if not names:
        return None
    return {name.strip().upper() for name in names if name.strip()}


def validate_worksheet_reference(worksheet: Any) -> Tuple[bool, str]:
    title = str(getattr(worksheet, "title", "")).strip()
    worksheet_id = getattr(worksheet, "id", None) or getattr(worksheet, "_properties", {}).get("sheetId")
    if not title:
        return False, "worksheet title is missing"
    if worksheet_id is None:
        return False, f"worksheet ID is missing for {title}"
    return True, ""


def iter_target_worksheets(spreadsheet: Any, requested: Optional[Set[str]]) -> Tuple[List[Any], List[str]]:
    worksheets = [
        worksheet
        for worksheet in with_retry(spreadsheet.worksheets, "list worksheets")
        if requested is None or worksheet.title.strip().upper() in requested
    ]
    missing: List[str] = []
    if requested:
        found = {worksheet.title.strip().upper() for worksheet in worksheets}
        missing = sorted(requested - found)
    return worksheets, missing


def classify_sheet_status(appended_count: int, reason: str) -> str:
    if appended_count > 0:
        return "updated"
    if "failure" in reason or "malformed" in reason:
        return "error"
    if reason.startswith("schema mismatch") or reason.startswith("schema missing") or reason == "Date_str contains no parseable dates":
        return "skipped"
    return "no_new_data"


def run_update(args: argparse.Namespace) -> Dict[str, Any]:
    if args.interval != "1d":
        raise ValueError("Only interval='1d' is supported for the ML pipeline schema")

    start_date = pd.to_datetime(args.start_date, errors="raise").date()
    max_data_date = safe_daily_end_date()
    requested = selected_worksheet_names(args)
    log(f"safe_daily_end_date={max_data_date.isoformat()} timezone={MARKET_TIMEZONE} cutoff={MARKET_DATA_CUTOFF.strftime('%H:%M')}")

    client = authorize_gspread(args.google_credentials)
    spreadsheet = with_retry(lambda: client.open_by_key(args.sheet_id), "open spreadsheet")

    rows_added = 0
    stocks_updated = 0
    failures = 0
    per_sheet: List[Dict[str, Any]] = []
    worksheets, missing_worksheets = iter_target_worksheets(spreadsheet, requested)
    for worksheet_name in missing_worksheets:
        failures += 1
        warning = f"Worksheet does not exist and will not be created: {worksheet_name}"
        log(warning)
        per_sheet.append(
            {
                "worksheet": worksheet_name,
                "status": "missing",
                "reason": warning,
                "rows_added": 0,
                "duration_seconds": 0.0,
            }
        )

    for worksheet in worksheets:
        started = time_module.monotonic()
        try:
            valid_worksheet, validation_reason = validate_worksheet_reference(worksheet)
            if not valid_worksheet:
                failures += 1
                per_sheet.append(
                    {
                        "worksheet": getattr(worksheet, "title", ""),
                        "status": "invalid",
                        "reason": validation_reason,
                        "rows_added": 0,
                        "duration_seconds": round(time_module.monotonic() - started, 3),
                    }
                )
                continue
            appended_count, reason = process_worksheet(worksheet, start_date, max_data_date, args.interval)
            status = classify_sheet_status(appended_count, reason)
            if status == "error":
                failures += 1
            if appended_count > 0:
                stocks_updated += 1
                rows_added += appended_count
            per_sheet.append(
                {
                    "worksheet": worksheet.title,
                    "status": status,
                    "reason": reason,
                    "rows_added": appended_count,
                    "duration_seconds": round(time_module.monotonic() - started, 3),
                }
            )
        except Exception as exc:
            failures += 1
            log(f"[{worksheet.title}] failed: unexpected error: {exc}")
            per_sheet.append(
                {
                    "worksheet": worksheet.title,
                    "status": "error",
                    "reason": str(exc),
                    "rows_added": 0,
                    "duration_seconds": round(time_module.monotonic() - started, 3),
                }
            )

    total_targets = len(worksheets) + len(missing_worksheets)
    if failures and failures == total_targets:
        status = "error"
    elif failures:
        status = "partial_error"
    elif rows_added == 0:
        status = "no_new_data"
    else:
        status = "ok"
    return {
        "status": status,
        "stocks_updated": stocks_updated,
        "rows_added": rows_added,
        "worksheet_count": len(worksheets),
        "missing_worksheet_count": len(missing_worksheets),
        "failures": failures,
        "worksheets": per_sheet,
    }


def main() -> None:
    try:
        result = run_update(parse_args())
    except Exception as exc:
        log(f"fatal: {exc}")
        result = {"status": "error", "error": str(exc)}
    print_json(result)
    raise SystemExit(1 if result.get("status") == "error" else 0)


if __name__ == "__main__":
    main()
