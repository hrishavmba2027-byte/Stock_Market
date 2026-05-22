from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

import Data_update


DEFAULT_OPERATIONAL_SHEET_ID = "1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o"
DEFAULT_HISTORICAL_TRAINING_SHEET_ID = "1_gpRclY80tH3r54b9R5HTXqKF7R7bnMoWugF9Fy-boI"
DEFAULT_ROLLING_OPERATIONAL_ROWS = 30


@dataclass
class ArchivalWorksheetResult:
    worksheet: str
    status: str
    rows_seen: int = 0
    valid_rows: int = 0
    malformed_rows: int = 0
    rolling_limit: int = DEFAULT_ROLLING_OPERATIONAL_ROWS
    candidate_rows: int = 0
    rows_appended: int = 0
    duplicate_rows_skipped: int = 0
    rows_removed_from_operational: int = 0
    oldest_archived_date: str = ""
    latest_archived_date: str = ""
    historical_schema_changed: bool = False
    historical_sorted: bool = False
    dry_run: bool = False
    reason: str = ""


def parse_positive_int(value: Any, default: int, *, name: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)
    if parsed <= 0:
        return int(default)
    return parsed


def rolling_operational_rows(default: int = DEFAULT_ROLLING_OPERATIONAL_ROWS) -> int:
    return parse_positive_int(os.environ.get("ROLLING_OPERATIONAL_ROWS"), default, name="ROLLING_OPERATIONAL_ROWS")


def column_number_to_letter(number: int) -> str:
    if number <= 0:
        raise ValueError("column number must be positive")
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def header_key(value: Any) -> str:
    return " ".join(str(value).strip().replace("_", " ").split()).lower()


def dedupe_headers(headers: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen: Dict[str, int] = {}
    for index, header in enumerate(headers):
        text = str(header).strip()
        if not text:
            text = f"Column_{index + 1}"
        key = header_key(text)
        count = seen.get(key, 0)
        seen[key] = count + 1
        output.append(text if count == 0 else f"{text}_{count + 1}")
    return output


def find_date_header_index(headers: Sequence[Any]) -> Optional[int]:
    preferred = ("Date_str", "Date", "Date_", "Datetime", "Time")
    normalized = [header_key(header) for header in headers]
    for wanted in preferred:
        key = header_key(wanted)
        if key in normalized:
            return normalized.index(key)
    for index, key in enumerate(normalized):
        if "date" in key or "time" in key:
            return index
    return None


def parse_sheet_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        if len(parsed) == 0 or pd.isna(parsed[0]):
            return None
        parsed = parsed[0]
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None) if hasattr(timestamp, "tz_convert") else timestamp.tz_localize(None)
    return timestamp.normalize()


def date_key(value: Any) -> Optional[str]:
    parsed = parse_sheet_date(value)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%d")


def pad_rows(values: Sequence[Sequence[Any]]) -> Tuple[List[List[Any]], int]:
    width = max((len(row) for row in values), default=0)
    return [list(row) + [""] * (width - len(row)) for row in values], width


def read_worksheet_values(worksheet: Any) -> List[List[Any]]:
    return Data_update.with_retry(worksheet.get_all_values, f"{worksheet.title}: get_all_values")


def find_worksheet_by_title(spreadsheet: Any, title: str) -> Optional[Any]:
    wanted = str(title).strip().upper()
    for worksheet in Data_update.with_retry(spreadsheet.worksheets, "list worksheets"):
        if str(getattr(worksheet, "title", "")).strip().upper() == wanted:
            return worksheet
    return None


def batch_update_rows(worksheet: Any, start_row: int, rows: List[List[Any]], width: int) -> None:
    if not rows:
        return
    end_col = column_number_to_letter(width)
    end_row = start_row + len(rows) - 1
    Data_update.with_retry(
        lambda: worksheet.batch_update(
            [{"range": f"A{start_row}:{end_col}{end_row}", "values": rows}],
            raw=True,
        ),
        f"{worksheet.title}: batch_update",
    )


def append_rows_raw(worksheet: Any, rows: List[List[Any]]) -> None:
    for start in range(0, len(rows), Data_update.APPEND_BATCH_SIZE):
        batch = rows[start : start + Data_update.APPEND_BATCH_SIZE]
        Data_update.with_retry(
            lambda batch=batch: worksheet.append_rows(batch, value_input_option="RAW"),
            f"{worksheet.title}: append_rows",
        )


def create_destination_worksheet(spreadsheet: Any, title: str, headers: List[str]) -> Any:
    worksheet = Data_update.with_retry(
        lambda: spreadsheet.add_worksheet(
            title=title,
            rows=max(100, len(headers) + 1),
            cols=max(1, len(headers)),
        ),
        f"{title}: add historical worksheet",
    )
    batch_update_rows(worksheet, 1, [headers], len(headers))
    return worksheet


def ensure_destination_headers(
    archive_spreadsheet: Any,
    archive_worksheet: Optional[Any],
    title: str,
    source_headers: List[str],
    *,
    dry_run: bool,
) -> Tuple[Optional[Any], List[str], bool, str]:
    source_headers = dedupe_headers(source_headers)
    if archive_worksheet is None:
        archive_worksheet = find_worksheet_by_title(archive_spreadsheet, title) if archive_spreadsheet is not None else None
    if archive_worksheet is None:
        if dry_run:
            return None, source_headers, True, "dry_run_destination_would_be_created"
        if archive_spreadsheet is None:
            return None, source_headers, False, "historical spreadsheet unavailable"
        archive_worksheet = create_destination_worksheet(archive_spreadsheet, title, source_headers)
        return archive_worksheet, source_headers, True, ""

    values = read_worksheet_values(archive_worksheet)
    if not values or not any(str(value).strip() for value in values[0]):
        if dry_run:
            return archive_worksheet, source_headers, True, "dry_run_headers_would_be_initialized"
        batch_update_rows(archive_worksheet, 1, [source_headers], len(source_headers))
        return archive_worksheet, source_headers, True, ""

    destination_headers = dedupe_headers(values[0])
    existing_keys = {header_key(header) for header in destination_headers}
    missing_headers = [header for header in source_headers if header_key(header) not in existing_keys]
    if not missing_headers:
        return archive_worksheet, destination_headers, False, ""

    updated_headers = destination_headers + missing_headers
    if dry_run:
        return archive_worksheet, updated_headers, True, "dry_run_headers_would_be_extended"
    batch_update_rows(archive_worksheet, 1, [updated_headers], len(updated_headers))
    return archive_worksheet, updated_headers, True, ""


def existing_date_keys(worksheet: Any, headers: Sequence[str]) -> Set[str]:
    values = read_worksheet_values(worksheet)
    if len(values) <= 1:
        return set()
    date_idx = find_date_header_index(headers)
    if date_idx is None:
        return set()
    keys: Set[str] = set()
    for row in values[1:]:
        if date_idx >= len(row):
            continue
        key = date_key(row[date_idx])
        if key:
            keys.add(key)
    return keys


def align_row_to_headers(row: Sequence[Any], source_headers: Sequence[str], destination_headers: Sequence[str]) -> List[Any]:
    source_by_key: Dict[str, int] = {}
    for index, header in enumerate(source_headers):
        source_by_key.setdefault(header_key(header), index)
    output: List[Any] = [""] * len(destination_headers)
    for dest_index, header in enumerate(destination_headers):
        source_index = source_by_key.get(header_key(header))
        if source_index is not None and source_index < len(row):
            output[dest_index] = row[source_index]
    return output


def sort_destination_chronologically(worksheet: Any, headers: Sequence[str], *, dry_run: bool) -> bool:
    values = read_worksheet_values(worksheet)
    if len(values) <= 2:
        return False
    padded, width = pad_rows(values)
    data_rows = padded[1:]
    date_idx = find_date_header_index(headers)
    if date_idx is None:
        return False

    valid_rows: List[Tuple[pd.Timestamp, int, List[Any]]] = []
    invalid_rows: List[Tuple[int, List[Any]]] = []
    for original_idx, row in enumerate(data_rows):
        if not any(str(value).strip() for value in row):
            invalid_rows.append((original_idx, row))
            continue
        parsed = parse_sheet_date(row[date_idx] if date_idx < len(row) else "")
        if parsed is None:
            invalid_rows.append((original_idx, row))
        else:
            valid_rows.append((parsed, original_idx, row))
    ordered = [row for _, row in sorted(invalid_rows, key=lambda item: item[0])]
    ordered.extend(row for _, _, row in sorted(valid_rows, key=lambda item: (item[0], item[1])))
    if ordered == data_rows:
        return False
    if not dry_run:
        batch_update_rows(worksheet, 2, ordered, width)
    return True


def select_rows_for_archival(
    values: List[List[Any]],
    keep_rows: int,
) -> Tuple[List[str], List[Tuple[str, pd.Timestamp, int, List[Any]]], List[List[Any]], Dict[str, int], str]:
    if not values:
        return [], [], [], {"rows_seen": 0, "valid_rows": 0, "malformed_rows": 0}, "worksheet_empty"
    padded, width = pad_rows(values)
    headers = dedupe_headers(padded[0])
    data_rows = [row + [""] * (width - len(row)) for row in padded[1:]]
    if not data_rows:
        return headers, [], [], {"rows_seen": 0, "valid_rows": 0, "malformed_rows": 0}, "worksheet_empty"

    date_idx = find_date_header_index(headers)
    if date_idx is None:
        return headers, [], data_rows, {"rows_seen": len(data_rows), "valid_rows": 0, "malformed_rows": len(data_rows)}, "date_column_missing"

    valid_rows: List[Tuple[pd.Timestamp, int, List[Any]]] = []
    invalid_rows: List[Tuple[int, List[Any]]] = []
    for original_idx, row in enumerate(data_rows):
        if not any(str(value).strip() for value in row):
            invalid_rows.append((original_idx, row))
            continue
        parsed = parse_sheet_date(row[date_idx] if date_idx < len(row) else "")
        if parsed is None:
            invalid_rows.append((original_idx, row))
        else:
            valid_rows.append((parsed, original_idx, row))

    stats = {
        "rows_seen": len(data_rows),
        "valid_rows": len(valid_rows),
        "malformed_rows": len(invalid_rows),
    }
    if len(valid_rows) <= keep_rows:
        retained = [row for _, row in sorted(invalid_rows, key=lambda item: item[0])]
        retained.extend(row for _, _, row in sorted(valid_rows, key=lambda item: (item[0], item[1])))
        return headers, [], retained, stats, "row_count_at_or_below_limit"

    sorted_valid = sorted(valid_rows, key=lambda item: (item[0], item[1]))
    archive_valid = sorted_valid[: len(sorted_valid) - keep_rows]
    keep_valid = sorted_valid[len(sorted_valid) - keep_rows :]
    archive_rows = [(date.strftime("%Y-%m-%d"), date, original_idx, row) for date, original_idx, row in archive_valid]
    retained = [row for _, row in sorted(invalid_rows, key=lambda item: item[0])]
    retained.extend(row for _, _, row in keep_valid)
    return headers, archive_rows, retained, stats, ""


def cleanup_source_after_archival(
    worksheet: Any,
    retained_rows: List[List[Any]],
    original_data_count: int,
    width: int,
    *,
    dry_run: bool,
) -> int:
    rows_to_remove = original_data_count - len(retained_rows)
    if rows_to_remove <= 0:
        return 0
    if dry_run:
        return rows_to_remove
    if retained_rows:
        batch_update_rows(worksheet, 2, retained_rows, width)
        Data_update.with_retry(
            lambda: worksheet.delete_rows(len(retained_rows) + 2, original_data_count + 1),
            f"{worksheet.title}: delete archived rows",
        )
    else:
        Data_update.with_retry(
            lambda: worksheet.delete_rows(2, original_data_count + 1),
            f"{worksheet.title}: delete all archived rows",
        )
    return rows_to_remove


def archive_old_rows_for_worksheet(
    source_worksheet: Any,
    *,
    archive_spreadsheet: Any = None,
    archive_worksheet: Any = None,
    keep_rows: int = DEFAULT_ROLLING_OPERATIONAL_ROWS,
    dry_run: bool = False,
) -> ArchivalWorksheetResult:
    title = str(getattr(source_worksheet, "title", "")).strip()
    if source_worksheet is None:
        return ArchivalWorksheetResult(worksheet="", status="skipped", reason="worksheet_missing", dry_run=dry_run)
    if not title:
        return ArchivalWorksheetResult(worksheet="", status="error", reason="worksheet title missing", dry_run=dry_run)

    keep_rows = parse_positive_int(keep_rows, DEFAULT_ROLLING_OPERATIONAL_ROWS, name="keep_rows")
    values = read_worksheet_values(source_worksheet)
    headers, archive_rows, retained_rows, stats, reason = select_rows_for_archival(values, keep_rows)
    rows_seen = int(stats.get("rows_seen", 0))
    width = max(len(headers), max((len(row) for row in values), default=0))
    result = ArchivalWorksheetResult(
        worksheet=title,
        status="skipped",
        rows_seen=rows_seen,
        valid_rows=int(stats.get("valid_rows", 0)),
        malformed_rows=int(stats.get("malformed_rows", 0)),
        rolling_limit=keep_rows,
        candidate_rows=len(archive_rows),
        dry_run=dry_run,
        reason=reason,
    )
    if not archive_rows:
        return result
    if archive_spreadsheet is None and archive_worksheet is None:
        result.status = "blocked"
        result.reason = "historical archive worksheet unavailable; operational cleanup skipped"
        return result

    destination, destination_headers, schema_changed, header_reason = ensure_destination_headers(
        archive_spreadsheet,
        archive_worksheet,
        title,
        headers,
        dry_run=dry_run,
    )
    result.historical_schema_changed = schema_changed
    if destination is None and not dry_run:
        result.status = "blocked"
        result.reason = header_reason or "historical archive worksheet unavailable; operational cleanup skipped"
        return result

    existing_dates = set() if dry_run or destination is None else existing_date_keys(destination, destination_headers)
    unique_by_date: Dict[str, Tuple[pd.Timestamp, int, List[Any]]] = {}
    duplicate_rows = 0
    for key, parsed_date, original_idx, row in archive_rows:
        if key in existing_dates:
            duplicate_rows += 1
            continue
        if key in unique_by_date:
            duplicate_rows += 1
        unique_by_date[key] = (parsed_date, original_idx, row)

    ordered_new = [
        (key, parsed_date, row)
        for key, (parsed_date, _, row) in sorted(unique_by_date.items(), key=lambda item: (item[1][0], item[1][1]))
    ]
    rows_to_append = [align_row_to_headers(row, headers, destination_headers) for _, _, row in ordered_new]
    result.rows_appended = len(rows_to_append)
    result.duplicate_rows_skipped = duplicate_rows
    if ordered_new:
        result.oldest_archived_date = ordered_new[0][0]
        result.latest_archived_date = ordered_new[-1][0]
    else:
        archived_dates = [key for key, _, _, _ in archive_rows]
        result.oldest_archived_date = min(archived_dates)
        result.latest_archived_date = max(archived_dates)

    if rows_to_append and not dry_run:
        assert destination is not None
        append_rows_raw(destination, rows_to_append)
        result.historical_sorted = sort_destination_chronologically(destination, destination_headers, dry_run=False)
    elif dry_run and rows_to_append:
        result.historical_sorted = True

    result.rows_removed_from_operational = cleanup_source_after_archival(
        source_worksheet,
        retained_rows,
        rows_seen,
        width,
        dry_run=dry_run,
    )
    result.status = "dry_run" if dry_run else "ok"
    result.reason = (
        "old operational rows archived before cleanup"
        if rows_to_append
        else "old operational rows already existed in archive; cleanup allowed"
    )
    if header_reason:
        result.reason = f"{result.reason}; {header_reason}"
    if result.malformed_rows:
        result.reason = f"{result.reason}; malformed rows retained in operational sheet"
    if not math.isfinite(result.rows_removed_from_operational):
        raise RuntimeError("invalid cleanup row count")
    return result
