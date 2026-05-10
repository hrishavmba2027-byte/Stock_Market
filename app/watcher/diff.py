from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from app.models.schemas import ChangeType, WorksheetChange


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalize_row(row: Iterable[Any]) -> str:
    values = ["" if value is None else str(value).strip() for value in row]
    while values and values[-1] == "":
        values.pop()
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def hash_row(row: Iterable[Any]) -> str:
    return hashlib.sha256(canonicalize_row(row).encode("utf-8")).hexdigest()


def hash_sheet(row_hashes: Dict[str, str]) -> str:
    joined = "\n".join(f"{idx}:{row_hashes[idx]}" for idx in sorted(row_hashes, key=lambda item: int(item)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def worksheet_id(worksheet: Any) -> str:
    value = getattr(worksheet, "id", None) or getattr(worksheet, "_properties", {}).get("sheetId")
    return str(value if value is not None else getattr(worksheet, "title", "unknown"))


def build_worksheet_snapshot(worksheet: Any, values: List[List[Any]]) -> Dict[str, Any]:
    row_hashes = {str(index): hash_row(row) for index, row in enumerate(values, start=1)}
    return {
        "worksheet_id": worksheet_id(worksheet),
        "title": str(getattr(worksheet, "title", "")).strip(),
        "row_count": len(values),
        "row_hashes": row_hashes,
        "sheet_hash": hash_sheet(row_hashes),
        "last_seen_at": now_iso(),
    }


def build_state_from_snapshots(snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": now_iso(),
        "worksheets": snapshots,
        "last_successful_run_at": None,
        "last_run_result": None,
    }


def diff_worksheet(
    previous: Optional[Dict[str, Any]],
    current: Optional[Dict[str, Any]],
) -> Optional[WorksheetChange]:
    if previous is None and current is None:
        return None
    if current is None:
        return WorksheetChange(
            worksheet_id=str(previous.get("worksheet_id", "")),
            title=str(previous.get("title", "")),
            change_types=[ChangeType.WORKSHEET_REMOVED],
            row_count_before=int(previous.get("row_count", 0)),
            row_count_after=0,
            sheet_hash_before=previous.get("sheet_hash"),
            sheet_hash_after=None,
        )
    if previous is None:
        rows = list(range(1, int(current.get("row_count", 0)) + 1))
        return WorksheetChange(
            worksheet_id=str(current.get("worksheet_id", "")),
            title=str(current.get("title", "")),
            change_types=[ChangeType.WORKSHEET_ADDED],
            added_rows=rows,
            row_count_before=0,
            row_count_after=int(current.get("row_count", 0)),
            sheet_hash_before=None,
            sheet_hash_after=current.get("sheet_hash"),
        )
    if previous.get("sheet_hash") == current.get("sheet_hash"):
        return None

    previous_hashes = previous.get("row_hashes", {}) or {}
    current_hashes = current.get("row_hashes", {}) or {}
    previous_count = int(previous.get("row_count", 0))
    current_count = int(current.get("row_count", 0))
    added_rows: List[int] = []
    modified_rows: List[int] = []

    for row_index in range(1, current_count + 1):
        key = str(row_index)
        old_hash = previous_hashes.get(key)
        new_hash = current_hashes.get(key)
        if old_hash is None and new_hash is not None:
            added_rows.append(row_index)
        elif old_hash != new_hash:
            modified_rows.append(row_index)

    change_types: List[ChangeType] = []
    if added_rows or current_count > previous_count:
        change_types.append(ChangeType.ROW_ADDED)
    if modified_rows:
        change_types.append(ChangeType.ROW_UPDATED)

    if not change_types:
        return None
    return WorksheetChange(
        worksheet_id=str(current.get("worksheet_id", "")),
        title=str(current.get("title", "")),
        change_types=change_types,
        added_rows=added_rows,
        modified_rows=modified_rows,
        row_count_before=previous_count,
        row_count_after=current_count,
        sheet_hash_before=previous.get("sheet_hash"),
        sheet_hash_after=current.get("sheet_hash"),
    )


def diff_snapshots(previous_state: Dict[str, Any], current_snapshots: Dict[str, Dict[str, Any]]) -> List[WorksheetChange]:
    previous_snapshots = previous_state.get("worksheets", {}) or {}
    changes: List[WorksheetChange] = []
    worksheet_ids = set(previous_snapshots) | set(current_snapshots)
    for worksheet_key in sorted(worksheet_ids):
        change = diff_worksheet(previous_snapshots.get(worksheet_key), current_snapshots.get(worksheet_key))
        if change is not None:
            changes.append(change)
    return changes


def has_actionable_changes(changes: List[WorksheetChange]) -> bool:
    actionable = {ChangeType.ROW_ADDED, ChangeType.ROW_UPDATED, ChangeType.WORKSHEET_ADDED}
    return any(set(change.change_types) & actionable for change in changes)

