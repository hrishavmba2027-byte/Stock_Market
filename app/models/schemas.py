from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "json"):
        return json.loads(model.json())
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


class ChangeType(str, Enum):
    ROW_ADDED = "row_added"
    ROW_UPDATED = "row_updated"
    WORKSHEET_ADDED = "worksheet_added"
    WORKSHEET_REMOVED = "worksheet_removed"


class RouteType(str, Enum):
    SKIP = "skip"
    INCREMENTAL_UPDATE = "incremental_update"
    FULL_UPDATE = "full_update"


class ExecutionMode(str, Enum):
    SKIP = "skip"
    INCREMENTAL = "incremental"
    FULL = "full"


class NotificationLevel(str, Enum):
    NONE = "none"
    SUCCESS = "success"
    FAILURE = "failure"


class RunStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    RUNNING = "running"


class RowChange(BaseModel):
    row_index: int
    change_type: ChangeType
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None


class WorksheetChange(BaseModel):
    worksheet_id: str
    title: str
    change_types: List[ChangeType]
    added_rows: List[int] = Field(default_factory=list)
    modified_rows: List[int] = Field(default_factory=list)
    row_count_before: int = 0
    row_count_after: int = 0
    sheet_hash_before: Optional[str] = None
    sheet_hash_after: Optional[str] = None

    @validator("title")
    def title_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("worksheet title cannot be empty")
        return value


class SheetChangeEvent(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    spreadsheet_id: str
    detected_at: str = Field(default_factory=utc_now)
    changes: List[WorksheetChange] = Field(default_factory=list)
    source: str = "watcher"

    @property
    def worksheet_titles(self) -> List[str]:
        seen = set()
        titles: List[str] = []
        for change in self.changes:
            title = change.title.strip()
            key = title.upper()
            if key not in seen:
                seen.add(key)
                titles.append(title)
        return titles


class WorkflowRunRequest(BaseModel):
    event: Optional[SheetChangeEvent] = None
    worksheets: List[str] = Field(default_factory=list)
    force: bool = False
    reason: str = "api"
    change_summary: Dict[str, Any] = Field(default_factory=dict)

    @validator("worksheets", pre=True)
    def normalize_worksheets(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = [part.strip() for part in value.split(",")]
        else:
            raw_values = [str(part).strip() for part in value]
        normalized: List[str] = []
        seen = set()
        for item in raw_values:
            if not item:
                continue
            key = item.upper()
            if key not in seen:
                seen.add(key)
                normalized.append(item)
        return normalized


class SubprocessResult(BaseModel):
    command: List[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    parsed_json: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0
    attempts: int = 1
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class WorkflowRunResult(BaseModel):
    status: RunStatus
    run_id: str
    route: RouteType
    worksheets: List[str] = Field(default_factory=list)
    started_at: str
    finished_at: Optional[str] = None
    duration_seconds: float = 0.0
    decision: Dict[str, Any] = Field(default_factory=dict)
    subprocess_result: Optional[Dict[str, Any]] = None
    update_result: Optional[Dict[str, Any]] = None
    prediction_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WorkflowStatus(BaseModel):
    service: str = "stock-market-automation"
    status: str = "ok"
    now: str = Field(default_factory=utc_now)
    last_run: Optional[Dict[str, Any]] = None
    state_exists: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)
