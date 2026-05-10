from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator

from app.models.schemas import ExecutionMode, NotificationLevel, RouteType


class WorkflowDecision(BaseModel):
    route: RouteType
    worksheets: List[str] = Field(default_factory=list)
    should_run: bool
    reason: str
    noise: bool = False
    execution_mode: ExecutionMode = ExecutionMode.SKIP
    notification_level: NotificationLevel = NotificationLevel.NONE

    @validator("worksheets", pre=True)
    def normalize_worksheets(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
        else:
            parts = [str(part).strip() for part in value]
        seen = set()
        normalized = []
        for item in parts:
            if not item:
                continue
            key = item.upper()
            if key not in seen:
                seen.add(key)
                normalized.append(item)
        return normalized

    @validator("should_run")
    def skip_routes_do_not_run(cls, value, values):
        route = values.get("route")
        if route == RouteType.SKIP and value:
            raise ValueError("skip route cannot set should_run=true")
        return value


class PredictionWriteDecision(BaseModel):
    should_write: bool
    worksheet: str
    row_number: int
    predicted_value: Optional[float] = None
    reason: str

    @validator("row_number")
    def row_number_must_be_data_row(cls, value: int) -> int:
        if value < 2:
            raise ValueError("prediction writes must target a data row")
        return value

    @validator("predicted_value")
    def prediction_must_be_finite_when_present(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("predicted value must be finite")
        return value
