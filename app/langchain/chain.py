from __future__ import annotations

from typing import Any, Dict, Iterable, List

try:
    from langchain_core.runnables import RunnableLambda
except Exception:  # pragma: no cover - fallback for environments without langchain-core
    class RunnableLambda:  # type: ignore
        def __init__(self, func):
            self.func = func

        def invoke(self, value):
            return self.func(value)

from app.langchain.schemas import PredictionWriteDecision, WorkflowDecision
from app.models.schemas import (
    ChangeType,
    ExecutionMode,
    NotificationLevel,
    RouteType,
    WorkflowRunRequest,
)


ACTIONABLE_CHANGE_TYPES = {
    ChangeType.ROW_ADDED,
    ChangeType.ROW_UPDATED,
    ChangeType.WORKSHEET_ADDED,
}


def _validate_decision(payload: Dict[str, Any]) -> WorkflowDecision:
    if hasattr(WorkflowDecision, "model_validate"):
        return WorkflowDecision.model_validate(payload)
    return WorkflowDecision.parse_obj(payload)


def _unique_worksheets(values: Iterable[str]) -> List[str]:
    seen = set()
    worksheets: List[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = item.upper()
        if key not in seen:
            seen.add(key)
            worksheets.append(item)
    return worksheets


def deterministic_route(request: WorkflowRunRequest) -> WorkflowDecision:
    if request.force:
        worksheets = _unique_worksheets(request.worksheets)
        route = RouteType.INCREMENTAL_UPDATE if worksheets else RouteType.FULL_UPDATE
        return _validate_decision(
            {
                "route": route,
                "worksheets": worksheets,
                "should_run": True,
                "reason": request.reason or "manual force run",
                "noise": False,
                "execution_mode": ExecutionMode.INCREMENTAL if worksheets else ExecutionMode.FULL,
                "notification_level": NotificationLevel.SUCCESS,
            }
        )

    worksheets = _unique_worksheets(request.worksheets)
    if request.event is not None:
        actionable_titles: List[str] = []
        for change in request.event.changes:
            change_types = set(change.change_types)
            if change_types & ACTIONABLE_CHANGE_TYPES:
                actionable_titles.append(change.title)
        worksheets = _unique_worksheets(actionable_titles or worksheets)

    if worksheets:
        return _validate_decision(
            {
                "route": RouteType.INCREMENTAL_UPDATE,
                "worksheets": worksheets,
                "should_run": True,
                "reason": request.reason or "actionable sheet change",
                "noise": False,
                "execution_mode": ExecutionMode.INCREMENTAL,
                "notification_level": NotificationLevel.SUCCESS,
            }
        )

    return _validate_decision(
        {
            "route": RouteType.SKIP,
            "worksheets": [],
            "should_run": False,
            "reason": "no actionable row additions or row updates",
            "noise": True,
            "execution_mode": ExecutionMode.SKIP,
            "notification_level": NotificationLevel.NONE,
        }
    )


def build_workflow_chain(mode: str = "deterministic"):
    # The deterministic Runnable is intentionally the default. Optional LLM-backed
    # routing can be added behind the same WorkflowDecision schema without changing
    # any Google Sheets, state, subprocess, or error handling code.
    return RunnableLambda(deterministic_route)


def validate_prediction_write(
    worksheet: str,
    row_number: int,
    predicted_value: Any,
    current_predicted_value: Any,
) -> PredictionWriteDecision:
    try:
        numeric_prediction = float(predicted_value)
    except (TypeError, ValueError):
        numeric_prediction = None

    try:
        current_flag = float(str(current_predicted_value).strip() or 0)
    except (TypeError, ValueError):
        current_flag = 0.0

    if numeric_prediction is None or numeric_prediction != numeric_prediction:
        payload = {
            "should_write": False,
            "worksheet": worksheet,
            "row_number": row_number,
            "predicted_value": None,
            "reason": "prediction value is missing or invalid",
        }
    elif current_flag != 0:
        payload = {
            "should_write": False,
            "worksheet": worksheet,
            "row_number": row_number,
            "predicted_value": numeric_prediction,
            "reason": "row is already marked predicted",
        }
    else:
        payload = {
            "should_write": True,
            "worksheet": worksheet,
            "row_number": row_number,
            "predicted_value": numeric_prediction,
            "reason": "row has valid prediction and predicted flag is 0",
        }

    if hasattr(PredictionWriteDecision, "model_validate"):
        return PredictionWriteDecision.model_validate(payload)
    return PredictionWriteDecision.parse_obj(payload)


def build_prediction_write_chain():
    return RunnableLambda(
        lambda payload: validate_prediction_write(
            worksheet=payload["worksheet"],
            row_number=payload["row_number"],
            predicted_value=payload["predicted_value"],
            current_predicted_value=payload.get("current_predicted_value", 0),
        )
    )
