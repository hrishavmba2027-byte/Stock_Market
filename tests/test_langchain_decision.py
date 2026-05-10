from app.langchain.chain import deterministic_route
from app.models.schemas import ChangeType, SheetChangeEvent, WorkflowRunRequest, WorksheetChange


def test_skip_without_changes():
    decision = deterministic_route(WorkflowRunRequest(reason="test"))
    assert decision.should_run is False
    assert decision.route == "skip"


def test_force_all_routes_full_update():
    decision = deterministic_route(WorkflowRunRequest(force=True, reason="manual"))
    assert decision.should_run is True
    assert decision.route == "full_update"
    assert decision.worksheets == []


def test_row_update_routes_incremental_sheet():
    event = SheetChangeEvent(
        spreadsheet_id="sheet",
        changes=[
            WorksheetChange(
                worksheet_id="1",
                title="RELIANCE",
                change_types=[ChangeType.ROW_UPDATED],
                modified_rows=[2],
                row_count_before=2,
                row_count_after=2,
            )
        ],
    )
    decision = deterministic_route(WorkflowRunRequest(event=event, reason="watcher_change"))
    assert decision.should_run is True
    assert decision.route == "incremental_update"
    assert decision.worksheets == ["RELIANCE"]

