from app.watcher.diff import build_state_from_snapshots, build_worksheet_snapshot, diff_snapshots, hash_row


class FakeWorksheet:
    def __init__(self, title="RELIANCE", sheet_id=1):
        self.title = title
        self.id = sheet_id


def test_hash_row_is_stable_for_equivalent_trailing_blanks():
    assert hash_row(["A", "B", ""]) == hash_row(["A", "B"])


def test_first_run_snapshot_has_no_diff_against_same_values():
    worksheet = FakeWorksheet()
    snapshot = build_worksheet_snapshot(worksheet, [["Date", "Close"], ["2026-01-01", "10"]])
    state = build_state_from_snapshots({snapshot["worksheet_id"]: snapshot})
    assert diff_snapshots(state, {snapshot["worksheet_id"]: snapshot}) == []


def test_row_addition_detection():
    worksheet = FakeWorksheet()
    before = build_worksheet_snapshot(worksheet, [["Date"], ["2026-01-01"]])
    after = build_worksheet_snapshot(worksheet, [["Date"], ["2026-01-01"], ["2026-01-02"]])
    changes = diff_snapshots(build_state_from_snapshots({before["worksheet_id"]: before}), {after["worksheet_id"]: after})
    assert len(changes) == 1
    assert changes[0].added_rows == [3]


def test_row_update_detection():
    worksheet = FakeWorksheet()
    before = build_worksheet_snapshot(worksheet, [["Date", "Close"], ["2026-01-01", "10"]])
    after = build_worksheet_snapshot(worksheet, [["Date", "Close"], ["2026-01-01", "11"]])
    changes = diff_snapshots(build_state_from_snapshots({before["worksheet_id"]: before}), {after["worksheet_id"]: after})
    assert len(changes) == 1
    assert changes[0].modified_rows == [2]

