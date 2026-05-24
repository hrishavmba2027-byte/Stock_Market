# Stock-Market Pipeline — Fixes & Run Guide

## How to Run the Workflow

```bash
cd ~/Documents/Stock_Market
./run_workflow.sh           # full run (Docker API, all sheets)
./run_workflow.sh --local   # direct Python (no Docker)
```

Or via the direct orchestrator:
```bash
./run_full_workflow.sh          # dry-run (default)
./run_full_workflow.sh --live   # real end-to-end run
```

---

## What Was Fixed (Root-Cause Summary)

### Bug 1 — Workflow silently skipped every time
**File:** `app/langchain/chain.py` · `deterministic_route()`

**Root cause:** When `force=False` (the default) and no specific worksheets were
supplied, `deterministic_route()` returned `should_run=False` with
`RouteType.SKIP`. The `WorkflowOrchestrator` saw this and skipped the entire run
without touching any data.

**Fix:** Replaced the SKIP return with a `FULL_UPDATE` / `should_run=True` return.
When no event or worksheet list is provided the workflow now runs a full refresh
of every worksheet every time.

---

### Bug 2 — Already-predicted rows never re-predicted (`predicted=1` guard)
**File:** `app/langchain/chain.py` · `validate_prediction_write()`

**Root cause:** An `elif current_flag != 0: → should_write=False` branch meant
that any row already marked `predicted=1` was permanently excluded from future
prediction writes. After the first run, every row in the sheet had `predicted=1`,
so subsequent runs wrote zero predictions.

**Fix:** Removed the `elif` block. Every row with a valid prediction value is now
written unconditionally.

---

### Bug 3 — Stale rows re-used instead of re-predicted
**File:** `main.py` · `prepare_stock_part()` / `process_payloads()` / `run_pipeline()`

**Root cause:** `valid_indices` was filtered to `predicted_values == 0`, so rows
already marked predicted were excluded from the inference input even before the
write step. Combined with Bug 2, no row was ever refreshed after the first run.

**Fix:** Added `force_all_rows=True` flag. When set:
- Every row past the model warm-up window (`seq_len = 20`) is eligible for
  inference on every run, regardless of its current `predicted` / forecast state.
- The eligibility assert is bypassed for the same reason.
- `run_pipeline()` always passes `force_all_rows=True`.

---

### Bug 4 — No full-sheet overwrite existed; only cell-level partial writes
**Files:** `main.py` · `app/services/google_sheet_updates.py`

**Root cause:** The old write path called `PredictionSheetUpdateService.update_
prediction_status()` which used `worksheet.batch_update()` to write only the
`predicted`, `Predicted_Close_Price`, and `Forecast_Close_T+N` columns row-by-row.
Feature-engineered indicator columns (RSI_14, MACD_12_26, etc.) were computed
in memory but never written back to the sheet. On the next run the sheet still
contained raw, un-engineered data.

**Fix (new function):** `overwrite_worksheet_with_engineered_data()` in
`app/services/google_sheet_updates.py`:
1. Takes the full engineered DataFrame (OHLCV + all 24 indicator columns) plus
   the model's `row_predictions` / `row_forecasts` dicts.
2. Merges predictions back by original sheet-row number.
3. Strips internal tracking columns (`__sheet_row_number`, `__sort_position`,
   forward labels).
4. Calls `worksheet.clear()` followed by a single `worksheet.update(all_values,
   "A1")` — one atomic clear-then-write, no partial appends.

**Fix (pipeline wiring):** `run_pipeline()` now calls
`overwrite_worksheet_with_engineered_data()` instead of
`update_google_predictions()`. The archival step (archive excess rows to TRAIN,
keep TEST ≤ 30 rows) runs unconditionally after each successful overwrite.

---

### Bug 5 — FE output discarded; Stage 7 result never reached Google Sheets
**File:** `main.py` · `prepare_stock_part()`

**Root cause:** The engineered DataFrame produced by `compute_indicators()` was
used only to build the model feature matrix and was then discarded. The sheet
only ever received the three prediction columns.

**Fix:** Added `engineered_frame: Optional[pd.DataFrame]` field to
`StockInferencePart`. `prepare_stock_part()` now stores `engineered_for_sheet =
engineered.copy()` after FE + leakage-column removal + NaN-row cleanup and
attaches it to the returned part. `run_pipeline()` reads `part.engineered_frame`
to perform the full-sheet overwrite.

---

## What Happens on Every Run Now

```
Data_update.py
  └─ Fetches new yfinance rows and APPENDS them to the TEST sheet

main.py  (called by run_pipeline)
  ├─ Loads ALL rows from TEST sheet
  ├─ compute_indicators()  ← Feature Engineering per symbol
  │     OHLCV + 24 indicator columns computed in memory
  ├─ run_inference()        ← Model predicts every row ≥ seq_len
  │     row_predictions {sheet_row → price}
  │     row_forecasts   {sheet_row → [T+1 … T+N]}
  ├─ overwrite_worksheet_with_engineered_data()
  │     worksheet.clear()
  │     worksheet.update(header + all rows with FE + predictions)
  │     ← ENTIRE TEST sheet replaced with fresh data every run
  └─ cleanup_google_sheet_latest_rows()
        archive excess rows → TRAIN sheet
        keep TEST ≤ 30 rows

sync_decision_features()
  └─ Writes 11 categorical decision columns to both sheets
```

---

## Files Modified

| File | Change |
|------|--------|
| `app/langchain/chain.py` | Removed SKIP gate; removed `predicted=1` guard |
| `app/services/google_sheet_updates.py` | Added `overwrite_worksheet_with_engineered_data()` |
| `main.py` | Added `engineered_frame` field; `force_all_rows` flag; full-overwrite write path |
| `run_full_workflow.sh` | (earlier fix) Xcode Python stub detection; `|| true` on version print |
| `run_workflow.sh` | (earlier fix) Shell-variable-in-Python heredoc; `FORCE_FLAG` env var |

---

## Validation Checklist

After running `./run_workflow.sh` (or `./run_full_workflow.sh --live`), verify:

1. **No silent skip:** The workflow log should NOT contain `"should_run": false`
   or `"route": "SKIP"`.

2. **TEST sheet overwritten:** Open the TEST spreadsheet. Every data row should
   have values in the indicator columns (RSI_14, MACD_12_26, etc.), `predicted=1`,
   a non-empty `Predicted_Close_Price`, and `Forecast_Close_T+1` through `T+5`.

3. **Row count ≤ 30:** The TEST sheet should have at most 30 data rows after
   cleanup. Excess rows should appear in the TRAIN sheet.

4. **Logs confirm overwrite:** Look for
   `"event": "worksheet_overwritten"` in the structured log output, confirming
   that `worksheet.clear()` + `worksheet.update()` ran successfully.

5. **No stale data:** Run the workflow twice in a row. On the second run the
   predictions should be freshly computed (not skipped). The
   `Predicted_Close_Price` values may differ slightly if the model is
   non-deterministic.

---

## Quick Sanity Commands

```bash
# Syntax-check all modified Python files
python3 -m py_compile app/langchain/chain.py app/services/google_sheet_updates.py main.py && echo "All OK"

# Confirm skip gate is gone
grep "RouteType.SKIP\|should_run.*False" app/langchain/chain.py || echo "OK: no SKIP gate"

# Confirm predicted=1 guard is gone
grep "already marked predicted" app/langchain/chain.py || echo "OK: guard removed"

# Confirm overwrite function exists
grep "def overwrite_worksheet_with_engineered_data" app/services/google_sheet_updates.py
```
