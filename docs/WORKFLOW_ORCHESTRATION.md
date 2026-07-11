# Stock Market Pipeline — Master Workflow Orchestration

This document describes `run_full_workflow.py` and `run_full_workflow.sh`, the
production-grade master orchestrator that activates the entire stock market ML
pipeline end-to-end, in a single deterministic sequence.

---

## 1. Workflow architecture

The repository already contains a complete, battle-tested pipeline, but its
logic was distributed across several entry points and there was no single
script that drove the *whole* thing in one explicit, validated sequence. The
orchestrator fills exactly that gap.

A deliberate design decision underpins the whole implementation: **the
orchestrator does not re-implement feature engineering, model inference,
Google Sheets I/O or the train/test rollover.** Those subsystems are mature and
correct. Re-writing them would introduce bugs and drift. Instead, the
orchestrator *drives* the existing components in the correct order, inserts
hard validation gates between them, and wraps everything in structured
logging, retry/backoff, atomic writes and failure isolation.

The components it drives:

- **`Data_update.py`** — yfinance ingestion. Detects the last date already in
  each worksheet, downloads only newer rows, deduplicates, and appends
  incrementally to the operational (TEST) Google Sheet in batches.
- **`Feature_Engineering.py`** — `compute_indicators()` computes 29 technical
  indicators plus 30 forward-return label columns. It is imported inline by
  `main.py` for per-symbol inference, and is also run directly by the
  orchestrator as an explicit up-front gate.
- **`main.py` → `run_pipeline()`** — the model engine. Loads worksheet
  payloads from Google Sheets, runs feature engineering per symbol, loads the
  Dense / LSTM / Transformer ensemble, runs inference, writes forecasts back
  into the sheet, and then performs the train/test rollover via
  `sheet_archival.archive_old_rows_for_worksheet()`.
- **`app/pipeline/metadata.py`** — self-healing `pipeline_metadata.json`
  (atomic writes, `.bak` backup, schema validation, schema migration).
- **`app/pipeline/startup.py`** — repository-level startup self-checks.
- **`app/services/sheet_archival.py`** — the TEST → TRAIN rollover engine.
- **`utils/google_auth.py`** — centralized service-account authentication.

The orchestrator sits *above* all of these as a thin, deterministic control
plane.

```
                    ┌──────────────────────────────────────┐
                    │        run_full_workflow.sh          │  launcher
                    │  env setup · venv · logging · safe   │
                    └───────────────────┬──────────────────┘
                                        │
                    ┌───────────────────▼──────────────────┐
                    │        run_full_workflow.py          │  orchestrator
                    │  15 stages · gates · retry · logging │
                    └───┬───────────────┬──────────────┬───┘
            drives      │               │              │
              ┌─────────▼──────┐  ┌─────▼───────┐  ┌────▼─────────────┐
              │ Data_update.py │  │  main.py    │  │ Feature_         │
              │ (yfinance →    │  │ run_pipeline│  │ Engineering.py   │
              │  TEST sheet)   │  │ (FE+infer+  │  │ compute_         │
              │                │  │  forecast+  │  │ indicators()     │
              │                │  │  rollover)  │  │                  │
              └────────────────┘  └──────┬──────┘  └──────────────────┘
                                         │ uses
                          ┌──────────────┼───────────────┐
                   ┌──────▼─────┐ ┌──────▼──────┐ ┌──────▼──────────┐
                   │ metadata.py│ │sheet_       │ │ google_auth.py  │
                   │(self-heal) │ │archival.py  │ │ (service acct)  │
                   └────────────┘ └─────────────┘ └─────────────────┘
```

---

## 2. Orchestration sequence (the 15 stages)

The orchestrator executes exactly the required order. Each stage is timed,
logged, and recorded in the run summary. Stages marked **GATE** abort the
entire workflow on failure — the orchestrator never continues past a failed
gate.

| #  | Stage | Gate | What it does |
|----|-------|------|--------------|
| 1  | Startup validation | **GATE** | Verifies credentials, directories, model checkpoints, metadata schema, Docker volumes, write-permissions and Google API connectivity. |
| 2  | Check YFinance for new data | **GATE** | Reads the last date in every TEST worksheet and probes yfinance to detect whether new rows exist. Read-only. |
| 3  | Download / append new market data | soft | Runs `Data_update.py` (live mode only) — downloads only new rows and appends them incrementally. |
| 4  | Update local workbook / data store | soft | Snapshots every worksheet to `outputs/workflow/data_store/<WS>.csv` as a deterministic local mirror. |
| 5  | Sync new rows into Google Sheets | soft | Verifies the append landed; checks for duplicate dates and schema preservation. |
| 6  | Validate workbook / sheet integrity | **GATE** | Confirms OHLCV schema, chronological ordering and absence of duplicate dates before any feature work. |
| 7  | Run Feature Engineering | **GATE** | Runs `compute_indicators()` on every worksheet and persists engineered frames to `outputs/workflow/engineered/`. |
| 8  | Validate engineered features | **GATE** | Confirms all 29 metadata feature columns exist, are non-empty, and detects schema drift / missing columns. |
| 9  | Update metadata | soft | Atomically records feature-engineering provenance into `pipeline_metadata.json`; re-validates schema after the write. |
| 10 | Run model pipeline (`main.py`) | **GATE** | Executes `main.py run_pipeline` — per-symbol FE + ensemble inference + forecast push + rollover. |
| 11 | Generate forecasts / predictions | soft | Confirms `predictions.csv` and `metrics.json` exist, are fresh, and reports forecast row counts. |
| 12 | Push forecasts to Google Sheets | soft | Verifies forecast write-back (live) and independently checks for duplicate forecast rows. |
| 13 | Train/test rollover validation | soft | Independently validates the rollover: TEST ≤ 30 rows, TRAIN grew, chronology + dedup preserved. |
| 14 | Validate final outputs | soft | Aggregate end-of-run checks: metadata valid, artifacts present, no unfinished stages. |
| 15 | Log workflow summary | soft | Writes `outputs/workflow/run_summary_<run_id>.json` and a timing table. Always runs, even after an abort. |

Why some stages are gates and others are soft: a gate failure means
downstream stages would operate on missing, stale or corrupt data, so the run
is aborted. A soft-stage failure marks the workflow `success_with_warnings` /
`degraded` but does not, by itself, invalidate the model run. Every failure —
gate or soft — is logged loudly; nothing fails silently.

### Dry-run vs live

Dry-run is the **default**. In dry-run the orchestrator performs every
read-only and compute stage but does **not** mutate any Google Sheet: stage 3
(`Data_update.py`) is skipped, stage 10 runs `main.py --dry-run` (computes
forecasts without writing), and stages 5, 12, 13 report simulations. Real
writes happen only with `--live`.

---

## 3. Workflow dependency graph

The orchestrator was designed from an explicit dependency analysis of the
repository:

**Scripts that generate data**
`Data_update.py` (yfinance → TEST sheet); `Feature_Engineering.py`
(`compute_indicators` → engineered frames).

**Scripts that consume data**
`main.py` (consumes the TEST sheet + engineered features);
`Feature_Engineering.py` (consumes raw OHLCV).

**Scripts that update Google Sheets**
`Data_update.py` (appends new OHLCV rows to TEST); `main.py` (writes
`Forecast_Close_T+n` columns to TEST, archives old rows into TRAIN, syncs
decision-feature columns).

**Scripts that perform inference**
`main.py` (`run_inference`, Dense / LSTM / Transformer ensemble).

**Scripts that perform feature engineering**
`Feature_Engineering.py` (`compute_indicators`), invoked inline by `main.py`
and explicitly by the orchestrator.

**Scripts that update metadata**
`app/pipeline/metadata.py` (`safe_save_metadata`, `record_run_result`),
invoked by `main.py`, `app_data.py` and the orchestrator.

**Dependency edges (must-run-before)**

```
yfinance ─▶ Data_update.py ─▶ TEST sheet ─▶ Feature_Engineering ─▶ engineered
                                                                       │
   pipeline_metadata.json (schema) ──────────────────────────────────┐ │
                                                                      ▼ ▼
                                              main.py run_pipeline (inference)
                                                                      │
                                          ┌───────────────────────────┤
                                          ▼                           ▼
                              forecasts → TEST sheet         rollover: TEST → TRAIN
```

The required 15-stage order is a topological sort of this graph: data must
exist before it is engineered; features must be valid before inference;
metadata must be valid before the model loads; forecasts must be produced
before they are pushed; the rollover must run after the forecast push.

---

## 4. Files created / modified

**Created**

- `run_full_workflow.py` — the master orchestrator (15 stages, ~1,800 lines).
- `run_full_workflow.sh` — the safe launcher.
- `WORKFLOW_ORCHESTRATION.md` — this document.

**Created at runtime** (by the orchestrator, not committed by hand)

- `outputs/workflow/` — run summaries (`run_summary_<run_id>.json`,
  `run_summary_latest.json`), the local data store and engineered frames.
- `logs/workflow/` — per-run human log, structured JSONL log, failures log,
  and launcher logs.
- `state/run_full_workflow.lock` — the concurrency lock (created and released
  per run).

**Modified**

- None. No existing script was edited.

**Protected — never touched**

- `outputs/Saved_Models/` — the orchestrator only ever *reads* this directory
  (to confirm `Dense.pt`, `LSTM.pt`, `Transformer.pt` exist and are non-empty).
  It is never written, moved or deleted.

---

## 5. Validation results

Validation was performed in an isolated Linux sandbox. The sandbox is **not**
your Mac: it has no network route to Google APIs, cannot `pip install`
`gspread` / `yfinance` / `torch`, and its bind-mount blocks file *deletion*
(though file *creation* works). Results within those constraints:

**Static checks — PASS**
- `python3 -m py_compile run_full_workflow.py` → compiles cleanly.
- `bash -n run_full_workflow.sh` → no syntax errors.
- `--help` renders the full CLI.

**Live orchestrator run (dry-run) — PASS for all reachable stages**
- Stage 1 (Startup validation) → **OK**. It correctly parsed the
  service-account credentials, found all three model checkpoints, validated
  the metadata schema (29 features, seq_len 20, horizons 1–5), and ran the
  write-probes. A first iteration of the write-probe surfaced a real bug — it
  treated a blocked *cleanup* (`unlink`) as "directory not writable". This was
  fixed: writability is now proven by a successful *write*, and a blocked
  cleanup is tolerated. After the fix, stage 1 passes.
- Stage 2 (Check YFinance) → **FAILED cleanly** on `ModuleNotFoundError: No
  module named 'gspread'` (sandbox has no Google libraries). This correctly
  triggered the gate: stages 3–14 were **skipped**, stage 15 still wrote the
  summary, and the process exited non-zero. This is exactly the
  "never continue past a failed gate, never fail silently" behavior.

**Feature-engineering path — PASS (synthetic data)**
- Stage 7 ran `compute_indicators()` on a synthetic 150-row OHLCV frame →
  150 rows × 62 columns, status OK.
- Stage 8 confirmed all 29 metadata feature columns present, labels present,
  no issues.
- Negative test: an engineered file with `RSI_14` and `MACD_12_26` removed
  caused stage 8 to raise `StageFailure` — schema drift / missing feature
  columns are detected and abort the run before inference.

**Deterministic helper tests — PASS**
- `_parse_last_json`, `retry_with_backoff` (transient recovered, deterministic
  fails fast with no retry), `is_transient_error` classification,
  `_parse_date` (multi-format), `atomic_write_json`, `_safe_market_date`
  (returns a weekday) — all pass.

**Not validatable in the sandbox** — stages 2–6 and 9–14 live behavior
(require Google Sheets + network), stage 10's `main.py` (requires `torch`),
and the Docker run (requires Docker on your Mac). Run these on your machine:

```bash
# 1. dry-run end-to-end (no sheet writes) — recommended first
./run_full_workflow.sh

# 2. live end-to-end (writes to your real sheets)
./run_full_workflow.sh --live

# 3. single stock
./run_full_workflow.sh --live --worksheets RELIANCE

# 4. inside Docker
docker compose --profile tools run --rm pipeline python run_full_workflow.py --live
# or, with the stack already up:
docker compose up --build -d
docker compose exec api python run_full_workflow.py --live
docker compose logs -f
```

After a live run, validate by inspecting `outputs/workflow/run_summary_latest.json`
(`overall_status` should be `success` or `success_with_warnings`), confirming
TEST worksheets hold ≤ 30 rows, TRAIN gained the archived rows, and
`pipeline_metadata.json` carries a fresh `workflow.last_run_id`.

---

## 6. Google Sheets sync strategy

There are two spreadsheets:

- **TEST / operational** — `1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o` —
  the rolling 30-day window the model runs against.
- **TRAIN / historical** — `1_gpRclY80tH3r54b9R5HTXqKF7R7bnMoWugF9Fy-boI` —
  the long-horizon archive used for training.

**Inbound sync (new market data).** `Data_update.py` reads the last `Date_str`
in each worksheet, requests from yfinance only `last_date + 1 … today`,
deduplicates against existing dates, and appends in batches of 500 with
`value_input_option=RAW`. New data is detected *before* any download, so
duplicate downloads and duplicate sheet rows are structurally impossible. The
orchestrator's stage 2 performs an independent read-only detection pass first.

**Outbound sync (forecasts).** `main.py` writes `Forecast_Close_T+1 … T+5`
columns back into the TEST worksheet via `PredictionSheetUpdateService`,
keyed on the existing row so timestamps are preserved and no duplicate
forecast rows are created. Stage 12 independently re-reads the sheet and flags
any duplicate dates.

**Credentials.** Authentication always uses
`credentials/Credentials_New.json` via `utils/google_auth.py`. The orchestrator
exports `GOOGLE_APPLICATION_CREDENTIALS` (and the legacy `GOOGLE_CREDENTIALS`)
to every child process, so `Data_update.py` and `main.py` authenticate
automatically. A single `gspread` client is authorized once per run and reused
across all stages to conserve API quota.

---

## 7. Train/test rollover logic

The rollover keeps the TEST sheet at a fixed rolling window and feeds the
TRAIN archive without losing or duplicating data. It is implemented inside
`main.py` (`cleanup_google_sheet_latest_rows` →
`sheet_archival.archive_old_rows_for_worksheet`) and runs automatically after
each successful forecast push. The orchestrator **reuses** that battle-tested
implementation and adds an **independent validation gate** (stage 13) rather
than re-doing the archival itself (which would risk double-archival).

The rollover guarantees:

- **Latest 30 days only in TEST** — `LATEST_SHEET_ROWS_TO_KEEP = 30`. Older
  rows are removed from the operational sheet once archived.
- **Oldest rows move to TRAIN** — surplus rows are appended to the matching
  worksheet in the historical spreadsheet.
- **Chronological ordering** — after appending, the TRAIN worksheet is sorted
  by date (`sort_destination_chronologically`).
- **Schema consistency** — `align_row_to_headers` maps each archived row onto
  the destination's header layout; `ensure_destination_headers` reconciles
  schema differences.
- **No duplicates** — every candidate row is checked against the set of dates
  already in TRAIN (`existing_date_keys`) and against other candidates;
  duplicates are skipped and counted.
- **No data loss** — operational rows are deleted **only after** a successful
  archive append; if the archive is unavailable the cleanup is *blocked*, not
  forced.
- **Atomic updates** — archive append, chronological sort and source cleanup
  happen as an ordered sequence; the source is never trimmed before the
  archive write succeeds.

Stage 13 independently confirms: every TEST worksheet has ≤ 30 data rows, the
TRAIN worksheet exists, its dates are strictly chronological, and it contains
no duplicate dates. In dry-run the rollover is skipped (it archives rows) and
stage 13 reports a simulation.

---

## 8. Retry / backoff strategy

All fragile operations are wrapped in `retry_with_backoff`:

- **Exponential backoff with jitter** — delay = `base_delay × 2^(attempt-1)`,
  capped at 60 s, plus up to 25% random jitter to avoid thundering-herd
  retries against the Google API.
- **Transient-only retries** — only transient errors are retried (quota / rate
  limit / 429 / 5xx / connection / timeout / SSL). Deterministic errors (bad
  schema, missing column, value errors) **fail fast on the first attempt** so
  real bugs are never masked by retry loops. This was verified by test.
- **Bounded attempts** — Google operations use `GOOGLE_RETRIES` (default 3);
  subprocesses use `SUBPROCESS_RETRIES` (default 2). Retry exhaustion raises
  loudly and is logged as `retry_exhausted`.
- **Layered** — `Data_update.py` and `sheet_archival.py` already have their
  own internal retry; the orchestrator adds an outer retry envelope, so a
  transient blip is absorbed at whichever layer sees it first.
- **API-quota protection** — one authorized client per run, reused everywhere;
  read-only detection in stage 2 avoids redundant downloads in stage 3.

---

## 9. Logging strategy

Every run produces three files in `logs/workflow/`, plus a launcher log:

- `workflow_<run_id>.log` — human-readable, one line per event:
  `timestamp [LEVEL] [component] message | key=value …`.
- `workflow_<run_id>.jsonl` — the same events as structured JSON lines, for
  machine parsing / log shipping.
- `failures_<run_id>.log` — WARNING / ERROR / CRITICAL events only, so a
  failure can be triaged without scrolling the full log.
- `launch_<timestamp>.log` — the launcher tees all stdout/stderr here.

Every event carries a **component tag** so logs can be sliced per subsystem,
covering exactly the required areas: `yfinance`, `sheets`, `feature_engineering`,
`metadata`, `model`, `forecast`, `rollover`, `retry`, `startup`, `validation`,
`stage`, `orchestrator`, `summary`. Retries, timing per stage, and a final
timing table are all logged. All three files plus stderr receive every event,
so `docker compose logs -f` shows the complete picture. The structured run
summary (`outputs/workflow/run_summary_<run_id>.json`) records per-stage
status, durations, details and errors.

Hidden-failure / stale-output / schema-drift detection is built in: stage 6
detects non-chronological or duplicate dates, stage 8 detects missing feature
columns and all-NaN columns, stage 11 flags artifacts older than one hour as
stale, and stage 14 flags any stage that did not finish.

---

## 10. Metadata handling

`pipeline_metadata.json` carries the model architecture contract (feature
columns, `seq_len`, horizons, quantiles) that `main.py` hard-validates before
loading the ensemble. The orchestrator treats it carefully:

- **Stage 1** calls `initialize_metadata_if_missing` and
  `repair_metadata_if_corrupted` — a missing or corrupt file is rebuilt from
  defaults or restored from the `.bak` backup, and the schema is validated. A
  schema that is still invalid after repair is a CRITICAL startup failure.
- **Stage 9** repairs and reloads the file, then records feature-engineering
  provenance under a namespaced `workflow` key (`last_run_id`,
  `last_feature_engineering_at`, the list of engineered worksheets, the run
  mode). It deliberately **never** touches the model-architecture keys.
- All writes go through `safe_save_metadata`: write to a sibling `.tmp` file,
  then `os.replace` (atomic on POSIX/APFS/ext4), then refresh the `.bak`
  backup. After the stage-9 write the file is reloaded and re-validated; if
  the write somehow corrupted the schema the stage fails.
- **Stage 14** re-confirms the schema is valid at end-of-run and that the
  fresh `last_run_id` is present.

Metadata is therefore never corrupted, never half-written, and always
self-heals.

---

## 11. Determinism & production safety

The workflow is deterministic and production-safe by construction:

- **Deterministic order** — the 15 stages always run in the same sequence; a
  topological sort of the dependency graph. No stage can run before its
  inputs exist and have been validated.
- **Gates, not guesses** — stages 1, 2, 6, 7, 8, 10 are hard gates. The
  workflow never continues past a failed feature-engineering or model
  stage; on a gate failure all later stages are explicitly skipped (recorded
  as `skipped`, not silently dropped) and the run exits non-zero.
- **Never silent** — every failure is written to the human log, the JSONL log,
  the failures log and stderr. The final summary lists every failed and
  degraded stage.
- **Idempotent / safe to re-run** — incremental yfinance append, date-keyed
  forecast writes and duplicate-aware archival mean a re-run never creates
  duplicate rows or corrupts history. The default dry-run mode makes a
  re-run completely side-effect-free.
- **Atomic writes** — the run summary, the local data store and metadata are
  all written via tmp-file + `os.replace`; a killed process never leaves a
  half-written file.
- **Concurrency-safe** — a `flock`-based lock in `state/` refuses to start a
  second concurrent run.
- **Failure isolation** — each stage runs inside its own try/except; a soft
  stage failure degrades the run without aborting it, a gate failure aborts
  cleanly, and an unexpected crash is caught, logged with a traceback, and
  still produces a summary.
- **Docker- and ARM64-safe** — all paths resolve through `BASE_DIR` and
  environment variables, so the same script runs identically on the host and
  in the `/app` container. The orchestrator itself imports no native ML
  libraries; `torch` is only loaded inside the `main.py` child process, which
  the Dockerfile already builds correctly for arm64.
- **Protected artifacts** — `outputs/Saved_Models/` is read-only to the
  workflow and is never modified or deleted.

---

## Appendix — quick reference

```bash
# default: dry-run, all worksheets, no sheet writes
./run_full_workflow.sh

# live end-to-end run
./run_full_workflow.sh --live

# live run, selected stocks
./run_full_workflow.sh --live --worksheets "RELIANCE,TCS"

# direct invocation (bypass the launcher)
python run_full_workflow.py --live

# inspect the last run
cat outputs/workflow/run_summary_latest.json
tail -f logs/workflow/workflow_*.log
```

Exit codes: `0` = success / success-with-warnings; `1` = failure or abort.
