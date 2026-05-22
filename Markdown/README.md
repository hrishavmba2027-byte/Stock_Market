# Stock Market Automation

Python replacement for the original `Stock Prediction Automation.json` n8n workflow. The system runs in Docker as two services:

- `api`: FastAPI backend for health, status, and workflow runs.
- `watcher`: continuous Google Sheets watcher that detects row additions and row updates in every worksheet tab.

The workflow is:

Google Sheets change -> watcher diff -> LangChain route decision -> deterministic Python workflow -> `app_data.py` -> `Data_update.py` -> `main.py` prediction pipeline -> Google Sheets update -> logs and optional Slack notification.

## Key Behavior

- Discovers all worksheet tabs dynamically.
- Polls every `WATCHER_POLL_SECONDS`, default `10`.
- Stores row hashes and worksheet state in `workflow_state.json`.
- First startup snapshots current sheet contents and does not process historical rows.
- Only changed worksheets are processed.
- Prediction writes refresh watcher state after successful runs to avoid self-trigger loops.
- LangChain is used only for bounded workflow routing and structured decision validation.

## Required Local Files

The project expects these files to exist:

- `credentials/stock-prices-495408-aa549faac3c5.json` or `stock-prices-495408-aa549faac3c5.json`
- `outputs/Saved_Models/Dense.pt`
- `outputs/Saved_Models/LSTM.pt`
- `outputs/Saved_Models/Transformer.pt`
- `outputs/pipeline_metadata.json`

Credential JSON files are intentionally ignored by git. Keep the Google service account JSON in `credentials/` locally, or set `GOOGLE_CREDENTIALS` to another local path.

The Google service account must have access to spreadsheet:

```text
1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o
```

## Docker Start

Place the Google service account file at `credentials/stock-prices-495408-aa549faac3c5.json`, then start the services:

```bash
docker compose build
docker compose up -d
```

Check the API:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

Watch logs:

```bash
docker compose logs -f api watcher
tail -f logs/app.log logs/error.log
```

## Manual Runs

Run one worksheet:

```bash
curl -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"force":true,"worksheets":["RELIANCE"],"reason":"manual"}'
```

Run all worksheets:

```bash
curl -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"force":true,"reason":"manual_all"}'
```

Run the workflow directly:

```bash
python3 app_data.py run --worksheets RELIANCE,TCS
```

## Environment Variables

Common settings:

```bash
SHEET_ID=1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o
GOOGLE_CREDENTIALS=/app/credentials/stock-prices-495408-aa549faac3c5.json
WORKFLOW_STATE_PATH=/app/workflow_state.json
WATCHER_POLL_SECONDS=10
LANGCHAIN_MODE=deterministic
MODEL_DIR=/app/outputs/Saved_Models
METADATA_PATH=/app/outputs/pipeline_metadata.json
OUTPUT_DIR=/app/outputs/main_inference
SLACK_WEBHOOK_URL=
SLACK_NOTIFY_FAILURES=true
SLACK_NOTIFY_SUCCESS=false
```

## Monthly Fine-Tuning

The monthly fine-tuning workflow is deterministic and externally triggered only.
Run it from GitHub Actions, cron, or a terminal; it does not watch or poll Google
Sheets on its own. Each run archives overflow rows from the operational sheet
into the historical training sheet before operational cleanup, then warm-starts
the existing `Dense.pt`, `LSTM.pt`, and `Transformer.pt` checkpoints only on
historical rows newer than `state/monthly_finetune_state.json`.

Dry-run without sheet or checkpoint writes:

```bash
python3 monthly_finetune.py \
  --google-credentials Credentials.json \
  --dry-run \
  --force-finetune
```

Monthly production run:

```bash
python3 monthly_finetune.py \
  --google-credentials Credentials.json
```

Cron example, first day of each month at 18:00 server time:

```cron
0 18 1 * * cd /Users/hrishavmajumder/Documents/Stock_Market && /usr/bin/python3 monthly_finetune.py --google-credentials Credentials.json >> logs/monthly_finetune.log 2>&1
```

The default training sheet is:

```text
1_gpRclY80tH3r54b9R5HTXqKF7R7bnMoWugF9Fy-boI
```

Useful environment settings:

```text
FORECAST_DAYS=15
ROLLING_OPERATIONAL_ROWS=30
FINE_TUNE_BATCH_ONLY_NEW_DATA=true
OPERATIONAL_SHEET_ID=1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o
HISTORICAL_TRAINING_SHEET_ID=1_gpRclY80tH3r54b9R5HTXqKF7R7bnMoWugF9Fy-boI
```

## Testing

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m compileall app app_data.py Data_update.py main.py monthly_finetune.py
pytest -q
```

Docker smoke test:

```bash
docker compose build
docker compose up -d
curl -f http://localhost:8000/health
curl -f http://localhost:8000/status
docker compose logs --tail=100 api watcher
```

Functional checks:

- Reset first-run behavior with `printf '{}\n' > workflow_state.json`, then start the watcher and confirm it snapshots without running predictions.
- Add a row in any worksheet and confirm one incremental run.
- Edit an existing row and confirm one incremental run.
- Make no changes and confirm no run.
- Configure `SLACK_WEBHOOK_URL`, force an error, and confirm failure notification.

## Debugging

- API logs: `logs/app.log`
- Error logs: `logs/error.log`
- Compose logs: `docker compose logs -f api watcher`
- State file: `workflow_state.json`
- Prediction outputs: `outputs/main_inference/`

If the watcher repeatedly triggers, inspect `workflow_state.json` and confirm the API run is succeeding. Successful watcher-triggered runs refresh state after prediction writes.

If Google auth fails, confirm the service account JSON path and that the spreadsheet is shared with the service account email.

If yfinance fails for one worksheet, that worksheet is logged as failed and other worksheets continue.

## Production Notes

- Keep `LANGCHAIN_MODE=deterministic` unless an LLM-backed router is explicitly added and validated.
- Persist `logs/`, `outputs/`, and `workflow_state.json`.
- Use Docker restart policies, already configured as `unless-stopped`.
- Back up `workflow_state.json` before deployments if avoiding repeated first-run snapshots matters.
- Increase `WATCHER_POLL_SECONDS` if Google API quota pressure appears.
