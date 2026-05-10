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

## Testing

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m compileall app app_data.py Data_update.py main.py
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
