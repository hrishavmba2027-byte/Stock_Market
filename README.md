# Stock Market ML Automation

Repository: <https://github.com/hrishavmba2027-byte/Stock_Market>

End-to-end NSE stock forecasting pipeline: OHLCV + news/social ingestion → feature
engineering (29 indicators + forward-return labels) → a PyTorch ensemble
(Dense / LSTM / Transformer) producing multi-horizon quantile forecasts → results
written back to Google Sheets. Ingested news and its sentiment analysis are
stored in Firebase (Firestore). Ships with a FastAPI server, a Google-Sheets
polling watcher, Docker Compose services, and scheduled GitHub Actions.

## Directory layout

```
Stock_Market/
├── README.md                  ← you are here (only doc kept at root)
├── main.py                    ← inference entry point (ensemble forecasts)
├── Data_update.py             ← yfinance OHLCV ingestion → Google Sheets
├── Feature_Engineering.py     ← indicator + label computation (imported, not run)
├── monthly_finetune.py        ← monthly model fine-tuning job
├── run_full_workflow.py       ← 15-stage end-to-end orchestrator
├── app_data.py                ← CLI + FastAPI wrapper (uvicorn app_data:app)
├── app.py                     ← FastAPI re-export (app.api.server)
├── app/                       ← API, watcher, pipeline, services packages
├── ingestion/                 ← news / reddit / X / fundamentals collectors
├── features/                  ← cross-sectional + sentiment features
├── mlops/                     ← model artifact upload (GitHub Releases)
├── utils/                     ← Google auth helpers
├── scripts/                   ← helper launchers & one-off tools
│   ├── run_full_workflow.sh   ← safe launcher for run_full_workflow.py
│   ├── run_workflow.sh        ← API/CLI/local workflow runner
│   ├── fix_environment.sh     ← macOS venv repair / bootstrap
│   └── trim_test_sheets.py    ← one-time operational-sheet trimmer
├── docs/                      ← all project documentation (see docs/)
│   └── archive/               ← superseded doc versions
├── Notebooks/                 ← model training notebooks (portable)
├── tests/                     ← pytest suite
├── Data/                      ← local workbooks / parquet artifacts
├── outputs/                   ← models, metadata, inference results
├── credentials/               ← Google + Firebase service-account JSONs (gitignored)
├── state/, logs/, cache/      ← runtime state
├── Dockerfile, docker-compose.yml
├── requirements.txt           ← runtime deps (+ requirements-dev.txt, -lock.txt)
└── .env.example               ← copy to .env and fill in
```

The seven Python files at the root are kept there deliberately — the Dockerfile,
docker-compose services, and GitHub Actions workflows all invoke them by that
path, and they import each other as top-level modules.

## Setup (fresh machine)

```bash
git clone https://github.com/hrishavmba2027-byte/Stock_Market.git
cd Stock_Market

# 1. Environment file — all paths inside are relative, so no edits needed
#    beyond filling in your own secrets/IDs.
cp .env.example .env

# 2. Google service-account JSON (gitignored) — place it here and point the
#    GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_CREDENTIALS entries in .env at it:
mkdir -p credentials
cp /path/to/your-service-account.json credentials/

# 2b. Firebase Admin SDK JSON (gitignored) — used to store ingested news and
#     its sentiment analysis in your Firebase project's Firestore.
#     Download it from Firebase console → Project settings → Service accounts
#     → "Generate new private key", then point FIREBASE_CREDENTIALS in .env at it:
cp /path/to/your-firebase-adminsdk.json credentials/Firebase_Credentials.json

# 3a. Python environment (manual)
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-dev.txt        # optional: test/dev tools

# 3b. …or the automated macOS bootstrap (creates venv, installs everything,
#     verifies imports and Apple-Silicon MPS support):
chmod +x scripts/fix_environment.sh
./scripts/fix_environment.sh
```

All scripts are meant to be run **from the repo root** with the venv activated.
Relative paths in `.env` (e.g. `./credentials/...`) resolve against the repo
root, so the same `.env` works on any machine.

> **Note:** run `git clone` from *outside* any existing checkout — cloning from
> inside the repo root creates a nested `Stock_Market/Stock_Market/` copy
> (harmless, gitignored, but delete it: `rm -rf Stock_Market/`).

### Verify a fresh clone (smoke test)

After the setup steps above, these three commands prove the clone works
end-to-end — Sheets auth, Firebase news storage, and model inference (the
only live write is the news upload; inference runs with `--dry-run`):

```bash
source venv/bin/activate
python Data_update.py --worksheets RELIANCE          # Sheets round-trip
python -m ingestion.news_ingest --tickers RELIANCE   # news → Firebase `news`
python main.py --worksheets RELIANCE --dry-run       # models load + 15-day forecast
```

## Running the pipeline

### Full end-to-end workflow (15 stages)

```bash
python run_full_workflow.py                          # dry-run (default, no sheet writes)
python run_full_workflow.py --live                   # real end-to-end run
python run_full_workflow.py --worksheets RELIANCE,TCS
python run_full_workflow.py --live --worksheets RELIANCE
python run_full_workflow.py --live --google-credentials credentials/your-sa.json
```

Or via the safe launcher (auto-detects venv/Python, loads `.env`, tees logs to
`logs/workflow/`):

```bash
./scripts/run_full_workflow.sh                       # dry-run
./scripts/run_full_workflow.sh --live                # real run
./scripts/run_full_workflow.sh --live --worksheets RELIANCE
```

### OHLCV data update (yfinance → Google Sheets)

```bash
python Data_update.py
python Data_update.py --sheet-id "$SHEET_ID" --google-credentials credentials/your-sa.json
python Data_update.py --worksheets RELIANCE,TCS --start-date 2015-01-01 --interval 1d
```

### Inference (ensemble forecasts)

```bash
python main.py                                       # Google Sheets source (default)
python main.py --dry-run                             # predict without writing to sheets
python main.py --worksheets RELIANCE,TCS --device cpu
python main.py --source workbook --workbook Data/nse_stock_data.xlsx
python main.py --latest-only                         # only the newest eligible row
python main.py --refresh-existing-forecasts          # overwrite existing forecast cells
python main.py --all-eligible-rows --plots           # full backfill + HTML plots
```

### Monthly fine-tuning

```bash
python monthly_finetune.py --dry-run                 # validate + train, no writes
python monthly_finetune.py                           # real fine-tune run
python monthly_finetune.py \
  --operational-sheet-id "$OPERATIONAL_SHEET_ID" \
  --historical-sheet-id  "$HISTORICAL_TRAINING_SHEET_ID" \
  --google-credentials   credentials/your-sa.json \
  --model-dir  outputs/Saved_Models \
  --metadata   outputs/pipeline_metadata.json \
  --output-dir outputs/monthly_finetune \
  --state-file state/monthly_finetune_state.json \
  --device cpu
python monthly_finetune.py --force-finetune --skip-yfinance-update
```

### API server + watcher (local, no Docker)

```bash
uvicorn app_data:app --host 0.0.0.0 --port 8000      # FastAPI server
python app_data.py api --host 0.0.0.0 --port 8000    # same, via the CLI wrapper
python app_data.py run --worksheets RELIANCE --dry-run  # one-shot update+predict
python -m app.watcher.service                        # Google Sheets polling daemon
```

Or the interactive runner (talks to a running API / Docker stack):

```bash
./scripts/run_workflow.sh                            # full run via API
./scripts/run_workflow.sh --stock RELIANCE           # single stock
./scripts/run_workflow.sh --stock "RELIANCE,TCS"     # multiple stocks
./scripts/run_workflow.sh --force                    # run even without sheet changes
./scripts/run_workflow.sh --dry-run                  # validate without writing
./scripts/run_workflow.sh --cli                      # via docker compose run
./scripts/run_workflow.sh --local                    # via local venv Python
./scripts/run_workflow.sh status                     # last-run status
./scripts/run_workflow.sh health                     # API health check
./scripts/run_workflow.sh logs                       # tail container logs
```

### Ingestion & feature jobs (news → Firebase)

News, sentiment analysis, Reddit, X, and fundamentals are stored in the
Firebase project pointed at by `FIREBASE_CREDENTIALS` in `.env` (Firestore
collections `news`, `sentiment_latest`, `fundamentals`, …). If that variable
is unset, writes fall back to the `GOOGLE_CREDENTIALS` service account.
Every module loads `.env` itself, so these run directly from the repo root
with the venv active — no Docker needed. Verified live:

```bash
python -m ingestion.news_ingest --tickers RELIANCE,TCS,INFY   # news → Firestore `news`
python -m features.sentiment                          # FinBERT analysis → `sentiment_latest`
```

The local `Data/archive/*.parquet` files are only upload retry-queues — they
are deleted once their rows reach Firestore. Sentiment therefore reads the
news back from the Firestore `news` collection when the local file is absent,
so the two commands above work standalone in any order.

All jobs:

```bash
python -m ingestion.collect_all                      # everything in one shot
python -m ingestion.collect_all --no-sentiment
python -m ingestion.collect_all --tickers RELIANCE,TCS
python -m ingestion.news_ingest                      # yfinance news (all NIFTY-50 tickers)
python -m ingestion.reddit_ingest                    # Reddit (anonymous scrape)
python -m ingestion.x_ingest                         # X/Twitter scrape
python -m ingestion.fundamentals --lookback-quarters 4
python -m features.cross_sectional                   # NIFTY/VIX index cache
```

Check what landed in Firebase at any time:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ingestion._firestore import init_firestore_client
c = init_firestore_client()
print('project:', c.project)
for d in c.collection('news').stream(): print(' news doc:', d.id)
"
```

### MLOps & utilities

```bash
python -m mlops.upload_models                        # upload .pt files to GitHub Release
python -m mlops.upload_models --tag models-current --only Dense LSTM
python scripts/trim_test_sheets.py --dry-run         # one-time sheet trimmer
python scripts/trim_test_sheets.py --keep 60 --worksheets RELIANCE,TCS
```

### Tests

```bash
pytest -q                                            # full suite
pytest tests/test_monthly_finetune.py -q             # single module
```

## Running live in phases (no Docker — recommended locally)

The local venv is the lightest way to run live: no image builds, no container
memory overhead, and PyTorch uses Apple-Silicon MPS automatically. Run stages
sequentially for a limited worksheet batch (verified live end-to-end):

```bash
source venv/bin/activate

# Phase 1 — append fresh OHLCV rows to the operational sheet (LIVE write)
python Data_update.py --worksheets RELIANCE

# Phase 2 — ensemble forecasts written back to the sheet (LIVE write)
python main.py --worksheets RELIANCE

# Phase 3 — news + sentiment analysis stored in Firebase (LIVE write)
python -m ingestion.news_ingest --tickers RELIANCE,TCS,INFY
python -m features.sentiment

# Repeat phases 1–2 for the next worksheet batch:
python Data_update.py --worksheets TCS,INFY
python main.py --worksheets TCS,INFY
```

## Docker (optional — for deployment parity)

```bash
docker compose build                                 # build the image
docker compose up -d api                             # API only (add watcher when needed)
docker compose ps                                    # service status
docker compose logs -f                               # follow logs
docker compose down                                  # stop everything
```

Docker (torch + FinBERT) is memory-hungry — prefer the local venv above on
laptops, start services one at a time, and stop them when done
(`docker compose stop api`). The containerized equivalents of the phased
live run (verified earlier) are:

```bash
docker compose --profile tools run --rm pipeline python Data_update.py --worksheets RELIANCE
docker compose --profile tools run --rm pipeline python main.py --worksheets RELIANCE --device cpu
```

One-shot jobs run through the `pipeline` service (opt-in `tools` profile):

```bash
docker compose --profile tools run --rm pipeline python run_full_workflow.py
docker compose --profile tools run --rm pipeline python run_full_workflow.py --live
docker compose --profile tools run --rm pipeline python run_full_workflow.py --live --worksheets RELIANCE,TCS
docker compose --profile tools run --rm pipeline python Data_update.py
docker compose --profile tools run --rm pipeline python main.py --dry-run
docker compose --profile tools run --rm pipeline python monthly_finetune.py --dry-run
docker compose --profile tools run --rm pipeline python -m ingestion.collect_all
docker compose --profile tools run --rm pipeline python -m features.sentiment
docker compose --profile tools run --rm pipeline pytest -q
```

## Notebooks

```bash
source venv/bin/activate
pip install jupyterlab
jupyter lab Notebooks/model_2_GPU_.ipynb             # current training notebook
```

The notebooks locate the repo root automatically (current dir → parents →
Kaggle paths). If you run them from an unusual location, set:

```bash
export STOCK_MARKET_ROOT=/path/to/Stock_Market
```

They read `TRAIN_END` / `TEST_END` / `BACK_TEST_START` / `BACK_TEST_END` from
`.env`. `Notebooks/model_2(GPU).ipynb` and `model_1.ipynb` are earlier
iterations kept for reference.

## Scheduled GitHub Actions

| Workflow | Schedule | Runs |
|---|---|---|
| `daily-data-collection.yml` | daily | `Data_update.py` + ingestion jobs |
| `daily-prediction.yml` | daily | `Data_update.py` → `features.cross_sectional` → `main.py` |
| `weekly-fundamentals.yml` | weekly | `ingestion.fundamentals` |
| `news-sentiment.yml` | scheduled | news ingest + `features.sentiment` |
| `monthly-finetune.yml` | 1st of month | `monthly_finetune.py` |
| `monthly-training.yml` | monthly | full retrain path |
| `run-full-workflow.yml` | manual dispatch | `run_full_workflow.py` (dry/live) |
| `upload-models.yml` | manual/after training | `mlops.upload_models` |
| `tests.yml` | on push/PR | `pytest` |

## Configuration (.env)

Copy `.env.example` → `.env` and fill in. Highlights:

| Variable | Purpose |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` / `GOOGLE_CREDENTIALS` | Path to service-account JSON (relative paths resolve against repo root) |
| `SHEET_ID` / `OPERATIONAL_SHEET_ID` | Live OHLCV Google Sheet |
| `HISTORICAL_TRAINING_SHEET_ID` | Training-archive sheet |
| `TRAIN_END`, `TEST_END`, `BACK_TEST_START`, `BACK_TEST_END` | Train/test/backtest split dates |
| `FORECAST_DAYS`, `ROLLING_OPERATIONAL_ROWS` | Forecast horizon & sheet retention |
| `DEVICE` | `auto` (CUDA → MPS → CPU), `cpu`, `cuda`, `mps` |
| `FIREBASE_CREDENTIALS` | Firebase Admin SDK JSON for the news + sentiment-analysis Firestore project; all Firestore writes prefer this, falling back to `GOOGLE_CREDENTIALS` when unset |
| `FIREBASE_PROJECT` | Optional explicit Firebase project id (default: read from the credentials JSON) |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` | Reddit ingestion |
| `CREDENTIALS_FILE` | Credentials filename used by the docker-compose mount |

## Documentation

Everything else lives in [docs/](docs/):

- [DEPLOYMENT.md](docs/DEPLOYMENT.md) — deployment guide
- [WORKFLOW_ORCHESTRATION.md](docs/WORKFLOW_ORCHESTRATION.md) — orchestrator internals
- [FIXES_AND_RUN_GUIDE.md](docs/FIXES_AND_RUN_GUIDE.md) — quick-start & fix log
- [ROADMAP.md](docs/ROADMAP.md) — project roadmap
- [ENVIRONMENT_REPAIR_GUIDE.md](docs/ENVIRONMENT_REPAIR_GUIDE.md) / [DIAGNOSTIC_REPORT.md](docs/DIAGNOSTIC_REPORT.md) — environment troubleshooting
- [P0_CHANGES.md](docs/P0_CHANGES.md) — P0 change log
- `docs/archive/` — superseded doc versions; `docs/Stock Prediction Automation.json` — n8n workflow export; `docs/compose-debugger-eval.html` — compose debugging report
