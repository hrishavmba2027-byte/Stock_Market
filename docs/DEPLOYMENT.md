# Free-Tier Hosting & Automation Guide

Run the whole pipeline **live, daily, zero cost** using only free tiers of services you already use. No credit card needed beyond GCP's free tier (which doesn't charge unless you exceed limits we won't come close to).

> **Scope of "free":** This guide hosts the pipeline currently shipped — OHLCV ingest, fundamentals → Firestore, news/Reddit/X scraping, FinBERT sentiment, cross-sectional features. The future Claude decision layer (P3) is *not* free — Anthropic charges per token. Budget separately when that ships.

---

## TL;DR Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Actions (the only compute you need)                     │
│  ────────────────────────────────────────                       │
│  Daily 11:00 UTC  → daily-prediction.yml   (after IST close)    │
│  Daily 18:00 UTC  → news-sentiment.yml     (social + FinBERT)   │
│  Weekly Sun 02:00 → weekly-fundamentals.yml (Firestore refresh) │
│  Monthly  04:00   → monthly-training.yml   (retrain + promote)  │
│  On push          → tests.yml              (pytest gate)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ reads secrets; writes parquet + .pt to GCS
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Google Cloud (free tier)                                       │
│  ────────────────────────                                       │
│  • Google Sheets   — OHLCV source of truth (already in use)     │
│  • Firestore       — quarterly fundamentals + indicator snapshot│
│                      (1 GB storage, 50k reads/d, 20k writes/d)  │
│  • Cloud Storage   — parquet archive + MODEL REGISTRY           │
│                      gs://bucket/models/{Dense,LSTM,Transformer}│
│                      gs://bucket/models/history/{run_id}/...    │
│                      (5 GB always free in us-east1)             │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Slack webhook (free)  ← daily PnL + error summaries            │
└─────────────────────────────────────────────────────────────────┘
```

**Model lifecycle (read this if you're wondering "how does daily prediction get a model?"):**

```
   training/train.py  ─ runs monthly ─►  writes .pt + metadata
                                              │
                                              ▼
                  ┌──────────────────────────────────────────────┐
                  │   GCS — gs://bucket/models/                  │
                  │   ───────────────────────                    │
                  │   Dense.pt              ◄── current pointer  │
                  │   LSTM.pt                                     │
                  │   Transformer.pt                              │
                  │   pipeline_metadata.json                      │
                  │                                               │
                  │   history/2026-04-01/{Dense.pt, …}            │
                  │   history/2026-05-01/{Dense.pt, …}            │
                  │   history/2026-06-01/{Dense.pt, …}  ← rollback│
                  └──────────────────────────────────────────────┘
                                              │
                                              ▼
   daily-prediction.yml  ─ runs daily ─►  downloads "current" → predicts
```

GitHub Actions runners are ephemeral — they cannot persist a 1.6 MB
`Transformer.pt` across runs. **GCS is the durable model registry.**
The daily workflow reads, the monthly workflow writes. The .pt files in
the repo (`outputs/Saved_Models/*.pt`) are the *bootstrap* — they get
uploaded to GCS once via [mlops/upload_models.py](mlops/upload_models.py),
then GCS is the source of truth from that point on.

**Why no always-on server?** The pipeline is *batch* by nature — daily-bar data, weekly fundamentals, decisions per stock per day. The existing FastAPI + 10s watcher polling exists for the n8n trigger style, but for a free swing-trading workflow we just run scheduled jobs. If you later need real-time triggers, see [Optional: Always-On VM](#optional-always-on-vm) at the end.

---

## Costs at this scale (all $0)

| Service | Free tier | Our usage |
|---|---|---|
| GitHub Actions (public repo) | unlimited minutes | ~30 min/day × 30 days ≈ 900 min/month |
| GitHub Actions (private repo) | 2,000 min/month | same ~900 min — fits |
| Firestore | 1 GB, 50k reads/d, 20k writes/d | 49 tickers × 4 quarters = 196 writes/week |
| Google Cloud Storage | 5 GB, 5k Class A ops/month | <100 MB parquet, ~100 ops/day |
| Google Sheets API | unlimited (rate-limited) | ~200 reads/day, ~50 writes/day |
| Reddit API (PRAW) | 60 req/min, 1k/day | ~5 subreddits × 1 call/day = trivial |
| yfinance | "polite" use | 49 tickers × 1 call/day |
| Slack webhook | unlimited | ≤ 10 messages/day |
| FinBERT | local, no API | model cached on Actions runner |

**Where you could blow the budget:** running the workflows every 5 minutes instead of daily, or accidentally pulling Reddit's `top of all time` instead of `new`. Both already capped in code.

---

## Prerequisites (15 min)

You need accounts for all of these. All free.

- [x] **GitHub** account → host this repo
- [x] **Google Cloud Platform** project → Firestore + GCS + Sheets API (you already have one — `stock-prices-495408`)
- [x] **Reddit developer app** → PRAW credentials
- [ ] **Slack workspace** with an Incoming Webhook (optional — but high-leverage for daily summaries)

---

## Step 1 — Push the code to GitHub

If the repo is already on GitHub, skip to step 2.

```bash
cd /Users/kingshukbarua/Documents/Stock_Market
git remote -v   # check if origin is set

# If not, create a new repo on github.com first, then:
git remote add origin git@github.com:<your-username>/stock-market-automation.git
git push -u origin main
```

**Public vs private:**
- **Public repo:** unlimited Actions minutes. Best for cost.
- **Private repo:** 2,000 free Actions minutes/month — still enough for our daily-only schedule.

Trade-off: a public repo means anyone can see your *code* but not your *secrets* (which live in GitHub Actions Secrets, encrypted). Your service-account JSON and Reddit creds stay private regardless.

---

## Step 2 — Enable Google Cloud services

Open <https://console.cloud.google.com/> and select your existing project (`stock-prices-495408`).

### 2.1 Enable APIs

In **APIs & Services → Library**, enable:

- [x] Google Sheets API (already enabled — you're using it)
- [ ] Cloud Firestore API
- [ ] Cloud Storage API
- [ ] IAM Service Account Credentials API

Click each, then "Enable".

### 2.2 Initialize Firestore (one-time)

In **Firestore → Create database**:
- Mode: **Native mode**
- Location: pick the region closest to NSE traders — **asia-south1 (Mumbai)** is ideal
- Click "Create"

This is one-click and the free tier auto-applies.

### 2.3 Create a Cloud Storage bucket

In **Cloud Storage → Buckets → Create**:
- Name: `<your-project-id>-stock-archive` (must be globally unique; suffix with random chars if taken)
- Location type: **Region**
- Location: **asia-south1** (same as Firestore — keep latency low)
- Storage class: **Standard**
- Public access: leave **Prevent public access** on
- Click "Create"

Free tier is **5 GB always free** in US regions (`us-east1`, `us-west1`, `us-central1`). asia-south1 is *not* in the free tier — see the trade-off below.

> **Trade-off — asia-south1 vs us-east1:**
> - asia-south1: same region as Firestore (fast), but storage is paid (~$0.026/GB-month — still pennies for <1 GB).
> - us-east1: free, but ~250ms farther from Firestore. For batch jobs (us only — fine).
>
> **Recommended:** put the bucket in **us-east1** to stay fully free. Reads happen in GitHub Actions (US-located runners anyway), and the latency to Firestore in asia-south1 doesn't matter for batch jobs.

### 2.4 Service-account permissions

Open your existing service account (`stock-prices-...@...iam.gserviceaccount.com`) under **IAM & Admin → Service Accounts**. Grant these additional roles:

- **Cloud Datastore User** (for Firestore reads/writes)
- **Storage Object Admin** (for GCS reads/writes)

The service account already has Sheets access from the original setup.

### 2.5 Download the service account JSON (if you don't already have it)

In the service account → **Keys → Add key → Create new key → JSON**. Save the file. This is the `GOOGLE_CREDENTIALS` referenced in `.env.example`.

⚠️ **Never commit this file.** `.gitignore` already excludes `credentials/*.json`.

---

## Step 3 — Reddit developer app

Open <https://www.reddit.com/prefs/apps>.

- Click **create another app...** at the bottom
- Type: **script**
- Name: `stock-market-automation`
- About URL: leave blank
- Redirect URI: `http://localhost:8080` (required field; never actually used)
- Click "create app"

Capture the two values:
- **client ID** = the random string under the app name (top-left)
- **client secret** = labelled "secret" on the right

These go into GitHub Secrets in the next step.

---

## Step 4 — GitHub Secrets

In your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**. Add each:

| Secret name | Value |
|---|---|
| `GOOGLE_CREDENTIALS_JSON` | **Paste the entire contents** of your service-account JSON file (open it in a text editor, copy all). |
| `SHEET_ID` | `1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o` |
| `GCS_BUCKET` | The bucket name from step 2.3 |
| `FIRESTORE_PROJECT` | Your GCP project ID (e.g. `stock-prices-495408`) |
| `REDDIT_CLIENT_ID` | From step 3 |
| `REDDIT_CLIENT_SECRET` | From step 3 |
| `REDDIT_USER_AGENT` | `stock-market-automation/0.1 by u/<your-reddit-username>` |
| `SLACK_WEBHOOK_URL` | (optional) Incoming webhook URL from your Slack workspace — see step 6 |

> The workflows read these via `${{ secrets.X }}` and reconstitute the credentials JSON on disk before running the Python jobs. They never appear in logs.

---

## Step 5 — Wire the GitHub Actions workflows

The repo ships five workflows in `.github/workflows/`:

| File | Schedule | What it does |
|---|---|---|
| [`tests.yml`](.github/workflows/tests.yml) | every push / PR | pytest suite (must be green before deploying) |
| [`daily-prediction.yml`](.github/workflows/daily-prediction.yml) | weekdays 11:00 UTC (16:30 IST) | Downloads current model from GCS → OHLCV update → predict → archive |
| [`news-sentiment.yml`](.github/workflows/news-sentiment.yml) | daily 18:00 UTC | News + Reddit + X + FinBERT sentiment |
| [`weekly-fundamentals.yml`](.github/workflows/weekly-fundamentals.yml) | Sundays 02:00 UTC | Quarterly fundamentals → Firestore + parquet |
| [`monthly-training.yml`](.github/workflows/monthly-training.yml) | 1st of month 04:00 UTC | Retrain ensemble → upload new weights to GCS as "current" |

**To enable** — once you push the code:

1. Go to the repo → **Actions** tab
2. GitHub will ask if you want to enable workflows for this repo — click "I understand my workflows, go ahead and enable them"
3. Each workflow shows a "Run workflow" button (manual trigger) — click it once to verify before relying on the cron

You can adjust schedules by editing the `cron:` line in each YAML. Crons use **UTC** (not your local time).

### Recommended timing (for NSE)

| When (IST) | Why | Cron (UTC) |
|---|---|---|
| 16:30 IST (weekdays) | NSE closes at 15:30; data settles by 16:00 | `0 11 * * 1-5` |
| 23:30 IST (daily) | Catch overnight US/global news cycle | `0 18 * * *` |
| Sun 07:30 IST | Weekly fundamentals refresh, before market opens Monday | `0 2 * * 0` |
| 1st of month 09:30 IST | Monthly retrain, before market opens that day | `0 4 1 * *` |

---

## Step 5.5 — Model lifecycle (bootstrap + monthly retrain)

This is the answer to *"the model is saved locally — how does the daily workflow find it?"*

### The split

- **Daily prediction** does **not** train. It downloads pre-trained weights from GCS, runs inference, archives the predictions. ~10 minutes per run.
- **Monthly training** trains from scratch on the latest 8 years of OHLCV + features, then **promotes** the new weights to the GCS "current" pointer. Old weights are kept under `models/history/<run_id>/` for rollback.

This is the same model-registry pattern most production ML stacks use, just scaled to a personal project with GCS instead of MLflow/SageMaker/Vertex.

### One-time bootstrap

You already have working models at `outputs/Saved_Models/{Dense,LSTM,Transformer}.pt` plus `outputs/pipeline_metadata.json`. Push them to GCS once:

```bash
# Locally, with GOOGLE_APPLICATION_CREDENTIALS pointing at your service-account JSON
export GCS_BUCKET=<your-bucket-name>
python -m mlops.upload_models
```

That uploads to **two** locations:
- `gs://<bucket>/models/{Dense,LSTM,Transformer}.pt` — the **"current" pointer** that daily-prediction reads
- `gs://<bucket>/models/history/<today>/{Dense,LSTM,Transformer}.pt` — a **versioned snapshot** so you can roll back later

After this one bootstrap, GCS is the source of truth. The `.pt` files in the repo can be deleted (they're 2 MB total — but no need).

### Verify the bootstrap

In GCS Console, check that you have:
```
gs://<bucket>/models/Dense.pt
gs://<bucket>/models/LSTM.pt
gs://<bucket>/models/Transformer.pt
gs://<bucket>/models/pipeline_metadata.json
gs://<bucket>/models/history/2026-05-20/...   (same four files)
```

### How daily-prediction uses it

[`daily-prediction.yml`](.github/workflows/daily-prediction.yml) has a step that runs **before** inference:

```yaml
- name: Download model artifacts from GCS (if cached)
  run: |
    # downloads gs://bucket/models/{Dense,LSTM,Transformer}.pt
    # → outputs/Saved_Models/
```

Then `main.py` loads from `outputs/Saved_Models/` exactly as it does locally. No code changes — the workflow just stages the files for it.

### Monthly retraining

[`monthly-training.yml`](.github/workflows/monthly-training.yml) is wired to run on the **1st of each month at 04:00 UTC**, gated behind a repo variable so it doesn't fail silently:

> **The training script itself is still pending** (ROADMAP P0.6 — extract from `Notebooks/model_2(GPU).ipynb` into `training/train.py`). Until that lands, monthly training is gated **off** and you retrain manually.

**Manual retraining flow (current — until P0.6 ships):**

1. Open `Notebooks/model_2(GPU).ipynb` locally or in Google Colab (Colab GPU is also free)
2. Train against the latest data — produces new `Dense.pt`, `LSTM.pt`, `Transformer.pt`, `pipeline_metadata.json`
3. Copy them to `outputs/Saved_Models/` and `outputs/pipeline_metadata.json`
4. Push to GCS:
   ```bash
   python -m mlops.upload_models --bucket <your-bucket> --run-id 2026-06-01-retrain
   ```
5. The next daily-prediction run picks up the new weights automatically (no code change, no re-deploy)

**Automated retraining flow (after P0.6 ships):**

1. Build `training/train.py` per ROADMAP P0
2. In your GitHub repo: **Settings → Secrets and variables → Actions → Variables → New repository variable** → name `RUN_TRAINING`, value `1`
3. From that month on, the workflow trains and uploads on its own
4. The Slack channel gets `:white_check_mark: monthly-training promoted run 2026-07-01 to current`

### Rolling back a bad retrain

If a new monthly model underperforms in live PnL, roll back without retraining:

```bash
# Restore last month's weights to the "current" pointer
python -m mlops.upload_models \
  --bucket <your-bucket> \
  --source-dir gs://... # (or download history/2026-05-01/* locally first, then run upload_models pointing at that dir)
```

The simpler rollback is one-liner with `gsutil`:

```bash
gsutil -m cp gs://<bucket>/models/history/2026-05-01/* gs://<bucket>/models/
```

Daily prediction will use the rolled-back weights starting from the next run.

### Why training in GitHub Actions is fine (for now)

GitHub Actions hard-caps a job at **6 hours**. Training the current Dense+LSTM+Transformer ensemble on ~8 years × 49 stocks of daily data is well under 2 hours on CPU. The `monthly-training.yml` workflow has `timeout-minutes: 350` (just under the cap) to be safe.

**When you'd outgrow this:**
- Universe expands to NIFTY 500 (10× tickers) — training time grows
- You start running 30-day quantile heads × 90 horizons (per ROADMAP P0.2) — that's a lot of LightGBM models
- You add neural seq2seq with longer training — likely needs a GPU

At that point, run training on **Google Colab (free GPU)** or **Kaggle Notebooks (free GPU)** monthly, save the outputs, run `mlops/upload_models.py` from your laptop. Daily prediction stays in GitHub Actions either way.

---

## Step 6 — Slack notifications (optional but high-leverage)

1. Go to <https://api.slack.com/apps> → **Create New App** → **From scratch**
2. Name it `stock-market`, pick your workspace
3. **Incoming Webhooks** → toggle **On** → **Add New Webhook to Workspace**
4. Pick the channel (e.g., `#stock-alerts`)
5. Copy the webhook URL → save it as the `SLACK_WEBHOOK_URL` GitHub secret

The pipeline already has Slack hooks ([app/services/slack.py](app/services/slack.py)) that fire on workflow failure if the webhook is set.

---

## Step 7 — Verify end-to-end

After your first push:

1. **Check Actions tab** → confirm `tests.yml` ran green
2. **Bootstrap the model registry** (one-time, from your laptop):
   ```bash
   export GCS_BUCKET=<your-bucket>
   export GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account.json
   python -m mlops.upload_models
   ```
   This pushes the four existing model files (`Dense.pt`, `LSTM.pt`, `Transformer.pt`, `pipeline_metadata.json`) to `gs://<bucket>/models/`. Without this, `daily-prediction.yml` has no weights to load.
3. **Manually trigger** `weekly-fundamentals.yml` (Actions → select workflow → Run workflow → main → Run)
4. **Manually trigger** `daily-prediction.yml` to confirm the model download path works end-to-end
5. **Verify Firestore** has one document per ticker under collection `fundamentals`; each document contains the recent quarter map at <https://console.cloud.google.com/firestore>
6. **Verify GCS** has parquet artifacts in your bucket under `archive/<date>/` and model artifacts under `models/`
7. **Check Slack** for the run summary

If anything fails, click the failed workflow → expand the step → look for the error. Common ones:
- `403 Permission denied` on Firestore → service account missing the **Cloud Datastore User** role
- `404 bucket not found` on GCS → bucket name typo in `GCS_BUCKET` secret
- `Reddit OAuth failed` → client ID / secret pasted with whitespace; re-add the secret

---

## Cost guardrails

These limits keep the free tier from accidentally tipping into paid:

1. **Firestore writes:** the weekly fundamentals job writes roughly one doc per ticker. Even if you misconfigure it to run hourly, that's still comfortably under the free write limit.
2. **GCS storage:** monitor in GCP Console → Cloud Storage → bucket → Configuration tab. If your bucket creeps above 4 GB, prune old archive partitions (the parquet files have a `date=YYYY-MM-DD` prefix).
3. **GitHub Actions minutes:** check **Settings → Billing** monthly if you're on a private repo. Each workflow run is reported.
4. **Reddit:** PRAW is rate-limited at 60 req/min; the code stays well within. Don't `time_module.sleep(0)`-tune the existing jobs.

You can set up a **GCP billing budget alert** at $1 to be safe:
- Billing → Budgets & alerts → Create budget → set amount to $1 → alert at 50% / 90% / 100%
- This will *email* you if anything ever costs more than $1; it doesn't auto-shut-off.

---

## Optional: Always-on VM (only if you need the real-time watcher)

If you ever want the Google Sheets watcher running real-time (10-second polling like in production), you can host it free on:

### Option A — Oracle Cloud "Always Free" (most generous)

- 4 ARM Ampere A1 cores + 24 GB RAM, free forever
- Requires credit card for verification (never charged on free tier)
- Setup: <https://www.oracle.com/cloud/free/>
- Run the existing `docker-compose up` on the VM; expose port 8000 via Cloudflare Tunnel (also free) instead of opening firewall ports

### Option B — GCP `e2-micro` (simplest if you're already on GCP)

- 1 shared vCPU + 1 GB RAM, free forever in `us-east1` / `us-west1` / `us-central1`
- Tight on RAM for FinBERT (~440 MB model + PyTorch overhead) — avoid running sentiment on this box; keep sentiment in GitHub Actions
- Setup: Compute Engine → Create instance → machine type **e2-micro** → region `us-east1`
- Same docker-compose deployment as Oracle

Honestly, for a swing-trading workflow at the 1–2 week horizon, **you don't need this.** Skip it unless you find yourself wanting Sheet-change → instant prediction (which won't change your PnL).

---

## What's *not* covered by free tier (future)

When you wire up P3 (Claude as decision-maker), you'll start paying:

- **Anthropic API:** budget ~$0.50–$5/run depending on model tier and stock count. With prompt caching, a daily run on NIFTY 50 stays under $2 with Opus.
- **Claude API key:** add as `ANTHROPIC_API_KEY` GitHub secret when ready.
- **Set the `CLAUDE_MAX_USD_PER_RUN` budget cap** in `.env` (default $2) so a runaway loop can't drain your credits.

Until then, this guide gives you a complete free pipeline through P1.

---

## Local dev (no hosting needed)

If you'd rather develop locally before pushing to GitHub:

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Copy .env.example → .env and fill in secrets
cp .env.example .env  # already done

# 3. Drop the service-account JSON at credentials/
mkdir -p credentials/
# (move your stock-prices-...json here)

# 4. Run a workflow manually
python -m ingestion.fundamentals --no-firestore  # dry run
python -m ingestion.news_ingest
python -m features.sentiment

# 5. Run tests
python -m pytest tests/
```

The same `.env` file works locally — no code changes needed between local and GitHub Actions.

---

## TL;DR runbook

```bash
# One-time setup
1. Enable Firestore + Cloud Storage + IAM API in GCP
2. Add Datastore User + Storage Object Admin to your service account
3. Create a GCS bucket in us-east1
4. Create a Reddit app at reddit.com/prefs/apps
5. Push this repo to GitHub
6. Add 7 secrets in GitHub repo settings (see Step 4)
7. Enable Actions in the repo
8. Bootstrap the model registry:
     python -m mlops.upload_models   # pushes outputs/Saved_Models/* → GCS

# Then it runs itself
- Daily at 11:00 UTC (weekdays): prediction  (downloads model from GCS)
- Daily at 18:00 UTC: sentiment refresh
- Sunday at 02:00 UTC: fundamentals refresh
- 1st of month at 04:00 UTC: training (gated until training/train.py ships)
- On every push: tests
```

You get a live, automated, free pipeline. Failures land in Slack. State lives in Firestore + GCS. You can develop locally with the same code path.
