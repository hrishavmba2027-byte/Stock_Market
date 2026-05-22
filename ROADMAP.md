# Stock Market Pipeline — Review & Roadmap (v2)

## Context

The project ([README.md](README.md)) is a FastAPI + Docker service that today polls a Google Sheet, fetches NSE OHLCV via yfinance, computes 29 technical indicators, and runs a 3-model PyTorch ensemble (Dense/LSTM/Transformer) that predicts *same-day Close*. The intended objective, however, is a **swing-trading strategy with a 1–2 week to 30-day holding window**, with **Claude (via LangChain) as the final decision-maker** producing structured `BUY / SHORT / HOLD` signals with explicit start and end dates.

This v2 plan reflects clarified scope:

1. **n8n is reference-only** ([Stock Prediction Automation.json](Stock%20Prediction%20Automation.json)) — the production orchestrator is **LangChain**, not n8n. The n8n workflow document is kept for trigger-design inspiration only and will not be deployed.
2. **Forecast target:** a **30-day daily price path with uncertainty bands** (quantile regression: q10/q50/q90 at each day from `t+1` to `t+30`).
3. **New data sources:** Reddit (PRAW), X / Twitter (snscrape or nitter front-end scraping), yfinance news, yfinance **quarterly** financials (income statement, balance sheet, cash flow).
4. **Sentiment pipeline:** NLP scoring of social + news text per stock per day, fed as features.
5. **Claude as final decision-maker:** receives ML forecast + uncertainty + sentiment + fundamentals + technical state via LangChain and returns a structured `{signal, entry_date, exit_date, rationale, confidence}`.
6. **Storage redesign:**
   - **Existing Google Sheet** — real daily OHLCV (unchanged, source of truth for prices).
   - **New Google Sheet** (link to be provided) — only the 30-day predicted price path + Claude's trade signal/dates per stock.
   - **Firestore** — latest snapshot of every technical indicator per stock (one document per ticker, overwritten on each run).
   - **Parquet archive** — historical indicators (and predictions) for training, backtesting, and drift monitoring.
7. **Paper-trading + backtest layer** — still in scope as the realized-PnL measurement layer (no live broker).
8. **Capital allocation layer** — Black-Litterman blends Claude's per-stock views with the market-cap prior; a Mean-CVaR convex optimizer then sizes positions from an initial capital `C` to maximize expected return subject to vol/CVaR/concentration constraints. See P4.2.

Output: phased roadmap (P0 → P4). **P1 is implemented** (see status panel below); P0, P2, P3, P4 remain as design.

---

## Implementation status

| Phase | Status | Notes |
|---|---|---|
| P0 — 30-day quantile forecast | ⏳ Pending | Same-day Close target still in place; no `training/` module yet. |
| P1 — Data sources + sentiment | ✅ Done | `ingestion/`, `features/sentiment.py`, `features/cross_sectional.py`. 33 new tests, 48 total passing. Package named `ingestion/` (not `data/`) due to macOS case-insensitive FS collision with existing `Data/` folder. |
| P2 — Storage redesign | ⏳ Partial | **Firestore writer for quarterly fundamentals landed** ([ingestion/fundamentals.py](ingestion/fundamentals.py) — collection `fundamentals`, doc id `{TICKER}`, recent quarters nested under `quarters`). General indicator-snapshot Firestore client, Parquet archive, and new predictions sheet still pending. |
| P3 — Claude decision layer | ⏳ Pending | `app/langchain/chain.py` still deterministic-only. |
| P4 — Backtest + portfolio + paper | ⏳ Pending | No backtester, optimizer, or paper-trading layer yet. |

---

## Part 1 — Current pipeline (snapshot of what exists today)

### 1.1 Data ingestion — [Data_update.py](Data_update.py)
- `yfinance` daily bars, NSE `.NS` tickers, 50 NIFTY constituents driven by Google Sheet worksheet names.
- Incremental fetch from `last Date_str + 1`, 4 PM IST cutoff via `safe_daily_end_date()` ([Data_update.py:201-205](Data_update.py#L201-L205)).
- Excel mirror at `Data/nse_stock_data.xlsx` (49 MB).

### 1.2 Feature engineering — [Feature_Engineering.py](Feature_Engineering.py)
29 indicators (SMA/EMA, RSI_14, MACD, Stoch, ATR, ADX, Bollinger, Williams %R, CCI, OBV, VWAP, MFI, daily/log returns).

### 1.3 Model & inference — [main.py](main.py)
- Target: `Close[t]`, regression. `seq_len=20` of 29 features → `Close[t]`. **No forward lookahead.**
- 3 PyTorch models: `DenseModel` ([main.py:95-112](main.py#L95-L112)), `LSTMModel` ([main.py:115-134](main.py#L115-L134)), `TransformerModel` ([main.py:137-176](main.py#L137-L176)).
- Weighted ensemble 0.2 / 0.3 / 0.5 from [outputs/pipeline_metadata.json](outputs/pipeline_metadata.json).
- Optional `inject_intraday_open` ([main.py:598-624](main.py#L598-L624)) — **must be removed**; it leaks the target.

### 1.4 Orchestration — `app/`
FastAPI (`/health`, `/status`, `/run`), watcher polling Google Sheets every 10 s ([app/watcher/service.py](app/watcher/service.py)), deterministic LangChain router ([app/langchain/chain.py](app/langchain/chain.py)), subprocess runner ([app/services/subprocess_runner.py](app/services/subprocess_runner.py)), Slack alerts ([app/services/slack.py](app/services/slack.py)). Two Docker services in [docker-compose.yml](docker-compose.yml). n8n + ngrok is **deprecated** in this plan.

### 1.5 Critical issues (still valid)
1. Same-day target ≠ swing-trade objective.
2. Reported test R² = −0.367 on scaled features; the 0.907 unscaled R² is misleading due to price autocorrelation.
3. Training notebooks (`Notebooks/model_2(GPU).ipynb`) are not in version-controlled training code — train/val/test split cannot be audited.
4. No backtest, no cost model, no risk layer.
5. Each stock modeled in isolation (no cross-sectional or regime signal).

---

## Part 2 — Target architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA SOURCES                                  │
│  yfinance OHLCV   yfinance quarterly fin.   yfinance news                │
│  Reddit (PRAW)    X (snscrape / nitter)                                  │
└──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
           ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       INGESTION & FEATURE LAYER                          │
│  Data_update.py (existing) ─► OHLCV ─► Feature_Engineering.py (29 ind.) │
│  ingestion/fundamentals.py  ─► quarterly P&L, BS, CF, ratios            │
│  ingestion/{news,reddit,x}_ingest.py ─► yfinance news + Reddit + X      │
│  features/sentiment.py      ─► FinBERT scoring per (ticker, date)       │
│  features/cross_sectional.py ─► NIFTY/VIX regime + ranks + rel-strength │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            STORAGE LAYER                                 │
│  Google Sheet (real OHLCV)  ─ source of truth, unchanged                 │
│  Firestore                  ─ latest indicator snapshot per ticker       │
│  Parquet archive            ─ historical indicators + sentiment + preds  │
│  Google Sheet (NEW)         ─ 30-day predicted path + Claude signal     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODELING LAYER                                   │
│  training/train.py   ─► quantile multi-horizon model                    │
│  produces q10/q50/q90 forecasts for t+1 … t+30                          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  LANGCHAIN DECISION LAYER (Claude)                       │
│  Inputs:  forecast path + bands, sentiment, fundamentals, indicators    │
│  Output:  {signal: BUY|SHORT|HOLD, entry_date, exit_date,               │
│            target_price, stop_loss, confidence, rationale}              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         BACKTEST + PAPER TRADING + MONITORING                            │
│  backtest/engine.py   paper_trading/   monitoring/drift.py              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3 — Phased roadmap

Five phases. Each is independently shippable. Dependencies: P0 must precede P3 and P4; P1 and P2 can run in parallel after P0.

---

### P0 — Reframe the model for a 30-day daily-path forecast

**Goal:** replace same-day Close regression with a leakage-free multi-horizon quantile forecaster that predicts the daily price path for `t+1 … t+30` with q10 / q50 / q90 bands.

**P0.1 — Label construction** ([Feature_Engineering.py](Feature_Engineering.py))
- For each row at time `t`, append 30 forward-return labels: `y_h = log(Close[t+h] / Close[t])` for `h ∈ {1, 2, …, 30}`.
- Last 30 rows per stock have no labels → exclude from training, keep for live inference only.
- Add explicit `has_labels` boolean column to make leakage bugs visible in code review.
- **Remove `inject_intraday_open`** ([main.py:598-624](main.py#L598-L624)) — it leaks `Open[t]` into the input window when the target is `Close[t]` or any forward value.

**P0.2 — Multi-horizon quantile model** (new: `training/model.py`)
- Three quantiles × 30 horizons = 90 output heads per stock.
- **Recommended architecture (start here):** **LightGBM** with pinball loss, one model per `(quantile, horizon)` — 90 models total, but each trains in seconds, parallelizable, gives feature importances. This is the practical workhorse for tabular financial features and almost always beats neural nets at the daily-bar horizon.
- **Alternative (after LightGBM baseline):** a single neural seq2seq with 90-output head + pinball loss. Use the existing Transformer scaffold ([main.py:137-176](main.py#L137-L176)) as a starting point but replace the single regression head with a `(3, 30)` output and pinball loss.
- Enforce non-crossing quantiles via post-hoc sorting per horizon.

**P0.3 — Leakage-free walk-forward CV** (new: `training/cv.py`)
- Time-ordered split with **embargo = 30 trading days** between train and validation to prevent label overlap (since labels extend 30 days into the future).
- Rolling window: 3-year train → 6-month validation → 6-month test, step 6 months.
- Scalers fit on the train fold only, saved per fold.

**P0.4 — Honest metrics** (new: `training/metrics.py`)
- Per horizon: pinball loss, MAE on returns, **Spearman IC** (cross-sectional rank correlation between predicted q50 and realized return).
- Calibration: empirical coverage of q10–q90 band (target 80%; if you see 40% your uncertainty is broken).
- Decile lift chart at the **terminal horizon (t+30)** and at the **canonical mid-horizon (t+10)**.

**P0.5 — Baselines before fancy models**
- Naive zero return at every horizon.
- ARIMA(1,1,0) per stock — surprisingly hard to beat at multi-week horizons.
- Ridge regression on the 29 features × 30 horizons.
- LSTM/Transformer must beat **ridge IC** on validation; if not, the architecture is not the bottleneck.

**P0.6 — Promote training out of notebooks** (new: `training/train.py`)
- CLI: `python -m training.train --horizons 1-30 --start 2017-01-01 --end 2024-06-30 --model lightgbm`.
- Versioned artifacts at `outputs/Saved_Models/<run_id>/`, plus a `current` symlink.

**Files touched:** [Feature_Engineering.py](Feature_Engineering.py), [main.py](main.py) (inference path rewrite for 90-output target), new `training/` package, [outputs/pipeline_metadata.json](outputs/pipeline_metadata.json) (schema bump: `horizons`, `quantiles`, `target_type`).

**Verification:** Validation Spearman IC at `h=10` > 0.04 (a defensible floor for daily-bar NSE swing models); IC stays positive but decaying out to `h=30`; q10–q90 empirical coverage between 75% and 85% across all horizons; ridge baseline reported in the same table.

---

### P1 — New data sources and sentiment ✅ Implemented

**Goal:** add the four new data layers (quarterly fundamentals, yfinance news, Reddit, X) and a sentiment pipeline that emits per-(ticker, date) features.

> **Implementation note:** the new package was named `ingestion/` rather than `data/`. macOS APFS is case-insensitive, so a lowercase `data/` Python package collides with the existing uppercase `Data/` OHLCV directory (the one that holds `nse_stock_data.xlsx`). Python's import resolution is case-sensitive even on a case-insensitive filesystem, so `import data` would fail. `ingestion/` keeps the new Python package and the existing `Data/` parquet/Excel storage cleanly separated.

**P1.1 — Quarterly fundamentals** (new: `ingestion/fundamentals.py`)
- Per ticker via `yfinance.Ticker(t)`:
  - `quarterly_financials` (income statement), `quarterly_balance_sheet`, `quarterly_cashflow`.
  - `info` for trailing ratios (P/E, P/B, ROE, debt/equity, market cap).
  - `earnings_dates` for the next reporting date.
- Derive features that are stable for ~90 days between reports: revenue YoY, EPS YoY, operating margin, ROE, debt/equity, FCF yield, P/E percentile vs sector.
- **Event features:** `days_to_earnings`, `in_earnings_window_5d`. At a 1–4 week horizon, earnings is the single biggest source of large moves — feature it both as a signal and as a *risk filter* (consider blocking new entries 3 days before earnings).
- **Two output sinks** (both written by a single refresh job):
  1. `Data/fundamentals.parquet` — flat one-row-per-ticker snapshot consumed by the ML feature pipeline.
  2. **Firestore collection `fundamentals`** — one document per ticker containing the *last 4 quarters*. Doc id = `{TICKER}` (e.g. `RELIANCE`). Schema:
     ```
     {
       "company_name": "Reliance Industries",
       "scrape_date":  "2026-05-20",
       "ticker":       "RELIANCE",
       "quarters": {
         "2025Q3": {
           "quarter_end_date": "2025-09-30",
           "financials": { revenue, net_income, operating_income, gross_profit,
                           operating_margin, net_margin, gross_margin,
                           total_equity, total_debt, total_assets,
                           debt_to_equity, roe, asset_turnover,
                           operating_cashflow, capex, free_cash_flow }
         }
       }
     }
     ```
     Idempotent: re-running the job overwrites the same company doc; the parquet archive holds retry staging.
- Refresh weekly. Disable Firestore writes for local dry-runs with `python -m ingestion.fundamentals --no-firestore`.

**P1.2 — News ingestion** (new: `ingestion/news_ingest.py`)
- yfinance: `Ticker(t).news` returns recent headlines with `providerPublishTime` and `link`. Store `{ticker, ts, title, source, url}`.
- Daily pull job; dedupe by URL hash; store in `Data/news.parquet`.

**P1.3 — Reddit ingestion** (new: `ingestion/reddit_ingest.py`)
- **PRAW** (Reddit API, free with personal-use registration). Credentials via `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` env vars (wired into [app/config/settings.py](app/config/settings.py)).
- Subreddits: `r/IndianStockMarket`, `r/IndiaInvestments`, `r/StockMarketIndia`, `r/DalalStreetTalks`, `r/Nifty50`, plus optionally `r/wallstreetbets` (rarely covers Indian names but useful for global risk sentiment).
- Match posts/comments to tickers by case-insensitive symbol regex (e.g. `\b(RELIANCE|TCS|INFY)\b`) and aliases (`Reliance Industries`, `Tata Consultancy`). Alias map at [ingestion/ticker_aliases.json](ingestion/ticker_aliases.json), regex compiled in [ingestion/aliases.py](ingestion/aliases.py) with word-boundary guards.
- Store `{ticker, ts, subreddit, title, body, score, num_comments, permalink}`.

**P1.4 — X / Twitter ingestion** (new: `ingestion/x_ingest.py`)
- **snscrape** as the primary path; **nitter** front-end HTML as fallback.
- **Caveats:** snscrape breaks every few months when X changes anti-scraping defenses; nitter instances die regularly. Engineer this as a *best-effort* feature — predictions must still work if X data is missing. Add a `x_available` boolean to the feature row and let the model learn the missing case.
- Query patterns: `$RELIANCE`, `#RELIANCE`, plus alias terms. Pre-filter spam (link-only tweets, low-author-followers, bot accounts via heuristics).
- Rate-limit yourself; cache aggressively to `Data/x_posts.parquet`.

**P1.5 — Sentiment scoring** (new: `features/sentiment.py`)
- **Model: FinBERT** (`ProsusAI/finbert` on HuggingFace) — finance-specific, runs on CPU, no API cost.
- For each text: probabilities over `{positive, neutral, negative}`. Aggregate per `(ticker, date)`:
  - `sent_mean_3d`, `sent_mean_7d` — exponentially weighted by post score / engagement.
  - `sent_volume_z` — z-score of post count vs the ticker's own 90-day baseline (volume spike often matters more than polarity).
  - `sent_pos_share`, `sent_neg_share`.
  - Per source: separate features for `news_*`, `reddit_*`, `x_*` (the three channels behave very differently — news is lagging, Reddit is contrarian, X is noisy but fast).
- Do **not** use Claude for sentiment scoring: per-post API cost will dominate at this volume and FinBERT is good enough. Claude is reserved for the final decision.

**P1.6 — Cross-sectional and regime features** (new: `features/cross_sectional.py`)

Implemented as a *separate module* rather than splicing into [Feature_Engineering.py](Feature_Engineering.py) — cross-sectional features need the *universe* of tickers at once, whereas `Feature_Engineering.compute_indicators` only sees one ticker at a time. The new module consumes `{ticker: DataFrame}` already enriched with per-ticker indicators and adds regime + rank + calendar columns.

- NIFTY 50 return + 20-day vol; INDIAVIX level + delta; sector index returns (cached at [Data/indices.parquet](Data/)).
- Relative strength: stock 20-day return minus NIFTY 20-day return; rolling 60-day beta.
- Cross-sectional rank: percentile of each indicator within the universe per date.
- Calendar: day-of-week, days-to-month-end, `days_to_earnings`.
- Missing-column guard: when index history is unavailable (fresh checkout, CI), regime features fall back to NaN rather than raising.

**Verification:**
- For 5 sample stocks × 5 sample dates, manually verify sentiment aggregation matches a hand computation.
- Fundamentals refresh job runs <60 s for full universe; second run hits cache >95%.
- Coverage report: % of (ticker, date) rows that have non-null sentiment per source. Target ≥ 70% for news/Reddit; X coverage will be lower (expected).

**Tests landed:** 33 new pytest cases under [tests/](tests/) — alias word-boundary safety, news shape handling (new vs legacy yfinance), Reddit ticker extraction, X spam filters + query builder, sentiment aggregation, fundamentals helpers + earnings-date selection, regime + rank normalization. Full suite passes (48 total).

**How to run end-to-end:**
1. `pip install -r requirements.txt`
2. `export REDDIT_CLIENT_ID=… REDDIT_CLIENT_SECRET=…`
3. `python -m ingestion.fundamentals` → `Data/fundamentals.parquet`
4. `python -m ingestion.news_ingest` → `Data/news.parquet`
5. `python -m ingestion.reddit_ingest` → `Data/reddit_posts.parquet`
6. `python -m ingestion.x_ingest` → `Data/x_posts.parquet` (best-effort)
7. `python -m features.sentiment` → `Data/sentiment_features.parquet`
8. `python -m features.cross_sectional` → `Data/indices.parquet` (then call `add_cross_sectional_features` from the training pipeline)

---

### P2 — Storage redesign

**Goal:** replace the Excel + single-sheet model with the four-tier storage described in Part 2.

**P2.1 — Firestore: latest indicator snapshot per stock** (new: `storage/firestore_client.py`)
- Google Cloud Firestore, native mode.
- Schema:
  - Collection `indicators_latest`, document id = ticker (e.g. `RELIANCE`).
  - Fields: every indicator (`RSI_14`, `MACD_12_26`, `ATR_14`, …, `sent_mean_7d`, …) as a top-level numeric field, plus `as_of_date`, `updated_at`.
  - One document per stock, **overwritten** on every pipeline run — no history kept here (history lives in Parquet, see P2.2).
- Auth: reuse the existing Google service account pattern (already used for Sheets at [app/services/google_sheets.py](app/services/google_sheets.py)); enable Firestore API on the same GCP project. Add `google-cloud-firestore` to [requirements.txt](requirements.txt).
- Idempotent: a re-run of the same date must produce the same document (no append, no drift).

**P2.2 — Parquet historical archive** (new: `storage/parquet_archive.py`)
- `Data/archive/indicators.parquet` — partition by year/month, columns: `ticker`, `date`, every indicator, every sentiment feature.
- `Data/archive/predictions.parquet` — every model run's q10/q50/q90 path with `run_id`, `model_version`, `as_of_date`.
- `Data/archive/decisions.parquet` — every Claude decision with full input/output for audit.
- Use `pyarrow` or `polars` for fast writes; partitioned by month for cheap queries.

**P2.3 — New Google Sheet for predicted prices + Claude signals** (new: `app/services/predictions_sheet.py`)
- Separate sheet ID, supplied by user via `PREDICTIONS_SHEET_ID` env var.
- One worksheet per stock (mirrors the existing real-data sheet structure).
- Columns: `as_of_date`, `h` (1…30), `predicted_date`, `pred_q10`, `pred_q50`, `pred_q90`, plus a summary block at the top for that run: `signal`, `entry_date`, `exit_date`, `target_price`, `stop_loss`, `confidence`, `rationale_short`.
- Writer reuses retry/backoff patterns from [Data_update.py:326-338](Data_update.py#L326-L338) and the gspread client at [app/services/google_sheets.py](app/services/google_sheets.py).
- **Overwrite-on-rerun** semantics — the sheet shows the *latest* predictions; full audit history lives in Parquet.

**P2.4 — Deprecate the local Excel mirror**
- `Data/nse_stock_data.xlsx` (49 MB) is replaced as a runtime artifact by Parquet. The Google Sheet remains the OHLCV source of truth. Keep the Excel file for one release as a fallback, then delete.

**Files touched:** new `storage/` package; [requirements.txt](requirements.txt) (`google-cloud-firestore`, `pyarrow`, `polars`); [app/config/settings.py](app/config/settings.py) (`PREDICTIONS_SHEET_ID`, `FIRESTORE_PROJECT`).

**Verification:**
- After a pipeline run, Firestore has exactly one `indicators_latest/<TICKER>` document per stock in the universe, and `as_of_date` equals the run date.
- `pq.read_table('Data/archive/indicators.parquet').to_pandas()` round-trips to a frame identical to the in-memory features.
- The new predictions sheet has 30 rows per stock (h=1..30) plus the summary header block.

---

### P3 — Claude as final decision-maker (LangChain)

**Goal:** wire LangChain so Claude receives a structured snapshot for one stock and returns a typed trading decision. This replaces the deterministic `app/langchain/chain.py` router for *decisions* — the router still handles workflow orchestration.

**P3.1 — Decision schema** (new: `app/langchain/decision_schema.py`)
```python
class TradeDecision(BaseModel):
    ticker: str
    signal: Literal["BUY", "SHORT", "HOLD"]
    entry_date: date          # must be within [t+1, t+5]
    exit_date: date           # must be within [entry_date+5, t+30]
    target_price: float
    stop_loss: float
    confidence: float         # 0..1
    rationale: str            # <= 300 words
    risk_flags: list[str]     # e.g. ["earnings_within_5d", "high_iv", "wide_q10_q90_band"]
```
Validated via Pydantic. Any output that fails validation triggers one retry with a corrective system message, then defaults to `HOLD`.

**P3.2 — Prompt builder** (new: `app/langchain/decision_prompt.py`)
- **System prompt (cached):** trading-rules constitution — definitions of BUY/SHORT/HOLD, risk policy, holding-window bounds, output schema, examples of good and bad reasoning. This block is large and reused on every call → place it under `cache_control: ephemeral` (Anthropic prompt caching) so the 5-minute cache covers the burst of decisions per pipeline run.
- **Per-stock context (uncached):**
  - 30-day q10/q50/q90 path as a compact table.
  - Current technical state from Firestore (snapshot of all indicators).
  - Last 4 quarters of fundamentals + next earnings date.
  - 7-day sentiment summary (news / Reddit / X separately) with 3 top headlines verbatim.
  - Index regime (NIFTY 20-day return, VIX level).
  - Historical realized outcomes for this stock's last N similar setups (retrieval — see P3.4).

**P3.3 — LangChain wiring** (extend [app/langchain/chain.py](app/langchain/chain.py))
- Use `langchain-anthropic` `ChatAnthropic(model="claude-opus-4-7")` for high-stakes decisions; fall back to `claude-haiku-4-5` for bulk runs / dry-runs to control cost.
- Structured output via `.with_structured_output(TradeDecision)` (LangChain handles JSON-mode + schema validation).
- Wrap in a `RunnableWithRetry` for transient API errors.
- Batch via `RunnableParallel` — one decision per stock in the universe, run in a single async fan-out.

**P3.4 — Retrieval of analogous historical setups** (new: `app/langchain/retrieval.py`)
- Optional but high-leverage: for each (ticker, date), find the 5 most similar historical setups by cosine similarity over a feature vector `[RSI_14, MACD_hist, ATR_14, 20d_return, sent_mean_7d, days_to_earnings]`. Look up their realized 30-day returns from Parquet and feed as a few-shot context block.
- Built on the Parquet archive (P2.2); no vector DB needed at this scale — `numpy` brute-force across <100k rows is sub-second.

**P3.5 — Cost & safety guardrails**
- Per-run budget cap (env var `CLAUDE_MAX_USD_PER_RUN`, default $2). Pre-estimate token cost, abort if exceeded.
- Dry-run mode that returns the prompt without calling the API — used in tests and CI.
- Log full `{prompt, response, parsed_decision, cost_usd}` to `Data/archive/decisions.parquet`.

**Files touched:** new `app/langchain/decision_schema.py`, `decision_prompt.py`, `retrieval.py`; extend [app/langchain/chain.py](app/langchain/chain.py); [requirements.txt](requirements.txt) (`langchain-anthropic`, `anthropic`).

**Verification:**
- 100% of decisions parse to a valid `TradeDecision` (validation gate is enforced).
- `exit_date - entry_date` is always within `[5, 30]` trading days.
- Prompt-cache hit rate > 70% after the first call of a batch (system block ≥ 1024 tokens).
- On a 100-stock dry-run, total cost < $1 (Haiku) or < $5 (Opus) with prompt caching enabled.

---

### P4 — Backtest, paper trading, monitoring

**Goal:** measure whether Claude's decisions actually make money after costs, and detect drift.

**P4.1 — Backtester** (new: `backtest/engine.py`)
- Replay: for each historical date, reconstruct what Claude would have seen (forecasts + fundamentals + sentiment as of that date) and re-run the decision (using cached Claude responses when available, or a deterministic surrogate model in fast-mode).
- Cost model: 15 bps slippage + Zerodha brokerage (₹20 or 0.03%, whichever lower) + STT 0.025% (sell side) + exchange + GST. Round-trip ≈ 50–80 bps on liquid NSE names.
- Honour the entry/exit dates Claude returned (no peeking).
- Survivorship-bias guard: train and backtest on historical NIFTY 50 *membership at each date*, not today's.
- Outputs: equity curve, Sharpe, Sortino, max DD, turnover, hit rate, avg holding days, per-signal-type PnL.

**P4.2 — Capital allocation & portfolio optimization** (new: `portfolio/`)

**Goal:** given an initial capital `C` (env `INITIAL_CAPITAL`, default ₹10,00,000), allocate it across the per-stock views Claude produced to **maximize expected return subject to explicit risk constraints**. This replaces ad-hoc top-K + equal-weight sizing with a proper convex optimization.

**P4.2.1 — Optimizer inputs (translating Claude's output)**

For each stock `i` in the universe on rebalance date `t`:
- **Expected return `μ_i`:** Claude's `target_price / current_price − 1`, *scaled by `confidence_i`* to shrink low-conviction views toward the prior. Sign flips for `SHORT`; `HOLD` is excluded.
- **View uncertainty `Ω_ii`:** width of the q90–q10 band from the forecast — wider band → higher view variance → less weight in the Bayesian blend (P4.2.2).
- **Covariance matrix `Σ`:** estimated from trailing 252-day log returns with **Ledoit-Wolf shrinkage** (raw sample covariance is hopelessly noisy at 50×50). Use `sklearn.covariance.LedoitWolf`.
- **Liquidity floor:** 20-day avg traded value > ₹50 cr (filter, not constraint).
- **Hard exclusions:** stocks with earnings in the next 3 days (uses P1.1).

**P4.2.2 — Optimization model: Black-Litterman → Mean-CVaR**

Two-stage. Black-Litterman gives stable expected returns; CVaR gives tail-aware sizing.

- **Stage 1 — Black-Litterman expected returns.**
  - Prior `π` = reverse-optimized from NIFTY 50 market-cap weights at risk aversion `δ = 2.5` (Idzorek convention).
  - Views matrix `P` is the identity on stocks Claude has a view on; view vector `Q = μ_Claude`; view-uncertainty `Ω = diag((band_width_i)²)`.
  - Posterior expected return `μ_BL = ((τΣ)⁻¹ + Pᵀ Ω⁻¹ P)⁻¹ ((τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q)` with `τ = 0.05`.
  - Stocks with no Claude view inherit the prior — they stay available for diversification without being forced to zero.

- **Stage 2 — Mean-CVaR optimization** (the actual allocation).
  - Maximize `wᵀ μ_BL − λ · CVaR_95(−wᵀr)` subject to constraints below.
  - **CVaR over variance** because at 1–4 week horizons return distributions are fat-tailed and asymmetric; variance penalizes upside and downside equally, CVaR only the downside.
  - `λ` (risk aversion) is calibrated so the historical *ex-post* annualized vol of the portfolio ≈ user-configured `TARGET_VOL` (default 15%).
  - Convex LP/QP solved via `cvxpy` with the `ECOS` or `CLARABEL` backend.

- **Alternatives considered (documented, not used by default):**
  - Hierarchical Risk Parity (HRP, López de Prado) — robust, no expected returns needed, but discards Claude's signal entirely. Keep as a *baseline* in the metrics table.
  - Kelly fractional — maximizes log-growth but overfits sample µ; if used, cap at quarter-Kelly per position.
  - Plain mean-variance — sensitive to µ noise, produces corner solutions; skip.

**P4.2.3 — Constraints (encoded as cvxpy constraints)**

- Capital: `Σ |w_i| · C ≤ C` (no leverage in v1).
- Long-only by default: `w_i ≥ 0`. Long-short variant flips this to `−w_max ≤ w_i ≤ w_max` for `SHORT` candidates.
- Per-position cap: `w_i ≤ 0.20` (max 20% in any one stock).
- Sector cap: `Σ_{i∈sector_s} w_i ≤ 0.30` for every sector `s`.
- Min position size: `w_i ≥ 0.02` if `w_i > 0` (avoids dust positions where round-trip brokerage exceeds expected edge; modelled as a mixed-integer constraint or solved via a two-pass relaxation).
- Turnover cap: `Σ |w_i − w_i,prev| ≤ 0.50` per rebalance (caps cost drag).
- Cash floor: `1 − Σ |w_i| ≥ 0.05` (5% buffer for fills + margin moves).

**P4.2.4 — Risk overlays (run *after* the optimizer)**

- Per-position ATR stop-loss (2× ATR) and time-stop at Claude's `exit_date`.
- Portfolio kill-switch: if 5-day drawdown > 8%, freeze new entries; existing positions continue to time-stop.
- Diversification floor: reject any solution where the *realized* 60-day pairwise return correlation between any two held positions exceeds 0.85 (prevents the "5 IT stocks during an IT rally" failure mode). If violated, drop the lower-confidence position and re-solve.

**P4.2.5 — Rebalancing cadence**

- **Weekly** (Monday open) for new entries; daily for exits (stops + Claude's `exit_date`). Weekly cadence matches the 1–4 week holding window and keeps turnover (and cost drag) manageable.
- Costs modeled inside the optimizer's turnover term: 50 bps round-trip on the changed weight (matches the cost model in P4.1).

**P4.2.6 — Implementation libraries**

- `PyPortfolioOpt` (`pypfopt`) — has built-in Black-Litterman (`black_litterman.BlackLittermanModel`) and CVaR (`efficient_frontier.EfficientCVaR`). Fastest path to a working version.
- `riskfolio-lib` — more advanced (EVaR, robust optimization, hierarchical methods). Use if pypfopt becomes limiting.
- `cvxpy` — for the custom mixed-integer + turnover constraints not covered by the libraries.
- All three are pure Python, no external solver licenses needed (ECOS / CLARABEL ship with cvxpy).

**Files:** new `portfolio/optimizer.py` (Black-Litterman + CVaR solver), `portfolio/constraints.py` (constraint builders), `portfolio/risk_overlay.py` (stops + kill-switch + diversification floor), `portfolio/allocate.py` (top-level entrypoint: takes Claude decisions + Σ + `C`, returns `{ticker: shares, ticker: notional}` allocation). Add `pypfopt`, `cvxpy`, `riskfolio-lib` to [requirements.txt](requirements.txt).

**Verification:**
- Solver returns a feasible solution for 50 stocks in < 2 s on a laptop.
- All hard constraints satisfied (assertion suite over weights post-solve).
- Backtest of the optimizer vs naive baselines (equal-weight top-5, HRP, inverse-vol) over 2022–2024 — target: Sharpe uplift ≥ 0.3 over the best baseline, max DD ≤ baseline's, turnover within cap.
- Sensitivity: perturb Claude's `μ` by ±20% and confirm portfolio weights do not flip more than 30% (stability check — if they do, view uncertainty `Ω` is too low and the optimizer is overfitting to noisy views).
- Edge cases: when Claude returns `HOLD` for every stock, optimizer falls back to cash; when only one BUY exists, single-position cap is respected (allocation ≤ 20% even if signal is huge).

**P4.3 — Paper trading** (new: `paper_trading/`)
- Daily job: read Claude's decisions, simulate fills at next-day open with slippage, persist positions in a `paper_portfolio` Firestore collection (latest) + Parquet (history).
- EOD mark-to-market, post a Slack summary using the existing webhook ([app/services/slack.py](app/services/slack.py)).

**P4.4 — Drift monitoring** (new: `monitoring/`)
- For each prediction, after `t+30` compute realized return at every horizon and compare to q50; track rolling 30-day Spearman IC at `h=10` and `h=30`.
- Track empirical coverage of the q10–q90 band — if it falls below 60%, model uncertainty is broken.
- Alert (Slack) when live IC drops below 50% of validation IC for 10 consecutive trading days.

**Verification:**
- 3-year out-of-sample backtest: Sharpe > 1.0 net of costs, max DD < 20%, IC stable across years.
- Paper trading for 4 weeks: daily PnL within 1σ of backtest's daily-PnL distribution.
- Synthetic-drift test: shuffle features in production traffic and confirm drift alert fires within 10 trading days.

---

## Part 4 — Extra accuracy / profit suggestions (point 7)

Beyond the phased work above, these are the highest-leverage improvements I'd consider, ordered by expected payoff per unit of effort.

1. **Sector + regime conditioning.** Train one model but pass sector one-hot + regime indicator (`vix_high/low`, `nifty_trend_up/down`) as features. Returns behave very differently across regimes at the 30-day horizon, and a single unconditional model averages them out.
2. **Volatility-of-volatility features.** `realized_vol_5d / realized_vol_60d` and `iv_atm` (option implied vol — available via NSE option chain scrape) are strong predictors of mean reversion vs continuation. Often the single biggest non-price feature for swing trading.
3. **Delivery % and FII/DII flows.** Daily NSE bhavcopy publishes delivery quantity per stock. High delivery on a green day is institutional accumulation — one of the cleanest swing signals on NSE specifically. Add as a feature; refresh daily from NSE.
4. **LightGBM before deep learning.** I'll repeat this because it matters: for tabular financial features at daily bars, gradient-boosted trees almost always beat LSTMs and Transformers, train in seconds, give feature importances, and are easier to debug. Use the neural ensemble only after LightGBM saturates.
5. **Pair / spread models.** Instead of predicting absolute return, predict `return(stock) − return(sector_index)`. The cross-sectional target has higher signal-to-noise because market-wide moves are removed.
6. **Conformal prediction for calibrated bands.** Quantile regression bands are usually mis-calibrated. Wrap the model in split conformal prediction on a held-out calibration fold to get bands with guaranteed empirical coverage. Cheap, well-supported by the `mapie` library.
7. **Position-aware loss.** Train with an asymmetric loss that penalizes the *direction-and-magnitude* error weighted by realized PnL after a fixed sizing rule — directly optimizes for trading utility rather than RMSE. Implementable as a custom LightGBM objective.
8. **Earnings-driven event model.** A separate model specifically for the 5 days around earnings — completely different feature set (surprise %, consensus revisions, guidance text sentiment). The mean model is bad here and degrades overall metrics if not separated.
9. **Options-implied features as labels.** Use 30-day at-the-money implied vol and 25-delta skew as auxiliary targets in a multi-task setup. The market already prices its 30-day expectations in options — borrowing that as a target acts as a strong regularizer.
10. **Walk-forward retraining cadence.** Retrain monthly on a rolling window, not once-and-deploy. Drift is real; retraining is the cheapest defense.
11. **Claude tool-use for arithmetic.** When Claude reasons about target/stop prices, expose a `compute_atr_stop(ticker, multiple)` tool — much more reliable than asking the model to multiply by 2 in its head. LangChain handles tool routing.
12. **Diversification floor.** A simple rule: cap portfolio correlation by rejecting candidates whose 60-day return correlation with already-selected positions exceeds 0.7. Eliminates the failure mode where the model picks 5 IT-sector stocks during an IT rally.

---

## Part 5 — Critical files / modules touched

| Phase | Path | Action |
|---|---|---|
| P0 | [Feature_Engineering.py](Feature_Engineering.py) | Add 30 forward-return labels, `has_labels` flag |
| P0 | [main.py](main.py) | Rewrite inference for multi-horizon quantile outputs; **delete** `inject_intraday_open` |
| P0 | `training/` (new) | `train.py`, `cv.py`, `metrics.py`, `model.py`, `baselines.py` |
| P0 | [outputs/pipeline_metadata.json](outputs/pipeline_metadata.json) | Schema: `horizons`, `quantiles`, `target_type` |
| P1 ✅ | `ingestion/` (new) | `fundamentals.py`, `news_ingest.py`, `reddit_ingest.py`, `x_ingest.py`, `aliases.py` |
| P1 ✅ | `features/sentiment.py` (new) | FinBERT scoring + per-source aggregation |
| P1 ✅ | `features/cross_sectional.py` (new) | NIFTY/VIX regime, rel-strength, beta, cross-sectional ranks, calendar |
| P1 ✅ | `ingestion/ticker_aliases.json` (new) | Alias map for social-text ticker matching (49 NIFTY 50 names) |
| P2 | `storage/` (new) | `firestore_client.py`, `parquet_archive.py` |
| P2 | `app/services/predictions_sheet.py` (new) | Writer for the second Google Sheet |
| P2 | [app/config/settings.py](app/config/settings.py) | Add `PREDICTIONS_SHEET_ID`, `FIRESTORE_PROJECT`, `CLAUDE_API_KEY`, `CLAUDE_MAX_USD_PER_RUN` |
| P3 | `app/langchain/decision_schema.py` (new) | Pydantic `TradeDecision` |
| P3 | `app/langchain/decision_prompt.py` (new) | System constitution + per-stock context |
| P3 | `app/langchain/retrieval.py` (new) | Analogous-setup retrieval over Parquet |
| P3 | [app/langchain/chain.py](app/langchain/chain.py) | Add Claude decision step; retain deterministic router for orchestration |
| P4 | `backtest/engine.py` (new) | Walk-forward backtester + NSE cost model |
| P4 | `portfolio/` (new) | `allocate.py`, `optimizer.py` (Black-Litterman + Mean-CVaR), `constraints.py`, `risk_overlay.py` |
| P4 | `paper_trading/` (new) | Paper broker + EOD Slack summary |
| P4 | `monitoring/` (new) | Live IC + coverage tracking, drift alerts |
| — | [requirements.txt](requirements.txt) | **Added in P1:** `pyarrow`, `praw`, `snscrape`, `beautifulsoup4`, `transformers`, `sentencepiece`, `google-cloud-firestore`. **Pending:** `lightgbm` (P0), `langchain-anthropic`, `anthropic` (P3), `polars`, `mapie` (P2), `pypfopt`, `cvxpy`, `riskfolio-lib` (P4). |
| — | n8n | **Deprecate** [Stock Prediction Automation.json](Stock%20Prediction%20Automation.json); keep file as reference, remove from deployment docs |

**Reuse (don't reinvent):**
- [app/services/subprocess_runner.py](app/services/subprocess_runner.py) — retry + JSON-parse pattern for backtest/training jobs.
- [app/services/slack.py](app/services/slack.py) — daily PnL summaries.
- [app/services/google_sheets.py](app/services/google_sheets.py) — base client for the new predictions sheet.
- [app/watcher/diff.py](app/watcher/diff.py) — row-hashing logic; extend for prediction-history tracking rather than rewriting.

---

## Part 6 — Verification plan

| Phase | How to verify it works |
|---|---|
| P0 | `python -m training.train --horizons 1-30 --model lightgbm` → fold metrics report. Validation IC@10 > 0.04, IC@30 > 0.02, q10–q90 coverage in [0.75, 0.85]. Ridge baseline reported alongside. |
| P1 ✅ | **Unit tests:** 33 new cases pass (full suite 48 ✓) — alias word-boundary safety, news shape handling (new + legacy yfinance), Reddit ticker extraction + cutoff, X spam filter + query builder, sentiment aggregation, fundamentals helpers + earnings-date selection, regime + rank normalization. **Live verification (run once you have creds + deps):** Coverage report ≥ 70% (ticker, date) rows have non-null `news_sent_mean_7d` and `reddit_sent_mean_7d`. FinBERT scoring of 10k posts < 5 min on CPU. Earnings dates correctly mapped for 49/49 stocks. |
| P2 | Firestore: exactly one `indicators_latest/<TICKER>` doc per stock, `as_of_date` correct. Parquet archive: round-trip equality with the in-memory frame. New predictions sheet: 30-row block per stock + decision summary header. |
| P3 | 100% of Claude outputs parse to `TradeDecision`. Prompt-cache hit rate > 70% on batch run. End-to-end test on 5 sample stocks costs < $0.25 with Haiku, < $1 with Opus. |
| P4 | Backtest 2022-01-01 → 2024-12-31 out-of-sample: Sharpe > 1.0, DD < 20% net of costs. Paper trading 4 weeks: daily PnL within 1σ of backtest distribution. Synthetic-drift test fires alert within 10 trading days. Optimizer: solves 50 stocks in <2 s, all hard constraints satisfied post-solve, Sharpe uplift ≥ 0.3 over equal-weight/HRP/inverse-vol baselines, ±20% perturbation of µ flips ≤30% of weight. |

---

## Open questions (revisit during implementation)

1. **Universe:** stay at NIFTY 50 or expand to NIFTY 200 / 500? Cross-sectional signal improves with universe size, but social data coverage falls off sharply outside NIFTY 100.
2. **Long-only vs long-short:** shorting NSE cash equities is hard for retail; futures cleaner but operationally heavier. Default in P4: long-only.
3. **Claude model tier:** Opus 4.7 for high-stakes daily decisions vs Haiku 4.5 for batch/dry-runs — confirm budget per run.
4. **When to revisit live broker integration:** after how many weeks of paper-trading PnL meeting the backtest distribution? Default suggestion: 8 weeks.
5. **X (Twitter) fallback plan:** if snscrape breaks and stays broken for > 2 weeks, do we add a paid API tier ($100/mo for X Basic) or drop the channel entirely?
6. **Firestore vs Parquet read latency:** at decision time, Claude needs the latest indicator snapshot. Firestore reads are fine for 50 stocks (~150 ms) but if the universe grows beyond ~500, batch-fetch into memory once per run.

---

## Recommended next step

P1 is shipped. The two natural next moves:

- **P0 (target reframe)** — unblocks everything downstream. Without forward-return labels and a leakage-free trainer, the new P1 features can't actually be trained against the 30-day target. This is the highest-leverage next commit and the prerequisite for P3 and P4.
- **P2 (storage redesign)** — independently shippable, can run in parallel with P0. Decouples the prediction layer from the legacy Excel workbook and prepares the predictions sheet that the user will provide a link for.

Defer P3 and P4 until P0 produces a real forecast — they consume its output and aren't useful before it exists.
