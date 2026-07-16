# Runbook — production pipelines & backtesting

How to run everything end-to-end: the four **production** cadence scripts and the
full **backtesting** flow (ingestion → simulation → results). All commands are run
from the repo root.

- Production writes to the live Google Sheet + Firestore and calls the LLM.
- Backtesting is **fully local** (no Firestore/Sheets) and strictly point-in-time.

---

## 0. One-time setup

```bash
# Fill in secrets in .env (kept in sync with .env.example):
#   GOOGLE_CREDENTIALS / GOOGLE_APPLICATION_CREDENTIALS  → service-account JSON
#   FIRESTORE_PROJECT, SHEET_ID (already set)
#   GLM_API_KEY                                          → REQUIRED for the LLM layer
```

Key env knobs (see `.env`): `RETRAIN_INTERVAL_DAYS=15`, `NEWS_LOOKBACK_DAYS=7`,
`FUNDAMENTALS_REFRESH_DAYS=15`, `FORECAST_DAYS=15`. Every orchestrator supports
`--dry-run` (print the stage commands without executing) and, except the daily,
`--continue-on-error`.

---

# Part A — Production

Four scripts, each a cadence. They chain existing modules as isolated
subprocesses (`scripts/_pipeline.py`), so one stage's failure never corrupts the
next. All live under `scripts/` and run with `python -m scripts.<name>`.

| Script | Cadence | What it does |
|---|---|---|
| `run_daily` | every trading day | OHLCV → feature engineering → **target/stop exit check** |
| `run_weekly` | every 7 days | fresh **news+sentiment** → LLM → **fund allocation** (reuses existing forecasts + fundamentals) |
| `run_fortnightly` | every 15 days | **full pipeline**: OHLCV→FE→train/forecast→news→sentiment→fundamentals→LLM→**allocation** |
| `run_suggestions` | on demand | **read-only**: existing news+forecast+fundamentals → LLM → **allocation** |

> **Fund allocation runs after every LLM step.** `run_weekly`, `run_fortnightly`
> and `run_suggestions` all end with `features.portfolio_allocation` (Markowitz +
> cash guard), which sizes the BUY set and writes the **final website-facing
> suggestions** to Firestore (see *Final suggestions collection* below). Skip it
> with `--no-allocate` on any of them.

### 1. Daily — `scripts/run_daily.py`
Keeps prices current and books exits when a held position touches its LLM target
(`sell_price`) or stoploss today. Production analogue of the backtest's per-bar
exit monitor (`backtesting/exits.py`). No training, no forecast, no LLM.

```bash
python -m scripts.run_daily              # OHLCV update + exit check
python -m scripts.run_daily --skip-ohlcv # only the exit check
python -m scripts.run_daily --dry-run    # report exits without writing state
```
- **Stages:** (1) OHLCV update (`ingestion.collect_all`, market-data step only) →
  (2) feature engineering (`Feature_Engineering.compute_indicators`, ATR14 for
  slippage) → (3) `scan_exits` on today's O/H/L/C vs stored `trade_suggestions`
  levels; booked fills update `state/portfolio_holdings.json`.
- **Reads:** holdings book `state/portfolio_holdings.json`; targets/stops from
  Firestore `trade_suggestions`. A held name with no active suggestion is held.
- **Writes:** updated holdings + `outputs/production_daily_monitor.xlsx`
  (`Price_Monitor`, `Exits`). Tune via `DAILY_COST_BPS_ROUNDTRIP`,
  `DAILY_SLIPPAGE_ATR_MULT`, `DAILY_HONOR_STOPLOSS`.

### 2. Weekly — `scripts/run_weekly.py`
Refreshes only the fast-moving signal (news sentiment) and re-decides. **Reuses
already-dumped forecasts** (operational sheet) and **already-stored fundamentals**
(Firestore) — no retrain, no re-forecast.

```bash
python -m scripts.run_weekly
python -m scripts.run_weekly --tickers RELIANCE,TCS
python -m scripts.run_weekly --force        # ignore fingerprints; re-send all
```
- **Stages:** (1) `ingestion.collect_all --no-market-data --no-cross-sectional
  --no-reddit --no-x` → news to Firestore + FinBERT `sentiment_latest`; (2)
  `features.trade_suggestions` → BUY/HOLD/AVOID (with target+stop) to
  `trade_suggestions`; (3) `features.portfolio_allocation` → Markowitz sizing +
  the `final_suggestions` collection. The incremental fingerprint re-sends only
  changed stocks. `--no-allocate` skips stage 3.

### 3. Fortnightly — `scripts/run_fortnightly.py`
The complete refresh, in the exact order requested.

```bash
python -m scripts.run_fortnightly                  # live, end-to-end
python -m scripts.run_fortnightly --workflow-dry-run   # forecast pipeline dry (no sheet writes)
```
- **Stages:** (1-3) `run_full_workflow.py --live` — OHLCV append → indicators →
  ensemble forecast → push to sheet → `monthly_finetune --if-due` (incremental
  warm-start retrain when `RETRAIN_INTERVAL_DAYS` elapsed); (4-5)
  `ingestion.collect_all` news+sentiment (skips OHLCV, already refreshed); (6)
  `ingestion.fundamentals` — latest **5 quarters**, rolling window → Firestore;
  (7) `features.trade_suggestions`; (8) `features.portfolio_allocation` →
  `final_suggestions` collection.
- Production "training" is always the **incremental** finetune (never a full
  retrain), which fires on this 15-day cadence.

### 4. On-demand — `scripts/run_suggestions.py`
Everything already present; just decide. Reads forecasts (sheet) + sentiment +
fundamentals (Firestore), runs the LLM, writes `trade_suggestions`. No collection,
no training.

```bash
python -m scripts.run_suggestions            # LLM + allocation → final_suggestions
python -m scripts.run_suggestions --no-allocate   # LLM only
```
- **Stages:** (1) `features.trade_suggestions`; (2) `features.portfolio_allocation`
  sizes the BUY set (Markowitz + cash guard) and writes `final_suggestions`.

### Final suggestions collection (website)
Every allocation run writes Firestore **`final_suggestions`** (env
`FINAL_SUGGESTIONS_FIRESTORE_COLLECTION`), one doc per stock (id = ticker), so the
front-end reads it directly:

| action | fields | shown on site? |
|---|---|---|
| `BUY` | `target_price`, `stop_loss`, `allocation_pct` (% of funds), `allocation_amount`, `buy_price`, `confidence`, `reason`, `expected_return_15d`, `risk_reward` | ✅ `display=true` |
| `AVOID` | `reason`, `recommended_order` (EXIT if held) | ✅ `display=true` |
| `HOLD` | `reason` | ❌ `display=false` (stored, but ignored by the front-end) |

Front-end query: `where display == true`, then split by `action` (BUY vs
AVOID/SELL). HOLD docs are persisted for completeness but carry `display=false`.
The raw LLM verdicts remain in `trade_suggestions`; `final_suggestions` is the
combined, allocation-enriched, presentation-ready view.

### Suggested schedule
```
daily        → python -m scripts.run_daily
weekly (Mon) → python -m scripts.run_weekly
every 15d    → python -m scripts.run_fortnightly     # supersedes that day's weekly
ad hoc       → python -m scripts.run_suggestions
```

---

# Part B — Backtesting (fully local, point-in-time)

Backtest window is **Jan 2020 → present** (`.env`: `BACKTEST_START_DATE=2020-01-01`,
`BACKTEST_END_DATE=` empty ⇒ up to today). Universe = `BACKTEST_TICKERS` (20).
Three stages: collect → simulate → results.

```
1. COLLECT   news + fundamentals for the window      → local files
2. SIMULATE  walk-forward replay + bookkeeping        → state + bookkeeping.xlsx
3. RESULTS   benchmark vs NIFTY 50 / Sensex / bonds    → results.xlsx + chart
```

### Setup
```bash
export BACKTEST_ENABLED=true         # required for news collection
export GLM_API_KEY=...               # required for the simulation's LLM calls
# optional NIFTY per-bar line in the sim report (results fetches live regardless):
python -m features.cross_sectional --output Data/archive/indices.parquet --start 2019-01-01
```

### 1. Collect data
```bash
python -m backtesting.collect_data                 # news + fundamentals
python -m backtesting.collect_data --only fundamentals
python -m backtesting.collect_data --only news
```
- **Fundamentals** — screener.in quarterly (~12q) + 10y annual → PIT parquet
  `Data/pit/fundamentals.parquet`, availability-dated (`scrape_date`): quarterly
  `quarter_end+~45d`; annual visible in **June** of the FY-close year. The sim
  passes the LLM a rolling **5 periods**, quarterly-preferred / annual-fallback.
- **News → sentiment** — Wayback CDX → FinBERT → 7-day windows into the local news
  workbook `Data/archive/backtest_news.xlsx` (needs `BACKTEST_ENABLED=true`).
- See [`backtesting/README.md`](backtesting/README.md) for full details/flags.

### 2. Run the simulation
Validate on a short slice first, then the full window:
```bash
python -m backtesting.walk_forward --start 2020-01-01 --end 2020-06-30   # slice
python -m backtesting.walk_forward --start 2020-01-01                    # full → present
```
Replays production PIT-sliced frames → cold/warm-start retrain every 15 trading
days → T+1..T+15 forecast → GLM BUY/HOLD/AVOID → Markowitz sizing with a cash
reserve → fills at next-bar open; sentiment re-checked every 7 days; target/stop
exits booked intraday. Resumable (`--no-resume` to restart). Outputs:
`outputs/backtest/state/backtest_state.json`, `backtest_bookkeeping.xlsx`
(LLM_Suggestions, Allocations, Trades, Exits, EquityCurve, Forecasts,
PnL_Statement, …), `outputs/backtest/report/`.

### 3. Build the benchmark result sheet
```bash
python -m backtesting.results                      # vs NIFTY 50, Sensex, gov bonds
python -m backtesting.results --bond-yield 0.072
```
`outputs/backtest/results.xlsx` (Summary / Benchmark_Stats / Equity) + chart,
rebased to ₹1 cr, with CAGR/Sharpe/Sortino/max-drawdown/alpha/beta.

### Full backtest sequence
```bash
export BACKTEST_ENABLED=true; export GLM_API_KEY=...
python -m features.cross_sectional --output Data/archive/indices.parquet --start 2019-01-01
python -m backtesting.collect_data
python -m backtesting.walk_forward --start 2020-01-01 --end 2020-06-30   # validate
python -m backtesting.walk_forward --start 2020-01-01                    # full → present
python -m backtesting.results
```

---

## Point-in-time guarantee (backtest)
Every input at a decision date `C` is cut at `C` — technicals (`shift(1)`-lagged,
frame ≤ C), model (trained ≤ C; cold-started on ≤2019, never on production
weights), news (`as_of_date ∈ [C-7d, C]`), fundamentals (availability-dated:
quarterly `+45d`, annual June-refresh). The **only** forward-looking value is the
model's own T+1..T+15 forecast. Execution fills at open of C+1; realized returns
appear only in post-hoc scoring.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `GLM_API_KEY not set` | Export it (both production suggestions and the backtest need it). |
| Backtest news says "disabled" | `export BACKTEST_ENABLED=true`. |
| screener.in HTTP 403/429 | `export SCREENER_COOKIE="sessionid=...; csrftoken=..."`; raise `--pause`. |
| Daily run: "no open positions" | `state/portfolio_holdings.json` is empty — run allocation first (`run_suggestions --allocate`). |
| A stage failed mid-pipeline | Re-run with `--continue-on-error`, or fix and re-run (each stage is idempotent). |
| Inspect commands without running | Add `--dry-run` to any orchestrator. |
