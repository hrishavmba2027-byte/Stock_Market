# P0 changes — summary

## Files

| File | Change |
|---|---|
| `Feature_Engineering.py` | Added forward-label helpers + `has_labels` flag; `compute_indicators` now auto-attaches `y_logret_h1 .. y_logret_h30` and `has_labels`. |
| `model_2_GPU_.ipynb` | Rewritten for multi-horizon quantile forecasting, env-driven sequential splits, no `Open[t]` injection. |
| `.env.example` | New — TRAIN_END / TEST_END / BACK_TEST_START / BACK_TEST_END. |

## Splits (date-based, sequential, env-driven)

Read from `.env` via `python-dotenv`:

```
TRAIN_END=2022-12-31
TEST_END=2023-06-30
BACK_TEST_START=2023-07-01
BACK_TEST_END=2023-12-31
```

- **Train** = `[TRAIN_END - 10y, TRAIN_END]`
- **Test**  = `(TRAIN_END, TEST_END]`
- **Backtest** = `[BACK_TEST_START, BACK_TEST_END]`

Last 30 rows of each training segment are dropped because their forward
labels are not available (`has_labels=False`). This is the 30-day embargo
required by ROADMAP P0.3.

## Target

- 30 forward log-returns per row: `y_h = log(Close[t+h] / Close[t])` for `h = 1..30`.
- Surfaced as columns `y_logret_h1 .. y_logret_h30` plus `has_labels` flag.

## Models

- Dense / LSTM / Transformer kept, but final layer now outputs
  `n_quantiles × n_horizons = 3 × 30 = 90` reshaped to `(batch, 3, 30)`.
- Loss: pinball (quantile) loss across all 90 outputs.
- Post-hoc non-crossing sort enforces `q10 <= q50 <= q90` per horizon.
- Ensemble weights unchanged: Dense 0.2 / LSTM 0.3 / Transformer 0.5.

## Leakage contract (enforced in code + sentinel-tested)

For prediction anchored at day `t`:

1. Window of features is **rows `[t - SEQ_LEN, t - 1]`** only. Row `t` is never read.
2. Feature scaler is fit on the train window ONLY, then frozen for test/backtest.
3. `inject_intraday_open` and `USE_INTRADAY_OPEN` are gone.
4. Smoke test corrupts `Close[t]` (and `Open[t]`) with `-1e9` for every row
   `>= t` and asserts the model's prediction is byte-identical.

## Function-argument hygiene

Every training / inference / backtest entry point takes all dependencies as
arguments — `device`, `use_amp`, `pin_memory`, `quantiles`, `feature_columns`,
`label_cols`, `seq_len`, `horizons`, `target_base_col`, scalers, etc. No
implicit reliance on notebook globals from inside utilities.

## Backtest

`run_backtest(...)` walks the backtest window for every stock and emits a
long-format dataframe with one row per `(symbol, anchor_date, horizon)`:

```
Symbol, Anchor_Date, Horizon, Anchor_Close,
Pred_LogRet_q10/q50/q90, Pred_Price_q10/q50/q90,
Realised_LogRet, Realised_Price
```

`Realised_*` is NaN for rows near the end of the backtest window where 30
forward bars are not yet available.

## Smoke tests passed (against synthetic data)

- Output shape `(N, 3, 30)`
- Byte-identical predictions under `Close[t]` and `Open[t]` sentinel corruption
- Every training row has `has_labels=True`
- `train < test < backtest` dates for all stocks
- Train window equals `[TRAIN_END - 10y, TRAIN_END]` exactly
