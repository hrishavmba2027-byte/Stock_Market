"""Point-in-time model training + forecasting for the walk-forward run.

Reuses the *production* training core (``monthly_finetune``) and inference core
(``main``) unchanged — only the data source (workbook frames sliced ≤ C) and the
artifact directory (``outputs/backtest/*``) differ, so the production model in
``outputs/Saved_Models`` is never touched.

* :func:`seed_train_at` — **cold start** at the first cutoff: fresh, randomly
  initialised Dense/LSTM/Transformer built from the metadata capacity, trained on
  all data ≤ C (higher epochs), saved to the backtest model dir. This is the key
  no-leakage step: the 2019 seed never loads the 2026-trained production weights.
* :func:`retrain_at` — **warm start**: fine-tune the backtest checkpoints on the
  data that became visible since the last retrain (frames still sliced ≤ C).
* :func:`forecast_at` — T+1..T+15 price path per symbol from the ≤ C model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtesting import data_access


def _import_cores():
    import main as forecast          # inference core (pulls in torch)
    import monthly_finetune as mf    # training core
    return forecast, mf


def _mf_args(settings: Any, *, epochs: Optional[int] = None, device: str = "cpu"):
    """Build a ``monthly_finetune`` args namespace pointed at the backtest paths.

    Uses the sys.argv-swap pattern from ``scripts/train_from_workbook.py`` so we
    reuse production hyper-parameter defaults, then override the model dir /
    metadata / epochs for the backtest.
    """
    _, mf = _import_cores()
    saved = sys.argv
    try:
        sys.argv = [
            "monthly_finetune.py",
            "--model-dir", str(settings.model_dir),
            "--metadata", str(settings.metadata_path),
            "--state-file", str(settings.finetune_state_file),
            "--output-dir", str(settings.output_dir),
            "--no-fine-tune-batch-only-new-data",
            "--device", device,
        ]
        args = mf.parse_args()
    finally:
        sys.argv = saved
    if epochs is not None:
        args.epochs = int(epochs)
    return args


def _build_cold_start_models(metadata: Dict[str, Any], device) -> Dict[str, Any]:
    """Instantiate fresh (randomly initialised) model modules from metadata."""
    forecast, _ = _import_cores()
    feature_columns, seq_len, feature_count, dense_input_size = forecast.validate_metadata(metadata)
    quantiles = list(metadata.get("quantiles", [0.1, 0.5, 0.9]))
    horizons = forecast.metadata_horizons(metadata)
    n_quantiles, n_horizons = len(quantiles), len(horizons)
    output_size = n_quantiles * n_horizons
    quantile_idx = forecast.find_metadata_index(quantiles, 0.5, "quantiles")
    bounds = forecast.model_horizon_bounds(metadata, n_horizons)
    cap = metadata.get("model_capacity", {})

    dense = forecast.DenseModel(
        input_size=dense_input_size, hidden_size=int(cap.get("Dense", {}).get("hidden_size", 128)),
        output_size=output_size, n_quantiles=n_quantiles, n_horizons=n_horizons,
        quantile_idx=quantile_idx, horizon_bounds=bounds,
    )
    lstm = forecast.LSTMModel(
        input_size=feature_count, hidden_size=int(cap.get("LSTM", {}).get("hidden_size", 64)),
        num_layers=int(cap.get("LSTM", {}).get("num_layers", 2)), output_size=output_size,
        n_quantiles=n_quantiles, n_horizons=n_horizons, quantile_idx=quantile_idx,
        horizon_bounds=bounds,
    )
    transformer = forecast.TransformerModel(
        input_size=feature_count, hidden_size=int(cap.get("Transformer", {}).get("hidden_size", 128)),
        num_heads=int(cap.get("Transformer", {}).get("num_heads", 4)),
        num_layers=int(cap.get("Transformer", {}).get("num_layers", 2)), output_size=output_size,
        n_quantiles=n_quantiles, n_horizons=n_horizons, quantile_idx=quantile_idx,
        max_seq_len=max(int(seq_len), 64), horizon_bounds=bounds,
    )
    return {"Dense": dense.to(device), "LSTM": lstm.to(device), "Transformer": transformer.to(device)}


def seed_train_at(frames: Dict[str, pd.DataFrame], cutoff: Any, settings: Any, *, device: str = "cpu") -> Dict[str, Any]:
    """Cold-start the backtest ensemble on all data ≤ ``cutoff`` (no prod weights)."""
    forecast, mf = _import_cores()
    args = _mf_args(settings, epochs=int(settings.seed_epochs), device=device)
    metadata = mf.get_or_create_metadata(Path(settings.metadata_path))
    forecast.validate_metadata(metadata)
    dev = forecast.choose_device(device)

    truncated = data_access.truncate_frames(frames, cutoff)
    cutoffs = {sym: None for sym in truncated}     # train on ALL rows ≤ C
    arrays = mf.build_finetune_arrays(truncated, metadata, args, cutoffs)
    if len(arrays.X_train) < int(args.min_train_samples) or len(arrays.X_val) == 0:
        raise RuntimeError(
            f"cold-start seed at {pd.Timestamp(cutoff).date()} has too few samples "
            f"(train={len(arrays.X_train)}, val={len(arrays.X_val)})"
        )
    models = _build_cold_start_models(metadata, dev)
    training = {name: mf.fine_tune_one_model(name, model, arrays, args, dev) for name, model in models.items()}
    evaluation = mf.evaluate_models(models, arrays, metadata, args, dev)
    mf.save_checkpoints_atomically(models, Path(settings.model_dir), metadata, dry_run=False)
    return {"mode": "seed", "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
            "train_samples": int(len(arrays.X_train)), "val_samples": int(len(arrays.X_val)),
            "evaluation": evaluation}


def retrain_at(
    frames: Dict[str, pd.DataFrame],
    cutoff: Any,
    settings: Any,
    *,
    prev_cutoff: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Warm-start fine-tune the backtest ensemble on data in (prev_cutoff, C]."""
    forecast, mf = _import_cores()
    args = _mf_args(settings, device=device)
    metadata = mf.get_or_create_metadata(Path(settings.metadata_path))
    forecast.validate_metadata(metadata)
    dev = forecast.choose_device(device)

    truncated = data_access.truncate_frames(frames, cutoff)
    prev = pd.Timestamp(prev_cutoff).normalize() if prev_cutoff is not None else None
    cutoffs = {sym: prev for sym in truncated}
    arrays = mf.build_finetune_arrays(truncated, metadata, args, cutoffs)
    if len(arrays.X_train) == 0 or len(arrays.X_val) == 0:
        return {"mode": "retrain", "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
                "skipped": "no new samples since last retrain"}
    models = forecast.load_models(Path(settings.model_dir), metadata, dev)
    training = {name: mf.fine_tune_one_model(name, model, arrays, args, dev) for name, model in models.items()}
    evaluation = mf.evaluate_models(models, arrays, metadata, args, dev)
    mf.save_checkpoints_atomically(models, Path(settings.model_dir), metadata, dry_run=False)
    return {"mode": "retrain", "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
            "train_samples": int(len(arrays.X_train)), "val_samples": int(len(arrays.X_val)),
            "evaluation": evaluation}


def forecast_at(
    frames: Dict[str, pd.DataFrame],
    cutoff: Any,
    settings: Any,
    *,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """T+1..T+forecast_days close-price path per symbol from the ≤ C backtest model."""
    forecast, mf = _import_cores()
    metadata = mf.get_or_create_metadata(Path(settings.metadata_path))
    forecast.validate_metadata(metadata)
    dev = forecast.choose_device(device)

    prev_days = os.environ.get("FORECAST_DAYS")
    os.environ["FORECAST_DAYS"] = str(int(settings.forecast_days))
    try:
        payloads = data_access.payloads_asof(frames, cutoff, metadata)
        if not payloads:
            return {}
        models = forecast.load_models(Path(settings.model_dir), metadata, dev)
        parts = []
        for payload in payloads:
            try:
                parts.append(forecast.prepare_stock_part(payload, metadata, latest_only=True))
            except (ValueError, RuntimeError):
                continue
        results, _ = forecast.run_inference(parts, models, metadata, dev)
        forecast_cols = forecast.forecast_close_columns(metadata)
    finally:
        if prev_days is None:
            os.environ.pop("FORECAST_DAYS", None)
        else:
            os.environ["FORECAST_DAYS"] = prev_days

    if results is None or results.empty:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for symbol, grp in results.groupby("Symbol"):
        row = grp.iloc[-1]                     # latest forecast row = anchor at C
        path: Dict[str, float] = {}
        for col in forecast_cols:
            if col in row and pd.notna(row[col]):
                path[col.replace("Forecast_Close_", "")] = float(row[col])
        if path:
            out[str(symbol).upper()] = path
    return out
