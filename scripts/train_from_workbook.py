#!/usr/bin/env python3
"""Train (replace) the production model from the completed local workbook.

The Google training sheet is at the 10M-cell limit, so we train directly from
``Data/archive/nse_stock_data_train_filled.xlsx`` instead. This reuses the exact
production training core from :mod:`monthly_finetune` (same feature engineering,
same warm-start, same atomic save) — only the *data source* differs: per-symbol
frames are built from the workbook instead of gspread worksheets.

Trains on the FULL history of every sheet (cutoffs = None ⇒ all rows), warm-starts
from the existing ``outputs/Saved_Models`` checkpoints, and atomically overwrites
them — i.e. replaces the production model. A timestamped backup of the prior
checkpoints should be taken first (the caller does this).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_WORKBOOK = _REPO_ROOT / "Data" / "archive" / "nse_stock_data_train_filled.xlsx"
REQUIRED = ["Date", "Date_str", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def _legacy_close_cols(header, title):
    out = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        name = f"{f}_{title}.NS"
        if name in header:
            out[f] = name
    return out if len(out) == 5 else None


def build_frames(workbook: Path, tickers: Optional[List[str]]) -> Dict[str, pd.DataFrame]:
    """One REQUIRED-columns OHLCV frame per sheet (OHLCV from the filled columns)."""
    xl = pd.ExcelFile(workbook)
    sheets = xl.sheet_names
    if tickers:
        wanted = {t.upper() for t in tickers}
        sheets = [s for s in sheets if s.upper() in wanted]
    frames: Dict[str, pd.DataFrame] = {}
    for s in sheets:
        df = pd.read_excel(xl, sheet_name=s)
        title = s.strip().upper()
        header = list(df.columns)
        # Prefer the standard mirror columns (now filled); fall back to legacy.
        if all(c in header for c in ("Open", "High", "Low", "Close", "Volume")) and \
                pd.to_numeric(df["Close"], errors="coerce").notna().any():
            o = df
            cols = {f: f for f in ("Open", "High", "Low", "Close", "Volume")}
        else:
            legacy = _legacy_close_cols(header, title)
            if legacy is None:
                log(f"[skip] {title}: no OHLCV columns")
                continue
            o = df
            cols = legacy
        frame = pd.DataFrame()
        frame["Date"] = pd.to_datetime(df["Date_str"], errors="coerce")
        frame["Date_str"] = df["Date_str"].astype(str)
        for f in ("Open", "High", "Low", "Close", "Volume"):
            frame[f] = pd.to_numeric(o[cols[f]], errors="coerce")
        frame["Adj Close"] = frame["Close"]
        frame = frame[REQUIRED].dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        frame = frame.drop_duplicates(subset=["Date_str"], keep="last").sort_values("Date").reset_index(drop=True)
        if len(frame) > 0:
            frames[title] = frame
    return frames


def main(argv: Optional[List[str]] = None) -> int:
    import monthly_finetune as mf

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    p.add_argument("--tickers", default=None, help="Comma-separated subset (default: all sheets).")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--model-dir", default=str(_REPO_ROOT / "outputs" / "Saved_Models"))
    p.add_argument("--metadata", default=str(_REPO_ROOT / "outputs" / "pipeline_metadata.json"))
    p.add_argument("--dry-run", action="store_true", help="Do everything except overwrite the checkpoints.")
    args = p.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    log(f"Building frames from {args.workbook} …")
    frames = build_frames(Path(args.workbook), tickers)
    log(f"Built {len(frames)} symbol frames "
        f"(rows: {min((len(f) for f in frames.values()), default=0)}–{max((len(f) for f in frames.values()), default=0)})")
    if not frames:
        log("No frames built — aborting")
        return 1

    # Build a monthly_finetune args namespace for the training hyper-params, then
    # drive its training core directly (bypassing gspread / archival).
    saved = sys.argv
    try:
        sys.argv = ["monthly_finetune.py",
                    "--model-dir", args.model_dir,
                    "--metadata", args.metadata,
                    "--no-fine-tune-batch-only-new-data",
                    "--device", args.device]
        mf_args = mf.parse_args()
    finally:
        sys.argv = saved

    metadata = mf.get_or_create_metadata(Path(args.metadata))
    mf.forecast.validate_metadata(metadata)
    device = mf.forecast.choose_device(mf_args.device)
    log(f"Device: {device}")

    cutoffs = {sym: None for sym in frames}  # train on ALL rows of every symbol
    arrays = mf.build_finetune_arrays(frames, metadata, mf_args, cutoffs)
    log(f"train_samples={len(arrays.X_train)} val_samples={len(arrays.X_val)} "
        f"symbols={len(arrays.summaries)}")
    if len(arrays.X_train) < int(mf_args.min_train_samples):
        log(f"Too few training samples ({len(arrays.X_train)}) — aborting")
        return 1
    if len(arrays.X_val) == 0:
        log("No validation samples — aborting (won't overwrite checkpoints)")
        return 1

    model_dir = Path(args.model_dir)
    models = mf.forecast.load_models(model_dir, metadata, device)
    training = {}
    for name, model in models.items():
        log(f"Fine-tuning {name} …")
        training[name] = mf.fine_tune_one_model(name, model, arrays, mf_args, device)
    evaluation = mf.evaluate_models(models, arrays, metadata, mf_args, device)
    overwrite = mf.save_checkpoints_atomically(models, model_dir, metadata, dry_run=bool(args.dry_run))

    result = {
        "status": "dry_run_ok" if args.dry_run else "ok",
        "symbols": len(frames),
        "train_samples": int(len(arrays.X_train)),
        "val_samples": int(len(arrays.X_val)),
        "checkpoint_overwrite": overwrite,
        "training": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in training.items()},
        "evaluation": evaluation,
        "model_dir": str(model_dir),
    }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
