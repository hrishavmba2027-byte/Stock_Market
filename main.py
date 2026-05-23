from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from app.pipeline.metadata import (
    get_or_create_metadata,
    record_run_result,
    safe_load_metadata,
)
from app.services.sheet_archival import (
    DEFAULT_HISTORICAL_TRAINING_SHEET_ID,
    archive_old_rows_for_worksheet,
    find_worksheet_by_title,
    parse_positive_int,
    rolling_operational_rows,
)
from app.services.google_sheet_updates import (
    PredictionSheetUpdateService,
    sync_decision_features,
)
from Feature_Engineering import (
    FeatureEngineeringError,
    INDICATOR_COLUMNS,
    compute_indicators,
    detect_date_column,
    ensure_date_column,
)


DEFAULT_SHEET_ID = "1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o"
TARGET_COL = "Close"
USE_INTRADAY_OPEN = True
BATCH_SIZE = 256
PREDICTED_COL = "predicted"
PREDICTED_PRICE_COL = "Predicted_Close_Price"
FORECAST_CLOSE_PREFIX = "Forecast_Close_T+"
MAX_FORECAST_HORIZON = 5
RECURSIVE_FORECAST_HORIZONS = 5
RECURSIVE_RETURN_CLIP = (-0.08, 0.08)
ROLLING_STABILIZATION_WINDOW = 20
LATEST_SHEET_ROWS_TO_KEEP = 30
SHEET_ROW_COL = "__sheet_row_number"
SORT_POSITION_COL = "__sort_position"
STATUS_OK = "ok"
STATUS_NO_NEW_DATA = "no_new_data"
STATUS_NO_VALID_START_POINT = "no_valid_start_point"
FORWARD_LABEL_PREFIX = "y_logret_h"
HAS_LABELS_COL = "has_labels"

PRICE_PREFIXES = {
    "open_": "Open",
    "high_": "High",
    "low_": "Low",
    "close_": "Close",
    "volume_": "Volume",
}
EXACT_PRICE_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
MODELS: Optional[Dict[str, nn.Module]] = None
MODELS_CACHE_KEY: Optional[Tuple[Any, ...]] = None
MODELS_LOCK = Lock()


class NoValidStartPointError(ValueError):
    pass


@dataclass
class SheetPayload:
    name: str
    frame: pd.DataFrame
    headers: List[str]
    data_row_count: int
    worksheet: Any = None


@dataclass
class StockInferencePart:
    symbol: str
    X: np.ndarray
    dates: np.ndarray
    sheet_rows: np.ndarray
    forecast_start_positions: np.ndarray
    actual_price: np.ndarray
    anchor_close: np.ndarray
    target_scaler: MinMaxScaler
    feature_scaler: MinMaxScaler
    current_open_values: np.ndarray
    headers: List[str]
    data_row_count: int
    forecast_base_frame: Optional[pd.DataFrame] = None
    forecast_date_col: Optional[str] = None
    worksheet: Any = None
    row_predictions: Dict[int, float] = field(default_factory=dict)
    row_forecasts: Dict[int, List[float]] = field(default_factory=dict)
    status: str = STATUS_OK
    eligible_target_count: int = 0
    last_valid_position: Optional[int] = None


def select_quantile_outputs(
    output: torch.Tensor,
    n_quantiles: int,
    n_horizons: int,
    quantile_idx: int,
) -> torch.Tensor:
    if output.dim() == 2:
        expected_width = n_quantiles * n_horizons
        if output.shape[1] != expected_width:
            raise ValueError(f"model output width mismatch: expected {expected_width}, got {output.shape[1]}")
        output = output.view(-1, n_quantiles, n_horizons)
    elif output.dim() != 3:
        raise ValueError(f"model output must be 2D or 3D, got {tuple(output.shape)}")
    if output.shape[1] != n_quantiles or output.shape[2] != n_horizons:
        raise ValueError(
            f"model output shape mismatch: expected (*, {n_quantiles}, {n_horizons}), got {tuple(output.shape)}"
        )
    return output[:, quantile_idx, :]


class DenseModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_quantiles: int,
        n_horizons: int,
        quantile_idx: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_horizons = n_horizons
        self.quantile_idx = quantile_idx
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return select_quantile_outputs(
            self.fc3(x),
            self.n_quantiles,
            self.n_horizons,
            self.quantile_idx,
        )


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        n_quantiles: int,
        n_horizons: int,
        quantile_idx: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_horizons = n_horizons
        self.quantile_idx = quantile_idx
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return select_quantile_outputs(
            self.head(hidden[-1]),
            self.n_quantiles,
            self.n_horizons,
            self.quantile_idx,
        )


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        output_size: int,
        n_quantiles: int,
        n_horizons: int,
        quantile_idx: int,
        dropout: float = 0.3,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_horizons = n_horizons
        self.quantile_idx = quantile_idx
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.position_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.position_embedding.shape[1]}"
            )
        x = self.input_projection(x)
        x = x + self.position_embedding[:, :seq_len, :]
        x = self.encoder(x)
        return select_quantile_outputs(
            self.head(x[:, -1, :]),
            self.n_quantiles,
            self.n_horizons,
            self.quantile_idx,
        )


def log(message: str) -> None:
    print(message, file=sys.stderr)


def configured_forecast_days(default: int = MAX_FORECAST_HORIZON) -> int:
    return parse_positive_int(os.environ.get("FORECAST_DAYS"), default, name="FORECAST_DAYS")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run inference-only stock close-price predictions and update Google Sheets."
    )
    parser.add_argument("--source", choices=["google", "workbook"], default="google")
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID)
    parser.add_argument("--google-credentials", default=None)
    parser.add_argument("--worksheet", default=None, help="Optional worksheet/stock name to process.")
    parser.add_argument("--worksheets", default=None, help="Comma-separated worksheet/stock names to process.")
    parser.add_argument("--workbook", default=str(script_dir / "Data" / "nse_stock_data.xlsx"))
    parser.add_argument("--model-dir", default=str(script_dir / "outputs" / "Saved_Models"))
    parser.add_argument("--metadata", default=str(script_dir / "outputs" / "pipeline_metadata.json"))
    parser.add_argument("--output-dir", default=str(script_dir / "outputs" / "main_inference"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--plots", action="store_true", help="Write Plotly HTML evaluation plots.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For Google Sheets source, read and predict without writing sheet updates.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Forecast only the latest eligible row in each worksheet.",
    )
    parser.add_argument(
        "--refresh-existing-forecasts",
        action="store_true",
        help="Refresh existing Forecast_Close_T+ columns even when those cells already contain values.",
    )
    parser.add_argument(
        "--all-eligible-rows",
        action="store_true",
        help=(
            "Ignore existing predicted checkpoints and forecast every row that can form "
            "a historical input window."
        ),
    )
    return parser.parse_args()


def parse_worksheet_filters(worksheets: Optional[str], worksheet: Optional[str] = None) -> Optional[Set[str]]:
    values: List[str] = []
    for source in (worksheets, worksheet):
        if not source:
            continue
        values.extend(part.strip() for part in str(source).split(","))
    normalized = {value.upper() for value in values if value}
    return normalized or None


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available")
    return torch.device(device_arg)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON from *path*.

    For pipeline_metadata.json use ``get_or_create_metadata`` / ``safe_load_metadata``
    instead — those functions add corruption recovery, atomic backups, and schema
    migration that this bare implementation lacks.

    Kept here to avoid breaking callers that load non-metadata JSON files.
    """
    with open(path, "r") as file:
        return json.load(file)


def torch_load_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def find_metadata_index(values: Sequence[Any], wanted: float, label: str) -> int:
    for idx, value in enumerate(values):
        if abs(float(value) - wanted) < 1e-9:
            return idx
    raise ValueError(f"metadata {label} must include {wanted:g}")


def metadata_horizons(metadata: Dict[str, Any]) -> List[int]:
    horizons = [int(horizon) for horizon in metadata.get("horizons", [])]
    expected_horizons = list(range(1, MAX_FORECAST_HORIZON + 1))
    if horizons != expected_horizons:
        raise ValueError(f"metadata horizons must be 1..{MAX_FORECAST_HORIZON}, got {horizons}")
    return horizons


def direct_trained_horizon_count(metadata: Dict[str, Any]) -> int:
    return len(metadata_horizons(metadata))


def is_forward_label_column(column: Any) -> bool:
    text = str(column).strip().lower()
    prefix = FORWARD_LABEL_PREFIX.lower()
    if not text.startswith(prefix):
        return False
    return text[len(prefix):].isdigit()


def quantile_output_contract(metadata: Dict[str, Any], output_size: int) -> Tuple[int, int, int]:
    quantiles = list(metadata.get("quantiles", []))
    horizons = metadata_horizons(metadata)
    if not quantiles or not horizons:
        raise ValueError("metadata must include quantiles and horizons for quantile checkpoints")
    n_quantiles = len(quantiles)
    n_horizons = len(horizons)
    expected_output_size = n_quantiles * n_horizons
    if int(output_size) != expected_output_size:
        raise ValueError(
            f"checkpoint output_size={output_size} but metadata requires "
            f"{n_quantiles} quantiles * {n_horizons} horizons = {expected_output_size}"
        )
    quantile_idx = find_metadata_index(quantiles, 0.5, "quantiles")
    return n_quantiles, n_horizons, quantile_idx


def infer_dense_model(state: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> DenseModel:
    hidden_size, input_size = state["fc1.weight"].shape
    output_size = int(state["fc3.weight"].shape[0])
    _, _, _, dense_input_size = validate_metadata(metadata)
    if int(input_size) != dense_input_size:
        raise ValueError(f"Dense checkpoint input_size={input_size} but metadata dense_input_size={dense_input_size}")
    output_contract = quantile_output_contract(metadata, output_size)
    return DenseModel(
        input_size=int(input_size),
        hidden_size=int(hidden_size),
        output_size=output_size,
        n_quantiles=output_contract[0],
        n_horizons=output_contract[1],
        quantile_idx=output_contract[2],
    )


def infer_lstm_model(state: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> LSTMModel:
    input_size = int(state["lstm.weight_ih_l0"].shape[1])
    hidden_size = int(state["lstm.weight_ih_l0"].shape[0] // 4)
    _, _, feature_count, _ = validate_metadata(metadata)
    if input_size != feature_count:
        raise ValueError(f"LSTM checkpoint input_size={input_size} but metadata feature_count={feature_count}")
    layer_indices = {
        int(key.split("lstm.weight_ih_l", 1)[1])
        for key in state
        if key.startswith("lstm.weight_ih_l")
    }
    output_size = int(state["head.3.weight"].shape[0])
    output_contract = quantile_output_contract(metadata, output_size)
    return LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=max(layer_indices) + 1,
        output_size=output_size,
        n_quantiles=output_contract[0],
        n_horizons=output_contract[1],
        quantile_idx=output_contract[2],
    )


def infer_transformer_heads(hidden_size: int, metadata: Dict[str, Any]) -> int:
    configured = metadata.get("model_capacity", {}).get("Transformer", {}).get("num_heads")
    if configured and hidden_size % int(configured) == 0:
        return int(configured)
    for candidate in (4, 8, 2, 1):
        if hidden_size % candidate == 0:
            return candidate
    return 1


def infer_transformer_model(state: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> TransformerModel:
    hidden_size, input_size = state["input_projection.weight"].shape
    _, _, feature_count, _ = validate_metadata(metadata)
    if int(input_size) != feature_count:
        raise ValueError(f"Transformer checkpoint input_size={input_size} but metadata feature_count={feature_count}")
    layer_indices = {
        int(key.split("encoder.layers.", 1)[1].split(".", 1)[0])
        for key in state
        if key.startswith("encoder.layers.")
    }
    output_size = int(state["head.4.weight"].shape[0])
    output_contract = quantile_output_contract(metadata, output_size)
    return TransformerModel(
        input_size=int(input_size),
        hidden_size=int(hidden_size),
        num_heads=infer_transformer_heads(int(hidden_size), metadata),
        num_layers=max(layer_indices) + 1,
        output_size=output_size,
        n_quantiles=output_contract[0],
        n_horizons=output_contract[1],
        quantile_idx=output_contract[2],
        max_seq_len=int(state["position_embedding"].shape[1]),
    )


def load_models(model_dir: Path, metadata: Dict[str, Any], device: torch.device) -> Dict[str, nn.Module]:
    loaders = {
        "Dense": lambda state: infer_dense_model(state, metadata),
        "LSTM": lambda state: infer_lstm_model(state, metadata),
        "Transformer": lambda state: infer_transformer_model(state, metadata),
    }
    models: Dict[str, nn.Module] = {}
    for model_name, factory in loaders.items():
        checkpoint_path = model_dir / f"{model_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        state = torch_load_state(checkpoint_path, device)
        model = factory(state).to(device)
        model.load_state_dict(state)
        model.eval()
        models[model_name] = model
        log(f"Loaded {model_name}: {checkpoint_path}")
    return models


def get_models(model_dir: Path, metadata: Dict[str, Any], device: torch.device) -> Dict[str, nn.Module]:
    global MODELS, MODELS_CACHE_KEY

    feature_columns, seq_len, feature_count, dense_input_size = validate_metadata(metadata)
    cache_key = (
        str(model_dir.resolve()),
        str(device),
        tuple(feature_columns),
        seq_len,
        feature_count,
        dense_input_size,
        tuple(metadata.get("quantiles", [])),
        tuple(metadata.get("horizons", [])),
        json.dumps(metadata.get("model_capacity", {}), sort_keys=True),
    )
    with MODELS_LOCK:
        if MODELS is None or MODELS_CACHE_KEY != cache_key:
            MODELS = load_models(model_dir, metadata, device)
            MODELS_CACHE_KEY = cache_key
        else:
            log("Using cached models")
        return MODELS


def canonical_column_name(column: Any) -> str:
    name = str(column).strip()
    lower = name.lower()
    if lower in EXACT_PRICE_COLUMNS:
        return EXACT_PRICE_COLUMNS[lower]
    for prefix, canonical in PRICE_PREFIXES.items():
        if lower.startswith(prefix):
            return canonical
    return name


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [canonical_column_name(column) for column in normalized.columns]
    normalized = normalized.loc[:, ~pd.Index(normalized.columns).duplicated()].copy()
    return normalized


def prepare_feature_engineering_input(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    rename_map = {}
    for column in prepared.columns:
        lower = str(column).strip().lower()
        if lower in EXACT_PRICE_COLUMNS:
            rename_map[column] = lower
    return prepared.rename(columns=rename_map)


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for column in numeric.columns:
        if numeric[column].dtype == object:
            numeric[column] = numeric[column].astype(str).str.replace(",", "", regex=False)
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.ffill().fillna(0)


def to_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def get_dates(df: pd.DataFrame, date_col: Optional[str]) -> np.ndarray:
    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        except TypeError:
            dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().any():
            return dates.to_numpy()
    return np.arange(len(df))


def forbidden_model_feature_columns(metadata: Dict[str, Any]) -> Set[str]:
    label_columns = set(metadata.get("label_columns", []))
    configured_horizons: List[int] = []
    for horizon in metadata.get("horizons", []):
        try:
            configured_horizons.append(int(horizon))
        except (TypeError, ValueError):
            continue
    # Feature_Engineering.py may still attach its default 30 forward labels.
    # Treat every canonical label up to that legacy range as target-only.
    max_label_horizon = max([30, MAX_FORECAST_HORIZON, *configured_horizons])
    label_columns.update(f"{FORWARD_LABEL_PREFIX}{h}" for h in range(1, max_label_horizon + 1))
    label_columns.add(HAS_LABELS_COL)
    return label_columns


def forward_label_like_columns(columns: Iterable[Any], metadata: Dict[str, Any]) -> List[Any]:
    forbidden = forbidden_model_feature_columns(metadata)
    return [
        column
        for column in columns
        if column in forbidden or is_forward_label_column(column)
    ]


def validate_metadata(metadata: Dict[str, Any]) -> Tuple[List[str], int, int, int]:
    metadata_horizons(metadata)
    feature_columns = list(metadata["feature_columns"])
    seq_len = int(metadata["seq_len"])
    feature_count = int(metadata["feature_count"])
    dense_input_size = int(metadata["dense_input_size"])
    leakage_columns = sorted(forward_label_like_columns(feature_columns, metadata))
    if leakage_columns:
        raise ValueError(f"Forward-label columns must not be model features: {leakage_columns}")
    if len(feature_columns) != feature_count:
        raise ValueError(
            f"Metadata feature_count={feature_count} but feature_columns has {len(feature_columns)} entries"
        )
    if seq_len * feature_count != dense_input_size:
        raise ValueError(
            f"Metadata dense_input_size mismatch: {seq_len} * {feature_count} != {dense_input_size}"
        )
    return feature_columns, seq_len, feature_count, dense_input_size


def find_dataframe_column(df: pd.DataFrame, column_name: str) -> Optional[Any]:
    wanted = column_name.strip().lower()
    for column in df.columns:
        if str(column).strip().lower() == wanted:
            return column
    return None


def missing_forecast_mask(df: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    for column_name in forecast_close_columns(metadata):
        column = find_dataframe_column(df, column_name)
        if column is None:
            mask[:] = True
            continue
        values = pd.to_numeric(df[column].replace("", np.nan), errors="coerce")
        mask |= ~np.isfinite(values.to_numpy(dtype=np.float64))
    return mask


def has_forecast_output_columns(df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
    return any(find_dataframe_column(df, column_name) is not None for column_name in forecast_close_columns(metadata))


def prepare_stock_part(
    payload: SheetPayload,
    metadata: Dict[str, Any],
    *,
    latest_only: bool = False,
    refresh_existing_forecasts: bool = False,
    all_eligible_rows: bool = False,
) -> StockInferencePart:
    feature_columns, seq_len, feature_count, _ = validate_metadata(metadata)
    raw_df = payload.frame.copy()
    if raw_df.empty:
        raise ValueError("worksheet has no data rows")

    if SHEET_ROW_COL not in raw_df.columns:
        raw_df[SHEET_ROW_COL] = np.arange(2, len(raw_df) + 2)

    df = prepare_feature_engineering_input(raw_df)
    date_col = detect_date_column(df)
    if date_col is not None:
        df = ensure_date_column(df, date_col)
    else:
        df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("no rows remain after date cleanup")

    df[SORT_POSITION_COL] = np.arange(len(df), dtype=np.int64)
    pred_col = next(
        (column for column in df.columns if str(column).strip().lower() == PREDICTED_COL),
        None,
    )
    if pred_col is None:
        df[PREDICTED_COL] = 0
        pred_col = PREDICTED_COL
        log(f"{payload.name}: predicted column not found; treating existing rows as unprocessed")

    predicted_values = to_numeric_series(df[pred_col]).fillna(0).to_numpy(dtype=np.float64)
    valid_prediction_indices = np.flatnonzero(predicted_values == 1)
    if all_eligible_rows:
        first_valid_position = min(max(seq_len - 1, 0), len(df) - 1)
        log(
            f"{payload.name}: all eligible row forecasting enabled; "
            f"using position {first_valid_position} as the initial prediction checkpoint"
        )
    elif len(valid_prediction_indices) == 0:
        first_valid_position = min(max(seq_len - 1, 0), len(df) - 1)
        log(
            f"{payload.name}: no rows marked predicted=1; "
            f"using position {first_valid_position} as the initial prediction checkpoint"
        )
    else:
        first_valid_position = max(int(valid_prediction_indices.min()), min(max(seq_len - 1, 0), len(df) - 1))

    forecast_missing_values = missing_forecast_mask(df, metadata)
    candidate_df = normalize_columns(df)
    candidate_date_col = canonical_column_name(date_col) if date_col else None
    candidate_positions = to_numeric_series(candidate_df[SORT_POSITION_COL]).astype(np.int64).to_numpy()
    assert np.array_equal(
        np.sort(candidate_positions),
        np.arange(len(df)),
    )
    assert first_valid_position < len(df)

    refresh_existing_forecasts = bool(refresh_existing_forecasts and has_forecast_output_columns(df, metadata))
    # Forecast columns are refreshed when present so older long-horizon outputs
    # are replaced by the 5-day hybrid forecast without changing processed flags.
    valid_indices = np.where(
        (candidate_positions > first_valid_position)
        & ((predicted_values == 0) | forecast_missing_values | refresh_existing_forecasts)
    )[0]
    if latest_only and len(valid_indices) > 0:
        valid_indices = valid_indices[-1:]
    eligible_target_count = int(len(valid_indices))

    if eligible_target_count == 0:
        log(
            f"Prepared {payload.name}: rows={len(raw_df)}, "
            f"first_valid_position={first_valid_position}, eligible_predictions=0"
        )
        return StockInferencePart(
            symbol=payload.name,
            X=np.empty((0, seq_len, feature_count), dtype=np.float32),
            dates=np.asarray([]),
            sheet_rows=np.asarray([], dtype=np.int64),
            forecast_start_positions=np.asarray([], dtype=np.int64),
            actual_price=np.asarray([], dtype=np.float32),
            anchor_close=np.asarray([], dtype=np.float32),
            target_scaler=MinMaxScaler(),
            feature_scaler=MinMaxScaler(),
            current_open_values=np.asarray([], dtype=np.float32),
            headers=payload.headers,
            data_row_count=payload.data_row_count,
            forecast_base_frame=candidate_df.copy(),
            forecast_date_col=candidate_date_col,
            worksheet=payload.worksheet,
            status=STATUS_NO_NEW_DATA,
            eligible_target_count=eligible_target_count,
            last_valid_position=first_valid_position,
        )

    if TARGET_COL not in candidate_df.columns:
        candidate_df[TARGET_COL] = np.nan
    candidate_df[TARGET_COL] = to_numeric_series(candidate_df[TARGET_COL])
    if USE_INTRADAY_OPEN:
        if "Open" not in feature_columns:
            raise ValueError("metadata feature_columns must include Open when USE_INTRADAY_OPEN is True")

    # ── Feature Engineering stage ────────────────────────────────────────────
    # compute_indicators now raises FeatureEngineeringError on missing OHLCV;
    # this propagates to process_payloads which records it in skipped[].
    fe_t0 = time.monotonic()
    log(f"[FE] Starting feature engineering for {payload.name} ({len(df)} rows)")
    with contextlib.redirect_stdout(sys.stderr):
        engineered = compute_indicators(df)
    fe_elapsed = time.monotonic() - fe_t0
    engineered = normalize_columns(engineered)

    # ── Post-FE structural validation ───────────────────────────────────────
    if TARGET_COL not in engineered.columns:
        raise ValueError(
            f"[FE] {payload.name}: Close column missing after feature engineering. "
            f"Available columns: {list(engineered.columns)}"
        )
    if SHEET_ROW_COL not in engineered.columns or SORT_POSITION_COL not in engineered.columns:
        raise ValueError(
            f"[FE] {payload.name}: Internal row-tracking columns were lost during "
            f"feature engineering ({SHEET_ROW_COL}, {SORT_POSITION_COL})."
        )

    # Verify expected indicator columns were actually produced
    _fe_missing = [c for c in INDICATOR_COLUMNS if c not in engineered.columns]
    if _fe_missing:
        log(
            f"[FE] WARNING {payload.name}: {len(_fe_missing)} indicator column(s) absent "
            f"from engineered output: {_fe_missing}"
        )
    else:
        log(
            f"[FE] {payload.name}: all {len(INDICATOR_COLUMNS)} indicator columns present "
            f"({len(engineered)} rows, elapsed={fe_elapsed:.3f}s)"
        )

    leakage_columns = forward_label_like_columns(engineered.columns, metadata)
    overlap = sorted(forward_label_like_columns(feature_columns, metadata))
    if overlap:
        raise ValueError(f"Forward-label columns must not be model features: {overlap}")
    # Forward labels are target-only columns; remove them before building any predictor matrix.
    engineered = engineered.drop(columns=leakage_columns, errors="ignore")

    engineered[TARGET_COL] = to_numeric_series(engineered[TARGET_COL])
    engineered = engineered.dropna(subset=[TARGET_COL])
    if engineered.empty:
        raise ValueError(
            f"[FE] {payload.name}: no valid OHLCV rows remain after feature engineering. "
            "All Close values are NaN or missing."
        )

    engineered_positions = to_numeric_series(engineered[SORT_POSITION_COL]).astype(np.int64).to_numpy()
    historical_mask = engineered_positions <= first_valid_position
    if not historical_mask.any():
        raise ValueError("no historical rows remain at or before the first predicted checkpoint")
    assert int(engineered_positions[historical_mask].max()) <= first_valid_position

    # Scalers are fit only on rows at or before the historical checkpoint.
    target_values = engineered.loc[historical_mask, [TARGET_COL]].to_numpy(dtype=np.float32)
    target_scaler = MinMaxScaler()
    target_scaler.fit(target_values)

    feature_frame = engineered.reindex(columns=feature_columns, fill_value=0)
    feature_frame = to_numeric_frame(feature_frame)
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_frame.loc[historical_mask])
    scaled_features = feature_scaler.transform(feature_frame).astype(np.float32)
    engineered_close = engineered[TARGET_COL].to_numpy(dtype=np.float32)

    order = np.argsort(engineered_positions, kind="stable")
    engineered_positions = engineered_positions[order]
    scaled_features = scaled_features[order]
    engineered_close = engineered_close[order]

    candidate_sheet_rows = to_numeric_series(candidate_df[SHEET_ROW_COL]).astype(np.int64).to_numpy()
    candidate_actual = candidate_df[TARGET_COL].to_numpy(dtype=np.float32)
    candidate_open = (
        to_numeric_series(candidate_df["Open"]).to_numpy(dtype=np.float32)
        if "Open" in candidate_df.columns
        else np.full(len(candidate_df), np.nan, dtype=np.float32)
    )
    candidate_dates = get_dates(candidate_df, candidate_date_col)

    X_sequences: List[np.ndarray] = []
    selected_positions: List[int] = []
    selected_rows: List[int] = []
    selected_actual: List[float] = []
    selected_anchor_close: List[float] = []
    selected_dates: List[Any] = []
    selected_open_values: List[float] = []

    for idx in valid_indices:
        original_position = int(candidate_positions[idx])
        anchor_position = original_position
        anchor_matches = np.flatnonzero(engineered_positions == int(anchor_position))
        if len(anchor_matches) == 0:
            continue
        history_indices = np.flatnonzero(engineered_positions <= int(anchor_position))
        if len(history_indices) < seq_len:
            continue
        window_indices = history_indices[-seq_len:]
        if not np.all(engineered_positions[window_indices] <= anchor_position):
            raise RuntimeError("sequence construction attempted to include a future row")
        if int(engineered_positions[window_indices[-1]]) != anchor_position:
            continue
        sequence = scaled_features[window_indices]
        if sequence.shape != (seq_len, feature_count):
            raise ValueError(f"sequence shape mismatch: expected {(seq_len, feature_count)}, got {sequence.shape}")
        anchor_close_value = float(engineered_close[window_indices[-1]])
        if not np.isfinite(anchor_close_value):
            raise ValueError(f"Anchor close is required before sheet row {int(candidate_sheet_rows[idx])}")
        if USE_INTRADAY_OPEN:
            # The selected row is the observed forecast anchor, so use its real
            # Open when present and fall back to the anchor Close only if absent.
            open_value = float(candidate_open[idx]) if idx < len(candidate_open) else float("nan")
            selected_open_values.append(open_value if np.isfinite(open_value) else float(anchor_close_value))
        X_sequences.append(sequence)
        selected_positions.append(original_position)
        selected_rows.append(int(candidate_sheet_rows[idx]))
        selected_actual.append(float(candidate_actual[idx]) if np.isfinite(candidate_actual[idx]) else np.nan)
        selected_anchor_close.append(anchor_close_value)
        selected_dates.append(candidate_dates[idx] if idx < len(candidate_dates) else idx)

    assert len(selected_positions) == len(set(selected_positions))
    assert all(
        (predicted_values[row_pos] == 0) or forecast_missing_values[row_pos] or refresh_existing_forecasts
        for row_pos in selected_positions
    )

    if X_sequences:
        X = np.stack(X_sequences).astype(np.float32)
    else:
        X = np.empty((0, seq_len, feature_count), dtype=np.float32)

    part_status = STATUS_OK if len(X) > 0 else STATUS_NO_NEW_DATA
    log(
        f"Prepared {payload.name}: rows={len(raw_df)}, "
        f"first_valid_position={first_valid_position}, "
        f"eligible_targets={eligible_target_count}, eligible_predictions={len(X)}"
    )
    return StockInferencePart(
        symbol=payload.name,
        X=X,
        dates=np.asarray(selected_dates),
        sheet_rows=np.asarray(selected_rows, dtype=np.int64),
        forecast_start_positions=np.asarray(selected_positions, dtype=np.int64),
        actual_price=np.asarray(selected_actual, dtype=np.float32),
        anchor_close=np.asarray(selected_anchor_close, dtype=np.float32),
        target_scaler=target_scaler,
        feature_scaler=feature_scaler,
        current_open_values=np.asarray(selected_open_values, dtype=np.float32),
        headers=payload.headers,
        data_row_count=payload.data_row_count,
        forecast_base_frame=candidate_df.copy(),
        forecast_date_col=candidate_date_col,
        worksheet=payload.worksheet,
        status=part_status,
        eligible_target_count=eligible_target_count,
        last_valid_position=first_valid_position,
    )


def validate_shapes(X: np.ndarray, metadata: Dict[str, Any]) -> None:
    _, seq_len, feature_count, dense_input_size = validate_metadata(metadata)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D, got {X.shape}")
    if X.shape[1] != seq_len or X.shape[2] != feature_count:
        raise ValueError(f"Expected X shape (*, {seq_len}, {feature_count}), got {X.shape}")
    actual_dense_size = X.shape[1] * X.shape[2]
    if actual_dense_size != dense_input_size:
        raise ValueError(f"Dense input mismatch: expected {dense_input_size}, got {actual_dense_size}")
    log(f"Combined X shape: {X.shape}; dense flattened size: {actual_dense_size}")


def inject_intraday_open(
    X: np.ndarray,
    current_open_values: Sequence[float],
    feature_scaler: MinMaxScaler,
    feature_columns: Sequence[str],
) -> np.ndarray:
    if not USE_INTRADAY_OPEN:
        return X
    if "Open" not in feature_columns:
        raise ValueError("feature_columns must include Open when USE_INTRADAY_OPEN is True")

    open_idx = list(feature_columns).index("Open")
    open_values = np.asarray(current_open_values, dtype=np.float32).reshape(-1)
    if len(open_values) != len(X):
        raise ValueError(f"Open value count mismatch: expected {len(X)}, got {len(open_values)}")
    if not np.isfinite(open_values).all():
        raise ValueError("Open(t) contains missing or non-finite values")

    current_features = pd.DataFrame(
        np.zeros((len(open_values), len(feature_columns)), dtype=np.float32),
        columns=list(feature_columns),
    )
    # Intraday override is limited to Open[t] for the prediction row.
    current_features.loc[:, "Open"] = open_values
    scaled_values = feature_scaler.transform(current_features).astype(np.float32)

    X[:, -1, open_idx] = scaled_values[:, open_idx]
    return X


def predict_model(model: nn.Module, model_name: str, X: np.ndarray, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    predictions: List[np.ndarray] = []
    expected_horizons = int(getattr(model, "n_horizons", MAX_FORECAST_HORIZON))
    with torch.no_grad():
        for (X_batch,) in loader:
            if X_batch.dim() != 3:
                raise ValueError(f"{model_name}: X batch must be 3D, got {X_batch.shape}")
            output = model(X_batch.to(device))
            output_values = output.detach().cpu().numpy()
            if output_values.ndim != 2 or output_values.shape[0] != len(X_batch):
                raise ValueError(f"{model_name}: expected one horizon vector per row, got output shape {tuple(output.shape)}")
            if output_values.shape[1] != expected_horizons:
                raise ValueError(
                    f"{model_name}: expected {expected_horizons} horizons, got output shape {tuple(output.shape)}"
                )
            predictions.append(output_values)
    if not predictions:
        return np.empty((0, expected_horizons), dtype=np.float32)
    return np.concatenate(predictions, axis=0).astype(np.float32)


def ensemble_predictions(
    model_predictions: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    if not model_predictions:
        return np.empty((0,), dtype=np.float32), {}

    available_weights = {
        model_name: max(0.0, float(weights.get(model_name, 0.0)))
        for model_name in model_predictions
    }
    total_weight = sum(available_weights.values())
    if total_weight <= 0:
        available_weights = {
            model_name: 1.0 / len(model_predictions)
            for model_name in model_predictions
        }
    else:
        available_weights = {
            model_name: weight / total_weight
            for model_name, weight in available_weights.items()
        }

    ensemble = np.zeros_like(next(iter(model_predictions.values())), dtype=np.float32)
    for model_name, predictions in model_predictions.items():
        ensemble += available_weights[model_name] * predictions
    return ensemble, available_weights


def uses_quantile_log_return_target(metadata: Dict[str, Any]) -> bool:
    return str(metadata.get("target_type", "")).strip().lower() == "multi_horizon_quantile_log_return"


def model_output_to_price(values: np.ndarray, part: StockInferencePart, metadata: Dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if uses_quantile_log_return_target(metadata):
        anchor_close = np.asarray(part.anchor_close, dtype=np.float32).reshape(-1)
        if len(anchor_close) != values.shape[0]:
            raise ValueError(f"Anchor close count mismatch: expected {values.shape[0]}, got {len(anchor_close)}")
        if not np.isfinite(anchor_close).all():
            raise ValueError("Anchor close contains missing or non-finite values")
        # q50 outputs are direct forward log returns for T+1 through T+5.
        return anchor_close.astype(np.float64)[:, None] * np.exp(values.astype(np.float64))
    if values.shape[1] != 1:
        raise ValueError(f"Non-quantile price inverse transform expects one output, got {values.shape[1]}")
    return part.target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1, 1)


def forecast_close_columns(metadata: Dict[str, Any]) -> List[str]:
    trained_horizons = metadata_horizons(metadata)
    forecast_days = configured_forecast_days(default=max(trained_horizons))
    return [f"{FORECAST_CLOSE_PREFIX}{horizon}" for horizon in range(1, forecast_days + 1)]


def finite_or_nan(value: Any) -> float:
    try:
        number = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def build_position_lookup(frame: pd.DataFrame) -> Dict[int, pd.Series]:
    if frame is None or frame.empty or SORT_POSITION_COL not in frame.columns:
        return {}
    positions = to_numeric_series(frame[SORT_POSITION_COL]).to_numpy(dtype=np.float64)
    lookup: Dict[int, pd.Series] = {}
    for idx, position in enumerate(positions):
        if np.isfinite(position):
            lookup[int(position)] = frame.iloc[idx]
    return lookup


def row_value(row: Optional[pd.Series], column: str) -> float:
    if row is None or column not in row.index:
        return float("nan")
    return finite_or_nan(row[column])


def last_finite_history_value(history: pd.DataFrame, column: str, fallback: float = float("nan")) -> float:
    if history.empty or column not in history.columns:
        return fallback
    values = to_numeric_series(history[column]).to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return fallback
    return float(finite[-1])


def rolling_history_mean(
    history: pd.DataFrame,
    column: str,
    window: int = ROLLING_STABILIZATION_WINDOW,
    fallback: float = float("nan"),
) -> float:
    if history.empty or column not in history.columns:
        return fallback
    values = to_numeric_series(history[column]).to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return fallback
    return float(np.mean(finite[-window:]))


def recent_range_fraction(history: pd.DataFrame, fallback: float = 0.02) -> float:
    required = {"High", "Low", TARGET_COL}
    if history.empty or not required.issubset(set(history.columns)):
        return fallback
    high = to_numeric_series(history["High"]).to_numpy(dtype=np.float64)
    low = to_numeric_series(history["Low"]).to_numpy(dtype=np.float64)
    close = to_numeric_series(history[TARGET_COL]).to_numpy(dtype=np.float64)
    valid = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (close > 0) & (high >= low)
    if not valid.any():
        return fallback
    ranges = (high[valid] - low[valid]) / close[valid]
    ranges = ranges[np.isfinite(ranges) & (ranges >= 0)]
    if len(ranges) == 0:
        return fallback
    return float(np.clip(np.median(ranges[-ROLLING_STABILIZATION_WINDOW:]), 0.005, 0.05))


def stabilized_high_low(history: pd.DataFrame, open_value: float, close_value: float) -> Tuple[float, float]:
    reference = max(abs(float(open_value)), abs(float(close_value)), 1.0)
    range_margin = reference * recent_range_fraction(history) * 0.5
    body_margin = abs(float(open_value) - float(close_value)) * 0.25
    margin = max(range_margin, body_margin)
    upper_base = max(float(open_value), float(close_value))
    lower_base = min(float(open_value), float(close_value))
    high_value = min(upper_base + margin, upper_base * 1.08)
    low_value = max(lower_base - margin, lower_base * 0.92, 0.01)
    if low_value > high_value:
        high_value = upper_base
        low_value = max(lower_base, 0.01)
    return float(high_value), float(low_value)


def next_rollout_date(history: pd.DataFrame, date_col: Optional[str]) -> Any:
    if not date_col or history.empty or date_col not in history.columns:
        return pd.NaT
    try:
        last_date = pd.to_datetime(history[date_col], errors="coerce").dropna()
    except TypeError:
        last_date = pd.to_datetime(history[date_col], errors="coerce").dropna()
    if last_date.empty:
        return pd.NaT
    return last_date.iloc[-1] + pd.offsets.BDay(1)


def current_step_open(row: Optional[pd.Series], previous_close: float) -> float:
    open_value = row_value(row, "Open")
    if np.isfinite(open_value):
        return float(open_value)
    if not np.isfinite(previous_close):
        raise ValueError("previous close must be finite when Open is unavailable")
    return float(previous_close)


def close_for_rollout_features(row: Optional[pd.Series], predicted_close: float) -> float:
    actual_close = row_value(row, TARGET_COL)
    if np.isfinite(actual_close):
        return float(actual_close)
    return float(predicted_close)


def make_rollout_row(
    template_columns: Sequence[Any],
    history: pd.DataFrame,
    base_row: Optional[pd.Series],
    position: int,
    date_col: Optional[str],
    open_value: float,
    predicted_close: float,
) -> pd.DataFrame:
    row_data: Dict[Any, Any] = {column: np.nan for column in template_columns}
    if base_row is not None:
        row_data.update(base_row.to_dict())

    close_value = close_for_rollout_features(base_row, predicted_close)
    high_value, low_value = stabilized_high_low(history, open_value, close_value)
    volume_value = rolling_history_mean(history, "Volume", fallback=float("nan"))
    if not np.isfinite(volume_value):
        volume_value = last_finite_history_value(history, "Volume", 0.0)

    row_data[SORT_POSITION_COL] = int(position)
    row_data["Open"] = float(open_value)
    row_data["High"] = float(high_value)
    row_data["Low"] = float(low_value)
    row_data[TARGET_COL] = float(close_value)
    row_data["Volume"] = float(volume_value)
    if date_col:
        if base_row is not None and date_col in base_row.index and pd.notna(base_row[date_col]):
            row_data[date_col] = base_row[date_col]
        else:
            row_data[date_col] = next_rollout_date(history, date_col)

    return pd.DataFrame([row_data], columns=list(template_columns))


def scaled_latest_feature_row(
    history: pd.DataFrame,
    part: StockInferencePart,
    metadata: Dict[str, Any],
) -> np.ndarray:
    feature_columns, _, feature_count, _ = validate_metadata(metadata)
    if history.empty:
        raise ValueError(f"{part.symbol}: recursive history is empty")
    with contextlib.redirect_stdout(sys.stderr):
        engineered = compute_indicators(history)
    engineered = normalize_columns(engineered)
    leakage_columns = forward_label_like_columns(engineered.columns, metadata)
    engineered = engineered.drop(columns=leakage_columns, errors="ignore")
    if engineered.empty:
        raise ValueError(f"{part.symbol}: recursive feature engineering produced no rows")
    feature_frame = engineered.reindex(columns=feature_columns, fill_value=0)
    feature_frame = to_numeric_frame(feature_frame)
    scaled = part.feature_scaler.transform(feature_frame).astype(np.float32)
    latest = scaled[-1]
    if latest.shape != (feature_count,):
        raise ValueError(f"{part.symbol}: recursive feature row shape mismatch: {latest.shape}")
    if not np.isfinite(latest).all():
        raise ValueError(f"{part.symbol}: recursive feature row contains non-finite values")
    return latest


def one_step_outputs_to_price(values: np.ndarray, previous_close: np.ndarray, part: StockInferencePart, metadata: Dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if uses_quantile_log_return_target(metadata):
        anchors = np.asarray(previous_close, dtype=np.float64).reshape(-1)
        if len(anchors) != len(values):
            raise ValueError(f"Anchor close count mismatch: expected {len(values)}, got {len(anchors)}")
        if not np.isfinite(anchors).all():
            raise ValueError("Anchor close contains missing or non-finite values")
        clipped_returns = np.clip(
            values.astype(np.float64),
            RECURSIVE_RETURN_CLIP[0],
            RECURSIVE_RETURN_CLIP[1],
        )
        return anchors * np.exp(clipped_returns)
    return part.target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def direct_forecast_part(
    part: StockInferencePart,
    models: Dict[str, nn.Module],
    metadata: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    feature_columns, seq_len, feature_count, _ = validate_metadata(metadata)
    forecast_steps = len(forecast_close_columns(metadata))
    trained_steps = direct_trained_horizon_count(metadata)
    direct_steps = min(trained_steps, forecast_steps)
    if len(part.X) == 0:
        return (
            np.empty((0, forecast_steps), dtype=np.float32),
            np.empty((0, forecast_steps), dtype=np.float32),
            {},
            {},
        )

    X_initial = np.asarray(part.X, dtype=np.float32).copy()
    if USE_INTRADAY_OPEN:
        X_initial = inject_intraday_open(
            X_initial,
            part.current_open_values,
            part.feature_scaler,
            feature_columns,
        )
    if X_initial.shape != (len(part.X), seq_len, feature_count):
        raise ValueError(
            f"{part.symbol}: direct X shape mismatch: expected {(len(part.X), seq_len, feature_count)}, "
            f"got {X_initial.shape}"
        )
    validate_shapes(X_initial, metadata)

    model_predictions = {
        model_name: predict_model(model, model_name, X_initial, device)
        for model_name, model in models.items()
    }
    ensemble_scaled, weights_used = ensemble_predictions(
        model_predictions,
        metadata.get("ensemble_weights", {}),
    )
    if ensemble_scaled.shape != (len(part.X), trained_steps):
        raise ValueError(
            f"{part.symbol}: direct model output shape mismatch: expected {(len(part.X), trained_steps)}, "
            f"got {ensemble_scaled.shape}"
        )
    direct_prices = model_output_to_price(ensemble_scaled, part, metadata).astype(np.float32)
    if direct_prices.shape != (len(part.X), trained_steps):
        raise ValueError(
            f"{part.symbol}: direct price shape mismatch: expected {(len(part.X), trained_steps)}, "
            f"got {direct_prices.shape}"
        )
    price_matrix = np.full((len(part.X), forecast_steps), np.nan, dtype=np.float32)
    scaled_matrix = np.full((len(part.X), forecast_steps), np.nan, dtype=np.float32)
    price_matrix[:, :direct_steps] = direct_prices[:, :direct_steps]
    scaled_matrix[:, :direct_steps] = ensemble_scaled[:, :direct_steps]
    if not np.isfinite(price_matrix).all():
        finite_direct = np.isfinite(price_matrix[:, :direct_steps]).all()
        if not finite_direct:
            raise ValueError(f"{part.symbol}: direct forecast prices contain non-finite values")
    return price_matrix, scaled_matrix.astype(np.float32), model_predictions, weights_used


def recursive_forecast_part(
    part: StockInferencePart,
    models: Dict[str, nn.Module],
    metadata: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    feature_columns, seq_len, feature_count, _ = validate_metadata(metadata)
    forecast_steps = len(forecast_close_columns(metadata))
    direct_steps = min(direct_trained_horizon_count(metadata), forecast_steps)
    count = len(part.X)
    if count == 0:
        return (
            np.empty((0, forecast_steps), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            {},
            {},
        )

    if part.forecast_base_frame is None or part.forecast_base_frame.empty:
        raise ValueError(f"{part.symbol}: recursive forecasting requires a source frame")
    if len(part.forecast_start_positions) != count:
        raise ValueError(f"{part.symbol}: forecast start position count mismatch")

    forecast_matrix, direct_scaled, direct_model_scaled, weights_used = direct_forecast_part(
        part,
        models,
        metadata,
        device,
    )
    if forecast_steps <= direct_steps:
        log(f"{part.symbol}: hybrid forecast uses direct trained horizons T+1..T+{forecast_steps}")
    else:
        log(
            f"{part.symbol}: hybrid forecast uses direct trained horizons T+1..T+{direct_steps} "
            f"and recursive one-step continuation T+{direct_steps + 1}..T+{forecast_steps}"
        )

    base_frame = part.forecast_base_frame.copy()
    base_positions = to_numeric_series(base_frame[SORT_POSITION_COL]).to_numpy(dtype=np.float64)
    base_lookup = build_position_lookup(base_frame)
    template_columns = list(base_frame.columns)

    states: List[Dict[str, Any]] = []
    for local_idx in range(count):
        start_position = int(part.forecast_start_positions[local_idx])
        history = base_frame.loc[np.isfinite(base_positions) & (base_positions <= start_position)].copy()
        history = history.sort_values(SORT_POSITION_COL, kind="stable").reset_index(drop=True)
        previous_close = float(part.anchor_close[local_idx])
        if not np.isfinite(previous_close):
            raise ValueError(f"{part.symbol}: previous close is required for recursive forecasting")
        sequence = np.asarray(part.X[local_idx], dtype=np.float32).copy()
        if sequence.shape != (seq_len, feature_count):
            raise ValueError(f"{part.symbol}: recursive sequence shape mismatch: {sequence.shape}")
        states.append(
            {
                "start_position": start_position,
                "history": history,
                "previous_close": previous_close,
                "sequence": sequence,
                "forecasts": [],
                "first_scaled": float(direct_scaled[local_idx, 0]) if direct_scaled.size else np.nan,
                "first_model_scaled": {
                    model_name: float(predictions[local_idx, 0])
                    for model_name, predictions in direct_model_scaled.items()
                },
            }
        )

    for step_idx in range(forecast_steps):
        step_sequences: List[np.ndarray] = []
        step_previous_close: List[float] = []
        step_open_values: List[float] = []
        step_rows: List[Optional[pd.Series]] = []

        for state in states:
            position = int(state["start_position"]) + step_idx + 1
            base_row = base_lookup.get(position)
            previous_close = float(state["previous_close"])
            open_value = current_step_open(base_row, previous_close)
            sequence = np.asarray(state["sequence"], dtype=np.float32).copy()
            sequence = inject_intraday_open(
                sequence.reshape(1, seq_len, feature_count),
                [open_value],
                part.feature_scaler,
                feature_columns,
            )[0]
            step_sequences.append(sequence)
            step_previous_close.append(previous_close)
            step_open_values.append(float(open_value))
            step_rows.append(base_row)

        if step_idx < direct_steps:
            predicted_close_values = forecast_matrix[:, step_idx].astype(np.float64)
            if not np.isfinite(predicted_close_values).all():
                raise ValueError(f"{part.symbol}: direct forecast prices contain non-finite values")
        else:
            X_step = np.stack(step_sequences).astype(np.float32)
            if step_idx == direct_steps:
                validate_shapes(X_step, metadata)
            elif X_step.shape != (count, seq_len, feature_count):
                raise ValueError(
                    f"{part.symbol}: recursive X shape mismatch at step {step_idx + 1}: "
                    f"expected {(count, seq_len, feature_count)}, got {X_step.shape}"
                )
            model_predictions = {
                model_name: predict_model(model, model_name, X_step, device)
                for model_name, model in models.items()
            }
            ensemble_scaled, weights_used = ensemble_predictions(
                model_predictions,
                metadata.get("ensemble_weights", {}),
            )
            one_step_scaled = ensemble_scaled[:, 0]
            predicted_close_values = one_step_outputs_to_price(
                one_step_scaled,
                np.asarray(step_previous_close, dtype=np.float64),
                part,
                metadata,
            )
            if not np.isfinite(predicted_close_values).all():
                raise ValueError(f"{part.symbol}: recursive forecast prices contain non-finite values")

        for local_idx, state in enumerate(states):
            predicted_close = float(predicted_close_values[local_idx])
            state["forecasts"].append(predicted_close)
            forecast_matrix[local_idx, step_idx] = predicted_close

            if step_idx + 1 >= forecast_steps:
                continue
            position = int(state["start_position"]) + step_idx + 1
            rollout_row = make_rollout_row(
                template_columns,
                state["history"],
                step_rows[local_idx],
                position,
                part.forecast_date_col,
                step_open_values[local_idx],
                predicted_close,
            )
            state["history"] = pd.concat([state["history"], rollout_row], ignore_index=True)
            state["previous_close"] = float(rollout_row.iloc[0][TARGET_COL])
            latest_scaled = scaled_latest_feature_row(state["history"], part, metadata)
            state["sequence"] = np.concatenate(
                [np.asarray(state["sequence"], dtype=np.float32)[1:], latest_scaled.reshape(1, -1)],
                axis=0,
            ).astype(np.float32)

    first_scaled = np.asarray([state["first_scaled"] for state in states], dtype=np.float32)
    first_model_scaled: Dict[str, np.ndarray] = {}
    for model_name in models:
        first_model_scaled[model_name] = np.asarray(
            [state["first_model_scaled"].get(model_name, np.nan) for state in states],
            dtype=np.float32,
        )
    return forecast_matrix, first_scaled, first_model_scaled, weights_used


def run_inference(
    parts: List[StockInferencePart],
    models: Dict[str, nn.Module],
    metadata: Dict[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    prediction_parts = [part for part in parts if len(part.X) > 0]
    if not prediction_parts:
        return empty_results_frame(), {}

    forecast_columns = forecast_close_columns(metadata)

    result_frames: List[pd.DataFrame] = []
    weights_used: Dict[str, float] = {}
    for part in prediction_parts:
        count = len(part.X)
        stock_price_matrix, stock_scaled_h1, first_model_scaled, part_weights = recursive_forecast_part(
            part,
            models,
            metadata,
            device,
        )
        if part_weights:
            weights_used = part_weights
        if stock_price_matrix.shape != (count, len(forecast_columns)):
            raise ValueError(
                f"forecast price shape mismatch: expected {(count, len(forecast_columns))}, "
                f"got {stock_price_matrix.shape}"
            )
        if not np.isfinite(stock_price_matrix).all():
            raise ValueError(f"{part.symbol}: forecast prices contain non-finite values")
        stock_price_h1 = stock_price_matrix[:, 0]
        part.row_predictions = {
            int(row_number): float(price)
            for row_number, price in zip(part.sheet_rows, stock_price_h1)
        }
        part.row_forecasts = {
            int(row_number): [float(value) for value in forecast_values]
            for row_number, forecast_values in zip(part.sheet_rows, stock_price_matrix)
        }

        result = pd.DataFrame(
            {
                "Symbol": part.symbol,
                "Sheet_Row": part.sheet_rows,
                "Date": pd.to_datetime(part.dates, errors="coerce", format="mixed"),
                "Actual_Price": part.actual_price,
                "Predicted_Scaled": stock_scaled_h1,
                "Predicted_Price": stock_price_h1.astype(np.float32),
            }
        )
        for col_idx, column in enumerate(forecast_columns):
            result[column] = stock_price_matrix[:, col_idx]
        for model_name, predictions in first_model_scaled.items():
            result[f"{model_name}_Scaled"] = predictions
        result_frames.append(result)

    results = pd.concat(result_frames, ignore_index=True)
    results.insert(0, "Sample_Index", np.arange(len(results)))
    results["Absolute_Error"] = np.where(
        results["Actual_Price"].notna(),
        np.abs(results["Actual_Price"] - results["Predicted_Price"]),
        np.nan,
    )
    results["Squared_Error"] = np.where(
        results["Actual_Price"].notna(),
        (results["Actual_Price"] - results["Predicted_Price"]) ** 2,
        np.nan,
    )
    return results, weights_used


def empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Sample_Index",
            "Symbol",
            "Sheet_Row",
            "Date",
            "Actual_Price",
            "Predicted_Scaled",
            "Predicted_Price",
            "Absolute_Error",
            "Squared_Error",
        ]
    )


def metric_value(value: float) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def calculate_metrics(y_true: Sequence[float], y_pred: Sequence[float], predictor_count: int) -> Dict[str, Optional[float]]:
    y_true_array = np.asarray(y_true, dtype=np.float64)
    y_pred_array = np.asarray(y_pred, dtype=np.float64)
    valid_mask = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
    y_true_array = y_true_array[valid_mask]
    y_pred_array = y_pred_array[valid_mask]
    n = len(y_true_array)
    if n == 0:
        return {
            "n": 0,
            "MSE": None,
            "RMSE": None,
            "MAE": None,
            "R2": None,
            "Adjusted_R2": None,
            "MAPE": None,
        }

    mse = mean_squared_error(y_true_array, y_pred_array)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_array, y_pred_array)
    r2 = r2_score(y_true_array, y_pred_array) if n > 1 and len(np.unique(y_true_array)) > 1 else np.nan
    adjusted_r2 = np.nan
    if n > predictor_count + 1 and np.isfinite(r2):
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - predictor_count - 1)
    nonzero_mask = y_true_array != 0
    mape = np.nan
    if nonzero_mask.any():
        mape = np.mean(np.abs((y_true_array[nonzero_mask] - y_pred_array[nonzero_mask]) / y_true_array[nonzero_mask])) * 100
    return {
        "n": int(n),
        "MSE": metric_value(mse),
        "RMSE": metric_value(rmse),
        "MAE": metric_value(mae),
        "R2": metric_value(r2),
        "Adjusted_R2": metric_value(adjusted_r2),
        "MAPE": metric_value(mape),
    }


def metrics_by_stock(results: pd.DataFrame, predictor_count: int) -> Dict[str, Dict[str, Optional[float]]]:
    stock_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    if results.empty:
        return stock_metrics
    for symbol, group in results.groupby("Symbol", sort=False):
        stock_metrics[str(symbol)] = calculate_metrics(
            group["Actual_Price"].to_numpy(),
            group["Predicted_Price"].to_numpy(),
            predictor_count,
        )
    return stock_metrics


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    return value


def write_metrics(
    output_dir: Path,
    results: pd.DataFrame,
    metadata: Dict[str, Any],
    weights_used: Dict[str, float],
    skipped: Iterable[str],
    source: str,
) -> Dict[str, Any]:
    predictor_count = int(metadata["seq_len"]) * int(metadata["feature_count"])
    metrics = {
        "source": source,
        "predictor_count": predictor_count,
        "weights_used": weights_used,
        "overall": calculate_metrics(
            results["Actual_Price"].to_numpy() if "Actual_Price" in results else [],
            results["Predicted_Price"].to_numpy() if "Predicted_Price" in results else [],
            predictor_count,
        ),
        "per_stock": metrics_by_stock(results, predictor_count),
        "skipped": list(skipped),
    }
    metrics_path = output_dir / "metrics.json"
    tmp_metrics = metrics_path.with_suffix(".tmp.json")
    try:
        with open(tmp_metrics, "w") as file:
            json.dump(sanitize_for_json(metrics), file, indent=2, allow_nan=False)
        tmp_metrics.replace(metrics_path)
    except Exception:
        tmp_metrics.unlink(missing_ok=True)
        raise
    log(f"Saved metrics (atomic): {metrics_path}")
    return metrics


def write_plotly_graphs(output_dir: Path, results: pd.DataFrame) -> None:
    plot_results = results.dropna(subset=["Actual_Price", "Predicted_Price"]).copy()
    if plot_results.empty:
        log("Skipping plots: no rows have both actual and predicted prices")
        return

    template = "plotly_white"
    line_fig = go.Figure()
    line_fig.add_trace(
        go.Scatter(
            x=plot_results["Sample_Index"],
            y=plot_results["Actual_Price"],
            mode="lines",
            name="Actual Price",
            line={"color": "#636EFA", "width": 2},
            customdata=np.stack(
                [
                    plot_results["Symbol"].astype(str),
                    plot_results["Date"].astype(str),
                    plot_results["Predicted_Price"],
                    plot_results["Absolute_Error"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Sample=%{x}<br>Actual=%{y:.4f}<br>"
                "Predicted=%{customdata[2]:.4f}<br>Error=%{customdata[3]:.4f}<br>"
                "Symbol=%{customdata[0]}<br>Date=%{customdata[1]}<extra></extra>"
            ),
        )
    )
    line_fig.add_trace(
        go.Scatter(
            x=plot_results["Sample_Index"],
            y=plot_results["Predicted_Price"],
            mode="lines",
            name="Predicted Price",
            line={"color": "#EF553B", "width": 2, "dash": "dot"},
        )
    )
    line_fig.update_layout(
        title="Actual Price vs Time and Predicted Price vs Time",
        xaxis_title="Time",
        yaxis_title="Close Price",
        template=template,
        hovermode="x unified",
        legend_title_text="Series",
        width=1500,
        height=650,
    )
    line_path = output_dir / "actual_vs_predicted_price_over_time.html"
    line_fig.write_html(line_path)

    scatter_fig = px.scatter(
        plot_results,
        x="Actual_Price",
        y="Predicted_Price",
        color="Symbol",
        hover_data=["Date", "Absolute_Error"],
        title="Actual vs Predicted Close Price",
        template=template,
        opacity=0.6,
    )
    min_price = min(plot_results["Actual_Price"].min(), plot_results["Predicted_Price"].min())
    max_price = max(plot_results["Actual_Price"].max(), plot_results["Predicted_Price"].max())
    scatter_fig.add_trace(
        go.Scatter(
            x=[min_price, max_price],
            y=[min_price, max_price],
            mode="lines",
            name="Perfect Prediction",
            line={"color": "black", "dash": "dash"},
        )
    )
    scatter_path = output_dir / "actual_vs_predicted_scatter.html"
    scatter_fig.write_html(scatter_path)

    error_fig = px.histogram(
        plot_results,
        x="Absolute_Error",
        color="Symbol",
        nbins=60,
        title="Absolute Error Distribution",
        template=template,
    )
    error_fig.update_layout(xaxis_title="Absolute Error", yaxis_title="Count", bargap=0.02)
    error_path = output_dir / "absolute_error_distribution.html"
    error_fig.write_html(error_path)
    log(f"Saved graph: {line_path}")
    log(f"Saved graph: {scatter_path}")
    log(f"Saved graph: {error_path}")


def make_unique_headers(headers: Sequence[Any]) -> List[str]:
    counts: Dict[str, int] = {}
    unique: List[str] = []
    for idx, header in enumerate(headers, start=1):
        base = str(header).strip() or f"Unnamed_{idx}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        unique.append(base if count == 0 else f"{base}_{count + 1}")
    return unique


def trim_sheet_values(values: List[List[Any]]) -> Tuple[List[str], List[List[Any]]]:
    if not values:
        return [], []
    width = max(len(row) for row in values)
    padded = [list(row) + [""] * (width - len(row)) for row in values]
    last_used = -1
    for row in padded:
        for idx, value in enumerate(row):
            if str(value).strip():
                last_used = max(last_used, idx)
    if last_used < 0:
        return [], []
    headers = [str(value).strip() for value in padded[0][: last_used + 1]]
    rows = [row[: last_used + 1] for row in padded[1:]]
    return headers, rows


def values_to_payload(name: str, values: List[List[Any]], worksheet: Any = None) -> SheetPayload:
    headers, rows = trim_sheet_values(values)
    if not headers:
        return SheetPayload(name=name, frame=pd.DataFrame(), headers=[], data_row_count=0, worksheet=worksheet)

    unique_headers = make_unique_headers(headers)
    records: List[List[Any]] = []
    sheet_rows: List[int] = []
    for row_number, row in enumerate(rows, start=2):
        if not any(str(value).strip() for value in row):
            continue
        records.append(row)
        sheet_rows.append(row_number)

    frame = pd.DataFrame(records, columns=unique_headers)
    if records:
        frame[SHEET_ROW_COL] = sheet_rows
    return SheetPayload(
        name=name,
        frame=frame,
        headers=headers,
        data_row_count=len(rows),
        worksheet=worksheet,
    )


def authorize_gspread(credentials_path: Optional[str]) -> Any:
    try:
        import gspread
        import google.auth
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise RuntimeError("Google Sheets support requires gspread and google-auth packages") from exc

    credential_file = (
        credentials_path
        or os.environ.get("GOOGLE_CREDENTIALS")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if credential_file:
        print(f"[google_auth] Using credentials : {credential_file}", flush=True)
        creds = Credentials.from_service_account_file(credential_file, scopes=GOOGLE_SCOPES)
        print("[google_auth] Service Account Loaded Successfully", flush=True)
    else:
        creds, _ = google.auth.default(scopes=GOOGLE_SCOPES)
    return gspread.authorize(creds)


def load_google_payloads(
    sheet_id: str,
    credentials_path: Optional[str],
    worksheet_filter: Optional[str] = None,
    worksheet_filters: Optional[Set[str]] = None,
) -> List[SheetPayload]:
    client = authorize_gspread(credentials_path)
    spreadsheet = client.open_by_key(sheet_id)
    requested = worksheet_filters or ({worksheet_filter.upper()} if worksheet_filter else None)
    payloads: List[SheetPayload] = []

    for worksheet in spreadsheet.worksheets():
        if requested and worksheet.title.upper() not in requested:
            continue
        values = worksheet.get_all_values()
        payloads.append(values_to_payload(worksheet.title, values, worksheet=worksheet))

    if requested and not payloads:
        raise RuntimeError(f"Worksheet(s) not found: {', '.join(sorted(requested))}")
    return payloads


def load_workbook_payloads(
    workbook: Path,
    worksheet_filter: Optional[str] = None,
    worksheet_filters: Optional[Set[str]] = None,
) -> List[SheetPayload]:
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")
    requested = worksheet_filters or ({worksheet_filter.upper()} if worksheet_filter else None)
    sheets = pd.read_excel(workbook, sheet_name=None)
    payloads: List[SheetPayload] = []
    for sheet_name, frame in sheets.items():
        if requested and sheet_name.upper() not in requested:
            continue
        df = frame.copy()
        df[SHEET_ROW_COL] = np.arange(2, len(df) + 2)
        payloads.append(
            SheetPayload(
                name=sheet_name,
                frame=df,
                headers=[str(column) for column in frame.columns],
                data_row_count=len(df),
            )
        )
    if requested and not payloads:
        raise RuntimeError(f"Worksheet(s) not found in workbook: {', '.join(sorted(requested))}")
    return payloads


def column_number_to_letter(number: int) -> str:
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def find_header_index(headers: List[str], column_name: str) -> Optional[int]:
    wanted = column_name.strip().lower()
    for idx, header in enumerate(headers):
        if str(header).strip().lower() == wanted:
            return idx
    return None


def find_date_header_index(headers: Sequence[Any]) -> Optional[int]:
    for idx, header in enumerate(headers):
        text = str(header).strip().lower()
        if text == "date" or "date" in text or "time" in text:
            return idx
    return None


def parse_sheet_date(value: Any) -> Optional[pd.Timestamp]:
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        if len(parsed) == 0 or pd.isna(parsed[0]):
            return None
        parsed = parsed[0]
    return pd.Timestamp(parsed)


def cleanup_google_sheet_latest_rows(
    worksheet: Any,
    *,
    keep_rows: Optional[int] = None,
    archive_spreadsheet: Any = None,
    archive_worksheet: Any = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if worksheet is None:
        return {"status": "skipped", "reason": "worksheet_missing", "rows_deleted": 0}
    keep = rolling_operational_rows(LATEST_SHEET_ROWS_TO_KEEP) if keep_rows is None else keep_rows
    result = archive_old_rows_for_worksheet(
        worksheet,
        archive_spreadsheet=archive_spreadsheet,
        archive_worksheet=archive_worksheet,
        keep_rows=int(keep),
        dry_run=dry_run,
    )
    return {
        "status": "ok" if result.status == "ok" else result.status,
        "reason": result.reason,
        "rows_deleted": int(result.rows_removed_from_operational),
        "valid_rows_before": int(result.valid_rows),
        "invalid_rows_before": int(result.malformed_rows),
        "rows_remaining": int(result.rolling_limit),
        "archived_rows": int(result.rows_appended),
        "duplicate_rows_skipped": int(result.duplicate_rows_skipped),
        "historical_schema_changed": bool(result.historical_schema_changed),
        "historical_sorted": bool(result.historical_sorted),
    }


def ensure_output_headers(headers: List[str]) -> Tuple[List[str], int, int]:
    updated = list(headers)
    predicted_idx = find_header_index(updated, PREDICTED_COL)
    if predicted_idx is None:
        updated.append(PREDICTED_COL)
        predicted_idx = len(updated) - 1

    price_idx = find_header_index(updated, PREDICTED_PRICE_COL)
    if price_idx is None:
        updated.append(PREDICTED_PRICE_COL)
        price_idx = len(updated) - 1
    return updated, predicted_idx, price_idx


def update_google_predictions(part: StockInferencePart) -> int:
    forecast_count = 0
    if part.row_forecasts:
        forecast_count = len(next(iter(part.row_forecasts.values())))
    service_module = sys.modules.get(PredictionSheetUpdateService.__module__)
    original_forecast_columns = getattr(service_module, "forecast_close_columns", None) if service_module else None
    if forecast_count and original_forecast_columns is not None:
        def _forecast_columns_for_part() -> List[str]:
            return [f"{FORECAST_CLOSE_PREFIX}{horizon}" for horizon in range(1, forecast_count + 1)]

        setattr(service_module, "forecast_close_columns", _forecast_columns_for_part)
    try:
        updated_count = PredictionSheetUpdateService().update_prediction_status(
            worksheet=part.worksheet,
            headers=part.headers,
            row_predictions=part.row_predictions,
            row_forecasts=part.row_forecasts,
        )
    finally:
        if original_forecast_columns is not None and service_module is not None:
            setattr(service_module, "forecast_close_columns", original_forecast_columns)
    if updated_count:
        log(f"Updated Google worksheet {part.symbol}: predicted_rows={updated_count}")
    return updated_count


def process_payloads(
    payloads: List[SheetPayload],
    metadata: Dict[str, Any],
    *,
    latest_only: bool = False,
    refresh_existing_forecasts: bool = False,
    all_eligible_rows: bool = False,
) -> Tuple[List[StockInferencePart], List[str]]:
    parts: List[StockInferencePart] = []
    skipped: List[str] = []
    for payload in payloads:
        if payload.frame.empty:
            skipped.append(f"{payload.name}: worksheet is empty")
            log(f"Skipped {payload.name}: worksheet is empty")
            continue
        try:
            parts.append(
                prepare_stock_part(
                    payload,
                    metadata,
                    latest_only=latest_only,
                    refresh_existing_forecasts=refresh_existing_forecasts,
                    all_eligible_rows=all_eligible_rows,
                )
            )
        except NoValidStartPointError as exc:
            skipped.append(f"{payload.name}: {exc}")
            log(f"Skipped {payload.name}: {exc}")
        except Exception as exc:
            skipped.append(f"{payload.name}: {exc}")
            log(f"Skipped {payload.name}: {exc}")
    return parts, skipped


def write_predictions(output_dir: Path, results: pd.DataFrame) -> Path:
    predictions_path = output_dir / "predictions.csv"
    tmp_path = predictions_path.with_suffix(".tmp.csv")
    try:
        results.to_csv(tmp_path, index=False)
        tmp_path.replace(predictions_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    log(f"Saved predictions (atomic): {predictions_path}")
    return predictions_path


def split_skipped_entry(entry: str) -> Tuple[str, str]:
    if ": " not in entry:
        return entry, entry
    worksheet, reason = entry.split(": ", 1)
    return worksheet, reason


def determine_summary_status(parts: List[StockInferencePart], skipped: List[str], predicted_count: int) -> str:
    if predicted_count > 0:
        return STATUS_OK

    has_valid_checkpoint = any(part.last_valid_position is not None for part in parts)
    has_no_valid_start = any(
        split_skipped_entry(entry)[1] == STATUS_NO_VALID_START_POINT
        for entry in skipped
    )
    if has_no_valid_start and not has_valid_checkpoint:
        return STATUS_NO_VALID_START_POINT
    return STATUS_NO_NEW_DATA


def build_worksheet_debug(parts: List[StockInferencePart], skipped: List[str]) -> List[Dict[str, Any]]:
    debug: List[Dict[str, Any]] = [
        {
            "worksheet": part.symbol,
            "status": part.status,
            "last_valid_position": part.last_valid_position,
            "eligible_target_count": part.eligible_target_count,
            "prepared_prediction_count": int(len(part.X)),
            "number_of_rows_processed": int(len(part.row_predictions)),
        }
        for part in parts
    ]
    for entry in skipped:
        worksheet, reason = split_skipped_entry(entry)
        debug.append(
            {
                "worksheet": worksheet,
                "status": reason if reason == STATUS_NO_VALID_START_POINT else "skipped",
                "reason": reason,
                "number_of_rows_processed": 0,
            }
        )
    return debug


def build_summary(
    args: argparse.Namespace,
    parts: List[StockInferencePart],
    skipped: List[str],
    weights_used: Dict[str, float],
    metrics_path: Path,
    predictions_path: Path,
    sheet_updates_written: bool,
    sheet_cleanup_results: Optional[List[Dict[str, Any]]] = None,
    sheet_update_errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    predicted_count = sum(len(part.row_predictions) for part in parts)
    status = determine_summary_status(parts, skipped, predicted_count)
    return {
        "status": status,
        "source": args.source,
        "sheet_id": args.sheet_id if args.source == "google" else None,
        "worksheet": args.worksheet,
        "worksheets": getattr(args, "worksheets", None),
        "processed_worksheet_count": len(parts),
        "number_of_rows_processed": predicted_count,
        "predicted_row_count": predicted_count,
        "skipped": skipped,
        "worksheet_debug": build_worksheet_debug(parts, skipped),
        "metrics_path": str(metrics_path),
        "predictions_csv_path": str(predictions_path),
        "weights_used": weights_used,
        "sheet_updates_written": sheet_updates_written,
        "sheet_cleanup_results": sheet_cleanup_results or [],
        "sheet_update_errors": sheet_update_errors or [],
    }


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata)
    model_dir = Path(args.model_dir)

    # Self-healing load: auto-creates from defaults if absent, recovers from
    # corruption, and applies schema migrations.  validate_metadata() below
    # provides the hard model-architecture check on top.
    metadata = get_or_create_metadata(metadata_path)
    validate_metadata(metadata)

    device = choose_device(args.device)
    log(f"Device: {device}")
    worksheet_filters = parse_worksheet_filters(getattr(args, "worksheets", None), getattr(args, "worksheet", None))

    if args.source == "google":
        payloads = load_google_payloads(
            args.sheet_id,
            args.google_credentials,
            getattr(args, "worksheet", None),
            worksheet_filters=worksheet_filters,
        )
    else:
        payloads = load_workbook_payloads(
            Path(args.workbook),
            getattr(args, "worksheet", None),
            worksheet_filters=worksheet_filters,
        )

    parts, skipped = process_payloads(
        payloads,
        metadata,
        latest_only=bool(getattr(args, "latest_only", False)),
        refresh_existing_forecasts=bool(getattr(args, "refresh_existing_forecasts", False)),
        all_eligible_rows=bool(getattr(args, "all_eligible_rows", False)),
    )
    has_prediction_data = any(len(part.X) > 0 for part in parts)
    if has_prediction_data:
        models = get_models(model_dir, metadata, device)
        results, weights_used = run_inference(parts, models, metadata, device)
    else:
        results = empty_results_frame()
        weights_used = {}
        log("No eligible prediction rows found; skipping model loading and inference")

    predictions_path = write_predictions(output_dir, results)
    write_metrics(output_dir, results, metadata, weights_used, skipped, args.source)
    metrics_path = output_dir / "metrics.json"
    if args.plots:
        write_plotly_graphs(output_dir, results)

    sheet_updates_written = False
    sheet_cleanup_results: List[Dict[str, Any]] = []
    sheet_update_errors: List[Dict[str, Any]] = []
    archive_spreadsheet = None
    archive_sheet_id = (
        os.environ.get("HISTORICAL_TRAINING_SHEET_ID")
        or os.environ.get("TRAINING_SHEET_ID")
        or DEFAULT_HISTORICAL_TRAINING_SHEET_ID
    )
    if args.source == "google" and not args.dry_run:
        try:
            archive_spreadsheet = authorize_gspread(args.google_credentials).open_by_key(archive_sheet_id)
        except Exception as exc:
            log(f"Historical archive spreadsheet unavailable; operational cleanup will be skipped: {exc}")
    if args.source == "google" and not args.dry_run:
        updated_row_count = 0
        for part in parts:
            try:
                part_updated_count = update_google_predictions(part)
            except Exception as exc:
                message = f"{part.symbol}: sheet update failed: {exc}"
                skipped.append(message)
                sheet_update_errors.append({"worksheet": part.symbol, "error": str(exc)})
                log(message)
                continue

            updated_row_count += part_updated_count
            if part_updated_count <= 0:
                continue

            try:
                archive_worksheet = (
                    find_worksheet_by_title(archive_spreadsheet, part.symbol)
                    if archive_spreadsheet is not None
                    else None
                )
                cleanup_result = cleanup_google_sheet_latest_rows(
                    part.worksheet,
                    archive_spreadsheet=archive_spreadsheet,
                    archive_worksheet=archive_worksheet,
                )
                cleanup_result["worksheet"] = part.symbol
                cleanup_result["historical_sheet_id"] = archive_sheet_id
                sheet_cleanup_results.append(cleanup_result)
                if cleanup_result.get("status") == STATUS_OK:
                    log(
                        f"Cleaned Google worksheet {part.symbol}: "
                        f"rows_deleted={cleanup_result.get('rows_deleted', 0)}"
                    )
                else:
                    log(
                        f"Skipped Google worksheet cleanup {part.symbol}: "
                        f"{cleanup_result.get('reason', 'unknown')}"
                    )
            except Exception as exc:
                cleanup_error = {"worksheet": part.symbol, "status": "error", "error": str(exc)}
                sheet_cleanup_results.append(cleanup_error)
                log(f"Google worksheet cleanup failed for {part.symbol}: {exc}")
        sheet_updates_written = updated_row_count > 0
    elif args.source == "google" and args.dry_run:
        log("Dry run enabled: Google Sheet updates were not written")

    # Decision-layer feature sync: write the 11 categorical decision columns
    # into BOTH Google Sheets. This is isolated from forecasting — any failure
    # is logged and recorded but never aborts the pipeline, and the new columns
    # are deliberately kept out of pipeline_metadata.json / model features.
    decision_feature_sync: Dict[str, Any] = {}
    if args.source == "google" and not args.dry_run:
        try:
            operational_spreadsheet = authorize_gspread(
                args.google_credentials
            ).open_by_key(args.sheet_id)
            decision_feature_sync = sync_decision_features(
                operational_spreadsheet, archive_spreadsheet
            )
            log(f"Decision feature sync: {decision_feature_sync.get('summary', 'n/a')}")
        except Exception as exc:
            decision_feature_sync = {"status": "error", "error": str(exc)}
            sheet_update_errors.append({"stage": "decision_feature_sync", "error": str(exc)})
            log(f"Decision feature sync skipped: {exc}")

    summary = build_summary(
        args=args,
        parts=parts,
        skipped=skipped,
        weights_used=weights_used,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        sheet_updates_written=sheet_updates_written,
        sheet_cleanup_results=sheet_cleanup_results,
        sheet_update_errors=sheet_update_errors,
    )
    summary["decision_feature_sync"] = decision_feature_sync
    return sanitize_for_json(summary)


def main() -> None:
    summary = run_pipeline(parse_args())
    print(json.dumps(summary, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
