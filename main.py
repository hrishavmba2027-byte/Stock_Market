from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
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

from app.services.google_sheet_updates import PredictionSheetUpdateService
from Feature_Engineering import compute_indicators, detect_date_column, ensure_date_column


DEFAULT_SHEET_ID = "1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o"
TARGET_COL = "Close"
USE_INTRADAY_OPEN = True
BATCH_SIZE = 256
PREDICTED_COL = "predicted"
PREDICTED_PRICE_COL = "Predicted_Close_Price"
SHEET_ROW_COL = "__sheet_row_number"
SORT_POSITION_COL = "__sort_position"
STATUS_OK = "ok"
STATUS_NO_NEW_DATA = "no_new_data"
STATUS_NO_VALID_START_POINT = "no_valid_start_point"

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
    actual_price: np.ndarray
    target_scaler: MinMaxScaler
    feature_scaler: MinMaxScaler
    current_open_values: np.ndarray
    headers: List[str]
    data_row_count: int
    worksheet: Any = None
    row_predictions: Dict[int, float] = field(default_factory=dict)
    status: str = STATUS_OK
    eligible_target_count: int = 0
    last_valid_position: Optional[int] = None


class DenseModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.3,
        max_seq_len: int = 512,
    ):
        super().__init__()
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
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
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
        return self.fc(x[:, -1, :])


def log(message: str) -> None:
    print(message, file=sys.stderr)


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
    parser.add_argument("--workbook", default=str(script_dir / "nse_stock_data.xlsx"))
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
    with open(path, "r") as file:
        return json.load(file)


def torch_load_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_dense_model(state: Dict[str, torch.Tensor]) -> DenseModel:
    hidden_size, input_size = state["fc1.weight"].shape
    return DenseModel(input_size=int(input_size), hidden_size=int(hidden_size))


def infer_lstm_model(state: Dict[str, torch.Tensor]) -> LSTMModel:
    input_size = int(state["lstm.weight_ih_l0"].shape[1])
    hidden_size = int(state["lstm.weight_ih_l0"].shape[0] // 4)
    layer_indices = {
        int(key.split("lstm.weight_ih_l", 1)[1])
        for key in state
        if key.startswith("lstm.weight_ih_l")
    }
    return LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=max(layer_indices) + 1)


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
    layer_indices = {
        int(key.split("encoder.layers.", 1)[1].split(".", 1)[0])
        for key in state
        if key.startswith("encoder.layers.")
    }
    return TransformerModel(
        input_size=int(input_size),
        hidden_size=int(hidden_size),
        num_heads=infer_transformer_heads(int(hidden_size), metadata),
        num_layers=max(layer_indices) + 1,
        max_seq_len=int(state["position_embedding"].shape[1]),
    )


def load_models(model_dir: Path, metadata: Dict[str, Any], device: torch.device) -> Dict[str, nn.Module]:
    loaders = {
        "Dense": lambda state: infer_dense_model(state),
        "LSTM": lambda state: infer_lstm_model(state),
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


def validate_metadata(metadata: Dict[str, Any]) -> Tuple[List[str], int, int, int]:
    feature_columns = list(metadata["feature_columns"])
    seq_len = int(metadata["seq_len"])
    feature_count = int(metadata["feature_count"])
    dense_input_size = int(metadata["dense_input_size"])
    if len(feature_columns) != feature_count:
        raise ValueError(
            f"Metadata feature_count={feature_count} but feature_columns has {len(feature_columns)} entries"
        )
    if seq_len * feature_count != dense_input_size:
        raise ValueError(
            f"Metadata dense_input_size mismatch: {seq_len} * {feature_count} != {dense_input_size}"
        )
    return feature_columns, seq_len, feature_count, dense_input_size


def prepare_stock_part(
    payload: SheetPayload,
    metadata: Dict[str, Any],
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
    if len(valid_prediction_indices) == 0:
        first_valid_position = min(max(seq_len - 1, 0), len(df) - 1)
        log(
            f"{payload.name}: no rows marked predicted=1; "
            f"using position {first_valid_position} as the initial prediction checkpoint"
        )
    else:
        first_valid_position = int(valid_prediction_indices.min())

    candidate_df = normalize_columns(df)
    candidate_date_col = canonical_column_name(date_col) if date_col else None
    candidate_positions = to_numeric_series(candidate_df[SORT_POSITION_COL]).astype(np.int64).to_numpy()
    assert np.array_equal(
        np.sort(candidate_positions),
        np.arange(len(df)),
    )
    assert first_valid_position < len(df)

    valid_indices = np.where(
        (candidate_positions > first_valid_position)
        & (predicted_values == 0)
    )[0]
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
            actual_price=np.asarray([], dtype=np.float32),
            target_scaler=MinMaxScaler(),
            feature_scaler=MinMaxScaler(),
            current_open_values=np.asarray([], dtype=np.float32),
            headers=payload.headers,
            data_row_count=payload.data_row_count,
            worksheet=payload.worksheet,
            status=STATUS_NO_NEW_DATA,
            eligible_target_count=eligible_target_count,
            last_valid_position=first_valid_position,
        )

    if TARGET_COL not in candidate_df.columns:
        candidate_df[TARGET_COL] = np.nan
    candidate_df[TARGET_COL] = to_numeric_series(candidate_df[TARGET_COL])
    candidate_open = None
    if USE_INTRADAY_OPEN:
        if "Open" not in feature_columns:
            raise ValueError("metadata feature_columns must include Open when USE_INTRADAY_OPEN is True")
        if "Open" not in candidate_df.columns:
            raise ValueError("Open column is required for intraday prediction")
        candidate_open = to_numeric_series(candidate_df["Open"]).to_numpy(dtype=np.float32)

    with contextlib.redirect_stdout(sys.stderr):
        engineered = compute_indicators(df, date_col)
    engineered = normalize_columns(engineered)
    if TARGET_COL not in engineered.columns:
        raise ValueError("could not identify a Close column after feature engineering")
    if SHEET_ROW_COL not in engineered.columns or SORT_POSITION_COL not in engineered.columns:
        raise ValueError("internal row tracking columns were lost during feature engineering")

    engineered[TARGET_COL] = to_numeric_series(engineered[TARGET_COL])
    engineered = engineered.dropna(subset=[TARGET_COL])
    if engineered.empty:
        raise ValueError("no valid OHLCV rows remain after feature engineering")

    engineered_positions = to_numeric_series(engineered[SORT_POSITION_COL]).astype(np.int64).to_numpy()
    historical_mask = engineered_positions <= first_valid_position
    if not historical_mask.any():
        raise ValueError("no historical rows remain at or before the first predicted checkpoint")

    target_values = engineered.loc[historical_mask, [TARGET_COL]].to_numpy(dtype=np.float32)
    target_scaler = MinMaxScaler()
    target_scaler.fit(target_values)

    feature_frame = engineered.reindex(columns=feature_columns, fill_value=0)
    feature_frame = to_numeric_frame(feature_frame)
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_frame.loc[historical_mask])
    scaled_features = feature_scaler.transform(feature_frame).astype(np.float32)

    order = np.argsort(engineered_positions, kind="stable")
    engineered_positions = engineered_positions[order]
    scaled_features = scaled_features[order]

    candidate_sheet_rows = to_numeric_series(candidate_df[SHEET_ROW_COL]).astype(np.int64).to_numpy()
    candidate_actual = candidate_df[TARGET_COL].to_numpy(dtype=np.float32)
    candidate_dates = get_dates(candidate_df, candidate_date_col)

    X_sequences: List[np.ndarray] = []
    selected_positions: List[int] = []
    selected_rows: List[int] = []
    selected_actual: List[float] = []
    selected_dates: List[Any] = []
    selected_open_values: List[float] = []

    for idx in valid_indices:
        original_position = int(candidate_positions[idx])
        target_position = original_position
        prior_indices = np.flatnonzero(engineered_positions < int(target_position))
        if len(prior_indices) < seq_len:
            continue
        sequence = scaled_features[prior_indices[-seq_len:]]
        if sequence.shape != (seq_len, feature_count):
            raise ValueError(f"sequence shape mismatch: expected {(seq_len, feature_count)}, got {sequence.shape}")
        if USE_INTRADAY_OPEN:
            current_open_value = candidate_open[idx]
            if not np.isfinite(current_open_value):
                raise ValueError(
                    f"Open value is required for intraday prediction at sheet row {int(candidate_sheet_rows[idx])}"
                )
            selected_open_values.append(float(current_open_value))
        X_sequences.append(sequence)
        selected_positions.append(original_position)
        selected_rows.append(int(candidate_sheet_rows[idx]))
        selected_actual.append(float(candidate_actual[idx]) if np.isfinite(candidate_actual[idx]) else np.nan)
        selected_dates.append(candidate_dates[idx] if idx < len(candidate_dates) else idx)

    assert len(selected_positions) == len(set(selected_positions))
    assert all(predicted_values[row_pos] == 0 for row_pos in selected_positions)

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
        actual_price=np.asarray(selected_actual, dtype=np.float32),
        target_scaler=target_scaler,
        feature_scaler=feature_scaler,
        current_open_values=np.asarray(selected_open_values, dtype=np.float32),
        headers=payload.headers,
        data_row_count=payload.data_row_count,
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
    current_features.loc[:, "Open"] = open_values
    scaled_values = feature_scaler.transform(current_features).astype(np.float32)

    X[:, -1, open_idx] = scaled_values[:, open_idx]
    return X


def predict_model(model: nn.Module, model_name: str, X: np.ndarray, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for (X_batch,) in loader:
            if X_batch.dim() != 3:
                raise ValueError(f"{model_name}: X batch must be 3D, got {X_batch.shape}")
            output = model(X_batch.to(device))
            predictions.append(output.detach().cpu().numpy().reshape(-1))
    if not predictions:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(predictions).astype(np.float32)


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


def run_inference(
    parts: List[StockInferencePart],
    models: Dict[str, nn.Module],
    metadata: Dict[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    prediction_parts = [part for part in parts if len(part.X) > 0]
    if not prediction_parts:
        return empty_results_frame(), {}

    feature_columns, _, _, _ = validate_metadata(metadata)
    X = np.concatenate(
        [
            inject_intraday_open(
                part.X.copy(),
                part.current_open_values,
                part.feature_scaler,
                feature_columns,
            )
            for part in prediction_parts
        ],
        axis=0,
    ).astype(np.float32)
    validate_shapes(X, metadata)

    model_predictions = {
        model_name: predict_model(model, model_name, X, device)
        for model_name, model in models.items()
    }
    ensemble_scaled, weights_used = ensemble_predictions(
        model_predictions,
        metadata.get("ensemble_weights", {}),
    )

    result_frames: List[pd.DataFrame] = []
    offset = 0
    for part in prediction_parts:
        count = len(part.X)
        stock_scaled = ensemble_scaled[offset : offset + count]
        stock_price = part.target_scaler.inverse_transform(stock_scaled.reshape(-1, 1)).reshape(-1)
        part.row_predictions = {
            int(row_number): float(price)
            for row_number, price in zip(part.sheet_rows, stock_price)
        }

        result = pd.DataFrame(
            {
                "Symbol": part.symbol,
                "Sheet_Row": part.sheet_rows,
                "Date": pd.to_datetime(part.dates, errors="coerce", format="mixed"),
                "Actual_Price": part.actual_price,
                "Predicted_Scaled": stock_scaled,
                "Predicted_Price": stock_price.astype(np.float32),
            }
        )
        for model_name, predictions in model_predictions.items():
            result[f"{model_name}_Scaled"] = predictions[offset : offset + count]
        result_frames.append(result)
        offset += count

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
    with open(metrics_path, "w") as file:
        json.dump(sanitize_for_json(metrics), file, indent=2, allow_nan=False)
    log(f"Saved metrics: {metrics_path}")
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

    credential_file = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credential_file:
        creds = Credentials.from_service_account_file(credential_file, scopes=GOOGLE_SCOPES)
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
    updated_count = PredictionSheetUpdateService().update_prediction_status(
        worksheet=part.worksheet,
        headers=part.headers,
        row_predictions=part.row_predictions,
    )
    if updated_count:
        log(f"Updated Google worksheet {part.symbol}: predicted_rows={updated_count}")
    return updated_count


def process_payloads(payloads: List[SheetPayload], metadata: Dict[str, Any]) -> Tuple[List[StockInferencePart], List[str]]:
    parts: List[StockInferencePart] = []
    skipped: List[str] = []
    for payload in payloads:
        if payload.frame.empty:
            skipped.append(f"{payload.name}: worksheet is empty")
            log(f"Skipped {payload.name}: worksheet is empty")
            continue
        try:
            parts.append(prepare_stock_part(payload, metadata))
        except NoValidStartPointError as exc:
            skipped.append(f"{payload.name}: {exc}")
            log(f"Skipped {payload.name}: {exc}")
        except Exception as exc:
            skipped.append(f"{payload.name}: {exc}")
            log(f"Skipped {payload.name}: {exc}")
    return parts, skipped


def write_predictions(output_dir: Path, results: pd.DataFrame) -> Path:
    predictions_path = output_dir / "predictions.csv"
    results.to_csv(predictions_path, index=False)
    log(f"Saved predictions: {predictions_path}")
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
    }


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata)
    model_dir = Path(args.model_dir)
    metadata = load_json(metadata_path)
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

    parts, skipped = process_payloads(payloads, metadata)
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
    if args.source == "google" and not args.dry_run:
        updated_row_count = 0
        for part in parts:
            updated_row_count += update_google_predictions(part)
        sheet_updates_written = updated_row_count > 0
    elif args.source == "google" and args.dry_run:
        log("Dry run enabled: Google Sheet updates were not written")

    summary = build_summary(
        args=args,
        parts=parts,
        skipped=skipped,
        weights_used=weights_used,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        sheet_updates_written=sheet_updates_written,
    )
    return sanitize_for_json(summary)


def main() -> None:
    summary = run_pipeline(parse_args())
    print(json.dumps(summary, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
