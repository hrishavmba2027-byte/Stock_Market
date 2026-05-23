import pandas as pd
import numpy as np
from openpyxl import load_workbook
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import TA-Lib equivalent (optional)
try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# Constants
REQUIRED_COLUMNS = {'open', 'high', 'low', 'close', 'volume'}
DATE_COLUMN_VARIANTS = ['date', 'Date', 'DATE', 'datetime', 'Datetime', 'time', 'Time']


def load_workbook_sheets(filepath):
    """
    Load all sheets from the workbook into a dictionary of DataFrames.
    
    Args:
        filepath: Path to the Excel workbook
        
    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    try:
        sheets = pd.read_excel(filepath, sheet_name=None)
        print(f"Loaded {len(sheets)} sheets from {filepath}")
        return sheets
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading workbook: {e}")
        sys.exit(1)


def detect_date_column(df):
    """
    Automatically detect the date column in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Column name if found, None otherwise
    """
    # Check for exact matches first
    for col in DATE_COLUMN_VARIANTS:
        if col in df.columns:
            return col
    
    # Check for columns containing 'date' or 'time'
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            return col
    
    # Check first column if it looks like dates
    if len(df.columns) > 0:
        first_col = df.columns[0]
        try:
            pd.to_datetime(df[first_col])
            return first_col
        except:
            pass
    
    return None


def ensure_date_column(df, date_col):
    """
    Ensure the date column is in datetime format and sort by date.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        
    Returns:
        DataFrame with date column as datetime and sorted
    """
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
        except TypeError:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Remove rows with invalid dates
        df = df.dropna(subset=[date_col])
        # Sort by date
        df = df.sort_values(by=date_col).reset_index(drop=True)
    return df


def find_ohlcv_columns(df):
    """
    Dynamically find OHLCV columns in the DataFrame.
    Handles patterns like: Open_SYMBOL, High_SYMBOL, etc.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping OHLCV names to actual column names, or None if not found
    """
    # Normalize column names to lowercase for matching
    col_lower = {col: col.lower() for col in df.columns}
    
    # Try exact matches first
    ohlcv_map = {}
    for ohlc_name in ['open', 'high', 'low', 'close', 'volume']:
        if ohlc_name in col_lower.values():
            ohlcv_map[ohlc_name] = [col for col, lower_col in col_lower.items() if lower_col == ohlc_name][0]
    
    # If we found all 5, return
    if len(ohlcv_map) == 5:
        return ohlcv_map
    
    # Otherwise, try pattern matching (e.g., Open_SYMBOL, High_SYMBOL)
    ohlcv_map = {}
    for col_name, col_lower_name in col_lower.items():
        if col_lower_name.startswith('open_'):
            ohlcv_map['open'] = col_name
        elif col_lower_name.startswith('high_'):
            ohlcv_map['high'] = col_name
        elif col_lower_name.startswith('low_'):
            ohlcv_map['low'] = col_name
        elif col_lower_name.startswith('close_'):
            ohlcv_map['close'] = col_name
        elif col_lower_name.startswith('volume_'):
            ohlcv_map['volume'] = col_name
    
    # Check if all OHLCV were found
    if len(ohlcv_map) == 5:
        return ohlcv_map
    
    return None


def compute_rsi(close, period=14):
    """Compute Relative Strength Index (RSI)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = np.where(loss != 0, gain / loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=close.index)


def compute_macd(close, fast=12, slow=26, signal=9):
    """
    Compute MACD, Signal Line, and Histogram.
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Compute Stochastic Oscillator (%K and %D).
    
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    denominator = highest_high - lowest_low
    k_percent = 100 * ((close - lowest_low) / denominator)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent


def compute_sma(close, period):
    """Compute Simple Moving Average (SMA)"""
    return close.rolling(window=period).mean()


def compute_ema(close, period):
    """Compute Exponential Moving Average (EMA)"""
    return close.ewm(span=period, adjust=False).mean()


def compute_adx(high, low, close, period=14):
    """Compute Average Directional Index (ADX)"""
    try:
        plus_dm = high.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > (low.diff() * -1)), 0)
        
        minus_dm = low.diff() * -1
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > high.diff()), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        tr_sum = tr.rolling(window=period).sum()
        
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr_sum)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr_sum)
        
        di_sum = (plus_di + minus_di).abs()
        di_diff = (plus_di - minus_di).abs()
        
        dx = 100 * (di_diff / di_sum)
        adx = dx.rolling(window=period).mean()
        
        return adx
    except Exception as e:
        print(f"      Warning: ADX calculation failed: {e}")
        return pd.Series(np.nan, index=close.index)


def compute_bollinger_bands(close, period=20, std_dev=2):
    """
    Compute Bollinger Bands (Upper, Middle, Lower).
    
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band


def compute_atr(high, low, close, period=14):
    """Compute Average True Range (ATR)"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_obv(close, volume):
    """Compute On-Balance Volume (OBV)"""
    price_diff = close.diff()
    price_diff[0] = 0  # First value is NaN
    obv = (np.sign(price_diff) * volume).fillna(0).cumsum()
    return obv


def compute_vwap(high, low, close, volume):
    """Compute Volume Weighted Average Price (VWAP)"""
    try:
        typical_price = (high + low + close) / 3
        cum_vol = volume.cumsum()
        cum_tp_vol = (typical_price * volume).cumsum()
        vwap = cum_tp_vol / cum_vol
        return vwap
    except Exception as e:
        print(f"      Warning: VWAP calculation failed: {e}")
        return pd.Series(np.nan, index=close.index)


def compute_daily_return(close):
    """Compute Daily Return (percentage change)"""
    return close.pct_change() * 100


def compute_log_return(close):
    """Compute Log Return"""
    return np.log(close / close.shift(1)) * 100


# ============================================================================
# P0.1 — Forward-return label construction
# ============================================================================
# These labels look INTO THE FUTURE — they must never be used as model inputs.
# They are the targets for the multi-horizon quantile forecaster (P0.2).
# ============================================================================

FORWARD_HORIZONS = tuple(range(1, 31))  # h = 1, 2, ..., 30
FORWARD_LABEL_PREFIX = "y_logret_h"
HAS_LABELS_COL = "has_labels"


def forward_label_columns(horizons=FORWARD_HORIZONS):
    """Canonical names for the forward-return label columns."""
    return [f"{FORWARD_LABEL_PREFIX}{h}" for h in horizons]


def compute_forward_log_returns(close, horizons=FORWARD_HORIZONS):
    """
    For each row at time t, compute y_h = log(Close[t+h] / Close[t]) for h in horizons.

    Implemented via close.shift(-h) so y_h at row t is undefined for the last h rows
    of the series — that's intentional. Those rows are NOT training-eligible and
    are flagged via `has_labels=False`.

    Returns:
        Dict[str, pd.Series] keyed by forward_label_columns(horizons).
    """
    labels = {}
    log_close = np.log(close)
    for h in horizons:
        labels[f"{FORWARD_LABEL_PREFIX}{h}"] = log_close.shift(-h) - log_close
    return labels


def attach_forward_labels(df, close_col="close", horizons=FORWARD_HORIZONS):
    """
    Attach 30 forward-return label columns plus a `has_labels` boolean flag to df.

    `has_labels=True` iff every forward label at that row is finite (i.e. the row
    has at least `max(horizons)` future bars available in the same frame).

    The flag is deliberately surfaced as a column so any downstream train/test
    code can `assert df.loc[train_mask, 'has_labels'].all()` and any leakage bug
    becomes visible in code review (per ROADMAP P0.1).
    """
    df_out = df.copy()
    labels = compute_forward_log_returns(df_out[close_col], horizons=horizons)
    label_cols = list(labels.keys())
    for name, series in labels.items():
        df_out[name] = series.values
    df_out[HAS_LABELS_COL] = df_out[label_cols].notna().all(axis=1)
    return df_out


def compute_roc(close, period=12):
    """Compute Rate of Change (ROC)"""
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    return roc


def compute_cci(high, low, close, period=20):
    """Compute Commodity Channel Index (CCI)"""
    try:
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=False
        )
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    except Exception as e:
        print(f"      Warning: CCI calculation failed: {e}")
        return pd.Series(np.nan, index=close.index)


def compute_williams_r(high, low, close, period=14):
    """Compute Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


def compute_mfi(high, low, close, volume, period=14):
    """Compute Money Flow Index (MFI)"""
    try:
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf_sum = positive_flow.rolling(window=period).sum()
        negative_mf_sum = negative_flow.rolling(window=period).sum()
        
        mfi_ratio = positive_mf_sum / negative_mf_sum
        mfi = 100 - (100 / (1 + mfi_ratio))
        return mfi
    except Exception as e:
        print(f"      Warning: MFI calculation failed: {e}")
        return pd.Series(np.nan, index=close.index)


# ============================================================================
# Decision-layer categorical features (Google Sheet visibility only)
# ----------------------------------------------------------------------------
# These columns are computed for Claude's decision-making and are written to
# the Google Sheets ONLY. They are intentionally NOT added to compute_indicators,
# so they never enter the model feature matrix or pipeline_metadata.json.
# All existing engineered features are preserved untouched.
# ============================================================================

DECISION_FEATURE_COLUMNS = [
    "RSI_7", "SMA_5_20", "SMA_20_50", "EMA_12_26", "EMA_26_50",
    "BB_indicator", "BB_trend_state", "BB_volatility_state",
    "BB_overbought_oversold", "RSI_indicator_7", "RSI_indicator_14",
]

# Minimum finite observations required to trust dynamic per-symbol percentiles.
RSI_BIN_MIN_HISTORY = 30
# Fallback (p25, p50, p75) — standard RSI rules — used when history is too short.
_RSI_FALLBACK_THRESHOLDS = (30.0, 50.0, 70.0)


def _finite(series):
    """Coerce to numeric and replace +/-inf with NaN (never raises)."""
    return pd.to_numeric(pd.Series(series), errors='coerce').replace([np.inf, -np.inf], np.nan)


def find_close_column(df):
    """Locate the Close column leniently (exact, then Close_SYMBOL pattern)."""
    for col in df.columns:
        if str(col).strip().lower() == 'close':
            return col
    for col in df.columns:
        if str(col).strip().lower().startswith('close_'):
            return col
    return None


def rsi_percentile_thresholds(rsi_values):
    """Dynamic (p25, p50, p75) from a symbol's RSI history; None if too sparse."""
    arr = _finite(pd.Series(list(rsi_values), dtype='float64')).dropna()
    if len(arr) < RSI_BIN_MIN_HISTORY:
        return None
    return (float(arr.quantile(0.25)), float(arr.quantile(0.50)), float(arr.quantile(0.75)))


def classify_rsi_bins(rsi_series, thresholds):
    """Map RSI to mutually-exclusive categorical bins.

    thresholds: (p25, p50, p75). When None, falls back to fixed RSI rules.
    Bins (checked in order, so they are mutually exclusive):
      > p75 -> highly overbought ; > p50 -> slightly overbought ;
      < p25 -> highly oversold   ; < p50 -> slightly oversold   ; else neutral.
    """
    p25, p50, p75 = thresholds if thresholds is not None else _RSI_FALLBACK_THRESHOLDS
    out = []
    for value in _finite(rsi_series):
        if pd.isna(value):
            out.append("")
        elif value > p75:
            out.append("highly overbought")
        elif value > p50:
            out.append("slightly overbought")
        elif value < p25:
            out.append("highly oversold")
        elif value < p50:
            out.append("slightly oversold")
        else:
            out.append("neutral")
    return out


def _classify_cross(fast, slow, up_label, down_label, neutral_label="neutral"):
    """Categorise a moving-average crossover; blank cell when inputs are NaN."""
    out = []
    for f, s in zip(_finite(fast), _finite(slow)):
        if pd.isna(f) or pd.isna(s):
            out.append("")
        elif f > s:
            out.append(up_label)
        elif f < s:
            out.append(down_label)
        else:
            out.append(neutral_label)
    return out


def bollinger_decision_states(close, period=20, std_dev=2):
    """Return (BB_indicator, BB_trend_state, BB_volatility_state, BB_overbought_oversold).

    Trend & overbought/oversold come from %B = (close - lower) / (upper - lower).
    Volatility comes from bandwidth = (upper - lower) / middle, compared against
    the symbol's own 25th/75th bandwidth percentiles (normal volatility when the
    symbol's history is too short to form percentiles).
    """
    close = _finite(close)
    upper, middle, lower = compute_bollinger_bands(close, period, std_dev)
    width = upper - lower
    pct_b = (close - lower) / width.where(width != 0, np.nan)
    bandwidth = width / middle.where(middle != 0, np.nan)
    bw_finite = _finite(bandwidth).dropna()
    if len(bw_finite) >= RSI_BIN_MIN_HISTORY:
        bw_low, bw_high = float(bw_finite.quantile(0.25)), float(bw_finite.quantile(0.75))
    else:
        bw_low = bw_high = None

    indicator, trend, volatility, obos = [], [], [], []
    for pb, bw in zip(pct_b, _finite(bandwidth)):
        if pd.isna(pb):
            t = o = ""
        else:
            o = "overbought" if pb >= 1.0 else ("oversold" if pb <= 0.0 else "neutral")
            t = "bullish" if pb > 0.6 else ("bearish" if pb < 0.4 else "neutral")
        if pd.isna(bw):
            v = ""
        elif bw_low is None:
            v = "normal volatility"
        elif bw > bw_high:
            v = "high volatility"
        elif bw < bw_low:
            v = "low volatility"
        else:
            v = "normal volatility"
        trend.append(t)
        volatility.append(v)
        obos.append(o)
        indicator.append(" | ".join(p for p in (t, o, v) if p))
    return indicator, trend, volatility, obos


def compute_decision_features(df, rsi7_thresholds=None, rsi14_thresholds=None):
    """Compute the 11 decision-layer categorical features for a single symbol.

    `df` must already be ordered chronologically so indicator look-back is
    correct. Returns a DataFrame of DECISION_FEATURE_COLUMNS aligned to df.index.
    Never raises: any unresolved input simply yields blank cells.

    The RSI bin thresholds are supplied by the caller (computed from the full
    cross-sheet history for the symbol); when None the bins fall back safely.
    """
    out = pd.DataFrame(index=df.index, columns=DECISION_FEATURE_COLUMNS, dtype=object)
    close_col = find_close_column(df)
    if close_col is None:
        return out.fillna("")
    try:
        close = _finite(df[close_col])
        close.index = df.index
        rsi7 = compute_rsi(close, 7)
        rsi14 = compute_rsi(close, 14)
        out["RSI_7"] = ["" if pd.isna(v) else round(float(v), 2) for v in _finite(rsi7)]
        out["SMA_5_20"] = _classify_cross(compute_sma(close, 5), compute_sma(close, 20),
                                          "bullish", "bearish")
        out["SMA_20_50"] = _classify_cross(compute_sma(close, 20), compute_sma(close, 50),
                                           "strong bullish", "strong bearish")
        out["EMA_12_26"] = _classify_cross(compute_ema(close, 12), compute_ema(close, 26),
                                           "bullish", "bearish")
        out["EMA_26_50"] = _classify_cross(compute_ema(close, 26), compute_ema(close, 50),
                                           "long-term bullish", "long-term bearish")
        bb_ind, bb_trend, bb_vol, bb_obos = bollinger_decision_states(close)
        out["BB_indicator"] = bb_ind
        out["BB_trend_state"] = bb_trend
        out["BB_volatility_state"] = bb_vol
        out["BB_overbought_oversold"] = bb_obos
        out["RSI_indicator_7"] = classify_rsi_bins(rsi7, rsi7_thresholds)
        out["RSI_indicator_14"] = classify_rsi_bins(rsi14, rsi14_thresholds)
    except Exception as exc:  # never break the caller's workflow
        print(f"      Warning: decision feature computation failed: {exc}")
    return out.fillna("")


def compute_decision_rsi_history(df, periods=(7, 14)):
    """Return {period: list[float]} of finite RSI values from a chronological frame.

    Used by the sheet sync to pool RSI history across BOTH sheets per symbol
    before deriving dynamic percentile thresholds.
    """
    history = {p: [] for p in periods}
    close_col = find_close_column(df)
    if close_col is None:
        return history
    close = _finite(df[close_col])
    for period in periods:
        try:
            history[period] = _finite(compute_rsi(close, period)).dropna().tolist()
        except Exception:
            history[period] = []
    return history


def compute_indicators(df, date_col):
    """
    Compute all technical indicators for the given DataFrame.
    
    Handles both daily and intraday data. Gracefully handles computation failures
    by filling with NaN values instead of crashing.
    
    Args:
        df: Input DataFrame with OHLCV data
        date_col: Name of the date column
        
    Returns:
        DataFrame with all indicators added (original columns + new indicator columns)
    """
    df_work = df.copy()
    
    # Find OHLCV columns dynamically
    ohlcv_map = find_ohlcv_columns(df_work)
    
    if ohlcv_map is None:
        print(f"      -> Skipping: Could not identify OHLCV columns")
        return df_work
    
    try:
        # Extract OHLCV as numeric, coercing errors to NaN
        open_prices = pd.to_numeric(df_work[ohlcv_map['open']], errors='coerce')
        high_prices = pd.to_numeric(df_work[ohlcv_map['high']], errors='coerce')
        low_prices = pd.to_numeric(df_work[ohlcv_map['low']], errors='coerce')
        close_prices = pd.to_numeric(df_work[ohlcv_map['close']], errors='coerce')
        volume = pd.to_numeric(df_work[ohlcv_map['volume']], errors='coerce')
        
        # Remove any NaN rows from OHLCV data
        valid_idx = ~(open_prices.isna() | high_prices.isna() | 
                      low_prices.isna() | close_prices.isna() | volume.isna())
        
        if not valid_idx.any():
            print(f"      -> Skipping: No valid OHLCV data")
            return df_work
        
        # Apply valid index filter using boolean indexing with .loc
        open_prices = open_prices.loc[valid_idx].reset_index(drop=True)
        high_prices = high_prices.loc[valid_idx].reset_index(drop=True)
        low_prices = low_prices.loc[valid_idx].reset_index(drop=True)
        close_prices = close_prices.loc[valid_idx].reset_index(drop=True)
        volume = volume.loc[valid_idx].reset_index(drop=True)
        df_work = df_work.loc[valid_idx].reset_index(drop=True)
        
        indicators = {}
        
        # RSI (14)
        try:
            indicators['RSI_14'] = compute_rsi(close_prices, 14)
        except Exception as e:
            print(f"      Warning: RSI failed: {e}")
            indicators['RSI_14'] = pd.Series(np.nan, index=range(len(df_work)))
        
        # MACD (12, 26, 9)
        try:
            macd, signal, hist = compute_macd(close_prices, 12, 26, 9)
            indicators['MACD_12_26'] = macd.reset_index(drop=True)
            indicators['MACD_Signal_9'] = signal.reset_index(drop=True)
            indicators['MACD_Histogram'] = hist.reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: MACD failed: {e}")
        
        # Stochastic Oscillator (%K and %D)
        try:
            k_percent, d_percent = compute_stochastic(high_prices, low_prices, close_prices, 14, 3)
            indicators['Stochastic_%K'] = k_percent.reset_index(drop=True)
            indicators['Stochastic_%D'] = d_percent.reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: Stochastic failed: {e}")
        
        # SMA (5, 20, 50)
        try:
            indicators['SMA_5'] = compute_sma(close_prices, 5).reset_index(drop=True)
            indicators['SMA_20'] = compute_sma(close_prices, 20).reset_index(drop=True)
            indicators['SMA_50'] = compute_sma(close_prices, 50).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: SMA failed: {e}")
        
        # EMA (12, 26, 50)
        try:
            indicators['EMA_12'] = compute_ema(close_prices, 12).reset_index(drop=True)
            indicators['EMA_26'] = compute_ema(close_prices, 26).reset_index(drop=True)
            indicators['EMA_50'] = compute_ema(close_prices, 50).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: EMA failed: {e}")
        
        # ADX (14)
        try:
            indicators['ADX_14'] = compute_adx(high_prices, low_prices, close_prices, 14).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: ADX failed: {e}")
        
        # Bollinger Bands (20, 2)
        try:
            upper_band, middle_band, lower_band = compute_bollinger_bands(close_prices, 20, 2)
            indicators['BB_Upper_20'] = upper_band.reset_index(drop=True)
            indicators['BB_Middle_20'] = middle_band.reset_index(drop=True)
            indicators['BB_Lower_20'] = lower_band.reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: Bollinger Bands failed: {e}")
        
        # ATR (14)
        try:
            indicators['ATR_14'] = compute_atr(high_prices, low_prices, close_prices, 14).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: ATR failed: {e}")
        
        # OBV
        try:
            indicators['OBV'] = compute_obv(close_prices, volume).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: OBV failed: {e}")
        
        # VWAP
        try:
            indicators['VWAP'] = compute_vwap(high_prices, low_prices, close_prices, volume).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: VWAP failed: {e}")
        
        # Daily Return
        try:
            indicators['Daily_Return_%'] = compute_daily_return(close_prices).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: Daily Return failed: {e}")
        
        # Log Return
        try:
            indicators['Log_Return_%'] = compute_log_return(close_prices).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: Log Return failed: {e}")
        
        # ROC (12)
        try:
            indicators['ROC_12'] = compute_roc(close_prices, 12).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: ROC failed: {e}")
        
        # CCI (20)
        try:
            indicators['CCI_20'] = compute_cci(high_prices, low_prices, close_prices, 20).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: CCI failed: {e}")
        
        # Williams %R (14)
        try:
            indicators['Williams_%R'] = compute_williams_r(high_prices, low_prices, close_prices, 14).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: Williams %R failed: {e}")
        
        # MFI (14)
        try:
            indicators['MFI_14'] = compute_mfi(high_prices, low_prices, close_prices, volume, 14).reset_index(drop=True)
        except Exception as e:
            print(f"      Warning: MFI failed: {e}")
        
        # Add all indicators to original dataframe
        for col_name, col_data in indicators.items():
            if isinstance(col_data, pd.Series):
                df_work[col_name] = col_data.shift(1).values
            else:
                df_work[col_name] = np.nan

        # P0.1 — attach 30 forward-return labels + has_labels flag.
        # These columns are TARGETS, not features. They must be stripped from
        # the feature matrix before training (see training code).
        try:
            close_for_labels = pd.to_numeric(
                df_work[ohlcv_map['close']], errors='coerce'
            ).reset_index(drop=True)
            forward_labels = compute_forward_log_returns(close_for_labels)
            for label_name, label_series in forward_labels.items():
                df_work[label_name] = label_series.values
            label_cols = list(forward_labels.keys())
            df_work[HAS_LABELS_COL] = df_work[label_cols].notna().all(axis=1).values
        except Exception as e:
            print(f"      Warning: forward label attachment failed: {e}")
            for label_name in forward_label_columns():
                df_work[label_name] = np.nan
            df_work[HAS_LABELS_COL] = False

        return df_work
    
    except Exception as e:
        print(f"      -> Error computing indicators: {e}")
        return df_work


def write_updated_workbook(sheets_dict, output_filepath):
    """
    Write updated sheets back to the Excel workbook using openpyxl.
    Preserves original sheet names and overwrites the file.
    
    Args:
        sheets_dict: Dictionary mapping sheet names to DataFrames
        output_filepath: Path to output Excel file
    """
    try:
        with pd.ExcelWriter(output_filepath, engine='openpyxl') as writer:
            for sheet_name, df in sheets_dict.items():
                # Remove entirely empty rows (all NaN)
                df_clean = df.dropna(how='all')
                
                # Write to Excel
                df_clean.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Successfully saved updated workbook to {output_filepath}")
    except Exception as e:
        print(f"Error writing workbook: {e}")
        sys.exit(1)


def main():
    """Main execution function"""
    print("=" * 70)
    print("NSE Stock Data - Technical Indicator Feature Engineering")
    print("=" * 70)
    
    # Canonical workbook location: <project_root>/Data/nse_stock_data.xlsx
    # (consistent with app/config/settings.py and Data_scraping.py output)
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'Data'
    input_file = data_dir / 'nse_stock_data.xlsx'
    output_file = data_dir / 'nse_stock_data.xlsx'

    # Check if input file exists
    if not input_file.exists():
        print(f"\nError: Input file '{input_file}' not found.")
        print(f"Generate it first by running:")
        print(f"  python3 Scripts/Data_scraping.py")
        print(f"  (downloads NIFTY-50 OHLCV data from yfinance into Data/nse_stock_data.xlsx)")
        sys.exit(1)
    
    # Load workbook
    print(f"\nLoading workbook: {input_file.name}")
    sheets = load_workbook_sheets(str(input_file))
    
    if not sheets:
        print("Error: Workbook contains no sheets.")
        sys.exit(1)
    
    # Process each sheet
    print(f"\nProcessing {len(sheets)} sheets...\n")
    processed_sheets = {}
    
    for sheet_name, df in sheets.items():
        print(f"Sheet: {sheet_name}")
        
        # Check if sheet is empty
        if df.empty:
            print(f"  -> Sheet is empty, skipping.")
            processed_sheets[sheet_name] = df
            continue
        
        print(f"  -> Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Detect date column
        date_col = detect_date_column(df)
        if date_col is None:
            print(f"  -> Could not detect date column, skipping indicators.")
            processed_sheets[sheet_name] = df
            continue
        
        print(f"  -> Date column detected: '{date_col}'")
        
        # Ensure date column is in proper format and sorted
        df = ensure_date_column(df, date_col)
        
        # Compute all indicators
        df = compute_indicators(df, date_col)
        
        processed_sheets[sheet_name] = df
        print(f"  -> Completed: {len(df.columns)} total columns\n")
    
    # Write updated workbook
    print("-" * 70)
    print("Writing updated workbook...")
    write_updated_workbook(processed_sheets, str(output_file))
    
    print("\n" + "=" * 70)
    print("✓ Feature engineering completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
