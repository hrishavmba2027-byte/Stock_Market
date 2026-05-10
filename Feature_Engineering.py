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
        if ohlc_name in col_lower:
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
                df_work[col_name] = col_data.values
            else:
                df_work[col_name] = np.nan
        
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
    
    # Define file paths relative to script location
    script_dir = Path(__file__).parent
    input_file = script_dir / 'nse_stock_data.xlsx'
    output_file = script_dir / 'nse_stock_data.xlsx'
    
    # Check if input file exists
    if not input_file.exists():
        print(f"\nError: Input file '{input_file}' not found.")
        print(f"Please ensure 'nse_stock_data.xlsx' exists in the same directory.")
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
