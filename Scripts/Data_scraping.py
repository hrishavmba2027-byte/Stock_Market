#!/usr/bin/env python3

import os
import argparse
from datetime import date, timedelta
import logging
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# -------------------------
# NIFTY 50 Stock Universe
# -------------------------
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS", "TITAN.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "NESTLEIND.NS", "ONGC.NS", "TECHM.NS", "HCLTECH.NS",
    "WIPRO.NS", "ADANIENT.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "INDUSINDBK.NS", "HDFCLIFE.NS", "SBILIFE.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "CIPLA.NS",
    "COALINDIA.NS", "GRASIM.NS", "APOLLOHOSP.NS", "SHREECEM.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "UPL.NS", "BPCL.NS", "IOC.NS", "HINDALCO.NS"
]


# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Download NIFTY 50 stock data from yfinance")

    parser.add_argument(
        "-s",
        "--symbols",
        type=str,
        default=",".join(NIFTY_50),
        help="Comma-separated NSE symbols (default: NIFTY 50)"
    )

    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "-i",
        "--interval",
        type=str,
        default="1d",
        help="Data interval (e.g., '1d', '1h', '5m')"
    )

    return parser.parse_args()


# -------------------------
# Fetch stock data
# -------------------------
def fetch_stock_data(symbol, start, end, interval):
    """
    Fetch raw OHLCV data for a single stock.
    
    For intraday data, restricts to last 60 days.
    Returns DataFrame with only raw columns or None if failed.
    """
    if interval != "1d":
        logger.warning(f"{symbol}: Intraday limited → using last 60 days")
        start = (pd.to_datetime(end) - timedelta(days=60)).strftime("%Y-%m-%d")

    try:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    except Exception as e:
        logger.warning(f"{symbol} failed: {e}")
        return None

    if df is None or df.empty:
        logger.warning(f"No data for {symbol}")
        return None

    # Keep only raw OHLCV columns
    raw_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_columns = [col for col in raw_columns if col in df.columns]

    if not available_columns:
        logger.warning(f"No raw OHLCV data for {symbol}")
        return None

    df = df[available_columns]

    return df


# -------------------------
# Update Excel file
# -------------------------
def update_excel(data_dict, output_path, interval):
    """
    Save stock data to Excel with one sheet per symbol.
    
    Handles file corruption, append mode, and adds Date_str column.
    """
    # Handle corrupted file
    if os.path.exists(output_path):
        try:
            pd.read_excel(output_path)
        except Exception:
            logger.warning("Corrupted Excel detected → deleting file")
            os.remove(output_path)

    mode = "a" if os.path.exists(output_path) else "w"

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
        mode=mode,
        if_sheet_exists="replace" if mode == "a" else None
    ) as writer:

        for symbol, df in data_dict.items():
            sheet_name = symbol.replace(".NS", "")

            # Reset index to make date a column
            df = df.reset_index()

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(map(str, col)).strip() for col in df.columns]
            else:
                df.columns = [str(c) for c in df.columns]

            # Get the date column name (usually "Date" or "Datetime")
            date_col = df.columns[0]

            # Ensure date column is datetime
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

            # Create Date_str column with appropriate format
            if interval == "1d":
                df["Date_str"] = df[date_col].dt.strftime("%Y-%m-%d")
            else:
                df["Date_str"] = df[date_col].dt.strftime("%Y-%m-%d %H:%M:%S")

            # Reorder columns: date_col, Date_str, then rest
            cols = [date_col, "Date_str"] + [col for col in df.columns if col not in [date_col, "Date_str"]]
            df = df[cols]

            df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(f"Saved: {sheet_name}")


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # Parse symbols and ensure .NS suffix
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    symbols = [s if s.endswith(".NS") else s + ".NS" for s in symbols]

    results = {}

    for sym in symbols:
        logger.info(f"Processing {sym}")

        df = fetch_stock_data(sym, args.start, args.end, args.interval)
        if df is None:
            continue

        results[sym] = df

    if not results:
        logger.error("No data collected")
        return

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(script_dir, "nse_stock_data.xlsx")

    update_excel(results, output_file, args.interval)

    logger.info("All done successfully!")


if __name__ == "__main__":
    main()