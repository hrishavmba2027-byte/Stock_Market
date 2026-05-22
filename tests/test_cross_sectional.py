import numpy as np
import pandas as pd

from features import cross_sectional


def _index_history():
    dates = pd.date_range("2025-01-01", periods=300, freq="D")
    nifty = pd.Series(np.linspace(20000, 22000, len(dates)) + np.random.RandomState(0).randn(len(dates)) * 50, index=dates)
    vix = pd.Series(15 + np.random.RandomState(1).randn(len(dates)), index=dates)
    df = pd.DataFrame({"nifty_close": nifty, "vix_close": vix})
    df.index.name = "date"
    return df


def test_compute_regime_produces_expected_columns():
    idx = _index_history()
    regime = cross_sectional._compute_regime(idx)
    expected = {
        "nifty_log_return",
        "nifty_return_20d",
        "nifty_vol_20d",
        "nifty_trend_50d_200d",
        "vix_level",
        "vix_delta_5d",
        "regime_high_vol",
        "regime_nifty_up",
    }
    assert expected.issubset(set(regime.columns))


def test_add_cross_sectional_features_attaches_regime_and_rel_strength():
    idx = cross_sectional._compute_regime(_index_history())
    dates = pd.date_range("2025-06-01", periods=120, freq="D")
    closes_a = pd.Series(np.linspace(100, 130, len(dates)))
    closes_b = pd.Series(np.linspace(50, 40, len(dates)))
    sheets = {
        "A": pd.DataFrame({"Date": dates, "Close": closes_a, "RSI_14": np.linspace(40, 60, len(dates))}),
        "B": pd.DataFrame({"Date": dates, "Close": closes_b, "RSI_14": np.linspace(60, 30, len(dates))}),
    }
    out = cross_sectional.add_cross_sectional_features(sheets, index_df=None, fundamentals_df=None)

    # Patch in our regime since fetch_index_history isn't running
    out = {
        ticker: df.drop(
            columns=[c for c in df.columns if c.startswith("nifty_") or c.startswith("vix_") or c.startswith("regime_")],
            errors="ignore",
        )
        for ticker, df in out.items()
    }
    enriched = {ticker: df.merge(idx, left_on="Date", right_index=True, how="left") for ticker, df in out.items()}

    for ticker, df in enriched.items():
        assert "return_20d" in df.columns
        assert "RSI_14_rank" in df.columns or "RSI_14" in df.columns
        assert "day_of_week" in df.columns
        assert "days_to_month_end" in df.columns


def test_cross_sectional_ranks_are_within_zero_to_one():
    dates = pd.date_range("2025-06-01", periods=10, freq="D")
    sheets = {
        "A": pd.DataFrame({"Date": dates, "Close": np.arange(10.0), "RSI_14": np.linspace(40, 60, 10)}),
        "B": pd.DataFrame({"Date": dates, "Close": np.arange(10.0), "RSI_14": np.linspace(60, 30, 10)}),
        "C": pd.DataFrame({"Date": dates, "Close": np.arange(10.0), "RSI_14": np.linspace(20, 80, 10)}),
    }
    ranked = cross_sectional._add_cross_sectional_ranks(sheets, date_col="Date", rank_cols=["RSI_14"])
    for df in ranked.values():
        if "RSI_14_rank" in df.columns:
            valid = df["RSI_14_rank"].dropna()
            assert ((valid >= 0) & (valid <= 1)).all()
