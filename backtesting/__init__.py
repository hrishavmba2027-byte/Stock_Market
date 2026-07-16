"""Backtesting package (Phase 3+).

A parallel, **fully local** pipeline: prices come from the local training
workbook and news/sentiment from a local Excel workbook (no Firestore, no Google
Sheet). It trains a SEPARATE, point-in-time model — the production pipeline and
its ``outputs/Saved_Models`` artifacts are never touched. All configuration comes
from :class:`app.config.backtest_settings.BacktestSettings`.
"""
