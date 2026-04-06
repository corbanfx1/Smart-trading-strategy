"""
Data Fetcher — downloads OHLCV data for XAUUSD across multiple timeframes.
Uses yfinance as primary source; resamples 1H → 4H internally.
"""
from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from config.settings import SYMBOL, TIMEFRAMES
from utils.logger import get_logger

log = get_logger("DataFetcher")


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, drop NaN/zero rows, ensure UTC index."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # yfinance sometimes returns multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower()

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required].copy()
    df = df.replace(0, np.nan).dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.sort_index(inplace=True)
    return df


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1H OHLCV bars into 4H bars."""
    ohlcv = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    df_4h = df_1h.resample("4h", closed="left", label="left").agg(ohlcv).dropna()
    return df_4h


def fetch_ohlcv(symbol: str = SYMBOL, timeframe_key: str = "1D") -> pd.DataFrame:
    """
    Download OHLCV for *symbol* at the given *timeframe_key*.
    Returns a clean DataFrame with UTC DatetimeIndex.
    """
    cfg = TIMEFRAMES[timeframe_key]
    log.info("Fetching %s | TF=%s | interval=%s period=%s",
             symbol, timeframe_key, cfg["interval"], cfg["period"])

    ticker = yf.Ticker(symbol)
    raw = ticker.history(
        interval=cfg["interval"],
        period=cfg["period"],
        auto_adjust=True,
        actions=False,
        prepost=False,
    )

    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol} @ {cfg['interval']}")

    df = _clean_df(raw)

    # Resample 1H → 4H when requested
    if timeframe_key == "4H":
        df = resample_to_4h(df)

    log.info("  ✓ %d bars (%s → %s)", len(df),
             df.index[0].date(), df.index[-1].date())
    return df


def fetch_all_timeframes(
    symbol: str = SYMBOL,
    fallback_synthetic: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all configured timeframes.
    If network is unavailable and *fallback_synthetic* is True,
    returns synthetic GBM data for offline testing/CI.
    """
    data: dict[str, pd.DataFrame] = {}

    for tf_key in ["1D", "4H", "1H", "15M"]:
        try:
            data[tf_key] = fetch_ohlcv(symbol, tf_key)
        except Exception as exc:
            log.error("Failed to fetch %s @ %s: %s", symbol, tf_key, exc)

    if not data and fallback_synthetic:
        log.warning("Network unavailable — using SYNTHETIC data for analysis")
        from data.synthetic import generate_all_timeframes
        data = generate_all_timeframes()

    return data


def get_current_price(symbol: str = SYMBOL) -> float:
    """Return the latest close price."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d", interval="1m")
    if hist.empty:
        raise RuntimeError("Cannot fetch current price")
    return float(hist["Close"].iloc[-1])
