"""
Data Fetcher – multi-timeframe OHLCV data retrieval
Supports yfinance (default) with a ccxt stub for crypto exchanges.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

import config as cfg


# ── yfinance interval → pandas resample rule ──────────────────────────────────
_RESAMPLE_MAP: Dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}


def fetch_ohlcv(
    symbol: str = cfg.DEFAULT_SYMBOL,
    interval: str = cfg.DEFAULT_INTERVAL,
    period: str = cfg.DEFAULT_PERIOD,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.

    Returns a DataFrame with columns:
        open, high, low, close, volume
    Indexed by UTC datetime.
    """
    if not _YF_AVAILABLE:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} [{interval}, {period}]")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"
    df.dropna(inplace=True)
    return df


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample a lower-timeframe OHLCV DataFrame to a higher timeframe.

    Parameters
    ----------
    df        : source OHLCV DataFrame (DatetimeIndex, UTC)
    target_tf : target timeframe string, e.g. '4h', '1D'
    """
    rule = _RESAMPLE_MAP.get(target_tf, target_tf)
    resampled = df.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    return resampled


def fetch_multi_tf(
    symbol: str = cfg.DEFAULT_SYMBOL,
    base_interval: str = "1h",
    period: str = cfg.DEFAULT_PERIOD,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for base timeframe and resample to 4H and Daily automatically.

    Returns a dict: {'1h': df, '4h': df, '1d': df}
    """
    base_df = fetch_ohlcv(symbol, base_interval, period)
    return {
        base_interval: base_df,
        "4h": resample_ohlcv(base_df, "4h"),
        "1d": resample_ohlcv(base_df, "1D"),
    }
