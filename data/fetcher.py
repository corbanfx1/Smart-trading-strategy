"""
Multi-Timeframe OHLCV Fetcher for XAUUSD
Pulls data from Yahoo Finance (GC=F = Gold Futures Continuous).
Results are disk-cached and refreshed only when the TTL expires.
"""
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch & cache OHLCV data for all configured timeframes."""

    def __init__(
        self,
        ticker: str = config.TICKER,
        cache_ttl_hours: int = 4,
    ) -> None:
        self.ticker = ticker
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        interval: str,
        period: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return clean OHLCV DataFrame for *interval* / *period*.

        yfinance does not natively support 4-hour candles, so we download
        1-hour data and resample to 4h internally.
        """
        cache_path = self._cache_path(interval)

        if not force_refresh and self._cache_valid(cache_path):
            logger.debug("Loading %s %s from disk cache.", self.ticker, interval)
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

        logger.info("Downloading %s %s  (period=%s) …", self.ticker, interval, period)

        try:
            if interval == "4h":
                raw = yf.download(
                    self.ticker,
                    interval="1h",
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                df = self._resample_ohlcv(raw, "4h")
            else:
                df = yf.download(
                    self.ticker,
                    interval=interval,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
        except Exception as exc:
            raise RuntimeError(
                f"yfinance download failed for {self.ticker} @ {interval}: {exc}"
            ) from exc

        df = self._clean(df)

        if df.empty:
            raise ValueError(
                f"No usable data returned for {self.ticker} @ {interval}."
            )

        with open(cache_path, "wb") as fh:
            pickle.dump(df, fh)

        logger.info(
            "  → %d bars  [%s … %s]",
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )
        return df

    def fetch_all(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for every timeframe defined in config.TIMEFRAMES."""
        return {
            tf: self.fetch(tf, params["period"], force_refresh)
            for tf, params in config.TIMEFRAMES.items()
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _cache_path(self, interval: str) -> str:
        safe_ticker = self.ticker.replace("=", "_")
        return os.path.join(config.CACHE_DIR, f"{safe_ticker}_{interval}.pkl")

    def _cache_valid(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        return age < self.cache_ttl

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise columns, strip timezone, drop bad rows."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().capitalize() for c in df.columns]

        needed = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in needed if c in df.columns]].copy()

        # Normalise index to timezone-naive UTC
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df.sort_index()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df[df["Close"] > 0]
        df = df[df["High"] >= df["Low"]]
        return df

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Aggregate 1-hour candles to a coarser timeframe."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip().capitalize() for c in df.columns]

        agg = {
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }
        agg = {k: v for k, v in agg.items() if k in df.columns}
        return df.resample(rule, label="left", closed="left").agg(agg).dropna(
            subset=["Close"]
        )
