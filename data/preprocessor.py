"""
Data Preprocessing & Multi-Timeframe Alignment
Cleans raw OHLCV, engineers log-returns, aligns higher-TF data onto the
primary (1-hour) index using forward-fill (no look-ahead).
"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Transform and align raw OHLCV DataFrames for the ML pipeline."""

    def __init__(self) -> None:
        self.primary_tf      = config.PRIMARY_TF
        self.intermediate_tf = config.INTERMEDIATE_TF
        self.anchor_tf       = config.ANCHOR_TF

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self, raw: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Return a dict of cleaned, enriched DataFrames keyed by interval."""
        processed = {}
        for tf, df in raw.items():
            processed[tf] = self._enrich(df.copy(), tf)
            logger.debug("Processed %s: %d rows", tf, len(processed[tf]))
        return processed

    def align_htf_features(
        self,
        primary_df: pd.DataFrame,
        htf_df: pd.DataFrame,
        htf_prefix: str,
        feature_cols: list,
    ) -> pd.DataFrame:
        """
        Forward-fill selected *feature_cols* from *htf_df* into *primary_df*.

        Uses reindex + ffill so HTF bar N is visible only from its open time
        onward on the primary TF — no look-ahead.
        """
        subset = htf_df[feature_cols].copy()
        subset.columns = [f"{htf_prefix}_{c}" for c in feature_cols]

        merged = primary_df.join(subset, how="left")
        merged[subset.columns] = merged[subset.columns].ffill()
        return merged

    def build_aligned_frame(
        self, processed: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge all timeframes onto the primary TF index.
        Returns a single DataFrame with columns from 1H, 4H and 1D.
        """
        primary = processed[self.primary_tf].copy()
        primary.columns = [f"1h_{c}" if c not in ("Open", "High", "Low", "Close", "Volume")
                           else c for c in primary.columns]

        # 4-Hour alignment
        htf4h = processed[self.intermediate_tf]
        propagate_4h = [c for c in htf4h.columns
                        if c not in ("Open", "High", "Low", "Close", "Volume")]
        primary = self.align_htf_features(primary, htf4h, "4h", propagate_4h)

        # Daily alignment
        htf1d = processed[self.anchor_tf]
        propagate_1d = [c for c in htf1d.columns
                        if c not in ("Open", "High", "Low", "Close", "Volume")]
        primary = self.align_htf_features(primary, htf1d, "1d", propagate_1d)

        # Drop rows before all HTF features are populated
        primary = primary.dropna(
            subset=[c for c in primary.columns if c.startswith("1d_")]
        )

        logger.info(
            "Aligned frame: %d rows, %d columns (primary=%s)",
            len(primary),
            len(primary.columns),
            self.primary_tf,
        )
        return primary

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _enrich(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Add derived columns that are TF-agnostic."""
        c = df["Close"]
        o = df["Open"]
        h = df["High"]
        lo = df["Low"]

        # Returns
        df["log_ret"]    = np.log(c / c.shift(1))
        df["ret_1"]      = c.pct_change(1)
        df["ret_5"]      = c.pct_change(5)
        df["ret_10"]     = c.pct_change(10)
        df["ret_20"]     = c.pct_change(20)

        # Candle anatomy
        rng = (h - lo).replace(0, np.nan)
        df["body"]       = (c - o) / c          # signed body as % of close
        df["wick_up"]    = (h - np.maximum(o, c)) / rng
        df["wick_down"]  = (np.minimum(o, c) - lo) / rng
        df["body_ratio"] = np.abs(c - o) / rng   # body / full range

        # Gap (open vs prev close)
        df["gap"]        = (o - c.shift(1)) / c.shift(1)

        # Intrabar range as fraction of close
        df["range_pct"]  = rng / c

        # Historical volatility (20-period rolling std of log returns, annualised)
        bars_per_year    = config.TIMEFRAMES[tf]["bars_per_day"] * 252
        df["hvol_20"]    = df["log_ret"].rolling(20).std() * np.sqrt(bars_per_year)
        df["hvol_60"]    = df["log_ret"].rolling(60).std() * np.sqrt(bars_per_year)

        return df
