"""
Multi-Timeframe Confluence Score Calculator
Aggregates trend, momentum and structure signals across 1H, 4H and 1D
and produces a single confluence score (−1 full bear → +1 full bull).
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class MTFConfluenceCalculator:
    """
    Compute per-bar MTF confluence features on the primary (1H) DataFrame
    that already has forward-filled 4H and 1D columns (via DataPreprocessor).
    """

    # Prefixes added by DataPreprocessor.align_htf_features
    _PREFIXES = {"1h": "", "4h": "4h_", "1d": "1d_"}
    _TF_WEIGHTS = {"1h": 0.20, "4h": 0.30, "1d": 0.50}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns *df* augmented with:
          - confluence_score         : weighted average direction score (−1 to +1)
          - confluence_bull          : bullish confluences (0–3)
          - confluence_bear          : bearish confluences (0–3)
          - mtf_aligned_bull         : 1 if all 3 TFs bullish
          - mtf_aligned_bear         : 1 if all 3 TFs bearish
          - mtf_trend_strength       : 0–1 agreement strength
        Per-TF sub-scores are also added as tf_score_1h / tf_score_4h / tf_score_1d.
        """
        out = df.copy()

        tf_scores = {}
        for tf, prefix in [("1h", ""), ("4h", "4h_"), ("1d", "1d_")]:
            tf_scores[tf] = self._tf_direction_score(out, prefix)
            out[f"tf_score_{tf}"] = tf_scores[tf]

        # Weighted confluence score
        out["confluence_score"] = sum(
            tf_scores[tf] * w for tf, w in self._TF_WEIGHTS.items()
        )

        # Count aligned timeframes
        out["confluence_bull"] = sum(
            (tf_scores[tf] > 0.2).astype(int) for tf in ["1h", "4h", "1d"]
        )
        out["confluence_bear"] = sum(
            (tf_scores[tf] < -0.2).astype(int) for tf in ["1h", "4h", "1d"]
        )

        out["mtf_aligned_bull"] = (out["confluence_bull"] == 3).astype(int)
        out["mtf_aligned_bear"] = (out["confluence_bear"] == 3).astype(int)

        out["mtf_trend_strength"] = out[["tf_score_1h", "tf_score_4h", "tf_score_1d"]]\
            .abs()\
            .mean(axis=1)

        # Momentum divergence flag
        out["mtf_momentum_div"] = self._detect_momentum_divergence(out)

        logger.debug("MTF confluence columns added.")
        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _tf_direction_score(df: pd.DataFrame, prefix: str) -> pd.Series:
        """
        Return a score ∈ [−1, +1] for a given timeframe using available
        indicator columns.  Missing columns are silently skipped.
        """
        scores = pd.DataFrame(index=df.index)

        def col(name):
            return df.get(f"{prefix}{name}", pd.Series(np.nan, index=df.index))

        # EMA trend stack  (full bull=+1, full bear=−1)
        ema_ts = col("ema_trend_score")
        if ema_ts.notna().any():
            scores["ema_trend"] = ema_ts * 2 - 1   # rescale 0–1 → −1–+1

        # RSI positioning
        rsi = col("rsi_14")
        if rsi.notna().any():
            scores["rsi_bias"] = (rsi - 50) / 50   # −1=oversold, +1=overbought

        # MACD cross
        mc = col("macd_cross")
        if mc.notna().any():
            scores["macd_dir"] = mc * 2 - 1

        # ADX directional
        adx  = col("adx_14")
        di_d = col("di_diff")
        if adx.notna().any() and di_d.notna().any():
            strong = (adx > 25).astype(float)
            scores["adx_dir"] = (di_d / (di_d.abs().rolling(20).mean() + 1e-9)).clip(-1, 1) * strong

        # Bollinger Band position
        bb_pos = col("bb_position")
        if bb_pos.notna().any():
            scores["bb_dir"] = (bb_pos - 0.5) * 2  # 0=lower band → −1, 1=upper band → +1

        # PSAR signal
        psar_sig = col("psar_signal")
        if psar_sig.notna().any():
            scores["psar_dir"] = psar_sig.clip(-1, 1)

        if scores.empty:
            return pd.Series(0.0, index=df.index)

        # Equal-weight average of available sub-scores
        return scores.mean(axis=1).clip(-1, 1)

    @staticmethod
    def _detect_momentum_divergence(df: pd.DataFrame) -> pd.Series:
        """
        Simple price/RSI divergence flag:
          +1 = bullish divergence (price new low, RSI higher low)
          −1 = bearish divergence (price new high, RSI lower high)
           0 = no divergence
        """
        n   = 14
        c   = df.get("Close", pd.Series(np.nan, index=df.index))
        rsi = df.get("rsi_14", df.get("1h_rsi_14", pd.Series(np.nan, index=df.index)))

        if rsi.isna().all():
            return pd.Series(0, index=df.index)

        roll_low_c   = c.rolling(n).min()
        roll_high_c  = c.rolling(n).max()
        roll_low_r   = rsi.rolling(n).min()
        roll_high_r  = rsi.rolling(n).max()

        bull_div = (
            (c == roll_low_c) &
            (rsi > roll_low_r)
        ).astype(int)

        bear_div = (
            (c == roll_high_c) &
            (rsi < roll_high_r)
        ).astype(int) * -1

        return (bull_div + bear_div).clip(-1, 1)
