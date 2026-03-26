"""
Master feature builder — combines technical + SMC + volume profile features.
"""
from __future__ import annotations

import pandas as pd
from features.technical import build_technical_features
from features.smc import build_smc_features
from features.volume_profile import build_volume_features
from utils.logger import get_logger

log = get_logger("FeatureBuilder")


def build_all_features(df: pd.DataFrame, include_volume_profile: bool = True) -> pd.DataFrame:
    """
    Build the full feature matrix from a raw OHLCV DataFrame.

    Parameters
    ----------
    df : raw OHLCV DataFrame (output of data.fetcher)
    include_volume_profile : VP features are slow; disable for quick runs

    Returns
    -------
    DataFrame with all features, NaN rows dropped.
    """
    log.info("Building feature matrix for %d bars…", len(df))

    tech = build_technical_features(df)
    smc  = build_smc_features(df.loc[tech.index])

    parts = [tech, smc]

    if include_volume_profile:
        vp = build_volume_features(df.loc[tech.index])
        parts.append(vp)

    combined = pd.concat(parts, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.dropna(inplace=True)

    log.info("  ✓ Feature matrix: %d rows × %d cols", *combined.shape)
    return combined


FEATURE_GROUPS = {
    "momentum":    ["rsi", "macd_line", "macd_hist", "stoch_k", "stoch_d",
                    "cci", "williams_r", "mfi"],
    "trend":       ["adx", "di_plus", "di_minus", "ema9", "ema21", "ema50",
                    "ema200", "cross_9_21", "cross_21_50", "price_vs_200"],
    "volatility":  ["atr", "bb_width", "bb_pct_b", "rv5", "rv20", "rv_ratio",
                    "garman_klass"],
    "volume":      ["obv", "vwap", "cvd", "cvd_z", "vol_ratio", "mfi"],
    "smc":         ["bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
                    "smc_trend", "in_bull_ob", "in_bear_ob", "bull_fvg", "bear_fvg",
                    "fvg_size", "sweep_highs", "sweep_lows", "dist_from_eq",
                    "in_discount", "in_premium"],
    "vol_profile": ["vp_poc", "vp_vah", "vp_val", "vp_in_va",
                    "price_vs_poc", "price_vs_vah", "price_vs_val"],
    "candle":      ["candle_body", "upper_wick", "lower_wick", "candle_body_norm",
                    "is_bullish", "engulfing", "doji", "hammer"],
}
