"""
Order Flow Analysis
────────────────────
• CVD (Cumulative Volume Delta) – estimated buy/sell pressure using candle geometry
• CVD Slope        – rising = buyers dominant, falling = sellers dominant
• CVD Divergence   – price/CVD divergence (leading reversal signal)
• Delta Flip       – bar-level shift from positive to negative delta (or vice versa)
• Absorption       – high volume + small body  → institutional counter-trend absorption
• Imbalance        – high volume + large body  → institutional momentum (order flow)
• Volume Profile   – rolling VPOC / VAH / VAL approximation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

import config as cfg


# ── CVD Computation ───────────────────────────────────────────────────────────

def compute_delta(df: pd.DataFrame) -> pd.Series:
    """
    Estimate per-bar delta (buy volume minus sell volume).

    Uses candle geometry to split total volume:
        buy_vol  = (close - low)  / (high - low) × volume
        sell_vol = (high - close) / (high - low) × volume
        delta    = buy_vol - sell_vol
    """
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    buy_vol  = (df["close"] - df["low"])  / candle_range * df["volume"]
    sell_vol = (df["high"]  - df["close"]) / candle_range * df["volume"]
    delta = buy_vol - sell_vol
    return delta.fillna(0).rename("delta")


def compute_cvd(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume Delta (running sum of per-bar delta)."""
    return compute_delta(df).cumsum().rename("cvd")


def cvd_slope(cvd: pd.Series, period: int = cfg.CVD_PERIOD) -> pd.Series:
    """
    CVD slope = CVD[i] - CVD[i - period].
    Positive → buying pressure building; Negative → selling pressure building.
    """
    return (cvd - cvd.shift(period)).rename("cvd_slope")


def cvd_divergence(
    df: pd.DataFrame,
    cvd: pd.Series,
    period: int = cfg.CVD_PERIOD,
) -> pd.DataFrame:
    """
    Identify bars where price and CVD move in opposite directions (divergence).

    Returns a DataFrame with two boolean columns:
        bull_div : price falling + CVD rising  → hidden bullish divergence
        bear_div : price rising  + CVD falling → hidden bearish divergence
    """
    price_chg = df["close"] - df["close"].shift(period)
    cvd_chg   = cvd - cvd.shift(period)

    bull_div = (price_chg < 0) & (cvd_chg > 0)
    bear_div = (price_chg > 0) & (cvd_chg < 0)

    return pd.DataFrame({"bull_div": bull_div, "bear_div": bear_div}, index=df.index)


def delta_flips(delta: pd.Series) -> pd.DataFrame:
    """
    Identify bars where delta flips sign.
    bull_flip : delta switches from negative to positive (buying pressure returns)
    bear_flip : delta switches from positive to negative (selling pressure returns)
    """
    bull_flip = (delta > 0) & (delta.shift(1) <= 0)
    bear_flip = (delta < 0) & (delta.shift(1) >= 0)
    return pd.DataFrame({"bull_flip": bull_flip, "bear_flip": bear_flip}, index=delta.index)


# ── Absorption & Imbalance ────────────────────────────────────────────────────

def detect_absorption(
    df: pd.DataFrame,
    vol_mult: float = cfg.ABSORPTION_VOL_MULT,
    body_max_pct: float = cfg.ABSORPTION_BODY_MAX,
    window: int = 20,
) -> pd.Series:
    """
    Absorption candles: volume ≥ vol_mult × avg_volume AND body ≤ body_max_pct × avg_body.
    Signals institutional counter-trend accumulation/distribution.
    """
    avg_vol  = df["volume"].rolling(window).mean()
    body     = (df["close"] - df["open"]).abs()
    avg_body = body.rolling(window).mean()

    absorption = (df["volume"] >= avg_vol * vol_mult) & (body <= avg_body * body_max_pct)
    return absorption.rename("absorption")


def detect_imbalance(
    df: pd.DataFrame,
    body_mult: float = cfg.IMBALANCE_BODY_MULT,
    window: int = 20,
) -> pd.DataFrame:
    """
    Imbalance candles: body ≥ body_mult × avg_body.
    Directional body classifies bull or bear imbalance.
    """
    body     = (df["close"] - df["open"]).abs()
    avg_body = body.rolling(window).mean()

    strong = body >= avg_body * body_mult
    bull_imb = strong & (df["close"] > df["open"])
    bear_imb = strong & (df["close"] < df["open"])
    return pd.DataFrame({"bull_imbalance": bull_imb, "bear_imbalance": bear_imb}, index=df.index)


# ── Volume Profile (VPOC) Approximation ───────────────────────────────────────

def rolling_vpoc(
    df: pd.DataFrame,
    window: int = 50,
    price_bins: int = 50,
) -> pd.Series:
    """
    Rolling Point of Control (VPOC) approximation.

    For each bar, compute a histogram of prices (using mid-price + wick coverage)
    weighted by volume over the past `window` bars.  Returns the price bin with
    the highest volume (VPOC).
    """
    n = len(df)
    vpoc_arr = np.full(n, np.nan)

    for i in range(window, n):
        win = df.iloc[i - window: i + 1]
        p_min = win["low"].min()
        p_max = win["high"].max()
        if p_max == p_min:
            vpoc_arr[i] = (p_max + p_min) / 2
            continue

        bins = np.linspace(p_min, p_max, price_bins + 1)
        vol_per_bin = np.zeros(price_bins)

        for _, row in win.iterrows():
            # Spread bar volume across price bins proportionally
            lo, hi = row["low"], row["high"]
            bar_range = hi - lo or (hi * 0.0001)
            for b in range(price_bins):
                bin_lo = bins[b]
                bin_hi = bins[b + 1]
                overlap = min(hi, bin_hi) - max(lo, bin_lo)
                if overlap > 0:
                    vol_per_bin[b] += row["volume"] * overlap / bar_range

        poc_bin = int(np.argmax(vol_per_bin))
        vpoc_arr[i] = (bins[poc_bin] + bins[poc_bin + 1]) / 2

    return pd.Series(vpoc_arr, index=df.index, name="vpoc")


# ── Full Order Flow Report ────────────────────────────────────────────────────

@dataclass
class OrderFlowReport:
    delta: pd.Series
    cvd: pd.Series
    cvd_slope: pd.Series
    divergence: pd.DataFrame
    flips: pd.DataFrame
    absorption: pd.Series
    imbalance: pd.DataFrame
    vpoc: pd.Series

    # Summary for the latest bar
    @property
    def latest(self) -> dict:
        i = -1
        return {
            "delta":         float(self.delta.iloc[i]),
            "cvd":           float(self.cvd.iloc[i]),
            "cvd_slope":     float(self.cvd_slope.iloc[i]),
            "cvd_slope_pos": bool(self.cvd_slope.iloc[i] > 0),
            "bull_div":      bool(self.divergence["bull_div"].iloc[i]),
            "bear_div":      bool(self.divergence["bear_div"].iloc[i]),
            "bull_flip":     bool(self.flips["bull_flip"].iloc[i]),
            "bear_flip":     bool(self.flips["bear_flip"].iloc[i]),
            "absorption":    bool(self.absorption.iloc[i]),
            "bull_imb":      bool(self.imbalance["bull_imbalance"].iloc[i]),
            "bear_imb":      bool(self.imbalance["bear_imbalance"].iloc[i]),
            "vpoc":          float(self.vpoc.iloc[i]) if not np.isnan(self.vpoc.iloc[i]) else None,
        }


def analyse_order_flow(df: pd.DataFrame) -> OrderFlowReport:
    """Run the full order flow analysis pipeline and return an OrderFlowReport."""
    delta  = compute_delta(df)
    cvd    = compute_cvd(df)
    slope  = cvd_slope(cvd)
    diverg = cvd_divergence(df, cvd)
    flips  = delta_flips(delta)
    absorp = detect_absorption(df)
    imbalance = detect_imbalance(df)
    vpoc   = rolling_vpoc(df)

    return OrderFlowReport(
        delta=delta,
        cvd=cvd,
        cvd_slope=slope,
        divergence=diverg,
        flips=flips,
        absorption=absorp,
        imbalance=imbalance,
        vpoc=vpoc,
    )
