"""
Institutional Phase Detection
───────────────────────────────
Three-phase model (Wyckoff / ICT hybrid):

  ACCUMULATION   – Smart money builds long positions.
                   Characteristics: low volatility (tight range), high absorption,
                   multiple SSL sweeps, CVD divergence (price down, CVD rising).

  MANIPULATION   – Stop hunts to clear liquidity before the real move.
                   Characteristics: liquidity sweep (EQH/EQL/BSL/SSL run),
                   CHoCH immediately following, spike in volume + absorption.

  DISTRIBUTION   – Smart money exits long (or builds short) positions.
                   Characteristics: high volatility, large range, bear imbalance
                   candles near premium, CVD divergence (price up, CVD falling).

  TRENDING       – Straightforward directional price action (BOS chain).

Diagnostic scores are computed per-bar. Each phase score is 0–5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from analysis.liquidity import LiquidityLevel, LiquiditySweep
from analysis.order_flow import OrderFlowReport
import config as cfg


@dataclass
class PhaseSnapshot:
    """Institutional phase assessment at the current bar."""
    accum_score: int       # 0–5
    manip_score: int       # 0–5
    distrib_score: int     # 0–5
    trending_score: int    # 0–5
    phase: str             # dominant phase label
    direction_hint: int    # +1 = expect upside, -1 = expect downside, 0 = unclear

    @property
    def phase_color(self) -> str:
        return {
            "ACCUMULATION": "#9C27B0",
            "MANIPULATION": "#FF9800",
            "DISTRIBUTION": "#EF5350",
            "TRENDING":     "#26A69A",
        }.get(self.phase, "#607D8B")


def detect_phase(
    df: pd.DataFrame,
    of_report: OrderFlowReport,
    sweeps: List[LiquiditySweep],
    trend_series: pd.Series,
    lookback: int = cfg.PHASE_LOOKBACK,
    accum_vol_thresh: float = cfg.ACCUM_VOL_THRESH,
    distrib_vol_thresh: float = cfg.DISTRIB_VOL_THRESH,
) -> PhaseSnapshot:
    """
    Derive the dominant institutional phase for the latest bar.

    Parameters
    ----------
    df           : OHLCV DataFrame
    of_report    : result from analyse_order_flow()
    sweeps       : liquidity sweeps detected
    trend_series : bar-by-bar trend series from detect_structure()
    lookback     : window for volatility comparison
    """
    n = len(df)
    if n < lookback:
        return PhaseSnapshot(0, 0, 0, 0, "TRENDING", 0)

    # ── Volatility regime ─────────────────────────────────────────────────────
    close_std = df["close"].rolling(lookback).std()
    avg_std   = close_std.rolling(lookback * 2).mean()
    vol_ratio = (close_std / avg_std.replace(0, np.nan)).iloc[-1]
    low_vol   = vol_ratio < accum_vol_thresh
    high_vol  = vol_ratio > distrib_vol_thresh

    # ── Recent order flow signals ─────────────────────────────────────────────
    w = min(5, n)   # look at last 5 bars
    latest_of = of_report

    recent_absorption   = latest_of.absorption.iloc[-w:].any()
    recent_bull_div     = latest_of.divergence["bull_div"].iloc[-w:].any()
    recent_bear_div     = latest_of.divergence["bear_div"].iloc[-w:].any()
    recent_bull_flip    = latest_of.flips["bull_flip"].iloc[-w:].any()
    recent_bear_flip    = latest_of.flips["bear_flip"].iloc[-w:].any()
    recent_bull_imb     = latest_of.imbalance["bull_imbalance"].iloc[-w:].any()
    recent_bear_imb     = latest_of.imbalance["bear_imbalance"].iloc[-w:].any()
    cvd_slope_pos       = float(latest_of.cvd_slope.iloc[-1]) > 0

    # ── Recent sweeps ─────────────────────────────────────────────────────────
    recent_sweep_bars = 5
    last_idx = n - 1
    recent_bull_sweep = any(
        s.kind == "BULL_SWEEP" and s.idx >= last_idx - recent_sweep_bars
        for s in sweeps
    )
    recent_bear_sweep = any(
        s.kind == "BEAR_SWEEP" and s.idx >= last_idx - recent_sweep_bars
        for s in sweeps
    )

    # ── Recent CHoCH (structural reversal) ───────────────────────────────────
    recent_trend_change = False
    if len(trend_series) > recent_sweep_bars:
        recent_trends = trend_series.iloc[-recent_sweep_bars:]
        recent_trend_change = (recent_trends.diff().abs() > 0).any()

    # ── Range tightness (another accumulation indicator) ─────────────────────
    recent_range = (df["high"].iloc[-w:] - df["low"].iloc[-w:]).mean()
    avg_range    = (df["high"] - df["low"]).rolling(lookback).mean().iloc[-1]
    tight_range  = recent_range < avg_range * 0.6

    # ── SCORE ACCUMULATION ────────────────────────────────────────────────────
    # Expect upside breakout from accumulation zone
    accum_score = sum([
        int(low_vol),
        int(tight_range),
        int(recent_absorption),
        int(recent_bull_div),
        int(recent_bull_sweep),   # SSL sweep = smart money engineered a buy entry
    ])

    # ── SCORE MANIPULATION ────────────────────────────────────────────────────
    # A stop hunt (sweep) followed by structural reversal = classic manipulation
    manip_score = sum([
        int(recent_bull_sweep or recent_bear_sweep),
        int(recent_trend_change),
        int(recent_absorption),
        int(recent_bull_flip or recent_bear_flip),
        int(not low_vol and not high_vol),   # moderate vol during manip
    ])

    # ── SCORE DISTRIBUTION ────────────────────────────────────────────────────
    # Smart money exiting / reversing at premium
    distrib_score = sum([
        int(high_vol),
        int(recent_bear_imb),
        int(recent_bear_div),
        int(recent_bear_flip),
        int(recent_bear_sweep),  # BSL sweep before reversal
    ])

    # ── SCORE TRENDING ────────────────────────────────────────────────────────
    trending_score = sum([
        int(not low_vol and not high_vol),
        int(recent_bull_imb or recent_bear_imb),
        int(cvd_slope_pos if recent_bull_imb else not cvd_slope_pos),
        int(not recent_trend_change),
        int(not recent_bull_sweep and not recent_bear_sweep),
    ])

    # ── Dominant phase ────────────────────────────────────────────────────────
    scores = {
        "ACCUMULATION": accum_score,
        "MANIPULATION": manip_score,
        "DISTRIBUTION": distrib_score,
        "TRENDING":     trending_score,
    }
    phase = max(scores, key=scores.__getitem__)

    # Direction hint
    direction_hint = 0
    if phase == "ACCUMULATION":
        direction_hint = 1    # expect upside
    elif phase == "MANIPULATION":
        # direction depends on which sweep occurred
        direction_hint = 1 if recent_bull_sweep else (-1 if recent_bear_sweep else 0)
    elif phase == "DISTRIBUTION":
        direction_hint = -1   # expect downside
    elif phase == "TRENDING":
        last_trend = int(trend_series.iloc[-1]) if len(trend_series) > 0 else 0
        direction_hint = last_trend

    return PhaseSnapshot(
        accum_score=accum_score,
        manip_score=manip_score,
        distrib_score=distrib_score,
        trending_score=trending_score,
        phase=phase,
        direction_hint=direction_hint,
    )


def phase_series(
    df: pd.DataFrame,
    of_report: OrderFlowReport,
    sweeps: List[LiquiditySweep],
    trend_series: pd.Series,
    lookback: int = cfg.PHASE_LOOKBACK,
) -> pd.Series:
    """
    Bar-by-bar phase label series (for visualisation / backtesting).
    NOTE: computationally expensive – use only on moderate-length DataFrames.
    """
    labels = []
    n = len(df)
    for i in range(n):
        end = i + 1
        sub_df  = df.iloc[:end]
        sub_trend = trend_series.iloc[:end]

        # Build a partial OrderFlowReport for this slice
        from analysis.order_flow import analyse_order_flow
        sub_of = analyse_order_flow(sub_df)

        # Sweeps up to bar i
        sub_sweeps = [s for s in sweeps if s.idx <= i]

        snap = detect_phase(sub_df, sub_of, sub_sweeps, sub_trend, lookback)
        labels.append(snap.phase)

    return pd.Series(labels, index=df.index, name="phase")
