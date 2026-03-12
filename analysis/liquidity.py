"""
Liquidity Level Analysis
─────────────────────────
• Buy-Side Liquidity  (BSL) – resting above swing highs (buy-stop orders)
• Sell-Side Liquidity (SSL) – resting below swing lows  (sell-stop orders)
• Equal Highs (EQH)         – two or more swing highs within threshold → dense BSL
• Equal Lows  (EQL)         – two or more swing lows  within threshold → dense SSL
• Liquidity Sweeps           – price briefly breaches a level and closes back inside
• Liquidity Runs             – price breaches a level and continues through (trend run)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config as cfg


@dataclass
class LiquidityLevel:
    idx: int
    timestamp: pd.Timestamp
    price: float
    kind: str              # 'BSL' | 'SSL' | 'EQH' | 'EQL'
    swept: bool = False
    swept_idx: Optional[int] = None
    ran_through: bool = False   # liquidity run (full break, not just wick)


@dataclass
class LiquiditySweep:
    idx: int
    timestamp: pd.Timestamp
    price: float
    kind: str              # 'BULL_SWEEP' (SSL swept) | 'BEAR_SWEEP' (BSL swept)
    level_swept: float


def detect_liquidity_levels(
    df: pd.DataFrame,
    lookback: int = cfg.LIQ_LOOKBACK,
    eq_thresh_pct: float = cfg.EQ_LEVEL_THRESH_PCT,
    max_levels: int = cfg.MAX_LIQ_LEVELS,
) -> Tuple[List[LiquidityLevel], List[LiquiditySweep]]:
    """
    Identify BSL/SSL from pivot highs/lows.  Detect EQH/EQL when multiple
    swing levels cluster within `eq_thresh_pct`%.  Mark sweeps on subsequent bars.

    Returns
    -------
    levels : list of LiquidityLevel
    sweeps : list of LiquiditySweep
    """
    n = len(df)
    levels: List[LiquidityLevel] = []
    sweeps: List[LiquiditySweep] = []

    # ── Pivot detection ───────────────────────────────────────────────────────
    pivot_highs: List[Tuple[int, float]] = []
    pivot_lows:  List[Tuple[int, float]] = []

    for i in range(lookback, n - lookback):
        window_h = df["high"].iloc[i - lookback: i + lookback + 1]
        window_l = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window_h.max():
            pivot_highs.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == window_l.min():
            pivot_lows.append((i, df["low"].iloc[i]))

    # ── Classify as BSL / EQH ────────────────────────────────────────────────
    for idx, price in pivot_highs:
        # Check if another pivot high is within threshold
        near_equals = [
            p for pi, p in pivot_highs
            if pi != idx and abs(p - price) / price * 100 <= eq_thresh_pct
        ]
        kind = "EQH" if near_equals else "BSL"
        levels.append(LiquidityLevel(
            idx=idx,
            timestamp=df.index[idx],
            price=price,
            kind=kind,
        ))

    # ── Classify as SSL / EQL ────────────────────────────────────────────────
    for idx, price in pivot_lows:
        near_equals = [
            p for pi, p in pivot_lows
            if pi != idx and abs(p - price) / price * 100 <= eq_thresh_pct
        ]
        kind = "EQL" if near_equals else "SSL"
        levels.append(LiquidityLevel(
            idx=idx,
            timestamp=df.index[idx],
            price=price,
            kind=kind,
        ))

    # Sort chronologically
    levels.sort(key=lambda l: l.idx)

    # ── Sweep / Run detection ─────────────────────────────────────────────────
    for lvl in levels:
        for k in range(lvl.idx + 1, n):
            bar = df.iloc[k]

            # BSL / EQH sweep: wick above, close back below
            if lvl.kind in ("BSL", "EQH"):
                if bar["high"] > lvl.price:
                    lvl.swept = True
                    lvl.swept_idx = k
                    if bar["close"] < lvl.price:
                        # Wick only → sweep (stop hunt), not a run
                        sweeps.append(LiquiditySweep(
                            idx=k,
                            timestamp=df.index[k],
                            price=bar["high"],
                            kind="BEAR_SWEEP",
                            level_swept=lvl.price,
                        ))
                    else:
                        # Close above → liquidity run (trend continuation)
                        lvl.ran_through = True
                    break

            # SSL / EQL sweep: wick below, close back above
            if lvl.kind in ("SSL", "EQL"):
                if bar["low"] < lvl.price:
                    lvl.swept = True
                    lvl.swept_idx = k
                    if bar["close"] > lvl.price:
                        sweeps.append(LiquiditySweep(
                            idx=k,
                            timestamp=df.index[k],
                            price=bar["low"],
                            kind="BULL_SWEEP",
                            level_swept=lvl.price,
                        ))
                    else:
                        lvl.ran_through = True
                    break

    return levels, sweeps


def recent_sweeps(sweeps: List[LiquiditySweep], n_bars: int = 3) -> List[LiquiditySweep]:
    """Return sweeps that occurred within the last n_bars."""
    if not sweeps:
        return []
    last_idx = sweeps[-1].idx
    return [s for s in sweeps if s.idx >= last_idx - n_bars]


def unswept_levels(
    levels: List[LiquidityLevel],
    current_idx: int,
    kind: Optional[str] = None,
) -> List[LiquidityLevel]:
    """Return liquidity levels that have not yet been swept."""
    return [
        l for l in levels
        if l.idx < current_idx
        and not l.swept
        and (kind is None or l.kind in kind)
    ]
