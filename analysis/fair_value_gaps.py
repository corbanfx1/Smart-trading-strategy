"""
Fair Value Gap (FVG) / Imbalance Detection
───────────────────────────────────────────
• Bullish FVG  – high[i-2] < low[i]   (gap between prior candle's high and current candle's low)
• Bearish FVG  – low[i-2]  > high[i]  (gap between prior candle's low  and current candle's high)
• Partial fill tracking
• Inverse FVG  – fully filled FVG that subsequently acts as a POI (support/resistance flip)
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

import config as cfg


@dataclass
class FVG:
    idx: int                   # bar index of the middle candle (bar[i-1])
    timestamp: pd.Timestamp
    top: float                 # upper bound of the gap
    bottom: float              # lower bound of the gap
    kind: str                  # 'BULL' | 'BEAR'
    size_pct: float            # gap size as % of mid-price
    filled: bool = False
    filled_idx: Optional[int] = None
    inverse: bool = False      # True once filled and price uses it as opposite POI

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom


def detect_fvgs(
    df: pd.DataFrame,
    min_size_pct: float = cfg.FVG_MIN_SIZE_PCT,
) -> List[FVG]:
    """
    Scan every bar triplet [i-2, i-1, i] for bullish or bearish FVGs.

    Minimum size filter (`min_size_pct`) removes noise.
    Fill status and inverse FVG are evaluated on subsequent bars.
    """
    fvgs: List[FVG] = []
    n = len(df)

    for i in range(2, n):
        h2  = df["high"].iloc[i - 2]
        l2  = df["low"].iloc[i - 2]
        l0  = df["low"].iloc[i]
        h0  = df["high"].iloc[i]
        mid = (df["close"].iloc[i] + df["open"].iloc[i]) / 2

        # Bullish FVG: candle[i-2] high is below candle[i] low
        if h2 < l0:
            gap_pct = (l0 - h2) / h2 * 100 if h2 > 0 else 0
            if gap_pct >= min_size_pct:
                fvgs.append(FVG(
                    idx=i - 1,
                    timestamp=df.index[i - 1],
                    top=l0,
                    bottom=h2,
                    kind="BULL",
                    size_pct=gap_pct,
                ))

        # Bearish FVG: candle[i-2] low is above candle[i] high
        elif l2 > h0:
            gap_pct = (l2 - h0) / l2 * 100 if l2 > 0 else 0
            if gap_pct >= min_size_pct:
                fvgs.append(FVG(
                    idx=i - 1,
                    timestamp=df.index[i - 1],
                    top=l2,
                    bottom=h0,
                    kind="BEAR",
                    size_pct=gap_pct,
                ))

    # ── Fill / Inverse tracking ───────────────────────────────────────────────
    for fvg in fvgs:
        for k in range(fvg.idx + 2, n):   # start 2 bars after FVG formed
            bar = df.iloc[k]
            if fvg.kind == "BULL" and bar["low"] <= fvg.bottom:
                fvg.filled = True
                fvg.filled_idx = k
                break
            if fvg.kind == "BEAR" and bar["high"] >= fvg.top:
                fvg.filled = True
                fvg.filled_idx = k
                break

        # Inverse FVG: if filled, check if price respects it as new POI
        if fvg.filled and fvg.filled_idx is not None:
            check_end = min(fvg.filled_idx + 5, n)
            for k in range(fvg.filled_idx + 1, check_end):
                bar = df.iloc[k]
                if fvg.kind == "BULL" and bar["close"] < fvg.bottom:
                    fvg.inverse = True
                    break
                if fvg.kind == "BEAR" and bar["close"] > fvg.top:
                    fvg.inverse = True
                    break

    return fvgs


def unfilled_fvgs(fvgs: List[FVG], current_idx: int) -> List[FVG]:
    """Return FVGs that have formed but not yet been filled."""
    return [f for f in fvgs if f.idx < current_idx and not f.filled]


def fvgs_at_price(
    fvgs: List[FVG],
    price: float,
    current_idx: int,
    kind: Optional[str] = None,
) -> List[FVG]:
    """Return unfilled FVGs whose zone contains `price`."""
    result = []
    for fvg in unfilled_fvgs(fvgs, current_idx):
        if kind and fvg.kind != kind:
            continue
        if fvg.bottom <= price <= fvg.top:
            result.append(fvg)
    return result
