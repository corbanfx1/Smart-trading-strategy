"""
Premium / Discount Zone Analysis
──────────────────────────────────
• Equilibrium (EQ) = 50% of the defined range
• Premium      = 50–75% of range   (sell zone – OBs / FVGs here are optimal shorts)
• Deep Premium = 75–100% of range  (extreme sell zone)
• Discount     = 25–50% of range   (buy zone  – OBs / FVGs here are optimal longs)
• Deep Discount = 0–25% of range   (extreme buy zone)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

import config as cfg


@dataclass
class PDContext:
    range_high: float
    range_low: float
    equilibrium: float
    p75: float         # deep premium threshold
    d25: float         # deep discount threshold
    current_price: float
    zone: str          # 'DEEP_PREMIUM' | 'PREMIUM' | 'EQUILIBRIUM' | 'DISCOUNT' | 'DEEP_DISCOUNT'
    pct_from_eq: float # signed % distance from equilibrium (positive = premium side)

    @property
    def in_premium(self) -> bool:
        return self.current_price > self.equilibrium

    @property
    def in_discount(self) -> bool:
        return self.current_price < self.equilibrium

    @property
    def in_deep_premium(self) -> bool:
        return self.current_price >= self.p75

    @property
    def in_deep_discount(self) -> bool:
        return self.current_price <= self.d25

    @property
    def optimal_long_zone(self) -> bool:
        """Price is in discount – ideal for long POIs."""
        return self.in_discount

    @property
    def optimal_short_zone(self) -> bool:
        """Price is in premium – ideal for short POIs."""
        return self.in_premium


def compute_pd_context(
    df: pd.DataFrame,
    range_length: int = cfg.PD_RANGE_LENGTH,
    premium_pct: float = cfg.PD_PREMIUM_PCT,
    discount_pct: float = cfg.PD_DISCOUNT_PCT,
    current_price: Optional[float] = None,
) -> PDContext:
    """
    Compute premium/discount context for the most recent `range_length` bars.

    Parameters
    ----------
    df            : OHLCV DataFrame
    range_length  : number of bars defining the range
    premium_pct   : fraction above which price is in deep premium (default 0.75)
    discount_pct  : fraction below which price is in deep discount (default 0.25)
    current_price : defaults to latest close
    """
    window = df.iloc[-range_length:] if len(df) >= range_length else df

    r_high = window["high"].max()
    r_low  = window["low"].min()
    r_range = r_high - r_low

    if r_range == 0:
        r_range = r_high * 0.001   # guard against flat data

    eq  = r_low + r_range * 0.50
    p75 = r_low + r_range * premium_pct
    d25 = r_low + r_range * discount_pct

    price = current_price if current_price is not None else df["close"].iloc[-1]

    if price >= p75:
        zone = "DEEP_PREMIUM"
    elif price > eq:
        zone = "PREMIUM"
    elif price <= d25:
        zone = "DEEP_DISCOUNT"
    elif price < eq:
        zone = "DISCOUNT"
    else:
        zone = "EQUILIBRIUM"

    pct_from_eq = (price - eq) / eq * 100

    return PDContext(
        range_high=r_high,
        range_low=r_low,
        equilibrium=eq,
        p75=p75,
        d25=d25,
        current_price=price,
        zone=zone,
        pct_from_eq=pct_from_eq,
    )


def pd_series(
    df: pd.DataFrame,
    range_length: int = cfg.PD_RANGE_LENGTH,
) -> pd.Series:
    """
    Return a bar-by-bar series of P/D zone labels.
    Useful for vectorised backtesting.
    """
    n = len(df)
    zones = []
    for i in range(n):
        start = max(0, i - range_length + 1)
        win = df.iloc[start: i + 1]
        r_h = win["high"].max()
        r_l = win["low"].min()
        r   = r_h - r_l or r_h * 0.001
        eq  = r_l + r * 0.50
        p75 = r_l + r * 0.75
        d25 = r_l + r * 0.25
        p   = df["close"].iloc[i]
        if p >= p75:
            zones.append("DEEP_PREMIUM")
        elif p > eq:
            zones.append("PREMIUM")
        elif p <= d25:
            zones.append("DEEP_DISCOUNT")
        elif p < eq:
            zones.append("DISCOUNT")
        else:
            zones.append("EQUILIBRIUM")
    return pd.Series(zones, index=df.index, name="pd_zone")
