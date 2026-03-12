"""
Market Structure Analysis
─────────────────────────
• Swing High / Swing Low detection
• Break of Structure (BOS)
• Change of Character (CHoCH)
• Trend state tracking
• Multi-timeframe bias derivation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config as cfg


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    idx: int               # bar index in the DataFrame
    timestamp: pd.Timestamp
    price: float
    kind: str              # 'high' | 'low'


@dataclass
class StructureEvent:
    idx: int
    timestamp: pd.Timestamp
    price: float
    event_type: str        # 'BOS_BULL' | 'BOS_BEAR' | 'CHOCH_BULL' | 'CHOCH_BEAR'
    broken_swing: float    # swing level that was broken


@dataclass
class MarketStructureResult:
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    events: List[StructureEvent] = field(default_factory=list)
    trend: pd.Series = field(default_factory=pd.Series)     # 1=bull, -1=bear, 0=neutral
    bias_4h: int = 0   # 1=bull, -1=bear, 0=neutral
    bias_daily: int = 0


# ── Swing Detection ───────────────────────────────────────────────────────────

def find_swings(
    df: pd.DataFrame,
    lookback: int = cfg.SWING_LOOKBACK,
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Detect swing highs and lows using a pivot-point method.

    A swing high at bar i: high[i] is the highest value in
        [i - lookback, i + lookback].
    A swing low  at bar i: low[i]  is the lowest  value in
        [i - lookback, i + lookback].
    """
    highs: List[SwingPoint] = []
    lows:  List[SwingPoint] = []
    n = len(df)

    for i in range(lookback, n - lookback):
        window_h = df["high"].iloc[i - lookback: i + lookback + 1]
        window_l = df["low"].iloc[i - lookback: i + lookback + 1]

        if df["high"].iloc[i] == window_h.max():
            highs.append(SwingPoint(
                idx=i,
                timestamp=df.index[i],
                price=df["high"].iloc[i],
                kind="high",
            ))
        if df["low"].iloc[i] == window_l.min():
            lows.append(SwingPoint(
                idx=i,
                timestamp=df.index[i],
                price=df["low"].iloc[i],
                kind="low",
            ))
    return highs, lows


# ── BOS / CHoCH Detection ─────────────────────────────────────────────────────

def detect_structure(
    df: pd.DataFrame,
    lookback: int = cfg.SWING_LOOKBACK,
) -> MarketStructureResult:
    """
    Full market structure analysis: swing detection → BOS / CHoCH → trend.

    Returns a MarketStructureResult with all swing points, structure events,
    and a bar-by-bar trend series.
    """
    swing_highs, swing_lows = find_swings(df, lookback)
    result = MarketStructureResult(swing_highs=swing_highs, swing_lows=swing_lows)

    n = len(df)
    trend_arr = np.zeros(n, dtype=int)
    current_trend = 0  # 0 = undefined, 1 = bull, -1 = bear

    # Pointers into the swing lists
    sh_ptr = 0  # next swing high to check
    sl_ptr = 0  # next swing low  to check

    for i in range(lookback, n):
        # Advance swing pointers up to bar i
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr].idx <= i:
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr].idx <= i:
            sl_ptr += 1

        # Most recent confirmed swings
        last_sh = swing_highs[sh_ptr - 1] if sh_ptr > 0 else None
        last_sl = swing_lows[sl_ptr - 1]  if sl_ptr > 0 else None

        bar_high = df["high"].iloc[i]
        bar_low  = df["low"].iloc[i]
        prev_high = df["high"].iloc[i - 1]
        prev_low  = df["low"].iloc[i - 1]

        # ── Bullish break of swing high ──────────────────────────────────────
        if last_sh is not None:
            broke_high = bar_high > last_sh.price and prev_high <= last_sh.price

            if broke_high:
                if current_trend >= 0:
                    # In uptrend (or neutral) → BOS bullish (trend continuation)
                    evt = StructureEvent(
                        idx=i,
                        timestamp=df.index[i],
                        price=bar_high,
                        event_type="BOS_BULL",
                        broken_swing=last_sh.price,
                    )
                else:
                    # In downtrend → CHoCH bullish (reversal signal)
                    evt = StructureEvent(
                        idx=i,
                        timestamp=df.index[i],
                        price=bar_high,
                        event_type="CHOCH_BULL",
                        broken_swing=last_sh.price,
                    )
                result.events.append(evt)
                current_trend = 1

        # ── Bearish break of swing low ───────────────────────────────────────
        if last_sl is not None:
            broke_low = bar_low < last_sl.price and prev_low >= last_sl.price

            if broke_low:
                if current_trend <= 0:
                    evt = StructureEvent(
                        idx=i,
                        timestamp=df.index[i],
                        price=bar_low,
                        event_type="BOS_BEAR",
                        broken_swing=last_sl.price,
                    )
                else:
                    evt = StructureEvent(
                        idx=i,
                        timestamp=df.index[i],
                        price=bar_low,
                        event_type="CHOCH_BEAR",
                        broken_swing=last_sl.price,
                    )
                result.events.append(evt)
                current_trend = -1

        trend_arr[i] = current_trend

    result.trend = pd.Series(trend_arr, index=df.index, name="trend")
    return result


# ── Multi-Timeframe Bias ──────────────────────────────────────────────────────

def htf_bias(df: pd.DataFrame, n_bars: int = 3) -> int:
    """
    Compute a simple directional bias from the last n_bars.
    Returns 1 (bullish), -1 (bearish), or 0 (neutral).
    """
    if len(df) < n_bars + 1:
        return 0
    closes = df["close"].iloc[-n_bars - 1:]
    highs  = df["high"].iloc[-n_bars - 1:]
    lows   = df["low"].iloc[-n_bars - 1:]

    bull = closes.iloc[-1] > closes.iloc[0] and highs.iloc[-1] > highs.iloc[0]
    bear = closes.iloc[-1] < closes.iloc[0] and lows.iloc[-1]  < lows.iloc[0]
    return 1 if bull else (-1 if bear else 0)


def mtf_bias(df_4h: pd.DataFrame, df_daily: pd.DataFrame) -> int:
    """
    Returns combined MTF bias:
    +1 if both 4H and Daily are bullish,
    -1 if both are bearish,
     0 otherwise.
    """
    b4h  = htf_bias(df_4h)
    bd   = htf_bias(df_daily)
    if b4h == 1 and bd >= 0:
        return 1
    if b4h == -1 and bd <= 0:
        return -1
    return 0
