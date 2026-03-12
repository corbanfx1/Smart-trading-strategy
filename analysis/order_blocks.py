"""
Order Block (OB) Detection
───────────────────────────
• Bullish OB  – last bearish candle before a bullish BOS/CHoCH
• Bearish OB  – last bullish candle before a bearish BOS/CHoCH
• Mitigation check – OB is "mitigated" when price returns ≥ 50% into the zone
• Breaker Block – a mitigated OB that then rejects price (polarity flip)
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from analysis.market_structure import StructureEvent
import config as cfg


@dataclass
class OrderBlock:
    idx_origin: int            # bar index of the originating candle
    timestamp: pd.Timestamp    # timestamp of the originating candle
    top: float                 # upper boundary of the OB zone
    bottom: float              # lower boundary of the OB zone
    kind: str                  # 'BULL' | 'BEAR'
    triggered_by: str          # 'BOS_BULL' | 'BOS_BEAR' | 'CHOCH_BULL' | 'CHOCH_BEAR'
    mitigated: bool = False
    mitigated_idx: Optional[int] = None
    breaker: bool = False      # becomes True if OB gets mitigated and then acts as resistance/support

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom


def detect_order_blocks(
    df: pd.DataFrame,
    events: List[StructureEvent],
    lookback: int = cfg.OB_LOOKBACK,
    mitigation_pct: float = cfg.OB_MITIGATION_PCT,
) -> List[OrderBlock]:
    """
    For every BOS/CHoCH event, locate the last opposing candle within `lookback`
    bars before the event bar.  That candle defines the Order Block zone.

    Mitigation is then checked on subsequent bars.
    """
    obs: List[OrderBlock] = []
    n = len(df)

    for evt in events:
        i = evt.idx
        looking_bull_ob = evt.event_type in ("BOS_BULL", "CHOCH_BULL")
        looking_bear_ob = evt.event_type in ("BOS_BEAR", "CHOCH_BEAR")

        # Search backwards from bar i for the first qualifying candle
        found = False
        for j in range(1, min(lookback + 1, i + 1)):
            bar = df.iloc[i - j]
            is_bearish = bar["close"] < bar["open"]
            is_bullish = bar["close"] > bar["open"]

            if looking_bull_ob and is_bearish:
                obs.append(OrderBlock(
                    idx_origin=i - j,
                    timestamp=df.index[i - j],
                    top=bar["open"],        # OB zone: body top to low
                    bottom=bar["low"],
                    kind="BULL",
                    triggered_by=evt.event_type,
                ))
                found = True
                break

            if looking_bear_ob and is_bullish:
                obs.append(OrderBlock(
                    idx_origin=i - j,
                    timestamp=df.index[i - j],
                    top=bar["high"],        # OB zone: high to body bottom
                    bottom=bar["open"],
                    kind="BEAR",
                    triggered_by=evt.event_type,
                ))
                found = True
                break

    # ── Check mitigation for every OB ────────────────────────────────────────
    for ob in obs:
        mitigation_level = ob.bottom + (ob.top - ob.bottom) * mitigation_pct

        for k in range(ob.idx_origin + 1, n):
            bar = df.iloc[k]
            if ob.kind == "BULL" and bar["low"] <= mitigation_level:
                ob.mitigated = True
                ob.mitigated_idx = k
                break
            if ob.kind == "BEAR" and bar["high"] >= mitigation_level:
                ob.mitigated = True
                ob.mitigated_idx = k
                break

        # Breaker: OB is mitigated AND subsequent price respects it (polarity flip)
        if ob.mitigated and ob.mitigated_idx is not None:
            check_end = min(ob.mitigated_idx + 5, n)
            for k in range(ob.mitigated_idx + 1, check_end):
                bar = df.iloc[k]
                if ob.kind == "BULL" and bar["close"] < ob.bottom:
                    ob.breaker = True
                    break
                if ob.kind == "BEAR" and bar["close"] > ob.top:
                    ob.breaker = True
                    break

    return obs


def active_obs(obs: List[OrderBlock], current_idx: int) -> List[OrderBlock]:
    """Return OBs that have been identified but not yet mitigated."""
    return [ob for ob in obs if ob.idx_origin < current_idx and not ob.mitigated]


def obs_in_zone(
    obs: List[OrderBlock],
    price: float,
    current_idx: int,
    kind: Optional[str] = None,
) -> List[OrderBlock]:
    """Return active OBs whose zone contains `price`."""
    result = []
    for ob in active_obs(obs, current_idx):
        if kind and ob.kind != kind:
            continue
        if ob.bottom <= price <= ob.top:
            result.append(ob)
    return result
