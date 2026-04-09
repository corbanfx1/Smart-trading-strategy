"""
Market Structure Analysis – Smart Money Concepts
Detects: swing highs/lows, support/resistance clusters, trend direction,
Break of Structure (BOS), Change of Character (CHoCH), Fair Value Gaps (FVG),
Order Blocks, and computes distance-to-level features.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

import config

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    index: int          # positional index in DataFrame
    timestamp: object   # DatetimeIndex label
    price: float
    kind: str           # 'high' or 'low'
    strength: float     # size of the swing in ATR units


@dataclass
class SRLevel:
    price: float
    touches: int
    kind: str           # 'support' or 'resistance' or 'both'
    strength: float     # normalised strength score


@dataclass
class OrderBlock:
    top: float
    bottom: float
    direction: str      # 'bullish' or 'bearish'
    timestamp: object
    tested: bool = False


@dataclass
class FVG:
    top: float
    bottom: float
    direction: str      # 'bullish' (gap up) or 'bearish' (gap down)
    timestamp: object
    filled: bool = False


@dataclass
class StructureAnalysis:
    swing_highs:   List[SwingPoint] = field(default_factory=list)
    swing_lows:    List[SwingPoint] = field(default_factory=list)
    sr_levels:     List[SRLevel]    = field(default_factory=list)
    order_blocks:  List[OrderBlock] = field(default_factory=list)
    fvgs:          List[FVG]        = field(default_factory=list)
    trend:         str              = "neutral"   # 'bullish', 'bearish', 'neutral'
    bos_signals:   List[dict]       = field(default_factory=list)
    choch_signals: List[dict]       = field(default_factory=list)


class MarketStructureAnalyzer:
    """Perform full market-structure analysis on an OHLCV DataFrame."""

    def __init__(
        self,
        swing_lookback: int = config.SWING_LOOKBACK,
        sr_tolerance:   float = config.SR_CLUSTER_TOLERANCE,
        max_sr_levels:  int = config.MAX_SR_LEVELS,
    ) -> None:
        self.swing_lookback = swing_lookback
        self.sr_tolerance   = sr_tolerance
        self.max_sr_levels  = max_sr_levels

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> StructureAnalysis:
        """Run full market-structure analysis and return structured result."""
        sa = StructureAnalysis()

        atr = self._compute_atr(df)

        sa.swing_highs, sa.swing_lows = self._detect_swings(df, atr)
        sa.trend                      = self._classify_trend(sa.swing_highs, sa.swing_lows)
        sa.sr_levels                  = self._cluster_sr(sa.swing_highs, sa.swing_lows)
        sa.bos_signals                = self._detect_bos(df, sa.swing_highs, sa.swing_lows)
        sa.choch_signals              = self._detect_choch(df, sa.swing_highs, sa.swing_lows)
        sa.order_blocks               = self._detect_order_blocks(df, atr)
        sa.fvgs                       = self._detect_fvg(df, atr)

        return sa

    def add_features(self, df: pd.DataFrame, sa: StructureAnalysis) -> pd.DataFrame:
        """Attach distance / signal features derived from *sa* to *df*."""
        out = df.copy()
        atr = self._compute_atr(df)

        c   = out["Close"]

        # Trend encoding
        trend_map = {"bullish": 1, "neutral": 0, "bearish": -1}
        out["ms_trend"] = trend_map.get(sa.trend, 0)

        # Distance to nearest support / resistance
        supports    = [lvl.price for lvl in sa.sr_levels if lvl.kind in ("support", "both")]
        resistances = [lvl.price for lvl in sa.sr_levels if lvl.kind in ("resistance", "both")]

        out["dist_support"]    = self._dist_to_nearest(c, supports, atr, "below")
        out["dist_resistance"] = self._dist_to_nearest(c, resistances, atr, "above")
        out["near_support"]    = (out["dist_support"] < 1.5).astype(int)
        out["near_resistance"] = (out["dist_resistance"] < 1.5).astype(int)

        # Swing structure features
        out["bars_since_sh"] = self._bars_since_swing(df, sa.swing_highs)
        out["bars_since_sl"] = self._bars_since_swing(df, sa.swing_lows)

        # BOS / CHoCH flags (event on bar)
        bos_idx   = {s["index"] for s in sa.bos_signals}
        choch_idx = {s["index"] for s in sa.choch_signals}
        out["bos"]   = out.index.map(lambda x: int(out.index.get_loc(x) in bos_idx))
        out["choch"] = out.index.map(lambda x: int(out.index.get_loc(x) in choch_idx))

        # Order block proximity
        out["in_bullish_ob"] = self._in_any_block(
            c, [ob for ob in sa.order_blocks if ob.direction == "bullish"]
        )
        out["in_bearish_ob"] = self._in_any_block(
            c, [ob for ob in sa.order_blocks if ob.direction == "bearish"]
        )

        # FVG proximity
        out["in_bullish_fvg"] = self._in_any_fvg(
            c, [f for f in sa.fvgs if f.direction == "bullish"]
        )
        out["in_bearish_fvg"] = self._in_any_fvg(
            c, [f for f in sa.fvgs if f.direction == "bearish"]
        )

        return out

    # ── Swing Detection ───────────────────────────────────────────────────────

    def _detect_swings(
        self, df: pd.DataFrame, atr: pd.Series
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        n       = self.swing_lookback
        highs   = df["High"].values
        lows    = df["Low"].values
        atr_arr = atr.values

        sh_idx = argrelextrema(highs, np.greater_equal, order=n)[0]
        sl_idx = argrelextrema(lows,  np.less_equal,    order=n)[0]

        swing_highs, swing_lows = [], []

        for i in sh_idx:
            if i < n or i >= len(df) - n:
                continue
            strength = (highs[i] - np.mean(highs[max(0, i-n):i])) / (atr_arr[i] + 1e-9)
            if strength > 0.3:
                swing_highs.append(
                    SwingPoint(i, df.index[i], highs[i], "high", float(strength))
                )

        for i in sl_idx:
            if i < n or i >= len(df) - n:
                continue
            strength = (np.mean(lows[max(0, i-n):i]) - lows[i]) / (atr_arr[i] + 1e-9)
            if strength > 0.3:
                swing_lows.append(
                    SwingPoint(i, df.index[i], lows[i], "low", float(strength))
                )

        return swing_highs, swing_lows

    # ── Trend Classification ──────────────────────────────────────────────────

    @staticmethod
    def _classify_trend(
        swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]
    ) -> str:
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "neutral"

        last_sh = sorted(swing_highs, key=lambda x: x.index)[-2:]
        last_sl = sorted(swing_lows,  key=lambda x: x.index)[-2:]

        hh = last_sh[-1].price > last_sh[-2].price   # higher high
        hl = last_sl[-1].price > last_sl[-2].price    # higher low
        lh = last_sh[-1].price < last_sh[-2].price    # lower high
        ll = last_sl[-1].price < last_sl[-2].price    # lower low

        bull_score = int(hh) + int(hl)
        bear_score = int(lh) + int(ll)

        if bull_score >= 2:
            return "bullish"
        if bear_score >= 2:
            return "bearish"
        return "neutral"

    # ── S/R Clustering ────────────────────────────────────────────────────────

    def _cluster_sr(
        self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]
    ) -> List[SRLevel]:
        levels: List[SRLevel] = []
        prices_kinds = (
            [(s.price, "resistance") for s in swing_highs]
            + [(s.price, "support")   for s in swing_lows]
        )

        if not prices_kinds:
            return levels

        prices = [p for p, _ in prices_kinds]
        base   = np.median(prices)
        tol    = base * self.sr_tolerance

        clusters: Dict[float, List[Tuple[float, str]]] = {}

        for price, kind in prices_kinds:
            merged = False
            for anchor in list(clusters.keys()):
                if abs(price - anchor) <= tol:
                    clusters[anchor].append((price, kind))
                    merged = True
                    break
            if not merged:
                clusters[price] = [(price, kind)]

        for anchor, members in clusters.items():
            avg_price = np.mean([p for p, _ in members])
            kinds     = [k for _, k in members]
            dominant  = max(set(kinds), key=kinds.count)
            if kinds.count("resistance") > 0 and kinds.count("support") > 0:
                dominant = "both"
            levels.append(
                SRLevel(
                    price=float(avg_price),
                    touches=len(members),
                    kind=dominant,
                    strength=len(members) / (len(prices_kinds) + 1e-9),
                )
            )

        levels.sort(key=lambda x: -x.touches)
        return levels[: self.max_sr_levels]

    # ── BOS / CHoCH ───────────────────────────────────────────────────────────

    def _detect_bos(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows:  List[SwingPoint],
    ) -> List[dict]:
        signals = []
        closes  = df["Close"].values

        for sh in swing_highs:
            future_mask = [i for i in range(sh.index + 1, len(closes))]
            for i in future_mask:
                if closes[i] > sh.price:
                    signals.append(
                        {"index": i, "timestamp": df.index[i],
                         "kind": "bullish_bos", "level": sh.price}
                    )
                    break

        for sl in swing_lows:
            future_mask = [i for i in range(sl.index + 1, len(closes))]
            for i in future_mask:
                if closes[i] < sl.price:
                    signals.append(
                        {"index": i, "timestamp": df.index[i],
                         "kind": "bearish_bos", "level": sl.price}
                    )
                    break

        return signals

    def _detect_choch(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows:  List[SwingPoint],
    ) -> List[dict]:
        """CHoCH = first BOS that goes against the current trend."""
        trend = self._classify_trend(swing_highs, swing_lows)
        choch = []

        if trend == "bullish":
            # Bearish CHoCH: break below swing low in bullish trend
            for sl in swing_lows[-3:]:
                for i in range(sl.index + 1, len(df)):
                    if df["Close"].iloc[i] < sl.price:
                        choch.append(
                            {"index": i, "timestamp": df.index[i],
                             "kind": "bearish_choch", "level": sl.price}
                        )
                        break
        elif trend == "bearish":
            for sh in swing_highs[-3:]:
                for i in range(sh.index + 1, len(df)):
                    if df["Close"].iloc[i] > sh.price:
                        choch.append(
                            {"index": i, "timestamp": df.index[i],
                             "kind": "bullish_choch", "level": sh.price}
                        )
                        break
        return choch

    # ── Order Blocks ──────────────────────────────────────────────────────────

    def _detect_order_blocks(
        self, df: pd.DataFrame, atr: pd.Series
    ) -> List[OrderBlock]:
        obs = []
        n   = min(config.ORDER_BLOCK_LOOKBACK, len(df) - 1)

        for i in range(2, n):
            # Bullish OB: last down candle before a strong up move
            if (df["Close"].iloc[i] < df["Open"].iloc[i] and           # down candle
                    df["Close"].iloc[i + 1] > df["Open"].iloc[i] and   # next is up
                    (df["Close"].iloc[i + 1] - df["Open"].iloc[i + 1])
                    > atr.iloc[i + 1]):                                 # impulsive
                obs.append(
                    OrderBlock(
                        top=df["Open"].iloc[i],
                        bottom=df["Low"].iloc[i],
                        direction="bullish",
                        timestamp=df.index[i],
                    )
                )
            # Bearish OB: last up candle before a strong down move
            elif (df["Close"].iloc[i] > df["Open"].iloc[i] and
                      df["Close"].iloc[i + 1] < df["Open"].iloc[i + 1] and
                      (df["Open"].iloc[i + 1] - df["Close"].iloc[i + 1])
                      > atr.iloc[i + 1]):
                obs.append(
                    OrderBlock(
                        top=df["High"].iloc[i],
                        bottom=df["Open"].iloc[i],
                        direction="bearish",
                        timestamp=df.index[i],
                    )
                )
        return obs[-20:]   # keep only the most recent

    # ── Fair Value Gaps ───────────────────────────────────────────────────────

    def _detect_fvg(self, df: pd.DataFrame, atr: pd.Series) -> List[FVG]:
        fvgs = []
        min_size = config.FVG_MIN_ATR_SIZE

        for i in range(1, len(df) - 1):
            gap_up   = df["Low"].iloc[i + 1] - df["High"].iloc[i - 1]
            gap_down = df["Low"].iloc[i - 1] - df["High"].iloc[i + 1]

            if gap_up > min_size * atr.iloc[i]:
                fvgs.append(
                    FVG(
                        top=df["Low"].iloc[i + 1],
                        bottom=df["High"].iloc[i - 1],
                        direction="bullish",
                        timestamp=df.index[i],
                    )
                )
            elif gap_down > min_size * atr.iloc[i]:
                fvgs.append(
                    FVG(
                        top=df["Low"].iloc[i - 1],
                        bottom=df["High"].iloc[i + 1],
                        direction="bearish",
                        timestamp=df.index[i],
                    )
                )
        return fvgs[-30:]

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h  = df["High"]
        lo = df["Low"]
        c  = df["Close"]
        tr = pd.concat(
            [h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _dist_to_nearest(
        c: pd.Series, levels: list, atr: pd.Series, side: str
    ) -> pd.Series:
        if not levels:
            return pd.Series(99.0, index=c.index)
        dist = pd.Series(99.0, index=c.index)
        for price in levels:
            if side == "below":
                d = (c - price) / (atr + 1e-9)
                dist = np.minimum(dist, d.clip(lower=0))
            else:
                d = (price - c) / (atr + 1e-9)
                dist = np.minimum(dist, d.clip(lower=0))
        return dist

    @staticmethod
    def _bars_since_swing(df: pd.DataFrame, swings: List[SwingPoint]) -> pd.Series:
        result = pd.Series(999, index=df.index)
        for sw in swings:
            loc = sw.index
            result.iloc[loc:] = range(len(result) - loc)
        return result.clip(upper=999)

    @staticmethod
    def _in_any_block(c: pd.Series, blocks: List[OrderBlock]) -> pd.Series:
        flag = pd.Series(0, index=c.index)
        for ob in blocks:
            flag |= ((c >= ob.bottom) & (c <= ob.top)).astype(int)
        return flag

    @staticmethod
    def _in_any_fvg(c: pd.Series, fvgs: List[FVG]) -> pd.Series:
        flag = pd.Series(0, index=c.index)
        for fvg in fvgs:
            flag |= ((c >= fvg.bottom) & (c <= fvg.top)).astype(int)
        return flag
