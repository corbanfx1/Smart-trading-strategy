"""
Smart Money Concepts (SMC) Feature Engineering.
Detects: BOS, CHoCH, Order Blocks, Fair Value Gaps, Liquidity Sweeps,
         Supply/Demand Zones, Equilibrium levels.
Directly mirrors what LuxAlgo SMC overlay shows on the charts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from config.settings import SMC_PARAMS as SP
from utils.logger import get_logger

log = get_logger("SMCFeatures")


# ─── SWING HIGH / LOW ──────────────────────────────────────────────────────────

def swing_highs_lows(df: pd.DataFrame,
                     lookback: int = SP["swing_lookback"]) -> pd.DataFrame:
    """
    Identify pivot swing highs/lows.
    A swing high is the highest high in [i-lookback : i+lookback+1].
    """
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)
    sh    = np.zeros(n)
    sl    = np.zeros(n)

    for i in range(lookback, n - lookback):
        if high[i] == max(high[max(0, i - lookback): i + lookback + 1]):
            sh[i] = high[i]
        if low[i] == min(low[max(0, i - lookback): i + lookback + 1]):
            sl[i] = low[i]

    return pd.DataFrame({
        "swing_high": sh,
        "swing_low":  sl,
    }, index=df.index)


# ─── MARKET STRUCTURE: BOS / CHoCH ─────────────────────────────────────────────

def market_structure(df: pd.DataFrame,
                     swing_lb: int = SP["swing_lookback"]) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).
    BOS   → price closes beyond a previous swing in the direction of trend
    CHoCH → price closes beyond a previous swing AGAINST the trend (reversal)
    Returns binary flags and direction.
    """
    swings   = swing_highs_lows(df, swing_lb)
    close    = df["close"].values
    n        = len(df)

    bos_bull  = np.zeros(n, int)   # bullish BOS
    bos_bear  = np.zeros(n, int)   # bearish BOS
    choch_bull = np.zeros(n, int)  # bullish CHoCH (bearish structure broken up)
    choch_bear = np.zeros(n, int)  # bearish CHoCH

    sh_vals = swings["swing_high"].values
    sl_vals = swings["swing_low"].values

    # Track last confirmed swing H/L
    last_sh = np.nan
    last_sl = np.nan
    trend   = 0  # +1 up, -1 down

    for i in range(swing_lb, n):
        if sh_vals[i] > 0:
            last_sh = sh_vals[i]
        if sl_vals[i] > 0:
            last_sl = sl_vals[i]

        if np.isnan(last_sh) or np.isnan(last_sl):
            continue

        # Bullish BOS: close above last swing high while trend is up
        if close[i] > last_sh and trend >= 0:
            bos_bull[i] = 1
            trend = 1
        # Bearish BOS: close below last swing low while trend is down
        elif close[i] < last_sl and trend <= 0:
            bos_bear[i] = 1
            trend = -1
        # CHoCH: structure break against existing trend
        elif close[i] > last_sh and trend < 0:
            choch_bull[i] = 1
            trend = 1
        elif close[i] < last_sl and trend > 0:
            choch_bear[i] = 1
            trend = -1

    return pd.DataFrame({
        "bos_bullish":   bos_bull,
        "bos_bearish":   bos_bear,
        "choch_bullish": choch_bull,
        "choch_bearish": choch_bear,
        "smc_trend":     np.where(bos_bull | choch_bull, 1,
                          np.where(bos_bear | choch_bear, -1, 0)),
    }, index=df.index)


# ─── ORDER BLOCKS ──────────────────────────────────────────────────────────────

def order_blocks(df: pd.DataFrame,
                 lookback: int = SP["ob_lookback"]) -> pd.DataFrame:
    """
    Identify bullish/bearish Order Blocks (OBs).
    Bullish OB: last bearish candle before a strong bullish impulse
    Bearish OB: last bullish candle before a strong bearish impulse
    Returns proximity flags (price is within OB zone).
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    bull_ob = np.zeros(n)  # price inside bullish OB
    bear_ob = np.zeros(n)  # price inside bearish OB
    bull_ob_top = np.full(n, np.nan)
    bull_ob_bot = np.full(n, np.nan)
    bear_ob_top = np.full(n, np.nan)
    bear_ob_bot = np.full(n, np.nan)

    ob_zones_bull: list[tuple[float, float, int]] = []  # (top, bot, formed_at)
    ob_zones_bear: list[tuple[float, float, int]] = []

    for i in range(2, n):
        # Bearish candle followed by bullish impulsive move → bullish OB
        if c[i - 1] < o[i - 1]:  # bearish candle
            impulse = (c[i] - o[i]) / max(h[i] - l[i], 1e-9)
            if impulse > 0.6:  # strong bullish body
                ob_zones_bull.append((h[i - 1], l[i - 1], i))

        # Bullish candle followed by bearish impulsive move → bearish OB
        if c[i - 1] > o[i - 1]:  # bullish candle
            impulse = (o[i] - c[i]) / max(h[i] - l[i], 1e-9)
            if impulse > 0.6:  # strong bearish body
                ob_zones_bear.append((h[i - 1], l[i - 1], i))

    # Mark proximity at each bar
    for i in range(n):
        price = c[i]
        for (top, bot, formed) in ob_zones_bull[-lookback:]:
            if bot <= price <= top:
                bull_ob[i] = 1
                bull_ob_top[i] = top
                bull_ob_bot[i] = bot
        for (top, bot, formed) in ob_zones_bear[-lookback:]:
            if bot <= price <= top:
                bear_ob[i] = 1
                bear_ob_top[i] = top
                bear_ob_bot[i] = bot

    return pd.DataFrame({
        "in_bull_ob":   bull_ob,
        "in_bear_ob":   bear_ob,
        "bull_ob_top":  bull_ob_top,
        "bull_ob_bot":  bull_ob_bot,
        "bear_ob_top":  bear_ob_top,
        "bear_ob_bot":  bear_ob_bot,
    }, index=df.index)


# ─── FAIR VALUE GAPS (FVG / IMBALANCE) ────────────────────────────────────────

def fair_value_gaps(df: pd.DataFrame,
                    threshold: float = SP["fvg_threshold"]) -> pd.DataFrame:
    """
    Bullish FVG: high[i-2] < low[i]  (gap between candle i-2 high and candle i low)
    Bearish FVG: low[i-2]  > high[i]
    """
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    bull_fvg = np.zeros(n, int)
    bear_fvg = np.zeros(n, int)
    fvg_size = np.zeros(n)

    for i in range(2, n):
        # Bullish FVG
        if h[i - 2] < l[i]:
            gap = (l[i] - h[i - 2]) / c[i]
            if gap >= threshold:
                bull_fvg[i] = 1
                fvg_size[i] = gap
        # Bearish FVG
        if l[i - 2] > h[i]:
            gap = (l[i - 2] - h[i]) / c[i]
            if gap >= threshold:
                bear_fvg[i] = 1
                fvg_size[i] = -gap

    # Is price inside a recent FVG?
    in_bull_fvg = np.zeros(n, int)
    in_bear_fvg = np.zeros(n, int)
    for i in range(2, n):
        if bull_fvg[i - 2]:
            if h[i - 4] < c[i] < l[i - 2] if i >= 4 else False:
                in_bull_fvg[i] = 1
        if bear_fvg[i - 2]:
            if h[i - 2] < c[i] < l[i - 4] if i >= 4 else False:
                in_bear_fvg[i] = 1

    return pd.DataFrame({
        "bull_fvg":    bull_fvg,
        "bear_fvg":    bear_fvg,
        "fvg_size":    fvg_size,
        "in_bull_fvg": in_bull_fvg,
        "in_bear_fvg": in_bear_fvg,
    }, index=df.index)


# ─── LIQUIDITY SWEEPS ──────────────────────────────────────────────────────────

def liquidity_sweeps(df: pd.DataFrame,
                     atr_col: str = "atr",
                     mult: float = SP["liquidity_mult"]) -> pd.DataFrame:
    """
    Detect equal highs/lows (liquidity pools) and their sweeps.
    Equal H/L: two swing points within *tolerance* of each other.
    Sweep: price breaks beyond the level by less than *mult* × ATR then reverses.
    """
    if atr_col not in df.columns:
        from features.technical import atr as _atr
        df = df.copy()
        df["atr"] = _atr(df).values

    h     = df["high"].values
    l     = df["low"].values
    c     = df["close"].values
    a     = df[atr_col].values
    n     = len(df)

    sweep_high = np.zeros(n, int)  # swept equal highs (bearish)
    sweep_low  = np.zeros(n, int)  # swept equal lows  (bullish)

    lookback = 30
    for i in range(lookback, n):
        window_h = h[i - lookback: i]
        window_l = l[i - lookback: i]
        eq_high  = np.percentile(window_h, 90)
        eq_low   = np.percentile(window_l, 10)
        tol      = mult * a[i]

        if h[i] > eq_high and (h[i] - eq_high) < tol and c[i] < eq_high:
            sweep_high[i] = 1  # fakeout above equal highs → bearish sweep
        if l[i] < eq_low and (eq_low - l[i]) < tol and c[i] > eq_low:
            sweep_low[i] = 1   # fakeout below equal lows → bullish sweep

    return pd.DataFrame({
        "sweep_highs": sweep_high,
        "sweep_lows":  sweep_low,
    }, index=df.index)


# ─── EQUILIBRIUM (50% RETRACEMENT) ────────────────────────────────────────────

def equilibrium_features(df: pd.DataFrame,
                          lookback: int = 50) -> pd.DataFrame:
    """
    Price position relative to the equilibrium (midpoint) of recent swing range.
    Above equilibrium → premium (potential short)
    Below equilibrium → discount (potential long)
    """
    hh = df["high"].rolling(lookback).max()
    ll = df["low"].rolling(lookback).min()
    eq = (hh + ll) / 2
    dist_from_eq = ((df["close"] - eq) / (hh - ll + 1e-9)).rename("dist_from_eq")
    in_discount   = (df["close"] < eq).astype(int).rename("in_discount")
    in_premium    = (df["close"] > eq).astype(int).rename("in_premium")
    eq_normalised = eq.rename("equilibrium")
    return pd.concat([dist_from_eq, in_discount, in_premium, eq_normalised], axis=1)


# ─── MASTER SMC FEATURE BUILDER ────────────────────────────────────────────────

def build_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge all SMC features."""
    log.debug("Building SMC features for %d bars", len(df))

    swings  = swing_highs_lows(df)
    mstruct = market_structure(df)
    obs     = order_blocks(df)
    fvgs    = fair_value_gaps(df)
    sweeps  = liquidity_sweeps(df)
    eqlib   = equilibrium_features(df)

    out = pd.concat([swings, mstruct, obs, fvgs, sweeps, eqlib], axis=1)
    log.debug("  ✓ %d SMC features", out.shape[1])
    return out
