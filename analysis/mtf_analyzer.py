"""
Multi-Timeframe Analyzer (Daily + 4H + 1H).
Performs a top-down structural bias assessment:
  1D  → macro trend & key S/R levels
  4H  → intermediate trend, order blocks, FVGs
  1H  → entry-level signals and fine structure

Produces a consolidated MTF bias dict consumed by the MetaFilter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from features.technical import atr, rsi, macd, adx, ema_features
from features.smc import market_structure, order_blocks, fair_value_gaps, swing_highs_lows
from config.settings import SIGNAL
from utils.logger import get_logger

log = get_logger("MTFAnalyzer")


# ─── SINGLE-TF BIAS ────────────────────────────────────────────────────────────

def _tf_bias(df: pd.DataFrame, label: str) -> dict:
    """
    Assess directional bias for a single timeframe.
    Returns a dict with:
      bias      : +1 / 0 / -1
      confidence: float 0-1
      reasons   : list of str
    """
    if len(df) < 50:
        return {"bias": 0, "confidence": 0.0, "reasons": ["insufficient_data"], "label": label}

    reasons  = []
    scores   = []

    # ── EMA Trend ──────────────────────────────────────────────────────────────
    emas = ema_features(df["close"])
    c    = df["close"].iloc[-1]
    e50  = emas["ema50"].iloc[-1]
    e200 = emas["ema200"].iloc[-1]

    if c > e50 > e200:
        scores.append(1)
        reasons.append("price_above_ema50_200")
    elif c < e50 < e200:
        scores.append(-1)
        reasons.append("price_below_ema50_200")
    else:
        scores.append(0)

    # ── ADX / DI ───────────────────────────────────────────────────────────────
    adx_df = adx(df)
    adx_v  = adx_df["adx"].iloc[-1]
    di_p   = adx_df["di_plus"].iloc[-1]
    di_m   = adx_df["di_minus"].iloc[-1]

    if adx_v > 25:
        trend_dir = 1 if di_p > di_m else -1
        scores.append(trend_dir)
        reasons.append(f"adx_trending_{adx_v:.0f}")
    else:
        scores.append(0)
        reasons.append("adx_ranging")

    # ── RSI Regime ─────────────────────────────────────────────────────────────
    rsi_v = rsi(df["close"]).iloc[-1]
    if rsi_v > 60:
        scores.append(1)
        reasons.append(f"rsi_bullish_{rsi_v:.0f}")
    elif rsi_v < 40:
        scores.append(-1)
        reasons.append(f"rsi_bearish_{rsi_v:.0f}")
    else:
        scores.append(0)

    # ── MACD ───────────────────────────────────────────────────────────────────
    macd_df = macd(df["close"])
    macd_h  = macd_df["macd_hist"].iloc[-1]
    macd_h_prev = macd_df["macd_hist"].iloc[-2] if len(macd_df) > 1 else 0
    if macd_h > 0 and macd_h > macd_h_prev:
        scores.append(1)
        reasons.append("macd_hist_bullish_expanding")
    elif macd_h < 0 and macd_h < macd_h_prev:
        scores.append(-1)
        reasons.append("macd_hist_bearish_expanding")
    else:
        scores.append(0)

    # ── SMC Market Structure ───────────────────────────────────────────────────
    ms = market_structure(df)
    last_ms = ms.iloc[-5:]  # last 5 bars
    if last_ms["bos_bullish"].any() or last_ms["choch_bullish"].any():
        scores.append(1)
        reasons.append("smc_bullish_structure")
    elif last_ms["bos_bearish"].any() or last_ms["choch_bearish"].any():
        scores.append(-1)
        reasons.append("smc_bearish_structure")
    else:
        trend_score = ms["smc_trend"].iloc[-1]
        scores.append(trend_score)

    # ── Price vs Key Levels ────────────────────────────────────────────────────
    swing_df = swing_highs_lows(df, lookback=15)
    recent_sh = swing_df["swing_high"].replace(0, np.nan).dropna()
    recent_sl = swing_df["swing_low"].replace(0, np.nan).dropna()

    if len(recent_sh) >= 2 and len(recent_sl) >= 2:
        higher_highs = recent_sh.iloc[-1] > recent_sh.iloc[-2]
        higher_lows  = recent_sl.iloc[-1] > recent_sl.iloc[-2]
        if higher_highs and higher_lows:
            scores.append(1)
            reasons.append("higher_highs_higher_lows")
        elif not higher_highs and not higher_lows:
            scores.append(-1)
            reasons.append("lower_highs_lower_lows")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    if not scores:
        return {"bias": 0, "confidence": 0.0, "reasons": reasons, "label": label}

    avg    = np.mean(scores)
    bias   = 1 if avg > 0.2 else (-1 if avg < -0.2 else 0)
    conf   = min(1.0, abs(avg))

    return {
        "bias":       bias,
        "confidence": conf,
        "reasons":    reasons,
        "label":      label,
        "raw_score":  avg,
        "rsi":        rsi_v,
        "adx":        adx_v,
        "close":      c,
    }


# ─── KEY LEVEL EXTRACTION ──────────────────────────────────────────────────────

def extract_key_levels(df_daily: pd.DataFrame,
                        df_4h:    pd.DataFrame,
                        current_price: float) -> dict:
    """
    Extract institutional key levels visible on the charts:
    - PDH/PDL (Previous Day High/Low)
    - PWH/PWL (Previous Week High/Low)
    - PMH/PML (Previous Month High/Low)
    - EQL (Equal Lows/Highs — potential liquidity)
    - Equilibrium of current swing range
    """
    levels = {}

    if len(df_daily) >= 2:
        levels["PDH"] = float(df_daily["high"].iloc[-2])
        levels["PDL"] = float(df_daily["low"].iloc[-2])

    if len(df_daily) >= 6:
        last_week = df_daily.iloc[-6:-1]
        levels["PWH"] = float(last_week["high"].max())
        levels["PWL"] = float(last_week["low"].min())

    if len(df_daily) >= 22:
        last_month = df_daily.iloc[-22:-1]
        levels["PMH"] = float(last_month["high"].max())
        levels["PML"] = float(last_month["low"].min())

    # Equilibrium of recent range
    recent = df_daily.iloc[-20:]
    eq = (recent["high"].max() + recent["low"].min()) / 2
    levels["EQ_20D"] = float(eq)

    # Distance from each level (normalised by ATR)
    atr_v = float(atr(df_daily).iloc[-1])
    near_levels = {}
    for name, lvl in levels.items():
        dist_atr = abs(current_price - lvl) / max(atr_v, 1e-9)
        near_levels[name] = {
            "level":    lvl,
            "dist_atr": round(dist_atr, 2),
            "side":     "above" if current_price > lvl else "below",
        }

    return near_levels


# ─── ORDER BLOCK MAP ───────────────────────────────────────────────────────────

def nearest_order_blocks(df: pd.DataFrame, current_price: float) -> dict:
    """Find the nearest bullish/bearish OBs above and below current price."""
    obs    = order_blocks(df)
    atr_v  = float(atr(df).iloc[-1])

    # Bullish OB (below price — potential support)
    bull_top = obs["bull_ob_top"].replace(np.nan, 0)
    bull_bot = obs["bull_ob_bot"].replace(np.nan, 0)
    valid_bull = bull_top[(bull_top > 0) & (bull_top < current_price)]

    # Bearish OB (above price — potential resistance)
    bear_top = obs["bear_ob_top"].replace(np.nan, 0)
    bear_bot = obs["bear_ob_bot"].replace(np.nan, 0)
    valid_bear = bear_bot[(bear_bot > 0) & (bear_bot > current_price)]

    result = {}
    if len(valid_bull) > 0:
        idx = valid_bull.idxmax()
        result["nearest_bull_ob"] = {
            "top": float(bull_top[idx]),
            "bot": float(bull_bot[idx]),
            "dist_atr": round((current_price - bull_top[idx]) / max(atr_v, 1e-9), 2),
        }
    if len(valid_bear) > 0:
        idx = valid_bear.idxmin()
        result["nearest_bear_ob"] = {
            "top": float(bear_top[idx]),
            "bot": float(bear_bot[idx]),
            "dist_atr": round((bear_bot[idx] - current_price) / max(atr_v, 1e-9), 2),
        }
    return result


# ─── MASTER MTF ANALYSIS ───────────────────────────────────────────────────────

class MTFAnalyzer:
    """
    Top-down multi-timeframe analysis engine.
    """

    def __init__(self):
        self.results: dict = {}

    def analyze(
        self,
        data: dict[str, pd.DataFrame],
        current_price: Optional[float] = None,
    ) -> dict:
        """
        Parameters
        ----------
        data : dict keyed by TF label {"1D": df, "4H": df, "1H": df, ...}
        current_price : latest price (defaults to last close of 1H/15M)
        """
        if not data:
            return {}

        # Determine current price
        if current_price is None:
            for tf in ["15M", "1H", "4H", "1D"]:
                if tf in data and len(data[tf]) > 0:
                    current_price = float(data[tf]["close"].iloc[-1])
                    break

        log.info("MTF Analysis | current_price=%.3f", current_price or 0)

        # ── Per-TF Bias ───────────────────────────────────────────────────────
        tf_biases = {}
        tf_signals = {}
        for tf_key, label in [("1D", "Daily"), ("4H", "4-Hour"), ("1H", "1-Hour")]:
            if tf_key in data:
                bias = _tf_bias(data[tf_key], label)
                tf_biases[tf_key]  = bias
                tf_signals[tf_key] = bias["bias"]
                log.info("  %s bias=%+d conf=%.2f | %s",
                         label, bias["bias"], bias["confidence"],
                         ", ".join(bias["reasons"][:3]))

        # ── Dominant Direction ────────────────────────────────────────────────
        signals   = [v for v in tf_signals.values() if v != 0]
        if signals:
            dominant  = max(set(signals), key=signals.count)
            mtf_score = np.mean(list(tf_signals.values()))
        else:
            dominant  = 0
            mtf_score = 0.0

        # ── Key Levels ────────────────────────────────────────────────────────
        key_levels = {}
        ob_levels  = {}
        if "1D" in data and current_price:
            key_levels = extract_key_levels(
                data["1D"],
                data.get("4H", data["1D"]),
                current_price,
            )
        if "4H" in data and current_price:
            ob_levels = nearest_order_blocks(data["4H"], current_price)

        # ── ATR Summary ───────────────────────────────────────────────────────
        atr_summary = {}
        for tf_key in ["1D", "4H", "1H"]:
            if tf_key in data and len(data[tf_key]) >= 15:
                a = float(atr(data[tf_key]).iloc[-1])
                atr_summary[tf_key] = round(a, 3)

        self.results = {
            "current_price": current_price,
            "tf_biases":     tf_biases,
            "tf_signals":    tf_signals,
            "dominant_dir":  dominant,
            "mtf_score":     round(mtf_score, 3),
            "key_levels":    key_levels,
            "ob_levels":     ob_levels,
            "atr_summary":   atr_summary,
        }

        log.info("MTF dominant=%+d score=%.3f ATR(1D)=%.2f",
                 dominant, mtf_score, atr_summary.get("1D", 0))
        return self.results

    def get_tf_signals(self) -> dict[str, int]:
        return self.results.get("tf_signals", {})

    def dominant_direction(self) -> int:
        return self.results.get("dominant_dir", 0)

    def print_report(self):
        r = self.results
        if not r:
            print("No MTF analysis results.")
            return
        sep = "─" * 62
        print(f"\n{sep}")
        print(f"  XAUUSD MTF ANALYSIS  │  Price: {r.get('current_price', 0):.3f}")
        print(sep)
        for tf_key, bias in r.get("tf_biases", {}).items():
            arrow = "▲" if bias["bias"] == 1 else ("▼" if bias["bias"] == -1 else "─")
            print(f"  [{bias['label']:10s}]  {arrow}  bias={bias['bias']:+d}  "
                  f"conf={bias['confidence']:.2f}  "
                  f"RSI={bias.get('rsi', 0):.1f}  ADX={bias.get('adx', 0):.1f}")
            for r_ in bias["reasons"][:3]:
                print(f"              ↳ {r_}")
        print(sep)
        dom = r.get("dominant_dir", 0)
        dom_str = "BULLISH ▲" if dom == 1 else ("BEARISH ▼" if dom == -1 else "NEUTRAL ─")
        print(f"  MTF Dominant Bias: {dom_str}  (score={r.get('mtf_score', 0):+.3f})")
        print(sep)
        print("  KEY LEVELS:")
        for name, info in r.get("key_levels", {}).items():
            print(f"    {name:8s}: {info['level']:.3f}  [{info['dist_atr']:.1f}×ATR {info['side']}]")
        print(sep)
        for name, ob in r.get("ob_levels", {}).items():
            tag = "BULL OB" if "bull" in name else "BEAR OB"
            print(f"  {tag}: {ob['bot']:.3f}–{ob['top']:.3f}  [{ob['dist_atr']:.1f}×ATR away]")
        print(f"{sep}\n")
