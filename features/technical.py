"""
Technical Indicators Feature Engineering.
Produces a rich set of momentum, trend, volatility, and volume features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from config.settings import LOOKBACK_PERIODS as LP
from utils.logger import get_logger

log = get_logger("TechnicalFeatures")


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _rma(series: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (used in RSI/ATR)."""
    return series.ewm(alpha=1 / n, adjust=False).mean()


# ─── ATR ───────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, n: int = LP["atr"]) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, n).rename("atr")


# ─── RSI ───────────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, n: int = LP["rsi"]) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    rs    = _rma(gain, n) / (_rma(loss, n) + 1e-9)
    return (100 - 100 / (1 + rs)).rename("rsi")


# ─── MACD ──────────────────────────────────────────────────────────────────────

def macd(series: pd.Series) -> pd.DataFrame:
    fast   = _ema(series, LP["macd_fast"])
    slow   = _ema(series, LP["macd_slow"])
    line   = (fast - slow).rename("macd_line")
    signal = _ema(line, LP["macd_signal"]).rename("macd_signal")
    hist   = (line - signal).rename("macd_hist")
    return pd.concat([line, signal, hist], axis=1)


# ─── BOLLINGER BANDS ───────────────────────────────────────────────────────────

def bollinger_bands(series: pd.Series,
                    n: int = LP["bb_period"],
                    k: float = LP["bb_std"]) -> pd.DataFrame:
    mid   = _sma(series, n).rename("bb_mid")
    std   = series.rolling(n).std()
    upper = (mid + k * std).rename("bb_upper")
    lower = (mid - k * std).rename("bb_lower")
    width = ((upper - lower) / mid).rename("bb_width")
    pct_b = ((series - lower) / (upper - lower + 1e-9)).rename("bb_pct_b")
    return pd.concat([mid, upper, lower, width, pct_b], axis=1)


# ─── ADX ───────────────────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, n: int = LP["adx"]) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    up   = high.diff()
    down = -low.diff()
    dm_p = (up.where((up > down) & (up > 0), 0.0))
    dm_m = (down.where((down > up) & (down > 0), 0.0))
    tr   = pd.concat([high - low,
                      (high - close.shift(1)).abs(),
                      (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_ = _rma(tr,   n)
    di_p = (100 * _rma(dm_p, n) / (atr_ + 1e-9)).rename("di_plus")
    di_m = (100 * _rma(dm_m, n) / (atr_ + 1e-9)).rename("di_minus")
    dx   = (100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9))
    adx_ = _rma(dx, n).rename("adx")
    return pd.concat([adx_, di_p, di_m], axis=1)


# ─── STOCHASTIC ────────────────────────────────────────────────────────────────

def stochastic(df: pd.DataFrame,
               k: int = LP["stoch_k"],
               d: int = LP["stoch_d"]) -> pd.DataFrame:
    low_k  = df["low"].rolling(k).min()
    high_k = df["high"].rolling(k).max()
    pct_k  = (100 * (df["close"] - low_k) / (high_k - low_k + 1e-9)).rename("stoch_k")
    pct_d  = _sma(pct_k, d).rename("stoch_d")
    return pd.concat([pct_k, pct_d], axis=1)


# ─── CCI ───────────────────────────────────────────────────────────────────────

def cci(df: pd.DataFrame, n: int = LP["cci"]) -> pd.Series:
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    mean = tp.rolling(n).mean()
    mad  = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((tp - mean) / (0.015 * mad + 1e-9)).rename("cci")


# ─── WILLIAMS %R ───────────────────────────────────────────────────────────────

def williams_r(df: pd.DataFrame, n: int = LP["williams_r"]) -> pd.Series:
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return (-100 * (hh - df["close"]) / (hh - ll + 1e-9)).rename("williams_r")


# ─── MFI ───────────────────────────────────────────────────────────────────────

def mfi(df: pd.DataFrame, n: int = LP["mfi"]) -> pd.Series:
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    rmf  = tp * df["volume"]
    diff = tp.diff()
    pos  = rmf.where(diff > 0, 0.0).rolling(n).sum()
    neg  = rmf.where(diff < 0, 0.0).rolling(n).sum()
    return (100 - 100 / (1 + pos / (neg + 1e-9))).rename("mfi")


# ─── OBV ───────────────────────────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum().rename("obv")


# ─── VWAP (Rolling) ────────────────────────────────────────────────────────────

def rolling_vwap(df: pd.DataFrame, n: int = LP["vwap_period"]) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return ((tp * df["volume"]).rolling(n).sum() /
            df["volume"].rolling(n).sum()).rename("vwap")


# ─── CVD (Cumulative Volume Delta) ─────────────────────────────────────────────

def cumulative_volume_delta(df: pd.DataFrame, n: int = LP["cvd_period"]) -> pd.DataFrame:
    """
    Approximate CVD using the close-relative-to-range heuristic:
      buy_vol  ≈ vol × (close − low) / (high − low)
      sell_vol ≈ vol × (high − close) / (high − low)
    """
    hl    = (df["high"] - df["low"]).replace(0, np.nan)
    buy_  = df["volume"] * (df["close"] - df["low"]) / hl
    sell_ = df["volume"] * (df["high"] - df["close"]) / hl
    delta = (buy_ - sell_).fillna(0)
    cvd   = delta.rolling(n).sum().rename("cvd")
    cvd_z = ((cvd - cvd.rolling(n).mean()) /
              (cvd.rolling(n).std() + 1e-9)).rename("cvd_z")
    return pd.concat([cvd, cvd_z], axis=1)


# ─── EMAs ──────────────────────────────────────────────────────────────────────

def ema_features(series: pd.Series) -> pd.DataFrame:
    e9   = _ema(series, LP["ema_fast"]).rename("ema9")
    e21  = _ema(series, LP["ema_medium"]).rename("ema21")
    e50  = _ema(series, LP["ema_slow"]).rename("ema50")
    e200 = _ema(series, LP["ema_trend"]).rename("ema200")
    # Normalised distance from price
    cross_9_21  = (e9 - e21).rename("cross_9_21")
    cross_21_50 = (e21 - e50).rename("cross_21_50")
    price_vs_200 = ((series - e200) / (e200 + 1e-9)).rename("price_vs_200")
    return pd.concat([e9, e21, e50, e200, cross_9_21, cross_21_50, price_vs_200], axis=1)


# ─── VOLATILITY FEATURES ───────────────────────────────────────────────────────

def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close    = df["close"]
    log_ret  = np.log(close / close.shift(1))
    rv_5     = log_ret.rolling(5).std().rename("rv5")
    rv_20    = log_ret.rolling(20).std().rename("rv20")
    rv_ratio = (rv_5 / (rv_20 + 1e-9)).rename("rv_ratio")
    garman   = (0.5 * np.log(df["high"] / df["low"]) ** 2 -
                (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
                ).rolling(20).mean().rename("garman_klass")
    return pd.concat([rv_5, rv_20, rv_ratio, garman], axis=1)


# ─── CANDLE PATTERN FEATURES ───────────────────────────────────────────────────

def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body   = (c - o).rename("candle_body")
    upper  = (h - pd.concat([o, c], axis=1).max(axis=1)).rename("upper_wick")
    lower  = (pd.concat([o, c], axis=1).min(axis=1) - l).rename("lower_wick")
    atr_   = atr(df)
    body_n = (body / (atr_ + 1e-9)).rename("candle_body_norm")
    is_bull = (c > o).astype(int).rename("is_bullish")
    # Engulfing
    prev_body = body.shift(1)
    engulf = ((body.abs() > prev_body.abs()) &
              (body * prev_body < 0)).astype(int).rename("engulfing")
    # Doji
    doji = ((body.abs() < 0.1 * (h - l + 1e-9))).astype(int).rename("doji")
    # Hammer/shooting star
    hammer = ((lower > 2 * body.abs()) & (upper < 0.3 * body.abs())).astype(int).rename("hammer")
    return pd.concat([body, upper, lower, body_n, is_bull, engulf, doji, hammer], axis=1)


# ─── MASTER FEATURE BUILDER ────────────────────────────────────────────────────

def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all technical indicators into a single DataFrame."""
    log.debug("Building technical features for %d bars", len(df))

    atr_    = atr(df)
    rsi_    = rsi(df["close"])
    macd_   = macd(df["close"])
    bb_     = bollinger_bands(df["close"])
    adx_    = adx(df)
    stoch_  = stochastic(df)
    cci_    = cci(df)
    wr_     = williams_r(df)
    mfi_    = mfi(df)
    obv_    = obv(df)
    vwap_   = rolling_vwap(df)
    cvd_    = cumulative_volume_delta(df)
    emas_   = ema_features(df["close"])
    vols_   = volatility_features(df)
    cndl_   = candle_features(df)

    # Volume normalised
    vol_sma = df["volume"].rolling(20).mean().rename("vol_sma20")
    vol_rat = (df["volume"] / (vol_sma + 1e-9)).rename("vol_ratio")

    # Combine
    out = pd.concat([
        df[["open", "high", "low", "close", "volume"]],
        atr_, rsi_, macd_, bb_, adx_, stoch_,
        cci_, wr_, mfi_, obv_, vwap_, cvd_, emas_, vols_, cndl_,
        vol_sma, vol_rat,
    ], axis=1)

    out.dropna(inplace=True)
    log.debug("  ✓ %d features, %d bars after dropna", out.shape[1], len(out))
    return out
