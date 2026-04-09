"""
Comprehensive Technical Indicator Feature Engineering
Produces ~70 features per bar used by both XGBoost and LSTM.
Depends on the `ta` library (pip install ta).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("'ta' library not found – install it: pip install ta")

import config

logger = logging.getLogger(__name__)


class TechnicalFeatureEngineer:
    """Compute all technical indicator features for a single OHLCV DataFrame."""

    def __init__(self) -> None:
        if not TA_AVAILABLE:
            raise ImportError("Install 'ta': pip install ta")

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* augmented with all technical features.

        Input columns required: Open, High, Low, Close, Volume
        """
        out = df.copy()
        c = out["Close"]
        h = out["High"]
        lo = out["Low"]
        v  = out.get("Volume", pd.Series(0, index=out.index))

        out = self._moving_averages(out, c)
        out = self._momentum(out, c, h, lo, v)
        out = self._volatility(out, c, h, lo)
        out = self._trend(out, c, h, lo)
        out = self._volume_indicators(out, c, h, lo, v)
        out = self._price_position(out, c, h, lo)

        return out

    # ── Private indicator groups ──────────────────────────────────────────────

    @staticmethod
    def _moving_averages(df: pd.DataFrame, c: pd.Series) -> pd.DataFrame:
        for p in config.EMA_PERIODS:
            col = f"ema_{p}"
            df[col]       = ta.trend.ema_indicator(c, window=p)
            df[f"dist_{col}"] = (c - df[col]) / df[col]   # relative distance

        for p in config.SMA_PERIODS:
            col = f"sma_{p}"
            df[col]       = ta.trend.sma_indicator(c, window=p)
            df[f"dist_{col}"] = (c - df[col]) / df[col]

        # Golden/death cross signals
        df["ema_cross_9_20"]   = (df["ema_9"]  > df["ema_20"]).astype(int)
        df["ema_cross_20_50"]  = (df["ema_20"] > df["ema_50"]).astype(int)
        df["ema_cross_50_200"] = (df["ema_50"] > df["ema_200"]).astype(int)
        df["ema_trend_score"]  = (
            df["ema_cross_9_20"] + df["ema_cross_20_50"] + df["ema_cross_50_200"]
        ) / 3  # 0 = full bear, 1 = full bull

        return df

    @staticmethod
    def _momentum(
        df: pd.DataFrame,
        c: pd.Series,
        h: pd.Series,
        lo: pd.Series,
        v: pd.Series,
    ) -> pd.DataFrame:
        # RSI
        df["rsi_14"] = ta.momentum.rsi(c, window=14)
        df["rsi_7"]  = ta.momentum.rsi(c, window=7)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(h, lo, c, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["stoch_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(int)

        # MACD
        macd_ind = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        df["macd"]        = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"]   = macd_ind.macd_diff()
        df["macd_cross"]  = (df["macd"] > df["macd_signal"]).astype(int)
        # Normalise histogram by ATR (computed later; placeholder for now)
        df["macd_hist_norm"] = df["macd_hist"] / (c.rolling(14).std() + 1e-9)

        # Williams %R
        df["williams_r"] = ta.momentum.williams_r(h, lo, c, lbp=14)

        # CCI
        df["cci_20"] = ta.trend.cci(h, lo, c, window=20)

        # Rate of Change
        df["roc_10"] = ta.momentum.roc(c, window=10)
        df["roc_20"] = ta.momentum.roc(c, window=20)

        # Momentum (n-period)
        df["mom_10"] = c - c.shift(10)

        return df

    @staticmethod
    def _volatility(
        df: pd.DataFrame, c: pd.Series, h: pd.Series, lo: pd.Series
    ) -> pd.DataFrame:
        # ATR
        atr = ta.volatility.average_true_range(h, lo, c, window=14)
        df["atr_14"]     = atr
        df["atr_14_pct"] = atr / c    # ATR as % of price

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        df["bb_upper"]    = bb.bollinger_hband()
        df["bb_lower"]    = bb.bollinger_lband()
        df["bb_mid"]      = bb.bollinger_mavg()
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_position"] = (c - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-9
        )  # 0=at lower band, 1=at upper band

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(h, lo, c, window=20)
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_lower"] = kc.keltner_channel_lband()
        df["kc_pos"]   = (c - df["kc_lower"]) / (
            df["kc_upper"] - df["kc_lower"] + 1e-9
        )

        # Squeeze (BB inside KC → low-vol compression)
        df["bb_squeeze"] = (
            (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])
        ).astype(int)

        # Ulcer index (drawdown risk)
        rolling_max = c.rolling(14).max()
        df["ulcer_14"] = np.sqrt(
            ((c - rolling_max) / (rolling_max + 1e-9) * 100).pow(2).rolling(14).mean()
        )

        return df

    @staticmethod
    def _trend(
        df: pd.DataFrame, c: pd.Series, h: pd.Series, lo: pd.Series
    ) -> pd.DataFrame:
        # ADX family
        adx_ind = ta.trend.ADXIndicator(h, lo, c, window=14)
        df["adx_14"]   = adx_ind.adx()
        df["plus_di"]  = adx_ind.adx_pos()
        df["minus_di"] = adx_ind.adx_neg()
        df["di_diff"]  = df["plus_di"] - df["minus_di"]

        # Aroon
        aroon = ta.trend.AroonIndicator(h, lo, window=25)
        df["aroon_up"]  = aroon.aroon_up()
        df["aroon_down"] = aroon.aroon_down()
        df["aroon_osc"] = aroon.aroon_indicator()

        # Parabolic SAR
        psar = ta.trend.PSARIndicator(h, lo, c)
        df["psar"]       = psar.psar()
        df["psar_bull"]  = psar.psar_up_indicator()
        df["psar_bear"]  = psar.psar_down_indicator()
        df["psar_signal"] = df["psar_bull"].fillna(0) - df["psar_bear"].fillna(0)

        # Ichimoku (simplified)
        ichi = ta.trend.IchimokuIndicator(h, lo, window1=9, window2=26, window3=52)
        df["ichi_conv"]  = ichi.ichimoku_conversion_line()
        df["ichi_base"]  = ichi.ichimoku_base_line()
        df["ichi_cross"] = (df["ichi_conv"] > df["ichi_base"]).astype(int)

        return df

    @staticmethod
    def _volume_indicators(
        df: pd.DataFrame,
        c: pd.Series,
        h: pd.Series,
        lo: pd.Series,
        v: pd.Series,
    ) -> pd.DataFrame:
        if v.sum() == 0:
            # No volume data available (e.g. spot FX feeds)
            for col in ["obv_norm", "obv_slope", "cmf_20", "mfi_14", "vwap_dist",
                        "vol_ratio", "vol_trend"]:
                df[col] = 0.0
            return df

        # OBV
        obv = ta.volume.on_balance_volume(c, v)
        obv_sma = obv.rolling(20).mean()
        df["obv_norm"]  = (obv - obv_sma) / (obv.rolling(20).std() + 1e-9)
        df["obv_slope"] = obv.diff(5) / (obv.abs().rolling(5).mean() + 1e-9)

        # Chaikin Money Flow
        df["cmf_20"] = ta.volume.chaikin_money_flow(h, lo, c, v, window=20)

        # Money Flow Index
        df["mfi_14"] = ta.volume.money_flow_index(h, lo, c, v, window=14)

        # VWAP (rolling daily proxy)
        df["vwap"]      = (v * (h + lo + c) / 3).rolling(24).sum() / v.rolling(24).sum()
        df["vwap_dist"] = (c - df["vwap"]) / (df["vwap"] + 1e-9)

        # Volume relative to 20-period average
        v_sma = v.rolling(20).mean()
        df["vol_ratio"] = v / (v_sma + 1e-9)
        df["vol_trend"] = (v > v_sma).astype(int)

        return df

    @staticmethod
    def _price_position(
        df: pd.DataFrame, c: pd.Series, h: pd.Series, lo: pd.Series
    ) -> pd.DataFrame:
        # Pivot points (standard daily pivot using rolling 20-bar high/low/close)
        n = 20
        h_roll = h.rolling(n).max()
        l_roll = lo.rolling(n).min()
        c_roll = c.shift(1)   # previous close as proxy
        pivot  = (h_roll + l_roll + c_roll) / 3
        r1 = 2 * pivot - l_roll
        r2 = pivot + (h_roll - l_roll)
        s1 = 2 * pivot - h_roll
        s2 = pivot - (h_roll - l_roll)

        atr = df.get("atr_14", h - lo)
        df["dist_pivot_r1"] = (r1 - c) / (atr + 1e-9)
        df["dist_pivot_r2"] = (r2 - c) / (atr + 1e-9)
        df["dist_pivot_s1"] = (c - s1) / (atr + 1e-9)
        df["dist_pivot_s2"] = (c - s2) / (atr + 1e-9)

        # 52-week high/low proximity (using available data)
        hi252 = h.rolling(min(252, len(h))).max()
        lo252 = lo.rolling(min(252, len(lo))).min()
        rng252 = (hi252 - lo252 + 1e-9)
        df["price_pos_252"] = (c - lo252) / rng252   # 0=52wk low, 1=52wk high

        return df
