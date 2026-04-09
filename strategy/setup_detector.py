"""
High-Probability Trade Setup Detector
Classifies each ensemble signal into a named setup type, calculates the
composite Setup Score (0–100), and computes entry / SL / TP levels.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ensemble.hybrid_ensemble import EnsembleSignal
import config

logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    # ── Identity ──────────────────────────────────────────────────────────────
    timestamp:      object
    direction:      str          # "long" | "short"
    setup_type:     str          # e.g. "Trend Continuation", "SR Bounce", …
    timeframe:      str          # Primary TF label

    # ── Scores ────────────────────────────────────────────────────────────────
    setup_score:    float        # 0–100 composite score
    confidence_pct: float        # 0–100 (= ensemble confidence × 100)
    xgb_pct:        float
    lstm_pct:       float
    vol_regime:     str

    # ── Price levels ──────────────────────────────────────────────────────────
    entry:  float
    stop:   float
    tp1:    float
    tp2:    float
    tp3:    float

    # ── R:R ───────────────────────────────────────────────────────────────────
    rr1:    float
    rr2:    float
    rr3:    float

    # ── Context ───────────────────────────────────────────────────────────────
    atr:            float
    near_support:   bool
    near_resistance: bool
    mtf_aligned:    bool
    mtf_confluence: float
    ms_trend:       str          # "bullish" | "bearish" | "neutral"
    notes:          List[str] = field(default_factory=list)

    @property
    def is_high_quality(self) -> bool:
        return self.setup_score >= 70 and self.rr1 >= config.MIN_RR_RATIO

    def summary(self) -> str:
        dir_arrow = "▲ LONG" if self.direction == "long" else "▼ SHORT"
        lines = [
            f"\n{'='*62}",
            f"  {config.ASSET_NAME}  |  {self.timeframe}  |  {dir_arrow}",
            f"  Setup: {self.setup_type}",
            f"  Score: {self.setup_score:.1f}/100  |  "
            f"Confidence: {self.confidence_pct:.1f}%  |  Regime: {self.vol_regime.upper()}",
            f"{'─'*62}",
            f"  Entry :  {self.entry:.2f}",
            f"  Stop  :  {self.stop:.2f}  "
            f"({abs(self.entry - self.stop):.2f} pts / "
            f"{abs(self.entry - self.stop)/self.entry*100:.2f}%)",
            f"  TP 1  :  {self.tp1:.2f}   R:R = 1:{self.rr1:.2f}",
            f"  TP 2  :  {self.tp2:.2f}   R:R = 1:{self.rr2:.2f}",
            f"  TP 3  :  {self.tp3:.2f}   R:R = 1:{self.rr3:.2f}",
            f"{'─'*62}",
            f"  XGBoost: {self.xgb_pct:.1f}%  "
            f"LSTM: {self.lstm_pct:.1f}%  "
            f"MTF: {'✓ Aligned' if self.mtf_aligned else '✗ Diverging'}",
            f"  Trend: {self.ms_trend.capitalize()}  |  ATR: {self.atr:.2f}",
        ]
        if self.notes:
            lines.append(f"  Notes: {'; '.join(self.notes)}")
        lines.append(f"{'='*62}\n")
        return "\n".join(lines)


class SetupDetector:
    """Convert ensemble signals + feature data into labelled TradeSetup objects."""

    SETUP_TYPES = {
        "trend_continuation": "Trend Continuation",
        "sr_bounce":          "S/R Bounce",
        "breakout":           "Breakout",
        "reversal":           "Reversal / CHoCH",
        "order_block":        "Order Block Reaction",
        "fvg_fill":           "Fair Value Gap Fill",
        "squeeze_release":    "Volatility Squeeze Release",
    }

    def __init__(self, timeframe_label: str = "1H") -> None:
        self.timeframe_label = timeframe_label

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        primary_df: pd.DataFrame,
        signals: List[EnsembleSignal],
    ) -> List[TradeSetup]:
        """Return a list of TradeSetup objects for every bar flagged as a setup."""
        setups = []
        for sig in signals:
            if not sig.is_setup:
                continue
            try:
                row = primary_df.loc[sig.timestamp]
            except KeyError:
                continue

            setup = self._build_setup(sig, row, primary_df)
            if setup is not None:
                setups.append(setup)

        logger.info(
            "Setup detector: %d setups found (score≥%.0f)",
            len(setups),
            config.SETUP_SCORE_THRESHOLD * 100,
        )
        return setups

    # ── Private builders ──────────────────────────────────────────────────────

    def _build_setup(
        self,
        sig: EnsembleSignal,
        row: pd.Series,
        df: pd.DataFrame,
    ) -> Optional[TradeSetup]:
        entry = float(row.get("Close", row.get("close", 0)))
        atr   = float(row.get("atr_14", entry * 0.005))

        if entry <= 0 or atr <= 0:
            return None

        # ── Entry / SL / TP ──────────────────────────────────────────────────
        if sig.direction == "long":
            stop = entry - config.ATR_STOP_MULTIPLIER * atr
            tp1  = entry + config.ATR_TP1_MULTIPLIER * atr
            tp2  = entry + config.ATR_TP2_MULTIPLIER * atr
            tp3  = entry + config.ATR_TP3_MULTIPLIER * atr
        else:
            stop = entry + config.ATR_STOP_MULTIPLIER * atr
            tp1  = entry - config.ATR_TP1_MULTIPLIER * atr
            tp2  = entry - config.ATR_TP2_MULTIPLIER * atr
            tp3  = entry - config.ATR_TP3_MULTIPLIER * atr

        risk  = abs(entry - stop)
        if risk == 0:
            return None

        rr1 = abs(tp1 - entry) / risk
        rr2 = abs(tp2 - entry) / risk
        rr3 = abs(tp3 - entry) / risk

        if rr1 < config.MIN_RR_RATIO:
            return None

        # ── Setup type classification ─────────────────────────────────────────
        setup_type = self._classify_type(sig, row)

        # ── Composite score ───────────────────────────────────────────────────
        score = self._compute_score(sig, row, rr1)

        # ── Context flags ─────────────────────────────────────────────────────
        ms_trend_code = int(row.get("ms_trend", 0))
        ms_trend_map  = {1: "bullish", 0: "neutral", -1: "bearish"}
        ms_trend      = ms_trend_map.get(ms_trend_code, "neutral")

        notes = self._build_notes(sig, row)

        return TradeSetup(
            timestamp       = sig.timestamp,
            direction       = sig.direction,
            setup_type      = setup_type,
            timeframe       = self.timeframe_label,
            setup_score     = round(score, 1),
            confidence_pct  = round(sig.confidence * 100, 1),
            xgb_pct         = round(
                (sig.xgb_prob_long if sig.direction == "long" else sig.xgb_prob_short) * 100, 1
            ),
            lstm_pct        = round(
                (sig.lstm_prob_long if sig.direction == "long" else sig.lstm_prob_short) * 100, 1
            ),
            vol_regime      = sig.vol_regime,
            entry           = round(entry, 2),
            stop            = round(stop, 2),
            tp1             = round(tp1, 2),
            tp2             = round(tp2, 2),
            tp3             = round(tp3, 2),
            rr1             = round(rr1, 2),
            rr2             = round(rr2, 2),
            rr3             = round(rr3, 2),
            atr             = round(atr, 2),
            near_support    = bool(row.get("near_support", False)),
            near_resistance = bool(row.get("near_resistance", False)),
            mtf_aligned     = sig.mtf_aligned,
            mtf_confluence  = sig.mtf_confluence,
            ms_trend        = ms_trend,
            notes           = notes,
        )

    def _classify_type(self, sig: EnsembleSignal, row: pd.Series) -> str:
        """Heuristic setup-type classifier."""
        bos   = bool(row.get("bos", 0))
        choch = bool(row.get("choch", 0))
        in_ob = bool(row.get("in_bullish_ob", 0) or row.get("in_bearish_ob", 0))
        in_fvg = bool(row.get("in_bullish_fvg", 0) or row.get("in_bearish_fvg", 0))
        near_sr = bool(row.get("near_support", 0) or row.get("near_resistance", 0))
        squeeze = bool(row.get("bb_squeeze", 0))
        ms_trend = int(row.get("ms_trend", 0))

        if choch:
            return self.SETUP_TYPES["reversal"]
        if in_ob:
            return self.SETUP_TYPES["order_block"]
        if in_fvg:
            return self.SETUP_TYPES["fvg_fill"]
        if squeeze:
            return self.SETUP_TYPES["squeeze_release"]
        if bos:
            return self.SETUP_TYPES["breakout"]
        if near_sr:
            return self.SETUP_TYPES["sr_bounce"]
        if ms_trend != 0:
            return self.SETUP_TYPES["trend_continuation"]
        return self.SETUP_TYPES["trend_continuation"]

    @staticmethod
    def _compute_score(
        sig: EnsembleSignal, row: pd.Series, rr1: float
    ) -> float:
        """
        Composite score out of 100:
          - Model confidence : 50 pts
          - MTF alignment    : 20 pts
          - R:R quality      : 15 pts
          - Vol regime       : 10 pts
          - Momentum confirm : 5 pts
        """
        model_pts = sig.confidence * 50

        mtf_pts   = 20 * (1.0 if sig.mtf_aligned else 0.4)

        rr_pts    = min(15, 15 * (rr1 - config.MIN_RR_RATIO) / (3.0 - config.MIN_RR_RATIO + 1e-9))
        rr_pts    = max(0, rr_pts)

        vol_pts   = 10 * sig.vol_regime_score

        # Momentum confirmation: RSI in right zone
        rsi = float(row.get("rsi_14", 50))
        if sig.direction == "long":
            mom_pts = 5 * np.clip((rsi - 40) / 30, 0, 1)
        else:
            mom_pts = 5 * np.clip((60 - rsi) / 30, 0, 1)

        return min(100, model_pts + mtf_pts + rr_pts + vol_pts + mom_pts)

    @staticmethod
    def _build_notes(sig: EnsembleSignal, row: pd.Series) -> List[str]:
        notes = []
        adx = float(row.get("adx_14", 0))
        if adx > 30:
            notes.append(f"Strong trend ADX={adx:.0f}")
        if sig.vol_regime == "high":
            notes.append("High vol regime – reduce size")
        if bool(row.get("choch", 0)):
            notes.append("CHoCH detected – potential reversal")
        if bool(row.get("bb_squeeze", 0)):
            notes.append("Bollinger squeeze – breakout imminent")
        rsi = float(row.get("rsi_14", 50))
        if sig.direction == "long" and rsi < 35:
            notes.append("RSI oversold – contrarian long opportunity")
        elif sig.direction == "short" and rsi > 65:
            notes.append("RSI overbought – contrarian short opportunity")
        return notes
