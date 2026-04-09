"""
Hybrid ML Ensemble – XGBoost + LSTM + ARCH/GARCH
Combines model outputs into a single directional score ∈ [0, 1] per bar,
along with a volatility-adjusted confidence band and regime flags.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class EnsembleSignal:
    """Structured output for a single bar's ensemble prediction."""
    timestamp:       object
    direction:       str        # "long" | "short" | "neutral"
    raw_score:       float      # weighted model consensus (0–1, direction-agnostic)
    direction_score: float      # signed score (-1 bear … +1 bull)
    confidence:      float      # final confidence after vol adjustment (0–1)
    xgb_prob_long:   float
    xgb_prob_short:  float
    lstm_prob_long:  float
    lstm_prob_short: float
    vol_regime:      str        # "low" | "medium" | "high"
    vol_regime_score: float     # multiplier 0.55–1.00
    mtf_confluence:  float      # -1 to +1
    mtf_aligned:     bool
    is_setup:        bool       # confidence > threshold AND mtf_aligned


class HybridEnsemble:
    """
    Weighted combination:
      score = w_xgb * P_xgb(direction)
            + w_lstm * P_lstm(direction)
            + w_garch * vol_regime_score (applied as multiplier)

    Setup criterion:
      1. confidence > SETUP_SCORE_THRESHOLD
      2. MTF alignment (if required by config)
    """

    def __init__(
        self,
        xgb_model,
        lstm_model,
        garch_model,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.xgb   = xgb_model
        self.lstm  = lstm_model
        self.garch = garch_model
        self.w     = weights or config.ENSEMBLE_WEIGHTS

    # ── Public API ────────────────────────────────────────────────────────────

    def score_latest(
        self,
        primary_df: pd.DataFrame,
        vol_forecast: Optional[Dict] = None,
    ) -> EnsembleSignal:
        """
        Generate an ensemble signal for the MOST RECENT bar in *primary_df*.
        """
        signals = self.score_all(primary_df, vol_forecast)
        return signals[-1]

    def score_all(
        self,
        primary_df: pd.DataFrame,
        vol_forecast: Optional[Dict] = None,
    ) -> List[EnsembleSignal]:
        """
        Score every bar in *primary_df*.  Returns a list of EnsembleSignal objects,
        padded with neutral signals at the head (LSTM warm-up window).
        """
        n        = len(primary_df)
        seq_len  = config.LSTM_SEQUENCE_LENGTH

        # ── XGBoost probabilities (N, 3) ─────────────────────────────────────
        try:
            xgb_proba = self.xgb.predict_proba(primary_df)   # [neutral, long, short]
        except Exception as exc:
            logger.warning("XGBoost predict failed: %s; using uniform.", exc)
            xgb_proba = np.full((n, 3), 1 / 3)

        # ── LSTM probabilities (M, 3) where M = N - seq_len ──────────────────
        try:
            lstm_proba_raw = self.lstm.predict_proba(primary_df)
            # Pad the warm-up head with uniform predictions
            pad            = np.full((n - len(lstm_proba_raw), 3), 1 / 3)
            lstm_proba     = np.vstack([pad, lstm_proba_raw])
        except Exception as exc:
            logger.warning("LSTM predict failed: %s; using uniform.", exc)
            lstm_proba = np.full((n, 3), 1 / 3)

        # ── GARCH vol info ────────────────────────────────────────────────────
        if vol_forecast is None:
            try:
                vol_forecast = self.garch.forecast()
            except Exception as exc:
                logger.warning("GARCH forecast failed: %s", exc)
                vol_forecast = {
                    "regime": "medium",
                    "regime_score": config.VOL_REGIME_MULTIPLIER["medium"],
                }

        regime       = vol_forecast.get("regime", "medium")
        regime_score = vol_forecast.get("regime_score", 0.80)

        # ── MTF confluence ────────────────────────────────────────────────────
        mtf_score   = primary_df.get("confluence_score",
                                     pd.Series(0.0, index=primary_df.index))
        mtf_aligned_bull = primary_df.get("mtf_aligned_bull",
                                          pd.Series(0, index=primary_df.index))
        mtf_aligned_bear = primary_df.get("mtf_aligned_bear",
                                          pd.Series(0, index=primary_df.index))

        signals = []
        for i in range(n):
            ts = primary_df.index[i]

            xp_neutral, xp_long, xp_short = xgb_proba[i]
            lp_neutral, lp_long, lp_short = lstm_proba[i]
            mtf_c = float(mtf_score.iloc[i])

            # Ensemble probabilities per direction
            bull_score = (
                self.w["xgboost"] * xp_long
                + self.w["lstm"]   * lp_long
            )
            bear_score = (
                self.w["xgboost"] * xp_short
                + self.w["lstm"]   * lp_short
            )

            # MTF bonus (0–0.1 depending on alignment strength)
            mtf_bonus = 0.05 * np.clip(mtf_c, -1, 1)
            if mtf_c > 0:
                bull_score = np.clip(bull_score + mtf_bonus, 0, 1)
            elif mtf_c < 0:
                bear_score = np.clip(bear_score - mtf_bonus, 0, 1)

            # Apply GARCH volatility multiplier
            bull_conf = bull_score * regime_score
            bear_conf = bear_score * regime_score

            if bull_conf > bear_conf:
                direction  = "long"
                confidence = bull_conf
            elif bear_conf > bull_conf:
                direction  = "short"
                confidence = bear_conf
            else:
                direction  = "neutral"
                confidence = 0.0

            # Signed direction score (useful for charting)
            dir_score = bull_conf - bear_conf

            # MTF alignment check
            aligned = bool(
                (direction == "long"  and bool(mtf_aligned_bull.iloc[i])) or
                (direction == "short" and bool(mtf_aligned_bear.iloc[i])) or
                (not config.MTF_ALIGNMENT_REQUIRED)
            )

            # Setup flag
            is_setup = (
                confidence >= config.SETUP_SCORE_THRESHOLD
                and (aligned or not config.MTF_ALIGNMENT_REQUIRED)
                and direction != "neutral"
            )

            signals.append(
                EnsembleSignal(
                    timestamp        = ts,
                    direction        = direction,
                    raw_score        = float((bull_score + bear_score) / 2),
                    direction_score  = float(dir_score),
                    confidence       = float(confidence),
                    xgb_prob_long    = float(xp_long),
                    xgb_prob_short   = float(xp_short),
                    lstm_prob_long   = float(lp_long),
                    lstm_prob_short  = float(lp_short),
                    vol_regime       = regime,
                    vol_regime_score = regime_score,
                    mtf_confluence   = mtf_c,
                    mtf_aligned      = aligned,
                    is_setup         = is_setup,
                )
            )

        n_setups = sum(1 for s in signals if s.is_setup)
        logger.info(
            "Ensemble scored %d bars → %d potential setups", n, n_setups
        )
        return signals

    def to_dataframe(self, signals: List[EnsembleSignal]) -> pd.DataFrame:
        """Convert list of EnsembleSignal to a tidy DataFrame."""
        records = [
            {
                "timestamp":        s.timestamp,
                "direction":        s.direction,
                "raw_score":        s.raw_score,
                "direction_score":  s.direction_score,
                "confidence":       s.confidence,
                "xgb_prob_long":    s.xgb_prob_long,
                "xgb_prob_short":   s.xgb_prob_short,
                "lstm_prob_long":   s.lstm_prob_long,
                "lstm_prob_short":  s.lstm_prob_short,
                "vol_regime":       s.vol_regime,
                "vol_regime_score": s.vol_regime_score,
                "mtf_confluence":   s.mtf_confluence,
                "mtf_aligned":      s.mtf_aligned,
                "is_setup":         s.is_setup,
            }
            for s in signals
        ]
        df = pd.DataFrame(records).set_index("timestamp")
        return df
