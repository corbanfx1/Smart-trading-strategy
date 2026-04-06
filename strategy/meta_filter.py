"""
Rule-Based Meta-Model Filter.
Aggregates XGBoost, LSTM, and GARCH signals with institutional SMC rules
to produce only high-probability XAUUSD trade setups.

Filter Stack:
  Layer 1 — Model Agreement   : XGB + LSTM must agree direction
  Layer 2 — Ensemble Threshold: Weighted probability ≥ threshold
  Layer 3 — GARCH Vol Filter  : Block trades in high/extreme vol regimes
  Layer 4 — CVD Confirmation  : Volume delta must confirm direction
  Layer 5 — SMC Structural    : BOS/CHoCH / OB / FVG / liquidity context
  Layer 6 — MTF Alignment     : At least 2/3 TFs must agree
  Layer 7 — Session Filter    : Active London/NY session
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from config.settings import META_FILTER as MF, SIGNAL
from utils.logger import get_logger

log = get_logger("MetaFilter")


# ─── INDIVIDUAL FILTER CHECKS ──────────────────────────────────────────────────

def check_model_agreement(xgb: dict, lstm: dict) -> tuple[bool, int]:
    """Both models must predict the same direction above minimum probability."""
    xgb_sig  = xgb.get("xgb_signal", 0)
    lstm_sig = lstm.get("lstm_signal", 0)

    if xgb_sig == 0 or lstm_sig == 0:
        return False, 0
    if xgb_sig != lstm_sig:
        return False, 0

    xgb_prob  = xgb.get("xgb_prob_long", 0.5)
    lstm_prob = lstm.get("lstm_prob_long", 0.5)

    if xgb_sig == 1:   # long setup
        ok = (xgb_prob  >= MF["xgb_min_prob"] and
              lstm_prob >= MF["lstm_min_prob"])
    else:              # short setup
        ok = (xgb_prob  <= 1 - MF["xgb_min_prob"] and
              lstm_prob <= 1 - MF["lstm_min_prob"])

    return ok, xgb_sig if ok else 0


def compute_ensemble_prob(xgb: dict, lstm: dict) -> tuple[float, float]:
    """
    Weighted ensemble probability.
    XGBoost weight: 0.55 (more interpretable, tree-based structure)
    LSTM weight:    0.45 (captures sequential patterns)
    """
    w_xgb  = 0.55
    w_lstm = 0.45

    p_long = (w_xgb  * xgb.get("xgb_prob_long",  0.5) +
              w_lstm * lstm.get("lstm_prob_long", 0.5))
    p_short = 1.0 - p_long
    return float(p_long), float(p_short)


def check_garch_regime(garch: dict) -> bool:
    """Block trades in extreme volatility regimes."""
    regime = garch.get("garch_regime", "normal")
    if regime == "high":
        log.debug("MetaFilter: GARCH regime=high — blocked")
        return False
    return True


def check_cvd_confirmation(features_row: pd.Series, direction: int) -> bool:
    """CVD must confirm direction."""
    if not MF["require_cvd_confirm"]:
        return True
    cvd_z = features_row.get("cvd_z", 0)
    if direction == 1 and cvd_z < -0.5:
        log.debug("MetaFilter: CVD bearish (%.2f) — long blocked", cvd_z)
        return False
    if direction == -1 and cvd_z > 0.5:
        log.debug("MetaFilter: CVD bullish (%.2f) — short blocked", cvd_z)
        return False
    return True


def check_smc_context(features_row: pd.Series, direction: int) -> tuple[bool, list[str]]:
    """
    SMC structural confirmation.
    For LONG: require discount zone + bull OB/FVG + sweep of lows OR CHoCH up
    For SHORT: require premium zone + bear OB/FVG + sweep of highs OR CHoCH down
    Returns (pass_flag, list_of_confluences).
    """
    confluences: list[str] = []
    score = 0

    if direction == 1:  # LONG
        if features_row.get("in_discount", 0):
            confluences.append("discount_zone")
            score += 1
        if features_row.get("in_bull_ob", 0):
            confluences.append("bull_order_block")
            score += 2
        if features_row.get("bull_fvg", 0) or features_row.get("in_bull_fvg", 0):
            confluences.append("bull_fvg")
            score += 1
        if features_row.get("sweep_lows", 0):
            confluences.append("liquidity_sweep_low")
            score += 2
        if features_row.get("choch_bullish", 0):
            confluences.append("choch_bullish")
            score += 2
        if features_row.get("bos_bullish", 0):
            confluences.append("bos_bullish")
            score += 1
        required = 3

    else:  # SHORT
        if features_row.get("in_premium", 0):
            confluences.append("premium_zone")
            score += 1
        if features_row.get("in_bear_ob", 0):
            confluences.append("bear_order_block")
            score += 2
        if features_row.get("bear_fvg", 0) or features_row.get("in_bear_fvg", 0):
            confluences.append("bear_fvg")
            score += 1
        if features_row.get("sweep_highs", 0):
            confluences.append("liquidity_sweep_high")
            score += 2
        if features_row.get("choch_bearish", 0):
            confluences.append("choch_bearish")
            score += 2
        if features_row.get("bos_bearish", 0):
            confluences.append("bos_bearish")
            score += 1
        required = 3

    ok = score >= required
    if not ok:
        log.debug("MetaFilter: SMC score=%d < %d — blocked. conf=%s",
                  score, required, confluences)
    return ok, confluences


def check_session_filter(ts: Optional[pd.Timestamp] = None) -> bool:
    """Only trade during active London or New York sessions (UTC)."""
    if not MF["require_active_session"]:
        return True
    if ts is None:
        ts = pd.Timestamp.utcnow()
    hour = ts.hour
    sessions = MF["active_sessions"]
    in_london = sessions["london"][0] <= hour < sessions["london"][1]
    in_ny     = sessions["new_york"][0] <= hour < sessions["new_york"][1]
    if not (in_london or in_ny):
        log.debug("MetaFilter: outside active session (UTC %02d:xx) — blocked", hour)
        return False
    return True


def check_mtf_alignment(tf_signals: dict[str, int],
                         direction:  int,
                         min_agree: int = MF["min_tf_alignment"]) -> tuple[bool, int]:
    """
    At least *min_agree* timeframe signals must match *direction*.
    The dominant TF bias must agree with the proposed trade direction,
    not merely agree amongst themselves.
    """
    if not tf_signals:
        return True, direction
    vals = [v for v in tf_signals.values() if v != 0]
    if not vals:
        log.debug("MetaFilter: MTF all neutral — blocked")
        return False, 0
    # Count how many TFs explicitly agree with the trade direction
    agree_count = sum(1 for v in vals if v == direction)
    ok = agree_count >= min_agree
    if not ok:
        log.debug(
            "MetaFilter: MTF direction=%+d agreement=%d/%d — blocked (signals=%s)",
            direction, agree_count, min_agree, tf_signals,
        )
    return ok, direction


# ─── MASTER META-FILTER ────────────────────────────────────────────────────────

class MetaModelFilter:
    """
    Combines all filter layers to produce a final high-probability signal.
    """

    def __init__(self):
        self.filter_stats = {
            "total_evaluated":   0,
            "model_agree_fail":  0,
            "ensemble_fail":     0,
            "garch_fail":        0,
            "cvd_fail":          0,
            "smc_fail":          0,
            "session_fail":      0,
            "mtf_fail":          0,
            "passed":            0,
        }

    def evaluate(
        self,
        xgb_pred:    dict,
        lstm_pred:   dict,
        garch_fc:    dict,
        features_row: pd.Series,
        tf_signals:  dict[str, int] = None,
        timestamp:   pd.Timestamp   = None,
    ) -> dict:
        """
        Run all filter layers.
        Returns a comprehensive signal dict.
        """
        self.filter_stats["total_evaluated"] += 1
        result = {
            "signal":          SIGNAL["NEUTRAL"],
            "direction":       0,
            "confidence":      0.0,
            "confluences":     [],
            "filters_passed":  [],
            "filters_failed":  [],
            "ensemble_prob":   0.5,
            "garch_regime":    garch_fc.get("garch_regime", "unknown"),
            "vol_scalar":      garch_fc.get("garch_vol_scalar", 1.0),
        }

        # ── Layer 1: Model Agreement ──────────────────────────────────────────
        agree, direction = check_model_agreement(xgb_pred, lstm_pred)
        if not agree:
            self.filter_stats["model_agree_fail"] += 1
            result["filters_failed"].append("model_agreement")
            return result
        result["filters_passed"].append("model_agreement")

        # ── Layer 2: Ensemble Probability ─────────────────────────────────────
        p_long, p_short = compute_ensemble_prob(xgb_pred, lstm_pred)
        result["ensemble_prob"] = p_long
        ens_prob = p_long if direction == 1 else p_short
        if ens_prob < MF["ensemble_threshold"]:
            self.filter_stats["ensemble_fail"] += 1
            result["filters_failed"].append("ensemble_threshold")
            return result
        result["filters_passed"].append("ensemble_threshold")

        # ── Layer 3: GARCH Volatility ──────────────────────────────────────────
        if not check_garch_regime(garch_fc):
            self.filter_stats["garch_fail"] += 1
            result["filters_failed"].append("garch_regime")
            return result
        result["filters_passed"].append("garch_regime")

        # ── Layer 4: CVD Confirmation ──────────────────────────────────────────
        if not check_cvd_confirmation(features_row, direction):
            self.filter_stats["cvd_fail"] += 1
            result["filters_failed"].append("cvd_confirmation")
            return result
        result["filters_passed"].append("cvd_confirmation")

        # ── Layer 5: SMC Context ───────────────────────────────────────────────
        smc_ok, confluences = check_smc_context(features_row, direction)
        if not smc_ok:
            self.filter_stats["smc_fail"] += 1
            result["filters_failed"].append("smc_context")
            return result
        result["confluences"]    = confluences
        result["filters_passed"].append("smc_context")

        # ── Layer 6: MTF Alignment ─────────────────────────────────────────────
        if tf_signals:
            mtf_ok, mtf_dir = check_mtf_alignment(tf_signals, direction)
            if not mtf_ok:
                self.filter_stats["mtf_fail"] += 1
                result["filters_failed"].append("mtf_alignment")
                return result
        result["filters_passed"].append("mtf_alignment")

        # ── Layer 7: Session Filter ────────────────────────────────────────────
        if not check_session_filter(timestamp):
            self.filter_stats["session_fail"] += 1
            result["filters_failed"].append("session_filter")
            return result
        result["filters_passed"].append("session_filter")

        # ── All Layers Passed ──────────────────────────────────────────────────
        self.filter_stats["passed"] += 1
        n_conf   = len(confluences)
        conf_score = min(1.0, ens_prob * (1 + 0.05 * n_conf))

        result.update({
            "direction":   direction,
            "signal":      SIGNAL["STRONG_BUY"] if (direction == 1 and ens_prob > 0.72)
                           else SIGNAL["BUY"]   if direction == 1
                           else SIGNAL["STRONG_SELL"] if ens_prob > 0.72
                           else SIGNAL["SELL"],
            "confidence":  conf_score,
            "n_confluences": n_conf,
        })

        log.info("MetaFilter PASS | dir=%+d | prob=%.3f | conf=%.3f | SMC=%s",
                 direction, ens_prob, conf_score, confluences)
        return result

    def get_stats(self) -> dict:
        total = max(self.filter_stats["total_evaluated"], 1)
        return {
            **self.filter_stats,
            "pass_rate": self.filter_stats["passed"] / total,
        }
