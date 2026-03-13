"""
Meta-Model  –  HARD-GATE + SOFT-SCORE
───────────────────────────────────────
Combines p_xgb (XGBoost-proxy) and p_lstm (LSTM-proxy) with rule-based
structure filters into a final institutional setup probability.

Architecture
────────────
  HARD-GATE layer:  Non-negotiable binary conditions.
                    ALL must pass → gate_open = True.
                    If gate_open = False → final_prob = 0.0, no trade.

  SOFT-SCORE layer: Weighted fusion of component scores.
                    soft_score = w_xgb*p_xgb + w_lstm*p_lstm
                                 + w_struct*p_struct + w_flow*p_flow
                                 + w_vp*p_vp

  CONVICTION tier:
                    soft_score >= 0.75 → CONVICTION_HIGH   (A+ setup)
                    soft_score >= 0.60 → CONVICTION_MEDIUM (B  setup)
                    soft_score >= 0.45 → CONVICTION_LOW    (C  setup, caution)
                    else               → NO_TRADE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from analysis.ml_scoring import FeatureVector, compute_p_xgb, compute_p_lstm
from analysis.market_structure import MarketStructureResult, StructureEvent
from analysis.order_blocks import OrderBlock, active_obs
from analysis.fair_value_gaps import FVG, unfilled_fvgs
from analysis.liquidity import LiquidityLevel, LiquiditySweep
from analysis.premium_discount import PDContext
from analysis.order_flow import OrderFlowReport
from analysis.volume_profile import VolumeProfileResult, VPOCMigration
from analysis.institutional_phase import PhaseSnapshot
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# HARD GATES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str      # human-readable explanation

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"  [{status}] {self.name}: {self.reason}"


def _gate_mtf_confluence(fv: FeatureVector, direction: str) -> GateResult:
    """4H and Daily must agree with the trade direction."""
    name = "MTF Confluence"
    if direction == "LONG":
        ok = fv.trend_4h >= 0 and fv.trend_daily >= 0 and (fv.trend_4h + fv.trend_daily) > 0
        reason = f"4H={'bull' if fv.trend_4h>0 else 'neut' if fv.trend_4h==0 else 'bear'}, " \
                 f"D={'bull' if fv.trend_daily>0 else 'neut' if fv.trend_daily==0 else 'bear'}"
    else:
        ok = fv.trend_4h <= 0 and fv.trend_daily <= 0 and (fv.trend_4h + fv.trend_daily) < 0
        reason = f"4H={'bear' if fv.trend_4h<0 else 'neut'}, D={'bear' if fv.trend_daily<0 else 'neut'}"
    return GateResult(name, ok, reason)


def _gate_structure_confirmation(
    fv: FeatureVector, direction: str
) -> GateResult:
    """CHoCH or BOS must have occurred recently on 1H."""
    name = "1H Structure"
    if direction == "LONG":
        ok = fv.choch_recency > 0.2 or fv.bos_recency > 0.4
        reason = f"CHoCH recency={fv.choch_recency:.2f}, BOS recency={fv.bos_recency:.2f}"
    else:
        ok = fv.choch_recency > 0.2 or fv.bos_recency > 0.4
        reason = f"CHoCH recency={fv.choch_recency:.2f}, BOS recency={fv.bos_recency:.2f}"
    return GateResult(name, ok, reason)


def _gate_pd_zone(fv: FeatureVector, direction: str) -> GateResult:
    """Price must be in the correct P/D zone for the trade direction."""
    name = "P/D Zone"
    if direction == "LONG":
        ok = fv.pd_encoded >= 0     # discount or equilibrium
        reason = f"pd_encoded={fv.pd_encoded:+.0f} ({'ok' if ok else 'PREMIUM – avoid longs'})"
    else:
        ok = fv.pd_encoded <= 0     # premium or equilibrium
        reason = f"pd_encoded={fv.pd_encoded:+.0f} ({'ok' if ok else 'DISCOUNT – avoid shorts'})"
    return GateResult(name, ok, reason)


def _gate_no_opposing_sweep(fv: FeatureVector, direction: str) -> GateResult:
    """
    No recent opposing sweep should have occurred.
    A recent BSL sweep while looking for longs = warning.
    """
    name = "No Opposing Sweep"
    if direction == "LONG":
        # BSL swept recently while looking for long = potential trap
        ok = not (fv.sweep_direction < 0 and fv.sweep_recency > 0.6)
        reason = ("BSL swept recently – possible bull trap" if not ok
                  else f"sweep_dir={fv.sweep_direction:+.0f}, recency={fv.sweep_recency:.2f}")
    else:
        ok = not (fv.sweep_direction > 0 and fv.sweep_recency > 0.6)
        reason = ("SSL swept recently – possible bear trap" if not ok
                  else f"sweep_dir={fv.sweep_direction:+.0f}, recency={fv.sweep_recency:.2f}")
    return GateResult(name, ok, reason)


def _gate_poi_exists(fv: FeatureVector, direction: str) -> GateResult:
    """A valid Point of Interest (OB or FVG) must exist near price."""
    name = "POI Exists"
    if direction == "LONG":
        ok = fv.price_in_bull_ob > 0 or fv.bull_fvg_count > 0
        reason = (f"in_bull_ob={fv.price_in_bull_ob:.0f}, "
                  f"bull_fvgs={fv.bull_fvg_count:.0f}")
    else:
        ok = fv.price_in_bear_ob > 0 or fv.bear_fvg_count > 0
        reason = (f"in_bear_ob={fv.price_in_bear_ob:.0f}, "
                  f"bear_fvgs={fv.bear_fvg_count:.0f}")
    return GateResult(name, ok, reason)


def _gate_cvd_not_opposing(fv: FeatureVector, direction: str) -> GateResult:
    """CVD slope must not be strongly against the trade direction."""
    name = "CVD Not Opposing"
    if direction == "LONG":
        ok = fv.cvd_slope_norm > -0.6    # allow mild negative (absorption zone)
        reason = f"cvd_slope={fv.cvd_slope_norm:+.2f}"
    else:
        ok = fv.cvd_slope_norm < 0.6
        reason = f"cvd_slope={fv.cvd_slope_norm:+.2f}"
    return GateResult(name, ok, reason)


def _gate_not_distribution_against(fv: FeatureVector, direction: str) -> GateResult:
    """Institutional phase should not be strongly opposing the trade."""
    name = "Phase Not Opposing"
    if direction == "LONG":
        ok = fv.phase_encoded > -0.5   # not distribution
        reason = f"phase_encoded={fv.phase_encoded:+.1f}"
    else:
        ok = fv.phase_encoded < 0.8    # not accumulation
        reason = f"phase_encoded={fv.phase_encoded:+.1f}"
    return GateResult(name, ok, reason)


# All gates for each direction
_LONG_GATES  = [
    _gate_mtf_confluence,
    _gate_structure_confirmation,
    _gate_pd_zone,
    _gate_no_opposing_sweep,
    _gate_poi_exists,
    _gate_cvd_not_opposing,
    _gate_not_distribution_against,
]

_SHORT_GATES = _LONG_GATES   # same gate functions, direction flips the logic


# ─────────────────────────────────────────────────────────────────────────────
# SOFT SCORE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def _p_structure(fv: FeatureVector, direction: str) -> float:
    """Structure sub-score: trend alignment + CHoCH/BOS recency."""
    if direction == "LONG":
        s = (max(fv.trend_1h, 0) + max(fv.trend_4h, 0) * 1.5 + max(fv.trend_daily, 0) * 0.8
             + fv.choch_recency * 2.5 + fv.bos_recency * 1.0) / 6.8
    else:
        s = (max(-fv.trend_1h, 0) + max(-fv.trend_4h, 0) * 1.5 + max(-fv.trend_daily, 0) * 0.8
             + fv.choch_recency * 2.5 + fv.bos_recency * 1.0) / 6.8
    return float(np.clip(s, 0.0, 1.0))


def _p_flow(fv: FeatureVector, direction: str) -> float:
    """Order flow sub-score: CVD, delta, absorption, imbalance."""
    if direction == "LONG":
        s = (max(fv.cvd_slope_norm, 0) * 2.5
             + max(fv.delta_flip, 0) * 2.0
             + fv.absorption * 1.0
             + fv.bull_imb * 1.5
             + fv.sweep_direction * max(fv.sweep_recency, 0) * 1.5) / 8.5
    else:
        s = (max(-fv.cvd_slope_norm, 0) * 2.5
             + max(-fv.delta_flip, 0) * 2.0
             + fv.absorption * 1.0
             + fv.bear_imb * 1.5
             + (-fv.sweep_direction) * max(fv.sweep_recency, 0) * 1.5) / 8.5
    return float(np.clip(s, 0.0, 1.0))


def _p_vp(fv: FeatureVector, direction: str) -> float:
    """Volume Profile sub-score: VPOC migration, value area position."""
    if direction == "LONG":
        vp_mig_ok = fv.vpoc_migration >= 0
        below_vah  = fv.price_vs_vah <= 0
        s = (int(vp_mig_ok) * 1.5 + int(below_vah) * 1.0
             + max(-fv.dist_from_vpoc_pct * 0.05, 0)) / 2.5
    else:
        vp_mig_ok = fv.vpoc_migration <= 0
        above_val  = fv.price_vs_vah >= 0
        s = (int(vp_mig_ok) * 1.5 + int(above_val) * 1.0
             + max(fv.dist_from_vpoc_pct * 0.05, 0)) / 2.5
    return float(np.clip(s, 0.0, 1.0))


# Soft-score weights (sum to 1.0)
_SOFT_WEIGHTS = {
    "p_xgb":   0.35,
    "p_lstm":  0.30,
    "p_struct": 0.15,
    "p_flow":  0.12,
    "p_vp":    0.08,
}


# ─────────────────────────────────────────────────────────────────────────────
# META-MODEL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetaModelResult:
    direction: str                     # "LONG" | "SHORT"
    gate_open: bool                    # True if all hard gates passed
    gates: List[GateResult] = field(default_factory=list)
    gates_passed: int = 0
    gates_total: int = 0

    p_xgb: float = 0.0
    p_lstm: float = 0.0
    p_struct: float = 0.0
    p_flow: float = 0.0
    p_vp: float = 0.0
    soft_score: float = 0.0

    # Final
    final_prob: float = 0.0           # 0 if gate_open=False, else soft_score
    conviction: str = "NO_TRADE"     # CONVICTION_HIGH/MEDIUM/LOW | NO_TRADE
    signal_label: str = ""

    def summary_lines(self) -> List[str]:
        lines = []
        lines.append(f"  Direction     : {self.direction}")
        lines.append(f"  Gate          : {'OPEN ✓' if self.gate_open else 'CLOSED ✗'}  "
                     f"({self.gates_passed}/{self.gates_total} passed)")
        for g in self.gates:
            lines.append(str(g))
        lines.append(f"  ─────────────────────────────────────────")
        lines.append(f"  p_xgb         : {self.p_xgb:.3f}  (w={_SOFT_WEIGHTS['p_xgb']:.2f})")
        lines.append(f"  p_lstm        : {self.p_lstm:.3f}  (w={_SOFT_WEIGHTS['p_lstm']:.2f})")
        lines.append(f"  p_struct      : {self.p_struct:.3f}  (w={_SOFT_WEIGHTS['p_struct']:.2f})")
        lines.append(f"  p_flow        : {self.p_flow:.3f}  (w={_SOFT_WEIGHTS['p_flow']:.2f})")
        lines.append(f"  p_vp          : {self.p_vp:.3f}  (w={_SOFT_WEIGHTS['p_vp']:.2f})")
        lines.append(f"  Soft Score    : {self.soft_score:.3f}")
        lines.append(f"  Final Prob    : {self.final_prob:.3f}")
        lines.append(f"  Conviction    : {self.conviction}  {self.signal_label}")
        return lines


def run_meta_model(
    fv: FeatureVector,
    p_xgb: float,
    p_lstm_long: float,
    p_lstm_short: float,
    direction: str,
) -> MetaModelResult:
    """
    Run the full HARD-GATE + SOFT-SCORE meta-model for a given direction.

    Parameters
    ----------
    fv            : FeatureVector for the current bar
    p_xgb         : XGBoost-proxy probability [0,1] (>0.5 = bullish)
    p_lstm_long   : LSTM-proxy long template similarity [0,1]
    p_lstm_short  : LSTM-proxy short template similarity [0,1]
    direction     : "LONG" or "SHORT"
    """
    result = MetaModelResult(direction=direction)
    result.gates_total = len(_LONG_GATES)

    # ── Hard gates ────────────────────────────────────────────────────────────
    gate_fns = _LONG_GATES if direction == "LONG" else _SHORT_GATES
    for fn in gate_fns:
        gr = fn(fv, direction)
        result.gates.append(gr)
        if gr.passed:
            result.gates_passed += 1

    result.gate_open = result.gates_passed == result.gates_total

    # ── Soft score ────────────────────────────────────────────────────────────
    # p_xgb is a bullish probability; invert for short
    p_xgb_dir = p_xgb if direction == "LONG" else (1.0 - p_xgb)
    p_lstm_dir = p_lstm_long if direction == "LONG" else p_lstm_short

    result.p_xgb   = float(np.clip(p_xgb_dir, 0.0, 1.0))
    result.p_lstm  = float(np.clip(p_lstm_dir, 0.0, 1.0))
    result.p_struct = _p_structure(fv, direction)
    result.p_flow   = _p_flow(fv, direction)
    result.p_vp     = _p_vp(fv, direction)

    result.soft_score = (
        result.p_xgb   * _SOFT_WEIGHTS["p_xgb"]
        + result.p_lstm  * _SOFT_WEIGHTS["p_lstm"]
        + result.p_struct * _SOFT_WEIGHTS["p_struct"]
        + result.p_flow  * _SOFT_WEIGHTS["p_flow"]
        + result.p_vp    * _SOFT_WEIGHTS["p_vp"]
    )

    result.final_prob = result.soft_score if result.gate_open else 0.0

    # ── Conviction tier ───────────────────────────────────────────────────────
    p = result.final_prob
    if not result.gate_open:
        result.conviction = "NO_TRADE"
        result.signal_label = "⛔ GATE BLOCKED"
    elif p >= 0.75:
        result.conviction = "CONVICTION_HIGH"
        result.signal_label = "🔥 A+ SETUP"
    elif p >= 0.60:
        result.conviction = "CONVICTION_MEDIUM"
        result.signal_label = "✅ B SETUP"
    elif p >= 0.45:
        result.conviction = "CONVICTION_LOW"
        result.signal_label = "⚠  C SETUP – CAUTION"
    else:
        result.conviction = "NO_TRADE"
        result.signal_label = "○ BELOW THRESHOLD"

    return result


def evaluate_both_directions(
    fv: FeatureVector,
    p_xgb: float,
    p_lstm_long: float,
    p_lstm_short: float,
) -> Tuple[MetaModelResult, MetaModelResult]:
    """Evaluate LONG and SHORT meta-models, return both."""
    long_result  = run_meta_model(fv, p_xgb, p_lstm_long, p_lstm_short, "LONG")
    short_result = run_meta_model(fv, p_xgb, p_lstm_long, p_lstm_short, "SHORT")
    return long_result, short_result


def dominant_signal(
    long_result: MetaModelResult,
    short_result: MetaModelResult,
) -> MetaModelResult:
    """
    Return the higher-conviction signal (or the one with higher final_prob
    if both are tradeable).  Prefers no-trade if neither exceeds 0.45.
    """
    tradeable_convictions = {"CONVICTION_HIGH", "CONVICTION_MEDIUM", "CONVICTION_LOW"}
    l_ok = long_result.conviction in tradeable_convictions
    s_ok = short_result.conviction in tradeable_convictions

    if l_ok and not s_ok:
        return long_result
    if s_ok and not l_ok:
        return short_result
    if l_ok and s_ok:
        return long_result if long_result.final_prob >= short_result.final_prob else short_result

    # Neither tradeable → return whichever has more gate passes
    return long_result if long_result.gates_passed >= short_result.gates_passed else short_result
