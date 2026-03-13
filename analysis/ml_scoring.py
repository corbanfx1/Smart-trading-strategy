"""
Proxy ML Scoring Engine
────────────────────────
Implements two complementary probability estimators without requiring
pre-trained weights or external ML libraries:

  p_xgb  – XGBoost-style proxy
            Feature engineering → weighted ensemble of decision-tree rules.
            Each "tree" is a conditional threshold rule that contributes a
            probability fragment. The ensemble mean approximates what a
            gradient-boosted classifier would output on these features.

  p_lstm – LSTM-style sequential proxy
            Sliding-window pattern matcher. Defines canonical "ideal sequences"
            for long and short setups (e.g., sweep → CHoCH → OB retest →
            delta flip → CVD confirmation). Scores recent bars against each
            template using a weighted Hamming-similarity measure.

Both scores are in [0, 1] where > 0.5 = directional bullish/bearish confidence.

See ml_meta.py for the Meta-model that combines these with hard gates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from analysis.market_structure import MarketStructureResult, StructureEvent
from analysis.order_blocks import OrderBlock, active_obs
from analysis.fair_value_gaps import FVG, unfilled_fvgs
from analysis.liquidity import LiquidityLevel, LiquiditySweep, unswept_levels
from analysis.premium_discount import PDContext
from analysis.order_flow import OrderFlowReport
from analysis.volume_profile import VolumeProfileResult, VPOCMigration
from analysis.institutional_phase import PhaseSnapshot
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VECTOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """
    26-feature vector extracted for each bar.
    Naming convention: positive values favour longs, negative favour shorts.
    """
    # ── Market Structure ──────────────────────────────────────────
    trend_1h: float           # -1 / 0 / 1
    trend_4h: float           # -1 / 0 / 1
    trend_daily: float        # -1 / 0 / 1
    mtf_confluence: float     # 0 / 1  (4H and Daily agree)
    choch_recency: float      # 0–1 (1 = CHoCH on current bar, 0 = very old)
    bos_recency: float        # 0–1
    dist_to_last_sh_pct: float # % gap from price to last swing high (negative = above)
    dist_to_last_sl_pct: float # % gap from price to last swing low  (positive = above)

    # ── P/D Zone ──────────────────────────────────────────────────
    pd_encoded: float         # -2 (deep prem) to +2 (deep disc)
    dist_from_eq_pct: float   # signed % from equilibrium

    # ── Order Blocks / FVG ────────────────────────────────────────
    bull_ob_count: float      # active bullish OBs
    bear_ob_count: float      # active bearish OBs
    price_in_bull_ob: float   # 0/1
    price_in_bear_ob: float   # 0/1
    bull_fvg_count: float     # unfilled bullish FVGs
    bear_fvg_count: float     # unfilled bearish FVGs

    # ── Liquidity ─────────────────────────────────────────────────
    sweep_recency: float      # 0–1 (1 = sweep on this bar)
    sweep_direction: float    # +1 = SSL swept (bullish), -1 = BSL swept (bearish)
    unswept_bsl_count: float
    unswept_ssl_count: float

    # ── Order Flow ────────────────────────────────────────────────
    cvd_slope_norm: float     # normalised CVD slope (-1 to +1)
    delta_flip: float         # +1 bull flip, -1 bear flip, 0 none
    absorption: float         # 0/1
    bull_imb: float           # 0/1
    bear_imb: float           # 0/1

    # ── Volume Profile ────────────────────────────────────────────
    dist_from_vpoc_pct: float # signed % (positive = above VPOC)
    price_vs_vah: float       # +1 above VAH, 0 inside VA, -1 below VAL
    vpoc_migration: float     # +1 rising, 0 flat, -1 falling

    # ── Institutional Phase ────────────────────────────────────────
    phase_encoded: float      # accum=+1, manip=0, distrib=-1, trend=0

    def as_array(self) -> np.ndarray:
        return np.array([
            self.trend_1h, self.trend_4h, self.trend_daily, self.mtf_confluence,
            self.choch_recency, self.bos_recency,
            self.dist_to_last_sh_pct, self.dist_to_last_sl_pct,
            self.pd_encoded, self.dist_from_eq_pct,
            self.bull_ob_count, self.bear_ob_count, self.price_in_bull_ob, self.price_in_bear_ob,
            self.bull_fvg_count, self.bear_fvg_count,
            self.sweep_recency, self.sweep_direction, self.unswept_bsl_count, self.unswept_ssl_count,
            self.cvd_slope_norm, self.delta_flip, self.absorption, self.bull_imb, self.bear_imb,
            self.dist_from_vpoc_pct, self.price_vs_vah, self.vpoc_migration,
            self.phase_encoded,
        ], dtype=float)


def extract_features(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_daily: pd.DataFrame,
    ms_1h: MarketStructureResult,
    ms_4h: MarketStructureResult,
    ms_daily: MarketStructureResult,
    obs: List[OrderBlock],
    fvgs: List[FVG],
    levels: List[LiquidityLevel],
    sweeps: List[LiquiditySweep],
    pd_ctx: PDContext,
    of_report: OrderFlowReport,
    vp: Optional[VolumeProfileResult],
    vp_migration: Optional[VPOCMigration],
    phase: PhaseSnapshot,
) -> FeatureVector:
    """Extract the 29-feature vector for the latest bar."""

    n = len(df_1h)
    last_idx = n - 1
    price = float(df_1h["close"].iloc[-1])

    # ── Trend ─────────────────────────────────────────────────────────────────
    t1h    = float(ms_1h.trend.iloc[-1])   if len(ms_1h.trend) > 0   else 0.0
    t4h    = float(ms_4h.trend.iloc[-1])   if len(ms_4h.trend) > 0   else 0.0
    tdaily = float(ms_daily.trend.iloc[-1]) if len(ms_daily.trend) > 0 else 0.0
    mtf_conf = 1.0 if (t4h > 0 and tdaily >= 0) or (t4h < 0 and tdaily <= 0) else 0.0

    # ── CHoCH / BOS recency (1 = this bar, 0 = very old, decay over 20 bars) ─
    DECAY = 20
    def event_recency(events: List[StructureEvent], kind_tokens: List[str]) -> float:
        for e in reversed(events):
            if any(t in e.event_type for t in kind_tokens):
                age = last_idx - e.idx
                return max(0.0, 1.0 - age / DECAY)
        return 0.0

    choch_rec = event_recency(ms_1h.events, ["CHOCH"])
    bos_rec   = event_recency(ms_1h.events, ["BOS"])

    # ── Distance to last swings ───────────────────────────────────────────────
    sh = ms_1h.swing_highs[-1].price if ms_1h.swing_highs else price
    sl = ms_1h.swing_lows[-1].price  if ms_1h.swing_lows  else price
    dist_sh = (price - sh) / sh * 100   # negative if below swing high
    dist_sl = (price - sl) / sl * 100   # positive if above swing low

    # ── P/D zone ──────────────────────────────────────────────────────────────
    pd_map = {"DEEP_PREMIUM": -2, "PREMIUM": -1, "EQUILIBRIUM": 0,
               "DISCOUNT": 1, "DEEP_DISCOUNT": 2}
    pd_enc = float(pd_map.get(pd_ctx.zone, 0))

    # ── OBs in zone ───────────────────────────────────────────────────────────
    act = active_obs(obs, last_idx)
    bull_obs = [o for o in act if o.kind == "BULL"]
    bear_obs = [o for o in act if o.kind == "BEAR"]
    tol = price * 0.005
    in_bull_ob = float(any(o.bottom - tol <= price <= o.top + tol for o in bull_obs))
    in_bear_ob = float(any(o.bottom - tol <= price <= o.top + tol for o in bear_obs))

    # ── FVG counts ────────────────────────────────────────────────────────────
    unfill = unfilled_fvgs(fvgs, last_idx)
    bfvg = sum(1 for f in unfill if f.kind == "BULL")
    sfvg = sum(1 for f in unfill if f.kind == "BEAR")

    # ── Sweeps ────────────────────────────────────────────────────────────────
    sw_rec = 0.0
    sw_dir = 0.0
    for sw in reversed(sweeps):
        age = last_idx - sw.idx
        if age <= DECAY:
            sw_rec = max(0.0, 1.0 - age / DECAY)
            sw_dir = 1.0 if sw.kind == "BULL_SWEEP" else -1.0
            break

    bsl_u = len(unswept_levels(levels, last_idx, kind=["BSL", "EQH"]))
    ssl_u = len(unswept_levels(levels, last_idx, kind=["SSL", "EQL"]))

    # ── Order flow ────────────────────────────────────────────────────────────
    cvd_s = float(of_report.cvd_slope.iloc[-1])
    cvd_max = float(of_report.cvd_slope.abs().max()) or 1.0
    cvd_norm = np.clip(cvd_s / cvd_max, -1.0, 1.0)

    bf = bool(of_report.flips["bull_flip"].iloc[-3:].any())
    bearf = bool(of_report.flips["bear_flip"].iloc[-3:].any())
    delta_flip_val = 1.0 if bf else (-1.0 if bearf else 0.0)

    abs_v  = float(of_report.absorption.iloc[-3:].any())
    bimb   = float(of_report.imbalance["bull_imbalance"].iloc[-3:].any())
    bearimb = float(of_report.imbalance["bear_imbalance"].iloc[-3:].any())

    # ── Volume Profile ────────────────────────────────────────────────────────
    if vp is not None:
        vdist = vp.dist_from_vpoc_pct(price)
        if price > vp.vah:
            pvvah = 1.0
        elif price < vp.val:
            pvvah = -1.0
        else:
            pvvah = 0.0
    else:
        vdist = 0.0
        pvvah = 0.0

    vmig = 0.0
    if vp_migration is not None:
        vmig = 1.0 if vp_migration.direction == "RISING" else \
               -1.0 if vp_migration.direction == "FALLING" else 0.0

    # ── Phase ─────────────────────────────────────────────────────────────────
    ph_map = {"ACCUMULATION": 1.0, "TRENDING": 0.5, "MANIPULATION": 0.0, "DISTRIBUTION": -1.0}
    ph_enc = ph_map.get(phase.phase, 0.0)

    return FeatureVector(
        trend_1h=t1h, trend_4h=t4h, trend_daily=tdaily, mtf_confluence=mtf_conf,
        choch_recency=choch_rec, bos_recency=bos_rec,
        dist_to_last_sh_pct=dist_sh, dist_to_last_sl_pct=dist_sl,
        pd_encoded=pd_enc, dist_from_eq_pct=pd_ctx.pct_from_eq,
        bull_ob_count=float(len(bull_obs)), bear_ob_count=float(len(bear_obs)),
        price_in_bull_ob=in_bull_ob, price_in_bear_ob=in_bear_ob,
        bull_fvg_count=float(bfvg), bear_fvg_count=float(sfvg),
        sweep_recency=sw_rec, sweep_direction=sw_dir,
        unswept_bsl_count=float(bsl_u), unswept_ssl_count=float(ssl_u),
        cvd_slope_norm=cvd_norm, delta_flip=delta_flip_val,
        absorption=abs_v, bull_imb=bimb, bear_imb=bearimb,
        dist_from_vpoc_pct=vdist, price_vs_vah=pvvah, vpoc_migration=vmig,
        phase_encoded=ph_enc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost-PROXY  (p_xgb)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#   Each "tree" is a weighted conditional rule.
#   Rule output: sigmoid-scaled partial probability.
#   Forest mean → final p_xgb in [0, 1].
#   p_xgb > 0.5 = bullish, < 0.5 = bearish.

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _tree_mtf_structure(fv: FeatureVector) -> float:
    """Tree 1: MTF trend confluence."""
    score = fv.trend_1h + fv.trend_4h * 1.5 + fv.trend_daily * 0.8 + fv.mtf_confluence * 2.0
    return _sigmoid(score * 0.6)

def _tree_choch_sweep_combo(fv: FeatureVector) -> float:
    """Tree 2: CHoCH + sweep combination (high-value entry signal)."""
    score = (fv.choch_recency * 3.0
             + fv.sweep_recency * 2.0
             + fv.sweep_direction * 1.5)
    return _sigmoid(score * 0.8)

def _tree_poi_quality(fv: FeatureVector) -> float:
    """Tree 3: Quality and direction of nearby POI."""
    long_poi  = fv.price_in_bull_ob + fv.bull_fvg_count * 0.3
    short_poi = fv.price_in_bear_ob + fv.bear_fvg_count * 0.3
    score = (long_poi - short_poi) * 2.0 + fv.pd_encoded * 0.5
    return _sigmoid(score * 0.7)

def _tree_pd_context(fv: FeatureVector) -> float:
    """Tree 4: Premium/Discount context – buy in discount, sell in premium."""
    score = fv.pd_encoded * 1.2 - fv.dist_from_eq_pct * 0.05
    return _sigmoid(score * 0.5)

def _tree_cvd_flow(fv: FeatureVector) -> float:
    """Tree 5: CVD slope + delta flip + absorption."""
    score = (fv.cvd_slope_norm * 2.5
             + fv.delta_flip * 2.0
             + fv.absorption * 0.5)
    return _sigmoid(score * 0.7)

def _tree_imbalance_direction(fv: FeatureVector) -> float:
    """Tree 6: Institutional imbalance candles."""
    score = fv.bull_imb * 2.0 - fv.bear_imb * 2.0 + fv.cvd_slope_norm
    return _sigmoid(score * 0.6)

def _tree_vp_context(fv: FeatureVector) -> float:
    """Tree 7: Volume Profile position – price leaving VPOC on volume = continuation."""
    score = (fv.vpoc_migration * 1.5
             - fv.dist_from_vpoc_pct * 0.1
             + fv.price_vs_vah * (-1.0))   # above VAH = potential rejection
    return _sigmoid(score * 0.5)

def _tree_liquidity_stack(fv: FeatureVector) -> float:
    """Tree 8: Unswept liquidity stack – price moves toward resting liquidity."""
    long_pull  = fv.unswept_bsl_count * 0.3    # BSL above pulls price up
    short_pull = fv.unswept_ssl_count * 0.3    # SSL below pulls price down
    score = long_pull - short_pull + fv.sweep_direction
    return _sigmoid(score * 0.4)

def _tree_swing_proximity(fv: FeatureVector) -> float:
    """Tree 9: Distance from swing levels – far below SH = room to run."""
    score = (-fv.dist_to_last_sh_pct * 0.15   # far from SH → more upside room
             + fv.dist_to_last_sl_pct * 0.15  # well above SL → confirmed bounce
             + fv.trend_1h * 0.5)
    return _sigmoid(score * 0.4)

def _tree_phase_alignment(fv: FeatureVector) -> float:
    """Tree 10: Institutional phase → directional hint alignment."""
    score = fv.phase_encoded * 2.0 + fv.mtf_confluence * 0.5
    return _sigmoid(score * 0.6)


# Tree weights (sum to 1.0, trained-equivalent using domain knowledge)
_XGB_TREES = [
    (_tree_mtf_structure,       0.18),
    (_tree_choch_sweep_combo,   0.16),
    (_tree_poi_quality,         0.14),
    (_tree_pd_context,          0.10),
    (_tree_cvd_flow,            0.14),
    (_tree_imbalance_direction, 0.08),
    (_tree_vp_context,          0.08),
    (_tree_liquidity_stack,     0.04),
    (_tree_swing_proximity,     0.04),
    (_tree_phase_alignment,     0.04),
]

def compute_p_xgb(fv: FeatureVector) -> float:
    """
    Weighted ensemble of rule-trees → XGBoost-proxy probability.
    Output in [0, 1].  > 0.5 = bullish bias.
    """
    total = sum(fn(fv) * w for fn, w in _XGB_TREES)
    return float(np.clip(total, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# LSTM-PROXY  (p_lstm)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#   Canonical "ideal sequences" for long and short setups defined as
#   ordered event templates.  Recent history (last SEQ_LEN bars) is encoded
#   as a binary event vector per bar. Similarity to each template → score.

SEQ_LEN = 8   # look back this many bars for sequence matching

def _encode_bar_events(
    idx: int,
    df: pd.DataFrame,
    ms_1h: MarketStructureResult,
    sweeps: List[LiquiditySweep],
    of_report: OrderFlowReport,
    obs: List[OrderBlock],
    fvgs: List[FVG],
    pd_ctx_series: pd.Series,
) -> Dict[str, bool]:
    """Encode the events that occurred at bar `idx` as a binary feature dict."""
    n = len(df)
    price = float(df["close"].iloc[idx])

    events_at = [e for e in ms_1h.events if e.idx == idx]

    has_choch_bull = any("CHOCH_BULL" in e.event_type for e in events_at)
    has_choch_bear = any("CHOCH_BEAR" in e.event_type for e in events_at)
    has_bos_bull   = any("BOS_BULL"   in e.event_type for e in events_at)
    has_bos_bear   = any("BOS_BEAR"   in e.event_type for e in events_at)

    sweeps_at = [s for s in sweeps if s.idx == idx]
    has_ssl_sweep = any(s.kind == "BULL_SWEEP" for s in sweeps_at)
    has_bsl_sweep = any(s.kind == "BEAR_SWEEP" for s in sweeps_at)

    has_delta_bull = bool(of_report.flips["bull_flip"].iloc[idx]) if idx < n else False
    has_delta_bear = bool(of_report.flips["bear_flip"].iloc[idx]) if idx < n else False
    has_absorption = bool(of_report.absorption.iloc[idx]) if idx < n else False
    has_bull_imb   = bool(of_report.imbalance["bull_imbalance"].iloc[idx]) if idx < n else False
    has_bear_imb   = bool(of_report.imbalance["bear_imbalance"].iloc[idx]) if idx < n else False

    act = active_obs(obs, idx)
    in_bull_ob = any(o.bottom <= price <= o.top for o in act if o.kind == "BULL")
    in_bear_ob = any(o.bottom <= price <= o.top for o in act if o.kind == "BEAR")

    zone = pd_ctx_series.iloc[idx] if idx < len(pd_ctx_series) else "EQUILIBRIUM"
    in_discount = zone in ("DISCOUNT", "DEEP_DISCOUNT")
    in_premium  = zone in ("PREMIUM", "DEEP_PREMIUM")

    cvd_pos = bool(of_report.cvd_slope.iloc[idx] > 0) if idx < n else False

    return {
        "choch_bull": has_choch_bull,
        "choch_bear": has_choch_bear,
        "bos_bull":   has_bos_bull,
        "bos_bear":   has_bos_bear,
        "ssl_sweep":  has_ssl_sweep,
        "bsl_sweep":  has_bsl_sweep,
        "delta_bull": has_delta_bull,
        "delta_bear": has_delta_bear,
        "absorption": has_absorption,
        "bull_imb":   has_bull_imb,
        "bear_imb":   has_bear_imb,
        "in_bull_ob": in_bull_ob,
        "in_bear_ob": in_bear_ob,
        "in_discount": in_discount,
        "in_premium":  in_premium,
        "cvd_pos":    cvd_pos,
    }


# ── Canonical sequences ───────────────────────────────────────────────────────
# Each sequence is a list of (event_key, expected_bool, weight) tuples.
# The sequence is ordered: earlier items should appear at earlier bars.
# Similarity = weighted fraction of matched events across the window.

_LONG_TEMPLATE = [
    # (event_key,      expected, weight)
    ("ssl_sweep",      True,   3.0),   # SSL swept (stop hunt)
    ("choch_bull",     True,   3.0),   # CHoCH bullish (structure shift)
    ("in_discount",    True,   1.5),   # price in discount zone
    ("in_bull_ob",     True,   2.5),   # OB retest
    ("absorption",     True,   1.5),   # institutional absorption at OB
    ("delta_bull",     True,   2.5),   # delta flip bullish
    ("cvd_pos",        True,   2.0),   # CVD slope positive
    ("bull_imb",       True,   1.5),   # bullish impulse away from OB
]

_SHORT_TEMPLATE = [
    ("bsl_sweep",      True,   3.0),
    ("choch_bear",     True,   3.0),
    ("in_premium",     True,   1.5),
    ("in_bear_ob",     True,   2.5),
    ("absorption",     True,   1.5),
    ("delta_bear",     True,   2.5),
    ("cvd_pos",        False,  2.0),   # CVD slope negative
    ("bear_imb",       True,   1.5),
]

_TOTAL_WEIGHT = sum(w for _, _, w in _LONG_TEMPLATE)


def _template_similarity(
    bar_events: List[Dict[str, bool]],
    template: List[Tuple[str, bool, float]],
) -> float:
    """
    Compute weighted similarity between a sequence of bar events and a template.

    The template is matched in order (early template items checked against
    early bars in the window).  Partial matches are scored proportionally.
    """
    if not bar_events or not template:
        return 0.0

    total_weight = sum(w for _, _, w in template)
    if total_weight == 0:
        return 0.0

    matched_weight = 0.0
    n_bars = len(bar_events)
    n_tmpl = len(template)

    # Assign each template step to the bar in the window that best matches it
    for t_idx, (key, expected, weight) in enumerate(template):
        # Map template step to bar window proportionally
        bar_idx = int(t_idx / n_tmpl * n_bars)
        bar_idx = min(bar_idx, n_bars - 1)

        # Check current bar AND ±1 neighbour
        for offset in [0, -1, 1]:
            bi = bar_idx + offset
            if 0 <= bi < n_bars:
                if bar_events[bi].get(key, False) == expected:
                    matched_weight += weight
                    break

    return matched_weight / total_weight


def compute_p_lstm(
    df: pd.DataFrame,
    ms_1h: MarketStructureResult,
    sweeps: List[LiquiditySweep],
    of_report: OrderFlowReport,
    obs: List[OrderBlock],
    fvgs: List[FVG],
    pd_zone_series: pd.Series,
    direction: str = "LONG",   # "LONG" or "SHORT"
    seq_len: int = SEQ_LEN,
) -> float:
    """
    LSTM-proxy: slide a window of `seq_len` bars, encode events at each bar,
    and measure similarity to the canonical template.

    Returns p_lstm in [0, 1].
    """
    n = len(df)
    last_idx = n - 1
    start_idx = max(0, last_idx - seq_len + 1)

    bar_events = [
        _encode_bar_events(i, df, ms_1h, sweeps, of_report, obs, fvgs, pd_zone_series)
        for i in range(start_idx, last_idx + 1)
    ]

    template = _LONG_TEMPLATE if direction == "LONG" else _SHORT_TEMPLATE
    similarity = _template_similarity(bar_events, template)

    # Apply a slight boost if the final bar contains the key entry events
    final = bar_events[-1] if bar_events else {}
    entry_bonus = 0.0
    if direction == "LONG" and (final.get("delta_bull") or final.get("in_bull_ob")):
        entry_bonus = 0.05
    elif direction == "SHORT" and (final.get("delta_bear") or final.get("in_bear_ob")):
        entry_bonus = 0.05

    return float(np.clip(similarity + entry_bonus, 0.0, 1.0))
