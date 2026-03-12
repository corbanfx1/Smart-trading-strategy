"""
High-Probability Trade Setup Scanner
──────────────────────────────────────
Aggregates all analysis modules into a single scored setup signal.

Scoring model (0–7 per direction):
  Long Setup:
    [1] MTF bias bullish   (Daily + 4H both agree)
    [2] 1H structure bull  (CHoCH ▲ or BOS ▲)
    [3] Price in discount  (below equilibrium)
    [4] Bullish POI nearby (OB or FVG in discount)
    [5] SSL swept recently (stop hunt complete)
    [6] Delta flip bullish (buying pressure returns)
    [7] CVD slope +ve or bull divergence

  Short Setup: mirror of above, inverted conditions.

  Signal fires when score ≥ MIN_LONG_SCORE / MIN_SHORT_SCORE.

Returns a SetupSignal dataclass containing the full diagnostic context.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from analysis.market_structure import (
    MarketStructureResult, StructureEvent, mtf_bias,
)
from analysis.order_blocks import OrderBlock, active_obs
from analysis.fair_value_gaps import FVG, unfilled_fvgs
from analysis.liquidity import LiquidityLevel, LiquiditySweep, recent_sweeps
from analysis.premium_discount import PDContext
from analysis.order_flow import OrderFlowReport
from analysis.institutional_phase import PhaseSnapshot
import config as cfg


@dataclass
class SetupSignal:
    """Full diagnostic package for a trade setup."""
    timestamp: pd.Timestamp
    direction: str          # 'LONG' | 'SHORT' | 'NONE'
    score: int              # 0–7
    max_score: int = 7

    # Sub-conditions (True = condition met)
    mtf_bias_ok: bool = False
    structure_ok: bool = False
    pd_zone_ok: bool = False
    poi_ok: bool = False        # OB or FVG in zone
    sweep_ok: bool = False
    delta_ok: bool = False
    cvd_ok: bool = False

    # Context
    mtf_bias_value: int = 0
    pd_context: Optional[PDContext] = None
    phase: Optional[PhaseSnapshot] = None
    triggering_event: Optional[StructureEvent] = None
    relevant_obs: List[OrderBlock] = field(default_factory=list)
    relevant_fvgs: List[FVG] = field(default_factory=list)
    relevant_sweeps: List[LiquiditySweep] = field(default_factory=list)

    @property
    def is_high_prob(self) -> bool:
        return self.score >= cfg.MIN_LONG_SCORE

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"  SETUP: {self.direction}  [{self.score}/{self.max_score}]  "
            f"{'★ HIGH PROB' if self.is_high_prob else '○ LOW PROB'}",
            f"  Time : {self.timestamp}",
            f"{'─'*55}",
            f"  MTF Bias  : {'✓' if self.mtf_bias_ok else '✗'}  ({'+' if self.mtf_bias_value==1 else '-' if self.mtf_bias_value==-1 else '●'})",
            f"  Structure : {'✓' if self.structure_ok else '✗'}  "
            f"({self.triggering_event.event_type if self.triggering_event else '—'})",
            f"  P/D Zone  : {'✓' if self.pd_zone_ok else '✗'}  "
            f"({self.pd_context.zone if self.pd_context else '—'})",
            f"  POI       : {'✓' if self.poi_ok else '✗'}  "
            f"(OBs={len(self.relevant_obs)}, FVGs={len(self.relevant_fvgs)})",
            f"  Sweep     : {'✓' if self.sweep_ok else '✗'}  "
            f"({len(self.relevant_sweeps)} recent)",
            f"  Delta     : {'✓' if self.delta_ok else '✗'}",
            f"  CVD       : {'✓' if self.cvd_ok else '✗'}",
        ]
        if self.phase:
            lines.append(f"  Phase     : {self.phase.phase}  "
                         f"(acc={self.phase.accum_score}, "
                         f"manip={self.phase.manip_score}, "
                         f"dist={self.phase.distrib_score})")
        lines.append(f"{'='*55}")
        return "\n".join(lines)


def scan_for_setups(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_daily: pd.DataFrame,
    ms_result: MarketStructureResult,
    obs: List[OrderBlock],
    fvgs: List[FVG],
    levels: List[LiquidityLevel],
    sweeps: List[LiquiditySweep],
    pd_ctx: PDContext,
    of_report: OrderFlowReport,
    phase: PhaseSnapshot,
) -> SetupSignal:
    """
    Evaluate the latest bar of df_1h against all analysis outputs and return
    a SetupSignal with direction and score.
    """
    n = len(df_1h)
    last_idx = n - 1
    ts = df_1h.index[-1]
    price = float(df_1h["close"].iloc[-1])

    # ── MTF Bias ─────────────────────────────────────────────────────────────
    bias = mtf_bias(df_4h, df_daily)
    bias_long  = bias == 1
    bias_short = bias == -1

    # ── Recent structure events ───────────────────────────────────────────────
    recent_events = [
        e for e in ms_result.events
        if e.idx >= last_idx - 5
    ]
    choch_bull = any(e.event_type == "CHOCH_BULL" for e in recent_events)
    bos_bull   = any(e.event_type == "BOS_BULL"   for e in recent_events)
    choch_bear = any(e.event_type == "CHOCH_BEAR" for e in recent_events)
    bos_bear   = any(e.event_type == "BOS_BEAR"   for e in recent_events)

    struct_long  = choch_bull or bos_bull
    struct_short = choch_bear or bos_bear

    trigger_long  = next((e for e in reversed(recent_events)
                          if e.event_type in ("CHOCH_BULL", "BOS_BULL")), None)
    trigger_short = next((e for e in reversed(recent_events)
                          if e.event_type in ("CHOCH_BEAR", "BOS_BEAR")), None)

    # ── P/D zone ─────────────────────────────────────────────────────────────
    pd_long  = pd_ctx.in_discount
    pd_short = pd_ctx.in_premium

    # ── POI near price ────────────────────────────────────────────────────────
    # Check if any active OB or unfilled FVG is within 1% of current price
    tol = price * 0.01
    act_obs = active_obs(obs, last_idx)
    unfill_fvgs = unfilled_fvgs(fvgs, last_idx)

    poi_long_obs  = [o for o in act_obs  if o.kind == "BULL" and price <= o.top + tol and price >= o.bottom - tol]
    poi_long_fvgs = [f for f in unfill_fvgs if f.kind == "BULL" and price <= f.top + tol and price >= f.bottom - tol]
    poi_short_obs  = [o for o in act_obs  if o.kind == "BEAR" and price <= o.top + tol and price >= o.bottom - tol]
    poi_short_fvgs = [f for f in unfill_fvgs if f.kind == "BEAR" and price <= f.top + tol and price >= f.bottom - tol]

    poi_long  = bool(poi_long_obs  or poi_long_fvgs)
    poi_short = bool(poi_short_obs or poi_short_fvgs)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    rswps = recent_sweeps(sweeps, n_bars=5)
    sweep_long  = any(s.kind == "BULL_SWEEP" for s in rswps)
    sweep_short = any(s.kind == "BEAR_SWEEP" for s in rswps)

    # ── Delta flip ────────────────────────────────────────────────────────────
    delta_long  = bool(of_report.flips["bull_flip"].iloc[-1]) or bool(of_report.flips["bull_flip"].iloc[-2:].any())
    delta_short = bool(of_report.flips["bear_flip"].iloc[-1]) or bool(of_report.flips["bear_flip"].iloc[-2:].any())

    # ── CVD ───────────────────────────────────────────────────────────────────
    cvd_slope_pos = float(of_report.cvd_slope.iloc[-1]) > 0
    bull_div      = bool(of_report.divergence["bull_div"].iloc[-3:].any())
    bear_div      = bool(of_report.divergence["bear_div"].iloc[-3:].any())
    cvd_long  = cvd_slope_pos or bull_div
    cvd_short = (not cvd_slope_pos) or bear_div

    # ── Score ─────────────────────────────────────────────────────────────────
    long_score = sum([
        int(bias_long),
        int(struct_long),
        int(pd_long),
        int(poi_long),
        int(sweep_long),
        int(delta_long),
        int(cvd_long),
    ])
    short_score = sum([
        int(bias_short),
        int(struct_short),
        int(pd_short),
        int(poi_short),
        int(sweep_short),
        int(delta_short),
        int(cvd_short),
    ])

    # ── Determine direction ───────────────────────────────────────────────────
    if long_score >= short_score and long_score >= cfg.MIN_LONG_SCORE:
        direction = "LONG"
        score = long_score
        sig = SetupSignal(
            timestamp=ts, direction=direction, score=score,
            mtf_bias_ok=bias_long, structure_ok=struct_long,
            pd_zone_ok=pd_long, poi_ok=poi_long, sweep_ok=sweep_long,
            delta_ok=delta_long, cvd_ok=cvd_long,
            mtf_bias_value=bias, pd_context=pd_ctx, phase=phase,
            triggering_event=trigger_long,
            relevant_obs=poi_long_obs, relevant_fvgs=poi_long_fvgs,
            relevant_sweeps=[s for s in rswps if s.kind == "BULL_SWEEP"],
        )
    elif short_score > long_score and short_score >= cfg.MIN_SHORT_SCORE:
        direction = "SHORT"
        score = short_score
        sig = SetupSignal(
            timestamp=ts, direction=direction, score=score,
            mtf_bias_ok=bias_short, structure_ok=struct_short,
            pd_zone_ok=pd_short, poi_ok=poi_short, sweep_ok=sweep_short,
            delta_ok=delta_short, cvd_ok=cvd_short,
            mtf_bias_value=bias, pd_context=pd_ctx, phase=phase,
            triggering_event=trigger_short,
            relevant_obs=poi_short_obs, relevant_fvgs=poi_short_fvgs,
            relevant_sweeps=[s for s in rswps if s.kind == "BEAR_SWEEP"],
        )
    else:
        direction = "NONE"
        score = max(long_score, short_score)
        sig = SetupSignal(
            timestamp=ts, direction=direction, score=score,
            mtf_bias_value=bias, pd_context=pd_ctx, phase=phase,
        )

    return sig
