"""
Institutional Risk Manager
Computes position sizing via Fixed Fractional, Volatility Targeting and
Kelly Criterion methods; also suggests partial-exit schedules.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from strategy.setup_detector import TradeSetup
import config

logger = logging.getLogger(__name__)


@dataclass
class PositionPlan:
    setup:             TradeSetup
    account_balance:   float
    risk_amount:       float         # $ at risk
    position_size:     float         # lots / contracts (fractional)
    max_loss:          float         # = risk_amount ($ per trade)
    kelly_size:        float         # Kelly-optimal size
    vol_target_size:   float         # volatility-targeting size
    recommended_size:  float         # conservative blend
    partial_exits:     List[Dict]    # staged exit schedule

    def summary(self) -> str:
        return (
            f"\n{'─'*50}\n"
            f"  POSITION PLAN  ({config.ASSET_NAME}  {self.setup.direction.upper()})\n"
            f"  Balance        : ${self.account_balance:,.0f}\n"
            f"  Risk (1%)      : ${self.risk_amount:,.0f}\n"
            f"  Rec. Size      : {self.recommended_size:.3f} lots\n"
            f"  Kelly Size     : {self.kelly_size:.3f} lots\n"
            f"  Vol-Target Size: {self.vol_target_size:.3f} lots\n"
            f"  Exit plan:\n"
            + "\n".join(
                f"    TP{p['tp_num']}: close {p['close_pct']*100:.0f}% @ "
                f"{p['price']:.2f} (R:R={p['rr']:.2f})"
                for p in self.partial_exits
            )
            + f"\n{'─'*50}\n"
        )


class RiskManager:
    """Compute institutional position plans for validated trade setups."""

    def __init__(
        self,
        account_balance: float = config.BACKTEST_INITIAL_CAPITAL,
        risk_pct:        float = config.RISK_PER_TRADE_PCT,
        kelly_fraction:  float = config.KELLY_FRACTION,
    ) -> None:
        self.account_balance = account_balance
        self.risk_pct        = risk_pct
        self.kelly_fraction  = kelly_fraction

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(self, setup: TradeSetup, win_rate: Optional[float] = None) -> PositionPlan:
        """
        Generate a complete position plan for *setup*.

        Args:
            setup:    validated TradeSetup
            win_rate: historical win-rate for Kelly sizing (if known)
        """
        risk_amount = self.account_balance * self.risk_pct
        risk_pts    = abs(setup.entry - setup.stop)

        if risk_pts == 0:
            risk_pts = setup.atr

        # Dollar value per point for XAUUSD / Gold futures
        # (1 standard lot = 100 oz, $1 per point)
        point_value = config.POINT_VALUE   # USD per point per lot

        # Fixed-fractional sizing
        ff_size = risk_amount / (risk_pts * point_value)

        # Kelly sizing
        kelly_size = self._kelly_size(
            setup, win_rate, risk_amount, risk_pts, point_value
        )

        # Volatility-targeting sizing (target 1% daily portfolio vol)
        vol_target_size = self._vol_target_size(setup, risk_amount, point_value)

        # Conservative blend: min of all three methods
        recommended = min(ff_size, kelly_size, vol_target_size)

        # Scale down in high-vol regime
        if setup.vol_regime == "high":
            recommended *= 0.5
        elif setup.vol_regime == "medium":
            recommended *= 0.75

        partial_exits = self._build_exit_schedule(setup)

        return PositionPlan(
            setup             = setup,
            account_balance   = self.account_balance,
            risk_amount       = round(risk_amount, 2),
            position_size     = round(ff_size, 4),
            max_loss          = round(risk_amount, 2),
            kelly_size        = round(kelly_size, 4),
            vol_target_size   = round(vol_target_size, 4),
            recommended_size  = round(max(0.001, recommended), 4),
            partial_exits     = partial_exits,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _kelly_size(
        self,
        setup:       TradeSetup,
        win_rate:    Optional[float],
        risk_amount: float,
        risk_pts:    float,
        point_value: float,
    ) -> float:
        w   = win_rate or 0.50          # default 50% if unknown
        b   = setup.rr1                 # reward / risk ratio at TP1
        raw = (b * w - (1 - w)) / b     # full Kelly fraction
        raw = max(0, raw)

        frac_kelly = self.kelly_fraction * raw
        return (frac_kelly * self.account_balance) / (risk_pts * point_value + 1e-9)

    def _vol_target_size(
        self, setup: TradeSetup, risk_amount: float, point_value: float
    ) -> float:
        """Size such that 1-day 1σ move = 1% of account."""
        target_vol_pct = 0.01
        atr_pct        = setup.atr / setup.entry
        target_pts     = target_vol_pct * self.account_balance / point_value
        return target_pts / (setup.entry * atr_pct + 1e-9)

    @staticmethod
    def _build_exit_schedule(setup: TradeSetup) -> List[Dict]:
        """Stage-out plan: 40% at TP1, 35% at TP2, 25% at TP3."""
        return [
            {"tp_num": 1, "price": setup.tp1, "close_pct": 0.40, "rr": setup.rr1},
            {"tp_num": 2, "price": setup.tp2, "close_pct": 0.35, "rr": setup.rr2},
            {"tp_num": 3, "price": setup.tp3, "close_pct": 0.25, "rr": setup.rr3},
        ]
