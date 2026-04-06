"""
Institutional Risk Manager.
Computes position size, stop-loss, take-profit, and trailing stop
using ATR-based dynamic levels with GARCH vol-scaling.
Enforces daily drawdown limits and max concurrent positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from config.settings import RISK
from utils.logger import get_logger

log = get_logger("RiskManager")


@dataclass
class TradeSetup:
    """Fully specified trade setup ready for execution."""
    # Identity
    trade_id:         str
    symbol:           str = "XAUUSD"
    direction:        int = 0      # +1 long, -1 short

    # Prices
    entry_price:      float = 0.0
    stop_loss:        float = 0.0
    take_profit_1:    float = 0.0  # partial TP (50% position)
    take_profit_2:    float = 0.0  # full TP
    trailing_stop:    float = 0.0  # initial trailing stop distance

    # Size
    lots:             float = 0.0
    risk_usd:         float = 0.0
    potential_reward: float = 0.0
    rr_ratio:         float = 0.0

    # Context
    confluence_score: float = 0.0
    confluences:      list  = field(default_factory=list)
    signal_strength:  int   = 0
    atr_14:           float = 0.0
    vol_regime:       str   = "normal"
    vol_scalar:       float = 1.0

    # Meta
    timestamp:        str   = ""
    status:           str   = "pending"   # pending / active / closed
    notes:            str   = ""

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


class RiskManager:
    """
    Computes position sizing and trade levels.
    Tracks daily P&L and enforces hard limits.
    """

    def __init__(self, config: dict = None):
        self.cfg          = config or RISK.copy()
        self.daily_pnl    = 0.0
        self.open_trades  = 0
        self.trade_count  = 0
        self._today       = date.today()

    # ─── DAILY RESET ───────────────────────────────────────────────────────────

    def _check_daily_reset(self):
        today = date.today()
        if today != self._today:
            log.info("RiskManager: new day — resetting daily P&L (was %.2f)", self.daily_pnl)
            self.daily_pnl = 0.0
            self._today = today

    # ─── GUARD RAILS ───────────────────────────────────────────────────────────

    def can_trade(self) -> tuple[bool, str]:
        self._check_daily_reset()

        max_dd = self.cfg["account_balance"] * self.cfg["max_daily_drawdown_pct"]
        if self.daily_pnl <= -max_dd:
            return False, f"daily_drawdown_limit_hit ({self.daily_pnl:.2f} USD)"

        if self.open_trades >= self.cfg["max_open_trades"]:
            return False, f"max_open_trades_reached ({self.open_trades})"

        return True, "ok"

    # ─── POSITION SIZING ───────────────────────────────────────────────────────

    def compute_position_size(
        self,
        entry:       float,
        stop_loss:   float,
        vol_scalar:  float = 1.0,
        confidence:  float = 1.0,
    ) -> tuple[float, float]:
        """
        Risk-based position sizing.
        risk_usd = account × risk_pct × vol_scalar × confidence_scale
        lots = risk_usd / (sl_distance_pips × pip_value)

        Returns (lots, risk_usd)
        """
        sl_dist    = abs(entry - stop_loss)
        if sl_dist < 1e-6:
            return 0.0, 0.0

        # Confidence scaling: reduce size for lower confidence signals
        conf_scale = max(0.5, min(1.0, confidence))

        risk_usd   = (self.cfg["account_balance"] *
                      self.cfg["risk_per_trade_pct"] *
                      vol_scalar *
                      conf_scale)

        # Gold: $1 per oz per $1 price move, 100 oz per standard lot
        lot_size   = self.cfg["lot_size"]  # oz per lot
        risk_per_lot = sl_dist * lot_size

        if risk_per_lot < 1e-6:
            return 0.0, 0.0

        lots = risk_usd / risk_per_lot
        lots = max(0.01, round(lots, 2))  # minimum 0.01 lots

        actual_risk = lots * sl_dist * lot_size
        return lots, actual_risk

    # ─── TRADE LEVELS ──────────────────────────────────────────────────────────

    def compute_trade_levels(
        self,
        direction:   int,
        entry:       float,
        atr_value:   float,
        sl_mult:     float = None,
        tp1_mult:    float = None,
        tp2_mult:    float = None,
    ) -> dict:
        """
        Compute SL and TP levels based on ATR multiples.
        Default: SL=1.5×ATR, TP1=2.25×ATR (1.5 R:R), TP2=3.75×ATR (2.5 R:R)
        """
        sl_m  = sl_mult  or self.cfg["atr_sl_multiplier"]
        tp1_m = tp1_mult or self.cfg["partial_tp_rr"] * sl_m
        tp2_m = tp2_mult or self.cfg["atr_tp_multiplier"]

        sl_dist  = sl_m  * atr_value
        tp1_dist = tp1_m * atr_value
        tp2_dist = tp2_m * atr_value
        trail    = self.cfg["trailing_stop_atr"] * atr_value

        if direction == 1:   # LONG
            sl   = entry - sl_dist
            tp1  = entry + tp1_dist
            tp2  = entry + tp2_dist
        else:                 # SHORT
            sl   = entry + sl_dist
            tp1  = entry - tp1_dist
            tp2  = entry - tp2_dist

        rr = tp2_dist / max(sl_dist, 1e-9)

        # Slippage adjustment
        slip = self.cfg["slippage_pips"]
        if direction == 1:
            entry_adj = entry + slip
            sl        = sl - slip
        else:
            entry_adj = entry - slip
            sl        = sl + slip

        return {
            "entry":         round(entry_adj, 3),
            "stop_loss":     round(sl, 3),
            "take_profit_1": round(tp1, 3),
            "take_profit_2": round(tp2, 3),
            "trailing_stop": round(trail, 3),
            "sl_dist":       round(sl_dist, 3),
            "tp2_dist":      round(tp2_dist, 3),
            "rr_ratio":      round(rr, 2),
        }

    # ─── BUILD TRADE SETUP ─────────────────────────────────────────────────────

    def build_trade_setup(
        self,
        direction:   int,
        entry:       float,
        atr_value:   float,
        signal_dict: dict,
        garch_dict:  dict,
        meta_result: dict,
    ) -> Optional[TradeSetup]:
        """
        Create a fully specified TradeSetup.
        Returns None if risk limits prevent the trade.
        """
        ok, reason = self.can_trade()
        if not ok:
            log.warning("RiskManager: cannot trade — %s", reason)
            return None

        vol_scalar  = garch_dict.get("garch_vol_scalar", 1.0)
        confidence  = meta_result.get("confidence", 0.5)
        vol_regime  = garch_dict.get("garch_regime", "normal")
        confluences = meta_result.get("confluences", [])

        levels = self.compute_trade_levels(direction, entry, atr_value)
        lots, risk_usd = self.compute_position_size(
            entry      = levels["entry"],
            stop_loss  = levels["stop_loss"],
            vol_scalar = vol_scalar,
            confidence = confidence,
        )

        if lots < 0.01:
            log.warning("RiskManager: position size too small (%.4f lots)", lots)
            return None

        # Check minimum R:R
        if levels["rr_ratio"] < self.cfg["default_rr"]:
            log.warning("RiskManager: R:R %.2f < %.2f — rejected",
                        levels["rr_ratio"], self.cfg["default_rr"])
            return None

        self.trade_count += 1
        trade_id = f"XAUUSD_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.trade_count:04d}"

        setup = TradeSetup(
            trade_id         = trade_id,
            direction        = direction,
            entry_price      = levels["entry"],
            stop_loss        = levels["stop_loss"],
            take_profit_1    = levels["take_profit_1"],
            take_profit_2    = levels["take_profit_2"],
            trailing_stop    = levels["trailing_stop"],
            lots             = lots,
            risk_usd         = round(risk_usd, 2),
            potential_reward = round(lots * levels["tp2_dist"] * self.cfg["lot_size"], 2),
            rr_ratio         = levels["rr_ratio"],
            confluence_score = confidence,
            confluences      = confluences,
            signal_strength  = meta_result.get("signal", 0),
            atr_14           = atr_value,
            vol_regime       = vol_regime,
            vol_scalar       = round(vol_scalar, 3),
            timestamp        = datetime.utcnow().isoformat() + "Z",
        )

        log.info(
            "TradeSetup | %s | dir=%+d | entry=%.3f SL=%.3f TP=%.3f "
            "| lots=%.2f risk=$%.0f R:R=%.2f | vol=%s conf=%.2f",
            trade_id, direction,
            setup.entry_price, setup.stop_loss, setup.take_profit_2,
            setup.lots, setup.risk_usd, setup.rr_ratio,
            vol_regime, confidence,
        )
        return setup

    def record_closed_trade(self, pnl: float):
        self.daily_pnl += pnl
        self.open_trades = max(0, self.open_trades - 1)
        log.info("Trade closed: PnL=%.2f | Daily PnL=%.2f", pnl, self.daily_pnl)
