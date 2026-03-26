"""
Trade Execution Engine.
Handles signal → setup → execution → monitoring lifecycle.
In live mode: logs signals + sends alerts.
In paper/backtest mode: simulates fills.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from strategy.risk_manager import RiskManager, TradeSetup
from config.settings import RISK, RESULTS_DIR
from utils.logger import get_logger

log = get_logger("TradeExecutor")

RESULTS_PATH = Path(RESULTS_DIR)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


class TradeExecutor:
    """
    Execution layer for the ML ensemble trading system.
    Modes: 'paper' (default) | 'live' (alert only, no auto-execution)
    """

    def __init__(self, mode: str = "paper"):
        self.mode          = mode
        self.risk_manager  = RiskManager()
        self.open_trades:  list[TradeSetup] = []
        self.closed_trades: list[dict]      = []
        self._trade_log_path = RESULTS_PATH / f"trades_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

    # ─── EXECUTE SIGNAL ────────────────────────────────────────────────────────

    def execute_signal(
        self,
        meta_result:  dict,
        garch_result: dict,
        features_row: pd.Series,
        current_price: float,
        atr_value:     float,
    ) -> Optional[TradeSetup]:
        """
        Convert a filtered meta-model signal into a concrete trade setup.
        In 'paper' mode: simulates immediate fill at current_price.
        In 'live'  mode: logs the setup and raises an alert.
        """
        direction = meta_result.get("direction", 0)
        if direction == 0:
            log.debug("No actionable signal from meta-filter.")
            return None

        setup = self.risk_manager.build_trade_setup(
            direction    = direction,
            entry        = current_price,
            atr_value    = atr_value,
            signal_dict  = {},
            garch_dict   = garch_result,
            meta_result  = meta_result,
        )

        if setup is None:
            return None

        if self.mode == "paper":
            self._paper_fill(setup)
        elif self.mode == "live":
            self._live_alert(setup)

        self._log_trade(setup)
        return setup

    # ─── PAPER FILL ────────────────────────────────────────────────────────────

    def _paper_fill(self, setup: TradeSetup):
        setup.status = "active"
        self.open_trades.append(setup)
        self.risk_manager.open_trades += 1
        log.info("[PAPER] Trade FILLED: %s", setup.trade_id)

    # ─── LIVE ALERT ────────────────────────────────────────────────────────────

    def _live_alert(self, setup: TradeSetup):
        """Print a structured alert (extend to Telegram/Discord/webhook as needed)."""
        dir_str = "LONG ▲" if setup.direction == 1 else "SHORT ▼"
        alert = (
            f"\n{'═'*60}\n"
            f"  🔔 XAUUSD TRADE ALERT  |  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"{'═'*60}\n"
            f"  Direction  : {dir_str}\n"
            f"  Entry      : {setup.entry_price:.3f}\n"
            f"  Stop Loss  : {setup.stop_loss:.3f}\n"
            f"  TP1 (50%)  : {setup.take_profit_1:.3f}\n"
            f"  TP2 (full) : {setup.take_profit_2:.3f}\n"
            f"  Lots       : {setup.lots:.2f}\n"
            f"  Risk $     : ${setup.risk_usd:.2f}\n"
            f"  Reward $   : ${setup.potential_reward:.2f}\n"
            f"  R:R        : {setup.rr_ratio:.2f}\n"
            f"  Vol Regime : {setup.vol_regime}\n"
            f"  Confidence : {setup.confluence_score:.2%}\n"
            f"  Confluences: {', '.join(setup.confluences)}\n"
            f"  ATR(14)    : {setup.atr_14:.3f}\n"
            f"  Trade ID   : {setup.trade_id}\n"
            f"{'═'*60}"
        )
        print(alert)
        log.info("[LIVE ALERT] %s", setup.trade_id)

    # ─── MARK-TO-MARKET ────────────────────────────────────────────────────────

    def update_positions(self, current_price: float):
        """Check open positions against current price; close if TP/SL hit."""
        still_open = []
        for trade in self.open_trades:
            if trade.direction == 1:  # LONG
                if current_price >= trade.take_profit_2:
                    pnl = ((trade.take_profit_2 - trade.entry_price) *
                           trade.lots * self.risk_manager.cfg["lot_size"])
                    self._close_trade(trade, "TP2", pnl, current_price)
                elif current_price <= trade.stop_loss:
                    pnl = ((trade.stop_loss - trade.entry_price) *
                           trade.lots * self.risk_manager.cfg["lot_size"])
                    self._close_trade(trade, "SL", pnl, current_price)
                else:
                    still_open.append(trade)
            else:  # SHORT
                if current_price <= trade.take_profit_2:
                    pnl = ((trade.entry_price - trade.take_profit_2) *
                           trade.lots * self.risk_manager.cfg["lot_size"])
                    self._close_trade(trade, "TP2", pnl, current_price)
                elif current_price >= trade.stop_loss:
                    pnl = ((trade.entry_price - trade.stop_loss) *
                           trade.lots * self.risk_manager.cfg["lot_size"])
                    self._close_trade(trade, "SL", pnl, current_price)
                else:
                    still_open.append(trade)
        self.open_trades = still_open

    def _close_trade(self, trade: TradeSetup, reason: str, pnl: float,
                     close_price: float):
        trade.status = "closed"
        record = {**trade.to_dict(),
                  "close_price": close_price,
                  "close_reason": reason,
                  "pnl_usd": round(pnl, 2),
                  "closed_at": datetime.utcnow().isoformat() + "Z"}
        self.closed_trades.append(record)
        self.risk_manager.record_closed_trade(pnl)
        self._log_trade_close(record)

    # ─── LOGGING ───────────────────────────────────────────────────────────────

    def _log_trade(self, setup: TradeSetup):
        with open(self._trade_log_path, "a") as f:
            f.write(json.dumps(setup.to_dict()) + "\n")

    def _log_trade_close(self, record: dict):
        close_log = RESULTS_PATH / f"closed_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(close_log, "a") as f:
            f.write(json.dumps(record) + "\n")
        emoji = "✅" if record["pnl_usd"] > 0 else "❌"
        log.info("%s Trade closed | %s | reason=%s | PnL=$%.2f",
                 emoji, record["trade_id"], record["close_reason"], record["pnl_usd"])

    # ─── PERFORMANCE SUMMARY ───────────────────────────────────────────────────

    def performance_summary(self) -> dict:
        if not self.closed_trades:
            return {"note": "no_closed_trades"}
        pnls  = [t["pnl_usd"] for t in self.closed_trades]
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p < 0]
        total = len(pnls)
        return {
            "total_trades": total,
            "win_rate":     len(wins) / total if total else 0,
            "total_pnl":    sum(pnls),
            "avg_win":      np.mean(wins) if wins else 0,
            "avg_loss":     np.mean(losses) if losses else 0,
            "profit_factor":abs(sum(wins)) / max(abs(sum(losses)), 1e-9),
            "max_drawdown": min(np.minimum.accumulate(np.cumsum(pnls))),
            "sharpe":       (np.mean(pnls) / (np.std(pnls) + 1e-9)) * np.sqrt(252),
        }
