"""
Vectorised Walk-Forward Backtesting Engine
Simulates each TradeSetup on historical price data and computes a full
suite of institutional performance metrics.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy.setup_detector import TradeSetup
import config

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    setup:        TradeSetup
    entry_price:  float
    entry_time:   object
    exit_price:   float
    exit_time:    object
    pnl_pts:      float
    pnl_pct:      float
    direction:    str
    outcome:      str      # "win_tp1" | "win_tp2" | "win_tp3" | "loss" | "stopped"
    bars_held:    int
    mfe:          float    # max favourable excursion
    mae:          float    # max adverse excursion


@dataclass
class BacktestResult:
    trades:             List[Trade] = field(default_factory=list)
    equity_curve:       pd.Series   = field(default_factory=pd.Series)
    total_return:       float       = 0.0
    cagr:               float       = 0.0
    sharpe_ratio:       float       = 0.0
    sortino_ratio:      float       = 0.0
    calmar_ratio:       float       = 0.0
    max_drawdown:       float       = 0.0
    max_dd_duration:    int         = 0
    win_rate:           float       = 0.0
    profit_factor:      float       = 0.0
    avg_rr:             float       = 0.0
    expectancy:         float       = 0.0
    n_trades:           int         = 0
    n_longs:            int         = 0
    n_shorts:           int         = 0
    avg_bars_held:      float       = 0.0
    best_trade_pct:     float       = 0.0
    worst_trade_pct:    float       = 0.0
    gross_profit:       float       = 0.0
    gross_loss:         float       = 0.0

    def summary(self) -> str:
        separator = "═" * 58
        lines = [
            f"\n{separator}",
            f"  BACKTEST RESULTS  –  {config.ASSET_NAME}",
            f"{separator}",
            f"  Total Return   : {self.total_return*100:>+8.2f}%",
            f"  CAGR           : {self.cagr*100:>+8.2f}%",
            f"  Sharpe Ratio   : {self.sharpe_ratio:>8.3f}",
            f"  Sortino Ratio  : {self.sortino_ratio:>8.3f}",
            f"  Calmar Ratio   : {self.calmar_ratio:>8.3f}",
            f"  Max Drawdown   : {self.max_drawdown*100:>8.2f}%",
            f"  Max DD Days    : {self.max_dd_duration:>8}",
            f"{'─'*58}",
            f"  Trades Total   : {self.n_trades:>8}",
            f"  Longs / Shorts : {self.n_longs} / {self.n_shorts}",
            f"  Win Rate       : {self.win_rate*100:>8.1f}%",
            f"  Profit Factor  : {self.profit_factor:>8.2f}",
            f"  Avg R:R        : {self.avg_rr:>8.2f}",
            f"  Expectancy     : {self.expectancy*100:>+8.2f}% per trade",
            f"  Avg Bars Held  : {self.avg_bars_held:>8.1f}",
            f"  Best Trade     : {self.best_trade_pct*100:>+8.2f}%",
            f"  Worst Trade    : {self.worst_trade_pct*100:>+8.2f}%",
            f"{separator}\n",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Simulate validated setups on OHLCV data.

    For each setup:
      1. Entry = setup.entry (next bar open, approximated as close)
      2. Simulation scans forward bar by bar
      3. Exit at TP1 (50%), TP2 (35%), TP3 (15%) or Stop
      4. Commission + slippage applied
    """

    def __init__(
        self,
        initial_capital: float = config.BACKTEST_INITIAL_CAPITAL,
        commission_pct:  float = config.BACKTEST_COMMISSION_PCT,
        slippage_pct:    float = config.BACKTEST_SLIPPAGE_PCT,
        risk_pct:        float = config.RISK_PER_TRADE_PCT,
    ) -> None:
        self.capital    = initial_capital
        self.commission = commission_pct
        self.slippage   = slippage_pct
        self.risk_pct   = risk_pct

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        setups:    List[TradeSetup],
        price_df:  pd.DataFrame,
        max_bars:  int = 100,
    ) -> BacktestResult:
        """
        Simulate every setup on *price_df* and return aggregated metrics.
        """
        trades = []
        for setup in setups:
            trade = self._simulate_trade(setup, price_df, max_bars)
            if trade is not None:
                trades.append(trade)

        return self._compute_metrics(trades, price_df)

    # ── Trade simulation ──────────────────────────────────────────────────────

    def _simulate_trade(
        self,
        setup:     TradeSetup,
        df:        pd.DataFrame,
        max_bars:  int,
    ) -> Optional[Trade]:
        try:
            start_pos = df.index.get_loc(setup.timestamp)
        except KeyError:
            return None

        if start_pos >= len(df) - 2:
            return None

        entry = setup.entry * (1 + (self.slippage if setup.direction == "long" else -self.slippage))
        stop  = setup.stop
        tp1   = setup.tp1
        tp2   = setup.tp2
        tp3   = setup.tp3
        d     = 1 if setup.direction == "long" else -1

        remaining = 1.0   # fraction of position still open
        pnl_pts   = 0.0
        mfe = mae  = 0.0
        exit_price = entry
        exit_time  = setup.timestamp
        outcome    = "stopped"
        bars_held  = 0

        for i in range(start_pos + 1, min(start_pos + 1 + max_bars, len(df))):
            h   = df["High"].iloc[i]
            lo  = df["Low"].iloc[i]
            bars_held = i - start_pos

            move    = d * (df["Close"].iloc[i] - entry)
            mfe     = max(mfe, move)
            mae     = min(mae, move)

            # Check stop
            stopped = (d == 1 and lo <= stop) or (d == -1 and h >= stop)
            if stopped:
                pnl_pts    += remaining * d * (stop - entry)
                exit_price  = stop
                exit_time   = df.index[i]
                outcome     = "loss"
                break

            # Check TP3
            hit_tp3 = (d == 1 and h >= tp3) or (d == -1 and lo <= tp3)
            if hit_tp3 and remaining > 0.1:
                pnl_pts    += remaining * d * (tp3 - entry)
                exit_price  = tp3
                exit_time   = df.index[i]
                outcome     = "win_tp3"
                remaining   = 0
                break

            # Check TP2
            hit_tp2 = (d == 1 and h >= tp2) or (d == -1 and lo <= tp2)
            if hit_tp2 and remaining >= 0.35:
                pnl_pts    += 0.35 * d * (tp2 - entry)
                remaining  -= 0.35
                exit_price  = tp2
                outcome     = "win_tp2"

            # Check TP1
            hit_tp1 = (d == 1 and h >= tp1) or (d == -1 and lo <= tp1)
            if hit_tp1 and remaining >= 0.50:
                pnl_pts    += 0.50 * d * (tp1 - entry)
                remaining  -= 0.50
                exit_price  = tp1
                if outcome == "stopped":
                    outcome = "win_tp1"

            if remaining < 0.01:
                exit_time = df.index[i]
                break
        else:
            # Time-expired exit
            exit_price = df["Close"].iloc[min(start_pos + max_bars, len(df) - 1)]
            exit_time  = df.index[min(start_pos + max_bars, len(df) - 1)]
            pnl_pts   += remaining * d * (exit_price - entry)
            outcome    = "win_tp1" if pnl_pts > 0 else "loss"

        # Commission
        cost     = (self.commission + self.slippage) * entry * 2
        pnl_pts -= cost
        pnl_pct  = pnl_pts / entry if entry > 0 else 0

        return Trade(
            setup       = setup,
            entry_price = round(entry, 4),
            entry_time  = setup.timestamp,
            exit_price  = round(exit_price, 4),
            exit_time   = exit_time,
            pnl_pts     = round(pnl_pts, 4),
            pnl_pct     = round(pnl_pct, 6),
            direction   = setup.direction,
            outcome     = outcome,
            bars_held   = bars_held,
            mfe         = round(mfe / entry if entry else 0, 6),
            mae         = round(mae / entry if entry else 0, 6),
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(
        self, trades: List[Trade], df: pd.DataFrame
    ) -> BacktestResult:
        if not trades:
            return BacktestResult()

        result = BacktestResult()
        result.trades   = trades
        result.n_trades = len(trades)
        result.n_longs  = sum(1 for t in trades if t.direction == "long")
        result.n_shorts = result.n_trades - result.n_longs

        pnl_pcts = np.array([t.pnl_pct for t in trades])

        winners  = pnl_pcts[pnl_pcts > 0]
        losers   = pnl_pcts[pnl_pcts <= 0]

        result.win_rate       = len(winners) / len(pnl_pcts)
        result.gross_profit   = float(winners.sum()) if len(winners) else 0.0
        result.gross_loss     = float(abs(losers.sum())) if len(losers) else 0.0
        result.profit_factor  = (
            result.gross_profit / result.gross_loss
            if result.gross_loss > 0 else float("inf")
        )
        result.best_trade_pct  = float(pnl_pcts.max()) if len(pnl_pcts) else 0
        result.worst_trade_pct = float(pnl_pcts.min()) if len(pnl_pcts) else 0
        result.avg_bars_held   = float(np.mean([t.bars_held for t in trades]))

        # R:R average
        rrs = [t.setup.rr1 for t in trades if t.outcome.startswith("win")]
        result.avg_rr = float(np.mean(rrs)) if rrs else 0.0

        # Expectancy
        result.expectancy = (
            result.win_rate * float(np.mean(winners)) if len(winners) else 0
        ) + (
            (1 - result.win_rate) * float(np.mean(losers)) if len(losers) else 0
        )

        # Equity curve
        equity        = np.cumprod(1 + pnl_pcts) * self.capital
        result.equity_curve = pd.Series(
            equity,
            index=[t.exit_time for t in trades],
        )

        result.total_return = float(equity[-1] / self.capital - 1)

        # CAGR
        n_days = max(
            (trades[-1].exit_time - trades[0].entry_time).days, 1
        ) if hasattr(trades[-1].exit_time, "days") else 365
        result.cagr = (1 + result.total_return) ** (365 / n_days) - 1

        # Drawdown
        roll_max       = np.maximum.accumulate(equity)
        dd             = (equity - roll_max) / roll_max
        result.max_drawdown = float(dd.min())

        # Max DD duration (bars)
        in_dd   = dd < 0
        changes = np.diff(in_dd.astype(int), prepend=0)
        starts  = np.where(changes == 1)[0]
        ends    = np.where(changes == -1)[0]
        if len(starts) and len(ends):
            durations = [
                e - s for s, e in zip(starts, ends[:len(starts)])
            ]
            result.max_dd_duration = int(max(durations)) if durations else 0

        # Sharpe (annualised, using 1H bars)
        ann_factor = np.sqrt(config.TIMEFRAMES[config.PRIMARY_TF]["bars_per_day"] * 252)
        if pnl_pcts.std() > 0:
            result.sharpe_ratio = float(pnl_pcts.mean() / pnl_pcts.std() * ann_factor)

        # Sortino (downside deviation)
        neg       = pnl_pcts[pnl_pcts < 0]
        down_std  = neg.std() if len(neg) > 1 else 1e-9
        result.sortino_ratio = float(pnl_pcts.mean() / down_std * ann_factor)

        # Calmar
        result.calmar_ratio = (
            abs(result.cagr / result.max_drawdown)
            if result.max_drawdown != 0 else float("inf")
        )

        logger.info("Backtest complete: %s", result.summary())
        return result
