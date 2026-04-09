"""
XAUUSD Institutional ML Trading System – Main Entry Point
══════════════════════════════════════════════════════════
Orchestrates the full pipeline:
  1. Fetch multi-timeframe OHLCV (yfinance)
  2. Engineer technical + market-structure + MTF-confluence features
  3. Train (or load) XGBoost / LSTM / GARCH models
  4. Run hybrid ensemble to score every bar
  5. Detect & validate high-probability setups
  6. Compute risk/position plans
  7. Backtest on held-out data
  8. Render TradingView-style interactive chart + HTML report

Usage:
  python main.py                  # full pipeline (auto-train if no saved models)
  python main.py --retrain        # force model retraining
  python main.py --no-backtest    # skip backtesting step
  python main.py --bars 500       # plot last N bars
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Local imports ──────────────────────────────────────────────────────────────
import config
from data.fetcher      import DataFetcher
from data.preprocessor import DataPreprocessor
from features.technical        import TechnicalFeatureEngineer
from features.market_structure import MarketStructureAnalyzer
from features.mtf_confluence   import MTFConfluenceCalculator
from models.xgboost_model  import XGBoostSetupClassifier
from models.lstm_model     import LSTMForecaster
from models.garch_model    import GARCHVolatilityModel
from ensemble.hybrid_ensemble  import HybridEnsemble
from strategy.setup_detector   import SetupDetector
from strategy.validator        import SetupValidator
from strategy.risk_manager     import RiskManager
from backtest.engine           import BacktestEngine
from visualization.charts      import ChartBuilder

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Colours (ANSI) ────────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


# ══════════════════════════════════════════════════════════════════════════════
#  Banner
# ══════════════════════════════════════════════════════════════════════════════

BANNER = f"""
{CYAN}{BOLD}
  ╔══════════════════════════════════════════════════════════════╗
  ║   XAUUSD  INSTITUTIONAL ML  TRADING SYSTEM                  ║
  ║   XGBoost + BiLSTM-Attention + ARCH/GARCH  Ensemble         ║
  ║   Multi-Timeframe: Daily · 4H · 1H                          ║
  ╚══════════════════════════════════════════════════════════════╝
{RESET}"""


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TradingPipeline:
    """End-to-end analysis pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> None:
        print(BANNER)
        t0 = time.time()

        # ── 1. Fetch data ─────────────────────────────────────────────────────
        self._section("1 / 8  DATA FETCH")
        raw = self._fetch_data()

        # ── 2. Preprocess & align ─────────────────────────────────────────────
        self._section("2 / 8  PREPROCESSING")
        processed, aligned = self._preprocess(raw)

        # ── 3. Feature engineering ────────────────────────────────────────────
        self._section("3 / 8  FEATURE ENGINEERING")
        featured, structure_1h = self._engineer_features(processed, aligned)

        # ── 4. Train or load models ───────────────────────────────────────────
        self._section("4 / 8  MODELS")
        xgb, lstm, garch, vol_fc = self._setup_models(featured, raw)

        # ── 5. Ensemble scoring ───────────────────────────────────────────────
        self._section("5 / 8  ENSEMBLE SCORING")
        ensemble  = HybridEnsemble(xgb, lstm, garch)
        signals   = ensemble.score_all(featured, vol_forecast=vol_fc)
        ens_df    = ensemble.to_dataframe(signals)

        # ── 6. Setup detection & validation ──────────────────────────────────
        self._section("6 / 8  SETUP DETECTION")
        detector   = SetupDetector(timeframe_label="1H")
        validator  = SetupValidator()
        raw_setups = detector.detect(featured, signals)
        valid_setups, rejected = validator.validate(raw_setups)

        print(f"\n  {BOLD}Detected {len(raw_setups)} candidate setups "
              f"→ {GREEN}{len(valid_setups)} validated{RESET}\n")

        # Risk plans for validated setups
        rm = RiskManager(account_balance=config.BACKTEST_INITIAL_CAPITAL)
        for setup in valid_setups[-5:]:   # show last 5
            plan = rm.plan(setup)
            print(setup.summary())
            print(plan.summary())

        # ── 7. Backtest ───────────────────────────────────────────────────────
        bt_result = None
        if not self.args.no_backtest and len(valid_setups) >= 3:
            self._section("7 / 8  BACKTEST")
            engine    = BacktestEngine()
            bt_result = engine.run(valid_setups, raw[config.PRIMARY_TF])
            print(bt_result.summary())
        else:
            self._section("7 / 8  BACKTEST  (skipped)")

        # ── 8. Chart & report ─────────────────────────────────────────────────
        self._section("8 / 8  CHART & REPORT")
        chart_path = self._render_chart(
            featured, valid_setups, structure_1h, ens_df
        )

        elapsed = time.time() - t0
        print(
            f"\n{GREEN}{BOLD}  Pipeline complete in {elapsed:.1f}s{RESET}\n"
            f"  Chart  → {CYAN}{chart_path}{RESET}\n"
        )

        if bt_result and bt_result.n_trades > 0:
            self._print_latest_setup_summary(valid_setups, featured)

    # ── Step implementations ──────────────────────────────────────────────────

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        fetcher = DataFetcher(
            ticker=config.TICKER,
            cache_ttl_hours=4,
        )
        raw = fetcher.fetch_all(force_refresh=self.args.retrain)
        for tf, df in raw.items():
            print(
                f"  {CYAN}{config.TIMEFRAMES[tf]['name']:8s}{RESET}  "
                f"{len(df):>5d} bars  "
                f"[{df.index[0].date()} → {df.index[-1].date()}]"
            )
        return raw

    @staticmethod
    def _preprocess(
        raw: Dict[str, pd.DataFrame]
    ):
        pp   = DataPreprocessor()
        proc = pp.process(raw)
        aligned = pp.build_aligned_frame(proc)
        return proc, aligned

    @staticmethod
    def _engineer_features(
        processed: Dict[str, pd.DataFrame],
        aligned:   pd.DataFrame,
    ):
        te  = TechnicalFeatureEngineer()
        msa = MarketStructureAnalyzer()
        mtf = MTFConfluenceCalculator()

        # Add technical indicators to primary (1H) processed data
        primary_tech = te.compute(processed[config.PRIMARY_TF])

        # Market structure on primary TF
        structure_1h = msa.analyze(primary_tech)
        primary_ms   = msa.add_features(primary_tech, structure_1h)

        # Also add tech indicators to 4H and daily for MTF column population
        for tf in [config.INTERMEDIATE_TF, config.ANCHOR_TF]:
            processed[tf] = te.compute(processed[tf])

        # Re-build aligned frame now that processed TFs have indicator cols
        pp      = DataPreprocessor()
        aligned = pp.build_aligned_frame(
            {tf: te.compute(processed[tf]) if tf != config.PRIMARY_TF else primary_ms
             for tf in config.TIMEFRAMES}
        )

        # MTF confluence on aligned frame
        featured = mtf.compute(aligned)

        # Carry forward GARCH conditional vol feature (filled later)
        featured["garch_cond_vol"] = 0.01
        featured["garch_regime"]   = 1

        n_feats = len([c for c in featured.columns if c not in
                       ("Open", "High", "Low", "Close", "Volume")])
        print(f"  {n_feats} engineered features on {len(featured)} primary bars")
        return featured, structure_1h

    def _setup_models(
        self,
        featured:  pd.DataFrame,
        raw:       Dict[str, pd.DataFrame],
    ):
        xgb   = XGBoostSetupClassifier()
        lstm  = LSTMForecaster()
        garch = GARCHVolatilityModel()

        xgb_path  = os.path.join(config.MODELS_DIR, xgb.MODEL_FNAME)
        lstm_path = os.path.join(config.MODELS_DIR, lstm.MODEL_FNAME)
        garch_path = os.path.join(config.MODELS_DIR, garch.MODEL_FNAME)

        # ── GARCH (fits on daily data) ────────────────────────────────────────
        if self.args.retrain or not os.path.exists(garch_path):
            print(f"  {YELLOW}Training GARCH on daily data …{RESET}")
            garch_metrics = garch.fit(raw[config.ANCHOR_TF])
            print(
                f"  GARCH AIC={garch_metrics.get('garch_aic', 'N/A'):.1f}  "
                f"GJR AIC={garch_metrics.get('gjr_aic', 'N/A'):.1f}"
                if garch_metrics.get('garch_aic') else "  GARCH fitted"
            )
        else:
            print(f"  {GREEN}Loading saved GARCH model{RESET}")
            garch.load()

        vol_forecast = garch.forecast()
        print(
            f"  Volatility regime: {BOLD}{vol_forecast['regime'].upper()}{RESET}  "
            f"(1-day cond.vol = {vol_forecast['cond_vol_pct']:.3f}%)"
        )

        # Add GARCH conditional vol series to featured DataFrame
        featured_with_garch = garch.add_features(featured, raw[config.ANCHOR_TF])

        # Build XGBoost target labels
        target = xgb.build_target(featured_with_garch)
        featured_with_garch["target"] = target

        # ── XGBoost ───────────────────────────────────────────────────────────
        if self.args.retrain or not os.path.exists(xgb_path):
            print(f"  {YELLOW}Training XGBoost ({config.XGB_PARAMS['n_estimators']} trees) …{RESET}")
            xgb_metrics = xgb.fit(featured_with_garch)
            print(
                f"  XGBoost CV accuracy: {xgb_metrics['mean_cv_acc']:.3f}"
            )
        else:
            print(f"  {GREEN}Loading saved XGBoost model{RESET}")
            xgb.load()

        # ── LSTM ──────────────────────────────────────────────────────────────
        if self.args.retrain or not os.path.exists(lstm_path):
            print(
                f"  {YELLOW}Training LSTM (epochs={config.LSTM_EPOCHS}, "
                f"seq={config.LSTM_SEQUENCE_LENGTH}) …{RESET}"
            )
            lstm_metrics = lstm.fit(featured_with_garch)
            best_val = lstm_metrics.get('best_val_loss', float('nan'))
            print(f"  LSTM best val loss: {best_val:.4f}")
        else:
            print(f"  {GREEN}Loading saved LSTM model{RESET}")
            lstm.load()

        # Store featured + garch back in featured
        featured.update(featured_with_garch[["garch_cond_vol", "garch_regime", "target"]])

        return xgb, lstm, garch, vol_forecast

    @staticmethod
    def _render_chart(
        featured:    pd.DataFrame,
        setups:      list,
        structure:   object,
        ens_df:      pd.DataFrame,
    ) -> str:
        cb    = ChartBuilder()
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"xauusd_analysis_{ts}.html"
        fig   = cb.build(
            df          = featured,
            setups      = setups,
            structure   = structure,
            ensemble_df = ens_df,
            last_n_bars = 400,
            title       = f"XAUUSD – ML Ensemble  ({ts})",
        )
        return cb.save(fig, fname)

    @staticmethod
    def _print_latest_setup_summary(setups: list, df: pd.DataFrame) -> None:
        if not setups:
            return
        latest = setups[-1]
        print(f"\n{BOLD}  ── LATEST HIGH-PROBABILITY SETUP ──{RESET}")
        print(latest.summary())

    @staticmethod
    def _section(label: str) -> None:
        print(f"\n{BOLD}{CYAN}  ◆  {label}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="XAUUSD Institutional ML Trading System"
    )
    p.add_argument("--retrain",      action="store_true",
                   help="Force model retraining")
    p.add_argument("--no-backtest",  action="store_true",
                   help="Skip backtesting step")
    p.add_argument("--bars",         type=int, default=400,
                   help="Number of bars to display in chart")
    p.add_argument("--ticker",       type=str, default=config.TICKER,
                   help="yfinance ticker (default: GC=F)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ticker != config.TICKER:
        config.TICKER = args.ticker

    pipeline = TradingPipeline(args)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  Interrupted by user.{RESET}")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        sys.exit(1)
