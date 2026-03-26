"""
XAUUSD Institutional-Grade ML Ensemble Trading System
──────────────────────────────────────────────────────
Entry point: trains all models, runs MTF analysis,
applies meta-filter, and outputs high-probability trade setups.

Usage:
  python main.py                  # full run: train + live analysis
  python main.py --mode paper     # paper trading mode
  python main.py --mode live      # live alert mode (no auto-execution)
  python main.py --skip-train     # skip retraining, use saved models
  python main.py --fast           # skip volume profile (faster)
"""
from __future__ import annotations

import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Core modules ─────────────────────────────────────────────────────────────
from data.fetcher        import fetch_all_timeframes, get_current_price
from features.builder    import build_all_features
from features.technical  import atr
from models.xgboost_model import XGBoostDirectionalModel
from models.lstm_model    import LSTMDirectionalModel
from models.garch_model   import GARCHVolatilityModel
from strategy.meta_filter import MetaModelFilter
from analysis.mtf_analyzer import MTFAnalyzer
from strategy.risk_manager import RiskManager
from execution.trade_executor import TradeExecutor
from config.settings import SYMBOL_DISPLAY, RESULTS_DIR
from utils.logger import get_logger

log = get_logger("Main")

RESULTS_PATH = Path(RESULTS_DIR)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# ─── BANNER ────────────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        XAUUSD  ML ENSEMBLE TRADING SYSTEM  v1.0             ║
║   XGBoost + LSTM + GARCH/ARCH  │  SMC Meta-Filter           ║
║   Multi-Timeframe: Daily + 4H + 1H                          ║
╚══════════════════════════════════════════════════════════════╝
""")


# ─── TRAIN ALL MODELS ──────────────────────────────────────────────────────────

def train_models(
    data:          dict,
    fast_features: bool = False,
    save:          bool = True,
) -> tuple[XGBoostDirectionalModel, LSTMDirectionalModel, GARCHVolatilityModel]:
    """Train XGBoost, LSTM, and GARCH on the 1H dataset."""
    log.info("=" * 60)
    log.info("TRAINING PHASE")
    log.info("=" * 60)

    # Use 1H as primary training timeframe (most data + best granularity)
    df_train = data.get("1H") or data.get("4H") or data.get("1D")
    if df_train is None or len(df_train) < 200:
        raise RuntimeError("Insufficient data for training")

    log.info("Training on %s bars of 1H data", len(df_train))

    # ── Feature matrix ────────────────────────────────────────────────────────
    log.info("Building feature matrix…")
    features = build_all_features(df_train, include_volume_profile=not fast_features)

    # ── GARCH ─────────────────────────────────────────────────────────────────
    log.info("Fitting GARCH model…")
    garch = GARCHVolatilityModel()
    garch.fit(df_train)

    # Append GARCH vol column to features
    features = garch.add_garch_features(features)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    log.info("Training XGBoost…")
    xgb = XGBoostDirectionalModel()
    xgb_metrics = xgb.fit(features)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    log.info("Training LSTM…")
    lstm = LSTMDirectionalModel()
    lstm_metrics = lstm.fit(features)

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        Path("models/saved").mkdir(parents=True, exist_ok=True)
        xgb.save()
        lstm.save()

    log.info("Training complete | XGB AUC=%.3f | LSTM AUC=%.3f",
             xgb_metrics.get("auc", 0), lstm_metrics.get("auc", 0))

    return xgb, lstm, garch


# ─── LIVE ANALYSIS ─────────────────────────────────────────────────────────────

def run_analysis(
    data:         dict,
    xgb:          XGBoostDirectionalModel,
    lstm:         LSTMDirectionalModel,
    garch:        GARCHVolatilityModel,
    executor:     TradeExecutor,
    fast_features: bool = False,
) -> dict:
    """
    Run the full analysis pipeline on current market data.
    Returns the complete signal report.
    """
    log.info("=" * 60)
    log.info("ANALYSIS PHASE — %s UTC", datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # ── MTF Analysis ─────────────────────────────────────────────────────────
    analyzer = MTFAnalyzer()
    mtf      = analyzer.analyze(data)
    analyzer.print_report()

    # ── Current Price & ATR ───────────────────────────────────────────────────
    current_price = mtf.get("current_price", 0)
    atr_1h = mtf["atr_summary"].get("1H", mtf["atr_summary"].get("4H", 5.0))
    atr_1d = mtf["atr_summary"].get("1D", atr_1h * 4)

    log.info("Current price: %.3f | ATR(1H)=%.3f | ATR(1D)=%.3f",
             current_price, atr_1h, atr_1d)

    # ── Build entry-TF feature matrix ─────────────────────────────────────────
    entry_df = data.get("1H") or data.get("4H")
    if entry_df is None:
        log.error("No 1H/4H data available for signal generation")
        return {}

    features = build_all_features(entry_df, include_volume_profile=not fast_features)
    features = garch.add_garch_features(features)

    if len(features) < 5:
        log.error("Feature matrix too short (%d rows)", len(features))
        return {}

    # ── Model Predictions ─────────────────────────────────────────────────────
    log.info("Running model predictions…")

    xgb_pred  = xgb.predict_latest(features)
    lstm_pred = lstm.predict_latest(features)
    garch_fc  = garch.forecast_volatility(entry_df)

    log.info("XGB  | prob_long=%.3f signal=%+d",
             xgb_pred["xgb_prob_long"], xgb_pred["xgb_signal"])
    log.info("LSTM | prob_long=%.3f signal=%+d",
             lstm_pred["lstm_prob_long"], lstm_pred["lstm_signal"])
    log.info("GARCH| cond_vol=%.4f regime=%s ann_vol=%.2f%%",
             garch_fc["garch_cond_vol"], garch_fc["garch_regime"],
             garch_fc["garch_ann_vol"] * 100)

    # ── Latest feature row for filter ─────────────────────────────────────────
    latest_row  = features.iloc[-1]
    tf_signals  = analyzer.get_tf_signals()

    # ── Meta-Model Filter ─────────────────────────────────────────────────────
    meta_filter = MetaModelFilter()
    meta_result = meta_filter.evaluate(
        xgb_pred     = xgb_pred,
        lstm_pred    = lstm_pred,
        garch_fc     = garch_fc,
        features_row = latest_row,
        tf_signals   = tf_signals,
        timestamp    = features.index[-1],
    )

    log.info("MetaFilter | signal=%d conf=%.3f filters_passed=%s",
             meta_result["signal"], meta_result["confidence"],
             meta_result.get("filters_passed", []))

    if meta_result.get("filters_failed"):
        log.info("         | filters_failed=%s", meta_result["filters_failed"])

    # ── Trade Setup ───────────────────────────────────────────────────────────
    setup = None
    if meta_result["direction"] != 0:
        setup = executor.execute_signal(
            meta_result   = meta_result,
            garch_result  = garch_fc,
            features_row  = latest_row,
            current_price = current_price,
            atr_value     = atr_1h,
        )

    # ── Compile Full Report ───────────────────────────────────────────────────
    report = {
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "symbol":         SYMBOL_DISPLAY,
        "current_price":  current_price,
        "mtf_analysis":   {
            "dominant_dir":  mtf["dominant_dir"],
            "mtf_score":     mtf["mtf_score"],
            "tf_signals":    mtf["tf_signals"],
            "key_levels":    {k: v["level"] for k, v in mtf["key_levels"].items()},
            "atr_summary":   mtf["atr_summary"],
        },
        "model_outputs":  {
            "xgboost": xgb_pred,
            "lstm":    lstm_pred,
            "garch":   garch_fc,
        },
        "meta_filter":    meta_result,
        "trade_setup":    setup.to_dict() if setup else None,
        "filter_stats":   meta_filter.get_stats(),
    }

    # Save report
    report_path = RESULTS_PATH / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved → %s", report_path)

    # ── Print Trade Setup ─────────────────────────────────────────────────────
    _print_final_signal(meta_result, setup, garch_fc, mtf)

    return report


# ─── PRINT FINAL SIGNAL ────────────────────────────────────────────────────────

def _print_final_signal(meta_result, setup, garch_fc, mtf):
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  FINAL SIGNAL OUTPUT  │  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(sep)

    direction = meta_result.get("direction", 0)
    if direction == 0:
        print("  SIGNAL: ── NO TRADE ──")
        failed = meta_result.get("filters_failed", [])
        if failed:
            print(f"  Blocked by: {', '.join(failed)}")
    else:
        dir_str = "LONG  ▲" if direction == 1 else "SHORT ▼"
        conf    = meta_result.get("confidence", 0)
        ens_p   = meta_result.get("ensemble_prob", 0.5)

        print(f"  SIGNAL:     ✅ {dir_str}")
        print(f"  Confidence: {conf:.1%}")
        print(f"  Ensemble P: {ens_p:.3f}")
        print(f"  MTF Score:  {mtf.get('mtf_score', 0):+.3f}")
        print(f"  Confluences ({len(meta_result.get('confluences', []))}):")
        for c in meta_result.get("confluences", []):
            print(f"    ↳ {c}")
        print(f"  Vol Regime: {garch_fc.get('garch_regime')} "
              f"(ann={garch_fc.get('garch_ann_vol', 0)*100:.1f}%)")

        if setup:
            print()
            print(f"  ┌─ TRADE PARAMETERS ───────────────────────────────┐")
            print(f"  │ Entry      : {setup.entry_price:>10.3f}                       │")
            print(f"  │ Stop Loss  : {setup.stop_loss:>10.3f}  ({abs(setup.entry_price - setup.stop_loss):.3f} pts)  │")
            print(f"  │ TP1 (50%) : {setup.take_profit_1:>10.3f}                       │")
            print(f"  │ TP2 (100%): {setup.take_profit_2:>10.3f}  R:R {setup.rr_ratio:.2f}           │")
            print(f"  │ Lots       : {setup.lots:>10.2f}                       │")
            print(f"  │ Risk $     : ${setup.risk_usd:>9.2f}                       │")
            print(f"  │ Reward $   : ${setup.potential_reward:>9.2f}                       │")
            print(f"  │ Trade ID   : {setup.trade_id:<35}│")
            print(f"  └──────────────────────────────────────────────────┘")

    print(f"{sep}\n")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XAUUSD ML Ensemble Trading System")
    parser.add_argument("--mode",       default="paper", choices=["paper", "live"])
    parser.add_argument("--skip-train", action="store_true",
                        help="Load saved models instead of retraining")
    parser.add_argument("--fast",       action="store_true",
                        help="Skip volume profile features (faster)")
    args = parser.parse_args()

    print_banner()
    log.info("Starting XAUUSD ML Ensemble System | mode=%s | fast=%s",
             args.mode, args.fast)

    # ── Fetch Data ────────────────────────────────────────────────────────────
    log.info("Fetching XAUUSD multi-timeframe data…")
    data = fetch_all_timeframes()

    if not data:
        log.error("Failed to fetch any data. Exiting.")
        sys.exit(1)

    available = list(data.keys())
    log.info("Data fetched: %s", available)

    # ── Train or Load Models ──────────────────────────────────────────────────
    xgb_path  = Path("models/saved/xgboost_xauusd.pkl")
    lstm_path = Path("models/saved/lstm_xauusd_meta.pkl")

    if args.skip_train and xgb_path.exists() and lstm_path.exists():
        log.info("Loading saved models…")
        xgb  = XGBoostDirectionalModel()
        lstm = LSTMDirectionalModel()
        garch= GARCHVolatilityModel()
        xgb.load()
        lstm.load()
        # GARCH always refitted on fresh data
        df_g = data.get("1H") or data.get("4H") or data.get("1D")
        if df_g is not None:
            garch.fit(df_g)
    else:
        xgb, lstm, garch = train_models(data, fast_features=args.fast)

    # ── Executor ──────────────────────────────────────────────────────────────
    executor = TradeExecutor(mode=args.mode)

    # ── Analysis ──────────────────────────────────────────────────────────────
    report = run_analysis(
        data          = data,
        xgb           = xgb,
        lstm          = lstm,
        garch         = garch,
        executor      = executor,
        fast_features = args.fast,
    )

    # ── Performance Summary ───────────────────────────────────────────────────
    perf = executor.performance_summary()
    if "note" not in perf:
        log.info("Performance: %s", json.dumps(perf, indent=2))

    log.info("System run complete.")
    return report


if __name__ == "__main__":
    main()
