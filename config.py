"""
XAUUSD Institutional-Grade ML Trading System – Central Configuration
All tuneable parameters live here; nothing is hard-coded elsewhere.
"""
import os

# ─── Asset ────────────────────────────────────────────────────────────────────
TICKER      = "GC=F"       # Gold Futures Continuous (yfinance proxy for XAUUSD)
ASSET_NAME  = "XAUUSD"
POINT_VALUE = 100          # USD per point move (standard gold futures)

# ─── Timeframes ───────────────────────────────────────────────────────────────
TIMEFRAMES = {
    "1d": {"name": "Daily",  "period": "5y",   "bars_per_day": 1},
    "4h": {"name": "4-Hour", "period": "730d",  "bars_per_day": 6},
    "1h": {"name": "1-Hour", "period": "730d",  "bars_per_day": 24},
}

PRIMARY_TF      = "1h"   # Signal generation timeframe
INTERMEDIATE_TF = "4h"   # Trend confirmation
ANCHOR_TF       = "1d"   # Macro trend anchor

# ─── Technical Indicator Periods ─────────────────────────────────────────────
ATR_PERIOD     = 14
RSI_PERIOD     = 14
RSI_FAST       = 7
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
BB_PERIOD      = 20
BB_STD         = 2.0
ADX_PERIOD     = 14
STOCH_K        = 14
STOCH_D        = 3
CCI_PERIOD     = 20
MFI_PERIOD     = 14
WILLIAMS_R     = 14
EMA_PERIODS    = [9, 20, 50, 200]
SMA_PERIODS    = [20, 50, 200]

# ─── Market Structure ─────────────────────────────────────────────────────────
SWING_LOOKBACK        = 5      # Bars each side for pivot detection
SR_CLUSTER_TOLERANCE  = 0.003  # 0.3% price tolerance for S/R clustering
MAX_SR_LEVELS         = 6      # Max levels to track per side
FVG_MIN_ATR_SIZE      = 0.4    # Minimum FVG size in ATR units
ORDER_BLOCK_LOOKBACK  = 50

# ─── ML Sequence / Label ─────────────────────────────────────────────────────
LSTM_SEQUENCE_LENGTH   = 60   # Input window (bars)
TARGET_FORWARD_BARS    = 5    # Forward bars for signal labelling
TARGET_ATR_MULTIPLIER  = 1.5  # ATR threshold to classify long/short

# ─── LSTM Hyper-parameters ───────────────────────────────────────────────────
LSTM_HIDDEN_SIZE   = 128
LSTM_NUM_LAYERS    = 2
LSTM_DROPOUT       = 0.3
LSTM_NUM_HEADS     = 4
LSTM_EPOCHS        = 100
LSTM_BATCH_SIZE    = 64
LSTM_LR            = 1e-3
LSTM_WEIGHT_DECAY  = 1e-4
LSTM_PATIENCE      = 15

# ─── XGBoost Hyper-parameters ────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
    "verbosity":        0,
}
XGB_EARLY_STOPPING = 50
XGB_CV_FOLDS       = 5

# ─── ARCH/GARCH Parameters ───────────────────────────────────────────────────
GARCH_P       = 1
GARCH_Q       = 1
GARCH_VOL     = "Garch"
GARCH_DIST    = "skewt"
GARCH_HORIZON = 5

# Annualised daily-return vol thresholds → regime labels
VOL_LOW_THRESHOLD  = 0.008   # < 0.8%  → "low"
VOL_HIGH_THRESHOLD = 0.018   # > 1.8%  → "high"

# ─── Ensemble ─────────────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {"xgboost": 0.40, "lstm": 0.40, "garch": 0.20}

VOL_REGIME_MULTIPLIER = {"low": 1.00, "medium": 0.80, "high": 0.55}

# ─── Setup Detection ─────────────────────────────────────────────────────────
SETUP_SCORE_THRESHOLD  = 0.65   # Minimum score (0–1) to flag a setup
MTF_ALIGNMENT_REQUIRED = True   # All 3 TFs must agree
MIN_RR_RATIO           = 1.5

# ─── Risk Management ─────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT    = 0.01    # 1% account risk per trade
MAX_STOP_PCT          = 0.015   # Hard cap: stop ≤ 1.5% of entry price
KELLY_FRACTION        = 0.25    # Conservative fractional Kelly

ATR_STOP_MULTIPLIER   = 1.5
ATR_TP1_MULTIPLIER    = 2.25    # 1 : 1.5 R
ATR_TP2_MULTIPLIER    = 4.50    # 1 : 3.0 R
ATR_TP3_MULTIPLIER    = 6.75    # 1 : 4.5 R

# ─── Backtesting ──────────────────────────────────────────────────────────────
BACKTEST_INITIAL_CAPITAL  = 100_000
BACKTEST_COMMISSION_PCT   = 0.0001
BACKTEST_SLIPPAGE_PCT     = 0.0001
WALK_FORWARD_TRAIN_MONTHS = 12
WALK_FORWARD_TEST_MONTHS  = 1

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for _d in [CACHE_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)
