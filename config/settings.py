"""
Institutional-Grade XAUUSD ML Ensemble Trading System
Configuration & Settings
"""

# ─── INSTRUMENT ────────────────────────────────────────────────────────────────
SYMBOL         = "GC=F"          # Yahoo Finance Gold Futures ticker
SYMBOL_DISPLAY = "XAUUSD"
BASE_CURRENCY  = "USD"

# ─── TIMEFRAMES (Yahoo Finance intervals) ──────────────────────────────────────
TIMEFRAMES = {
    "1D":  {"interval": "1d",  "period": "2y",   "label": "Daily"},
    "4H":  {"interval": "1h",  "period": "60d",  "label": "4-Hour"},   # fetch 1h, resample to 4H
    "1H":  {"interval": "1h",  "period": "30d",  "label": "1-Hour"},
    "15M": {"interval": "15m", "period": "7d",   "label": "15-Minute"},
}

PRIMARY_TF   = "1D"
SECONDARY_TF = "4H"
ENTRY_TF     = "1H"

# ─── ML MODEL PARAMS ───────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.03,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 1.5,    # slight upweight for minority class
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}

LSTM_PARAMS = {
    "sequence_length":  60,
    "lstm_units":       [128, 64],
    "dropout":          0.3,
    "dense_units":      32,
    "batch_size":       32,
    "epochs":           50,
    "patience":         10,        # early stopping
    "learning_rate":    0.001,
    "val_split":        0.15,
}

GARCH_PARAMS = {
    "p": 1,
    "q": 1,
    "dist": "skewt",               # skewed-t for fat tails
    "vol": "Garch",
    "mean": "AR",
    "lags": 1,
    "forecast_horizon": 5,
}

# ─── FEATURE ENGINEERING ───────────────────────────────────────────────────────
LOOKBACK_PERIODS = {
    "swing_high_low":  20,
    "volume_profile":  50,
    "atr":             14,
    "rsi":             14,
    "macd_fast":       12,
    "macd_slow":       26,
    "macd_signal":     9,
    "bb_period":       20,
    "bb_std":          2,
    "ema_fast":        9,
    "ema_medium":      21,
    "ema_slow":        50,
    "ema_trend":       200,
    "adx":             14,
    "cci":             20,
    "stoch_k":         14,
    "stoch_d":         3,
    "williams_r":      14,
    "mfi":             14,
    "obv_period":      20,
    "vwap_period":     20,
    "cvd_period":      20,
}

SMC_PARAMS = {
    "swing_lookback":   10,         # bars each side to detect swing H/L
    "fvg_threshold":    0.001,      # 0.1% minimum FVG gap
    "ob_lookback":      50,         # order block search window
    "bos_lookback":     30,         # break of structure window
    "zone_tolerance":   0.002,      # 0.2% for zone proximity
    "liquidity_mult":   1.5,        # ATR multiplier for liquidity sweeps
    "choch_confirm":    3,          # bars needed to confirm CHoCH
}

# ─── META-MODEL FILTER (Rule-Based) ────────────────────────────────────────────
META_FILTER = {
    # Minimum probabilities from each model to pass the filter
    "xgb_min_prob":          0.60,
    "lstm_min_prob":         0.58,
    # Ensemble threshold to generate a signal
    "ensemble_threshold":    0.62,
    # Volatility regime thresholds (annualised vol)
    "vol_low_threshold":     0.08,
    "vol_high_threshold":    0.35,
    # CVD confirmation: require CVD alignment
    "require_cvd_confirm":   True,
    # MTF alignment: how many timeframes must agree
    "min_tf_alignment":      2,      # out of 3 (1D, 4H, 1H)
    # Session filter (UTC hours)
    "active_sessions": {
        "london":    (7,  16),
        "new_york":  (12, 21),
        "overlap":   (12, 16),
    },
    "require_active_session": True,
    # Minimum ATR-based move for valid setup
    "min_atr_multiplier":    0.5,
    # News/high-impact block: avoid trading within N minutes of NFP etc.
    "news_buffer_mins":      30,
    # Max spread in pips to trade
    "max_spread_pips":       3.0,
}

# ─── RISK MANAGEMENT ───────────────────────────────────────────────────────────
RISK = {
    "account_balance":       100_000,   # USD notional
    "risk_per_trade_pct":    0.01,       # 1% per trade
    "max_daily_drawdown_pct":0.03,       # 3% daily DD limit
    "max_open_trades":       3,
    "default_rr":            2.5,        # minimum R:R
    "atr_sl_multiplier":     1.5,        # SL = 1.5 × ATR(14)
    "atr_tp_multiplier":     3.5,        # TP = 3.5 × ATR(14) → ~2.33 R:R
    "trailing_stop_atr":     1.0,        # trail at 1× ATR
    "partial_tp_pct":        0.5,        # take 50% profit at first TP
    "partial_tp_rr":         1.5,        # first TP at 1.5 R:R
    "lot_size":              100,        # oz per standard lot (gold)
    "pip_value":             1.0,        # $1 per $1 move in gold
    "commission_per_lot":    7.0,        # broker commission USD/lot
    "slippage_pips":         1.5,
}

# ─── LOGGING & OUTPUT ──────────────────────────────────────────────────────────
LOG_DIR     = "logs"
RESULTS_DIR = "results"
MODEL_DIR   = "models/saved"
LOG_LEVEL   = "INFO"

# ─── SIGNAL LABELS ─────────────────────────────────────────────────────────────
SIGNAL = {
    "STRONG_BUY":  2,
    "BUY":         1,
    "NEUTRAL":     0,
    "SELL":       -1,
    "STRONG_SELL":-2,
}
