"""
LSTM Sequential Model for XAUUSD price direction forecasting.
Architecture: 2-layer stacked LSTM + Dropout + Dense → Sigmoid
Uses sequences of engineered features to capture temporal dependencies.
"""
from __future__ import annotations

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config.settings import LSTM_PARAMS
from data.preprocessor import generate_labels, build_sequences, scale_features
from utils.logger import get_logger

log = get_logger("LSTMModel")

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Lazy TensorFlow import to avoid load time on non-GPU machines
_tf = None
_keras = None


def _get_tf():
    global _tf, _keras
    if _tf is None:
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            _tf = tf
            _keras = tf.keras
            log.info("TensorFlow %s loaded", tf.__version__)
        except ImportError:
            log.warning("TensorFlow not available — LSTM will run in stub mode")
    return _tf, _keras


class LSTMDirectionalModel:
    """
    Stacked LSTM classifier.
    Produces probability of a long setup for the next bar.
    Falls back to a simple trend-following heuristic if TF is unavailable.
    """

    def __init__(self, params: dict = None):
        self.params  = params or LSTM_PARAMS.copy()
        self.model   = None
        self.scaler  = None
        self.feature_cols: list[str] = []
        self.trained  = False
        self.tf_available = False
        self._metrics: dict = {}

    # ─── MODEL ARCHITECTURE ────────────────────────────────────────────────────

    def _build_model(self, n_features: int):
        tf, keras = _get_tf()
        if tf is None:
            return None

        seq_len  = self.params["seq_len"] = self.params["sequence_length"]
        units    = self.params["lstm_units"]
        drop     = self.params["dropout"]
        dense_u  = self.params["dense_units"]
        lr       = self.params["learning_rate"]

        inp = keras.Input(shape=(seq_len, n_features))
        x   = keras.layers.LSTM(units[0], return_sequences=True,
                                 kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
        x   = keras.layers.Dropout(drop)(x)
        x   = keras.layers.LSTM(units[1], return_sequences=False,
                                 kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x   = keras.layers.Dropout(drop)(x)
        x   = keras.layers.Dense(dense_u, activation="relu")(x)
        x   = keras.layers.BatchNormalization()(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        return model

    # ─── FEATURE SELECTION ─────────────────────────────────────────────────────

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        # Use a curated subset for LSTM (avoid too many redundant features)
        preferred = [
            "close", "rsi", "macd_line", "macd_hist", "macd_signal",
            "bb_pct_b", "bb_width", "adx", "di_plus", "di_minus",
            "stoch_k", "stoch_d", "cci", "williams_r", "mfi",
            "atr", "rv5", "rv20", "rv_ratio",
            "cross_9_21", "cross_21_50", "price_vs_200",
            "cvd", "cvd_z", "vol_ratio",
            "smc_trend", "in_bull_ob", "in_bear_ob",
            "bull_fvg", "bear_fvg", "fvg_size",
            "sweep_highs", "sweep_lows",
            "dist_from_eq", "in_discount",
            "candle_body_norm", "is_bullish", "engulfing",
        ]
        return [c for c in preferred if c in df.columns]

    # ─── TRAINING ──────────────────────────────────────────────────────────────

    def fit(self, df_features: pd.DataFrame) -> dict:
        tf, keras = _get_tf()
        self.tf_available = tf is not None

        log.info("LSTM: preparing training data…")
        labels = generate_labels(df_features)
        valid  = labels[labels != 99]
        df     = df_features.loc[valid.index].copy()
        y_raw  = valid.loc[df.index]
        y      = (y_raw == 1).astype(int).values.astype(np.float32)

        self.feature_cols = self._select_features(df)
        X     = df[self.feature_cols]
        n     = len(X)
        split = int(n * (1 - self.params["val_split"]))

        X_tr = X.iloc[:split]
        X_te = X.iloc[split:]
        y_tr = y[:split]
        y_te = y[split:]

        X_tr_sc, X_te_sc, self.scaler = scale_features(X_tr, X_te)

        seq_len  = self.params["sequence_length"]
        X_tr_seq, y_tr_seq = build_sequences(X_tr_sc.values, y_tr, seq_len)
        X_te_seq, y_te_seq = build_sequences(X_te_sc.values, y_te, seq_len)

        if len(X_tr_seq) < 100:
            log.warning("Insufficient data for LSTM (%d sequences). Using heuristic.", len(X_tr_seq))
            self.trained = False
            return {"note": "insufficient_data"}

        if not self.tf_available:
            log.warning("TensorFlow unavailable — LSTM running in heuristic mode")
            self.trained = False
            return {"note": "no_tensorflow"}

        self.model = self._build_model(len(self.feature_cols))

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=self.params["patience"],
                restore_best_weights=True, mode="max"),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        log.info("LSTM training: %d train seqs, %d features, %d epochs",
                 len(X_tr_seq), len(self.feature_cols), self.params["epochs"])

        history = self.model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_te_seq, y_te_seq),
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        probs = self.model.predict(X_te_seq, verbose=0).flatten()
        preds = (probs >= 0.5).astype(int)

        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        self._metrics = {
            "accuracy": accuracy_score(y_te_seq, preds),
            "auc":      roc_auc_score(y_te_seq, probs) if len(set(y_te_seq)) > 1 else 0.5,
            "f1":       f1_score(y_te_seq, preds, zero_division=0),
        }
        log.info("LSTM Val | acc=%.3f | AUC=%.3f | F1=%.3f",
                 self._metrics["accuracy"], self._metrics["auc"],
                 self._metrics["f1"])
        self.trained = True
        return self._metrics

    # ─── PREDICTION ────────────────────────────────────────────────────────────

    def predict_latest(self, df_features: pd.DataFrame) -> dict:
        """Return LSTM probability for the latest bar."""
        if not self.trained or self.model is None:
            return self._heuristic_predict(df_features)

        X  = df_features[self.feature_cols]
        seq_len = self.params["sequence_length"]
        if len(X) < seq_len:
            return self._heuristic_predict(df_features)

        recent = X.tail(seq_len)
        sc     = pd.DataFrame(
            self.scaler.transform(recent),
            index=recent.index,
            columns=recent.columns,
        )
        seq    = sc.values[np.newaxis, :, :]           # (1, seq_len, features)
        prob   = float(self.model.predict(seq, verbose=0)[0, 0])
        signal = 1 if prob >= 0.55 else (-1 if prob <= 0.45 else 0)
        return {
            "lstm_prob_long":  prob,
            "lstm_prob_short": 1 - prob,
            "lstm_signal":     signal,
        }

    def _heuristic_predict(self, df_features: pd.DataFrame) -> dict:
        """
        Fallback when TF unavailable: simple trend heuristic.
        Uses RSI + MACD + SMC trend as a proxy.
        """
        row  = df_features.iloc[-1]
        score = 0.0
        if "rsi" in df_features.columns:
            rsi_v = row.get("rsi", 50)
            score += (rsi_v - 50) / 100.0
        if "macd_hist" in df_features.columns:
            score += np.tanh(row.get("macd_hist", 0) * 100) * 0.3
        if "smc_trend" in df_features.columns:
            score += row.get("smc_trend", 0) * 0.2
        if "cvd_z" in df_features.columns:
            score += np.tanh(row.get("cvd_z", 0)) * 0.2

        prob = float(np.clip(0.5 + score, 0.05, 0.95))
        signal = 1 if prob >= 0.55 else (-1 if prob <= 0.45 else 0)
        return {
            "lstm_prob_long":  prob,
            "lstm_prob_short": 1 - prob,
            "lstm_signal":     signal,
            "lstm_mode":       "heuristic",
        }

    # ─── PERSISTENCE ───────────────────────────────────────────────────────────

    def save(self, path: str = None):
        base = str(MODEL_DIR / "lstm_xauusd")
        if self.trained and self.model is not None:
            self.model.save(base + ".keras")
        state = {
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "params": self.params,
            "trained": self.trained,
            "metrics": self._metrics,
        }
        with open(base + "_meta.pkl", "wb") as f:
            pickle.dump(state, f)
        log.info("LSTM model saved → %s", base)

    def load(self, path: str = None):
        base = str(MODEL_DIR / "lstm_xauusd")
        meta_path = base + "_meta.pkl"
        keras_path = base + ".keras"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                state = pickle.load(f)
            self.scaler       = state["scaler"]
            self.feature_cols = state["feature_cols"]
            self.params       = state["params"]
            self.trained      = state["trained"]
            self._metrics     = state.get("metrics", {})
        if self.trained and os.path.exists(keras_path):
            tf, keras = _get_tf()
            if tf is not None:
                self.model = keras.models.load_model(keras_path)
        log.info("LSTM model loaded ← %s", base)
