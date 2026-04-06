"""
XGBoost Directional Classifier.
Predicts probability of bullish/bearish movement for XAUUSD.
Includes walk-forward cross-validation, feature importance, and SHAP analysis.
"""
from __future__ import annotations

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, precision_score, recall_score,
                              f1_score)
from sklearn.preprocessing import LabelEncoder

from config.settings import XGBOOST_PARAMS, RISK
from data.preprocessor import generate_labels, scale_features, walk_forward_splits
from utils.logger import get_logger

log = get_logger("XGBoostModel")

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class XGBoostDirectionalModel:
    """
    XGBoost classifier for XAUUSD directional prediction.
    Binary output: 1 = long opportunity, 0 = short/neutral.
    """

    def __init__(self, params: dict = None):
        self.params  = params or XGBOOST_PARAMS.copy()
        self.model:  Optional[XGBClassifier] = None
        self.scaler  = None
        self.feature_cols: list[str] = []
        self.threshold = 0.55
        self._metrics: dict = {}

    # ─── FEATURE SELECTION ─────────────────────────────────────────────────────

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        """Drop raw OHLCV + label-related cols; keep engineered features."""
        exclude = {"open", "high", "low", "close", "volume", "label",
                   "label_bin", "equilibrium", "vp_poc", "vp_vah", "vp_val",
                   "ema9", "ema21", "ema50", "ema200",   # use cross instead
                   "bb_mid", "bb_upper", "bb_lower",     # use normalised
                   "bull_ob_top", "bull_ob_bot",
                   "bear_ob_top", "bear_ob_bot"}
        return [c for c in df.columns if c not in exclude and
                not c.startswith("swing_") and
                pd.api.types.is_numeric_dtype(df[c])]

    # ─── TRAINING ──────────────────────────────────────────────────────────────

    def fit(self, df_features: pd.DataFrame) -> dict:
        """
        Walk-forward train on *df_features*.
        Returns aggregate OOS metrics.
        """
        log.info("XGBoost: generating labels…")
        labels = generate_labels(df_features)

        # Align & filter
        valid = labels[labels != 99]
        df    = df_features.loc[valid.index].copy()
        y_raw = valid.loc[df.index]

        # Convert ternary → binary: 1=long, 0=short/neutral
        y = (y_raw == 1).astype(int)

        self.feature_cols = self._select_features(df)
        X = df[self.feature_cols]

        log.info("XGBoost training: %d samples, %d features, pos_rate=%.2f%%",
                 len(X), len(self.feature_cols), 100 * y.mean())

        splits = walk_forward_splits(X, n_splits=5)
        all_preds, all_probs, all_true = [], [], []

        for fold_i, (tr_idx, te_idx) in enumerate(splits):
            X_tr = X.loc[tr_idx]
            X_te = X.loc[te_idx]
            y_tr = y.loc[tr_idx]
            y_te = y.loc[te_idx]

            X_tr_sc, X_te_sc, scaler = scale_features(X_tr, X_te)

            model = XGBClassifier(**self.params)
            model.fit(
                X_tr_sc, y_tr,
                eval_set=[(X_te_sc, y_te)],
                verbose=False,
            )

            probs = model.predict_proba(X_te_sc)[:, 1]
            preds = (probs >= self.threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_te.values)

            fold_acc = accuracy_score(y_te, preds)
            log.info("  Fold %d/%d: acc=%.3f, AUC=%.3f",
                     fold_i + 1, len(splits), fold_acc,
                     roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else 0.5)

        # Final model on full data
        X_sc, _, self.scaler = scale_features(X, X.iloc[:1])
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_sc, y, verbose=False)

        # OOS metrics
        self._metrics = {
            "accuracy":  accuracy_score(all_true, all_preds),
            "precision": precision_score(all_true, all_preds, zero_division=0),
            "recall":    recall_score(all_true, all_preds, zero_division=0),
            "f1":        f1_score(all_true, all_preds, zero_division=0),
            "auc":       roc_auc_score(all_true, all_probs)
                         if len(set(all_true)) > 1 else 0.5,
        }
        log.info("XGBoost OOS | acc=%.3f | AUC=%.3f | F1=%.3f",
                 self._metrics["accuracy"], self._metrics["auc"],
                 self._metrics["f1"])
        return self._metrics

    # ─── PREDICTION ────────────────────────────────────────────────────────────

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Return (n, 2) probability matrix for the latest bars."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X = df_features[self.feature_cols].tail(100)
        X_sc = pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns,
        )
        return self.model.predict_proba(X_sc)

    def predict_latest(self, df_features: pd.DataFrame) -> dict:
        """Return prediction for the most recent bar."""
        probs = self.predict_proba(df_features)
        latest_prob = float(probs[-1, 1])
        signal = 1 if latest_prob >= self.threshold else (-1 if latest_prob <= 1 - self.threshold else 0)
        return {
            "xgb_prob_long":  latest_prob,
            "xgb_prob_short": float(probs[-1, 0]),
            "xgb_signal":     signal,
        }

    # ─── FEATURE IMPORTANCE ────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        imp = pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)
        return imp.head(top_n).to_frame("importance")

    # ─── PERSISTENCE ───────────────────────────────────────────────────────────

    def save(self, path: str = None):
        path = path or str(MODEL_DIR / "xgboost_xauusd.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": self.model,
                         "scaler": self.scaler,
                         "feature_cols": self.feature_cols,
                         "threshold": self.threshold,
                         "metrics": self._metrics}, f)
        log.info("XGBoost model saved → %s", path)

    def load(self, path: str = None):
        path = path or str(MODEL_DIR / "xgboost_xauusd.pkl")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model        = state["model"]
        self.scaler       = state["scaler"]
        self.feature_cols = state["feature_cols"]
        self.threshold    = state["threshold"]
        self._metrics     = state.get("metrics", {})
        log.info("XGBoost model loaded ← %s", path)
