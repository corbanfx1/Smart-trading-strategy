"""
XGBoost Setup Classifier
Predicts trade setup direction: 0=Neutral, 1=Long, 2=Short.
Trained with time-series cross-validation (walk-forward splits).
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("xgboost not installed – pip install xgboost")

import config

logger = logging.getLogger(__name__)

# Columns to drop before feeding to XGBoost (raw OHLCV + non-numeric)
_DROP_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "bb_upper", "bb_lower", "bb_mid",
    "ema_9", "ema_20", "ema_50", "ema_200",
    "sma_20", "sma_50", "sma_200",
    "psar", "vwap",
    "kc_upper", "kc_lower",
]


class XGBoostSetupClassifier:
    """
    Three-class classifier: Neutral (0), Long (1), Short (2).

    The target label is constructed from *n*-bar forward returns:
      Long  if max_gain > ATR_MULTIPLIER × ATR before max_loss occurs
      Short if max_loss > ATR_MULTIPLIER × ATR before max_gain occurs
      Neutral otherwise
    """

    MODEL_FNAME = "xgb_setup.pkl"

    def __init__(self) -> None:
        if not XGB_AVAILABLE:
            raise ImportError("Install xgboost: pip install xgboost")
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.label_enc = LabelEncoder()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Build forward-looking labels.  Returns a Series aligned to *df*.
        NaN at the tail (insufficient look-forward data) is later dropped.
        """
        c        = df["Close"].values
        atr      = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df))
        n        = config.TARGET_FORWARD_BARS
        thr      = config.TARGET_ATR_MULTIPLIER
        labels   = np.zeros(len(df), dtype=np.int8)

        for i in range(len(df) - n):
            future_c     = c[i + 1: i + n + 1]
            max_gain     = np.max(future_c - c[i])
            max_loss     = np.max(c[i] - future_c)
            threshold    = thr * atr[i]

            if max_gain >= threshold and max_gain > max_loss:
                labels[i] = 1    # Long
            elif max_loss >= threshold and max_loss > max_gain:
                labels[i] = 2    # Short
            # else 0 = Neutral

        result = pd.Series(labels, index=df.index, name="target")
        result.iloc[-n:] = np.nan   # tail has no valid forward data
        return result

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw OHLCV / collinear columns; return numeric feature matrix."""
        drop = [c for c in _DROP_COLS if c in df.columns]
        feats = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])

        # Also drop duplicate HTF price columns
        for col in list(feats.columns):
            if any(feats[col].equals(df.get("Close", feats[col]))):
                if col not in ("Close",) and "close" in col.lower():
                    feats = feats.drop(columns=[col])

        return feats

    def fit(self, df: pd.DataFrame, n_cv_folds: int = config.XGB_CV_FOLDS) -> Dict:
        """Train with TimeSeriesSplit CV. Returns metrics dict."""
        target  = self.build_target(df)
        feats   = self.prepare_features(df)

        # Align
        valid   = target.dropna().index
        X       = feats.loc[valid].fillna(0)
        y       = target.loc[valid].astype(int)

        self.feature_names = list(X.columns)
        logger.info(
            "XGBoost training: %d samples, %d features, label dist=%s",
            len(X), len(self.feature_names),
            dict(y.value_counts().sort_index()),
        )

        tscv    = TimeSeriesSplit(n_splits=n_cv_folds)
        cv_results = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(
                **config.XGB_PARAMS,
                early_stopping_rounds=config.XGB_EARLY_STOPPING,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds     = model.predict(X_val)
            acc       = (preds == y_val.values).mean()
            cv_results.append({"fold": fold, "val_acc": acc, "n_val": len(y_val)})
            logger.debug("  Fold %d: val_acc=%.3f", fold, acc)

        # Retrain on full data
        self.model = xgb.XGBClassifier(**config.XGB_PARAMS)
        self.model.fit(X, y, verbose=False)

        mean_acc = np.mean([r["val_acc"] for r in cv_results])
        logger.info("XGBoost CV mean accuracy: %.3f", mean_acc)

        self.save()
        return {"cv_results": cv_results, "mean_cv_acc": mean_acc}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, 3) probability matrix [P(neutral), P(long), P(short)]."""
        if self.model is None:
            self._try_load()
        feats = self.prepare_features(df)
        feats = feats.reindex(columns=self.feature_names, fill_value=0).fillna(0)
        return self.model.predict_proba(feats)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (0=Neutral, 1=Long, 2=Short)."""
        return np.argmax(self.predict_proba(df), axis=1)

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        imp = pd.DataFrame(
            {"feature": self.feature_names,
             "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)
        return imp.head(top_n)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        logger.info("XGBoost model saved → %s", path)

    def load(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        payload = joblib.load(path)
        self.model         = payload["model"]
        self.feature_names = payload["features"]
        logger.info("XGBoost model loaded ← %s", path)

    def _try_load(self) -> None:
        path = os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        if os.path.exists(path):
            self.load(path)
        else:
            raise RuntimeError(
                "XGBoost model not trained. Call .fit() first."
            )
