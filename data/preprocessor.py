"""
Preprocessor — label generation, normalisation, train/test splitting.
Produces forward-looking labels for supervised ML training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from config.settings import RISK
from utils.logger import get_logger

log = get_logger("Preprocessor")


# ─── LABEL GENERATION ──────────────────────────────────────────────────────────

def generate_labels(
    df: pd.DataFrame,
    atr_col: str = "atr",
    tp_mult: float = RISK["atr_tp_multiplier"],
    sl_mult: float = RISK["atr_sl_multiplier"],
    horizon:  int  = 20,
) -> pd.Series:
    """
    Forward-looking ternary label:
      +1  → price reaches TP before SL within *horizon* bars  (LONG)
       0  → neither hit within horizon                         (NEUTRAL)
      -1  → price reaches SL before TP                        (SHORT)

    TP = close + tp_mult * ATR
    SL = close - sl_mult * ATR
    """
    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    atr    = df[atr_col].values
    labels = np.zeros(len(df), dtype=int)

    for i in range(len(df) - horizon):
        tp = close[i] + tp_mult * atr[i]
        sl = close[i] - sl_mult * atr[i]

        for j in range(i + 1, min(i + horizon + 1, len(df))):
            if high[j] >= tp:
                labels[i] = 1
                break
            if low[j] <= sl:
                labels[i] = -1
                break

    # last *horizon* bars are unlabelled — drop later
    labels[-horizon:] = 99
    series = pd.Series(labels, index=df.index, name="label")
    return series


def label_binary_direction(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Simple binary: 1 if close[t+horizon] > close[t], else 0."""
    future = df["close"].shift(-horizon)
    label  = (future > df["close"]).astype(int)
    label.name = "label_bin"
    return label


# ─── NORMALISATION ─────────────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """RobustScaler fit on train, applied to both splits."""
    scaler  = RobustScaler()
    X_tr_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_te_sc = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    return X_tr_sc, X_te_sc, scaler


# ─── WALK-FORWARD SPLIT ────────────────────────────────────────────────────────

def walk_forward_splits(
    df: pd.DataFrame,
    n_splits:     int = 5,
    test_ratio:   float = 0.15,
    gap:          int = 5,
) -> list[tuple[pd.Index, pd.Index]]:
    """
    Generate *n_splits* expanding walk-forward train/test index pairs.
    *gap* bars are excluded between train and test to prevent leakage.
    """
    n = len(df)
    test_size  = int(n * test_ratio)
    min_train  = int(n * 0.40)
    splits     = []

    step = (n - min_train - test_size) // max(n_splits - 1, 1)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_start = train_end + gap
        test_end   = test_start + test_size
        if test_end > n:
            break
        train_idx = df.index[:train_end]
        test_idx  = df.index[test_start:test_end]
        splits.append((train_idx, test_idx))

    log.info("Walk-forward: %d splits, test_size=%d bars each", len(splits), test_size)
    return splits


# ─── SEQUENCE BUILDER FOR LSTM ─────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape flat feature array into (samples, seq_len, features) for LSTM."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
