"""
Bidirectional LSTM + Multi-Head Attention Forecaster
Predicts the direction of the next N candles: 0=Neutral, 1=Up, 2=Down.
Architecture: BiLSTM(128) → BiLSTM(64) → MultiHeadAttention → Dense → Softmax
"""
import logging
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed – pip install torch")

import config

logger = logging.getLogger(__name__)


# ── Neural Network ─────────────────────────────────────────────────────────────

class _BiLSTMAttention(object if not TORCH_AVAILABLE else nn.Module):
    """
    Bidirectional LSTM with multi-head self-attention.

    Input:  (batch, seq_len, input_size)
    Output: (batch, 3)  softmax probabilities [neutral, up, down]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int    = config.LSTM_HIDDEN_SIZE,
        num_layers: int     = config.LSTM_NUM_LAYERS,
        dropout: float      = config.LSTM_DROPOUT,
        num_heads: int      = config.LSTM_NUM_HEADS,
        num_classes: int    = 3,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("Install PyTorch: pip install torch")
        super().__init__()

        self.hidden_size = hidden_size
        d_model = hidden_size * 2   # bidirectional doubles the size

        self.lstm1 = nn.LSTM(
            input_size, hidden_size,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            d_model, hidden_size // 2,
            num_layers=1, batch_first=True, bidirectional=True,
        )

        d_model2 = hidden_size  # (hidden_size//2) * 2

        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model2)
        self.dropout  = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            d_model2, num_heads, dropout=0.1, batch_first=True
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model2),
            nn.Linear(d_model2, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out1, _ = self.lstm1(x)
        out1    = self.norm1(self.dropout(out1))

        out2, _ = self.lstm2(out1)
        out2    = self.norm2(self.dropout(out2))

        attn_out, _ = self.attention(out2, out2, out2)
        attn_out    = attn_out + out2               # residual

        last = attn_out[:, -1, :]                  # last timestep
        return torch.softmax(self.head(last), dim=-1)


# ── Wrapper class ──────────────────────────────────────────────────────────────

class LSTMForecaster:
    """End-to-end LSTM trainer / predictor with per-feature scaling."""

    MODEL_FNAME  = "lstm_model.pt"
    SCALER_FNAME = "lstm_scaler.pkl"

    # Features to use as LSTM input (subset of engineered features)
    FEATURE_COLS = [
        "Close", "log_ret", "body", "wick_up", "wick_down",
        "rsi_14", "rsi_7",
        "macd_hist_norm", "stoch_k", "stoch_d",
        "atr_14_pct", "bb_position", "bb_width", "bb_squeeze",
        "ema_trend_score", "adx_14", "di_diff", "aroon_osc",
        "obv_norm", "vol_ratio",
        "ms_trend", "dist_support", "dist_resistance",
        "confluence_score", "mtf_trend_strength",
        "roc_10", "roc_20", "psar_signal", "cci_20",
        "kc_pos", "hvol_20",
    ]

    def __init__(self, device: Optional[str] = None) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("Install PyTorch: pip install torch")
        self.device   = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model:   Optional[_BiLSTMAttention] = None
        self.scaler:  Optional[StandardScaler]   = None
        self.seq_len: int = config.LSTM_SEQUENCE_LENGTH
        self.feat_cols: List[str] = self.FEATURE_COLS

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_sequences(
        self, df: pd.DataFrame, with_labels: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Slice the DataFrame into overlapping windows of length *seq_len*.

        Returns:
          X: (N, seq_len, n_features)
          y: (N,) int labels (only when with_labels=True and target column exists)
        """
        available = [c for c in self.feat_cols if c in df.columns]
        data = df[available].fillna(0).values.astype(np.float32)

        if self.scaler is None:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)

        # Update feat_cols to only available ones
        self.feat_cols = available

        n   = self.seq_len
        X   = np.stack([data[i: i + n] for i in range(len(data) - n)], axis=0)

        if not with_labels or "target" not in df.columns:
            return X, None

        y_raw = df["target"].values[n:]
        valid = ~np.isnan(y_raw)
        return X[valid], y_raw[valid].astype(np.int64)

    def fit(self, df: pd.DataFrame) -> Dict:
        """Train the LSTM on the provided DataFrame (must contain 'target' col)."""
        X, y = self.build_sequences(df, with_labels=True)
        n_feat = X.shape[2]

        self.model = _BiLSTMAttention(input_size=n_feat).to(self.device)
        optimizer  = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LSTM_LR,
            weight_decay=config.LSTM_WEIGHT_DECAY,
        )
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.LSTM_EPOCHS, eta_min=1e-5
        )
        criterion  = nn.CrossEntropyLoss()

        # Train / val split (80/20, no shuffle → temporal order preserved)
        split      = int(0.8 * len(X))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        ds_tr  = TensorDataset(
            torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)
        )
        ds_val = TensorDataset(
            torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)
        )
        dl_tr  = DataLoader(ds_tr,  batch_size=config.LSTM_BATCH_SIZE, shuffle=False)
        dl_val = DataLoader(ds_val, batch_size=config.LSTM_BATCH_SIZE, shuffle=False)

        best_val_loss  = float("inf")
        patience_count = 0
        history        = []

        for epoch in range(1, config.LSTM_EPOCHS + 1):
            # ── Train ───
            self.model.train()
            tr_loss = 0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(xb)

            # ── Validate ───
            self.model.eval()
            val_loss = 0.0
            correct  = 0
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    val_loss += criterion(logits, yb).item() * len(xb)
                    correct  += (logits.argmax(1) == yb).sum().item()

            tr_loss  /= len(ds_tr)
            val_loss /= len(ds_val)
            val_acc   = correct / len(ds_val)
            scheduler.step()

            history.append(
                {"epoch": epoch, "tr_loss": tr_loss,
                 "val_loss": val_loss, "val_acc": val_acc}
            )

            if epoch % 10 == 0:
                logger.debug(
                    "LSTM Epoch %3d | tr_loss=%.4f val_loss=%.4f val_acc=%.3f",
                    epoch, tr_loss, val_loss, val_acc,
                )

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_state     = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= config.LSTM_PATIENCE:
                    logger.info("LSTM early stop at epoch %d.", epoch)
                    break

        self.model.load_state_dict(best_state)
        self.save()
        return {"history": history, "best_val_loss": best_val_loss}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, 3) probability matrix for the last bar in each sequence window."""
        if self.model is None:
            self._try_load()

        X, _ = self.build_sequences(df, with_labels=False)
        if len(X) == 0:
            return np.array([[1 / 3, 1 / 3, 1 / 3]])

        self.model.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            proba = self.model(tensor).cpu().numpy()
        return proba   # (N, 3)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        model_path  = os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        scaler_path = os.path.join(config.MODELS_DIR, self.SCALER_FNAME)

        torch.save(
            {
                "state_dict":  self.model.state_dict(),
                "input_size":  len(self.feat_cols),
                "feat_cols":   self.feat_cols,
            },
            model_path,
        )
        joblib.dump(self.scaler, scaler_path)
        logger.info("LSTM model saved → %s", model_path)

    def load(self) -> None:
        model_path  = os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        scaler_path = os.path.join(config.MODELS_DIR, self.SCALER_FNAME)

        payload        = torch.load(model_path, map_location=self.device)
        self.feat_cols = payload["feat_cols"]
        n_feat         = payload["input_size"]

        self.model = _BiLSTMAttention(input_size=n_feat).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        logger.info("LSTM model loaded ← %s", model_path)

    def _try_load(self) -> None:
        model_path = os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        if os.path.exists(model_path):
            self.load()
        else:
            raise RuntimeError("LSTM model not trained. Call .fit() first.")


# Fix missing Dict import
from typing import Dict  # noqa: E402 (needed inside methods)
