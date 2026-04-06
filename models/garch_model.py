"""
ARCH/GARCH Volatility Model for XAUUSD.
Fits AR(1)-GARCH(1,1) with skewed-t innovations.
Produces:
  - Conditional volatility forecasts (5-bar ahead)
  - Volatility regime classification (low/normal/high)
  - Annualised VaR and CVaR estimates
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

from config.settings import GARCH_PARAMS as GP
from utils.logger import get_logger

log = get_logger("GARCHModel")

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class GARCHVolatilityModel:
    """
    AR(1)-GARCH(1,1) volatility model.
    Falls back to rolling-std estimate if arch package unavailable.
    """

    def __init__(self, params: dict = None):
        self.params  = params or GP.copy()
        self.model   = None
        self.result  = None
        self.fitted   = False
        self._arch_available = False

    # ─── HELPERS ───────────────────────────────────────────────────────────────

    @staticmethod
    def _log_returns(close: pd.Series) -> pd.Series:
        return np.log(close / close.shift(1)).dropna() * 100  # scaled ×100

    # ─── FIT ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame):
        """Fit GARCH to the close price series."""
        returns = self._log_returns(df["close"])
        log.info("GARCH fitting on %d return observations…", len(returns))

        try:
            from arch import arch_model
            self._arch_available = True

            am = arch_model(
                returns,
                vol=self.params["vol"],
                p=self.params["p"],
                q=self.params["q"],
                mean=self.params["mean"],
                lags=self.params["lags"],
                dist=self.params["dist"],
            )
            self.result = am.fit(
                disp="off",
                show_warning=False,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            self.model  = am
            self.fitted = True

            # Log model summary (key params)
            params_ = self.result.params
            log.info("GARCH fitted | omega=%.6f alpha=%.4f beta=%.4f",
                     params_.get("omega", np.nan),
                     params_.get("alpha[1]", np.nan),
                     params_.get("beta[1]", np.nan))
        except ImportError:
            log.warning("arch package unavailable — GARCH using rolling-std fallback")
            self._arch_available = False
            self.fitted = False

    # ─── FORECAST ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bars_per_year(df: pd.DataFrame) -> float:
        """
        Infer bars-per-year from the DataFrame's DatetimeIndex frequency.
        Falls back to median timedelta inspection when freq is not set.
        """
        if isinstance(df.index, pd.DatetimeIndex) and len(df) >= 2:
            try:
                freq = df.index.freq or df.index.inferred_freq
                if freq is not None:
                    freq_str = str(freq).upper()
                    if "D" in freq_str and "H" not in freq_str:
                        return 252.0
                    if "4H" in freq_str or "4T" in freq_str:
                        return 252.0 * 6
                    if freq_str.startswith("H") or freq_str.startswith("60"):
                        return 252.0 * 24
                    if "15" in freq_str:
                        return 252.0 * 96
            except Exception:
                pass
            # Fallback: use median bar duration
            deltas = df.index[1:] - df.index[:-1]
            median_sec = float(np.median([d.total_seconds() for d in deltas]))
            if median_sec > 0:
                return (365.25 * 24 * 3600) / median_sec
        return 252.0  # safe default (daily)

    def forecast_volatility(self, df: pd.DataFrame,
                             horizon: int = None) -> dict:
        """
        Forecast conditional volatility *horizon* bars ahead.
        Returns vol forecast, vol regime, annualised vol, VaR/CVaR.
        Annualisation uses the actual bar frequency of *df* so the model
        is correctly calibrated whether fit on 1H, 4H, or daily data.
        """
        horizon = horizon or self.params["forecast_horizon"]
        returns = self._log_returns(df["close"])
        current_vol_rv = float(returns.rolling(20).std().iloc[-1])
        bars_per_year  = self._bars_per_year(df)

        if self.fitted and self._arch_available:
            try:
                fc = self.result.forecast(horizon=horizon, reindex=False)
                var_h = fc.variance.iloc[-1].values          # (horizon,)
                vol_h = np.sqrt(var_h)                       # % per bar

                # Annualise using the correct bars-per-year for this timeframe
                ann_vol  = float(vol_h[0]) * np.sqrt(bars_per_year) / 100
                cond_vol = float(vol_h[0]) / 100

            except Exception as e:
                log.warning("GARCH forecast failed (%s), using rolling std", e)
                cond_vol = current_vol_rv / 100
                ann_vol  = cond_vol * np.sqrt(bars_per_year)
        else:
            cond_vol = current_vol_rv / 100
            ann_vol  = cond_vol * np.sqrt(bars_per_year)

        # Volatility regime
        if ann_vol < 0.08:
            regime = "low"
        elif ann_vol < 0.20:
            regime = "normal"
        elif ann_vol < 0.35:
            regime = "elevated"
        else:
            regime = "high"

        # Historical VaR (95%) and CVaR
        var_95  = float(np.percentile(returns, 5)) / 100
        cvar_95 = float(returns[returns <= np.percentile(returns, 5)].mean()) / 100

        # Position size adjustment based on vol regime
        vol_scalar = max(0.25, min(1.5, 0.15 / (ann_vol + 1e-9)))

        return {
            "garch_cond_vol":    cond_vol,
            "garch_ann_vol":     ann_vol,
            "garch_regime":      regime,
            "garch_var_95":      var_95,
            "garch_cvar_95":     cvar_95,
            "garch_vol_scalar":  vol_scalar,
            "garch_rv20":        current_vol_rv / 100,
        }

    def add_garch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append bar-by-bar GARCH conditional volatility estimates to df.
        Used as input feature for XGBoost/LSTM.
        """
        returns = self._log_returns(df["close"])

        if self.fitted and self._arch_available:
            try:
                cond_vol = self.result.conditional_volatility
                # Align index
                cond_vol.index = returns.index
                df = df.copy()
                df.loc[cond_vol.index, "garch_vol"] = cond_vol.values / 100
            except Exception:
                df["garch_vol"] = returns.rolling(20).std() / 100
        else:
            df = df.copy()
            df["garch_vol"] = returns.rolling(20).std().reindex(df.index) / 100

        # Standardised vol
        rv20 = returns.rolling(20).std()
        rv5  = returns.rolling(5).std()
        df["vol_regime_score"] = (rv5.reindex(df.index) /
                                  (rv20.reindex(df.index) + 1e-9))
        return df
