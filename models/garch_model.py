"""
ARCH/GARCH Volatility Model
Fits GARCH(1,1) and GJR-GARCH(1,1) to gold return series.
Outputs: conditional volatility forecast, VaR estimates, and regime label.
"""
import logging
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from arch import arch_model
    from arch.univariate import GARCH, GJR_GARCH, SkewStudent
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch library not installed – pip install arch")

import config

logger = logging.getLogger(__name__)


class GARCHVolatilityModel:
    """
    Dual-model GARCH ensemble:
      1. GARCH(1,1)       – baseline symmetric volatility
      2. GJR-GARCH(1,1)  – asymmetric (leverage) effects

    Forecasts conditional volatility 1–5 bars ahead and classifies the
    current regime as low / medium / high.
    """

    MODEL_FNAME = "garch_model.pkl"

    def __init__(self) -> None:
        if not ARCH_AVAILABLE:
            raise ImportError("Install arch: pip install arch")
        self.garch_result    = None
        self.gjr_result      = None
        self.returns_std:    Optional[float] = None  # unconditional std
        self._is_fitted      = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, daily_df: pd.DataFrame) -> Dict:
        """
        Fit GARCH and GJR-GARCH on the daily log-return series.

        Args:
            daily_df: DataFrame with a 'Close' column (daily frequency).
        Returns:
            Dict of fit summaries.
        """
        returns = self._get_returns(daily_df)
        self.returns_std = float(returns.std())

        logger.info(
            "Fitting GARCH on %d daily observations (σ_unconditional=%.4f)",
            len(returns), self.returns_std,
        )

        # GARCH(1,1) with skewed-t innovations
        try:
            garch_spec = arch_model(
                returns * 100,          # scale to percentage returns
                vol="Garch",
                p=config.GARCH_P,
                q=config.GARCH_Q,
                dist="skewt",
                mean="Constant",
            )
            self.garch_result = garch_spec.fit(disp="off", show_warning=False)
            logger.debug("GARCH fitted: AIC=%.2f", self.garch_result.aic)
        except Exception as exc:
            logger.warning("GARCH fitting failed: %s", exc)
            self.garch_result = None

        # GJR-GARCH(1,1) – captures asymmetric volatility (bad news > good news)
        try:
            gjr_spec = arch_model(
                returns * 100,
                vol="GARCH",
                p=config.GARCH_P,
                o=1,                    # GJR term
                q=config.GARCH_Q,
                dist="skewt",
                mean="Constant",
            )
            self.gjr_result = gjr_spec.fit(disp="off", show_warning=False)
            logger.debug("GJR-GARCH fitted: AIC=%.2f", self.gjr_result.aic)
        except Exception as exc:
            logger.warning("GJR-GARCH fitting failed: %s", exc)
            self.gjr_result = None

        self._is_fitted = True
        self.save()

        return {
            "garch_aic":    self.garch_result.aic if self.garch_result else None,
            "gjr_aic":      self.gjr_result.aic   if self.gjr_result   else None,
            "n_obs":        len(returns),
            "unconditional_vol": self.returns_std,
        }

    def forecast(self, horizon: int = config.GARCH_HORIZON) -> Dict:
        """
        Produce a *horizon*-step ahead volatility forecast.

        Returns:
          {
            "cond_vol_1":   current 1-step conditional vol (% daily),
            "cond_vol_mean": mean over horizon,
            "regime":        "low" | "medium" | "high",
            "regime_score":  1.0 (low) … 0.0 (high vol → reduced position),
            "var_95":        1-day 95% VaR (fractional),
            "var_99":        1-day 99% VaR (fractional),
          }
        """
        if not self._is_fitted:
            self._try_load()

        cond_vol_pct = self._best_vol_forecast(horizon)  # as % return
        cond_vol     = cond_vol_pct / 100                 # back to fractional

        regime = self._classify_regime(cond_vol)

        # VaR (normal approximation)
        var_95 = 1.645 * cond_vol
        var_99 = 2.326 * cond_vol

        regime_score = config.VOL_REGIME_MULTIPLIER.get(regime, 0.80)

        return {
            "cond_vol_1":    round(cond_vol,     6),
            "cond_vol_pct":  round(cond_vol_pct, 4),
            "cond_vol_mean": round(cond_vol,     6),
            "regime":        regime,
            "regime_score":  regime_score,
            "var_95":        round(var_95, 6),
            "var_99":        round(var_99, 6),
        }

    def add_features(self, df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach time-varying conditional volatility from GARCH to *df*.
        Uses the in-sample conditional variance series.
        """
        if not self._is_fitted:
            return df

        out = df.copy()
        cond_vol_series = self._extract_conditional_vol(daily_df)

        if cond_vol_series is not None:
            # Forward-fill daily cond_vol onto the primary (finer) TF
            cond_vol_ff = cond_vol_series.reindex(
                out.index, method="ffill"
            )
            out["garch_cond_vol"] = cond_vol_ff / 100   # fractional
            out["garch_regime"]   = out["garch_cond_vol"].apply(
                lambda v: self._regime_code(self._classify_regime(v))
            )
        else:
            out["garch_cond_vol"] = self.returns_std or 0.01
            out["garch_regime"]   = 1  # medium by default

        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _best_vol_forecast(self, horizon: int) -> float:
        """Return the 1-step ahead conditional volatility (in % terms)."""
        result = self.gjr_result or self.garch_result
        if result is None:
            return (self.returns_std or 0.01) * 100

        try:
            fc = result.forecast(horizon=horizon, reindex=False)
            # conditional variance is in (%)^2 space; take sqrt
            var_h = fc.variance.iloc[-1].values
            vol_1 = float(np.sqrt(var_h[0]))
        except Exception as exc:
            logger.debug("Forecast error: %s", exc)
            vol_1 = (self.returns_std or 0.01) * 100

        return vol_1

    def _extract_conditional_vol(self, daily_df: pd.DataFrame) -> Optional[pd.Series]:
        result = self.gjr_result or self.garch_result
        if result is None:
            return None
        try:
            return np.sqrt(result.conditional_volatility)
        except Exception:
            return None

    @staticmethod
    def _get_returns(df: pd.DataFrame) -> pd.Series:
        c = df["Close"]
        r = np.log(c / c.shift(1)).dropna()
        return r

    @staticmethod
    def _classify_regime(cond_vol_frac: float) -> str:
        if cond_vol_frac < config.VOL_LOW_THRESHOLD:
            return "low"
        if cond_vol_frac < config.VOL_HIGH_THRESHOLD:
            return "medium"
        return "high"

    @staticmethod
    def _regime_code(regime: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(regime, 1)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        path = os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        joblib.dump(
            {
                "garch_result":  self.garch_result,
                "gjr_result":    self.gjr_result,
                "returns_std":   self.returns_std,
            },
            path,
        )
        logger.info("GARCH model saved → %s", path)

    def load(self) -> None:
        path = os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        payload             = joblib.load(path)
        self.garch_result   = payload["garch_result"]
        self.gjr_result     = payload["gjr_result"]
        self.returns_std    = payload["returns_std"]
        self._is_fitted     = True
        logger.info("GARCH model loaded ← %s", path)

    def _try_load(self) -> None:
        path = os.path.join(config.MODELS_DIR, self.MODEL_FNAME)
        if os.path.exists(path):
            self.load()
        else:
            logger.warning("GARCH model not found; using unconditional volatility.")
            self._is_fitted   = True
            self.returns_std  = 0.01
