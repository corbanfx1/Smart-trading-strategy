"""
Synthetic XAUUSD data generator for offline testing and CI validation.
Generates realistic GBM + mean-reversion price paths with GARCH-like volatility.
Matches current chart context: ~4,434 price, bearish bias, elevated vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


def generate_xauusd(
    n_bars:      int   = 500,
    interval:    str   = "1H",
    start_price: float = 4_434.0,
    ann_vol:     float = 0.18,
    drift:       float = -0.0002,   # slight bearish as per chart context
    seed:        int   = 42,
) -> pd.DataFrame:
    """
    Generate synthetic XAUUSD OHLCV bars.
    Uses GBM with GARCH(1,1)-like conditional vol clustering.
    """
    rng = np.random.default_rng(seed)

    # ── Timeframe params ──────────────────────────────────────────────────────
    intervals = {
        "1D":  {"minutes": 1440, "bars_per_year": 252},
        "4H":  {"minutes": 240,  "bars_per_year": 252 * 6},
        "1H":  {"minutes": 60,   "bars_per_year": 252 * 24},
        "15M": {"minutes": 15,   "bars_per_year": 252 * 96},
    }
    cfg = intervals.get(interval, intervals["1H"])
    dt  = cfg["minutes"] / (252 * 1440)     # fraction of year per bar
    sigma_bar = ann_vol * np.sqrt(dt)

    # ── GARCH(1,1) vol process ────────────────────────────────────────────────
    omega = 0.00001
    alpha = 0.08
    beta  = 0.90
    sigma2 = np.full(n_bars, sigma_bar ** 2)
    eps    = np.zeros(n_bars)
    for t in range(1, n_bars):
        eps[t]    = rng.normal(0, np.sqrt(sigma2[t - 1]))
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma_t = np.sqrt(sigma2)

    # ── Log returns ───────────────────────────────────────────────────────────
    log_ret = drift * dt + sigma_t * rng.standard_normal(n_bars)

    # ── Price path ────────────────────────────────────────────────────────────
    close = start_price * np.exp(np.cumsum(log_ret))

    # ── OHLCV construction ────────────────────────────────────────────────────
    bar_range = sigma_t * start_price * 2.5   # realistic intra-bar range
    open_  = np.empty(n_bars)
    high   = np.empty(n_bars)
    low    = np.empty(n_bars)
    volume = np.empty(n_bars)

    open_[0] = start_price
    for i in range(n_bars):
        open_[i]  = close[i - 1] if i > 0 else start_price
        r         = bar_range[i]
        # Add small wick noise
        wick_up   = abs(rng.normal(0, r * 0.3))
        wick_down = abs(rng.normal(0, r * 0.3))
        if close[i] >= open_[i]:
            high[i] = close[i] + wick_up
            low[i]  = open_[i] - wick_down
        else:
            high[i] = open_[i] + wick_up
            low[i]  = close[i] - wick_down
        volume[i] = abs(rng.normal(10_000, 3_000)) * (1 + 3 * abs(log_ret[i]) / sigma_bar)

    # ── Timestamps ────────────────────────────────────────────────────────────
    end_ts = pd.Timestamp("2026-03-26 16:00:00", tz="UTC")
    freq   = {"1D": "D", "4H": "4h", "1H": "h", "15M": "15min"}[interval]
    idx    = pd.date_range(end=end_ts, periods=n_bars, freq=freq, tz="UTC")

    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=idx)

    return df


def generate_all_timeframes(start_price: float = 4_434.0) -> dict[str, pd.DataFrame]:
    """Generate synthetic data for all configured timeframes."""
    return {
        "1D":  generate_xauusd(500,  "1D",  start_price),
        "4H":  generate_xauusd(720,  "4H",  start_price),
        "1H":  generate_xauusd(720,  "1H",  start_price),
        "15M": generate_xauusd(672,  "15M", start_price),
    }
