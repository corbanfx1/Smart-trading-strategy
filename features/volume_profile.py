"""
Volume Profile Feature Engineering.
Computes Point of Control (POC), Value Area High/Low, HVN/LVN proximity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("VolumeProfile")


def compute_volume_profile(
    df: pd.DataFrame,
    n_bins:   int = 50,
    lookback: int = 100,
) -> pd.DataFrame:
    """
    Rolling volume profile over *lookback* bars.
    Returns POC, VAH, VAL, and price position relative to each.
    """
    n      = len(df)
    poc    = np.full(n, np.nan)
    vah    = np.full(n, np.nan)
    val_   = np.full(n, np.nan)
    in_va  = np.zeros(n, int)

    for i in range(lookback, n):
        window = df.iloc[i - lookback: i]
        lo_w   = window["low"].min()
        hi_w   = window["high"].max()

        if hi_w <= lo_w:
            continue

        edges  = np.linspace(lo_w, hi_w, n_bins + 1)
        vol_bins = np.zeros(n_bins)

        for _, row in window.iterrows():
            # Distribute bar volume proportionally across price bins it spans
            bar_lo = row["low"]
            bar_hi = row["high"]
            bar_vol = row["volume"]
            span   = bar_hi - bar_lo
            if span < 1e-9:
                # single price bin
                idx = np.searchsorted(edges, (bar_lo + bar_hi) / 2) - 1
                idx = np.clip(idx, 0, n_bins - 1)
                vol_bins[idx] += bar_vol
            else:
                for b in range(n_bins):
                    overlap = min(edges[b + 1], bar_hi) - max(edges[b], bar_lo)
                    if overlap > 0:
                        vol_bins[b] += bar_vol * overlap / span

        poc_idx   = int(np.argmax(vol_bins))
        poc[i]    = (edges[poc_idx] + edges[poc_idx + 1]) / 2

        # Value Area = 70% of volume around POC
        total_vol = vol_bins.sum()
        va_target = total_vol * 0.70
        va_vol    = vol_bins[poc_idx]
        lo_idx    = poc_idx
        hi_idx    = poc_idx

        while va_vol < va_target and (lo_idx > 0 or hi_idx < n_bins - 1):
            add_above = vol_bins[hi_idx + 1] if hi_idx < n_bins - 1 else 0
            add_below = vol_bins[lo_idx - 1] if lo_idx > 0 else 0
            if add_above >= add_below:
                hi_idx += 1
                va_vol += add_above
            else:
                lo_idx -= 1
                va_vol += add_below

        vah[i]   = edges[hi_idx + 1]
        val_[i]  = edges[lo_idx]
        price    = df["close"].iloc[i]
        in_va[i] = int(val_[i] <= price <= vah[i])

    price_vs_poc = ((df["close"].values - poc) / (poc + 1e-9))
    price_vs_vah = ((df["close"].values - vah) / (vah + 1e-9))
    price_vs_val = ((df["close"].values - val_) / (val_ + 1e-9))

    return pd.DataFrame({
        "vp_poc":        poc,
        "vp_vah":        vah,
        "vp_val":        val_,
        "vp_in_va":      in_va,
        "price_vs_poc":  price_vs_poc,
        "price_vs_vah":  price_vs_vah,
        "price_vs_val":  price_vs_val,
    }, index=df.index)


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build volume profile features (uses simplified lookback for speed)."""
    log.debug("Building volume profile features")
    vp = compute_volume_profile(df, n_bins=30, lookback=50)
    log.debug("  ✓ %d volume profile features", vp.shape[1])
    return vp
