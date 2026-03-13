"""
Volume Profile – Full Implementation
──────────────────────────────────────
• Fixed-Range Volume Profile  (user-defined bar window)
• Rolling Session VP          (per-session or rolling N-bar)
• Point of Control (POC / VPOC)
• Value Area High / Low       (default 70% of total volume)
• High Volume Nodes (HVN)     – price bins with above-average volume
• Low Volume Nodes  (LVN)     – price bins with below-average volume (fast-travel zones)
• VPOC migration              – rising / falling / flat (institutional bias signal)
• Price position relative to VA, VPOC, HVN/LVN
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config as cfg


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VolumeProfileResult:
    vpoc: float                   # Point of Control (highest volume price)
    vah: float                    # Value Area High  (top of 70% VA)
    val: float                    # Value Area Low   (bottom of 70% VA)
    total_volume: float
    value_area_pct: float         # default 0.70
    price_bins: np.ndarray        # bin midpoints
    volume_per_bin: np.ndarray    # volume at each bin
    hvn_levels: List[float]       # High Volume Node levels
    lvn_levels: List[float]       # Low Volume Node levels

    # Derived
    @property
    def value_area_range(self) -> float:
        return self.vah - self.val

    def price_position(self, price: float) -> str:
        """Classify price relative to the Value Area."""
        if price > self.vah:
            return "ABOVE_VA"
        elif price < self.val:
            return "BELOW_VA"
        elif price >= self.vpoc:
            return "VA_HIGH"
        else:
            return "VA_LOW"

    def dist_from_vpoc_pct(self, price: float) -> float:
        return (price - self.vpoc) / self.vpoc * 100 if self.vpoc > 0 else 0.0

    def nearest_hvn(self, price: float) -> Optional[float]:
        if not self.hvn_levels:
            return None
        return min(self.hvn_levels, key=lambda h: abs(h - price))

    def nearest_lvn(self, price: float) -> Optional[float]:
        if not self.lvn_levels:
            return None
        return min(self.lvn_levels, key=lambda l: abs(l - price))


@dataclass
class VPOCMigration:
    direction: str          # 'RISING' | 'FALLING' | 'FLAT'
    magnitude_pct: float    # % change over the migration window
    bars: int               # window used


# ── Core volume profile builder ───────────────────────────────────────────────

def build_volume_profile(
    df: pd.DataFrame,
    price_bins: int = 100,
    value_area_pct: float = 0.70,
    hvn_threshold: float = 1.5,    # HVN if volume > avg × threshold
    lvn_threshold: float = 0.5,    # LVN if volume < avg × threshold
) -> VolumeProfileResult:
    """
    Compute a volume profile for a given OHLCV DataFrame slice.

    Volume is distributed across price bins using the bar's range,
    weighted proportionally by where within the candle the volume lies.
    """
    p_min = df["low"].min()
    p_max = df["high"].max()

    if p_max == p_min:
        p_max = p_min * 1.001

    edges = np.linspace(p_min, p_max, price_bins + 1)
    bin_mids = (edges[:-1] + edges[1:]) / 2
    vol_bins = np.zeros(price_bins)

    for _, row in df.iterrows():
        lo, hi   = row["low"], row["high"]
        vol      = row["volume"]
        bar_rng  = hi - lo or hi * 0.0001

        for b in range(price_bins):
            b_lo = edges[b]
            b_hi = edges[b + 1]
            overlap = min(hi, b_hi) - max(lo, b_lo)
            if overlap > 0:
                vol_bins[b] += vol * overlap / bar_rng

    # ── POC (highest volume bin) ──────────────────────────────────────────────
    poc_idx  = int(np.argmax(vol_bins))
    vpoc     = float(bin_mids[poc_idx])

    # ── Value Area (70% of total volume centered on POC) ─────────────────────
    total_vol    = float(vol_bins.sum())
    target_vol   = total_vol * value_area_pct
    va_vol       = vol_bins[poc_idx]
    lo_idx       = poc_idx
    hi_idx       = poc_idx

    while va_vol < target_vol:
        add_lo = vol_bins[lo_idx - 1] if lo_idx > 0 else 0
        add_hi = vol_bins[hi_idx + 1] if hi_idx < price_bins - 1 else 0

        if add_lo == 0 and add_hi == 0:
            break
        if add_hi >= add_lo:
            hi_idx  += 1
            va_vol  += add_hi
        else:
            lo_idx  -= 1
            va_vol  += add_lo

    vah = float(edges[hi_idx + 1])
    val = float(edges[lo_idx])

    # ── HVN / LVN ────────────────────────────────────────────────────────────
    avg_bin_vol = vol_bins.mean()
    hvn_levels  = [float(bin_mids[b]) for b in range(price_bins)
                   if vol_bins[b] > avg_bin_vol * hvn_threshold]
    lvn_levels  = [float(bin_mids[b]) for b in range(price_bins)
                   if 0 < vol_bins[b] < avg_bin_vol * lvn_threshold]

    return VolumeProfileResult(
        vpoc=vpoc,
        vah=vah,
        val=val,
        total_volume=total_vol,
        value_area_pct=value_area_pct,
        price_bins=bin_mids,
        volume_per_bin=vol_bins,
        hvn_levels=hvn_levels,
        lvn_levels=lvn_levels,
    )


# ── Rolling VPOC series ───────────────────────────────────────────────────────

def rolling_vp(
    df: pd.DataFrame,
    window: int = 50,
    price_bins: int = 50,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """
    Compute rolling VPOC, VAH, VAL for every bar.
    Returns a DataFrame indexed like df with columns: vpoc, vah, val.
    """
    n = len(df)
    vpocs = np.full(n, np.nan)
    vahs  = np.full(n, np.nan)
    vals  = np.full(n, np.nan)

    for i in range(window, n):
        win = df.iloc[i - window: i + 1]
        try:
            vp = build_volume_profile(win, price_bins=price_bins,
                                      value_area_pct=value_area_pct)
            vpocs[i] = vp.vpoc
            vahs[i]  = vp.vah
            vals[i]  = vp.val
        except Exception:
            pass

    return pd.DataFrame({"vpoc": vpocs, "vah": vahs, "val": vals}, index=df.index)


# ── VPOC Migration ────────────────────────────────────────────────────────────

def detect_vpoc_migration(
    rolling_vp_df: pd.DataFrame,
    window: int = 10,
    flat_threshold_pct: float = 0.05,
) -> VPOCMigration:
    """
    Assess VPOC migration direction over the last `window` bars.

    Rising VPOC = institutional buyers pushing value higher (bullish)
    Falling VPOC = institutional sellers pressing value lower (bearish)
    Flat VPOC   = price acceptance at current level (consolidation)
    """
    vpoc_series = rolling_vp_df["vpoc"].dropna()
    if len(vpoc_series) < window:
        return VPOCMigration("FLAT", 0.0, window)

    recent = vpoc_series.iloc[-window:]
    start  = float(recent.iloc[0])
    end    = float(recent.iloc[-1])

    mag_pct = (end - start) / start * 100 if start > 0 else 0.0

    if abs(mag_pct) < flat_threshold_pct:
        direction = "FLAT"
    elif mag_pct > 0:
        direction = "RISING"
    else:
        direction = "FALLING"

    return VPOCMigration(direction=direction, magnitude_pct=mag_pct, bars=window)


# ── Session-based VP (London / NY / Asia) ────────────────────────────────────

def session_volume_profiles(
    df: pd.DataFrame,
    price_bins: int = 60,
    value_area_pct: float = 0.70,
) -> dict:
    """
    Split the DataFrame by trading session (UTC hours) and build a
    volume profile per session.

    Returns dict: {'london': VP, 'newyork': VP, 'asia': VP}
    Sessions:
        Asia:     00:00 – 08:00 UTC
        London:   07:00 – 16:00 UTC
        New York: 13:00 – 22:00 UTC  (overlaps London 13-16)
    """
    if not hasattr(df.index, 'hour'):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)

    hour = df.index.hour
    sessions = {
        "asia":     df[(hour >= 0)  & (hour < 8)],
        "london":   df[(hour >= 7)  & (hour < 16)],
        "newyork":  df[(hour >= 13) & (hour < 22)],
        "overlap":  df[(hour >= 13) & (hour < 16)],   # London/NY overlap – highest liquidity
    }

    results = {}
    for name, sdf in sessions.items():
        if len(sdf) >= 5:
            try:
                results[name] = build_volume_profile(sdf, price_bins, value_area_pct)
            except Exception:
                results[name] = None
        else:
            results[name] = None

    return results
