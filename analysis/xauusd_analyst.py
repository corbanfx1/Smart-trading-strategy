"""
XAUUSD Institutional Analyst
──────────────────────────────
Full pipeline runner and report generator for Gold (XAU/USD).

Pipeline:
  1.  Fetch multi-TF data  (1H base, 4H bias, Daily bias)
  2.  Market structure     (1H, 4H, Daily)
  3.  Order Blocks + FVG
  4.  Liquidity levels + sweeps
  5.  Premium / Discount context
  6.  Order Flow (CVD / delta / absorption)
  7.  Volume Profile (rolling VPOC/VAH/VAL, session VPs, migration)
  8.  Institutional Phase
  9.  Feature extraction (29-feature vector)
  10. p_xgb  (XGBoost-proxy)
  11. p_lstm (LSTM-proxy, long + short)
  12. Meta-model (HARD-GATE + SOFT-SCORE)
  13. Risk parameters (SL below OB, TP at liquidity / next key level)
  14. Formatted institutional brief

XAUUSD-specific context:
  • Key sessions: Asia (thin), London open (7 UTC), NY open (13 UTC),
    London/NY overlap (13–16 UTC) – peak liquidity, highest sweep probability
  • DXY inverse correlation flagged as a contextual note
  • Volatility regime: ATR-based
  • Key structural levels for Gold: round numbers ($100 increments),
    prior session highs/lows, FOMC/NFP reaction levels
"""

from __future__ import annotations

import sys
import math
from datetime import datetime, timezone
from typing import Optional, Dict

import numpy as np
import pandas as pd

from data.fetcher import fetch_multi_tf
from analysis.market_structure import detect_structure, htf_bias
from analysis.order_blocks import detect_order_blocks, active_obs
from analysis.fair_value_gaps import detect_fvgs, unfilled_fvgs
from analysis.liquidity import detect_liquidity_levels, recent_sweeps
from analysis.premium_discount import compute_pd_context, pd_series
from analysis.order_flow import analyse_order_flow
from analysis.volume_profile import (
    build_volume_profile, rolling_vp, detect_vpoc_migration,
    session_volume_profiles,
)
from analysis.institutional_phase import detect_phase
from analysis.ml_scoring import extract_features, compute_p_xgb, compute_p_lstm
from analysis.ml_meta import evaluate_both_directions, dominant_signal
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# XAUUSD CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
XAUUSD_SYMBOL  = "GC=F"          # Gold futures (yfinance)
XAUUSD_SPOT    = "XAUUSD=X"      # Spot (limited yfinance coverage)
PIP_VALUE      = 0.01            # 1 pip = $0.01 for XAU/USD
TICK_SIZE      = 0.10            # $0.10 minimum move
ATR_PERIOD     = 14
SESSION_HOURS  = {
    "Asia":    (0, 8),
    "London":  (7, 16),
    "NY":      (13, 22),
    "Overlap": (13, 16),
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _current_session(utc_hour: int) -> str:
    sessions = []
    for name, (start, end) in SESSION_HOURS.items():
        if start <= utc_hour < end:
            sessions.append(name)
    return " + ".join(sessions) if sessions else "Off-Hours"


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _round_price(p: float, tick: float = 0.10) -> str:
    """Format price to nearest tick."""
    rounded = round(p / tick) * tick
    return f"{rounded:.2f}"


def _nearest_round_number(price: float, step: float = 100.0) -> float:
    """Find nearest $100 / $50 round number for Gold."""
    return round(price / step) * step


def _bar_as_str(row: pd.Series, ts: pd.Timestamp) -> str:
    return (f"O={row['open']:.2f} H={row['high']:.2f} "
            f"L={row['low']:.2f} C={row['close']:.2f} "
            f"V={row['volume']:,.0f}")


def _pct(a: float, b: float) -> str:
    if b == 0:
        return "0.00%"
    return f"{(a - b) / b * 100:+.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

def _compute_risk_params(
    direction: str,
    entry: float,
    obs,
    fvgs,
    levels,
    pd_ctx,
    atr: float,
    last_idx: int,
) -> Dict:
    """
    Compute SL, TP1, TP2, TP3 and R:R.

    Stop Loss:
      LONG:  below the bottom of the nearest bullish OB (or 1.5× ATR)
      SHORT: above the top  of the nearest bearish OB (or 1.5× ATR)

    Take Profit:
      TP1: nearest unswept opposing liquidity level
      TP2: VPOC or equilibrium
      TP3: 3× ATR from entry or next key round number
    """
    act = active_obs(obs, last_idx)
    unfill = unfilled_fvgs(fvgs, last_idx)

    if direction == "LONG":
        # SL: below nearest bull OB bottom, or 1.5 ATR
        bull_obs = sorted([o for o in act if o.kind == "BULL"],
                          key=lambda o: abs(o.midpoint - entry))
        if bull_obs:
            sl = bull_obs[0].bottom - TICK_SIZE * 3
        else:
            sl = entry - atr * 1.5
        sl = min(sl, entry - atr)   # at least 1 ATR

        # TP: opposing liquidity (BSL levels)
        bsl_levels = sorted(
            [l for l in levels if not l.swept and l.kind in ("BSL", "EQH") and l.price > entry],
            key=lambda l: l.price
        )
        tp1 = bsl_levels[0].price - TICK_SIZE if bsl_levels else entry + atr * 2
        tp2 = bsl_levels[1].price - TICK_SIZE if len(bsl_levels) > 1 else entry + atr * 3
        tp3 = _nearest_round_number(entry + atr * 4, 50)
        if tp3 <= tp2:
            tp3 = tp2 + atr * 1.5

    else:  # SHORT
        bear_obs = sorted([o for o in act if o.kind == "BEAR"],
                          key=lambda o: abs(o.midpoint - entry))
        if bear_obs:
            sl = bear_obs[0].top + TICK_SIZE * 3
        else:
            sl = entry + atr * 1.5
        sl = max(sl, entry + atr)

        ssl_levels = sorted(
            [l for l in levels if not l.swept and l.kind in ("SSL", "EQL") and l.price < entry],
            key=lambda l: l.price, reverse=True
        )
        tp1 = ssl_levels[0].price + TICK_SIZE if ssl_levels else entry - atr * 2
        tp2 = ssl_levels[1].price + TICK_SIZE if len(ssl_levels) > 1 else entry - atr * 3
        tp3 = _nearest_round_number(entry - atr * 4, 50)
        if tp3 >= tp2:
            tp3 = tp2 - atr * 1.5

    risk   = abs(entry - sl)
    reward1 = abs(entry - tp1)
    reward2 = abs(entry - tp2)
    rr1    = reward1 / risk if risk > 0 else 0.0
    rr2    = reward2 / risk if risk > 0 else 0.0

    return {
        "entry": entry,
        "sl":    sl,
        "tp1":   tp1,
        "tp2":   tp2,
        "tp3":   tp3,
        "risk_pts":  risk,
        "rr1":   rr1,
        "rr2":   rr2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_xauusd_analysis(
    symbol: str = XAUUSD_SYMBOL,
    interval: str = "1h",
    period: str = "60d",
    verbose: bool = True,
) -> str:
    """
    Execute the full institutional XAUUSD analysis and return a formatted brief.
    """

    now_utc = datetime.now(timezone.utc)
    session = _current_session(now_utc.hour)
    lines = []

    def hdr(text: str, width: int = 70):
        lines.append("═" * width)
        lines.append(f"  {text}")
        lines.append("═" * width)

    def sub(text: str):
        lines.append(f"  ── {text}")

    def row(label: str, value: str, pad: int = 22):
        lines.append(f"  {label:<{pad}}: {value}")

    # ── 0. Header ─────────────────────────────────────────────────────────────
    hdr(f"XAUUSD  INSTITUTIONAL ANALYSIS  |  {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    row("Symbol",  symbol)
    row("Session", session)
    row("Base TF", interval)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    lines.append("")
    sub("Fetching data …")
    try:
        tf_data = fetch_multi_tf(symbol, base_interval=interval, period=period)
    except Exception as e:
        lines.append(f"  [ERROR] {e}")
        return "\n".join(lines)

    df_1h    = tf_data[interval]
    df_4h    = tf_data["4h"]
    df_daily = tf_data["1d"]
    price    = float(df_1h["close"].iloc[-1])
    last_bar = df_1h.iloc[-1]
    n        = len(df_1h)

    row("Latest close", f"${price:,.2f}")
    row("Latest bar",   _bar_as_str(last_bar, df_1h.index[-1]))
    row("1H bars",      str(n))
    row("4H bars",      str(len(df_4h)))
    row("Daily bars",   str(len(df_daily)))

    atr = _atr(df_1h)
    row("ATR(14) 1H",   f"${atr:.2f}  ({atr/price*100:.2f}%)")

    # ── 2. Market structure ───────────────────────────────────────────────────
    lines.append("")
    sub("Market Structure")
    ms_1h    = detect_structure(df_1h,    lookback=cfg.SWING_LOOKBACK)
    ms_4h    = detect_structure(df_4h,    lookback=cfg.SWING_LOOKBACK)
    ms_daily = detect_structure(df_daily, lookback=cfg.SWING_LOOKBACK)

    t1h    = int(ms_1h.trend.iloc[-1])    if len(ms_1h.trend)    > 0 else 0
    t4h    = int(ms_4h.trend.iloc[-1])    if len(ms_4h.trend)    > 0 else 0
    tdaily = int(ms_daily.trend.iloc[-1]) if len(ms_daily.trend) > 0 else 0

    _bias = lambda v: "BULLISH ▲" if v > 0 else "BEARISH ▼" if v < 0 else "NEUTRAL ●"
    row("Daily trend",  _bias(tdaily))
    row("4H trend",     _bias(t4h))
    row("1H trend",     _bias(t1h))

    confluence = (t4h > 0 and tdaily >= 0) or (t4h < 0 and tdaily <= 0)
    row("MTF Confluence", "YES ✓" if confluence else "NO – mixed signals")

    recent_events_1h = [e for e in ms_1h.events if e.idx >= n - 10]
    if recent_events_1h:
        last_e = recent_events_1h[-1]
        row("Last Structure Event", f"{last_e.event_type}  @${last_e.price:,.2f}")
    if ms_1h.swing_highs:
        sh = ms_1h.swing_highs[-1]
        row("Last Swing High",  f"${sh.price:,.2f}  [{sh.timestamp.strftime('%m-%d %H:%M')}]")
    if ms_1h.swing_lows:
        sl = ms_1h.swing_lows[-1]
        row("Last Swing Low",   f"${sl.price:,.2f}  [{sl.timestamp.strftime('%m-%d %H:%M')}]")

    # ── 3. Order Blocks ───────────────────────────────────────────────────────
    lines.append("")
    sub("Order Blocks")
    obs = detect_order_blocks(df_1h, ms_1h.events)
    act = active_obs(obs, n - 1)
    bull_act = [o for o in act if o.kind == "BULL"]
    bear_act = [o for o in act if o.kind == "BEAR"]
    row("Active Bull OBs",  str(len(bull_act)))
    row("Active Bear OBs",  str(len(bear_act)))

    tol = price * 0.005
    near_ob = [o for o in act if o.bottom - tol <= price <= o.top + tol]
    if near_ob:
        for o in near_ob[:3]:
            row(f"  Price in {o.kind} OB",
                f"${o.bottom:.2f} – ${o.top:.2f}  "
                f"({_pct(price, o.midpoint)} from mid)")

    # ── 4. Fair Value Gaps ────────────────────────────────────────────────────
    lines.append("")
    sub("Fair Value Gaps")
    fvgs = detect_fvgs(df_1h)
    unfill = unfilled_fvgs(fvgs, n - 1)
    bull_fvg = [f for f in unfill if f.kind == "BULL"]
    bear_fvg = [f for f in unfill if f.kind == "BEAR"]
    row("Unfilled Bull FVGs", str(len(bull_fvg)))
    row("Unfilled Bear FVGs", str(len(bear_fvg)))
    near_fvg = [f for f in unfill if f.bottom - tol <= price <= f.top + tol]
    if near_fvg:
        for f in near_fvg[:2]:
            row(f"  Price in {f.kind} FVG",
                f"${f.bottom:.2f} – ${f.top:.2f}  ({f.size_pct:.2f}%)")

    # ── 5. Liquidity ──────────────────────────────────────────────────────────
    lines.append("")
    sub("Liquidity Levels")
    levels, sweeps = detect_liquidity_levels(df_1h)
    unswept_bsl = [l for l in levels if not l.swept and l.kind in ("BSL", "EQH") and l.price > price]
    unswept_ssl = [l for l in levels if not l.swept and l.kind in ("SSL", "EQL") and l.price < price]
    row("Unswept BSL above",  str(len(unswept_bsl)))
    row("Unswept SSL below",  str(len(unswept_ssl)))

    if unswept_bsl:
        nearest_bsl = min(unswept_bsl, key=lambda l: l.price)
        row("Nearest BSL",   f"${nearest_bsl.price:.2f}  ({_pct(nearest_bsl.price, price)} from price)")
    if unswept_ssl:
        nearest_ssl = max(unswept_ssl, key=lambda l: l.price)
        row("Nearest SSL",   f"${nearest_ssl.price:.2f}  ({_pct(nearest_ssl.price, price)} from price)")

    rswps = recent_sweeps(sweeps, n_bars=6)
    if rswps:
        sw = rswps[-1]
        row("Recent Sweep",  f"{sw.kind}  @${sw.price:.2f}  [{df_1h.index[sw.idx].strftime('%m-%d %H:%M')}]")
    else:
        row("Recent Sweep",  "None detected")

    # ── 6. Premium / Discount ─────────────────────────────────────────────────
    lines.append("")
    sub("Premium / Discount")
    pd_ctx = compute_pd_context(df_1h)
    pd_ser = pd_series(df_1h)
    row("Range",        f"${pd_ctx.range_low:.2f} – ${pd_ctx.range_high:.2f}  "
                        f"(${pd_ctx.range_high - pd_ctx.range_low:.2f} span)")
    row("Equilibrium",  f"${pd_ctx.equilibrium:.2f}")
    row("75% Premium",  f"${pd_ctx.p75:.2f}")
    row("25% Discount", f"${pd_ctx.d25:.2f}")
    row("Current Zone", f"{pd_ctx.zone}  ({pd_ctx.pct_from_eq:+.2f}% from EQ)")

    # ── 7. Order Flow ─────────────────────────────────────────────────────────
    lines.append("")
    sub("Order Flow  (CVD / Delta)")
    of_report = analyse_order_flow(df_1h)
    of_latest = of_report.latest

    cvd_slope_pos = of_latest["cvd_slope_pos"]
    row("CVD (cumulative)", f"{of_latest['cvd']:+,.0f}")
    row("CVD Slope",        "POSITIVE ▲ – buying pressure" if cvd_slope_pos
                            else "NEGATIVE ▼ – selling pressure")
    row("CVD Divergence",   ("BULL DIV ↑ – hidden strength" if of_latest["bull_div"]
                              else "BEAR DIV ↓ – hidden weakness" if of_latest["bear_div"]
                              else "None"))
    row("Delta (bar)",      f"{of_latest['delta']:+,.0f}")
    row("Delta Flip",       ("BULL FLIP ▲" if of_latest["bull_flip"]
                              else "BEAR FLIP ▼" if of_latest["bear_flip"]
                              else "None"))
    row("Absorption",       "DETECTED ⚠" if of_latest["absorption"] else "No")
    row("Bull Imbalance",   "YES" if of_latest["bull_imb"] else "No")
    row("Bear Imbalance",   "YES" if of_latest["bear_imb"] else "No")

    # ── 8. Volume Profile ─────────────────────────────────────────────────────
    lines.append("")
    sub("Volume Profile")
    vp_window = min(cfg.PD_RANGE_LENGTH, n - 1)
    vp = build_volume_profile(df_1h.iloc[-vp_window:])
    rvp = rolling_vp(df_1h, window=vp_window)
    vp_mig = detect_vpoc_migration(rvp)

    row("VPOC",            f"${vp.vpoc:.2f}  ({_pct(price, vp.vpoc)} from price)")
    row("VAH (70%)",       f"${vp.vah:.2f}")
    row("VAL (70%)",       f"${vp.val:.2f}")
    row("Price vs VA",     vp.price_position(price))
    row("VPOC Migration",  f"{vp_mig.direction}  ({vp_mig.magnitude_pct:+.2f}% / {vp_mig.bars}bars)")

    if vp.hvn_levels:
        nearest_hvn = vp.nearest_hvn(price)
        row("Nearest HVN",  f"${nearest_hvn:.2f}  ({_pct(nearest_hvn, price)} from price)")
    if vp.lvn_levels:
        nearest_lvn = vp.nearest_lvn(price)
        row("Nearest LVN",  f"${nearest_lvn:.2f}  ({_pct(nearest_lvn, price)} from price)")

    # Session VPs
    sess_vps = session_volume_profiles(df_1h)
    for sess_name in ("london", "newyork", "overlap"):
        svp = sess_vps.get(sess_name)
        if svp:
            row(f"  {sess_name.capitalize()} VPOC",
                f"${svp.vpoc:.2f}  |  VA ${svp.val:.2f}–${svp.vah:.2f}")

    # ── 9. Institutional Phase ────────────────────────────────────────────────
    lines.append("")
    sub("Institutional Phase")
    phase = detect_phase(df_1h, of_report, sweeps, ms_1h.trend)
    row("Phase",           phase.phase)
    row("Direction Hint",  "UP ▲" if phase.direction_hint == 1
                           else "DOWN ▼" if phase.direction_hint == -1
                           else "NEUTRAL")
    row("Scores",          f"accum={phase.accum_score}  manip={phase.manip_score}  "
                           f"dist={phase.distrib_score}  trend={phase.trending_score}")

    # ── 10. XAUUSD-specific context ───────────────────────────────────────────
    lines.append("")
    sub("XAUUSD Contextual Notes")
    round_lvl = _nearest_round_number(price, 100)
    row("Nearest $100 level",  f"${round_lvl:.2f}  ({_pct(round_lvl, price)} from price)")
    row("50-handle level",     f"${_nearest_round_number(price, 50):.2f}")
    row("Session",             f"{session}  (peak liquidity: London/NY overlap 13–16 UTC)")
    row("Volatility",          f"ATR=${atr:.2f}  ({'HIGH' if atr > 20 else 'NORMAL' if atr > 8 else 'LOW'})")

    # ── 11. ML Feature Vector ─────────────────────────────────────────────────
    lines.append("")
    sub("ML Feature Vector  (29 features)")
    fv = extract_features(
        df_1h, df_4h, df_daily,
        ms_1h, ms_4h, ms_daily,
        obs, fvgs, levels, sweeps, pd_ctx, of_report,
        vp, vp_mig, phase,
    )
    fv_arr = fv.as_array()
    row("trend_1h / 4h / D",  f"{fv.trend_1h:+.0f} / {fv.trend_4h:+.0f} / {fv.trend_daily:+.0f}")
    row("mtf_confluence",     f"{fv.mtf_confluence:.0f}")
    row("choch_recency",      f"{fv.choch_recency:.3f}")
    row("bos_recency",        f"{fv.bos_recency:.3f}")
    row("pd_encoded",         f"{fv.pd_encoded:+.0f}  (−2=DPrem … +2=DDisc)")
    row("dist_from_eq_pct",   f"{fv.dist_from_eq_pct:+.2f}%")
    row("cvd_slope_norm",     f"{fv.cvd_slope_norm:+.3f}")
    row("delta_flip",         f"{fv.delta_flip:+.0f}  (+1=bull, −1=bear)")
    row("sweep_recency",      f"{fv.sweep_recency:.3f}  dir={fv.sweep_direction:+.0f}")
    row("price_in_bull_ob",   f"{fv.price_in_bull_ob:.0f}")
    row("price_in_bear_ob",   f"{fv.price_in_bear_ob:.0f}")
    row("dist_from_vpoc",     f"{fv.dist_from_vpoc_pct:+.2f}%")
    row("price_vs_vah",       f"{fv.price_vs_vah:+.0f}  (+1=above, −1=below)")
    row("vpoc_migration",     f"{fv.vpoc_migration:+.0f}  (+1=rising, −1=falling)")
    row("phase_encoded",      f"{fv.phase_encoded:+.1f}")

    # ── 12. p_xgb  ────────────────────────────────────────────────────────────
    lines.append("")
    sub("p_xgb  (XGBoost-proxy  |  10-tree ensemble)")
    p_xgb = compute_p_xgb(fv)
    row("p_xgb (bull prob)",  f"{p_xgb:.4f}  ({'BULLISH' if p_xgb > 0.55 else 'BEARISH' if p_xgb < 0.45 else 'NEUTRAL'})")
    row("p_xgb confidence",   f"{'HIGH' if abs(p_xgb - 0.5) > 0.20 else 'MODERATE' if abs(p_xgb - 0.5) > 0.10 else 'LOW'}")

    # ── 13. p_lstm ────────────────────────────────────────────────────────────
    lines.append("")
    sub("p_lstm  (LSTM-proxy  |  8-bar sequential pattern)")
    p_lstm_long  = compute_p_lstm(df_1h, ms_1h, sweeps, of_report, obs, fvgs, pd_ser, "LONG")
    p_lstm_short = compute_p_lstm(df_1h, ms_1h, sweeps, of_report, obs, fvgs, pd_ser, "SHORT")
    row("p_lstm LONG  (seq match)",  f"{p_lstm_long:.4f}")
    row("p_lstm SHORT (seq match)",  f"{p_lstm_short:.4f}")
    row("Sequence bias",             ("LONG  ▲" if p_lstm_long > p_lstm_short
                                       else "SHORT ▼" if p_lstm_short > p_lstm_long
                                       else "NEUTRAL"))

    # ── 14. Meta-Model ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("═" * 70)
    lines.append("  META-MODEL  |  HARD-GATE + SOFT-SCORE")
    lines.append("═" * 70)

    long_meta, short_meta = evaluate_both_directions(fv, p_xgb, p_lstm_long, p_lstm_short)
    best = dominant_signal(long_meta, short_meta)

    for meta in [long_meta, short_meta]:
        lines.append(f"\n  [{meta.direction}]")
        lines.extend(meta.summary_lines())

    # ── 15. Risk Parameters ───────────────────────────────────────────────────
    lines.append("")
    lines.append("═" * 70)
    lines.append("  TRADE SETUP  &  RISK MANAGEMENT")
    lines.append("═" * 70)

    if best.gate_open and best.conviction != "NO_TRADE":
        rp = _compute_risk_params(
            best.direction, price, obs, fvgs, levels, pd_ctx, atr, n - 1
        )
        lines.append(f"\n  Direction     : {best.direction}")
        lines.append(f"  Conviction    : {best.conviction}  {best.signal_label}")
        lines.append(f"  Final Prob    : {best.final_prob:.3f}")
        lines.append(f"")
        lines.append(f"  Entry         : ${rp['entry']:,.2f}  (market / limit on OB/FVG retest)")
        lines.append(f"  Stop Loss     : ${rp['sl']:,.2f}  (risk ${rp['risk_pts']:.2f} / {rp['risk_pts']/price*100:.2f}%)")
        lines.append(f"  TP1 (1:1+)    : ${rp['tp1']:,.2f}  (R:R  {rp['rr1']:.2f}:1)")
        lines.append(f"  TP2           : ${rp['tp2']:,.2f}  (R:R  {rp['rr2']:.2f}:1)")
        lines.append(f"  TP3 (runner)  : ${rp['tp3']:,.2f}")
        lines.append(f"")
        lines.append(f"  Partial exit  : 50% at TP1, 30% at TP2, trail 20% to TP3")
        lines.append(f"  Invalidation  : Close {'below' if best.direction == 'LONG' else 'above'} "
                     f"${rp['sl']:,.2f} on 1H candle close")
    else:
        lines.append(f"\n  Dominant side : {best.direction}")
        lines.append(f"  Gate status   : {'OPEN' if best.gate_open else 'CLOSED'}  "
                     f"({best.gates_passed}/{best.gates_total} gates passed)")
        lines.append(f"  Signal        : {best.signal_label}")
        lines.append(f"  Action        : STAND ASIDE – conditions not met for institutional entry")
        if best.gates_total - best.gates_passed > 0:
            failed = [g for g in best.gates if not g.passed]
            lines.append(f"  Failing gates :")
            for g in failed:
                lines.append(f"    ✗ {g.name}: {g.reason}")

    # ── 16. Summary Table ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("═" * 70)
    lines.append("  ANALYSIS SUMMARY")
    lines.append("═" * 70)
    summary = [
        ("XAUUSD price",    f"${price:,.2f}"),
        ("MTF Bias",        f"{_bias(t4h)} / {_bias(tdaily)}"),
        ("1H Structure",    _bias(t1h)),
        ("P/D Zone",        pd_ctx.zone),
        ("Phase",           phase.phase),
        ("CVD Slope",       "POSITIVE ▲" if cvd_slope_pos else "NEGATIVE ▼"),
        ("VPOC",            f"${vp.vpoc:.2f}  [{vp_mig.direction}]"),
        ("p_xgb",           f"{p_xgb:.3f}"),
        ("p_lstm L / S",    f"{p_lstm_long:.3f} / {p_lstm_short:.3f}"),
        ("Meta – Long",     f"{long_meta.final_prob:.3f}  {long_meta.conviction}"),
        ("Meta – Short",    f"{short_meta.final_prob:.3f}  {short_meta.conviction}"),
        ("SIGNAL",          f"{best.direction}  {best.signal_label}"),
    ]
    for k, v in summary:
        lines.append(f"  {k:<22}: {v}")

    lines.append("")
    lines.append("  Disclaimer: Educational/research use only. Not financial advice.")
    lines.append("═" * 70)

    return "\n".join(lines)
