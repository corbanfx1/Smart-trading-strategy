"""
Smart Trading Framework – Main Entry Point
──────────────────────────────────────────
Runs a full multi-timeframe analysis pipeline and prints the setup signal.

Usage:
    python main.py                              # BTC-USD, 1h, 60d
    python main.py --symbol ETH-USD
    python main.py --symbol AAPL --interval 1h --period 90d
    python main.py --symbol BTC-USD --chart     # save chart.html
"""

from __future__ import annotations

import argparse
import sys

import config as cfg
from data.fetcher import fetch_multi_tf
from analysis.market_structure import detect_structure, mtf_bias
from analysis.order_blocks import detect_order_blocks
from analysis.fair_value_gaps import detect_fvgs
from analysis.liquidity import detect_liquidity_levels
from analysis.premium_discount import compute_pd_context
from analysis.order_flow import analyse_order_flow
from analysis.institutional_phase import detect_phase
from analysis.setup_scanner import scan_for_setups


def run_analysis(
    symbol: str = cfg.DEFAULT_SYMBOL,
    interval: str = cfg.DEFAULT_INTERVAL,
    period: str = cfg.DEFAULT_PERIOD,
    save_chart: bool = False,
    chart_path: str = "chart.html",
    verbose: bool = True,
) -> None:
    """
    Full analysis pipeline:
        1. Fetch multi-timeframe data
        2. Market structure (BOS/CHoCH/swings)
        3. Order blocks
        4. Fair value gaps
        5. Liquidity levels + sweeps
        6. Premium/Discount context
        7. Order flow (CVD/delta/absorption/imbalance)
        8. Institutional phase
        9. Setup scoring + signal
       10. Optional chart output
    """

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print(f"\n  Fetching {symbol} [{interval}, {period}] …")
    try:
        tf_data = fetch_multi_tf(symbol, base_interval=interval, period=period)
    except Exception as e:
        print(f"  [ERROR] Data fetch failed: {e}")
        sys.exit(1)

    df_1h    = tf_data[interval]
    df_4h    = tf_data["4h"]
    df_daily = tf_data["1d"]

    if verbose:
        print(f"  1H bars  : {len(df_1h)}")
        print(f"  4H bars  : {len(df_4h)}")
        print(f"  Daily bars: {len(df_daily)}")
        print(f"  Latest close: {df_1h['close'].iloc[-1]:.4f}")

    # ── 2. Market Structure ───────────────────────────────────────────────────
    print("\n  Running market structure analysis …")
    ms_result = detect_structure(df_1h, lookback=cfg.SWING_LOOKBACK)
    bias      = mtf_bias(df_4h, df_daily)

    if verbose:
        print(f"  Swing highs  : {len(ms_result.swing_highs)}")
        print(f"  Swing lows   : {len(ms_result.swing_lows)}")
        print(f"  Structure events: {len(ms_result.events)}")
        current_trend = int(ms_result.trend.iloc[-1])
        trend_lbl = "BULLISH" if current_trend == 1 else "BEARISH" if current_trend == -1 else "NEUTRAL"
        bias_lbl  = "BULLISH" if bias == 1 else "BEARISH" if bias == -1 else "NEUTRAL"
        print(f"  Current 1H trend : {trend_lbl}")
        print(f"  MTF Bias (4H+D)  : {bias_lbl}")

    # ── 3. Order Blocks ───────────────────────────────────────────────────────
    print("\n  Detecting order blocks …")
    obs = detect_order_blocks(df_1h, ms_result.events,
                              lookback=cfg.OB_LOOKBACK,
                              mitigation_pct=cfg.OB_MITIGATION_PCT)
    active = [o for o in obs if not o.mitigated]
    if verbose:
        print(f"  Total OBs: {len(obs)}  (active: {len(active)}, mitigated: {len(obs)-len(active)})")

    # ── 4. Fair Value Gaps ────────────────────────────────────────────────────
    print("\n  Detecting fair value gaps …")
    fvgs = detect_fvgs(df_1h, min_size_pct=cfg.FVG_MIN_SIZE_PCT)
    unfilled = [f for f in fvgs if not f.filled]
    if verbose:
        print(f"  Total FVGs: {len(fvgs)}  (unfilled: {len(unfilled)})")

    # ── 5. Liquidity ──────────────────────────────────────────────────────────
    print("\n  Detecting liquidity levels …")
    levels, sweeps = detect_liquidity_levels(
        df_1h,
        lookback=cfg.LIQ_LOOKBACK,
        eq_thresh_pct=cfg.EQ_LEVEL_THRESH_PCT,
    )
    unswept = [l for l in levels if not l.swept]
    if verbose:
        print(f"  Liquidity levels : {len(levels)}  (unswept: {len(unswept)})")
        print(f"  Sweeps detected  : {len(sweeps)}")

    # ── 6. Premium / Discount ─────────────────────────────────────────────────
    print("\n  Computing Premium/Discount context …")
    pd_ctx = compute_pd_context(df_1h, range_length=cfg.PD_RANGE_LENGTH)
    if verbose:
        print(f"  Zone             : {pd_ctx.zone}")
        print(f"  Equilibrium      : {pd_ctx.equilibrium:.4f}")
        print(f"  Range            : {pd_ctx.range_low:.4f} – {pd_ctx.range_high:.4f}")
        print(f"  Dist. from EQ    : {pd_ctx.pct_from_eq:+.2f}%")

    # ── 7. Order Flow ─────────────────────────────────────────────────────────
    print("\n  Analysing order flow …")
    of_report = analyse_order_flow(df_1h)
    latest_of = of_report.latest
    if verbose:
        print(f"  Delta (latest)   : {latest_of['delta']:+.0f}")
        print(f"  CVD slope        : {'POSITIVE ▲' if latest_of['cvd_slope_pos'] else 'NEGATIVE ▼'}")
        print(f"  CVD divergence   : {'bull' if latest_of['bull_div'] else 'bear' if latest_of['bear_div'] else 'none'}")
        print(f"  Absorption       : {'YES' if latest_of['absorption'] else 'no'}")
        print(f"  Delta flip       : {'bull ▲' if latest_of['bull_flip'] else 'bear ▼' if latest_of['bear_flip'] else '—'}")
        vpoc = latest_of.get("vpoc")
        if vpoc:
            print(f"  VPOC (rolling)   : {vpoc:.4f}")

    # ── 8. Institutional Phase ────────────────────────────────────────────────
    print("\n  Diagnosing institutional phase …")
    phase = detect_phase(
        df_1h, of_report, sweeps, ms_result.trend,
        lookback=cfg.PHASE_LOOKBACK,
    )
    if verbose:
        print(f"  Phase            : {phase.phase}")
        print(f"  Direction hint   : {'UP ▲' if phase.direction_hint == 1 else 'DOWN ▼' if phase.direction_hint == -1 else 'NEUTRAL'}")
        print(f"  Scores  accum={phase.accum_score}  manip={phase.manip_score}  "
              f"dist={phase.distrib_score}  trend={phase.trending_score}")

    # ── 9. Setup Signal ───────────────────────────────────────────────────────
    print("\n  Scanning for high-probability setup …")
    signal = scan_for_setups(
        df_1h, df_4h, df_daily,
        ms_result, obs, fvgs, levels, sweeps, pd_ctx, of_report, phase,
    )
    print(f"\n{signal.summary()}")

    # ── 10. Chart ─────────────────────────────────────────────────────────────
    if save_chart:
        try:
            from visualization.chart import build_chart, save_html
            chart_title = f"{symbol} | {interval} | Smart Trading Framework"
            fig = build_chart(
                df=df_1h,
                obs=obs,
                fvgs=fvgs,
                levels=levels,
                sweeps=sweeps,
                events=ms_result.events,
                pd_ctx=pd_ctx,
                of_report=of_report,
                signal=signal,
                title=chart_title,
            )
            save_html(fig, chart_path)
        except ImportError:
            print("  [WARN] plotly not installed – chart skipped.")

    return signal


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Smart Trading Framework – TradingView Chart Analysis"
    )
    parser.add_argument("--symbol",   default=cfg.DEFAULT_SYMBOL,   help="Ticker symbol (default: BTC-USD)")
    parser.add_argument("--interval", default=cfg.DEFAULT_INTERVAL,  help="Base timeframe (default: 1h)")
    parser.add_argument("--period",   default=cfg.DEFAULT_PERIOD,    help="Lookback period (default: 60d)")
    parser.add_argument("--chart",    action="store_true",            help="Save interactive chart to chart.html")
    parser.add_argument("--chart-path", default="chart.html",        help="Output path for chart HTML")
    parser.add_argument("--quiet",    action="store_true",            help="Suppress verbose output")
    args = parser.parse_args()

    run_analysis(
        symbol=args.symbol,
        interval=args.interval,
        period=args.period,
        save_chart=args.chart,
        chart_path=args.chart_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
