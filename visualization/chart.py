"""
Chart Visualisation
────────────────────
Plotly-based charting for the Smart Trading Framework.

Renders:
  • Candlestick chart (main panel)
  • Order Block zones
  • Fair Value Gap zones
  • Liquidity levels (dashed lines)
  • Premium / Discount zones (shaded)
  • BOS / CHoCH labels
  • Sweep markers
  • CVD panel (sub-chart)
  • Volume panel (sub-chart)
  • Setup signal annotations
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

from analysis.order_blocks import OrderBlock
from analysis.fair_value_gaps import FVG
from analysis.liquidity import LiquidityLevel, LiquiditySweep
from analysis.premium_discount import PDContext
from analysis.market_structure import StructureEvent
from analysis.setup_scanner import SetupSignal
from analysis.order_flow import OrderFlowReport


# ── Color palette ─────────────────────────────────────────────────────────────
C_BULL        = "#26A69A"
C_BEAR        = "#EF5350"
C_OB_BULL     = "rgba(38,166,154,0.15)"
C_OB_BEAR     = "rgba(239,83,80,0.15)"
C_FVG_BULL    = "rgba(38,166,154,0.10)"
C_FVG_BEAR    = "rgba(239,83,80,0.10)"
C_PREMIUM     = "rgba(239,83,80,0.06)"
C_DISCOUNT    = "rgba(38,166,154,0.06)"
C_EQ          = "rgba(158,158,158,0.8)"
C_SWEEP_BULL  = "#64FFDA"
C_SWEEP_BEAR  = "#FF9800"
C_SIGNAL_LONG  = "#26A69A"
C_SIGNAL_SHORT = "#EF5350"
C_BG          = "#0D0D1A"
C_GRID        = "#1E1E2E"


def _require_plotly():
    if not _PLOTLY:
        raise ImportError("plotly is not installed. Run: pip install plotly")


def build_chart(
    df: pd.DataFrame,
    obs: Optional[List[OrderBlock]] = None,
    fvgs: Optional[List[FVG]] = None,
    levels: Optional[List[LiquidityLevel]] = None,
    sweeps: Optional[List[LiquiditySweep]] = None,
    events: Optional[List[StructureEvent]] = None,
    pd_ctx: Optional[PDContext] = None,
    of_report: Optional[OrderFlowReport] = None,
    signal: Optional[SetupSignal] = None,
    title: str = "Smart Trading Framework",
    show_volume: bool = True,
    show_cvd: bool = True,
    max_obs: int = 10,
    max_fvgs: int = 15,
    max_levels: int = 10,
) -> "go.Figure":
    """
    Build a full analysis chart.

    Parameters
    ----------
    df         : OHLCV DataFrame (1H or chosen entry TF)
    obs        : list of OrderBlock objects
    fvgs       : list of FVG objects
    levels     : list of LiquidityLevel objects
    sweeps     : list of LiquiditySweep objects
    events     : list of StructureEvent objects (BOS/CHoCH)
    pd_ctx     : PDContext for the current bar
    of_report  : OrderFlowReport for CVD panel
    signal     : SetupSignal to annotate
    """
    _require_plotly()

    n_rows = 1 + int(show_volume) + int(show_cvd)
    row_heights = [0.60]
    if show_volume:
        row_heights.append(0.20)
    if show_cvd:
        row_heights.append(0.20)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    x = df.index

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        increasing_line_color=C_BULL,
        decreasing_line_color=C_BEAR,
        increasing_fillcolor=C_BULL,
        decreasing_fillcolor=C_BEAR,
        name="Price",
        showlegend=False,
        line=dict(width=1),
    ), row=1, col=1)

    # ── Premium / Discount zones ──────────────────────────────────────────────
    if pd_ctx is not None:
        x0, x1 = x[0], x[-1]
        # Premium
        fig.add_shape(type="rect", x0=x0, x1=x1,
                      y0=pd_ctx.equilibrium, y1=pd_ctx.range_high,
                      fillcolor=C_PREMIUM, line=dict(width=0), layer="below", row=1, col=1)
        # Discount
        fig.add_shape(type="rect", x0=x0, x1=x1,
                      y0=pd_ctx.range_low, y1=pd_ctx.equilibrium,
                      fillcolor=C_DISCOUNT, line=dict(width=0), layer="below", row=1, col=1)
        # Equilibrium line
        fig.add_hline(y=pd_ctx.equilibrium, line_dash="dot",
                      line_color=C_EQ, line_width=1, row=1, col=1,
                      annotation_text="EQ", annotation_font_color=C_EQ)

    # ── Order Blocks ──────────────────────────────────────────────────────────
    if obs:
        displayed = [o for o in obs if not o.mitigated][-max_obs:]
        for ob in displayed:
            color = C_OB_BULL if ob.kind == "BULL" else C_OB_BEAR
            border = C_BULL   if ob.kind == "BULL" else C_BEAR
            ts = df.index[min(ob.idx_origin, len(df) - 1)]
            fig.add_shape(
                type="rect",
                x0=ts, x1=x[-1],
                y0=ob.bottom, y1=ob.top,
                fillcolor=color,
                line=dict(color=border, width=1, dash="dot"),
                row=1, col=1,
            )
            fig.add_annotation(
                x=ts, y=ob.top,
                text=f"OB {'▲' if ob.kind == 'BULL' else '▼'}",
                font=dict(color=C_BULL if ob.kind == "BULL" else C_BEAR, size=9),
                showarrow=False, xanchor="left", yanchor="bottom",
                row=1, col=1,
            )

    # ── Fair Value Gaps ───────────────────────────────────────────────────────
    if fvgs:
        displayed = [f for f in fvgs if not f.filled][-max_fvgs:]
        for fvg in displayed:
            color = C_FVG_BULL if fvg.kind == "BULL" else C_FVG_BEAR
            border = C_BULL    if fvg.kind == "BULL" else C_BEAR
            ts = df.index[min(fvg.idx, len(df) - 1)]
            fig.add_shape(
                type="rect",
                x0=ts, x1=x[-1],
                y0=fvg.bottom, y1=fvg.top,
                fillcolor=color,
                line=dict(color=border, width=0.5, dash="dash"),
                row=1, col=1,
            )

    # ── Liquidity levels ──────────────────────────────────────────────────────
    if levels:
        active_levels = [l for l in levels if not l.swept][-max_levels:]
        for lvl in active_levels:
            color = C_BEAR if lvl.kind in ("BSL", "EQH") else C_BULL
            label = lvl.kind
            ts = df.index[min(lvl.idx, len(df) - 1)]
            fig.add_shape(
                type="line",
                x0=ts, x1=x[-1],
                y0=lvl.price, y1=lvl.price,
                line=dict(color=color, width=1, dash="dash"),
                row=1, col=1,
            )
            fig.add_annotation(
                x=x[-1], y=lvl.price,
                text=f"  {label}",
                font=dict(color=color, size=8),
                showarrow=False, xanchor="left",
                row=1, col=1,
            )

    # ── Sweeps ────────────────────────────────────────────────────────────────
    if sweeps:
        for sw in sweeps:
            if sw.idx >= len(df):
                continue
            ts = df.index[sw.idx]
            color = C_SWEEP_BULL if sw.kind == "BULL_SWEEP" else C_SWEEP_BEAR
            symbol = "triangle-up" if sw.kind == "BULL_SWEEP" else "triangle-down"
            price  = df["low"].iloc[sw.idx] if sw.kind == "BULL_SWEEP" else df["high"].iloc[sw.idx]
            fig.add_trace(go.Scatter(
                x=[ts], y=[price],
                mode="markers+text",
                marker=dict(symbol=symbol, size=12, color=color),
                text=["SWEEP"],
                textposition="bottom center" if sw.kind == "BULL_SWEEP" else "top center",
                textfont=dict(size=8, color=color),
                showlegend=False,
            ), row=1, col=1)

    # ── BOS / CHoCH labels ────────────────────────────────────────────────────
    if events:
        for evt in events:
            if evt.idx >= len(df):
                continue
            ts = df.index[evt.idx]
            is_bull = "BULL" in evt.event_type
            is_choch = "CHOCH" in evt.event_type
            color  = "#64FFDA" if is_choch and is_bull else \
                     "#FF9800" if is_choch else \
                     C_BULL if is_bull else C_BEAR
            label  = evt.event_type.replace("_", " ")
            price  = df["high"].iloc[evt.idx] if is_bull else df["low"].iloc[evt.idx]
            ypos   = "top center" if is_bull else "bottom center"
            fig.add_annotation(
                x=ts, y=price,
                text=label,
                font=dict(color=color, size=9, family="monospace"),
                showarrow=True,
                arrowhead=2, arrowsize=0.8, arrowcolor=color,
                ay=-20 if is_bull else 20,
                row=1, col=1,
            )

    # ── Setup signal ──────────────────────────────────────────────────────────
    if signal and signal.direction != "NONE" and signal.is_high_prob:
        ts = signal.timestamp
        price = df["close"].iloc[-1]
        color = C_SIGNAL_LONG if signal.direction == "LONG" else C_SIGNAL_SHORT
        arrow_dir = "▲" if signal.direction == "LONG" else "▼"
        label = f"🎯 {signal.direction} [{signal.score}/{signal.max_score}]"
        fig.add_annotation(
            x=ts, y=price,
            text=label,
            font=dict(color="white", size=11, family="monospace"),
            bgcolor=color,
            bordercolor=color,
            showarrow=True,
            arrowhead=2, arrowcolor=color,
            ay=-40 if signal.direction == "LONG" else 40,
            row=1, col=1,
        )

    # ── Volume sub-chart ──────────────────────────────────────────────────────
    if show_volume:
        vol_row = 2
        vol_colors = [C_BULL if c >= o else C_BEAR
                      for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=x, y=df["volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
            marker_line_width=0,
        ), row=vol_row, col=1)
        fig.update_yaxes(title_text="Vol", row=vol_row, col=1)

    # ── CVD sub-chart ─────────────────────────────────────────────────────────
    if show_cvd and of_report is not None:
        cvd_row = 2 + int(show_volume)
        cvd = of_report.cvd
        slope = of_report.cvd_slope
        cvd_color = [C_BULL if s > 0 else C_BEAR for s in slope]
        fig.add_trace(go.Scatter(
            x=x, y=cvd,
            mode="lines",
            line=dict(color=C_BULL, width=1),
            name="CVD",
            showlegend=False,
        ), row=cvd_row, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color=C_EQ,
                      line_width=0.5, row=cvd_row, col=1)
        fig.update_yaxes(title_text="CVD", row=cvd_row, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(color="#BB86FC", size=14)),
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        font=dict(color="#BDBDBD", size=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=700 + 150 * (int(show_volume) + int(show_cvd)),
        margin=dict(l=60, r=120, t=50, b=40),
    )
    for row in range(1, n_rows + 1):
        fig.update_xaxes(
            showgrid=True, gridcolor=C_GRID, gridwidth=0.5,
            zeroline=False, row=row, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=C_GRID, gridwidth=0.5,
            zeroline=False, row=row, col=1,
        )

    return fig


def show(fig: "go.Figure") -> None:
    """Open the chart in the default browser."""
    _require_plotly()
    fig.show()


def save_html(fig: "go.Figure", path: str = "chart.html") -> None:
    """Save the chart to an HTML file."""
    _require_plotly()
    fig.write_html(path)
    print(f"Chart saved to {path}")
