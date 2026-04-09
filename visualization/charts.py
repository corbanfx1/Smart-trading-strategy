"""
TradingView-Style Interactive Chart Generator
Multi-panel Plotly chart: Candlestick + Volume + RSI + MACD + ADX.
Setup annotations, S/R levels, EMA ribbon, Bollinger Bands, ensemble score bar.
"""
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategy.setup_detector import TradeSetup
from features.market_structure import StructureAnalysis
import config

logger = logging.getLogger(__name__)

# ── TradingView Dark Theme Palette ─────────────────────────────────────────────
TV = {
    "bg":       "#131722",
    "panel":    "#1e222d",
    "grid":     "#2a2e39",
    "text":     "#b2b5be",
    "text_hi":  "#d1d4dc",
    "bull":     "#26a69a",   # green candle
    "bear":     "#ef5350",   # red candle
    "wick":     "#787b86",
    "vol_bull": "#26a69a40",
    "vol_bear": "#ef535040",
    "ema_9":    "#e91e63",
    "ema_20":   "#f48fb1",
    "ema_50":   "#ff9800",
    "ema_200":  "#4caf50",
    "bb_fill":  "rgba(100,149,237,0.08)",
    "bb_line":  "rgba(100,149,237,0.6)",
    "support":  "#26a69a",
    "resist":   "#ef5350",
    "setup_l":  "#26a69a",
    "setup_s":  "#ef5350",
    "score":    "#ab47bc",
}


class ChartBuilder:
    """Build a TradingView-style multi-panel interactive Plotly chart."""

    def __init__(self, n_rows: int = 5) -> None:
        self.n_rows = n_rows

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        df:             pd.DataFrame,
        setups:         Optional[List[TradeSetup]]  = None,
        structure:      Optional[StructureAnalysis] = None,
        ensemble_df:    Optional[pd.DataFrame]      = None,
        last_n_bars:    int = 300,
        title:          str = "XAUUSD – ML Ensemble Analysis",
    ) -> go.Figure:
        """Return a fully-annotated Plotly Figure."""
        plot_df = df.iloc[-last_n_bars:].copy()

        fig = make_subplots(
            rows=self.n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            row_heights=[0.50, 0.12, 0.12, 0.13, 0.13],
            subplot_titles=("", "Volume", "RSI", "MACD", "ADX / Ensemble Score"),
        )

        self._add_candles(fig, plot_df)
        self._add_ema_ribbon(fig, plot_df)
        self._add_bollinger(fig, plot_df)
        self._add_volume(fig, plot_df)
        self._add_rsi(fig, plot_df)
        self._add_macd(fig, plot_df)
        self._add_adx_and_score(fig, plot_df, ensemble_df)

        if structure:
            self._add_sr_levels(fig, plot_df, structure)
            self._add_order_blocks(fig, plot_df, structure)
            self._add_fvgs(fig, plot_df, structure)

        if setups:
            self._add_setups(fig, plot_df, setups)

        self._apply_theme(fig, title)
        return fig

    def save(
        self,
        fig:      go.Figure,
        filename: str = "xauusd_analysis.html",
    ) -> str:
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        path = os.path.join(config.REPORTS_DIR, filename)
        fig.write_html(path, include_plotlyjs="cdn")
        logger.info("Chart saved → %s", path)
        return path

    # ── Panel builders ────────────────────────────────────────────────────────

    def _add_candles(self, fig: go.Figure, df: pd.DataFrame) -> None:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"],
                low=df["Low"],   close=df["Close"],
                increasing_line_color=TV["bull"],
                decreasing_line_color=TV["bear"],
                increasing_fillcolor=TV["bull"],
                decreasing_fillcolor=TV["bear"],
                line=dict(width=1),
                whiskerwidth=0,
                name="XAUUSD",
                showlegend=False,
            ),
            row=1, col=1,
        )

    def _add_ema_ribbon(self, fig: go.Figure, df: pd.DataFrame) -> None:
        specs = [("ema_9", TV["ema_9"], 1), ("ema_20", TV["ema_20"], 1),
                 ("ema_50", TV["ema_50"], 1.5), ("ema_200", TV["ema_200"], 2)]
        for col, colour, width in specs:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col],
                        mode="lines",
                        line=dict(color=colour, width=width),
                        name=col.upper(), opacity=0.8,
                        showlegend=True,
                    ),
                    row=1, col=1,
                )

    def _add_bollinger(self, fig: go.Figure, df: pd.DataFrame) -> None:
        if "bb_upper" not in df.columns:
            return
        fig.add_trace(
            go.Scatter(
                x=pd.concat([pd.Series(df.index), pd.Series(df.index[::-1])]),
                y=pd.concat([df["bb_upper"], df["bb_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor=TV["bb_fill"],
                line=dict(color="rgba(0,0,0,0)"),
                name="BB", showlegend=False,
            ),
            row=1, col=1,
        )
        for col, dash in [("bb_upper", "dot"), ("bb_lower", "dot"), ("bb_mid", "dash")]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col],
                        mode="lines",
                        line=dict(color=TV["bb_line"], width=1, dash=dash),
                        showlegend=False,
                    ),
                    row=1, col=1,
                )

    def _add_volume(self, fig: go.Figure, df: pd.DataFrame) -> None:
        if "Volume" not in df.columns:
            return
        colors = [
            TV["vol_bull"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
            else TV["vol_bear"]
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], marker_color=colors,
                   name="Volume", showlegend=False),
            row=2, col=1,
        )

    def _add_rsi(self, fig: go.Figure, df: pd.DataFrame) -> None:
        if "rsi_14" not in df.columns:
            return
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["rsi_14"],
                mode="lines", line=dict(color="#7e57c2", width=1.5),
                name="RSI(14)",
            ),
            row=3, col=1,
        )
        for level, colour, dash in [(70, TV["bear"], "dot"), (30, TV["bull"], "dot"),
                                     (50, TV["wick"], "dash")]:
            fig.add_hline(
                y=level, row=3, col=1,
                line=dict(color=colour, width=1, dash=dash),
                annotation_text=str(level),
                annotation_font=dict(color=TV["text"], size=9),
            )

    def _add_macd(self, fig: go.Figure, df: pd.DataFrame) -> None:
        if "macd" not in df.columns:
            return
        hist    = df.get("macd_hist", pd.Series(0, index=df.index))
        hcolors = [TV["bull"] if v >= 0 else TV["bear"] for v in hist]

        fig.add_trace(
            go.Bar(x=df.index, y=hist, marker_color=hcolors,
                   name="MACD Hist", showlegend=False, opacity=0.7),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"],
                       mode="lines", line=dict(color="#2196f3", width=1.2),
                       name="MACD"),
            row=4, col=1,
        )
        if "macd_signal" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["macd_signal"],
                           mode="lines", line=dict(color="#ff9800", width=1.2),
                           name="Signal"),
                row=4, col=1,
            )

    def _add_adx_and_score(
        self,
        fig:         go.Figure,
        df:          pd.DataFrame,
        ensemble_df: Optional[pd.DataFrame],
    ) -> None:
        if "adx_14" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["adx_14"],
                    mode="lines", line=dict(color="#ffb300", width=1.5),
                    name="ADX(14)",
                ),
                row=5, col=1,
            )
            fig.add_hline(y=25, row=5, col=1,
                          line=dict(color=TV["wick"], width=1, dash="dot"))

        if ensemble_df is not None and "confidence" in ensemble_df.columns:
            ens = ensemble_df.reindex(df.index).ffill()
            score_scaled = ens["confidence"] * 100
            score_colors = [
                TV["bull"] if d == "long" else TV["bear"] if d == "short" else TV["wick"]
                for d in ens.get("direction", pd.Series("neutral", index=ens.index))
            ]
            fig.add_trace(
                go.Bar(
                    x=df.index, y=score_scaled,
                    marker_color=score_colors, opacity=0.6,
                    name="Ensemble Score", showlegend=True,
                ),
                row=5, col=1,
            )

    def _add_sr_levels(
        self,
        fig:       go.Figure,
        df:        pd.DataFrame,
        structure: StructureAnalysis,
    ) -> None:
        for lvl in structure.sr_levels:
            colour = TV["support"] if lvl.kind == "support" else (
                TV["resist"] if lvl.kind == "resistance" else "#9e9e9e"
            )
            fig.add_hline(
                y=lvl.price, row=1, col=1,
                line=dict(color=colour, width=1, dash="dash"),
                annotation_text=f"{lvl.kind.capitalize()} {lvl.price:.0f} (×{lvl.touches})",
                annotation_position="right",
                annotation_font=dict(color=colour, size=9),
            )

    def _add_order_blocks(
        self, fig: go.Figure, df: pd.DataFrame, structure: StructureAnalysis
    ) -> None:
        x0, x1 = df.index[0], df.index[-1]
        for ob in structure.order_blocks[-8:]:
            colour = "rgba(38,166,154,0.12)" if ob.direction == "bullish" else "rgba(239,83,80,0.12)"
            fig.add_vrect(
                x0=ob.timestamp, x1=x1,
                fillcolor=colour, opacity=1,
                layer="below", line_width=0,
                annotation_text=f"{ob.direction[:4].upper()} OB",
                annotation_font=dict(size=8, color=TV["text"]),
                row=1, col=1,
            )

    def _add_fvgs(
        self, fig: go.Figure, df: pd.DataFrame, structure: StructureAnalysis
    ) -> None:
        for fvg in structure.fvgs[-6:]:
            if fvg.filled:
                continue
            colour = "rgba(38,166,154,0.07)" if fvg.direction == "bullish" else "rgba(239,83,80,0.07)"
            fig.add_hrect(
                y0=fvg.bottom, y1=fvg.top,
                fillcolor=colour, opacity=1,
                layer="below", line_width=0,
                annotation_text="FVG",
                annotation_font=dict(size=8, color=TV["text"]),
                row=1, col=1,
            )

    def _add_setups(
        self,
        fig:    go.Figure,
        df:     pd.DataFrame,
        setups: List[TradeSetup],
    ) -> None:
        for s in setups:
            if s.timestamp not in df.index:
                continue
            colour = TV["setup_l"] if s.direction == "long" else TV["setup_s"]
            symbol = "triangle-up" if s.direction == "long" else "triangle-down"
            y      = df.loc[s.timestamp, "Low"] * 0.999 if s.direction == "long" \
                     else df.loc[s.timestamp, "High"] * 1.001

            # Entry marker
            fig.add_trace(
                go.Scatter(
                    x=[s.timestamp], y=[y],
                    mode="markers+text",
                    marker=dict(symbol=symbol, size=14, color=colour),
                    text=[f"  {s.setup_score:.0f}"],
                    textfont=dict(color=colour, size=9),
                    textposition="middle right",
                    name=f"{s.direction.capitalize()} Setup",
                    showlegend=False,
                    hovertext=(
                        f"{s.setup_type}<br>"
                        f"Score: {s.setup_score:.1f}/100<br>"
                        f"Entry: {s.entry:.2f}<br>"
                        f"SL: {s.stop:.2f}<br>"
                        f"TP1: {s.tp1:.2f}  R:R {s.rr1:.1f}"
                    ),
                    hoverinfo="text",
                ),
                row=1, col=1,
            )

            # Stop / TP horizontal lines (dashed, short)
            ts_end = df.index[min(df.index.get_loc(s.timestamp) + 20, len(df) - 1)]
            for price, lc, label in [
                (s.stop, TV["bear"],   "SL"),
                (s.tp1,  "#4caf50",    "TP1"),
                (s.tp2,  "#81c784",    "TP2"),
            ]:
                fig.add_shape(
                    type="line",
                    x0=s.timestamp, x1=ts_end,
                    y0=price, y1=price,
                    line=dict(color=lc, width=1, dash="dot"),
                    row=1, col=1,
                )

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self, fig: go.Figure, title: str) -> None:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(color=TV["text_hi"], size=16),
                x=0.5,
            ),
            paper_bgcolor=TV["bg"],
            plot_bgcolor=TV["panel"],
            font=dict(color=TV["text"], family="'Roboto Mono', monospace"),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor=TV["grid"],
                font=dict(size=10),
            ),
            margin=dict(l=60, r=30, t=60, b=40),
            height=900,
        )
        for axis in fig.layout:
            if axis.startswith(("xaxis", "yaxis")):
                fig.layout[axis].update(
                    showgrid=True,
                    gridcolor=TV["grid"],
                    gridwidth=1,
                    zeroline=False,
                    tickfont=dict(color=TV["text"], size=9),
                    linecolor=TV["grid"],
                )
