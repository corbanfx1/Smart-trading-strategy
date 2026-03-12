# Smart Trading Framework

Multi-timeframe TradingView chart analysis tool implementing an ICT / SMC methodology.

---

## Methodology

```
Daily / 4H  →  Bias (directional context)
1H          →  Timing & trigger
Entry TF    →  Precision entry (OB / FVG retest)
```

### Analysis Pipeline

| Layer | Components |
|---|---|
| **Bias** | Daily + 4H Higher Highs/Lows consensus |
| **Structure** | BOS (Break of Structure) · CHoCH (Change of Character) · Swing H/L |
| **POI** | Order Blocks (OB) · Fair Value Gaps (FVG) · Breaker Blocks |
| **Liquidity** | BSL/SSL · Equal Highs/Lows · Sweeps · Runs |
| **Context** | Premium / Discount / Equilibrium zones |
| **Order Flow** | CVD slope · CVD divergence · Delta flips · Absorption · Imbalances · VPOC |
| **Phase** | Accumulation · Manipulation · Distribution · Trending |
| **Signal** | 7-point scoring system (fires at ≥ 4/7) |

---

## Repository Structure

```
Smart-trading-strategy/
│
├── pine_scripts/
│   └── smart_trading_framework.pine   ← TradingView indicator (paste into Pine Editor)
│
├── analysis/
│   ├── market_structure.py            ← Swing detection, BOS, CHoCH, MTF bias
│   ├── order_blocks.py                ← OB detection + mitigation + breaker blocks
│   ├── fair_value_gaps.py             ← FVG detection + fill tracking + inverse FVG
│   ├── liquidity.py                   ← BSL/SSL/EQH/EQL + sweep/run detection
│   ├── premium_discount.py            ← P/D zone computation (bar-by-bar + latest)
│   ├── order_flow.py                  ← CVD, delta, absorption, imbalance, VPOC
│   ├── institutional_phase.py         ← Accumulation/Manipulation/Distribution scorer
│   └── setup_scanner.py               ← Aggregates all modules → SetupSignal
│
├── data/
│   └── fetcher.py                     ← yfinance OHLCV fetch + multi-TF resampler
│
├── visualization/
│   └── chart.py                       ← Plotly interactive chart builder
│
├── main.py                            ← CLI entry point
├── config.py                          ← All tunable parameters
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run analysis (prints setup signal to terminal)
python main.py --symbol BTC-USD --interval 1h --period 60d

# 3. Generate interactive HTML chart
python main.py --symbol ETH-USD --chart

# 4. Other examples
python main.py --symbol AAPL     --interval 1h  --period 90d --chart
python main.py --symbol GC=F     --interval 1h  --period 60d          # Gold futures
python main.py --symbol EURUSD=X --interval 1h  --period 30d         # Forex
```

---

## TradingView Pine Script

1. Open TradingView → Pine Editor (bottom panel)
2. Paste the contents of `pine_scripts/smart_trading_framework.pine`
3. Click **Add to chart**
4. Configure via the **Settings** panel (inputs are grouped by category)

### Indicator Features

| Feature | Description |
|---|---|
| **Dashboard** | Top-right info table: bias, structure, zone, CVD, phase, scores |
| **OB Zones** | Shaded boxes extending right until mitigated |
| **FVG Zones** | Shaded imbalance gaps |
| **BSL / SSL** | Dashed liquidity lines with EQH/EQL detection |
| **BOS / CHoCH** | Labelled with connecting dashed lines |
| **Sweeps** | `SWEEP BSL ⚡` / `SWEEP SSL ⚡` on the sweep bar |
| **CVD Background** | Subtle green/red tint showing CVD slope regime |
| **Setup Signal** | `🎯 LONG [6/7]` / `🎯 SHORT [5/7]` labels |
| **Alerts** | Configurable alerts for setups, CHoCH, sweeps |

---

## Setup Scoring (7 Conditions)

### Long Setup
1. MTF bias bullish (Daily >= 0 AND 4H bullish)
2. 1H structure bullish (CHoCH up or BOS up)
3. Price below equilibrium (discount zone)
4. Bullish OB or FVG within 1% of price
5. SSL swept recently (stop hunt complete)
6. Delta flip bullish (buying pressure returning)
7. CVD slope positive OR bullish CVD divergence

### Short Setup  *(mirror of above, inverted)*

Signal fires at >= 4 / 7 conditions met.  Score >= 5 = high conviction.

---

## Order Flow Concepts

### CVD (Cumulative Volume Delta)
Estimated per-bar:
```
buy_vol  = (close - low)  / (high - low) x volume
sell_vol = (high - close) / (high - low) x volume
delta    = buy_vol - sell_vol
CVD      = sum(delta)
```

| Signal | Meaning |
|---|---|
| CVD slope rising | Buyers in control |
| CVD slope falling | Sellers in control |
| Price down + CVD up | Hidden bullish divergence - potential reversal |
| Price up + CVD down | Hidden bearish divergence - potential reversal |
| Delta flip at POI | Momentum shift at key level - high-value entry trigger |

### Absorption vs Imbalance
| Pattern | Volume | Body | Interpretation |
|---|---|---|---|
| **Absorption** | High (> 1.5x avg) | Small (< 0.5x avg) | Institutional counter-trend accumulation |
| **Imbalance** | Moderate | Large (> 2x avg) | Institutional directional order flow |

---

## Institutional Phase Model

| Phase | Characteristics | Expected Move |
|---|---|---|
| **Accumulation** | Low vol, tight range, absorption, SSL sweeps, bull CVD div | Upside breakout |
| **Manipulation** | Liquidity sweep + CHoCH, spike volume + absorption | Reversal in sweep direction |
| **Distribution** | High vol, large range, bear imbalances, bear CVD div | Downside |
| **Trending** | Moderate vol, directional BOS chain, no trend change | Continuation |

---

## Configuration (`config.py`)

Key parameters to tune for different instruments / timeframes:

```python
SWING_LOOKBACK      = 10      # increase for higher TF charts
OB_LOOKBACK         = 15      # bars before BOS to search for OB
FVG_MIN_SIZE_PCT    = 0.05    # increase for volatile assets (crypto: 0.10)
EQ_LEVEL_THRESH_PCT = 0.10    # increase for assets with choppy wicks
PD_RANGE_LENGTH     = 50      # bars for premium/discount range
CVD_PERIOD          = 14      # CVD slope measurement period
MIN_LONG_SCORE      = 4       # lower to 3 for more signals (less quality)
MIN_SHORT_SCORE     = 4
```

---

## Disclaimer

This framework is for **educational and research purposes only**.
It does not constitute financial advice. Past signals are not indicative of future results.
Always manage risk with stop-losses and appropriate position sizing.