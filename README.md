# XAUUSD Institutional-Grade ML Ensemble Trading System

Hybrid ML ensemble (XGBoost + LSTM + ARCH/GARCH) with a 7-layer rule-based Meta-Model filter for institutional-grade XAUUSD trade signal generation.

## Architecture

```
Data Pipeline (yfinance / synthetic fallback)
        │
        ▼
Feature Engineering
   ├── Technical: RSI, MACD, BB, ADX, Stoch, CCI, ATR, OBV, CVD, EMAs, VWAP
   ├── SMC:       BOS, CHoCH, Order Blocks, FVGs, Liquidity Sweeps, Equilibrium
   └── Volume:    Volume Profile, POC, VAH/VAL, HVN/LVN proximity
        │
        ▼
 ┌──────────────────────────────────────┐
 │          ML ENSEMBLE                 │
 │  XGBoost (55%)  +  LSTM (45%)       │
 │  Directional probability classifier │
 └──────────────────┬───────────────────┘
                    │
        ┌───────────▼──────────┐
        │   ARCH/GARCH(1,1)    │
        │   Volatility model   │
        │  Regime + VaR/CVaR   │
        └───────────┬──────────┘
                    │
        ┌───────────▼──────────────────────┐
        │     7-LAYER META-FILTER          │
        │  1. Model Agreement              │
        │  2. Ensemble Probability >=0.62  │
        │  3. GARCH Vol Regime             │
        │  4. CVD Confirmation             │
        │  5. SMC Structural Confluence    │
        │  6. MTF Alignment (2/3 TFs)      │
        │  7. Session Filter (Lon/NY)      │
        └───────────┬──────────────────────┘
                    │
        ┌───────────▼──────────────────────┐
        │      RISK MANAGER                │
        │  ATR-based SL/TP levels          │
        │  GARCH vol-scaled position size  │
        │  1% risk/trade, 3% daily DD cap  │
        │  R:R >= 2.5 enforcement          │
        └───────────┬──────────────────────┘
                    │
                TRADE SETUP
```

## Multi-Timeframe Analysis

| Timeframe | Role | Indicators |
|-----------|------|------------|
| **1D** | Macro trend + key S/R | EMA200, ADX, BOS/CHoCH, PDH/PDL, PWH/PWL, PMH/PML |
| **4H** | Intermediate structure | Order Blocks, FVGs, swing H/L, equilibrium |
| **1H** | Entry-level signals | Fine SMC structure, CVD, entry precision |

## Quick Start

```bash
pip install -r requirements.txt

# Full run (train + analysis + signal)
python main.py

# Skip retraining (use saved models)
python main.py --skip-train

# Fast mode (no volume profile)
python main.py --fast

# Live alert mode
python main.py --mode live
```

## Chart Context (2026-03-26)

From the provided TradingView charts:
- **Price**: ~4,434-4,437
- **1D**: Major downtrend from ~5,600; price near equilibrium zone; CVD -247K
- **4H**: Bearish trend with BOS; PDH/PDL/PWL visible; CVD -25K
- **1H**: CHoCH + BOS signals; supply zones overhead; EQL lows tested; CVD -27K
- **SMC Bias**: BEARISH - price in premium/equilibrium, sweep of equal lows visible
- **GARCH**: Elevated volatility (est. 35-40% ann.) - protective filter active

## Signal Output Example

```
==============================================================
  FINAL SIGNAL OUTPUT  |  2026-03-26 16:04 UTC
==============================================================
  SIGNAL: SHORT
  Confidence: 71.2%
  Ensemble P: 0.847
  MTF Score:  -0.667
  Confluences (5):
    - premium_zone
    - bear_order_block
    - liquidity_sweep_high
    - choch_bearish
    - bear_fvg
  Vol Regime: elevated (ann=22.1%)

  TRADE PARAMETERS:
    Entry      :   4437.000
    Stop Loss  :   4504.000  (67.000 pts)
    TP1 (50%) :   4336.500
    TP2 (100%):   4202.500  R:R 2.50
    Lots       :       1.49
    Risk USD   :  $1000.00
    Reward USD :  $3500.00
```

## File Structure

```
Smart-trading-strategy/
├── config/settings.py          # All parameters (models, risk, filters)
├── data/
│   ├── fetcher.py              # yfinance multi-TF data download
│   ├── preprocessor.py         # Label generation, walk-forward splits
│   └── synthetic.py            # GBM+GARCH synthetic data (offline testing)
├── features/
│   ├── technical.py            # RSI, MACD, BB, ADX, ATR, CVD, etc.
│   ├── smc.py                  # BOS, CHoCH, OBs, FVGs, liquidity sweeps
│   ├── volume_profile.py       # POC, VAH, VAL
│   └── builder.py              # Master feature combiner
├── models/
│   ├── xgboost_model.py        # Walk-forward XGBoost classifier
│   ├── lstm_model.py           # Stacked LSTM (TF optional)
│   └── garch_model.py          # AR(1)-GARCH(1,1) vol model
├── strategy/
│   ├── meta_filter.py          # 7-layer rule-based filter
│   └── risk_manager.py         # ATR SL/TP + vol-scaled position sizing
├── analysis/
│   └── mtf_analyzer.py         # Daily + 4H + 1H structural bias
├── execution/
│   └── trade_executor.py       # Paper/live execution + P&L tracking
├── main.py                     # Entry point
└── requirements.txt
```

## Risk Parameters

| Parameter | Value |
|-----------|-------|
| Risk per trade | 1% of account |
| Max daily drawdown | 3% |
| SL multiplier | 1.5 x ATR(14) |
| TP multiplier | 3.5 x ATR(14) |
| Min R:R | 2.5 |
| Partial TP | 50% at 1.5 R:R |
| Max open trades | 3 |
| Vol scaling | GARCH-adaptive |
