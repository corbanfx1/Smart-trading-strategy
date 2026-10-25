# XAUUSD Institutional ML Scalper: critique, v1.2 changes, ML roadmap

Scope: `pine/XAUUSD_Institutional_ML_Scalper_v1_1_FIXED.pine` (reviewed) and `pine/XAUUSD_Institutional_ML_Scalper_v1_2.pine` (revised).
Objective chosen: positive expectancy and profit factor, not raw win rate.

**Status.** I could not compile Pine or run a backtest in this environment. Nothing below is a performance claim. v1.2 must be compiled in TradingView, then evaluated walk-forward (Strategy Tester, "Walk-forward" mode, several months of data) before you trust any number it prints.

---

## 1. Critique of v1.1

### 1.1 Logic bugs and design flaws (highest impact first)

| # | Finding | Why it matters |
|---|---|---|
| 1 | **"Exhaustion" regime is direction-blind.** `exhaustion` is true when RSI > 72 *or* < 28. `exhaustionLong` only checks `exhaustion`, a recent sweep, a CHOCH and `close > emaFast`. | A "reversal long" can fire at RSI 75 (buying a blow-off top). The setup is half-random in direction. |
| 2 | **The probability gate swamps the EV gate.** Break-even for a 2R:1R binary payoff is p ≈ 0.33 (plus costs). `minProbability` = 0.55 sits far above that, and `minExpectedR` = 0.15 is satisfied at p ≈ 0.39. | The 0.55 cut selects only the most extreme raw logistic outputs. The model learns a base rate near 0.3–0.4, so it rarely reaches 0.55 and trades on noise at the tail. |
| 3 | **Label ≠ trade.** Labels are "2R before 1R within 12 bars". Live trades scale out 25% at 1R, 50% at 2R and 25% at 3R, with a time exit. A timeout is labelled a loss, but live it realises a partial win or loss. | The model optimises a different payoff than the one you trade. The "EV = 3p − 1" formula is for a binary 2:1 bet, not this ladder. |
| 4 | **No break-even.** All three exits keep the original stop, so a trade that reaches +1R can still return to −1R. | It gives up the main benefit of scaling out and lowers expectancy. |
| 5 | **Cost is not in the backtest edge.** `commission_value = 0` and slippage is 10 ticks. The $0.40 cost only appears in the filter. With ATR near $2 and a 1.2–3.5 ATR stop, one round trip is about 0.1–0.25R. | A 5-minute scalper that pays about 0.2R per trade needs far more than a 55% hit rate. The Strategy Tester flatters it unless slippage and spread are set per broker. |
| 6 | **Overlapping, autocorrelated labels.** Every candidate bar is queued, so one setup can produce 3–5 near-identical samples. The "n = 50" warm-up is really about 10 independent events. | SGD over-fits clusters, and the confidence it implies is false. |
| 7 | **Feature set is thin and partly redundant.** Five features, linear, and the `regimeFeature` number is not a categorical variable. `deltaScore` (close-location × tick volume) is collinear with `momentum` and wick position. | There is no interaction between regime and features, no room-to-liquidity term, no time-of-day term and no HTF trend strength. |
| 8 | **Hand-scaled macro inputs.** DXY ROC/0.20, US10Y ROC/0.50 and VIX ROC/2.0 are arbitrary and non-stationary. 5-minute DXY and US10Y data have gaps and different trading hours. | The signal-to-noise ratio is low, and the scale changes with volatility. |
| 9 | **"Liquidity" is mostly noise.** The sweep test counts any 3-bar pivot (`swingLen = 3`) as a liquidity pool, on the same footing as prior-day high/low and the Asia range. | Most 5-minute 3-bar pivots are not stops that institutions target. All sweeps carry the same weight. |
| 10 | **Zones never invalidate.** FVGs and OBs stay valid for `zoneLife` bars even after price closes through them. Only the latest zone is tracked. | You can retest a dead zone. |
| 11 | **Risk controls are incomplete.** There is a daily loss limit, but no peak-to-trough drawdown throttle, no end-of-session flatten, no shock-bar filter and only three manual news slots. | The usual way scalpers fail is a bad news bar or a losing streak. |
| 12 | **Hit-rate expectations.** With the 1.2 ATR minimum stop, the 2R target is at least 2.4 ATR away, and a 12-bar random walk covers about 3.5 ATR in total. A random walk hits +2R before −1R about 33% of the time, and less with the time limit. | The setup layer has to supply real edge to clear costs. Treat the p ≥ 0.55 default as unrealistic. |

### 1.2 What v1.1 already does well

- Correct no-repaint pattern for HTF data (`[1]` plus `lookahead_on`) and for the daily and macro series.
- Labels mature before a new signal is scored, and walk-forward or frozen OOS modes exist.
- Entries fill at the next open with exits attached to the actual fill price.
- Daily loss limit is latched, so a later recovery does not reopen the gate.
- The code is honest in its comments: the scores are uncalibrated and the delta is a proxy.

---

## 2. What v1.2 changes

| Area | v1.2 |
|---|---|
| **Architecture** | The rule layer is the primary model (candidate generator). The ML layer is a meta-model that takes or skips each candidate and sizes it (meta-labelling). |
| **Exhaustion bug** | Split into `exhaustionBull` (RSI ≤ 30 within the last 8 bars) and `exhaustionBear` (RSI ≥ 70). Longs fire only after sell-side washouts, shorts only after buy-side blow-offs. |
| **Labels** | Each label is the realised net R of the actual ladder: 25% at 1R, 50% at 2R, 25% at 3R, break-even after TP1, time exit at market, and the sample's own cost subtracted. Ties resolve stop-first. |
| **Break-even** | Live strategy re-issues the TP2 and TP3 legs with the stop at entry once the TP1 leg fills. |
| **De-duplication** | At most one queued sample per side per `dedupeBars` (default 3). `minLabels` raised to 150. |
| **Learners** | Two online learners: a logistic model for P(net R > 0) and an NLMS regression for expected R. Each is global weights plus per-regime deviations with stronger L2 on the regime part, so rare regimes borrow strength from the global fit. |
| **Features** | 11 features: momentum, HTF trend strength in ATR units, confluence, z-scored macro, delta proxy, volatility state, ADX, room to the next major liquidity level in R, session quality, liquidity-pool quality (prior-day > Asia > swing) and candle close location. |
| **Calibration** | Ten-bin online calibration of the logistic score, shrunk toward the raw score with 15 pseudo-counts. |
| **EV** | Calibrated P combined with *empirical* average win and loss R (EV_p), and the regression EV. The gate uses `min(EV_p, EV_reg)`, so both models must agree. The p gate now defaults to 0.45 and is a floor, not the main filter. |
| **Skill monitor** | A running Brier score versus base-rate variance is shown on the dashboard. A negative skill figure means the model is not adding information, and you should not trade it live. |
| **Cost and room gates** | Skips trades where round-trip cost exceeds `maxCostR` (default 0.20R) or where a major level is closer than `minRoomR` (default 1R). |
| **Risk** | Optional confidence-scaled sizing (0.5x–1.5x of base risk, from ensemble EV). A drawdown throttle halves risk at 5% from peak and halts at 10%. |
| **Event safety** | Daily 08:30 ET and 14:00 ET release windows are blocked. A shock-bar filter (range > 3x ATR) stands aside for 6 bars. Positions are flattened at session end. |
| **Structure** | Mitigated FVGs and OBs are invalidated once price closes through them. Reversal setups only need the HTF not to be opposed, while trend setups still need full 15m + 1H alignment. |
| **Macro** | Each series is z-scored against its own 200-bar history before the weights apply. |

### Honest limitations of v1.2

- It is still online learning on a few hundred to a few thousand samples. Expect a noisy model, especially at first.
- It is untested. I could not compile it, so expect minor syntax fixes.
- Volume on spot and CFD gold is tick volume, so `deltaScore` is a weak proxy, not order flow.
- Defaults (`minProbability`, `minExpectedR`, `maxCostR`, `minRoomR`, stop band) are reasoned starting points, not optimised values.
- Strategy Tester `slippage` and `commission` in the `strategy()` header are fixed constants. Set them to your broker's real spread before reading any result.

---

## 3. Institutional approaches worth adding (roadmap)

Ordered by expected value for the effort, for a gold intraday desk-style system.

1. **Meta-labelling with a proper learner (offline).** Keep the rules as the primary signal and train gradient-boosted trees (LightGBM or XGBoost) to predict take or skip and a size. Use triple-barrier labels (profit-take, stop, vertical time barrier), sample weights by label uniqueness and purged, embargoed K-fold cross-validation. Export the result as thresholds or lookup tables, or run it off-platform and send signals by webhook.
2. **Regime model.** Replace the hand thresholds with a hidden Markov model or Markov-switching model on returns, realised volatility and volume features. Use the posterior regime probability as a feature and as a hard gate. For gold, this catches volatility clustering and session structure.
3. **Fractional differentiation and stationary features.** Use fractionally differentiated price and volatility-scaled returns instead of raw levels, so features keep memory without being non-stationary.
4. **Probability calibration and conformal gating.** Isotonic or Platt calibration, then conformal prediction to trade only when the interval excludes break-even. This is a stricter version of v1.2's binned calibration.
5. **Real order flow where available.** On futures (GC/MGC), use footprint or L2-derived features: delta, absorption, stacked imbalance and queue depletion. TradingView exposes some of these in Pine; check your plan and the current docs before relying on them.
6. **Cross-asset and macro signals.** Real yields (TIPS) and breakeven inflation instead of nominal US10Y, USDJPY and EURUSD as dollar-strength cross-checks, and rate-sensitive futures positioning. Gate macro weight on a rolling correlation with gold, because the DXY link breaks in risk-off regimes.
7. **Event-aware models.** Use a calendar API to model post-release behaviour explicitly (NFP, CPI, FOMC), either as a separate strategy or as a no-trade filter with a volatility-decay re-entry.
8. **Execution and position sizing.** Volatility-targeted sizing with fractional Kelly on *calibrated* EV, correlation-aware caps if you add other symbols, and an execution layer that works passive entries when the spread is wide.
9. **Validation discipline.** Combinatorial purged cross-validation, deflated Sharpe ratio and probability of backtest overfitting before putting any parameters live. Run at least three separate regimes (for example trending, ranging and high-volatility) out of sample.

---

## 4. Suggested validation checklist before trusting v1.2

1. Compile it, then set `slippage` and commission to your broker's real spread.
2. Run **Walk-forward** mode over at least 6–12 months of 5-minute data. Check the dashboard skill figure is positive and stable.
3. Run **Frozen OOS** with a cutoff roughly two-thirds through the sample. Only post-cutoff trades count.
4. Compare net profit factor and max drawdown with each new gate toggled off, one at a time, to confirm each earns its place.
5. Forward-test on a demo account before real capital. Treat all results as provisional.
