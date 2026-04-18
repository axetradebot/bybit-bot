# Future Work — Deferred High-Effort Improvements

This file tracks the high-effort recommendations from the Apr-2026 deep
strategy review that were intentionally deferred. They should be picked
up after the medium/low-effort changes have been validated by the
simulation suite.

The numbering matches the original review's "PnL Impact" table.

---

## 13. Liquidation-cascade contrarian strategy (HIGH effort, high reward)

**Idea.** When 1h liquidation volume crosses the 99th percentile of the
trailing 30-day distribution AND price has moved > 3 × ATR within the
hour, fade the move. Mean-reversion edge is real but execution is
delicate.

**Why deferred.**
- Needs a new strategy file plus careful entry-timing model
  (pivot inside the cascade window, not at the extreme).
- Requires CoinGlass + Bybit liquidation streams to be in agreement
  (the current `liq_volume_1h` indicator only carries Bybit data).
- Optimal SL is "beyond cascade extreme" — needs intra-cascade peak
  detection rather than a fixed ATR multiple.

**Implementation sketch.**
1. New `src/strategies/strategy_liq_cascade.py`.
2. Trigger: `liq_volume_1h > p99_30d` AND `abs(close - close[-12]) > 3 * atr_14`.
3. Entry: limit at the 30-min midpoint between the cascade extreme and
   the current price.
4. SL: cascade extreme ± 0.3 × ATR.
5. TP1: VWAP. TP2: 2 × R from entry.
6. Block in trending regimes (ADX > 30 + same-direction trend).
7. Backtest on the LUNA/3AC June-2022 and FTX Nov-2022 windows first.

---

## 14. Funding-rate carry / arbitrage strategy (HIGH effort, steady reward)

**Idea.** Predicted funding > 0.05 %/8h (or < -0.03 %/8h) is a
recurrent mean-reversion edge: the perp typically reverts toward the
index. Two flavours:
  a. **Naked** carry — short the perp when funding is extreme positive
     (or long when extreme negative), accepting the directional risk
     in exchange for collecting the funding payment.
  b. **Delta-neutral** arb — short perp + long spot (or vice versa)
     to harvest funding with hedged price risk.

**Why deferred.**
- Naked carry is straightforward to add but interacts heavily with the
  rest of the portfolio (correlated with mean-reversion strategies in
  positive-funding regimes).
- Delta-neutral arb requires Bybit Spot integration (not yet wired in
  `OrderManager`) plus careful collateral / leverage accounting.
- Fee math is sensitive: at maker entry + maker exit on both legs, the
  break-even funding rate is ~0.04 % per cycle on Bybit VIP0.

**Implementation sketch.**
1. New `src/strategies/strategy_funding_carry.py` for naked variant.
2. Add `predicted_funding_8h` extra to `indicators_5m.extras` from
   the Bybit `tickers` stream (already partially captured by
   `_predicted_funding` in the listener — promote it to a real column).
3. Entry trigger: `predicted_funding > 0.0005` AND price within ±1 ATR
   of VWAP (avoid entering after a momentum push).
4. Exit: at next funding fix OR if funding mean-reverts to < 0.0001.
5. For the arb variant: add a `BybitSpotOrderManager` mirror class,
   open spot leg first, then perp leg, with a strict ratio check
   before either side fills.

---

## 15. Microstructure feature pack (HIGH effort, lifts every strategy)

**Idea.** Recompute features from `raw_trades` tick data and surface
them on the 5m bar so every existing strategy can use them as filters
or confidence boosters. Key features:
- Top-of-book imbalance (rolling 30 s, 60 s, 5 min)
- Trade-direction imbalance (signed-volume EMA)
- Order-flow toxicity (VPIN bucket flow)
- Effective spread, mid-price drift over the bar
- Trade-size distribution (large vs small print imbalance)

**Why deferred.**
- `raw_trades` is a tick-level table; recomputing features for 3 years
  × 8 symbols at scale needs either a TimescaleDB continuous
  aggregate or a separate Spark/DuckDB pipeline.
- Storage is non-trivial (~50–150 GB depending on symbol set).
- Each feature needs an ablation A/B test before being added to a
  strategy — large work matrix.

**Implementation sketch.**
1. New `src/indicators/microstructure.py` with pure-numpy / numba
   computations against a chunked iterator over `raw_trades`.
2. Persist to a new `microstructure_5m` table (same PK as
   `indicators_5m`).
3. Expose via `INDICATOR_COLS` extension in `src/backtest/simulator.py`.
4. For each existing strategy, add an optional filter
   (e.g. `bb_squeeze` requires `tob_imbalance > 0.6` for longs) and
   re-run the ablation grid in `scripts/ablation_filters.py`.

---

## Other items deferred until medium/low-effort changes are validated

These were referenced in the review but are not "high effort" — they
will be easier (and less error-prone) to add once the v2 risk manager
and simulator are in place.

### Stat-arb pair (BTC/ETH cointegration)
Engle–Granger cointegration test on rolling 60d windows, Z-score
reversion entry on the residual. Adds an uncorrelated sleeve; needs
two simultaneous orders and careful per-leg accounting. Slot in after
the Bybit Spot integration arrives for #14.

### Cross-exchange basis arbitrage (Bybit perp vs Binance perp)
High alpha during stress events but needs Binance API plumbing,
cross-exchange latency monitoring, and a separate execution loop.
Schedule after the post-only maker work has stabilised live fees.

### Better SL/TP scaling via Garman–Klass / Yang–Zhang vol forecasts
ATR is backward-looking; replacing it with Yang–Zhang vol on
overnight/intraday components, or a Hawkes-process intraday vol
estimator, will tighten SLs in calm regimes and widen them in active
ones. Requires a vol-forecast module and a re-fit of every strategy's
SL/TP multipliers via the simulation suite.

### Bybit Portfolio Margin (PM) cross-margin pooled equity
Once equity > 25 k USD, switching from isolated-per-position to PM
frees ~30 % notional capacity. Requires verifying that all existing
risk-manager assumptions (per-position liq distance) translate into
the cross-margin world.

### Continuous-aggregate views in TimescaleDB
For year-3 query performance on `candles_5m` and `indicators_5m`, add
hourly + daily continuous aggregates for analytics. Pure-Ops change.

### Test suite
No `tests/` folder exists. Add at minimum a golden-master 1-week
backtest per strategy that asserts PnL equality on every commit, plus
unit tests for the new risk-manager primitives (`KellySizer`,
`PortfolioRiskGate`, `BayesianRegimeFilter`).
