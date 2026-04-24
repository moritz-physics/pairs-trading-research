# CLAUDE.md — pairs-trading-research

## Project

Research project implementing cointegration-based pairs trading on US
equities. Produces analysis, figures, and a writeup — not a production
trading system. Deliberately honest about what does and doesn't work
over the evaluated period.

Owner: physics/AI master's student. Prioritize correctness and teaching
value. Prefer clear code over clever code.

## Relationship to Backtest-Engine

This repo depends on `Backtest-Engine` (installed as an editable
dependency at `../Backtest-Engine`). Key imports available:

    from backtester.data.loader import load_prices, load_ohlcv, to_returns
    from backtester.data.rates import load_risk_free_rate, to_daily_rate
    from backtester.backtest.engine import run_backtest, BacktestResult
    from backtester.costs.linear import LinearCost
    from backtester.metrics.performance import (
        sharpe_ratio, sortino_ratio, max_drawdown, drawdown_series,
        annualized_return, annualized_volatility, cagr, calmar_ratio,
        hit_rate, turnover,
    )

Engine contract: run_backtest(signal, returns, cost_model, signal_lag=1,
cash_rate=None) -> BacktestResult. Signal and returns are DataFrames
with matching indices and columns. signal_lag=1 is non-negotiable and
enforces anti-look-ahead.

DO NOT modify Backtest-Engine from this repo. If a framework change is
genuinely needed, raise it so we can add it to Backtest-Engine in a
separate commit with its own tests.

## Non-Negotiables (inherited from Backtest-Engine)

1. No look-ahead bias. All features causal.
2. No silent NaN handling.
3. Costs always modeled.
4. Reproducibility: seed all randomness.

## Domain-Specific Conventions

- Cointegration tests: use statsmodels Engle-Granger (coint) and
  Johansen (coint_johansen). Default significance level 0.05 unless
  stated.
- Hedge ratios: two approaches will be implemented — OLS (fixed) and
  Kalman filter (time-varying). Each lives in its own module.
- Z-score signals: entry at |z| > 2, exit at |z| < 0.5, hard stop at
  |z| > 4 (assume the pair broke and exit). These defaults are
  parameters, not magic numbers — all thresholds are explicit arguments.
- Universe: start with sector-clustered US large caps (S&P 500
  constituents). Cointegration is more likely within sectors than across.

## Code Style

- src/pairs_trading/ contains reusable logic: selection, hedge ratio
  estimators, signal generators. Small pure functions where possible.
- scripts/NN_description.py are runnable entry points that compose the
  library. Each produces at least one figure in results/.
- Tests in tests/ mirror the module structure. Mock any external data.

## Performance Awareness

- Prefer vectorized pandas/numpy. Loops only when unavoidable.
- When a computation is slow (>5s), profile before optimizing. Don't
  guess.
- If we end up needing numba or similar, we will justify it with a
  measurement.

## Known framework issues

Issues observed in `Backtest-Engine` from this repo. Do not fix them
here — fix in `Backtest-Engine` with its own tests. This section
documents active workarounds so future scripts know what to avoid.

- **Engine `cash_rate` is applied per-asset and summed (bug).** In
  `run_backtest`, the cash contribution is computed as
  `(1 - |position|) * cash_rate` *per asset*, then the row-sum across
  assets produces the portfolio return. For a multi-asset panel where
  gross exposure is less than the number of assets, this overcounts
  cash — e.g. a 2-asset panel with both positions at 0 earns
  `2 * rf_daily` instead of `1 * rf_daily`. Any strategy whose per-day
  `sum(|w_i|) < n_assets` is affected (which includes every partially
  invested or flat-portion strategy).

  *Current workaround (pairs scripts):* compound the DTB3 daily series
  directly for the cash-only benchmark row instead of routing zero-
  weights through `run_backtest`. The strategy and in-position
  benchmarks (static-spread, long-y B&H) are left on the engine because
  their gross exposure sums to the right figure for their own reporting
  — but the Sharpe of the long-y B&H row is slightly overstated because
  the engine applies cash to the zero-weight leg.

  *Tracking:* to be fixed in `Backtest-Engine` in the next session.

## References

- Gatev, Goetzmann, Rouwenhorst (2006), "Pairs Trading: Performance of
  a Relative-Value Arbitrage Rule"
- Vidyamurthy (2004), "Pairs Trading: Quantitative Methods and Analysis"
- Chan, "Algorithmic Trading" — Kalman filter pairs chapter