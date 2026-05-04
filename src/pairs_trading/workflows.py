"""Reusable end-to-end workflows for pair backtests.

Sessions 02 and 03 share the same OLS-hedge pipeline for the
benchmark row of their comparison tables.  Rather than duplicating
the logic, we extract it into :func:`run_pair_backtest_ols` and call
it from both scripts.  The function is deliberately narrow: OLS
hedge, rolling z, state-machine signal, backtest.  Benchmark curves
(static spread, long-y B&H, cash) stay in the scripts that need them.

The returned dict has exactly the six keys documented in
:func:`run_pair_backtest_ols` — no more, no less — so callers can
rely on a stable contract.
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from backtester.backtest.engine import BacktestResult, run_backtest
from backtester.costs.linear import LinearCost
from backtester.data.loader import load_prices, to_returns
from backtester.data.rates import load_risk_free_rate, to_daily_rate

from pairs_trading.hedge_ratio import ols_hedge_ratio
from pairs_trading.selection import compute_half_life
from pairs_trading.signals import (
    compute_live_spread,
    spread_position_to_asset_weights,
    zscore,
    zscore_signal,
)

MIN_Z_WINDOW = 20


class OLSPairBacktestOutput(TypedDict):
    """Return type of :func:`run_pair_backtest_ols`."""

    result: BacktestResult
    beta: float
    half_life: float
    weights: pd.DataFrame
    zscore: pd.Series
    pair: tuple[str, str]


def build_cash_rate(
    index: pd.DatetimeIndex,
    validation_start: str,
    validation_end: str,
) -> pd.Series:
    """DTB3 daily rate aligned to *index*, ffilled over non-trading days."""
    rf_annual = load_risk_free_rate(
        validation_start, validation_end, series="DTB3"
    )
    rf_daily = to_daily_rate(rf_annual, periods_per_year=252, method="simple")
    rf_daily = rf_daily.reindex(rf_daily.index.union(index)).ffill()
    rf_daily = rf_daily.reindex(index).bfill()
    rf_daily.name = "DTB3_daily"
    return rf_daily


def run_pair_backtest_ols(
    pair: tuple[str, str],
    selection_window: tuple[str, str],
    validation_window: tuple[str, str],
    cost_bps: float = 5.0,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
) -> OLSPairBacktestOutput:
    """Run the session-02 OLS-hedge pair backtest on a single pair.

    The hedge ratio beta is estimated on the selection window and
    FROZEN.  The rolling z-score window is 2 * half_life (floored at
    ``MIN_Z_WINDOW``).  The backtest runs on validation-window simple
    returns only.  All anti-look-ahead invariants enumerated in
    ``scripts/02_pair_backtest.py`` apply here unchanged.

    Parameters
    ----------
    pair : (y_ticker, x_ticker)
        First element is the "long-leg when long-spread" asset.
    selection_window : (start, end)
        ISO-date strings defining the OLS fit window.
    validation_window : (start, end)
        ISO-date strings defining the backtest window.
    cost_bps : float, default 5.0
        Linear cost per unit of |Δposition|, in basis points.
    z_entry, z_exit, z_stop : float
        State-machine thresholds.  Must satisfy
        ``0 < z_exit < z_entry < z_stop``.

    Returns
    -------
    OLSPairBacktestOutput
        TypedDict with keys:
        - ``result`` : :class:`backtester.backtest.engine.BacktestResult`
        - ``beta`` : frozen OLS hedge ratio (float)
        - ``half_life`` : spread half-life in days (float)
        - ``weights`` : (T, 2) DataFrame, pre-shift target weights
        - ``zscore`` : validation-window z-score Series
        - ``pair`` : the input ``(y, x)`` tuple
    """
    y_ticker, x_ticker = pair
    selection_start, selection_end = selection_window
    validation_start, validation_end = validation_window

    # 1. Load prices across the full span.
    prices = load_prices(
        [y_ticker, x_ticker],
        selection_start,
        validation_end,
        alignment="inner",
    )
    y_full = prices[y_ticker]
    x_full = prices[x_ticker]

    # 2. Selection slice for OLS.
    sel_mask = (prices.index >= selection_start) & (prices.index <= selection_end)
    y_sel = y_full.loc[sel_mask]
    x_sel = x_full.loc[sel_mask]

    # 3. Freeze OLS hedge ratio.
    hedge = ols_hedge_ratio(y_sel, x_sel, log_prices=True)

    # 4. Half-life → rolling z-score window.
    half_life = compute_half_life(hedge.spread)
    window = max(int(round(2 * half_life)), MIN_Z_WINDOW)

    # 5. Live spread on full concat, no intercept.
    live_spread_full = compute_live_spread(
        y_full, x_full,
        beta=hedge.beta,
        alpha=hedge.alpha,
        log_prices=True,
        use_intercept_in_spread=False,
    )

    # 6. Rolling z-score on full series, slice to validation.
    z_full = zscore(live_spread_full, window=window)
    val_mask = (prices.index >= validation_start) & (prices.index <= validation_end)
    z_val = z_full.loc[val_mask]

    # 7. State-machine signal → raw asset weights with fixed beta.
    spread_pos = zscore_signal(
        z_val, entry_z=z_entry, exit_z=z_exit, stop_z=z_stop
    )
    weights = spread_position_to_asset_weights(
        spread_pos, hedge_ratio=hedge.beta, pair=pair
    )

    # 8. Simple returns on the validation window; drop first row (NaN from diff).
    prices_val = prices.loc[val_mask, [y_ticker, x_ticker]]
    returns_val = to_returns(prices_val, method="simple").iloc[1:]
    weights = weights.reindex(returns_val.index)[[y_ticker, x_ticker]]

    # 9. Cash rate on returns calendar.
    rf_daily = build_cash_rate(
        returns_val.index, validation_start, validation_end
    )

    # 10. Run backtest (signal_lag=1 enforced by the engine).
    result = run_backtest(
        signal=weights,
        returns=returns_val,
        cost_model=LinearCost(cost_bps),
        signal_lag=1,
        cash_rate=rf_daily,
    )

    return OLSPairBacktestOutput(
        result=result,
        beta=hedge.beta,
        half_life=half_life,
        weights=weights,
        zscore=z_val,
        pair=pair,
    )
