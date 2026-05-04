"""Regression test for pairs_trading.workflows.run_pair_backtest_ols.

Locks in the session-02 GOOG/GOOGL OLS-hedge result so future
refactors can't silently change the backtest semantics without the
test flagging it.

Number history
--------------
Session 02 commit 4c0204c reported a net Sharpe of ~0.73 using the
legacy Backtest-Engine where cash_rate was applied per-asset and
summed across columns.  For a 2-asset panel on FLAT days, the
strategy earned ``2 * rf_daily`` per day instead of ``1 * rf_daily``
— an excess return of ``rf`` per flat day that was not real.

The engine was fixed in commit ec70a18 (portfolio-level cash
accrual).  Post-fix, the strategy's flat-day return is exactly
``rf_daily`` and its **excess** return over rf on flat days is zero,
as it should be.  That removes a hidden tailwind and the
rf-subtracted Sharpe drops from ~0.73 to ~0.37.

This test locks in the POST-FIX number.  The strategy did not get
worse; the prior number was overstated.
"""

from __future__ import annotations

from backtester.metrics.performance import sharpe_ratio

from pairs_trading.workflows import build_cash_rate, run_pair_backtest_ols


# Post-engine-fix excess-return Sharpe for GOOG/GOOGL on the session-02
# validation window.  If this number moves by more than ``TOLERANCE``,
# something non-trivial has changed in the workflow or the engine —
# investigate, do not update the tolerance.
POST_FIX_SHARPE = 0.37
TOLERANCE = 0.05


def test_goog_googl_ols_sharpe_within_tolerance():
    out = run_pair_backtest_ols(
        pair=("GOOG", "GOOGL"),
        selection_window=("2015-01-01", "2020-12-31"),
        validation_window=("2021-01-01", "2022-12-31"),
    )
    net = out["result"].portfolio_net_returns
    rf_daily = build_cash_rate(net.index, "2021-01-01", "2022-12-31")
    sharpe = sharpe_ratio(net, rf=rf_daily)
    assert abs(sharpe - POST_FIX_SHARPE) < TOLERANCE, (
        f"GOOG/GOOGL net Sharpe (rf=DTB3) = {sharpe:.4f}, "
        f"expected {POST_FIX_SHARPE} ± {TOLERANCE}. "
        f"Investigate before updating the tolerance."
    )


def test_output_contract():
    """The output dict has exactly the six documented keys."""
    out = run_pair_backtest_ols(
        pair=("GOOG", "GOOGL"),
        selection_window=("2015-01-01", "2020-12-31"),
        validation_window=("2021-01-01", "2022-12-31"),
    )
    assert set(out.keys()) == {
        "result", "beta", "half_life", "weights", "zscore", "pair",
    }
    assert isinstance(out["beta"], float)
    assert isinstance(out["half_life"], float)
    assert out["pair"] == ("GOOG", "GOOGL")
    assert list(out["weights"].columns) == ["GOOG", "GOOGL"]
