"""Unit tests for ``pairs_trading.portfolio.build_pairs_portfolio``.

The portfolio combines outputs of ``run_pair_backtest_ols`` and runs
``run_backtest`` on the aggregate.  These tests stub out the workflow
and the data loaders with synthetic panels so we can construct the
exact scenarios the spec calls for: disjoint activity, overlapping
activity, σ-driven weighting, shared tickers, and no-look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.backtest.engine import BacktestResult, run_backtest
from backtester.costs.linear import LinearCost

from pairs_trading import portfolio as portfolio_mod
from pairs_trading.portfolio import (
    PREFER_SLOW_HL_BAND,
    _compute_static_weights,
    build_pairs_portfolio,
)


SELECTION = ("2018-01-02", "2018-12-31")
VALIDATION = ("2019-01-02", "2019-06-28")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _val_dates(n: int = 30) -> pd.DatetimeIndex:
    return pd.date_range(VALIDATION[0], periods=n, freq="B")


def _make_weights(
    pair: tuple[str, str], pattern: list[int], beta: float, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Build a (T, 2) weights DataFrame from a {-1, 0, +1} position pattern.

    Sign convention from ``signals.py``: long-spread → (+1, -beta).
    """
    y, x = pair
    pos = pd.Series(pattern, index=dates, dtype=float)
    return pd.DataFrame(
        {y: pos, x: -beta * pos},
        index=dates,
    )


def _fake_workflow_output(
    pair: tuple[str, str],
    weights: pd.DataFrame,
    half_life: float,
    beta: float,
    net_returns: pd.Series | None = None,
) -> dict:
    """Mimic the dict returned by ``run_pair_backtest_ols``.

    Only the keys consumed by ``portfolio.build_pairs_portfolio`` are
    populated.  ``result`` only needs ``portfolio_net_returns`` set if
    the test exercises the inverse_vol selection-window path.
    """
    if net_returns is None:
        net_returns = pd.Series(0.0, index=weights.index, name="portfolio_net")
    res = BacktestResult(
        positions=weights.fillna(0.0),
        gross_returns=pd.DataFrame(0.0, index=weights.index, columns=weights.columns),
        net_returns=pd.DataFrame(0.0, index=weights.index, columns=weights.columns),
        cost_series=pd.DataFrame(0.0, index=weights.index, columns=weights.columns),
        turnover_series=pd.DataFrame(0.0, index=weights.index, columns=weights.columns),
        portfolio_gross_returns=net_returns.copy(),
        portfolio_net_returns=net_returns.copy(),
    )
    return {
        "result": res,
        "beta": beta,
        "half_life": half_life,
        "weights": weights,
        "zscore": pd.Series(np.nan, index=weights.index),
        "pair": pair,
    }


def _patch_workflow(
    monkeypatch: pytest.MonkeyPatch,
    pair_to_output: dict[tuple[str, str], dict],
    selection_pair_to_returns: dict[tuple[str, str], pd.Series] | None = None,
) -> None:
    """Replace ``run_pair_backtest_ols`` with a deterministic stub.

    For each call, returns the output registered for ``pair``.  When
    ``validation_window == selection_window`` (the inverse_vol code
    path), returns a dummy output whose ``portfolio_net_returns`` come
    from ``selection_pair_to_returns[pair]`` if provided; otherwise it
    re-uses the validation output.
    """
    selection_map = selection_pair_to_returns or {}

    def fake_run_pair_backtest_ols(
        pair, selection_window, validation_window, **_kwargs
    ):
        if validation_window == selection_window:
            base = pair_to_output[pair]
            sel_rets = selection_map.get(
                pair, base["result"].portfolio_net_returns
            )
            # Build a fresh fake output where the only field consumed
            # downstream — portfolio_net_returns — is the selection one.
            return _fake_workflow_output(
                pair=pair,
                weights=base["weights"],
                half_life=base["half_life"],
                beta=base["beta"],
                net_returns=sel_rets,
            )
        return pair_to_output[pair]

    monkeypatch.setattr(
        portfolio_mod, "run_pair_backtest_ols", fake_run_pair_backtest_ols
    )


def _patch_data_loaders(
    monkeypatch: pytest.MonkeyPatch,
    prices: pd.DataFrame,
    rf_daily_value: float = 0.0,
) -> None:
    """Stub ``load_prices`` and ``build_cash_rate`` with fixed panels."""

    def fake_load_prices(tickers, start, end, alignment="inner"):
        return prices[list(tickers)].copy()

    def fake_build_cash_rate(index, start, end):
        s = pd.Series(rf_daily_value, index=index, name="DTB3_daily")
        return s

    monkeypatch.setattr(portfolio_mod, "load_prices", fake_load_prices)
    monkeypatch.setattr(portfolio_mod, "build_cash_rate", fake_build_cash_rate)


def _flat_prices(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Constant-100 prices — all asset returns are zero, so portfolio
    P&L is exactly the cash contribution.  Lets us isolate combination
    semantics from return arithmetic.
    """
    return pd.DataFrame(100.0, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_capital_budget_one_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single pair: portfolio K=1 when active, share=1.0, combined
    weights identical to the pair's own weights."""
    pair = ("Y", "X")
    dates = _val_dates(20)
    pattern = [0] * 5 + [1] * 5 + [-1] * 5 + [0] * 5
    w = _make_weights(pair, pattern, beta=1.0, dates=dates)
    pair_out = _fake_workflow_output(pair, w, half_life=20.0, beta=1.0)

    _patch_workflow(monkeypatch, {pair: pair_out})
    # Fake prices span exactly the fake-weights dates so the engine's
    # returns_val.index matches dates[1:].
    _patch_data_loaders(monkeypatch, _flat_prices(["Y", "X"], dates))

    res = build_pairs_portfolio(
        pairs=[pair],
        selection_window=SELECTION,
        validation_window=(str(dates[0].date()), str(dates[-1].date())),
        weighting="equal",
        cost_bps=0.0,
    )

    # combined weights index = dates[1:] (run_backtest drops the first
    # row from to_returns(...).iloc[1:]).
    cw = res.combined_weights
    pw = w.loc[cw.index]
    pd.testing.assert_frame_equal(cw[["Y", "X"]], pw[["Y", "X"]], check_names=False)

    # K_t matches activity.
    expected_K = (pw["Y"] != 0).astype(int)
    pd.testing.assert_series_equal(
        res.K_history, expected_K.rename("K"), check_names=False
    )

    # share of the only pair is 1.0 on active days, 0 on flat.
    share = res.pair_weights_history["Y/X"]
    np.testing.assert_array_equal(
        share.to_numpy(),
        (pw["Y"] != 0).astype(float).to_numpy(),
    )


def test_capital_budget_two_pairs_disjoint_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pairs, non-overlapping active windows.  K_t in {0, 1};
    when one pair is active, that pair takes the full pair-strategy
    capital and combined_weights match it exactly."""
    pa, pb = ("A", "B"), ("C", "D")
    dates = _val_dates(20)
    pattern_a = [1] * 8 + [0] * 12         # active first 8 days
    pattern_b = [0] * 12 + [-1] * 8        # active last 8 days, no overlap
    wa = _make_weights(pa, pattern_a, beta=1.0, dates=dates)
    wb = _make_weights(pb, pattern_b, beta=1.0, dates=dates)

    _patch_workflow(monkeypatch, {
        pa: _fake_workflow_output(pa, wa, half_life=20.0, beta=1.0),
        pb: _fake_workflow_output(pb, wb, half_life=30.0, beta=1.0),
    })
    _patch_data_loaders(monkeypatch, _flat_prices(["A", "B", "C", "D"], dates))

    res = build_pairs_portfolio(
        pairs=[pa, pb],
        selection_window=SELECTION,
        validation_window=(str(dates[0].date()), str(dates[-1].date())),
        weighting="equal",
        cost_bps=0.0,
    )

    K = res.K_history
    assert (K <= 1).all(), "Expected K_t in {0, 1} for disjoint pairs"
    # When pair A is active, combined weight for A equals wa.
    cw = res.combined_weights
    wa_aligned = wa.loc[cw.index]
    wb_aligned = wb.loc[cw.index]
    a_active = (wa_aligned["A"] != 0)
    b_active = (wb_aligned["C"] != 0)
    np.testing.assert_allclose(
        cw.loc[a_active, "A"].to_numpy(),
        wa_aligned.loc[a_active, "A"].to_numpy(),
    )
    np.testing.assert_allclose(
        cw.loc[b_active, "C"].to_numpy(),
        wb_aligned.loc[b_active, "C"].to_numpy(),
    )


def test_capital_budget_two_pairs_overlapping_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pairs, fully overlapping active windows.  K_t = 2 on active
    days; with equal weighting each pair receives share = 0.5; combined
    weights = 0.5 * (wa + wb).  Pair-strategy capital sums to 1.0 on
    active days; asset-level gross is ~2.0."""
    pa, pb = ("A", "B"), ("C", "D")
    dates = _val_dates(20)
    pattern = [0] * 5 + [1] * 10 + [0] * 5
    wa = _make_weights(pa, pattern, beta=1.0, dates=dates)
    wb = _make_weights(pb, pattern, beta=1.0, dates=dates)

    _patch_workflow(monkeypatch, {
        pa: _fake_workflow_output(pa, wa, half_life=20.0, beta=1.0),
        pb: _fake_workflow_output(pb, wb, half_life=30.0, beta=1.0),
    })
    _patch_data_loaders(monkeypatch, _flat_prices(["A", "B", "C", "D"], dates))

    res = build_pairs_portfolio(
        pairs=[pa, pb],
        selection_window=SELECTION,
        validation_window=(str(dates[0].date()), str(dates[-1].date())),
        weighting="equal",
        cost_bps=0.0,
    )

    cw = res.combined_weights
    K = res.K_history
    active_idx = K.index[K == 2]
    assert len(active_idx) > 0
    # Each pair contributes 0.5 of its raw weights when both active.
    for ticker, raw in [("A", wa["A"]), ("B", wa["B"]), ("C", wb["C"]), ("D", wb["D"])]:
        np.testing.assert_allclose(
            cw.loc[active_idx, ticker].to_numpy(),
            0.5 * raw.loc[active_idx].to_numpy(),
        )

    # Pair-strategy capital sums to 1.0 on active days (the invariant).
    share_sum = res.pair_weights_history.sum(axis=1)
    np.testing.assert_allclose(
        share_sum.loc[active_idx].to_numpy(),
        np.ones(len(active_idx)),
    )

    # Asset-level gross exposure on active days = 2.0 (two beta=1 pairs
    # each scaled to 0.5: 0.5*(1+1) + 0.5*(1+1) = 2.0).
    gross = cw.abs().sum(axis=1)
    np.testing.assert_allclose(
        gross.loc[active_idx].to_numpy(),
        2.0 * np.ones(len(active_idx)),
    )


def test_inverse_vol_weighting_normalization() -> None:
    """σ_A = 0.10, σ_B = 0.30 → static weights (10, 3.33) → after
    per-day normalization on active days, shares (0.75, 0.25)."""
    pa, pb = ("A", "B"), ("C", "D")
    diagnostics = {
        "A/B": {"half_life": 20.0, "sigma_selection": 0.10,
                "round_trips_selection": 50, "beta": 1.0},
        "C/D": {"half_life": 30.0, "sigma_selection": 0.30,
                "round_trips_selection": 50, "beta": 1.0},
    }
    sw, excluded = _compute_static_weights(
        [pa, pb], "inverse_vol", diagnostics
    )
    assert excluded == []
    np.testing.assert_allclose(sw["A/B"], 10.0)
    np.testing.assert_allclose(sw["C/D"], 1.0 / 0.30, rtol=1e-9)
    # Normalize (both active simultaneously) and verify 0.75 / 0.25.
    total = sw.sum()
    shares = sw / total
    np.testing.assert_allclose(shares["A/B"], 0.75, rtol=1e-3)
    np.testing.assert_allclose(shares["C/D"], 0.25, rtol=1e-3)


def test_no_lookahead_in_weight_estimation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """σ for inverse_vol must come from the SELECTION-window backtest,
    not the validation one.  Construct synthetic returns where the
    selection σ is 0.05 and the validation σ is 0.50, and assert
    diagnostics['sigma_selection'] reflects the selection estimate."""
    pa = ("A", "B")
    dates = _val_dates(20)
    pattern = [0] * 5 + [1] * 10 + [0] * 5
    wa = _make_weights(pa, pattern, beta=1.0, dates=dates)

    # 252-trading-day series with controlled std.  Selection: tight
    # (annualized σ ≈ 0.05).  Validation: wide (annualized σ ≈ 0.50).
    rng = np.random.default_rng(7)
    sel_idx = pd.date_range("2018-01-02", periods=252, freq="B")
    val_idx = pd.date_range("2019-01-02", periods=120, freq="B")
    sel_rets = pd.Series(
        rng.standard_normal(252) * (0.05 / np.sqrt(252)), index=sel_idx
    )
    val_rets = pd.Series(
        rng.standard_normal(120) * (0.50 / np.sqrt(252)), index=val_idx
    )
    val_out = _fake_workflow_output(pa, wa, half_life=20.0, beta=1.0,
                                    net_returns=val_rets)

    _patch_workflow(
        monkeypatch,
        pair_to_output={pa: val_out},
        selection_pair_to_returns={pa: sel_rets},
    )
    _patch_data_loaders(monkeypatch, _flat_prices(["A", "B"], dates))

    res = build_pairs_portfolio(
        pairs=[pa],
        selection_window=SELECTION,
        validation_window=(str(dates[0].date()), str(dates[-1].date())),
        weighting="inverse_vol",
        cost_bps=0.0,
    )

    sigma = res.pair_diagnostics["A/B"]["sigma_selection"]
    # σ_target = 0.05 (annualized).  Sample std on 252 obs with seed 7
    # is within ~10% of the target — generous tolerance.
    assert 0.04 < sigma < 0.06, (
        f"sigma_selection={sigma:.4f}, expected ~0.05 (selection); "
        f"if it's near 0.50, the workflow is using validation data."
    )


def test_aggregation_handles_shared_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pairs share asset A. Combined A weight = signed sum of
    contributions (not absolute sum)."""
    pa, pb = ("A", "B"), ("A", "C")
    dates = _val_dates(20)
    # Pair 1 long spread → A=+1, B=-1.
    # Pair 2 short spread → A=-1, C=+1.
    # Both active at the same time → A from pair1 is +1, from pair2 is -1.
    pattern_p1 = [0] * 5 + [+1] * 10 + [0] * 5
    pattern_p2 = [0] * 5 + [-1] * 10 + [0] * 5
    w1 = _make_weights(pa, pattern_p1, beta=1.0, dates=dates)
    w2 = _make_weights(pb, pattern_p2, beta=1.0, dates=dates)

    _patch_workflow(monkeypatch, {
        pa: _fake_workflow_output(pa, w1, half_life=20.0, beta=1.0),
        pb: _fake_workflow_output(pb, w2, half_life=30.0, beta=1.0),
    })
    _patch_data_loaders(monkeypatch, _flat_prices(["A", "B", "C"], dates))

    res = build_pairs_portfolio(
        pairs=[pa, pb],
        selection_window=SELECTION,
        validation_window=(str(dates[0].date()), str(dates[-1].date())),
        weighting="equal",
        cost_bps=0.0,
    )

    cw = res.combined_weights
    K = res.K_history
    active_idx = K.index[K == 2]
    assert len(active_idx) > 0

    # Pair 1 (A,B) long-spread contributes (A=+1, B=-1).
    # Pair 2 (A,C) short-spread (pattern -1) contributes (A=-1, C=+1).
    # Each pair gets share=0.5 → combined A = 0.5*(+1) + 0.5*(-1) = 0.
    np.testing.assert_allclose(
        cw.loc[active_idx, "A"].to_numpy(),
        np.zeros(len(active_idx)),
        atol=1e-12,
    )
    # B and C are not shared; each carries 0.5 of its single pair's contribution.
    np.testing.assert_allclose(
        cw.loc[active_idx, "B"].to_numpy(),
        0.5 * w1.loc[active_idx, "B"].to_numpy(),
    )
    np.testing.assert_allclose(
        cw.loc[active_idx, "C"].to_numpy(),
        0.5 * w2.loc[active_idx, "C"].to_numpy(),
    )


def test_prefer_slow_drops_pairs_outside_band() -> None:
    """``prefer_slow`` excludes pairs with HL outside [5, 250].  Other
    schemes keep them."""
    pa, pb, pc = ("A", "B"), ("C", "D"), ("E", "F")
    diagnostics = {
        "A/B": {"half_life": 25.0, "sigma_selection": 0.10,
                "round_trips_selection": 50, "beta": 1.0},
        "C/D": {"half_life": 2.0, "sigma_selection": 0.10,    # too fast
                "round_trips_selection": 50, "beta": 1.0},
        "E/F": {"half_life": 400.0, "sigma_selection": 0.10,  # too slow
                "round_trips_selection": 50, "beta": 1.0},
    }
    sw, excluded = _compute_static_weights(
        [pa, pb, pc], "prefer_slow", diagnostics
    )
    assert list(sw.index) == ["A/B"]
    np.testing.assert_allclose(sw["A/B"], 25.0)
    excluded_pairs = {p for p, _ in excluded}
    assert excluded_pairs == {pb, pc}

    # Equal scheme keeps all three.
    sw_eq, excl_eq = _compute_static_weights(
        [pa, pb, pc], "equal", diagnostics
    )
    assert list(sw_eq.index) == ["A/B", "C/D", "E/F"]
    assert excl_eq == []
