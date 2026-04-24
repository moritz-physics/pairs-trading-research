"""Tests for pairs_trading.signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.signals import (
    compute_live_spread,
    spread_position_to_asset_weights,
    zscore,
    zscore_signal,
)


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------


def test_zscore_no_lookahead():
    """z at index t must equal z computed on spread[:t+1] with same window."""
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    spread = pd.Series(rng.standard_normal(n), index=idx)
    window = 30

    full = zscore(spread, window=window)

    # Pick a few points in the second half and recompute using only the past.
    for t in [50, 100, 150, 199]:
        truncated = zscore(spread.iloc[: t + 1], window=window)
        assert np.isclose(full.iloc[t], truncated.iloc[t], equal_nan=True)


def test_zscore_strict_min_periods():
    """Default min_periods=window → first window-1 values are NaN."""
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    spread = pd.Series(np.arange(50, dtype=float), index=idx)
    z = zscore(spread, window=10)
    assert z.iloc[:9].isna().all()
    assert not z.iloc[9:].isna().any()


# ---------------------------------------------------------------------------
# zscore_signal — state machine
# ---------------------------------------------------------------------------


def _signal(z_values, **kw):
    idx = pd.date_range("2020-01-01", periods=len(z_values), freq="B")
    z = pd.Series(z_values, index=idx)
    return zscore_signal(z, **kw)


def test_signal_state_machine_long_then_flat():
    # z crosses below -entry, holds, then crosses above -exit → exit to 0.
    z_vals = [0.0, -1.5, -2.5, -2.8, -1.0, -0.3, 0.1]
    pos = _signal(z_vals, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    # Day 0: flat. Day 1: |z|<entry, flat. Day 2: z<-2 → long.
    assert list(pos.to_numpy()) == [0, 0, 1, 1, 1, 0, 0]


def test_signal_state_machine_short_then_flat():
    z_vals = [0.0, 1.5, 2.5, 3.0, 1.0, 0.2, -0.1]
    pos = _signal(z_vals, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    assert list(pos.to_numpy()) == [0, 0, -1, -1, -1, 0, 0]


def test_signal_reentry_after_normal_exit():
    """Normal exit (non-stop) allows immediate re-entry on next extreme z."""
    # long → exit → short, all without any stop-out.
    z_vals = [-2.5, -0.3, 0.0, 2.5]
    pos = _signal(z_vals, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    assert list(pos.to_numpy()) == [1, 0, 0, -1]


def test_signal_stop_out_locks_until_reset():
    """After a stop, re-entry is blocked until |z| re-enters exit band.

    Construct: enter short at z=2.5, stop out at z=4.5, z then slowly
    decays.  Even if it crosses back above entry_z=2.0 again while
    above exit_z=0.5, no new entry should fire.  Only after |z|<0.5
    does a subsequent crossing of entry_z re-arm and trigger entry.
    """
    z_vals = [
        2.5,   # enter short
        4.5,   # stop-out → flat, locked
        3.0,   # still above entry but locked → flat
        2.5,   # still locked → flat
        1.0,   # above exit → still locked
        0.2,   # below exit_z → unlock; but no entry this bar
        2.5,   # crosses entry → enter short again
    ]
    pos = _signal(z_vals, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    assert list(pos.to_numpy()) == [-1, 0, 0, 0, 0, 0, -1]


def test_signal_rejects_bad_thresholds():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    z = pd.Series([0.0, 0.0, 0.0], index=idx)
    with pytest.raises(ValueError):
        zscore_signal(z, entry_z=2.0, exit_z=2.0, stop_z=4.0)
    with pytest.raises(ValueError):
        zscore_signal(z, entry_z=5.0, exit_z=0.5, stop_z=4.0)


def test_signal_nan_z_is_flat():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    z = pd.Series([np.nan, np.nan, -3.0, -1.0], index=idx)
    pos = zscore_signal(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    assert list(pos.to_numpy()) == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# spread_position_to_asset_weights
# ---------------------------------------------------------------------------


def test_position_to_weights_long_spread():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    sp = pd.Series([1, 0, -1], index=idx)
    w = spread_position_to_asset_weights(sp, hedge_ratio=1.034, pair=("GOOG", "GOOGL"))
    assert list(w.columns) == ["GOOG", "GOOGL"]
    assert w.loc[idx[0], "GOOG"] == pytest.approx(1.0)
    assert w.loc[idx[0], "GOOGL"] == pytest.approx(-1.034)
    assert w.loc[idx[1], "GOOG"] == 0.0
    assert w.loc[idx[1], "GOOGL"] == 0.0
    assert w.loc[idx[2], "GOOG"] == pytest.approx(-1.0)
    assert w.loc[idx[2], "GOOGL"] == pytest.approx(1.034)


# ---------------------------------------------------------------------------
# compute_live_spread
# ---------------------------------------------------------------------------


def test_compute_live_spread_drops_alpha_by_default():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    y = pd.Series([10.0, 11.0, 10.5, 12.0, 11.5], index=idx)
    x = pd.Series([10.0, 10.5, 10.2, 11.0, 10.8], index=idx)
    s_no_int = compute_live_spread(y, x, beta=1.0, alpha=5.0, log_prices=False)
    s_with_int = compute_live_spread(
        y, x, beta=1.0, alpha=5.0, log_prices=False, use_intercept_in_spread=True
    )
    # Without intercept: y - x.  With intercept: y - 5 - x.
    assert np.allclose(s_no_int.to_numpy(), (y - x).to_numpy())
    assert np.allclose(s_with_int.to_numpy(), (y - 5.0 - x).to_numpy())
