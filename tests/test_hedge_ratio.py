"""Tests for pairs_trading.hedge_ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.hedge_ratio import OLSHedgeResult, ols_hedge_ratio


def test_ols_recovers_known_beta():
    """y = 3 * x + noise (raw levels) — OLS should recover beta≈3."""
    rng = np.random.default_rng(42)
    n = 500
    x = np.cumsum(rng.standard_normal(n)) + 100.0  # positive levels
    noise = rng.standard_normal(n) * 0.5
    y = 3.0 * x + 10.0 + noise
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    y_s = pd.Series(y, index=idx)
    x_s = pd.Series(x, index=idx)

    res = ols_hedge_ratio(y_s, x_s, log_prices=False)

    assert isinstance(res, OLSHedgeResult)
    assert abs(res.beta - 3.0) < 0.05
    assert abs(res.alpha - 10.0) < 1.0
    # Spread is the residual on the training window, same index.
    assert res.spread.index.equals(idx)
    assert res.spread.abs().mean() < 1.0  # small residuals given noise ~0.5


def test_log_prices_flag_changes_beta():
    """log_prices=True vs False fits different models → different betas."""
    rng = np.random.default_rng(0)
    n = 400
    # Geometric random walk so log-prices are near-linear and raw is curved.
    returns = rng.standard_normal(n) * 0.01
    x = 100.0 * np.exp(np.cumsum(returns))
    y = 2.0 * x + rng.standard_normal(n) * 0.2 + 5.0
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    y_s = pd.Series(y, index=idx)
    x_s = pd.Series(x, index=idx)

    raw_beta = ols_hedge_ratio(y_s, x_s, log_prices=False).beta
    log_beta = ols_hedge_ratio(y_s, x_s, log_prices=True).beta

    # Raw should recover ~2; log should differ materially.
    assert abs(raw_beta - 2.0) < 0.1
    assert abs(log_beta - raw_beta) > 0.2


def test_log_prices_rejects_nonpositive():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    y = pd.Series(np.arange(10, dtype=float) + 1.0, index=idx)
    x = pd.Series(np.arange(10, dtype=float), index=idx)  # has a 0
    with pytest.raises(ValueError, match="positive"):
        ols_hedge_ratio(y, x, log_prices=True)


def test_index_mismatch_raises():
    y = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
    x = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-02", periods=3))
    with pytest.raises(ValueError, match="index"):
        ols_hedge_ratio(y, x, log_prices=False)


def test_nan_input_raises():
    idx = pd.date_range("2020-01-01", periods=5)
    y = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=idx)
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(ValueError, match="NaN"):
        ols_hedge_ratio(y, x, log_prices=False)
