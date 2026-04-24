"""Tests for pairs_trading.selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.selection import (
    compute_half_life,
    engle_granger_test,
    scan_pairs,
)


def _ar1_noise(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(n) * sigma
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def _random_walk(n: int, rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    return np.cumsum(rng.standard_normal(n) * sigma)


def test_engle_granger_cointegrated_synthetic():
    rng = np.random.default_rng(42)
    n = 500
    x = _random_walk(n, rng)
    noise = _ar1_noise(n, phi=0.7, sigma=0.5, rng=rng)
    y = 2.0 * x + noise
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    y_s = pd.Series(y, index=idx)
    x_s = pd.Series(x, index=idx)

    res = engle_granger_test(y_s, x_s)
    assert res.pvalue < 0.01
    assert abs(res.hedge_ratio - 2.0) < 0.05


def test_engle_granger_independent_random_walks():
    rng = np.random.default_rng(123)
    n = 400
    x = _random_walk(n, rng)
    y = _random_walk(n, rng)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    res = engle_granger_test(pd.Series(y, index=idx), pd.Series(x, index=idx))
    assert res.pvalue > 0.05


def test_half_life_known():
    rng = np.random.default_rng(7)
    phi = 0.9
    n = 5000
    spread = _ar1_noise(n, phi=phi, sigma=1.0, rng=rng)
    hl = compute_half_life(pd.Series(spread))
    expected = -np.log(2) / np.log(phi)
    assert abs(hl - expected) / expected < 0.10


def test_half_life_non_mean_reverting():
    # Explosive AR(1) with phi>1 guarantees the AR(1) slope b = phi-1 > 0,
    # which is the exact branch compute_half_life is supposed to flag.
    rng = np.random.default_rng(11)
    n = 2000
    phi = 1.01
    e = rng.standard_normal(n) * 0.5
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    hl = compute_half_life(pd.Series(y))
    assert np.isinf(hl)


def test_half_life_random_walk():
    # Random walks (phi = 1) have b = 0 in expectation; in a finite sample
    # the OLS slope lands on either side of zero. Either outcome must yield
    # a "not mean-reverting" answer: inf, or a half-life large enough that
    # any sane downstream filter rejects it.
    n = 2000
    results = []
    # Seeds chosen so at least one lands on b >= 0 (inf branch) under the
    # current implementation, while the others exercise the "tiny negative
    # slope" path that should still produce a clearly non-tradable half-life.
    for seed in (1, 2, 3, 21, 42):
        rng = np.random.default_rng(seed)
        y = np.cumsum(rng.standard_normal(n) * 0.5)
        hl = compute_half_life(pd.Series(y))
        assert np.isinf(hl) or hl > 200, f"seed={seed}: hl={hl}"
        results.append(hl)
    assert any(np.isinf(hl) for hl in results), (
        "Expected at least one seed to land on the b >= 0 branch (inf), "
        f"got {results}"
    )


def test_scan_pairs_filters():
    rng = np.random.default_rng(2024)
    n = 600
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    a = _random_walk(n, rng)
    # B is cointegrated with A: B = 1.5*A + AR(1) noise
    b = 1.5 * a + _ar1_noise(n, phi=0.6, sigma=0.3, rng=rng)
    # C is an independent random walk
    c = _random_walk(n, rng)
    # Use positive levels so log_prices works
    base = 100.0
    prices = pd.DataFrame(
        {"AAA": np.exp(a / 10 + np.log(base)),
         "BBB": np.exp(b / 10 + np.log(base)),
         "CCC": np.exp(c / 10 + np.log(base))},
        index=idx,
    )
    out = scan_pairs(prices, max_pvalue=0.05, max_half_life=200, multiple_testing="none")
    pairs = set(zip(out["asset_y"], out["asset_x"]))
    assert ("AAA", "BBB") in pairs
    assert ("AAA", "CCC") not in pairs
    assert ("BBB", "CCC") not in pairs


def test_scan_pairs_multiple_correction_applied():
    """With 10 assets (45 pairs) all marginally significant raw, Bonferroni
    should kill them all while 'none' keeps them."""
    rng = np.random.default_rng(99)
    n = 300
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    # Build 10 series that all share a common stochastic trend weakly —
    # enough to get raw p < 0.05 but not p < 0.05/45.
    common = _random_walk(n, rng, sigma=1.0)
    data = {}
    for i in range(10):
        idio = _ar1_noise(n, phi=0.95, sigma=1.5, rng=rng)
        series = common + idio
        data[f"T{i:02d}"] = np.exp((series - series.min() + 1.0) / 20)
    prices = pd.DataFrame(data, index=idx)

    raw = scan_pairs(prices, max_pvalue=0.05, max_half_life=10_000, multiple_testing="none")
    bonf = scan_pairs(
        prices, max_pvalue=0.05, max_half_life=10_000, multiple_testing="bonferroni"
    )
    assert len(raw) > 0, "sanity: some raw p<0.05 expected in this construction"
    assert len(bonf) < len(raw), "Bonferroni must shrink the filtered set"


def test_scan_pairs_alphabetical_ordering():
    """Column order of input must not affect pair direction."""
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    a = _random_walk(n, rng)
    b = 1.2 * a + _ar1_noise(n, phi=0.5, sigma=0.3, rng=rng)
    base = 100.0
    p1 = pd.DataFrame(
        {"AAA": np.exp(a / 10 + np.log(base)), "BBB": np.exp(b / 10 + np.log(base))},
        index=idx,
    )
    p2 = p1[["BBB", "AAA"]]
    r1 = scan_pairs(p1, max_pvalue=1.0, max_half_life=np.inf, multiple_testing="none")
    r2 = scan_pairs(p2, max_pvalue=1.0, max_half_life=np.inf, multiple_testing="none")
    assert r1["asset_y"].iloc[0] == "AAA"
    assert r1["asset_x"].iloc[0] == "BBB"
    assert r2["asset_y"].iloc[0] == "AAA"
    assert r2["asset_x"].iloc[0] == "BBB"
    assert r1["hedge_ratio"].iloc[0] == pytest.approx(r2["hedge_ratio"].iloc[0])
