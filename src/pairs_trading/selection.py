"""Cointegration primitives for pairs selection.

Two non-stationary series ``y_t`` and ``x_t`` are *cointegrated* if there
exists ``beta`` such that ``y_t - beta * x_t`` is stationary.  Economically:
both share a common stochastic trend and the spread mean-reverts.

This module provides three primitives and one scanner:

- :func:`engle_granger_test` — two-step test.  Regress ``y`` on ``x`` via
  OLS, then ADF on the residuals.  ``H0``: no cointegration.  Reject at
  ``p < 0.05``.  Returns the hedge ratio and the fitted spread.
- :func:`johansen_test` — multivariate, reduced-rank VAR eigenvalue test.
  For two series we test ``H0``: rank=0 using the trace statistic and
  the 95% critical value.  Independent of Engle-Granger and useful as a
  robustness check.
- :func:`compute_half_life` — AR(1) mean-reversion speed of the spread.
  A pair can be cointegrated with ``p=0.01`` yet have a half-life of
  400 days; that pair is untradable.  Always check both.
- :func:`scan_pairs` — runs the above on every unordered pair in a price
  panel with multiple-testing correction.

Convention: callers pass **log-prices** to ``engle_granger_test`` and
``johansen_test``.  The functions themselves are signal-agnostic (they
operate on whatever series you give them).  ``scan_pairs`` takes raw
prices and applies ``np.log`` internally when ``log_prices=True``.

Pair ordering: ``scan_pairs`` defines ``(y, x)`` by alphabetical ticker
order so each unordered pair is tested exactly once, reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass(frozen=True)
class CointResult:
    """Engle-Granger cointegration result for a single pair."""

    pvalue: float
    test_statistic: float
    hedge_ratio: float
    intercept: float
    spread: pd.Series
    half_life: float


@dataclass(frozen=True)
class JohansenResult:
    """Johansen cointegration result for a pair (rank=0 hypothesis)."""

    trace_stat: float
    crit_value_95: float
    cointegrated: bool


def compute_half_life(spread: pd.Series) -> float:
    """Estimate spread mean-reversion half-life from an AR(1) fit.

    Fits ``Δspread_t = a + b * spread_{t-1} + ε`` by OLS.  Under the
    AR(1) model ``spread_t = (1+b) * spread_{t-1} + …``, so the decay
    factor per step is ``(1+b)`` and the half-life solves
    ``(1+b)^h = 1/2``, giving ``h = -log(2) / log(1+b)``.

    Parameters
    ----------
    spread : pd.Series
        Spread series, assumed daily.  NaNs are dropped.

    Returns
    -------
    float
        Half-life in days.  ``np.inf`` when ``b >= 0`` (no mean
        reversion) or when ``1+b <= 0`` (would produce a complex log).
    """
    s = spread.dropna()
    s_lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    s_lag, delta = s_lag.align(delta, join="inner")
    X = sm.add_constant(s_lag.values)
    b = sm.OLS(delta.values, X).fit().params[1]
    if b >= 0 or (1.0 + b) <= 0:
        return float(np.inf)
    return float(-np.log(2.0) / np.log(1.0 + b))


def engle_granger_test(y: pd.Series, x: pd.Series) -> CointResult:
    """Engle-Granger two-step cointegration test.

    Step 1: regress ``y = alpha + beta * x + ε`` by OLS.
    Step 2: ADF unit-root test on the residuals (``trend="c"``).

    ``H0``: residuals have a unit root (no cointegration).

    Parameters
    ----------
    y, x : pd.Series
        Series with a common DatetimeIndex.  Pass log-prices by
        convention — this function is signal-agnostic but the p-value
        and hedge ratio are only meaningful on level series.

    Returns
    -------
    CointResult
        p-value, test statistic, OLS hedge ratio and intercept, the
        fitted spread, and its half-life.
    """
    y, x = y.align(x, join="inner")
    X = sm.add_constant(x.values)
    ols = sm.OLS(y.values, X).fit()
    intercept, hedge_ratio = float(ols.params[0]), float(ols.params[1])
    spread = y - hedge_ratio * x - intercept
    t_stat, pvalue, _ = coint(y.values, x.values, trend="c")
    return CointResult(
        pvalue=float(pvalue),
        test_statistic=float(t_stat),
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        spread=spread,
        half_life=compute_half_life(spread),
    )


def johansen_test(y: pd.Series, x: pd.Series) -> JohansenResult:
    """Johansen cointegration test for a 2-series system, rank=0.

    Uses ``det_order=0`` (constant in cointegration relation, no trend)
    and ``k_ar_diff=1``.  Reports the trace statistic against the 95%
    critical value for the null of zero cointegrating vectors.
    """
    y, x = y.align(x, join="inner")
    data = np.column_stack([y.values, x.values])
    res = coint_johansen(data, det_order=0, k_ar_diff=1)
    trace_stat = float(res.lr1[0])
    crit_95 = float(res.cvt[0, 1])
    return JohansenResult(
        trace_stat=trace_stat,
        crit_value_95=crit_95,
        cointegrated=trace_stat > crit_95,
    )


def scan_pairs(
    prices: pd.DataFrame,
    max_pvalue: float = 0.05,
    max_half_life: float = 60,
    log_prices: bool = True,
    multiple_testing: Literal["none", "bonferroni", "fdr_bh"] = "fdr_bh",
) -> pd.DataFrame:
    """Scan all unordered pairs in a price panel for cointegration.

    For each pair, runs Engle-Granger, Johansen (independent), and
    computes the spread half-life.  Applies a multiple-testing
    correction to the Engle-Granger p-values across the full scan.

    Filtering
    ---------
    A pair is kept if BOTH:
    - ``pvalue_adjusted <= max_pvalue`` (or raw ``pvalue`` when
      ``multiple_testing="none"``), AND
    - ``half_life <= max_half_life``.

    ``max_pvalue`` applies to **adjusted** p-values whenever correction
    is on — this is deliberate.  With n≈135 tests at α=0.05, ~7 false
    positives are expected by chance; the correction is the discipline.

    Pair direction: alphabetical — ``asset_y`` is the ticker that sorts
    first, ``asset_x`` the second.  Each unordered pair is tested once.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide panel of prices with tickers as columns, DatetimeIndex
        rows.  No NaNs.
    max_pvalue : float, default 0.05
        Filter threshold on ``pvalue_adjusted``.
    max_half_life : float, default 60
        Filter threshold on half-life, in days.
    log_prices : bool, default True
        If True, take ``np.log(prices)`` before testing.  Standard
        literature convention; makes the spread a log-return
        differential and hedge ratios scale-invariant.
    multiple_testing : {"none", "bonferroni", "fdr_bh"}, default "fdr_bh"
        Correction applied across all tested pairs.  FDR (Benjamini-
        Hochberg) controls the expected false discovery rate and is
        less conservative than Bonferroni.

    Sub-window robustness
    ---------------------
    For any pair with raw EG ``pvalue < 0.05`` (i.e. tests as cointegrated
    on the full window), the function also runs EG on the first and second
    halves of the window and reports ``pvalue_first_half`` and
    ``pvalue_second_half``. These are surfaced for diagnostics — they are
    not used as filters here. Pairs with both halves significant are more
    likely to reflect a structural relationship than a single-regime
    artifact. Pairs whose full-window p is high get NaN in these columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``asset_y, asset_x, pvalue, pvalue_adjusted,
        significant_after_correction, pvalue_first_half,
        pvalue_second_half, test_statistic, hedge_ratio, half_life,
        johansen_trace_stat, johansen_crit_95, johansen_cointegrated``.
        Rows are filtered pairs, sorted ascending by the effective
        p-value.
    """
    if prices.isna().any().any():
        raise ValueError("scan_pairs requires a NaN-free price panel.")

    data = np.log(prices) if log_prices else prices
    tickers = sorted(data.columns)

    # "Tests as cointegrated" — gate for the sub-window robustness check.
    SUB_WINDOW_GATE = 0.05

    rows: list[dict] = []
    for a, b in combinations(tickers, 2):
        y, x = data[a], data[b]
        eg = engle_granger_test(y, x)
        jo = johansen_test(y, x)

        if eg.pvalue < SUB_WINDOW_GATE:
            mid = len(y) // 2
            _, p_first, _ = coint(y.iloc[:mid].values, x.iloc[:mid].values, trend="c")
            _, p_second, _ = coint(y.iloc[mid:].values, x.iloc[mid:].values, trend="c")
        else:
            p_first = np.nan
            p_second = np.nan

        rows.append(
            {
                "asset_y": a,
                "asset_x": b,
                "pvalue": eg.pvalue,
                "pvalue_first_half": float(p_first),
                "pvalue_second_half": float(p_second),
                "test_statistic": eg.test_statistic,
                "hedge_ratio": eg.hedge_ratio,
                "half_life": eg.half_life,
                "johansen_trace_stat": jo.trace_stat,
                "johansen_crit_95": jo.crit_value_95,
                "johansen_cointegrated": jo.cointegrated,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df["pvalue_adjusted"] = []
        df["significant_after_correction"] = []
        return df

    # Significance is always judged at alpha=0.05 (standard). The filter
    # threshold ``max_pvalue`` is a separate knob — it lets callers request
    # a looser or stricter cut without redefining what "significant" means.
    SIG_ALPHA = 0.05
    pvals = df["pvalue"].to_numpy()
    if multiple_testing == "none":
        df["pvalue_adjusted"] = pvals
        df["significant_after_correction"] = pvals < SIG_ALPHA
        effective = pvals
    else:
        _, p_adj, _, _ = multipletests(pvals, alpha=SIG_ALPHA, method=multiple_testing)
        df["pvalue_adjusted"] = p_adj
        df["significant_after_correction"] = p_adj < SIG_ALPHA
        effective = p_adj

    keep = (effective <= max_pvalue) & (df["half_life"].to_numpy() <= max_half_life)
    df = df.loc[keep].copy()
    df = df.sort_values("pvalue_adjusted" if multiple_testing != "none" else "pvalue").reset_index(
        drop=True
    )
    cols = [
        "asset_y",
        "asset_x",
        "pvalue",
        "pvalue_adjusted",
        "significant_after_correction",
        "pvalue_first_half",
        "pvalue_second_half",
        "test_statistic",
        "hedge_ratio",
        "half_life",
        "johansen_trace_stat",
        "johansen_crit_95",
        "johansen_cointegrated",
    ]
    return df[cols]


# ---------------------------------------------------------------------------
# Data windows — imported by later scripts so they don't re-specify dates.
# Three-way split to avoid in-sample bias:
#   SELECTION: fit cointegration / pick pairs here (session 01).
#   VALIDATION: backtest and tune parameters here (session 02+).
#   HOLDOUT: touched exactly once at the end of the project.
#
# 3 years of daily data is on the thin side for cointegration tests,
# particularly with the 2020 COVID regime break injecting a large common
# factor. Extending to 2015 gives ~1500 trading days and multiple regime
# periods, increasing statistical power while keeping the validation and
# holdout periods untouched. We keep 2020 in the selection window rather
# than excluding it — pairs that don't survive distress periods are not
# real pairs.
# ---------------------------------------------------------------------------

SELECTION_START = "2015-01-01"
SELECTION_END = "2020-12-31"
VALIDATION_START = "2021-01-01"
VALIDATION_END = "2022-12-31"
HOLDOUT_START = "2023-01-01"
HOLDOUT_END = "2024-12-31"
