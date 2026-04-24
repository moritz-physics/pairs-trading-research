"""OLS hedge ratio estimation for pairs trading.

Two cointegrated series ``y_t`` and ``x_t`` admit a stationary linear
combination ``y_t - beta * x_t - alpha``.  The *hedge ratio* ``beta``
is the OLS slope of ``y`` on ``x``; ``alpha`` is the intercept.  In
pairs-trading the downstream signal is built from the spread, so we
keep both parameters available while being explicit about which one
enters the live spread used at trading time.

Training-data-only API
----------------------
The single exported function, :func:`ols_hedge_ratio`, takes one
contiguous block of (training) data and returns the fit plus the
in-sample spread.  It does *not* offer a predict-on-new-data path by
design — the backtest script is expected to freeze ``beta`` on the
selection window and then compute the live spread manually on the
validation window.  This structurally prevents re-estimating the
hedge ratio on out-of-sample data.

Alpha convention
----------------
``alpha`` is reported for transparency but is intentionally *not* used
to form the live spread in the signal path.  A 2015-2020 intercept is
a stale level estimate by the time the 2021-2022 validation window
starts; the rolling z-score's centring term handles the mean better.
See :func:`pairs_trading.signals.compute_live_spread` for the live
convention (``y - beta * x`` by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class OLSHedgeResult:
    """Result of an OLS hedge-ratio fit on a single training window.

    Attributes
    ----------
    alpha : float
        OLS intercept.  Stored for transparency; not used in the live
        spread by default (see module docstring).
    beta : float
        OLS slope — the hedge ratio.  Fixed thereafter and used to
        form the live spread on out-of-sample data.
    spread : pd.Series
        In-sample residual ``y - alpha - beta * x`` on the training
        window, in whatever units the inputs were passed (log or raw).
    log_prices : bool
        True if the fit was performed on log-prices.
    """

    alpha: float
    beta: float
    spread: pd.Series
    log_prices: bool


def ols_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    log_prices: bool = True,
) -> OLSHedgeResult:
    """Estimate the OLS hedge ratio of ``y`` on ``x`` on training data.

    Parameters
    ----------
    y, x : pd.Series
        Price series for the two legs of the pair, aligned on a shared
        DatetimeIndex.  Both must be strictly positive when
        ``log_prices=True``.
    log_prices : bool, default True
        If True, take ``np.log`` of both inputs before regressing.
        Matches the convention used in :mod:`pairs_trading.selection`.

    Returns
    -------
    OLSHedgeResult

    Raises
    ------
    ValueError
        If the two series do not share an index, or if NaN is present,
        or if ``log_prices=True`` and any input is non-positive.
    """
    if not y.index.equals(x.index):
        raise ValueError(
            "y and x must share an identical index; "
            f"y has {len(y)} rows, x has {len(x)} rows."
        )
    if y.isna().any() or x.isna().any():
        raise ValueError("NaN detected in input series; clean before fitting.")

    if log_prices:
        if (y <= 0).any() or (x <= 0).any():
            raise ValueError(
                "log_prices=True requires strictly positive inputs."
            )
        y_fit = np.log(y)
        x_fit = np.log(x)
    else:
        y_fit = y
        x_fit = x

    design = sm.add_constant(x_fit.to_numpy())
    model = sm.OLS(y_fit.to_numpy(), design).fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])

    spread = pd.Series(
        y_fit.to_numpy() - alpha - beta * x_fit.to_numpy(),
        index=y.index,
        name="spread",
    )

    return OLSHedgeResult(
        alpha=alpha,
        beta=beta,
        spread=spread,
        log_prices=log_prices,
    )
