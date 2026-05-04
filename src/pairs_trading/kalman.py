"""Kalman-filter hedge ratio for pairs trading.

Session 03 replacement for the fixed-OLS hedge used in session 02.
The Kalman filter treats the hedge ratio as a latent state that drifts
over time (random walk).  Each observation updates the belief about
today's ratio using only past data to form the prediction — so the
innovation ``e_t = y_t - H_t @ x_hat_{t|t-1}`` is strictly causal and
is the natural trade signal.

Model
-----
State (2-vector, one per timestep):

    [alpha_t, beta_t]^T = [alpha_{t-1}, beta_{t-1}]^T + w_t,
        w_t ~ N(0, Q)

Observation (scalar):

    y_t = alpha_t + beta_t * x_t + v_t,    v_t ~ N(0, R)

The 2-state (alpha, beta) form is preferred over the 1-state (beta
only) form because even near-duplicate equities like GOOG/GOOGL have a
persistent level offset that would otherwise be absorbed into beta and
pollute the spread.  Cost: one extra state dim, one extra element in
the Kalman gain — trivial.

Kalman recursion
----------------
``F = I_2`` (random walk), ``H_t = [1, x_t]`` (row vector, rebuilt
every step from the current x).  Because the observation is scalar,
the innovation variance ``S_t`` is scalar and no matrix inversion is
needed anywhere in the loop.

Predict:

    x_hat_{t|t-1} = x_hat_{t-1|t-1}
    P_{t|t-1}     = P_{t-1|t-1} + Q

Update:

    e_t = y_t - H_t @ x_hat_{t|t-1}                      # innovation
    S_t = H_t @ P_{t|t-1} @ H_t^T + R                    # scalar
    K_t = P_{t|t-1} @ H_t^T / S_t                        # 2-vector
    x_hat_{t|t} = x_hat_{t|t-1} + K_t * e_t
    P_{t|t}     = (I - K_t @ H_t) @ P_{t|t-1}

No look-ahead
-------------
The spread at timestep *t* is ``e_t``, and ``e_t`` depends only on
``x_hat_{t|t-1}`` — the state estimated from data through *t-1*.
Today's ``y_t`` enters only in the update step, AFTER the innovation
is fixed.  This is the causality property of the recursion and is
regression-tested in ``tests/test_kalman.py::test_kalman_innovation_no_lookahead``.

Hyperparameters
---------------
``R`` (scalar) — observation-noise variance.  Day-to-day spread shocks
not attributable to a drifting hedge ratio.
``Q`` (2x2) — process-noise covariance.  Controls how fast the filter
tracks changes in (alpha, beta).  Large Q: fast tracking, noisy state.
Small Q: slow, steady.  Only the ratio ``Q / R`` affects the Kalman
gain (rescaling both by the same constant is invisible at the trading
signal level).

Defaults ``R=1e-3``, ``Q=1e-4 * I_2`` are the Chan (2013) textbook
starting point.  They are **not** tuned on validation data; see the
anti-look-ahead audit block in ``scripts/03_kalman_pair_backtest.py``.

Implementation note
-------------------
The recursion is inherently sequential — each step depends on the
previous state and covariance — so an explicit Python loop is used
with numpy arrays internally.  Conversion to pandas Series happens
once at the end to avoid per-step pandas overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanHedgeResult:
    """Output of :func:`kalman_hedge_ratio`.

    Attributes
    ----------
    alphas : pd.Series
        Filtered alpha_{t|t} at each timestep.
    betas : pd.Series
        Filtered beta_{t|t} at each timestep — the time-varying hedge
        ratio used for position sizing.
    spreads : pd.Series
        Innovations e_t = y_t - (alpha_{t|t-1} + beta_{t|t-1} * x_t).
        This IS the trade signal — the model's prediction residual,
        using only data through t-1.
    innovation_variance : pd.Series
        S_t — variance of the innovation at each step.  Useful for
        adaptive thresholds, not used by the default backtest.
    state_cov_history : list[np.ndarray]
        List of 2x2 posterior covariances P_{t|t}, one per timestep.
        Diagnostic (e.g., to plot uncertainty decay).
    """

    alphas: pd.Series
    betas: pd.Series
    spreads: pd.Series
    innovation_variance: pd.Series
    state_cov_history: list[np.ndarray]


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    Q: np.ndarray | None = None,
    R: float = 1e-3,
    initial_state: np.ndarray | None = None,
    initial_cov: np.ndarray | None = None,
) -> KalmanHedgeResult:
    """Run a 2-state Kalman filter for a time-varying hedge ratio.

    Parameters
    ----------
    y, x : pd.Series
        Observed series on a shared DatetimeIndex.  Pass log-prices
        for consistency with the rest of the project; the function
        itself is signal-agnostic.
    Q : np.ndarray of shape (2, 2), optional
        Process-noise covariance.  Defaults to ``1e-4 * I_2``.
    R : float, default 1e-3
        Observation-noise variance (scalar).
    initial_state : np.ndarray of shape (2,), optional
        Prior mean for ``[alpha, beta]``.  Defaults to ``[0.0, 1.0]``
        (no level offset, equal-weighted).
    initial_cov : np.ndarray of shape (2, 2), optional
        Prior covariance of the state.  Defaults to ``I_2`` — mildly
        informative; tightens quickly as observations come in.  Use a
        larger initial_cov for a truly diffuse prior.

    Returns
    -------
    KalmanHedgeResult

    Raises
    ------
    ValueError
        If y and x do not share an index, have NaNs, or hyperparameter
        shapes are wrong.
    """
    if not y.index.equals(x.index):
        raise ValueError(
            "y and x must share an identical index; "
            f"y has {len(y)} rows, x has {len(x)} rows."
        )
    if y.isna().any() or x.isna().any():
        raise ValueError("NaN detected in input series; clean before filtering.")

    if Q is None:
        Q = np.eye(2) * 1e-4
    if initial_state is None:
        initial_state = np.array([0.0, 1.0], dtype=float)
    if initial_cov is None:
        initial_cov = np.eye(2)

    Q = np.asarray(Q, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float).reshape(2)
    initial_cov = np.asarray(initial_cov, dtype=float)
    if Q.shape != (2, 2):
        raise ValueError(f"Q must be (2, 2); got {Q.shape}.")
    if initial_cov.shape != (2, 2):
        raise ValueError(
            f"initial_cov must be (2, 2); got {initial_cov.shape}."
        )
    if R <= 0:
        raise ValueError(f"R must be positive; got {R}.")

    y_arr = y.to_numpy(dtype=float)
    x_arr = x.to_numpy(dtype=float)
    n = len(y_arr)

    alphas = np.empty(n)
    betas = np.empty(n)
    spreads = np.empty(n)
    innov_var = np.empty(n)
    cov_history: list[np.ndarray] = []

    state = initial_state.copy()
    P = initial_cov.copy()
    I2 = np.eye(2)

    for t in range(n):
        # Predict: F = I, so x_hat_{t|t-1} = x_hat_{t-1|t-1}.
        P_pred = P + Q
        state_pred = state  # no copy needed; we rebind state below.

        # Observation design H_t = [1, x_t].
        H = np.array([1.0, x_arr[t]])

        # Innovation (scalar).  Uses state_pred — data through t-1 only.
        y_pred = H @ state_pred
        e = y_arr[t] - y_pred

        # Innovation variance (scalar) and Kalman gain (2-vector).
        S = H @ P_pred @ H + R
        K = (P_pred @ H) / S

        # Posterior state and covariance.
        state = state_pred + K * e
        # (I - K @ H) form; K is (2,), H is (2,), so K[:, None] @ H[None, :]
        # is the 2x2 outer product.
        P = (I2 - np.outer(K, H)) @ P_pred

        alphas[t] = state[0]
        betas[t] = state[1]
        spreads[t] = e
        innov_var[t] = S
        cov_history.append(P.copy())

    idx = y.index
    return KalmanHedgeResult(
        alphas=pd.Series(alphas, index=idx, name="alpha"),
        betas=pd.Series(betas, index=idx, name="beta"),
        spreads=pd.Series(spreads, index=idx, name="spread"),
        innovation_variance=pd.Series(innov_var, index=idx, name="S"),
        state_cov_history=cov_history,
    )


def spread_position_to_asset_weights_tv(
    spread_position: pd.Series,
    beta_series: pd.Series,
    pair: tuple[str, str],
) -> pd.DataFrame:
    """Time-varying analogue of :func:`pairs_trading.signals.spread_position_to_asset_weights`.

    At each timestep *t*, weights are ``(signal_t, -signal_t * beta_t)``
    on ``(y, x)``, where ``beta_t`` comes from the Kalman filter's
    filtered estimate at *t*.  Downstream, the backtest engine applies
    ``signal_lag=1`` so the position held on day *t+1* uses beta from
    day *t* — causal.

    Parameters
    ----------
    spread_position : pd.Series
        {-1, 0, +1} spread position.
    beta_series : pd.Series
        Filtered hedge ratio at each timestep.  Must share the index
        of ``spread_position``.
    pair : tuple[str, str]
        ``(y_ticker, x_ticker)``.

    Returns
    -------
    pd.DataFrame
        Two columns labelled by ``pair``, indexed by
        ``spread_position.index``.
    """
    if not spread_position.index.equals(beta_series.index):
        raise ValueError(
            "spread_position and beta_series must share an identical index."
        )
    y_ticker, x_ticker = pair
    pos = spread_position.astype(float)
    weights = pd.DataFrame(
        {
            y_ticker: pos,
            x_ticker: -beta_series.astype(float) * pos,
        },
        index=spread_position.index,
    )
    return weights
