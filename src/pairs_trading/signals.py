"""Z-score signals and spread-position-to-weights conversion.

Three primitives:

- :func:`compute_live_spread` — forms the out-of-sample spread from a
  frozen hedge ratio.  Defaults to ``y - beta * x`` (no intercept);
  ``use_intercept_in_spread=True`` opts back in to ``y - alpha - beta * x``.
- :func:`zscore` — causal rolling z-score with strict ``min_periods``.
- :func:`zscore_signal` — path-dependent state machine that maps a
  z-score series to positions in ``{-1, 0, +1}`` on the *spread*.
  Entry at ``|z| > entry_z``, exit on crossing back inside ``exit_z``,
  stop-out at ``|z| > stop_z`` with a full reset requirement before
  re-arming.
- :func:`spread_position_to_asset_weights` — converts a spread position
  to two-asset weights ``(y, x) = (sign, -sign * beta)``.

Sign convention
---------------
``+1`` (long spread)  = long y, short ``beta`` units of x.  Entered when
z is sufficiently **negative** (spread undershoots its mean).
``-1`` (short spread) = short y, long ``beta`` units of x.  Entered when
z is sufficiently **positive** (spread overshoots).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_live_spread(
    y: pd.Series,
    x: pd.Series,
    beta: float,
    alpha: float = 0.0,
    log_prices: bool = True,
    use_intercept_in_spread: bool = False,
) -> pd.Series:
    """Form the out-of-sample spread from a frozen hedge ratio.

    Parameters
    ----------
    y, x : pd.Series
        Price series on a shared DatetimeIndex.
    beta : float
        Frozen hedge ratio from the selection-window fit.
    alpha : float, default 0.0
        Intercept from the selection-window fit.  Only used when
        ``use_intercept_in_spread=True``.
    log_prices : bool, default True
        If True, the spread is computed on ``np.log(y)`` and
        ``np.log(x)`` — must match the fit convention.
    use_intercept_in_spread : bool, default False
        If False (default), the live spread is ``y - beta * x``.  The
        rolling z-score handles the mean.  If True, the spread is
        ``y - alpha - beta * x``, pinning it to the selection-window
        mean; use only for explicit experiments comparing conventions.

    Returns
    -------
    pd.Series
        Spread series, same index as the inputs.
    """
    if not y.index.equals(x.index):
        raise ValueError("y and x must share an identical index.")

    y_use = np.log(y) if log_prices else y
    x_use = np.log(x) if log_prices else x

    if use_intercept_in_spread:
        spread = y_use - alpha - beta * x_use
    else:
        spread = y_use - beta * x_use
    spread.name = "spread"
    return spread


def zscore(
    spread: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling z-score of a spread.

    Strictly causal: ``z_t = (spread_t - mean_{t-window+1..t}) /
    std_{t-window+1..t}``.  Uses pandas ``.rolling(window)``, which
    includes day ``t`` itself — no future data.

    Parameters
    ----------
    spread : pd.Series
        Spread series.
    window : int
        Rolling-window length in observations.
    min_periods : int or None, default None
        If None, defaults to ``window`` — strict, producing NaN until
        the window fills.

    Returns
    -------
    pd.Series
        Z-score, NaN where the window is not yet full or where the
        rolling std is zero.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    if min_periods is None:
        min_periods = window

    roll = spread.rolling(window=window, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std()
    # Avoid division by zero — a constant-in-window spread yields NaN.
    z = (spread - mean) / std.where(std > 0)
    z.name = "zscore"
    return z


def zscore_signal(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> pd.Series:
    """Map a z-score series to spread positions ``{-1, 0, +1}``.

    Path-dependent state machine (entry/exit/stop semantics are defined
    over time, not pointwise).  Implemented as an explicit Python loop
    for clarity — the input is daily over a few years, so speed is a
    non-issue.

    State transitions, applied each day in order:

    * From **flat (0)**:
      - ``z < -entry_z``  → **+1** (long spread).
      - ``z > +entry_z``  → **-1** (short spread).
      - ``|z| > stop_z``  → stay flat and enter *locked* mode — no
        entries allowed until ``|z|`` returns below ``exit_z``.
    * From **long (+1)**:
      - ``z >= -exit_z``  → 0 (take profit / normal exit).
      - ``z < -stop_z``   → 0 and enter locked mode (pair broke).
    * From **short (-1)**:
      - ``z <= +exit_z``  → 0.
      - ``z > +stop_z``   → 0 and enter locked mode.
    * **Locked (flat, post-stop)**:
      - Re-arm only once ``|z| < exit_z``; then normal entry rules apply.

    Any day with NaN z-score forces flat, non-locking.

    Parameters
    ----------
    zscore : pd.Series
        Output of :func:`zscore`.
    entry_z, exit_z, stop_z : float
        Band thresholds.  Must satisfy ``0 < exit_z < entry_z < stop_z``.

    Returns
    -------
    pd.Series
        Integer-valued position series aligned to ``zscore.index``.
    """
    if not (0 < exit_z < entry_z < stop_z):
        raise ValueError(
            "Require 0 < exit_z < entry_z < stop_z; got "
            f"exit_z={exit_z}, entry_z={entry_z}, stop_z={stop_z}."
        )

    position = 0
    locked = False
    out = np.zeros(len(zscore), dtype=np.int8)
    values = zscore.to_numpy()

    for i, z in enumerate(values):
        if np.isnan(z):
            out[i] = 0
            continue

        if position == 0:
            if locked:
                if abs(z) < exit_z:
                    locked = False  # re-arm; stay flat this bar
            else:
                if z < -entry_z and z > -stop_z:
                    position = 1
                elif z > entry_z and z < stop_z:
                    position = -1
                elif abs(z) > stop_z:
                    # Entered extreme from flat; treat as locked.
                    locked = True
        elif position == 1:
            if z < -stop_z:
                position = 0
                locked = True
            elif z >= -exit_z:
                position = 0
        elif position == -1:
            if z > stop_z:
                position = 0
                locked = True
            elif z <= exit_z:
                position = 0

        out[i] = position

    return pd.Series(out, index=zscore.index, name="spread_position").astype(int)


def spread_position_to_asset_weights(
    spread_position: pd.Series,
    hedge_ratio: float,
    pair: tuple[str, str],
) -> pd.DataFrame:
    """Convert a ``{-1, 0, +1}`` spread position to two-asset weights.

    Weights are *raw* (unnormalized): when long the spread, exposure
    is ``(+1, -beta)`` on ``(y, x)``.  Gross exposure when active is
    ``1 + |beta|``, typically ~2.0 for the pairs we test.  Net
    exposure is ``1 - beta`` (near zero for beta≈1: dollar-neutral).

    Parameters
    ----------
    spread_position : pd.Series
        Output of :func:`zscore_signal`.
    hedge_ratio : float
        The ``beta`` used to form the spread.
    pair : tuple[str, str]
        ``(y_ticker, x_ticker)``.  The first is the "long-leg when
        long-spread" asset.

    Returns
    -------
    pd.DataFrame
        Two columns labelled by ``pair``, indexed by
        ``spread_position.index``.
    """
    y_ticker, x_ticker = pair
    weights = pd.DataFrame(
        {
            y_ticker: spread_position.astype(float),
            x_ticker: -hedge_ratio * spread_position.astype(float),
        },
        index=spread_position.index,
    )
    return weights
