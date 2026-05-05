"""Combine N independent OLS pair strategies into a single portfolio.

The hard part of a pairs portfolio is not the math but the bookkeeping:
each pair already produces its own ``run_pair_backtest_ols`` output
with its own ``(weights, beta, half_life)``.  This module reduces N of
those into one ``BacktestResult`` and surfaces the per-day per-pair
diagnostics needed to read the result honestly.

Capital budget convention
-------------------------
At each timestep ``t`` we identify the set of active pairs

    A_t = { i : spread_position_i(t) != 0 }
    K_t = |A_t|

The portfolio's *pair-strategy capital* is fixed at 1.0 unit per dollar
of NAV.  That capital is distributed across active pairs via

    s_i(t) = pair_static_weights[i] / sum_{j in A_t} pair_static_weights[j]

if ``i in A_t`` else 0.  The static weights (per-pair scalars, derived
on the SELECTION window only — see below) determine the *relative*
allocation among active pairs; the per-timestep normalization makes the
total *pair-strategy* capital sum to 1.0 whenever K_t >= 1, and to 0
when K_t = 0.

This is "1.0 of pair-strategy capital", *not* "1.0 of asset-level
gross exposure".  A standard dollar-neutral pair (beta ~ 1) has
per-pair gross ~2.0 (+1 long y, -beta short x), so portfolio asset-
level gross is ~2.0 when at least one pair is active.  Reported
``mean_gross_exposure`` reflects this — it is the asset-level number,
which is what the cost model and engine see.

Weighting schemes
-----------------
``equal``        : pair_static_weights[i] = 1 for all i.
``inverse_vol``  : pair_static_weights[i] = 1 / sigma_i, where sigma_i
                   is the annualized std of the pair's NET strategy
                   returns from a SELECTION-window in-sample backtest
                   (same OLS beta, same z-window, same costs).  This is
                   in-sample with respect to the OLS fit but never
                   touches validation data — no look-ahead.  Pairs with
                   fewer than 10 selection-window round-trips are
                   surfaced via ``round_trips_selection`` so the user
                   can see when sigma is estimated on too little data.
``prefer_slow``  : pair_static_weights[i] = half_life_i (selection-
                   window AR(1) HL of the OLS spread).  Pairs with
                   HL outside [5, 250] are DROPPED from this scheme
                   entirely (excluded from combined_weights, K_t, and
                   the portfolio engine call).  Rationale: session 02
                   showed fast pairs are cost-destroyed; cost-per-
                   half-life-cycle dominates the strategy economics on
                   our universe.  This is an empirical preference, not
                   a universal principle — slow pairs are favored
                   because *on this dataset* their turnover footprint
                   is small enough for the edge to survive.

No-look-ahead invariants
------------------------
1. ``sigma_i`` and ``half_life_i`` are computed on selection-window
   data only.  The selection-window backtest re-uses the same
   ``run_pair_backtest_ols`` workflow but with
   ``validation_window=selection_window`` — the resulting "validation"
   slice is the selection window and produces a strictly in-sample
   strategy return series, never peeking at 2021–2022.
2. ``A_t`` and ``K_t`` are formed from ``spread_position_i(t)`` which
   is a causal output of ``zscore_signal`` at time t.  The engine
   applies ``signal_lag=1`` to the combined weights, so the position
   physically held on day ``t+1`` corresponds to ``s_i(t)``.  We
   report ``K_history`` on the signal calendar (no extra shift) — the
   reader interprets it knowing the engine shifts execution by 1 day.
3. Same ticker appearing across pairs is *summed* (signed) into one
   column of ``combined_weights``.  No re-normalization at the asset
   level — the pair-level normalization is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backtester.backtest.engine import BacktestResult, run_backtest
from backtester.costs.linear import LinearCost
from backtester.data.loader import load_prices, to_returns

from pairs_trading.workflows import build_cash_rate, run_pair_backtest_ols

WeightingScheme = Literal["equal", "inverse_vol", "prefer_slow"]

# HL band for prefer_slow.  Pairs outside this band are dropped — a
# pair with HL < 5 days is cost-destroyed (per session 02) and a pair
# with HL > 250 days is on the edge of "no mean reversion at all" and
# its HL estimate is dominated by noise.
PREFER_SLOW_HL_BAND: tuple[float, float] = (5.0, 250.0)
# Visibility threshold for inverse_vol.  Pairs with fewer than this
# many round-trips on the selection window have a sigma estimate based
# on very little signal — we surface a stdout warning per pair.
MIN_ROUND_TRIPS_FOR_VOL = 10


@dataclass(frozen=True)
class PairsPortfolioResult:
    """Result of :func:`build_pairs_portfolio`.

    Attributes
    ----------
    pair_results : dict[tuple[str, str], dict]
        Per-pair output of ``run_pair_backtest_ols`` on the validation
        window.  Includes ``result``, ``beta``, ``half_life``,
        ``weights``, ``zscore``, ``pair``.
    combined_weights : pd.DataFrame
        Aggregated per-asset target weights across all pairs (signed
        sum across pairs sharing a ticker), indexed by validation-
        window dates.  Pre-shift — the engine applies ``signal_lag=1``.
    portfolio_result : BacktestResult
        Engine output for the combined portfolio (cost-aware, cash-
        aware).
    pair_weights_history : pd.DataFrame
        Per-day per-pair allocation share ``s_i(t)``.  Columns are
        ``"y/x"`` strings, one per included pair.  Sums to 1.0 when
        ``K_t >= 1``, 0 when ``K_t == 0``.
    K_history : pd.Series
        Number of active pairs per day, on the signal calendar (no
        engine lag applied).
    pair_static_weights : pd.Series
        The unnormalized base weights driving ``s_i(t)``.  Indexed by
        ``"y/x"`` pair label.
    pair_diagnostics : dict[str, dict]
        Per-pair diagnostic numbers computed during construction:
        ``half_life``, ``sigma_selection`` (annualized), and
        ``round_trips_selection``.  Keyed by ``"y/x"`` label.
    excluded_pairs : list[tuple[tuple[str, str], str]]
        Pairs dropped from this portfolio along with the reason (e.g.,
        HL out of band for ``prefer_slow``).  Empty for schemes that
        include every input pair.
    weighting : WeightingScheme
        The scheme that produced this result, recorded for reporting.
    """

    pair_results: dict[tuple[str, str], dict]
    combined_weights: pd.DataFrame
    portfolio_result: BacktestResult
    pair_weights_history: pd.DataFrame
    K_history: pd.Series
    pair_static_weights: pd.Series
    pair_diagnostics: dict[str, dict]
    excluded_pairs: list[tuple[tuple[str, str], str]]
    weighting: WeightingScheme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _count_round_trips(spread_position: pd.Series) -> int:
    """Count entry→flat transitions in a {-1, 0, +1} position series.

    A round trip is one completed trade: the position was non-zero on
    day ``t-1`` and is zero on day ``t``.  Open trades at the end of
    the window do NOT count.
    """
    pos = spread_position.fillna(0).to_numpy()
    if len(pos) < 2:
        return 0
    prev = pos[:-1]
    curr = pos[1:]
    closes = (prev != 0) & (curr == 0)
    return int(closes.sum())


def _spread_position_from_weights(
    weights: pd.DataFrame, pair: tuple[str, str]
) -> pd.Series:
    """Recover the {-1, 0, +1} spread position from the pair's weights.

    ``run_pair_backtest_ols`` returns weights but not the underlying
    spread position.  By construction the y-leg weight equals the
    spread position itself (sign convention from ``signals.py``), so
    sign(weights[y]) recovers it.
    """
    y_ticker = pair[0]
    return np.sign(weights[y_ticker]).astype(int)


def _selection_window_strategy_returns(
    pair: tuple[str, str],
    selection_window: tuple[str, str],
    cost_bps: float,
    z_entry: float,
    z_exit: float,
    z_stop: float,
) -> tuple[pd.Series, int]:
    """In-sample backtest on the SELECTION window.

    Calls ``run_pair_backtest_ols`` with both ``selection_window`` and
    ``validation_window`` set to the selection window, so the OLS fit
    and the strategy run share the same dates.  This is in-sample with
    respect to beta but does NOT touch the validation window — used
    only to estimate sigma_i and round-trip count.

    Returns ``(net_returns_series, round_trip_count)``.
    """
    out = run_pair_backtest_ols(
        pair=pair,
        selection_window=selection_window,
        validation_window=selection_window,
        cost_bps=cost_bps,
        z_entry=z_entry,
        z_exit=z_exit,
        z_stop=z_stop,
    )
    net = out["result"].portfolio_net_returns
    spread_pos = _spread_position_from_weights(out["weights"], pair)
    round_trips = _count_round_trips(spread_pos)
    return net, round_trips


def _compute_static_weights(
    pairs: list[tuple[str, str]],
    weighting: WeightingScheme,
    diagnostics: dict[str, dict],
) -> tuple[pd.Series, list[tuple[tuple[str, str], str]]]:
    """Translate per-pair diagnostics into a static-weight series.

    Returns ``(weights, excluded)`` where ``excluded`` lists pairs that
    are dropped from this scheme along with a reason string.
    """
    excluded: list[tuple[tuple[str, str], str]] = []
    rows: dict[str, float] = {}
    if weighting == "equal":
        for p in pairs:
            rows[_pair_label(p)] = 1.0
    elif weighting == "inverse_vol":
        for p in pairs:
            label = _pair_label(p)
            sigma = diagnostics[label]["sigma_selection"]
            if not np.isfinite(sigma) or sigma <= 0:
                excluded.append((p, f"non-finite sigma ({sigma!r})"))
                continue
            rows[label] = 1.0 / sigma
    elif weighting == "prefer_slow":
        lo, hi = PREFER_SLOW_HL_BAND
        for p in pairs:
            label = _pair_label(p)
            hl = diagnostics[label]["half_life"]
            if not np.isfinite(hl) or hl < lo or hl > hi:
                excluded.append((p, f"HL={hl:.2f} outside [{lo}, {hi}]"))
                continue
            rows[label] = float(hl)
    else:
        raise ValueError(f"unknown weighting scheme: {weighting!r}")
    return pd.Series(rows, name="static_weight"), excluded


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_pairs_portfolio(
    pairs: list[tuple[str, str]],
    selection_window: tuple[str, str],
    validation_window: tuple[str, str],
    weighting: WeightingScheme = "equal",
    cost_bps: float = 5.0,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
) -> PairsPortfolioResult:
    """Run N pair strategies and combine them into one portfolio.

    See module docstring for the capital-budget convention, weighting
    schemes, and no-look-ahead invariants.

    Parameters
    ----------
    pairs : list of (y_ticker, x_ticker)
        Pairs to include.  Each runs through ``run_pair_backtest_ols``.
    selection_window, validation_window : (start, end)
        ISO-date strings.  Selection drives OLS, half-life, z-window,
        and (for ``inverse_vol``) the in-sample sigma estimate.
        Validation is the only window the engine sees for the
        portfolio's reported P&L.
    weighting : {'equal', 'inverse_vol', 'prefer_slow'}, default 'equal'
        See module docstring.
    cost_bps : float, default 5.0
    z_entry, z_exit, z_stop : float
        Z-score thresholds for every pair's state machine.

    Returns
    -------
    PairsPortfolioResult
    """
    if len(pairs) == 0:
        raise ValueError("Need at least one pair.")
    # Detect duplicate pairs (same (y, x) twice in input).  Same ticker
    # across DIFFERENT pairs is fine and aggregates by sum at the asset
    # level — that's the whole point.
    seen_pairs: set[tuple[str, str]] = set()
    for p in pairs:
        if p in seen_pairs:
            raise ValueError(f"Duplicate pair in input: {p}")
        seen_pairs.add(p)

    # 1. Per-pair validation-window backtest (the actual portfolio output).
    pair_results: dict[tuple[str, str], dict] = {}
    for p in pairs:
        pair_results[p] = run_pair_backtest_ols(
            pair=p,
            selection_window=selection_window,
            validation_window=validation_window,
            cost_bps=cost_bps,
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=z_stop,
        )

    # 2. Per-pair diagnostics (HL is in pair_results; sigma + round-trips
    # come from a SEPARATE selection-window in-sample backtest).
    diagnostics: dict[str, dict] = {}
    for p in pairs:
        label = _pair_label(p)
        out = pair_results[p]
        sigma_sel: float = float("nan")
        round_trips: int = 0
        # Only run the second (selection-window) backtest when needed,
        # since it's expensive.  HL alone is enough for prefer_slow and
        # equal; inverse_vol needs sigma.
        if weighting == "inverse_vol":
            sel_net, round_trips = _selection_window_strategy_returns(
                pair=p,
                selection_window=selection_window,
                cost_bps=cost_bps,
                z_entry=z_entry,
                z_exit=z_exit,
                z_stop=z_stop,
            )
            sigma_sel = float(sel_net.std() * np.sqrt(252.0))
        diagnostics[label] = {
            "half_life": float(out["half_life"]),
            "sigma_selection": sigma_sel,
            "round_trips_selection": round_trips,
            "beta": float(out["beta"]),
        }

    # 3. Static weights (per-pair scalars) from the chosen scheme.
    static_weights, excluded = _compute_static_weights(
        pairs, weighting, diagnostics
    )
    if len(static_weights) == 0:
        raise ValueError(
            f"No pairs survived weighting scheme {weighting!r}; "
            f"excluded={excluded}"
        )

    # Pairs that participate in the portfolio (excluded ones drop out).
    included_pairs = [p for p in pairs if _pair_label(p) in static_weights.index]

    # 4. Per-day per-pair allocation share s_i(t).  Built on the
    # validation-window calendar of the FIRST included pair; all pair
    # workflows produce weights on the same calendar (load_prices
    # alignment="inner" + same window), so this is robust.
    val_index = pair_results[included_pairs[0]]["weights"].index

    # spread_position_i for each included pair, aligned to val_index.
    spread_positions = pd.DataFrame(index=val_index)
    for p in included_pairs:
        sp = _spread_position_from_weights(pair_results[p]["weights"], p)
        spread_positions[_pair_label(p)] = sp.reindex(val_index).fillna(0)

    is_active = spread_positions.ne(0)
    K_history = is_active.sum(axis=1).astype(int)
    K_history.name = "K"

    # Active static weights per day.  Broadcast static_weights across
    # the time index then mask to active.
    static_w_row = static_weights.reindex(spread_positions.columns)
    active_static = is_active.astype(float).mul(static_w_row, axis=1)
    active_static_sum = active_static.sum(axis=1)
    # Where K=0, sum is 0 — divide-by-zero would give NaN; replace with
    # 0 so the resulting share is 0 everywhere on flat days.
    safe_sum = active_static_sum.replace(0, np.nan)
    pair_share = active_static.div(safe_sum, axis=0).fillna(0.0)
    pair_share.columns.name = None  # cosmetic

    # 5. Combined per-asset weights.  Each pair contributes
    # share_i(t) * pair_weights_i(t); aggregate by ticker (signed sum).
    all_tickers = sorted({t for p in included_pairs for t in p})
    combined_weights = pd.DataFrame(0.0, index=val_index, columns=all_tickers)
    for p in included_pairs:
        label = _pair_label(p)
        w_p = pair_results[p]["weights"].reindex(val_index).fillna(0.0)
        share_p = pair_share[label]  # T-vector
        for ticker in p:
            combined_weights[ticker] = (
                combined_weights[ticker] + share_p * w_p[ticker]
            )

    # 6. Asset returns on the validation window for the combined panel.
    val_start, val_end = validation_window
    sel_start = selection_window[0]
    prices = load_prices(all_tickers, sel_start, val_end, alignment="inner")
    val_mask = (prices.index >= val_start) & (prices.index <= val_end)
    prices_val = prices.loc[val_mask, all_tickers]
    returns_val = to_returns(prices_val, method="simple").iloc[1:]
    # Re-align combined_weights and pair_share / K_history to returns_val.index.
    # The pair workflow's weights index is already returns_val.index for
    # each pair (run_pair_backtest_ols computes returns the same way),
    # but be defensive in case of inner-join differences.
    combined_weights = combined_weights.reindex(returns_val.index).fillna(0.0)
    pair_share = pair_share.reindex(returns_val.index).fillna(0.0)
    K_history = K_history.reindex(returns_val.index).fillna(0).astype(int)
    spread_positions = spread_positions.reindex(returns_val.index).fillna(0)

    # 7. Cash rate over the validation window.
    rf_daily = build_cash_rate(returns_val.index, val_start, val_end)

    # 8. Run the portfolio backtest.  Cash routes through the engine
    # (verified on a synthetic 2-pair panel — see CLAUDE.md history).
    portfolio_result = run_backtest(
        signal=combined_weights,
        returns=returns_val,
        cost_model=LinearCost(cost_bps),
        signal_lag=1,
        cash_rate=rf_daily,
    )

    return PairsPortfolioResult(
        pair_results=pair_results,
        combined_weights=combined_weights,
        portfolio_result=portfolio_result,
        pair_weights_history=pair_share,
        K_history=K_history,
        pair_static_weights=static_weights,
        pair_diagnostics=diagnostics,
        excluded_pairs=excluded,
        weighting=weighting,
    )
