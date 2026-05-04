"""Session 02 — out-of-sample OLS-hedge pair backtest.

Scope
-----
Two cointegrated pairs from session 01:

  - GOOG / GOOGL  (share-class arb, HL ~25d — primary candidate)
  - GLD  / IAU    (gold-ETF arb,   HL ~5d  — secondary; cost-heavy)

For each pair we freeze an OLS hedge ratio on the SELECTION window
(2015-2020) and backtest a z-score mean-reversion strategy on the
VALIDATION window (2021-2022) only.  Fixed hedge ratio; Kalman is
session 03.

The OLS pipeline itself lives in :func:`pairs_trading.workflows.run_pair_backtest_ols`
so session 03 can reuse it for its OLS comparison row.  This script
composes the workflow with the per-pair benchmarks (static-spread
hold, long-y B&H, cash-only) and the figure + CSV output.

Anti-look-ahead audit
---------------------
Pairs trading has several subtle paths for future information to leak
into a backtest.  We enumerate them here so reviewers can verify the
reasoning against the code below:

  1. Hedge ratio (beta) is estimated with OLS on [SELECTION_START,
     SELECTION_END] log-prices and then held FIXED for the entire
     validation window.  There is no in-sample refit path in
     `ols_hedge_ratio`; the function signature takes one training
     block and returns a frozen result.

  2. Alpha (intercept) is NOT used to form the live spread.  A
     2015-2020 intercept is a stale level by 2021; the rolling
     z-score's centring term handles the spread mean.  We keep alpha
     on the result object for transparency only.  See
     `compute_live_spread(use_intercept_in_spread=False)` (default).

  3. Rolling z-score is strictly causal: pandas `.rolling(window)`
     includes day t itself and only past observations.  We set
     min_periods=window so early values are NaN until the window is
     full (the state-machine treats NaN as "flat, no transition").

  4. Z-score warm-up uses selection-window prices concatenated with
     validation-window prices, but only as HISTORY: we compute the
     spread and z-score on the concatenated series, then slice z to
     the validation window.  Selection prices are legitimate past
     data at the start of validation — not leakage.  This avoids a
     dead ~30-day warm-up inside the validation window while keeping
     every z-value computed from data at-or-before its own timestamp.

  5. The backtest engine enforces signal_lag=1 before computing PnL,
     so today's position comes from yesterday's signal.  Today's
     signal can therefore safely use today's z-score.

  6. Cash rate (DTB3) is forward-filled only.  FRED publishes after
     the trading day closes, and we only use the previously-published
     rate on any given day — no future rates ever seep in.

Outputs
-------
  results/02_pair_backtest.png  — 3x2 figure (z-score, equity, drawdown).
  results/02_pair_metrics.csv   — per-pair metrics + benchmark rows.
  stdout                        — metrics summary + exposure diagnostics.

Notes on expectations
---------------------
Sharpe > ~1.5 on this validation window would be surprising for a
first-pass OLS pair trade and should be treated as a red flag, not a
result.  See the end of the script for the turnover / exposure /
signal plot that prints if that threshold trips.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester.backtest.engine import run_backtest
from backtester.costs.linear import LinearCost
from backtester.data.loader import load_prices, to_returns
from backtester.metrics.performance import (
    annualized_return,
    annualized_volatility,
    cagr,
    calmar_ratio,
    drawdown_series,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)
from pairs_trading.selection import (
    SELECTION_END,
    SELECTION_START,
    VALIDATION_END,
    VALIDATION_START,
)
from pairs_trading.signals import spread_position_to_asset_weights
from pairs_trading.workflows import build_cash_rate, run_pair_backtest_ols

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FIGURE_PATH = RESULTS_DIR / "02_pair_backtest.png"
METRICS_CSV = RESULTS_DIR / "02_pair_metrics.csv"

# Strategy-level constants (thresholds are explicit per CLAUDE.md — these
# are defaults, not magic).
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
COST_BPS = 5.0  # LinearCost(5.0) == 5 bps per unit of |Δposition|.

PAIRS: list[tuple[str, str]] = [
    ("GOOG", "GOOGL"),
    ("GLD", "IAU"),
]


# ---------------------------------------------------------------------------
# Per-pair pipeline
# ---------------------------------------------------------------------------


def _metrics_row(
    label: str,
    net_returns: pd.Series,
    gross_returns: pd.Series | None,
    positions: pd.DataFrame | None,
    rf_daily: pd.Series,
) -> dict[str, float | str]:
    """Compute the metric bundle we report for every strategy or benchmark."""
    row: dict[str, float | str] = {"label": label}
    row["ann_return"] = annualized_return(net_returns)
    row["ann_vol"] = annualized_volatility(net_returns)
    row["sharpe"] = sharpe_ratio(net_returns, rf=rf_daily)
    row["sortino"] = sortino_ratio(net_returns, rf=rf_daily)
    row["cagr"] = cagr(net_returns)
    row["max_drawdown"] = max_drawdown(net_returns)
    row["calmar"] = calmar_ratio(net_returns)
    row["hit_rate"] = hit_rate(net_returns)
    if positions is not None:
        row["turnover_ann"] = turnover(positions)
        abs_row = positions.abs().sum(axis=1)
        row["mean_gross_exposure"] = float(abs_row.mean())
        row["mean_net_exposure"] = float(positions.sum(axis=1).mean())
        row["pct_days_active"] = float((abs_row > 0).mean())
    else:
        row["turnover_ann"] = np.nan
        row["mean_gross_exposure"] = np.nan
        row["mean_net_exposure"] = np.nan
        row["pct_days_active"] = np.nan
    if gross_returns is not None:
        row["sharpe_gross"] = sharpe_ratio(gross_returns, rf=rf_daily)
    else:
        row["sharpe_gross"] = np.nan
    return row


def run_pair(
    pair: tuple[str, str],
    rows: list[dict[str, float | str]],
    axes_row: dict[str, plt.Axes],
) -> dict[str, object]:
    """Compose the OLS workflow with per-pair benchmarks + plotting."""
    y_ticker, x_ticker = pair
    pair_label = f"{y_ticker}/{x_ticker}"
    print(f"\n=== {pair_label} ===")

    # 1. The OLS workflow runs steps 1–10 of the original script.
    out = run_pair_backtest_ols(
        pair=pair,
        selection_window=(SELECTION_START, SELECTION_END),
        validation_window=(VALIDATION_START, VALIDATION_END),
        cost_bps=COST_BPS,
        z_entry=ENTRY_Z,
        z_exit=EXIT_Z,
        z_stop=STOP_Z,
    )
    result = out["result"]
    beta = out["beta"]
    half_life = out["half_life"]
    z_val = out["zscore"]
    window = max(int(round(2 * half_life)), 20)
    print(
        f"  OLS fit (selection, log-prices): beta={beta:.4f}  "
        f"half-life={half_life:.2f}d  z-window={window}d"
    )

    strat_net = result.portfolio_net_returns
    strat_gross = result.portfolio_gross_returns
    positions = result.positions

    # 2. Reload returns for the benchmark backtests.  load_prices is cached.
    prices = load_prices(
        [y_ticker, x_ticker], SELECTION_START, VALIDATION_END, alignment="inner"
    )
    val_mask = (prices.index >= VALIDATION_START) & (prices.index <= VALIDATION_END)
    prices_val = prices.loc[val_mask, [y_ticker, x_ticker]]
    returns_val = to_returns(prices_val, method="simple").iloc[1:]
    rf_daily = build_cash_rate(returns_val.index, VALIDATION_START, VALIDATION_END)

    # 3. Benchmark: static spread (always +1 spread position).
    static_pos = pd.Series(1, index=z_val.index)
    static_weights = spread_position_to_asset_weights(
        static_pos, hedge_ratio=beta, pair=pair
    ).reindex(returns_val.index)[[y_ticker, x_ticker]]
    static_result = run_backtest(
        signal=static_weights,
        returns=returns_val,
        cost_model=LinearCost(COST_BPS),
        signal_lag=1,
        cash_rate=rf_daily,
    )

    # 4. Benchmark: long-y buy-and-hold.
    bh_weights = pd.DataFrame(
        {y_ticker: 1.0, x_ticker: 0.0}, index=returns_val.index
    )[[y_ticker, x_ticker]]
    bh_result = run_backtest(
        signal=bh_weights,
        returns=returns_val,
        cost_model=LinearCost(COST_BPS),
        signal_lag=1,
        cash_rate=rf_daily,
    )

    # 5. Benchmark: cash-only.  Post-engine-fix this could route zero-
    # weights through run_backtest, but compounding the daily DTB3
    # directly is exactly equivalent and one fewer moving part.
    cash_only_returns = rf_daily.copy()
    cash_only_returns.name = "cash_only"

    # 6. Metric rows.
    rows.append({
        "pair": pair_label, "beta": beta, "half_life": half_life,
        "z_window": window,
        **_metrics_row(
            f"{pair_label} strategy (net)",
            strat_net, strat_gross, positions, rf_daily,
        ),
    })
    rows.append({
        "pair": pair_label, "beta": beta, "half_life": half_life,
        "z_window": window,
        **_metrics_row(
            f"{pair_label} long-{y_ticker} B&H",
            bh_result.portfolio_net_returns,
            bh_result.portfolio_gross_returns,
            bh_result.positions,
            rf_daily,
        ),
    })
    rows.append({
        "pair": pair_label, "beta": beta, "half_life": half_life,
        "z_window": window,
        **_metrics_row(
            f"{pair_label} static-spread",
            static_result.portfolio_net_returns,
            static_result.portfolio_gross_returns,
            static_result.positions,
            rf_daily,
        ),
    })
    rows.append({
        "pair": pair_label, "beta": beta, "half_life": half_life,
        "z_window": window,
        **_metrics_row(
            f"{pair_label} cash-only",
            cash_only_returns,
            cash_only_returns,
            None,
            rf_daily,
        ),
    })

    _plot_pair(
        axes_row,
        pair_label=pair_label,
        z=z_val.reindex(positions.index),
        positions=positions,
        strat_net=strat_net,
        strat_gross=strat_gross,
        cash_net=cash_only_returns,
    )

    # 7. Red-flag diagnostic: Sharpe > 1.5 should be scrutinized.
    strat_sharpe = sharpe_ratio(strat_net, rf=rf_daily)
    if strat_sharpe > 1.5:
        abs_exp = positions.abs().sum(axis=1)
        print(
            f"  [!] Sharpe {strat_sharpe:.2f} > 1.5 on first-pass OLS pair — "
            f"suspect. Turnover(ann)={turnover(positions):.2f}, "
            f"mean gross exposure={abs_exp.mean():.3f}, "
            f"pct days active={(abs_exp > 0).mean():.2%}.  "
            f"Inspect scripts/02_pair_backtest.py outputs before trusting."
        )

    return {
        "pair": pair_label,
        "z_val": z_val,
        "positions": positions,
        "strat_net": strat_net,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_pair(
    axes_row: dict[str, plt.Axes],
    pair_label: str,
    z: pd.Series,
    positions: pd.DataFrame,
    strat_net: pd.Series,
    strat_gross: pd.Series,
    cash_net: pd.Series,
) -> None:
    # Row 1: z-score with band lines and in-position shading.
    ax_z = axes_row["z"]
    ax_z.plot(z.index, z.values, color="black", linewidth=0.8, label="z-score")
    for level, style in [
        (ENTRY_Z, ("--", "tab:blue")),
        (-ENTRY_Z, ("--", "tab:blue")),
        (EXIT_Z, (":", "grey")),
        (-EXIT_Z, (":", "grey")),
        (STOP_Z, ("--", "tab:red")),
        (-STOP_Z, ("--", "tab:red")),
    ]:
        ls, color = style
        ax_z.axhline(level, linestyle=ls, color=color, linewidth=0.8, alpha=0.7)
    pos_sign = np.sign(positions.iloc[:, 0].to_numpy())
    _shade_positions(ax_z, z.index.to_numpy(), pos_sign)
    ax_z.set_title(f"{pair_label} — validation-window z-score")
    ax_z.set_ylabel("z")
    ax_z.grid(alpha=0.3)

    # Row 2: equity curves (net, gross, cash-only).
    ax_eq = axes_row["eq"]
    for series, color, label in [
        (strat_net, "tab:blue", "strategy (net)"),
        (strat_gross, "tab:green", "strategy (gross)"),
        (cash_net, "grey", "cash only"),
    ]:
        equity = (1.0 + series).cumprod()
        ax_eq.plot(equity.index, equity.values, color=color, linewidth=1.2, label=label)
    ax_eq.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ax_eq.set_title(f"{pair_label} — equity curves")
    ax_eq.set_ylabel("equity (starting 1.0)")
    ax_eq.legend(loc="best", fontsize=8)
    ax_eq.grid(alpha=0.3)

    # Row 3: net-return drawdown.
    ax_dd = axes_row["dd"]
    dd = drawdown_series(strat_net)
    ax_dd.fill_between(dd.index, dd.values, 0.0, color="tab:red", alpha=0.4)
    ax_dd.set_title(f"{pair_label} — strategy (net) drawdown")
    ax_dd.set_ylabel("drawdown")
    ax_dd.grid(alpha=0.3)


def _shade_positions(
    ax: plt.Axes, dates: np.ndarray, pos_sign: np.ndarray
) -> None:
    """Shade contiguous in-position intervals on an axis."""
    n = len(dates)
    i = 0
    while i < n:
        s = pos_sign[i]
        if s == 0:
            i += 1
            continue
        j = i
        while j < n and pos_sign[j] == s:
            j += 1
        color = "tab:green" if s > 0 else "tab:red"
        ax.axvspan(dates[i], dates[min(j, n - 1)], color=color, alpha=0.12)
        i = j


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Session 02 — OLS-hedge pair backtest")
    print(f"  SELECTION:  {SELECTION_START} .. {SELECTION_END}  (fit beta here)")
    print(f"  VALIDATION: {VALIDATION_START} .. {VALIDATION_END}  (backtest here)")
    print(f"  Entry |z|>{ENTRY_Z}, exit |z|<{EXIT_Z}, stop |z|>{STOP_Z}; "
          f"cost={COST_BPS} bps per unit trade.")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    rows: list[dict[str, float | str]] = []
    axes_per_pair: list[dict[str, plt.Axes]] = [
        {"z": axes[0, 0], "eq": axes[1, 0], "dd": axes[2, 0]},
        {"z": axes[0, 1], "eq": axes[1, 1], "dd": axes[2, 1]},
    ]
    for pair, axes_row in zip(PAIRS, axes_per_pair, strict=True):
        run_pair(pair, rows, axes_row)

    fig.suptitle(
        "Session 02 — pair backtest on validation window (2021–2022)", y=1.00
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    col_order = [
        "pair", "label", "beta", "half_life", "z_window",
        "ann_return", "ann_vol", "sharpe", "sharpe_gross", "sortino",
        "cagr", "max_drawdown", "calmar", "hit_rate", "turnover_ann",
        "mean_gross_exposure", "mean_net_exposure", "pct_days_active",
    ]
    df = df[col_order]
    df.to_csv(METRICS_CSV, index=False, float_format="%.6f")

    print("\nMetrics summary")
    print("---------------")
    summary_cols = [
        "label", "sharpe", "sharpe_gross", "ann_return", "ann_vol",
        "max_drawdown", "turnover_ann", "mean_gross_exposure",
        "mean_net_exposure", "pct_days_active",
    ]
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda v: f"{v:7.3f}" if pd.notna(v) else "    nan",
    ):
        print(df[summary_cols].to_string(index=False))

    print(f"\nWrote {FIGURE_PATH}")
    print(f"Wrote {METRICS_CSV}")
    print(
        "\nExposure note: mean_gross_exposure = mean of |w_y|+|w_x|; "
        "mean_net_exposure = mean of w_y+w_x; pct_days_active = fraction of "
        "days with any non-zero position.  A dollar-neutral beta≈1 pair has "
        "net~0 always and gross~2 when active."
    )


if __name__ == "__main__":
    main()
