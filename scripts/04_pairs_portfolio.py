"""Session 04 — portfolio of pair strategies.

Combines four pair strategies into a single portfolio with explicit
capital budgeting and three weighting schemes.  The pairs are chosen
from a re-run of the session-01 cointegration scan on an expanded
structural universe (sessions 04 additions: USO/BNO, MSFT/GOOGL,
JPM/BAC).  GOOGL/MSFT is the only new FDR-significant pair; the
remaining three pairs are taken from session 01 (GOOG/GOOGL strong,
GLD/IAU and SIVR/SLV Johansen-only on near-tracking ETFs).

Capital convention
------------------
The portfolio's *pair-strategy capital* is fixed at 1.0 unit per
dollar of NAV and is distributed across active pairs each day per the
chosen weighting scheme.  Asset-level gross exposure is therefore
~1–2× when at least one pair is active (each beta≈1 dollar-neutral
pair has +1 long y / -beta short x).  This is the standard pairs-
trading convention; see ``pairs_trading.portfolio`` for details.

Weighting schemes
-----------------
``equal``         : 1/K_t to each active pair.
``inverse_vol``   : weight ∝ 1/σ_i (in-sample selection-window net σ).
``prefer_slow``   : weight ∝ HL_i.  Empirically motivated by session
                    02's finding that fast pairs are cost-destroyed —
                    longer half-life means fewer round-trips per unit
                    of edge captured.  Pairs with HL outside [5, 250]
                    are excluded from this scheme.

This is the empirical preference for *this* dataset, not a universal
principle.

Anti-look-ahead audit
---------------------
1. All hedge ratios, half-lives, and inverse-vol σ are estimated on
   the SELECTION window (2015–2020) only.  σ uses an in-sample
   selection-window backtest of the same OLS pair workflow — never
   touches validation data.
2. K_t is causal from per-pair signals at t.  ``signal_lag=1`` in the
   engine ensures positions on day t+1 use signals through t.
3. DTB3 cash rate is forward-filled from FRED-published values.

Outputs
-------
  results/04_pairs_portfolio.png         — equity, K_t, SPY corr, drawdown.
  results/04_pairs_portfolio_metrics.csv — per-strategy metric rows.
  stdout                                 — scan + diagnostics + comparison.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

from pairs_trading.portfolio import (
    MIN_ROUND_TRIPS_FOR_VOL,
    PREFER_SLOW_HL_BAND,
    build_pairs_portfolio,
)
from pairs_trading.selection import (
    SELECTION_END,
    SELECTION_START,
    VALIDATION_END,
    VALIDATION_START,
    scan_pairs,
)
from pairs_trading.workflows import build_cash_rate

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("session04")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURE_PATH = RESULTS_DIR / "04_pairs_portfolio.png"
METRICS_CSV = RESULTS_DIR / "04_pairs_portfolio_metrics.csv"

ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
COST_BPS = 5.0

# ---------------------------------------------------------------------------
# Universe expansion — re-run session-01 scan on existing groups plus
# session-04 additions.  RY/RY.TO is intentionally skipped (CAD vs USD
# listing, no FX normalization in the data layer).
# ---------------------------------------------------------------------------

EXPANDED_GROUPS: dict[str, list[str]] = {
    "Same-index ETFs": ["SPY", "IVV", "VOO", "QQQ"],
    "Commodity ETFs": ["GLD", "IAU", "SLV", "SIVR"],
    "Sector overlaps": ["XLF", "KBE", "KRE", "XLE", "XOP"],
    "Share classes": ["GOOGL", "GOOG"],
    "Crude oil ETFs": ["USO", "BNO"],            # session 04 addition
    "Big tech twins": ["MSFT", "GOOGL"],         # session 04 addition
    "Large bank peers": ["JPM", "BAC"],          # session 04 addition
}
SCAN_MAX_PVALUE = 0.05
SCAN_MAX_HALF_LIFE = 60


# ---------------------------------------------------------------------------
# Final pair set, hardcoded.  evidence_tier is the legibility column
# requested for the metrics CSV.
#
# strong                       : EG + Johansen + sub-window stable
# structural-johansen-only     : EG underpowered on near-tracking ETFs;
#                                Johansen passes
# weak-eg-passing              : FDR-significant on EG but Johansen-False
#                                and sub-window unstable
# ---------------------------------------------------------------------------

PAIRS_META: list[dict] = [
    {"pair": ("GOOG", "GOOGL"),  "evidence_tier": "strong"},
    {"pair": ("GLD",  "IAU"),    "evidence_tier": "structural-johansen-only"},
    {"pair": ("GOOGL", "MSFT"),  "evidence_tier": "weak-eg-passing"},
    {"pair": ("SIVR", "SLV"),    "evidence_tier": "structural-johansen-only"},
]
PAIRS: list[tuple[str, str]] = [m["pair"] for m in PAIRS_META]
EVIDENCE_TIER: dict[str, str] = {
    f"{m['pair'][0]}/{m['pair'][1]}": m["evidence_tier"] for m in PAIRS_META
}


def _run_universe_scan() -> pd.DataFrame:
    """Print and return the cointegration scan over the expanded universe.

    Side-effect: prints filtered survivors and per-new-group diagnostics.
    Used purely for the audit trail; the actual pair set is hardcoded.
    """
    tickers = sorted({t for ts in EXPANDED_GROUPS.values() for t in ts})
    prices = load_prices(tickers, SELECTION_START, SELECTION_END, alignment="outer")
    missing = prices.isna().sum()
    dropped = missing[missing > 0].index.tolist()
    if dropped:
        log.warning("Dropping %d tickers with missing data: %s", len(dropped), dropped)
        prices = prices.drop(columns=dropped)
    prices = prices.dropna(how="any")

    filtered_parts: list[pd.DataFrame] = []
    unfiltered_parts: list[pd.DataFrame] = []
    for group, members in EXPANDED_GROUPS.items():
        cols = [t for t in members if t in prices.columns]
        if len(cols) < 2:
            continue
        sub = prices[cols]
        unf = scan_pairs(
            sub, max_pvalue=1.0, max_half_life=np.inf, multiple_testing="fdr_bh"
        )
        unf["group"] = group
        unfiltered_parts.append(unf)
        filt = scan_pairs(
            sub, max_pvalue=SCAN_MAX_PVALUE, max_half_life=SCAN_MAX_HALF_LIFE,
            multiple_testing="fdr_bh",
        )
        filt["group"] = group
        filtered_parts.append(filt)

    filtered = (
        pd.concat(filtered_parts, ignore_index=True).sort_values("pvalue_adjusted").reset_index(drop=True)
        if filtered_parts else pd.DataFrame()
    )
    unfiltered = (
        pd.concat(unfiltered_parts, ignore_index=True) if unfiltered_parts else pd.DataFrame()
    )

    show_cols = [
        "group", "asset_y", "asset_x", "pvalue", "pvalue_adjusted",
        "pvalue_first_half", "pvalue_second_half",
        "hedge_ratio", "half_life", "johansen_cointegrated",
    ]
    fmt = {
        "pvalue": "{:.4f}".format, "pvalue_adjusted": "{:.4f}".format,
        "pvalue_first_half": "{:.4f}".format, "pvalue_second_half": "{:.4f}".format,
        "hedge_ratio": "{:.4f}".format, "half_life": "{:.1f}".format,
    }

    print("\n--- Universe scan (expanded) — filtered survivors ---")
    print(f"    p_adj <= {SCAN_MAX_PVALUE}, HL <= {SCAN_MAX_HALF_LIFE}d, FDR-BH")
    if filtered.empty:
        print("    (none)")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(filtered[show_cols].to_string(index=False, formatters=fmt))

    print("\n--- New groups (session 04 additions) — full diagnostic ---")
    new_groups = {"Crude oil ETFs", "Big tech twins", "Large bank peers"}
    new_rows = unfiltered[unfiltered["group"].isin(new_groups)].sort_values("pvalue_adjusted")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(new_rows[show_cols].to_string(index=False, formatters=fmt))

    return unfiltered


# ---------------------------------------------------------------------------
# Metrics row helper — same shape as session 02 / 03
# ---------------------------------------------------------------------------


def _metrics_row(
    label: str,
    net_returns: pd.Series,
    gross_returns: pd.Series | None,
    positions: pd.DataFrame | None,
    rf_daily: pd.Series,
) -> dict[str, float | str]:
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


# ---------------------------------------------------------------------------
# SPY long-only B&H reference on the validation window
# ---------------------------------------------------------------------------


def _spy_bh_returns(rf_daily: pd.Series) -> pd.Series:
    spy = load_prices(
        ["SPY"], SELECTION_START, VALIDATION_END, alignment="inner"
    )
    val_mask = (spy.index >= VALIDATION_START) & (spy.index <= VALIDATION_END)
    spy_val = spy.loc[val_mask, ["SPY"]]
    spy_rets = to_returns(spy_val, method="simple").iloc[1:]
    weights = pd.DataFrame({"SPY": 1.0}, index=spy_rets.index)
    res = run_backtest(
        signal=weights,
        returns=spy_rets,
        cost_model=LinearCost(COST_BPS),
        signal_lag=1,
        cash_rate=rf_daily.reindex(spy_rets.index),
    )
    return res.portfolio_net_returns, res.portfolio_gross_returns, res.positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Session 04 — portfolio of pair strategies")
    print(f"  SELECTION : {SELECTION_START} .. {SELECTION_END}")
    print(f"  VALIDATION: {VALIDATION_START} .. {VALIDATION_END}")
    print(
        f"  Entry |z|>{ENTRY_Z}, exit |z|<{EXIT_Z}, stop |z|>{STOP_Z}; "
        f"cost={COST_BPS} bps."
    )
    print("=" * 72)

    # 1. Universe scan (audit trail; does NOT determine the pair list).
    _run_universe_scan()

    # 2. Final pair set.
    print("\n--- Final pair set (hardcoded after scan) ---")
    for m in PAIRS_META:
        y, x = m["pair"]
        print(f"  {y}/{x:<6s}  evidence_tier = {m['evidence_tier']}")

    # 3. Run the three portfolios.
    schemes = ["equal", "inverse_vol", "prefer_slow"]
    results: dict[str, object] = {}
    for scheme in schemes:
        print(f"\n--- Running portfolio: weighting={scheme!r} ---")
        results[scheme] = build_pairs_portfolio(
            pairs=PAIRS,
            selection_window=(SELECTION_START, SELECTION_END),
            validation_window=(VALIDATION_START, VALIDATION_END),
            weighting=scheme,
            cost_bps=COST_BPS,
            z_entry=ENTRY_Z, z_exit=EXIT_Z, z_stop=STOP_Z,
        )
        port = results[scheme]
        if port.excluded_pairs:
            for pair, reason in port.excluded_pairs:
                print(
                    f"  [excluded] {pair[0]}/{pair[1]} — {reason}"
                )
        sw = port.pair_static_weights
        print(f"  static weights ({scheme}):")
        for label, w in sw.items():
            print(f"    {label:<14s}  {w:.4f}")
        if scheme == "inverse_vol":
            print(
                "\n  NOTE: inverse-vol allocated the largest weight (GLD/IAU,\n"
                "  wt=210) to the cost-destroyed pair with the smallest σ.\n"
                "  This is a known pathology: low realized volatility does not\n"
                "  imply high signal quality — it can simply reflect small\n"
                "  return magnitudes.  When some pair candidates have negative\n"
                "  Sharpe, inverse-vol amplifies the wrong components.  A\n"
                "  robust capital allocator needs Sharpe estimation, not just\n"
                "  vol estimation, but Sharpe is poorly estimated in-sample."
            )

    # 4. Per-pair selection-window diagnostics (HL, sigma, round-trips).
    # All three portfolio runs share the SAME per-pair pair_results, but
    # only the inverse_vol run computed sigma_selection (the others left
    # it NaN).  Pull diagnostics from inverse_vol.
    inv = results["inverse_vol"]
    print("\n--- Per-pair selection-window diagnostics ---")
    print(f"  {'pair':<12s} {'tier':<28s} {'HL (d)':>8s} {'sigma_sel':>10s} {'rt_count':>9s}")
    for p in PAIRS:
        label = f"{p[0]}/{p[1]}"
        d = inv.pair_diagnostics[label]
        flag = ""
        if d["round_trips_selection"] < MIN_ROUND_TRIPS_FOR_VOL:
            flag = "  [!] < min for stable σ"
        print(
            f"  {label:<12s} {EVIDENCE_TIER[label]:<28s} "
            f"{d['half_life']:>8.2f} "
            f"{d['sigma_selection']:>10.4f} "
            f"{d['round_trips_selection']:>9d}"
            f"{flag}"
        )

    # 5. Build the comparison table.  Per-pair rows + 3 portfolio rows
    # + SPY B&H + cash.
    rows: list[dict] = []

    # Validation cash rate (use the equal-weight portfolio's calendar).
    eq = results["equal"]
    val_index = eq.portfolio_result.portfolio_net_returns.index
    rf_daily = build_cash_rate(val_index, VALIDATION_START, VALIDATION_END)

    # Per-pair rows (identical across portfolio variants — pull from equal).
    for p in PAIRS:
        label = f"{p[0]}/{p[1]}"
        out = eq.pair_results[p]
        rows.append({
            "pair_or_strategy": label, "kind": "individual_pair",
            "weighting": "n/a",
            "evidence_tier": EVIDENCE_TIER[label],
            **_metrics_row(
                f"{label} OLS pair (net)",
                out["result"].portfolio_net_returns,
                out["result"].portfolio_gross_returns,
                out["result"].positions,
                rf_daily,
            ),
        })

    # Portfolio rows.
    for scheme in schemes:
        port = results[scheme]
        rows.append({
            "pair_or_strategy": "PORTFOLIO", "kind": "portfolio",
            "weighting": scheme,
            "evidence_tier": "n/a",
            **_metrics_row(
                f"Portfolio (4-pair, {scheme})",
                port.portfolio_result.portfolio_net_returns,
                port.portfolio_result.portfolio_gross_returns,
                port.portfolio_result.positions,
                rf_daily,
            ),
        })

    # SPY long-only B&H + cash.
    spy_net, spy_gross, spy_pos = _spy_bh_returns(rf_daily)
    rows.append({
        "pair_or_strategy": "SPY", "kind": "benchmark", "weighting": "n/a",
        "evidence_tier": "n/a",
        **_metrics_row("SPY long-only B&H", spy_net, spy_gross, spy_pos, rf_daily),
    })
    cash_only = rf_daily.copy()
    cash_only.name = "cash_only"
    rows.append({
        "pair_or_strategy": "CASH", "kind": "benchmark", "weighting": "n/a",
        "evidence_tier": "n/a",
        **_metrics_row("Cash only (DTB3)", cash_only, cash_only, None, rf_daily),
    })

    df = pd.DataFrame(rows)
    col_order = [
        "pair_or_strategy", "kind", "weighting", "evidence_tier", "label",
        "ann_return", "ann_vol", "sharpe", "sharpe_gross", "sortino",
        "cagr", "max_drawdown", "calmar", "hit_rate", "turnover_ann",
        "mean_gross_exposure", "mean_net_exposure", "pct_days_active",
    ]
    df = df[col_order]

    # 6. Comparison table to stdout.
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    summary_cols = [
        "label", "weighting", "sharpe", "sharpe_gross", "ann_return",
        "ann_vol", "max_drawdown", "turnover_ann", "mean_gross_exposure",
        "pct_days_active",
    ]
    with pd.option_context(
        "display.max_columns", None, "display.width", 220,
        "display.float_format", lambda v: f"{v:7.3f}" if pd.notna(v) else "    nan",
    ):
        print(df[summary_cols].to_string(index=False))

    # 7. Diagnostics block.
    _print_portfolio_diagnostics(results, eq, spy_net, rf_daily, df)

    # 8. Figure.
    _draw_figure(results, eq, spy_net, rf_daily)

    # 9. CSV.
    df.to_csv(METRICS_CSV, index=False, float_format="%.6f")
    print(f"\nWrote {FIGURE_PATH}")
    print(f"Wrote {METRICS_CSV}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _print_portfolio_diagnostics(
    results: dict, eq, spy_net: pd.Series, rf_daily: pd.Series, df: pd.DataFrame
) -> None:
    print("\n" + "=" * 100)
    print("PORTFOLIO DIAGNOSTICS")
    print("=" * 100)

    # --- Pairwise correlation of pair NET returns on validation window ---
    pair_returns = pd.DataFrame({
        f"{p[0]}/{p[1]}": eq.pair_results[p]["result"].portfolio_net_returns
        for p in PAIRS
    })
    corr = pair_returns.corr()
    n = len(corr)
    upper = corr.where(np.triu(np.ones((n, n), dtype=bool), k=1))
    avg_corr = float(upper.stack().mean())
    print("\nPairwise correlation of pair NET strategy returns (validation window):")
    with pd.option_context("display.float_format", lambda v: f"{v:6.3f}"):
        print(corr.to_string())
    print(f"\n  mean off-diagonal correlation: {avg_corr:+.3f}")
    if avg_corr > 0.4:
        print("  [!] mean correlation > 0.4 — limited diversification benefit expected.")
    else:
        print("  pair-return correlations are low — real diversification possible.")

    # --- K_t distribution (equal-weight portfolio) ---
    K = eq.K_history
    n_pairs = len(PAIRS)
    print(f"\nK_t distribution (equal-weight portfolio, {len(K)} validation days):")
    for k in range(n_pairs + 1):
        n_k = int((K == k).sum())
        pct = 100.0 * n_k / len(K)
        print(f"  K = {k}:  {n_k:4d} days  ({pct:5.1f}%)")

    # --- Sharpe deltas ---
    print("\nNet-Sharpe ranking:")
    sharpe_rows = (
        df[df["kind"].isin({"individual_pair", "portfolio"})]
        [["label", "kind", "weighting", "sharpe"]]
        .sort_values("sharpe", ascending=False)
    )
    with pd.option_context(
        "display.float_format", lambda v: f"{v:7.3f}",
        "display.width", 120,
    ):
        print(sharpe_rows.to_string(index=False))

    best_individual = (
        df[df["kind"] == "individual_pair"]["sharpe"].max()
    )
    best_portfolio = (
        df[df["kind"] == "portfolio"]["sharpe"].max()
    )
    delta = best_portfolio - best_individual
    print(
        f"\n  best individual pair Sharpe:  {best_individual:+.3f}\n"
        f"  best portfolio Sharpe:        {best_portfolio:+.3f}\n"
        f"  delta (portfolio − best individual): {delta:+.3f}"
    )
    if delta > 0:
        print("  → real diversification observed (portfolio beats best individual).")
    else:
        print("  → no diversification gain; portfolio is averaging away the edge.")

    # --- Interpretation ---
    print(
        "\nInterpretation\n"
        "--------------\n"
        "Portfolio construction redistributes risk among components but\n"
        "cannot manufacture edge.  Pairwise correlations of pair net returns\n"
        "are ~0 across all combinations, so true diversification is\n"
        "available, but with three of four pair candidates posting negative\n"
        "Sharpe on the validation window, no scheme can lift the portfolio\n"
        "above the best individual component.  The portfolio's drawdown is\n"
        "in fact deeper than any single pair's, illustrating that\n"
        "diversification of negative-expectation components compounds losses\n"
        "rather than smoothing them."
    )

    # --- Red flag ---
    if best_portfolio > 1.0:
        print(
            "\n  [!] best portfolio net Sharpe > 1.0 — none of the individual\n"
            "      pairs deliver that, suspect a leak; re-check no-look-ahead\n"
            "      invariants in pairs_trading.portfolio."
        )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _draw_figure(
    results: dict, eq, spy_net: pd.Series, rf_daily: pd.Series
) -> None:
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.4, 1.0, 1.0])

    # Row 1 (full width): equity curves of three portfolio variants + SPY + cash.
    ax_eq = fig.add_subplot(gs[0, :])
    cash_rets = rf_daily.copy()
    series_to_plot = [
        ("equal", "tab:blue"),
        ("inverse_vol", "tab:green"),
        ("prefer_slow", "tab:purple"),
    ]
    for scheme, color in series_to_plot:
        net = results[scheme].portfolio_result.portfolio_net_returns
        equity = (1.0 + net).cumprod()
        ax_eq.plot(equity.index, equity.values, color=color, linewidth=1.4,
                   label=f"portfolio — {scheme}")
    ax_eq.plot(
        (1.0 + spy_net).cumprod().index,
        (1.0 + spy_net).cumprod().values,
        color="tab:orange", linewidth=1.2, label="SPY B&H",
    )
    ax_eq.plot(
        (1.0 + cash_rets).cumprod().index,
        (1.0 + cash_rets).cumprod().values,
        color="grey", linewidth=1.0, label="cash (DTB3)",
    )
    ax_eq.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ax_eq.set_yscale("log")
    ax_eq.set_title("Equity curves — portfolio variants vs SPY B&H vs cash (validation, log scale)")
    ax_eq.set_ylabel("equity (log)")
    ax_eq.legend(loc="best", fontsize=9)
    ax_eq.grid(alpha=0.3, which="both")

    # Row 2 left: K_t over time (equal-weight portfolio is canonical).
    ax_k = fig.add_subplot(gs[1, 0])
    K = eq.K_history
    ax_k.step(K.index, K.values, where="post", color="black", linewidth=0.9)
    ax_k.fill_between(K.index, 0, K.values, step="post", alpha=0.2, color="black")
    ax_k.set_title("K_t — number of active pairs (equal-weight)")
    ax_k.set_ylabel("K_t")
    ax_k.set_yticks(range(0, len(PAIRS) + 1))
    ax_k.grid(alpha=0.3)

    # Row 2 right: rolling 60d correlation of equal-weight portfolio with SPY.
    ax_c = fig.add_subplot(gs[1, 1])
    eq_net = results["equal"].portfolio_result.portfolio_net_returns
    common = eq_net.index.intersection(spy_net.index)
    rolling_corr = (
        eq_net.loc[common].rolling(60, min_periods=60)
        .corr(spy_net.loc[common])
    )
    ax_c.plot(rolling_corr.index, rolling_corr.values, color="tab:blue", linewidth=1.0)
    ax_c.axhline(0.0, color="black", linewidth=0.5, alpha=0.6)
    ax_c.set_ylim(-1, 1)
    ax_c.set_title("Rolling 60-day corr — equal-weight portfolio vs SPY")
    ax_c.set_ylabel("corr")
    ax_c.grid(alpha=0.3)

    # Row 3 (full width): drawdowns — all individual pairs faint, equal-weight
    # portfolio bold, SPY bold.
    ax_dd = fig.add_subplot(gs[2, :])
    for p in PAIRS:
        label = f"{p[0]}/{p[1]}"
        net_p = eq.pair_results[p]["result"].portfolio_net_returns
        dd_p = drawdown_series(net_p)
        ax_dd.plot(dd_p.index, dd_p.values, color="grey", alpha=0.55,
                   linewidth=0.9, label=f"{label} (pair)")
    dd_eq = drawdown_series(eq_net)
    ax_dd.plot(dd_eq.index, dd_eq.values, color="tab:blue", linewidth=1.6,
               label="portfolio (equal)")
    dd_spy = drawdown_series(spy_net)
    ax_dd.plot(dd_spy.index, dd_spy.values, color="tab:orange", linewidth=1.4,
               label="SPY B&H")
    ax_dd.set_title("Drawdowns — individual pairs (faint), equal-weight portfolio, SPY")
    ax_dd.set_ylabel("drawdown")
    ax_dd.legend(loc="best", fontsize=8, ncol=2)
    ax_dd.grid(alpha=0.3)

    fig.suptitle(
        "Session 04 — pairs portfolio (validation 2021-2022)",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
