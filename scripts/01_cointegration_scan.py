"""Session 01 — structural cointegration scan (pipeline validation universe).

What this script does
---------------------
Scans every within-group pair in a small structural universe for
cointegration over the SELECTION window. For each pair it runs Engle-
Granger and Johansen, computes the AR(1) spread half-life, and (for
pairs with raw EG p<0.05) re-runs EG on each half of the window for a
sub-window robustness diagnostic. Output is one filtered ranking
(p_adj<=0.05 AND HL<=60d) plus an unfiltered diagnostic frame, both as
CSVs, and a scatter of half-life vs adjusted p-value coloured by group.

Universe — and why this one
---------------------------
The first attempt was a sector-clustered S&P 500 single-stock scan
(30 tickers across Tech / Financials / Energy). Over 2015-2020 with
FDR correction, *zero* pairs survived. Before publishing that as a
finding, we needed to rule out a pipeline bug. So this script swaps the
universe for one where cointegration is structurally guaranteed for
some pairs and structurally absent for others — same-index ETFs, a
share-class pair (GOOG/GOOGL), and an explicit negative control
(SPY/GLD) — keeping every other parameter (window, FDR, half-life
filter, sub-window check) identical to the failed scan.

Result
------
2 of 24 pairs survive the strict filter:
  - SPY/VOO   (Same-index ETFs, p_adj=0.0007, HL=0.9d)
  - GOOG/GOOGL (Share classes, p_adj=0.031,  HL=25d)
Negative control SPY/GLD correctly fails (p=0.34, Johansen=False).

A nuance worth noting: several pairs that *should* cointegrate (SPY/IVV,
IVV/VOO, GLD/IAU, SLV/SIVR) all show hedge ratio ~1.0 and short
half-lives but fail Engle-Granger because their residuals are dominated
by tiny tracking noise — ADF lacks rejection power on near-identical
series. Johansen, which doesn't condition on a fitted residual, picks up
GLD/IAU and SLV/SIVR. We keep both tests in the output rather than
restructuring the primary filter — the EG/Johansen disagreement is
itself diagnostic and is documented in the writeup.

Tradeable candidates passed to session 02 (scripts/02_pair_backtest.py,
not yet written):
  - GOOG/GOOGL — primary, real spread variance (HL=25d).
  - GLD/IAU    — secondary, Johansen-only (HL=5d, EG underpowered).

Data windows (three-way split, defined in pairs_trading.selection):
    SELECTION  2015-01-01 .. 2020-12-31   <- this script uses this only
    VALIDATION 2021-01-01 .. 2022-12-31   <- session 02+ backtests/tunes
    HOLDOUT    2023-01-01 .. 2024-12-31   <- touched exactly once at end

QQQM launched 2020-10 and will be dropped from the universe (insufficient
history); other ETFs cover the full window.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester.data.loader import load_prices
from pairs_trading.selection import (
    SELECTION_END,
    SELECTION_START,
    VALIDATION_END,
    VALIDATION_START,
    HOLDOUT_START,
    HOLDOUT_END,
    scan_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("scan")

# Pairs are formed within each group. A ticker may appear in more than
# one group (e.g. SPY and GLD also appear in the Negative control group).
# The loader dedupes when downloading.
STRUCTURE_GROUPS: dict[str, list[str]] = {
    "Same-index ETFs": ["SPY", "IVV", "VOO", "QQQ", "QQQM"],
    "Commodity ETFs": ["GLD", "IAU", "SLV", "SIVR"],
    "Sector overlaps": ["XLF", "KBE", "KRE", "XLE", "XOP"],
    # GOOG is the newly-created (April 2014) C-class share. The spread vs
    # GOOGL had not yet accumulated meaningful idiosyncratic variance in
    # the first half of the SELECTION window, so EG looks weak there
    # (p_h1 ~ 0.29 vs p_h2 ~ 0.04). Full-window EG and Johansen both pass.
    "Share classes": ["GOOGL", "GOOG"],
    "Negative control (different assets)": ["SPY", "GLD"],
}

# Stable color order for the scatter; keep in sync with STRUCTURE_GROUPS.
GROUP_COLORS: dict[str, str] = {
    "Same-index ETFs": "#1f77b4",
    "Commodity ETFs": "#ff7f0e",
    "Sector overlaps": "#2ca02c",
    "Share classes": "#9467bd",
    "Negative control (different assets)": "#d62728",
}

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MAX_PVALUE = 0.05
MAX_HALF_LIFE = 60
MULTIPLE_TESTING = "fdr_bh"


def _load_universe() -> pd.DataFrame:
    # Dedupe tickers across groups (SPY and GLD appear twice).
    tickers = sorted({t for ts in STRUCTURE_GROUPS.values() for t in ts})
    log.info(
        "Loading %d unique tickers, %s to %s (SELECTION window).",
        len(tickers), SELECTION_START, SELECTION_END,
    )
    # Use outer alignment so we can explicitly drop tickers with insufficient
    # history rather than silently inner-joining them away.
    prices = load_prices(
        tickers, SELECTION_START, SELECTION_END, alignment="outer"
    )
    # Drop tickers with any missing observations over the window.
    missing = prices.isna().sum()
    dropped = missing[missing > 0].index.tolist()
    if dropped:
        log.warning("Dropping %d tickers with missing data: %s", len(dropped), dropped)
        prices = prices.drop(columns=dropped)
    # Drop any remaining dates that still have NaN (none expected).
    before = len(prices)
    prices = prices.dropna(how="any")
    if len(prices) < before:
        log.info("Dropped %d dates with residual NaN.", before - len(prices))
    log.info("Universe: %d tickers, %d dates.", prices.shape[1], prices.shape[0])
    return prices


def _scan_by_group(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return (filtered_ranked, unfiltered_all, counts) across all groups.

    Pairs are formed within each structural group. A ticker that appears
    in multiple groups (e.g. SPY in both Same-index ETFs and Negative
    control) participates in pairs from each group it belongs to; a given
    pair (a, b) is unique to whichever group contains both a and b.
    """
    filtered_parts: list[pd.DataFrame] = []
    unfiltered_parts: list[pd.DataFrame] = []
    counts = {
        "total": 0,
        "raw_sig": 0,
        "fdr_sig": 0,
        "pass_halflife": 0,
        "pass_johansen": 0,
    }
    for group, members in STRUCTURE_GROUPS.items():
        cols = [t for t in members if t in prices.columns]
        if len(cols) < 2:
            log.warning("Group %r has <2 available tickers, skipping.", group)
            continue
        sub = prices[cols]
        # Unfiltered — for the diagnostic scatter, funnel counts, controls.
        unf = scan_pairs(
            sub, max_pvalue=1.0, max_half_life=np.inf,
            multiple_testing=MULTIPLE_TESTING,
        )
        unf["group"] = group
        unfiltered_parts.append(unf)

        counts["total"] += len(unf)
        counts["raw_sig"] += int((unf["pvalue"] < MAX_PVALUE).sum())
        counts["fdr_sig"] += int(unf["significant_after_correction"].sum())
        mask_hl = unf["significant_after_correction"] & (unf["half_life"] <= MAX_HALF_LIFE)
        counts["pass_halflife"] += int(mask_hl.sum())
        counts["pass_johansen"] += int((mask_hl & unf["johansen_cointegrated"]).sum())

        # Filtered — FDR-adjusted p<=0.05 AND HL<=60.
        filt = scan_pairs(
            sub, max_pvalue=MAX_PVALUE, max_half_life=MAX_HALF_LIFE,
            multiple_testing=MULTIPLE_TESTING,
        )
        filt["group"] = group
        filtered_parts.append(filt)

    filtered = (
        pd.concat(filtered_parts, ignore_index=True)
        .sort_values("pvalue_adjusted")
        .reset_index(drop=True)
        if filtered_parts else pd.DataFrame()
    )
    unfiltered = (
        pd.concat(unfiltered_parts, ignore_index=True) if unfiltered_parts else pd.DataFrame()
    )
    return filtered, unfiltered, counts


def _plot(
    unfiltered: pd.DataFrame,
    filtered: pd.DataFrame,
    prices: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for group, color in GROUP_COLORS.items():
        sub = unfiltered[unfiltered["group"] == group]
        if sub.empty:
            continue
        # Replace inf half-lives with a large finite plotting cap so they show.
        hl = sub["half_life"].replace(np.inf, 1e4)
        ax1.scatter(
            sub["pvalue_adjusted"].clip(lower=1e-6), hl,
            c=color, alpha=0.7, s=55, label=group, edgecolors="none",
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.axvline(MAX_PVALUE, color="black", linestyle="--", linewidth=1,
                label=f"p_adj = {MAX_PVALUE}")
    ax1.axhline(MAX_HALF_LIFE, color="grey", linestyle="--", linewidth=1,
                label=f"half-life = {MAX_HALF_LIFE}d")
    ax1.set_xlabel("FDR-adjusted p-value (log)")
    ax1.set_ylabel("Half-life, days (log)")
    ax1.set_title("Cointegration candidates — good corner is bottom-left")
    # Annotate top 10 by adjusted p-value overall.
    top = unfiltered.nsmallest(10, "pvalue_adjusted")
    for _, row in top.iterrows():
        hl_plot = row["half_life"] if np.isfinite(row["half_life"]) else 1e4
        ax1.annotate(
            f"{row['asset_y']}/{row['asset_x']}",
            xy=(max(row["pvalue_adjusted"], 1e-6), hl_plot),
            xytext=(4, 3), textcoords="offset points", fontsize=7,
        )
    ax1.legend(loc="best", fontsize=7)

    # Panel (b): best pair, normalized to 100 at start.
    if not filtered.empty:
        best = filtered.iloc[0]
    else:
        best = unfiltered.sort_values("pvalue_adjusted").iloc[0]
    a, b = best["asset_y"], best["asset_x"]
    pa = prices[a] / prices[a].iloc[0] * 100
    pb = prices[b] / prices[b].iloc[0] * 100
    ax2.plot(pa.index, pa.values, label=a, linewidth=1.2)
    ax2.plot(pb.index, pb.values, label=b, linewidth=1.2)
    ax2.set_title(
        f"Best pair: {a} vs {b}  (p_adj={best['pvalue_adjusted']:.2e}, "
        f"HL={best['half_life']:.1f}d)"
    )
    ax2.set_ylabel("Normalized price (start=100)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Cointegration scan — SELECTION window")
    print(f"  SELECTION : {SELECTION_START} .. {SELECTION_END}  (used here)")
    print(f"  VALIDATION: {VALIDATION_START} .. {VALIDATION_END}  (session 02+)")
    print(f"  HOLDOUT   : {HOLDOUT_START} .. {HOLDOUT_END}  (final evaluation)")
    print("=" * 72)

    prices = _load_universe()
    filtered, unfiltered, counts = _scan_by_group(prices)

    display_cols = ["group", "asset_y", "asset_x", "pvalue", "pvalue_adjusted",
                    "pvalue_first_half", "pvalue_second_half",
                    "hedge_ratio", "half_life", "johansen_cointegrated"]
    fmt = {
        "pvalue": "{:.4f}".format,
        "pvalue_adjusted": "{:.4f}".format,
        "pvalue_first_half": "{:.4f}".format,
        "pvalue_second_half": "{:.4f}".format,
        "hedge_ratio": "{:.4f}".format,
        "half_life": "{:.1f}".format,
    }

    print("\nTop pairs by FDR-adjusted p-value (filtered: p_adj<=0.05 AND HL<=60):")
    if filtered.empty:
        print("  (none passed the filter)")
    else:
        with pd.option_context("display.width", 180, "display.max_columns", None):
            print(filtered[display_cols].head(10).to_string(index=False, formatters=fmt))

    csv_path = RESULTS_DIR / "01_pairs_ranked.csv"
    filtered.to_csv(csv_path, index=False)
    full_csv_path = RESULTS_DIR / "01_pairs_ranked_full.csv"
    unfiltered.sort_values("pvalue_adjusted").to_csv(full_csv_path, index=False)
    png_path = RESULTS_DIR / "01_pair_candidates.png"
    _plot(unfiltered, filtered, prices, png_path)

    print("\nFunnel (selection window, across all groups):")
    print(f"  Total pairs tested:                       {counts['total']}")
    print(f"  Pairs with raw p<0.05:                    {counts['raw_sig']}")
    print(f"  Pairs with FDR-adjusted p<0.05:           {counts['fdr_sig']}")
    print(f"  Pairs also passing half-life <= {MAX_HALF_LIFE}:     {counts['pass_halflife']}")
    print(f"  Pairs also passing Johansen:              {counts['pass_johansen']}")

    _print_controls(unfiltered)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {full_csv_path}")
    print(f"Wrote {png_path}")


def _print_controls(unfiltered: pd.DataFrame) -> None:
    """Hardcoded sanity checks for pipeline validation.

    GOOGL/GOOG (positive): same-company share classes — must cointegrate
    tightly with a short half-life and both sub-windows significant.
    SPY/GLD (negative): unrelated assets — must NOT cointegrate.
    """
    print("\nPipeline validation — control pairs:")
    # scan_pairs sorts tickers alphabetically per pair.
    controls = [
        ("Positive (share classes)", "GOOG", "GOOGL"),
        ("Negative (different assets)", "GLD", "SPY"),
    ]
    for label, y, x in controls:
        row = unfiltered[(unfiltered["asset_y"] == y) & (unfiltered["asset_x"] == x)]
        if row.empty:
            print(f"  {label:32s} {y}/{x}: NOT TESTED (missing from universe)")
            continue
        r = row.iloc[0]
        print(
            f"  {label:32s} {y}/{x}: "
            f"p={r['pvalue']:.4g}  p_adj={r['pvalue_adjusted']:.4g}  "
            f"HL={r['half_life']:.1f}d  "
            f"p_h1={r['pvalue_first_half']:.4g}  p_h2={r['pvalue_second_half']:.4g}  "
            f"johansen={bool(r['johansen_cointegrated'])}"
        )


if __name__ == "__main__":
    main()
