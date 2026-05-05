# pairs-trading-research

A research investigation into cointegration-based pairs trading on US
equities, 2015–2024. Pairs are selected on 2015–2020 with FDR-corrected
Engle-Granger and Johansen tests, backtested on 2021–2022 with explicit
transaction costs, signal lag, and a real cash rate, and a 2023–2024
holdout window is reserved for a single end-of-project evaluation.
The project sits on top of a separately developed backtesting framework
([Backtest-Engine](https://github.com/moritz-physics/Backtest-Engine))
and is deliberately honest about what works and what does not over the
evaluated period — every section reports both positive findings and
clean negatives.

![Two pair backtests on the validation window](results/02_pair_backtest.png)

## Findings

**F1 — A multiple-testing-corrected single-stock scan produces zero
robust pairs.** Across an S&P 500 sector-clustered single-stock
universe over 2015–2020, no pair survived FDR-Benjamini-Hochberg
correction at α=0.05 combined with a 60-day half-life filter and a
sub-window stability check. This is the signature finding of the
post-2003 pairs-trading literature reproduced on a recent window: the
cross-sectional structure has been arbitraged out of US large-cap
single names. See `results/01_pair_candidates.png` and
`results/01_pairs_ranked_full.csv`.

**F2 — The pipeline is validated against explicit positive and
negative controls.** Before publishing the null result we re-ran the
identical scan on a structural universe — share classes, same-index
ETFs, same-underlying commodity ETFs — where cointegration is
structurally guaranteed for some pairs and structurally absent for
others. GOOG/GOOGL passes (FDR-adjusted p=0.031, half-life 25 days);
the negative control SPY/GLD correctly fails (p=0.34, Johansen False).
A nuance worth flagging: several near-tracking ETF pairs (SPY/IVV,
GLD/IAU) fail Engle-Granger because their residuals are dominated by
tracking noise too small for ADF to reject — Johansen, which does not
condition on a fitted residual, picks them up. We surface both tests
rather than collapsing to one.

**F3 — On the validation window an OLS-hedged GOOG/GOOGL trade
delivers a small honest positive net Sharpe.** With β frozen on
selection-window log-prices and a rolling z-score driving entries at
|z|>2 / exits at |z|<0.5 / stops at |z|>4 with 5 bps round-trip cost
and DTB3 cash accrual on idle days, GOOG/GOOGL posts net Sharpe +0.37
(gross +1.23) with maximum drawdown −1.83%. The faster GLD/IAU pair
illustrates the cost mechanism that defeats most candidates: at a
~5-day half-life it round-trips ~50× per year, gross Sharpe is +2.71,
and 5 bps per trade is enough to drag net Sharpe to −3.41. See
`results/02_pair_backtest.png` and `results/02_pair_metrics.csv`.

**F4 — The session-02 work surfaced a multi-asset cash accrual bug in
the underlying Backtest-Engine framework.** When a strategy is
partially deployed (some legs at zero weight on a given day), the
engine was failing to accrue the risk-free rate on the idle capital,
biasing reported Sharpe upward for any pair strategy that sits in cash
much of the time. The bug was fixed in Backtest-Engine in a separate
commit with a regression test, and session 02 was rerun against the
patched engine. The fix lowered GOOG/GOOGL's reported net Sharpe — an
example of methodological rigor producing more conservative numbers,
not better ones.

**F5 — A Kalman-filter hedge ratio underperforms fixed OLS on the same
pair, by a mechanism we identify directly.** Replacing the fixed β
with a 2-state Kalman filter (α, β) at Chan's default
Q=10⁻⁴·I drops GOOG/GOOGL's net Sharpe from +0.37 to −1.17. The
mechanism is not slippage or tuning — the validation-window AR(1)
coefficient of the OLS spread is +0.96, while the Kalman innovation
at default Q has AR(1) = +0.08. The filter is doing what it is
designed to do: absorbing the predictable mean-reversion structure
into the state estimate and emitting orthogonal innovations. A Q-sweep
over four orders of magnitude shows the filtered β std stays at
~0.002–0.003 throughout — there is no β drift on this pair to track —
and net Sharpe degrades monotonically as Q grows. Adaptiveness here
is the failure mode. See `results/03_kalman_pair_backtest.png` and
`results/03_kalman_diagnostics.md`.

**F6 — A four-pair portfolio with three weighting schemes
underperforms the best single pair, despite essentially zero
pairwise correlation among pair returns.** Combining GOOG/GOOGL,
GLD/IAU, GOOGL/MSFT, and SIVR/SLV under equal-weight, inverse-vol,
and a half-life-preferring scheme produces validation-window net
Sharpes of −1.29, −1.40, and −0.99 respectively, against +0.37 for
GOOG/GOOGL standalone. Pair returns are nearly uncorrelated, so
diversification is genuinely available — but with three of four
candidates carrying negative-Sharpe edges, no allocation rule can
average them up to a winner. Inverse-vol specifically misallocates:
it concentrates capital on GLD/IAU (lowest realised σ) precisely
because that pair is cost-destroyed and emits small returns. See
`results/04_pairs_portfolio.png` and
`results/04_pairs_portfolio_metrics.csv`.

## Methodology

**Three-way data split.** Selection 2015-01-01 to 2020-12-31 (pair
selection and hedge-ratio fitting). Validation 2021-01-01 to
2022-12-31 (every backtest in the project). Holdout 2023-01-01 to
2024-12-31, untouched. Window constants live in
`src/pairs_trading/selection.py` and are imported everywhere, so
there is one source of truth.

**Cointegration testing.** Engle-Granger (regress `y` on `x`, ADF on
residuals) is the primary test, with Johansen run independently as a
robustness check that does not condition on a fitted residual.
P-values are corrected across the full scan with Benjamini-Hochberg
FDR at α=0.05; pairs are filtered on adjusted p-value AND
AR(1)-implied spread half-life ≤ 60 days. For any pair with raw
EG p<0.05 we additionally run EG on the first and second halves of
the selection window and surface those p-values as a sub-window
stability diagnostic.

**Backtest assumptions.** All strategies use the Backtest-Engine
contract: `signal_lag=1` (yesterday's signal drives today's position),
`LinearCost(5.0)` (5 bps per unit of |Δposition|), and a daily DTB3
cash rate accrued on idle capital. Pair weights are dollar-neutral on
the chosen β: a +1 spread position is `(+1·y, −β·x)`, scaled to a
target gross of 1.0 of pair-strategy capital and aggregated across
active pairs at the portfolio level (capital budget normalisation each
day, see `src/pairs_trading/portfolio.py`). Z-scores use a rolling
window equal to twice the selection-window half-life, floored at 20
days.

**Universe choice.** The single-stock S&P 500 sector scan was the
first attempt; it produced no FDR-survivors and is reported as F1.
The scans that drive sessions 02–04 use a smaller structural universe
(share classes, same-index ETFs, same-underlying commodity ETFs,
sector overlaps, plus three session-04 additions: USO/BNO, MSFT/GOOGL,
JPM/BAC) on which positive controls *should* survive. This is the
right design for a teaching project — we want to know that the
pipeline can find structure where structure exists, not just that it
fails to invent structure where none does.

## Repository layout

```
src/pairs_trading/
    selection.py     # cointegration tests, half-life, FDR scan, window constants
    hedge_ratio.py   # OLS hedge ratio and live spread construction
    kalman.py        # 2-state Kalman filter (alpha, beta) for time-varying hedge
    signals.py       # rolling z-score and entry/exit/stop state machine
    workflows.py     # end-to-end OLS pair backtest used by sessions 02-04
    portfolio.py     # multi-pair portfolio with three weighting schemes

scripts/
    01_cointegration_scan.py      # session 01 entry point
    02_pair_backtest.py           # session 02 entry point
    03_kalman_pair_backtest.py    # session 03 entry point
    04_pairs_portfolio.py         # session 04 entry point

tests/                # mirrors src/, mocks any external data
results/              # PNGs and CSVs produced by the scripts
docs/writeup.md       # narrative writeup of the full research arc
notebooks/research_walkthrough.ipynb  # readable end-to-end walkthrough
```

## Quick start

Clone and install (uses [uv](https://docs.astral.sh/uv/)):

```sh
git clone https://github.com/moritz-physics/pairs-trading-research.git
cd pairs-trading-research
uv sync
```

Run the cleanest entry point:

```sh
uv run python scripts/02_pair_backtest.py
```

Outputs land in `results/`: each session writes one PNG and one CSV
named with its session number. Standard-output text duplicates the
metrics table and the diagnostics.

## Limitations and honest caveats

- **Survivorship bias.** Tickers come from yfinance using current
  membership; we do not reconstruct point-in-time S&P 500 membership
  or include delisted names. F1's null result is therefore robust
  (delisted names would not generate cointegration any easier), but
  any positive finding inherits a survivorship tilt.
- **Two-year validation window.** 2021–2022 is one tightening cycle
  and one growth-stock drawdown — the conclusions are about *this
  period*, not about pairs trading in all regimes.
- **Four pairs is not a portfolio.** The session-04 portfolio result
  is illustrative of how capital allocation interacts with
  per-component edge, not a conclusion about portfolio construction
  in general.
- **DTB3 as cash proxy.** DTB3 (3-month T-bill) slightly overstates
  true overnight cash returns by ~10–20 bps; reported net Sharpes
  benefit by a similar order of magnitude.
- **Holdout reserved.** The 2023–2024 window has not been touched.
  Evaluating the surviving GOOG/GOOGL OLS strategy on that window is
  the natural next step; we have not done it because doing so once
  ends the project.

## What I learned

- There is a real difference between *statistically* cointegrated and
  *tradably* cointegrated. Most surviving pairs in standard scans
  fail at one of the next two filters — half-life or transaction cost
  — long before they reach a backtest.
- Multiple-testing correction reshapes the candidate list more than
  any other single methodological choice. Without it the single-stock
  scan produces dozens of "candidates"; with it, none.
- Diversification is a redistribution operator, not a creation
  operator. Combining negative-edge pairs with low pairwise
  correlation produces a portfolio that hits its drawdown faster, not
  slower, than the worst component on its own.
- The Kalman filter is the right tool when β has independent prior
  evidence of drift, and the wrong tool otherwise. On a structurally
  stable pair the filter whitens exactly the autocorrelation that the
  z-score strategy is trying to trade. The mechanism is visible in
  the AR(1) coefficient of the innovation series.
- Diagnostic reporting beats inspection. The framework bug surfaced
  not because anyone was looking for it but because the scripts log
  cash-accrual on idle days as a routine line. If a number you do not
  scrutinise can be wrong, eventually it is.

## References

- Gatev, Goetzmann, Rouwenhorst (2006). "Pairs Trading: Performance
  of a Relative-Value Arbitrage Rule." *Review of Financial Studies*
  19(3).
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and
  Analysis.* Wiley.
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their
  Rationale.* Wiley. (Kalman filter pairs chapter.)
- Engle, R. and Granger, C. (1987). "Co-integration and Error
  Correction: Representation, Estimation, and Testing."
  *Econometrica* 55(2).

## Related repository

This research repo depends on
[Backtest-Engine](https://github.com/moritz-physics/Backtest-Engine)
as an editable install (`../Backtest-Engine` in `pyproject.toml`).
The two repos are co-developed: framework changes live in
Backtest-Engine with their own tests, and research changes live here.
The clearest illustration of the relationship is F4 — a multi-asset
cash accrual bug surfaced by the session-02 diagnostics in this repo
was fixed in Backtest-Engine in a separate commit with a regression
test, and session 02 was then rerun against the patched engine.
Bidirectional development is the point of keeping the framework
separate.

## License and author

MIT — see `LICENSE`. Author: Moritz Heidtmann
(`heidtmann.moritz@gmail.com`).
