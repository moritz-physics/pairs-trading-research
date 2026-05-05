# A Cointegration-Based Pairs Trading Investigation, 2015–2024

A research note describing what we tried, what we found, and what we
deliberately did not claim. The companion code lives in
`scripts/01_cointegration_scan.py` through
`scripts/04_pairs_portfolio.py` and the figures referenced below are
in `results/`.

## 1. Motivation and research question

Pairs trading is one of the canonical relative-value strategies and
one of the cleanest pedagogical vehicles for the broader idea of
trading deviations from a stationary linear combination. The seminal
empirical result (Gatev, Goetzmann, Rouwenhorst, 2006) found that
distance-based pairs trading on US stocks earned ~11% annualised
excess returns from 1962–2002. The follow-up literature, which we
take as our starting point rather than a destination, finds that the
strategy's edge has decayed substantially after 2003 — by some
estimates to statistical insignificance once realistic costs are
applied.

The question we set out to answer is narrower than "does pairs
trading work today." We asked: on a recent ten-year window
(2015–2024) of US equities, with a methodologically careful pipeline
— multiple-testing-corrected cointegration tests, a held-out
validation window, explicit transaction costs, no in-sample tuning —
what survives? And, separately, what does the failure mode look like
when something doesn't survive? The latter turns out to be the more
interesting half of the project.

The deliverable is not a strategy. It is a sequence of honest
diagnostics, plus one small surviving signal (GOOG/GOOGL on
validation) and one cleanly negative portfolio finding, all built on
top of a backtesting framework
([Backtest-Engine](https://github.com/moritz-physics/Backtest-Engine))
that the author developed in parallel.

## 2. Methodology overview

We describe the pipeline once, in the order an actual run executes.

**Three-way data split.** Selection 2015-01-01 to 2020-12-31, used
for cointegration testing and OLS hedge-ratio fitting. Validation
2021-01-01 to 2022-12-31, used for every backtest in the project.
Holdout 2023-01-01 to 2024-12-31, untouched. Window constants live
in `src/pairs_trading/selection.py` and are imported everywhere.

**Engle-Granger test.** For two log-price series `y_t` and `x_t`, we
run OLS

    y_t = α + β x_t + ε_t

and ADF on the fitted residuals `ε̂_t = y_t − α̂ − β̂ x_t` against
the null of a unit root. Rejection at α=0.05 (or, after multiple-
testing correction, at adjusted α=0.05) is read as evidence of
cointegration. The OLS β doubles as the hedge ratio.

**Half-life.** For a candidate spread `s_t` we fit AR(1)

    Δs_t = a + b · s_{t−1} + η_t

and report

    HL = −log 2 / log(1 + b)

as the spread's mean-reversion half-life in days. Pairs with HL > 60
days are dropped: even when statistically cointegrated, slow pairs
do not produce enough round-trips for a noisy estimate of expected
return to be meaningfully positive after costs.

**Johansen test.** Run independently on the same pair as a 2-series
trace test against the null of zero cointegrating vectors
(`det_order=0`, `k_ar_diff=1`). Johansen does not condition on a
fitted residual, so it has different power to Engle-Granger; we keep
both and surface their disagreement explicitly.

**FDR correction.** Across the full scan of unordered pairs (n
varies by universe) we apply Benjamini-Hochberg to the
Engle-Granger p-values:

    p_(i) ≤ (i / n) · α     ⇒  reject H0 for tests 1..i*

with α=0.05. With n ≈ 100, ~5 raw rejections are expected by chance;
without correction the candidate list is dominated by noise.

**Sub-window stability.** For any pair with raw EG p<0.05, we re-run
EG on the first and second halves of the selection window and
surface both p-values. Pairs that pass on the full window but fail
on one half are flagged as single-regime artifacts rather than
filtered out automatically — judgment is preserved at the writer's
desk, not buried in the scan.

**Z-score signal.** The trade signal is the rolling z-score of the
live spread `s_t = log y_t − β̂ log x_t` (no intercept; the rolling
window's centring term handles spread mean). Window length =
`max(2 · HL, 20)`. Entries at |z|>2, exits at |z|<0.5, hard stop at
|z|>4 (assume the pair has broken). Thresholds are passed as
explicit arguments, not magic numbers.

**Kalman alternative.** Where used (session 03), the OLS β is
replaced with the filtered β_{t|t} from a 2-state Kalman recursion
on (α_t, β_t) with state evolution `x_{t+1} = x_t + w_t`,
`w_t ∼ N(0, Q)` and observation `y_t = [1, x_t]·x_t + v_t`,
`v_t ∼ N(0, R)`. The filter's innovation `e_t = y_t − [1, x_t] ·
x̂_{t|t−1}` uses the predicted state through `t−1` only and is
therefore causal. We report `Q = 10⁻⁴ · I_2`, `R = 10⁻³` (Chan
2013 defaults) and a Q sweep at 10⁻⁶ and 10⁻² for sensitivity.

**Backtest contract.** Every backtest is run through Backtest-
Engine's `run_backtest` with `signal_lag=1` (today's position
is yesterday's signal), `LinearCost(5.0)` (5 bps per unit of
|Δposition|), and `cash_rate=DTB3` (forward-filled FRED rate, idle
capital accrues the previously-published DTB3). Pair weights are
dollar-neutral on β: a +1 spread position is `(+1·y, −β·x)`.

## 3. Universe construction

The first scan was a sector-clustered S&P 500 single-stock universe:
30 large caps spanning Tech, Financials, and Energy on 2015–2020.
After FDR correction with the half-life filter and the sub-window
stability check, **zero** pairs survived. This is not a pathology of
the pipeline; it is consistent with the published finding that
post-2003 single-stock pairs trading on US large caps has been
heavily arbitraged out. We report the result as F1.

Before publishing F1 we needed to rule out a pipeline bug. We
re-ran the identical scan on a structural universe where
cointegration is *guaranteed* for some pairs and *guaranteed
absent* for others — share classes (GOOG/GOOGL), same-index ETFs
(SPY/IVV/VOO/QQQ), same-underlying commodity ETFs (GLD/IAU,
SLV/SIVR), sector overlaps, and an explicit negative control
(SPY/GLD). Every other parameter (window, FDR, half-life filter,
sub-window check) was held identical to the failed scan.

Two pairs survived the strict filter:

- **SPY/VOO** (Same-index ETFs, FDR-adjusted p=7×10⁻⁴, HL=0.9d)
- **GOOG/GOOGL** (Share classes, FDR-adjusted p=0.031, HL=25d)

The negative control SPY/GLD correctly failed (raw p=0.34, Johansen
False).

A nuance worth flagging: several near-tracking ETF pairs (SPY/IVV,
GLD/IAU, SLV/SIVR) all have hedge ratios near 1.0 and very short
half-lives but fail Engle-Granger because the residuals are
dominated by tracking noise too small for ADF to reject as
stationary against a unit-root null. Johansen, which doesn't
condition on a fitted residual, picks up GLD/IAU and SLV/SIVR. We
keep both tests in the output rather than restructuring the primary
filter — the disagreement is itself diagnostic.

The pairs carried into the rest of the project are GOOG/GOOGL
(strong Engle-Granger plus Johansen, real spread variance to trade),
GLD/IAU (Johansen-only, used to demonstrate the cost-destruction
mechanism), and three additional structural pairs introduced for
session 04's portfolio construction (GOOGL/MSFT, SIVR/SLV, plus the
pre-existing GLD/IAU).

See `results/01_pair_candidates.png` for the scatter of FDR-adjusted
p-value vs half-life, coloured by structural group, with the gating
thresholds drawn in.

## 4. OLS baseline (session 02)

With the structural pairs in hand, session 02 implements the
simplest possible OLS pair backtest. β is estimated on log-prices
over the selection window and frozen; the live spread is `log y_t −
β̂ log x_t`; a rolling z-score with window `max(2 · HL, 20)` drives
entries, exits, and stops; weights are dollar-neutral; trades cost
5 bps; idle capital earns DTB3.

**Results.** On GOOG/GOOGL the strategy posts net Sharpe **+0.37**
with annualised volatility 1.85%, max drawdown −1.83%, and 33.7% of
days active. The gross Sharpe is +1.23, so costs eat about two
thirds of the gross edge — but a third survives, and the strategy
beats both static-spread (-0.25 net Sharpe) and long-y B&H
(0.16) on Sharpe. This is small. It is not zero.

GLD/IAU illustrates the cost mechanism that defeats most candidates.
The pair has a half-life of ~5 days and round-trips ~50× per year.
Gross Sharpe is +2.71 — there is a real signal. Net Sharpe is
**−3.41**. Five basis points per trade is enough to flip the sign,
because the strategy pays the cost ~50 times to capture an annual
volatility of 0.49% gross. Fast pairs are cost-destroyed pairs.

The figure `results/02_pair_backtest.png` shows both pairs in the
same panel — z-score with bands and in-position shading,
net/gross/cash equity curves, and net-return drawdown. The visual
contrast between the two pairs makes the cost story legible at a
glance.

![Two pair backtests on the validation window](../results/02_pair_backtest.png)

## 5. The framework bug interlude

While reviewing session 02's exposure diagnostics we noticed that
on days when both legs of a pair were at zero weight the portfolio
return was *exactly zero*, not the daily DTB3 rate. With strategies
that are in-position only ~30% of days, this is a substantial
silent overstate of net performance. The bug was in
Backtest-Engine: the multi-asset cash-accrual path was zeroed when
the position vector was zeroed, instead of being accrued on the
idle capital.

The fix lives in Backtest-Engine in a separate commit with a
regression test. After the patch, session 02 was rerun, and the
GOOG/GOOGL net Sharpe number reported above is the post-fix figure
— lower than the pre-fix figure was. Methodological rigor sometimes
buys you smaller numbers; that is what it should buy you.

## 6. Kalman filter (session 03)

A natural next step from a frozen-β OLS hedge is a time-varying β.
The Kalman filter is the textbook tool, and Chan (2013) walks
through exactly this construction with the same recommended priors
on a daily-equity pair. We implemented the 2-state filter on (α, β)
in `src/pairs_trading/kalman.py`, wired it through the same backtest
contract as session 02, and compared.

**Result.** GOOG/GOOGL net Sharpe drops from **+0.37 (OLS)** to
**−1.17 (Kalman, default Q)**. GLD/IAU goes from −3.41 to −3.30.
The Kalman version is not a small underperformance — it is a
sign-flipping degradation on the only pair with positive net edge.

This is the kind of result that demands a mechanism, not a
hyperparameter excuse. Session 03 includes a diagnostic block,
reproduced verbatim from `results/03_kalman_diagnostics.md`, that
isolates two facts.

**Q-sensitivity sweep.** Three values of `Q = q · I_2` at fixed
R = 10⁻³, same z-window floor of 20 days, same costs:

| Q                     | β std (val) | innov. HL (sel) | z-window | % days active | Sharpe (net) | Sharpe (gross) |
|-----------------------|-------------|------------------|----------|----------------|---------------|-----------------|
| 1e-6  (tight)         | 0.0022      | 2.23 d           | 20       | 11.6 %         | +0.24         | +1.45           |
| 1e-4  (Chan default)  | 0.0026      | 0.28 d           | 20       | 5.2  %         | −1.17         | +0.75           |
| 1e-2  (loose)         | 0.0027      | inf (no MR)      | 20       | 4.8  %         | −0.98         | +1.20           |

β std stays at 0.002–0.003 across four orders of magnitude in q.
GOOG/GOOGL share a single underlying business — there is empirically
no β drift to track, regardless of how much the filter is allowed to
adapt. Net Sharpe degrades monotonically as q grows.

**AR(1) of the trade signal.** The mechanism is visible one level
deeper, in the autocorrelation of the series the strategy is
z-scoring:

| Series                                 | AR(1)   |
|----------------------------------------|---------|
| OLS spread (validation)                | +0.96   |
| Kalman innovation, Q=1e-6 (tight)      | +0.75   |
| Kalman innovation, Q=1e-4 (default)    | +0.08   |
| Kalman innovation, Q=1e-2 (loose)      | −0.14   |

The OLS spread has AR(1) +0.96 — a strong, persistent
mean-reversion structure mirroring the ~25-day half-life from the
selection-window AR(1) fit. This is what a z-score strategy needs
to trade.

At Chan's default Q the Kalman innovation has AR(1) +0.08 —
essentially white noise. This is not a bug. The innovation is, by
construction, the part of `y_t` that is *not* predicted by the
state estimate. A Kalman filter run on a series with persistent
mean reversion will absorb that persistence into its state and
emit innovations orthogonal to the past. The z-score-based
strategy then has nothing to trade.

Tightening Q to 10⁻⁶ throttles the filter's adaptation rate, AR(1)
recovers to +0.75, and net Sharpe climbs back to +0.24 — but still
trails OLS by ~0.13. The remaining gap is the cost of letting β
move at all on a pair whose true β is constant. Loosening Q to
10⁻² overshoots: β chases intraday noise, AR(1) goes mildly
negative, and the half-life is undefined.

The conclusion is that for structurally stable pairs the Kalman
filter is not an upgrade to OLS — it is an *adaptiveness penalty*
proportional to Q, with no drift to compensate. The filter is the
right tool when β has independent prior evidence of time variation
(regime-changing macro relationships, evolving sector composition,
funding-spread structural breaks), and the wrong tool here.

![Kalman vs OLS pair backtest](../results/03_kalman_pair_backtest.png)

## 7. Portfolio construction (session 04)

The final session attempts to combine pair strategies into a
portfolio. The pair set is GOOG/GOOGL (strong evidence),
GLD/IAU and SIVR/SLV (Johansen-only structural pairs, included to
populate the portfolio), and GOOGL/MSFT (the only new
FDR-significant pair from an expanded universe scan, on weak
evidence). Three weighting schemes are run:

- **Equal**: 1/K_t to each active pair.
- **Inverse-vol**: weight ∝ 1/σ_i, with σ_i estimated on
  selection-window in-sample backtests of each pair (same OLS β,
  same z-window, same costs — never touches validation data).
- **Prefer-slow**: weight ∝ HL_i, with pairs outside [5, 250] days
  excluded. Empirically motivated by the cost-destruction finding
  in session 02.

A daily capital-budget normalisation distributes 1.0 unit of
pair-strategy capital across active pairs each day; idle capital
accrues DTB3.

**Pairwise correlations.** The off-diagonal correlation of the four
pairs' net strategy returns is essentially zero on the validation
window. By the standard intuition this is exactly the regime where
diversification should help.

**Result.** Validation-window net Sharpes:

- GOOG/GOOGL standalone: **+0.37**
- Portfolio, equal-weight: **−1.29**
- Portfolio, inverse-vol: **−1.40**
- Portfolio, prefer-slow: **−0.99**

The best portfolio underperforms the best single component by ~1.4
in Sharpe units. Diversification redistributes risk; it does not
manufacture edge. With three of four candidates carrying
negative-Sharpe edges, no allocation rule can average them up to a
winner. The portfolio drawdown is in fact deeper than any single
pair's, illustrating that diversification of negative-expectation
components compounds losses rather than smoothing them.

Inverse-vol deserves a separate note. It assigned its largest
weight to GLD/IAU because that pair has the lowest realised σ —
but GLD/IAU is the cost-destroyed pair. Low realised volatility is
not the same as high signal quality; it can simply reflect small
return magnitudes, including small negative return magnitudes. A
robust capital allocator needs Sharpe estimation, not vol
estimation, but in-sample Sharpe is poorly estimated. Inverse-vol
at this scale is not a cautious default — it is an active bet that
the lowest-σ pair has the highest *unobserved* edge.

![Portfolio of pairs](../results/04_pairs_portfolio.png)

## 8. Discussion

Six findings, considered together, support a cautious reading. F1
reproduces the post-2003 finding that single-stock pairs trading
on US large caps does not survive multiple-testing correction. F2
verifies that the pipeline is not over- or under-rejecting by
construction. F3 produces a small honest positive number on a
share-class arb. F4 lowers that number by fixing a bug in our own
infrastructure. F5 fails to improve on F3 by a more sophisticated
hedge model, and identifies the mechanism precisely. F6 fails to
improve on F3 by portfolio construction, again with a clean
mechanism.

The picture is consistent: there is residual edge in structural
pairs that share an actual underlying claim (share classes,
co-listings, ADRs), and approximately none in pairs that share only
a statistical relationship. Adding sophistication on top of weak
edges does not rescue them; it usually makes them worse, in ways
the diagnostics make legible.

Should pairs trading be treated as a dead strategy on US large
caps? For a retail trader paying 5+ bps per round-trip, the
honest answer is yes — the strategy as a *general* approach to US
large caps does not appear viable on this window. For a
sophisticated practitioner with sub-bp execution, alternative
universes (international markets, dual-listed equities, ADR/local
pairs, cross-exchange variants), and access to richer hedge models
than fixed OLS or 2-state Kalman, there are still pockets of edge
— GOOG/GOOGL and SPY/VOO are existence proofs that the structural
class survives. The strategy is not dead so much as no longer
indiscriminate.

This project does not establish that pairs trading is universally
unprofitable — it cannot, with one universe, one window, and one
cost regime. It does establish that on this universe, this window,
and this cost regime, only the structurally guaranteed pairs
survive, and only the slowest of those produce a positive net
Sharpe.

## 9. Future directions

The **2023–2024 holdout** has been deliberately untouched. The
natural next step is a single end-of-project run of the surviving
GOOG/GOOGL OLS strategy on that window. If the net Sharpe stays
positive, that is meaningful out-of-sample evidence of edge
persistence; if it does not, that is meaningful evidence of
strategy decay. We do not run it twice.

**Universe expansion.** International dual-listings (e.g.
Royal Dutch Shell A/B before consolidation, BHP/RIO across LSE/ASX),
ADR/local pairs (e.g. TSM US/2330 TT), and cross-exchange variants
are the universes where structural cointegration arguably still
survives at scale. The pipeline as built generalises to any
NaN-free price panel; the bottleneck is data, not code.

**Cost models.** A linear 5 bps cost is a coarse approximation;
real execution involves spread, market impact, and a non-linear
relationship to trade size. A more sophisticated cost model would
likely deepen rather than reverse F3 and F6, but the magnitude
matters for any practitioner-facing claim.

**Higher frequencies.** The fast-pair-cost-destruction story is
specific to daily data with retail-grade costs. At intraday
frequencies with sub-bp costs the trade-off changes; the present
pipeline does not address that regime.

## 10. Acknowledgments

The Backtest-Engine framework was developed in parallel with this
research repo (see
[Backtest-Engine](https://github.com/moritz-physics/Backtest-Engine)).
Several design choices in the framework — the explicit `signal_lag`
contract, the multi-asset cash-rate path, the `LinearCost` interface
— were sharpened by the demands of running pair strategies through
it, and at least one bug (F4) was found by this repo's diagnostics.

The work was iterated with assistance from Claude Code as a
pair-programming partner on design decisions, anti-look-ahead audits,
and writeup structure. All findings, code, and writing are the
author's own.
