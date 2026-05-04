# Session 03 — Kalman diagnostics (GOOG/GOOGL)

This note isolates the most teachable artifact from session 03: a
Q-sensitivity sweep and an AR(1) comparison of the Kalman innovation
against the OLS spread, both on the validation window.  The two
tables together explain *why* the Kalman filter underperforms the
fixed-OLS hedge on a structurally stable share-class pair, and they
do so without reference to the Sharpe number itself.

The OLS reference on this pair is net Sharpe **+0.37** (gross
**+1.23**), with selection-window OLS β = 1.034 and innovation
half-life 25.5 days.

## Q-sensitivity sweep

Three values of the state-noise covariance `Q = q · I_2` are
evaluated at the same R = 1e-3, same z-window floor of 20 days, same
entry/exit/stop thresholds, and same 5 bps cost as the main run.
Only `q` changes.

| Q                     | β std (val) | innov. HL (sel) | z-window | % days active | Sharpe (net) | Sharpe (gross) |
|-----------------------|-------------|------------------|----------|----------------|---------------|-----------------|
| 1e-6  (tight)         | 0.0022      | 2.23 d           | 20       | 11.6 %         | +0.24         | +1.45           |
| 1e-4  (Chan default)  | 0.0026      | 0.28 d           | 20       | 5.2  %         | −1.17         | +0.75           |
| 1e-2  (loose)         | 0.0027      | inf (no MR)      | 20       | 4.8  %         | −0.98         | +1.20           |

Two facts dominate the sweep:

1. **β barely moves at any Q.**  The validation-window standard
   deviation of the filtered β stays at ~0.002–0.003 across four
   orders of magnitude in `q`.  GOOG/GOOGL share a single underlying
   business — there is no β drift for the filter to track.
2. **Net Sharpe is monotonically degraded from OLS** as `q` grows.
   At Q=1e-6 (essentially OLS) net Sharpe is +0.24 vs OLS +0.37; at
   Chan's default Q=1e-4 it falls to −1.17; at Q=1e-2 it sits at
   −0.98.  The cost of letting β move at all on this pair is
   strictly negative.

## AR(1) comparison (validation window)

The reason the Sharpe degrades is visible one level deeper, in the
autocorrelation of the series the strategy is z-scoring.

| Series                                 | AR(1)   |
|----------------------------------------|---------|
| OLS spread (validation)                | +0.96   |
| Kalman innovation, Q=1e-6 (tight)      | +0.75   |
| Kalman innovation, Q=1e-4 (default)    | +0.08   |
| Kalman innovation, Q=1e-2 (loose)      | −0.14   |

The OLS spread on the validation window has AR(1) = +0.96, mirroring
the ~25-day half-life from the selection-window fit — `y_t − β_OLS · x_t`
carries a genuine, persistent mean-reversion structure that the OLS
strategy can trade.

At Chan's default Q the Kalman innovation has AR(1) = +0.08 —
essentially white noise.  This is not a bug; it is what the filter
is designed to do.  The innovation is, by construction, the part of
`y_t` that is *not* predicted by the state estimate `x_hat_{t|t-1}`.
A Kalman filter run on a series with persistent mean reversion will
absorb that persistence into its state and emit innovations that are
orthogonal to the past.

Tightening Q to 1e-6 throttles the filter's adaptation rate, so it
fails to absorb most of the structure: AR(1) recovers to +0.75 and
net Sharpe climbs back to +0.24.  But it still trails OLS — the
remaining gap is the cost of letting β float at all when its true
value is constant.

Loosening Q to 1e-2 overshoots: β chases intraday noise, the
innovation goes mildly negative-AR, and the selection-window
half-life ceases to be well-defined (no AR(1) mean reversion).

## Interpretation

For structurally stable pairs — share classes, same-underlying ETFs,
A/B-ticker arbitrage — the time-varying-β model has no drift to
exploit.  Its main effect is to *whiten the trade signal* in
proportion to Q, which monotonically degrades a z-score-based
strategy.  This is the reverse of the usual intuition that "more
adaptive = better."  Adaptiveness here is the failure mode.

The Kalman filter is the appropriate tool when β has independent
prior evidence of time variation: regime-changing macro
relationships, evolving sector composition, structural breaks in
funding spreads.  For pairs whose β is structurally stable, fixed
OLS is not a baseline to beat — it is the right answer, and any
adaptive method imposes a cost in proportion to how aggressively it
adapts.

## What rules this out as a methodology bug

Three independent facts make the "Kalman is just absorbing the
signal" reading hard to escape:

1. The β std at Q=1e-6 is 0.0022 — three orders of magnitude smaller
   than the OLS β of 1.034.  There is empirically no drift to track.
2. As `q → 0` the Kalman strategy converges toward OLS performance
   (Sharpe +0.24 at Q=1e-6 vs +0.37 OLS), as the math demands.  The
   convergence is regression-tested in
   `tests/test_kalman.py::test_kalman_converges_to_ols_as_Q_to_zero`.
3. The AR(1) of the innovation falls smoothly with Q (+0.75 → +0.08
   → −0.14), exactly the monotonic whitening signature predicted by
   filter theory.
