"""Tests for pairs_trading.kalman."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pairs_trading.kalman import (
    KalmanHedgeResult,
    kalman_hedge_ratio,
    spread_position_to_asset_weights_tv,
)


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, freq="B")


def _simulate_pair(
    betas: np.ndarray,
    alpha: float = 0.0,
    x_drift: float = 0.0005,
    x_vol: float = 0.02,
    eps_sigma: float = 0.005,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Synthesize y and x with a prescribed beta path.

    x is a cumulated log-return process, y = alpha + beta_t * x + eps.
    The beta schedule is an array of length n.
    """
    n = len(betas)
    idx = _make_index(n)
    rng = np.random.default_rng(seed)
    x_rets = rng.normal(x_drift, x_vol, n)
    x = np.cumsum(x_rets)
    eps = rng.normal(0, eps_sigma, n)
    y = alpha + betas * x + eps
    return pd.Series(y, index=idx, name="y"), pd.Series(x, index=idx, name="x")


class TestKalmanRecoversConstantBeta:
    def test_constant_beta_recovered(self):
        n = 400
        true_beta = 2.0
        betas = np.full(n, true_beta)
        y, x = _simulate_pair(betas, alpha=0.0, eps_sigma=0.005, seed=1)

        result = kalman_hedge_ratio(y, x)

        # After ~30 days of warmup, filtered beta must track true beta.
        tail = result.betas.iloc[30:]
        assert (tail - true_beta).abs().max() < 0.1, (
            f"max|beta - 2.0| on warm tail = "
            f"{(tail - true_beta).abs().max():.4f}"
        )
        # Alpha should hover near 0.
        alpha_tail = result.alphas.iloc[30:]
        assert alpha_tail.abs().max() < 0.2


class TestKalmanTracksDriftingBeta:
    def test_step_change_is_tracked(self):
        n = 600
        betas = np.concatenate([np.full(n // 2, 2.0), np.full(n - n // 2, 3.0)])
        # Slightly larger Q so the filter adapts within a reasonable horizon.
        y, x = _simulate_pair(betas, alpha=0.0, eps_sigma=0.005, seed=2)

        result = kalman_hedge_ratio(y, x, Q=np.eye(2) * 1e-3, R=1e-3)

        first_half_tail = result.betas.iloc[100:290]
        second_half_tail = result.betas.iloc[-100:]

        # Pre-step: close to 2.0.
        assert (first_half_tail - 2.0).abs().mean() < 0.2
        # Post-step (far enough after 300 that the filter has adapted):
        # close to 3.0.
        assert (second_half_tail - 3.0).abs().mean() < 0.2
        # And directionally: the mean of the second-half tail should be
        # higher than that of the first-half tail.
        assert second_half_tail.mean() > first_half_tail.mean() + 0.5


class TestKalmanInnovationNoLookahead:
    """Critical causality test: the spread at time t must not depend on
    any data after time t.  We run the filter on the full series and on
    a truncated prefix; the overlap must match to numerical precision.
    """

    def test_truncation_equivalence(self):
        n = 300
        betas = np.linspace(1.5, 2.5, n)  # gently drifting
        y, x = _simulate_pair(betas, alpha=0.1, eps_sigma=0.003, seed=7)

        full = kalman_hedge_ratio(y, x)

        # Truncate to first k observations and rerun.
        for k in (50, 120, 237):
            trunc = kalman_hedge_ratio(y.iloc[: k + 1], x.iloc[: k + 1])
            np.testing.assert_allclose(
                full.spreads.iloc[: k + 1].to_numpy(),
                trunc.spreads.to_numpy(),
                atol=1e-12,
                rtol=0,
                err_msg=f"spreads diverge on prefix up to k={k}",
            )
            np.testing.assert_allclose(
                full.betas.iloc[: k + 1].to_numpy(),
                trunc.betas.to_numpy(),
                atol=1e-12,
                rtol=0,
            )
            np.testing.assert_allclose(
                full.alphas.iloc[: k + 1].to_numpy(),
                trunc.alphas.to_numpy(),
                atol=1e-12,
                rtol=0,
            )


class TestKalmanReducesToOLSAsQToZero:
    def test_zero_process_noise_converges_to_ols(self):
        """With Q near zero, the filter is an online OLS.  After many
        observations the terminal beta must match the OLS estimate.
        """
        n = 1500
        true_beta = 1.75
        betas = np.full(n, true_beta)
        y, x = _simulate_pair(
            betas, alpha=0.4, eps_sigma=0.01, x_vol=0.02, seed=11
        )

        # Q essentially zero -> state is (near-)constant; the filter
        # averages all data.
        result = kalman_hedge_ratio(
            y, x, Q=np.eye(2) * 1e-12, R=1.0, initial_cov=np.eye(2) * 1e3
        )

        # Compare terminal filtered (alpha, beta) to batch OLS.
        X = sm.add_constant(x.to_numpy())
        ols = sm.OLS(y.to_numpy(), X).fit()
        ols_alpha, ols_beta = float(ols.params[0]), float(ols.params[1])

        assert abs(result.betas.iloc[-1] - ols_beta) < 1e-2, (
            f"Kalman beta {result.betas.iloc[-1]:.6f} vs OLS beta "
            f"{ols_beta:.6f}"
        )
        assert abs(result.alphas.iloc[-1] - ols_alpha) < 1e-2


class TestKalmanInitialUncertaintyDecays:
    def test_diffuse_prior_shrinks(self):
        n = 200
        betas = np.full(n, 1.2)
        y, x = _simulate_pair(betas, alpha=0.0, eps_sigma=0.01, seed=13)

        diffuse = np.eye(2) * 1e6
        result = kalman_hedge_ratio(y, x, initial_cov=diffuse)

        P0 = diffuse.copy()
        P100 = result.state_cov_history[100]

        # Beta uncertainty at t=100 must be a small fraction of prior.
        ratio = P100[1, 1] / P0[1, 1]
        assert ratio < 0.01, (
            f"P[1,1] shrinkage ratio at t=100 was {ratio:.4g}, expected <0.01"
        )


class TestKalmanSpreadSignConvention:
    def test_positive_innovation_when_y_exceeds_prediction(self):
        """With default prior [alpha=0, beta=1] and y_0 chosen so that
        y_0 > 0 + 1 * x_0, the very first innovation e_0 must be > 0.

        Locks in the sign convention e_t = y_t - H_t * x_hat_{t|t-1}
        (predicted residual, with positive = "y above prediction").
        Flipping this sign would silently invert every trade.
        """
        idx = _make_index(5)
        # Start x at 1.0 and y at 2.0 — default prediction is y_pred=1.0,
        # so e_0 = 2.0 - 1.0 = 1.0 > 0.
        y = pd.Series([2.0, 2.01, 2.02, 2.03, 2.04], index=idx)
        x = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)

        result = kalman_hedge_ratio(y, x)

        assert result.spreads.iloc[0] > 0, (
            f"Expected positive innovation at t=0 (y > predicted), "
            f"got spreads[0] = {result.spreads.iloc[0]}"
        )


class TestKalmanInputValidation:
    def test_mismatched_index_raises(self):
        idx1 = _make_index(10)
        idx2 = _make_index(11)
        y = pd.Series(np.arange(10, dtype=float), index=idx1)
        x = pd.Series(np.arange(11, dtype=float), index=idx2)
        with pytest.raises(ValueError, match="identical index"):
            kalman_hedge_ratio(y, x)

    def test_nan_raises(self):
        idx = _make_index(10)
        y = pd.Series([1.0] * 10, index=idx)
        x = pd.Series([1.0] * 9 + [np.nan], index=idx)
        with pytest.raises(ValueError, match="NaN"):
            kalman_hedge_ratio(y, x)

    def test_bad_Q_shape_raises(self):
        idx = _make_index(5)
        y = pd.Series(np.zeros(5), index=idx)
        x = pd.Series(np.zeros(5), index=idx)
        with pytest.raises(ValueError, match=r"Q must be \(2, 2\)"):
            kalman_hedge_ratio(y, x, Q=np.eye(3))

    def test_non_positive_R_raises(self):
        idx = _make_index(5)
        y = pd.Series(np.zeros(5), index=idx)
        x = pd.Series(np.zeros(5), index=idx)
        with pytest.raises(ValueError, match="R must be positive"):
            kalman_hedge_ratio(y, x, R=0.0)


class TestTimeVaryingWeights:
    def test_tv_weights_basic(self):
        idx = _make_index(5)
        signal = pd.Series([0, 1, 1, -1, 0], index=idx)
        betas = pd.Series([1.0, 1.2, 0.9, 1.1, 1.0], index=idx)

        w = spread_position_to_asset_weights_tv(
            signal, betas, pair=("A", "B")
        )

        assert list(w.columns) == ["A", "B"]
        np.testing.assert_array_almost_equal(
            w["A"].values, [0.0, 1.0, 1.0, -1.0, 0.0]
        )
        # x-weight = -signal * beta (elementwise).
        np.testing.assert_array_almost_equal(
            w["B"].values, [0.0, -1.2, -0.9, 1.1, 0.0]
        )

    def test_tv_weights_index_mismatch_raises(self):
        signal = pd.Series([0, 1], index=_make_index(2))
        betas = pd.Series([1.0, 1.1, 1.2], index=_make_index(3))
        with pytest.raises(ValueError, match="share an identical index"):
            spread_position_to_asset_weights_tv(
                signal, betas, pair=("A", "B")
            )


class TestKalmanResultDataclass:
    def test_result_is_frozen(self):
        idx = _make_index(3)
        y = pd.Series([1.0, 1.1, 1.2], index=idx)
        x = pd.Series([0.5, 0.55, 0.6], index=idx)
        r = kalman_hedge_ratio(y, x)
        assert isinstance(r, KalmanHedgeResult)
        with pytest.raises(Exception):
            r.betas = r.alphas  # frozen
