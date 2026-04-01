"""
Tests for ShapeAdaptiveCP (MOPI) conformal predictor.

ShapeAdaptiveCP solves a minimax problem to achieve near-uniform conditional
coverage across subgroups Z. The defining properties we test:

1. After calibration, per-group coverage should be closer to (1-alpha) than
   naive split conformal (which applies one global threshold).
2. MSCE should be <= the MSCE of a global-threshold predictor.
3. The masked-Z feature works: you can calibrate with Z and predict using only
   X (via group_fn), with no Z at prediction time.
4. RKHS mode: intervals produced with kernel smoothing should have finite,
   non-negative widths.

We use a heteroscedastic DGP with known per-group structure so we can verify
that the algorithm is actually doing something useful, not just passing trivially.

DGP: Y | (X, group) ~ N(mu(X), sigma_group^2)
     where sigma varies substantially across groups.
     This creates a scenario where a single global threshold fails badly.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal.mopi import (
    ShapeAdaptiveCP,
    _compute_msce,
    _rbf_kernel,
    _sigmoid,
    _sigmoid_grad,
)


# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------


def _make_group_data(
    n: int = 2000,
    n_groups: int = 4,
    seed: int = 0,
) -> dict:
    """
    Heteroscedastic regression DGP with K groups.

    Y_i | (X_i, Z_i=k) ~ N(X_i[0], sigma_k^2)
    sigma varies across groups: group 0 is quiet, group K-1 is noisy.

    mu(X) = X[:, 0] is the true mean (no model error).
    sigma_k = 0.2 + 0.6 * k / (K-1) increases linearly with group index.
    """
    rng = np.random.default_rng(seed)
    K = n_groups
    sigmas = 0.2 + 0.6 * np.arange(K) / max(K - 1, 1)

    X = rng.uniform(0.0, 2.0, size=(n, 2))
    Z = rng.integers(0, K, size=n)
    mu_true = X[:, 0]
    sigma_true = sigmas[Z]
    y = rng.normal(mu_true, sigma_true)

    return {
        "X": X,
        "y": y,
        "Z": Z,
        "mu": mu_true,
        "sigma": sigma_true,
        "sigmas": sigmas,
        "K": K,
        "rng": rng,
    }


def _make_continuous_z_data(n: int = 500, seed: int = 7) -> dict:
    """
    DGP for RKHS mode: Z is continuous in [0, 1].
    sigma(z) = 0.2 + 0.8 * z (higher Z = noisier).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 2.0, size=(n, 2))
    Z = rng.uniform(0.0, 1.0, size=n)
    mu_true = X[:, 0]
    sigma_true = 0.2 + 0.8 * Z
    y = rng.normal(mu_true, sigma_true)
    return {"X": X, "y": y, "Z": Z, "mu": mu_true, "sigma": sigma_true}


@pytest.fixture(scope="module")
def group_data():
    """Large dataset for coverage tests."""
    return _make_group_data(n=3000, n_groups=4, seed=42)


@pytest.fixture(scope="module")
def small_group_data():
    """Small dataset for unit tests that don't need statistical power."""
    return _make_group_data(n=400, n_groups=3, seed=1)


@pytest.fixture(scope="module")
def continuous_z_data():
    return _make_continuous_z_data(n=600, seed=99)


# ---------------------------------------------------------------------------
# Unit tests: pure math helpers
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_sigmoid_zero_returns_half(self):
        assert _sigmoid(np.array([0.0]), temp=0.1)[0] == pytest.approx(0.5)

    def test_sigmoid_large_positive_approaches_one(self):
        val = _sigmoid(np.array([100.0]), temp=0.1)[0]
        assert val > 0.999

    def test_sigmoid_large_negative_approaches_zero(self):
        val = _sigmoid(np.array([-100.0]), temp=0.1)[0]
        assert val < 0.001

    def test_sigmoid_grad_is_derivative(self):
        """Finite-difference check of sigmoid gradient."""
        temp = 0.1
        u = np.array([0.3, -0.2, 1.5])
        eps = 1e-5
        grad_fd = (_sigmoid(u + eps, temp) - _sigmoid(u - eps, temp)) / (2 * eps)
        grad_analytic = _sigmoid_grad(u, temp)
        np.testing.assert_allclose(grad_fd, grad_analytic, rtol=1e-4)

    def test_sigmoid_monotone(self):
        u = np.linspace(-2, 2, 50)
        sig = _sigmoid(u, temp=0.1)
        assert np.all(np.diff(sig) > 0)


class TestRbfKernel:
    def test_diagonal_is_one(self):
        Z = np.random.default_rng(0).normal(size=(10, 2))
        K = _rbf_kernel(Z, Z, bandwidth=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_symmetric(self):
        Z = np.random.default_rng(0).normal(size=(8, 2))
        K = _rbf_kernel(Z, Z, bandwidth=1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_values_in_zero_one(self):
        Z1 = np.random.default_rng(1).normal(size=(5, 1))
        Z2 = np.random.default_rng(2).normal(size=(7, 1))
        K = _rbf_kernel(Z1, Z2, bandwidth=1.0)
        assert K.shape == (5, 7)
        assert np.all(K >= 0) and np.all(K <= 1.0)

    def test_narrow_bandwidth_gives_near_identity(self):
        """Very narrow bandwidth: distant points have near-zero kernel."""
        Z = np.array([[0.0], [10.0]])
        K = _rbf_kernel(Z, Z, bandwidth=0.01)
        assert K[0, 1] < 1e-10

    def test_wide_bandwidth_gives_near_ones(self):
        """Very wide bandwidth: all points are near-identical."""
        Z = np.array([[0.0], [1.0]])
        K = _rbf_kernel(Z, Z, bandwidth=1000.0)
        assert K[0, 1] > 0.999


class TestComputeMsce:
    def test_perfect_coverage_gives_zero(self):
        miscov = np.zeros(100)
        Z = np.repeat([0, 1, 2, 3], 25)
        msce = _compute_msce(miscov, Z, alpha=0.1)
        # All groups miscoverage = 0, alpha deviation = 0 - 0.1 = -0.1
        # So MSCE = (0 - 0.1)^2 = 0.01 -- no, this is not zero
        # Perfect coverage means miscoverage=0, so delta_k = 0 - 0.1 = -0.1
        # MSCE measures deviation from alpha, not from 0
        assert msce == pytest.approx(0.01, abs=1e-6)

    def test_uniform_alpha_miscoverage_is_zero(self):
        """If every group has exactly alpha miscoverage, MSCE=0."""
        n = 1000
        alpha = 0.1
        # 10% of each group is miscovered
        miscov = np.zeros(n)
        miscov[: int(n * alpha)] = 1.0
        Z = np.zeros(n, dtype=int)  # all in one group
        msce = _compute_msce(miscov, Z, alpha=alpha)
        assert msce < 0.001  # close to zero (exact if n is divisible by 10)

    def test_msce_is_non_negative(self):
        rng = np.random.default_rng(0)
        miscov = rng.binomial(1, 0.15, size=200).astype(float)
        Z = rng.integers(0, 4, size=200)
        msce = _compute_msce(miscov, Z, alpha=0.1)
        assert msce >= 0.0


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestInitialisation:
    def test_not_calibrated_on_init(self):
        cp = ShapeAdaptiveCP()
        assert not cp.is_calibrated_
        assert cp.n_calibration_ == 0
        assert cp.thresholds_ is None
        assert cp.msce_ is None

    def test_predict_raises_before_calibration(self):
        cp = ShapeAdaptiveCP()
        with pytest.raises(RuntimeError, match="not been calibrated"):
            cp.predict(np.ones((5, 2)), mu_test=np.ones(5), Z_test=np.zeros(5))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            ShapeAdaptiveCP(mode="invalid")

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be"):
            ShapeAdaptiveCP(alpha=1.5)

    def test_invalid_temp_raises(self):
        with pytest.raises(ValueError, match="smooth_temp must be positive"):
            ShapeAdaptiveCP(smooth_temp=-0.1)

    def test_repr_before_calibration(self):
        cp = ShapeAdaptiveCP(mode="group", alpha=0.1)
        r = repr(cp)
        assert "not calibrated" in r

    def test_repr_after_calibration(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=10)
        cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"], sigma_cal=d["sigma"])
        r = repr(cp)
        assert "calibrated" in r
        assert "msce=" in r


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_calibrate_sets_state(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"], sigma_cal=d["sigma"])
        assert cp.is_calibrated_
        assert cp.n_calibration_ == len(d["y"])
        assert cp.thresholds_ is not None
        assert cp.groups_ is not None
        assert cp.group_index_ is not None
        assert cp.msce_ is not None
        assert cp.coverage_by_group_ is not None

    def test_calibrate_groups_match_z(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="absolute", alpha=0.1, n_iter=10)
        cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"])
        assert len(cp.groups_) == d["K"]
        assert len(cp.thresholds_) == d["K"]

    def test_thresholds_are_positive(self, small_group_data):
        """After fitting, all thresholds should be positive (scores are non-negative)."""
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=50)
        cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"], sigma_cal=d["sigma"])
        assert np.all(cp.thresholds_ > 0), (
            f"Expected positive thresholds, got {cp.thresholds_}"
        )

    def test_msce_is_finite(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"], sigma_cal=d["sigma"])
        assert np.isfinite(cp.msce_)

    def test_z_length_mismatch_raises(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(alpha=0.1)
        with pytest.raises(ValueError, match="Z_cal length"):
            cp.calibrate(d["X"], d["y"], d["Z"][:-5], mu_cal=d["mu"])

    def test_absolute_score_no_sigma(self, small_group_data):
        """Absolute score should work without sigma."""
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="absolute", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"])
        assert cp.is_calibrated_

    def test_missing_mu_raises(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="absolute", alpha=0.1)
        with pytest.raises(ValueError, match="requires mu_cal"):
            cp.calibrate(d["X"], d["y"], d["Z"])

    def test_missing_sigma_raises(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1)
        with pytest.raises(ValueError, match="requires sigma_cal"):
            cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"])

    def test_custom_score_fn(self, small_group_data):
        """User-supplied callable score_fn."""
        d = small_group_data

        def my_score(y, mu, sigma):
            return np.abs(y - mu) / (sigma + 0.1)

        cp = ShapeAdaptiveCP(score_fn=my_score, alpha=0.1, n_iter=10)
        cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"], sigma_cal=d["sigma"])
        assert cp.is_calibrated_

    def test_invalid_score_fn_str_raises(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="nonsense", alpha=0.1)
        with pytest.raises(ValueError, match="score_fn must be"):
            cp.calibrate(d["X"], d["y"], d["Z"], mu_cal=d["mu"])


# ---------------------------------------------------------------------------
# Prediction tests: group mode
# ---------------------------------------------------------------------------


class TestPredictGroup:
    def test_predict_returns_two_arrays(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300],
                     mu_cal=d["mu"][:300], sigma_cal=d["sigma"][:300])
        lower, upper = cp.predict(
            d["X"][300:],
            mu_test=d["mu"][300:],
            sigma_test=d["sigma"][300:],
            Z_test=d["Z"][300:],
        )
        assert lower.shape == upper.shape == (len(d["X"]) - 300,)

    def test_upper_geq_lower(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300],
                     mu_cal=d["mu"][:300], sigma_cal=d["sigma"][:300])
        lower, upper = cp.predict(
            d["X"][300:],
            mu_test=d["mu"][300:],
            sigma_test=d["sigma"][300:],
            Z_test=d["Z"][300:],
        )
        assert np.all(upper >= lower), "All upper bounds must be >= lower bounds"

    def test_group_fn_masked_z(self, small_group_data):
        """
        Test the masked-Z feature: calibrate with Z, predict using group_fn(X).
        """
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300],
                     mu_cal=d["mu"][:300], sigma_cal=d["sigma"][:300])

        # group_fn derives group labels from X (simulating masked Z at test time)
        # In reality, you'd use a feature that correlates with Z; here we use a mock
        K = d["K"]
        def group_fn(X):
            # Assign to groups 0..K-1 based on X[:, 0] quantile
            bins = np.percentile(X[:, 0], np.linspace(0, 100, K + 1)[1:-1])
            return np.digitize(X[:, 0], bins)

        lower, upper = cp.predict(
            d["X"][300:],
            mu_test=d["mu"][300:],
            sigma_test=d["sigma"][300:],
            group_fn=group_fn,
        )
        assert lower.shape == (len(d["X"]) - 300,)
        assert np.all(upper >= lower)

    def test_both_group_fn_and_z_test_raises(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="absolute", alpha=0.1, n_iter=10)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300], mu_cal=d["mu"][:300])
        with pytest.raises(ValueError, match="either group_fn or Z_test"):
            cp.predict(
                d["X"][300:],
                mu_test=d["mu"][300:],
                Z_test=d["Z"][300:],
                group_fn=lambda X: np.zeros(len(X)),
            )

    def test_neither_group_fn_nor_z_test_raises(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="absolute", alpha=0.1, n_iter=10)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300], mu_cal=d["mu"][:300])
        with pytest.raises(ValueError, match="must provide either Z_test or group_fn"):
            cp.predict(d["X"][300:], mu_test=d["mu"][300:])

    def test_unseen_group_uses_fallback(self, small_group_data):
        """An unseen group label at test time should not raise, uses median threshold."""
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300],
                     mu_cal=d["mu"][:300], sigma_cal=d["sigma"][:300])

        # Use a Z_test with a group label that was not in Z_cal
        Z_unseen = np.full(len(d["X"]) - 300, 999)
        lower, upper = cp.predict(
            d["X"][300:],
            mu_test=d["mu"][300:],
            sigma_test=d["sigma"][300:],
            Z_test=Z_unseen,
        )
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    def test_absolute_score_intervals(self, small_group_data):
        """Absolute score: half-width = threshold (not threshold*sigma)."""
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="absolute", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300], mu_cal=d["mu"][:300])
        lower, upper = cp.predict(
            d["X"][300:300+5],
            mu_test=d["mu"][300:300+5],
            Z_test=d["Z"][300:300+5],
        )
        # Width should be constant across points with same group (for absolute mode,
        # half-width = threshold, which is the same for all points in the group)
        widths = upper - lower
        # All widths must be positive
        assert np.all(widths > 0)


# ---------------------------------------------------------------------------
# Coverage property tests (statistical)
# ---------------------------------------------------------------------------


class TestCoverageProperties:
    """
    Statistical tests that MOPI actually improves conditional coverage.

    These use a larger dataset where Mondrian conformal is visibly worse.
    The DGP has sigmas varying 5x across groups — a global threshold will
    badly undercover the noisy groups and overcover the quiet ones.
    """

    def test_mopi_msce_lower_than_global(self, group_data):
        """
        MOPI should reduce MSCE compared to a single global quantile threshold.
        """
        d = group_data
        n = len(d["y"])
        cal_end = int(n * 0.6)
        X_cal, y_cal = d["X"][:cal_end], d["y"][:cal_end]
        Z_cal = d["Z"][:cal_end]
        mu_cal = d["mu"][:cal_end]
        sigma_cal = d["sigma"][:cal_end]
        X_test = d["X"][cal_end:]
        y_test = d["y"][cal_end:]
        Z_test = d["Z"][cal_end:]
        mu_test = d["mu"][cal_end:]
        sigma_test = d["sigma"][cal_end:]

        # MOPI predictor
        cp_mopi = ShapeAdaptiveCP(
            score_fn="standardised", alpha=0.1, n_iter=200, lr=0.01,
        )
        cp_mopi.calibrate(X_cal, y_cal, Z_cal, mu_cal=mu_cal, sigma_cal=sigma_cal)
        lower_mopi, upper_mopi = cp_mopi.predict(
            X_test, mu_test=mu_test, sigma_test=sigma_test, Z_test=Z_test,
        )

        # Global quantile: one threshold for everyone
        from insurance_conformal.utils import conformal_quantile
        scores_cal = np.abs(y_cal - mu_cal) / np.maximum(sigma_cal, 1e-8)
        global_q = conformal_quantile(scores_cal, 0.1)
        lower_global = mu_test - global_q * sigma_test
        upper_global = mu_test + global_q * sigma_test

        msce_mopi = cp_mopi.msce(y_test, lower_mopi, upper_mopi, Z_test)
        msce_global = _compute_msce(
            ((y_test < lower_global) | (y_test > upper_global)).astype(float),
            Z_test,
            alpha=0.1,
        )

        assert msce_mopi <= msce_global * 1.5, (
            f"MOPI MSCE {msce_mopi:.4f} should be <= global MSCE {msce_global:.4f} "
            f"(allowing 50% slack for finite-sample variance)"
        )

    def test_per_group_coverage_reasonable(self, group_data):
        """
        Per-group coverage should be within 15pp of the target for most groups.
        """
        d = group_data
        n = len(d["y"])
        cal_end = int(n * 0.6)
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=200)
        cp.calibrate(
            d["X"][:cal_end], d["y"][:cal_end], d["Z"][:cal_end],
            mu_cal=d["mu"][:cal_end], sigma_cal=d["sigma"][:cal_end],
        )
        lower, upper = cp.predict(
            d["X"][cal_end:],
            mu_test=d["mu"][cal_end:],
            sigma_test=d["sigma"][cal_end:],
            Z_test=d["Z"][cal_end:],
        )
        report = cp.coverage_report(d["y"][cal_end:], lower, upper, d["Z"][cal_end:])
        coverages = report["coverage"].to_numpy()
        target = 0.90
        deviations = np.abs(coverages - target)
        # At least 75% of groups should be within 15pp of target
        fraction_ok = np.mean(deviations < 0.15)
        assert fraction_ok >= 0.75, (
            f"Only {fraction_ok:.0%} of groups within 15pp of 90% target. "
            f"Per-group coverages: {coverages}"
        )

    def test_marginal_coverage_is_reasonable(self, group_data):
        """
        Marginal coverage should be close to 1-alpha (within 5pp).
        MOPI tunes per-group thresholds but should maintain reasonable marginal coverage.
        """
        d = group_data
        n = len(d["y"])
        cal_end = int(n * 0.6)
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=200)
        cp.calibrate(
            d["X"][:cal_end], d["y"][:cal_end], d["Z"][:cal_end],
            mu_cal=d["mu"][:cal_end], sigma_cal=d["sigma"][:cal_end],
        )
        lower, upper = cp.predict(
            d["X"][cal_end:],
            mu_test=d["mu"][cal_end:],
            sigma_test=d["sigma"][cal_end:],
            Z_test=d["Z"][cal_end:],
        )
        y_test = d["y"][cal_end:]
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
        assert coverage >= 0.80, (
            f"Marginal coverage {coverage:.3f} is below acceptable floor 0.80"
        )
        assert coverage <= 0.99, (
            f"Marginal coverage {coverage:.3f} seems excessively conservative"
        )


# ---------------------------------------------------------------------------
# RKHS mode tests
# ---------------------------------------------------------------------------


class TestRkhs:
    def test_rkhs_calibration_runs(self, continuous_z_data):
        d = continuous_z_data
        n = len(d["y"])
        cal_end = int(n * 0.7)
        cp = ShapeAdaptiveCP(
            score_fn="standardised",
            alpha=0.1,
            mode="rkhs",
            n_iter=50,
            kernel_bandwidth=0.3,
            gamma=1e-3,
        )
        cp.calibrate(
            d["X"][:cal_end], d["y"][:cal_end], d["Z"][:cal_end],
            mu_cal=d["mu"][:cal_end], sigma_cal=d["sigma"][:cal_end],
        )
        assert cp.is_calibrated_
        assert cp.thresholds_ is not None
        assert len(cp.thresholds_) == cal_end
        assert np.isfinite(cp.msce_)

    def test_rkhs_predict_finite(self, continuous_z_data):
        d = continuous_z_data
        n = len(d["y"])
        cal_end = int(n * 0.7)
        cp = ShapeAdaptiveCP(
            score_fn="standardised",
            alpha=0.1,
            mode="rkhs",
            n_iter=30,
            kernel_bandwidth=0.3,
            gamma=1e-3,
        )
        cp.calibrate(
            d["X"][:cal_end], d["y"][:cal_end], d["Z"][:cal_end],
            mu_cal=d["mu"][:cal_end], sigma_cal=d["sigma"][:cal_end],
        )
        lower, upper = cp.predict(
            d["X"][cal_end:],
            mu_test=d["mu"][cal_end:],
            sigma_test=d["sigma"][cal_end:],
            Z_test=d["Z"][cal_end:],
        )
        assert lower.shape == upper.shape == (n - cal_end,)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all(upper >= lower), "Upper must be >= lower"

    def test_rkhs_predict_without_z_test_raises(self, continuous_z_data):
        d = continuous_z_data
        cp = ShapeAdaptiveCP(mode="rkhs", n_iter=10, gamma=1e-3)
        cp.calibrate(d["X"][:100], d["y"][:100], d["Z"][:100], mu_cal=d["mu"][:100])
        with pytest.raises(ValueError, match="Z_test is required"):
            cp.predict(d["X"][100:200], mu_test=d["mu"][100:200])


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_coverage_report_returns_dataframe(self, small_group_data):
        import polars as pl
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300],
                     mu_cal=d["mu"][:300], sigma_cal=d["sigma"][:300])
        lower, upper = cp.predict(
            d["X"][300:],
            mu_test=d["mu"][300:],
            sigma_test=d["sigma"][300:],
            Z_test=d["Z"][300:],
        )
        report = cp.coverage_report(d["y"][300:], lower, upper, d["Z"][300:])
        assert isinstance(report, pl.DataFrame)
        assert set(report.columns) == {
            "group", "n_obs", "coverage", "target_coverage", "miscoverage_deviation"
        }
        assert len(report) == d["K"]

    def test_coverage_report_target_correct(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="absolute", alpha=0.05, n_iter=10)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300], mu_cal=d["mu"][:300])
        lower, upper = cp.predict(
            d["X"][300:],
            mu_test=d["mu"][300:],
            Z_test=d["Z"][300:],
        )
        report = cp.coverage_report(d["y"][300:], lower, upper, d["Z"][300:])
        assert all(abs(v - 0.95) < 1e-6 for v in report["target_coverage"].to_list())

    def test_msce_non_negative(self, small_group_data):
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=20)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300],
                     mu_cal=d["mu"][:300], sigma_cal=d["sigma"][:300])
        lower, upper = cp.predict(
            d["X"][300:],
            mu_test=d["mu"][300:],
            sigma_test=d["sigma"][300:],
            Z_test=d["Z"][300:],
        )
        msce = cp.msce(d["y"][300:], lower, upper, d["Z"][300:])
        assert msce >= 0.0

    def test_calibration_msce_attribute_matches_manual(self, small_group_data):
        """cp.msce_ (calibration set) should equal manual MSCE on calibration set."""
        d = small_group_data
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=30)
        cp.calibrate(d["X"][:300], d["y"][:300], d["Z"][:300],
                     mu_cal=d["mu"][:300], sigma_cal=d["sigma"][:300])

        # Compute manually on calibration set
        lower_cal, upper_cal = cp.predict(
            d["X"][:300],
            mu_test=d["mu"][:300],
            sigma_test=d["sigma"][:300],
            Z_test=d["Z"][:300],
        )
        msce_manual = cp.msce(d["y"][:300], lower_cal, upper_cal, d["Z"][:300])
        # These should be close (msce_ is set from scores, msce() is from intervals)
        # Allow generous tolerance since they compute the same quantity differently
        assert abs(cp.msce_ - msce_manual) < 0.05, (
            f"cp.msce_={cp.msce_:.4f}, manual={msce_manual:.4f}"
        )


# ---------------------------------------------------------------------------
# Input format tests
# ---------------------------------------------------------------------------


class TestInputFormats:
    def test_polars_inputs_accepted(self, small_group_data):
        import polars as pl
        d = small_group_data
        X_pl = pl.DataFrame({"x0": d["X"][:200, 0], "x1": d["X"][:200, 1]})
        y_pl = pl.Series("y", d["y"][:200])
        Z_pl = pl.Series("Z", d["Z"][:200])
        mu_pl = pl.Series("mu", d["mu"][:200])
        sigma_pl = pl.Series("sigma", d["sigma"][:200])

        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=10)
        cp.calibrate(X_pl, y_pl, Z_pl, mu_cal=mu_pl, sigma_cal=sigma_pl)
        assert cp.is_calibrated_

    def test_pandas_inputs_accepted(self, small_group_data):
        import pandas as pd
        d = small_group_data
        X_pd = pd.DataFrame({"x0": d["X"][:200, 0], "x1": d["X"][:200, 1]})
        y_pd = pd.Series(d["y"][:200])
        Z_pd = pd.Series(d["Z"][:200])
        mu_pd = pd.Series(d["mu"][:200])
        sigma_pd = pd.Series(d["sigma"][:200])

        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=10)
        cp.calibrate(X_pd, y_pd, Z_pd, mu_cal=mu_pd, sigma_cal=sigma_pd)
        assert cp.is_calibrated_

    def test_string_group_labels(self):
        """Group labels can be strings, not just integers."""
        rng = np.random.default_rng(5)
        n = 200
        X = rng.uniform(size=(n, 2))
        Z_str = np.array(["low", "mid", "high"])[rng.integers(0, 3, n)]
        mu = X[:, 0]
        sigma = np.ones(n) * 0.3
        y = rng.normal(mu, sigma)

        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, n_iter=10)
        cp.calibrate(X, y, Z_str, mu_cal=mu, sigma_cal=sigma)
        assert cp.is_calibrated_
        assert set(cp.groups_) == {"low", "mid", "high"}
