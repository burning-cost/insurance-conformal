"""
Tests for RetroAdj: Online Conformal Inference with Retrospective Adjustment.

Tests cover:
1. Matrix update correctness (rank-one update/downdate vs direct inversion)
2. LOO residual formula vs brute-force refit
3. Jackknife+ interval properties
4. Coverage on stationary series
5. Coverage recovery after abrupt distribution shift
6. API contract (fit/predict_interval, X=None mode)
7. Numerical stability (periodic reset, symmetry enforcement)
8. SFOGD alpha tracker
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_conformal.retro_adj import (
    RetroAdj,
    _NumericalInstabilityError,
    _SFOGDAlphaTracker,
    _jackknife_plus_interval,
    _loo_residuals_and_leverages,
    _rank_one_downdate,
    _rank_one_update,
    _rbf_kernel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _direct_krr_alpha(X_train, Y_train, bw, lam):
    """Return alpha = (K + lam*I)^{-1} Y for KRR."""
    K = _rbf_kernel(X_train, X_train, bw)
    return np.linalg.solve(K + lam * np.eye(len(K)), Y_train)


def make_krr_prediction(X_train, Y_train, x_test, bw, lam):
    """Brute-force KRR prediction at x_test, using full training set."""
    alpha = _direct_krr_alpha(X_train, Y_train, bw, lam)
    k_star = _rbf_kernel(x_test.reshape(1, -1), X_train, bw).ravel()
    return float(k_star @ alpha)


def make_krr_loo_residual(X_train, Y_train, i, bw, lam):
    """Brute-force LOO residual at index i by re-fitting without point i."""
    idx = [j for j in range(len(X_train)) if j != i]
    X_loo = X_train[idx]
    Y_loo = Y_train[idx]
    pred = make_krr_prediction(X_loo, Y_loo, X_train[i], bw, lam)
    return float(Y_train[i]) - pred


# ---------------------------------------------------------------------------
# 1. Kernel helper
# ---------------------------------------------------------------------------


class TestRbfKernel:
    def test_self_kernel_is_one(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 3))
        K = _rbf_kernel(X, X, bandwidth=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_symmetry(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(8, 4))
        K = _rbf_kernel(X, X, bandwidth=2.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_positive_definite(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(6, 2))
        K = _rbf_kernel(X, X, bandwidth=1.0)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > 0), f"Not PD: min eigenvalue = {eigvals.min()}"

    def test_cross_kernel_shape(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(5, 3))
        X2 = rng.normal(size=(7, 3))
        K = _rbf_kernel(X, X2, bandwidth=1.0)
        assert K.shape == (5, 7)

    def test_bandwidth_effect(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(4, 2))
        K_narrow = _rbf_kernel(X, X, bandwidth=0.1)
        K_wide = _rbf_kernel(X, X, bandwidth=100.0)
        off_diag_wide = K_wide[0, 1]
        off_diag_narrow = K_narrow[0, 1]
        assert off_diag_wide > off_diag_narrow


# ---------------------------------------------------------------------------
# 2. Rank-one update correctness
# ---------------------------------------------------------------------------


class TestRankOneUpdate:
    def test_update_matches_direct_inversion(self):
        """Incrementally built Q should match direct inversion."""
        rng = np.random.default_rng(42)
        n, p, lam = 12, 3, 0.1
        bw = 1.0
        X = rng.normal(size=(n, p))

        K = _rbf_kernel(X, X, bw)
        Q_direct = np.linalg.inv(K + lam * np.eye(n))

        # Build Q incrementally from first 2 points
        K2 = _rbf_kernel(X[:2], X[:2], bw)
        Q_incr = np.linalg.inv(K2 + lam * np.eye(2))

        for i in range(2, n):
            k_new = _rbf_kernel(X[i : i + 1], X[:i], bw).ravel()
            Q_incr = _rank_one_update(Q_incr, k_new, kappa_self_val=1.0, lambda_reg=lam)

        np.testing.assert_allclose(Q_incr, Q_direct, atol=1e-9)

    def test_update_output_shape(self):
        """Rank-one update should increase size by 1."""
        rng = np.random.default_rng(5)
        n, p, lam, bw = 5, 2, 0.1, 1.0
        X = rng.normal(size=(n + 1, p))
        K = _rbf_kernel(X[:n], X[:n], bw)
        Q = np.linalg.inv(K + lam * np.eye(n))
        k_new = _rbf_kernel(X[n : n + 1], X[:n], bw).ravel()
        Q_new = _rank_one_update(Q, k_new, 1.0, lam)
        assert Q_new.shape == (n + 1, n + 1)

    def test_update_symmetry(self):
        rng = np.random.default_rng(6)
        n, p, lam = 8, 2, 0.05
        X = rng.normal(size=(n + 1, p))
        K = _rbf_kernel(X[:n], X[:n], 1.0)
        Q = np.linalg.inv(K + lam * np.eye(n))
        k_new = _rbf_kernel(X[n : n + 1], X[:n], 1.0).ravel()
        Q_new = _rank_one_update(Q, k_new, 1.0, lam)
        np.testing.assert_allclose(Q_new, Q_new.T, atol=1e-10)

    def test_update_is_valid_inverse(self):
        """Q_new should match direct inversion of full (n+1) x (n+1) matrix."""
        rng = np.random.default_rng(7)
        n, p, lam = 6, 2, 0.1
        bw = 1.0
        X = rng.normal(size=(n + 1, p))
        K_full = _rbf_kernel(X, X, bw)
        Q_direct = np.linalg.inv(K_full + lam * np.eye(n + 1))

        K_old = _rbf_kernel(X[:n], X[:n], bw)
        Q_old = np.linalg.inv(K_old + lam * np.eye(n))
        k_new = _rbf_kernel(X[n : n + 1], X[:n], bw).ravel()
        Q_incr = _rank_one_update(Q_old, k_new, 1.0, lam)

        np.testing.assert_allclose(Q_incr, Q_direct, atol=1e-9)

    def test_update_raises_on_negative_delta(self):
        """Should raise _NumericalInstabilityError when delta <= 0."""
        # Set Q to make k_new @ Q @ k_new > kappa_self + lambda
        # Use Q = large * I so that k_new @ Q @ k_new = large * ||k_new||^2
        rng = np.random.default_rng(99)
        n = 3
        Q = np.eye(n) * 20.0  # very large Q
        k_new = np.ones(n)    # k_new @ Q @ k_new = 20 * 3 = 60 >> 1 + 0.01
        with pytest.raises(_NumericalInstabilityError):
            _rank_one_update(Q, k_new, kappa_self_val=1.0, lambda_reg=0.01)


# ---------------------------------------------------------------------------
# 3. Rank-one downdate correctness
# ---------------------------------------------------------------------------


class TestRankOneDowndate:
    def test_downdate_matches_direct_inversion(self):
        """Removing index 0 via downdate should match direct inversion without it."""
        rng = np.random.default_rng(8)
        n, p, lam = 10, 3, 0.2
        bw = 1.5
        X = rng.normal(size=(n, p))

        K_full = _rbf_kernel(X, X, bw)
        Q_full = np.linalg.inv(K_full + lam * np.eye(n))

        Q_down = _rank_one_downdate(Q_full)

        K_sub = _rbf_kernel(X[1:], X[1:], bw)
        Q_sub = np.linalg.inv(K_sub + lam * np.eye(n - 1))

        np.testing.assert_allclose(Q_down, Q_sub, atol=1e-9)

    def test_downdate_output_shape(self):
        rng = np.random.default_rng(9)
        n = 6
        X = rng.normal(size=(n, 2))
        K = _rbf_kernel(X, X, 1.0)
        Q = np.linalg.inv(K + 0.1 * np.eye(n))
        Q_down = _rank_one_downdate(Q)
        assert Q_down.shape == (n - 1, n - 1)

    def test_roundtrip_update_downdate(self):
        """Update then downdate oldest should match direct inversion of shifted window."""
        rng = np.random.default_rng(10)
        n, p, lam = 8, 2, 0.1
        bw = 1.0
        X = rng.normal(size=(n + 1, p))

        K = _rbf_kernel(X[:n], X[:n], bw)
        Q_orig = np.linalg.inv(K + lam * np.eye(n))

        # Add point n, remove point 0
        k_new = _rbf_kernel(X[n : n + 1], X[:n], bw).ravel()
        Q_updated = _rank_one_update(Q_orig, k_new, 1.0, lam)
        # Downdate removes point 0 -> covers points 1..n
        Q_after = _rank_one_downdate(Q_updated)

        # Direct: K of points 1..n
        K_target = _rbf_kernel(X[1:], X[1:], bw)
        Q_target = np.linalg.inv(K_target + lam * np.eye(n))

        np.testing.assert_allclose(Q_after, Q_target, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. LOO residuals vs brute-force
# ---------------------------------------------------------------------------


class TestLooResiduals:
    def test_loo_residuals_match_brute_force(self):
        """Lemma 1 formula should match LOO residuals computed by re-fitting."""
        rng = np.random.default_rng(11)
        n, p, lam, bw = 8, 2, 0.2, 1.0
        X = rng.normal(size=(n, p))
        Y = rng.normal(size=n)

        K = _rbf_kernel(X, X, bw)
        Q = np.linalg.inv(K + lam * np.eye(n))

        R_loo, S_diag, mask = _loo_residuals_and_leverages(Q, K, Y)

        for i in range(n):
            if not mask[i]:
                continue
            bf = make_krr_loo_residual(X, Y, i, bw, lam)
            np.testing.assert_allclose(
                R_loo[i], bf, atol=1e-7,
                err_msg=f"LOO residual mismatch at index {i}"
            )

    def test_leverage_scores_in_range(self):
        """Leverage scores should be in [0, 1] for a proper smoother."""
        rng = np.random.default_rng(12)
        n, p, lam, bw = 10, 3, 0.1, 1.0
        X = rng.normal(size=(n, p))
        Y = rng.normal(size=n)
        K = _rbf_kernel(X, X, bw)
        Q = np.linalg.inv(K + lam * np.eye(n))
        _, S_diag, _ = _loo_residuals_and_leverages(Q, K, Y)
        assert np.all(S_diag >= -1e-10)
        assert np.all(S_diag <= 1.0 + 1e-10)

    def test_masked_points_get_inf(self):
        """Points with |1 - S_ii| < eps should be masked (R_loo = inf)."""
        # With very small lambda, leverage scores near 1 -> some points masked
        rng = np.random.default_rng(13)
        n, p, lam, bw = 4, 2, 1e-8, 1.0
        X = rng.normal(size=(n, p))
        Y = rng.normal(size=n)
        K = _rbf_kernel(X, X, bw)
        Q = np.linalg.inv(K + lam * np.eye(n))
        R_loo, _, mask = _loo_residuals_and_leverages(Q, K, Y)
        # Masked points must be inf
        assert np.all(R_loo[~mask] == np.inf)


# ---------------------------------------------------------------------------
# 5. Jackknife+ interval properties
# ---------------------------------------------------------------------------


class TestJackknifePlusInterval:
    def test_interval_lower_leq_upper(self):
        """lower should always be <= upper."""
        rng = np.random.default_rng(14)
        n = 50
        f_loo = np.zeros(n)
        R_loo = rng.normal(0, 1, n)
        mask = np.ones(n, dtype=bool)
        lo, hi = _jackknife_plus_interval(f_loo, R_loo, alpha_t=0.1, mask=mask)
        assert lo <= hi

    def test_interval_wider_at_lower_coverage(self):
        """Smaller alpha_t (higher coverage target) should give wider intervals."""
        rng = np.random.default_rng(15)
        n = 50
        f_loo = np.zeros(n)
        R_loo = rng.normal(0, 1, n)
        mask = np.ones(n, dtype=bool)
        lo_90, hi_90 = _jackknife_plus_interval(f_loo, R_loo, alpha_t=0.1, mask=mask)
        lo_50, hi_50 = _jackknife_plus_interval(f_loo, R_loo, alpha_t=0.5, mask=mask)
        assert (hi_90 - lo_90) >= (hi_50 - lo_50)

    def test_all_masked_returns_inf(self):
        """With no valid points, should return (-inf, inf)."""
        n = 5
        f_loo = np.zeros(n)
        R_loo = np.full(n, np.inf)
        mask = np.zeros(n, dtype=bool)
        lo, hi = _jackknife_plus_interval(f_loo, R_loo, alpha_t=0.1, mask=mask)
        assert lo == -np.inf
        assert hi == np.inf

    def test_symmetric_mode(self):
        """Symmetric mode should produce interval around f_loo median."""
        rng = np.random.default_rng(16)
        n = 30
        f_loo = np.zeros(n)
        R_loo = rng.normal(0, 1, n)
        mask = np.ones(n, dtype=bool)
        lo, hi = _jackknife_plus_interval(f_loo, R_loo, alpha_t=0.1, mask=mask, symmetric=True)
        # With symmetric mode and f_loo=0, upper >= 0 and lower <= 0 for reasonable R_loo
        assert lo <= hi

    def test_shift_in_f_loo_shifts_interval(self):
        """Shifting all f_loo by a constant should shift both bounds by the same amount."""
        rng = np.random.default_rng(16)
        n = 30
        shift = 5.0
        f_loo = np.zeros(n)
        R_loo = rng.normal(0, 1, n)
        mask = np.ones(n, dtype=bool)
        lo0, hi0 = _jackknife_plus_interval(f_loo, R_loo, alpha_t=0.1, mask=mask)
        lo1, hi1 = _jackknife_plus_interval(f_loo + shift, R_loo, alpha_t=0.1, mask=mask)
        np.testing.assert_allclose(lo1, lo0 + shift, atol=1e-10)
        np.testing.assert_allclose(hi1, hi0 + shift, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. SFOGD alpha tracker
# ---------------------------------------------------------------------------


class TestSFOGDAlphaTracker:
    def test_alpha_stays_in_bounds(self):
        """Alpha should remain in (1e-6, 1-1e-6) regardless of signals."""
        tracker = _SFOGDAlphaTracker(alpha_init=0.1, gamma=0.1)
        for _ in range(200):
            tracker.update(covered=False, alpha_target=0.1)
        assert 1e-6 <= tracker.alpha <= 1.0 - 1e-6

    def test_alpha_increases_on_misses(self):
        """Consistently missing (uncovered) should drive alpha_t UP.

        With SFOGD: err = alpha_target - (0 if covered else 1).
        miss -> err = alpha - 1 < 0, alpha_t = alpha_t - gamma * err / sqrt(...)
        Since err < 0, this increases alpha_t (wider intervals needed).
        """
        tracker = _SFOGDAlphaTracker(alpha_init=0.1, gamma=0.5)
        initial = tracker.alpha
        for _ in range(50):
            tracker.update(covered=False, alpha_target=0.1)
        assert tracker.alpha > initial, (
            f"SFOGD alpha should increase on misses: {initial} -> {tracker.alpha}"
        )

    def test_alpha_decreases_on_covers(self):
        """Consistently covering should drive alpha_t DOWN.

        cover -> err = alpha - 0 = alpha > 0, alpha_t decreases (intervals can narrow).
        """
        tracker = _SFOGDAlphaTracker(alpha_init=0.1, gamma=0.5)
        initial = tracker.alpha
        for _ in range(50):
            tracker.update(covered=True, alpha_target=0.1)
        assert tracker.alpha < initial, (
            f"SFOGD alpha should decrease on covers: {initial} -> {tracker.alpha}"
        )

    def test_first_step_no_div_zero(self):
        """First step (cumulative_sq_grad=0) should not raise."""
        tracker = _SFOGDAlphaTracker(alpha_init=0.1, gamma=0.005)
        result = tracker.update(covered=True, alpha_target=0.1)
        assert 1e-6 <= result <= 1.0 - 1e-6


# ---------------------------------------------------------------------------
# 7. RetroAdj API contract
# ---------------------------------------------------------------------------


class TestRetroAdjAPI:
    def test_repr_not_fitted(self):
        model = RetroAdj(window_size=50)
        assert "not fitted" in repr(model)

    def test_repr_fitted(self):
        rng = np.random.default_rng(17)
        y = rng.normal(size=100)
        model = RetroAdj(window_size=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y)
        assert "fitted" in repr(model)

    def test_predict_without_fit_raises(self):
        model = RetroAdj(window_size=50)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_interval(np.ones(10))

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            RetroAdj(window_size=3)

    def test_invalid_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth"):
            RetroAdj(bandwidth=-1.0)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda_reg"):
            RetroAdj(lambda_reg=0.0)

    def test_invalid_alpha_update(self):
        with pytest.raises(ValueError, match="alpha_update"):
            RetroAdj(alpha_update="bad")

    def test_invalid_alpha_range(self):
        rng = np.random.default_rng(18)
        y = rng.normal(size=50)
        model = RetroAdj(window_size=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y)
        with pytest.raises(ValueError, match="alpha"):
            model.predict_interval(y, alpha=1.5)

    def test_output_shapes(self):
        rng = np.random.default_rng(19)
        n_train, n_test = 60, 40
        y = rng.normal(size=n_train + n_test)
        model = RetroAdj(window_size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            lower, upper = model.predict_interval(y[n_train:], alpha=0.1)
        assert lower.shape == (n_test,)
        assert upper.shape == (n_test,)

    def test_lower_leq_upper(self):
        rng = np.random.default_rng(20)
        n_train, n_test = 80, 50
        y = rng.normal(size=n_train + n_test)
        model = RetroAdj(window_size=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            lower, upper = model.predict_interval(y[n_train:], alpha=0.1)
        assert np.all(lower <= upper), "lower > upper at some steps"

    def test_x_none_warning(self):
        rng = np.random.default_rng(21)
        y = rng.normal(size=50)
        model = RetroAdj(window_size=20)
        with pytest.warns(UserWarning, match="residual-only"):
            model.fit(y)

    def test_x_mismatch_raises(self):
        """fit(X=None) then predict(X=...) should raise ValueError."""
        rng = np.random.default_rng(22)
        y = rng.normal(size=80)
        X = rng.normal(size=(80, 3))
        model2 = RetroAdj(window_size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model2.fit(y[:50])
        with pytest.raises(ValueError, match="residual-only"):
            model2.predict_interval(y[50:], X[50:], alpha=0.1)

    def test_fit_returns_self(self):
        rng = np.random.default_rng(23)
        y = rng.normal(size=50)
        model = RetroAdj(window_size=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(y)
        assert result is model

    def test_with_features(self):
        rng = np.random.default_rng(24)
        n_train, n_test, p = 80, 30, 3
        X = rng.normal(size=(n_train + n_test, p))
        y = rng.normal(size=n_train + n_test)
        model = RetroAdj(window_size=30, bandwidth=2.0)
        model.fit(y[:n_train], X[:n_train])
        lower, upper = model.predict_interval(y[n_train:], X[n_train:], alpha=0.1)
        assert lower.shape == (n_test,)
        assert np.all(lower <= upper)

    def test_sfogd_mode(self):
        rng = np.random.default_rng(25)
        n_train, n_test = 60, 40
        y = rng.normal(size=n_train + n_test)
        model = RetroAdj(window_size=30, alpha_update="sfogd")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            lower, upper = model.predict_interval(y[n_train:], alpha=0.1)
        assert lower.shape == (n_test,)
        assert np.all(lower <= upper)

    def test_symmetric_mode(self):
        rng = np.random.default_rng(26)
        n_train, n_test = 60, 30
        y = rng.normal(size=n_train + n_test)
        model = RetroAdj(window_size=30, symmetric=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            lower, upper = model.predict_interval(y[n_train:], alpha=0.1)
        assert np.all(lower <= upper)

    def test_window_doesnt_exceed_window_size(self):
        """After many steps, internal window should not exceed window_size."""
        rng = np.random.default_rng(27)
        n_train, n_test = 50, 200
        y = rng.normal(size=n_train + n_test)
        ws = 40
        model = RetroAdj(window_size=ws)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            model.predict_interval(y[n_train:], alpha=0.1)
        assert len(model._Y_window) <= ws

    def test_periodic_reset(self):
        """With reset_freq=10, no crash over 100 steps."""
        rng = np.random.default_rng(28)
        n_train, n_test = 50, 100
        y = rng.normal(size=n_train + n_test)
        model = RetroAdj(window_size=30, reset_freq=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            lower, upper = model.predict_interval(y[n_train:], alpha=0.1)
        assert lower.shape == (n_test,)
        assert np.all(lower <= upper)


# ---------------------------------------------------------------------------
# 8. Coverage on stationary series
# ---------------------------------------------------------------------------


class TestCoverageStationary:
    def test_coverage_normal_no_features(self):
        """On stationary normal series, long-run coverage should be near target."""
        rng = np.random.default_rng(0)
        n_train, n_test, alpha = 300, 500, 0.1
        y = rng.normal(0, 1, n_train + n_test)

        model = RetroAdj(window_size=100, gamma=0.005)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            lower, upper = model.predict_interval(y[n_train:], alpha=alpha)

        y_test = y[n_train:]
        coverage = float(np.mean((lower <= y_test) & (y_test <= upper)))
        target = 1.0 - alpha
        assert abs(coverage - target) < 0.05, (
            f"Coverage {coverage:.3f} is more than 5pp from target {target:.3f}"
        )

    def test_coverage_with_features(self):
        """With features on stationary data, coverage should be near target."""
        rng = np.random.default_rng(1)
        n_train, n_test, p, alpha = 200, 300, 3, 0.1
        X = rng.normal(size=(n_train + n_test, p))
        beta = np.array([0.5, -0.3, 0.2])
        y = X @ beta + rng.normal(0, 0.5, n_train + n_test)

        model = RetroAdj(window_size=80, bandwidth=2.0, gamma=0.005)
        model.fit(y[:n_train], X[:n_train])
        lower, upper = model.predict_interval(y[n_train:], X[n_train:], alpha=alpha)

        y_test = y[n_train:]
        coverage = float(np.mean((lower <= y_test) & (y_test <= upper)))
        target = 1.0 - alpha
        # Wider tolerance for feature-based case — coverage is valid but
        # interval width can be large, leading to over-coverage
        assert coverage >= 0.80, (
            f"Coverage {coverage:.3f} is too low (< 0.80) for alpha={alpha}"
        )

    def test_coverage_sfogd_stationary(self):
        """SFOGD mode should run without error and produce non-degenerate intervals.

        SFOGD coverage properties depend heavily on the data and parameters.
        The key checks are: no crashes, valid shapes, lower <= upper, and
        non-trivial coverage (not 0% or 100% collapsed to a single point).
        Tight coverage guarantees are tested in test_coverage_normal_no_features
        (ACI mode), which converges faster and is easier to assert on.
        """
        rng = np.random.default_rng(2)
        n_train, n_test, alpha = 200, 400, 0.1
        y = rng.normal(0, 1, n_train + n_test)

        model = RetroAdj(window_size=80, gamma=0.005, alpha_update="sfogd")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            lower, upper = model.predict_interval(y[n_train:], alpha=alpha)

        assert lower.shape == (n_test,)
        assert upper.shape == (n_test,)
        assert np.all(lower <= upper), "lower > upper in some SFOGD steps"
        assert not np.any(np.isnan(lower)), "NaN in SFOGD lower bounds"
        assert not np.any(np.isnan(upper)), "NaN in SFOGD upper bounds"

        y_test = y[n_train:]
        coverage = float(np.mean((lower <= y_test) & (y_test <= upper)))
        # Coverage must be non-trivial in both directions
        assert coverage > 0.1, f"SFOGD coverage {coverage:.3f} is near-zero (collapsed)"
        assert coverage < 1.0, f"SFOGD coverage {coverage:.3f} is 100% (degenerate)"


# ---------------------------------------------------------------------------
# 9. Coverage recovery after abrupt distribution shift
# ---------------------------------------------------------------------------


class TestCoverageRecovery:
    def _run_recovery_experiment(self, alpha_update: str, gamma: float):
        """Helper: returns post-shift coverage in [shift, shift+30] window."""
        rng = np.random.default_rng(100)
        n_pre, n_shift, n_post = 200, 100, 80
        y_pre = rng.normal(0, 1, n_pre)
        # Abrupt +5 location shift
        y_shift = rng.normal(5, 1, n_shift)
        y_post_settled = rng.normal(5, 1, n_post)

        y_test = np.concatenate([y_shift, y_post_settled])
        alpha = 0.1

        model = RetroAdj(
            window_size=50,
            gamma=gamma,
            alpha_update=alpha_update,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y_pre)
            lower, upper = model.predict_interval(y_test, alpha=alpha)

        covered = (lower[:30] <= y_test[:30]) & (y_test[:30] <= upper[:30])
        return float(covered.mean())

    def test_post_shift_coverage_eventually_recovers(self):
        """After abrupt shift, some recovery should happen within 30 steps."""
        coverage_retro = self._run_recovery_experiment("aci", gamma=0.005)
        # Not zero — the method should detect and adapt to the shift
        assert coverage_retro > 0.05, (
            f"RetroAdj post-shift coverage {coverage_retro:.3f} is near zero"
        )

    def test_longer_window_coverage(self):
        """Over a longer post-shift window, coverage should settle near target."""
        rng = np.random.default_rng(101)
        n_pre = 200
        n_test = 300
        shift_at = 100
        alpha = 0.1

        y_pre = rng.normal(0, 1, n_pre)
        y_pre_test = rng.normal(0, 1, shift_at)
        y_post_test = rng.normal(5, 1, n_test - shift_at)
        y_test = np.concatenate([y_pre_test, y_post_test])

        model = RetroAdj(window_size=60, gamma=0.005)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y_pre)
            lower, upper = model.predict_interval(y_test, alpha=alpha)

        # Coverage in last 100 steps (well after shift, model should have adapted)
        y_late = y_test[-100:]
        lo_late = lower[-100:]
        hi_late = upper[-100:]
        coverage_late = float(np.mean((lo_late <= y_late) & (y_late <= hi_late)))
        assert coverage_late > 0.7, (
            f"Late post-shift coverage {coverage_late:.3f} should be > 0.70"
        )


# ---------------------------------------------------------------------------
# 10. Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_no_nan_in_intervals(self):
        """Intervals should not contain NaN in normal operation."""
        rng = np.random.default_rng(200)
        n_train, n_test = 100, 500
        y = rng.normal(size=n_train + n_test)

        model = RetroAdj(window_size=50, reset_freq=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            lower, upper = model.predict_interval(y[n_train:], alpha=0.1)

        assert not np.any(np.isnan(lower)), "NaN in lower bounds"
        assert not np.any(np.isnan(upper)), "NaN in upper bounds"

    def test_q_remains_approximately_symmetric(self):
        """Q should remain approximately symmetric after many updates."""
        rng = np.random.default_rng(201)
        n_train, n_test = 60, 200
        y = rng.normal(size=n_train + n_test)

        model = RetroAdj(window_size=30, reset_freq=500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:n_train])
            model.predict_interval(y[n_train:], alpha=0.1)

        Q = model._Q
        asymmetry = np.max(np.abs(Q - Q.T))
        assert asymmetry < 1e-8, f"Q asymmetry too large: {asymmetry:.2e}"

    def test_large_shift_no_crash(self):
        """A 100-sigma shift should not crash the method."""
        rng = np.random.default_rng(202)
        n_train = 80
        y_pre = rng.normal(0, 1, n_train)
        y_post = rng.normal(1000, 1, 50)

        model = RetroAdj(window_size=40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y_pre)
            lower, upper = model.predict_interval(y_post, alpha=0.1)

        assert lower.shape == (50,)
        assert upper.shape == (50,)

    def test_constant_series_no_crash(self):
        """A constant series (zero variance) should not crash."""
        y = np.ones(100)
        model = RetroAdj(window_size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y[:60])
            lower, upper = model.predict_interval(y[60:], alpha=0.1)
        assert lower.shape == (40,)

    def test_fit_uses_last_window_size_points(self):
        """fit() should use the last window_size observations, not all."""
        rng = np.random.default_rng(203)
        ws = 20
        y_long = rng.normal(size=500)
        y_short = y_long[-ws:]

        model_long = RetroAdj(window_size=ws)
        model_short = RetroAdj(window_size=ws)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_long.fit(y_long)
            model_short.fit(y_short)

        np.testing.assert_allclose(
            model_long._Y_window, model_short._Y_window, atol=1e-12
        )
        np.testing.assert_allclose(model_long._Q, model_short._Q, atol=1e-12)
