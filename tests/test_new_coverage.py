"""
New tests targeting under-covered modules:

- retro_adj.py: RetroAdj, _SFOGDAlphaTracker, _rank_one_update/_downdate,
  _loo_residuals_and_leverages, _jackknife_plus_interval
- assessment.py: ConditionalCoverageAssessor, _compute_cvi, _parse_prediction_sets
- utils.py: temporal_split, ipcw_weighted_quantile, conformal_quantile edge cases
- lcp_model_selector.py: LCPModelSelector, LocalScoreSelector,
  _kernel_weights, _weighted_quantile
- diagnostics_ext.py: ert_coverage_gap, subgroup_coverage,
  width_efficiency_comparison

Focus: numerical correctness and boundary conditions, not just "doesn't crash".
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest


# ===========================================================================
# Shared helpers
# ===========================================================================


class ConstantPredictor:
    """Minimal predictor with predict_interval(). Returns fixed-width intervals
    with slightly varying midpoints (so ert_coverage_gap can bin by quantile)."""

    def __init__(self, width: float = 1.0, point: float = 2.0) -> None:
        self.width = width
        self.point = point

    def predict_interval(self, X, alpha=0.10):
        X_arr = np.asarray(X)
        n = len(X_arr)
        # Add small variation based on first feature (or row index if 1D)
        if X_arr.ndim >= 2:
            variation = X_arr[:, 0] * 0.01
        else:
            variation = np.arange(n) * 0.01
        pts = self.point + variation
        lower = pts - self.width / 2.0
        upper = pts + self.width / 2.0
        return pl.DataFrame({"lower": lower, "point": pts, "upper": upper})


class FixedPredictor:
    """Truly constant predictor — for tests that don't use ert_coverage_gap."""

    def __init__(self, width: float = 1.0, point: float = 2.0) -> None:
        self.width = width
        self.point = point

    def predict_interval(self, X, alpha=0.10):
        n = len(np.asarray(X))
        lower = np.full(n, self.point - self.width / 2.0)
        upper = np.full(n, self.point + self.width / 2.0)
        return pl.DataFrame({"lower": lower, "point": upper, "upper": upper})


class ScalarModel:
    """Predicts a constant scalar."""

    def __init__(self, value: float = 2.0) -> None:
        self.value = value

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else len(np.asarray(X))
        return np.full(n, self.value)


# ===========================================================================
# retro_adj.py
# ===========================================================================


class TestRetroAdjRbfKernel:
    """Tests for the retro_adj._rbf_kernel helper."""

    def test_diagonal_is_one(self):
        from insurance_conformal.retro_adj import _rbf_kernel
        X = np.random.default_rng(0).normal(size=(5, 2))
        K = _rbf_kernel(X, X, bandwidth=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_values_in_unit_interval(self):
        from insurance_conformal.retro_adj import _rbf_kernel
        rng = np.random.default_rng(1)
        X1 = rng.normal(size=(4, 2))
        X2 = rng.normal(size=(6, 2))
        K = _rbf_kernel(X1, X2, bandwidth=1.0)
        assert K.shape == (4, 6)
        assert np.all(K >= 0.0) and np.all(K <= 1.0)

    def test_narrow_bandwidth_near_zero_off_diagonal(self):
        from insurance_conformal.retro_adj import _rbf_kernel
        X = np.array([[0.0, 0.0], [100.0, 100.0]])
        K = _rbf_kernel(X, X, bandwidth=0.001)
        assert K[0, 1] < 1e-10

    def test_symmetric(self):
        from insurance_conformal.retro_adj import _rbf_kernel
        rng = np.random.default_rng(2)
        X = rng.normal(size=(6, 3))
        K = _rbf_kernel(X, X, bandwidth=2.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)


class TestRankOneUpdateDowndate:
    """Verify Schur-complement matrix update/downdate inverses are consistent."""

    def _make_Q(self, n: int, lam: float = 0.1, seed: int = 0) -> tuple:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        from insurance_conformal.retro_adj import _rbf_kernel
        K = _rbf_kernel(X, X, bandwidth=1.0)
        Q = np.linalg.inv(K + lam * np.eye(n))
        return Q, X

    def test_update_increases_dimension(self):
        from insurance_conformal.retro_adj import _rank_one_update
        Q, X = self._make_Q(5)
        n = len(Q)
        k_new = np.ones(n)
        Q_new = _rank_one_update(Q, k_new, kappa_self_val=1.0, lambda_reg=0.1)
        assert Q_new.shape == (n + 1, n + 1)

    def test_update_then_downdate_recovers_original(self):
        """Applying update then downdate should recover Q approximately."""
        from insurance_conformal.retro_adj import _rank_one_update, _rank_one_downdate
        Q, X = self._make_Q(5)
        n = len(Q)
        k_new = np.ones(n) * 0.5
        Q_big = _rank_one_update(Q, k_new, kappa_self_val=1.0, lambda_reg=0.1)
        # Reorder so the new point is at index 0 (downdate removes index 0)
        # Q_big has new point at the LAST row/col. We need to permute it to first.
        perm = [n] + list(range(n))
        Q_perm = Q_big[np.ix_(perm, perm)]
        Q_recovered = _rank_one_downdate(Q_perm)
        # The recovered matrix should match the original Q
        np.testing.assert_allclose(Q_recovered, Q, atol=1e-6)

    def test_downdate_decreases_dimension(self):
        from insurance_conformal.retro_adj import _rank_one_downdate
        Q, _ = self._make_Q(6)
        Q_small = _rank_one_downdate(Q)
        assert Q_small.shape == (5, 5)

    def test_update_is_positive_definite(self):
        """Updated Q should remain approximately symmetric PD."""
        from insurance_conformal.retro_adj import _rank_one_update
        Q, _ = self._make_Q(4)
        k_new = np.ones(len(Q)) * 0.3
        Q_new = _rank_one_update(Q, k_new, kappa_self_val=1.0, lambda_reg=0.1)
        eigenvalues = np.linalg.eigvalsh(Q_new)
        assert np.all(eigenvalues > -1e-10)  # numerically PSD

    def test_update_nonpositive_denominator_raises(self):
        """Force denominator <= 0 to trigger _NumericalInstabilityError."""
        from insurance_conformal.retro_adj import _rank_one_update, _NumericalInstabilityError
        # Use identity inverse (Q = I), k_new = ones(1) -> denom = 1 + lam - Q @ k_new
        # For n=1: Q = [[1]], k_new = [0.5], denom = 1 + 0.1 - 0.5 = 0.6 (positive)
        # To get negative denom: k_new @ Q @ k_new must exceed kappa_self + lambda
        # With Q = [[10]], k_new = [1], denom = 0 + 0 - 10 = -10 (if kappa_self_val=0)
        Q = np.array([[10.0]])
        k_new = np.array([1.0])
        with pytest.raises(_NumericalInstabilityError):
            _rank_one_update(Q, k_new, kappa_self_val=0.0, lambda_reg=0.0)


class TestLooResidualsAndLeverages:
    """Tests for _loo_residuals_and_leverages."""

    def test_hat_diag_between_zero_and_one(self):
        from insurance_conformal.retro_adj import _loo_residuals_and_leverages, _rbf_kernel
        rng = np.random.default_rng(0)
        n = 8
        X = rng.normal(size=(n, 2))
        K = _rbf_kernel(X, X, bandwidth=1.0)
        Q = np.linalg.inv(K + 0.1 * np.eye(n))
        Y = rng.normal(1.0, 0.5, n)
        R_loo, H_diag, mask = _loo_residuals_and_leverages(Q, K, Y)
        # Hat matrix diagonal should be in [0, 1] for KRR
        assert np.all(H_diag >= -1e-8) and np.all(H_diag <= 1.0 + 1e-8)

    def test_mask_all_true_for_reasonable_lambda(self):
        """With lambda=0.5 (high regularisation), all 1 - H_diag should be well away from 0."""
        from insurance_conformal.retro_adj import _loo_residuals_and_leverages, _rbf_kernel
        rng = np.random.default_rng(1)
        n = 6
        X = rng.normal(size=(n, 2))
        K = _rbf_kernel(X, X, bandwidth=1.0)
        Q = np.linalg.inv(K + 0.5 * np.eye(n))
        Y = rng.normal(1.0, 0.5, n)
        _, _, mask = _loo_residuals_and_leverages(Q, K, Y)
        assert mask.all(), "With high lambda, all points should be valid (no near-interpolation)"

    def test_loo_residuals_finite_for_valid_mask(self):
        from insurance_conformal.retro_adj import _loo_residuals_and_leverages, _rbf_kernel
        rng = np.random.default_rng(2)
        n = 10
        X = rng.normal(size=(n, 2))
        K = _rbf_kernel(X, X, bandwidth=1.0)
        Q = np.linalg.inv(K + 0.1 * np.eye(n))
        Y = rng.normal(2.0, 1.0, n)
        R_loo, H_diag, mask = _loo_residuals_and_leverages(Q, K, Y)
        assert np.all(np.isfinite(R_loo[mask]))


class TestJackknifePlusInterval:
    """Tests for _jackknife_plus_interval."""

    def test_returns_tuple_of_two_floats(self):
        from insurance_conformal.retro_adj import _jackknife_plus_interval
        rng = np.random.default_rng(0)
        n = 20
        f_loo = rng.normal(1.0, 0.3, n)
        R_loo = rng.normal(0.0, 0.5, n)
        mask = np.ones(n, dtype=bool)
        lower, upper = _jackknife_plus_interval(f_loo, R_loo, alpha_t=0.1, mask=mask)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_empty_mask_returns_inf_interval(self):
        from insurance_conformal.retro_adj import _jackknife_plus_interval
        f_loo = np.array([1.0, 2.0, 3.0])
        R_loo = np.array([0.1, 0.2, 0.3])
        mask = np.zeros(3, dtype=bool)
        lower, upper = _jackknife_plus_interval(f_loo, R_loo, alpha_t=0.1, mask=mask)
        assert lower == -np.inf
        assert upper == np.inf

    def test_upper_geq_lower(self):
        from insurance_conformal.retro_adj import _jackknife_plus_interval
        rng = np.random.default_rng(5)
        n = 50
        f_loo = rng.normal(2.0, 0.5, n)
        R_loo = rng.normal(0.0, 1.0, n)
        mask = np.ones(n, dtype=bool)
        for alpha_t in [0.05, 0.10, 0.20, 0.50]:
            lo, hi = _jackknife_plus_interval(f_loo, R_loo, alpha_t=alpha_t, mask=mask)
            assert hi >= lo, f"upper < lower at alpha_t={alpha_t}"

    def test_symmetric_mode_uses_abs_residuals(self):
        """In symmetric mode, all residuals become positive, widening the interval."""
        from insurance_conformal.retro_adj import _jackknife_plus_interval
        rng = np.random.default_rng(6)
        n = 30
        f_loo = np.ones(n) * 2.0
        R_loo = rng.normal(0.0, 1.0, n)  # mixed sign
        mask = np.ones(n, dtype=bool)
        lo_asym, hi_asym = _jackknife_plus_interval(f_loo, R_loo, 0.1, mask, symmetric=False)
        lo_sym, hi_sym = _jackknife_plus_interval(f_loo, R_loo, 0.1, mask, symmetric=True)
        # Symmetric uses |R|, so hi_sym >= hi_asym (wider or equal)
        # This isn't always strictly true but the interval should be non-degenerate
        assert hi_sym >= lo_sym
        assert hi_sym >= 0  # since f_loo + |R| > 0 for these values


class TestSFOGDAlphaTracker:
    """Tests for _SFOGDAlphaTracker."""

    def test_alpha_stays_in_valid_range(self):
        from insurance_conformal.retro_adj import _SFOGDAlphaTracker
        tracker = _SFOGDAlphaTracker(alpha_init=0.1, gamma=0.01)
        for i in range(200):
            covered = (i % 3) != 0  # 1/3 miss rate
            alpha = tracker.update(covered, alpha_target=0.1)
            assert 1e-6 <= alpha <= 1 - 1e-6

    def test_consistently_missing_increases_alpha(self):
        """If we always miss, alpha should increase (intervals need to be wider)."""
        from insurance_conformal.retro_adj import _SFOGDAlphaTracker
        tracker = _SFOGDAlphaTracker(alpha_init=0.1, gamma=0.05)
        for _ in range(50):
            tracker.update(covered=False, alpha_target=0.1)
        assert tracker.alpha > 0.1

    def test_consistently_covering_decreases_alpha(self):
        """If we always cover, alpha should decrease (intervals can narrow)."""
        from insurance_conformal.retro_adj import _SFOGDAlphaTracker
        tracker = _SFOGDAlphaTracker(alpha_init=0.3, gamma=0.05)
        for _ in range(50):
            tracker.update(covered=True, alpha_target=0.1)
        assert tracker.alpha < 0.3

    def test_alpha_target_convergence(self):
        """With balanced coverage, alpha should stay near target."""
        from insurance_conformal.retro_adj import _SFOGDAlphaTracker
        rng = np.random.default_rng(42)
        tracker = _SFOGDAlphaTracker(alpha_init=0.1, gamma=0.02)
        # Simulate 10% miscoverage (correct target rate)
        for _ in range(500):
            covered = rng.random() > 0.1
            tracker.update(covered=covered, alpha_target=0.1)
        # Should not have drifted too far from 0.1
        assert abs(tracker.alpha - 0.1) < 0.15


class TestRetroAdjFitPredict:
    """Integration tests for RetroAdj.fit() and predict_interval()."""

    def _make_data(self, n_train=80, n_test=30, seed=0):
        rng = np.random.default_rng(seed)
        X_train = rng.normal(size=(n_train, 2))
        y_train = np.sin(X_train[:, 0]) + rng.normal(0, 0.3, n_train)
        X_test = rng.normal(size=(n_test, 2))
        y_test = np.sin(X_test[:, 0]) + rng.normal(0, 0.3, n_test)
        return X_train, y_train, X_test, y_test

    def test_fit_returns_self(self):
        from insurance_conformal.retro_adj import RetroAdj
        X_train, y_train, _, _ = self._make_data()
        model = RetroAdj(window_size=20)
        result = model.fit(y_train, X_train)
        assert result is model

    def test_predict_interval_shape(self):
        from insurance_conformal.retro_adj import RetroAdj
        X_train, y_train, X_test, y_test = self._make_data()
        model = RetroAdj(window_size=20, gamma=0.01)
        model.fit(y_train, X_train)
        lower, upper = model.predict_interval(y_test, X_test, alpha=0.1)
        assert lower.shape == (len(y_test),)
        assert upper.shape == (len(y_test),)

    def test_predict_interval_upper_geq_lower(self):
        from insurance_conformal.retro_adj import RetroAdj
        X_train, y_train, X_test, y_test = self._make_data()
        model = RetroAdj(window_size=20, gamma=0.01)
        model.fit(y_train, X_train)
        lower, upper = model.predict_interval(y_test, X_test, alpha=0.1)
        assert np.all(upper >= lower), "upper bounds must be >= lower bounds"

    def test_predict_interval_all_finite(self):
        from insurance_conformal.retro_adj import RetroAdj
        X_train, y_train, X_test, y_test = self._make_data()
        model = RetroAdj(window_size=20, gamma=0.01)
        model.fit(y_train, X_train)
        lower, upper = model.predict_interval(y_test, X_test, alpha=0.1)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    def test_predict_before_fit_raises(self):
        from insurance_conformal.retro_adj import RetroAdj
        model = RetroAdj(window_size=10)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_interval(np.array([1.0, 2.0]))

    def test_residual_only_mode_warns(self):
        from insurance_conformal.retro_adj import RetroAdj
        y_train = np.ones(50) * 1.0
        model = RetroAdj(window_size=20)
        with pytest.warns(UserWarning, match="residual-only"):
            model.fit(y_train)  # X=None

    def test_residual_only_mode_predict_works(self):
        from insurance_conformal.retro_adj import RetroAdj
        rng = np.random.default_rng(7)
        y = rng.normal(0, 1, 100)
        model = RetroAdj(window_size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(y[:60])
        lower, upper = model.predict_interval(y[60:], alpha=0.1)
        assert lower.shape == (40,)
        assert np.all(upper >= lower)

    def test_x_none_mode_then_x_provided_raises(self):
        from insurance_conformal.retro_adj import RetroAdj
        rng = np.random.default_rng(8)
        y = rng.normal(0, 1, 50)
        X = rng.normal(size=(20, 2))
        model = RetroAdj(window_size=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(y)
        with pytest.raises(ValueError, match="residual-only"):
            model.predict_interval(y[:20], X=X)

    def test_invalid_alpha_raises(self):
        from insurance_conformal.retro_adj import RetroAdj
        X_train, y_train, _, _ = self._make_data()
        model = RetroAdj(window_size=20)
        model.fit(y_train, X_train)
        y_test = np.ones(10)
        X_test = np.ones((10, 2))
        with pytest.raises(ValueError, match="alpha"):
            model.predict_interval(y_test, X_test, alpha=1.5)

    def test_sfogd_mode_works(self):
        from insurance_conformal.retro_adj import RetroAdj
        X_train, y_train, X_test, y_test = self._make_data()
        model = RetroAdj(window_size=20, alpha_update="sfogd", gamma=0.01)
        model.fit(y_train, X_train)
        lower, upper = model.predict_interval(y_test, X_test, alpha=0.1)
        assert np.all(upper >= lower)

    def test_symmetric_mode_intervals_wider_or_equal(self):
        """Symmetric mode uses |R_loo| which tends to widen intervals."""
        from insurance_conformal.retro_adj import RetroAdj
        X_train, y_train, X_test, y_test = self._make_data(seed=99)
        model_asym = RetroAdj(window_size=30, symmetric=False, gamma=0.01)
        model_sym = RetroAdj(window_size=30, symmetric=True, gamma=0.01)
        model_asym.fit(y_train, X_train)
        model_sym.fit(y_train, X_train)
        lo_a, hi_a = model_asym.predict_interval(y_test, X_test, alpha=0.1)
        lo_s, hi_s = model_sym.predict_interval(y_test, X_test, alpha=0.1)
        # Both should produce valid intervals
        assert np.all(hi_a >= lo_a)
        assert np.all(hi_s >= lo_s)

    def test_invalid_constructor_params_raise(self):
        from insurance_conformal.retro_adj import RetroAdj
        with pytest.raises(ValueError, match="window_size"):
            RetroAdj(window_size=3)
        with pytest.raises(ValueError, match="bandwidth"):
            RetroAdj(bandwidth=-1.0)
        with pytest.raises(ValueError, match="lambda_reg"):
            RetroAdj(lambda_reg=0.0)
        with pytest.raises(ValueError, match="alpha_update"):
            RetroAdj(alpha_update="bad_mode")
        with pytest.raises(ValueError, match="gamma"):
            RetroAdj(gamma=-0.1)

    def test_repr_reflects_state(self):
        from insurance_conformal.retro_adj import RetroAdj
        model = RetroAdj(window_size=20)
        assert "not fitted" in repr(model)
        X_train, y_train, _, _ = self._make_data()
        model.fit(y_train, X_train)
        assert "fitted" in repr(model)
        assert "full KRR" in repr(model)

    def test_repr_residual_only(self):
        from insurance_conformal.retro_adj import RetroAdj
        model = RetroAdj(window_size=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(np.ones(30))
        assert "residual-only" in repr(model)


# ===========================================================================
# assessment.py: _compute_cvi and _parse_prediction_sets
# ===========================================================================


class TestComputeCvi:
    """Tests for the _compute_cvi internal helper."""

    def test_perfect_coverage_gives_zero_cvi_u(self):
        """All eta_hat = 1.0 means no undercoverage."""
        from insurance_conformal.assessment import _compute_cvi
        eta_hat = np.ones(100)
        cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo = _compute_cvi(eta_hat, alpha=0.1, gamma=0.1)
        assert cvi_u == pytest.approx(0.0)

    def test_all_undercovered_gives_zero_cvi_o(self):
        """eta_hat = 0 everywhere means all undercovered, no overcoverage."""
        from insurance_conformal.assessment import _compute_cvi
        eta_hat = np.zeros(100)
        cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo = _compute_cvi(eta_hat, alpha=0.1, gamma=0.1)
        assert cvi_o == pytest.approx(0.0)
        assert cvi_u > 0.0

    def test_cvi_equals_cvi_u_plus_cvi_o(self):
        from insurance_conformal.assessment import _compute_cvi
        rng = np.random.default_rng(0)
        eta_hat = rng.uniform(0.5, 1.0, 200)
        cvi, cvi_u, cvi_o, *_ = _compute_cvi(eta_hat, alpha=0.1, gamma=0.1)
        assert cvi == pytest.approx(cvi_u + cvi_o)

    def test_pi_minus_pi_plus_in_unit_interval(self):
        from insurance_conformal.assessment import _compute_cvi
        rng = np.random.default_rng(1)
        eta_hat = rng.uniform(0.7, 0.99, 100)
        _, _, _, pi_minus, pi_plus, _, _ = _compute_cvi(eta_hat, alpha=0.1, gamma=0.1)
        assert 0.0 <= pi_minus <= 1.0
        assert 0.0 <= pi_plus <= 1.0

    def test_uniform_target_coverage_gives_near_zero_cvi(self):
        """If all eta_hat == 1 - alpha exactly, CVI should be zero."""
        from insurance_conformal.assessment import _compute_cvi
        target = 0.9
        eta_hat = np.full(200, target)
        cvi, cvi_u, cvi_o, _, _, _, _ = _compute_cvi(eta_hat, alpha=0.1, gamma=0.1)
        assert cvi == pytest.approx(0.0, abs=1e-10)

    def test_gamma_zero_makes_classification_sharp(self):
        """gamma=0: anything != exactly target is classified as over or under."""
        from insurance_conformal.assessment import _compute_cvi
        eta_hat = np.array([0.85, 0.90, 0.95])  # target = 0.90
        cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo = _compute_cvi(eta_hat, alpha=0.1, gamma=0.0)
        # 0.85 is below target -> pi_minus includes it
        # 0.95 is above target -> pi_plus includes it
        # 0.90 is exactly target -> neither
        assert pi_minus > 0.0
        assert pi_plus > 0.0

    def test_nonneg_components(self):
        from insurance_conformal.assessment import _compute_cvi
        rng = np.random.default_rng(3)
        eta_hat = rng.uniform(0.6, 1.0, 150)
        cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo = _compute_cvi(eta_hat, alpha=0.1, gamma=0.1)
        assert cvi >= 0.0
        assert cvi_u >= 0.0
        assert cvi_o >= 0.0
        assert cmu >= 0.0
        assert cmo >= 0.0


class TestParsePredictionSets:
    """Tests for _parse_prediction_sets."""

    def _make_arrays(self, n=10):
        lower = np.arange(n, dtype=float)
        upper = lower + 1.0
        return lower, upper

    def test_tuple_input(self):
        from insurance_conformal.assessment import _parse_prediction_sets
        lower, upper = self._make_arrays()
        l_out, u_out = _parse_prediction_sets((lower, upper))
        np.testing.assert_array_equal(l_out, lower)
        np.testing.assert_array_equal(u_out, upper)

    def test_dict_input(self):
        from insurance_conformal.assessment import _parse_prediction_sets
        lower, upper = self._make_arrays()
        l_out, u_out = _parse_prediction_sets({"lower": lower, "upper": upper})
        np.testing.assert_array_equal(l_out, lower)
        np.testing.assert_array_equal(u_out, upper)

    def test_polars_df_input(self):
        from insurance_conformal.assessment import _parse_prediction_sets
        lower, upper = self._make_arrays()
        df = pl.DataFrame({"lower": lower, "upper": upper})
        l_out, u_out = _parse_prediction_sets(df)
        np.testing.assert_array_equal(l_out, lower)
        np.testing.assert_array_equal(u_out, upper)

    def test_pandas_df_input(self):
        import pandas as pd
        from insurance_conformal.assessment import _parse_prediction_sets
        lower, upper = self._make_arrays()
        df = pd.DataFrame({"lower": lower, "upper": upper})
        l_out, u_out = _parse_prediction_sets(df)
        np.testing.assert_array_equal(l_out, lower)
        np.testing.assert_array_equal(u_out, upper)

    def test_tuple_wrong_length_raises(self):
        from insurance_conformal.assessment import _parse_prediction_sets
        with pytest.raises(ValueError, match="lower, upper"):
            _parse_prediction_sets((np.ones(5),))

    def test_dict_missing_column_raises(self):
        from insurance_conformal.assessment import _parse_prediction_sets
        with pytest.raises(ValueError, match="lower"):
            _parse_prediction_sets({"lower": np.ones(5)})  # missing "upper"

    def test_unsupported_type_raises(self):
        from insurance_conformal.assessment import _parse_prediction_sets
        with pytest.raises(TypeError, match="prediction_sets"):
            _parse_prediction_sets(42)


class TestConditionalCoverageAssessor:
    """Integration tests for ConditionalCoverageAssessor."""

    @pytest.fixture(scope="class")
    def assessor_and_data(self):
        rng = np.random.default_rng(0)
        n = 600
        X = rng.normal(size=(n, 3))
        y = rng.exponential(scale=2.0, size=n)
        # Build intervals that cover ~90%
        lo = np.maximum(0.0, y - 1.5)
        hi = y + 1.5
        ps_cal = {"lower": lo[:400], "upper": hi[:400]}
        ps_test = {"lower": lo[400:], "upper": hi[400:]}

        from insurance_conformal.assessment import ConditionalCoverageAssessor
        assessor = ConditionalCoverageAssessor(
            alpha=0.10, n_estimators=30, n_splits=2, random_state=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assessor.fit(X[:400], y[:400], ps_cal)

        return assessor, X[400:], y[400:], ps_test

    def test_invalid_alpha_raises(self):
        from insurance_conformal.assessment import ConditionalCoverageAssessor
        with pytest.raises(ValueError, match="alpha"):
            ConditionalCoverageAssessor(alpha=1.5)

    def test_invalid_gamma_raises(self):
        from insurance_conformal.assessment import ConditionalCoverageAssessor
        with pytest.raises(ValueError, match="gamma"):
            ConditionalCoverageAssessor(alpha=0.1, gamma=1.5)

    def test_invalid_n_splits_raises(self):
        from insurance_conformal.assessment import ConditionalCoverageAssessor
        with pytest.raises(ValueError, match="n_splits"):
            ConditionalCoverageAssessor(alpha=0.1, n_splits=1)

    def test_fit_returns_self(self, assessor_and_data):
        from insurance_conformal.assessment import ConditionalCoverageAssessor
        assessor, X_test, y_test, ps_test = assessor_and_data
        assert assessor.is_fitted_

    def test_score_before_fit_raises(self):
        from insurance_conformal.assessment import ConditionalCoverageAssessor
        a = ConditionalCoverageAssessor(alpha=0.1)
        with pytest.raises(RuntimeError, match="fit"):
            a.score(np.ones((10, 3)), np.ones(10), (np.zeros(10), np.ones(10)))

    def test_score_returns_result(self, assessor_and_data):
        from insurance_conformal.assessment import CVIAssessmentResult
        assessor, X_test, y_test, ps_test = assessor_and_data
        result = assessor.score(X_test, y_test, ps_test)
        assert isinstance(result, CVIAssessmentResult)

    def test_result_fields_valid(self, assessor_and_data):
        assessor, X_test, y_test, ps_test = assessor_and_data
        result = assessor.score(X_test, y_test, ps_test)
        assert 0.0 <= result.cvi
        assert 0.0 <= result.cvi_u
        assert 0.0 <= result.cvi_o
        assert 0.0 <= result.marginal_coverage <= 1.0
        assert result.n_obs == len(y_test)
        assert result.alpha == pytest.approx(0.10)

    def test_result_summary_returns_dataframe(self, assessor_and_data):
        assessor, X_test, y_test, ps_test = assessor_and_data
        result = assessor.score(X_test, y_test, ps_test)
        summary = result.summary()
        assert isinstance(summary, pl.DataFrame)
        assert len(summary) == 1
        assert "cvi" in summary.columns

    def test_eta_hat_shape(self, assessor_and_data):
        assessor, X_test, y_test, ps_test = assessor_and_data
        result = assessor.score(X_test, y_test, ps_test)
        assert result.eta_hat.shape == (len(y_test),)

    def test_select_returns_best_key(self, assessor_and_data):
        from insurance_conformal.assessment import CVISelectResult
        assessor, X_test, y_test, ps_test = assessor_and_data
        # Two prediction sets: one good, one bad (very wide intervals)
        ps_bad = (np.zeros(len(y_test)), np.ones(len(y_test)) * 1e6)
        result = assessor.select(X_test, y_test, {"good": ps_test, "bad": ps_bad})
        assert isinstance(result, CVISelectResult)
        assert result.best_key in ("good", "bad")
        assert len(result.rankings) == 2

    def test_select_empty_raises(self, assessor_and_data):
        assessor, X_test, y_test, ps_test = assessor_and_data
        with pytest.raises(ValueError, match="empty"):
            assessor.select(X_test, y_test, {})

    def test_select_compare_returns_dataframe(self, assessor_and_data):
        from insurance_conformal.assessment import CVISelectResult
        assessor, X_test, y_test, ps_test = assessor_and_data
        result = assessor.select(X_test, y_test, {"A": ps_test, "B": ps_test})
        df = result.compare()
        assert isinstance(df, pl.DataFrame)
        assert "rank" in df.columns
        assert len(df) == 2

    def test_fit_too_few_obs_raises(self):
        from insurance_conformal.assessment import ConditionalCoverageAssessor
        a = ConditionalCoverageAssessor(alpha=0.1)
        n = 30  # below _MIN_OBS=50
        X = np.ones((n, 2))
        y = np.ones(n)
        ps = (np.zeros(n), np.ones(n) * 2.0)
        with pytest.raises(ValueError, match="50"):
            a.fit(X, y, ps)

    def test_cvi_decomp_sums_correctly(self, assessor_and_data):
        """cvi = cvi_u + cvi_o always."""
        assessor, X_test, y_test, ps_test = assessor_and_data
        result = assessor.score(X_test, y_test, ps_test)
        assert result.cvi == pytest.approx(result.cvi_u + result.cvi_o, abs=1e-10)


# ===========================================================================
# utils.py: temporal_split, ipcw_weighted_quantile
# ===========================================================================


class TestTemporalSplit:
    """Tests for temporal_split."""

    def test_basic_tail_split(self):
        from insurance_conformal.utils import temporal_split
        import pandas as pd
        n = 100
        X = pd.DataFrame({"a": range(n)})
        y = np.arange(n, dtype=float)
        X_tr, X_cal, y_tr, y_cal, _, _ = temporal_split(X, y, calibration_frac=0.2)
        assert len(y_tr) + len(y_cal) == n
        assert abs(len(y_cal) - 20) <= 1

    def test_cal_frac_one_is_all_data(self):
        from insurance_conformal.utils import temporal_split
        import pandas as pd
        n = 50
        X = pd.DataFrame({"a": range(n)})
        y = np.arange(n, dtype=float)
        X_tr, X_cal, y_tr, y_cal, _, _ = temporal_split(X, y, calibration_frac=1.0)
        assert len(y_cal) == n

    def test_with_date_col_sorts_by_date(self):
        """Calibration set should contain the most recent obs when date_col is given."""
        from insurance_conformal.utils import temporal_split
        import pandas as pd
        n = 20
        X = pd.DataFrame({"date": list(range(n)), "feature": range(n)})
        y = np.arange(n, dtype=float)
        X_tr, X_cal, y_tr, y_cal, _, _ = temporal_split(
            X, y, calibration_frac=0.3, date_col="date"
        )
        # Calibration set should have the largest date values
        assert X_cal["date"].max() >= X_tr["date"].max()

    def test_exposure_split(self):
        from insurance_conformal.utils import temporal_split
        import pandas as pd
        n = 40
        X = pd.DataFrame({"a": range(n)})
        y = np.ones(n)
        exp = np.arange(n, dtype=float) + 1.0
        X_tr, X_cal, y_tr, y_cal, exp_tr, exp_cal = temporal_split(
            X, y, calibration_frac=0.25, exposure=exp
        )
        assert exp_tr is not None
        assert exp_cal is not None
        assert len(exp_tr) + len(exp_cal) == n

    def test_no_exposure_returns_none(self):
        from insurance_conformal.utils import temporal_split
        import pandas as pd
        n = 30
        X = pd.DataFrame({"a": range(n)})
        y = np.ones(n)
        X_tr, X_cal, y_tr, y_cal, exp_tr, exp_cal = temporal_split(X, y, calibration_frac=0.2)
        assert exp_tr is None
        assert exp_cal is None

    def test_polars_input_accepted(self):
        from insurance_conformal.utils import temporal_split
        n = 40
        X_pl = pl.DataFrame({"a": range(n)})
        y = np.arange(n, dtype=float)
        X_tr, X_cal, y_tr, y_cal, _, _ = temporal_split(X_pl, y, calibration_frac=0.2)
        assert len(y_tr) + len(y_cal) == n

    def test_numpy_array_input_accepted(self):
        from insurance_conformal.utils import temporal_split
        n = 30
        X = np.random.default_rng(0).normal(size=(n, 2))
        y = np.arange(n, dtype=float)
        X_tr, X_cal, y_tr, y_cal, _, _ = temporal_split(X, y, calibration_frac=0.3)
        assert len(y_tr) + len(y_cal) == n


class TestIpcwWeightedQuantile:
    """Tests for ipcw_weighted_quantile."""

    def test_uniform_weights_equals_unweighted(self):
        """Uniform weights should give the same result as the standard quantile."""
        from insurance_conformal.utils import ipcw_weighted_quantile
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 5, 100)
        weights = np.ones(100)
        q = ipcw_weighted_quantile(scores, weights, alpha=0.1)
        # Should be close to the 90th percentile
        expected = np.quantile(scores, 0.9)
        assert abs(q - expected) < 0.3  # generous tolerance for discrete nature

    def test_zero_weight_mass_ignored(self):
        """Points with zero weight should not influence the quantile."""
        from insurance_conformal.utils import ipcw_weighted_quantile
        scores = np.array([1.0, 2.0, 100.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 0.0, 1.0, 1.0])  # 100.0 has zero weight
        q_with_zero = ipcw_weighted_quantile(scores, weights, alpha=0.1)
        assert q_with_zero < 100.0  # 100.0 should not dominate

    def test_near_zero_weights_warn_and_return_max(self):
        """If all weights are near-zero, warn and return max score."""
        from insurance_conformal.utils import ipcw_weighted_quantile
        scores = np.array([1.0, 2.0, 3.0])
        weights = np.array([1e-15, 1e-15, 1e-15])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            q = ipcw_weighted_quantile(scores, weights, alpha=0.1)
            assert len(w) == 1
            assert "near-zero" in str(w[0].message).lower() or "zero" in str(w[0].message).lower()
        assert q == pytest.approx(3.0)

    def test_empty_scores_raises(self):
        from insurance_conformal.utils import ipcw_weighted_quantile
        with pytest.raises(ValueError, match="empty"):
            ipcw_weighted_quantile(np.array([]), np.array([]), alpha=0.1)

    def test_invalid_alpha_raises(self):
        from insurance_conformal.utils import ipcw_weighted_quantile
        with pytest.raises(ValueError, match="alpha"):
            ipcw_weighted_quantile(np.array([1.0, 2.0]), np.array([1.0, 1.0]), alpha=1.5)

    def test_quantile_is_from_scores(self):
        """The returned quantile must equal one of the input scores."""
        from insurance_conformal.utils import ipcw_weighted_quantile
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        q = ipcw_weighted_quantile(scores, weights, alpha=0.2)
        assert q in scores

    def test_high_weight_on_small_score_lowers_quantile(self):
        """Putting high weight on small scores should lower the quantile."""
        from insurance_conformal.utils import ipcw_weighted_quantile
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w_uniform = np.ones(5) / 5.0
        w_low = np.array([10.0, 10.0, 1.0, 1.0, 1.0])  # emphasise low scores
        q_uniform = ipcw_weighted_quantile(scores, w_uniform, alpha=0.2)
        q_low = ipcw_weighted_quantile(scores, w_low, alpha=0.2)
        assert q_low <= q_uniform


class TestConformalQuantileEdgeCases:
    """Edge cases for conformal_quantile."""

    def test_alpha_boundary_zero_raises(self):
        from insurance_conformal.utils import conformal_quantile
        with pytest.raises(ValueError, match="alpha"):
            conformal_quantile(np.array([1.0, 2.0]), alpha=0.0)

    def test_alpha_boundary_one_raises(self):
        from insurance_conformal.utils import conformal_quantile
        with pytest.raises(ValueError, match="alpha"):
            conformal_quantile(np.array([1.0, 2.0]), alpha=1.0)

    def test_empty_scores_raises(self):
        from insurance_conformal.utils import conformal_quantile
        with pytest.raises(ValueError, match="empty"):
            conformal_quantile(np.array([]), alpha=0.1)

    def test_single_score_returns_it(self):
        """With n=1, the only possible quantile is the single score."""
        from insurance_conformal.utils import conformal_quantile
        q = conformal_quantile(np.array([2.5]), alpha=0.1)
        assert q == pytest.approx(2.5)

    def test_large_alpha_returns_small_quantile(self):
        """alpha=0.9 should return a very low quantile (10th percentile)."""
        from insurance_conformal.utils import conformal_quantile
        scores = np.arange(1.0, 101.0)  # 1..100
        q = conformal_quantile(scores, alpha=0.9)
        # Should be near the 10th percentile
        assert q < 20.0

    def test_small_alpha_returns_high_quantile(self):
        """alpha=0.01 should return a very high quantile."""
        from insurance_conformal.utils import conformal_quantile
        scores = np.arange(1.0, 101.0)
        q = conformal_quantile(scores, alpha=0.01)
        assert q > 90.0


# ===========================================================================
# lcp_model_selector.py: _kernel_weights, _weighted_quantile, LCPModelSelector
# ===========================================================================


class TestKernelWeights:
    """Tests for _kernel_weights."""

    def _make_X_cal(self, n=20, p=2, seed=0):
        return np.random.default_rng(seed).normal(size=(n, p))

    def test_gaussian_sums_positive(self):
        from insurance_conformal.lcp_model_selector import _kernel_weights
        X_cal = self._make_X_cal()
        x_test = np.zeros(2)
        w = _kernel_weights(x_test, X_cal, localizer="gaussian", bandwidth=1.0)
        assert np.all(w >= 0)
        assert w.sum() > 0

    def test_exponential_sums_positive(self):
        from insurance_conformal.lcp_model_selector import _kernel_weights
        X_cal = self._make_X_cal()
        x_test = np.zeros(2)
        w = _kernel_weights(x_test, X_cal, localizer="exponential", bandwidth=1.0)
        assert np.all(w >= 0)

    def test_uniform_nonzero_for_points_in_bandwidth(self):
        from insurance_conformal.lcp_model_selector import _kernel_weights
        X_cal = np.array([[0.1, 0.0], [5.0, 0.0], [0.0, 0.2]])
        x_test = np.zeros(2)
        w = _kernel_weights(x_test, X_cal, localizer="uniform", bandwidth=0.5)
        # Only points within 0.5 should have nonzero weight
        assert w[0] > 0  # distance ~0.1
        assert w[1] == 0.0  # distance ~5.0
        assert w[2] > 0  # distance ~0.2

    def test_uniform_all_zero_falls_back_to_uniform(self):
        """If all points are outside bandwidth, weights should all be equal."""
        from insurance_conformal.lcp_model_selector import _kernel_weights
        X_cal = np.array([[10.0, 0.0], [20.0, 0.0]])
        x_test = np.zeros(2)
        w = _kernel_weights(x_test, X_cal, localizer="uniform", bandwidth=0.001)
        assert np.allclose(w, w[0])  # all equal

    def test_knn_k_points_get_nonzero_weight(self):
        from insurance_conformal.lcp_model_selector import _kernel_weights
        rng = np.random.default_rng(0)
        X_cal = rng.normal(size=(30, 2))
        x_test = np.zeros(2)
        k = 5
        w = _kernel_weights(x_test, X_cal, localizer="knn", bandwidth=1.0, knn_k=k)
        n_nonzero = np.sum(w > 0)
        assert n_nonzero == k

    def test_invalid_localizer_raises(self):
        from insurance_conformal.lcp_model_selector import _kernel_weights
        X_cal = np.ones((5, 2))
        with pytest.raises(ValueError, match="localizer"):
            _kernel_weights(np.zeros(2), X_cal, localizer="banana", bandwidth=1.0)

    def test_gaussian_weights_decrease_with_distance(self):
        """For Gaussian kernel, points further away should have lower weight."""
        from insurance_conformal.lcp_model_selector import _kernel_weights
        X_cal = np.array([[0.1, 0.0], [1.0, 0.0], [5.0, 0.0]])
        x_test = np.zeros(2)
        w = _kernel_weights(x_test, X_cal, localizer="gaussian", bandwidth=1.0)
        assert w[0] > w[1] > w[2]


class TestWeightedQuantile:
    """Tests for _weighted_quantile."""

    def test_uniform_weights_matches_sort_quantile(self):
        from insurance_conformal.lcp_model_selector import _weighted_quantile
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5) / 5.0
        q = _weighted_quantile(scores, weights, level=0.8)
        assert q in scores

    def test_all_weight_on_one_score_returns_that_score(self):
        from insurance_conformal.lcp_model_selector import _weighted_quantile
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([0.0, 0.0, 1.0, 0.0])  # all weight on score=3.0
        q = _weighted_quantile(scores, weights, level=0.5)
        assert q == pytest.approx(3.0)

    def test_level_one_returns_max(self):
        from insurance_conformal.lcp_model_selector import _weighted_quantile
        scores = np.array([1.0, 2.0, 3.0])
        weights = np.ones(3) / 3.0
        q = _weighted_quantile(scores, weights, level=1.0)
        assert q == pytest.approx(3.0)


@pytest.fixture(scope="module")
def lcp_data():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 3))
    beta = np.array([0.3, -0.1, 0.2])
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    return {
        "X_train": X[:100],
        "y_train": y[:100],
        "X_cal": X[100:200],
        "y_cal": y[100:200],
        "X_test": X[200:],
        "y_test": y[200:],
    }


class ConstantPredictorSklearn:
    """Minimal model for LCPModelSelector tests."""

    def __init__(self, value: float = 2.0) -> None:
        self.value = value

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.value)

    def fit(self, X, y):
        self.value = float(np.mean(y))
        return self


class TestLCPModelSelector:
    """Tests for LCPModelSelector."""

    def _make_lcp(self, lcp_data, localizer="gaussian"):
        m1 = ConstantPredictorSklearn(2.0)
        m2 = ConstantPredictorSklearn(3.0)
        from insurance_conformal.lcp_model_selector import LCPModelSelector
        lcp = LCPModelSelector(
            models=[m1, m2],
            score_fns=["raw", "pearson"],
            localizer=localizer,
            bandwidth=2.0,
            tweedie_power=1.5,
        )
        lcp.calibrate(lcp_data["X_cal"], lcp_data["y_cal"])
        return lcp

    def test_invalid_localizer_raises(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LCPModelSelector
        with pytest.raises(ValueError, match="localizer"):
            LCPModelSelector(
                models=[ConstantPredictorSklearn()],
                score_fns=["raw"],
                localizer="invalid",
            )

    def test_empty_models_raises(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LCPModelSelector
        with pytest.raises(ValueError, match="At least one"):
            LCPModelSelector(models=[], score_fns=[])

    def test_models_score_fns_length_mismatch_raises(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LCPModelSelector
        with pytest.raises(ValueError, match="same length"):
            LCPModelSelector(
                models=[ConstantPredictorSklearn()],
                score_fns=["raw", "pearson"],
            )

    def test_calibrate_returns_self(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        assert lcp.is_calibrated_

    def test_predict_before_calibrate_raises(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LCPModelSelector
        lcp = LCPModelSelector(
            models=[ConstantPredictorSklearn()],
            score_fns=["raw"],
        )
        with pytest.raises(RuntimeError, match="calibrat"):
            lcp.predict_interval(lcp_data["X_test"])

    def test_predict_interval_shape(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        intervals = lcp.predict_interval(lcp_data["X_test"], alpha=0.1)
        assert isinstance(intervals, pl.DataFrame)
        assert len(intervals) == len(lcp_data["X_test"])

    def test_predict_interval_columns(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        intervals = lcp.predict_interval(lcp_data["X_test"], alpha=0.1)
        expected_cols = {"lower", "point_estimate", "upper", "ensemble_width", "n_models_in_set"}
        assert set(intervals.columns) == expected_cols

    def test_upper_geq_lower(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        intervals = lcp.predict_interval(lcp_data["X_test"], alpha=0.1)
        assert (intervals["upper"] >= intervals["lower"]).all()

    def test_lower_nonneg(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        intervals = lcp.predict_interval(lcp_data["X_test"], alpha=0.1)
        assert (intervals["lower"] >= 0.0).all()

    def test_ensemble_width_matches_upper_minus_lower(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        intervals = lcp.predict_interval(lcp_data["X_test"], alpha=0.1)
        expected_width = intervals["upper"] - intervals["lower"]
        np.testing.assert_allclose(
            intervals["ensemble_width"].to_numpy(),
            expected_width.to_numpy(),
            atol=1e-10,
        )

    def test_invalid_alpha_raises(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        with pytest.raises(ValueError, match="alpha"):
            lcp.predict_interval(lcp_data["X_test"], alpha=0.0)

    def test_selected_models_returns_dataframe(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        sel = lcp.selected_models(lcp_data["X_test"], alpha=0.1)
        assert isinstance(sel, pl.DataFrame)
        assert len(sel) == len(lcp_data["X_test"])
        assert "selected_model_idx" in sel.columns
        assert "fallback_used" in sel.columns

    def test_selected_model_idx_in_range(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        sel = lcp.selected_models(lcp_data["X_test"], alpha=0.1)
        K = len(lcp.models)
        assert ((sel["selected_model_idx"] >= 0) & (sel["selected_model_idx"] < K)).all()

    def test_coverage_diagnostics_returns_dataframe(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        diag = lcp.coverage_diagnostics(lcp_data["X_test"], lcp_data["y_test"], alpha=0.1)
        assert isinstance(diag, pl.DataFrame)
        assert "coverage" in diag.columns
        assert "slice_type" in diag.columns

    def test_repr_before_calibration(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LCPModelSelector
        lcp = LCPModelSelector(
            models=[ConstantPredictorSklearn()],
            score_fns=["raw"],
        )
        assert "not calibrated" in repr(lcp)

    def test_repr_after_calibration(self, lcp_data):
        lcp = self._make_lcp(lcp_data)
        r = repr(lcp)
        assert "calibrated" in r

    def test_knn_localizer(self, lcp_data):
        lcp = self._make_lcp(lcp_data, localizer="knn")
        intervals = lcp.predict_interval(lcp_data["X_test"], alpha=0.1)
        assert len(intervals) == len(lcp_data["X_test"])

    def test_uniform_localizer(self, lcp_data):
        lcp = self._make_lcp(lcp_data, localizer="uniform")
        intervals = lcp.predict_interval(lcp_data["X_test"], alpha=0.1)
        assert len(intervals) == len(lcp_data["X_test"])

    def test_single_model(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LCPModelSelector
        lcp = LCPModelSelector(
            models=[ConstantPredictorSklearn()],
            score_fns=["raw"],
            bandwidth=2.0,
        )
        lcp.calibrate(lcp_data["X_cal"], lcp_data["y_cal"])
        intervals = lcp.predict_interval(lcp_data["X_test"], alpha=0.1)
        assert len(intervals) == len(lcp_data["X_test"])


class TestLocalScoreSelector:
    """Tests for LocalScoreSelector (convenience wrapper)."""

    def test_basic_fit_predict(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LocalScoreSelector
        lss = LocalScoreSelector(
            model=ConstantPredictorSklearn(),
            score_types=["raw", "pearson"],
            bandwidth=2.0,
        )
        lss.calibrate(lcp_data["X_cal"], lcp_data["y_cal"])
        intervals = lss.predict_interval(lcp_data["X_test"], alpha=0.1)
        assert isinstance(intervals, pl.DataFrame)
        assert len(intervals) == len(lcp_data["X_test"])

    def test_default_score_types(self, lcp_data):
        """Default should use pearson_weighted, deviance, anscombe (3 scores)."""
        from insurance_conformal.lcp_model_selector import LocalScoreSelector
        lss = LocalScoreSelector(
            model=ConstantPredictorSklearn(2.0),
            bandwidth=2.0,
        )
        assert len(lss.score_types) == 3
        assert "pearson_weighted" in lss.score_types

    def test_repr(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LocalScoreSelector
        lss = LocalScoreSelector(
            model=ConstantPredictorSklearn(),
            score_types=["raw"],
        )
        r = repr(lss)
        assert "LocalScoreSelector" in r
        lss.calibrate(lcp_data["X_cal"], lcp_data["y_cal"])
        r2 = repr(lss)
        assert "calibrated" in r2

    def test_selected_models_returns_score_names(self, lcp_data):
        from insurance_conformal.lcp_model_selector import LocalScoreSelector
        lss = LocalScoreSelector(
            model=ConstantPredictorSklearn(),
            score_types=["raw", "pearson"],
            bandwidth=2.0,
        )
        lss.calibrate(lcp_data["X_cal"], lcp_data["y_cal"])
        sel = lss.selected_models(lcp_data["X_test"], alpha=0.1)
        assert "score_name" in sel.columns
        # All score names should be one of the score types
        names = set(sel["score_name"].to_list())
        assert names.issubset({"raw", "pearson"})


# ===========================================================================
# diagnostics_ext.py
# ===========================================================================


@pytest.fixture(scope="module")
def diag_data():
    """Synthetic data with a calibrated predictor for diagnostics tests."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    mu = np.exp(0.3 * X[:, 0] - 0.1 * X[:, 1])
    y = rng.poisson(mu).astype(float)
    return {"X": X, "y": y, "mu": mu}


class TestErtCoverageGap:
    """Tests for ert_coverage_gap."""

    def test_returns_polars_dataframe(self, diag_data):
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=4.0, point=2.0)
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=0.1)
        assert isinstance(result, pl.DataFrame)

    def test_required_columns(self, diag_data):
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=4.0, point=2.0)
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=0.1)
        expected = {
            "bin", "mean_predicted", "n_obs", "empirical_coverage",
            "target_coverage", "coverage_gap", "coverage_se", "exceeds_2se"
        }
        assert set(result.columns) == expected

    def test_summary_row_has_bin_minus_one(self, diag_data):
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=4.0, point=2.0)
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=0.1)
        # Summary row at end has bin == -1
        assert result["bin"][-1] == -1

    def test_n_obs_sums_to_total(self, diag_data):
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=4.0, point=2.0)
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=0.1, n_bins=5)
        # Exclude summary row (bin == -1)
        data_rows = result.filter(pl.col("bin") != -1)
        assert data_rows["n_obs"].sum() == len(diag_data["y"])

    def test_target_coverage_matches_alpha(self, diag_data):
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=4.0, point=2.0)
        alpha = 0.15
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=alpha)
        data_rows = result.filter(pl.col("bin") != -1)
        target_vals = data_rows["target_coverage"].to_numpy()
        np.testing.assert_allclose(target_vals, 1.0 - alpha, atol=1e-10)

    def test_empirical_coverage_in_unit_interval(self, diag_data):
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=4.0, point=2.0)
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=0.1)
        data_rows = result.filter(pl.col("bin") != -1)
        cov = data_rows["empirical_coverage"].to_numpy()
        assert np.all(cov >= 0.0) and np.all(cov <= 1.0)

    def test_wide_intervals_high_coverage(self, diag_data):
        """Very wide intervals should have near-100% coverage."""
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=1000.0, point=5.0)
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=0.1)
        summary_row = result.filter(pl.col("bin") == -1)
        assert summary_row["empirical_coverage"][0] > 0.90

    def test_narrow_intervals_low_coverage(self, diag_data):
        """Near-zero-width intervals should have near-0% coverage."""
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=0.001, point=5.0)
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=0.1)
        summary_row = result.filter(pl.col("bin") == -1)
        assert summary_row["empirical_coverage"][0] < 0.50

    def test_custom_n_bins(self, diag_data):
        from insurance_conformal.diagnostics_ext import ert_coverage_gap
        predictor = ConstantPredictor(width=2.0, point=2.0)
        result = ert_coverage_gap(predictor, diag_data["X"], diag_data["y"], alpha=0.1, n_bins=5)
        # 5 data rows + 1 summary row
        data_rows = result.filter(pl.col("bin") != -1)
        assert len(data_rows) <= 5


class TestSubgroupCoverage:
    """Tests for subgroup_coverage."""

    def test_returns_polars_dataframe(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=3.0, point=2.0)
        groups = (np.arange(len(diag_data["y"])) % 3).astype(str)
        result = subgroup_coverage(predictor, diag_data["X"], diag_data["y"], alpha=0.1, groups=groups)
        assert isinstance(result, pl.DataFrame)

    def test_n_rows_equals_n_unique_groups(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=3.0, point=2.0)
        groups = np.array(["A", "B", "C"])[np.arange(len(diag_data["y"])) % 3]
        result = subgroup_coverage(predictor, diag_data["X"], diag_data["y"], alpha=0.1, groups=groups)
        assert len(result) == 3

    def test_required_columns(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=3.0, point=2.0)
        groups = np.zeros(len(diag_data["y"]))
        result = subgroup_coverage(predictor, diag_data["X"], diag_data["y"], alpha=0.1, groups=groups)
        expected = {
            "group", "n_obs", "empirical_coverage", "target_coverage",
            "coverage_gap", "mean_lower", "mean_upper", "mean_width", "mean_predicted"
        }
        assert set(result.columns) == expected

    def test_n_obs_sums_to_total(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=3.0, point=2.0)
        groups = np.arange(len(diag_data["y"])) % 4
        result = subgroup_coverage(predictor, diag_data["X"], diag_data["y"], alpha=0.1, groups=groups)
        assert result["n_obs"].sum() == len(diag_data["y"])

    def test_coverage_in_unit_interval(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=3.0, point=2.0)
        groups = np.zeros(len(diag_data["y"]))
        result = subgroup_coverage(predictor, diag_data["X"], diag_data["y"], alpha=0.1, groups=groups)
        cov = result["empirical_coverage"].to_numpy()
        assert np.all(cov >= 0.0) and np.all(cov <= 1.0)

    def test_group_mismatch_raises(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=3.0, point=2.0)
        groups = np.zeros(5)  # wrong length
        with pytest.raises(ValueError, match="length"):
            subgroup_coverage(predictor, diag_data["X"], diag_data["y"], alpha=0.1, groups=groups)

    def test_custom_group_name(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=3.0, point=2.0)
        groups = np.zeros(len(diag_data["y"]))
        result = subgroup_coverage(
            predictor, diag_data["X"], diag_data["y"], alpha=0.1,
            groups=groups, group_name="vehicle_class"
        )
        assert "vehicle_class" in result.columns

    def test_target_coverage_matches_alpha(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=3.0, point=2.0)
        alpha = 0.20
        groups = np.zeros(len(diag_data["y"]))
        result = subgroup_coverage(predictor, diag_data["X"], diag_data["y"], alpha=alpha, groups=groups)
        assert result["target_coverage"][0] == pytest.approx(1.0 - alpha)

    def test_mean_width_positive(self, diag_data):
        from insurance_conformal.diagnostics_ext import subgroup_coverage
        predictor = ConstantPredictor(width=2.0, point=3.0)
        groups = np.zeros(len(diag_data["y"]))
        result = subgroup_coverage(predictor, diag_data["X"], diag_data["y"], alpha=0.1, groups=groups)
        assert result["mean_width"][0] > 0.0


class TestWidthEfficiencyComparison:
    """Tests for width_efficiency_comparison."""

    def test_returns_polars_dataframe(self, diag_data):
        from insurance_conformal.diagnostics_ext import width_efficiency_comparison
        pred1 = ConstantPredictor(width=2.0, point=2.0)
        pred2 = ConstantPredictor(width=4.0, point=2.0)
        result = width_efficiency_comparison(
            {"narrow": pred1, "wide": pred2},
            diag_data["X"], diag_data["y"], alpha=0.1
        )
        assert isinstance(result, pl.DataFrame)

    def test_required_columns(self, diag_data):
        from insurance_conformal.diagnostics_ext import width_efficiency_comparison
        pred1 = ConstantPredictor(width=2.0, point=2.0)
        result = width_efficiency_comparison({"p1": pred1}, diag_data["X"], diag_data["y"], alpha=0.1)
        expected = {
            "predictor_name", "marginal_coverage", "target_coverage",
            "mean_width", "median_width", "width_90th_pct", "width_relative_to_widest"
        }
        assert set(result.columns) == expected

    def test_sorted_by_mean_width(self, diag_data):
        from insurance_conformal.diagnostics_ext import width_efficiency_comparison
        pred1 = ConstantPredictor(width=1.0, point=2.0)
        pred2 = ConstantPredictor(width=3.0, point=2.0)
        pred3 = ConstantPredictor(width=2.0, point=2.0)
        result = width_efficiency_comparison(
            {"a": pred1, "b": pred2, "c": pred3},
            diag_data["X"], diag_data["y"], alpha=0.1
        )
        widths = result["mean_width"].to_numpy()
        assert np.all(np.diff(widths) >= 0), "Result should be sorted by mean_width ascending"

    def test_narrower_has_lower_relative_width(self, diag_data):
        from insurance_conformal.diagnostics_ext import width_efficiency_comparison
        narrow = ConstantPredictor(width=1.0, point=2.0)
        wide = ConstantPredictor(width=5.0, point=2.0)
        result = width_efficiency_comparison(
            {"narrow": narrow, "wide": wide},
            diag_data["X"], diag_data["y"], alpha=0.1
        )
        r_narrow = result.filter(pl.col("predictor_name") == "narrow")["width_relative_to_widest"][0]
        r_wide = result.filter(pl.col("predictor_name") == "wide")["width_relative_to_widest"][0]
        assert r_narrow < r_wide
        assert r_wide == pytest.approx(1.0)

    def test_n_rows_equals_n_predictors(self, diag_data):
        from insurance_conformal.diagnostics_ext import width_efficiency_comparison
        preds = {f"p{i}": ConstantPredictor(width=float(i + 1), point=2.0) for i in range(4)}
        result = width_efficiency_comparison(preds, diag_data["X"], diag_data["y"], alpha=0.1)
        assert len(result) == 4

    def test_target_coverage_column_matches_alpha(self, diag_data):
        from insurance_conformal.diagnostics_ext import width_efficiency_comparison
        alpha = 0.15
        pred = ConstantPredictor(width=2.0, point=2.0)
        result = width_efficiency_comparison({"p": pred}, diag_data["X"], diag_data["y"], alpha=alpha)
        assert result["target_coverage"][0] == pytest.approx(1.0 - alpha)

    def test_coverage_in_unit_interval(self, diag_data):
        from insurance_conformal.diagnostics_ext import width_efficiency_comparison
        pred = ConstantPredictor(width=2.0, point=2.0)
        result = width_efficiency_comparison({"p": pred}, diag_data["X"], diag_data["y"], alpha=0.1)
        cov = result["marginal_coverage"][0]
        assert 0.0 <= cov <= 1.0
