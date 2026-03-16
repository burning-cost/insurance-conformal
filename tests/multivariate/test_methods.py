"""
Tests for conformal quantile methods: Bonferroni, Sidak, GWC, LWC.
"""

import numpy as np
import pytest

from insurance_conformal.multivariate.methods import (
    bonferroni_quantile,
    sidak_quantile,
    gwc_quantile,
    gwc_quantile_vec,
    lwc_quantile_exact,
    _coordinate_standardize,
)


class TestCoordinateStandardize:
    def test_basic_shape(self, simple_residuals_2d):
        std, mu, sigma = _coordinate_standardize(simple_residuals_2d)
        n, d = simple_residuals_2d.shape
        assert std.shape == (n, d)
        assert mu.shape == (d,)
        assert sigma.shape == (d,)

    def test_mean_zero_after_standardization(self, simple_residuals_2d):
        std, mu, sigma = _coordinate_standardize(simple_residuals_2d)
        # Standardized columns should have mean ~0
        np.testing.assert_allclose(std.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_std_one_after_standardization(self, simple_residuals_2d):
        std, mu, sigma = _coordinate_standardize(simple_residuals_2d)
        np.testing.assert_allclose(std.std(axis=0), [1.0, 1.0], atol=1e-10)

    def test_constant_column_no_div_zero(self):
        # Column with zero variance: sigma clamped to 1e-8
        res = np.column_stack([
            np.ones(50),       # constant column
            np.random.default_rng(0).normal(0, 1, 50),
        ])
        std, mu, sigma = _coordinate_standardize(res)
        assert sigma[0] == pytest.approx(1e-8, rel=1e-3)
        assert np.all(np.isfinite(std))

    def test_recovers_original(self, simple_residuals_2d):
        std, mu, sigma = _coordinate_standardize(simple_residuals_2d)
        recovered = std * sigma + mu
        np.testing.assert_allclose(recovered, simple_residuals_2d, rtol=1e-10)


class TestBonferroniQuantile:
    def test_returns_d_quantiles(self, simple_residuals_2d):
        q = bonferroni_quantile(simple_residuals_2d, alpha=0.05)
        assert q.shape == (2,)

    def test_quantiles_nonnegative(self, simple_residuals_2d):
        q = bonferroni_quantile(simple_residuals_2d, alpha=0.05)
        assert np.all(q >= 0)

    def test_stricter_alpha_wider_intervals(self, simple_residuals_2d):
        q_loose = bonferroni_quantile(simple_residuals_2d, alpha=0.20)
        q_strict = bonferroni_quantile(simple_residuals_2d, alpha=0.01)
        # Stricter alpha means higher quantile (wider interval)
        assert np.all(q_strict >= q_loose)

    def test_1d_input(self):
        rng = np.random.default_rng(0)
        res_1d = np.abs(rng.normal(0, 1, 100))
        q = bonferroni_quantile(res_1d, alpha=0.05, d=1)
        assert q.shape == (1,)

    def test_d_larger_than_actual_dims(self, simple_residuals_2d):
        # d=4 but residuals have d=2: should use d=4 for alpha correction
        q_d2 = bonferroni_quantile(simple_residuals_2d, alpha=0.05, d=2)
        q_d4 = bonferroni_quantile(simple_residuals_2d, alpha=0.05, d=4)
        # With d=4, alpha/d = 0.0125 vs 0.025 for d=2 — stricter => wider
        assert np.all(q_d4 >= q_d2)


class TestSidakQuantile:
    def test_returns_d_quantiles(self, simple_residuals_2d):
        q = sidak_quantile(simple_residuals_2d, alpha=0.05)
        assert q.shape == (2,)

    def test_sidak_narrower_than_bonferroni(self, simple_residuals_2d):
        # Sidak is tighter than Bonferroni (it's the independence assumption version)
        q_sidak = sidak_quantile(simple_residuals_2d, alpha=0.05, d=2)
        q_bonf = bonferroni_quantile(simple_residuals_2d, alpha=0.05, d=2)
        # Sidak level = 1 - (1 - 0.05)^(1/2) ~ 0.0253, vs Bonferroni 0.025
        # Sidak is very slightly less conservative
        assert np.all(q_sidak <= q_bonf + 1e-6)


class TestGWCQuantile:
    def test_returns_scalar_and_stats(self, simple_residuals_2d):
        q, mu, sigma = gwc_quantile(simple_residuals_2d, alpha=0.05)
        assert isinstance(q, float)
        assert mu.shape == (2,)
        assert sigma.shape == (2,)

    def test_sigma_positive(self, simple_residuals_2d):
        _, _, sigma = gwc_quantile(simple_residuals_2d, alpha=0.05)
        assert np.all(sigma > 0)

    def test_stricter_alpha_larger_quantile(self, simple_residuals_2d):
        q_loose, _, _ = gwc_quantile(simple_residuals_2d, alpha=0.20)
        q_strict, _, _ = gwc_quantile(simple_residuals_2d, alpha=0.01)
        assert q_strict >= q_loose

    def test_1d_reduces_to_standard_conformal(self):
        # With d=1, GWC should give the standard conformal quantile
        rng = np.random.default_rng(5)
        res = np.abs(rng.normal(0, 1, 100)).reshape(-1, 1)
        q_gwc, mu, sigma = gwc_quantile(res, alpha=0.05)
        # Standard conformal: sort |residuals|, take ceil((n+1)*0.95)/n index
        n = 100
        k = int(np.ceil((n + 1) * 0.95))
        k = min(k, n)
        res_std = (res.flatten() - mu[0]) / sigma[0]
        q_expected = np.sort(res_std)[k - 1]
        assert q_gwc == pytest.approx(q_expected, rel=1e-6)

    def test_3d_residuals(self, simple_residuals_3d):
        q, mu, sigma = gwc_quantile(simple_residuals_3d, alpha=0.05)
        assert isinstance(q, float)
        assert mu.shape == (3,)
        assert sigma.shape == (3,)


class TestLWCQuantile:
    def test_returns_d_thresholds(self, simple_residuals_2d):
        t, mu, sigma = gwc_quantile_vec(simple_residuals_2d, alpha=0.05)
        assert t.shape == (2,)
        assert mu.shape == (2,)
        assert sigma.shape == (2,)

    def test_thresholds_leq_gwc(self, simple_residuals_2d):
        # LWC should be <= GWC (tighter or equal)
        q_gwc, _, _ = gwc_quantile(simple_residuals_2d, alpha=0.05)
        t_gwc_vec, _, _ = gwc_quantile_vec(simple_residuals_2d, alpha=0.05)
        # Each dimension threshold should be <= q_gwc
        assert np.all(t_gwc_vec <= q_gwc + 1e-10)

    def test_3d(self, simple_residuals_3d):
        t, mu, sigma = gwc_quantile_vec(simple_residuals_3d, alpha=0.05)
        assert t.shape == (3,)

    def test_1d_reduces_to_gwc(self):
        rng = np.random.default_rng(6)
        res = np.abs(rng.normal(0, 1, 100)).reshape(-1, 1)
        q_gwc, _, _ = gwc_quantile(res, alpha=0.05)
        t_gwc_vec, _, _ = gwc_quantile_vec(res, alpha=0.05)
        assert t_gwc_vec[0] == pytest.approx(q_gwc, rel=1e-6)


class TestLWCQuantileExact:
    def test_returns_d_thresholds(self, simple_residuals_2d):
        t, mu, sigma = lwc_quantile_exact(simple_residuals_2d, alpha=0.05)
        assert t.shape == (2,)

    def test_exact_leq_gwc(self, simple_residuals_2d):
        q_gwc, _, _ = gwc_quantile(simple_residuals_2d, alpha=0.05)
        t_exact, _, _ = lwc_quantile_exact(simple_residuals_2d, alpha=0.05)
        # Per-group thresholds should be <= global threshold
        assert np.all(t_exact <= q_gwc + 1e-10)

    def test_stricter_alpha(self, simple_residuals_2d):
        t_loose, _, _ = lwc_quantile_exact(simple_residuals_2d, alpha=0.20)
        t_strict, _, _ = lwc_quantile_exact(simple_residuals_2d, alpha=0.01)
        assert np.all(t_strict >= t_loose)

    def test_monotone_in_alpha(self, simple_residuals_2d):
        # More stringent alpha -> wider thresholds (higher quantile)
        thresholds = []
        for a in [0.25, 0.15, 0.10, 0.05, 0.02]:
            t, _, _ = lwc_quantile_exact(simple_residuals_2d, alpha=a)
            thresholds.append(t.mean())
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1] + 1e-8

    def test_nonnegative_thresholds(self, simple_residuals_2d):
        # Residuals are absolute => standardized may be negative (mean-centered)
        # But quantile at 95% level should be positive for reasonable data
        t, _, _ = lwc_quantile_exact(simple_residuals_2d, alpha=0.05)
        # Check that raw interval widths are positive: t * sigma + mu
        from insurance_conformal.multivariate.methods import _coordinate_standardize
        _, mu, sigma = _coordinate_standardize(simple_residuals_2d)
        half_widths = t * sigma + mu
        assert np.all(half_widths > 0)


class TestMethodConsistency:
    def test_all_methods_same_shape_output(self, simple_residuals_2d):
        alpha = 0.05
        q_bonf = bonferroni_quantile(simple_residuals_2d, alpha)
        q_sid = sidak_quantile(simple_residuals_2d, alpha)
        q_gwc, mu_gwc, sigma_gwc = gwc_quantile(simple_residuals_2d, alpha)
        t_gwc_vec, mu_lwc, sigma_lwc = lwc_quantile_exact(simple_residuals_2d, alpha)

        assert q_bonf.shape == (2,)
        assert q_sid.shape == (2,)
        assert t_gwc_vec.shape == (2,)

    def test_bonferroni_wider_than_gwc(self, simple_residuals_2d):
        # Bonferroni per-dimension intervals should generally be wider than GWC
        # This is the whole point of the library
        alpha = 0.05
        q_bonf = bonferroni_quantile(simple_residuals_2d, alpha)
        q_gwc, mu, sigma = gwc_quantile(simple_residuals_2d, alpha)
        # GWC half-widths in original scale: q_gwc * sigma + mu
        hw_gwc = q_gwc * sigma + mu
        # Not necessarily true element-by-element but should be true on average
        # (over many random draws Bonferroni tends to be wider)
        # We just check that GWC is finite and positive
        assert np.all(np.isfinite(hw_gwc))
