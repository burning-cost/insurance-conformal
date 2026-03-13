"""
Tests for nonconformity score functions.
"""

import numpy as np
import pytest

from insurance_conformal.multivariate.scores import (
    absolute_residual_score,
    normalized_residual_score,
    poisson_deviance_score,
    gamma_deviance_score,
    tweedie_deviance_score,
)


class TestAbsoluteResidualScore:
    def test_basic_scalar(self):
        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([1.5, 1.5, 3.5])
        r = absolute_residual_score(y, y_hat)
        np.testing.assert_allclose(r, [0.5, 0.5, 0.5])

    def test_zero_residuals(self):
        y = np.array([1.0, 2.0])
        r = absolute_residual_score(y, y)
        np.testing.assert_allclose(r, [0.0, 0.0])

    def test_2d_input(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_hat = np.array([[0.5, 2.5], [3.5, 3.5]])
        r = absolute_residual_score(y, y_hat)
        assert r.shape == (2, 2)
        np.testing.assert_allclose(r, [[0.5, 0.5], [0.5, 0.5]])

    def test_always_nonnegative(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 100)
        y_hat = rng.normal(0, 1, 100)
        r = absolute_residual_score(y, y_hat)
        assert np.all(r >= 0)

    def test_symmetry(self):
        y = np.array([1.0, 2.0])
        y_hat = np.array([2.0, 1.0])
        r1 = absolute_residual_score(y, y_hat)
        r2 = absolute_residual_score(y_hat, y)
        np.testing.assert_allclose(r1, r2)


class TestNormalizedResidualScore:
    def test_basic(self):
        y = np.array([1.0, 3.0])
        y_hat = np.array([0.0, 1.0])
        sigma = np.array([1.0, 2.0])
        r = normalized_residual_score(y, y_hat, sigma)
        np.testing.assert_allclose(r, [1.0, 1.0])

    def test_negative_sigma_raises(self):
        y = np.array([1.0])
        y_hat = np.array([0.0])
        sigma = np.array([-1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            normalized_residual_score(y, y_hat, sigma)

    def test_zero_sigma_raises(self):
        y = np.array([1.0])
        y_hat = np.array([1.0])
        sigma = np.array([0.0])
        with pytest.raises(ValueError, match="strictly positive"):
            normalized_residual_score(y, y_hat, sigma)

    def test_larger_sigma_smaller_score(self):
        y = np.array([2.0])
        y_hat = np.array([1.0])
        r_small = normalized_residual_score(y, y_hat, np.array([0.5]))
        r_large = normalized_residual_score(y, y_hat, np.array([2.0]))
        assert r_small[0] > r_large[0]


class TestPoissonDevianceScore:
    def test_zero_claim_formula(self):
        # y=0: deviance = 2 * mu
        y = np.array([0.0])
        mu = np.array([0.1])
        r = poisson_deviance_score(y, mu)
        np.testing.assert_allclose(r, [2 * 0.1], rtol=1e-6)

    def test_perfect_prediction(self):
        # y == mu: deviance = 0
        y = np.array([2.0, 3.0])
        r = poisson_deviance_score(y, y)
        np.testing.assert_allclose(r, [0.0, 0.0], atol=1e-10)

    def test_always_nonnegative(self):
        rng = np.random.default_rng(1)
        y = rng.poisson(0.2, 200).astype(float)
        y_hat = rng.uniform(0.05, 0.5, 200)
        r = poisson_deviance_score(y, y_hat)
        assert np.all(r >= 0)

    def test_larger_miss_larger_score(self):
        y = np.array([1.0])
        r_small = poisson_deviance_score(y, np.array([0.9]))
        r_large = poisson_deviance_score(y, np.array([0.1]))
        assert r_large[0] > r_small[0]

    def test_clip_handles_near_zero_pred(self):
        # Should not raise even with tiny predictions
        y = np.array([0.0, 1.0])
        y_hat = np.array([1e-10, 1e-10])
        r = poisson_deviance_score(y, y_hat)
        assert np.all(np.isfinite(r))


class TestGammaDevianceScore:
    def test_perfect_prediction(self):
        y = np.array([1000.0, 2000.0])
        r = gamma_deviance_score(y, y)
        np.testing.assert_allclose(r, [0.0, 0.0], atol=1e-10)

    def test_always_nonnegative(self):
        rng = np.random.default_rng(2)
        y = rng.gamma(2.0, 1000.0, 200)
        y_hat = rng.gamma(2.0, 900.0, 200)
        r = gamma_deviance_score(y, y_hat)
        assert np.all(r >= 0)

    def test_symmetric_ish(self):
        # Gamma deviance is not perfectly symmetric but roughly so
        y = np.array([2000.0])
        r1 = gamma_deviance_score(y, np.array([2200.0]))
        r2 = gamma_deviance_score(y, np.array([1800.0]))
        # Both should be small positive numbers
        assert r1[0] > 0 and r2[0] > 0

    def test_scale_invariance(self):
        # Gamma deviance is scale-invariant (relative errors)
        y1 = np.array([1000.0])
        y2 = np.array([2000.0])
        r1 = gamma_deviance_score(y1, np.array([1200.0]))
        r2 = gamma_deviance_score(y2, np.array([2400.0]))
        # Same relative error: 20%, so deviance should be the same
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestTweedieDevianceScore:
    def test_p1_matches_poisson(self):
        y = np.array([0.0, 1.0, 2.0])
        y_hat = np.array([0.1, 0.2, 1.5])
        r_tweedie = tweedie_deviance_score(y, y_hat, p=1)
        r_poisson = poisson_deviance_score(y, y_hat)
        np.testing.assert_allclose(r_tweedie, r_poisson, rtol=1e-6)

    def test_p2_matches_gamma(self):
        y = np.array([1000.0, 2000.0])
        y_hat = np.array([900.0, 2200.0])
        r_tweedie = tweedie_deviance_score(y, y_hat, p=2)
        r_gamma = gamma_deviance_score(y, y_hat)
        np.testing.assert_allclose(r_tweedie, r_gamma, rtol=1e-5)

    def test_p0_matches_mse(self):
        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([1.5, 1.5, 2.5])
        r = tweedie_deviance_score(y, y_hat, p=0)
        expected = (y - y_hat) ** 2
        np.testing.assert_allclose(r, expected, rtol=1e-10)

    def test_compound_poisson_p15(self):
        # p=1.5 is the classic compound Poisson-Gamma
        rng = np.random.default_rng(3)
        y = rng.gamma(1.5, 100.0, 50)
        y_hat = rng.gamma(1.5, 90.0, 50)
        r = tweedie_deviance_score(y, y_hat, p=1.5)
        assert np.all(r >= 0)
        assert np.all(np.isfinite(r))

    def test_always_nonnegative(self):
        rng = np.random.default_rng(4)
        y = rng.uniform(0.1, 10.0, 100)
        y_hat = rng.uniform(0.1, 10.0, 100)
        for p in [0.5, 1.0, 1.5, 2.0, 2.5]:
            r = tweedie_deviance_score(y, y_hat, p=p)
            assert np.all(r >= 0), f"Negative deviance for p={p}"
