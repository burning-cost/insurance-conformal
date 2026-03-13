"""
Tests for insurance-specific loss functions.
"""

import numpy as np
import pytest

from insurance_conformal.risk.losses import (
    shortfall_loss,
    scaled_shortfall_loss,
    coverage_loss,
    interval_width_loss,
    xl_recovery_loss,
    exposure_weighted_mean,
)


class TestShortfallLoss:
    def test_zero_when_claim_below_upper(self):
        y = np.array([100.0, 200.0, 50.0])
        upper = np.array([150.0, 200.0, 100.0])
        premium = np.array([200.0, 250.0, 150.0])
        result = shortfall_loss(y, upper, premium)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_positive_when_claim_exceeds_upper(self):
        y = np.array([300.0])
        upper = np.array([200.0])
        premium = np.array([250.0])
        result = shortfall_loss(y, upper, premium)
        expected = (300 - 200) / 250
        np.testing.assert_allclose(result, [expected])

    def test_normalised_by_premium_not_upper(self):
        y = np.array([500.0])
        upper = np.array([300.0])
        premium = np.array([400.0])
        result = shortfall_loss(y, upper, premium)
        # (500 - 300) / 400 = 0.5
        np.testing.assert_allclose(result, [0.5])

    def test_non_positive_premium_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            shortfall_loss(np.array([100.0]), np.array([80.0]), np.array([0.0]))

    def test_mixed_claims(self):
        y = np.array([100.0, 600.0, 200.0])
        upper = np.array([200.0, 500.0, 200.0])
        premium = np.array([200.0, 500.0, 200.0])
        result = shortfall_loss(y, upper, premium)
        expected = np.array([0.0, (600 - 500) / 500, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        n = 1000
        y = rng.gamma(2, 500, size=n)
        upper = rng.gamma(2, 400, size=n)
        premium = rng.uniform(300, 800, size=n)
        result = shortfall_loss(y, upper, premium)
        assert np.all(result >= 0)


class TestScaledShortfallLoss:
    def test_lambda_1_equals_shortfall_when_premium_equals_upper(self):
        """At lambda=1, upper = 1 * base_premium = base_premium."""
        rng = np.random.default_rng(7)
        y = rng.gamma(2, 500, size=100)
        base_premium = rng.uniform(300, 700, size=100)

        result = scaled_shortfall_loss(y, 1.0, base_premium)
        expected = shortfall_loss(y, base_premium, base_premium)
        np.testing.assert_allclose(result, expected)

    def test_large_lambda_gives_zero_loss(self):
        """Very large lambda means all claims are covered."""
        y = np.array([1000.0, 2000.0, 500.0])
        base = np.array([500.0, 800.0, 300.0])
        result = scaled_shortfall_loss(y, 100.0, base)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_monotone_in_lambda(self):
        """Loss must be non-increasing in lambda."""
        rng = np.random.default_rng(13)
        y = rng.gamma(3, 400, size=200)
        base = rng.uniform(200, 600, size=200)
        lambdas = np.linspace(0.5, 4.0, 50)
        losses = np.array([scaled_shortfall_loss(y, lam, base).mean() for lam in lambdas])
        assert np.all(np.diff(losses) <= 1e-10)

    def test_bounded_by_B(self):
        """Loss should be at most max(y) / min(premium) in absolute terms."""
        y = np.array([500.0])
        base = np.array([100.0])
        result = scaled_shortfall_loss(y, 0.0, base)
        # max shortfall = 500 / 100 = 5.0
        assert result[0] <= 6.0  # generous bound


class TestCoverageLoss:
    def test_zero_when_inside(self):
        y = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.0])
        upper = np.array([1.5, 2.5, 4.0])
        result = coverage_loss(y, lower, upper)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_one_when_outside(self):
        y = np.array([0.0, 5.0])
        lower = np.array([1.0, 1.0])
        upper = np.array([3.0, 3.0])
        result = coverage_loss(y, lower, upper)
        np.testing.assert_array_equal(result, np.ones(2))

    def test_boundary_included(self):
        y = np.array([1.0, 3.0])  # Exactly at boundary
        lower = np.array([1.0, 1.0])
        upper = np.array([3.0, 3.0])
        result = coverage_loss(y, lower, upper)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_values_are_0_or_1(self):
        rng = np.random.default_rng(55)
        n = 500
        y = rng.normal(0, 1, size=n)
        lower = -1.0 * np.ones(n)
        upper = 1.0 * np.ones(n)
        result = coverage_loss(y, lower, upper)
        assert set(result.tolist()).issubset({0.0, 1.0})


class TestIntervalWidthLoss:
    def test_basic_width(self):
        lower = np.array([0.0, 10.0])
        upper = np.array([5.0, 15.0])
        result = interval_width_loss(lower, upper, scale=1.0)
        np.testing.assert_allclose(result, [5.0, 5.0])

    def test_normalised_by_scale(self):
        lower = np.array([0.0])
        upper = np.array([1000.0])
        result = interval_width_loss(lower, upper, scale=2000.0)
        np.testing.assert_allclose(result, [0.5])

    def test_zero_width_intervals(self):
        lower = np.array([5.0, 10.0])
        upper = np.array([5.0, 10.0])
        result = interval_width_loss(lower, upper)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_non_positive_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            interval_width_loss(np.array([0.0]), np.array([1.0]), scale=0.0)


class TestXLRecoveryLoss:
    def test_no_recovery_below_excess(self):
        """Claims below excess have zero actual recovery."""
        y = np.array([500.0, 800.0])
        excess = 1000.0
        limit = 5000.0
        estimated = np.array([100.0, 200.0])
        result = xl_recovery_loss(y, excess, limit, estimated)
        # Actual recovery = 0 for both; estimated > 0, so loss = max(0 - est, 0) = 0
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_recovery_above_excess(self):
        """Standard XL layer: claim = 3000, excess = 1000, limit = 5000."""
        y = np.array([3000.0])
        excess = 1000.0
        limit = 5000.0
        actual_recovery = 2000.0  # min(max(3000 - 1000, 0), 5000)
        estimated = np.array([1500.0])
        result = xl_recovery_loss(y, excess, limit, estimated)
        expected = (actual_recovery - 1500.0) / 5000.0
        np.testing.assert_allclose(result, [expected])

    def test_recovery_capped_at_limit(self):
        """Claim well above limit: recovery = limit."""
        y = np.array([50000.0])
        excess = 1000.0
        limit = 5000.0
        estimated = np.array([4000.0])
        result = xl_recovery_loss(y, excess, limit, estimated)
        # actual = 5000, estimated = 4000, loss = (5000 - 4000) / 5000 = 0.2
        np.testing.assert_allclose(result, [0.2])

    def test_bounded_by_1(self):
        """Loss normalised by limit, so always <= 1."""
        rng = np.random.default_rng(33)
        y = rng.uniform(0, 20000, size=500)
        estimated = rng.uniform(0, 5000, size=500)
        result = xl_recovery_loss(y, excess=2000, limit=5000, estimated_recovery=estimated)
        assert np.all(result <= 1.0)
        assert np.all(result >= 0.0)

    def test_negative_excess_raises(self):
        with pytest.raises(ValueError, match="excess"):
            xl_recovery_loss(np.array([1000.0]), excess=-100, limit=5000, estimated_recovery=np.array([200.0]))

    def test_zero_limit_raises(self):
        with pytest.raises(ValueError, match="limit"):
            xl_recovery_loss(np.array([1000.0]), excess=500, limit=0.0, estimated_recovery=np.array([200.0]))


class TestExposureWeightedMean:
    def test_equal_exposure_equals_unweighted_mean(self):
        losses = np.array([1.0, 2.0, 3.0])
        exposure = np.ones(3)
        result = exposure_weighted_mean(losses, exposure)
        np.testing.assert_allclose(result, 2.0)

    def test_half_exposure_counts_half(self):
        losses = np.array([0.0, 1.0])
        exposure = np.array([1.0, 0.5])
        # weighted: (0 * 1 + 1 * 0.5) / (1 + 0.5) = 0.5 / 1.5 = 1/3
        result = exposure_weighted_mean(losses, exposure)
        np.testing.assert_allclose(result, 1 / 3)

    def test_zero_exposure_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            exposure_weighted_mean(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
