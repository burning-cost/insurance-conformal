"""
Tests for PremiumSufficiencyController.

Key property to verify: E[max(Y - lambda_hat * premium, 0) / premium] <= alpha
on a fresh test set drawn from the same distribution.
"""

import numpy as np
import pytest

from insurance_conformal.risk import PremiumSufficiencyController


def make_motor_data(n: int, seed: int = 42, cap_multiplier: float = 5.0):
    """
    Synthetic motor portfolio. Premium ~ Uniform(200, 800).
    Claims ~ Gamma(shape=1.5, scale=premium * 0.6), capped at cap_multiplier * premium.
    Zero-inflation: 40% of policies have zero claims.

    Capping by policy limit is realistic and ensures B is well-defined.
    """
    rng = np.random.default_rng(seed)
    premium = rng.uniform(200, 800, size=n)
    claim_indicator = rng.binomial(1, 0.6, size=n)
    claim_severity = rng.gamma(1.5, premium * 0.6, size=n)
    # Cap claims at cap_multiplier * premium (policy limit)
    claim_severity = np.minimum(claim_severity, cap_multiplier * premium)
    y = claim_indicator * claim_severity
    return y, premium


class TestPremiumSufficiencyBasic:

    def test_calibrate_returns_self(self):
        y, prem = make_motor_data(500)
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        result = psc.calibrate(y, prem)
        assert result is psc

    def test_is_calibrated_flag(self):
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        assert not psc.is_calibrated_
        y, prem = make_motor_data(500)
        psc.calibrate(y, prem)
        assert psc.is_calibrated_

    def test_lambda_hat_positive(self):
        y, prem = make_motor_data(500)
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        psc.calibrate(y, prem)
        assert psc.lambda_hat_ > 0

    def test_lambda_hat_in_grid(self):
        grid = np.linspace(0.5, 3.0, 100)
        y, prem = make_motor_data(500)
        psc = PremiumSufficiencyController(alpha=0.05, lambda_grid=grid, B=5.0)
        psc.calibrate(y, prem)
        assert psc.lambda_hat_ in grid

    def test_predict_output_shape(self):
        y, prem = make_motor_data(500)
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        psc.calibrate(y, prem)

        prem_new = np.array([300.0, 400.0, 600.0])
        result = psc.predict(prem_new)

        assert result.shape == (3, 4)
        assert "base_premium" in result.columns
        assert "upper_bound" in result.columns
        assert "safety_loading" in result.columns
        assert "lambda_hat" in result.columns

    def test_upper_bound_equals_lambda_times_premium(self):
        y, prem = make_motor_data(500)
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        psc.calibrate(y, prem)

        prem_new = np.array([300.0, 500.0])
        result = psc.predict(prem_new)

        expected_upper = psc.lambda_hat_ * prem_new
        np.testing.assert_allclose(
            result["upper_bound"].to_numpy(), expected_upper
        )

    def test_predict_before_calibrate_raises(self):
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            psc.predict(np.array([300.0]))

    def test_shortfall_report_columns(self):
        y, prem = make_motor_data(500)
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        psc.calibrate(y, prem)

        report = psc.shortfall_report(y, prem)
        expected_cols = {"decile", "mean_premium", "n_obs", "mean_shortfall", "max_shortfall", "pct_underpriced"}
        assert expected_cols.issubset(set(report.columns))

    def test_higher_alpha_gives_smaller_lambda(self):
        """Higher risk tolerance should yield a smaller (less conservative) lambda."""
        y, prem = make_motor_data(1000, seed=7)
        grid = np.linspace(0.5, 6.0, 500)

        psc_tight = PremiumSufficiencyController(alpha=0.01, lambda_grid=grid, B=5.0)
        psc_tight.calibrate(y, prem)

        psc_loose = PremiumSufficiencyController(alpha=0.20, lambda_grid=grid, B=5.0)
        psc_loose.calibrate(y, prem)

        assert psc_loose.lambda_hat_ <= psc_tight.lambda_hat_, (
            f"Loose alpha should give smaller lambda. "
            f"Got loose={psc_loose.lambda_hat_:.4f}, tight={psc_tight.lambda_hat_:.4f}"
        )

    def test_larger_n_gives_tighter_lambda(self):
        """
        More calibration data reduces the finite-sample correction B/(n+1).
        With the same DGP, large n should give smaller or equal lambda_hat.

        We cap claims at 3x premium (B=3) so the correction is well-behaved
        and the DGP is tractable.
        """
        grid = np.linspace(0.5, 4.0, 300)
        B = 3.0
        alpha = 0.10

        y_small, prem_small = make_motor_data(100, seed=42, cap_multiplier=3.0)
        y_large, prem_large = make_motor_data(3000, seed=42, cap_multiplier=3.0)

        psc_small = PremiumSufficiencyController(alpha=alpha, lambda_grid=grid, B=B)
        psc_small.calibrate(y_small, prem_small)

        psc_large = PremiumSufficiencyController(alpha=alpha, lambda_grid=grid, B=B)
        psc_large.calibrate(y_large, prem_large)

        # Large n should give lambda <= small n (finite-sample correction is smaller)
        # Allow generous tolerance since single realisations can vary
        assert psc_large.lambda_hat_ <= psc_small.lambda_hat_ + 0.5

    def test_guarantee_holds_on_test_set(self):
        """
        Core statistical guarantee: E[shortfall at lambda_hat] <= alpha.

        CRC guarantees E[L_{n+1}(lambda_hat)] <= alpha, where the expectation
        is over the joint distribution of calibration set and test point.
        Equivalently, the mean empirical risk across many (cal, test) replicates
        should be <= alpha. Individual replicates can exceed alpha — that's
        expected due to sampling variance (not a violation of the guarantee).

        We verify the mean across 20 replicates is <= alpha + small tolerance.
        """
        alpha = 0.08
        n_cal = 1000
        n_test = 5000
        n_replicates = 20
        B = 5.0  # Claims capped at 5x premium

        empirical_risks = []
        for seed in range(n_replicates):
            y_cal, prem_cal = make_motor_data(n_cal, seed=seed, cap_multiplier=5.0)
            y_test, prem_test = make_motor_data(n_test, seed=seed + 1000, cap_multiplier=5.0)

            psc = PremiumSufficiencyController(
                alpha=alpha,
                lambda_grid=np.linspace(0.5, 6.0, 400),
                B=B,
            )
            psc.calibrate(y_cal, prem_cal)

            upper_test = psc.lambda_hat_ * prem_test
            shortfall = np.maximum(y_test - upper_test, 0.0) / prem_test
            empirical_risks.append(float(shortfall.mean()))

        mean_risk = np.mean(empirical_risks)
        # Mean across replicates should be <= alpha. Allow small tolerance for
        # finite replicates (20 replicates x 5000 test points is not infinite).
        assert mean_risk <= alpha + 0.02, (
            f"Mean empirical risk across replicates is {mean_risk:.4f}, "
            f"expected <= {alpha + 0.02:.4f}. "
            f"This suggests the CRC calibration is not working correctly."
        )


class TestPremiumSufficiencyEdgeCases:

    def test_negative_claims_raises(self):
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        with pytest.raises(ValueError, match="negative"):
            psc.calibrate(np.array([-10.0, 100.0]), np.array([200.0, 300.0]))

    def test_zero_premium_raises(self):
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        with pytest.raises(ValueError, match="strictly positive"):
            psc.calibrate(np.array([100.0, 200.0]), np.array([0.0, 300.0]))

    def test_mismatched_lengths_raises(self):
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        with pytest.raises(ValueError, match="same length"):
            psc.calibrate(np.array([100.0, 200.0]), np.array([300.0]))

    def test_predict_zero_premium_raises(self):
        y, prem = make_motor_data(200)
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        psc.calibrate(y, prem)
        with pytest.raises(ValueError, match="strictly positive"):
            psc.predict(np.array([0.0, 300.0]))

    def test_all_zero_claims(self):
        """All zero claims should give lambda_hat at the minimum of the grid."""
        y = np.zeros(500)
        prem = np.ones(500) * 400.0
        psc = PremiumSufficiencyController(alpha=0.05, lambda_grid=np.linspace(0.5, 3.0, 100), B=5.0)
        psc.calibrate(y, prem)
        # With zero shortfall, any lambda works. Should get the smallest grid value.
        assert psc.lambda_hat_ == pytest.approx(0.5, abs=0.1)

    def test_exposure_adjusts_claims(self):
        """
        With exposure=0.5, claims are annualised (divided by 0.5 = doubled).
        Annualised claims are larger, so lambda_hat should be larger.

        Use claims capped at 2x premium so B=2 is sufficient and the
        annualised version (4x) stays within B=4.
        """
        rng = np.random.default_rng(42)
        n = 500
        prem = rng.uniform(300, 600, size=n)
        # Claims bounded by 1.5x premium: y/prem <= 1.5, y/(0.5) / prem <= 3
        y = np.minimum(rng.gamma(1.2, prem * 0.4, size=n), 1.5 * prem)
        exposure = 0.5 * np.ones(n)

        # B = 3 because annualised claims are y/0.5 and max(y/0.5 - upper, 0)/prem <= 3
        psc_no_exp = PremiumSufficiencyController(
            alpha=0.10, lambda_grid=np.linspace(0.5, 4.0, 200), B=3.0
        )
        psc_no_exp.calibrate(y, prem)

        psc_exp = PremiumSufficiencyController(
            alpha=0.10, lambda_grid=np.linspace(0.5, 4.0, 200), B=3.0
        )
        psc_exp.calibrate(y, prem, exposure=exposure)

        # Annualised claims (y/0.5 = 2y) are larger, so lambda_hat should be >= original
        assert psc_exp.lambda_hat_ >= psc_no_exp.lambda_hat_ - 0.01  # Allow small tolerance

    def test_risk_summary_keys(self):
        y, prem = make_motor_data(300)
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        psc.calibrate(y, prem)
        summary = psc.risk_summary()
        expected_keys = {"lambda_hat", "alpha", "n_calibration", "B", "corrected_risk_at_lambda"}
        assert expected_keys.issubset(set(summary.keys()))

    def test_repr_before_calibrate(self):
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        assert "not calibrated" in repr(psc)

    def test_repr_after_calibrate(self):
        y, prem = make_motor_data(300)
        psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
        psc.calibrate(y, prem)
        r = repr(psc)
        assert "lambda_hat" in r
        assert "300" in r

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            PremiumSufficiencyController(alpha=0.0, B=5.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            PremiumSufficiencyController(alpha=1.5, B=5.0)

    def test_non_increasing_lambda_grid_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            PremiumSufficiencyController(alpha=0.05, lambda_grid=np.array([3.0, 2.0, 1.0]), B=5.0)
