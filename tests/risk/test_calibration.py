"""
Tests for the core CRC calibration algorithm.
"""

import numpy as np
import pytest

from insurance_conformal.risk.calibration import (
    conformal_risk_calibration,
    MonotoneLambdaSearch,
)


class TestConformalRiskCalibration:
    """Tests for conformal_risk_calibration()."""

    def test_basic_guarantee_holds(self):
        """
        Empirical guarantee: when we run CRC many times and check whether
        the population expected loss at lambda_hat is <= alpha, it should
        hold with frequency >= 1 - delta (here we use delta=0 since the
        guarantee is unconditional under exchangeability).
        """
        rng = np.random.default_rng(42)
        n = 500
        m = 200
        alpha = 0.10
        B = 1.0

        # True DGP: loss decreases monotonically in lambda
        # losses[i, j] = max(0, 1 - lambda_j) * U_i where U_i ~ Uniform(0, 1)
        # Population E[L(lambda)] = max(0, 1 - lambda) / 2
        lambdas = np.linspace(0, 1.5, m)
        u = rng.uniform(0, 1, size=n)
        losses = np.outer(u, np.maximum(0, 1 - lambdas))  # shape (n, m)

        lambda_hat, idx, risk_curve = conformal_risk_calibration(losses, lambdas, alpha=alpha, B=B)

        # True population risk at lambda_hat
        true_risk = max(0, 1 - lambda_hat) / 2
        assert true_risk <= alpha + B / (n + 1) + 0.01, (
            f"Population risk {true_risk:.4f} exceeds alpha={alpha} by more than tolerance"
        )
        assert lambda_hat >= 0
        assert len(risk_curve) == m

    def test_finite_sample_correction_matters(self):
        """
        With n=10, the correction B/(n+1) is ~0.09 for B=1.
        A naive implementation (no correction) would find a smaller lambda.
        """
        n = 10
        m = 100
        alpha = 0.10
        B = 1.0

        rng = np.random.default_rng(99)
        lambdas = np.linspace(0, 2, m)
        u = rng.uniform(0, 1, size=n)
        losses = np.outer(u, np.maximum(0, 1.5 - lambdas))

        lambda_hat, idx, risk_curve = conformal_risk_calibration(losses, lambdas, alpha=alpha, B=B)

        # The corrected risk at lambda_hat must be <= alpha
        assert risk_curve[idx] <= alpha + 1e-9

        # Verify correction: corrected = (n/(n+1)) * empirical + B/(n+1)
        empirical = losses.mean(axis=0)
        expected_corrected = (n / (n + 1)) * empirical + B / (n + 1)
        np.testing.assert_allclose(risk_curve, expected_corrected, rtol=1e-10)

    def test_returns_smallest_valid_lambda(self):
        """lambda_hat should be the SMALLEST lambda with corrected risk <= alpha."""
        n = 1000
        m = 100
        alpha = 0.15
        B = 1.0

        lambdas = np.linspace(0, 1, m)
        rng = np.random.default_rng(7)
        u = rng.uniform(0, 1, size=n)
        losses = np.outer(u, np.maximum(0, 1 - lambdas))

        lambda_hat, idx, risk_curve = conformal_risk_calibration(losses, lambdas, alpha=alpha, B=B)

        # All lambdas before idx should have corrected risk > alpha
        if idx > 0:
            assert np.all(risk_curve[:idx] > alpha)

    def test_alpha_too_small_raises(self):
        """When alpha is smaller than minimum achievable risk, raise RuntimeError."""
        n = 50
        m = 50
        lambdas = np.linspace(0, 1, m)
        rng = np.random.default_rng(123)
        # Loss never drops below 0.3
        losses = 0.3 + 0.1 * rng.uniform(0, 1, size=(n, m))

        with pytest.raises(RuntimeError, match="No lambda"):
            conformal_risk_calibration(losses, lambdas, alpha=0.01, B=1.0)

    def test_alpha_out_of_range_raises(self):
        lambdas = np.linspace(0, 1, 10)
        losses = np.zeros((20, 10))
        with pytest.raises(ValueError, match="alpha must be in"):
            conformal_risk_calibration(losses, lambdas, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            conformal_risk_calibration(losses, lambdas, alpha=1.5)

    def test_shape_mismatch_raises(self):
        lambdas = np.linspace(0, 1, 10)
        losses = np.zeros((20, 7))  # Wrong: 7 != 10
        with pytest.raises(ValueError):
            conformal_risk_calibration(losses, lambdas, alpha=0.1)

    def test_1d_losses_single_lambda(self):
        """1D losses should work when lambdas has exactly 1 element."""
        n = 100
        lambdas = np.array([1.0])
        losses = np.zeros(n)  # All zero loss
        lambda_hat, idx, risk_curve = conformal_risk_calibration(losses, lambdas, alpha=0.05)
        assert lambda_hat == 1.0
        assert idx == 0

    def test_large_B(self):
        """Custom B should work correctly."""
        n = 200
        m = 50
        lambdas = np.linspace(0, 10, m)
        B = 5.0
        rng = np.random.default_rng(55)
        losses = B * rng.uniform(0, 1, size=(n, m)) * np.maximum(0, 1 - lambdas / 10)

        lambda_hat, idx, risk_curve = conformal_risk_calibration(
            losses, lambdas, alpha=0.2, B=B
        )
        assert risk_curve[idx] <= 0.2 + 1e-9

    def test_risk_curve_monotone(self):
        """For a truly monotone loss function, risk curve should be non-increasing."""
        n = 500
        m = 100
        lambdas = np.linspace(0, 2, m)
        rng = np.random.default_rng(11)
        u = rng.uniform(0, 1, size=n)
        # Strictly monotone: loss = max(0, 1 - lambda) * u
        losses = np.outer(u, np.maximum(0, 1 - lambdas))

        _, _, risk_curve = conformal_risk_calibration(losses, lambdas, alpha=0.1)
        diffs = np.diff(risk_curve)
        assert np.all(diffs <= 1e-10), "Risk curve should be non-increasing for monotone loss"


class TestMonotoneLambdaSearch:
    """Tests for MonotoneLambdaSearch."""

    def test_compute_losses_shape(self):
        def my_loss(y, lam, base):
            return np.maximum(y - lam * base, 0) / base

        n = 100
        m = 50
        search = MonotoneLambdaSearch(
            loss_fn=my_loss,
            lambda_grid=np.linspace(0.5, 3.0, m),
        )
        y = np.random.default_rng(42).gamma(2, 500, size=n)
        base = np.ones(n) * 600

        losses = search.compute_losses(y, base=base)
        assert losses.shape == (n, m)

    def test_calibrate_returns_tuple(self):
        def my_loss(y, lam, base):
            return np.maximum(y - lam * base, 0) / base

        rng = np.random.default_rng(7)
        n = 300
        y = rng.gamma(2, 500, size=n)
        base = np.ones(n) * 600

        search = MonotoneLambdaSearch(
            loss_fn=my_loss,
            lambda_grid=np.linspace(0.1, 5.0, 200),
        )
        lambda_hat, risk_curve = search.calibrate(y, alpha=0.10, B=1.0, base=base)

        assert isinstance(lambda_hat, float)
        assert lambda_hat > 0
        assert len(risk_curve) == 200

    def test_losses_monotone_in_lambda(self):
        """For the scaled shortfall loss, losses should decrease as lambda increases."""
        def shortfall(y, lam, base):
            return np.maximum(y - lam * base, 0) / base

        rng = np.random.default_rng(99)
        n = 200
        y = rng.gamma(3, 400, size=n)
        base = np.ones(n) * 500

        search = MonotoneLambdaSearch(
            loss_fn=shortfall,
            lambda_grid=np.linspace(0.5, 4.0, 100),
        )
        losses = search.compute_losses(y, base=base)
        # Mean loss should be non-increasing
        mean_loss = losses.mean(axis=0)
        assert np.all(np.diff(mean_loss) <= 1e-10)
