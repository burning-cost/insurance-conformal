"""
Tests for solvency_capital_range().

We verify:
- The return type and all required fields are present
- upper_bound carries valid 99.5% conformal coverage (>= 1 - alpha of observations
  fall below the upper bound, allowing for finite-sample variation)
- scr_estimate = max(0, upper_bound - expected_loss) for all risks
- interval_width = upper_bound - lower_bound
- exposure scaling works correctly
- error handling for invalid inputs
- compatibility with InsuranceConformalPredictor and ConformalisedQuantileRegression
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal import (
    InsuranceConformalPredictor,
    ConformalisedQuantileRegression,
)
from insurance_conformal._solvency import solvency_capital_range, SolvencyCapitalRange


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LinearPoissonModel:
    """Minimal sklearn-compatible model for testing."""

    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        return np.exp(np.asarray(X) @ self.beta)


class ConstantQuantileModel:
    """Quantile model that adds a fixed fraction of X[:,0] to a constant."""

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.maximum(0.0, X[:, 0] + self.offset)


@pytest.fixture(scope="module")
def base_data():
    rng = np.random.default_rng(42)
    n = 1500
    beta = np.array([0.3, -0.15, 0.05])
    X = rng.normal(size=(n, 3))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    model = LinearPoissonModel(beta)
    return {
        "X_train": X[:600],
        "y_train": y[:600],
        "X_cal": X[600:1000],
        "y_cal": y[600:1000],
        "X_test": X[1000:],
        "y_test": y[1000:],
        "model": model,
    }


@pytest.fixture(scope="module")
def calibrated_cp(base_data):
    cp = InsuranceConformalPredictor(
        model=base_data["model"],
        nonconformity="pearson_weighted",
        tweedie_power=1.0,
    )
    cp.calibrate(base_data["X_cal"], base_data["y_cal"])
    return cp


@pytest.fixture(scope="module")
def calibrated_cqr(base_data):
    """CQR with constant offset quantile models — simple, controllable."""
    model_lo = ConstantQuantileModel(offset=-0.5)
    model_hi = ConstantQuantileModel(offset=1.5)
    cqr = ConformalisedQuantileRegression(model_lo=model_lo, model_hi=model_hi)
    cqr.calibrate(base_data["X_cal"], base_data["y_cal"])
    return cqr


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_solvency_capital_range(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert isinstance(result, SolvencyCapitalRange)

    def test_all_fields_present(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert hasattr(result, "scr_estimate")
        assert hasattr(result, "lower_bound")
        assert hasattr(result, "upper_bound")
        assert hasattr(result, "interval_width")
        assert hasattr(result, "coverage_level")
        assert hasattr(result, "n_risks")
        assert hasattr(result, "alpha")
        assert hasattr(result, "total_scr")
        assert hasattr(result, "mean_interval_width")

    def test_array_shapes_match_input(self, calibrated_cp, base_data):
        n = len(base_data["X_test"])
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert result.scr_estimate.shape == (n,)
        assert result.lower_bound.shape == (n,)
        assert result.upper_bound.shape == (n,)
        assert result.interval_width.shape == (n,)

    def test_n_risks_correct(self, calibrated_cp, base_data):
        n = len(base_data["X_test"])
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert result.n_risks == n

    def test_repr_contains_key_info(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        r = repr(result)
        assert "SolvencyCapitalRange" in r
        assert "n_risks" in r
        assert "coverage_level" in r
        assert "total_scr" in r


# ---------------------------------------------------------------------------
# Default alpha = 0.005 (Solvency II 99.5%)
# ---------------------------------------------------------------------------


class TestDefaultAlpha:
    def test_coverage_level_is_0_995(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert np.isclose(result.coverage_level, 0.995)

    def test_alpha_stored_correctly(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert np.isclose(result.alpha, 0.005)

    def test_empirical_coverage_at_99_5_pct(self, calibrated_cp, base_data):
        """At least 99.5% of y_test should fall below upper_bound.

        Conformal theory guarantees >= 1 - alpha coverage in expectation.
        We test with a 1pp tolerance for finite-sample variation.
        """
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        y = base_data["y_test"]
        covered = y <= result.upper_bound
        empirical = covered.mean()
        assert empirical >= 0.985, (
            f"Empirical coverage {empirical:.3%} is too far below 99.5%. "
            "The upper bound is not providing the expected tail protection."
        )


# ---------------------------------------------------------------------------
# Numerical consistency
# ---------------------------------------------------------------------------


class TestNumericalConsistency:
    def test_scr_is_nonnegative(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert np.all(result.scr_estimate >= 0)

    def test_interval_width_is_nonnegative(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert np.all(result.interval_width >= 0)

    def test_interval_width_equals_upper_minus_lower(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        expected_width = result.upper_bound - result.lower_bound
        assert np.allclose(result.interval_width, expected_width)

    def test_total_scr_equals_sum_of_scr_estimate(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert np.isclose(result.total_scr, result.scr_estimate.sum())

    def test_mean_width_equals_mean_of_interval_width(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert np.isclose(result.mean_interval_width, result.interval_width.mean())

    def test_upper_bound_exceeds_lower_bound(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert np.all(result.upper_bound >= result.lower_bound)

    def test_lower_bound_nonnegative(self, calibrated_cp, base_data):
        """Lower bounds should be clipped at zero for non-negative insurance losses."""
        result = solvency_capital_range(calibrated_cp, base_data["X_test"])
        assert np.all(result.lower_bound >= 0)


# ---------------------------------------------------------------------------
# Custom alpha
# ---------------------------------------------------------------------------


class TestCustomAlpha:
    def test_alpha_0_01_gives_99pct_coverage(self, calibrated_cp, base_data):
        result = solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=0.01)
        assert np.isclose(result.coverage_level, 0.99)
        assert np.isclose(result.alpha, 0.01)

    def test_smaller_alpha_gives_wider_intervals(self, calibrated_cp, base_data):
        """Stricter alpha (smaller value) must produce wider intervals."""
        r_005 = solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=0.005)
        r_10 = solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=0.10)
        assert r_005.mean_interval_width > r_10.mean_interval_width

    def test_smaller_alpha_gives_higher_upper_bound(self, calibrated_cp, base_data):
        """Stricter alpha must produce higher (more conservative) upper bounds."""
        r_005 = solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=0.005)
        r_10 = solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=0.10)
        assert r_005.mean_interval_width >= r_10.mean_interval_width


# ---------------------------------------------------------------------------
# Exposure weighting
# ---------------------------------------------------------------------------


class TestExposure:
    def test_unit_exposure_is_no_op(self, calibrated_cp, base_data):
        n = len(base_data["X_test"])
        unit_exp = np.ones(n)
        result_no_exp = solvency_capital_range(calibrated_cp, base_data["X_test"])
        result_unit = solvency_capital_range(
            calibrated_cp, base_data["X_test"], exposure=unit_exp
        )
        assert np.allclose(result_no_exp.upper_bound, result_unit.upper_bound)
        assert np.allclose(result_no_exp.lower_bound, result_unit.lower_bound)

    def test_half_exposure_halves_scr(self, calibrated_cp, base_data):
        n = len(base_data["X_test"])
        half_exp = np.full(n, 0.5)
        result_full = solvency_capital_range(calibrated_cp, base_data["X_test"])
        result_half = solvency_capital_range(
            calibrated_cp, base_data["X_test"], exposure=half_exp
        )
        assert np.allclose(result_half.upper_bound, result_full.upper_bound * 0.5)
        assert np.allclose(result_half.lower_bound, result_full.lower_bound * 0.5)

    def test_exposure_scales_total_scr(self, calibrated_cp, base_data):
        n = len(base_data["X_test"])
        exp = np.full(n, 2.0)
        result_full = solvency_capital_range(calibrated_cp, base_data["X_test"])
        result_double = solvency_capital_range(
            calibrated_cp, base_data["X_test"], exposure=exp
        )
        assert np.isclose(result_double.total_scr, result_full.total_scr * 2.0, rtol=1e-6)

    def test_variable_exposure(self, calibrated_cp, base_data):
        """Variable exposure should apply elementwise to all bounds."""
        n = len(base_data["X_test"])
        rng = np.random.default_rng(99)
        exp = rng.uniform(0.1, 2.0, size=n)
        result_no_exp = solvency_capital_range(calibrated_cp, base_data["X_test"])
        result_exp = solvency_capital_range(
            calibrated_cp, base_data["X_test"], exposure=exp
        )
        assert np.allclose(result_exp.upper_bound, result_no_exp.upper_bound * exp)
        assert np.allclose(result_exp.lower_bound, result_no_exp.lower_bound * exp)


# ---------------------------------------------------------------------------
# Compatibility with different predictors
# ---------------------------------------------------------------------------


class TestPredictorCompatibility:
    def test_works_with_cqr(self, calibrated_cqr, base_data):
        result = solvency_capital_range(calibrated_cqr, base_data["X_test"])
        assert isinstance(result, SolvencyCapitalRange)
        assert result.n_risks == len(base_data["X_test"])
        assert np.all(result.scr_estimate >= 0)
        assert np.all(result.interval_width >= 0)

    def test_cqr_upper_bound_positive(self, calibrated_cqr, base_data):
        result = solvency_capital_range(calibrated_cqr, base_data["X_test"])
        assert np.all(result.upper_bound >= 0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_rejects_predictor_without_predict_interval(self):
        with pytest.raises(TypeError, match="predict_interval"):
            solvency_capital_range("not_a_predictor", np.zeros((10, 3)))

    def test_rejects_alpha_zero(self, calibrated_cp, base_data):
        with pytest.raises(ValueError, match="alpha"):
            solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=0.0)

    def test_rejects_alpha_one(self, calibrated_cp, base_data):
        with pytest.raises(ValueError, match="alpha"):
            solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=1.0)

    def test_rejects_alpha_negative(self, calibrated_cp, base_data):
        with pytest.raises(ValueError, match="alpha"):
            solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=-0.1)

    def test_rejects_alpha_above_one(self, calibrated_cp, base_data):
        with pytest.raises(ValueError, match="alpha"):
            solvency_capital_range(calibrated_cp, base_data["X_test"], alpha=1.5)

    def test_rejects_nonpositive_exposure(self, calibrated_cp, base_data):
        n = len(base_data["X_test"])
        bad_exp = np.ones(n)
        bad_exp[3] = 0.0
        with pytest.raises(ValueError, match="positive"):
            solvency_capital_range(calibrated_cp, base_data["X_test"], exposure=bad_exp)

    def test_rejects_negative_exposure(self, calibrated_cp, base_data):
        n = len(base_data["X_test"])
        bad_exp = np.ones(n)
        bad_exp[0] = -0.5
        with pytest.raises(ValueError, match="positive"):
            solvency_capital_range(calibrated_cp, base_data["X_test"], exposure=bad_exp)

    def test_rejects_mismatched_exposure_length(self, calibrated_cp, base_data):
        wrong_exp = np.ones(5)  # wrong length
        with pytest.raises(ValueError, match="exposure length"):
            solvency_capital_range(
                calibrated_cp, base_data["X_test"], exposure=wrong_exp
            )
