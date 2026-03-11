"""
Tests for HongModelFree and HongTransformConformal.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal.hong import HongModelFree, HongTransformConformal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hong_data():
    rng = np.random.default_rng(77)
    n = 800
    p = 3
    beta = np.array([0.5, -0.2, 0.1])
    X = rng.normal(size=(n, p))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)

    return {
        "X_train": X[:400],
        "y_train": y[:400],
        "X_cal": X[400:600],
        "y_cal": y[400:600],
        "X_test": X[600:],
        "y_test": y[600:],
        "beta": beta,
        "mu_test": mu[600:],
    }


def simple_model_predict(X):
    """Simple linear model predict for tests."""
    X = np.asarray(X)
    beta = np.array([0.5, -0.2, 0.1])
    return np.exp(X @ beta)


# ---------------------------------------------------------------------------
# HongModelFree
# ---------------------------------------------------------------------------


class TestHongModelFree:
    def test_repr_not_fitted(self):
        hmf = HongModelFree()
        assert "not fitted" in repr(hmf)

    def test_repr_fitted(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        r = repr(hmf)
        assert "n_train=" in r
        assert "400" in r

    def test_fit_stores_data(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        assert hmf.X_train_ is not None
        assert hmf.y_train_ is not None
        assert hmf.n_train_ == 400

    def test_predict_without_fit_raises(self, hong_data):
        hmf = HongModelFree()
        with pytest.raises(RuntimeError, match="fit"):
            hmf.predict_interval(hong_data["X_test"])

    def test_output_columns(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        result = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        assert isinstance(result, pl.DataFrame)
        assert "lower" in result.columns
        assert "upper" in result.columns
        assert len(result) == len(hong_data["X_test"])

    def test_lower_nonnegative(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        result = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (result["lower"] >= 0).all()

    def test_lower_leq_upper(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        result = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (result["lower"] <= result["upper"]).all()

    def test_invalid_alpha(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        with pytest.raises(ValueError, match="alpha"):
            hmf.predict_interval(hong_data["X_test"], alpha=1.5)

    def test_smaller_alpha_gives_wider_intervals(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        r_wide = hmf.predict_interval(hong_data["X_test"], alpha=0.05)
        r_narrow = hmf.predict_interval(hong_data["X_test"], alpha=0.30)
        w_wide = (r_wide["upper"] - r_wide["lower"]).mean()
        w_narrow = (r_narrow["upper"] - r_narrow["lower"]).mean()
        assert w_wide >= w_narrow

    def test_fit_returns_self(self, hong_data):
        hmf = HongModelFree()
        result = hmf.fit(hong_data["X_train"], hong_data["y_train"])
        assert result is hmf

    def test_coverage_marginal(self, hong_data):
        """HongModelFree should achieve ~marginal coverage on Poisson data."""
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        alpha = 0.10
        result = hmf.predict_interval(hong_data["X_test"], alpha=alpha)
        y = hong_data["y_test"]
        lower = result["lower"].to_numpy()
        upper = result["upper"].to_numpy()
        coverage = float(((y >= lower) & (y <= upper)).mean())
        # Model-free conformal may be conservative, but should cover
        # We just check it's not trivially broken (coverage > 0.5)
        assert coverage > 0.5, f"Suspiciously low coverage: {coverage:.3%}"


# ---------------------------------------------------------------------------
# HongTransformConformal
# ---------------------------------------------------------------------------


class TestHongTransformConformal:
    def test_requires_callable(self):
        with pytest.raises(ValueError, match="callable"):
            HongTransformConformal(h="not_a_function")

    def test_repr_not_calibrated(self):
        htc = HongTransformConformal(h=simple_model_predict)
        assert "not calibrated" in repr(htc)

    def test_repr_calibrated(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        assert "calibrated on 200 obs" in repr(htc)

    def test_calibrate_sets_state(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        assert htc.is_calibrated_
        assert htc.n_calibration_ == 200
        assert htc.cal_residuals_ is not None
        assert len(htc.cal_residuals_) == 200

    def test_residuals_have_correct_sign(self, hong_data):
        """Residuals W_i = Y_i - h(X_i) can be positive or negative."""
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        # Should have both positive and negative residuals for a reasonable model
        resids = htc.cal_residuals_
        assert np.any(resids > 0)
        assert np.any(resids < 0)

    def test_calibrate_returns_self(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        result = htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        assert result is htc

    def test_predict_without_calibrate_raises(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        with pytest.raises(RuntimeError, match="calibrate"):
            htc.predict_interval(hong_data["X_test"])

    def test_output_columns(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        result = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"lower", "point", "upper"}
        assert len(result) == len(hong_data["X_test"])

    def test_lower_nonnegative(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        result = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (result["lower"] >= 0).all()

    def test_lower_leq_point_leq_upper(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        result = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (result["lower"] <= result["point"]).all()
        assert (result["point"] <= result["upper"]).all()

    def test_invalid_alpha(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        with pytest.raises(ValueError, match="alpha"):
            htc.predict_interval(hong_data["X_test"], alpha=0.0)

    def test_smaller_alpha_gives_wider_intervals(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        r_wide = htc.predict_interval(hong_data["X_test"], alpha=0.05)
        r_narrow = htc.predict_interval(hong_data["X_test"], alpha=0.30)
        w_wide = (r_wide["upper"] - r_wide["lower"]).mean()
        w_narrow = (r_narrow["upper"] - r_narrow["lower"]).mean()
        assert w_wide > w_narrow

    def test_coverage_marginal(self, hong_data):
        """HongTransformConformal should achieve marginal coverage."""
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        alpha = 0.10
        result = htc.predict_interval(hong_data["X_test"], alpha=alpha)
        y = hong_data["y_test"]
        lower = result["lower"].to_numpy()
        upper = result["upper"].to_numpy()
        coverage = float(((y >= lower) & (y <= upper)).mean())
        # Allow 6pp tolerance for n=200 calibration
        assert coverage >= 1 - alpha - 0.06, (
            f"Coverage {coverage:.3%} below target {1-alpha:.0%} - 6pp tolerance"
        )

    def test_predict_upper_bound(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        ub = htc.predict_upper_bound(hong_data["X_test"], alpha=0.10)
        assert isinstance(ub, np.ndarray)
        assert len(ub) == len(hong_data["X_test"])
        # Upper bound should be >= point prediction for most cases
        h_test = simple_model_predict(hong_data["X_test"])
        # The one-sided bound uses full alpha, so should be tighter than 2-sided
        two_sided = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        # upper bound from one-sided <= upper from two-sided (uses full alpha vs alpha/2)
        assert np.mean(ub <= two_sided["upper"].to_numpy()) > 0.9

    def test_predict_upper_bound_coverage(self, hong_data):
        """One-sided upper bound should cover with prob >= 1-alpha."""
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        alpha = 0.10
        ub = htc.predict_upper_bound(hong_data["X_test"], alpha=alpha)
        y = hong_data["y_test"]
        coverage = float((y <= ub).mean())
        assert coverage >= 1 - alpha - 0.06

    def test_lambda_h_works(self, hong_data):
        """h can be a lambda or arbitrary callable."""
        y_mean = float(hong_data["y_train"].mean())
        htc = HongTransformConformal(h=lambda X: np.full(len(X), y_mean))
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        result = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        # All point predictions should equal the constant
        assert np.allclose(result["point"].to_numpy(), y_mean)

    def test_recalibrate_clears_cache(self, hong_data):
        htc = HongTransformConformal(h=simple_model_predict)
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        htc.predict_interval(hong_data["X_test"], alpha=0.10)
        assert 0.10 in htc.cal_quantiles_
        htc.calibrate(hong_data["X_cal"], hong_data["y_cal"])
        assert len(htc.cal_quantiles_) == 0
