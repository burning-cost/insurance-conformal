"""
Tests for LocallyWeightedConformal.

The spread model is fitted on |Pearson residuals| from the training set.
We use a minimal mock CatBoost stand-in (a constant predictor) to avoid
requiring a real CatBoost installation in environments where it isn't available.
Tests that genuinely require CatBoost are marked with a pytest.importorskip.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal.locally_weighted import LocallyWeightedConformal


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class ConstantSpreadModel:
    """Returns a constant spread prediction for all inputs."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)

    def fit(self, X, y):
        return self


class VaryingSpreadModel:
    """Returns spread that varies with first feature — for diagnostics tests."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # Spread proportional to exp(first feature), clipped > 0
        return np.clip(np.exp(X[:, 0] * 0.5), 0.5, 3.0)

    def fit(self, X, y):
        return self


class LinearModel:
    """E[Y] = exp(X @ beta)."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(X) @ self.beta)


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(99)
    n = 1500
    p = 3
    beta = np.array([0.4, -0.2, 0.15])
    X = rng.normal(size=(n, p))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)

    model = LinearModel(beta)

    return {
        "X_train": X[:900],
        "y_train": y[:900],
        "X_cal": X[900:1200],
        "y_cal": y[900:1200],
        "X_test": X[1200:],
        "y_test": y[1200:],
        "model": model,
    }


def make_lw_with_mock_spread(data, tweedie_power=1.0):
    """Build a LocallyWeightedConformal with spread model bypassed via monkeypatching."""
    lw = LocallyWeightedConformal(model=data["model"], tweedie_power=tweedie_power)
    # Inject a constant spread model directly to avoid requiring CatBoost
    lw.spread_model_ = ConstantSpreadModel(value=1.0)
    return lw


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self, synthetic_data):
        lw = LocallyWeightedConformal(model=synthetic_data["model"])
        assert lw.tweedie_power == 1.5
        assert lw.clip_spread_at == 0.01
        assert lw.spread_model_ is None
        assert not lw.is_calibrated_

    def test_custom_power(self, synthetic_data):
        lw = LocallyWeightedConformal(model=synthetic_data["model"], tweedie_power=2.0)
        assert lw.tweedie_power == 2.0

    def test_custom_spread_params(self, synthetic_data):
        params = {"iterations": 50, "verbose": False}
        lw = LocallyWeightedConformal(
            model=synthetic_data["model"], spread_model_params=params
        )
        assert lw.spread_model_params["iterations"] == 50
        # Default params should be merged
        assert "learning_rate" in lw.spread_model_params

    def test_repr_not_fitted(self, synthetic_data):
        lw = LocallyWeightedConformal(model=synthetic_data["model"])
        assert "not fitted" in repr(lw)

    def test_repr_fitted_not_calibrated(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data)
        r = repr(lw)
        assert "fitted" in r
        assert "not calibrated" in r

    def test_repr_calibrated(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        r = repr(lw)
        assert "calibrated on" in r


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_calibrate_without_fit_raises(self, synthetic_data):
        lw = LocallyWeightedConformal(model=synthetic_data["model"])
        with pytest.raises(RuntimeError, match="Spread model has not been fitted"):
            lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])

    def test_calibrate_sets_state(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        assert lw.is_calibrated_
        assert lw.n_calibration_ == len(synthetic_data["y_cal"])
        assert lw.cal_scores_ is not None

    def test_cal_scores_positive(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        assert np.all(lw.cal_scores_ >= 0)

    def test_returns_self(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        result = lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        assert result is lw

    def test_recalibrate_clears_cache(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        lw.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert 0.10 in lw.cal_quantiles_
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        assert len(lw.cal_quantiles_) == 0

    def test_calibrate_mismatched_lengths_raises(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        y_short = synthetic_data["y_cal"][:-1]
        with pytest.raises(ValueError, match="lengths must match"):
            lw.calibrate(synthetic_data["X_cal"], y_short)


# ---------------------------------------------------------------------------
# predict_interval
# ---------------------------------------------------------------------------


class TestPredictInterval:
    def test_predict_without_calibration_raises(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            lw.predict_interval(synthetic_data["X_test"])

    def test_output_shape_and_columns(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        result = lw.predict_interval(synthetic_data["X_test"], alpha=0.10)

        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"lower", "point", "upper", "spread"}
        assert len(result) == len(synthetic_data["X_test"])

    def test_lower_leq_point_leq_upper(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        result = lw.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert (result["lower"] <= result["point"]).all()
        assert (result["point"] <= result["upper"]).all()

    def test_lower_nonnegative(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        result = lw.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert (result["lower"] >= 0).all()

    def test_invalid_alpha(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        with pytest.raises(ValueError, match="alpha"):
            lw.predict_interval(synthetic_data["X_test"], alpha=1.5)

    def test_wider_intervals_at_smaller_alpha(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        i_wide = lw.predict_interval(synthetic_data["X_test"], alpha=0.05)
        i_narrow = lw.predict_interval(synthetic_data["X_test"], alpha=0.30)
        w_wide = (i_wide["upper"] - i_wide["lower"]).mean()
        w_narrow = (i_narrow["upper"] - i_narrow["lower"]).mean()
        assert w_wide > w_narrow

    def test_spread_column_positive(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        result = lw.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert (result["spread"] > 0).all()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class TestCoverage:
    TOLERANCE = 0.05

    def test_marginal_coverage(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        alpha = 0.10
        intervals = lw.predict_interval(synthetic_data["X_test"], alpha=alpha)
        y = synthetic_data["y_test"]
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        coverage = float(((y >= lower) & (y <= upper)).mean())
        assert coverage >= 1 - alpha - self.TOLERANCE, (
            f"Coverage {coverage:.3%} is more than {self.TOLERANCE:.0%} below target {1-alpha:.0%}"
        )


# ---------------------------------------------------------------------------
# Coverage diagnostics
# ---------------------------------------------------------------------------


class TestCoverageDiagnostics:
    def test_output_columns(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        df = lw.coverage_diagnostics(
            synthetic_data["X_test"], synthetic_data["y_test"], alpha=0.10
        )
        assert isinstance(df, pl.DataFrame)
        expected_cols = {
            "slice_type", "bin", "mean_predicted", "mean_spread",
            "n_obs", "coverage", "target_coverage"
        }
        assert expected_cols.issubset(set(df.columns))

    def test_has_both_slice_types(self, synthetic_data):
        lw = LocallyWeightedConformal(model=synthetic_data["model"], tweedie_power=1.0)
        lw.spread_model_ = VaryingSpreadModel()
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        df = lw.coverage_diagnostics(
            synthetic_data["X_test"], synthetic_data["y_test"], alpha=0.10
        )
        slice_types = df["slice_type"].unique().to_list()
        assert "pred_decile" in slice_types
        assert "spread_quartile" in slice_types

    def test_coverage_values_in_range(self, synthetic_data):
        lw = make_lw_with_mock_spread(synthetic_data, tweedie_power=1.0)
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        df = lw.coverage_diagnostics(
            synthetic_data["X_test"], synthetic_data["y_test"], alpha=0.10
        )
        assert (df["coverage"] >= 0).all()
        assert (df["coverage"] <= 1).all()


# ---------------------------------------------------------------------------
# Backend selection (auto / sklearn)
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_default_backend_is_auto(self, synthetic_data):
        lw = LocallyWeightedConformal(model=synthetic_data["model"])
        assert lw.backend == "auto"

    def test_invalid_backend_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="backend must be one of"):
            LocallyWeightedConformal(model=synthetic_data["model"], backend="xgboost")

    def test_sklearn_backend_fits_without_optional_deps(self, synthetic_data):
        """sklearn backend must work with no optional dependencies installed."""
        lw = LocallyWeightedConformal(
            model=synthetic_data["model"],
            tweedie_power=1.0,
            backend="sklearn",
        )
        lw.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        assert lw.spread_model_ is not None
        assert lw.backend_ == "sklearn"

    def test_sklearn_backend_calibrates_and_predicts(self, synthetic_data):
        """Full pipeline with sklearn backend must produce valid intervals."""
        lw = LocallyWeightedConformal(
            model=synthetic_data["model"],
            tweedie_power=1.0,
            backend="sklearn",
        )
        lw.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        lw.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        intervals = lw.predict_interval(synthetic_data["X_test"], alpha=0.10)

        assert isinstance(intervals, pl.DataFrame)
        assert set(intervals.columns) == {"lower", "point", "upper", "spread"}
        assert (intervals["lower"] >= 0).all()
        assert (intervals["lower"] <= intervals["upper"]).all()

    def test_auto_backend_resolves_and_records(self, synthetic_data):
        """backend='auto' must set backend_ after fit()."""
        lw = LocallyWeightedConformal(
            model=synthetic_data["model"],
            tweedie_power=1.0,
            backend="auto",
        )
        assert lw.backend_ is None  # not yet resolved
        lw.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        # After fit, backend_ should be one of the valid concrete backends
        assert lw.backend_ in ("catboost", "lightgbm", "sklearn")

    def test_auto_repr_shows_resolution(self, synthetic_data):
        """repr() for backend='auto' should show what was resolved."""
        lw = LocallyWeightedConformal(
            model=synthetic_data["model"],
            tweedie_power=1.0,
            backend="auto",
        )
        lw.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        r = repr(lw)
        # Should contain "auto ->" to signal the resolved backend
        assert "auto ->" in r
