"""
Tests for FrequencySeverityConformal.

The key behavioural contract being tested:
1. Coverage is >= 1 - alpha (marginal, finite-sample valid).
2. Intervals are finite and positive width.
3. Calibration uses composed prediction mu_hat(x), not observed d_i.
4. Zero-claim observations are handled correctly.
5. LightGBM spread model works as a drop-in alternative.
6. Edge cases: single calibration point, all-zero claims.

We use sklearn GLMs (PoissonRegressor / GammaRegressor) as the base models
throughout, to avoid requiring CatBoost for the main model. The default
spread model (CatBoost) is swapped for a constant mock in most tests to
keep the test suite environment-agnostic. Tests that actually require
CatBoost or LightGBM are marked with importorskip.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal.claims import FrequencySeverityConformal


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------


class ConstantModel:
    """Returns a constant prediction regardless of input."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


class LinearFreqModel:
    """E[d] = exp(X @ beta). Pre-fitted — fit() is a no-op."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def fit(self, X, y):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(X) @ self.beta)


class LinearSevModel:
    """E[y | x, d] = c * d + X @ gamma. Fit uses least squares on first column only."""

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSevModel":
        # X has shape (n, p+1) where last column is frequency
        self._coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X) @ self._coef


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_freqsev():
    """
    Synthetic frequency-severity dataset.

    Frequency: Poisson with mean exp(0.3*x1 - 0.2*x2).
    Severity: Gamma with mean 500 * freq * exp(0.1*x3).
    """
    rng = np.random.default_rng(2024)
    n = 2000
    p = 4
    X = rng.normal(size=(n, p))

    # True frequency
    freq_true = np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1])
    d = rng.poisson(freq_true).astype(float)

    # True severity (only for claims)
    sev_mean = 500.0 * (d + 0.01) * np.exp(0.1 * X[:, 2])
    y = np.where(d > 0, rng.gamma(shape=2.0, scale=sev_mean / 2.0), 0.0)

    n_train = 1200
    n_cal = 500

    return {
        "X_train": X[:n_train],
        "d_train": d[:n_train],
        "y_train": y[:n_train],
        "X_cal": X[n_train : n_train + n_cal],
        "d_cal": d[n_train : n_train + n_cal],
        "y_cal": y[n_train : n_train + n_cal],
        "X_test": X[n_train + n_cal :],
        "d_test": d[n_train + n_cal :],
        "y_test": y[n_train + n_cal :],
        "n_train": n_train,
        "n_cal": n_cal,
    }


def make_fitted_fs(data, spread_model=None) -> FrequencySeverityConformal:
    """Build a FrequencySeverityConformal with mock models already fitted."""
    freq_model = LinearFreqModel(beta=np.array([0.3, -0.2, 0.0, 0.0]))
    sev_model = LinearSevModel()
    if spread_model is None:
        spread_model = ConstantModel(value=200.0)

    fs = FrequencySeverityConformal(
        freq_model=freq_model,
        sev_model=sev_model,
        spread_model=spread_model,
        alpha=0.1,
    )
    fs.fit(data["X_train"], data["d_train"], data["y_train"])
    return fs


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults_stored(self, synthetic_freqsev):
        fs = make_fitted_fs(synthetic_freqsev)
        assert fs.alpha == 0.1
        assert not fs.is_calibrated_

    def test_invalid_alpha_raises(self, synthetic_freqsev):
        with pytest.raises(ValueError, match="alpha"):
            FrequencySeverityConformal(
                freq_model=ConstantModel(1.0),
                sev_model=LinearSevModel(),
                alpha=1.5,
            )

    def test_repr_not_fitted(self):
        fs = FrequencySeverityConformal(
            freq_model=ConstantModel(1.0),
            sev_model=LinearSevModel(),
        )
        assert "not fitted" in repr(fs)

    def test_repr_calibrated(self, synthetic_freqsev):
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        assert "calibrated on" in repr(fs)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_sets_models(self, synthetic_freqsev):
        fs = make_fitted_fs(synthetic_freqsev)
        assert fs.freq_model_ is not None
        assert fs.sev_model_ is not None
        assert fs.spread_model_ is not None

    def test_fit_returns_self(self, synthetic_freqsev):
        freq_model = LinearFreqModel(beta=np.array([0.3, -0.2, 0.0, 0.0]))
        sev_model = LinearSevModel()
        fs = FrequencySeverityConformal(
            freq_model=freq_model,
            sev_model=sev_model,
            spread_model=ConstantModel(200.0),
        )
        result = fs.fit(
            synthetic_freqsev["X_train"],
            synthetic_freqsev["d_train"],
            synthetic_freqsev["y_train"],
        )
        assert result is fs

    def test_fit_no_claims_raises(self, synthetic_freqsev):
        d_zero = np.zeros(len(synthetic_freqsev["d_train"]))
        freq_model = LinearFreqModel(beta=np.array([0.3, -0.2, 0.0, 0.0]))
        sev_model = LinearSevModel()
        fs = FrequencySeverityConformal(
            freq_model=freq_model,
            sev_model=sev_model,
            spread_model=ConstantModel(200.0),
        )
        with pytest.raises(ValueError, match="No claim observations"):
            fs.fit(synthetic_freqsev["X_train"], d_zero, synthetic_freqsev["y_train"])


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_calibrate_without_fit_raises(self, synthetic_freqsev):
        fs = FrequencySeverityConformal(
            freq_model=ConstantModel(1.0),
            sev_model=LinearSevModel(),
            spread_model=ConstantModel(200.0),
        )
        with pytest.raises(RuntimeError, match="not been fitted"):
            fs.calibrate(
                synthetic_freqsev["X_cal"],
                synthetic_freqsev["d_cal"],
                synthetic_freqsev["y_cal"],
            )

    def test_calibrate_sets_state(self, synthetic_freqsev):
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        assert fs.is_calibrated_
        assert fs.n_calibration_ == synthetic_freqsev["n_cal"]
        assert fs.cal_scores_ is not None

    def test_cal_scores_nonnegative(self, synthetic_freqsev):
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        assert np.all(fs.cal_scores_ >= 0)

    def test_recalibrate_clears_quantile_cache(self, synthetic_freqsev):
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        fs.predict_interval(synthetic_freqsev["X_test"], alpha=0.10)
        assert 0.10 in fs.cal_quantiles_
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        assert len(fs.cal_quantiles_) == 0


# ---------------------------------------------------------------------------
# predict_interval()
# ---------------------------------------------------------------------------


class TestPredictInterval:
    def _calibrated_fs(self, data):
        fs = make_fitted_fs(data)
        fs.calibrate(data["X_cal"], data["d_cal"], data["y_cal"])
        return fs

    def test_predict_without_calibration_raises(self, synthetic_freqsev):
        fs = make_fitted_fs(synthetic_freqsev)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            fs.predict_interval(synthetic_freqsev["X_test"])

    def test_output_is_polars_dataframe(self, synthetic_freqsev):
        fs = self._calibrated_fs(synthetic_freqsev)
        result = fs.predict_interval(synthetic_freqsev["X_test"])
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, synthetic_freqsev):
        fs = self._calibrated_fs(synthetic_freqsev)
        result = fs.predict_interval(synthetic_freqsev["X_test"])
        assert set(result.columns) == {"lower", "point", "upper"}

    def test_output_row_count(self, synthetic_freqsev):
        fs = self._calibrated_fs(synthetic_freqsev)
        result = fs.predict_interval(synthetic_freqsev["X_test"])
        assert len(result) == len(synthetic_freqsev["X_test"])

    def test_lower_leq_upper(self, synthetic_freqsev):
        """lower <= upper is the conformal guarantee. point can theoretically be
        outside [lower, upper] for unconstrained sev models that predict negative
        values — but the interval bounds themselves must be ordered."""
        fs = self._calibrated_fs(synthetic_freqsev)
        result = fs.predict_interval(synthetic_freqsev["X_test"])
        assert (result["lower"] <= result["upper"]).all()

    def test_point_leq_upper(self, synthetic_freqsev):
        """Point prediction should be <= upper bound (upper = point + half_width)."""
        fs = self._calibrated_fs(synthetic_freqsev)
        result = fs.predict_interval(synthetic_freqsev["X_test"])
        assert (result["point"] <= result["upper"]).all()

    def test_lower_nonnegative(self, synthetic_freqsev):
        fs = self._calibrated_fs(synthetic_freqsev)
        result = fs.predict_interval(synthetic_freqsev["X_test"])
        assert (result["lower"] >= 0).all()

    def test_interval_widths_positive(self, synthetic_freqsev):
        fs = self._calibrated_fs(synthetic_freqsev)
        result = fs.predict_interval(synthetic_freqsev["X_test"])
        widths = result["upper"] - result["lower"]
        assert (widths > 0).all()

    def test_intervals_finite(self, synthetic_freqsev):
        fs = self._calibrated_fs(synthetic_freqsev)
        result = fs.predict_interval(synthetic_freqsev["X_test"])
        assert np.all(np.isfinite(result["lower"].to_numpy()))
        assert np.all(np.isfinite(result["upper"].to_numpy()))

    def test_wider_at_lower_alpha(self, synthetic_freqsev):
        fs = self._calibrated_fs(synthetic_freqsev)
        wide = fs.predict_interval(synthetic_freqsev["X_test"], alpha=0.05)
        narrow = fs.predict_interval(synthetic_freqsev["X_test"], alpha=0.30)
        w_wide = (wide["upper"] - wide["lower"]).mean()
        w_narrow = (narrow["upper"] - narrow["lower"]).mean()
        assert w_wide > w_narrow

    def test_invalid_alpha_raises(self, synthetic_freqsev):
        fs = self._calibrated_fs(synthetic_freqsev)
        with pytest.raises(ValueError, match="alpha"):
            fs.predict_interval(synthetic_freqsev["X_test"], alpha=2.0)

    def test_default_alpha_used(self, synthetic_freqsev):
        """predict_interval() with no alpha arg uses self.alpha."""
        fs = make_fitted_fs(synthetic_freqsev)
        fs.alpha = 0.15
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        result_default = fs.predict_interval(synthetic_freqsev["X_test"])
        result_explicit = fs.predict_interval(synthetic_freqsev["X_test"], alpha=0.15)
        # Widths should be identical
        w1 = (result_default["upper"] - result_default["lower"]).to_numpy()
        w2 = (result_explicit["upper"] - result_explicit["lower"]).to_numpy()
        np.testing.assert_array_equal(w1, w2)


# ---------------------------------------------------------------------------
# Coverage guarantee
# ---------------------------------------------------------------------------


class TestCoverage:
    TOLERANCE = 0.06  # allow 6% slack for the finite sample

    def test_marginal_coverage(self, synthetic_freqsev):
        """Empirical coverage must be >= 1-alpha (marginal conformal guarantee)."""
        alpha = 0.10
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        result = fs.predict_interval(synthetic_freqsev["X_test"], alpha=alpha)
        y_test = synthetic_freqsev["y_test"]
        lower = result["lower"].to_numpy()
        upper = result["upper"].to_numpy()
        coverage = float(((y_test >= lower) & (y_test <= upper)).mean())
        assert coverage >= 1 - alpha - self.TOLERANCE, (
            f"Coverage {coverage:.3%} is more than {self.TOLERANCE:.0%} below "
            f"target {1 - alpha:.0%}"
        )


# ---------------------------------------------------------------------------
# Composed prediction test
# ---------------------------------------------------------------------------


class TestComposedPrediction:
    """
    Verify that calibration uses mu_hat(x), not observed d_i.

    We create a frequency model that always predicts 2.0 regardless of input.
    We intercept calls to sev_model_.predict at calibration time and record
    what frequency values were appended. They should all be 2.0, not the
    observed d values.
    """

    def test_calibration_uses_mu_hat_not_observed_d(self, synthetic_freqsev):
        recorded_freq_values = []

        class RecordingModel:
            """Records what frequency values were passed during predict()."""

            def __init__(self):
                self._inner = LinearSevModel()

            def fit(self, X, y):
                self._inner.fit(X, y)
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                # Last column is the appended frequency
                recorded_freq_values.append(X[:, -1].copy())
                return self._inner.predict(X)

        constant_freq = 2.0
        freq_model = ConstantModel(value=constant_freq)
        sev_model = RecordingModel()

        fs = FrequencySeverityConformal(
            freq_model=freq_model,
            sev_model=sev_model,
            spread_model=ConstantModel(value=200.0),
        )
        fs.fit(
            synthetic_freqsev["X_train"],
            synthetic_freqsev["d_train"],
            synthetic_freqsev["y_train"],
        )

        # Clear recording before calibrate
        recorded_freq_values.clear()
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )

        # The freq values passed to sev model at calibration should be mu_hat(x)
        # which is always 2.0, not the observed d values
        assert len(recorded_freq_values) > 0, "sev_model.predict was not called"
        for freq_vals in recorded_freq_values:
            np.testing.assert_allclose(
                freq_vals,
                constant_freq,
                err_msg=(
                    "Calibration passed observed d values instead of mu_hat(x). "
                    "This breaks the conformal coverage guarantee."
                ),
            )

        # Verify observed d values were NOT used (they vary)
        d_cal = synthetic_freqsev["d_cal"]
        assert not np.allclose(d_cal, constant_freq), (
            "Test data degenerate: all d_cal == 2.0, test cannot distinguish"
        )


# ---------------------------------------------------------------------------
# Zero-claim handling
# ---------------------------------------------------------------------------


class TestZeroClaimHandling:
    def test_zero_claims_in_calibration(self, synthetic_freqsev):
        """Calibration set with many zero-claim policies should work fine."""
        # Zero-claim observations have y=0. With our spread model returning
        # a constant, their score = |0 - point| / sigma. This is fine.
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        n_zero = (synthetic_freqsev["d_cal"] == 0).sum()
        # Our synthetic data should have some zero-claim observations
        assert n_zero > 0, "Test requires some zero-claim calibration observations"
        assert fs.is_calibrated_

    def test_prediction_with_all_zero_input(self, synthetic_freqsev):
        """predict_interval works even if X_new would produce near-zero frequency."""
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            synthetic_freqsev["X_cal"],
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        # X that gives very low frequency (large negative first feature)
        X_low = np.column_stack([
            np.full(10, -10.0),
            np.zeros((10, 3)),
        ])
        result = fs.predict_interval(X_low, alpha=0.10)
        assert isinstance(result, pl.DataFrame)
        assert (result["lower"] >= 0).all()
        assert np.all(np.isfinite(result["upper"].to_numpy()))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_calibration_point(self, synthetic_freqsev):
        """Single calibration observation should produce a quantile == that score."""
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            synthetic_freqsev["X_cal"][:1],
            synthetic_freqsev["d_cal"][:1],
            synthetic_freqsev["y_cal"][:1],
        )
        assert fs.n_calibration_ == 1
        result = fs.predict_interval(synthetic_freqsev["X_test"][:5], alpha=0.10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5

    def test_polars_input(self, synthetic_freqsev):
        """FrequencySeverityConformal accepts Polars DataFrames as input."""
        fs = make_fitted_fs(synthetic_freqsev)
        fs.calibrate(
            pl.DataFrame(synthetic_freqsev["X_cal"]),
            synthetic_freqsev["d_cal"],
            synthetic_freqsev["y_cal"],
        )
        result = fs.predict_interval(
            pl.DataFrame(synthetic_freqsev["X_test"]), alpha=0.10
        )
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# LightGBM backend on LocallyWeightedConformal
# ---------------------------------------------------------------------------


class TestLightGBMBackend:
    """Tests for LocallyWeightedConformal with backend='lightgbm'."""

    def test_invalid_backend_raises(self):
        from insurance_conformal.locally_weighted import LocallyWeightedConformal

        class DummyModel:
            def predict(self, X):
                return np.ones(len(X))

        with pytest.raises(ValueError, match="backend"):
            LocallyWeightedConformal(model=DummyModel(), backend="xgboost")

    def test_lightgbm_backend_repr(self):
        from insurance_conformal.locally_weighted import LocallyWeightedConformal

        lgbm = pytest.importorskip("lightgbm")

        class DummyModel:
            def predict(self, X):
                return np.ones(len(X))

        lw = LocallyWeightedConformal(model=DummyModel(), backend="lightgbm")
        assert "lightgbm" in repr(lw)

    def test_lightgbm_fit_calibrate_predict(self):
        """Full fit/calibrate/predict cycle with LightGBM spread model."""
        lgbm = pytest.importorskip("lightgbm")

        from insurance_conformal.locally_weighted import LocallyWeightedConformal

        rng = np.random.default_rng(42)
        n = 600
        X = rng.normal(size=(n, 3))
        beta = np.array([0.4, -0.2, 0.1])
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)

        class TrueModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.exp(np.asarray(X) @ beta)

        model = TrueModel()

        lw = LocallyWeightedConformal(
            model=model,
            tweedie_power=1.0,
            backend="lightgbm",
            lgbm_spread_model_params={"n_estimators": 50, "verbose": -1},
        )
        lw.fit(X[:400], y[:400])
        lw.calibrate(X[400:500], y[400:500])
        result = lw.predict_interval(X[500:], alpha=0.10)

        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"lower", "point", "upper", "spread"}
        assert (result["lower"] >= 0).all()
        assert (result["lower"] <= result["upper"]).all()

    def test_lightgbm_coverage(self):
        """LightGBM backend achieves >= 1-alpha empirical coverage."""
        lgbm = pytest.importorskip("lightgbm")

        from insurance_conformal.locally_weighted import LocallyWeightedConformal

        rng = np.random.default_rng(7)
        n = 1000
        X = rng.normal(size=(n, 3))
        beta = np.array([0.3, -0.15, 0.1])
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)

        class TrueModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.exp(np.asarray(X) @ beta)

        lw = LocallyWeightedConformal(
            model=TrueModel(),
            tweedie_power=1.0,
            backend="lightgbm",
            lgbm_spread_model_params={"n_estimators": 50, "verbose": -1},
        )
        alpha = 0.10
        lw.fit(X[:600], y[:600])
        lw.calibrate(X[600:800], y[600:800])
        result = lw.predict_interval(X[800:], alpha=alpha)

        y_test = y[800:]
        lower = result["lower"].to_numpy()
        upper = result["upper"].to_numpy()
        coverage = float(((y_test >= lower) & (y_test <= upper)).mean())
        assert coverage >= 1 - alpha - 0.06, (
            f"LightGBM backend coverage {coverage:.3%} below target {1-alpha:.0%}"
        )

    def test_catboost_backend_still_works(self):
        """Explicit backend='catboost' still works (not broken by the new parameter)."""
        catboost = pytest.importorskip("catboost")

        from insurance_conformal.locally_weighted import LocallyWeightedConformal

        rng = np.random.default_rng(13)
        n = 600
        X = rng.normal(size=(n, 3))
        beta = np.array([0.3, -0.2, 0.1])
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)

        class TrueModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.exp(np.asarray(X) @ beta)

        lw = LocallyWeightedConformal(
            model=TrueModel(),
            tweedie_power=1.0,
            backend="catboost",
            spread_model_params={"iterations": 50, "verbose": False},
        )
        lw.fit(X[:400], y[:400])
        lw.calibrate(X[400:500], y[400:500])
        result = lw.predict_interval(X[500:], alpha=0.10)
        assert isinstance(result, pl.DataFrame)
        assert (result["lower"] >= 0).all()


# ---------------------------------------------------------------------------
# TweediePearsonScore alias
# ---------------------------------------------------------------------------


class TestTweediePearsonScoreAlias:
    def test_alias_exists(self):
        from insurance_conformal.claims import TweediePearsonScore

        assert TweediePearsonScore is not None

    def test_alias_is_same_class(self):
        from insurance_conformal.claims import TweedePearsonScore, TweediePearsonScore

        assert TweediePearsonScore is TweedePearsonScore

    def test_alias_works(self):
        from insurance_conformal.claims import TweediePearsonScore

        score = TweediePearsonScore(p=1.5)
        y = np.array([1.0, 2.0, 0.0])
        y_hat = np.array([1.2, 1.8, 0.5])
        scores = score.score(y, y_hat)
        assert scores.shape == (3,)
        assert np.all(scores >= 0)
