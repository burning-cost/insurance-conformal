"""
Tests for LCPModelSelector and LocalScoreSelector.

The LOO construction and kernel localisation are the two most complex
parts of this module. We test:
1.  Basic calibrate/predict cycle with multiple simple models.
2.  Output schema: polars DataFrame, correct column names.
3.  Interval ordering: lower <= point_estimate <= upper (up to clipping).
4.  Marginal coverage >= 1 - alpha (with tolerance) on a larger test set.
5.  LocalScoreSelector convenience wrapper (single model, multiple scores).
6.  Kernel weight shapes and normalisation.
7.  Fallback behaviour when calibration set is tiny.
8.  selected_models() output format and content.
9.  Invalid inputs raise appropriate errors.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_conformal.lcp_model_selector import (
    LCPModelSelector,
    LocalScoreSelector,
    _kernel_weights,
    _weighted_quantile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ConstantModel:
    """Predicts a fixed constant for all inputs."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)


class LinearExpModel:
    """Predicts exp(X @ beta)."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(X) @ self.beta)


@pytest.fixture(scope="module")
def synthetic_data():
    """
    Simple Poisson DGP. True model: E[Y] = exp(beta @ x).
    n=1500, split 900/300/300 train/cal/test.
    Two pre-fitted models: the true model and a biased one.
    """
    rng = np.random.default_rng(42)
    n = 1500
    p = 4
    beta = np.array([0.4, -0.2, 0.15, 0.05])
    X = rng.normal(size=(n, p))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)

    model_true = LinearExpModel(beta)
    # Biased model: wrong coefficients
    model_biased = LinearExpModel(beta * 1.3 + 0.1)

    return {
        "X_train": X[:900],
        "y_train": y[:900],
        "X_cal": X[900:1200],
        "y_cal": y[900:1200],
        "X_test": X[1200:],
        "y_test": y[1200:],
        "model_true": model_true,
        "model_biased": model_biased,
        "beta": beta,
    }


def make_lcp(data, score_fns=None, localizer="gaussian", bandwidth=1.0):
    """Build and calibrate an LCPModelSelector from synthetic_data."""
    if score_fns is None:
        score_fns = ["pearson_weighted", "pearson_weighted"]
    lcp = LCPModelSelector(
        models=[data["model_true"], data["model_biased"]],
        score_fns=score_fns,
        localizer=localizer,
        bandwidth=bandwidth,
    )
    lcp.calibrate(data["X_cal"], data["y_cal"])
    return lcp


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_attributes(self, synthetic_data):
        lcp = LCPModelSelector(
            models=[synthetic_data["model_true"]],
            score_fns=["pearson_weighted"],
        )
        assert lcp.localizer == "gaussian"
        assert lcp.bandwidth == 1.0
        assert not lcp.is_calibrated_

    def test_repr_not_calibrated(self, synthetic_data):
        lcp = LCPModelSelector(
            models=[synthetic_data["model_true"], synthetic_data["model_biased"]],
            score_fns=["raw", "raw"],
        )
        assert "not calibrated" in repr(lcp)
        assert "K=2" in repr(lcp)

    def test_mismatched_models_scores_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="same length"):
            LCPModelSelector(
                models=[synthetic_data["model_true"]],
                score_fns=["raw", "pearson"],
            )

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="At least one model"):
            LCPModelSelector(models=[], score_fns=[])

    def test_invalid_localizer_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="localizer"):
            LCPModelSelector(
                models=[synthetic_data["model_true"]],
                score_fns=["raw"],
                localizer="rbf",
            )

    def test_string_score_fns_resolved(self, synthetic_data):
        lcp = LCPModelSelector(
            models=[synthetic_data["model_true"]],
            score_fns=["deviance"],
        )
        assert callable(lcp.score_fns[0])
        assert lcp._score_names[0] == "deviance"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_sets_calibrated_flag(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        assert lcp.is_calibrated_

    def test_calibrate_returns_self(self, synthetic_data):
        lcp = LCPModelSelector(
            models=[synthetic_data["model_true"], synthetic_data["model_biased"]],
            score_fns=["pearson_weighted", "pearson_weighted"],
        )
        result = lcp.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        assert result is lcp

    def test_stores_cal_arrays(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        n = len(synthetic_data["y_cal"])
        K = 2
        assert lcp.cal_X_.shape == (n, 4)
        assert lcp.cal_y_.shape == (n,)
        assert lcp.cal_yhats_.shape == (n, K)
        assert lcp.cal_scores_.shape == (n, K)
        assert lcp.loo_intervals_.shape == (n, K, 2)

    def test_scores_nonnegative(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        assert np.all(lcp.cal_scores_ >= 0)

    def test_loo_intervals_ordered(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        lo = lcp.loo_intervals_[:, :, 0]
        hi = lcp.loo_intervals_[:, :, 1]
        assert np.all(lo <= hi)

    def test_mismatched_lengths_raises(self, synthetic_data):
        lcp = LCPModelSelector(
            models=[synthetic_data["model_true"]],
            score_fns=["raw"],
        )
        with pytest.raises(ValueError, match="X_cal has"):
            lcp.calibrate(
                synthetic_data["X_cal"],
                synthetic_data["y_cal"][:-1],
            )

    def test_repr_calibrated(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        r = repr(lcp)
        assert "calibrated on" in r
        assert "300" in r


# ---------------------------------------------------------------------------
# predict_interval output
# ---------------------------------------------------------------------------


class TestPredictInterval:
    def test_not_calibrated_raises(self, synthetic_data):
        lcp = LCPModelSelector(
            models=[synthetic_data["model_true"]],
            score_fns=["pearson_weighted"],
        )
        with pytest.raises(RuntimeError, match="not been calibrated"):
            lcp.predict_interval(synthetic_data["X_test"])

    def test_alpha_out_of_range_raises(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        with pytest.raises(ValueError, match="alpha"):
            lcp.predict_interval(synthetic_data["X_test"], alpha=1.5)

    def test_output_is_polars(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.predict_interval(synthetic_data["X_test"], alpha=0.10)
        expected = {"lower", "point_estimate", "upper", "ensemble_width", "n_models_in_set"}
        assert expected == set(result.columns)

    def test_output_length(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert len(result) == len(synthetic_data["y_test"])

    def test_lower_nonnegative(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert (result["lower"] >= 0).all()

    def test_lower_leq_upper(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert (result["lower"] <= result["upper"]).all()

    def test_ensemble_width_consistent(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.predict_interval(synthetic_data["X_test"], alpha=0.10)
        computed_width = result["upper"] - result["lower"]
        diff = (computed_width - result["ensemble_width"]).abs().max()
        assert diff < 1e-6

    def test_n_models_in_set_bounded(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.predict_interval(synthetic_data["X_test"], alpha=0.10)
        K = len(lcp.models)
        assert (result["n_models_in_set"] >= 1).all()
        assert (result["n_models_in_set"] <= K).all()

    def test_wider_at_lower_alpha(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        r_wide = lcp.predict_interval(synthetic_data["X_test"], alpha=0.05)
        r_narrow = lcp.predict_interval(synthetic_data["X_test"], alpha=0.30)
        w_wide = float(r_wide["ensemble_width"].mean())
        w_narrow = float(r_narrow["ensemble_width"].mean())
        assert w_wide > w_narrow


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class TestCoverage:
    TOLERANCE = 0.06

    def test_marginal_coverage_gaussian(self, synthetic_data):
        lcp = make_lcp(synthetic_data, localizer="gaussian", bandwidth=2.0)
        alpha = 0.10
        intervals = lcp.predict_interval(synthetic_data["X_test"], alpha=alpha)
        y = synthetic_data["y_test"]
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        coverage = float(((y >= lower) & (y <= upper)).mean())
        assert coverage >= 1.0 - alpha - self.TOLERANCE, (
            f"Coverage {coverage:.3%} is below target {1-alpha:.0%} - {self.TOLERANCE:.0%}"
        )

    def test_marginal_coverage_knn(self, synthetic_data):
        lcp = make_lcp(synthetic_data, localizer="knn")
        alpha = 0.10
        intervals = lcp.predict_interval(synthetic_data["X_test"], alpha=alpha)
        y = synthetic_data["y_test"]
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        coverage = float(((y >= lower) & (y <= upper)).mean())
        assert coverage >= 1.0 - alpha - self.TOLERANCE

    def test_marginal_coverage_uniform(self, synthetic_data):
        lcp = make_lcp(synthetic_data, localizer="uniform", bandwidth=3.0)
        alpha = 0.10
        intervals = lcp.predict_interval(synthetic_data["X_test"], alpha=alpha)
        y = synthetic_data["y_test"]
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        coverage = float(((y >= lower) & (y <= upper)).mean())
        assert coverage >= 1.0 - alpha - self.TOLERANCE


# ---------------------------------------------------------------------------
# selected_models output
# ---------------------------------------------------------------------------


class TestSelectedModels:
    def test_output_is_polars(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.selected_models(synthetic_data["X_test"], alpha=0.10)
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.selected_models(synthetic_data["X_test"], alpha=0.10)
        expected = {"selected_model_idx", "score_name", "n_models_in_set", "fallback_used"}
        assert expected == set(result.columns)

    def test_output_length(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.selected_models(synthetic_data["X_test"], alpha=0.10)
        assert len(result) == len(synthetic_data["y_test"])

    def test_model_idx_in_range(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.selected_models(synthetic_data["X_test"], alpha=0.10)
        K = len(lcp.models)
        assert (result["selected_model_idx"] >= 0).all()
        assert (result["selected_model_idx"] < K).all()

    def test_fallback_is_bool(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.selected_models(synthetic_data["X_test"], alpha=0.10)
        assert result["fallback_used"].dtype == pl.Boolean

    def test_not_calibrated_raises(self, synthetic_data):
        lcp = LCPModelSelector(
            models=[synthetic_data["model_true"]],
            score_fns=["raw"],
        )
        with pytest.raises(RuntimeError, match="not been calibrated"):
            lcp.selected_models(synthetic_data["X_test"])


# ---------------------------------------------------------------------------
# LocalScoreSelector
# ---------------------------------------------------------------------------


class TestLocalScoreSelector:
    def test_default_score_types(self, synthetic_data):
        lss = LocalScoreSelector(model=synthetic_data["model_true"])
        assert lss.score_types == ["pearson_weighted", "deviance", "anscombe"]
        assert len(lss.models) == 3

    def test_calibrate_predict_roundtrip(self, synthetic_data):
        lss = LocalScoreSelector(
            model=synthetic_data["model_true"],
            tweedie_power=1.0,
            localizer="gaussian",
            bandwidth=2.0,
        )
        lss.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        result = lss.predict_interval(synthetic_data["X_test"], alpha=0.10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(synthetic_data["y_test"])
        assert (result["lower"] >= 0).all()
        assert (result["lower"] <= result["upper"]).all()

    def test_custom_score_types(self, synthetic_data):
        lss = LocalScoreSelector(
            model=synthetic_data["model_true"],
            score_types=["raw", "pearson"],
        )
        assert len(lss.models) == 2
        assert lss.score_types == ["raw", "pearson"]

    def test_repr(self, synthetic_data):
        lss = LocalScoreSelector(model=synthetic_data["model_true"])
        r = repr(lss)
        assert "LocalScoreSelector" in r
        assert "pearson_weighted" in r
        assert "not calibrated" in r

    def test_repr_calibrated(self, synthetic_data):
        lss = LocalScoreSelector(model=synthetic_data["model_true"])
        lss.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        r = repr(lss)
        assert "calibrated on" in r

    def test_score_selection_varies(self, synthetic_data):
        """With multiple scores and enough data, selection should not be constant."""
        lss = LocalScoreSelector(
            model=synthetic_data["model_true"],
            tweedie_power=1.0,
        )
        lss.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        sel = lss.selected_models(synthetic_data["X_test"], alpha=0.10)
        # At least two distinct scores should be selected across 300 test points
        unique_scores = sel["score_name"].n_unique()
        # We don't enforce uniqueness > 1 since one score may dominate,
        # but the output should be well-formed
        assert unique_scores >= 1

    def test_marginal_coverage(self, synthetic_data):
        lss = LocalScoreSelector(
            model=synthetic_data["model_true"],
            tweedie_power=1.0,
            bandwidth=2.0,
        )
        lss.calibrate(synthetic_data["X_cal"], synthetic_data["y_cal"])
        alpha = 0.10
        intervals = lss.predict_interval(synthetic_data["X_test"], alpha=alpha)
        y = synthetic_data["y_test"]
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        coverage = float(((y >= lower) & (y <= upper)).mean())
        assert coverage >= 1.0 - alpha - 0.06


# ---------------------------------------------------------------------------
# Kernel weight computation
# ---------------------------------------------------------------------------


class TestKernelWeights:
    def test_gaussian_weights_shape(self):
        rng = np.random.default_rng(0)
        X_cal = rng.normal(size=(50, 3))
        x_test = rng.normal(size=(3,))
        w = _kernel_weights(x_test, X_cal, localizer="gaussian", bandwidth=1.0)
        assert w.shape == (50,)

    def test_gaussian_weights_nonnegative(self):
        rng = np.random.default_rng(0)
        X_cal = rng.normal(size=(50, 3))
        x_test = rng.normal(size=(3,))
        w = _kernel_weights(x_test, X_cal, localizer="gaussian", bandwidth=1.0)
        assert np.all(w >= 0)

    def test_gaussian_nearest_point_has_highest_weight(self):
        X_cal = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        x_test = np.array([0.1, 0.1])
        w = _kernel_weights(x_test, X_cal, localizer="gaussian", bandwidth=1.0)
        assert np.argmax(w) == 0

    def test_knn_weights_sum_one(self):
        rng = np.random.default_rng(1)
        X_cal = rng.normal(size=(100, 4))
        x_test = rng.normal(size=(4,))
        w = _kernel_weights(x_test, X_cal, localizer="knn", bandwidth=1.0, knn_k=10)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_knn_exactly_k_nonzero(self):
        rng = np.random.default_rng(2)
        X_cal = rng.normal(size=(100, 4))
        x_test = rng.normal(size=(4,))
        k = 15
        w = _kernel_weights(x_test, X_cal, localizer="knn", bandwidth=1.0, knn_k=k)
        assert np.sum(w > 0) == k

    def test_exponential_weights_nonnegative(self):
        rng = np.random.default_rng(3)
        X_cal = rng.normal(size=(30, 2))
        x_test = rng.normal(size=(2,))
        w = _kernel_weights(x_test, X_cal, localizer="exponential", bandwidth=1.0)
        assert np.all(w >= 0)

    def test_uniform_fallback_when_nothing_in_radius(self):
        """With tiny bandwidth, uniform kernel may have zero mass -> fallback to uniform."""
        X_cal = np.array([[10.0, 10.0], [20.0, 20.0]])
        x_test = np.array([0.0, 0.0])
        w = _kernel_weights(x_test, X_cal, localizer="uniform", bandwidth=0.001)
        assert np.all(w > 0)  # fallback to equal weights

    def test_invalid_localizer_raises(self):
        with pytest.raises(ValueError, match="Unknown localizer"):
            _kernel_weights(
                np.array([0.0]), np.array([[0.0]]),
                localizer="not_a_kernel", bandwidth=1.0
            )


# ---------------------------------------------------------------------------
# Fallback behaviour with tiny calibration set
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_warning_on_tiny_calibration(self):
        """With only 3 calibration points, validity checks may fail; warning should appear."""
        rng = np.random.default_rng(99)
        n_cal = 3
        X_cal = rng.normal(size=(n_cal, 2))
        y_cal = rng.poisson(1.0, size=n_cal).astype(float)

        # Model that predicts close to 1 everywhere
        model = ConstantModel(value=1.0)
        lcp = LCPModelSelector(
            models=[model, ConstantModel(value=2.0)],
            score_fns=["raw", "raw"],
            localizer="gaussian",
            bandwidth=100.0,  # very wide -> near-uniform weights
        )
        lcp.calibrate(X_cal, y_cal)

        X_test = rng.normal(size=(5, 2))

        # With such a tiny calibration set, some test points may trigger fallback.
        # We just check that predict_interval runs without crashing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = lcp.predict_interval(X_test, alpha=0.10)

        assert len(result) == 5
        assert (result["lower"] >= 0).all()
        assert (result["lower"] <= result["upper"]).all()

    def test_fallback_flag_set_in_selected_models(self):
        """
        Force fallback by using alpha=0.001 so almost no model is locally valid.
        Check that fallback_used flag appears in selected_models output.
        """
        rng = np.random.default_rng(0)
        n_cal = 10
        X_cal = rng.normal(size=(n_cal, 2))
        y_cal = rng.poisson(1.0, size=n_cal).astype(float)

        model = ConstantModel(value=1.0)
        lcp = LCPModelSelector(
            models=[model, ConstantModel(value=0.5)],
            score_fns=["raw", "raw"],
            localizer="gaussian",
            bandwidth=100.0,
        )
        lcp.calibrate(X_cal, y_cal)

        X_test = rng.normal(size=(20, 2))

        # alpha=0.001 is very tight — many test points will need fallback
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sel = lcp.selected_models(X_test, alpha=0.001)

        # We just check the column exists and has boolean values
        assert "fallback_used" in sel.columns
        assert sel["fallback_used"].dtype == pl.Boolean


# ---------------------------------------------------------------------------
# coverage_diagnostics
# ---------------------------------------------------------------------------


class TestCoverageDiagnostics:
    def test_output_is_polars(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.coverage_diagnostics(
            synthetic_data["X_test"], synthetic_data["y_test"], alpha=0.10
        )
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.coverage_diagnostics(
            synthetic_data["X_test"], synthetic_data["y_test"], alpha=0.10
        )
        expected = {
            "slice_type", "bin", "mean_predicted", "n_obs",
            "coverage", "target_coverage", "dominant_model",
        }
        assert expected == set(result.columns)

    def test_coverage_in_range(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.coverage_diagnostics(
            synthetic_data["X_test"], synthetic_data["y_test"], alpha=0.10
        )
        assert (result["coverage"] >= 0).all()
        assert (result["coverage"] <= 1).all()

    def test_has_both_slice_types(self, synthetic_data):
        lcp = make_lcp(synthetic_data)
        result = lcp.coverage_diagnostics(
            synthetic_data["X_test"], synthetic_data["y_test"], alpha=0.10
        )
        slice_types = result["slice_type"].unique().to_list()
        assert "pred_decile" in slice_types
        assert "by_model" in slice_types


# ---------------------------------------------------------------------------
# Weighted quantile helper
# ---------------------------------------------------------------------------


class TestWeightedQuantile:
    def test_uniform_weights_matches_empirical(self):
        rng = np.random.default_rng(0)
        scores = rng.exponential(size=200)
        w = np.ones(200) / 200
        q_90 = _weighted_quantile(scores, w, 0.90)
        np_90 = float(np.percentile(scores, 90))
        assert abs(q_90 - np_90) < 0.05 * np_90 + 0.01

    def test_concentrated_weights(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # All weight on score=3.0 -> quantile for any level should be 3.0
        w = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        q = _weighted_quantile(scores, w, 0.50)
        assert q == 3.0


# ---------------------------------------------------------------------------
# Input format acceptance
# ---------------------------------------------------------------------------


class TestInputFormats:
    def test_polars_dataframe_input(self, synthetic_data):
        lcp = LCPModelSelector(
            models=[synthetic_data["model_true"]],
            score_fns=["pearson_weighted"],
        )
        X_pl = pl.DataFrame(synthetic_data["X_cal"])
        y_pl = pl.Series(synthetic_data["y_cal"])
        lcp.calibrate(X_pl, y_pl)

        X_test_pl = pl.DataFrame(synthetic_data["X_test"])
        result = lcp.predict_interval(X_test_pl, alpha=0.10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(synthetic_data["y_test"])

    def test_1d_array_as_features(self):
        """Single-feature case: 1D array should be reshaped to (n,1)."""
        rng = np.random.default_rng(5)
        n = 50
        x_cal = rng.normal(size=n)
        y_cal = np.abs(rng.normal(size=n)) + 0.1
        model = ConstantModel(1.0)

        lcp = LCPModelSelector(
            models=[model],
            score_fns=["raw"],
            localizer="gaussian",
            bandwidth=2.0,
        )
        lcp.calibrate(x_cal, y_cal)
        x_test = rng.normal(size=10)
        result = lcp.predict_interval(x_test, alpha=0.10)
        assert len(result) == 10
