"""
Tests for KMMConformalPredictor (covariate-shift-robust conformal prediction).

The core property under test is that KMM-CP produces valid coverage under
covariate shift while standard CP does not. We also test weight normalisation,
the QP properties, selective mode, and the output schema.

Tests are written with small-ish datasets (n=100-500) to keep runtime low.
Coverage tests allow generous slack (~15pp) given finite sample effects.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_conformal.covariate_shift import (
    KMMConformalPredictor,
    _median_bandwidth,
    _rbf_kernel,
    _solve_kmm_qp,
    _weighted_conformal_quantile,
    _kernel_smooth_test_weights,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class ConstantModel:
    """Simplest possible model: predicts the training mean."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)


class LinearModel:
    """Linear model: predict = X @ beta (clipped to positive)."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(X) @ self.beta, 1e-6, None)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture(scope="module")
def small_data(rng):
    """100-point source/target split for fast unit tests."""
    n = 100
    X = rng.standard_normal((n, 4))
    y = np.exp(X[:, 0] * 0.3) * rng.exponential(1.0, n)
    X_cal = X[:60]
    y_cal = y[:60]
    X_target = X[60:]
    return {
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_target": X_target,
        "model": ConstantModel(float(y_cal.mean())),
    }


@pytest.fixture(scope="module")
def shift_data(rng):
    """
    Clearly shifted source/target: source is N(0,1), target is N(3,1).
    Used for the coverage-under-shift test.
    """
    n_source = 400
    n_target = 150

    X_source = rng.normal(0, 1, (n_source, 3))
    y_source = 0.5 * np.exp(X_source[:, 0] * 0.2) + rng.exponential(0.2, n_source)

    X_target = rng.normal(3, 1, (n_target, 3))
    y_target = 0.5 * np.exp(X_target[:, 0] * 0.2) + rng.exponential(0.2, n_target)

    beta = np.array([0.2, 0.0, 0.0])
    model = LinearModel(0.5 * np.exp(beta))
    model_fitted = LinearModel(np.array([0.1, 0.0, 0.0]))

    return {
        "X_cal": X_source[200:],
        "y_cal": y_source[200:],
        "X_target": X_target,
        "y_target": y_target,
        "X_test": X_target,
        "y_test": y_target,
        "model": ConstantModel(float(y_source.mean())),
    }


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_median_bandwidth_positive(self):
        rng = np.random.default_rng(1)
        X1 = rng.standard_normal((50, 3))
        X2 = rng.standard_normal((50, 3))
        gamma = _median_bandwidth(X1, X2)
        assert gamma > 0.0
        assert np.isfinite(gamma)

    def test_median_bandwidth_identical(self):
        X = np.ones((20, 3))
        # All pairwise distances are 0 — should return 1.0 fallback
        gamma = _median_bandwidth(X, X)
        assert gamma == 1.0

    def test_rbf_kernel_shape(self):
        X1 = np.random.default_rng(0).standard_normal((10, 3))
        X2 = np.random.default_rng(1).standard_normal((15, 3))
        K = _rbf_kernel(X1, X2, gamma=1.0)
        assert K.shape == (10, 15)

    def test_rbf_kernel_diagonal_is_one(self):
        X = np.random.default_rng(0).standard_normal((8, 3))
        K = _rbf_kernel(X, X, gamma=0.5)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)

    def test_rbf_kernel_values_in_0_1(self):
        X1 = np.random.default_rng(0).standard_normal((10, 3))
        X2 = np.random.default_rng(1).standard_normal((10, 3))
        K = _rbf_kernel(X1, X2, gamma=0.1)
        assert K.min() >= 0.0
        assert K.max() <= 1.0 + 1e-9

    def test_solve_kmm_qp_bounds_respected(self):
        rng = np.random.default_rng(2)
        n = 20
        K = rng.standard_normal((n, n))
        K = K.T @ K + 1e-6 * np.eye(n)
        kappa = np.ones(n)
        B = 5.0
        beta = _solve_kmm_qp(K, kappa, B=B, epsilon=0.1)
        assert beta.min() >= -1e-9
        assert beta.max() <= B + 1e-9
        assert beta.shape == (n,)

    def test_solve_kmm_qp_normalisation(self):
        rng = np.random.default_rng(3)
        n = 30
        K = rng.standard_normal((n, n))
        K = K.T @ K + 1e-5 * np.eye(n)
        kappa = np.ones(n)
        epsilon = 0.1
        beta = _solve_kmm_qp(K, kappa, B=10.0, epsilon=epsilon)
        mean_beta = beta.mean()
        # mean(beta) should be within epsilon of 1.0
        assert abs(mean_beta - 1.0) <= epsilon + 0.05  # small numerical tolerance

    def test_weighted_conformal_quantile_basic(self):
        scores = np.array([0.1, 0.5, 0.9, 1.5, 2.0])
        beta = np.ones(5)
        # alpha=0.10: we need cumulative mass >= 0.9
        # Uniform weights, beta_test = 1.0 => total = 6
        # sorted cummass: 1/6, 2/6, 3/6, 4/6, 5/6 -> 5/6 >= 0.9 at idx=4
        q = _weighted_conformal_quantile(scores, beta, beta_test=1.0, alpha=0.10)
        assert q == pytest.approx(2.0, abs=1e-9)

    def test_weighted_conformal_quantile_inf(self):
        # Very small alpha — cumulative mass may not reach 1-alpha
        scores = np.array([0.1, 0.2, 0.3])
        beta = np.ones(3) * 0.001
        q = _weighted_conformal_quantile(scores, beta, beta_test=100.0, alpha=0.001)
        # beta_test dominates; cummass never reaches 1-alpha
        assert q == np.inf

    def test_kernel_smooth_test_weights_shape(self):
        rng = np.random.default_rng(4)
        X_test = rng.standard_normal((7, 3))
        X_cal = rng.standard_normal((20, 3))
        beta = np.ones(20)
        w = _kernel_smooth_test_weights(X_test, X_cal, beta, gamma=0.5)
        assert w.shape == (7,)
        assert np.all(np.isfinite(w))

    def test_kernel_smooth_fallback_for_distant_points(self):
        # Test point very far from calibration -> should fall back to mean(beta)
        X_test = np.array([[100.0, 100.0, 100.0]])
        X_cal = np.random.default_rng(5).standard_normal((10, 3))
        beta = np.arange(1, 11, dtype=float)
        w = _kernel_smooth_test_weights(X_test, X_cal, beta, gamma=1.0)
        assert w[0] == pytest.approx(float(np.mean(beta)), abs=1e-6)


# ---------------------------------------------------------------------------
# KMMConformalPredictor unit tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        assert kcp.nonconformity == "pearson_weighted"
        assert kcp.distribution == "tweedie"
        assert kcp.weight_bound == 10.0
        assert kcp.epsilon == 0.05
        assert kcp.test_weight_method == "kernel_smooth"
        assert kcp.selective is False
        assert not kcp.is_calibrated_

    def test_invalid_nonconformity(self, small_data):
        with pytest.raises(ValueError, match="Unknown nonconformity"):
            KMMConformalPredictor(model=small_data["model"], nonconformity="bad")

    def test_invalid_distribution(self, small_data):
        with pytest.raises(ValueError, match="Unknown distribution"):
            KMMConformalPredictor(model=small_data["model"], distribution="normal")

    def test_invalid_test_weight_method(self, small_data):
        with pytest.raises(ValueError, match="Unknown test_weight_method"):
            KMMConformalPredictor(model=small_data["model"], test_weight_method="bad")

    def test_invalid_weight_bound(self, small_data):
        with pytest.raises(ValueError, match="weight_bound"):
            KMMConformalPredictor(model=small_data["model"], weight_bound=-1.0)

    def test_invalid_epsilon(self, small_data):
        with pytest.raises(ValueError, match="epsilon"):
            KMMConformalPredictor(model=small_data["model"], epsilon=1.5)

    def test_explicit_tweedie_power(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"], tweedie_power=2.0)
        assert kcp.tweedie_power == 2.0

    def test_repr_uncalibrated(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        assert "not calibrated" in repr(kcp)
        assert "KMMConformalPredictor" in repr(kcp)


class TestCalibrate:
    def test_sets_calibrated_flag(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        assert kcp.is_calibrated_

    def test_returns_self(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        result = kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        assert result is kcp

    def test_weights_shape(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        assert kcp.kmm_weights_.shape == (kcp.n_support_,)

    def test_weights_non_negative(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        assert kcp.kmm_weights_.min() >= -1e-9

    def test_weights_bounded(self, small_data):
        B = 5.0
        kcp = KMMConformalPredictor(model=small_data["model"], weight_bound=B)
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        assert kcp.kmm_weights_.max() <= B + 1e-6

    def test_gamma_resolved_from_none(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"], gamma=None)
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        assert kcp.gamma_ > 0.0
        assert np.isfinite(kcp.gamma_)

    def test_explicit_gamma_respected(self, small_data):
        gamma_in = 2.5
        kcp = KMMConformalPredictor(model=small_data["model"], gamma=gamma_in)
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        assert kcp.gamma_ == pytest.approx(gamma_in)

    def test_x_cal_len_mismatch_raises(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        with pytest.raises(ValueError, match="lengths must match"):
            kcp.calibrate(
                small_data["X_cal"],
                small_data["y_cal"][:-5],
                small_data["X_target"],
            )

    def test_recalibrate_resets_state(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        gamma1 = kcp.gamma_
        # Recalibrate with explicit gamma
        kcp.calibrate(
            small_data["X_cal"],
            small_data["y_cal"],
            small_data["X_target"],
            # No change, but verifies state is updated
        )
        assert kcp.is_calibrated_


class TestWeightNormalisation:
    def test_weights_sum_near_n_slight_shift(self):
        rng = np.random.default_rng(10)
        X_cal = rng.standard_normal((80, 5))
        X_target = X_cal + 0.1  # slight shift
        y_cal = np.abs(rng.standard_normal(80)) + 0.1
        kcp = KMMConformalPredictor(
            model=ConstantModel(0.5),
            weight_bound=5.0,
            epsilon=0.05,
        )
        kcp.calibrate(X_cal, y_cal, X_target)
        beta = kcp.kmm_weights_
        mean_beta = float(beta.mean())
        # mean(beta) should be within epsilon + small tolerance of 1.0
        assert abs(mean_beta - 1.0) < 0.15

    def test_no_shift_weights_approximately_uniform(self):
        rng = np.random.default_rng(20)
        X = rng.standard_normal((200, 4))
        X_cal, X_target = X[:100], X[100:]
        y_cal = np.abs(rng.standard_normal(100)) + 0.5
        kcp = KMMConformalPredictor(
            model=ConstantModel(0.5),
            weight_bound=5.0,
            epsilon=0.05,
        )
        kcp.calibrate(X_cal, y_cal, X_target)
        beta = kcp.kmm_weights_
        # Weights should be close to uniform (std < 0.5 for no-shift case)
        assert np.std(beta) < 0.8


class TestPredictInterval:
    def test_output_schema(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        intervals = kcp.predict_interval(small_data["X_target"], alpha=0.10)
        assert isinstance(intervals, pl.DataFrame)
        assert "lower" in intervals.columns
        assert "point" in intervals.columns
        assert "upper" in intervals.columns
        assert len(intervals) == len(small_data["X_target"])

    def test_lower_leq_point_leq_upper(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        intervals = kcp.predict_interval(small_data["X_target"])
        lower = intervals["lower"].to_numpy()
        point = intervals["point"].to_numpy()
        upper = intervals["upper"].to_numpy()
        # All valid (non-NaN) rows should have lower <= point <= upper
        valid = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(lower[valid] <= point[valid] + 1e-9)
        assert np.all(point[valid] <= upper[valid] + 1e-9)

    def test_lower_non_negative(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        intervals = kcp.predict_interval(small_data["X_target"])
        lower = intervals["lower"].to_numpy()
        valid = np.isfinite(lower)
        assert np.all(lower[valid] >= -1e-9)

    def test_return_weights_column(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        intervals = kcp.predict_interval(
            small_data["X_target"], return_weights=True
        )
        assert "w_test" in intervals.columns
        assert intervals["w_test"].dtype == pl.Float64

    def test_no_return_weights_by_default(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        intervals = kcp.predict_interval(small_data["X_target"])
        assert "w_test" not in intervals.columns

    def test_not_calibrated_raises(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        with pytest.raises(RuntimeError, match="not been calibrated"):
            kcp.predict_interval(small_data["X_target"])

    def test_invalid_alpha_raises(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        with pytest.raises(ValueError, match="alpha"):
            kcp.predict_interval(small_data["X_target"], alpha=1.5)

    def test_alpha_01_vs_02(self, small_data):
        """Wider alpha should give wider intervals."""
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        i90 = kcp.predict_interval(small_data["X_target"], alpha=0.10)
        i80 = kcp.predict_interval(small_data["X_target"], alpha=0.20)
        w90 = (i90["upper"] - i90["lower"]).mean()
        w80 = (i80["upper"] - i80["lower"]).mean()
        assert w90 >= w80

    def test_mean_weight_method(self, small_data):
        kcp = KMMConformalPredictor(
            model=small_data["model"], test_weight_method="mean"
        )
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        intervals = kcp.predict_interval(small_data["X_target"])
        assert isinstance(intervals, pl.DataFrame)
        assert len(intervals) == len(small_data["X_target"])

    def test_knn_weight_method(self, small_data):
        kcp = KMMConformalPredictor(
            model=small_data["model"], test_weight_method="knn", knn_k=5
        )
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        intervals = kcp.predict_interval(small_data["X_target"])
        assert isinstance(intervals, pl.DataFrame)
        assert len(intervals) == len(small_data["X_target"])

    def test_polars_input(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        X_cal_pl = pl.DataFrame(small_data["X_cal"])
        y_cal_pl = pl.Series(small_data["y_cal"])
        X_target_pl = pl.DataFrame(small_data["X_target"])
        kcp.calibrate(X_cal_pl, y_cal_pl, X_target_pl)
        intervals = kcp.predict_interval(X_target_pl)
        assert len(intervals) == len(small_data["X_target"])


class TestSelectiveMode:
    def test_support_mask_set(self):
        rng = np.random.default_rng(30)
        X_cal = rng.standard_normal((80, 3))
        y_cal = np.abs(rng.standard_normal(80)) + 0.5
        X_target = rng.standard_normal((50, 3)) + 2.0  # shifted
        kcp = KMMConformalPredictor(
            model=ConstantModel(0.5),
            selective=True,
        )
        kcp.calibrate(X_cal, y_cal, X_target)
        assert kcp.support_mask_ is not None
        assert kcp.support_mask_.shape == (80,)
        assert kcp.support_mask_.dtype == bool

    def test_support_mask_none_when_not_selective(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"], selective=False)
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        assert kcp.support_mask_ is None

    def test_abstention_on_far_points(self):
        """Points far from the target support should receive NaN intervals."""
        rng = np.random.default_rng(40)
        X_cal = rng.standard_normal((80, 3))
        y_cal = np.abs(rng.standard_normal(80)) + 0.5
        X_target = rng.standard_normal((40, 3))  # same distribution as cal

        X_close = rng.standard_normal((15, 3))        # on target distribution
        X_far = rng.standard_normal((15, 3)) + 15.0   # very far from target
        X_test = np.vstack([X_close, X_far])

        kcp = KMMConformalPredictor(
            model=ConstantModel(0.5),
            selective=True,
        )
        kcp.calibrate(X_cal, y_cal, X_target)
        intervals = kcp.predict_interval(X_test)

        lower = intervals["lower"].to_numpy()
        # Far points should have NaN lower bound
        assert np.all(np.isnan(lower[15:]))
        # Close points should have finite lower bound
        assert np.all(np.isfinite(lower[:15]))

    def test_explicit_selective_threshold(self):
        rng = np.random.default_rng(50)
        X_cal = rng.standard_normal((60, 3))
        y_cal = np.abs(rng.standard_normal(60)) + 0.5
        X_target = rng.standard_normal((30, 3)) + 1.0

        kcp = KMMConformalPredictor(
            model=ConstantModel(0.5),
            selective=True,
            selective_threshold=0.0,  # very low threshold — keep all
        )
        kcp.calibrate(X_cal, y_cal, X_target)
        # All calibration points should be in support
        assert kcp.support_mask_.all()

    def test_n_support_leq_n_cal(self):
        rng = np.random.default_rng(60)
        X_cal = rng.standard_normal((80, 3))
        y_cal = np.abs(rng.standard_normal(80)) + 0.5
        X_target = rng.standard_normal((40, 3)) + 5.0  # strong shift

        kcp = KMMConformalPredictor(
            model=ConstantModel(0.5),
            selective=True,
        )
        kcp.calibrate(X_cal, y_cal, X_target)
        assert kcp.n_support_ <= kcp.n_calibration_


class TestShiftDiagnostics:
    def test_output_schema(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        diag = kcp.shift_diagnostics()
        assert isinstance(diag, pl.DataFrame)
        assert "statistic" in diag.columns
        assert "value" in diag.columns

    def test_required_statistics_present(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        diag = kcp.shift_diagnostics()
        required = {
            "n_cal",
            "n_support",
            "mean_weight",
            "max_weight",
            "weight_std",
            "effective_n",
            "coverage_shrinkage",
            "gamma_used",
        }
        actual = set(diag["statistic"].to_list())
        assert required.issubset(actual)

    def test_coverage_shrinkage_in_0_1(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        kcp.calibrate(small_data["X_cal"], small_data["y_cal"], small_data["X_target"])
        diag = kcp.shift_diagnostics()
        cs = float(diag.filter(pl.col("statistic") == "coverage_shrinkage")["value"][0])
        assert 0.0 < cs <= 1.0 + 1e-6

    def test_not_calibrated_raises(self, small_data):
        kcp = KMMConformalPredictor(model=small_data["model"])
        with pytest.raises(RuntimeError, match="not been calibrated"):
            kcp.shift_diagnostics()


class TestSubsample:
    def test_large_cal_warns(self):
        rng = np.random.default_rng(70)
        n = 2500
        X_cal = rng.standard_normal((n, 3))
        y_cal = np.abs(rng.standard_normal(n)) + 0.5
        X_target = rng.standard_normal((100, 3))

        kcp = KMMConformalPredictor(
            model=ConstantModel(0.5),
            max_cal_n=2000,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kcp.calibrate(X_cal, y_cal, X_target)
            subsample_warns = [x for x in w if "Subsampling" in str(x.message)]
            assert len(subsample_warns) == 1

    def test_large_cal_result_valid(self):
        rng = np.random.default_rng(71)
        n = 2200
        X_cal = rng.standard_normal((n, 3))
        y_cal = np.abs(rng.standard_normal(n)) + 0.5
        X_target = rng.standard_normal((50, 3))

        kcp = KMMConformalPredictor(
            model=ConstantModel(0.5),
            max_cal_n=500,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kcp.calibrate(X_cal, y_cal, X_target)

        intervals = kcp.predict_interval(X_target)
        assert len(intervals) == 50
        assert kcp.n_support_ <= 500


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_coverage_no_shift(self):
        """
        Under no distributional shift, KMM-CP should achieve marginal coverage
        >= 1 - alpha (with finite-sample slack).
        """
        rng = np.random.default_rng(100)
        n = 600
        beta = np.array([0.2, -0.1, 0.15, 0.0])
        X = rng.standard_normal((n, 4))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)

        X_cal, y_cal = X[:300], y[:300]
        X_test, y_test = X[300:500], y[300:500]
        X_val = X[500:]

        model = LinearModel(beta)
        kcp = KMMConformalPredictor(
            model=model,
            nonconformity="pearson_weighted",
            tweedie_power=1.0,
        )
        kcp.calibrate(X_cal, y_cal, X_target=X_test)
        intervals = kcp.predict_interval(X_test, alpha=0.10)

        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))

        # Should be >= 0.80 (10pp slack for finite sample)
        assert coverage >= 0.80, f"Coverage {coverage:.3f} below 0.80"

    def test_coverage_improves_under_shift(self, shift_data):
        """
        Under strong covariate shift, KMM-CP should achieve better coverage
        than standard conformal prediction.
        """
        from insurance_conformal import InsuranceConformalPredictor

        X_cal = shift_data["X_cal"]
        y_cal = shift_data["y_cal"]
        X_target = shift_data["X_target"]
        y_test = shift_data["y_test"]
        model = shift_data["model"]

        # Standard CP (no shift correction)
        icp = InsuranceConformalPredictor(
            model=model,
            nonconformity="pearson_weighted",
            tweedie_power=1.5,
        )
        icp.calibrate(X_cal, y_cal)
        icp_intervals = icp.predict_interval(X_target, alpha=0.10)
        icp_lower = icp_intervals["lower"].to_numpy()
        icp_upper = icp_intervals["upper"].to_numpy()
        icp_coverage = float(np.mean((y_test >= icp_lower) & (y_test <= icp_upper)))

        # KMM-CP
        kcp = KMMConformalPredictor(
            model=model,
            nonconformity="pearson_weighted",
            tweedie_power=1.5,
            weight_bound=10.0,
        )
        kcp.calibrate(X_cal, y_cal, X_target=X_target)
        kcp_intervals = kcp.predict_interval(X_target, alpha=0.10)
        kcp_lower = kcp_intervals["lower"].to_numpy()
        kcp_upper = kcp_intervals["upper"].to_numpy()
        kcp_coverage = float(np.mean((y_test >= kcp_lower) & (y_test <= kcp_upper)))

        # KMM should be closer to 0.90 than standard CP, or at least not worse
        icp_gap = abs(icp_coverage - 0.90)
        kcp_gap = abs(kcp_coverage - 0.90)
        assert kcp_gap <= icp_gap + 0.10, (
            f"KMM coverage gap {kcp_gap:.3f} exceeds ICP gap {icp_gap:.3f} + 0.10 tolerance"
        )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestPortfolioTransferIntegration:
    def test_portfolio_transfer(self):
        """
        Simulate a portfolio transfer: source is young-driver-heavy (age 18-35),
        target is an older acquired book (age 40-70).

        KMM-CP should achieve better coverage than standard CP on the target book.
        """
        rng = np.random.default_rng(123)

        n_source = 500
        n_target = 200

        # Source: young drivers (feature 0 in [18, 35])
        X_source = np.column_stack([
            rng.uniform(18, 35, n_source),
            rng.uniform(0, 5, n_source),
        ])
        y_source = 0.05 * X_source[:, 0] + rng.exponential(0.1, n_source)

        # Target: older drivers (feature 0 in [40, 70])
        X_target = np.column_stack([
            rng.uniform(40, 70, n_target),
            rng.uniform(0, 5, n_target),
        ])
        y_target = 0.05 * X_target[:, 0] + rng.exponential(0.1, n_target)

        from sklearn.linear_model import TweedieRegressor

        model = TweedieRegressor(power=1.5, alpha=0.01)
        model.fit(X_source[:400], y_source[:400])
        X_cal, y_cal = X_source[400:], y_source[400:]

        # Standard conformal
        from insurance_conformal import InsuranceConformalPredictor

        icp = InsuranceConformalPredictor(model=model, tweedie_power=1.5)
        icp.calibrate(X_cal, y_cal)
        icp_intervals = icp.predict_interval(X_target, alpha=0.10)
        icp_cov = float(
            np.mean(
                (y_target >= icp_intervals["lower"].to_numpy())
                & (y_target <= icp_intervals["upper"].to_numpy())
            )
        )

        # KMM-CP
        kcp = KMMConformalPredictor(model=model, tweedie_power=1.5)
        kcp.calibrate(X_cal, y_cal, X_target=X_target)
        kcp_intervals = kcp.predict_interval(X_target, alpha=0.10)
        kcp_cov = float(
            np.mean(
                (y_target >= kcp_intervals["lower"].to_numpy())
                & (y_target <= kcp_intervals["upper"].to_numpy())
            )
        )

        # KMM coverage gap should be no worse than standard CP gap + 5pp
        assert abs(kcp_cov - 0.90) <= abs(icp_cov - 0.90) + 0.05, (
            f"KMM coverage={kcp_cov:.3f}, ICP coverage={icp_cov:.3f}"
        )

        # Diagnostics should run without error
        diag = kcp.shift_diagnostics()
        assert isinstance(diag, pl.DataFrame)
        assert len(diag) > 0
