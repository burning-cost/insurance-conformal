"""
Tests for ConformalisedQuantileRegression (CQR).

CQR wraps a pair of quantile models (lower and upper quantile) and uses a
conformal calibration step to correct for systematic miscoverage. The key
property to verify: for any alpha and any exchangeable data generating process,
the empirical coverage on a fresh test set must be >= 1 - alpha.

We use a simple heteroscedastic regression DGP where the true conditional
quantiles are known analytically. This lets us test both the score computation
and the coverage guarantee without needing a real GBM.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal.cqr import ConformalisedQuantileRegression, _cqr_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TrueQuantileModel:
    """
    Oracle quantile model that returns the exact conditional quantiles.

    For Y ~ N(mu(X), sigma(X)^2), the alpha-quantile is
    mu(X) + z_alpha * sigma(X). We use this to test CQR without model error.
    """

    def __init__(self, alpha: float, mu_fn, sigma_fn) -> None:
        from scipy.stats import norm
        self.z = norm.ppf(alpha)
        self.mu_fn = mu_fn
        self.sigma_fn = sigma_fn

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return self.mu_fn(X) + self.z * self.sigma_fn(X)


class ConstantQuantileModel:
    """Returns a fixed constant for all inputs."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)


class LinearQuantileModel:
    """Returns beta0 + X @ beta for all inputs."""

    def __init__(self, intercept: float, slope: float) -> None:
        self.intercept = intercept
        self.slope = slope

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return self.intercept + X[:, 0] * self.slope


def _make_data(n: int = 1000, seed: int = 0) -> dict:
    """
    Heteroscedastic regression DGP: Y ~ N(2*x1, (0.5 + x1)^2).
    Features: x1 ~ Uniform(0, 2).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 2.0, size=n)
    X = x1.reshape(-1, 1)
    mu = 2.0 * x1
    sigma = 0.5 + x1
    y = rng.normal(mu, sigma)

    def mu_fn(X):
        return 2.0 * X[:, 0]

    def sigma_fn(X):
        return 0.5 + X[:, 0]

    return {"X": X, "y": y, "mu_fn": mu_fn, "sigma_fn": sigma_fn, "rng": rng}


@pytest.fixture(scope="module")
def hetero_data():
    return _make_data(n=2000, seed=42)


# ---------------------------------------------------------------------------
# Unit tests for _cqr_score
# ---------------------------------------------------------------------------


class TestCqrScore:
    def test_score_negative_when_inside(self):
        """Observation inside [q_lo, q_hi] gives negative score."""
        y = np.array([1.5])
        q_lo = np.array([1.0])
        q_hi = np.array([2.0])
        score = _cqr_score(y, q_lo, q_hi)
        assert score[0] < 0

    def test_score_zero_at_boundary(self):
        """Observation exactly at upper bound gives score = 0."""
        y = np.array([2.0])
        q_lo = np.array([1.0])
        q_hi = np.array([2.0])
        score = _cqr_score(y, q_lo, q_hi)
        assert score[0] == pytest.approx(0.0)

    def test_score_positive_when_outside_upper(self):
        """Observation above q_hi gives positive score."""
        y = np.array([3.0])
        q_lo = np.array([1.0])
        q_hi = np.array([2.0])
        score = _cqr_score(y, q_lo, q_hi)
        assert score[0] == pytest.approx(1.0)

    def test_score_positive_when_outside_lower(self):
        """Observation below q_lo gives positive score."""
        y = np.array([0.5])
        q_lo = np.array([1.0])
        q_hi = np.array([2.0])
        score = _cqr_score(y, q_lo, q_hi)
        assert score[0] == pytest.approx(0.5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            _cqr_score(np.array([1.0, 2.0]), np.array([0.5]), np.array([1.5]))

    def test_batch_shape_preserved(self):
        n = 50
        rng = np.random.default_rng(0)
        y = rng.normal(size=n)
        q_lo = y - 1.0
        q_hi = y + 1.0
        scores = _cqr_score(y, q_lo, q_hi)
        assert scores.shape == (n,)


# ---------------------------------------------------------------------------
# Initialisation and state tests
# ---------------------------------------------------------------------------


class TestInitialisation:
    def test_not_calibrated_on_init(self):
        cqr = ConformalisedQuantileRegression(
            model_lo=ConstantQuantileModel(0.0),
            model_hi=ConstantQuantileModel(1.0),
        )
        assert not cqr.is_calibrated_
        assert cqr.n_calibration_ == 0

    def test_predict_raises_before_calibration(self):
        cqr = ConformalisedQuantileRegression(
            model_lo=ConstantQuantileModel(0.0),
            model_hi=ConstantQuantileModel(1.0),
        )
        X = np.ones((5, 2))
        with pytest.raises(RuntimeError, match="not been calibrated"):
            cqr.predict_interval(X, alpha=0.10)

    def test_repr_before_calibration(self):
        cqr = ConformalisedQuantileRegression(
            model_lo=ConstantQuantileModel(0.0),
            model_hi=ConstantQuantileModel(1.0),
        )
        r = repr(cqr)
        assert "not calibrated" in r

    def test_repr_after_calibration(self):
        cqr = ConformalisedQuantileRegression(
            model_lo=ConstantQuantileModel(0.0),
            model_hi=ConstantQuantileModel(1.0),
        )
        X = np.ones((20, 2))
        y = np.ones(20) * 0.5
        cqr.calibrate(X, y)
        r = repr(cqr)
        assert "20" in r

    def test_calibrate_sets_state(self):
        cqr = ConformalisedQuantileRegression(
            model_lo=ConstantQuantileModel(-1.0),
            model_hi=ConstantQuantileModel(1.0),
        )
        X = np.zeros((30, 1))
        y = np.zeros(30)
        cqr.calibrate(X, y)
        assert cqr.is_calibrated_
        assert cqr.n_calibration_ == 30
        assert cqr.cal_scores_ is not None
        assert len(cqr.cal_scores_) == 30

    def test_length_mismatch_raises(self):
        cqr = ConformalisedQuantileRegression(
            model_lo=ConstantQuantileModel(0.0),
            model_hi=ConstantQuantileModel(1.0),
        )
        X = np.ones((10, 2))
        y = np.ones(8)
        with pytest.raises(ValueError, match="lengths must match"):
            cqr.calibrate(X, y)

    def test_invalid_alpha_raises(self):
        cqr = ConformalisedQuantileRegression(
            model_lo=ConstantQuantileModel(0.0),
            model_hi=ConstantQuantileModel(1.0),
        )
        cqr.calibrate(np.ones((20, 1)), np.ones(20) * 0.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            cqr.predict_interval(np.ones((5, 1)), alpha=1.5)


# ---------------------------------------------------------------------------
# Coverage guarantee tests
# ---------------------------------------------------------------------------


class TestCoverageGuarantee:
    """
    The defining property of CQR: marginal coverage >= 1 - alpha.

    We use oracle quantile models (true conditional quantiles) so that any
    coverage shortfall comes only from finite-sample variation, not model error.
    We allow a 3pp tolerance below the nominal level.
    """

    def test_coverage_alpha_10(self, hetero_data):
        alpha = 0.10
        d = hetero_data
        n = len(d["y"])
        # Split: 60% train (used to fit quantile models, not needed here since
        # we use oracle models), 20% calibration, 20% test
        cal_end = int(n * 0.8)
        X_cal, y_cal = d["X"][:cal_end], d["y"][:cal_end]
        X_test, y_test = d["X"][cal_end:], d["y"][cal_end:]

        lo = TrueQuantileModel(alpha / 2, d["mu_fn"], d["sigma_fn"])
        hi = TrueQuantileModel(1 - alpha / 2, d["mu_fn"], d["sigma_fn"])

        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)
        cqr.calibrate(X_cal, y_cal)
        intervals = cqr.predict_interval(X_test, alpha=alpha)

        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))

        assert coverage >= (1 - alpha) - 0.03, (
            f"Coverage {coverage:.3f} is below target {1 - alpha:.3f} - 3pp tolerance"
        )

    def test_coverage_alpha_05(self, hetero_data):
        alpha = 0.05
        d = hetero_data
        n = len(d["y"])
        cal_end = int(n * 0.8)
        X_cal, y_cal = d["X"][:cal_end], d["y"][:cal_end]
        X_test, y_test = d["X"][cal_end:], d["y"][cal_end:]

        lo = TrueQuantileModel(alpha / 2, d["mu_fn"], d["sigma_fn"])
        hi = TrueQuantileModel(1 - alpha / 2, d["mu_fn"], d["sigma_fn"])

        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)
        cqr.calibrate(X_cal, y_cal)
        intervals = cqr.predict_interval(X_test, alpha=alpha)

        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))

        assert coverage >= (1 - alpha) - 0.03

    def test_correction_is_finite(self, hetero_data):
        """The conformal correction quantile should always be finite."""
        d = hetero_data
        lo = TrueQuantileModel(0.05, d["mu_fn"], d["sigma_fn"])
        hi = TrueQuantileModel(0.95, d["mu_fn"], d["sigma_fn"])
        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)
        cqr.calibrate(d["X"], d["y"])
        q = cqr.correction_quantile(0.10)
        assert np.isfinite(q)


# ---------------------------------------------------------------------------
# Interval structure tests
# ---------------------------------------------------------------------------


class TestIntervalStructure:
    def test_output_columns(self, hetero_data):
        d = hetero_data
        lo = TrueQuantileModel(0.05, d["mu_fn"], d["sigma_fn"])
        hi = TrueQuantileModel(0.95, d["mu_fn"], d["sigma_fn"])
        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)
        cqr.calibrate(d["X"][:800], d["y"][:800])
        result = cqr.predict_interval(d["X"][800:], alpha=0.10)

        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"lower", "q_lo", "q_hi", "upper"}

    def test_upper_geq_lower(self, hetero_data):
        d = hetero_data
        lo = TrueQuantileModel(0.05, d["mu_fn"], d["sigma_fn"])
        hi = TrueQuantileModel(0.95, d["mu_fn"], d["sigma_fn"])
        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)
        cqr.calibrate(d["X"][:800], d["y"][:800])
        result = cqr.predict_interval(d["X"][800:], alpha=0.10)

        assert (result["upper"] >= result["lower"]).all()

    def test_clip_lower_enforced(self):
        """Lower bound must be clipped at clip_lower=0.0 (insurance default)."""
        lo_model = ConstantQuantileModel(-5.0)  # very negative lower quantile
        hi_model = ConstantQuantileModel(5.0)
        cqr = ConformalisedQuantileRegression(
            model_lo=lo_model, model_hi=hi_model, clip_lower=0.0
        )
        X = np.ones((20, 1))
        y = np.ones(20)
        cqr.calibrate(X, y)
        result = cqr.predict_interval(np.ones((10, 1)), alpha=0.10)
        assert (result["lower"] >= 0.0).all()

    def test_heteroscedastic_width(self, hetero_data):
        """
        CQR intervals should be wider for high-variance observations.

        DGP: sigma(x) = 0.5 + x1. Intervals for x1 ~ 1.8 should be
        wider than for x1 ~ 0.2.
        """
        d = hetero_data
        lo = TrueQuantileModel(0.05, d["mu_fn"], d["sigma_fn"])
        hi = TrueQuantileModel(0.95, d["mu_fn"], d["sigma_fn"])
        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)
        cqr.calibrate(d["X"][:1600], d["y"][:1600])

        X_low = np.array([[0.2]] * 100)
        X_high = np.array([[1.8]] * 100)

        res_low = cqr.predict_interval(X_low, alpha=0.10)
        res_high = cqr.predict_interval(X_high, alpha=0.10)

        width_low = (res_low["upper"] - res_low["lower"]).mean()
        width_high = (res_high["upper"] - res_high["lower"]).mean()

        assert width_high > width_low, (
            f"Expected wider intervals for high-variance x1=1.8 (width={width_high:.3f}) "
            f"than x1=0.2 (width={width_low:.3f})"
        )

    def test_polars_input(self, hetero_data):
        """CQR must accept Polars DataFrames as input."""
        d = hetero_data
        lo = TrueQuantileModel(0.05, d["mu_fn"], d["sigma_fn"])
        hi = TrueQuantileModel(0.95, d["mu_fn"], d["sigma_fn"])
        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)

        X_pl = pl.DataFrame({"x1": d["X"][:800, 0]})
        y_pl = pl.Series("y", d["y"][:800])
        cqr.calibrate(X_pl, y_pl)

        X_test_pl = pl.DataFrame({"x1": d["X"][800:900, 0]})
        result = cqr.predict_interval(X_test_pl, alpha=0.10)
        assert len(result) == 100

    def test_recalibrate_clears_cache(self):
        """Re-calibrating on a different dataset should update the correction."""
        lo = ConstantQuantileModel(0.0)
        hi = ConstantQuantileModel(1.0)
        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)

        # First calibration: scores all around 0.5
        y1 = np.ones(50) * 0.5
        cqr.calibrate(np.zeros((50, 1)), y1)
        q1 = cqr.correction_quantile(0.10)

        # Second calibration: scores much larger (y outside [0, 1])
        y2 = np.ones(50) * 3.0
        cqr.calibrate(np.zeros((50, 1)), y2)
        q2 = cqr.correction_quantile(0.10)

        assert q2 > q1, "Re-calibration on worse-covered data should increase correction"

    def test_coverage_by_decile_returns_dataframe(self, hetero_data):
        d = hetero_data
        lo = TrueQuantileModel(0.05, d["mu_fn"], d["sigma_fn"])
        hi = TrueQuantileModel(0.95, d["mu_fn"], d["sigma_fn"])
        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi, clip_lower=-np.inf)
        cqr.calibrate(d["X"][:1500], d["y"][:1500])
        diag = cqr.coverage_by_decile(d["X"][1500:], d["y"][1500:], alpha=0.10)

        assert isinstance(diag, pl.DataFrame)
        expected_cols = {"decile", "mean_midpoint", "mean_width", "n_obs", "coverage", "target_coverage"}
        assert set(diag.columns) == expected_cols
        assert len(diag) <= 10
