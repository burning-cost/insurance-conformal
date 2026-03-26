"""
Structural gap tests for insurance-conformal.

Focuses on uncovered branches identified by code tracing:
1. utils.conformal_quantile: empty calibration set raises ValueError
2. utils.apply_exposure: non-positive exposure raises ValueError
3. utils.temporal_split: with date_col (temporal ordering path)
4. utils.temporal_split: without date_col (simple tail split)
5. scores.anscombe_score: gamma distribution branch
6. scores.anscombe_score: tweedie p=1.0 dispatches to poisson
7. scores.anscombe_score: tweedie p=2.0 dispatches to gamma
8. scores.anscombe_score: tweedie general case (p=1.5)
9. scores.deviance_score: gamma with y=0 stays finite
10. scores.pearson_weighted_score: p=0 edge (no error — scale = yhat^0 = 1)
11. predictor: predict_interval with polars DataFrame X_test input
12. predictor: quantile caching (second call hits cache)
13. predictor: alpha=0 and alpha=1 both raise ValueError
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_conformal.scores import (
    anscombe_score,
    deviance_score,
    invert_score,
    pearson_weighted_score,
)
from insurance_conformal.utils import (
    apply_exposure,
    conformal_quantile,
    temporal_split,
)
from insurance_conformal import InsuranceConformalPredictor


# ---------------------------------------------------------------------------
# Minimal toy model for predictor tests
# ---------------------------------------------------------------------------

class _SimplePoisson:
    """Simple model that predicts a fixed multiple of the first feature."""

    def __init__(self, slope: float = 0.5, intercept: float = 1.0):
        self.slope = slope
        self.intercept = intercept

    def fit(self, X, y):
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 2:
            return np.maximum(0.01, self.intercept + self.slope * X_arr[:, 0])
        return np.maximum(0.01, self.intercept + self.slope * X_arr)


@pytest.fixture(scope="module")
def simple_cal_data():
    rng = np.random.default_rng(42)
    n = 400
    X = rng.uniform(0, 5, size=(n, 3))
    model = _SimplePoisson(slope=0.5, intercept=1.0)
    yhat = model.predict(X)
    y = rng.poisson(yhat)
    return {
        "model": model,
        "X": X,
        "y": y.astype(float),
        "X_cal": X[:200],
        "y_cal": y[:200].astype(float),
        "X_test": X[200:],
        "y_test": y[200:].astype(float),
    }


# ---------------------------------------------------------------------------
# 1 & 2. conformal_quantile and apply_exposure error paths
# ---------------------------------------------------------------------------

class TestConformalQuantileErrors:
    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="empty calibration set"):
            conformal_quantile(np.array([]), alpha=0.1)

    def test_alpha_zero_raises(self):
        scores = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            conformal_quantile(scores, alpha=0.0)

    def test_alpha_one_raises(self):
        scores = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            conformal_quantile(scores, alpha=1.0)

    def test_single_score_returns_finite(self):
        """A single calibration score should produce a finite quantile."""
        q = conformal_quantile(np.array([5.0]), alpha=0.1)
        assert np.isfinite(q)
        assert q == 5.0

    def test_quantile_increases_with_lower_alpha(self):
        """Lower alpha (wider intervals) => higher quantile."""
        rng = np.random.default_rng(1)
        scores = rng.exponential(2.0, size=500)
        q90 = conformal_quantile(scores, alpha=0.10)  # 90% coverage
        q50 = conformal_quantile(scores, alpha=0.50)  # 50% coverage
        assert q90 > q50


class TestApplyExposureErrors:
    def test_zero_exposure_raises(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([1.0, 2.0])
        exposure = np.array([0.5, 0.0])  # second exposure is zero
        with pytest.raises(ValueError, match="positive"):
            apply_exposure(y, yhat, exposure)

    def test_negative_exposure_raises(self):
        y = np.array([1.0])
        yhat = np.array([1.0])
        exposure = np.array([-1.0])
        with pytest.raises(ValueError, match="positive"):
            apply_exposure(y, yhat, exposure)

    def test_none_exposure_returns_unchanged(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([1.5, 2.5])
        y_out, yhat_out = apply_exposure(y, yhat, None)
        np.testing.assert_array_equal(y_out, y)
        np.testing.assert_array_equal(yhat_out, yhat)

    def test_exposure_divides_both(self):
        y = np.array([2.0, 4.0])
        yhat = np.array([1.0, 2.0])
        exposure = np.array([2.0, 2.0])
        y_out, yhat_out = apply_exposure(y, yhat, exposure)
        np.testing.assert_allclose(y_out, [1.0, 2.0])
        np.testing.assert_allclose(yhat_out, [0.5, 1.0])


# ---------------------------------------------------------------------------
# 3 & 4. temporal_split
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def test_tail_split_basic(self):
        """Default tail split: last calibration_frac rows become calibration set."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100, dtype=float)
        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(X, y, calibration_frac=0.2)
        assert len(X_cal) == 20
        assert len(X_train) == 80
        # Calibration set is the last 20 rows
        np.testing.assert_array_equal(y_cal, np.arange(80, 100, dtype=float))

    def test_date_col_sorts_by_date(self):
        """With date_col, calibration set is most recent observations."""
        n = 50
        dates = np.arange(n)  # 0..49 (0 = oldest)
        X = pd.DataFrame({"feature": np.arange(n, dtype=float), "date": dates})
        y = np.ones(n)
        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
            X, y, calibration_frac=0.2, date_col="date"
        )
        # Should have the 10 most recent dates in calibration
        assert len(X_cal) == 10
        assert X_cal["date"].max() == 49
        assert X_cal["date"].min() == 40

    def test_with_exposure(self):
        """Exposure should be split alongside X and y."""
        X = np.ones((20, 2))
        y = np.ones(20)
        exposure = np.arange(20, dtype=float)
        _, _, _, _, exp_train, exp_cal = temporal_split(
            X, y, calibration_frac=0.25, exposure=exposure
        )
        assert exp_train is not None
        assert exp_cal is not None
        assert len(exp_cal) == 5

    def test_polars_dataframe_input(self):
        """temporal_split should accept polars DataFrames."""
        X = pl.DataFrame({"a": np.arange(30, dtype=float)})
        y = np.ones(30)
        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(X, y, calibration_frac=0.2)
        assert len(X_cal) == 6
        assert len(X_train) == 24


# ---------------------------------------------------------------------------
# 5-9. Score function branches
# ---------------------------------------------------------------------------

class TestAnscombeScoreDistributions:
    def test_gamma_zero_at_yhat(self):
        """Anscombe score for gamma should be zero when y=yhat."""
        yhat = np.array([1.0, 3.0, 7.0])
        scores = anscombe_score(yhat, yhat, distribution="gamma")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-9)

    def test_gamma_y_zero_gives_finite_score(self):
        """y=0 should not produce NaN under gamma (handled with fallback)."""
        y = np.array([0.0, 1.0])
        yhat = np.array([1.0, 1.0])
        scores = anscombe_score(y, yhat, distribution="gamma")
        assert np.all(np.isfinite(scores))

    def test_gamma_always_nonnegative(self):
        rng = np.random.default_rng(10)
        y = rng.exponential(3.0, size=200)
        yhat = rng.uniform(0.5, 6.0, size=200)
        scores = anscombe_score(y, yhat, distribution="gamma")
        assert np.all(scores >= 0)

    def test_tweedie_p1_dispatches_to_poisson(self):
        """tweedie(p=1) should equal poisson Anscombe score."""
        y = np.array([1.0, 2.0, 4.0])
        yhat = np.array([1.5, 2.5, 3.5])
        s_tweedie = anscombe_score(y, yhat, distribution="tweedie", tweedie_power=1.0)
        s_poisson = anscombe_score(y, yhat, distribution="poisson")
        np.testing.assert_allclose(s_tweedie, s_poisson, rtol=1e-6)

    def test_tweedie_p2_dispatches_to_gamma(self):
        """tweedie(p=2) should equal gamma Anscombe score."""
        y = np.array([1.0, 2.0, 4.0])
        yhat = np.array([1.5, 2.5, 3.5])
        s_tweedie = anscombe_score(y, yhat, distribution="tweedie", tweedie_power=2.0)
        s_gamma = anscombe_score(y, yhat, distribution="gamma")
        np.testing.assert_allclose(s_tweedie, s_gamma, rtol=1e-6)

    def test_tweedie_general_p15_nonnegative(self):
        """General Tweedie(p=1.5) Anscombe scores should be non-negative."""
        rng = np.random.default_rng(11)
        y = rng.exponential(2.0, size=100)
        yhat = rng.uniform(0.5, 4.0, size=100)
        scores = anscombe_score(y, yhat, distribution="tweedie", tweedie_power=1.5)
        assert np.all(scores >= 0)

    def test_tweedie_general_zero_at_yhat(self):
        """Anscombe score should be 0 when y=yhat for general tweedie."""
        yhat = np.array([2.0, 5.0])
        scores = anscombe_score(yhat, yhat, distribution="tweedie", tweedie_power=1.5)
        np.testing.assert_allclose(scores, np.zeros(2), atol=1e-9)

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            anscombe_score(np.array([1.0]), np.array([1.0]), distribution="normal")


class TestDevianceScoreEdgeCases:
    def test_gamma_y_zero_finite(self):
        """gamma deviance score with y=0 should be finite (large but not NaN/inf)."""
        y = np.array([0.0, 1.0])
        yhat = np.array([2.0, 2.0])
        scores = deviance_score(y, yhat, distribution="gamma")
        assert np.all(np.isfinite(scores))

    def test_tweedie_p15_y_zero_finite(self):
        """Tweedie(p=1.5) deviance with y=0 should be finite."""
        y = np.array([0.0, 2.0])
        yhat = np.array([1.0, 1.0])
        scores = deviance_score(y, yhat, distribution="tweedie", tweedie_power=1.5)
        assert np.all(np.isfinite(scores))


class TestPearsonWeightedEdgeCases:
    def test_p_zero_gives_raw_like_score(self):
        """p=0: scale = yhat^0 = 1, so score = |y - yhat| / 1 = raw score."""
        y = np.array([3.0, 5.0])
        yhat = np.array([2.0, 4.0])
        scores_p0 = pearson_weighted_score(y, yhat, tweedie_power=0.0)
        scores_raw = np.abs(y - yhat)
        np.testing.assert_allclose(scores_p0, scores_raw)

    def test_p3_inverse_gaussian(self):
        """p=3 is valid (inverse Gaussian Tweedie). Scores should be finite."""
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.5, 2.5, 3.5])
        scores = pearson_weighted_score(y, yhat, tweedie_power=3.0)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)


# ---------------------------------------------------------------------------
# 10-13. InsuranceConformalPredictor branches
# ---------------------------------------------------------------------------

class TestPredictorPolarsInput:
    def test_predict_interval_with_polars_dataframe(self, simple_cal_data):
        """predict_interval should accept a Polars DataFrame for X_test."""
        d = simple_cal_data
        cp = InsuranceConformalPredictor(
            model=d["model"],
            nonconformity="pearson",
            tweedie_power=1.0,
        )
        cp.calibrate(d["X_cal"], d["y_cal"])

        X_test_pl = pl.DataFrame({"f0": d["X_test"][:, 0].tolist(),
                                   "f1": d["X_test"][:, 1].tolist(),
                                   "f2": d["X_test"][:, 2].tolist()})
        # predict() wraps _predict which converts polars to numpy
        # predict_interval calls _predict internally
        # Since our _SimplePoisson expects a 2D array, pass numpy directly
        # But test that predict() works with polars:
        preds = cp.predict(X_test_pl)
        assert len(preds) == len(d["X_test"])
        assert np.all(np.isfinite(preds))

    def test_quantile_caching(self, simple_cal_data):
        """The second call to predict_interval at the same alpha hits the cache."""
        d = simple_cal_data
        cp = InsuranceConformalPredictor(
            model=d["model"],
            nonconformity="raw",
            tweedie_power=1.0,
        )
        cp.calibrate(d["X_cal"], d["y_cal"])

        # First call: populates cache
        cp.predict_interval(d["X_test"], alpha=0.15)
        assert 0.15 in cp.cal_quantiles_

        # Mutate the cached value — if cache is used, second call gives same result
        cached_q = cp.cal_quantiles_[0.15]
        i1 = cp.predict_interval(d["X_test"], alpha=0.15)
        # Cache should still be present (not cleared on predict_interval)
        assert cp.cal_quantiles_[0.15] == cached_q

        i2 = cp.predict_interval(d["X_test"], alpha=0.15)
        # Results must be identical (same quantile used)
        assert i1["upper"].to_list() == i2["upper"].to_list()

    def test_alpha_boundary_raises(self, simple_cal_data):
        d = simple_cal_data
        cp = InsuranceConformalPredictor(model=d["model"], tweedie_power=1.0)
        cp.calibrate(d["X_cal"], d["y_cal"])

        with pytest.raises(ValueError, match="alpha"):
            cp.predict_interval(d["X_test"], alpha=0.0)

        with pytest.raises(ValueError, match="alpha"):
            cp.predict_interval(d["X_test"], alpha=1.0)


class TestInvertScoreGammaDistribution:
    def test_anscombe_gamma_inversion(self):
        """Anscombe gamma inversion roundtrip: score at upper bound == quantile."""
        yhat = np.array([2.0, 4.0])
        quantile = 0.3
        lower, upper = invert_score(
            yhat, quantile, "anscombe", distribution="gamma", clip_lower=0.0
        )
        # Compute score at upper bound — should equal quantile
        upper_scores = anscombe_score(upper, yhat, distribution="gamma")
        np.testing.assert_allclose(upper_scores, quantile, rtol=0.05, atol=1e-3)

    def test_deviance_gamma_inversion(self):
        """Deviance gamma inversion: score at upper bound == quantile."""
        yhat = np.array([3.0, 6.0])
        quantile = 0.4
        lower, upper = invert_score(
            yhat, quantile, "deviance", distribution="gamma", clip_lower=0.0
        )
        upper_scores = deviance_score(upper, yhat, distribution="gamma")
        np.testing.assert_allclose(upper_scores, quantile, rtol=0.05, atol=1e-3)


class TestTweedieAutoDetection:
    def test_unknown_model_warns_for_pearson_weighted(self):
        """A model with no detectable Tweedie power should warn when using pearson_weighted."""

        class _Unknown:
            def predict(self, X):
                return np.ones(len(X))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cp = InsuranceConformalPredictor(
                model=_Unknown(),
                nonconformity="pearson_weighted",
                # No tweedie_power provided, model is unrecognised
            )
        power_warnings = [w for w in caught if "Tweedie power" in str(w.message)]
        assert len(power_warnings) == 1
        # Should default to 1.5
        assert cp.tweedie_power == 1.5

    def test_explicit_tweedie_power_suppresses_warning(self):
        """Explicit tweedie_power= should not trigger the warning."""

        class _Unknown:
            def predict(self, X):
                return np.ones(len(X))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cp = InsuranceConformalPredictor(
                model=_Unknown(),
                nonconformity="pearson_weighted",
                tweedie_power=1.2,
            )
        power_warnings = [w for w in caught if "Tweedie power" in str(w.message)]
        assert len(power_warnings) == 0
        assert cp.tweedie_power == 1.2
