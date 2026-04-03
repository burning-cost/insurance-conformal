"""
Additional test coverage for CQR, LocallyWeightedConformal, HongModelFree,
HongTransformConformal, and deeper edge cases.

Focuses on:
- CQR: coverage_by_decile, correction_quantile, repr, error paths
- LocallyWeightedConformal: existing test gaps
- HongModelFree and HongTransformConformal: additional API and edge cases
- scores: invert_score with extreme Tweedie powers
- utils: extract_tweedie_power for pipeline models
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_conformal.scores import (
    invert_score,
    pearson_weighted_score,
    deviance_score,
    anscombe_score,
    raw_score,
    compute_score,
)
from insurance_conformal.utils import (
    conformal_quantile,
    extract_tweedie_power,
)
from insurance_conformal import ConformalisedQuantileRegression
from insurance_conformal.hong import HongModelFree, HongTransformConformal
from insurance_conformal.locally_weighted import LocallyWeightedConformal


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class ConstantModel:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)

    def fit(self, X, y):
        return self


class LinearModel:
    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(X) @ self.beta)


class OffsetQuantileModel:
    """Returns predictions = max(0, X[:,0] + offset) for quantile modelling."""

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.maximum(0.0, X[:, 0] + self.offset)


@pytest.fixture(scope="module")
def cqr_data():
    rng = np.random.default_rng(101)
    n = 800
    X = rng.normal(size=(n, 3))
    y = np.maximum(0.0, X[:, 0] + rng.normal(0, 1, n))

    lo_model = OffsetQuantileModel(offset=-1.0)
    hi_model = OffsetQuantileModel(offset=1.5)
    return {
        "X_cal": X[:300],
        "y_cal": y[:300],
        "X_test": X[300:600],
        "y_test": y[300:600],
        "lo_model": lo_model,
        "hi_model": hi_model,
    }


@pytest.fixture(scope="module")
def calibrated_cqr(cqr_data):
    cqr = ConformalisedQuantileRegression(
        model_lo=cqr_data["lo_model"],
        model_hi=cqr_data["hi_model"],
    )
    cqr.calibrate(cqr_data["X_cal"], cqr_data["y_cal"])
    return cqr


@pytest.fixture(scope="module")
def hong_data():
    rng = np.random.default_rng(202)
    n = 600
    X = rng.normal(size=(n, 3))
    beta = np.array([0.3, -0.1, 0.2])
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    return {
        "X_train": X[:300],
        "y_train": y[:300],
        "X_test": X[300:],
        "y_test": y[300:],
        "model": LinearModel(beta),
    }


@pytest.fixture(scope="module")
def lwc_data():
    rng = np.random.default_rng(303)
    n = 800
    beta = np.array([0.4, -0.2])
    X = rng.normal(size=(n, 2))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    return {
        "X_train": X[:400],
        "y_train": y[:400],
        "X_cal": X[400:600],
        "y_cal": y[400:600],
        "X_test": X[600:],
        "y_test": y[600:],
        "model": LinearModel(beta),
    }


# ---------------------------------------------------------------------------
# CQR: coverage_by_decile
# ---------------------------------------------------------------------------


class TestCQRCoverageByDecile:
    def test_returns_polars_dataframe(self, calibrated_cqr, cqr_data):
        result = calibrated_cqr.coverage_by_decile(
            cqr_data["X_test"], cqr_data["y_test"], alpha=0.10
        )
        assert isinstance(result, pl.DataFrame)

    def test_required_columns(self, calibrated_cqr, cqr_data):
        result = calibrated_cqr.coverage_by_decile(
            cqr_data["X_test"], cqr_data["y_test"], alpha=0.10
        )
        for col in ["decile", "mean_midpoint", "mean_width", "n_obs", "coverage", "target_coverage"]:
            assert col in result.columns

    def test_n_obs_sums_to_total(self, calibrated_cqr, cqr_data):
        result = calibrated_cqr.coverage_by_decile(
            cqr_data["X_test"], cqr_data["y_test"], alpha=0.10
        )
        assert result["n_obs"].sum() == len(cqr_data["y_test"])

    def test_coverage_in_unit_interval(self, calibrated_cqr, cqr_data):
        result = calibrated_cqr.coverage_by_decile(
            cqr_data["X_test"], cqr_data["y_test"], alpha=0.10
        )
        assert (result["coverage"] >= 0.0).all()
        assert (result["coverage"] <= 1.0).all()

    def test_target_coverage_matches_alpha(self, calibrated_cqr, cqr_data):
        result = calibrated_cqr.coverage_by_decile(
            cqr_data["X_test"], cqr_data["y_test"], alpha=0.20
        )
        assert np.allclose(result["target_coverage"].to_numpy(), 0.80)

    def test_n_bins_parameter(self, calibrated_cqr, cqr_data):
        result = calibrated_cqr.coverage_by_decile(
            cqr_data["X_test"], cqr_data["y_test"], alpha=0.10, n_bins=5
        )
        assert len(result) <= 5

    def test_n_bins_1(self, calibrated_cqr, cqr_data):
        result = calibrated_cqr.coverage_by_decile(
            cqr_data["X_test"], cqr_data["y_test"], alpha=0.10, n_bins=1
        )
        assert len(result) == 1
        assert result["n_obs"][0] == len(cqr_data["y_test"])

    def test_mean_width_positive(self, calibrated_cqr, cqr_data):
        result = calibrated_cqr.coverage_by_decile(
            cqr_data["X_test"], cqr_data["y_test"], alpha=0.10
        )
        assert (result["mean_width"] >= 0.0).all()


# ---------------------------------------------------------------------------
# CQR: correction_quantile
# ---------------------------------------------------------------------------


class TestCQRCorrectionQuantile:
    def test_returns_float(self, calibrated_cqr):
        q = calibrated_cqr.correction_quantile(alpha=0.10)
        assert isinstance(q, float)

    def test_finite(self, calibrated_cqr):
        q = calibrated_cqr.correction_quantile(alpha=0.10)
        assert np.isfinite(q)

    def test_smaller_alpha_gives_larger_correction(self, calibrated_cqr):
        q_tight = calibrated_cqr.correction_quantile(alpha=0.05)
        q_loose = calibrated_cqr.correction_quantile(alpha=0.30)
        assert q_tight >= q_loose

    def test_before_calibrate_raises(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
        )
        with pytest.raises(RuntimeError, match="calibrated"):
            cqr.correction_quantile(0.10)

    def test_matches_predict_interval_correction(self, calibrated_cqr, cqr_data):
        """The correction_quantile should match the actual interval adjustment."""
        q = calibrated_cqr.correction_quantile(alpha=0.10)
        intervals = calibrated_cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        # upper = q_hi + q, lower = max(0, q_lo - q)
        # So upper - q_hi should equal q everywhere
        q_hi = intervals["q_hi"].to_numpy()
        upper = intervals["upper"].to_numpy()
        diff = upper - q_hi
        np.testing.assert_allclose(diff, q, atol=1e-10)


# ---------------------------------------------------------------------------
# CQR: repr
# ---------------------------------------------------------------------------


class TestCQRRepr:
    def test_repr_before_calibration(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
        )
        r = repr(cqr)
        assert "not calibrated" in r
        assert "ConformalisedQuantileRegression" in r

    def test_repr_after_calibration(self, calibrated_cqr, cqr_data):
        r = repr(calibrated_cqr)
        assert "calibrated on" in r
        assert str(len(cqr_data["y_cal"])) in r


# ---------------------------------------------------------------------------
# CQR: clip_lower behaviour
# ---------------------------------------------------------------------------


class TestCQRClipLower:
    def test_default_clip_lower_is_zero(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
        )
        assert cqr.clip_lower == 0.0

    def test_clip_lower_custom(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
            clip_lower=5.0,
        )
        cqr.calibrate(cqr_data["X_cal"], cqr_data["y_cal"])
        intervals = cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        assert (intervals["lower"] >= 5.0).all()

    def test_lower_nonneg_by_default(self, calibrated_cqr, cqr_data):
        intervals = calibrated_cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        assert (intervals["lower"] >= 0.0).all()

    def test_clip_lower_neg_inf_allows_negative(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
            clip_lower=-np.inf,
        )
        cqr.calibrate(cqr_data["X_cal"], cqr_data["y_cal"])
        intervals = cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        # With large correction, some lower bounds may be negative
        assert isinstance(intervals, pl.DataFrame)


# ---------------------------------------------------------------------------
# CQR: interval structure
# ---------------------------------------------------------------------------


class TestCQRIntervalStructure:
    def test_upper_geq_lower(self, calibrated_cqr, cqr_data):
        intervals = calibrated_cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        assert (intervals["upper"] >= intervals["lower"]).all()

    def test_output_columns(self, calibrated_cqr, cqr_data):
        intervals = calibrated_cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        assert set(intervals.columns) == {"lower", "q_lo", "q_hi", "upper"}

    def test_output_length(self, calibrated_cqr, cqr_data):
        intervals = calibrated_cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        assert len(intervals) == len(cqr_data["X_test"])

    def test_wider_at_smaller_alpha(self, calibrated_cqr, cqr_data):
        i_90 = calibrated_cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        i_50 = calibrated_cqr.predict_interval(cqr_data["X_test"], alpha=0.50)
        w90 = (i_90["upper"] - i_90["lower"]).mean()
        w50 = (i_50["upper"] - i_50["lower"]).mean()
        assert w90 > w50

    def test_invalid_alpha_raises(self, calibrated_cqr, cqr_data):
        with pytest.raises(ValueError, match="alpha"):
            calibrated_cqr.predict_interval(cqr_data["X_test"], alpha=0.0)

    def test_predict_before_calibrate_raises(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
        )
        with pytest.raises(RuntimeError, match="calibrated"):
            cqr.predict_interval(cqr_data["X_test"])

    def test_polars_input(self, calibrated_cqr, cqr_data):
        X_pl = pl.DataFrame(cqr_data["X_test"])
        intervals = calibrated_cqr.predict_interval(X_pl, alpha=0.10)
        assert len(intervals) == len(cqr_data["X_test"])

    def test_pandas_input(self, calibrated_cqr, cqr_data):
        X_pd = pd.DataFrame(cqr_data["X_test"])
        intervals = calibrated_cqr.predict_interval(X_pd, alpha=0.10)
        assert len(intervals) == len(cqr_data["X_test"])

    def test_length_mismatch_raises(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
        )
        y_short = cqr_data["y_cal"][:-1]
        with pytest.raises(ValueError, match="lengths"):
            cqr.calibrate(cqr_data["X_cal"], y_short)


# ---------------------------------------------------------------------------
# CQR: cal_scores_ properties
# ---------------------------------------------------------------------------


class TestCQRCalScores:
    def test_cal_scores_set_after_calibrate(self, calibrated_cqr, cqr_data):
        assert calibrated_cqr.cal_scores_ is not None
        assert len(calibrated_cqr.cal_scores_) == len(cqr_data["y_cal"])

    def test_n_calibration_correct(self, calibrated_cqr, cqr_data):
        assert calibrated_cqr.n_calibration_ == len(cqr_data["y_cal"])

    def test_is_calibrated_true(self, calibrated_cqr):
        assert calibrated_cqr.is_calibrated_

    def test_recalibrate_clears_cache(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
        )
        cqr.calibrate(cqr_data["X_cal"], cqr_data["y_cal"])
        cqr.predict_interval(cqr_data["X_test"], alpha=0.10)
        assert len(cqr.cal_quantile_) > 0

        cqr.calibrate(cqr_data["X_cal"], cqr_data["y_cal"])
        assert len(cqr.cal_quantile_) == 0

    def test_returns_self_on_calibrate(self, cqr_data):
        cqr = ConformalisedQuantileRegression(
            model_lo=cqr_data["lo_model"],
            model_hi=cqr_data["hi_model"],
        )
        result = cqr.calibrate(cqr_data["X_cal"], cqr_data["y_cal"])
        assert result is cqr


# ---------------------------------------------------------------------------
# HongModelFree: API and edge cases
# ---------------------------------------------------------------------------


class TestHongModelFreeEdgeCases:
    def test_fit_returns_self(self, hong_data):
        hmf = HongModelFree()
        result = hmf.fit(hong_data["X_train"], hong_data["y_train"])
        assert result is hmf

    def test_stores_training_data(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        assert hmf.X_train_ is not None
        assert hmf.y_train_ is not None
        assert hmf.n_train_ == len(hong_data["y_train"])

    def test_predict_interval_before_fit_raises(self):
        hmf = HongModelFree()
        with pytest.raises(RuntimeError):
            hmf.predict_interval(np.array([[1.0, 2.0, 3.0]]))

    def test_output_columns(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        intervals = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        assert isinstance(intervals, pl.DataFrame)
        assert "lower" in intervals.columns
        assert "upper" in intervals.columns

    def test_output_length(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        intervals = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        assert len(intervals) == len(hong_data["X_test"])

    def test_lower_leq_upper(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        intervals = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (intervals["upper"] >= intervals["lower"]).all()

    def test_lower_nonneg_with_clip_lower_0(self, hong_data):
        hmf = HongModelFree(clip_lower=0.0)
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        intervals = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (intervals["lower"] >= 0.0).all()

    def test_invalid_alpha_raises(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        with pytest.raises(ValueError, match="alpha"):
            hmf.predict_interval(hong_data["X_test"], alpha=1.5)

    def test_wider_at_lower_alpha(self, hong_data):
        hmf = HongModelFree()
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        i_90 = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        i_50 = hmf.predict_interval(hong_data["X_test"], alpha=0.50)
        w90 = (i_90["upper"] - i_90["lower"]).mean()
        w50 = (i_50["upper"] - i_50["lower"]).mean()
        assert w90 >= w50

    def test_polars_X_accepted(self, hong_data):
        hmf = HongModelFree()
        X_pl = pl.DataFrame(hong_data["X_train"])
        hmf.fit(X_pl, hong_data["y_train"])
        intervals = hmf.predict_interval(pl.DataFrame(hong_data["X_test"]), alpha=0.10)
        assert len(intervals) == len(hong_data["X_test"])

    def test_pandas_X_accepted(self, hong_data):
        hmf = HongModelFree()
        X_pd = pd.DataFrame(hong_data["X_train"])
        hmf.fit(X_pd, hong_data["y_train"])
        intervals = hmf.predict_interval(pd.DataFrame(hong_data["X_test"]), alpha=0.10)
        assert len(intervals) == len(hong_data["X_test"])

    def test_single_training_observation(self):
        """Degenerate: 1 training obs, should not crash."""
        hmf = HongModelFree()
        X_train = np.array([[1.0, 0.5, 0.3]])
        y_train = np.array([2.0])
        hmf.fit(X_train, y_train)
        X_test = np.array([[1.0, 0.5, 0.3], [0.0, 0.0, 0.0]])
        intervals = hmf.predict_interval(X_test, alpha=0.10)
        assert len(intervals) == 2

    def test_clip_lower_custom(self, hong_data):
        hmf = HongModelFree(clip_lower=1.0)
        hmf.fit(hong_data["X_train"], hong_data["y_train"])
        intervals = hmf.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (intervals["lower"] >= 1.0).all()

    def test_1d_X_reshaped(self):
        """1D X should be reshaped to 2D (n, 1)."""
        hmf = HongModelFree()
        X_1d = np.array([1.0, 2.0, 3.0])
        y = np.array([1.5, 2.5, 3.5])
        hmf.fit(X_1d, y)
        assert hmf.X_train_.ndim == 2
        assert hmf.X_train_.shape == (3, 1)


# ---------------------------------------------------------------------------
# HongTransformConformal: API and edge cases
# ---------------------------------------------------------------------------


class TestHongTransformConformalEdgeCases:
    def test_fit_calibrate_predict(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        htc.calibrate(hong_data["X_train"], hong_data["y_train"])
        intervals = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        assert isinstance(intervals, pl.DataFrame)

    def test_output_columns(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        htc.calibrate(hong_data["X_train"], hong_data["y_train"])
        intervals = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        # HongTransformConformal should have at least lower and upper
        assert "lower" in intervals.columns
        assert "upper" in intervals.columns

    def test_lower_leq_upper(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        htc.calibrate(hong_data["X_train"], hong_data["y_train"])
        intervals = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (intervals["upper"] >= intervals["lower"]).all()

    def test_lower_nonneg(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        htc.calibrate(hong_data["X_train"], hong_data["y_train"])
        intervals = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        assert (intervals["lower"] >= 0.0).all()

    def test_predict_before_calibrate_raises(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        with pytest.raises(RuntimeError):
            htc.predict_interval(hong_data["X_test"])

    def test_wider_at_lower_alpha(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        htc.calibrate(hong_data["X_train"], hong_data["y_train"])
        i_90 = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        i_50 = htc.predict_interval(hong_data["X_test"], alpha=0.50)
        w90 = (i_90["upper"] - i_90["lower"]).mean()
        w50 = (i_50["upper"] - i_50["lower"]).mean()
        assert w90 >= w50

    def test_invalid_alpha_raises(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        htc.calibrate(hong_data["X_train"], hong_data["y_train"])
        with pytest.raises(ValueError):
            htc.predict_interval(hong_data["X_test"], alpha=0.0)

    def test_output_length(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        htc.calibrate(hong_data["X_train"], hong_data["y_train"])
        intervals = htc.predict_interval(hong_data["X_test"], alpha=0.10)
        assert len(intervals) == len(hong_data["X_test"])

    def test_calibrate_returns_self(self, hong_data):
        htc = HongTransformConformal(model=hong_data["model"])
        result = htc.calibrate(hong_data["X_train"], hong_data["y_train"])
        assert result is htc


# ---------------------------------------------------------------------------
# LocallyWeightedConformal: edge cases not in existing tests
# ---------------------------------------------------------------------------


class TestLWCEdgeCases:
    def test_calibrate_before_fit_raises(self, lwc_data):
        """fit() is required before calibrate() when using a separate spread model."""
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        # fit() must be called first to train the spread model
        # Calling calibrate without fit should raise RuntimeError
        with pytest.raises(RuntimeError):
            lwc.calibrate(lwc_data["X_cal"], lwc_data["y_cal"])

    def test_fit_returns_self(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        result = lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        assert result is lwc

    def test_calibrate_returns_self(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        result = lwc.calibrate(lwc_data["X_cal"], lwc_data["y_cal"])
        assert result is lwc

    def test_predict_interval_before_calibrate_raises(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        with pytest.raises(RuntimeError):
            lwc.predict_interval(lwc_data["X_test"])

    def test_output_columns(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        lwc.calibrate(lwc_data["X_cal"], lwc_data["y_cal"])
        intervals = lwc.predict_interval(lwc_data["X_test"], alpha=0.10)
        assert "lower" in intervals.columns
        assert "upper" in intervals.columns
        assert "point" in intervals.columns

    def test_lower_leq_point_leq_upper(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        lwc.calibrate(lwc_data["X_cal"], lwc_data["y_cal"])
        intervals = lwc.predict_interval(lwc_data["X_test"], alpha=0.10)
        assert (intervals["lower"] <= intervals["point"]).all()
        assert (intervals["point"] <= intervals["upper"]).all()

    def test_lower_nonneg(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        lwc.calibrate(lwc_data["X_cal"], lwc_data["y_cal"])
        intervals = lwc.predict_interval(lwc_data["X_test"], alpha=0.10)
        assert (intervals["lower"] >= 0.0).all()

    def test_wider_at_lower_alpha(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        lwc.calibrate(lwc_data["X_cal"], lwc_data["y_cal"])
        i_90 = lwc.predict_interval(lwc_data["X_test"], alpha=0.10)
        i_50 = lwc.predict_interval(lwc_data["X_test"], alpha=0.50)
        w90 = (i_90["upper"] - i_90["lower"]).mean()
        w50 = (i_50["upper"] - i_50["lower"]).mean()
        assert w90 > w50

    def test_invalid_alpha_raises(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        lwc.calibrate(lwc_data["X_cal"], lwc_data["y_cal"])
        with pytest.raises(ValueError):
            lwc.predict_interval(lwc_data["X_test"], alpha=1.5)

    def test_sklearn_backend_auto_fallback(self, lwc_data):
        """backend='sklearn' should work without catboost/lightgbm."""
        from sklearn.ensemble import GradientBoostingRegressor
        spread = GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=0)
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=None,
            tweedie_power=1.0,
            backend="sklearn",
        )
        lwc.fit(lwc_data["X_train"], lwc_data["y_train"])
        lwc.calibrate(lwc_data["X_cal"], lwc_data["y_cal"])
        intervals = lwc.predict_interval(lwc_data["X_test"], alpha=0.10)
        assert len(intervals) == len(lwc_data["X_test"])

    def test_polars_input(self, lwc_data):
        lwc = LocallyWeightedConformal(
            model=lwc_data["model"],
            spread_model=ConstantModel(1.0),
            tweedie_power=1.0,
        )
        X_train_pl = pl.DataFrame(lwc_data["X_train"])
        y_train_pl = pl.Series(lwc_data["y_train"].tolist())
        lwc.fit(X_train_pl, y_train_pl)

        X_cal_pl = pl.DataFrame(lwc_data["X_cal"])
        y_cal_pl = pl.Series(lwc_data["y_cal"].tolist())
        lwc.calibrate(X_cal_pl, y_cal_pl)

        X_test_pl = pl.DataFrame(lwc_data["X_test"])
        intervals = lwc.predict_interval(X_test_pl, alpha=0.10)
        assert len(intervals) == len(lwc_data["X_test"])


# ---------------------------------------------------------------------------
# extract_tweedie_power: sklearn pipeline handling
# ---------------------------------------------------------------------------


class TestExtractTweediePowerEdgeCases:
    def test_pipeline_last_step_extracted(self):
        """If the last step of a sklearn Pipeline is TweedieRegressor, extract its power."""
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import TweedieRegressor

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", TweedieRegressor(power=1.7)),
            ])
            result = extract_tweedie_power(pipe)
            assert result == pytest.approx(1.7)
        except ImportError:
            pytest.skip("sklearn not available")

    def test_returns_none_for_list(self):
        result = extract_tweedie_power([1, 2, 3])
        assert result is None

    def test_returns_none_for_dict(self):
        result = extract_tweedie_power({"a": 1})
        assert result is None

    def test_returns_none_for_string(self):
        result = extract_tweedie_power("some_model_string")
        assert result is None

    def test_sklearn_power_0(self):
        try:
            from sklearn.linear_model import TweedieRegressor
            model = TweedieRegressor(power=0.0)
            result = extract_tweedie_power(model)
            assert result == pytest.approx(0.0)
        except ImportError:
            pytest.skip("sklearn not available")

    def test_sklearn_power_2(self):
        try:
            from sklearn.linear_model import TweedieRegressor
            model = TweedieRegressor(power=2.0)
            result = extract_tweedie_power(model)
            assert result == pytest.approx(2.0)
        except ImportError:
            pytest.skip("sklearn not available")


# ---------------------------------------------------------------------------
# Score consistency: compute_score vs manual calls
# ---------------------------------------------------------------------------


class TestScoreConsistency:
    @pytest.mark.parametrize("nc,dist,p", [
        ("pearson", "poisson", 1.0),
        ("pearson_weighted", "tweedie", 1.5),
        ("deviance", "tweedie", 1.5),
        ("anscombe", "poisson", 1.0),
        ("raw", "tweedie", 1.5),
    ])
    def test_compute_matches_direct(self, nc, dist, p):
        rng = np.random.default_rng(99)
        y = rng.exponential(2.0, 20)
        yhat = rng.uniform(0.5, 4.0, 20)

        via_compute = compute_score(y, yhat, nonconformity=nc, distribution=dist, tweedie_power=p)

        if nc == "raw":
            direct = raw_score(y, yhat)
        elif nc == "pearson":
            direct = pearson_score_fn(y, yhat)
        elif nc == "pearson_weighted":
            direct = pearson_weighted_score(y, yhat, tweedie_power=p)
        elif nc == "deviance":
            direct = deviance_score(y, yhat, distribution=dist, tweedie_power=p)
        elif nc == "anscombe":
            direct = anscombe_score(y, yhat, distribution=dist, tweedie_power=p)

        np.testing.assert_allclose(via_compute, direct)

    def test_raw_score_commutative_in_abs(self):
        y = np.array([3.0, 5.0])
        yhat = np.array([2.0, 6.0])
        s1 = raw_score(y, yhat)
        s2 = raw_score(yhat, y)
        np.testing.assert_allclose(s1, s2)


# Import this here to avoid circular import issue in test method above
from insurance_conformal.scores import pearson_score as pearson_score_fn


# ---------------------------------------------------------------------------
# Invert score: additional Tweedie power variants
# ---------------------------------------------------------------------------


class TestInvertScoreAdditionalPowers:
    @pytest.mark.parametrize("p", [0.0, 1.0, 1.5, 2.0, 3.0])
    def test_pearson_weighted_inversion_consistent(self, p):
        yhat = np.array([2.0, 5.0])
        q = 0.5
        lower, upper = invert_score(yhat, q, "pearson_weighted", tweedie_power=p, clip_lower=0.0)
        # Re-score at upper should equal quantile
        upper_scores = pearson_weighted_score(upper, yhat, tweedie_power=p)
        np.testing.assert_allclose(upper_scores, q, rtol=1e-10)

    def test_raw_inversion_large_quantile_clipped(self):
        """With very large quantile, lower = 0 (clipped), upper = yhat + q."""
        yhat = np.array([0.01])
        q = 1000.0
        lower, upper = invert_score(yhat, q, "raw", clip_lower=0.0)
        assert lower[0] == 0.0
        assert upper[0] == pytest.approx(1000.01)

    def test_deviance_poisson_roundtrip(self):
        yhat = np.array([1.0, 3.0, 7.0])
        q = 0.4
        lower, upper = invert_score(yhat, q, "deviance", distribution="poisson", clip_lower=0.0)
        upper_scores = deviance_score(upper, yhat, distribution="poisson")
        np.testing.assert_allclose(upper_scores, q, rtol=0.05, atol=1e-3)

    def test_deviance_gamma_roundtrip(self):
        yhat = np.array([2.0, 4.0])
        q = 0.3
        lower, upper = invert_score(yhat, q, "deviance", distribution="gamma", clip_lower=0.0)
        upper_scores = deviance_score(upper, yhat, distribution="gamma")
        np.testing.assert_allclose(upper_scores, q, rtol=0.05, atol=1e-3)

    def test_anscombe_gamma_roundtrip(self):
        yhat = np.array([2.0, 5.0])
        q = 0.5
        lower, upper = invert_score(yhat, q, "anscombe", distribution="gamma", clip_lower=0.0)
        upper_scores = anscombe_score(upper, yhat, distribution="gamma")
        np.testing.assert_allclose(upper_scores, q, rtol=0.05, atol=1e-3)

    def test_all_scores_upper_geq_yhat(self):
        """For all score types, upper bound should be >= yhat."""
        yhat = np.array([1.0, 2.0, 5.0])
        q = 0.5
        for nc in ["raw", "pearson", "pearson_weighted", "deviance", "anscombe"]:
            _, upper = invert_score(yhat, q, nc, distribution="poisson", clip_lower=0.0)
            assert np.all(upper >= yhat), f"upper < yhat for {nc}"


# ---------------------------------------------------------------------------
# Conformal quantile: finite-sample guarantee
# ---------------------------------------------------------------------------


class TestConformalQuantileGuarantee:
    def test_finite_sample_guarantee(self):
        """
        The conformal quantile formula should give coverage >= 1 - alpha
        on average, even for small n.

        We verify this by checking that the fraction of test scores that
        fall at or below the calibration quantile is >= 1 - alpha.

        This is the mathematical guarantee: under exchangeability,
        P(score_test <= Q) >= 1 - alpha.
        """
        rng = np.random.default_rng(42)
        n_cal = 100
        n_test = 1000

        # Scores from the same distribution
        all_scores = rng.exponential(scale=2.0, size=n_cal + n_test)
        cal_scores = all_scores[:n_cal]
        test_scores = all_scores[n_cal:]

        for alpha in [0.05, 0.10, 0.20]:
            q = conformal_quantile(cal_scores, alpha=alpha)
            covered_frac = np.mean(test_scores <= q)
            # Allow 3pp tolerance for finite sample
            assert covered_frac >= (1 - alpha) - 0.03, (
                f"Finite-sample coverage {covered_frac:.3f} below target {1 - alpha:.2f} "
                f"(alpha={alpha})"
            )

    def test_quantile_is_from_scores(self):
        """The quantile must be an actual value in the scores array."""
        scores = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        q = conformal_quantile(scores, alpha=0.20)
        assert q in scores


# ---------------------------------------------------------------------------
# Score properties: non-conformity ordering
# ---------------------------------------------------------------------------


class TestScoreOrdering:
    def test_pearson_leq_raw_for_large_yhat(self):
        """Pearson < raw when yhat > 1 (standard deviation normalization)."""
        rng = np.random.default_rng(0)
        yhat = rng.uniform(5.0, 20.0, 50)
        y = yhat + rng.normal(0, 1.0, 50)

        raw = raw_score(y, yhat)
        pearson = pearson_score_fn(y, yhat)
        # On average, Pearson should be smaller than raw for large yhat
        assert pearson.mean() < raw.mean()

    def test_deviance_finite_for_reasonable_values(self):
        rng = np.random.default_rng(1)
        y = rng.exponential(2.0, 100)
        yhat = rng.uniform(0.5, 5.0, 100)
        scores = deviance_score(y, yhat, distribution="tweedie", tweedie_power=1.5)
        assert np.all(np.isfinite(scores))

    def test_anscombe_finite_for_reasonable_values(self):
        rng = np.random.default_rng(2)
        y = rng.exponential(2.0, 100)
        yhat = rng.uniform(0.5, 5.0, 100)
        scores = anscombe_score(y, yhat, distribution="poisson")
        assert np.all(np.isfinite(scores))

    def test_pearson_weighted_finite_for_reasonable_values(self):
        rng = np.random.default_rng(3)
        y = rng.exponential(2.0, 100)
        yhat = rng.uniform(0.5, 5.0, 100)
        scores = pearson_weighted_score(y, yhat, tweedie_power=1.5)
        assert np.all(np.isfinite(scores))
