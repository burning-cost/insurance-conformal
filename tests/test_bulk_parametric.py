"""
Bulk parametric tests covering many input combinations for scores, utils, and predictor.

These are data-driven tests that run the same checks across many seeds,
tweedie powers, distributions, and input shapes. They catch regressions
where a specific parameter combination breaks.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal.scores import (
    compute_score,
    invert_score,
    raw_score,
    pearson_score,
    pearson_weighted_score,
    deviance_score,
    anscombe_score,
)
from insurance_conformal.utils import conformal_quantile, ipcw_weighted_quantile
from insurance_conformal import InsuranceConformalPredictor


# ---------------------------------------------------------------------------
# compute_score: parametric sweep over all combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted", "deviance", "anscombe"])
@pytest.mark.parametrize("dist", ["poisson", "gamma", "tweedie"])
@pytest.mark.parametrize("p", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_score_parametric(nc, dist, p, seed):
    """Every combination of score/distribution/power should produce finite, non-negative scores."""
    rng = np.random.default_rng(seed)
    y = rng.exponential(2.0, 30)
    yhat = rng.uniform(0.5, 5.0, 30)

    # Skip gamma/poisson with specific p values (they're valid but redundant)
    scores = compute_score(y, yhat, nonconformity=nc, distribution=dist, tweedie_power=p)
    assert scores.shape == (30,)
    assert np.all(np.isfinite(scores)), f"Non-finite scores for {nc}/{dist}/p={p}/seed={seed}"
    assert np.all(scores >= 0), f"Negative scores for {nc}/{dist}/p={p}/seed={seed}"


# ---------------------------------------------------------------------------
# invert_score: parametric over scores and yhat ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted"])
@pytest.mark.parametrize("q", [0.1, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("yhat_val", [0.5, 1.0, 5.0, 10.0])
def test_invert_score_bounds_ordering(nc, q, yhat_val):
    """lower <= upper for all score types and quantile values."""
    yhat = np.array([yhat_val])
    lower, upper = invert_score(yhat, q, nc, clip_lower=0.0)
    assert lower[0] <= upper[0] + 1e-10, f"lower > upper for {nc}, q={q}, yhat={yhat_val}"


@pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted"])
@pytest.mark.parametrize("q", [0.1, 0.5, 1.0, 2.0])
def test_invert_score_lower_nonneg(nc, q):
    """lower >= 0 with clip_lower=0."""
    yhat = np.array([0.5, 1.0, 2.0, 5.0])
    lower, upper = invert_score(yhat, q, nc, clip_lower=0.0)
    assert np.all(lower >= 0.0), f"Negative lower for {nc}, q={q}"


# ---------------------------------------------------------------------------
# conformal_quantile: parametric over sample sizes and distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 10, 50, 200, 1000])
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_conformal_quantile_parametric(n, alpha):
    """conformal_quantile should always return finite scalar for valid inputs."""
    rng = np.random.default_rng(n + int(alpha * 100))
    scores = rng.exponential(2.0, n)
    q = conformal_quantile(scores, alpha)
    assert np.isfinite(q)
    assert isinstance(q, float)
    assert q >= scores.min()
    assert q <= scores.max()


@pytest.mark.parametrize("dist_fn,kwargs", [
    (np.random.default_rng(0).exponential, {"scale": 2.0, "size": 100}),
    (np.random.default_rng(1).lognormal, {"mean": 0, "sigma": 1, "size": 100}),
    (np.random.default_rng(2).gamma, {"shape": 2, "scale": 1, "size": 100}),
])
def test_conformal_quantile_various_distributions(dist_fn, kwargs):
    """conformal_quantile should work for various score distributions."""
    scores = dist_fn(**kwargs)
    q = conformal_quantile(scores, alpha=0.10)
    assert np.isfinite(q)
    assert q > 0


# ---------------------------------------------------------------------------
# ipcw_weighted_quantile: parametric over weight patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 0.30])
def test_ipcw_parametric_alpha(alpha):
    rng = np.random.default_rng(int(alpha * 100))
    scores = rng.exponential(2.0, 100)
    weights = rng.uniform(0.1, 2.0, 100)
    q = ipcw_weighted_quantile(scores, weights, alpha)
    assert np.isfinite(q)
    assert isinstance(q, float)
    assert q >= scores.min()
    assert q <= scores.max()


@pytest.mark.parametrize("n", [10, 50, 200])
def test_ipcw_different_sizes(n):
    rng = np.random.default_rng(n)
    scores = rng.exponential(2.0, n)
    weights = rng.uniform(0.5, 1.5, n)
    q = ipcw_weighted_quantile(scores, weights, alpha=0.10)
    assert np.isfinite(q)


# ---------------------------------------------------------------------------
# pearson_weighted_score: parametric over Tweedie powers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
def test_pearson_weighted_various_powers(p):
    rng = np.random.default_rng(int(p * 10))
    y = rng.exponential(2.0, 50)
    yhat = rng.uniform(0.5, 5.0, 50)
    scores = pearson_weighted_score(y, yhat, tweedie_power=p)
    assert scores.shape == (50,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


# ---------------------------------------------------------------------------
# InsuranceConformalPredictor: parametric over nonconformity scores
# ---------------------------------------------------------------------------


class LogLinearModel:
    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        return np.exp(np.asarray(X) @ self.beta)


@pytest.fixture(scope="module")
def parametric_data():
    rng = np.random.default_rng(42)
    n = 1000
    beta = np.array([0.3, -0.1])
    X = rng.normal(size=(n, 2))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    return {
        "X_cal": X[:400],
        "y_cal": y[:400],
        "X_test": X[400:],
        "y_test": y[400:],
        "model": LogLinearModel(beta),
    }


@pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted", "deviance", "anscombe"])
def test_predictor_all_scores_produce_valid_intervals(nc, parametric_data):
    cp = InsuranceConformalPredictor(
        model=parametric_data["model"],
        nonconformity=nc,
        distribution="poisson",
        tweedie_power=1.0,
    )
    cp.calibrate(parametric_data["X_cal"], parametric_data["y_cal"])
    intervals = cp.predict_interval(parametric_data["X_test"], alpha=0.10)

    assert len(intervals) == len(parametric_data["X_test"])
    assert (intervals["lower"] >= 0.0).all()
    assert (intervals["lower"] <= intervals["upper"]).all()
    assert (intervals["lower"] <= intervals["point"]).all()
    assert (intervals["point"] <= intervals["upper"]).all()
    # All finite
    for col in ["lower", "point", "upper"]:
        assert np.all(np.isfinite(intervals[col].to_numpy()))


@pytest.mark.parametrize("alpha", [0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50])
def test_predictor_valid_intervals_at_various_alphas(alpha, parametric_data):
    cp = InsuranceConformalPredictor(
        model=parametric_data["model"],
        nonconformity="pearson_weighted",
        tweedie_power=1.0,
    )
    cp.calibrate(parametric_data["X_cal"], parametric_data["y_cal"])
    intervals = cp.predict_interval(parametric_data["X_test"], alpha=alpha)

    assert (intervals["lower"] >= 0.0).all()
    assert (intervals["lower"] <= intervals["upper"]).all()
    assert len(intervals) == len(parametric_data["X_test"])


# ---------------------------------------------------------------------------
# Score shape checks: various input sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 10, 100, 1000])
def test_all_scores_correct_shape(n):
    rng = np.random.default_rng(n)
    y = rng.exponential(2.0, n)
    yhat = rng.uniform(0.5, 5.0, n)

    for nc in ["raw", "pearson", "pearson_weighted"]:
        scores = compute_score(y, yhat, nonconformity=nc, distribution="tweedie", tweedie_power=1.5)
        assert scores.shape == (n,), f"Wrong shape for {nc} with n={n}"


# ---------------------------------------------------------------------------
# Zero observations: y=0 handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted"])
def test_zero_observations_finite(nc):
    """y=0 is a common case in insurance (no claims). All scores should handle it."""
    y = np.zeros(20)
    yhat = np.ones(20) * 2.0
    scores = compute_score(y, yhat, nonconformity=nc, distribution="poisson")
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


def test_zero_observations_deviance_poisson():
    y = np.zeros(20)
    yhat = np.ones(20) * 2.0
    scores = deviance_score(y, yhat, distribution="poisson")
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


def test_zero_observations_anscombe_poisson():
    y = np.zeros(20)
    yhat = np.ones(20) * 2.0
    scores = anscombe_score(y, yhat, distribution="poisson")
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


# ---------------------------------------------------------------------------
# Score symmetry: |y - yhat| = |yhat - y|
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nc", ["raw"])
def test_raw_score_symmetric(nc):
    """raw_score is symmetric: s(y, yhat) == s(yhat, y)."""
    rng = np.random.default_rng(0)
    y = rng.exponential(2.0, 50)
    yhat = rng.uniform(0.5, 5.0, 50)
    s1 = raw_score(y, yhat)
    s2 = raw_score(yhat, y)
    np.testing.assert_allclose(s1, s2)


# ---------------------------------------------------------------------------
# conformal_quantile: finite-sample size formula verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,alpha,expected_level", [
    # level = ceil((n+1)(1-alpha)) / n, capped at 1.0
    (10, 0.10, min(np.ceil(11 * 0.90) / 10, 1.0)),
    (10, 0.50, min(np.ceil(11 * 0.50) / 10, 1.0)),
    (20, 0.10, min(np.ceil(21 * 0.90) / 20, 1.0)),
    (100, 0.10, min(np.ceil(101 * 0.90) / 100, 1.0)),
])
def test_conformal_quantile_level_formula(n, alpha, expected_level):
    scores = np.arange(1, n + 1, dtype=float)
    q = conformal_quantile(scores, alpha)
    expected_q = np.quantile(scores, expected_level, method="higher")
    assert q == pytest.approx(expected_q)


# ---------------------------------------------------------------------------
# Predictor: all score types with deviance and anscombe on calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nc,dist", [
    ("deviance", "poisson"),
    ("deviance", "tweedie"),
    ("anscombe", "poisson"),
    ("anscombe", "tweedie"),
])
def test_predictor_deviance_anscombe_scores(nc, dist, parametric_data):
    cp = InsuranceConformalPredictor(
        model=parametric_data["model"],
        nonconformity=nc,
        distribution=dist,
        tweedie_power=1.0,
    )
    cp.calibrate(parametric_data["X_cal"], parametric_data["y_cal"])
    assert cp.is_calibrated_
    assert cp.n_calibration_ == len(parametric_data["y_cal"])
    assert np.all(cp.cal_scores_ >= 0)

    intervals = cp.predict_interval(parametric_data["X_test"], alpha=0.10)
    assert len(intervals) == len(parametric_data["X_test"])
    assert (intervals["lower"] <= intervals["upper"]).all()
