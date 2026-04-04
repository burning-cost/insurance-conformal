"""
Property-based and parametric tests for insurance-conformal.

These tests verify mathematical invariants that should hold across many
combinations of inputs: score monotonicity, interval width ordering,
conformal coverage guarantee, and numerical stability under extreme values.

Rather than testing specific numbers, these tests verify relationships
that must always hold. They catch regressions in mathematical correctness
that point tests might miss.
"""

from __future__ import annotations

import warnings

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
# Shared model fixture
# ---------------------------------------------------------------------------


class LogLinearModel:
    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(X) @ self.beta)


@pytest.fixture(scope="module")
def base_predictor():
    rng = np.random.default_rng(42)
    n = 1200
    beta = np.array([0.4, -0.15, 0.1])
    X = rng.normal(size=(n, 3))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    model = LogLinearModel(beta)
    cp = InsuranceConformalPredictor(model=model, nonconformity="pearson_weighted", tweedie_power=1.0)
    cp.calibrate(X[:400], y[:400])
    return cp, X[400:800], y[400:800]


# ---------------------------------------------------------------------------
# Property: Score is non-negative for all input combinations
# ---------------------------------------------------------------------------


class TestScoreNonNegativity:
    """All non-conformity scores must be non-negative."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_raw_nonneg_random(self, seed):
        rng = np.random.default_rng(seed)
        y = rng.exponential(2.0, 200)
        yhat = rng.uniform(0.1, 8.0, 200)
        assert np.all(raw_score(y, yhat) >= 0)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_pearson_nonneg_random(self, seed):
        rng = np.random.default_rng(seed)
        y = rng.exponential(2.0, 100)
        yhat = rng.uniform(0.1, 8.0, 100)
        assert np.all(pearson_score(y, yhat) >= 0)

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
    def test_pearson_weighted_nonneg(self, p):
        rng = np.random.default_rng(0)
        y = rng.exponential(2.0, 100)
        yhat = rng.uniform(0.5, 5.0, 100)
        assert np.all(pearson_weighted_score(y, yhat, tweedie_power=p) >= 0)

    @pytest.mark.parametrize("dist,p", [
        ("poisson", 1.0), ("gamma", 2.0), ("tweedie", 1.5), ("tweedie", 3.0)
    ])
    def test_deviance_nonneg(self, dist, p):
        rng = np.random.default_rng(0)
        y = rng.exponential(2.0, 100)
        yhat = rng.uniform(0.5, 5.0, 100)
        scores = deviance_score(y, yhat, distribution=dist, tweedie_power=p)
        assert np.all(scores >= 0)

    @pytest.mark.parametrize("dist,p", [
        ("poisson", 1.0), ("gamma", 2.0), ("tweedie", 1.5)
    ])
    def test_anscombe_nonneg(self, dist, p):
        rng = np.random.default_rng(0)
        y = rng.exponential(2.0, 100)
        yhat = rng.uniform(0.5, 5.0, 100)
        scores = anscombe_score(y, yhat, distribution=dist, tweedie_power=p)
        assert np.all(scores >= 0)


# ---------------------------------------------------------------------------
# Property: Score is zero at y=yhat for differentiable scores
# ---------------------------------------------------------------------------


class TestScoreZeroAtPerfectPrediction:
    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0])
    def test_pearson_weighted_zero_at_perfect(self, p):
        yhat = np.array([1.0, 2.0, 5.0, 10.0])
        scores = pearson_weighted_score(yhat, yhat, tweedie_power=p)
        np.testing.assert_allclose(scores, np.zeros(4), atol=1e-12)

    def test_deviance_zero_at_perfect_poisson(self):
        yhat = np.array([0.5, 1.0, 2.0, 5.0])
        scores = deviance_score(yhat, yhat, distribution="poisson")
        np.testing.assert_allclose(scores, np.zeros(4), atol=1e-10)

    def test_deviance_zero_at_perfect_gamma(self):
        yhat = np.array([0.5, 1.0, 2.0])
        scores = deviance_score(yhat, yhat, distribution="gamma")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_deviance_zero_at_perfect_tweedie(self):
        yhat = np.array([1.0, 3.0, 7.0])
        for p in [1.5, 1.8]:
            scores = deviance_score(yhat, yhat, distribution="tweedie", tweedie_power=p)
            # Tweedie deviance uses numerical integration; allow ~1e-6 residual
            np.testing.assert_allclose(scores, np.zeros(3), atol=1e-6)

    def test_anscombe_zero_at_perfect_poisson(self):
        yhat = np.array([1.0, 4.0, 9.0])
        scores = anscombe_score(yhat, yhat, distribution="poisson")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_anscombe_zero_at_perfect_gamma(self):
        yhat = np.array([2.0, 3.0, 5.0])
        scores = anscombe_score(yhat, yhat, distribution="gamma")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_raw_zero_at_perfect(self):
        yhat = np.array([1.0, 2.0, 3.0])
        scores = raw_score(yhat, yhat)
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-12)

    def test_pearson_zero_at_perfect(self):
        yhat = np.array([1.0, 4.0, 9.0])
        scores = pearson_score(yhat, yhat)
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-12)


# ---------------------------------------------------------------------------
# Property: Inversion roundtrip for closed-form scores
# ---------------------------------------------------------------------------


class TestInversionRoundtrip:
    """
    For closed-form scores (raw, pearson, pearson_weighted), the inversion
    must be exact: score(invert(q, yhat), yhat) == q.
    """

    @pytest.mark.parametrize("yhat_val", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_raw_inversion_roundtrip(self, yhat_val):
        yhat = np.array([yhat_val])
        q = 0.7
        _, upper = invert_score(yhat, q, "raw", clip_lower=0.0)
        score_at_upper = raw_score(upper, yhat)
        np.testing.assert_allclose(score_at_upper, [q], atol=1e-10)

    @pytest.mark.parametrize("yhat_val", [1.0, 2.0, 4.0, 9.0])
    def test_pearson_inversion_roundtrip(self, yhat_val):
        yhat = np.array([yhat_val])
        q = 0.5
        _, upper = invert_score(yhat, q, "pearson", clip_lower=0.0)
        score_at_upper = pearson_score(upper, yhat)
        np.testing.assert_allclose(score_at_upper, [q], atol=1e-10)

    @pytest.mark.parametrize("p,yhat_val", [
        (1.0, 2.0), (1.5, 3.0), (2.0, 4.0)
    ])
    def test_pearson_weighted_inversion_roundtrip(self, p, yhat_val):
        yhat = np.array([yhat_val])
        q = 0.3
        _, upper = invert_score(yhat, q, "pearson_weighted", tweedie_power=p, clip_lower=0.0)
        score_at_upper = pearson_weighted_score(upper, yhat, tweedie_power=p)
        np.testing.assert_allclose(score_at_upper, [q], atol=1e-10)


# ---------------------------------------------------------------------------
# Property: Interval width monotone in alpha
# ---------------------------------------------------------------------------


class TestIntervalWidthMonotone:
    """
    Smaller alpha (higher coverage target) must produce wider intervals.
    """

    @pytest.fixture(scope="class")
    def cp(self):
        rng = np.random.default_rng(7)
        n = 800
        beta = np.array([0.3, -0.1])
        X = rng.normal(size=(n, 2))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)
        model = LogLinearModel(beta)
        cp = InsuranceConformalPredictor(model=model, tweedie_power=1.0)
        cp.calibrate(X[:300], y[:300])
        return cp, X[300:]

    @pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted"])
    def test_width_monotone_in_alpha(self, nc, cp):
        predictor, X_test = cp
        # Create a fresh predictor for this nc
        rng = np.random.default_rng(7)
        n = 800
        beta = np.array([0.3, -0.1])
        X = rng.normal(size=(n, 2))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)
        model = LogLinearModel(beta)
        cp_nc = InsuranceConformalPredictor(model=model, nonconformity=nc, tweedie_power=1.0)
        cp_nc.calibrate(X[:300], y[:300])

        alphas = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
        mean_widths = []
        for a in alphas:
            intervals = cp_nc.predict_interval(X[300:], alpha=a)
            w = (intervals["upper"] - intervals["lower"]).mean()
            mean_widths.append(float(w))

        # Widths should decrease as alpha increases (larger alpha = less coverage = narrower)
        for i in range(len(mean_widths) - 1):
            assert mean_widths[i] >= mean_widths[i + 1] - 1e-6, (
                f"Width not monotone for {nc} at alphas {alphas[i]} vs {alphas[i+1]}: "
                f"{mean_widths[i]:.4f} vs {mean_widths[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# Property: Conformal quantile monotone in alpha
# ---------------------------------------------------------------------------


class TestConformalQuantileMonotone:
    """
    conformal_quantile(scores, alpha) must be non-increasing in alpha.
    """

    @pytest.mark.parametrize("n", [20, 100, 500])
    def test_monotone_in_alpha_for_various_n(self, n):
        rng = np.random.default_rng(n)
        scores = rng.exponential(2.0, n)
        alphas = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
        quantiles = [conformal_quantile(scores, a) for a in alphas]
        for i in range(len(quantiles) - 1):
            assert quantiles[i] >= quantiles[i + 1] - 1e-10, (
                f"Quantile not monotone at n={n}: q[{alphas[i]}]={quantiles[i]:.4f} "
                f"< q[{alphas[i+1]}]={quantiles[i+1]:.4f}"
            )

    @pytest.mark.parametrize("seed", range(5))
    def test_monotone_random_scores(self, seed):
        rng = np.random.default_rng(seed)
        scores = rng.lognormal(0, 1, 200)
        q_01 = conformal_quantile(scores, 0.05)
        q_50 = conformal_quantile(scores, 0.50)
        assert q_01 >= q_50


# ---------------------------------------------------------------------------
# Property: Lower bound always <= point <= upper bound
# ---------------------------------------------------------------------------


class TestBoundsOrdering:
    @pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted"])
    def test_lower_leq_point_leq_upper(self, nc, base_predictor):
        cp, X_test, y_test = base_predictor
        rng = np.random.default_rng(0)
        n = 800
        beta = np.array([0.4, -0.15, 0.1])
        X = rng.normal(size=(n, 3))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)
        model = LogLinearModel(beta)
        cp_nc = InsuranceConformalPredictor(model=model, nonconformity=nc, tweedie_power=1.0)
        cp_nc.calibrate(X[:400], y[:400])
        intervals = cp_nc.predict_interval(X[400:], alpha=0.10)
        assert (intervals["lower"] <= intervals["point"]).all()
        assert (intervals["point"] <= intervals["upper"]).all()

    def test_lower_nonneg_for_default_clip(self, base_predictor):
        cp, X_test, y_test = base_predictor
        intervals = cp.predict_interval(X_test, alpha=0.10)
        assert (intervals["lower"] >= 0.0).all()

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20, 0.30])
    def test_bounds_ordering_multiple_alphas(self, alpha, base_predictor):
        cp, X_test, y_test = base_predictor
        intervals = cp.predict_interval(X_test, alpha=alpha)
        assert (intervals["lower"] <= intervals["upper"]).all()


# ---------------------------------------------------------------------------
# Property: IPCW quantile monotone in alpha
# ---------------------------------------------------------------------------


class TestIpcwQuantileMonotone:
    @pytest.mark.parametrize("seed", range(5))
    def test_monotone_in_alpha(self, seed):
        rng = np.random.default_rng(seed)
        n = 100
        scores = rng.exponential(2.0, n)
        weights = rng.uniform(0.1, 2.0, n)
        alphas = [0.05, 0.10, 0.20, 0.30, 0.50]
        quantiles = [ipcw_weighted_quantile(scores, weights, a) for a in alphas]
        for i in range(len(quantiles) - 1):
            assert quantiles[i] >= quantiles[i + 1] - 1e-10

    def test_returns_score_from_input(self):
        """The returned quantile must be one of the input score values."""
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 10, 50)
        weights = rng.uniform(0.5, 2.0, 50)
        q = ipcw_weighted_quantile(scores, weights, alpha=0.20)
        assert q in scores


# ---------------------------------------------------------------------------
# Property: Coverage guarantee
# ---------------------------------------------------------------------------


class TestCoverageGuaranteeProperty:
    """
    Verify the conformal coverage guarantee holds across different nonconformity
    scores and alpha values. This is the core theorem we depend on.
    """

    TOLERANCE = 0.06  # 6pp tolerance for finite samples

    @pytest.mark.parametrize("nc", ["raw", "pearson"])
    def test_coverage_holds_for_scores(self, nc):
        rng = np.random.default_rng(99)
        n = 2000
        beta = np.array([0.5, -0.3, 0.2])
        X = rng.normal(size=(n, 3))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)

        model = LogLinearModel(beta)
        cp = InsuranceConformalPredictor(model=model, nonconformity=nc, tweedie_power=1.0)
        cp.calibrate(X[:800], y[:800])

        intervals = cp.predict_interval(X[800:], alpha=0.10)
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        y_test = y[800:]
        covered = (y_test >= lower) & (y_test <= upper)
        coverage = covered.mean()

        assert coverage >= 0.90 - self.TOLERANCE, (
            f"Coverage {coverage:.3%} < {0.90 - self.TOLERANCE:.3%} for {nc}"
        )

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 0.30])
    def test_coverage_at_various_alphas(self, alpha):
        rng = np.random.default_rng(77)
        n = 2000
        beta = np.array([0.4, -0.2])
        X = rng.normal(size=(n, 2))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)

        model = LogLinearModel(beta)
        cp = InsuranceConformalPredictor(model=model, nonconformity="raw", tweedie_power=1.0)
        cp.calibrate(X[:800], y[:800])

        intervals = cp.predict_interval(X[800:], alpha=alpha)
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        y_test = y[800:]
        covered = (y_test >= lower) & (y_test <= upper)
        coverage = covered.mean()

        assert coverage >= (1 - alpha) - self.TOLERANCE, (
            f"Coverage {coverage:.3%} < target {1 - alpha:.3%} (alpha={alpha})"
        )


# ---------------------------------------------------------------------------
# Property: Score invariance checks
# ---------------------------------------------------------------------------


class TestScoreInvarianceProperties:
    def test_raw_scale_invariant(self):
        """raw_score(c*y, c*yhat) = c * raw_score(y, yhat) for c > 0."""
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.5, 1.5, 2.5])
        c = 2.0
        s1 = raw_score(y, yhat)
        s2 = raw_score(c * y, c * yhat)
        np.testing.assert_allclose(s2, c * s1)

    def test_pearson_score_at_unit_yhat(self):
        """At yhat=1, pearson_score = |y - 1|."""
        y = np.array([0.0, 1.0, 2.0, 3.0])
        yhat = np.ones(4)
        scores = pearson_score(y, yhat)
        np.testing.assert_allclose(scores, np.abs(y - 1.0))

    def test_pearson_weighted_reduces_to_pearson_p1(self):
        """pearson_weighted(y, yhat, p=1) == pearson_score(y, yhat)."""
        rng = np.random.default_rng(0)
        y = rng.exponential(2.0, 50)
        yhat = rng.uniform(0.5, 5.0, 50)
        pw = pearson_weighted_score(y, yhat, tweedie_power=1.0)
        ps = pearson_score(y, yhat)
        np.testing.assert_allclose(pw, ps, rtol=1e-10)

    def test_deviance_p1_equals_poisson(self):
        """deviance(y, yhat, tweedie, p=1) == deviance(y, yhat, poisson)."""
        rng = np.random.default_rng(1)
        y = rng.exponential(2.0, 50)
        yhat = rng.uniform(0.5, 5.0, 50)
        d_t1 = deviance_score(y, yhat, distribution="tweedie", tweedie_power=1.0)
        d_p = deviance_score(y, yhat, distribution="poisson")
        np.testing.assert_allclose(d_t1, d_p, rtol=1e-6)

    def test_deviance_p2_equals_gamma(self):
        """deviance(y, yhat, tweedie, p=2) == deviance(y, yhat, gamma)."""
        rng = np.random.default_rng(2)
        y = rng.exponential(2.0, 50)
        yhat = rng.uniform(0.5, 5.0, 50)
        d_t2 = deviance_score(y, yhat, distribution="tweedie", tweedie_power=2.0)
        d_g = deviance_score(y, yhat, distribution="gamma")
        np.testing.assert_allclose(d_t2, d_g, rtol=1e-6)

    def test_anscombe_p1_equals_poisson(self):
        rng = np.random.default_rng(3)
        y = rng.exponential(2.0, 50)
        yhat = rng.uniform(0.5, 5.0, 50)
        a_t1 = anscombe_score(y, yhat, distribution="tweedie", tweedie_power=1.0)
        a_p = anscombe_score(y, yhat, distribution="poisson")
        np.testing.assert_allclose(a_t1, a_p, rtol=1e-6)

    def test_anscombe_p2_equals_gamma(self):
        rng = np.random.default_rng(4)
        y = rng.exponential(2.0, 50)
        yhat = rng.uniform(0.5, 5.0, 50)
        a_t2 = anscombe_score(y, yhat, distribution="tweedie", tweedie_power=2.0)
        a_g = anscombe_score(y, yhat, distribution="gamma")
        np.testing.assert_allclose(a_t2, a_g, rtol=1e-6)


# ---------------------------------------------------------------------------
# Property: Score increases monotonically with residual magnitude
# ---------------------------------------------------------------------------


class TestScoreMonotonicity:
    """
    For a fixed yhat, all scores should be monotone increasing in |y - yhat|.
    """

    def test_raw_monotone_in_residual(self):
        yhat = np.full(5, 2.0)
        y_vals = np.array([2.0, 2.5, 3.0, 4.0, 5.0])  # increasing residual
        scores = raw_score(y_vals, yhat)
        assert np.all(np.diff(scores) > 0)

    def test_pearson_monotone_in_residual(self):
        yhat = np.full(5, 4.0)
        y_vals = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        scores = pearson_score(y_vals, yhat)
        assert np.all(np.diff(scores) > 0)

    def test_pearson_weighted_monotone_in_residual(self):
        yhat = np.full(5, 3.0)
        y_vals = np.array([3.0, 3.5, 4.0, 5.0, 6.0])
        scores = pearson_weighted_score(y_vals, yhat, tweedie_power=1.5)
        assert np.all(np.diff(scores) > 0)

    def test_deviance_poisson_monotone_in_residual(self):
        yhat = np.full(4, 2.0)
        y_vals = np.array([2.0, 2.5, 3.0, 4.0])
        scores = deviance_score(y_vals, yhat, distribution="poisson")
        assert np.all(np.diff(scores) >= 0)

    def test_anscombe_poisson_monotone_in_residual(self):
        yhat = np.full(4, 2.0)
        y_vals = np.array([2.0, 2.5, 3.0, 4.0])
        scores = anscombe_score(y_vals, yhat, distribution="poisson")
        assert np.all(np.diff(scores) >= 0)


# ---------------------------------------------------------------------------
# Boundary: single observation in calibration set
# ---------------------------------------------------------------------------


class TestSingleObservationCalibration:
    @pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted"])
    def test_single_cal_obs_produces_intervals(self, nc):
        rng = np.random.default_rng(0)
        beta = np.array([0.3, -0.1])
        X = rng.normal(size=(100, 2))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)

        model = LogLinearModel(beta)
        cp = InsuranceConformalPredictor(model=model, nonconformity=nc, tweedie_power=1.0)
        cp.calibrate(X[:1], y[:1])
        intervals = cp.predict_interval(X[1:], alpha=0.10)
        assert len(intervals) == 99
        assert (intervals["lower"] <= intervals["upper"]).all()


# ---------------------------------------------------------------------------
# Numerical stability: extreme yhat values
# ---------------------------------------------------------------------------


class TestNumericalStabilityExtreme:
    def test_very_small_yhat_pearson(self):
        """Very small yhat should produce finite scores."""
        y = np.array([0.0, 0.01, 0.1])
        yhat = np.array([1e-5, 1e-5, 1e-5])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            scores = pearson_score(y, yhat)
        assert np.all(np.isfinite(scores))

    def test_large_yhat_raw_finite(self):
        y = np.array([1e6])
        yhat = np.array([1e6 + 100.0])
        scores = raw_score(y, yhat)
        assert np.isfinite(scores[0])
        assert scores[0] == pytest.approx(100.0)

    def test_large_yhat_pearson_weighted_finite(self):
        y = np.array([1e4])
        yhat = np.array([1e4])
        scores = pearson_weighted_score(y, yhat, tweedie_power=1.5)
        assert np.isfinite(scores[0])
        assert scores[0] == pytest.approx(0.0, abs=1e-10)

    def test_large_n_scores_all_finite(self):
        rng = np.random.default_rng(0)
        n = 50_000
        y = rng.exponential(2.0, n)
        yhat = rng.uniform(0.5, 10.0, n)
        scores = compute_score(y, yhat, nonconformity="pearson_weighted",
                               distribution="tweedie", tweedie_power=1.5)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)


# ---------------------------------------------------------------------------
# Boundary: invert_score with quantile=0
# ---------------------------------------------------------------------------


class TestInvertScoreQuantileZero:
    @pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted"])
    def test_zero_quantile_gives_point_prediction(self, nc):
        """At q=0, inversion gives [yhat, yhat] (zero width), clipped at 0."""
        yhat = np.array([1.0, 3.0, 7.0])
        lower, upper = invert_score(yhat, 0.0, nc, clip_lower=0.0)
        np.testing.assert_allclose(lower, yhat, atol=1e-10)
        np.testing.assert_allclose(upper, yhat, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge: conformal_quantile with all identical scores
# ---------------------------------------------------------------------------


class TestConformalQuantileDegenerate:
    def test_all_identical_scores(self):
        scores = np.full(100, 5.0)
        for alpha in [0.05, 0.10, 0.20]:
            q = conformal_quantile(scores, alpha)
            assert q == 5.0

    def test_two_distinct_values(self):
        scores = np.array([1.0, 10.0])
        q_tight = conformal_quantile(scores, alpha=0.10)
        q_loose = conformal_quantile(scores, alpha=0.60)
        assert q_tight >= q_loose

    def test_sorted_vs_unsorted(self):
        """Order of scores should not matter."""
        rng = np.random.default_rng(0)
        scores = rng.exponential(2.0, 100)
        q_orig = conformal_quantile(scores, alpha=0.10)
        q_sorted = conformal_quantile(np.sort(scores), alpha=0.10)
        assert q_orig == q_sorted

    def test_large_n_stable(self):
        rng = np.random.default_rng(1)
        scores = rng.lognormal(0, 2, 100_000)
        q = conformal_quantile(scores, alpha=0.10)
        assert np.isfinite(q)
        assert q > 0
