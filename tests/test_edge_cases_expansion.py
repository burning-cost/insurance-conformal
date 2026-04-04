"""
Expanded edge-case, error-path, and boundary-condition tests.

Targets modules that had incomplete coverage:
- scores.py: extreme Tweedie powers, zero variance, large inputs, shape edge cases
- utils.py: ipcw_weighted_quantile edge cases, temporal_split edge cases
- predictor.py: summary(), coverage_by_decile(), predict() method, edge case inputs
- _solvency.py: SolvencyCapitalRange repr, exposure edge cases, error paths
- diagnostics_ext.py: subgroup_coverage, width_efficiency_comparison
- diagnostics.py (CoverageDiagnostics): additional property and method tests
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_conformal.scores import (
    NonconformityScore,
    anscombe_score,
    compute_score,
    deviance_score,
    invert_score,
    pearson_score,
    pearson_weighted_score,
    raw_score,
)
from insurance_conformal.utils import (
    apply_exposure,
    as_numpy,
    conformal_quantile,
    ipcw_weighted_quantile,
    temporal_split,
)
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal._solvency import solvency_capital_range, SolvencyCapitalRange
from insurance_conformal.diagnostics import CoverageDiagnostics
from insurance_conformal.diagnostics_ext import (
    ert_coverage_gap,
    subgroup_coverage,
    width_efficiency_comparison,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


class LinearPoissonModel:
    """Log-linear Poisson model for testing."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(X) @ self.beta)


@pytest.fixture(scope="module")
def calibrated_predictor():
    rng = np.random.default_rng(77)
    n = 1500
    beta = np.array([0.4, -0.2, 0.1])
    X = rng.normal(size=(n, 3))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)

    model = LinearPoissonModel(beta)
    cp = InsuranceConformalPredictor(model=model, nonconformity="pearson_weighted", tweedie_power=1.0)
    cp.calibrate(X[:500], y[:500])
    return cp, X[500:1000], y[500:1000], X[1000:], y[1000:]


@pytest.fixture(scope="module")
def simple_data():
    rng = np.random.default_rng(88)
    n = 600
    beta = np.array([0.3, -0.1])
    X = rng.normal(size=(n, 2))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    model = LinearPoissonModel(beta)
    return {
        "X_cal": X[:200], "y_cal": y[:200],
        "X_test": X[200:400], "y_test": y[200:400],
        "model": model,
    }


# ---------------------------------------------------------------------------
# Scores: NonconformityScore enum
# ---------------------------------------------------------------------------


class TestNonconformityScoreEnum:
    def test_all_values_accessible(self):
        values = {s.value for s in NonconformityScore}
        assert values == {"raw", "pearson", "pearson_weighted", "deviance", "anscombe"}

    def test_enum_from_string(self):
        assert NonconformityScore("raw") == NonconformityScore.RAW
        assert NonconformityScore("deviance") == NonconformityScore.DEVIANCE

    def test_enum_is_str(self):
        # NonconformityScore inherits from str
        assert isinstance(NonconformityScore.RAW, str)

    def test_invalid_enum_raises(self):
        with pytest.raises(ValueError):
            NonconformityScore("not_a_score")


# ---------------------------------------------------------------------------
# Scores: raw_score edge cases
# ---------------------------------------------------------------------------


class TestRawScoreEdgeCases:
    def test_single_element(self):
        result = raw_score(np.array([3.0]), np.array([1.5]))
        np.testing.assert_allclose(result, [1.5])

    def test_2d_like_input_flattened(self):
        """raw_score should handle 1D inputs properly."""
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.0, 2.0, 3.0])
        result = raw_score(y, yhat)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_mismatched_shapes_raise(self):
        with pytest.raises(Exception):
            # Use shapes that cannot broadcast: (3,) vs (2,) raises ValueError
            raw_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_large_values(self):
        y = np.array([1e6])
        yhat = np.array([1e6 + 1.0])
        result = raw_score(y, yhat)
        np.testing.assert_allclose(result, [1.0])

    def test_list_input_accepted(self):
        result = raw_score([1.0, 2.0], [1.5, 1.5])
        np.testing.assert_allclose(result, [0.5, 0.5])


# ---------------------------------------------------------------------------
# Scores: pearson_score edge cases
# ---------------------------------------------------------------------------


class TestPearsonScoreEdgeCases:
    def test_negative_prediction_clipped_and_warned(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pearson_score(np.array([1.0]), np.array([-1.0]))
            assert any("clipped" in str(warning.message).lower() for warning in w)

    def test_multiple_negative_predictions_warned(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pearson_score(np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 1.0]))
            assert len(w) == 1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            pearson_score(np.array([1.0, 2.0]), np.array([1.0]))

    def test_yhat_equals_y_gives_zero(self):
        y = np.array([2.0, 4.0, 1.0])
        result = pearson_score(y, y)
        np.testing.assert_allclose(result, np.zeros(3))

    def test_large_yhat_small_residual(self):
        """For large yhat, Pearson score is small relative to raw."""
        y = np.array([100.0])
        yhat = np.array([99.0])
        ps = pearson_score(y, yhat)
        rs = raw_score(y, yhat)
        assert ps < rs


# ---------------------------------------------------------------------------
# Scores: pearson_weighted_score edge cases
# ---------------------------------------------------------------------------


class TestPearsonWeightedEdgeCases:
    def test_p0_gives_raw_residual(self):
        """At p=0, yhat^(0/2) = 1, so pearson_weighted equals raw."""
        y = np.array([3.0, 5.0])
        yhat = np.array([2.0, 4.0])
        pw = pearson_weighted_score(y, yhat, tweedie_power=0.0)
        rs = raw_score(y, yhat)
        np.testing.assert_allclose(pw, rs)

    def test_p3_gives_narrow_intervals_for_large_yhat(self):
        """Higher p means stronger normalisation for large yhat."""
        y = np.array([10.0])
        yhat = np.array([9.0])
        pw_p2 = pearson_weighted_score(y, yhat, tweedie_power=2.0)
        pw_p3 = pearson_weighted_score(y, yhat, tweedie_power=3.0)
        # Higher p → stronger denominator → smaller score
        assert pw_p3 < pw_p2

    def test_yhat_equals_y_gives_zero(self):
        y = np.array([5.0])
        result = pearson_weighted_score(y, y, tweedie_power=1.5)
        np.testing.assert_allclose(result, [0.0])

    def test_output_shape_preserved(self):
        rng = np.random.default_rng(0)
        n = 150
        y = rng.uniform(0.1, 10, n)
        yhat = rng.uniform(0.1, 10, n)
        result = pearson_weighted_score(y, yhat, tweedie_power=1.5)
        assert result.shape == (n,)

    def test_all_same_yhat_gives_same_score(self):
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.full(3, 4.0)
        scores = pearson_weighted_score(y, yhat, tweedie_power=2.0)
        # Denominator is same for all, so scores just differ by |y - yhat|
        expected = np.abs(y - yhat) / (4.0 ** 1.0)
        np.testing.assert_allclose(scores, expected)


# ---------------------------------------------------------------------------
# Scores: deviance_score edge cases
# ---------------------------------------------------------------------------


class TestDevianceScoreEdgeCases:
    def test_poisson_with_y_equal_yhat(self):
        yhat = np.array([0.5, 1.0, 3.0])
        scores = deviance_score(yhat, yhat, distribution="poisson")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_gamma_with_y_equal_yhat(self):
        yhat = np.array([1.0, 2.0, 5.0])
        scores = deviance_score(yhat, yhat, distribution="gamma")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_gamma_zero_y_stays_finite(self):
        """y=0 with gamma distribution; log(0) handled with fallback."""
        y = np.array([0.0, 1.0])
        yhat = np.array([1.0, 1.0])
        scores = deviance_score(y, yhat, distribution="gamma")
        assert np.all(np.isfinite(scores))

    def test_tweedie_power_15_nonneg(self):
        rng = np.random.default_rng(3)
        y = rng.exponential(2.0, 100)
        yhat = rng.uniform(0.5, 5.0, 100)
        scores = deviance_score(y, yhat, distribution="tweedie", tweedie_power=1.5)
        assert np.all(scores >= 0)

    def test_tweedie_power_3_nonneg(self):
        rng = np.random.default_rng(4)
        y = rng.exponential(3.0, 50)
        yhat = rng.uniform(1.0, 8.0, 50)
        scores = deviance_score(y, yhat, distribution="tweedie", tweedie_power=3.0)
        assert np.all(scores >= 0)

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            deviance_score(np.array([1.0]), np.array([1.0]), distribution="exponential")

    def test_case_insensitive_distribution(self):
        """Distribution string should be case-insensitive."""
        y = np.array([2.0])
        yhat = np.array([1.5])
        s1 = deviance_score(y, yhat, distribution="Poisson")
        s2 = deviance_score(y, yhat, distribution="poisson")
        np.testing.assert_allclose(s1, s2)

    def test_large_y_values(self):
        """Very large y values should produce finite scores."""
        y = np.array([1e4])
        yhat = np.array([1e3])
        score = deviance_score(y, yhat, distribution="poisson")
        assert np.isfinite(score[0])


# ---------------------------------------------------------------------------
# Scores: anscombe_score edge cases
# ---------------------------------------------------------------------------


class TestAnscombeScoreEdgeCases:
    def test_gamma_zero_at_yhat(self):
        yhat = np.array([1.0, 2.0, 5.0])
        scores = anscombe_score(yhat, yhat, distribution="gamma")
        np.testing.assert_allclose(scores, np.zeros(3), atol=1e-10)

    def test_tweedie_general_zero_at_yhat(self):
        yhat = np.array([2.0, 4.0])
        scores = anscombe_score(yhat, yhat, distribution="tweedie", tweedie_power=1.5)
        np.testing.assert_allclose(scores, np.zeros(2), atol=1e-10)

    def test_tweedie_p2_equals_gamma(self):
        y = np.array([2.0, 4.0, 1.0])
        yhat = np.array([1.5, 3.5, 0.8])
        a_t2 = anscombe_score(y, yhat, distribution="tweedie", tweedie_power=2.0)
        a_g = anscombe_score(y, yhat, distribution="gamma")
        np.testing.assert_allclose(a_t2, a_g, rtol=1e-6)

    def test_poisson_nonnegative_for_y_zero(self):
        """y=0 is a valid observation for Poisson."""
        y = np.array([0.0, 0.0])
        yhat = np.array([1.0, 2.0])
        scores = anscombe_score(y, yhat, distribution="poisson")
        assert np.all(scores >= 0)
        assert np.all(np.isfinite(scores))

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            anscombe_score(np.array([1.0]), np.array([1.0]), distribution="lognormal")

    def test_shape_preserved(self):
        n = 75
        y = np.ones(n) * 2.0
        yhat = np.ones(n) * 1.5
        result = anscombe_score(y, yhat, distribution="poisson")
        assert result.shape == (n,)

    def test_tweedie_p3_nonnegative(self):
        rng = np.random.default_rng(10)
        y = rng.exponential(2.0, 50)
        yhat = rng.uniform(0.5, 5.0, 50)
        scores = anscombe_score(y, yhat, distribution="tweedie", tweedie_power=3.0)
        assert np.all(scores >= 0)


# ---------------------------------------------------------------------------
# Scores: compute_score dispatch
# ---------------------------------------------------------------------------


class TestComputeScoreDispatch:
    @pytest.mark.parametrize("nc", ["raw", "pearson", "pearson_weighted", "deviance", "anscombe"])
    def test_all_scores_produce_nonneg(self, nc):
        rng = np.random.default_rng(0)
        y = rng.exponential(2.0, 30)
        yhat = rng.uniform(0.5, 4.0, 30)
        scores = compute_score(y, yhat, nonconformity=nc, distribution="poisson")
        assert np.all(scores >= 0), f"Negative score for {nc}"

    @pytest.mark.parametrize("dist", ["poisson", "gamma", "tweedie"])
    def test_deviance_all_distributions(self, dist):
        rng = np.random.default_rng(1)
        y = rng.exponential(2.0, 20)
        yhat = rng.uniform(0.5, 4.0, 20)
        scores = compute_score(y, yhat, nonconformity="deviance", distribution=dist)
        assert np.all(scores >= 0)

    def test_invalid_score_name_raises(self):
        with pytest.raises(ValueError):
            compute_score(np.array([1.0]), np.array([1.0]), "invalid_score")


# ---------------------------------------------------------------------------
# Scores: invert_score edge cases
# ---------------------------------------------------------------------------


class TestInvertScoreEdgeCases:
    def test_raw_quantile_zero_gives_point_predictions(self):
        """At q=0, inversion gives [yhat, yhat] (width=0), clipped at 0."""
        yhat = np.array([1.0, 2.0, 5.0])
        lower, upper = invert_score(yhat, 0.0, "raw", clip_lower=0.0)
        np.testing.assert_allclose(lower, yhat)
        np.testing.assert_allclose(upper, yhat)

    def test_pearson_large_quantile_wide_intervals(self):
        yhat = np.array([4.0])
        lower, upper = invert_score(yhat, 100.0, "pearson", clip_lower=0.0)
        assert upper[0] > 100.0
        assert lower[0] == 0.0  # clipped at zero

    def test_pearson_weighted_p2_symmetry(self):
        """For p=2 and yhat=1, half_width=q, so intervals are yhat ± q."""
        yhat = np.array([1.0])
        q = 0.5
        lower, upper = invert_score(yhat, q, "pearson_weighted", tweedie_power=2.0, clip_lower=0.0)
        # half_width = q * yhat^(2/2) = q * 1 = 0.5
        np.testing.assert_allclose(upper, [1.5])
        np.testing.assert_allclose(lower, [0.5])

    def test_lower_never_exceeds_upper(self):
        yhat = np.array([0.1, 1.0, 5.0, 20.0])
        for nc in ["raw", "pearson", "pearson_weighted"]:
            lower, upper = invert_score(yhat, 0.5, nc, clip_lower=0.0)
            assert np.all(lower <= upper), f"lower > upper for {nc}"

    def test_clip_lower_custom_value(self):
        yhat = np.array([1.0])
        lower, upper = invert_score(yhat, 5.0, "raw", clip_lower=2.0)
        assert lower[0] >= 2.0

    def test_deviance_lower_leq_upper(self):
        yhat = np.array([1.0, 3.0])
        lower, upper = invert_score(yhat, 0.3, "deviance", distribution="poisson", clip_lower=0.0)
        assert np.all(lower <= upper)

    def test_anscombe_lower_leq_upper(self):
        yhat = np.array([2.0, 4.0])
        lower, upper = invert_score(yhat, 0.4, "anscombe", distribution="poisson", clip_lower=0.0)
        assert np.all(lower <= upper)

    def test_upper_geq_yhat_for_all_scores(self):
        yhat = np.array([2.0, 5.0])
        for nc in ["raw", "pearson", "pearson_weighted"]:
            _, upper = invert_score(yhat, 1.0, nc, clip_lower=0.0)
            assert np.all(upper >= yhat), f"upper < yhat for {nc}"


# ---------------------------------------------------------------------------
# Utils: ipcw_weighted_quantile edge cases
# ---------------------------------------------------------------------------


class TestIpcwWeightedQuantileEdgeCases:
    def test_basic_returns_valid_quantile(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        q = ipcw_weighted_quantile(scores, weights, alpha=0.10)
        assert q in scores

    def test_uniform_weights_vs_alpha(self):
        """With uniform weights, lower alpha → higher quantile."""
        scores = np.linspace(0, 10, 100)
        weights = np.ones(100)
        q_low = ipcw_weighted_quantile(scores, weights, alpha=0.10)
        q_high = ipcw_weighted_quantile(scores, weights, alpha=0.40)
        assert q_low >= q_high

    def test_concentrated_weights_pull_quantile(self):
        scores = np.array([1.0, 10.0])
        # All weight on the first (smaller) score
        weights_lo = np.array([100.0, 0.01])
        q_lo = ipcw_weighted_quantile(scores, weights_lo, alpha=0.10)
        # All weight on the second (larger) score
        weights_hi = np.array([0.01, 100.0])
        q_hi = ipcw_weighted_quantile(scores, weights_hi, alpha=0.10)
        assert q_lo <= q_hi

    def test_single_observation(self):
        scores = np.array([3.0])
        weights = np.array([1.0])
        q = ipcw_weighted_quantile(scores, weights, alpha=0.10)
        assert q == 3.0

    def test_empty_scores_raise(self):
        with pytest.raises(ValueError, match="empty"):
            ipcw_weighted_quantile(np.array([]), np.array([]), alpha=0.10)

    def test_invalid_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ipcw_weighted_quantile(np.array([1.0, 2.0]), np.ones(2), alpha=0.0)

    def test_invalid_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ipcw_weighted_quantile(np.array([1.0, 2.0]), np.ones(2), alpha=1.0)

    def test_zero_weights_returns_max_with_warning(self):
        scores = np.array([1.0, 5.0, 3.0])
        weights = np.zeros(3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            q = ipcw_weighted_quantile(scores, weights, alpha=0.10)
            assert any("near-zero" in str(warning.message).lower() for warning in w)
        assert q == 5.0

    def test_near_zero_weights_returns_max_with_warning(self):
        scores = np.array([1.0, 5.0])
        weights = np.array([1e-15, 1e-15])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            q = ipcw_weighted_quantile(scores, weights, alpha=0.10)
        assert q == 5.0

    def test_returns_float(self):
        scores = np.array([1.0, 2.0, 3.0])
        weights = np.ones(3)
        q = ipcw_weighted_quantile(scores, weights, alpha=0.20)
        assert isinstance(q, float)

    def test_large_alpha_returns_small_quantile(self):
        scores = np.linspace(0, 10, 50)
        weights = np.ones(50)
        q_large_alpha = ipcw_weighted_quantile(scores, weights, alpha=0.90)
        q_small_alpha = ipcw_weighted_quantile(scores, weights, alpha=0.10)
        assert q_large_alpha <= q_small_alpha

    def test_weight_normalisation_does_not_affect_result(self):
        """Multiplying all weights by a constant should not change the quantile."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        q1 = ipcw_weighted_quantile(scores, weights, alpha=0.20)
        q2 = ipcw_weighted_quantile(scores, weights * 1000, alpha=0.20)
        assert q1 == q2


# ---------------------------------------------------------------------------
# Utils: conformal_quantile additional edge cases
# ---------------------------------------------------------------------------


class TestConformalQuantileEdgeCases:
    def test_large_n_convergence(self):
        """With many observations, quantile should be close to the expected value."""
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 10, 10_000)
        q = conformal_quantile(scores, alpha=0.10)
        # Should be close to the 90th percentile
        expected = np.percentile(scores, 90)
        assert abs(q - expected) < 0.5

    def test_alpha_very_small(self):
        scores = np.arange(1, 101, dtype=float)
        q = conformal_quantile(scores, alpha=0.01)
        # Should be the maximum or near-maximum
        assert q >= 98.0

    def test_alpha_very_large(self):
        scores = np.arange(1, 101, dtype=float)
        q = conformal_quantile(scores, alpha=0.99)
        # Should be near the minimum
        assert q <= 5.0

    def test_returns_float(self):
        scores = np.array([1.0, 2.0, 3.0])
        q = conformal_quantile(scores, alpha=0.20)
        assert isinstance(q, float)

    def test_scores_all_same(self):
        """When all scores are equal, quantile should be that value."""
        scores = np.full(50, 3.14)
        q = conformal_quantile(scores, alpha=0.10)
        assert q == 3.14

    def test_negative_scores_valid(self):
        """Negative non-conformity scores (unusual but valid input) should work."""
        scores = np.array([-3.0, -1.0, 0.0, 1.0, 2.0])
        q = conformal_quantile(scores, alpha=0.20)
        assert np.isfinite(q)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_various_n(self, n):
        scores = np.arange(1, n + 1, dtype=float)
        q = conformal_quantile(scores, alpha=0.20)
        assert np.isfinite(q)
        assert q >= scores.min()


# ---------------------------------------------------------------------------
# Utils: apply_exposure edge cases
# ---------------------------------------------------------------------------


class TestApplyExposureEdgeCases:
    def test_none_exposure_returns_unchanged(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([1.5, 2.5])
        y_out, yhat_out = apply_exposure(y, yhat, exposure=None)
        np.testing.assert_array_equal(y_out, y)
        np.testing.assert_array_equal(yhat_out, yhat)

    def test_exposure_divides_both(self):
        y = np.array([2.0, 4.0])
        yhat = np.array([3.0, 6.0])
        exposure = np.array([0.5, 1.0])
        y_out, yhat_out = apply_exposure(y, yhat, exposure=exposure)
        np.testing.assert_allclose(y_out, [4.0, 4.0])
        np.testing.assert_allclose(yhat_out, [6.0, 6.0])

    def test_zero_exposure_raises(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([1.0, 2.0])
        exposure = np.array([0.5, 0.0])
        with pytest.raises(ValueError, match="positive"):
            apply_exposure(y, yhat, exposure=exposure)

    def test_negative_exposure_raises(self):
        with pytest.raises(ValueError, match="positive"):
            apply_exposure(
                np.array([1.0]), np.array([1.0]), exposure=np.array([-0.5])
            )

    def test_unit_exposure_no_change(self):
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.5, 2.5, 3.5])
        exposure = np.ones(3)
        y_out, yhat_out = apply_exposure(y, yhat, exposure=exposure)
        np.testing.assert_allclose(y_out, y)
        np.testing.assert_allclose(yhat_out, yhat)

    def test_list_exposure_accepted(self):
        y = np.array([2.0])
        yhat = np.array([2.0])
        y_out, yhat_out = apply_exposure(y, yhat, exposure=[0.5])
        np.testing.assert_allclose(y_out, [4.0])
        np.testing.assert_allclose(yhat_out, [4.0])


# ---------------------------------------------------------------------------
# Utils: temporal_split edge cases
# ---------------------------------------------------------------------------


class TestTemporalSplitEdgeCases:
    def test_very_small_calibration_frac(self):
        """calibration_frac=0.01 on n=100 should give at least 1 cal observation."""
        n = 100
        X = pd.DataFrame({"a": np.arange(n)})
        y = np.ones(n)
        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(X, y, calibration_frac=0.01)
        assert len(X_cal) >= 1
        assert len(X_train) + len(X_cal) == n

    def test_large_calibration_frac(self):
        n = 100
        X = pd.DataFrame({"a": np.arange(n)})
        y = np.ones(n)
        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(X, y, calibration_frac=0.9)
        assert len(X_cal) == 90
        assert len(X_train) == 10

    def test_numpy_array_X(self):
        """X as a numpy array should work (gets wrapped in DataFrame)."""
        n = 50
        X = np.random.default_rng(0).normal(size=(n, 3))
        y = np.ones(n)
        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(X, y, calibration_frac=0.2)
        assert len(X_cal) == 10
        assert len(X_train) == 40

    def test_date_col_as_array(self):
        """date_col can be passed as an array instead of a column name."""
        n = 100
        X = pd.DataFrame({"feature": np.random.randn(n)})
        dates = np.arange(n)
        y = np.ones(n)
        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
            X, y, calibration_frac=0.2, date_col=dates
        )
        assert len(X_cal) == 20
        assert len(X_train) == 80

    def test_exposure_none_when_not_provided(self):
        n = 50
        X = pd.DataFrame({"a": np.arange(n)})
        y = np.ones(n)
        _, _, _, _, exp_train, exp_cal = temporal_split(X, y, calibration_frac=0.2)
        assert exp_train is None
        assert exp_cal is None

    def test_total_preserved_with_small_n(self):
        n = 5
        X = pd.DataFrame({"a": np.arange(n)})
        y = np.ones(n)
        X_train, X_cal, y_train, y_cal, _, _ = temporal_split(X, y, calibration_frac=0.4)
        assert len(X_train) + len(X_cal) == n


# ---------------------------------------------------------------------------
# Utils: as_numpy additional cases
# ---------------------------------------------------------------------------


class TestAsNumpyEdgeCases:
    def test_pandas_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = as_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_scalar_int(self):
        result = as_numpy(5)
        assert isinstance(result, np.ndarray)

    def test_empty_list(self):
        result = as_numpy([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_nested_list(self):
        result = as_numpy([[1, 2], [3, 4]])
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Predictor: predict() method
# ---------------------------------------------------------------------------


class TestPredictorPredictMethod:
    def test_predict_before_calibrate(self, simple_data):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        # predict() should work without calibration (it just calls model.predict)
        result = cp.predict(simple_data["X_test"])
        assert isinstance(result, np.ndarray)
        assert len(result) == len(simple_data["X_test"])

    def test_predict_nonneg_for_poisson(self, simple_data):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        result = cp.predict(simple_data["X_test"])
        assert np.all(result > 0)

    def test_predict_consistent_with_interval_point(self, simple_data):
        """predict() and the 'point' column of predict_interval() should agree."""
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])

        point_from_predict = cp.predict(simple_data["X_test"])
        intervals = cp.predict_interval(simple_data["X_test"], alpha=0.10)
        point_from_interval = intervals["point"].to_numpy()

        np.testing.assert_allclose(point_from_predict, point_from_interval)

    def test_predict_with_polars_df(self, simple_data):
        """Polars DataFrame input to predict() should work."""
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        X_pl = pl.DataFrame(simple_data["X_test"])
        result = cp.predict(X_pl)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(simple_data["X_test"])


# ---------------------------------------------------------------------------
# Predictor: summary() method
# ---------------------------------------------------------------------------


class TestPredictorSummary:
    def test_summary_runs_without_error(self, simple_data, capsys):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])
        cp.summary(simple_data["X_test"], simple_data["y_test"], alpha=0.10)
        captured = capsys.readouterr()
        assert "coverage" in captured.out.lower()

    def test_summary_shows_score_name(self, simple_data, capsys):
        cp = InsuranceConformalPredictor(
            model=simple_data["model"],
            nonconformity="pearson_weighted",
            tweedie_power=1.0,
        )
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])
        cp.summary(simple_data["X_test"], simple_data["y_test"], alpha=0.10)
        captured = capsys.readouterr()
        assert "pearson_weighted" in captured.out

    def test_summary_different_alpha(self, simple_data, capsys):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])
        cp.summary(simple_data["X_test"], simple_data["y_test"], alpha=0.20)
        captured = capsys.readouterr()
        assert "80.0%" in captured.out or "80%" in captured.out


# ---------------------------------------------------------------------------
# Predictor: tweedie power auto-detection warning
# ---------------------------------------------------------------------------


class TestPredictorTweediePowerWarning:
    def test_warns_when_power_undetectable(self):
        """An opaque model without TweedieRegressor type should warn."""
        class OpaquePredictModel:
            def predict(self, X):
                return np.ones(len(X))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp = InsuranceConformalPredictor(
                model=OpaquePredictModel(),
                nonconformity="pearson_weighted",
                # No tweedie_power= passed, model not TweedieRegressor
            )
            assert any("1.5" in str(warning.message) or "auto-detect" in str(warning.message).lower()
                       for warning in w)

    def test_no_warn_when_power_provided(self):
        class OpaquePredictModel:
            def predict(self, X):
                return np.ones(len(X))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp = InsuranceConformalPredictor(
                model=OpaquePredictModel(),
                nonconformity="pearson_weighted",
                tweedie_power=1.5,
            )
            power_warns = [x for x in w if "auto-detect" in str(x.message).lower()
                          or "tweedie" in str(x.message).lower()]
            assert len(power_warns) == 0

    def test_no_warn_for_raw_score(self):
        class OpaquePredictModel:
            def predict(self, X):
                return np.ones(len(X))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp = InsuranceConformalPredictor(
                model=OpaquePredictModel(),
                nonconformity="raw",
            )
            power_warns = [x for x in w if "auto-detect" in str(x.message).lower()]
            assert len(power_warns) == 0


# ---------------------------------------------------------------------------
# Predictor: quantile caching
# ---------------------------------------------------------------------------


class TestQuantileCaching:
    def test_quantile_cached_on_second_call(self, simple_data):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])

        # First call populates cache
        assert 0.10 not in cp.cal_quantiles_
        cp.predict_interval(simple_data["X_test"], alpha=0.10)
        assert 0.10 in cp.cal_quantiles_

        # Second call uses cached value
        q1 = cp.cal_quantiles_[0.10]
        cp.predict_interval(simple_data["X_test"], alpha=0.10)
        q2 = cp.cal_quantiles_[0.10]
        assert q1 == q2

    def test_different_alphas_cached_separately(self, simple_data):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])

        cp.predict_interval(simple_data["X_test"], alpha=0.10)
        cp.predict_interval(simple_data["X_test"], alpha=0.20)
        assert 0.10 in cp.cal_quantiles_
        assert 0.20 in cp.cal_quantiles_
        assert cp.cal_quantiles_[0.10] != cp.cal_quantiles_[0.20]

    def test_recalibrate_clears_cache(self, simple_data):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])
        cp.predict_interval(simple_data["X_test"], alpha=0.10)
        assert len(cp.cal_quantiles_) > 0

        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])
        assert len(cp.cal_quantiles_) == 0


# ---------------------------------------------------------------------------
# Predictor: coverage_by_decile with different n_bins
# ---------------------------------------------------------------------------


class TestCoverageByDecileVariants:
    @pytest.fixture(scope="class")
    def cp_calibrated(self, simple_data):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])
        return cp

    def test_n_bins_1(self, simple_data):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])
        result = cp.coverage_by_decile(simple_data["X_test"], simple_data["y_test"], n_bins=1)
        assert len(result) == 1
        assert result["n_obs"][0] == len(simple_data["y_test"])

    def test_n_bins_5(self, simple_data):
        cp = InsuranceConformalPredictor(model=simple_data["model"], tweedie_power=1.0)
        cp.calibrate(simple_data["X_cal"], simple_data["y_cal"])
        result = cp.coverage_by_decile(simple_data["X_test"], simple_data["y_test"], n_bins=5)
        assert len(result) <= 5
        assert result["n_obs"].sum() == len(simple_data["y_test"])


# ---------------------------------------------------------------------------
# SolvencyCapitalRange: repr and error handling
# ---------------------------------------------------------------------------


class TestSolvencyCapitalRange:
    def test_repr(self):
        result = SolvencyCapitalRange(
            scr_estimate=np.array([10.0]),
            lower_bound=np.array([0.5]),
            upper_bound=np.array([15.0]),
            interval_width=np.array([14.5]),
            coverage_level=0.995,
            n_risks=1,
            alpha=0.005,
            total_scr=10.0,
            mean_interval_width=14.5,
        )
        r = repr(result)
        assert "SolvencyCapitalRange" in r
        assert "n_risks=1" in r

    def test_repr_contains_coverage_level(self):
        result = SolvencyCapitalRange(
            scr_estimate=np.array([5.0, 3.0]),
            lower_bound=np.array([0.0, 0.0]),
            upper_bound=np.array([8.0, 6.0]),
            interval_width=np.array([8.0, 6.0]),
            coverage_level=0.99,
            n_risks=2,
            alpha=0.01,
            total_scr=8.0,
            mean_interval_width=7.0,
        )
        r = repr(result)
        assert "0.990" in r


class TestSolvencyCapitalRangeFunction:
    @pytest.fixture(scope="class")
    def cp_for_scr(self):
        rng = np.random.default_rng(55)
        n = 1000
        beta = np.array([0.3, -0.1])
        X = rng.normal(size=(n, 2))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)
        model = LinearPoissonModel(beta)
        cp = InsuranceConformalPredictor(model=model, tweedie_power=1.0)
        cp.calibrate(X[:400], y[:400])
        return cp, X[400:], y[400:]

    def test_basic_return_type(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        result = solvency_capital_range(cp, X_test, alpha=0.10)
        assert isinstance(result, SolvencyCapitalRange)

    def test_scr_nonneg(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        result = solvency_capital_range(cp, X_test, alpha=0.10)
        assert np.all(result.scr_estimate >= 0)

    def test_interval_width_matches(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        result = solvency_capital_range(cp, X_test, alpha=0.10)
        expected_width = result.upper_bound - result.lower_bound
        np.testing.assert_allclose(result.interval_width, expected_width)

    def test_coverage_level_matches_alpha(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        result = solvency_capital_range(cp, X_test, alpha=0.05)
        assert result.coverage_level == pytest.approx(0.95)
        assert result.alpha == 0.05

    def test_n_risks_matches_input_length(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        result = solvency_capital_range(cp, X_test, alpha=0.10)
        assert result.n_risks == len(X_test)

    def test_total_scr_equals_sum(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        result = solvency_capital_range(cp, X_test, alpha=0.10)
        assert result.total_scr == pytest.approx(result.scr_estimate.sum())

    def test_mean_width_equals_mean(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        result = solvency_capital_range(cp, X_test, alpha=0.10)
        assert result.mean_interval_width == pytest.approx(result.interval_width.mean())

    def test_invalid_alpha_raises(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        with pytest.raises(ValueError, match="alpha"):
            solvency_capital_range(cp, X_test, alpha=0.0)

    def test_alpha_above_1_raises(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        with pytest.raises(ValueError, match="alpha"):
            solvency_capital_range(cp, X_test, alpha=1.1)

    def test_no_predict_interval_raises_type_error(self, cp_for_scr):
        _, X_test, _ = cp_for_scr
        with pytest.raises(TypeError, match="predict_interval"):
            solvency_capital_range("not_a_predictor", X_test)

    def test_exposure_scaling(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        n = len(X_test)
        exposure = np.full(n, 0.5)

        result_no_exp = solvency_capital_range(cp, X_test, alpha=0.10)
        result_with_exp = solvency_capital_range(cp, X_test, alpha=0.10, exposure=exposure)

        # With exposure=0.5, all bounds should be half of no-exposure
        np.testing.assert_allclose(
            result_with_exp.upper_bound, result_no_exp.upper_bound * 0.5, rtol=1e-10
        )

    def test_exposure_length_mismatch_raises(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        with pytest.raises(ValueError, match="length"):
            solvency_capital_range(cp, X_test, alpha=0.10, exposure=np.ones(3))

    def test_negative_exposure_raises(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        n = len(X_test)
        bad_exposure = np.full(n, -0.5)
        with pytest.raises(ValueError, match="positive"):
            solvency_capital_range(cp, X_test, alpha=0.10, exposure=bad_exposure)

    def test_zero_exposure_raises(self, cp_for_scr):
        cp, X_test, y_test = cp_for_scr
        exposure = np.ones(len(X_test))
        exposure[5] = 0.0
        with pytest.raises(ValueError, match="positive"):
            solvency_capital_range(cp, X_test, alpha=0.10, exposure=exposure)

    def test_wider_intervals_at_lower_alpha(self, cp_for_scr):
        """Lower alpha = higher coverage = wider intervals."""
        cp, X_test, y_test = cp_for_scr
        r_tight = solvency_capital_range(cp, X_test, alpha=0.20)
        r_wide = solvency_capital_range(cp, X_test, alpha=0.05)
        assert r_wide.mean_interval_width > r_tight.mean_interval_width


# ---------------------------------------------------------------------------
# diagnostics_ext: subgroup_coverage additional tests
# ---------------------------------------------------------------------------


class TestSubgroupCoverageEdgeCases:
    @pytest.fixture(scope="class")
    def diag_setup(self):
        rng = np.random.default_rng(33)
        n = 800
        beta = np.array([0.4, -0.2, 0.1])
        X = rng.normal(size=(n, 3))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)
        model = LinearPoissonModel(beta)

        cp = InsuranceConformalPredictor(model=model, nonconformity="pearson_weighted", tweedie_power=1.0)
        cp.calibrate(X[:300], y[:300])
        return cp, X[300:600], y[300:600]

    def test_returns_polars_dataframe(self, diag_setup):
        cp, X_test, y_test = diag_setup
        groups = np.where(X_test[:, 0] > 0, "high", "low")
        result = subgroup_coverage(cp, X_test, y_test, alpha=0.10, groups=groups)
        assert isinstance(result, pl.DataFrame)

    def test_required_columns_present(self, diag_setup):
        cp, X_test, y_test = diag_setup
        groups = np.where(X_test[:, 0] > 0, "A", "B")
        result = subgroup_coverage(cp, X_test, y_test, alpha=0.10, groups=groups)
        for col in ["n_obs", "empirical_coverage", "target_coverage", "coverage_gap"]:
            assert col in result.columns

    def test_n_obs_sums_to_total(self, diag_setup):
        cp, X_test, y_test = diag_setup
        groups = (X_test[:, 0] > 0).astype(int)
        result = subgroup_coverage(cp, X_test, y_test, alpha=0.10, groups=groups)
        assert result["n_obs"].sum() == len(y_test)

    def test_coverage_in_unit_interval(self, diag_setup):
        cp, X_test, y_test = diag_setup
        groups = np.where(X_test[:, 0] > 0, "A", "B")
        result = subgroup_coverage(cp, X_test, y_test, alpha=0.10, groups=groups)
        assert (result["empirical_coverage"] >= 0.0).all()
        assert (result["empirical_coverage"] <= 1.0).all()

    def test_target_coverage_matches_alpha(self, diag_setup):
        cp, X_test, y_test = diag_setup
        groups = np.where(X_test[:, 0] > 0, "A", "B")
        result = subgroup_coverage(cp, X_test, y_test, alpha=0.15, groups=groups)
        assert np.allclose(result["target_coverage"].to_numpy(), 0.85)

    def test_single_group(self, diag_setup):
        """All observations in one group: should produce a 1-row result."""
        cp, X_test, y_test = diag_setup
        groups = np.full(len(y_test), "all")
        result = subgroup_coverage(cp, X_test, y_test, alpha=0.10, groups=groups)
        assert len(result) == 1
        assert result["n_obs"][0] == len(y_test)

    def test_many_groups(self, diag_setup):
        """One observation per group (n unique groups = n observations)."""
        cp, X_test, y_test = diag_setup
        groups = np.arange(len(y_test)).astype(str)
        result = subgroup_coverage(cp, X_test, y_test, alpha=0.10, groups=groups)
        assert len(result) == len(y_test)

    def test_group_name_parameter(self, diag_setup):
        cp, X_test, y_test = diag_setup
        groups = np.where(X_test[:, 0] > 0, "A", "B")
        result = subgroup_coverage(
            cp, X_test, y_test, alpha=0.10, groups=groups, group_name="vehicle_class"
        )
        assert "vehicle_class" in result.columns

    def test_length_mismatch_raises(self, diag_setup):
        cp, X_test, y_test = diag_setup
        groups = np.array(["A", "B"])  # wrong length
        with pytest.raises(ValueError):
            subgroup_coverage(cp, X_test, y_test, alpha=0.10, groups=groups)


# ---------------------------------------------------------------------------
# diagnostics_ext: width_efficiency_comparison additional tests
# ---------------------------------------------------------------------------


class TestWidthEfficiencyComparisonEdgeCases:
    @pytest.fixture(scope="class")
    def two_predictors(self):
        rng = np.random.default_rng(44)
        n = 800
        beta = np.array([0.4, -0.2])
        X = rng.normal(size=(n, 2))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)
        model = LinearPoissonModel(beta)

        cp1 = InsuranceConformalPredictor(model=model, nonconformity="raw", tweedie_power=1.0)
        cp1.calibrate(X[:300], y[:300])

        cp2 = InsuranceConformalPredictor(model=model, nonconformity="pearson_weighted", tweedie_power=1.0)
        cp2.calibrate(X[:300], y[:300])

        return cp1, cp2, X[300:600], y[300:600]

    def test_returns_polars_dataframe(self, two_predictors):
        cp1, cp2, X_test, y_test = two_predictors
        result = width_efficiency_comparison(
            predictors={"raw": cp1, "pearson_weighted": cp2},
            X_test=X_test,
            y_test=y_test,
            alpha=0.10,
        )
        assert isinstance(result, pl.DataFrame)

    def test_has_required_columns(self, two_predictors):
        cp1, cp2, X_test, y_test = two_predictors
        result = width_efficiency_comparison(
            predictors={"raw": cp1, "pearson_weighted": cp2},
            X_test=X_test,
            y_test=y_test,
            alpha=0.10,
        )
        assert "predictor_name" in result.columns
        assert "mean_width" in result.columns
        assert "marginal_coverage" in result.columns

    def test_n_rows_equals_n_predictors(self, two_predictors):
        cp1, cp2, X_test, y_test = two_predictors
        result = width_efficiency_comparison(
            predictors={"A": cp1, "B": cp2},
            X_test=X_test,
            y_test=y_test,
            alpha=0.10,
        )
        assert len(result) == 2

    def test_coverage_in_unit_interval(self, two_predictors):
        cp1, cp2, X_test, y_test = two_predictors
        result = width_efficiency_comparison(
            predictors={"raw": cp1, "pw": cp2},
            X_test=X_test,
            y_test=y_test,
            alpha=0.10,
        )
        assert (result["marginal_coverage"] >= 0.0).all()
        assert (result["marginal_coverage"] <= 1.0).all()

    def test_pearson_weighted_narrower_than_raw(self, two_predictors):
        """pearson_weighted should produce narrower mean intervals than raw."""
        cp1, cp2, X_test, y_test = two_predictors
        result = width_efficiency_comparison(
            predictors={"raw": cp1, "pearson_weighted": cp2},
            X_test=X_test,
            y_test=y_test,
            alpha=0.10,
        )
        raw_row = result.filter(pl.col("predictor_name") == "raw")
        pw_row = result.filter(pl.col("predictor_name") == "pearson_weighted")
        assert pw_row["mean_width"][0] < raw_row["mean_width"][0]


# ---------------------------------------------------------------------------
# ERT coverage gap additional tests
# ---------------------------------------------------------------------------


class TestErtCoverageGapEdgeCases:
    @pytest.fixture(scope="class")
    def ert_setup(self):
        rng = np.random.default_rng(22)
        n = 1200
        beta = np.array([0.3, -0.15])
        X = rng.normal(size=(n, 2))
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)
        model = LinearPoissonModel(beta)
        cp = InsuranceConformalPredictor(model=model, nonconformity="pearson_weighted", tweedie_power=1.0)
        cp.calibrate(X[:400], y[:400])
        return cp, X[400:800], y[400:800]

    def test_returns_polars_dataframe(self, ert_setup):
        cp, X_test, y_test = ert_setup
        result = ert_coverage_gap(cp, X_test, y_test, alpha=0.10)
        assert isinstance(result, pl.DataFrame)

    def test_has_required_columns(self, ert_setup):
        cp, X_test, y_test = ert_setup
        result = ert_coverage_gap(cp, X_test, y_test, alpha=0.10)
        for col in ["bin", "empirical_coverage", "target_coverage", "coverage_gap"]:
            assert col in result.columns

    def test_has_summary_row(self, ert_setup):
        """The last row should have bin=-1 (the summary sentinel)."""
        cp, X_test, y_test = ert_setup
        result = ert_coverage_gap(cp, X_test, y_test, alpha=0.10)
        assert result["bin"][-1] == -1

    def test_n_obs_positive(self, ert_setup):
        cp, X_test, y_test = ert_setup
        result = ert_coverage_gap(cp, X_test, y_test, alpha=0.10)
        assert (result["n_obs"] > 0).all()

    def test_target_coverage_matches_alpha(self, ert_setup):
        cp, X_test, y_test = ert_setup
        result = ert_coverage_gap(cp, X_test, y_test, alpha=0.20)
        # All rows should have target_coverage = 0.80
        assert np.allclose(result["target_coverage"].to_numpy(), 0.80)

    def test_n_bins_parameter(self, ert_setup):
        cp, X_test, y_test = ert_setup
        result = ert_coverage_gap(cp, X_test, y_test, alpha=0.10, n_bins=5)
        # 5 bins + 1 summary row = 6 rows max
        assert len(result) <= 6
