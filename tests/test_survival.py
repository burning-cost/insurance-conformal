"""
Tests for ConformalisedSurvival and supporting classes.

Uses synthetic right-censored data generated from exponential distributions:
  T ~ Exp(lambda_T)   (event time)
  C ~ Exp(lambda_C)   (censoring time)
  T_tilde = min(T, C), E = 1{T <= C}

We use a simple oracle survival model that knows the true exponential distribution
as the survival model, and a KaplanMeierCensoringModel as the censoring model.
This is the best-case scenario for coverage — both models are well-specified —
which lets us test the mechanism without needing lifelines or scikit-survival.

Coverage tests use generous tolerance (~15pp) because:
1. The guarantee is asymptotic, not finite-sample.
2. Coverage is evaluated only on uncensored observations (a subset of test data).
3. Small n makes empirical coverage noisy.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal.survival import (
    ConformalisedSurvival,
    KaplanMeierCensoringModel,
    SurvivalModelProtocol,
    CensoringModelProtocol,
)
from insurance_conformal.utils import ipcw_weighted_quantile


# ---------------------------------------------------------------------------
# Synthetic data and mock models
# ---------------------------------------------------------------------------


def make_survival_data(
    n: int = 500,
    lambda_t: float = 0.2,
    lambda_c: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic right-censored survival data.

    T ~ Exp(lambda_t), C ~ Exp(lambda_c), independent.
    T_tilde = min(T, C), E = 1{T <= C}.

    Returns X, t, e, t_true where t_true is the unobserved event time T.
    X is a (n, 2) feature matrix (not used by the oracle models but needed for
    API compatibility).
    """
    rng = np.random.default_rng(seed)
    T = rng.exponential(1.0 / lambda_t, size=n)
    C = rng.exponential(1.0 / lambda_c, size=n)
    t = np.minimum(T, C)
    e = (T <= C).astype(float)
    X = rng.standard_normal((n, 2))
    return X, t, e, T


class ExponentialSurvivalModel:
    """
    Oracle survival model for Exp(lambda_t) distribution.

    predict_quantile(X, alpha) returns the alpha-quantile of Exp(lambda_t):
        q_alpha = -log(1 - alpha) / lambda_t
    """

    def __init__(self, lambda_t: float) -> None:
        self.lambda_t = lambda_t

    def predict_quantile(self, X: np.ndarray, alpha: float) -> np.ndarray:
        q = -np.log(1.0 - alpha) / self.lambda_t
        return np.full(len(X), q)


class NoisySurvivalModel:
    """
    Slightly misspecified survival model that adds noise to the oracle quantile.
    Used to test behaviour under partial model misspecification.
    """

    def __init__(self, lambda_t: float, noise_scale: float = 0.1, seed: int = 0) -> None:
        self.lambda_t = lambda_t
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

    def predict_quantile(self, X: np.ndarray, alpha: float) -> np.ndarray:
        q = -np.log(1.0 - alpha) / self.lambda_t
        noise = self.rng.normal(0, self.noise_scale * q, size=len(X))
        return np.clip(q + noise, 1e-6, None)


# ---------------------------------------------------------------------------
# Tests: ipcw_weighted_quantile (shared utility)
# ---------------------------------------------------------------------------


class TestIpcwWeightedQuantile:
    def test_uniform_weights_equals_empirical_quantile(self):
        """With uniform weights, the weighted quantile should match the
        standard empirical quantile at level ceil((n+1)(1-alpha))/n."""
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 10, size=100)
        weights = np.ones(100)
        q = ipcw_weighted_quantile(scores, weights, alpha=0.10)
        # Should be a value in the distribution
        assert scores.min() <= q <= scores.max()

    def test_higher_weighted_points_pull_quantile(self):
        """Concentrating weight on high-score points should push the quantile up."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights_low = np.array([10.0, 10.0, 10.0, 1.0, 1.0])
        weights_high = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
        q_low = ipcw_weighted_quantile(scores, weights_low, alpha=0.10)
        q_high = ipcw_weighted_quantile(scores, weights_high, alpha=0.10)
        assert q_high >= q_low

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ipcw_weighted_quantile(np.array([1.0, 2.0]), np.ones(2), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ipcw_weighted_quantile(np.array([1.0, 2.0]), np.ones(2), alpha=1.0)

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ipcw_weighted_quantile(np.array([]), np.array([]), alpha=0.10)

    def test_degenerate_weights_returns_max(self):
        """Near-zero weights should trigger a warning and return max score."""
        scores = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])
        with pytest.warns(UserWarning, match="near-zero"):
            q = ipcw_weighted_quantile(scores, weights, alpha=0.10)
        assert q == 3.0

    def test_single_observation(self):
        q = ipcw_weighted_quantile(np.array([5.0]), np.array([1.0]), alpha=0.10)
        assert q == 5.0


# ---------------------------------------------------------------------------
# Tests: KaplanMeierCensoringModel
# ---------------------------------------------------------------------------


class TestKaplanMeierCensoringModel:
    def test_fit_returns_self(self):
        km = KaplanMeierCensoringModel()
        X, t, e, _ = make_survival_data(n=100)
        result = km.fit(t, e)
        assert result is km

    def test_survival_at_zero_is_one(self):
        km = KaplanMeierCensoringModel()
        X, t, e, _ = make_survival_data(n=200, seed=1)
        km.fit(t, e)
        surv = km.predict_survival_at(X[:5], np.zeros(5))
        assert np.all(surv >= 0.99)

    def test_survival_is_decreasing(self):
        km = KaplanMeierCensoringModel()
        X, t, e, _ = make_survival_data(n=300, seed=2)
        km.fit(t, e)
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        X_dummy = np.zeros((5, 2))
        surv = km.predict_survival_at(X_dummy, times)
        # Survival should be non-increasing
        for i in range(len(surv) - 1):
            assert surv[i] >= surv[i + 1] - 1e-6

    def test_sample_truncated_shape(self):
        km = KaplanMeierCensoringModel()
        X, t, e, _ = make_survival_data(n=300, seed=3)
        km.fit(t, e)
        rng = np.random.default_rng(0)
        t_min = np.array([0.5, 1.0, 1.5])
        X_sub = np.zeros((3, 2))
        samples = km.sample_truncated(X_sub, t_min, n_draws=20, rng=rng)
        assert samples.shape == (3,)

    def test_sample_truncated_above_tmin(self):
        km = KaplanMeierCensoringModel()
        X, t, e, _ = make_survival_data(n=500, seed=4)
        km.fit(t, e)
        rng = np.random.default_rng(0)
        t_min = np.array([0.5, 1.0, 2.0])
        X_sub = np.zeros((3, 2))
        samples = km.sample_truncated(X_sub, t_min, n_draws=50, rng=rng)
        # Sampled censoring times should be >= t_min (truncation)
        assert np.all(samples >= t_min - 1e-9)

    def test_unfitted_raises(self):
        km = KaplanMeierCensoringModel()
        with pytest.raises(AssertionError):
            km.predict_survival_at(np.zeros((2, 2)), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Tests: ConformalisedSurvival constructor validation
# ---------------------------------------------------------------------------


class TestConformalisedSurvivalConstructor:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            ConformalisedSurvival(
                survival_model=None,
                censoring_model=None,
                method="adaptive_cutoff",
            )

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ConformalisedSurvival(
                survival_model=None,
                censoring_model=None,
                alpha=0.0,
            )
        with pytest.raises(ValueError, match="alpha"):
            ConformalisedSurvival(
                survival_model=None,
                censoring_model=None,
                alpha=1.0,
            )

    def test_invalid_n_impute_raises(self):
        with pytest.raises(ValueError, match="n_impute"):
            ConformalisedSurvival(
                survival_model=None,
                censoring_model=None,
                n_impute=0,
            )

    def test_invalid_cutoff_raises(self):
        with pytest.raises(ValueError, match="cutoff"):
            ConformalisedSurvival(
                survival_model=None,
                censoring_model=None,
                cutoff=-1.0,
            )

    def test_repr_uncalibrated(self):
        cs = ConformalisedSurvival(
            survival_model=None,
            censoring_model=None,
            alpha=0.10,
        )
        r = repr(cs)
        assert "not calibrated" in r
        assert "alpha=0.1" in r


# ---------------------------------------------------------------------------
# Tests: ConformalisedSurvival calibrate
# ---------------------------------------------------------------------------


class TestConformalisedSurvivalCalibrate:
    def setup_method(self):
        self.lambda_t = 0.2
        self.lambda_c = 0.3
        X, t, e, T = make_survival_data(n=400, lambda_t=self.lambda_t,
                                         lambda_c=self.lambda_c, seed=10)
        self.X, self.t, self.e, self.T = X, t, e, T
        self.km = KaplanMeierCensoringModel()
        self.km.fit(t, e)
        self.surv_model = ExponentialSurvivalModel(lambda_t=self.lambda_t)

    def test_calibrate_sets_state(self):
        cs = ConformalisedSurvival(
            survival_model=self.surv_model,
            censoring_model=self.km,
            alpha=0.10,
            n_impute=5,
            random_state=42,
        )
        result = cs.calibrate(self.X, self.t, self.e)
        assert result is cs
        assert cs.is_calibrated_
        assert cs.cal_scores_ is not None
        assert cs.ipcw_weights_ is not None
        assert cs.calibration_quantile_ is not None
        assert cs.cutoff_ is not None
        assert cs.n_calibration_ == 400
        assert cs.n_filtered_ > 0
        assert cs.n_filtered_ <= cs.n_calibration_

    def test_calibrate_returns_self_for_chaining(self):
        cs = ConformalisedSurvival(
            survival_model=self.surv_model,
            censoring_model=self.km,
            alpha=0.10,
            n_impute=5,
        )
        result = cs.calibrate(self.X, self.t, self.e)
        assert result is cs

    def test_calibrate_mismatched_lengths_raises(self):
        cs = ConformalisedSurvival(
            survival_model=self.surv_model,
            censoring_model=self.km,
        )
        with pytest.raises(ValueError, match="same length"):
            cs.calibrate(self.X[:10], self.t[:5], self.e[:10])

    def test_calibrate_invalid_event_indicator_raises(self):
        cs = ConformalisedSurvival(
            survival_model=self.surv_model,
            censoring_model=self.km,
        )
        bad_e = self.e.copy()
        bad_e[0] = 2.0  # invalid: only 0 or 1 allowed
        with pytest.raises(ValueError, match="0.*1"):
            cs.calibrate(self.X, self.t, bad_e)

    def test_calibrate_explicit_cutoff(self):
        cs = ConformalisedSurvival(
            survival_model=self.surv_model,
            censoring_model=self.km,
            cutoff=2.0,
            n_impute=5,
        )
        cs.calibrate(self.X, self.t, self.e)
        assert cs.cutoff_ == 2.0

    def test_calibrate_cutoff_defaults_to_median(self):
        cs = ConformalisedSurvival(
            survival_model=self.surv_model,
            censoring_model=self.km,
            cutoff=None,
            n_impute=5,
        )
        cs.calibrate(self.X, self.t, self.e)
        expected_cutoff = float(np.median(self.t))
        assert abs(cs.cutoff_ - expected_cutoff) < 1e-9

    def test_calibration_summary_returns_dataframe(self):
        cs = ConformalisedSurvival(
            survival_model=self.surv_model,
            censoring_model=self.km,
            n_impute=5,
        )
        cs.calibrate(self.X, self.t, self.e)
        summary = cs.calibration_summary()
        assert isinstance(summary, pl.DataFrame)
        assert "statistic" in summary.columns
        assert "value" in summary.columns
        stats = summary["statistic"].to_list()
        assert "n_calibration" in stats
        assert "n_filtered" in stats
        assert "calibration_quantile" in stats


# ---------------------------------------------------------------------------
# Tests: predict_lower_bound
# ---------------------------------------------------------------------------


class TestPredictLowerBound:
    def setup_method(self):
        self.lambda_t = 0.2
        self.lambda_c = 0.3
        X, t, e, T = make_survival_data(n=400, lambda_t=self.lambda_t,
                                         lambda_c=self.lambda_c, seed=20)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        surv_model = ExponentialSurvivalModel(lambda_t=self.lambda_t)
        self.cs = ConformalisedSurvival(
            survival_model=surv_model,
            censoring_model=km,
            alpha=0.10,
            n_impute=5,
            random_state=0,
        )
        self.cs.calibrate(X, t, e)
        self.X_test, self.t_test, self.e_test, self.T_test = make_survival_data(
            n=300, lambda_t=self.lambda_t, lambda_c=self.lambda_c, seed=99
        )

    def test_raises_if_not_calibrated(self):
        cs = ConformalisedSurvival(
            survival_model=ExponentialSurvivalModel(0.2),
            censoring_model=KaplanMeierCensoringModel(),
        )
        with pytest.raises(RuntimeError, match="calibrated"):
            cs.predict_lower_bound(self.X_test)

    def test_output_is_polars_dataframe(self):
        result = self.cs.predict_lower_bound(self.X_test)
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self):
        result = self.cs.predict_lower_bound(self.X_test)
        assert "lower_bound" in result.columns
        assert "q_alpha_model" in result.columns

    def test_output_shape(self):
        result = self.cs.predict_lower_bound(self.X_test)
        assert len(result) == len(self.X_test)

    def test_lower_bound_nonnegative(self):
        result = self.cs.predict_lower_bound(self.X_test)
        lb = result["lower_bound"].to_numpy()
        assert np.all(lb >= 0.0)

    def test_lower_bound_le_model_quantile(self):
        """
        After the conformal correction, the lower bound should be at most the
        raw model quantile (assuming the calibration quantile is non-negative).
        If the correction is positive, L_hat = q_model - Q_hat < q_model.
        """
        result = self.cs.predict_lower_bound(self.X_test)
        lb = result["lower_bound"].to_numpy()
        q_model = result["q_alpha_model"].to_numpy()
        # Lower bound is clipped at 0, so if q_model > 0, lb <= q_model
        # (calibration quantile can be negative if model over-predicts,
        # in which case lb > q_model is possible after clipping — but
        # this should not happen with the oracle model)
        assert np.all(lb <= q_model + 1e-9)

    def test_coverage_on_uncensored_gte_1_minus_alpha(self):
        """
        Empirical coverage on uncensored test observations should be >= 1-alpha.

        The coverage is asymptotic, so we allow 15pp slack for finite samples.
        The oracle model should comfortably exceed 75% coverage for alpha=0.10.
        """
        result = self.cs.predict_lower_bound(self.X_test)
        lb = result["lower_bound"].to_numpy()

        uncensored = self.e_test == 1
        covered = lb[uncensored] <= self.T_test[uncensored]
        empirical_coverage = covered.mean()

        target = 0.90
        tolerance = 0.15  # generous for finite-sample asymptotic method
        assert empirical_coverage >= target - tolerance, (
            f"Coverage {empirical_coverage:.3f} is more than {tolerance:.2f} "
            f"below target {target:.2f}"
        )

    def test_alpha_warning_on_mismatch(self):
        """predict_lower_bound should warn if alpha differs from calibration alpha."""
        with pytest.warns(UserWarning, match="alpha"):
            self.cs.predict_lower_bound(self.X_test, alpha=0.05)


# ---------------------------------------------------------------------------
# Tests: coverage_diagnostics
# ---------------------------------------------------------------------------


class TestCoverageDiagnostics:
    def setup_method(self):
        lambda_t = 0.2
        lambda_c = 0.3
        X, t, e, T = make_survival_data(n=400, lambda_t=lambda_t,
                                         lambda_c=lambda_c, seed=30)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        surv_model = ExponentialSurvivalModel(lambda_t=lambda_t)
        self.cs = ConformalisedSurvival(
            survival_model=surv_model,
            censoring_model=km,
            alpha=0.10,
            n_impute=5,
            random_state=0,
        )
        self.cs.calibrate(X, t, e)
        self.X_test, self.t_test, self.e_test, self.T_test = make_survival_data(
            n=300, lambda_t=lambda_t, lambda_c=lambda_c, seed=88
        )

    def test_raises_if_not_calibrated(self):
        cs = ConformalisedSurvival(
            survival_model=ExponentialSurvivalModel(0.2),
            censoring_model=KaplanMeierCensoringModel(),
        )
        with pytest.raises(RuntimeError, match="calibrated"):
            cs.coverage_diagnostics(self.X_test, self.t_test, self.e_test)

    def test_output_is_polars_dataframe(self):
        result = self.cs.coverage_diagnostics(self.X_test, self.t_test, self.e_test)
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self):
        result = self.cs.coverage_diagnostics(self.X_test, self.t_test, self.e_test)
        expected_cols = {
            "n_total",
            "n_uncensored",
            "empirical_coverage",
            "target_coverage",
            "coverage_gap",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_n_total_correct(self):
        result = self.cs.coverage_diagnostics(self.X_test, self.t_test, self.e_test)
        assert result["n_total"][0] == len(self.t_test)

    def test_n_uncensored_correct(self):
        result = self.cs.coverage_diagnostics(self.X_test, self.t_test, self.e_test)
        expected = int((self.e_test == 1).sum())
        assert result["n_uncensored"][0] == expected

    def test_target_coverage_matches_alpha(self):
        result = self.cs.coverage_diagnostics(self.X_test, self.t_test, self.e_test)
        assert abs(result["target_coverage"][0] - 0.90) < 1e-9

    def test_coverage_gap_consistent(self):
        result = self.cs.coverage_diagnostics(self.X_test, self.t_test, self.e_test)
        target = result["target_coverage"][0]
        empirical = result["empirical_coverage"][0]
        gap = result["coverage_gap"][0]
        assert abs(gap - (target - empirical)) < 1e-9

    def test_all_censored_warns(self):
        """All-censored test set should warn and return NaN coverage."""
        all_censored_e = np.zeros(len(self.t_test))
        with pytest.warns(UserWarning, match="No uncensored"):
            result = self.cs.coverage_diagnostics(
                self.X_test, self.t_test, all_censored_e
            )
        assert np.isnan(result["empirical_coverage"][0])


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_calibrated(self):
        lambda_t = 0.2
        X, t, e, _ = make_survival_data(n=200, lambda_t=lambda_t, seed=50)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        cs = ConformalisedSurvival(
            survival_model=ExponentialSurvivalModel(lambda_t),
            censoring_model=km,
            alpha=0.10,
            n_impute=5,
        )
        cs.calibrate(X, t, e)
        r = repr(cs)
        assert "calibrated" in r
        assert "cutoff=" in r


# ---------------------------------------------------------------------------
# Tests: fixed_cutoff vs auto-cutoff comparison
# ---------------------------------------------------------------------------


class TestCutoffBehaviour:
    def test_smaller_cutoff_retains_more_points(self):
        """
        A smaller cutoff should retain more calibration points (less filtering).
        """
        X, t, e, _ = make_survival_data(n=500, seed=60)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        surv_model = ExponentialSurvivalModel(0.2)

        cs_small = ConformalisedSurvival(
            survival_model=surv_model,
            censoring_model=km,
            cutoff=0.5,  # small cutoff: most points have C_hat >= 0.5
            n_impute=5,
        )
        cs_large = ConformalisedSurvival(
            survival_model=surv_model,
            censoring_model=km,
            cutoff=5.0,  # large cutoff: few points have C_hat >= 5
            n_impute=5,
        )

        cs_small.calibrate(X, t, e)
        cs_large.calibrate(X, t, e)

        assert cs_small.n_filtered_ >= cs_large.n_filtered_

    def test_noisy_model_still_achieves_coverage(self):
        """
        With a slightly misspecified survival model, coverage should still be
        reasonable when the censoring model is correct (double robustness).
        """
        X_cal, t_cal, e_cal, _ = make_survival_data(n=600, lambda_t=0.2, seed=70)
        X_test, t_test, e_test, T_test = make_survival_data(
            n=400, lambda_t=0.2, seed=71
        )
        km = KaplanMeierCensoringModel()
        km.fit(t_cal, e_cal)

        # Misspecified survival model (adds noise to quantile predictions)
        noisy = NoisySurvivalModel(lambda_t=0.2, noise_scale=0.2, seed=0)
        cs = ConformalisedSurvival(
            survival_model=noisy,
            censoring_model=km,
            alpha=0.10,
            n_impute=10,
            random_state=0,
        )
        cs.calibrate(X_cal, t_cal, e_cal)
        result = cs.predict_lower_bound(X_test)
        lb = result["lower_bound"].to_numpy()

        uncensored = e_test == 1
        covered = lb[uncensored] <= T_test[uncensored]
        empirical_coverage = covered.mean()

        # More generous tolerance for misspecified model
        assert empirical_coverage >= 0.70, (
            f"Coverage {empirical_coverage:.3f} is too low even with 20pp tolerance"
        )
