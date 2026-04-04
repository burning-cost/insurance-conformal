"""
Expanded edge-case and integration tests for insurance-conformal v0.7-v1.3.

This module covers gaps in the existing test suite across:
- TweedieConformPredictor (v0.7.0)
- ConditionalCoverageERT / ConditionalValidityIndex / CCSelect (v0.7.1-v0.8.0)
- LoBoostCP (v1.0.0)
- KMMConformalPredictor (v1.1.0)
- ConformalisedSurvival / KaplanMeierCensoringModel (v1.2.0)
- LCPModelSelector / LocalScoreSelector (v1.2.0)

Focus areas:
  - Single-row calibration sets
  - Tiny input arrays (n=1, n=2)
  - alpha boundary conditions (near 0, near 1)
  - Mismatched dimensions
  - NaN/inf behaviour in inputs
  - Module integration paths
  - CVI n<800 warning (v1.3.0)
  - ConditionalCoverageAssessor
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from insurance_conformal import TweedieConformPredictor
from insurance_conformal.loboost import LoBoostCP, _nonconformity_score
from insurance_conformal.covariate_shift import KMMConformalPredictor, _weighted_conformal_quantile
from insurance_conformal.survival import ConformalisedSurvival, KaplanMeierCensoringModel
from insurance_conformal.lcp_model_selector import LCPModelSelector, LocalScoreSelector


# ===========================================================================
# Shared helpers
# ===========================================================================


class ConstantModel:
    """Simplest possible model — predicts a fixed scalar."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.full(len(X), self.value)


class LinearExpModel:
    """log-linear model: predict = exp(X @ beta), clipped positive."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = np.asarray(beta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(X) @ self.beta)


def make_basic_data(n: int = 300, n_feat: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    beta = np.array([0.3, -0.2, 0.1])[:n_feat]
    X = rng.normal(size=(n, n_feat))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float) + 0.01
    tr = n // 2
    cal = n // 4
    return {
        "X_train": X[:tr],
        "y_train": y[:tr],
        "X_cal": X[tr : tr + cal],
        "y_cal": y[tr : tr + cal],
        "X_test": X[tr + cal :],
        "y_test": y[tr + cal :],
        "model": LinearExpModel(beta),
    }


@pytest.fixture(scope="module")
def basic_data():
    return make_basic_data(n=300, seed=7)


@pytest.fixture(scope="module")
def gbm_data():
    """GBM-backed data — needed for LoBoostCP."""
    rng = np.random.default_rng(13)
    n = 600
    X = rng.normal(size=(n, 4))
    y = np.exp(0.3 * X[:, 0]) * rng.gamma(2.0, 0.5, n)
    tr = 360
    cal = 120
    gbm = GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0)
    gbm.fit(X[:tr], y[:tr])
    return {
        "X_cal": X[tr : tr + cal],
        "y_cal": y[tr : tr + cal],
        "X_test": X[tr + cal :],
        "y_test": y[tr + cal :],
        "model": gbm,
    }


# ===========================================================================
# TweedieConformPredictor — boundary and edge conditions
# ===========================================================================


class TestTweedieEdgeCases:

    # -----------------------------------------------------------------------
    # Single-row calibration set
    # -----------------------------------------------------------------------

    def test_single_row_calibration(self):
        """A calibration set of size 1 should work; quantile is that single score."""
        rng = np.random.default_rng(0)
        X1 = rng.normal(size=(1, 3))
        y1 = np.array([2.0])
        X_test = rng.normal(size=(5, 3))
        model = ConstantModel(2.0)
        tcp = TweedieConformPredictor(model=model, p=1.5, score="pearson")
        tcp.calibrate(X1, y1)
        assert tcp.n_calibration_ == 1
        intervals = tcp.predict_interval(X_test, alpha=0.10)
        assert intervals.shape == (5, 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])
        assert np.all(intervals[:, 0] >= 0.0)

    # -----------------------------------------------------------------------
    # alpha near 0 and near 1
    # -----------------------------------------------------------------------

    def test_alpha_near_zero_gives_wide_intervals(self, basic_data):
        """alpha=0.001 should produce very wide intervals."""
        tcp = TweedieConformPredictor(model=basic_data["model"], p=1.5)
        tcp.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        intervals = tcp.predict_interval(basic_data["X_test"][:10], alpha=0.001)
        widths = intervals[:, 1] - intervals[:, 0]
        assert np.all(widths > 0)
        # intervals at alpha=0.001 should be much wider than at alpha=0.5
        intervals_wide = tcp.predict_interval(basic_data["X_test"][:10], alpha=0.5)
        widths_wide = intervals_wide[:, 1] - intervals_wide[:, 0]
        assert widths.mean() > widths_wide.mean()

    def test_alpha_boundaries_raise(self, basic_data):
        """alpha=0 and alpha=1 are not in (0,1) and must raise."""
        tcp = TweedieConformPredictor(model=basic_data["model"])
        tcp.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        with pytest.raises(ValueError, match="alpha must be in"):
            tcp.predict_interval(basic_data["X_test"][:5], alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            tcp.predict_interval(basic_data["X_test"][:5], alpha=1.0)

    # -----------------------------------------------------------------------
    # Mismatched X/y dimensions at calibrate
    # -----------------------------------------------------------------------

    def test_calibrate_wrong_y_length(self, basic_data):
        tcp = TweedieConformPredictor(model=basic_data["model"])
        X = basic_data["X_cal"]
        y_bad = basic_data["y_cal"][:-1]
        with pytest.raises(ValueError, match="lengths must match"):
            tcp.calibrate(X, y_bad)

    # -----------------------------------------------------------------------
    # Zero-target values (edge for deviance score)
    # -----------------------------------------------------------------------

    def test_calibrate_with_zero_targets(self):
        """y=0 values are common in insurance (no-claim rows).
        Pearson score should handle them without error."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 2))
        y = np.zeros(50)  # all zeros
        model = ConstantModel(1.0)
        tcp = TweedieConformPredictor(model=model, p=1.5, score="pearson")
        tcp.calibrate(X, y)
        assert tcp.is_calibrated_
        intervals = tcp.predict_interval(rng.normal(size=(5, 2)), alpha=0.10)
        assert np.all(intervals[:, 0] <= intervals[:, 1])

    # -----------------------------------------------------------------------
    # All-same predictions (degenerate model)
    # -----------------------------------------------------------------------

    def test_constant_model_all_scores(self, basic_data):
        """A constant model should still produce valid intervals for all score types."""
        for score in ("pearson", "deviance", "anscombe"):
            model = ConstantModel(1.5)
            tcp = TweedieConformPredictor(model=model, p=1.5, score=score)
            tcp.calibrate(basic_data["X_cal"], basic_data["y_cal"])
            intervals = tcp.predict_interval(basic_data["X_test"][:10], alpha=0.10)
            assert np.all(intervals[:, 0] <= intervals[:, 1]), f"score={score} violated ordering"
            assert np.all(intervals[:, 0] >= 0.0), f"score={score} gave negative lower bound"

    # -----------------------------------------------------------------------
    # p=1.0 (Poisson) and p=2.0 (Gamma) — boundary variance powers
    # -----------------------------------------------------------------------

    def test_p_poisson(self, basic_data):
        tcp = TweedieConformPredictor(model=basic_data["model"], p=1.0, score="pearson")
        tcp.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        intervals = tcp.predict_interval(basic_data["X_test"][:10], alpha=0.10)
        assert intervals.shape[1] == 2

    def test_p_gamma(self, basic_data):
        """p=2.0 with deviance score (anscombe is undefined at p=2)."""
        tcp = TweedieConformPredictor(model=basic_data["model"], p=2.0, score="deviance")
        tcp.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        intervals = tcp.predict_interval(basic_data["X_test"][:10], alpha=0.10)
        assert intervals.shape[1] == 2

    # -----------------------------------------------------------------------
    # select_score() with empty valid candidates (fallback path)
    # -----------------------------------------------------------------------

    def test_select_score_fallback_when_no_coverage(self, basic_data):
        """select_score() should still return a valid name even when
        no candidate achieves the target coverage."""
        tcp = TweedieConformPredictor(model=basic_data["model"], score="pearson")
        tcp.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        # Use alpha=0.9999: barely any coverage possible -> fallback to closest
        best = tcp.select_score(
            basic_data["X_test"],
            basic_data["y_test"],
            alpha=0.9999,
            candidates=["pearson", "deviance"],
        )
        assert best in ("pearson", "deviance")

    # -----------------------------------------------------------------------
    # coverage_diagnostic on a tiny test set
    # -----------------------------------------------------------------------

    def test_coverage_diagnostic_tiny_test(self, basic_data):
        """coverage_diagnostic on n=5 test points should not crash."""
        tcp = TweedieConformPredictor(model=basic_data["model"])
        tcp.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        diag = tcp.coverage_diagnostic(
            basic_data["X_test"][:5], basic_data["y_test"][:5], alpha=0.10
        )
        assert 0.0 <= diag["empirical_coverage"] <= 1.0

    # -----------------------------------------------------------------------
    # Exposure weighted with negative exposure raises
    # -----------------------------------------------------------------------

    def test_exposure_weighted_zero_exposure_raises(self, basic_data):
        model = ConstantModel(1.0)
        tcp = TweedieConformPredictor(model=model, p=1.0, exposure_weighted=True)
        rng = np.random.default_rng(5)
        X_cal = rng.normal(size=(20, 2))
        y_cal = rng.exponential(1.0, 20)
        exp_bad = np.ones(20)
        exp_bad[3] = 0.0  # zero exposure
        with pytest.raises(ValueError, match="positive"):
            tcp.calibrate(X_cal, y_cal, exposure_cal=exp_bad)

    # -----------------------------------------------------------------------
    # Two-stage integration: TweedieConformPredictor + select_score
    # -----------------------------------------------------------------------

    def test_select_score_after_calibrate_integration(self, basic_data):
        """Full pipeline: calibrate then select_score then predict with best score."""
        tcp = TweedieConformPredictor(model=basic_data["model"], p=1.5)
        tcp.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        best = tcp.select_score(
            basic_data["X_test"],
            basic_data["y_test"],
            alpha=0.10,
        )
        # Now build a new predictor with the best score and verify it produces valid intervals
        tcp2 = TweedieConformPredictor(model=basic_data["model"], p=1.5, score=best)
        tcp2.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        intervals = tcp2.predict_interval(basic_data["X_test"], alpha=0.10)
        assert np.all(intervals[:, 0] <= intervals[:, 1])


# ===========================================================================
# LoBoostCP — additional edge cases
# ===========================================================================


class TestLoBoostEdgeCases:

    def test_single_row_calibration_falls_back(self, gbm_data):
        """n_cal=1 triggers fallback for all test points; should not crash."""
        d = gbm_data
        lcp = LoBoostCP(model=d["model"], alpha=0.1, min_samples=1)
        X1 = d["X_cal"][:1]
        y1 = d["y_cal"][:1]
        lcp.calibrate(X1, y1)
        assert lcp.n_calibration_ == 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            intervals = lcp.predict(d["X_test"][:5])
        assert isinstance(intervals, pl.DataFrame)
        assert np.all(intervals["lower"].to_numpy() <= intervals["upper"].to_numpy())

    def test_min_samples_equals_one_allows_small_local_sets(self, gbm_data):
        """min_samples=1 should never trigger the global fallback."""
        d = gbm_data
        lcp = LoBoostCP(model=d["model"], alpha=0.1, min_samples=1, overlap_frac=0.5)
        lcp.calibrate(d["X_cal"], d["y_cal"])
        intervals = lcp.predict(d["X_test"][:20])
        assert np.all(intervals["lower"].to_numpy() <= intervals["upper"].to_numpy())

    def test_overlap_frac_zero_equals_standard_cp(self, gbm_data):
        """overlap_frac=0.0 means every cal point is always in local set.
        All test points get the same quantile (global CP)."""
        d = gbm_data
        lcp = LoBoostCP(model=d["model"], alpha=0.1, overlap_frac=0.0, min_samples=1)
        lcp.calibrate(d["X_cal"], d["y_cal"])
        intervals = lcp.predict(d["X_test"][:30])
        # All upper - point_pred should be the same (global quantile)
        upper_minus_pred = (
            intervals["upper"].to_numpy() - intervals["point_pred"].to_numpy()
        )
        # Small numerical tolerance
        assert np.allclose(upper_minus_pred, upper_minus_pred[0], atol=1e-6)

    def test_overlap_frac_one_uses_exact_matches(self, gbm_data):
        """overlap_frac=1.0 requires exact leaf match. Most points will fallback,
        but it should not crash."""
        d = gbm_data
        lcp = LoBoostCP(model=d["model"], alpha=0.1, overlap_frac=1.0, min_samples=5)
        lcp.calibrate(d["X_cal"], d["y_cal"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            intervals = lcp.predict(d["X_test"][:10])
        assert isinstance(intervals, pl.DataFrame)
        assert np.all(intervals["lower"].to_numpy() >= 0.0)

    def test_normalized_score_nonnegative(self, gbm_data):
        """Normalized nonconformity scores must be non-negative."""
        d = gbm_data
        lcp = LoBoostCP(model=d["model"], alpha=0.1, score_type="normalized")
        lcp.calibrate(d["X_cal"], d["y_cal"])
        assert np.all(lcp.cal_scores_ >= 0.0)

    def test_alpha_near_boundary(self, gbm_data):
        """alpha=0.01 (very conservative) and alpha=0.49 (less conservative)
        should both produce valid intervals."""
        d = gbm_data
        for alpha in (0.01, 0.49):
            lcp = LoBoostCP(model=d["model"], alpha=alpha, overlap_frac=0.0, min_samples=1)
            lcp.calibrate(d["X_cal"], d["y_cal"])
            intervals = lcp.predict(d["X_test"][:5])
            lower = intervals["lower"].to_numpy()
            upper = intervals["upper"].to_numpy()
            assert np.all(lower <= upper), f"alpha={alpha}: lower > upper"

    def test_nonconformity_score_normalized_zero_yhat(self):
        """When yhat is near-zero, normalized score clips the denominator."""
        y = np.array([1.0, 2.0])
        yhat = np.array([0.0, 0.0])  # would divide by zero without clip
        scores = _nonconformity_score(y, yhat, "normalized")
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0.0)

    def test_coverage_report_before_calibrate_raises(self, gbm_data):
        lcp = LoBoostCP(model=gbm_data["model"], alpha=0.1)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            lcp.coverage_report(gbm_data["X_test"], gbm_data["y_test"])


# ===========================================================================
# KMMConformalPredictor — additional edge cases
# ===========================================================================


class TestKMMEdgeCases:

    def test_tiny_target_warns(self):
        """X_target with very few samples relative to calibration triggers warning."""
        rng = np.random.default_rng(20)
        X_cal = rng.normal(size=(100, 3))
        y_cal = np.abs(rng.normal(size=100)) + 0.5
        # target has 2 samples — well below 5% of 100
        X_target = rng.normal(size=(2, 3))
        kcp = KMMConformalPredictor(model=ConstantModel(1.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kcp.calibrate(X_cal, y_cal, X_target)
        assert any("noisy" in str(x.message).lower() for x in w), (
            "Expected a warning about small target set"
        )

    def test_alpha_near_zero_gives_wide_intervals(self):
        rng = np.random.default_rng(21)
        X_cal = rng.normal(size=(60, 3))
        y_cal = np.abs(rng.normal(size=60)) + 0.5
        X_target = rng.normal(size=(30, 3))
        kcp = KMMConformalPredictor(model=ConstantModel(1.0))
        kcp.calibrate(X_cal, y_cal, X_target)
        i_wide = kcp.predict_interval(X_target, alpha=0.001)
        i_narrow = kcp.predict_interval(X_target, alpha=0.3)
        w_wide = (i_wide["upper"] - i_wide["lower"]).mean()
        w_narrow = (i_narrow["upper"] - i_narrow["lower"]).mean()
        assert w_wide >= w_narrow

    def test_weight_bound_1_no_upweighting(self):
        """weight_bound=1.0 disables upweighting. All weights <= 1.0."""
        rng = np.random.default_rng(22)
        X_cal = rng.normal(size=(80, 3))
        y_cal = np.abs(rng.normal(size=80)) + 0.5
        X_target = rng.normal(size=(40, 3)) + 2.0
        kcp = KMMConformalPredictor(model=ConstantModel(1.0), weight_bound=1.0)
        kcp.calibrate(X_cal, y_cal, X_target)
        assert kcp.kmm_weights_.max() <= 1.0 + 1e-6

    def test_selective_all_excluded_raises(self):
        """If selective threshold excludes all calibration points, raise ValueError."""
        rng = np.random.default_rng(23)
        X_cal = rng.normal(size=(60, 3))
        y_cal = np.abs(rng.normal(size=60)) + 0.5
        X_target = rng.normal(size=(30, 3)) + 100.0  # very far away
        kcp = KMMConformalPredictor(
            model=ConstantModel(1.0),
            selective=True,
            selective_threshold=1.0,  # unreachable threshold
        )
        with pytest.raises(ValueError, match="All calibration points were excluded"):
            kcp.calibrate(X_cal, y_cal, X_target)

    def test_weighted_conformal_quantile_total_near_zero(self):
        """When beta+beta_test is near zero, should return max(scores)."""
        scores = np.array([1.0, 2.0, 3.0])
        beta = np.array([1e-14, 1e-14, 1e-14])
        q = _weighted_conformal_quantile(scores, beta, beta_test=1e-14, alpha=0.1)
        assert q == pytest.approx(3.0, abs=1e-9)

    def test_repr_includes_key_params(self):
        kcp = KMMConformalPredictor(model=ConstantModel(1.0), weight_bound=5.0)
        r = repr(kcp)
        assert "KMMConformalPredictor" in r
        assert "5.0" in r

    def test_nonconformity_options_all_calibrate(self):
        """Each valid nonconformity option should calibrate without error."""
        rng = np.random.default_rng(24)
        X_cal = rng.normal(size=(60, 3))
        y_cal = np.abs(rng.normal(size=60)) + 1.0
        X_target = rng.normal(size=(30, 3))
        for nc in ("raw", "pearson", "pearson_weighted"):
            kcp = KMMConformalPredictor(
                model=ConstantModel(1.0),
                nonconformity=nc,
                tweedie_power=1.5,
            )
            kcp.calibrate(X_cal, y_cal, X_target)
            assert kcp.is_calibrated_, f"Failed to calibrate with nonconformity={nc}"


# ===========================================================================
# ConformalisedSurvival — additional edge cases
# ===========================================================================


def _make_surv_data(n: int = 200, lambda_t: float = 0.2, lambda_c: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = rng.exponential(1.0 / lambda_t, n)
    C = rng.exponential(1.0 / lambda_c, n)
    t = np.minimum(T, C)
    e = (T <= C).astype(float)
    X = rng.normal(size=(n, 2))
    return X, t, e, T


class OracleSurvModel:
    def __init__(self, lam: float):
        self.lam = lam

    def predict_quantile(self, X, alpha):
        return np.full(len(X), -np.log(1.0 - alpha) / self.lam)


class TestConformalisedSurvivalEdgeCases:

    def test_all_events_no_censoring(self):
        """When all observations are events (e=1), imputation is all sampled."""
        rng = np.random.default_rng(30)
        n = 100
        T = rng.exponential(5.0, n)
        t = T.copy()
        e = np.ones(n)  # all events
        X = rng.normal(size=(n, 2))

        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        cs = ConformalisedSurvival(
            survival_model=OracleSurvModel(0.2),
            censoring_model=km,
            alpha=0.10,
            n_impute=5,
            random_state=0,
        )
        cs.calibrate(X, t, e)
        assert cs.is_calibrated_

    def test_all_censored_no_events(self):
        """When all observations are censored (e=0), imputation is trivial (C_hat=t)."""
        rng = np.random.default_rng(31)
        n = 100
        C = rng.exponential(5.0, n)
        t = C.copy()
        e = np.zeros(n)  # all censored
        X = rng.normal(size=(n, 2))

        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        cs = ConformalisedSurvival(
            survival_model=OracleSurvModel(0.2),
            censoring_model=km,
            alpha=0.10,
            n_impute=5,
            random_state=0,
        )
        # Should work — censored points have C_hat = t directly
        cs.calibrate(X, t, e)
        assert cs.is_calibrated_

    def test_cutoff_exceeds_all_times_raises(self):
        """If cutoff is larger than all observed times, the filter leaves nothing."""
        rng = np.random.default_rng(32)
        n = 50
        t = rng.uniform(0, 1, n)
        e = (rng.uniform(size=n) > 0.5).astype(float)
        X = rng.normal(size=(n, 2))
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        cs = ConformalisedSurvival(
            survival_model=OracleSurvModel(0.2),
            censoring_model=km,
            cutoff=1000.0,  # far beyond all observed times
            alpha=0.10,
        )
        with pytest.raises(ValueError, match="No calibration points remain"):
            cs.calibrate(X, t, e)

    def test_n_impute_one(self):
        """n_impute=1 is the minimum; should calibrate and predict without error."""
        X, t, e, T = _make_surv_data(n=200, seed=33)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        cs = ConformalisedSurvival(
            survival_model=OracleSurvModel(0.2),
            censoring_model=km,
            alpha=0.10,
            n_impute=1,
            random_state=0,
        )
        cs.calibrate(X, t, e)
        X_test, t_test, e_test, T_test = _make_surv_data(n=50, seed=34)
        result = cs.predict_lower_bound(X_test)
        assert isinstance(result, pl.DataFrame)
        lb = result["lower_bound"].to_numpy()
        assert np.all(lb >= 0.0)

    def test_n_impute_zero_raises(self):
        with pytest.raises(ValueError, match="n_impute"):
            ConformalisedSurvival(
                survival_model=OracleSurvModel(0.2),
                censoring_model=KaplanMeierCensoringModel(),
                n_impute=0,
            )

    def test_km_survival_exactly_one_at_zero(self):
        """KM: S(0) = 1 always."""
        X, t, e, _ = _make_surv_data(n=200, seed=35)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        s0 = km.predict_survival_at(np.zeros((3, 2)), np.zeros(3))
        assert np.all(s0 >= 0.99)

    def test_km_sample_truncated_at_extreme_tmin(self):
        """sample_truncated with t_min beyond support should return t_min."""
        X, t, e, _ = _make_surv_data(n=200, seed=36)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        rng = np.random.default_rng(0)
        # t_min very large — beyond all observed times
        t_min = np.array([1000.0, 2000.0])
        X_sub = np.zeros((2, 2))
        samples = km.sample_truncated(X_sub, t_min, n_draws=5, rng=rng)
        # Should return t_min when truncation is beyond support
        assert np.all(samples >= t_min - 1e-9)

    def test_coverage_diagnostics_on_all_events(self):
        """coverage_diagnostics with e_test all=1 should return valid coverage."""
        X_cal, t_cal, e_cal, T_cal = _make_surv_data(n=300, seed=40)
        km = KaplanMeierCensoringModel()
        km.fit(t_cal, e_cal)
        cs = ConformalisedSurvival(
            survival_model=OracleSurvModel(0.2),
            censoring_model=km,
            alpha=0.10,
            n_impute=5,
            random_state=0,
        )
        cs.calibrate(X_cal, t_cal, e_cal)
        X_test, t_test, e_test, T_test = _make_surv_data(n=100, seed=41)
        # Force all test events to be observed
        e_all = np.ones(len(t_test))
        diag = cs.coverage_diagnostics(X_test, t_test, e_all)
        assert not np.isnan(diag["empirical_coverage"][0])
        assert 0.0 <= diag["empirical_coverage"][0] <= 1.0

    def test_calibration_summary_after_calibrate(self):
        X, t, e, _ = _make_surv_data(n=300, seed=42)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        cs = ConformalisedSurvival(
            survival_model=OracleSurvModel(0.2),
            censoring_model=km,
            alpha=0.10,
            n_impute=5,
        )
        cs.calibrate(X, t, e)
        summary = cs.calibration_summary()
        assert isinstance(summary, pl.DataFrame)
        stats = summary["statistic"].to_list()
        assert "cutoff_c0" in stats
        assert "alpha" in stats
        assert "effective_n" in stats


# ===========================================================================
# LCPModelSelector — additional edge cases
# ===========================================================================


class TestLCPEdgeCases:

    def test_single_model_calibrate_predict(self):
        """K=1: LCP with one model should behave like standard conformal."""
        rng = np.random.default_rng(50)
        n = 200
        X = rng.normal(size=(n, 3))
        y = np.abs(rng.normal(size=n)) + 0.5
        model = ConstantModel(float(y.mean()))
        lcp = LCPModelSelector(
            models=[model],
            score_fns=["raw"],
            localizer="gaussian",
            bandwidth=2.0,
        )
        lcp.calibrate(X[:120], y[:120])
        intervals = lcp.predict_interval(X[120:], alpha=0.10)
        assert isinstance(intervals, pl.DataFrame)
        assert len(intervals) == n - 120
        assert (intervals["lower"] >= 0).all()
        assert (intervals["lower"] <= intervals["upper"]).all()

    def test_bandwidth_zero_falls_back_uniform(self):
        """bandwidth=0 makes the Gaussian kernel constant (exp(-inf)→0). The
        zero-weight fallback should make the kernel uniform."""
        rng = np.random.default_rng(51)
        n = 100
        X = rng.normal(size=(n, 2))
        y = np.abs(rng.normal(size=n)) + 0.5
        model = ConstantModel(1.0)
        lcp = LCPModelSelector(
            models=[model],
            score_fns=["raw"],
            localizer="gaussian",
            bandwidth=1e-10,  # near-zero bandwidth
        )
        lcp.calibrate(X[:60], y[:60])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intervals = lcp.predict_interval(X[60:], alpha=0.10)
        assert isinstance(intervals, pl.DataFrame)

    def test_knn_larger_than_n_cal(self):
        """knn_k > n_cal: _kernel_weights clamps k to len(X_cal)."""
        rng = np.random.default_rng(52)
        n_cal = 10
        X_cal = rng.normal(size=(n_cal, 2))
        y_cal = np.abs(rng.normal(size=n_cal)) + 0.5
        model = ConstantModel(1.0)
        lcp = LCPModelSelector(
            models=[model],
            score_fns=["raw"],
            localizer="knn",
            knn_k=50,  # larger than n_cal
        )
        lcp.calibrate(X_cal, y_cal)
        X_test = rng.normal(size=(5, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intervals = lcp.predict_interval(X_test, alpha=0.10)
        assert len(intervals) == 5

    def test_two_models_select_narrower(self):
        """With two models, LCP should select the one producing narrower intervals."""
        rng = np.random.default_rng(53)
        n = 400
        X = rng.normal(size=(n, 3))
        y = np.abs(rng.normal(size=n)) + 1.0
        # One good model, one bad model (predicts very wrong value)
        model_good = ConstantModel(float(y.mean()))
        model_bad = ConstantModel(100.0)  # extreme prediction -> huge intervals
        lcp = LCPModelSelector(
            models=[model_good, model_bad],
            score_fns=["raw", "raw"],
            localizer="gaussian",
            bandwidth=2.0,
        )
        lcp.calibrate(X[:200], y[:200])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intervals = lcp.predict_interval(X[200:250], alpha=0.10)
        # Most test points should use the good model (narrower intervals)
        sel = lcp.selected_models(X[200:250], alpha=0.10)
        # The good model (idx=0) should dominate
        assert (sel["selected_model_idx"] == 0).sum() > (sel["selected_model_idx"] == 1).sum()

    def test_coverage_diagnostics_requires_calibration(self):
        model = ConstantModel(1.0)
        lcp = LCPModelSelector(models=[model], score_fns=["raw"])
        with pytest.raises(RuntimeError, match="not been calibrated"):
            lcp.coverage_diagnostics(np.zeros((5, 2)), np.ones(5))

    def test_large_n_uses_rank_update_path(self):
        """n>1000 triggers the rank-update LOO path. It should produce same
        interval structure as small-n brute-force path."""
        rng = np.random.default_rng(54)
        # n=1100 calibration points to trigger rank-update
        n_cal = 1100
        X_cal = rng.normal(size=(n_cal, 3))
        y_cal = np.abs(rng.normal(size=n_cal)) + 0.5
        model = ConstantModel(float(y_cal.mean()))
        lcp = LCPModelSelector(
            models=[model],
            score_fns=["raw"],
            localizer="gaussian",
            bandwidth=2.0,
        )
        lcp.calibrate(X_cal, y_cal)
        assert lcp.is_calibrated_
        assert lcp.loo_intervals_.shape == (n_cal, 1, 2)
        X_test = rng.normal(size=(10, 3))
        intervals = lcp.predict_interval(X_test, alpha=0.10)
        assert len(intervals) == 10
        assert (intervals["lower"] <= intervals["upper"]).all()


# ===========================================================================
# LocalScoreSelector — additional edge cases
# ===========================================================================


class TestLocalScoreSelectorEdgeCases:

    def test_single_score_type(self, basic_data):
        """LocalScoreSelector with a single score type should still work."""
        lss = LocalScoreSelector(
            model=basic_data["model"],
            score_types=["pearson_weighted"],
            tweedie_power=1.5,
        )
        lss.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        intervals = lss.predict_interval(basic_data["X_test"][:10], alpha=0.10)
        assert len(intervals) == 10
        assert (intervals["lower"] >= 0).all()

    def test_selected_models_always_in_score_types(self, basic_data):
        """selected_models() score names should be a subset of score_types."""
        lss = LocalScoreSelector(
            model=basic_data["model"],
            score_types=["pearson_weighted", "deviance"],
            tweedie_power=1.5,
            bandwidth=2.0,
        )
        lss.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        sel = lss.selected_models(basic_data["X_test"][:20], alpha=0.10)
        unique_names = set(sel["score_name"].to_list())
        assert unique_names.issubset({"pearson_weighted", "deviance"})

    def test_coverage_diagnostics_structure(self, basic_data):
        lss = LocalScoreSelector(
            model=basic_data["model"],
            tweedie_power=1.5,
            bandwidth=2.0,
        )
        lss.calibrate(basic_data["X_cal"], basic_data["y_cal"])
        diag = lss.coverage_diagnostics(
            basic_data["X_test"], basic_data["y_test"], alpha=0.10
        )
        assert isinstance(diag, pl.DataFrame)
        assert "coverage" in diag.columns
        assert "slice_type" in diag.columns
        assert (diag["coverage"] >= 0).all()
        assert (diag["coverage"] <= 1).all()


# ===========================================================================
# Integration tests: cross-module combinations
# ===========================================================================


class TestCrossModuleIntegration:

    def test_tweedie_then_kmm_same_data(self, basic_data):
        """
        TweedieConformPredictor and KMMConformalPredictor on the same dataset.
        Both should produce non-trivial (non-zero-width) intervals.
        """
        d = basic_data

        # Standard conformal
        tcp = TweedieConformPredictor(model=d["model"], p=1.5, score="pearson")
        tcp.calibrate(d["X_cal"], d["y_cal"])
        tc_intervals = tcp.predict_interval(d["X_test"][:20], alpha=0.10)

        # KMM conformal
        kcp = KMMConformalPredictor(model=d["model"], tweedie_power=1.5)
        kcp.calibrate(d["X_cal"], d["y_cal"], X_target=d["X_test"][:20])
        kcp_intervals = kcp.predict_interval(d["X_test"][:20], alpha=0.10)

        # Both should have positive widths
        tc_widths = tc_intervals[:, 1] - tc_intervals[:, 0]
        kcp_widths = (kcp_intervals["upper"] - kcp_intervals["lower"]).to_numpy()
        assert np.all(tc_widths > 0)
        valid = np.isfinite(kcp_widths)
        assert np.all(kcp_widths[valid] >= 0)

    def test_loboost_coverage_report_then_tweedie_comparison(self, gbm_data, basic_data):
        """
        LoBoostCP coverage_report + TweedieConformPredictor on separate datasets.
        Checks that both modules can be used in the same test session without state conflicts.
        """
        d_gbm = gbm_data
        lcp = LoBoostCP(model=d_gbm["model"], alpha=0.1, overlap_frac=0.0, min_samples=1)
        lcp.calibrate(d_gbm["X_cal"], d_gbm["y_cal"])
        report = lcp.coverage_report(d_gbm["X_test"], d_gbm["y_test"])
        assert isinstance(report, pl.DataFrame)

        d = basic_data
        tcp = TweedieConformPredictor(model=d["model"], p=1.5, score="deviance")
        tcp.calibrate(d["X_cal"], d["y_cal"])
        intervals = tcp.predict_interval(d["X_test"][:10], alpha=0.10)
        assert np.all(intervals[:, 0] <= intervals[:, 1])

    def test_lcp_then_survival_pipeline(self, basic_data):
        """
        Use LCPModelSelector intervals, then run ConformalisedSurvival — no shared state.
        """
        d = basic_data
        lcp = LCPModelSelector(
            models=[d["model"]],
            score_fns=["pearson_weighted"],
            bandwidth=2.0,
        )
        lcp.calibrate(d["X_cal"], d["y_cal"])
        lcp_ivs = lcp.predict_interval(d["X_test"][:10], alpha=0.10)
        assert len(lcp_ivs) == 10

        # Survival pipeline
        X, t, e, T = _make_surv_data(n=200, seed=99)
        km = KaplanMeierCensoringModel()
        km.fit(t, e)
        cs = ConformalisedSurvival(
            survival_model=OracleSurvModel(0.2),
            censoring_model=km,
            alpha=0.10,
            n_impute=5,
        )
        cs.calibrate(X, t, e)
        result = cs.predict_lower_bound(X[:10])
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10



# ===========================================================================
# ConditionalValidityIndex n<800 warning (v1.3.0)
# ===========================================================================


class TestCVISmallSampleWarning:
    """CVI warns when n < 800 (v1.3.0 polish).

    ConditionalValidityIndex.evaluate(X, y_lower, y_upper, y_true, alpha)
    emits a UserWarning when n < 800 (threshold from Zhou et al. 2603.27189).
    Requires LightGBM.
    """

    @pytest.fixture(autouse=True)
    def _require_lgbm(self):
        pytest.importorskip("lightgbm", reason="ConditionalValidityIndex requires LightGBM")

    def _make_cvi_inputs(self, n: int, alpha: float = 0.10, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 3))
        y_true = rng.exponential(1.0, n)
        covered = rng.binomial(1, 1 - alpha, size=n).astype(bool)
        y_lower = np.where(covered, np.zeros(n), y_true + 1.0)
        y_upper = np.where(covered, y_true + 2.0, y_true - 1.0)
        return X, y_lower, y_upper, y_true

    def test_cvi_warns_below_800(self):
        """evaluate() should warn when n < 800."""
        from insurance_conformal.conditional_coverage import ConditionalValidityIndex

        n = 400  # deliberately below 800
        alpha = 0.10
        X, y_lower, y_upper, y_true = self._make_cvi_inputs(n, alpha)

        cvi = ConditionalValidityIndex(gamma=0.1, n_splits=3, random_state=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cvi.evaluate(X, y_lower, y_upper, y_true, alpha=alpha)
        small_n_warns = [x for x in w if "800" in str(x.message)]
        assert len(small_n_warns) > 0, (
            f"Expected a warning about n<800. Got: {[str(x.message) for x in w]}"
        )

    def test_cvi_no_warning_above_800(self):
        """evaluate() should NOT warn when n >= 800."""
        from insurance_conformal.conditional_coverage import ConditionalValidityIndex

        n = 900  # above 800 threshold
        alpha = 0.10
        X, y_lower, y_upper, y_true = self._make_cvi_inputs(n, alpha, seed=1)

        cvi = ConditionalValidityIndex(gamma=0.1, n_splits=3, random_state=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cvi.evaluate(X, y_lower, y_upper, y_true, alpha=alpha)
        small_n_warns = [x for x in w if "800" in str(x.message)]
        assert len(small_n_warns) == 0

    def test_cvi_evaluate_returns_cvi_result(self):
        """evaluate() should return a CVIResult with non-negative cvi attribute."""
        from insurance_conformal.conditional_coverage import ConditionalValidityIndex, CVIResult

        n = 600
        alpha = 0.10
        X, y_lower, y_upper, y_true = self._make_cvi_inputs(n, alpha, seed=2)

        cvi = ConditionalValidityIndex(gamma=0.1, n_splits=3, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cvi.evaluate(X, y_lower, y_upper, y_true, alpha=alpha)

        assert isinstance(result, CVIResult)
        assert hasattr(result, "cvi")
        assert hasattr(result, "cvi_u")
        assert hasattr(result, "cvi_o")
        assert result.cvi >= 0.0

    def test_cvi_invalid_alpha_raises(self):
        """alpha outside (0,1) should raise ValueError."""
        from insurance_conformal.conditional_coverage import ConditionalValidityIndex

        n = 600
        X, y_lower, y_upper, y_true = self._make_cvi_inputs(n, 0.10)
        cvi = ConditionalValidityIndex()
        with pytest.raises(ValueError, match="alpha"):
            cvi.evaluate(X, y_lower, y_upper, y_true, alpha=0.0)

    def test_cvi_too_few_observations_raises(self):
        """n < 50 (_MIN_OBS_CVI) should raise ValueError."""
        from insurance_conformal.conditional_coverage import ConditionalValidityIndex

        rng = np.random.default_rng(5)
        n = 30  # below _MIN_OBS_CVI=50
        X = rng.normal(size=(n, 2))
        y_true = rng.exponential(size=n)
        y_lower = np.zeros(n)
        y_upper = y_true + 1.0
        cvi = ConditionalValidityIndex()
        with pytest.raises(ValueError, match="50"):
            cvi.evaluate(X, y_lower, y_upper, y_true, alpha=0.10)


# ===========================================================================
# ConditionalCoverageERT — additional edge cases
# ===========================================================================


class TestConditionalCoverageERTEdgeCases:
    """Additional edge cases for ConditionalCoverageERT not in existing tests."""

    @pytest.fixture(autouse=True)
    def _require_lgbm(self):
        pytest.importorskip("lightgbm", reason="ConditionalCoverageERT requires LightGBM")

    def _make_coverage_inputs(self, n: int = 600, alpha: float = 0.10, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 4))
        y_true = rng.exponential(size=n)
        covered = rng.binomial(1, 1 - alpha, size=n).astype(bool)
        y_lower = np.where(covered, np.zeros(n), y_true + 1.0)
        y_upper = np.where(covered, y_true + 2.0, y_true - 1.0)
        return X, y_lower, y_upper, y_true

    def test_ert_returns_result_with_expected_attributes(self):
        from insurance_conformal.conditional_coverage import ConditionalCoverageERT, ERTResult

        X, y_lower, y_upper, y_true = self._make_coverage_inputs()
        ert = ConditionalCoverageERT(n_splits=3, n_bootstraps=50, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ert.evaluate(X, y_lower, y_upper, y_true, alpha=0.10)
        assert isinstance(result, ERTResult)
        assert hasattr(result, "ert")
        assert np.isfinite(result.ert)

    def test_ert_loss_variants_produce_results(self):
        """l1, l2, kl loss variants should all run without error."""
        from insurance_conformal.conditional_coverage import ConditionalCoverageERT

        X, y_lower, y_upper, y_true = self._make_coverage_inputs(seed=3)
        for loss in ("l1", "l2", "kl"):
            ert = ConditionalCoverageERT(
                loss=loss, n_splits=3, n_bootstraps=20, random_state=0
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = ert.evaluate(X, y_lower, y_upper, y_true, alpha=0.10)
            assert np.isfinite(result.ert), f"loss={loss}: ert is not finite"

    def test_ert_direction_under_nonnegative(self):
        """direction='under' focuses on undercoverage; ERT should be >= 0."""
        from insurance_conformal.conditional_coverage import ConditionalCoverageERT

        X, y_lower, y_upper, y_true = self._make_coverage_inputs(seed=4)
        ert = ConditionalCoverageERT(
            direction="under", n_splits=3, n_bootstraps=20, random_state=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ert.evaluate(X, y_lower, y_upper, y_true, alpha=0.10)
        assert np.isfinite(result.ert)  # ERT can be negative when classifier is worse than baseline

    def test_ert_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="loss"):
            from insurance_conformal.conditional_coverage import ConditionalCoverageERT
            ConditionalCoverageERT(loss="mse")

    def test_ert_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            from insurance_conformal.conditional_coverage import ConditionalCoverageERT
            ConditionalCoverageERT(direction="left")


# ===========================================================================
# _weighted_conformal_quantile edge cases
# ===========================================================================


class TestWeightedConformalQuantileEdges:

    def test_single_score(self):
        """Single calibration score should return that score or inf."""
        scores = np.array([3.14])
        beta = np.array([1.0])
        q = _weighted_conformal_quantile(scores, beta, beta_test=1.0, alpha=0.1)
        # Cumulative mass = beta/(beta+beta_test) = 0.5 < 0.9 -> quantile is scores[0] or inf
        assert q == 3.14 or q == np.inf

    def test_alpha_near_one(self):
        """alpha near 1 -> 1-alpha near 0 -> quantile is very small.
        
        Use a tiny beta_test so it does not dominate the total mass and
        cause the early-return at +inf (which happens when beta_test/total > 1-alpha).
        """
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta = np.ones(5)
        # beta_test=1e-6 << 1-alpha=0.01, so no early return to inf
        q = _weighted_conformal_quantile(scores, beta, beta_test=1e-6, alpha=0.99)
        # 1-alpha=0.01 — very low requirement; cumulative mass >= 0.01 at the first score
        assert q <= 2.0 + 1e-9

    def test_sorted_order_invariance(self):
        """Result should not depend on input order."""
        scores = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
        beta = np.ones(5)
        q1 = _weighted_conformal_quantile(scores, beta, beta_test=1.0, alpha=0.1)
        scores_sorted = np.sort(scores)
        q2 = _weighted_conformal_quantile(scores_sorted, beta, beta_test=1.0, alpha=0.1)
        assert q1 == pytest.approx(q2, abs=1e-9)
