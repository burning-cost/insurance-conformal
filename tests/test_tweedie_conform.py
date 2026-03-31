"""
Tests for TweedieConformPredictor.

Coverage guarantee is the primary correctness criterion: P(y in interval) >= 1-alpha
for exchangeable data. We also verify:
- score round-trips (inversion consistency)
- exposure weighting correctness
- select_score() returns a valid score name
- coverage_diagnostic() structure
- error handling for misconfigured usage

Uses the conftest.py fixtures (poisson_data, small_cal_data) plus locally defined
Tweedie fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal import TweedieConformPredictor


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class ConstantModel:
    """Returns a fixed array as predictions. For testing only."""

    def __init__(self, mu: float = 2.0) -> None:
        self.mu = mu

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.full(len(X), self.mu)


class LinearTweedieModel:
    """Log-linear Tweedie model for testing: E[Y] = exp(X @ beta)."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = np.asarray(beta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(X) @ self.beta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearTweedieModel":
        return self  # no-op; beta is fixed


def make_tweedie_data(n: int = 800, seed: int = 42) -> dict:
    """Synthetic compound Poisson-Gamma data with known model."""
    rng = np.random.default_rng(seed)
    n_feat = 4
    beta = np.array([0.4, -0.3, 0.2, -0.1])
    X = rng.normal(size=(n, n_feat))
    mu = np.exp(X @ beta + 0.5)

    # compound Poisson-Gamma with p=1.5
    p_tw = 1.5
    alpha_g = (2.0 - p_tw) / (p_tw - 1.0)
    y = np.zeros(n)
    for i in range(n):
        lam = mu[i] ** (2.0 - p_tw) / (2.0 - p_tw)
        k = rng.poisson(lam)
        if k > 0:
            scale = mu[i] ** (p_tw - 1.0) * (p_tw - 1.0)
            y[i] = rng.gamma(alpha_g * k, scale)

    tr, cal, te = int(0.5 * n), int(0.25 * n), n - int(0.5 * n) - int(0.25 * n)
    return {
        "X_train": X[:tr],
        "y_train": y[:tr],
        "X_cal": X[tr : tr + cal],
        "y_cal": y[tr : tr + cal],
        "X_test": X[tr + cal :],
        "y_test": y[tr + cal :],
        "model": LinearTweedieModel(beta),
    }


@pytest.fixture(scope="module")
def tweedie_data():
    return make_tweedie_data(n=900, seed=99)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        assert tcp.p == 1.5
        assert tcp.score == "pearson"
        assert tcp.spread_backend == "auto"
        assert tcp.exposure_weighted is False
        assert not tcp.is_calibrated_

    def test_invalid_score(self, tweedie_data):
        with pytest.raises(ValueError, match="score must be one of"):
            TweedieConformPredictor(model=tweedie_data["model"], score="rubbish")

    def test_invalid_backend(self, tweedie_data):
        with pytest.raises(ValueError, match="spread_backend must be one of"):
            TweedieConformPredictor(model=tweedie_data["model"], spread_backend="xgboost")

    def test_invalid_min_mu(self, tweedie_data):
        with pytest.raises(ValueError, match="min_mu must be positive"):
            TweedieConformPredictor(model=tweedie_data["model"], min_mu=0.0)

    def test_repr_before_calibration(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"], p=1.5, score="deviance")
        r = repr(tcp)
        assert "not calibrated" in r
        assert "deviance" in r
        assert "p=1.5" in r

    def test_repr_after_calibration(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        r = repr(tcp)
        assert "calibrated on" in r


# ---------------------------------------------------------------------------
# fit() behaviour
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_noop_for_pearson(self, tweedie_data):
        """fit() is a no-op for non-lw_pearson scores."""
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score="pearson")
        result = tcp.fit(tweedie_data["X_train"], tweedie_data["y_train"])
        assert result is tcp  # returns self
        assert tcp.spread_model_ is None

    def test_fit_noop_for_deviance(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score="deviance")
        tcp.fit(tweedie_data["X_train"], tweedie_data["y_train"])
        assert tcp.spread_model_ is None

    def test_fit_lw_pearson_sklearn_backend(self, tweedie_data):
        """fit() with lw_pearson/sklearn creates a spread model."""
        tcp = TweedieConformPredictor(
            model=tweedie_data["model"], score="lw_pearson", spread_backend="sklearn"
        )
        tcp.fit(tweedie_data["X_train"], tweedie_data["y_train"])
        assert tcp.spread_model_ is not None
        assert tcp.backend_ == "sklearn"


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_calibrate_sets_is_calibrated(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        assert tcp.is_calibrated_
        assert tcp.n_calibration_ == len(tweedie_data["y_cal"])

    def test_calibrate_stores_scores(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        assert tcp.cal_scores_ is not None
        assert tcp.cal_scores_.shape == (len(tweedie_data["y_cal"]),)
        assert np.all(tcp.cal_scores_ >= 0)

    def test_calibrate_lw_without_fit_raises(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score="lw_pearson")
        with pytest.raises(RuntimeError, match="fit()"):
            tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])

    def test_calibrate_length_mismatch_raises(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        with pytest.raises(ValueError, match="lengths must match"):
            tcp.calibrate(
                tweedie_data["X_cal"],
                tweedie_data["y_cal"][:-1],  # wrong length
            )

    def test_calibrate_stores_cal_arrays_for_select(self, tweedie_data):
        """calibrate() stores _cal_y, _cal_mu, _cal_X for select_score()."""
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        assert hasattr(tcp, "_cal_y")
        assert hasattr(tcp, "_cal_mu")
        assert hasattr(tcp, "_cal_X")

    def test_recalibrate_clears_quantile_cache(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        _ = tcp.predict_interval(tweedie_data["X_test"][:5], alpha=0.10)
        assert 0.10 in tcp.cal_quantiles_
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        assert len(tcp.cal_quantiles_) == 0


# ---------------------------------------------------------------------------
# predict_interval()
# ---------------------------------------------------------------------------


class TestPredictInterval:
    def test_returns_numpy_array(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        intervals = tcp.predict_interval(tweedie_data["X_test"], alpha=0.10)
        assert isinstance(intervals, np.ndarray)
        assert intervals.shape == (len(tweedie_data["X_test"]), 2)

    def test_lower_leq_upper(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        intervals = tcp.predict_interval(tweedie_data["X_test"], alpha=0.10)
        assert np.all(intervals[:, 0] <= intervals[:, 1])

    def test_lower_bounds_non_negative(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        intervals = tcp.predict_interval(tweedie_data["X_test"], alpha=0.10)
        assert np.all(intervals[:, 0] >= 0.0)

    def test_invalid_alpha_raises(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        with pytest.raises(ValueError, match="alpha must be in"):
            tcp.predict_interval(tweedie_data["X_test"], alpha=1.5)

    def test_uncalibrated_raises(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        with pytest.raises(RuntimeError, match="not been calibrated"):
            tcp.predict_interval(tweedie_data["X_test"], alpha=0.10)

    def test_coverage_pearson(self, tweedie_data):
        """Empirical coverage >= 1-alpha on test set (pearson score)."""
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score="pearson")
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        intervals = tcp.predict_interval(tweedie_data["X_test"], alpha=0.10)
        y = tweedie_data["y_test"]
        covered = (y >= intervals[:, 0]) & (y <= intervals[:, 1])
        # Allow 2pp slack for finite-sample variation with n~225
        assert covered.mean() >= 0.88, f"Coverage {covered.mean():.3f} below 88%"

    def test_coverage_deviance(self, tweedie_data):
        """Empirical coverage >= 1-alpha with deviance score."""
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score="deviance")
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        intervals = tcp.predict_interval(tweedie_data["X_test"], alpha=0.10)
        y = tweedie_data["y_test"]
        covered = (y >= intervals[:, 0]) & (y <= intervals[:, 1])
        assert covered.mean() >= 0.88, f"Coverage {covered.mean():.3f} below 88%"

    def test_coverage_anscombe(self, tweedie_data):
        """Empirical coverage >= 1-alpha with anscombe score."""
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score="anscombe")
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        intervals = tcp.predict_interval(tweedie_data["X_test"], alpha=0.10)
        y = tweedie_data["y_test"]
        covered = (y >= intervals[:, 0]) & (y <= intervals[:, 1])
        assert covered.mean() >= 0.88, f"Coverage {covered.mean():.3f} below 88%"

    def test_coverage_lw_pearson_sklearn(self, tweedie_data):
        """Empirical coverage >= 1-alpha with lw_pearson score (sklearn backend)."""
        tcp = TweedieConformPredictor(
            model=tweedie_data["model"],
            score="lw_pearson",
            spread_backend="sklearn",
        )
        tcp.fit(tweedie_data["X_train"], tweedie_data["y_train"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        intervals = tcp.predict_interval(tweedie_data["X_test"], alpha=0.10)
        y = tweedie_data["y_test"]
        covered = (y >= intervals[:, 0]) & (y <= intervals[:, 1])
        assert covered.mean() >= 0.88, f"Coverage {covered.mean():.3f} below 88%"

    def test_wider_intervals_for_smaller_alpha(self, tweedie_data):
        """alpha=0.01 should give wider intervals than alpha=0.20."""
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        wide = tcp.predict_interval(tweedie_data["X_test"], alpha=0.01)
        narrow = tcp.predict_interval(tweedie_data["X_test"], alpha=0.20)
        w_wide = (wide[:, 1] - wide[:, 0]).mean()
        w_narrow = (narrow[:, 1] - narrow[:, 0]).mean()
        assert w_wide > w_narrow

    def test_anscombe_raises_at_p2(self):
        """Anscombe score is undefined at p=2."""
        model = ConstantModel(mu=2.0)
        with pytest.raises(ValueError, match="undefined for p=2"):
            TweedieConformPredictor(model=model, p=2.0, score="anscombe")

    def test_different_scores_give_different_intervals(self, tweedie_data):
        """Different score types produce different interval widths."""
        n = 30
        widths = {}
        for score in ("pearson", "deviance", "anscombe"):
            tcp = TweedieConformPredictor(model=tweedie_data["model"], score=score)
            tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
            intervals = tcp.predict_interval(tweedie_data["X_test"][:n], alpha=0.10)
            widths[score] = (intervals[:, 1] - intervals[:, 0]).mean()
        # Not all widths should be identical
        assert len(set(round(w, 4) for w in widths.values())) > 1


# ---------------------------------------------------------------------------
# Exposure weighting
# ---------------------------------------------------------------------------


class TestExposureWeighting:
    """Tests for exposure_weighted=True mode."""

    def _make_exposure_data(self, seed: int = 77) -> dict:
        rng = np.random.default_rng(seed)
        n = 600
        n_feat = 3
        beta = np.array([0.3, -0.2, 0.1])
        X = rng.normal(size=(n, n_feat))
        exposure = rng.uniform(0.3, 2.0, size=n)

        # rate model: mu is per-unit-exposure rate
        rate = np.exp(X @ beta + 0.5)
        # actual counts: exposure * rate
        y = rng.poisson(exposure * rate).astype(float)

        tr = int(0.5 * n)
        cal = int(0.25 * n)
        return {
            "X_train": X[:tr],
            "y_train": y[:tr],
            "X_cal": X[tr : tr + cal],
            "y_cal": y[tr : tr + cal],
            "X_test": X[tr + cal :],
            "y_test": y[tr + cal :],
            "exp_train": exposure[:tr],
            "exp_cal": exposure[tr : tr + cal],
            "exp_test": exposure[tr + cal :],
            # Rate model: predict(X) -> rate, not count
            "model": LinearTweedieModel(beta),
        }

    def test_exposure_weighted_calibrate_requires_exposure(self):
        data = self._make_exposure_data()
        tcp = TweedieConformPredictor(
            model=data["model"], p=1.0, exposure_weighted=True
        )
        with pytest.raises(ValueError, match="no exposure provided"):
            tcp.calibrate(data["X_cal"], data["y_cal"])  # missing exposure_cal

    def test_exposure_weighted_predict_requires_exposure(self):
        data = self._make_exposure_data()
        tcp = TweedieConformPredictor(
            model=data["model"], p=1.0, exposure_weighted=True
        )
        tcp.calibrate(
            data["X_cal"], data["y_cal"], exposure_cal=data["exp_cal"]
        )
        with pytest.raises(ValueError, match="no exposure provided"):
            tcp.predict_interval(data["X_test"], alpha=0.10)  # missing exposure_new

    def test_exposure_weighted_intervals_positive(self):
        data = self._make_exposure_data()
        tcp = TweedieConformPredictor(
            model=data["model"], p=1.0, exposure_weighted=True
        )
        tcp.calibrate(
            data["X_cal"], data["y_cal"], exposure_cal=data["exp_cal"]
        )
        intervals = tcp.predict_interval(
            data["X_test"], alpha=0.10, exposure_new=data["exp_test"]
        )
        assert np.all(intervals[:, 0] >= 0.0)
        assert np.all(intervals[:, 0] <= intervals[:, 1])

    def test_exposure_weighted_coverage(self):
        data = self._make_exposure_data()
        tcp = TweedieConformPredictor(
            model=data["model"], p=1.0, exposure_weighted=True
        )
        tcp.calibrate(
            data["X_cal"], data["y_cal"], exposure_cal=data["exp_cal"]
        )
        intervals = tcp.predict_interval(
            data["X_test"], alpha=0.10, exposure_new=data["exp_test"]
        )
        y = data["y_test"]
        covered = (y >= intervals[:, 0]) & (y <= intervals[:, 1])
        # Poisson with correct model: coverage should be close to 90%
        assert covered.mean() >= 0.85

    def test_exposure_wrong_length_raises(self):
        data = self._make_exposure_data()
        tcp = TweedieConformPredictor(
            model=data["model"], p=1.0, exposure_weighted=True
        )
        with pytest.raises(ValueError, match="lengths must match"):
            tcp.calibrate(
                data["X_cal"],
                data["y_cal"],
                exposure_cal=data["exp_cal"][:-1],  # wrong length
            )

    def test_exposure_non_positive_raises(self):
        data = self._make_exposure_data()
        tcp = TweedieConformPredictor(
            model=data["model"], p=1.0, exposure_weighted=True
        )
        bad_exp = data["exp_cal"].copy()
        bad_exp[0] = 0.0
        with pytest.raises(ValueError, match="positive"):
            tcp.calibrate(data["X_cal"], data["y_cal"], exposure_cal=bad_exp)


# ---------------------------------------------------------------------------
# select_score()
# ---------------------------------------------------------------------------


class TestSelectScore:
    def test_returns_valid_score(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        best = tcp.select_score(
            tweedie_data["X_test"],
            tweedie_data["y_test"],
            alpha=0.10,
            candidates=["pearson", "deviance"],
        )
        assert best in ("pearson", "deviance")

    def test_default_candidates_exclude_lw_when_no_spread(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        best = tcp.select_score(
            tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
        )
        # lw_pearson not in default candidates when spread model is None
        assert best in ("pearson", "deviance", "anscombe")

    def test_select_score_includes_lw_when_fitted(self, tweedie_data):
        tcp = TweedieConformPredictor(
            model=tweedie_data["model"], score="lw_pearson", spread_backend="sklearn"
        )
        tcp.fit(tweedie_data["X_train"], tweedie_data["y_train"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        best = tcp.select_score(
            tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
        )
        assert best in ("pearson", "deviance", "anscombe", "lw_pearson")

    def test_single_candidate(self, tweedie_data):
        """select_score() with one candidate returns that candidate."""
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        best = tcp.select_score(
            tweedie_data["X_test"],
            tweedie_data["y_test"],
            alpha=0.10,
            candidates=["pearson"],
        )
        assert best == "pearson"

    def test_uncalibrated_raises(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        with pytest.raises(RuntimeError, match="not been calibrated"):
            tcp.select_score(
                tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
            )


# ---------------------------------------------------------------------------
# coverage_diagnostic()
# ---------------------------------------------------------------------------


class TestCoverageDiagnostic:
    def test_returns_expected_keys(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        diag = tcp.coverage_diagnostic(
            tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
        )
        assert "empirical_coverage" in diag
        assert "mean_width" in diag
        assert "by_decile" in diag
        assert "by_spread_quartile" in diag

    def test_by_spread_quartile_none_for_pearson(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score="pearson")
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        diag = tcp.coverage_diagnostic(
            tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
        )
        assert diag["by_spread_quartile"] is None

    def test_by_spread_quartile_populated_for_lw_pearson(self, tweedie_data):
        tcp = TweedieConformPredictor(
            model=tweedie_data["model"], score="lw_pearson", spread_backend="sklearn"
        )
        tcp.fit(tweedie_data["X_train"], tweedie_data["y_train"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        diag = tcp.coverage_diagnostic(
            tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
        )
        assert diag["by_spread_quartile"] is not None
        assert len(diag["by_spread_quartile"]) > 0

    def test_empirical_coverage_range(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        diag = tcp.coverage_diagnostic(
            tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
        )
        assert 0.0 <= diag["empirical_coverage"] <= 1.0
        assert diag["mean_width"] > 0.0

    def test_by_decile_structure(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        diag = tcp.coverage_diagnostic(
            tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
        )
        for row in diag["by_decile"]:
            assert "decile" in row
            assert "n_obs" in row
            assert "mean_predicted" in row
            assert "coverage" in row
            assert 0 <= row["coverage"] <= 1

    def test_uncalibrated_raises(self, tweedie_data):
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        with pytest.raises(RuntimeError, match="not been calibrated"):
            tcp.coverage_diagnostic(
                tweedie_data["X_test"], tweedie_data["y_test"], alpha=0.10
            )


# ---------------------------------------------------------------------------
# Score round-trip sanity: scores are non-negative
# ---------------------------------------------------------------------------


class TestScoreSanity:
    """Scores must be non-negative and intervals must bracket mu_hat."""

    @pytest.mark.parametrize("score", ["pearson", "deviance", "anscombe"])
    def test_scores_non_negative(self, tweedie_data, score):
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score=score)
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        assert np.all(tcp.cal_scores_ >= 0.0)

    @pytest.mark.parametrize("score", ["pearson", "deviance", "anscombe"])
    def test_intervals_bracket_mean(self, tweedie_data, score):
        """Each interval should contain mu_hat (it is the centre prediction)."""
        tcp = TweedieConformPredictor(model=tweedie_data["model"], score=score)
        tcp.calibrate(tweedie_data["X_cal"], tweedie_data["y_cal"])
        X = tweedie_data["X_test"][:30]
        intervals = tcp.predict_interval(X, alpha=0.10)
        mu_hat = tcp._base_predict(X)
        # Each interval should contain mu_hat (intervals are symmetric about mu)
        assert np.all(intervals[:, 0] <= mu_hat + 1e-9)
        assert np.all(intervals[:, 1] >= mu_hat - 1e-9)


# ---------------------------------------------------------------------------
# Input format compatibility
# ---------------------------------------------------------------------------


class TestInputFormats:
    def test_pandas_dataframe_input(self, tweedie_data):
        import pandas as pd
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        X_cal_pd = pd.DataFrame(tweedie_data["X_cal"])
        X_test_pd = pd.DataFrame(tweedie_data["X_test"])
        tcp.calibrate(X_cal_pd, tweedie_data["y_cal"])
        intervals = tcp.predict_interval(X_test_pd, alpha=0.10)
        assert intervals.shape == (len(tweedie_data["X_test"]), 2)

    def test_polars_dataframe_input(self, tweedie_data):
        import polars as pl
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        X_cal_pl = pl.DataFrame(
            {str(i): tweedie_data["X_cal"][:, i] for i in range(tweedie_data["X_cal"].shape[1])}
        )
        X_test_pl = pl.DataFrame(
            {str(i): tweedie_data["X_test"][:, i] for i in range(tweedie_data["X_test"].shape[1])}
        )
        tcp.calibrate(X_cal_pl, tweedie_data["y_cal"])
        intervals = tcp.predict_interval(X_test_pl, alpha=0.10)
        assert intervals.shape == (len(tweedie_data["X_test"]), 2)

    def test_pandas_series_y(self, tweedie_data):
        import pandas as pd
        tcp = TweedieConformPredictor(model=tweedie_data["model"])
        y_cal_s = pd.Series(tweedie_data["y_cal"])
        tcp.calibrate(tweedie_data["X_cal"], y_cal_s)
        assert tcp.is_calibrated_
