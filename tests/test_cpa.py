"""
Tests for insurance_conformal.cpa — ReliabilityEstimator, CPASelector,
CPAResult, CPASelectResult.

Design
------
All tests use synthetic prediction sets. No conformal predictor is run.
Coverage patterns are constructed explicitly so we know the ground truth.

Two canonical data scenarios:
- uniform_coverage: coverage indicators are i.i.d. Bernoulli(0.90), independent
  of X. The reliability estimator should predict near 0.90 everywhere; CVI
  should be near zero.
- conditional_miscoverage: X[:,0] > 0 -> coverage ~0.55, X[:,0] <= 0 ->
  coverage ~1.00. Strong conditional miscoverage. CVI_U should be positive
  and the heteroscedastic predictor should have higher CVI than the uniform one
  when scored against the same fitted reliability estimator.

Statistical tests use n=800-1000 with clear signal (0.55 vs 1.00 coverage)
to make results reliable without being slow.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_conformal.cpa import (
    CPAResult,
    CPASelectResult,
    CPASelector,
    ReliabilityEstimator,
    _compute_cvi,
    _parse_intervals,
)


# ===========================================================================
# Synthetic data helpers
# ===========================================================================


def _make_uniform(n: int = 800, alpha: float = 0.10, seed: int = 0) -> dict:
    """Coverage independent of features — should give low CVI."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = rng.exponential(scale=2.0, size=n)
    covered = rng.binomial(1, 1.0 - alpha, size=n).astype(bool)
    lower = np.where(covered, y - 1.0, y + 1.0)
    upper = np.where(covered, y + 1.0, y - 1.0)
    return {"X": X, "y": y, "lower": lower, "upper": upper, "alpha": alpha}


def _make_heteroscedastic(n: int = 800, alpha: float = 0.10, seed: int = 1) -> dict:
    """X[:,0] > 0: coverage 0.55 (severe), X[:,0] <= 0: coverage 1.00."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = rng.exponential(scale=2.0, size=n)
    prob = np.where(X[:, 0] > 0, 0.55, 1.00)
    covered = rng.binomial(1, prob).astype(bool)
    lower = np.where(covered, y - 1.0, y + 1.0)
    upper = np.where(covered, y + 1.0, y - 1.0)
    return {"X": X, "y": y, "lower": lower, "upper": upper, "alpha": alpha}


def _make_small_selector(alpha: float = 0.10, seed: int = 0, n_cal: int = 800) -> CPASelector:
    """Fit a CPASelector with fast settings for constructor/API tests."""
    from sklearn.linear_model import LogisticRegression
    cal = _make_uniform(n=n_cal, alpha=alpha, seed=seed)
    # LogisticRegression is fast and has no hyperparameters to tune
    selector = CPASelector(alpha=alpha, gamma=0.1, calibrate=False, random_state=0)
    selector.fit(cal["X"], cal["y"], cal["lower"], cal["upper"])
    return selector


# ===========================================================================
# Tests: _compute_cvi
# ===========================================================================


class TestComputeCVI:
    """Unit tests for the CVI decomposition formula."""

    def test_all_at_target_gives_zero(self):
        """eta_hat all equal to 1-alpha -> CVI = 0."""
        alpha = 0.10
        eta = np.full(100, 1.0 - alpha)
        cvi, cvi_u, cvi_o, *_ = _compute_cvi(eta, alpha, gamma=0.1)
        assert cvi == 0.0
        assert cvi_u == 0.0
        assert cvi_o == 0.0

    def test_all_undercovered_gives_positive_cvi_u(self):
        """eta_hat well below target -> CVI_U > 0, CVI_O == 0."""
        alpha = 0.10
        eta = np.full(100, 0.50)  # far below 0.90
        cvi, cvi_u, cvi_o, *_ = _compute_cvi(eta, alpha, gamma=0.1)
        assert cvi_u > 0.0
        assert cvi_o == 0.0
        assert np.isclose(cvi, cvi_u)

    def test_all_overcovered_gives_positive_cvi_o(self):
        """eta_hat well above target -> CVI_O > 0, CVI_U == 0."""
        alpha = 0.10
        eta = np.full(100, 1.0)  # far above 0.90
        cvi, cvi_u, cvi_o, *_ = _compute_cvi(eta, alpha, gamma=0.1)
        assert cvi_o > 0.0
        assert cvi_u == 0.0
        assert np.isclose(cvi, cvi_o)

    def test_decomposition_identity(self):
        """CVI = CVI_U + CVI_O always."""
        rng = np.random.default_rng(42)
        eta = rng.uniform(0.0, 1.0, size=500)
        cvi, cvi_u, cvi_o, *_ = _compute_cvi(eta, alpha=0.10, gamma=0.1)
        assert np.isclose(cvi, cvi_u + cvi_o, atol=1e-12)

    def test_cvi_non_negative(self):
        rng = np.random.default_rng(7)
        eta = rng.uniform(0.0, 1.0, size=200)
        cvi, cvi_u, cvi_o, *_ = _compute_cvi(eta, alpha=0.10, gamma=0.1)
        assert cvi >= 0.0
        assert cvi_u >= 0.0
        assert cvi_o >= 0.0

    def test_pi_minus_and_pi_plus_in_unit_interval(self):
        rng = np.random.default_rng(11)
        eta = rng.uniform(0.0, 1.0, size=300)
        _, _, _, pi_minus, pi_plus, *_ = _compute_cvi(eta, alpha=0.10, gamma=0.1)
        assert 0.0 <= pi_minus <= 1.0
        assert 0.0 <= pi_plus <= 1.0

    def test_gamma_zero_larger_cvi(self):
        """With gamma=0, any deviation from 1-alpha penalised. CVI >= CVI at gamma=0.1."""
        rng = np.random.default_rng(99)
        eta = rng.uniform(0.80, 1.00, size=300)
        cvi_loose, *_ = _compute_cvi(eta, alpha=0.10, gamma=0.1)
        cvi_strict, *_ = _compute_cvi(eta, alpha=0.10, gamma=0.0)
        assert cvi_strict >= cvi_loose


# ===========================================================================
# Tests: _parse_intervals
# ===========================================================================


class TestParseIntervals:
    def test_tuple(self):
        lo = np.array([1.0, 2.0, 3.0])
        hi = np.array([2.0, 3.0, 4.0])
        l, u = _parse_intervals((lo, hi))
        np.testing.assert_array_equal(l, lo)
        np.testing.assert_array_equal(u, hi)

    def test_dict(self):
        ps = {"lower": np.array([0.5, 1.5]), "upper": np.array([1.5, 2.5])}
        l, u = _parse_intervals(ps)
        np.testing.assert_array_equal(l, ps["lower"])
        np.testing.assert_array_equal(u, ps["upper"])

    def test_polars_dataframe(self):
        df = pl.DataFrame({"lower": [1.0, 2.0], "upper": [2.0, 3.0]})
        l, u = _parse_intervals(df)
        assert list(l) == [1.0, 2.0]
        assert list(u) == [2.0, 3.0]

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="lower, upper"):
            _parse_intervals((np.array([1.0]),))

    def test_dict_missing_upper_raises(self):
        with pytest.raises(ValueError, match="upper"):
            _parse_intervals({"lower": np.array([1.0])})

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            _parse_intervals(42)


# ===========================================================================
# Tests: ReliabilityEstimator — constructor
# ===========================================================================


class TestReliabilityEstimatorConstructor:
    def test_default_estimator_is_none(self):
        re = ReliabilityEstimator()
        assert re.estimator is None
        assert not re.is_fitted_

    def test_calibrate_default_true(self):
        re = ReliabilityEstimator()
        assert re.calibrate is True

    def test_custom_estimator_stored(self):
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        re = ReliabilityEstimator(estimator=clf)
        assert re.estimator is clf

    def test_not_fitted_initially(self):
        re = ReliabilityEstimator()
        assert not re.is_fitted_


# ===========================================================================
# Tests: ReliabilityEstimator — fit()
# ===========================================================================


class TestReliabilityEstimatorFit:
    def test_fit_returns_self(self):
        d = _make_uniform(n=100, seed=0)
        re = ReliabilityEstimator(calibrate=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = re.fit(d["X"], d["y"], d["lower"], d["upper"])
        assert result is re

    def test_fit_marks_fitted(self):
        d = _make_uniform(n=100, seed=1)
        re = ReliabilityEstimator(calibrate=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            re.fit(d["X"], d["y"], d["lower"], d["upper"])
        assert re.is_fitted_

    def test_fit_too_small_raises(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 2))
        y = rng.exponential(size=20)
        re = ReliabilityEstimator(calibrate=False)
        with pytest.raises(ValueError, match="30"):
            re.fit(X, y, y - 1, y + 1)

    def test_fit_warns_below_500(self):
        d = _make_uniform(n=100, seed=2)
        re = ReliabilityEstimator(calibrate=False)
        with pytest.warns(UserWarning, match="500"):
            re.fit(d["X"], d["y"], d["lower"], d["upper"])

    def test_fit_no_warn_at_500(self):
        d = _make_uniform(n=500, seed=3)
        re = ReliabilityEstimator(calibrate=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            re.fit(d["X"], d["y"], d["lower"], d["upper"])

    def test_fit_mismatched_lower_length_raises(self):
        d = _make_uniform(n=100, seed=4)
        re = ReliabilityEstimator(calibrate=False)
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                re.fit(d["X"], d["y"], d["lower"][:50], d["upper"])

    def test_fit_with_custom_classifier(self):
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0)
        d = _make_uniform(n=200, seed=5)
        re = ReliabilityEstimator(estimator=clf, calibrate=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            re.fit(d["X"], d["y"], d["lower"], d["upper"])
        assert re.is_fitted_


# ===========================================================================
# Tests: ReliabilityEstimator — predict_coverage_proba()
# ===========================================================================


class TestReliabilityEstimatorPredict:
    @pytest.fixture(scope="class")
    def fitted_re(self):
        d = _make_uniform(n=500, seed=10)
        re = ReliabilityEstimator(calibrate=False, random_state=0)
        re.fit(d["X"], d["y"], d["lower"], d["upper"])
        return re

    def test_predict_shape(self, fitted_re):
        X_test = np.random.default_rng(0).normal(size=(100, 4))
        eta = fitted_re.predict_coverage_proba(X_test)
        assert eta.shape == (100,)

    def test_predict_in_unit_interval(self, fitted_re):
        X_test = np.random.default_rng(1).normal(size=(100, 4))
        eta = fitted_re.predict_coverage_proba(X_test)
        assert np.all(eta >= 0.0)
        assert np.all(eta <= 1.0)

    def test_predict_before_fit_raises(self):
        re = ReliabilityEstimator()
        X = np.random.default_rng(0).normal(size=(10, 4))
        with pytest.raises(RuntimeError, match="fit"):
            re.predict_coverage_proba(X)

    def test_predict_1d_input_accepted(self, fitted_re):
        """1D input (single feature) should be reshaped internally."""
        X_1d = np.random.default_rng(5).normal(size=50)
        eta = fitted_re.predict_coverage_proba(X_1d)
        assert eta.shape == (50,)


# ===========================================================================
# Tests: CPASelector — constructor validation
# ===========================================================================


class TestCPASelectorConstructor:
    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            CPASelector(alpha=0.0)

    def test_invalid_alpha_one(self):
        with pytest.raises(ValueError, match="alpha"):
            CPASelector(alpha=1.0)

    def test_invalid_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma"):
            CPASelector(alpha=0.10, gamma=-0.1)

    def test_invalid_gamma_one(self):
        with pytest.raises(ValueError, match="gamma"):
            CPASelector(alpha=0.10, gamma=1.0)

    def test_defaults(self):
        sel = CPASelector(alpha=0.10)
        assert sel.alpha == 0.10
        assert sel.gamma == 0.1
        assert sel.calibrate is True
        assert not sel.is_fitted_
        assert sel.reliability_estimator_ is None


# ===========================================================================
# Tests: CPASelector — fit()
# ===========================================================================


class TestCPASelectorFit:
    def test_fit_returns_self(self):
        d = _make_uniform(n=500, seed=20)
        sel = CPASelector(alpha=0.10, calibrate=False)
        result = sel.fit(d["X"], d["y"], d["lower"], d["upper"])
        assert result is sel

    def test_fit_marks_fitted(self):
        d = _make_uniform(n=500, seed=21)
        sel = CPASelector(alpha=0.10, calibrate=False)
        sel.fit(d["X"], d["y"], d["lower"], d["upper"])
        assert sel.is_fitted_

    def test_fit_creates_reliability_estimator(self):
        d = _make_uniform(n=500, seed=22)
        sel = CPASelector(alpha=0.10, calibrate=False)
        sel.fit(d["X"], d["y"], d["lower"], d["upper"])
        assert isinstance(sel.reliability_estimator_, ReliabilityEstimator)
        assert sel.reliability_estimator_.is_fitted_


# ===========================================================================
# Tests: CPASelector — score()
# ===========================================================================


class TestCPASelectorScore:
    @pytest.fixture(scope="class")
    def selector_and_test(self):
        sel = _make_small_selector(seed=30)
        test = _make_uniform(n=300, seed=130)
        return sel, test

    def test_score_returns_cpa_result(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert isinstance(result, CPAResult)

    def test_cvi_decomposition_identity(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert np.isclose(result.cvi, result.cvi_u + result.cvi_o, atol=1e-12)

    def test_cvi_non_negative(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert result.cvi >= 0.0
        assert result.cvi_u >= 0.0
        assert result.cvi_o >= 0.0

    def test_pi_minus_in_unit_interval(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert 0.0 <= result.pi_minus <= 1.0

    def test_pi_plus_in_unit_interval(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert 0.0 <= result.pi_plus <= 1.0

    def test_marginal_coverage_in_unit_interval(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert 0.0 <= result.marginal_coverage <= 1.0

    def test_eta_hat_shape(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert result.eta_hat.shape == (result.n_obs,)

    def test_eta_hat_in_unit_interval(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert np.all(result.eta_hat >= 0.0)
        assert np.all(result.eta_hat <= 1.0)

    def test_alpha_and_gamma_stored(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert np.isclose(result.alpha, 0.10)
        assert np.isclose(result.gamma, 0.1)

    def test_n_obs_matches_input(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert result.n_obs == len(test["y"])

    def test_score_before_fit_raises(self):
        sel = CPASelector(alpha=0.10)
        d = _make_uniform(n=100)
        with pytest.raises(RuntimeError, match="fit"):
            sel.score(d["X"], d["y"], (d["lower"], d["upper"]))

    def test_score_mismatched_length_raises(self, selector_and_test):
        sel, test = selector_and_test
        with pytest.raises(ValueError):
            sel.score(test["X"], test["y"], (test["lower"][:50], test["upper"]))

    def test_score_accepts_dict_intervals(self, selector_and_test):
        sel, test = selector_and_test
        ps = {"lower": test["lower"], "upper": test["upper"]}
        result = sel.score(test["X"], test["y"], ps)
        assert isinstance(result, CPAResult)

    def test_score_accepts_polars_intervals(self, selector_and_test):
        sel, test = selector_and_test
        ps = pl.DataFrame({"lower": test["lower"], "upper": test["upper"]})
        result = sel.score(test["X"], test["y"], ps)
        assert isinstance(result, CPAResult)

    def test_repr_format(self, selector_and_test):
        sel, test = selector_and_test
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        r = repr(result)
        assert "CPAResult" in r
        assert "cvi=" in r
        assert "cvi_u=" in r


# ===========================================================================
# Tests: CPASelector — select()
# ===========================================================================


class TestCPASelectorSelect:
    @pytest.fixture(scope="class")
    def select_setup(self):
        cal = _make_uniform(n=800, seed=40)
        sel = CPASelector(alpha=0.10, gamma=0.1, calibrate=False, random_state=0)
        sel.fit(cal["X"], cal["y"], cal["lower"], cal["upper"])

        test_u = _make_uniform(n=400, alpha=0.10, seed=41)
        test_h = _make_heteroscedastic(n=400, alpha=0.10, seed=42)

        ps = {
            "uniform": (test_u["lower"], test_u["upper"]),
            "heteroscedastic": (test_h["lower"], test_h["upper"]),
        }
        return sel, test_u["X"], test_u["y"], ps

    def test_select_returns_result_type(self, select_setup):
        sel, X, y, ps = select_setup
        result = sel.select(X, y, ps)
        assert isinstance(result, CPASelectResult)

    def test_best_key_is_in_rankings(self, select_setup):
        sel, X, y, ps = select_setup
        result = sel.select(X, y, ps)
        assert result.best_key in result.rankings

    def test_rankings_sorted_ascending_cvi(self, select_setup):
        sel, X, y, ps = select_setup
        result = sel.select(X, y, ps)
        cvi_values = [result.scores[k].cvi for k in result.rankings]
        assert cvi_values == sorted(cvi_values)

    def test_scores_keys_match_prediction_sets(self, select_setup):
        sel, X, y, ps = select_setup
        result = sel.select(X, y, ps)
        assert set(result.scores.keys()) == set(ps.keys())

    def test_select_before_fit_raises(self):
        sel = CPASelector(alpha=0.10)
        d = _make_uniform(n=100)
        with pytest.raises(RuntimeError, match="fit"):
            sel.select(d["X"], d["y"], {"p": (d["lower"], d["upper"])})

    def test_select_empty_dict_raises(self, select_setup):
        sel, X, y, _ = select_setup
        with pytest.raises(ValueError, match="empty"):
            sel.select(X, y, {})

    def test_select_repr(self, select_setup):
        sel, X, y, ps = select_setup
        result = sel.select(X, y, ps)
        r = repr(result)
        assert "CPASelectResult" in r
        assert "best_key" in r


# ===========================================================================
# Tests: CPASelectResult.compare()
# ===========================================================================


class TestCPASelectResultCompare:
    @pytest.fixture(scope="class")
    def compare_result(self):
        cal = _make_uniform(n=800, seed=50)
        sel = CPASelector(alpha=0.10, gamma=0.1, calibrate=False, random_state=0)
        sel.fit(cal["X"], cal["y"], cal["lower"], cal["upper"])

        test = _make_uniform(n=300, seed=51)
        ps = {
            "a": (test["lower"], test["upper"]),
            "b": (test["lower"] * 1.1, test["upper"] * 0.9),
        }
        return sel.select(test["X"], test["y"], ps)

    def test_compare_returns_polars(self, compare_result):
        df = compare_result.compare()
        assert isinstance(df, pl.DataFrame)

    def test_compare_row_count(self, compare_result):
        df = compare_result.compare()
        assert len(df) == 2

    def test_compare_sorted_by_cvi(self, compare_result):
        df = compare_result.compare()
        cvi_vals = df["cvi"].to_numpy()
        assert np.all(np.diff(cvi_vals) >= -1e-12)

    def test_compare_rank_starts_at_one(self, compare_result):
        df = compare_result.compare()
        assert df["rank"].min() == 1

    def test_compare_best_has_rank_one(self, compare_result):
        df = compare_result.compare()
        best_name = df.filter(pl.col("rank") == 1)["key"][0]
        assert best_name == compare_result.best_key

    def test_compare_expected_columns(self, compare_result):
        df = compare_result.compare()
        expected = {"key", "cvi", "cvi_u", "cvi_o", "pi_minus", "pi_plus",
                    "marginal_coverage", "rank"}
        assert expected.issubset(set(df.columns))


# ===========================================================================
# Tests: statistical properties
# ===========================================================================


class TestStatisticalProperties:
    """
    Statistical claims about CVI behaviour on well-separated synthetic data.
    Uses n=800 with coverage gap 0.55 vs 1.00 — strong enough to be reliable.
    """

    @pytest.fixture(scope="class")
    def fitted_selector_and_data(self):
        """Fit selector on uniform cal; score uniform and heteroscedastic test."""
        cal = _make_uniform(n=800, alpha=0.10, seed=60)
        sel = CPASelector(alpha=0.10, gamma=0.05, calibrate=False, random_state=42)
        sel.fit(cal["X"], cal["y"], cal["lower"], cal["upper"])

        test_u = _make_uniform(n=600, alpha=0.10, seed=61)
        test_h = _make_heteroscedastic(n=600, alpha=0.10, seed=62)

        return sel, test_u, test_h

    def test_hetero_cvi_u_higher_than_uniform(self, fitted_selector_and_data):
        """
        Prediction intervals with conditional miscoverage should have higher
        CVI_U than intervals with uniform coverage.
        """
        sel, test_u, test_h = fitted_selector_and_data
        res_u = sel.score(test_u["X"], test_u["y"], (test_u["lower"], test_u["upper"]))
        res_h = sel.score(test_h["X"], test_h["y"], (test_h["lower"], test_h["upper"]))
        assert res_h.cvi_u >= res_u.cvi_u, (
            f"Expected hetero CVI_U ({res_h.cvi_u:.4f}) >= uniform CVI_U ({res_u.cvi_u:.4f})"
        )

    def test_hetero_cvi_higher_than_uniform(self, fitted_selector_and_data):
        sel, test_u, test_h = fitted_selector_and_data
        res_u = sel.score(test_u["X"], test_u["y"], (test_u["lower"], test_u["upper"]))
        res_h = sel.score(test_h["X"], test_h["y"], (test_h["lower"], test_h["upper"]))
        assert res_h.cvi >= res_u.cvi, (
            f"Expected hetero CVI ({res_h.cvi:.4f}) >= uniform CVI ({res_u.cvi:.4f})"
        )

    def test_select_prefers_uniform_predictor(self):
        """CC-Select should pick the predictor with more uniform coverage."""
        cal = _make_uniform(n=800, alpha=0.10, seed=70)
        sel = CPASelector(alpha=0.10, gamma=0.1, calibrate=False, random_state=7)
        sel.fit(cal["X"], cal["y"], cal["lower"], cal["upper"])

        test_u = _make_uniform(n=600, alpha=0.10, seed=71)
        test_h = _make_heteroscedastic(n=600, alpha=0.10, seed=72)

        X, y = test_u["X"], test_u["y"]
        ps = {
            "uniform": (test_u["lower"], test_u["upper"]),
            "heteroscedastic": (test_h["lower"], test_h["upper"]),
        }
        result = sel.select(X, y, ps)
        assert result.best_key == "uniform", (
            f"Expected 'uniform' to win, got {result.best_key!r}. "
            f"CVI: uniform={result.scores['uniform'].cvi:.4f}, "
            f"heteroscedastic={result.scores['heteroscedastic'].cvi:.4f}"
        )

    def test_uniform_marginal_coverage_near_target(self):
        """Marginal coverage on uniform data should be near 0.90."""
        cal = _make_uniform(n=800, alpha=0.10, seed=80)
        sel = CPASelector(alpha=0.10, calibrate=False)
        sel.fit(cal["X"], cal["y"], cal["lower"], cal["upper"])
        test = _make_uniform(n=500, alpha=0.10, seed=81)
        result = sel.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert abs(result.marginal_coverage - 0.90) < 0.05

    def test_reliability_estimator_used_directly(self):
        """
        ReliabilityEstimator can be used independently for segment audits.
        Young driver segment (X[:,0] < -1) should get predicted coverage
        different from rest when trained on heteroscedastic data.
        """
        d = _make_heteroscedastic(n=800, alpha=0.10, seed=90)
        re = ReliabilityEstimator(calibrate=False, random_state=0)
        re.fit(d["X"], d["y"], d["lower"], d["upper"])

        X_test = np.random.default_rng(91).normal(size=(400, 4))
        eta = re.predict_coverage_proba(X_test)
        # eta should vary — not constant — because conditional coverage is non-uniform
        assert eta.std() > 0.01, (
            f"Expected eta_hat to vary but got std={eta.std():.4f}. "
            "ReliabilityEstimator should detect the conditional coverage pattern."
        )
