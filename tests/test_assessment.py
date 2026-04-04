"""
Tests for insurance_conformal.assessment — ConditionalCoverageAssessor and
related dataclasses (CVIAssessmentResult, CVISelectResult).

Design
------
All tests use synthetic prediction sets — we do not run a full conformal
predictor. This keeps the suite fast and the coverage patterns explicit.

Two canonical data fixtures:
- uniform_data: coverage indicators are i.i.d. Bernoulli(0.90), independent
  of X. The reliability estimator should produce eta_hat near 0.90 everywhere,
  giving CVI near zero.
- heteroscedastic_data: X[:,0] > 0 -> coverage 0.55, X[:,0] <= 0 -> coverage
  1.00. Strong conditional miscoverage. CVI_U should be clearly positive.

Skips
-----
None — all code paths use sklearn (always available).
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_conformal.assessment import (
    ConditionalCoverageAssessor,
    CVIAssessmentResult,
    CVISelectResult,
    _parse_prediction_sets,
)


# ===========================================================================
# Synthetic data helpers
# ===========================================================================


def _make_uniform(n: int = 800, alpha: float = 0.10, seed: int = 0) -> dict:
    """Coverage independent of features — should produce low CVI."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = rng.exponential(scale=2.0, size=n)
    covered = rng.binomial(1, 1.0 - alpha, size=n).astype(bool)
    lower = np.where(covered, y - 1.0, y + 1.0)
    upper = np.where(covered, y + 1.0, y - 1.0)
    return {"X": X, "y": y, "lower": lower, "upper": upper, "alpha": alpha}


def _make_heteroscedastic(n: int = 800, alpha: float = 0.10, seed: int = 1) -> dict:
    """
    X[:,0] > 0: coverage 0.55 (severe undercoverage).
    X[:,0] <= 0: coverage 1.00 (overcoverage).
    CVI_U should be clearly positive.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = rng.exponential(scale=2.0, size=n)
    prob = np.where(X[:, 0] > 0, 0.55, 1.00)
    covered = rng.binomial(1, prob).astype(bool)
    lower = np.where(covered, y - 1.0, y + 1.0)
    upper = np.where(covered, y + 1.0, y - 1.0)
    return {"X": X, "y": y, "lower": lower, "upper": upper, "alpha": alpha}


def _make_assessor_and_data(seed: int = 0, alpha: float = 0.10, n: int = 800):
    """Fit an assessor on uniform calibration data, return assessor + cal + test splits."""
    cal = _make_uniform(n=n, alpha=alpha, seed=seed)
    test = _make_uniform(n=300, alpha=alpha, seed=seed + 99)
    assessor = ConditionalCoverageAssessor(
        alpha=alpha,
        gamma=0.1,
        n_estimators=50,  # small for test speed
        max_depth=3,
        n_splits=3,
        random_state=0,
    )
    assessor.fit(
        cal["X"], cal["y"], (cal["lower"], cal["upper"])
    )
    return assessor, cal, test


# ===========================================================================
# Tests: ConditionalCoverageAssessor constructor validation
# ===========================================================================


class TestConstructorValidation:
    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            ConditionalCoverageAssessor(alpha=0.0)

    def test_invalid_alpha_one(self):
        with pytest.raises(ValueError, match="alpha"):
            ConditionalCoverageAssessor(alpha=1.0)

    def test_invalid_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma"):
            ConditionalCoverageAssessor(alpha=0.10, gamma=-0.1)

    def test_invalid_gamma_one(self):
        with pytest.raises(ValueError, match="gamma"):
            ConditionalCoverageAssessor(alpha=0.10, gamma=1.0)

    def test_invalid_n_splits(self):
        with pytest.raises(ValueError, match="n_splits"):
            ConditionalCoverageAssessor(alpha=0.10, n_splits=1)

    def test_invalid_subsample(self):
        with pytest.raises(ValueError, match="subsample"):
            ConditionalCoverageAssessor(alpha=0.10, subsample=0.0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ConditionalCoverageAssessor(alpha=0.10, learning_rate=0.0)

    def test_defaults_stored(self):
        a = ConditionalCoverageAssessor(alpha=0.10)
        assert a.alpha == 0.10
        assert a.gamma == 0.1
        assert a.n_splits == 5
        assert a.calibrate_proba is True
        assert not a.is_fitted_


# ===========================================================================
# Tests: fit()
# ===========================================================================


class TestFit:
    def test_fit_marks_fitted(self):
        d = _make_uniform(n=200, seed=0)
        a = ConditionalCoverageAssessor(alpha=0.10, n_estimators=20, n_splits=2, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            a.fit(d["X"], d["y"], (d["lower"], d["upper"]))
        assert a.is_fitted_

    def test_fit_returns_self(self):
        d = _make_uniform(n=200, seed=1)
        a = ConditionalCoverageAssessor(alpha=0.10, n_estimators=20, n_splits=2, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = a.fit(d["X"], d["y"], (d["lower"], d["upper"]))
        assert result is a

    def test_fit_too_small_raises(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 2))
        y = rng.normal(size=30)
        a = ConditionalCoverageAssessor(alpha=0.10, n_splits=2)
        with pytest.raises(ValueError, match="50"):
            a.fit(X, y, (y - 1, y + 1))

    def test_fit_warns_below_800(self):
        d = _make_uniform(n=200, seed=2)
        a = ConditionalCoverageAssessor(alpha=0.10, n_estimators=20, n_splits=2, random_state=0)
        with pytest.warns(UserWarning, match="800"):
            a.fit(d["X"], d["y"], (d["lower"], d["upper"]))

    def test_fit_no_warn_at_800(self):
        d = _make_uniform(n=800, seed=3)
        a = ConditionalCoverageAssessor(alpha=0.10, n_estimators=20, n_splits=2, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise
            a.fit(d["X"], d["y"], (d["lower"], d["upper"]))

    def test_fit_mismatched_lengths_raises(self):
        d = _make_uniform(n=200, seed=4)
        a = ConditionalCoverageAssessor(alpha=0.10, n_splits=2)
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a.fit(d["X"], d["y"], (d["lower"][:100], d["upper"]))  # wrong length

    def test_fit_accepts_polars_prediction_sets(self):
        d = _make_uniform(n=200, seed=5)
        ps = pl.DataFrame({"lower": d["lower"], "upper": d["upper"]})
        a = ConditionalCoverageAssessor(alpha=0.10, n_estimators=20, n_splits=2, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a.fit(d["X"], d["y"], ps)
        assert a.is_fitted_

    def test_fit_accepts_dict_prediction_sets(self):
        d = _make_uniform(n=200, seed=6)
        ps = {"lower": d["lower"], "upper": d["upper"]}
        a = ConditionalCoverageAssessor(alpha=0.10, n_estimators=20, n_splits=2, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a.fit(d["X"], d["y"], ps)
        assert a.is_fitted_


# ===========================================================================
# Tests: score()
# ===========================================================================


class TestScore:
    @pytest.fixture(scope="class")
    def fitted_assessor_and_test(self):
        assessor, _, test = _make_assessor_and_data(seed=10)
        return assessor, test

    def test_score_returns_result_type(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert isinstance(result, CVIAssessmentResult)

    def test_cvi_u_plus_cvi_o_equals_cvi(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert np.isclose(result.cvi_u + result.cvi_o, result.cvi, atol=1e-12)

    def test_cvi_non_negative(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert result.cvi >= 0.0

    def test_cvi_u_non_negative(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert result.cvi_u >= 0.0

    def test_cvi_o_non_negative(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert result.cvi_o >= 0.0

    def test_pi_minus_in_unit_interval(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert 0.0 <= result.pi_minus <= 1.0

    def test_pi_plus_in_unit_interval(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert 0.0 <= result.pi_plus <= 1.0

    def test_marginal_coverage_in_unit_interval(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert 0.0 <= result.marginal_coverage <= 1.0

    def test_eta_hat_shape(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert result.eta_hat.shape == (result.n_obs,)

    def test_eta_hat_in_unit_interval(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert np.all(result.eta_hat >= 0.0)
        assert np.all(result.eta_hat <= 1.0)

    def test_alpha_stored(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert np.isclose(result.alpha, 0.10)

    def test_gamma_stored(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert np.isclose(result.gamma, 0.1)

    def test_n_obs_matches_input(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        assert result.n_obs == len(test["y"])

    def test_score_without_fit_raises(self):
        a = ConditionalCoverageAssessor(alpha=0.10)
        d = _make_uniform(n=100, seed=99)
        with pytest.raises(RuntimeError, match="fit"):
            a.score(d["X"], d["y"], (d["lower"], d["upper"]))

    def test_score_mismatched_lengths_raises(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        with pytest.raises(ValueError):
            a.score(test["X"], test["y"], (test["lower"][:100], test["upper"]))

    def test_repr_contains_key_fields(self, fitted_assessor_and_test):
        a, test = fitted_assessor_and_test
        result = a.score(test["X"], test["y"], (test["lower"], test["upper"]))
        r = repr(result)
        assert "CVIAssessmentResult" in r
        assert "cvi=" in r
        assert "cvi_u=" in r


# ===========================================================================
# Tests: CVIAssessmentResult.summary()
# ===========================================================================


class TestSummary:
    def test_summary_returns_polars(self):
        assessor, _, test = _make_assessor_and_data(seed=20)
        result = assessor.score(test["X"], test["y"], (test["lower"], test["upper"]))
        s = result.summary()
        assert isinstance(s, pl.DataFrame)

    def test_summary_one_row(self):
        assessor, _, test = _make_assessor_and_data(seed=21)
        result = assessor.score(test["X"], test["y"], (test["lower"], test["upper"]))
        s = result.summary()
        assert len(s) == 1

    def test_summary_expected_columns(self):
        assessor, _, test = _make_assessor_and_data(seed=22)
        result = assessor.score(test["X"], test["y"], (test["lower"], test["upper"]))
        s = result.summary()
        expected = {"cvi", "cvi_u", "cvi_o", "pi_minus", "pi_plus",
                    "cmu", "cmo", "marginal_coverage", "alpha", "gamma", "n_obs"}
        assert expected.issubset(set(s.columns))


# ===========================================================================
# Tests: select()
# ===========================================================================


class TestSelect:
    @pytest.fixture(scope="class")
    def select_setup(self):
        """
        Two prediction sets: one with near-uniform coverage, one with
        severe conditional miscoverage. Assessor trained on uniform cal.
        """
        cal = _make_uniform(n=800, seed=30)
        assessor = ConditionalCoverageAssessor(
            alpha=0.10, gamma=0.1, n_estimators=50, max_depth=3, n_splits=3, random_state=0
        )
        assessor.fit(cal["X"], cal["y"], (cal["lower"], cal["upper"]))

        test_uniform = _make_uniform(n=400, alpha=0.10, seed=31)
        test_hetero = _make_heteroscedastic(n=400, alpha=0.10, seed=32)

        ps = {
            "uniform": (test_uniform["lower"], test_uniform["upper"]),
            "heteroscedastic": (test_hetero["lower"], test_hetero["upper"]),
        }
        return assessor, test_uniform["X"], test_uniform["y"], ps

    def test_select_returns_result_type(self, select_setup):
        assessor, X, y, ps = select_setup
        result = assessor.select(X, y, ps)
        assert isinstance(result, CVISelectResult)

    def test_best_key_in_rankings(self, select_setup):
        assessor, X, y, ps = select_setup
        result = assessor.select(X, y, ps)
        assert result.best_key in result.rankings

    def test_rankings_sorted_by_cvi(self, select_setup):
        assessor, X, y, ps = select_setup
        result = assessor.select(X, y, ps)
        cvi_values = [result.scores[k].cvi for k in result.rankings]
        assert cvi_values == sorted(cvi_values)

    def test_scores_keys_match_prediction_sets(self, select_setup):
        assessor, X, y, ps = select_setup
        result = assessor.select(X, y, ps)
        assert set(result.scores.keys()) == set(ps.keys())

    def test_select_without_fit_raises(self):
        a = ConditionalCoverageAssessor(alpha=0.10)
        d = _make_uniform(n=100)
        with pytest.raises(RuntimeError, match="fit"):
            a.select(d["X"], d["y"], {"p": (d["lower"], d["upper"])})

    def test_select_empty_raises(self, select_setup):
        assessor, X, y, _ = select_setup
        with pytest.raises(ValueError, match="empty"):
            assessor.select(X, y, {})

    def test_select_repr(self, select_setup):
        assessor, X, y, ps = select_setup
        result = assessor.select(X, y, ps)
        r = repr(result)
        assert "CVISelectResult" in r
        assert "best_key" in r


# ===========================================================================
# Tests: CVISelectResult.compare()
# ===========================================================================


class TestCompare:
    @pytest.fixture(scope="class")
    def compare_result(self):
        cal = _make_uniform(n=800, seed=40)
        assessor = ConditionalCoverageAssessor(
            alpha=0.10, n_estimators=50, max_depth=3, n_splits=3, random_state=0
        )
        assessor.fit(cal["X"], cal["y"], (cal["lower"], cal["upper"]))
        test = _make_uniform(n=300, seed=41)
        ps = {
            "a": (test["lower"], test["upper"]),
            "b": (test["lower"] * 1.1, test["upper"] * 0.9),  # slightly different
        }
        return assessor.select(test["X"], test["y"], ps)

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
# Tests: _parse_prediction_sets
# ===========================================================================


class TestParsePredictionSets:
    def test_tuple(self):
        lower = np.array([1.0, 2.0, 3.0])
        upper = np.array([2.0, 3.0, 4.0])
        lo, hi = _parse_prediction_sets((lower, upper))
        np.testing.assert_array_equal(lo, lower)
        np.testing.assert_array_equal(hi, upper)

    def test_polars_dataframe(self):
        df = pl.DataFrame({"lower": [1.0, 2.0], "upper": [2.0, 3.0]})
        lo, hi = _parse_prediction_sets(df)
        assert list(lo) == [1.0, 2.0]
        assert list(hi) == [2.0, 3.0]

    def test_dict(self):
        ps = {"lower": np.array([0.5, 1.5]), "upper": np.array([1.5, 2.5])}
        lo, hi = _parse_prediction_sets(ps)
        np.testing.assert_array_equal(lo, ps["lower"])
        np.testing.assert_array_equal(hi, ps["upper"])

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="lower, upper"):
            _parse_prediction_sets((np.array([1.0]),))

    def test_missing_upper_col_raises(self):
        with pytest.raises(ValueError, match="upper"):
            _parse_prediction_sets({"lower": np.array([1.0])})

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            _parse_prediction_sets(42)


# ===========================================================================
# Tests: statistical — heteroscedastic data produces higher CVI_U
# ===========================================================================


class TestStatisticalProperties:
    """
    These tests make statistical claims about CVI behaviour. They use large enough
    samples and strong enough conditional miscoverage to be reliable.
    """

    @pytest.fixture(scope="class")
    def uniform_vs_hetero(self):
        """Fit on uniform cal, score both uniform and heteroscedastic test sets."""
        cal = _make_uniform(n=800, alpha=0.10, seed=50)
        assessor = ConditionalCoverageAssessor(
            alpha=0.10, gamma=0.05,  # tight gamma to be sensitive
            n_estimators=100, max_depth=4, n_splits=3, random_state=42
        )
        assessor.fit(cal["X"], cal["y"], (cal["lower"], cal["upper"]))

        test_u = _make_uniform(n=600, alpha=0.10, seed=51)
        test_h = _make_heteroscedastic(n=600, alpha=0.10, seed=52)

        res_u = assessor.score(test_u["X"], test_u["y"], (test_u["lower"], test_u["upper"]))
        res_h = assessor.score(test_h["X"], test_h["y"], (test_h["lower"], test_h["upper"]))
        return res_u, res_h

    def test_hetero_cvi_u_higher_than_uniform(self, uniform_vs_hetero):
        """
        Heteroscedastic miscoverage should produce higher CVI_U than uniform.
        The half of the data with X[:,0]>0 gets 55% coverage at a 90% target —
        a reliable estimator should detect this.
        """
        res_u, res_h = uniform_vs_hetero
        assert res_h.cvi_u >= res_u.cvi_u, (
            f"Expected hetero CVI_U ({res_h.cvi_u:.4f}) >= uniform CVI_U ({res_u.cvi_u:.4f})"
        )

    def test_select_prefers_uniform_predictor(self):
        """
        CC-Select should pick the predictor with more uniform conditional coverage.
        """
        cal = _make_uniform(n=800, alpha=0.10, seed=60)
        assessor = ConditionalCoverageAssessor(
            alpha=0.10, gamma=0.1, n_estimators=100, max_depth=4, n_splits=3, random_state=7
        )
        assessor.fit(cal["X"], cal["y"], (cal["lower"], cal["upper"]))

        test = _make_uniform(n=600, alpha=0.10, seed=61)
        test_h = _make_heteroscedastic(n=600, alpha=0.10, seed=62)

        # Use the same X and y for both — only intervals differ
        X, y = test["X"], test["y"]
        ps = {
            "uniform": (test["lower"], test["upper"]),
            "heteroscedastic": (test_h["lower"], test_h["upper"]),
        }
        sel = assessor.select(X, y, ps)
        assert sel.best_key == "uniform", (
            f"Expected 'uniform' to win, got {sel.best_key!r}. "
            f"CVI scores: {sel.scores['uniform'].cvi:.4f} vs "
            f"{sel.scores['heteroscedastic'].cvi:.4f}"
        )
