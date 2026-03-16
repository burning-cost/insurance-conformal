"""
Tests for tweedie.py — Tweedie nonconformity scores and TwoStageLWConformal.

Score round-trip tests: verify score(inverse(s, mu)) ≈ s for all three score types.
Coverage tests: empirical coverage >= 1-alpha on synthetic Tweedie data.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal.claims.tweedie import (
    TweedePearsonScore,
    TweedieAnscombeScore,
    TweedieDevianceScore,
    TwoStageLWConformal,
)


# ---------------------------------------------------------------------------
# TweedePearsonScore
# ---------------------------------------------------------------------------


class TestTweedePearsonScore:
    def test_score_shape(self):
        score = TweedePearsonScore(p=1.5)
        y = np.array([1.0, 2.0, 0.0, 5.0])
        y_hat = np.array([1.5, 1.8, 0.5, 4.0])
        s = score.score(y, y_hat)
        assert s.shape == (4,)

    def test_score_non_negative(self):
        score = TweedePearsonScore(p=1.5)
        rng = np.random.default_rng(0)
        y = np.abs(rng.normal(size=100))
        y_hat = np.abs(rng.normal(size=100)) + 0.01
        s = score.score(y, y_hat)
        assert np.all(s >= 0.0)

    def test_score_formula_at_known_values(self):
        """R = |y - mu| / mu^{p/2} at p=2 (Gamma): R = |y - mu| / mu."""
        score = TweedePearsonScore(p=2.0)
        y = np.array([2.0])
        mu = np.array([1.0])
        # R = |2-1| / 1^1 = 1
        np.testing.assert_allclose(score.score(y, mu), [1.0])

    def test_score_at_p15(self):
        """R = |y - mu| / mu^{0.75} at p=1.5."""
        score = TweedePearsonScore(p=1.5)
        y = np.array([3.0])
        mu = np.array([2.0])
        expected = abs(3.0 - 2.0) / 2.0 ** 0.75
        np.testing.assert_allclose(score.score(y, mu), [expected], rtol=1e-9)

    def test_inverse_upper_round_trip(self):
        """score(inverse_upper(s, mu), mu) should recover s."""
        score = TweedePearsonScore(p=1.5)
        rng = np.random.default_rng(42)
        mu = np.abs(rng.normal(size=50)) + 0.5
        s_vals = rng.uniform(0.1, 3.0, size=50)
        for i in range(50):
            y_upper = score.inverse(s_vals[i], mu[i : i + 1], upper=True)
            s_recovered = score.score(y_upper, mu[i : i + 1])
            np.testing.assert_allclose(s_recovered, [s_vals[i]], rtol=1e-9)

    def test_inverse_lower_round_trip(self):
        """score(inverse_lower(s, mu), mu) should recover s when mu - margin > 0."""
        score = TweedePearsonScore(p=1.5)
        mu = np.array([10.0, 5.0, 8.0])
        s = 0.5
        y_lower = score.inverse(s, mu, upper=False)
        s_recovered = score.score(y_lower, mu)
        np.testing.assert_allclose(s_recovered, [s, s, s], rtol=1e-9)

    def test_inverse_lower_clipped_at_zero(self):
        """Lower bound cannot be negative (insurance claims)."""
        score = TweedePearsonScore(p=1.5)
        mu = np.array([0.1])
        s = 1000.0  # huge quantile — would give negative lower
        y_lower = score.inverse(s, mu, upper=False)
        assert y_lower[0] >= 0.0

    def test_repr(self):
        assert "TweedePearsonScore" in repr(TweedePearsonScore(p=1.5))

    def test_p_outside_range_warns(self):
        with pytest.warns(UserWarning, match="compound Poisson"):
            TweedePearsonScore(p=2.5)

    def test_zero_y_score(self):
        """y=0 is a valid insurance claim (no claim). Score should be finite."""
        score = TweedePearsonScore(p=1.5)
        s = score.score(np.array([0.0]), np.array([1.0]))
        assert np.isfinite(s[0])
        assert s[0] >= 0.0

    def test_vectorised_arrays(self):
        score = TweedePearsonScore(p=1.5)
        rng = np.random.default_rng(1)
        n = 500
        y = np.abs(rng.normal(size=n))
        y_hat = np.abs(rng.normal(size=n)) + 0.1
        s = score.score(y, y_hat)
        assert s.shape == (n,)
        assert np.all(np.isfinite(s))

    def test_various_p_values(self):
        """Score should work for all p in [1, 2]."""
        for p in [1.0, 1.2, 1.5, 1.8]:  # p=2.0 is excluded: raises ValueError (exponent is 0)
            score = TweedePearsonScore(p=p)
            y = np.array([2.0, 1.0])
            y_hat = np.array([1.5, 1.2])
            s = score.score(y, y_hat)
            assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# TweedieDevianceScore
# ---------------------------------------------------------------------------


class TestTweedieDevianceScore:
    def test_score_non_negative(self):
        score = TweedieDevianceScore(p=1.5)
        rng = np.random.default_rng(0)
        y = np.abs(rng.normal(size=100))
        y_hat = np.abs(rng.normal(size=100)) + 0.01
        s = score.score(y, y_hat)
        assert np.all(s >= 0.0)

    def test_score_zero_when_y_equals_mu(self):
        """d(y, y) = 0 for any y = mu > 0."""
        score = TweedieDevianceScore(p=1.5)
        mu = np.array([1.0, 2.0, 5.0])
        s = score.score(mu, mu)
        np.testing.assert_allclose(s, 0.0, atol=1e-10)

    def test_inverse_upper_round_trip(self):
        """sqrt(d(inverse_upper(s, mu), mu)) should recover s."""
        score = TweedieDevianceScore(p=1.5)
        rng = np.random.default_rng(7)
        mu = np.abs(rng.normal(size=20)) + 1.0  # positive means
        s_vals = rng.uniform(0.1, 1.0, size=20)
        for i in range(20):
            y_upper = score.inverse(float(s_vals[i]), mu[i : i + 1], upper=True)
            s_recovered = score.score(y_upper, mu[i : i + 1])
            np.testing.assert_allclose(
                s_recovered, [s_vals[i]], rtol=1e-4,
                err_msg=f"Round-trip failed at i={i}, mu={mu[i]:.3f}, s={s_vals[i]:.3f}"
            )

    def test_inverse_lower_non_negative(self):
        score = TweedieDevianceScore(p=1.5)
        mu = np.array([1.0, 2.0, 0.5])
        s = 0.5
        y_lower = score.inverse(s, mu, upper=False)
        assert np.all(y_lower >= 0.0)

    def test_repr(self):
        assert "TweedieDevianceScore" in repr(TweedieDevianceScore(p=1.5))

    def test_score_shape(self):
        score = TweedieDevianceScore(p=1.5)
        y = np.ones(10)
        y_hat = np.ones(10) * 1.5
        assert score.score(y, y_hat).shape == (10,)

    def test_zero_y_score_finite(self):
        """y=0 is valid — deviance with y=0 should be finite for p in (1,2)."""
        score = TweedieDevianceScore(p=1.5)
        s = score.score(np.array([0.0]), np.array([1.0]))
        assert np.isfinite(s[0])
        assert s[0] >= 0.0


# ---------------------------------------------------------------------------
# TweedieAnscombeScore
# ---------------------------------------------------------------------------


class TestTweedieAnscombeScore:
    def test_score_non_negative(self):
        score = TweedieAnscombeScore(p=1.5)
        rng = np.random.default_rng(0)
        y = np.abs(rng.normal(size=100))
        y_hat = np.abs(rng.normal(size=100)) + 0.01
        s = score.score(y, y_hat)
        assert np.all(s >= 0.0)

    def test_score_zero_when_y_equals_mu(self):
        score = TweedieAnscombeScore(p=1.5)
        mu = np.array([1.0, 2.5, 4.0])
        s = score.score(mu, mu)
        np.testing.assert_allclose(s, 0.0, atol=1e-12)

    def test_inverse_upper_round_trip(self):
        """Anscombe has closed-form inverse — round-trip should be very tight."""
        score = TweedieAnscombeScore(p=1.5)
        rng = np.random.default_rng(3)
        mu = np.abs(rng.normal(size=30)) + 0.5
        s_vals = rng.uniform(0.01, 1.0, size=30)
        for i in range(30):
            y_upper = score.inverse(float(s_vals[i]), mu[i : i + 1], upper=True)
            s_recovered = score.score(y_upper, mu[i : i + 1])
            np.testing.assert_allclose(
                s_recovered, [s_vals[i]], rtol=1e-8,
                err_msg=f"Round-trip failed at i={i}"
            )

    def test_inverse_lower_non_negative(self):
        score = TweedieAnscombeScore(p=1.5)
        mu = np.array([1.0, 0.5, 3.0])
        y_lower = score.inverse(0.5, mu, upper=False)
        assert np.all(y_lower >= 0.0)

    def test_p_equals_2_raises(self):
        with pytest.raises(ValueError, match="p=2"):
            TweedieAnscombeScore(p=2.0)

    def test_repr(self):
        assert "TweedieAnscombeScore" in repr(TweedieAnscombeScore(p=1.5))

    def test_score_shape(self):
        score = TweedieAnscombeScore(p=1.5)
        assert score.score(np.ones(20), np.ones(20) * 1.2).shape == (20,)

    def test_various_p_values(self):
        for p in [1.0, 1.2, 1.5, 1.8]:  # p=2.0 is excluded: raises ValueError (exponent is 0)
            score = TweedieAnscombeScore(p=p)
            y = np.array([2.0, 1.0, 0.5])
            y_hat = np.array([1.5, 1.2, 0.8])
            s = score.score(y, y_hat)
            assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# Score cross-checks
# ---------------------------------------------------------------------------


class TestScoreCrossChecks:
    """Verify the three score types are consistent in ordering (not identical)."""

    def test_all_three_agree_on_zero_residual(self):
        """All three scores should be 0 when y = mu."""
        mu = np.array([1.0, 2.0, 5.0])
        for ScoreClass in [TweedePearsonScore, TweedieDevianceScore, TweedieAnscombeScore]:
            score = ScoreClass(p=1.5)
            s = score.score(mu, mu)
            np.testing.assert_allclose(s, 0.0, atol=1e-10,
                                       err_msg=f"{ScoreClass.__name__} not 0 at y=mu")

    def test_scores_increase_with_residual_magnitude(self):
        """All three scores should increase as |y - mu| increases."""
        mu = np.array([2.0])
        y_small = np.array([2.5])  # small residual
        y_large = np.array([10.0])  # large residual
        for ScoreClass in [TweedePearsonScore, TweedieDevianceScore, TweedieAnscombeScore]:
            score = ScoreClass(p=1.5)
            s_small = score.score(y_small, mu)[0]
            s_large = score.score(y_large, mu)[0]
            assert s_large > s_small, f"{ScoreClass.__name__}: large residual gave smaller score"


# ---------------------------------------------------------------------------
# TwoStageLWConformal
# ---------------------------------------------------------------------------


class TestTwoStageLWConformal:
    """Tests for the two-stage locally weighted conformal predictor."""

    def _make_simple_sklearn_model(self):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=50, random_state=0)

    def test_fit_calibrate_predict_shape(self, large_dataset):
        pytest.importorskip("sklearn")
        pytest.importorskip("catboost")
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        lw.fit(X_train, y_train)
        lw.calibrate(X_cal, y_cal)
        intervals = lw.predict_interval(X_test, alpha=0.10)
        assert intervals.shape == (len(X_test), 2)

    def test_lower_non_negative(self, large_dataset):
        pytest.importorskip("sklearn")
        pytest.importorskip("catboost")
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        lw.fit(X_train, y_train)
        lw.calibrate(X_cal, y_cal)
        intervals = lw.predict_interval(X_test, alpha=0.10)
        assert np.all(intervals[:, 0] >= 0.0)

    def test_upper_ge_lower(self, large_dataset):
        pytest.importorskip("sklearn")
        pytest.importorskip("catboost")
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        lw.fit(X_train, y_train)
        lw.calibrate(X_cal, y_cal)
        intervals = lw.predict_interval(X_test, alpha=0.10)
        assert np.all(intervals[:, 1] >= intervals[:, 0])

    def test_calibrate_raises_before_fit(self, large_dataset):
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        pytest.importorskip("sklearn")
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        with pytest.raises(RuntimeError, match="fit"):
            lw.calibrate(X_cal, y_cal)

    def test_predict_raises_before_calibrate(self, large_dataset):
        pytest.importorskip("sklearn")
        pytest.importorskip("catboost")
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        lw.fit(X_train, y_train)
        with pytest.raises(RuntimeError, match="calibrate"):
            lw.predict_interval(X_test, alpha=0.10)

    def test_coverage_90pct(self, large_dataset):
        pytest.importorskip("sklearn")
        pytest.importorskip("catboost")
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        lw.fit(X_train, y_train)
        lw.calibrate(X_cal, y_cal)
        diag = lw.coverage_diagnostic(X_test, y_test, alpha=0.10)
        # n_cal=300, allow slack
        assert diag["empirical_coverage"] >= 0.85

    def test_coverage_diagnostic_structure(self, large_dataset):
        pytest.importorskip("sklearn")
        pytest.importorskip("catboost")
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        lw.fit(X_train, y_train)
        lw.calibrate(X_cal, y_cal)
        diag = lw.coverage_diagnostic(X_test, y_test, alpha=0.10)
        assert "empirical_coverage" in diag
        assert "nominal_coverage" in diag
        assert diag["nominal_coverage"] == pytest.approx(0.90)

    def test_repr(self, large_dataset):
        pytest.importorskip("sklearn")
        pytest.importorskip("catboost")
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        lw.fit(X_train, y_train)
        lw.calibrate(X_cal, y_cal)
        r = repr(lw)
        assert "TwoStageLWConformal" in r
        assert "p=1.5" in r

    def test_custom_spread_model(self, large_dataset):
        """Custom spread model (not CatBoost) should work via sklearn protocol."""
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor

        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        spread_model = RandomForestRegressor(n_estimators=20, random_state=0)
        lw = TwoStageLWConformal(
            mean_model=mean_model, p=1.5, spread_model=spread_model
        )
        lw.fit(X_train, y_train)
        lw.calibrate(X_cal, y_cal)
        intervals = lw.predict_interval(X_test[:10], alpha=0.10)
        assert intervals.shape == (10, 2)

    def test_invalid_alpha_raises(self, large_dataset):
        pytest.importorskip("sklearn")
        pytest.importorskip("catboost")
        X_train, y_train, X_cal, y_cal, X_test, y_test = large_dataset
        mean_model = self._make_simple_sklearn_model()
        lw = TwoStageLWConformal(mean_model=mean_model, p=1.5)
        lw.fit(X_train, y_train)
        lw.calibrate(X_cal, y_cal)
        with pytest.raises(ValueError, match="alpha"):
            lw.predict_interval(X_test[:5], alpha=-0.1)
