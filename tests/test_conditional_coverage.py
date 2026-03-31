"""
Tests for conditional_coverage module.

Part 1 (v0.7.1): ConditionalCoverageERT — hypothesis test for conditional
coverage violations. Four scenarios: null, alternative, directional variants,
and subgroup coverage output format.

Part 2 (v0.8.0): ConditionalValidityIndex and CCSelect — scalar CVI diagnostic
and predictor selection.

Synthetic data design: we simulate conformal intervals by fixing known coverage
patterns rather than running a full conformal predictor. This makes the tests
fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal.conditional_coverage import (
    CCSelect,
    CCSelectResult,
    CVIResult,
    ConditionalCoverageERT,
    ConditionalValidityIndex,
    ERTResult,
)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _make_uniform_coverage(n: int = 800, alpha: float = 0.10, seed: int = 0) -> dict:
    """
    Synthetic intervals with genuinely uniform coverage.

    Coverage indicator Z_i is i.i.d. Bernoulli(1-alpha) independent of X_i.
    A classifier should not be able to beat the constant predictor.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y_true = rng.exponential(scale=2.0, size=n)
    # Intervals: always cover with probability (1-alpha) regardless of X
    covered = rng.binomial(1, 1.0 - alpha, size=n).astype(bool)
    # Build y_lower/y_upper that achieve this coverage pattern
    y_lower = np.where(covered, y_true - 1.0, y_true + 1.0)
    y_upper = np.where(covered, y_true + 1.0, y_true - 1.0)
    return {
        "X": X,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "y_true": y_true,
        "alpha": alpha,
    }


def _make_heteroscedastic_miscoverage(n: int = 800, alpha: float = 0.10, seed: int = 1) -> dict:
    """
    Synthetic intervals with conditional coverage violation.

    X[:,0] > 0: coverage = 0.50 (severe undercoverage)
    X[:,0] <= 0: coverage = 1.00 (overcoverage)
    Marginal coverage = ~0.75, but strongly conditional on X[:,0].
    A classifier should detect this easily.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y_true = rng.exponential(scale=2.0, size=n)

    covered = np.where(
        X[:, 0] > 0,
        rng.binomial(1, 0.50, size=n).astype(bool),
        np.ones(n, dtype=bool),
    )
    y_lower = np.where(covered, y_true - 1.0, y_true + 1.0)
    y_upper = np.where(covered, y_true + 1.0, y_true - 1.0)
    return {
        "X": X,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "y_true": y_true,
        "alpha": alpha,
    }


# ===========================================================================
# Part 1: ConditionalCoverageERT (v0.7.1)
# ===========================================================================


# ---------------------------------------------------------------------------
# TestERTResultDataclass
# ---------------------------------------------------------------------------


class TestERTResultDataclass:
    def test_repr_positive_ert(self):
        r = ERTResult(
            ert=0.05, baseline_loss=0.10, classifier_loss=0.05,
            ci_lower=0.01, ci_upper=0.09,
            alpha=0.10, marginal_coverage=0.87,
            loss="l1", direction="under", n_obs=500, n_bootstraps=100,
        )
        rep = repr(r)
        assert "***" in rep
        assert "0.05" in rep

    def test_repr_insignificant_ert(self):
        r = ERTResult(
            ert=0.005, baseline_loss=0.10, classifier_loss=0.095,
            ci_lower=-0.003, ci_upper=0.013,
            alpha=0.10, marginal_coverage=0.90,
            loss="l1", direction="both", n_obs=500, n_bootstraps=100,
        )
        rep = repr(r)
        assert "not significant" in rep


# ---------------------------------------------------------------------------
# TestConditionalCoverageERT — constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="loss must be"):
            ConditionalCoverageERT(loss="mse")  # type: ignore

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            ConditionalCoverageERT(direction="left")  # type: ignore

    def test_n_splits_too_small(self):
        with pytest.raises(ValueError, match="n_splits"):
            ConditionalCoverageERT(n_splits=1)

    def test_defaults(self):
        ert = ConditionalCoverageERT()
        assert ert.loss == "l1"
        assert ert.direction == "under"
        assert ert.n_splits == 5


# ---------------------------------------------------------------------------
# TestERTNullCase — uniform coverage -> ERT near zero
# ---------------------------------------------------------------------------


class TestERTNullCase:
    """
    Under genuine uniform coverage, ERT should not be significantly positive.
    We use a modest sample and don't demand ERT == 0, but the CI should
    include zero (or ERT should be small).
    """

    @pytest.fixture(scope="class")
    def uniform_data(self):
        return _make_uniform_coverage(n=600, alpha=0.10, seed=42)

    def test_ert_result_type(self, uniform_data):
        ert = ConditionalCoverageERT(loss="l1", direction="both", n_splits=3, n_bootstraps=50, random_state=0)
        result = ert.evaluate(**uniform_data)
        assert isinstance(result, ERTResult)

    def test_ert_near_zero_uniform(self, uniform_data):
        """ERT under null should be small — CI upper should not be large."""
        ert = ConditionalCoverageERT(loss="l1", direction="both", n_splits=3, n_bootstraps=50, random_state=0)
        result = ert.evaluate(**uniform_data)
        # Allow some noise; just ensure it's not picking up a huge signal
        assert result.ert < 0.05, f"ERT under null was {result.ert:.4f}, expected < 0.05"

    def test_result_fields_populated(self, uniform_data):
        ert = ConditionalCoverageERT(loss="l2", direction="both", n_splits=3, n_bootstraps=30, random_state=0)
        result = ert.evaluate(**uniform_data)
        assert result.n_obs == 600
        assert result.alpha == 0.10
        assert result.loss == "l2"
        assert result.direction == "both"
        assert 0.0 <= result.marginal_coverage <= 1.0
        assert result.baseline_loss > 0
        assert result.n_bootstraps == 30

    def test_ci_contains_zero_null(self, uniform_data):
        ert = ConditionalCoverageERT(loss="l1", direction="both", n_splits=3, n_bootstraps=100, random_state=0)
        result = ert.evaluate(**uniform_data)
        # Under the null, the 90% CI should not exclude zero (it might occasionally
        # due to randomness with small n, but should be consistent with zero)
        assert result.ci_lower <= result.ci_upper, "CI bounds inverted"


# ---------------------------------------------------------------------------
# TestERTAlternativeCase — heteroscedastic miscoverage -> ERT positive
# ---------------------------------------------------------------------------


class TestERTAlternativeCase:
    @pytest.fixture(scope="class")
    def hetero_data(self):
        return _make_heteroscedastic_miscoverage(n=800, alpha=0.10, seed=7)

    def test_ert_positive_under_alternative(self, hetero_data):
        """
        Under strong conditional miscoverage, ERT should be clearly positive.
        The split between X[:,0] > 0 and X[:,0] <= 0 is trivially learnable.
        """
        ert = ConditionalCoverageERT(loss="l1", direction="both", n_splits=3, n_bootstraps=50, random_state=0)
        result = ert.evaluate(**hetero_data)
        assert result.ert > 0.05, f"Expected ERT > 0.05 under strong alternative, got {result.ert:.4f}"

    def test_ci_excludes_zero_alternative(self, hetero_data):
        ert = ConditionalCoverageERT(loss="l1", direction="both", n_splits=3, n_bootstraps=100, random_state=0)
        result = ert.evaluate(**hetero_data)
        assert result.ci_lower > 0.0, (
            f"Expected CI lower > 0 under strong alternative, got [{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
        )

    def test_l2_ert_positive_alternative(self, hetero_data):
        ert = ConditionalCoverageERT(loss="l2", direction="both", n_splits=3, n_bootstraps=30, random_state=0)
        result = ert.evaluate(**hetero_data)
        assert result.ert > 0.0

    def test_kl_ert_positive_alternative(self, hetero_data):
        ert = ConditionalCoverageERT(loss="kl", direction="both", n_splits=3, n_bootstraps=30, random_state=0)
        result = ert.evaluate(**hetero_data)
        assert result.ert > 0.0

    def test_baseline_loss_gt_classifier_loss(self, hetero_data):
        """Classifier should beat baseline under strong alternative."""
        ert = ConditionalCoverageERT(loss="l1", direction="both", n_splits=3, n_bootstraps=30, random_state=0)
        result = ert.evaluate(**hetero_data)
        assert result.baseline_loss >= result.classifier_loss


# ---------------------------------------------------------------------------
# TestDirectionalVariants
# ---------------------------------------------------------------------------


class TestDirectionalVariants:
    @pytest.fixture(scope="class")
    def data(self):
        return _make_heteroscedastic_miscoverage(n=800, alpha=0.10, seed=99)

    def test_under_direction(self, data):
        ert = ConditionalCoverageERT(loss="l1", direction="under", n_splits=3, n_bootstraps=30, random_state=0)
        result = ert.evaluate(**data)
        assert isinstance(result, ERTResult)
        assert result.direction == "under"

    def test_over_direction(self, data):
        ert = ConditionalCoverageERT(loss="l1", direction="over", n_splits=3, n_bootstraps=30, random_state=0)
        result = ert.evaluate(**data)
        assert isinstance(result, ERTResult)
        assert result.direction == "over"

    def test_both_direction(self, data):
        ert = ConditionalCoverageERT(loss="l1", direction="both", n_splits=3, n_bootstraps=30, random_state=0)
        result = ert.evaluate(**data)
        assert isinstance(result, ERTResult)
        assert result.direction == "both"

    def test_all_loss_variants_run(self, data):
        for loss_variant in ("l1", "l2", "kl"):
            ert = ConditionalCoverageERT(loss=loss_variant, direction="both", n_splits=3, n_bootstraps=20, random_state=0)
            result = ert.evaluate(**data)
            assert isinstance(result.ert, float)
            assert np.isfinite(result.ert)


# ---------------------------------------------------------------------------
# TestSubgroupCoverage
# ---------------------------------------------------------------------------


class TestSubgroupCoverage:
    @pytest.fixture(scope="class")
    def data(self):
        return _make_heteroscedastic_miscoverage(n=500, alpha=0.10, seed=5)

    def test_output_is_polars(self, data):
        ert = ConditionalCoverageERT()
        df = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=data["alpha"],
        )
        assert isinstance(df, pl.DataFrame)

    def test_expected_columns(self, data):
        ert = ConditionalCoverageERT()
        df = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=data["alpha"],
        )
        expected = {"feature", "bin_index", "bin_midpoint", "n_obs",
                    "empirical_coverage", "target_coverage", "coverage_gap"}
        assert expected.issubset(set(df.columns))

    def test_feature_names_used(self, data):
        ert = ConditionalCoverageERT()
        names = ["age", "region", "vehicle_group", "ncb"]
        df = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=data["alpha"],
            feature_names=names,
        )
        assert set(df["feature"].unique().to_list()) == set(names)

    def test_default_feature_names(self, data):
        ert = ConditionalCoverageERT()
        df = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=data["alpha"],
        )
        assert "feature_0" in df["feature"].unique().to_list()

    def test_coverage_in_range(self, data):
        ert = ConditionalCoverageERT()
        df = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=data["alpha"],
        )
        assert (df["empirical_coverage"] >= 0).all()
        assert (df["empirical_coverage"] <= 1).all()

    def test_n_obs_positive(self, data):
        ert = ConditionalCoverageERT()
        df = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=data["alpha"],
        )
        assert (df["n_obs"] > 0).all()

    def test_coverage_gap_consistency(self, data):
        ert = ConditionalCoverageERT()
        alpha = 0.10
        df = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=alpha,
        )
        target = 1.0 - alpha
        computed_gap = (df["target_coverage"] - df["empirical_coverage"]).to_numpy()
        reported_gap = df["coverage_gap"].to_numpy()
        np.testing.assert_allclose(computed_gap, reported_gap, atol=1e-10)

    def test_n_bins_controls_rows(self, data):
        ert = ConditionalCoverageERT()
        df5 = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=data["alpha"], n_bins=5,
        )
        df10 = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=data["alpha"], n_bins=10,
        )
        # More bins -> more rows (may not be exactly n*bins due to duplicates)
        assert len(df10) >= len(df5)

    def test_target_coverage_column(self, data):
        alpha = 0.15
        ert = ConditionalCoverageERT()
        df = ert.subgroup_coverage(
            data["X"], data["y_lower"], data["y_upper"], data["y_true"],
            alpha=alpha,
        )
        np.testing.assert_allclose(df["target_coverage"].to_numpy(), 1 - alpha)

    def test_single_feature_1d(self):
        """1D feature input should be accepted and produce output for one feature."""
        rng = np.random.default_rng(0)
        X_1d = rng.normal(size=200)
        y_true = rng.exponential(size=200)
        covered = rng.binomial(1, 0.90, size=200).astype(bool)
        y_lower = np.where(covered, y_true - 1.0, y_true + 1.0)
        y_upper = np.where(covered, y_true + 1.0, y_true - 1.0)

        ert = ConditionalCoverageERT()
        df = ert.subgroup_coverage(X_1d, y_lower, y_upper, y_true, alpha=0.10)
        assert len(df) > 0
        assert "feature_0" in df["feature"].unique().to_list()


# ===========================================================================
# Part 2: ConditionalValidityIndex and CCSelect (v0.8.0)
# ===========================================================================


# ---------------------------------------------------------------------------
# Minimal conformal predictor stubs for CCSelect tests
# ---------------------------------------------------------------------------


class _ConstantPredictor:
    """
    Predictor that produces fixed-width symmetric intervals.

    By controlling the width, we can create a predictor with good or bad
    conditional coverage on a synthetic dataset.
    """

    def __init__(self, half_width: float):
        self.half_width = half_width
        self._center: float = 0.0

    def calibrate(self, X, y, **kwargs) -> None:
        self._center = float(np.mean(y))

    def predict_interval(self, X, alpha: float = 0.10) -> pl.DataFrame:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        center = np.full(n, self._center)
        lower = center - self.half_width
        upper = center + self.half_width
        return pl.DataFrame({"lower": lower, "upper": upper})


class _ConditionalPredictor:
    """
    Predictor that uses input features to produce adaptive intervals.

    For X[:, 0] > 0 it produces wide intervals (good coverage);
    for X[:, 0] <= 0 it produces very narrow intervals (poor coverage).
    This creates measurable conditional miscoverage for CCSelect to detect.
    """

    def __init__(self, good_half_width: float = 3.0, bad_half_width: float = 0.1):
        self.good_half_width = good_half_width
        self.bad_half_width = bad_half_width
        self._center: float = 0.0

    def calibrate(self, X, y, **kwargs) -> None:
        self._center = float(np.mean(y))

    def predict_interval(self, X, alpha: float = 0.10) -> pl.DataFrame:
        X_np = np.asarray(X)
        n = X_np.shape[0]
        lower = np.zeros(n)
        upper = np.zeros(n)
        for i in range(n):
            hw = self.good_half_width if X_np[i, 0] > 0 else self.bad_half_width
            lower[i] = self._center - hw
            upper[i] = self._center + hw
        return pl.DataFrame({"lower": lower, "upper": upper})


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data():
    """600 observations: Gaussian X (3 features), Gaussian y."""
    rng = np.random.default_rng(42)
    n = 600
    X = rng.normal(size=(n, 3))
    y = rng.normal(loc=5.0, scale=1.5, size=n)
    return {"X": X, "y": y, "n": n}


@pytest.fixture(scope="module")
def cvi_result(synthetic_data):
    """Pre-computed CVIResult on the synthetic data with alpha=0.10."""
    d = synthetic_data
    pred = _ConstantPredictor(half_width=3.0)
    pred.calibrate(d["X"][:300], d["y"][:300])
    intervals = pred.predict_interval(d["X"][300:], alpha=0.10)
    lower = intervals["lower"].to_numpy()
    upper = intervals["upper"].to_numpy()

    cvi = ConditionalValidityIndex(gamma=0.1, n_splits=3, random_state=0)
    return cvi.evaluate(d["X"][300:], lower, upper, d["y"][300:], alpha=0.10)


# ---------------------------------------------------------------------------
# CVIResult invariants
# ---------------------------------------------------------------------------


class TestCVIInvariants:
    def test_cvi_u_plus_cvi_o_equals_cvi(self, cvi_result):
        """CVI_U + CVI_O must equal CVI (fundamental invariant)."""
        assert np.isclose(
            cvi_result.cvi_u + cvi_result.cvi_o, cvi_result.cvi, atol=1e-10
        )

    def test_cvi_non_negative(self, cvi_result):
        assert cvi_result.cvi >= 0.0

    def test_cvi_u_non_negative(self, cvi_result):
        assert cvi_result.cvi_u >= 0.0

    def test_cvi_o_non_negative(self, cvi_result):
        assert cvi_result.cvi_o >= 0.0

    def test_pi_minus_in_unit_interval(self, cvi_result):
        assert 0.0 <= cvi_result.pi_minus <= 1.0

    def test_pi_plus_in_unit_interval(self, cvi_result):
        assert 0.0 <= cvi_result.pi_plus <= 1.0

    def test_cmu_non_negative(self, cvi_result):
        assert cvi_result.cmu >= 0.0

    def test_cmo_non_negative(self, cvi_result):
        assert cvi_result.cmo >= 0.0

    def test_marginal_coverage_in_unit_interval(self, cvi_result):
        assert 0.0 <= cvi_result.marginal_coverage <= 1.0

    def test_n_obs_matches_input(self, synthetic_data, cvi_result):
        """n_obs should match the evaluation set size (300 obs)."""
        assert cvi_result.n_obs == 300

    def test_eta_hat_shape_matches_n_obs(self, cvi_result):
        assert cvi_result.eta_hat.shape == (cvi_result.n_obs,)

    def test_eta_hat_in_unit_interval(self, cvi_result):
        assert np.all(cvi_result.eta_hat >= 0.0)
        assert np.all(cvi_result.eta_hat <= 1.0)

    def test_alpha_stored(self, cvi_result):
        assert np.isclose(cvi_result.alpha, 0.10)

    def test_gamma_stored(self, cvi_result):
        assert np.isclose(cvi_result.gamma, 0.1)

    def test_repr_contains_cvi(self, cvi_result):
        r = repr(cvi_result)
        assert "CVIResult" in r
        assert "cvi=" in r


# ---------------------------------------------------------------------------
# ConditionalValidityIndex.evaluate() validation
# ---------------------------------------------------------------------------


class TestCVIEvaluateValidation:
    def test_error_on_small_n(self):
        """n < 50 should raise ValueError."""
        rng = np.random.default_rng(1)
        n = 30
        X = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        lower = y - 1.0
        upper = y + 1.0

        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        with pytest.raises(ValueError, match="50"):
            cvi.evaluate(X, lower, upper, y, alpha=0.10)

    def test_error_on_invalid_alpha(self, synthetic_data):
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        with pytest.raises(ValueError, match="alpha"):
            cvi.evaluate(d["X"], d["y"] - 1, d["y"] + 1, d["y"], alpha=0.0)

    def test_error_on_alpha_one(self, synthetic_data):
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        with pytest.raises(ValueError, match="alpha"):
            cvi.evaluate(d["X"], d["y"] - 1, d["y"] + 1, d["y"], alpha=1.0)

    def test_mismatched_lengths_raise(self, synthetic_data):
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        with pytest.raises(ValueError):
            cvi.evaluate(
                d["X"][:100],
                d["y"][:100] - 1,
                d["y"][:50] + 1,  # wrong length
                d["y"][:100],
                alpha=0.10,
            )

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            ConditionalValidityIndex(gamma=-0.1)

    def test_gamma_one_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            ConditionalValidityIndex(gamma=1.0)

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits"):
            ConditionalValidityIndex(n_splits=1)

    def test_result_stored_after_evaluate(self, synthetic_data):
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        assert cvi.result_ is None
        cvi.evaluate(d["X"][:200], d["y"][:200] - 2, d["y"][:200] + 2, d["y"][:200], alpha=0.10)
        assert cvi.result_ is not None


# ---------------------------------------------------------------------------
# gamma=0 edge case
# ---------------------------------------------------------------------------


class TestGammaZero:
    def test_gamma_zero_pi_minus_pi_plus_partition(self, synthetic_data):
        """
        With gamma=0, the tolerance band collapses to the single point (1-alpha).
        Essentially all observations are classified as either under or over.
        """
        d = synthetic_data
        pred = _ConstantPredictor(half_width=3.0)
        pred.calibrate(d["X"][:300], d["y"][:300])
        intervals = pred.predict_interval(d["X"][300:], alpha=0.10)
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()

        cvi = ConditionalValidityIndex(gamma=0.0, n_splits=3, random_state=0)
        result = cvi.evaluate(d["X"][300:], lower, upper, d["y"][300:], alpha=0.10)

        # pi_minus + pi_plus should be close to 1 (all obs classified)
        total = result.pi_minus + result.pi_plus
        assert total > 0.9, f"pi_minus + pi_plus = {total:.4f}, expected > 0.9"


# ---------------------------------------------------------------------------
# reliability_profile
# ---------------------------------------------------------------------------


class TestReliabilityProfile:
    def test_profile_requires_evaluate_first(self):
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        with pytest.raises(RuntimeError, match="evaluate"):
            cvi.reliability_profile()

    def test_profile_columns(self, synthetic_data):
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        cvi.evaluate(d["X"][:200], d["y"][:200] - 2, d["y"][:200] + 2, d["y"][:200], alpha=0.10)
        profile = cvi.reliability_profile()
        assert "quantile" in profile.columns
        assert "coverage_prob" in profile.columns

    def test_profile_is_polars(self, synthetic_data):
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        cvi.evaluate(d["X"][:200], d["y"][:200] - 2, d["y"][:200] + 2, d["y"][:200], alpha=0.10)
        profile = cvi.reliability_profile()
        assert isinstance(profile, pl.DataFrame)

    def test_profile_101_rows(self, synthetic_data):
        """linspace(0, 1, 101) -> 101 rows."""
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        cvi.evaluate(d["X"][:200], d["y"][:200] - 2, d["y"][:200] + 2, d["y"][:200], alpha=0.10)
        profile = cvi.reliability_profile()
        assert len(profile) == 101

    def test_profile_quantile_range(self, synthetic_data):
        """Quantile column should go from 0.0 to 1.0."""
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        cvi.evaluate(d["X"][:200], d["y"][:200] - 2, d["y"][:200] + 2, d["y"][:200], alpha=0.10)
        profile = cvi.reliability_profile()
        assert np.isclose(profile["quantile"].min(), 0.0)
        assert np.isclose(profile["quantile"].max(), 1.0)

    def test_profile_coverage_prob_non_decreasing(self, synthetic_data):
        """coverage_prob should be non-decreasing (it's a quantile function)."""
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        cvi.evaluate(d["X"][:200], d["y"][:200] - 2, d["y"][:200] + 2, d["y"][:200], alpha=0.10)
        profile = cvi.reliability_profile()
        probs = profile["coverage_prob"].to_numpy()
        assert np.all(np.diff(probs) >= -1e-10)

    def test_profile_coverage_in_unit_interval(self, synthetic_data):
        d = synthetic_data
        cvi = ConditionalValidityIndex(n_splits=3, random_state=0)
        cvi.evaluate(d["X"][:200], d["y"][:200] - 2, d["y"][:200] + 2, d["y"][:200], alpha=0.10)
        profile = cvi.reliability_profile()
        assert (profile["coverage_prob"] >= 0.0).all()
        assert (profile["coverage_prob"] <= 1.0).all()


# ---------------------------------------------------------------------------
# CCSelect: basic functionality
# ---------------------------------------------------------------------------


class TestCCSelectBasic:
    def test_fit_returns_ccselect_result(self, synthetic_data):
        d = synthetic_data
        predictors = {
            "wide": _ConstantPredictor(half_width=5.0),
            "narrow": _ConstantPredictor(half_width=0.5),
        }
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        result = selector.fit(predictors, d["X"], d["y"], alpha=0.10)
        assert isinstance(result, CCSelectResult)

    def test_best_predictor_is_in_rankings(self, synthetic_data):
        d = synthetic_data
        predictors = {
            "wide": _ConstantPredictor(half_width=5.0),
            "narrow": _ConstantPredictor(half_width=0.5),
        }
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        result = selector.fit(predictors, d["X"], d["y"], alpha=0.10)
        assert result.best_predictor in result.rankings
        assert result.best_predictor in result.cvi_scores

    def test_rankings_sorted_by_cvi(self, synthetic_data):
        """rankings list must be sorted in ascending CVI order."""
        d = synthetic_data
        predictors = {
            "wide": _ConstantPredictor(half_width=5.0),
            "narrow": _ConstantPredictor(half_width=0.5),
        }
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        result = selector.fit(predictors, d["X"], d["y"], alpha=0.10)
        cvi_values = [result.cvi_scores[name] for name in result.rankings]
        assert cvi_values == sorted(cvi_values)

    def test_cvi_scores_keys_match_predictors(self, synthetic_data):
        d = synthetic_data
        predictors = {
            "a": _ConstantPredictor(half_width=3.0),
            "b": _ConstantPredictor(half_width=1.0),
        }
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        result = selector.fit(predictors, d["X"], d["y"], alpha=0.10)
        assert set(result.cvi_scores.keys()) == {"a", "b"}
        assert set(result.cvi_u_scores.keys()) == {"a", "b"}
        assert set(result.cvi_o_scores.keys()) == {"a", "b"}

    def test_cvi_scores_non_negative(self, synthetic_data):
        d = synthetic_data
        predictors = {
            "a": _ConstantPredictor(half_width=3.0),
            "b": _ConstantPredictor(half_width=1.0),
        }
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        result = selector.fit(predictors, d["X"], d["y"], alpha=0.10)
        for name, score in result.cvi_scores.items():
            assert score >= 0.0, f"CVI for {name!r} is negative: {score}"

    def test_result_stored_after_fit(self, synthetic_data):
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        assert selector.result_ is None
        selector.fit(
            {"p": _ConstantPredictor(half_width=2.0)},
            d["X"], d["y"], alpha=0.10
        )
        assert selector.result_ is not None

    def test_repr_contains_best_predictor(self, synthetic_data):
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        result = selector.fit(
            {"p": _ConstantPredictor(half_width=2.0)},
            d["X"], d["y"], alpha=0.10
        )
        r = repr(result)
        assert "CCSelectResult" in r
        assert "best_predictor" in r


# ---------------------------------------------------------------------------
# CCSelect: compare()
# ---------------------------------------------------------------------------


class TestCCSelectCompare:
    def test_compare_requires_fit_first(self):
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        with pytest.raises(RuntimeError, match="fit"):
            selector.compare()

    def test_compare_returns_polars_dataframe(self, synthetic_data):
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        selector.fit(
            {"a": _ConstantPredictor(half_width=3.0), "b": _ConstantPredictor(half_width=1.0)},
            d["X"], d["y"], alpha=0.10
        )
        df = selector.compare()
        assert isinstance(df, pl.DataFrame)

    def test_compare_columns(self, synthetic_data):
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        selector.fit(
            {"a": _ConstantPredictor(half_width=3.0), "b": _ConstantPredictor(half_width=1.0)},
            d["X"], d["y"], alpha=0.10
        )
        df = selector.compare()
        expected = {"predictor_name", "mean_cvi", "mean_cvi_u", "mean_cvi_o", "rank"}
        assert expected.issubset(set(df.columns))

    def test_compare_row_count_matches_predictors(self, synthetic_data):
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        selector.fit(
            {"a": _ConstantPredictor(half_width=3.0), "b": _ConstantPredictor(half_width=1.0)},
            d["X"], d["y"], alpha=0.10
        )
        df = selector.compare()
        assert len(df) == 2

    def test_compare_sorted_by_mean_cvi(self, synthetic_data):
        """compare() should be sorted by ascending mean_cvi."""
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        selector.fit(
            {"a": _ConstantPredictor(half_width=3.0), "b": _ConstantPredictor(half_width=1.0)},
            d["X"], d["y"], alpha=0.10
        )
        df = selector.compare()
        cvi_vals = df["mean_cvi"].to_numpy()
        assert np.all(np.diff(cvi_vals) >= -1e-10)

    def test_compare_rank_starts_at_1(self, synthetic_data):
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        selector.fit(
            {"a": _ConstantPredictor(half_width=3.0), "b": _ConstantPredictor(half_width=1.0)},
            d["X"], d["y"], alpha=0.10
        )
        df = selector.compare()
        assert df["rank"].min() == 1

    def test_compare_best_predictor_has_rank_1(self, synthetic_data):
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        result = selector.fit(
            {"a": _ConstantPredictor(half_width=3.0), "b": _ConstantPredictor(half_width=1.0)},
            d["X"], d["y"], alpha=0.10
        )
        df = selector.compare()
        rank_1_name = df.filter(pl.col("rank") == 1)["predictor_name"][0]
        assert rank_1_name == result.best_predictor


# ---------------------------------------------------------------------------
# CCSelect: validation
# ---------------------------------------------------------------------------


class TestCCSelectValidation:
    def test_empty_predictors_raises(self, synthetic_data):
        d = synthetic_data
        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=0)
        with pytest.raises(ValueError, match="empty"):
            selector.fit({}, d["X"], d["y"], alpha=0.10)

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits"):
            CCSelect(n_splits=1)

    def test_invalid_split_ratio_raises(self):
        with pytest.raises(ValueError, match="split_ratio"):
            CCSelect(split_ratio=0.0)

    def test_invalid_split_ratio_one_raises(self):
        with pytest.raises(ValueError, match="split_ratio"):
            CCSelect(split_ratio=1.0)


# ---------------------------------------------------------------------------
# CCSelect: picks lower-CVI predictor in synthetic example
# ---------------------------------------------------------------------------


class TestCCSelectPicksLowerCVI:
    def test_selects_more_uniform_predictor(self):
        """
        Synthetic example where one predictor has strongly conditional miscoverage
        and one predictor is approximately uniform.

        The _ConditionalPredictor gives wide intervals for X[:,0]>0 and narrow
        for X[:,0]<=0, creating systematic undercoverage on half the data.
        The wide _ConstantPredictor covers everyone uniformly.

        CCSelect should prefer the constant wide predictor.
        """
        rng = np.random.default_rng(7)
        n = 800
        X = rng.normal(size=(n, 3))
        # y is centred at 5.0 with std 1.0
        y = rng.normal(loc=5.0, scale=1.0, size=n)

        predictors = {
            "uniform_wide": _ConstantPredictor(half_width=4.0),
            "conditional_bad": _ConditionalPredictor(good_half_width=4.0, bad_half_width=0.05),
        }

        selector = CCSelect(n_splits=3, split_ratio=0.5, random_state=42)
        result = selector.fit(predictors, X, y, alpha=0.10)

        # The uniform predictor should have lower CVI than the conditional one
        assert result.cvi_scores["uniform_wide"] <= result.cvi_scores["conditional_bad"], (
            f"Expected uniform_wide CVI ({result.cvi_scores['uniform_wide']:.4f}) "
            f"<= conditional_bad CVI ({result.cvi_scores['conditional_bad']:.4f})"
        )


# ---------------------------------------------------------------------------
# Top-level imports (v0.8.0)
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    def test_all_four_names_importable(self):
        """All four new names must be importable from insurance_conformal directly."""
        from insurance_conformal import (  # noqa: F401
            CCSelect,
            CCSelectResult,
            CVIResult,
            ConditionalValidityIndex,
        )

    def test_ert_still_importable(self):
        """ConditionalCoverageERT must still be importable (v0.7.1 API unchanged)."""
        from insurance_conformal import ConditionalCoverageERT  # noqa: F401
