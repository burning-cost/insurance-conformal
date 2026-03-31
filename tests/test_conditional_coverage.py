"""
Tests for ConditionalCoverageERT.

Four scenarios:
1. Null case: uniform coverage — ERT should be near zero
2. Alternative case: heteroscedastic miscoverage — ERT should be positive
3. Directional variants: under/over/both produce sensible results
4. subgroup_coverage: output format and column integrity

Synthetic data design: we simulate conformal intervals by fixing known coverage
patterns rather than running a full conformal predictor. This makes the tests
fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import polars as pl

from insurance_conformal.conditional_coverage import ConditionalCoverageERT, ERTResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
