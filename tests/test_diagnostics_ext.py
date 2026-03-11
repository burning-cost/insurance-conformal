"""
Tests for diagnostics_ext: ert_coverage_gap, subgroup_coverage,
width_efficiency_comparison.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.diagnostics_ext import (
    ert_coverage_gap,
    subgroup_coverage,
    width_efficiency_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LinearPoissonModel:
    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        return np.exp(np.asarray(X) @ self.beta)


@pytest.fixture(scope="module")
def diag_data():
    rng = np.random.default_rng(55)
    n = 1200
    beta = np.array([0.5, -0.25, 0.1])
    X = rng.normal(size=(n, 3))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    model = LinearPoissonModel(beta)

    return {
        "X_cal": X[:400],
        "y_cal": y[:400],
        "X_test": X[400:800],
        "y_test": y[400:800],
        "X_test2": X[800:],
        "y_test2": y[800:],
        "model": model,
    }


@pytest.fixture(scope="module")
def cp_std(diag_data):
    cp = InsuranceConformalPredictor(
        model=diag_data["model"],
        nonconformity="pearson_weighted",
        tweedie_power=1.0,
    )
    cp.calibrate(diag_data["X_cal"], diag_data["y_cal"])
    return cp


@pytest.fixture(scope="module")
def cp_raw(diag_data):
    cp = InsuranceConformalPredictor(
        model=diag_data["model"],
        nonconformity="raw",
        tweedie_power=1.0,
    )
    cp.calibrate(diag_data["X_cal"], diag_data["y_cal"])
    return cp


# ---------------------------------------------------------------------------
# ert_coverage_gap
# ---------------------------------------------------------------------------


class TestErtCoverageGap:
    def test_output_columns(self, cp_std, diag_data):
        df = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=0.10
        )
        expected = {
            "bin", "mean_predicted", "n_obs", "empirical_coverage",
            "target_coverage", "coverage_gap", "coverage_se", "exceeds_2se"
        }
        assert expected.issubset(set(df.columns))

    def test_output_is_polars(self, cp_std, diag_data):
        df = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=0.10
        )
        assert isinstance(df, pl.DataFrame)

    def test_has_summary_row(self, cp_std, diag_data):
        df = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=0.10
        )
        assert -1 in df["bin"].to_list()

    def test_n_bins_affects_row_count(self, cp_std, diag_data):
        df5 = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=0.10, n_bins=5
        )
        df10 = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=0.10, n_bins=10
        )
        # +1 for summary row
        assert len(df5) == 6
        assert len(df10) == 11

    def test_coverage_in_range(self, cp_std, diag_data):
        df = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=0.10
        )
        # Exclude summary row (bin=-1)
        df_bins = df.filter(pl.col("bin") != -1)
        assert (df_bins["empirical_coverage"] >= 0).all()
        assert (df_bins["empirical_coverage"] <= 1).all()

    def test_target_coverage_correct(self, cp_std, diag_data):
        alpha = 0.15
        df = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=alpha
        )
        df_bins = df.filter(pl.col("bin") != -1)
        assert np.allclose(df_bins["target_coverage"].to_numpy(), 1 - alpha)

    def test_exceeds_2se_is_bool(self, cp_std, diag_data):
        df = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=0.10
        )
        # Should be boolean column
        assert df["exceeds_2se"].dtype == pl.Boolean

    def test_coverage_gap_sign(self, cp_std, diag_data):
        """coverage_gap = target - empirical; positive means undercoverage."""
        df = ert_coverage_gap(
            cp_std, diag_data["X_test"], diag_data["y_test"], alpha=0.10
        )
        df_bins = df.filter(pl.col("bin") != -1)
        target = df_bins["target_coverage"].to_numpy()
        empirical = df_bins["empirical_coverage"].to_numpy()
        gap = df_bins["coverage_gap"].to_numpy()
        assert np.allclose(gap, target - empirical)


# ---------------------------------------------------------------------------
# subgroup_coverage
# ---------------------------------------------------------------------------


class TestSubgroupCoverage:
    def test_output_columns(self, cp_std, diag_data):
        groups = np.random.choice(["A", "B", "C"], size=len(diag_data["y_test"]))
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=0.10, groups=groups, group_name="segment"
        )
        expected = {
            "segment", "n_obs", "empirical_coverage", "target_coverage",
            "coverage_gap", "mean_lower", "mean_upper", "mean_width", "mean_predicted"
        }
        assert expected.issubset(set(df.columns))

    def test_output_is_polars(self, cp_std, diag_data):
        groups = np.ones(len(diag_data["y_test"]), dtype=int)
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=0.10, groups=groups
        )
        assert isinstance(df, pl.DataFrame)

    def test_all_groups_present(self, cp_std, diag_data):
        groups = np.array(["motor"] * 200 + ["property"] * 200)
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=0.10, groups=groups
        )
        assert set(df["group"].to_list()) == {"motor", "property"}

    def test_n_obs_sums_to_total(self, cp_std, diag_data):
        groups = np.random.choice(["A", "B"], size=len(diag_data["y_test"]))
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=0.10, groups=groups
        )
        assert df["n_obs"].sum() == len(diag_data["y_test"])

    def test_mismatched_groups_raises(self, cp_std, diag_data):
        groups = np.ones(10)  # wrong length
        with pytest.raises(ValueError, match="groups length"):
            subgroup_coverage(
                cp_std, diag_data["X_test"], diag_data["y_test"],
                alpha=0.10, groups=groups
            )

    def test_coverage_in_range(self, cp_std, diag_data):
        groups = np.random.choice(["A", "B", "C"], size=len(diag_data["y_test"]))
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=0.10, groups=groups
        )
        assert (df["empirical_coverage"] >= 0).all()
        assert (df["empirical_coverage"] <= 1).all()

    def test_mean_width_positive(self, cp_std, diag_data):
        groups = np.random.choice([1, 2], size=len(diag_data["y_test"]))
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=0.10, groups=groups
        )
        assert (df["mean_width"] > 0).all()

    def test_single_group(self, cp_std, diag_data):
        groups = np.ones(len(diag_data["y_test"]), dtype=int)
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=0.10, groups=groups
        )
        assert len(df) == 1

    def test_target_coverage_column(self, cp_std, diag_data):
        groups = np.random.choice(["A", "B"], size=len(diag_data["y_test"]))
        alpha = 0.12
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=alpha, groups=groups
        )
        assert np.allclose(df["target_coverage"].to_numpy(), 1 - alpha)

    def test_default_group_column_name(self, cp_std, diag_data):
        groups = np.ones(len(diag_data["y_test"]))
        df = subgroup_coverage(
            cp_std, diag_data["X_test"], diag_data["y_test"],
            alpha=0.10, groups=groups
        )
        assert "group" in df.columns


# ---------------------------------------------------------------------------
# width_efficiency_comparison
# ---------------------------------------------------------------------------


class TestWidthEfficiencyComparison:
    def test_output_columns(self, cp_std, cp_raw, diag_data):
        result = width_efficiency_comparison(
            predictors={"standard": cp_std, "raw": cp_raw},
            X_test=diag_data["X_test"],
            y_test=diag_data["y_test"],
            alpha=0.10,
        )
        expected = {
            "predictor_name", "marginal_coverage", "target_coverage",
            "mean_width", "median_width", "width_90th_pct",
            "width_relative_to_widest"
        }
        assert expected.issubset(set(result.columns))

    def test_output_is_polars(self, cp_std, cp_raw, diag_data):
        result = width_efficiency_comparison(
            predictors={"standard": cp_std, "raw": cp_raw},
            X_test=diag_data["X_test"],
            y_test=diag_data["y_test"],
        )
        assert isinstance(result, pl.DataFrame)

    def test_row_count_matches_predictors(self, cp_std, cp_raw, diag_data):
        result = width_efficiency_comparison(
            predictors={"p1": cp_std, "p2": cp_raw},
            X_test=diag_data["X_test"],
            y_test=diag_data["y_test"],
        )
        assert len(result) == 2

    def test_sorted_by_mean_width(self, cp_std, cp_raw, diag_data):
        result = width_efficiency_comparison(
            predictors={"standard": cp_std, "raw": cp_raw},
            X_test=diag_data["X_test"],
            y_test=diag_data["y_test"],
        )
        widths = result["mean_width"].to_numpy()
        assert np.all(np.diff(widths) >= 0)

    def test_relative_width_leq_1(self, cp_std, cp_raw, diag_data):
        result = width_efficiency_comparison(
            predictors={"standard": cp_std, "raw": cp_raw},
            X_test=diag_data["X_test"],
            y_test=diag_data["y_test"],
        )
        assert (result["width_relative_to_widest"] <= 1.0).all()
        # Widest predictor should have ratio = 1.0
        assert np.isclose(result["width_relative_to_widest"].max(), 1.0)

    def test_all_predictor_names_present(self, cp_std, cp_raw, diag_data):
        result = width_efficiency_comparison(
            predictors={"pearson_weighted": cp_std, "raw": cp_raw},
            X_test=diag_data["X_test"],
            y_test=diag_data["y_test"],
        )
        assert set(result["predictor_name"].to_list()) == {"pearson_weighted", "raw"}

    def test_single_predictor(self, cp_std, diag_data):
        result = width_efficiency_comparison(
            predictors={"only": cp_std},
            X_test=diag_data["X_test"],
            y_test=diag_data["y_test"],
        )
        assert len(result) == 1
        assert np.isclose(result["width_relative_to_widest"][0], 1.0)

    def test_target_coverage_correct(self, cp_std, diag_data):
        alpha = 0.15
        result = width_efficiency_comparison(
            predictors={"p": cp_std},
            X_test=diag_data["X_test"],
            y_test=diag_data["y_test"],
            alpha=alpha,
        )
        assert np.isclose(result["target_coverage"][0], 1 - alpha)
