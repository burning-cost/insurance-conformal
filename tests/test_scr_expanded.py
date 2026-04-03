"""
Expanded tests for SCRReport: coverage_validation_table, to_markdown,
policy_ids, error handling, and edge cases.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.scr import SCRReport


class LinearModel:
    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        return np.exp(np.asarray(X) @ self.beta)


@pytest.fixture(scope="module")
def scr_setup():
    rng = np.random.default_rng(77)
    n = 1200
    beta = np.array([0.3, -0.1, 0.05])
    X = rng.normal(size=(n, 3))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    model = LinearModel(beta)
    cp = InsuranceConformalPredictor(model=model, nonconformity="pearson_weighted", tweedie_power=1.0)
    cp.calibrate(X[:400], y[:400])
    return {
        "cp": cp,
        "X_test": X[400:800],
        "y_test": y[400:800],
        "n_test": 400,
    }


@pytest.fixture(scope="module")
def scr_report(scr_setup):
    return SCRReport(predictor=scr_setup["cp"])


# ---------------------------------------------------------------------------
# coverage_validation_table
# ---------------------------------------------------------------------------


class TestCoverageValidationTable:
    def test_returns_polars_dataframe(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"]
        )
        assert isinstance(result, pl.DataFrame)

    def test_required_columns(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"]
        )
        for col in ["alpha", "target_coverage", "empirical_coverage",
                    "n_covered", "n_total", "coverage_shortfall", "meets_requirement"]:
            assert col in result.columns

    def test_default_5_alpha_levels(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"]
        )
        assert len(result) == 5

    def test_custom_alphas(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"],
            alphas=[0.10, 0.20]
        )
        assert len(result) == 2

    def test_n_total_consistent(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"]
        )
        assert (result["n_total"] == scr_setup["n_test"]).all()

    def test_coverage_in_unit_interval(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"]
        )
        assert (result["empirical_coverage"] >= 0.0).all()
        assert (result["empirical_coverage"] <= 1.0).all()

    def test_target_coverage_matches_alpha(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"],
            alphas=[0.10, 0.20]
        )
        for row in result.iter_rows(named=True):
            assert row["target_coverage"] == pytest.approx(1.0 - row["alpha"])

    def test_n_covered_leq_n_total(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"]
        )
        assert (result["n_covered"] <= result["n_total"]).all()

    def test_coverage_shortfall_nonneg(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"]
        )
        assert (result["coverage_shortfall"] >= 0.0).all()

    def test_ordered_by_alpha(self, scr_report, scr_setup):
        result = scr_report.coverage_validation_table(
            scr_setup["X_test"], scr_setup["y_test"]
        )
        alphas = result["alpha"].to_numpy()
        assert np.all(np.diff(alphas) >= 0)

    def test_stores_last_table(self, scr_report, scr_setup):
        scr_report.coverage_validation_table(scr_setup["X_test"], scr_setup["y_test"])
        assert scr_report._last_coverage_table is not None


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------


class TestToMarkdown:
    def test_to_markdown_before_table_raises(self, scr_setup):
        fresh_report = SCRReport(predictor=scr_setup["cp"])
        with pytest.raises(RuntimeError, match="coverage_validation_table"):
            fresh_report.to_markdown()

    def test_to_markdown_returns_string(self, scr_report, scr_setup):
        scr_report.coverage_validation_table(scr_setup["X_test"], scr_setup["y_test"])
        md = scr_report.to_markdown()
        assert isinstance(md, str)

    def test_to_markdown_contains_header(self, scr_report, scr_setup):
        scr_report.coverage_validation_table(scr_setup["X_test"], scr_setup["y_test"])
        md = scr_report.to_markdown()
        assert "SCR" in md or "Coverage" in md

    def test_to_markdown_contains_alpha(self, scr_report, scr_setup):
        scr_report.coverage_validation_table(scr_setup["X_test"], scr_setup["y_test"])
        md = scr_report.to_markdown()
        assert "0.10" in md or "alpha" in md.lower()

    def test_to_markdown_nonempty(self, scr_report, scr_setup):
        scr_report.coverage_validation_table(scr_setup["X_test"], scr_setup["y_test"])
        md = scr_report.to_markdown()
        assert len(md) > 100


# ---------------------------------------------------------------------------
# SCRReport with policy_ids
# ---------------------------------------------------------------------------


class TestSCRReportWithPolicyIds:
    def test_policy_ids_in_output(self, scr_setup):
        ids = np.arange(scr_setup["n_test"])
        report = SCRReport(predictor=scr_setup["cp"], policy_ids=ids)
        result = report.solvency_capital_requirement(scr_setup["X_test"])
        assert "policy_id" in result.columns

    def test_policy_ids_length_mismatch_raises(self, scr_setup):
        ids = np.arange(10)  # wrong length
        report = SCRReport(predictor=scr_setup["cp"], policy_ids=ids)
        with pytest.raises(ValueError, match="length"):
            report.solvency_capital_requirement(scr_setup["X_test"])

    def test_without_policy_ids_no_policy_column(self, scr_setup):
        report = SCRReport(predictor=scr_setup["cp"])
        result = report.solvency_capital_requirement(scr_setup["X_test"])
        assert "policy_id" not in result.columns

    def test_string_policy_ids(self, scr_setup):
        ids = [f"POL{i:04d}" for i in range(scr_setup["n_test"])]
        report = SCRReport(predictor=scr_setup["cp"], policy_ids=ids)
        result = report.solvency_capital_requirement(scr_setup["X_test"])
        assert "policy_id" in result.columns
        assert len(result) == scr_setup["n_test"]


# ---------------------------------------------------------------------------
# SCR: alpha variants
# ---------------------------------------------------------------------------


class TestSCRAlphaVariants:
    def test_alpha_001(self, scr_report, scr_setup):
        """99.9% coverage — very wide intervals."""
        result = scr_report.solvency_capital_requirement(scr_setup["X_test"], alpha=0.001)
        assert isinstance(result, pl.DataFrame)
        assert "scr_component" in result.columns

    def test_alpha_01(self, scr_report, scr_setup):
        """99% coverage — Lloyd's stress test level."""
        result = scr_report.solvency_capital_requirement(scr_setup["X_test"], alpha=0.01)
        assert isinstance(result, pl.DataFrame)

    def test_wider_at_lower_alpha(self, scr_report, scr_setup):
        """Lower alpha = higher coverage requirement = wider SCR intervals."""
        r_strict = scr_report.solvency_capital_requirement(scr_setup["X_test"], alpha=0.005)
        r_loose = scr_report.solvency_capital_requirement(scr_setup["X_test"], alpha=0.20)
        # Strict should have larger SCR components on average
        assert r_strict["scr_component"].mean() >= r_loose["scr_component"].mean()

    def test_invalid_alpha_zero_raises(self, scr_report, scr_setup):
        with pytest.raises(ValueError):
            scr_report.solvency_capital_requirement(scr_setup["X_test"], alpha=0.0)

    def test_invalid_alpha_above_1_raises(self, scr_report, scr_setup):
        with pytest.raises(ValueError):
            scr_report.solvency_capital_requirement(scr_setup["X_test"], alpha=1.5)


# ---------------------------------------------------------------------------
# SCR: output column name with different alphas
# ---------------------------------------------------------------------------


class TestSCROutputColumnName:
    def test_column_name_at_default_alpha(self, scr_report, scr_setup):
        result = scr_report.solvency_capital_requirement(scr_setup["X_test"])
        # Default alpha=0.005 → 99.5% column
        upper_col = [c for c in result.columns if "upper" in c.lower()]
        assert len(upper_col) == 1
        assert "99.5" in upper_col[0] or "995" in upper_col[0]

    def test_column_name_at_custom_alpha(self, scr_report, scr_setup):
        result = scr_report.solvency_capital_requirement(scr_setup["X_test"], alpha=0.10)
        upper_col = [c for c in result.columns if "upper" in c.lower()]
        assert len(upper_col) == 1
        # 90.0% column
        assert "90.0" in upper_col[0] or "900" in upper_col[0]
