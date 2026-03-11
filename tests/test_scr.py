"""
Tests for SCRReport.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.scr import SCRReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LinearPoissonModel:
    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        return np.exp(np.asarray(X) @ self.beta)


@pytest.fixture(scope="module")
def scr_data():
    rng = np.random.default_rng(123)
    n = 1200
    beta = np.array([0.4, -0.2, 0.1])
    X = rng.normal(size=(n, 3))
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    model = LinearPoissonModel(beta)

    return {
        "X_cal": X[:400],
        "y_cal": y[:400],
        "X_test": X[400:800],
        "y_test": y[400:800],
        "model": model,
    }


@pytest.fixture(scope="module")
def calibrated_predictor(scr_data):
    cp = InsuranceConformalPredictor(
        model=scr_data["model"],
        nonconformity="pearson_weighted",
        tweedie_power=1.0,
    )
    cp.calibrate(scr_data["X_cal"], scr_data["y_cal"])
    return cp


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSCRReportConstruction:
    def test_from_conformal(self, calibrated_predictor):
        scr = SCRReport.from_conformal(calibrated_predictor)
        assert scr.predictor is calibrated_predictor

    def test_direct_constructor(self, calibrated_predictor):
        scr = SCRReport(predictor=calibrated_predictor)
        assert scr.predictor is calibrated_predictor

    def test_rejects_non_predictor(self):
        with pytest.raises(TypeError, match="predict_interval"):
            SCRReport(predictor="not_a_predictor")

    def test_with_policy_ids(self, calibrated_predictor):
        ids = np.arange(100)
        scr = SCRReport(predictor=calibrated_predictor, policy_ids=ids)
        assert scr._policy_ids is not None

    def test_repr(self, calibrated_predictor):
        scr = SCRReport(predictor=calibrated_predictor)
        r = repr(scr)
        assert "SCRReport" in r


# ---------------------------------------------------------------------------
# solvency_capital_requirement
# ---------------------------------------------------------------------------


class TestSCR:
    def test_output_columns_default_alpha(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        result = scr.solvency_capital_requirement(scr_data["X_test"])
        assert isinstance(result, pl.DataFrame)
        assert "expected_loss" in result.columns
        assert "scr_component" in result.columns
        assert "coverage_level" in result.columns
        assert len(result) == len(scr_data["X_test"])

    def test_coverage_level_column(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        result = scr.solvency_capital_requirement(scr_data["X_test"], alpha=0.005)
        assert np.allclose(result["coverage_level"].to_numpy(), 0.995)

    def test_scr_component_nonnegative(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        result = scr.solvency_capital_requirement(scr_data["X_test"])
        assert (result["scr_component"] >= 0).all()

    def test_scr_component_equals_upper_minus_expected(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        result = scr.solvency_capital_requirement(scr_data["X_test"], alpha=0.005)
        # Find the upper bound column
        upper_col = [c for c in result.columns if "pct" in c][0]
        expected = result["expected_loss"].to_numpy()
        upper = result[upper_col].to_numpy()
        computed_scr = np.maximum(0.0, upper - expected)
        assert np.allclose(result["scr_component"].to_numpy(), computed_scr)

    def test_with_policy_ids(self, calibrated_predictor, scr_data):
        n = len(scr_data["X_test"])
        ids = np.arange(n)
        scr = SCRReport(predictor=calibrated_predictor, policy_ids=ids)
        result = scr.solvency_capital_requirement(scr_data["X_test"])
        assert "policy_id" in result.columns

    def test_mismatched_policy_ids_raises(self, calibrated_predictor, scr_data):
        ids = np.arange(10)  # wrong length
        scr = SCRReport(predictor=calibrated_predictor, policy_ids=ids)
        with pytest.raises(ValueError, match="policy_ids length"):
            scr.solvency_capital_requirement(scr_data["X_test"])

    def test_invalid_alpha(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        with pytest.raises(ValueError, match="alpha"):
            scr.solvency_capital_requirement(scr_data["X_test"], alpha=1.5)

    def test_upper_bound_column_name_reflects_alpha(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        result_005 = scr.solvency_capital_requirement(scr_data["X_test"], alpha=0.005)
        result_01 = scr.solvency_capital_requirement(scr_data["X_test"], alpha=0.01)
        upper_cols_005 = [c for c in result_005.columns if "99.5" in c]
        upper_cols_01 = [c for c in result_01.columns if "99.0" in c]
        assert len(upper_cols_005) == 1
        assert len(upper_cols_01) == 1


# ---------------------------------------------------------------------------
# coverage_validation_table
# ---------------------------------------------------------------------------


class TestCoverageValidationTable:
    def test_output_columns(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        table = scr.coverage_validation_table(scr_data["X_test"], scr_data["y_test"])
        expected_cols = {
            "alpha", "target_coverage", "empirical_coverage",
            "n_covered", "n_total", "coverage_shortfall", "meets_requirement"
        }
        assert expected_cols.issubset(set(table.columns))

    def test_default_alphas(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        table = scr.coverage_validation_table(scr_data["X_test"], scr_data["y_test"])
        alphas = table["alpha"].to_list()
        assert 0.005 in alphas
        assert 0.05 in alphas

    def test_custom_alphas(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        table = scr.coverage_validation_table(
            scr_data["X_test"], scr_data["y_test"], alphas=[0.10, 0.20]
        )
        assert len(table) == 2
        assert set(table["alpha"].to_list()) == {0.10, 0.20}

    def test_n_total_consistent(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        table = scr.coverage_validation_table(scr_data["X_test"], scr_data["y_test"])
        n = len(scr_data["y_test"])
        assert (table["n_total"] == n).all()

    def test_coverage_in_range(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        table = scr.coverage_validation_table(scr_data["X_test"], scr_data["y_test"])
        assert (table["empirical_coverage"] >= 0).all()
        assert (table["empirical_coverage"] <= 1).all()

    def test_target_coverage_correct(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        table = scr.coverage_validation_table(
            scr_data["X_test"], scr_data["y_test"], alphas=[0.10]
        )
        assert np.isclose(table["target_coverage"][0], 0.90)

    def test_sorted_by_alpha(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        table = scr.coverage_validation_table(scr_data["X_test"], scr_data["y_test"])
        alphas = table["alpha"].to_numpy()
        assert np.all(np.diff(alphas) >= 0)


# ---------------------------------------------------------------------------
# to_markdown and aggregate_scr
# ---------------------------------------------------------------------------


class TestMarkdownAndAggregate:
    def test_to_markdown_before_validation_raises(self, calibrated_predictor):
        scr = SCRReport(predictor=calibrated_predictor)
        with pytest.raises(RuntimeError, match="coverage_validation_table"):
            scr.to_markdown()

    def test_to_markdown_returns_string(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        scr.coverage_validation_table(scr_data["X_test"], scr_data["y_test"])
        md = scr.to_markdown()
        assert isinstance(md, str)
        assert "SCR Coverage Validation" in md
        assert "| Alpha |" in md

    def test_markdown_contains_all_alphas(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        scr.coverage_validation_table(
            scr_data["X_test"], scr_data["y_test"], alphas=[0.005, 0.01, 0.05]
        )
        md = scr.to_markdown()
        assert "0.005" in md
        assert "0.010" in md
        assert "0.050" in md

    def test_aggregate_scr_before_calculation_raises(self, calibrated_predictor):
        scr = SCRReport(predictor=calibrated_predictor)
        with pytest.raises(RuntimeError, match="solvency_capital_requirement"):
            scr.aggregate_scr()

    def test_aggregate_scr_keys(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        scr.solvency_capital_requirement(scr_data["X_test"], alpha=0.005)
        agg = scr.aggregate_scr()
        assert "total_expected_loss" in agg
        assert "total_scr" in agg
        assert "scr_ratio" in agg
        assert "n_risks" in agg
        assert abs(agg["alpha"] - 0.005) < 1e-10

    def test_aggregate_scr_ratio_positive(self, calibrated_predictor, scr_data):
        scr = SCRReport(predictor=calibrated_predictor)
        scr.solvency_capital_requirement(scr_data["X_test"])
        agg = scr.aggregate_scr()
        assert agg["scr_ratio"] >= 0
        assert agg["total_scr"] >= 0
