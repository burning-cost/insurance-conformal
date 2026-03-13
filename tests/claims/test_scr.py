"""
Tests for scr.py — SCRReport.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from insurance_conformal.claims.hong import HongConformal
from insurance_conformal.claims.scr import SCRReport


def make_tweedie_data(n=1000, p=5, seed=42, p_tweedie=1.5):
    """Generate synthetic Tweedie compound Poisson-Gamma data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 0] = np.abs(X[:, 0])
    beta = rng.normal(size=p) * 0.3
    mu = np.exp(1.5 + X @ beta)
    alpha_g = (2.0 - p_tweedie) / (p_tweedie - 1.0)
    y = np.zeros(n)
    for i in range(n):
        lam = mu[i] ** (2.0 - p_tweedie) / (2.0 - p_tweedie)
        k = rng.poisson(lam)
        if k > 0:
            y[i] = rng.gamma(alpha_g * k, mu[i] ** (p_tweedie - 1.0) * (p_tweedie - 1.0))
    return X, y


@pytest.fixture(scope="module")
def fitted_hong_and_data():
    X_all, y_all = make_tweedie_data(n=400, seed=20)
    X_train, y_train = X_all[:300], y_all[:300]
    X_test, y_test = X_all[300:], y_all[300:]
    hc = HongConformal().fit(X_train, y_train)
    return hc, X_test, y_test


class TestSCRReportAPI:
    def test_init_accepts_predictor(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        assert report.predictor is hc

    def test_init_rejects_non_predictor(self):
        with pytest.raises(TypeError, match="predict_interval"):
            SCRReport("not a predictor")

    def test_repr(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        assert "SCRReport" in repr(report)
        assert "HongConformal" in repr(report)

    def test_scr_returns_float(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        scr = report.solvency_capital_requirement(X_test, alpha=0.005)
        assert isinstance(scr, float)
        assert scr > 0.0

    def test_scr_is_max_upper_bound(self, fitted_hong_and_data):
        """SCR should equal max of upper prediction bounds."""
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        scr = report.solvency_capital_requirement(X_test, alpha=0.005)
        intervals = hc.predict_interval(X_test, alpha=0.005)
        assert scr == pytest.approx(float(intervals[:, 1].max()))

    def test_scr_increases_with_tighter_coverage(self, fitted_hong_and_data):
        """SCR at 99.5% should be >= SCR at 95%."""
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        scr_995 = report.solvency_capital_requirement(X_test, alpha=0.005)
        scr_95 = report.solvency_capital_requirement(X_test, alpha=0.05)
        assert scr_995 >= scr_95

    def test_aggregate_reserve(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        agg = report.aggregate_reserve(X_test, alpha=0.05)
        intervals = hc.predict_interval(X_test, alpha=0.05)
        expected = float(intervals[:, 1].sum())
        assert agg == pytest.approx(expected)

    def test_aggregate_reserve_exceeds_scr(self, fitted_hong_and_data):
        """Aggregate reserve (sum) > portfolio SCR (max) for n > 1."""
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        agg = report.aggregate_reserve(X_test, alpha=0.05)
        scr = report.solvency_capital_requirement(X_test, alpha=0.05)
        assert agg > scr  # n=100 > 1


class TestSCRCoverageTable:
    def test_coverage_table_returns_dataframe(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        df = report.coverage_table(X_test, y_test)
        assert isinstance(df, pd.DataFrame)

    def test_coverage_table_columns(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        df = report.coverage_table(X_test, y_test)
        required_cols = {
            "alpha", "nominal_coverage", "empirical_coverage",
            "mean_width", "n_test"
        }
        assert required_cols.issubset(set(df.columns))

    def test_coverage_table_has_correct_n_rows(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        alphas = [0.005, 0.01, 0.05]
        df = report.coverage_table(X_test, y_test, alphas=alphas)
        assert len(df) == 3

    def test_coverage_table_alphas_correct(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        alphas = [0.005, 0.10]
        df = report.coverage_table(X_test, y_test, alphas=alphas)
        np.testing.assert_allclose(df["alpha"].values, alphas)

    def test_nominal_coverage_equals_1_minus_alpha(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        df = report.coverage_table(X_test, y_test)
        np.testing.assert_allclose(
            df["nominal_coverage"].values,
            1.0 - df["alpha"].values,
            atol=1e-12
        )

    def test_empirical_coverage_in_valid_range(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        df = report.coverage_table(X_test, y_test)
        assert np.all(df["empirical_coverage"] >= 0.0)
        assert np.all(df["empirical_coverage"] <= 1.0)

    def test_mean_width_positive(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        df = report.coverage_table(X_test, y_test)
        assert np.all(df["mean_width"] > 0.0)

    def test_width_increases_with_coverage(self, fitted_hong_and_data):
        """Wider intervals = higher coverage."""
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        df = report.coverage_table(X_test, y_test, alphas=[0.10, 0.05, 0.01])
        # As nominal_coverage increases, mean_width should increase
        widths = df.sort_values("nominal_coverage")["mean_width"].values
        assert np.all(np.diff(widths) >= 0)

    def test_conformal_coverage_guarantee(self, fitted_hong_and_data):
        """Empirical coverage should be >= nominal for a valid conformal predictor."""
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        df = report.coverage_table(X_test, y_test, alphas=[0.10, 0.05, 0.20])
        for _, row in df.iterrows():
            # Allow 5pp slack for 100 test samples (finite sample variation)
            assert row["empirical_coverage"] >= row["nominal_coverage"] - 0.05, (
                f"Coverage below nominal at alpha={row['alpha']}: "
                f"emp={row['empirical_coverage']:.3f}, nom={row['nominal_coverage']:.3f}"
            )


class TestSCRHTMLOutput:
    def test_to_html_returns_string(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        html = report.to_html(X_test, y_test)
        assert isinstance(html, str)

    def test_to_html_contains_doctype(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        html = report.to_html(X_test, y_test)
        assert "<!DOCTYPE html>" in html

    def test_to_html_contains_table(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        html = report.to_html(X_test, y_test)
        assert "<table>" in html
        assert "</table>" in html

    def test_to_html_contains_predictor_name(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        html = report.to_html(X_test, y_test)
        assert "HongConformal" in html

    def test_to_html_uses_cached_table(self, fitted_hong_and_data):
        """After coverage_table(), to_html() should work without X_test/y_test."""
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        report.coverage_table(X_test, y_test)
        html = report.to_html()  # No args — uses cache
        assert "<table>" in html

    def test_to_html_raises_without_data(self, fitted_hong_and_data):
        hc, _, _ = fitted_hong_and_data
        report = SCRReport(hc)
        with pytest.raises(ValueError, match="coverage_table"):
            report.to_html()

    def test_to_html_solvency_reference(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        html = report.to_html(X_test, y_test)
        assert "Solvency" in html


class TestSCRJSONOutput:
    def test_to_json_returns_string(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        js = report.to_json(X_test, y_test)
        assert isinstance(js, str)

    def test_to_json_valid_json(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        js = report.to_json(X_test, y_test)
        parsed = json.loads(js)
        assert isinstance(parsed, dict)

    def test_to_json_structure(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        parsed = json.loads(report.to_json(X_test, y_test))
        assert "predictor" in parsed
        assert "coverage_table" in parsed
        assert "methodology" in parsed
        assert parsed["predictor"] == "HongConformal"

    def test_to_json_coverage_table_is_list(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        parsed = json.loads(report.to_json(X_test, y_test))
        assert isinstance(parsed["coverage_table"], list)
        assert len(parsed["coverage_table"]) > 0

    def test_to_json_scr_alpha_is_0005(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        parsed = json.loads(report.to_json(X_test, y_test))
        assert parsed["methodology"]["scr_alpha"] == 0.005

    def test_to_json_uses_cached_table(self, fitted_hong_and_data):
        hc, X_test, y_test = fitted_hong_and_data
        report = SCRReport(hc)
        report.coverage_table(X_test, y_test)
        js = report.to_json()  # Uses cache
        parsed = json.loads(js)
        assert "coverage_table" in parsed

    def test_to_json_raises_without_data(self, fitted_hong_and_data):
        hc, _, _ = fitted_hong_and_data
        report = SCRReport(hc)
        with pytest.raises(ValueError):
            report.to_json()
