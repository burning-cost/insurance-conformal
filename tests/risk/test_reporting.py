"""
Tests for reporting utilities.
"""

import numpy as np
import pytest

from insurance_conformal.risk.reporting import (
    premium_sufficiency_report,
    solvency_ii_model_error_note,
    risk_curve_dataframe,
)


class TestPremiumSufficiencyReport:

    def test_returns_dataframe(self):
        import polars as pl
        result = premium_sufficiency_report(
            lambda_hat=1.35,
            alpha=0.05,
            n_calibration=1000,
            B=5.0,
            corrected_risk=0.048,
        )
        assert isinstance(result, pl.DataFrame)

    def test_has_expected_columns(self):
        result = premium_sufficiency_report(1.35, 0.05, 1000, 5.0, 0.048)
        assert "metric" in result.columns
        assert "value" in result.columns
        assert "interpretation" in result.columns

    def test_gwp_adds_row(self):
        without = premium_sufficiency_report(1.35, 0.05, 1000, 5.0, 0.048)
        with_gwp = premium_sufficiency_report(1.35, 0.05, 1000, 5.0, 0.048, portfolio_gwp=10e6)
        assert with_gwp.shape[0] == without.shape[0] + 1


class TestSolvencyIINote:

    def test_returns_string(self):
        result = solvency_ii_model_error_note(0.05, 1.35, 1000)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_key_terms(self):
        result = solvency_ii_model_error_note(0.05, 1.35, 1000)
        assert "conformal risk control" in result.lower() or "conformal" in result.lower()
        assert "1,000" in result
        assert "1.350" in result

    def test_alpha_formatted(self):
        result = solvency_ii_model_error_note(0.03, 1.20, 500)
        assert "3.0%" in result or "3%" in result


class TestRiskCurveDataframe:

    def test_shape(self):
        lambdas = np.linspace(0.5, 3.0, 100)
        risk = np.linspace(0.5, 0.01, 100)  # Decreasing
        result = risk_curve_dataframe(lambdas, risk, alpha=0.05, lambda_hat=1.5)
        assert result.shape == (100, 4)

    def test_columns(self):
        lambdas = np.linspace(0.5, 3.0, 50)
        risk = np.linspace(0.3, 0.01, 50)
        result = risk_curve_dataframe(lambdas, risk, alpha=0.05, lambda_hat=1.5)
        assert "lambda" in result.columns
        assert "corrected_risk" in result.columns
        assert "below_target" in result.columns
        assert "is_lambda_hat" in result.columns

    def test_below_target_matches_alpha(self):
        lambdas = np.array([1.0, 2.0, 3.0])
        risk = np.array([0.10, 0.05, 0.02])
        result = risk_curve_dataframe(lambdas, risk, alpha=0.06, lambda_hat=2.0)
        below = result["below_target"].to_numpy()
        np.testing.assert_array_equal(below, [False, True, True])
