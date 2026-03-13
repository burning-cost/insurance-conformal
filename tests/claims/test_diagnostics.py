"""
Tests for diagnostics.py — conditional_coverage_gap, calibration_plot_data,
calibration_plot, width_by_covariate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_conformal.claims.hong import HongConformal
from insurance_conformal.claims.diagnostics import (
    calibration_plot_data,
    conditional_coverage_gap,
    width_by_covariate,
)


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
def fitted_predictor():
    X_all, y_all = make_tweedie_data(n=450, seed=30)
    X_train, y_train = X_all[:300], y_all[:300]
    X_test, y_test = X_all[300:], y_all[300:]
    hc = HongConformal().fit(X_train, y_train)
    return hc, X_test, y_test


class TestConditionalCoverageGap:
    def test_returns_dict(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = (X_test[:, 1] > 0).astype(int)
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.10, groups=groups)
        assert isinstance(result, dict)

    def test_required_keys(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = (X_test[:, 1] > 0).astype(int)
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.10, groups=groups)
        for key in ["overall_coverage", "nominal_coverage", "group_results", "max_gap", "min_gap"]:
            assert key in result

    def test_group_results_is_dataframe(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = (X_test[:, 1] > 0).astype(int)
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.10, groups=groups)
        assert isinstance(result["group_results"], pd.DataFrame)

    def test_group_results_columns(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = (X_test[:, 1] > 0).astype(int)
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.10, groups=groups)
        df = result["group_results"]
        for col in ["group", "n", "coverage", "width", "coverage_gap"]:
            assert col in df.columns

    def test_two_groups_gives_two_rows(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = (X_test[:, 1] > 0).astype(int)
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.10, groups=groups)
        assert len(result["group_results"]) == 2

    def test_nominal_coverage_correct(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = np.ones(len(y_test), dtype=int)
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.05, groups=groups)
        assert result["nominal_coverage"] == pytest.approx(0.95)

    def test_string_groups(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = np.where(X_test[:, 1] > 0, "high", "low")
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.10, groups=groups)
        df = result["group_results"]
        assert set(df["group"].values) == {"high", "low"}

    def test_overall_coverage_in_range(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = (X_test[:, 1] > 0).astype(int)
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.10, groups=groups)
        assert 0.0 <= result["overall_coverage"] <= 1.0

    def test_many_groups(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = np.arange(len(y_test)) % 5  # 5 groups
        result = conditional_coverage_gap(hc, X_test, y_test, alpha=0.10, groups=groups)
        assert len(result["group_results"]) == 5

    def test_invalid_alpha_raises(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        groups = np.ones(len(y_test))
        with pytest.raises(ValueError, match="alpha"):
            conditional_coverage_gap(hc, X_test, y_test, alpha=1.5, groups=groups)


class TestCalibrationPlotData:
    def test_returns_dataframe(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = calibration_plot_data(hc, X_test, y_test)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = calibration_plot_data(hc, X_test, y_test)
        for col in ["alpha", "nominal_coverage", "empirical_coverage", "mean_width", "coverage_gap"]:
            assert col in df.columns

    def test_default_n_rows(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = calibration_plot_data(hc, X_test, y_test)
        assert len(df) == 50  # default 50 alpha values

    def test_custom_alpha_grid(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        grid = np.array([0.05, 0.10, 0.20])
        df = calibration_plot_data(hc, X_test, y_test, alpha_grid=grid)
        assert len(df) == 3

    def test_nominal_coverage_equals_1_minus_alpha(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        grid = np.array([0.05, 0.10, 0.20])
        df = calibration_plot_data(hc, X_test, y_test, alpha_grid=grid)
        np.testing.assert_allclose(
            df["nominal_coverage"].values,
            1.0 - df["alpha"].values,
            atol=1e-12
        )

    def test_coverage_gap_is_emp_minus_nom(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = calibration_plot_data(hc, X_test, y_test)
        np.testing.assert_allclose(
            df["coverage_gap"].values,
            df["empirical_coverage"].values - df["nominal_coverage"].values,
            atol=1e-12
        )

    def test_empirical_coverage_in_01(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = calibration_plot_data(hc, X_test, y_test)
        assert np.all(df["empirical_coverage"] >= 0.0)
        assert np.all(df["empirical_coverage"] <= 1.0)

    def test_conformal_coverage_non_decreasing_with_nominal(self, fitted_predictor):
        """Empirical coverage should weakly increase with nominal coverage."""
        hc, X_test, y_test = fitted_predictor
        df = calibration_plot_data(hc, X_test, y_test).sort_values("nominal_coverage")
        # Allow small violations due to discrete order statistics
        violations = np.sum(np.diff(df["empirical_coverage"].values) < -0.1)
        assert violations == 0, f"{violations} large violations in calibration monotonicity"


class TestCalibrationPlot:
    def test_returns_axes(self, fitted_predictor):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from insurance_conformal.claims.diagnostics import calibration_plot

        hc, X_test, y_test = fitted_predictor
        ax = calibration_plot(hc, X_test, y_test, alpha_grid=np.linspace(0.05, 0.5, 10))
        assert ax is not None
        plt.close("all")

    def test_accepts_existing_axes(self, fitted_predictor):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from insurance_conformal.claims.diagnostics import calibration_plot

        hc, X_test, y_test = fitted_predictor
        fig, ax = plt.subplots()
        returned_ax = calibration_plot(
            hc, X_test, y_test,
            alpha_grid=np.linspace(0.05, 0.5, 5),
            ax=ax
        )
        assert returned_ax is ax
        plt.close("all")

    def test_raises_without_matplotlib(self, fitted_predictor, monkeypatch):
        """Should raise ImportError if matplotlib not installed."""
        import sys
        from insurance_conformal.claims.diagnostics import calibration_plot

        hc, X_test, y_test = fitted_predictor
        # Mock matplotlib as unavailable
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

        with pytest.raises((ImportError, TypeError)):
            calibration_plot(hc, X_test, y_test)


class TestWidthByCovariate:
    def test_returns_dataframe(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = width_by_covariate(hc, X_test, y_test, alpha=0.10, covariate_idx=0)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = width_by_covariate(hc, X_test, y_test, alpha=0.10, covariate_idx=0)
        for col in ["bin_centre", "mean_width", "mean_coverage", "n"]:
            assert col in df.columns

    def test_default_n_bins(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = width_by_covariate(hc, X_test, y_test, alpha=0.10, covariate_idx=0)
        assert len(df) == 10

    def test_custom_n_bins(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = width_by_covariate(hc, X_test, y_test, alpha=0.10, covariate_idx=0, n_bins=5)
        assert len(df) == 5

    def test_mean_width_positive(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = width_by_covariate(hc, X_test, y_test, alpha=0.10, covariate_idx=0)
        assert np.all(df["mean_width"] > 0.0)

    def test_coverage_in_range(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        df = width_by_covariate(hc, X_test, y_test, alpha=0.10, covariate_idx=0)
        assert np.all(df["mean_coverage"] >= 0.0)
        assert np.all(df["mean_coverage"] <= 1.0)

    def test_n_sums_to_n_test(self, fitted_predictor):
        """Bins should partition the test set."""
        hc, X_test, y_test = fitted_predictor
        df = width_by_covariate(hc, X_test, y_test, alpha=0.10, covariate_idx=0)
        assert df["n"].sum() == len(y_test)

    def test_different_covariate_indices(self, fitted_predictor):
        hc, X_test, y_test = fitted_predictor
        for idx in range(X_test.shape[1]):
            df = width_by_covariate(hc, X_test, y_test, alpha=0.10, covariate_idx=idx)
            assert len(df) > 0
