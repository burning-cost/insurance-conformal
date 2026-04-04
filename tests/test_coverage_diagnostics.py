"""
Tests for CoverageDiagnostics from insurance_conformal.diagnostics.

This module had zero dedicated test coverage. CoverageDiagnostics is a
standalone class that wraps any prediction interval result and provides
marginal coverage, decile diagnostics, and (optionally) matplotlib plots.
We avoid the plotting methods here to keep the test suite headless.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_conformal.diagnostics import CoverageDiagnostics


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_perfect_intervals(n: int = 200, seed: int = 0) -> dict:
    """Generate intervals that always cover the true value."""
    rng = np.random.default_rng(seed)
    y_true = rng.exponential(scale=2.0, size=n)
    y_pred = y_true * rng.uniform(0.8, 1.2, size=n)
    y_lower = np.zeros(n)
    y_upper = y_true + 1.0  # always covers
    return dict(y_true=y_true, y_lower=y_lower, y_upper=y_upper, y_pred=y_pred)


def _make_zero_coverage_intervals(n: int = 200, seed: int = 1) -> dict:
    """Generate intervals that never cover the true value."""
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(5.0, 10.0, size=n)
    # Vary predictions so coverage_by_decile can form non-degenerate bins
    y_pred = rng.uniform(2.0, 4.0, size=n)
    y_lower = np.zeros(n)
    y_upper = np.full(n, 1.0)  # always misses since y_true > 1
    return dict(y_true=y_true, y_lower=y_lower, y_upper=y_upper, y_pred=y_pred)


def _make_typical_intervals(n: int = 500, alpha: float = 0.10, seed: int = 42) -> dict:
    """Generate realistic intervals with approximately correct coverage."""
    rng = np.random.default_rng(seed)
    y_true = rng.poisson(lam=2.0, size=n).astype(float)
    # Use varying predictions so coverage_by_decile can form non-degenerate bins.
    # Constant predictions cause pd.qcut/cut to produce all-NaN labels.
    y_pred = np.clip(y_true * rng.uniform(0.8, 1.2, size=n), 0.1, None)
    q = np.quantile(np.abs(y_true - y_pred), 1 - alpha)
    y_lower = np.maximum(0.0, y_pred - q)
    y_upper = y_pred + q
    return dict(y_true=y_true, y_lower=y_lower, y_upper=y_upper, y_pred=y_pred, alpha=alpha)


@pytest.fixture(scope="module")
def perfect_diag():
    d = _make_perfect_intervals(n=200)
    return CoverageDiagnostics(
        y_true=d["y_true"],
        y_lower=d["y_lower"],
        y_upper=d["y_upper"],
        y_pred=d["y_pred"],
        alpha=0.10,
    )


@pytest.fixture(scope="module")
def zero_diag():
    d = _make_zero_coverage_intervals(n=200)
    return CoverageDiagnostics(
        y_true=d["y_true"],
        y_lower=d["y_lower"],
        y_upper=d["y_upper"],
        y_pred=d["y_pred"],
        alpha=0.10,
    )


@pytest.fixture(scope="module")
def typical_diag():
    d = _make_typical_intervals(n=500, alpha=0.10)
    return CoverageDiagnostics(
        y_true=d["y_true"],
        y_lower=d["y_lower"],
        y_upper=d["y_upper"],
        y_pred=d["y_pred"],
        alpha=0.10,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_construction(self):
        diag = CoverageDiagnostics(
            y_true=np.array([1.0, 2.0, 3.0]),
            y_lower=np.array([0.5, 1.5, 2.5]),
            y_upper=np.array([1.5, 2.5, 3.5]),
        )
        assert diag.y_true.shape == (3,)
        assert diag.y_lower.shape == (3,)
        assert diag.y_upper.shape == (3,)
        assert diag.alpha is None

    def test_with_alpha(self):
        diag = CoverageDiagnostics(
            y_true=np.array([1.0, 2.0]),
            y_lower=np.array([0.5, 1.5]),
            y_upper=np.array([1.5, 2.5]),
            alpha=0.10,
        )
        assert diag.alpha == 0.10

    def test_with_y_pred(self):
        y_pred = np.array([1.2, 2.1, 3.05])
        diag = CoverageDiagnostics(
            y_true=np.array([1.0, 2.0, 3.0]),
            y_lower=np.array([0.5, 1.5, 2.5]),
            y_upper=np.array([1.5, 2.5, 3.5]),
            y_pred=y_pred,
        )
        np.testing.assert_array_equal(diag.y_pred, y_pred)

    def test_without_y_pred_uses_midpoint(self):
        """When y_pred is not provided, it should be (lower + upper) / 2."""
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        diag = CoverageDiagnostics(
            y_true=np.array([1.0, 2.0, 3.0]),
            y_lower=lower,
            y_upper=upper,
        )
        expected_pred = (lower + upper) / 2.0
        np.testing.assert_allclose(diag.y_pred, expected_pred)

    def test_pandas_series_input(self):
        import pandas as pd
        diag = CoverageDiagnostics(
            y_true=pd.Series([1.0, 2.0]),
            y_lower=pd.Series([0.5, 1.5]),
            y_upper=pd.Series([1.5, 2.5]),
        )
        assert isinstance(diag.y_true, np.ndarray)

    def test_polars_series_input(self):
        diag = CoverageDiagnostics(
            y_true=pl.Series([1.0, 2.0]),
            y_lower=pl.Series([0.5, 1.5]),
            y_upper=pl.Series([1.5, 2.5]),
        )
        assert isinstance(diag.y_true, np.ndarray)

    def test_list_input(self):
        diag = CoverageDiagnostics(
            y_true=[1.0, 2.0, 3.0],
            y_lower=[0.5, 1.5, 2.5],
            y_upper=[1.5, 2.5, 3.5],
        )
        assert isinstance(diag.y_true, np.ndarray)
        assert len(diag.y_true) == 3

    def test_single_observation(self):
        """Single observation edge case."""
        diag = CoverageDiagnostics(
            y_true=np.array([2.0]),
            y_lower=np.array([1.0]),
            y_upper=np.array([3.0]),
            alpha=0.10,
        )
        assert diag.marginal_coverage == 1.0

    def test_all_zeros(self):
        """All zeros is a valid (degenerate) case."""
        n = 10
        diag = CoverageDiagnostics(
            y_true=np.zeros(n),
            y_lower=np.zeros(n),
            y_upper=np.zeros(n),
        )
        # y_true == 0 is in [0, 0], so all covered
        assert diag.marginal_coverage == 1.0


# ---------------------------------------------------------------------------
# marginal_coverage property
# ---------------------------------------------------------------------------


class TestMarginalCoverage:
    def test_perfect_coverage(self, perfect_diag):
        assert perfect_diag.marginal_coverage == 1.0

    def test_zero_coverage(self, zero_diag):
        assert zero_diag.marginal_coverage == 0.0

    def test_typical_coverage_in_range(self, typical_diag):
        cov = typical_diag.marginal_coverage
        assert 0.0 <= cov <= 1.0

    def test_exact_coverage_manual(self):
        """Verify the coverage computation manually."""
        # 3 out of 4 covered
        diag = CoverageDiagnostics(
            y_true=np.array([1.0, 2.0, 3.0, 10.0]),
            y_lower=np.array([0.5, 1.5, 2.5, 0.0]),
            y_upper=np.array([1.5, 2.5, 3.5, 5.0]),  # 10.0 > 5.0: not covered
        )
        assert diag.marginal_coverage == 0.75

    def test_boundary_covered(self):
        """Values exactly on the boundary should be considered covered."""
        diag = CoverageDiagnostics(
            y_true=np.array([1.0, 3.0]),
            y_lower=np.array([1.0, 2.0]),
            y_upper=np.array([2.0, 3.0]),
        )
        # Both are at exact boundary: 1.0 >= 1.0 and 1.0 <= 2.0 → covered
        # 3.0 >= 2.0 and 3.0 <= 3.0 → covered
        assert diag.marginal_coverage == 1.0

    def test_coverage_float_type(self, typical_diag):
        assert isinstance(typical_diag.marginal_coverage, float)

    def test_coverage_returns_scalar(self, perfect_diag):
        cov = perfect_diag.marginal_coverage
        assert np.isscalar(cov)


# ---------------------------------------------------------------------------
# mean_width and median_width properties
# ---------------------------------------------------------------------------


class TestWidthProperties:
    def test_mean_width_nonnegative(self, typical_diag):
        assert typical_diag.mean_width >= 0.0

    def test_median_width_nonnegative(self, typical_diag):
        assert typical_diag.median_width >= 0.0

    def test_mean_width_zero_when_degenerate(self):
        """If all intervals have zero width, mean_width = 0."""
        diag = CoverageDiagnostics(
            y_true=np.array([1.0, 2.0]),
            y_lower=np.array([1.0, 2.0]),
            y_upper=np.array([1.0, 2.0]),
        )
        assert diag.mean_width == 0.0

    def test_mean_width_manual(self):
        diag = CoverageDiagnostics(
            y_true=np.array([1.0, 2.0, 3.0]),
            y_lower=np.array([0.0, 1.0, 2.0]),
            y_upper=np.array([2.0, 3.0, 4.0]),
        )
        # All widths = 2.0
        assert diag.mean_width == 2.0

    def test_median_width_manual(self):
        diag = CoverageDiagnostics(
            y_true=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            y_lower=np.zeros(5),
            y_upper=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        assert diag.median_width == 3.0

    def test_mean_width_float_type(self, typical_diag):
        assert isinstance(typical_diag.mean_width, float)

    def test_median_width_float_type(self, typical_diag):
        assert isinstance(typical_diag.median_width, float)

    def test_mean_geq_zero_for_any_input(self):
        rng = np.random.default_rng(0)
        lower = rng.uniform(0, 5, 100)
        upper = lower + rng.uniform(0, 3, 100)
        diag = CoverageDiagnostics(
            y_true=rng.exponential(2, 100),
            y_lower=lower,
            y_upper=upper,
        )
        assert diag.mean_width >= 0.0


# ---------------------------------------------------------------------------
# coverage_by_decile
# ---------------------------------------------------------------------------


class TestCoverageByDecile:
    def test_returns_polars_dataframe(self, typical_diag):
        result = typical_diag.coverage_by_decile()
        assert isinstance(result, pl.DataFrame)

    def test_required_columns(self, typical_diag):
        result = typical_diag.coverage_by_decile()
        required = {"decile", "mean_predicted", "mean_width", "n_obs", "coverage", "target_coverage"}
        assert required.issubset(set(result.columns))

    def test_n_obs_sums_to_total(self, typical_diag):
        result = typical_diag.coverage_by_decile()
        assert result["n_obs"].sum() == len(typical_diag.y_true)

    def test_coverage_in_unit_interval(self, typical_diag):
        result = typical_diag.coverage_by_decile()
        assert (result["coverage"] >= 0.0).all()
        assert (result["coverage"] <= 1.0).all()

    def test_n_bins_parameter(self, typical_diag):
        result = typical_diag.coverage_by_decile(n_bins=5)
        assert len(result) <= 5

    def test_default_10_bins(self, typical_diag):
        result = typical_diag.coverage_by_decile()
        assert len(result) <= 10

    def test_mean_width_nonnegative(self, typical_diag):
        result = typical_diag.coverage_by_decile()
        assert (result["mean_width"] >= 0.0).all()

    def test_target_coverage_nan_when_alpha_none(self):
        """When alpha is not provided, target_coverage should be NaN."""
        rng = np.random.default_rng(0)
        n = 100
        diag = CoverageDiagnostics(
            y_true=rng.exponential(2, n),
            y_lower=np.zeros(n),
            y_upper=rng.uniform(1, 5, n),
        )
        result = diag.coverage_by_decile()
        import math
        assert all(math.isnan(v) for v in result["target_coverage"].to_list())

    def test_target_coverage_matches_alpha_when_set(self, typical_diag):
        result = typical_diag.coverage_by_decile()
        assert np.allclose(result["target_coverage"].to_numpy(), 0.90)

    def test_perfect_coverage_all_ones(self, perfect_diag):
        result = perfect_diag.coverage_by_decile()
        assert (result["coverage"] == 1.0).all()

    def test_zero_coverage_all_zeros(self, zero_diag):
        result = zero_diag.coverage_by_decile()
        assert (result["coverage"] == 0.0).all()

    def test_single_observation_fallback(self):
        """Single observation: should not crash."""
        diag = CoverageDiagnostics(
            y_true=np.array([2.0]),
            y_lower=np.array([1.0]),
            y_upper=np.array([3.0]),
            alpha=0.10,
        )
        result = diag.coverage_by_decile()
        assert isinstance(result, pl.DataFrame)

    def test_n_bins_1(self, typical_diag):
        result = typical_diag.coverage_by_decile(n_bins=1)
        assert isinstance(result, pl.DataFrame)
        assert len(result) >= 1

    def test_n_bins_20(self):
        rng = np.random.default_rng(7)
        n = 400
        y_true = rng.poisson(2.0, n).astype(float)
        y_pred = np.clip(y_true * rng.uniform(0.8, 1.2, n), 0.1, None)
        y_lower = np.zeros(n)
        y_upper = np.full(n, 5.0)
        diag = CoverageDiagnostics(y_true=y_true, y_lower=y_lower, y_upper=y_upper, y_pred=y_pred, alpha=0.10)
        result = diag.coverage_by_decile(n_bins=20)
        assert len(result) <= 20
        assert result["n_obs"].sum() == n

    def test_all_same_predicted_value(self):
        """When all predictions are identical, qcut falls back to cut."""
        n = 100
        diag = CoverageDiagnostics(
            y_true=np.ones(n) * 2.0,
            y_lower=np.ones(n),
            y_upper=np.ones(n) * 3.0,
            y_pred=np.ones(n) * 2.0,  # all same
            alpha=0.10,
        )
        result = diag.coverage_by_decile()
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Internal state: _covered and _widths
# ---------------------------------------------------------------------------


class TestInternalState:
    def test_covered_array_is_boolean(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.5, 1.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5])
        diag = CoverageDiagnostics(y_true=y, y_lower=lo, y_upper=hi)
        assert diag._covered.dtype == bool

    def test_widths_array_correct(self):
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([2.0, 3.0, 6.0])
        diag = CoverageDiagnostics(y_true=np.ones(3), y_lower=lo, y_upper=hi)
        np.testing.assert_allclose(diag._widths, [2.0, 2.0, 4.0])

    def test_covered_correct_values(self):
        y = np.array([1.0, 5.0, 2.0])
        lo = np.array([0.5, 6.0, 1.5])
        hi = np.array([1.5, 7.0, 2.5])
        diag = CoverageDiagnostics(y_true=y, y_lower=lo, y_upper=hi)
        # 1.0 in [0.5, 1.5] → True; 5.0 in [6.0, 7.0] → False; 2.0 in [1.5, 2.5] → True
        np.testing.assert_array_equal(diag._covered, [True, False, True])

    def test_covered_all_true(self, perfect_diag):
        assert perfect_diag._covered.all()

    def test_covered_all_false(self, zero_diag):
        assert not zero_diag._covered.any()

    def test_widths_nonnegative_for_valid_intervals(self, typical_diag):
        assert np.all(typical_diag._widths >= 0.0)


# ---------------------------------------------------------------------------
# Large n stress tests
# ---------------------------------------------------------------------------


class TestLargeN:
    def test_large_n_construction(self):
        rng = np.random.default_rng(99)
        n = 10_000
        y_true = rng.poisson(3.0, n).astype(float)
        y_pred = np.full(n, 3.0)
        y_lower = np.zeros(n)
        y_upper = np.full(n, 8.0)
        diag = CoverageDiagnostics(
            y_true=y_true, y_lower=y_lower, y_upper=y_upper,
            y_pred=y_pred, alpha=0.10
        )
        cov = diag.marginal_coverage
        assert 0.0 <= cov <= 1.0

    def test_large_n_coverage_by_decile(self):
        rng = np.random.default_rng(100)
        n = 5_000
        y_true = rng.exponential(2.0, n)
        y_pred = rng.uniform(1, 4, n)
        y_lower = np.zeros(n)
        y_upper = y_true + 1.0
        diag = CoverageDiagnostics(
            y_true=y_true, y_lower=y_lower, y_upper=y_upper,
            y_pred=y_pred, alpha=0.10
        )
        result = diag.coverage_by_decile()
        assert result["n_obs"].sum() == n


# ---------------------------------------------------------------------------
# Edge cases: extreme values
# ---------------------------------------------------------------------------


class TestExtremeValues:
    def test_very_large_upper_bounds(self):
        """Upper bound = 1e10 should not cause overflow."""
        n = 50
        diag = CoverageDiagnostics(
            y_true=np.ones(n),
            y_lower=np.zeros(n),
            y_upper=np.full(n, 1e10),
            alpha=0.10,
        )
        assert diag.marginal_coverage == 1.0
        assert np.isfinite(diag.mean_width)

    def test_very_small_upper_bounds(self):
        """Upper bound very close to zero, true values positive: near-zero coverage."""
        n = 50
        diag = CoverageDiagnostics(
            y_true=np.ones(n) * 10.0,
            y_lower=np.zeros(n),
            y_upper=np.full(n, 1e-10),
            alpha=0.10,
        )
        assert diag.marginal_coverage == 0.0

    def test_negative_lower_bound(self):
        """Negative lower bounds are numerically valid."""
        diag = CoverageDiagnostics(
            y_true=np.array([0.5, 1.5]),
            y_lower=np.array([-1.0, -0.5]),
            y_upper=np.array([1.0, 2.0]),
        )
        # Both 0.5 and 1.5 are covered
        assert diag.marginal_coverage == 1.0

    def test_all_true_values_equal(self):
        """When all true values are the same constant."""
        n = 50
        diag = CoverageDiagnostics(
            y_true=np.full(n, 3.0),
            y_lower=np.full(n, 2.0),
            y_upper=np.full(n, 4.0),
            alpha=0.10,
        )
        assert diag.marginal_coverage == 1.0
        assert diag.mean_width == 2.0

    def test_widths_when_bounds_equal(self):
        """When lower == upper, width is 0."""
        n = 10
        diag = CoverageDiagnostics(
            y_true=np.full(n, 2.0),
            y_lower=np.full(n, 2.0),
            y_upper=np.full(n, 2.0),
        )
        assert diag.mean_width == 0.0
        assert diag.median_width == 0.0
        assert diag.marginal_coverage == 1.0  # y_true == 2.0 in [2.0, 2.0]
