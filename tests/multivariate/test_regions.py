"""
Tests for JointPredictionSet region representations.
"""

import numpy as np
import pytest

from insurance_conformal.multivariate.regions import JointPredictionSet


def _make_simple_set(n=50, dims=None):
    if dims is None:
        dims = ["frequency", "severity"]
    rng = np.random.default_rng(0)
    lower = {k: rng.uniform(0.0, 0.5, n) for k in dims}
    upper = {k: lower[k] + rng.uniform(0.1, 2.0, n) for k in dims}
    return JointPredictionSet(
        lower=lower, upper=upper, dimensions=dims, alpha=0.05, method="lwc"
    )


class TestJointPredictionSetCreation:
    def test_basic_creation(self):
        s = _make_simple_set()
        assert s.n_obs == 50
        assert s.alpha == 0.05
        assert s.method == "lwc"
        assert not s.one_sided

    def test_one_sided_creation(self):
        rng = np.random.default_rng(1)
        n = 20
        lower = {"a": np.zeros(n), "b": np.zeros(n)}
        upper = {"a": rng.uniform(1, 5, n), "b": rng.uniform(10, 50, n)}
        s = JointPredictionSet(lower, upper, ["a", "b"], alpha=0.005, method="gwc", one_sided=True)
        assert s.one_sided

    def test_inconsistent_shapes_raise(self):
        lower = {"a": np.zeros(10), "b": np.zeros(10)}
        upper = {"a": np.ones(10), "b": np.ones(20)}  # mismatched
        with pytest.raises(ValueError, match="Inconsistent shapes"):
            JointPredictionSet(lower, upper, ["a", "b"], alpha=0.05, method="lwc")

    def test_3d_creation(self):
        s = _make_simple_set(dims=["flood", "fire", "sub"])
        assert s.n_obs == 50
        assert len(s.dimensions) == 3


class TestVolume:
    def test_volume_shape(self):
        s = _make_simple_set(n=30)
        vol = s.volume()
        assert vol.shape == (30,)

    def test_volume_positive(self):
        s = _make_simple_set()
        assert np.all(s.volume() > 0)

    def test_volume_is_product_of_widths(self):
        n = 10
        lower = {"a": np.zeros(n), "b": np.zeros(n)}
        upper = {"a": np.ones(n) * 2.0, "b": np.ones(n) * 3.0}
        s = JointPredictionSet(lower, upper, ["a", "b"], alpha=0.05, method="gwc")
        expected_vol = 2.0 * 3.0
        np.testing.assert_allclose(s.volume(), expected_vol)


class TestContains:
    def test_midpoints_inside(self):
        s = _make_simple_set(n=20)
        mid = {k: (s.lower[k] + s.upper[k]) / 2 for k in s.dimensions}
        assert np.all(s.contains(mid))

    def test_lower_bound_on_boundary(self):
        s = _make_simple_set(n=20)
        # Exactly at lower bound should be inside (>=)
        at_lower = {k: s.lower[k] for k in s.dimensions}
        assert np.all(s.contains(at_lower))

    def test_upper_bound_on_boundary(self):
        s = _make_simple_set(n=20)
        at_upper = {k: s.upper[k] for k in s.dimensions}
        assert np.all(s.contains(at_upper))

    def test_far_outside_not_inside(self):
        n = 20
        lower = {"a": np.zeros(n), "b": np.zeros(n)}
        upper = {"a": np.ones(n), "b": np.ones(n)}
        s = JointPredictionSet(lower, upper, ["a", "b"], alpha=0.05, method="gwc")
        # y far outside
        y = {"a": np.ones(n) * 10.0, "b": np.zeros(n)}
        covered = s.contains(y)
        assert not np.any(covered)

    def test_1d_ndarray_input(self):
        rng = np.random.default_rng(0)
        n = 20
        lower = {"out": np.zeros(n)}
        upper = {"out": np.ones(n) * 5.0}
        s = JointPredictionSet(lower, upper, ["out"], alpha=0.05, method="gwc")
        y = rng.uniform(0, 5, n)
        covered = s.contains(y)
        assert covered.dtype == bool
        assert np.all(covered)

    def test_2d_ndarray_input(self):
        s = _make_simple_set(n=20)
        mid_arr = np.column_stack([
            (s.lower[k] + s.upper[k]) / 2 for k in s.dimensions
        ])
        covered = s.contains(mid_arr)
        assert np.all(covered)

    def test_missing_dimension_raises(self):
        s = _make_simple_set(n=10)
        y = {"frequency": np.zeros(10)}  # missing 'severity'
        with pytest.raises(ValueError, match="missing dimension"):
            s.contains(y)


class TestCoverageCheck:
    def test_joint_coverage_returns_float(self):
        s = _make_simple_set(n=30)
        mid = {k: (s.lower[k] + s.upper[k]) / 2 for k in s.dimensions}
        cov = s.joint_coverage_check(mid)
        assert isinstance(cov, float)
        assert cov == pytest.approx(1.0)

    def test_marginal_coverage_returns_dict(self):
        s = _make_simple_set(n=30)
        mid = {k: (s.lower[k] + s.upper[k]) / 2 for k in s.dimensions}
        rates = s.marginal_coverage_rates(mid)
        assert isinstance(rates, dict)
        assert set(rates.keys()) == set(s.dimensions)

    def test_outside_point_zero_joint_coverage(self):
        n = 20
        lower = {"a": np.ones(n) * 2.0, "b": np.zeros(n)}
        upper = {"a": np.ones(n) * 3.0, "b": np.ones(n)}
        s = JointPredictionSet(lower, upper, ["a", "b"], alpha=0.05, method="gwc")
        y = {"a": np.zeros(n), "b": np.zeros(n)}  # a is outside [2, 3]
        cov = s.joint_coverage_check(y)
        assert cov == pytest.approx(0.0)


class TestMarginalIntervals:
    def test_widths_positive(self):
        s = _make_simple_set()
        widths = s.marginal_intervals()
        for k, w in widths.items():
            assert np.all(w > 0)

    def test_widths_eq_upper_minus_lower(self):
        s = _make_simple_set()
        widths = s.marginal_intervals()
        for k in s.dimensions:
            np.testing.assert_allclose(widths[k], s.upper[k] - s.lower[k])


class TestSummaryAndRepr:
    def test_summary_keys(self):
        s = _make_simple_set()
        summary = s.summary()
        assert "n_obs" in summary
        assert "mean_volume" in summary
        assert "mean_widths" in summary
        assert "dimensions" in summary

    def test_repr(self):
        s = _make_simple_set()
        r = repr(s)
        assert "JointPredictionSet" in r
        assert "n=" in r


class TestToPolars:
    def test_to_polars_columns(self):
        s = _make_simple_set(n=10)
        try:
            df = s.to_polars()
            import polars as pl
            expected = {"frequency_lower", "frequency_upper", "severity_lower", "severity_upper"}
            assert set(df.columns) == expected
            assert df.height == 10
        except ImportError:
            pytest.skip("polars not available")

    def test_to_polars_values(self):
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not available")

        n = 5
        lower = {"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        upper = {"a": np.array([2.0, 3.0, 4.0, 5.0, 6.0])}
        s = JointPredictionSet(lower, upper, ["a"], alpha=0.05, method="gwc")
        df = s.to_polars()
        np.testing.assert_allclose(df["a_lower"].to_numpy(), lower["a"])
        np.testing.assert_allclose(df["a_upper"].to_numpy(), upper["a"])
