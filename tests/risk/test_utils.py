"""
Tests for utility functions.
"""

import numpy as np
import pytest

from insurance_conformal.risk.utils import (
    as_numpy,
    validate_same_length,
    check_positive,
    check_non_negative,
)


class TestAsNumpy:

    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = as_numpy(arr)
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == float

    def test_list_conversion(self):
        result = as_numpy([1, 2, 3])
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_2d_flattened(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = as_numpy(arr)
        assert result.ndim == 1
        assert len(result) == 4

    def test_integer_to_float(self):
        result = as_numpy(np.array([1, 2, 3], dtype=int))
        assert result.dtype == float

    def test_polars_series(self):
        import polars as pl
        s = pl.Series([1.0, 2.0, 3.0])
        result = as_numpy(s)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


class TestValidateSameLength:

    def test_equal_lengths_ok(self):
        validate_same_length(np.array([1, 2]), np.array([3, 4]))

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            validate_same_length(np.array([1, 2]), np.array([3, 4, 5]))

    def test_named_error_message(self):
        with pytest.raises(ValueError, match="y_cal"):
            validate_same_length(
                np.array([1, 2]), np.array([3, 4, 5]),
                names=["y_cal", "premium_cal"]
            )


class TestCheckPositive:

    def test_positive_ok(self):
        check_positive(np.array([1.0, 2.0, 0.5]))

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            check_positive(np.array([1.0, 0.0, 2.0]))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            check_positive(np.array([-1.0, 2.0]))


class TestCheckNonNegative:

    def test_zero_ok(self):
        check_non_negative(np.array([0.0, 1.0, 2.0]))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            check_non_negative(np.array([1.0, -0.1]))
