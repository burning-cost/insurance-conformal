"""
Expanded tests for risk utility functions.
"""

from __future__ import annotations

import numpy as np
import pytest
import polars as pl
import pandas as pd

from insurance_conformal.risk.utils import (
    as_numpy,
    validate_same_length,
    check_positive,
    check_non_negative,
)


class TestAsNumpyExpanded:
    def test_float_scalar(self):
        result = as_numpy(3.14)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float

    def test_empty_list(self):
        result = as_numpy([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        assert result.dtype == float

    def test_pandas_series(self):
        s = pd.Series([1, 2, 3])
        result = as_numpy(s)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_polars_dataframe_1d(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = as_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1  # raveled

    def test_integer_array_converted_to_float(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = as_numpy(arr)
        assert result.dtype == float

    def test_2d_numpy_raveled(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = as_numpy(arr)
        assert result.ndim == 1
        assert len(result) == 4

    def test_tuple_input(self):
        result = as_numpy((1.0, 2.0, 3.0))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


class TestValidateSameLengthExpanded:
    def test_three_arrays_equal_length(self):
        a = np.ones(5)
        b = np.ones(5)
        c = np.ones(5)
        validate_same_length(a, b, c)  # should not raise

    def test_three_arrays_mismatch_raises(self):
        a = np.ones(5)
        b = np.ones(5)
        c = np.ones(6)
        with pytest.raises(ValueError):
            validate_same_length(a, b, c)

    def test_single_array_ok(self):
        validate_same_length(np.ones(10))

    def test_error_contains_lengths(self):
        a = np.ones(3)
        b = np.ones(5)
        with pytest.raises(ValueError, match="3"):
            validate_same_length(a, b)

    def test_named_arrays_in_error(self):
        a = np.ones(3)
        b = np.ones(4)
        with pytest.raises(ValueError, match="foo"):
            validate_same_length(a, b, names=["foo", "bar"])

    def test_empty_arrays_same_length(self):
        a = np.array([])
        b = np.array([])
        validate_same_length(a, b)  # both length 0, ok


class TestCheckPositiveExpanded:
    def test_all_positive(self):
        check_positive(np.array([0.001, 1.0, 100.0]))

    def test_one_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            check_positive(np.array([1.0, 0.0, 2.0]))

    def test_all_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            check_positive(np.array([-1.0, -2.0]))

    def test_error_count_in_message(self):
        arr = np.array([1.0, 0.0, -1.0, 2.0, -3.0])
        with pytest.raises(ValueError, match="3"):
            check_positive(arr)

    def test_large_positive_array_ok(self):
        check_positive(np.ones(10_000))

    def test_custom_name_in_error(self):
        with pytest.raises(ValueError, match="premium"):
            check_positive(np.array([1.0, 0.0]), name="premium")

    def test_very_small_positive_ok(self):
        check_positive(np.array([1e-300]))


class TestCheckNonNegativeExpanded:
    def test_zero_ok(self):
        check_non_negative(np.array([0.0, 1.0, 2.0]))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            check_non_negative(np.array([0.0, -0.001, 1.0]))

    def test_all_zero_ok(self):
        check_non_negative(np.zeros(10))

    def test_custom_name_in_error(self):
        with pytest.raises(ValueError, match="claims"):
            check_non_negative(np.array([-1.0]), name="claims")

    def test_large_positive_array_ok(self):
        check_non_negative(np.ones(10_000))

    def test_very_small_negative_raises(self):
        with pytest.raises(ValueError):
            check_non_negative(np.array([1.0, -1e-15]))
