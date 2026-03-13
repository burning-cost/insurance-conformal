"""
Utility functions shared across insurance-conformal-risk modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(x: Any) -> np.ndarray:
    """
    Convert array-like input to a 1D numpy float64 array.

    Accepts numpy arrays, lists, pandas Series/DataFrame columns,
    and polars Series. Returns a 1D float64 array.
    """
    if isinstance(x, np.ndarray):
        return x.ravel().astype(float)

    # Polars Series or DataFrame column
    try:
        import polars as pl
        if isinstance(x, (pl.Series, pl.DataFrame)):
            arr = x.to_numpy()
            return arr.ravel().astype(float)
    except ImportError:
        pass

    # Pandas Series or array-like
    try:
        import pandas as pd
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.to_numpy().ravel().astype(float)
    except ImportError:
        pass

    return np.asarray(x, dtype=float).ravel()


def validate_same_length(*arrays: np.ndarray, names: list[str] | None = None) -> None:
    """Raise ValueError if arrays have different lengths."""
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        if names is not None:
            detail = ", ".join(f"{n}={l}" for n, l in zip(names, lengths))
        else:
            detail = str(lengths)
        raise ValueError(f"Arrays must have the same length. Got: {detail}")


def check_positive(arr: np.ndarray, name: str = "array") -> None:
    """Raise ValueError if any element is non-positive."""
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive (found {(arr <= 0).sum()} non-positive values)")


def check_non_negative(arr: np.ndarray, name: str = "array") -> None:
    """Raise ValueError if any element is negative."""
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative (found {(arr < 0).sum()} negative values)")
