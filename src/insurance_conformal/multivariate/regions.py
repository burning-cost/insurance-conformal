"""
Prediction region representations.

JointPredictionSet is the output of JointConformalPredictor.predict(). It holds
per-dimension lower and upper bounds as arrays — one bound per test observation.

Design: we store bounds as dicts keyed by dimension name (matching the model
dict the user passed in). This keeps the API readable:
    joint_set.lower['frequency']   # array of freq lower bounds
    joint_set.upper['severity']    # array of sev upper bounds

For the SCR (one-sided) case, lower bounds are all 0.0 and users only care
about upper bounds.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray


class JointPredictionSet:
    """
    Joint hyperrectangular prediction set for multi-output models.

    Attributes
    ----------
    lower : dict[str, ndarray]
        Per-dimension lower bounds, shape (n_test,).
    upper : dict[str, ndarray]
        Per-dimension upper bounds, shape (n_test,).
    dimensions : list[str]
        Dimension names in order.
    n_obs : int
        Number of test observations.
    alpha : float
        Miscoverage level the set was calibrated for.
    method : str
        Method used: 'bonferroni', 'sidak', 'gwc', 'lwc'.
    one_sided : bool
        True if this is a one-sided (SCR / upper-tail only) set.
    """

    def __init__(
        self,
        lower: Dict[str, NDArray],
        upper: Dict[str, NDArray],
        dimensions: List[str],
        alpha: float,
        method: str,
        one_sided: bool = False,
    ):
        self.lower = {k: np.asarray(v, dtype=float) for k, v in lower.items()}
        self.upper = {k: np.asarray(v, dtype=float) for k, v in upper.items()}
        self.dimensions = dimensions
        self.alpha = alpha
        self.method = method
        self.one_sided = one_sided

        # Validate shapes
        shapes = {k: v.shape for k, v in self.lower.items()}
        shapes.update({k: v.shape for k, v in self.upper.items()})
        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise ValueError(f"Inconsistent shapes in lower/upper bounds: {shapes}")
        self.n_obs = list(self.lower.values())[0].shape[0]

    def marginal_intervals(self) -> Dict[str, NDArray]:
        """
        Return per-dimension interval widths (upper - lower), shape (n_obs,) each.
        """
        return {k: self.upper[k] - self.lower[k] for k in self.dimensions}

    def volume(self) -> NDArray[np.floating]:
        """
        Hyperrectangle volume per test observation, shape (n_obs,).

        Volume = product of interval widths across all dimensions.
        Use as an efficiency metric: lower volume = tighter prediction set.
        For one-sided sets, volume = product of upper bounds (lower = 0).
        """
        widths = self.marginal_intervals()
        vol = np.ones(self.n_obs, dtype=float)
        for k in self.dimensions:
            vol *= widths[k]
        return vol

    def contains(
        self,
        y: Union[Dict[str, NDArray], NDArray],
    ) -> NDArray[np.bool_]:
        """
        Check which observations have y within the prediction set.

        y : dict or ndarray
            True outcomes. If dict, keys must match dimensions.
            If ndarray shape (n_obs, d), columns match dimensions in order.

        Returns
        -------
        ndarray of bool, shape (n_obs,)
            True where ALL dimensions are covered simultaneously.
        """
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                y = {self.dimensions[0]: y}
            elif y.ndim == 2:
                y = {k: y[:, i] for i, k in enumerate(self.dimensions)}
            else:
                raise ValueError("y must be 1D or 2D ndarray")

        covered = np.ones(self.n_obs, dtype=bool)
        for k in self.dimensions:
            if k not in y:
                raise ValueError(f"y missing dimension '{k}'")
            y_k = np.asarray(y[k], dtype=float)
            covered &= (y_k >= self.lower[k]) & (y_k <= self.upper[k])
        return covered

    def joint_coverage_check(
        self,
        Y_test: Union[Dict[str, NDArray], NDArray],
    ) -> float:
        """
        Empirical joint coverage rate on test set.

        Returns float in [0, 1]. Should be >= 1 - alpha.
        """
        return float(np.mean(self.contains(Y_test)))

    def marginal_coverage_rates(
        self,
        Y_test: Union[Dict[str, NDArray], NDArray],
    ) -> Dict[str, float]:
        """
        Per-dimension empirical marginal coverage rates.

        Returns dict mapping dimension name -> float in [0, 1].
        """
        if isinstance(Y_test, np.ndarray):
            if Y_test.ndim == 2:
                Y_test = {k: Y_test[:, i] for i, k in enumerate(self.dimensions)}
            else:
                Y_test = {self.dimensions[0]: Y_test}

        rates = {}
        for k in self.dimensions:
            y_k = np.asarray(Y_test[k], dtype=float)
            rates[k] = float(np.mean((y_k >= self.lower[k]) & (y_k <= self.upper[k])))
        return rates

    def to_polars(self) -> "polars.DataFrame":
        """
        Export bounds to a Polars DataFrame.

        Columns: {dim}_lower, {dim}_upper for each dimension.
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for to_polars(). Install with: pip install polars"
            )

        data = {}
        for k in self.dimensions:
            data[f"{k}_lower"] = self.lower[k]
            data[f"{k}_upper"] = self.upper[k]
        return pl.DataFrame(data)

    def summary(self) -> Dict[str, object]:
        """Return a summary dict of the prediction set properties."""
        return {
            "n_obs": self.n_obs,
            "alpha": self.alpha,
            "method": self.method,
            "one_sided": self.one_sided,
            "dimensions": self.dimensions,
            "mean_volume": float(np.mean(self.volume())),
            "mean_widths": {
                k: float(np.mean(self.marginal_intervals()[k]))
                for k in self.dimensions
            },
        }

    def __repr__(self) -> str:
        s = self.summary()
        widths_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in s["mean_widths"].items()
        )
        return (
            f"JointPredictionSet("
            f"n={self.n_obs}, alpha={self.alpha}, method='{self.method}', "
            f"mean_widths=[{widths_str}])"
        )
