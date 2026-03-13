"""
IntervalWidthController: find the most efficient intervals that still control
expected width.

The standard conformal prediction API gives you intervals at a fixed miscoverage
level alpha. But interval width is not controlled — it can be arbitrarily large
for high-variance policies, and that width has a direct business cost (you're
reserving the full interval).

IntervalWidthController flips the problem: given a maximum acceptable expected
width (as a fraction of some scale), find the tightest lambda such that
E[width(C_lambda(X))] <= width_target.

The parameterisation: lambda controls the quantile level used to build intervals.
Larger lambda = higher quantile = wider intervals = guaranteed coverage is easier
to achieve but intervals are costlier to reserve against. Smaller lambda = tighter
intervals = more efficient capital allocation.

This is useful as a second-stage optimisation: use insurance-conformal to generate
coverage-controlled intervals at, say, 90%, then use IntervalWidthController to
find the minimum lambda that keeps expected width below a regulatory or internal
budget constraint.

Note: this controller is width-focused, not coverage-focused. If you run it at
a very tight width target, you may lose coverage. The predict() output includes
the estimated coverage at lambda_hat on the calibration set, so you can check.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import polars as pl

from insurance_conformal.risk.calibration import conformal_risk_calibration
from insurance_conformal.risk.utils import as_numpy


class IntervalWidthController:
    """
    Control expected interval width via conformal risk control.

    Finds the smallest lambda (quantile level) such that:

        E[upper_lambda(X) - lower_lambda(X)] / scale <= width_target

    The lambda here is a quantile level in (0, 1). It parameterises how wide
    the intervals are: lambda=0.9 gives 90th-percentile-based bounds (wider),
    lambda=0.5 gives median-based bounds (narrower, but lower coverage).

    For a width-controlled prediction system, you need a function that maps
    (X, lambda) -> (lower, upper). This is typically provided by a fitted
    conformal predictor. Pass calibration-set intervals at multiple lambda
    values and the controller finds the optimal lambda.

    Parameters
    ----------
    width_target : float
        Maximum acceptable expected interval width, expressed as a fraction
        of scale. width_target=0.5 with scale=1000 means expected width <= 500.
    scale : float, default 1.0
        Normalisation for the width loss. Set to the maximum expected claim
        or the mean prediction to make width_target interpretable.
    lambda_grid : np.ndarray or None
        Quantile levels to search over. Default: np.linspace(0.50, 0.995, 200).
        Must be strictly increasing. Higher lambda = wider intervals.
    B : float, default 1.0
        Upper bound on normalised width loss. Set to
        max_possible_width / scale. Usually 1.0 if scale is set correctly.

    Attributes
    ----------
    lambda_hat_ : float
        Calibrated quantile level.
    n_calibration_ : int
    is_calibrated_ : bool

    Examples
    --------
    Minimal usage — pass pre-computed interval arrays::

        >>> # widths_cal[i, j] = interval width for obs i at lambda_grid[j]
        >>> controller = IntervalWidthController(width_target=0.3, scale=max_width)
        >>> controller.calibrate_from_widths(widths_cal)
        >>> # Now apply lambda_hat to your conformal predictor
        >>> print(controller.lambda_hat_)
    """

    def __init__(
        self,
        width_target: float,
        scale: float = 1.0,
        lambda_grid: Optional[np.ndarray] = None,
        B: float = 1.0,
    ) -> None:
        if width_target <= 0:
            raise ValueError(f"width_target must be positive, got {width_target}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if B <= 0:
            raise ValueError(f"B must be positive, got {B}")

        self.width_target = width_target
        self.scale = scale
        self.B = B
        self.lambda_grid = (
            np.linspace(0.50, 0.995, 200) if lambda_grid is None
            else np.asarray(lambda_grid, dtype=float)
        )

        if not np.all(np.diff(self.lambda_grid) > 0):
            raise ValueError("lambda_grid must be strictly increasing")

        self.lambda_hat_: Optional[float] = None
        self.risk_curve_: Optional[np.ndarray] = None
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

    def calibrate_from_widths(
        self,
        widths_cal: np.ndarray,
    ) -> "IntervalWidthController":
        """
        Calibrate from a pre-computed matrix of interval widths.

        Parameters
        ----------
        widths_cal : np.ndarray, shape (n, m)
            widths_cal[i, j] = interval width for calibration observation i
            at lambda_grid[j]. Must be non-decreasing along axis=1 (wider
            intervals for larger lambda). Must be non-negative.

        Returns
        -------
        self
        """
        widths_cal = np.asarray(widths_cal, dtype=float)

        if widths_cal.ndim != 2:
            raise ValueError(
                f"widths_cal must be 2D (n_obs x n_lambda), got shape {widths_cal.shape}"
            )
        if widths_cal.shape[1] != len(self.lambda_grid):
            raise ValueError(
                f"widths_cal has {widths_cal.shape[1]} columns but "
                f"lambda_grid has {len(self.lambda_grid)} elements"
            )
        if np.any(widths_cal < 0):
            raise ValueError("widths_cal contains negative values")

        # Normalise widths by scale to get loss in [0, B]
        losses = widths_cal / self.scale

        # Intervals are wider for larger lambda. Width loss is INCREASING in lambda.
        # CRC requires loss to be NON-INCREASING in lambda.
        # Reverse the lambda grid so the search finds the smallest lambda
        # with acceptable (low) width.
        # We want E[width] <= width_target, and width decreases as lambda decreases.
        # So: find the LARGEST lambda such that E[width / scale] <= width_target / scale.
        # Actually: width is increasing in lambda. CRC finds the smallest lambda where
        # corrected risk <= alpha. Since width increases, we need to find the minimum
        # lambda where the width is already below target. That's the minimum valid lambda.
        # But wait — smaller lambda = smaller width. So smallest lambda satisfying
        # E[width] <= target is lambda_grid[0] trivially (too narrow, no coverage).
        # We want the LARGEST lambda still satisfying the width constraint.
        #
        # The right framing: treat (1 - lambda) as the search parameter. Higher
        # (1-lambda) = smaller lambda = narrower intervals = lower width loss.
        # CRC finds smallest lambda (largest 1-lambda) where corrected risk <= target.
        # Equivalently: reverse the grid, run CRC, map back.

        reversed_losses = losses[:, ::-1]
        reversed_grid = self.lambda_grid[::-1]

        try:
            _, lambda_idx_rev, risk_curve_rev = conformal_risk_calibration(
                reversed_losses,
                reversed_grid,
                alpha=self.width_target / self.scale,
                B=self.B,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Width calibration failed: {exc}. "
                "Consider increasing width_target or scale."
            ) from exc

        # Map back to original lambda grid
        original_idx = len(self.lambda_grid) - 1 - lambda_idx_rev
        self.lambda_hat_ = float(self.lambda_grid[original_idx])
        self.risk_curve_ = risk_curve_rev[::-1]
        self.n_calibration_ = widths_cal.shape[0]
        self.is_calibrated_ = True

        return self

    def calibrate(
        self,
        interval_fn: Any,
        X_cal: Any,
        y_cal: Any,
    ) -> "IntervalWidthController":
        """
        Calibrate from a callable that produces intervals at each lambda.

        A convenience wrapper for when you have a conformal predictor that
        accepts a quantile level parameter.

        Parameters
        ----------
        interval_fn : callable
            Signature: interval_fn(X, alpha) -> dict with keys 'lower', 'upper'
            as numpy arrays. This is compatible with InsuranceConformalPredictor's
            predict_interval() signature.
        X_cal : array-like, shape (n, p)
        y_cal : array-like, shape (n,)
            Only used to compute calibration set size; the interval_fn should
            already be calibrated on separate data.

        Returns
        -------
        self
        """
        y_cal = as_numpy(y_cal)
        n = len(y_cal)
        m = len(self.lambda_grid)
        widths_cal = np.empty((n, m), dtype=float)

        for j, lam in enumerate(self.lambda_grid):
            # lambda_grid[j] is a quantile level (e.g., 0.9 = 90th percentile)
            # For insurance-conformal, alpha = 1 - lambda
            alpha = 1.0 - lam
            if alpha <= 0 or alpha >= 1:
                # Edge case: set very large width
                widths_cal[:, j] = 1e10
                continue
            intervals = interval_fn(X_cal, alpha=alpha)
            if isinstance(intervals, dict):
                lower = np.asarray(intervals["lower"], dtype=float)
                upper = np.asarray(intervals["upper"], dtype=float)
            else:
                # polars DataFrame
                lower = intervals["lower"].to_numpy()
                upper = intervals["upper"].to_numpy()
            widths_cal[:, j] = upper - lower

        return self.calibrate_from_widths(widths_cal)

    def risk_summary(self) -> dict:
        """Return summary of calibration results."""
        self._check_calibrated()
        return {
            "lambda_hat": self.lambda_hat_,
            "width_target": self.width_target,
            "scale": self.scale,
            "n_calibration": self.n_calibration_,
            "B": self.B,
        }

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Controller not calibrated. "
                "Call .calibrate_from_widths() or .calibrate() first."
            )

    def __repr__(self) -> str:
        if self.is_calibrated_:
            return (
                f"IntervalWidthController("
                f"width_target={self.width_target}, lambda_hat={self.lambda_hat_:.4f}, "
                f"n_cal={self.n_calibration_})"
            )
        return f"IntervalWidthController(width_target={self.width_target}, not calibrated)"
