"""
PremiumSufficiencyController: control expected underpricing shortfall.

The problem: your GBM outputs a predicted pure premium p(X). After uplift and
expense loading, you charge some multiple of p(X). How often does the claim
exceed what you expected to pay? And by how much, on average?

Standard conformal prediction answers: 'the claim exceeds the upper bound on
at most alpha% of policies' (coverage control). That says nothing about the
magnitude of the excess.

Conformal risk control answers: 'the expected shortfall — claims above the
upper bound, normalised by premium — is at most alpha'. You control the
financial exposure from underpriced policies, not just the count.

The lambda here is a multiplier on the base premium. lambda=1.2 means the
upper bound is 120% of the predicted pure premium. The calibration finds the
smallest lambda such that E[max(claim - lambda*premium, 0)/premium] <= alpha.

This is the use case actuaries immediately understand: it's a bound on the
expected loss from model error, expressed as a fraction of premium income.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import polars as pl

from insurance_conformal.risk.calibration import (
    conformal_risk_calibration,
    MonotoneLambdaSearch,
)
from insurance_conformal.risk.losses import scaled_shortfall_loss
from insurance_conformal.risk.utils import as_numpy


class PremiumSufficiencyController:
    """
    Distribution-free control of expected premium shortfall.

    Finds the smallest loading factor lambda* such that the expected
    shortfall from underpriced policies is at most alpha:

        E[max(Y - lambda* * premium, 0) / premium] <= alpha

    This is a marginal guarantee over the calibration distribution. It does
    not guarantee that any particular segment (e.g., young drivers) is
    individually controlled — check shortfall_by_decile() for that.

    Parameters
    ----------
    alpha : float
        Target expected shortfall level. alpha=0.05 means expected shortfall
        is at most 5% of premium income. Typical values: 0.02 to 0.10.
    lambda_grid : np.ndarray or None
        Grid of multipliers to search over. If None, defaults to
        np.linspace(0.5, 5.0, 500). Must be sorted ascending. Larger lambda
        = higher upper bound = lower shortfall (monotone).
    B : float
        Upper bound on the normalised shortfall loss (max_claim / min_premium
        in the calibration set). Required — no default. Setting B=1.0 is
        almost always wrong for insurance claims where a single large claim
        can far exceed premium. Use B = np.max(y_cal / premium_cal) or the
        policy limit divided by the minimum premium in your portfolio.

    Attributes
    ----------
    lambda_hat_ : float
        Calibrated multiplier. Set after calibrate().
    risk_curve_ : np.ndarray
        Corrected empirical risk at each grid lambda. Set after calibrate().
    n_calibration_ : int
        Calibration set size.
    is_calibrated_ : bool

    Examples
    --------
    >>> psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
    >>> psc.calibrate(y_cal, premium_cal)
    >>> result = psc.predict(premium_new)
    >>> print(result.head())

    For integration with insurance-conformal intervals::

        >>> from insurance_conformal import InsuranceConformalPredictor
        >>> cp = InsuranceConformalPredictor(model=fitted_gbm)
        >>> cp.calibrate(X_cal, y_cal)
        >>> upper_cal = cp.predict_interval(X_cal)["upper"].to_numpy()
        >>> # Now calibrate risk control on top of conformal intervals
        >>> psc = PremiumSufficiencyController(alpha=0.03, B=5.0)
        >>> psc.calibrate(y_cal, upper_cal)
    """

    def __init__(
        self,
        alpha: float,
        lambda_grid: Optional[np.ndarray] = None,
        B: Optional[float] = None,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if B is None:
            raise ValueError(
                "B must be provided. B is the upper bound on the normalised shortfall loss "
"(max_claim / min_premium in the calibration set). Default B=1.0 is almost always wrong "
"for insurance claims where single large claims can far exceed the premium. "
"Set B = np.max(y_cal / premium_cal) or use the policy limit / minimum premium."
            )
        if B <= 0:
            raise ValueError(f"B must be positive, got {B}")

        self.alpha = alpha
        self.B = B
        self.lambda_grid = (
            np.linspace(0.5, 5.0, 500) if lambda_grid is None
            else np.asarray(lambda_grid, dtype=float)
        )

        if not np.all(np.diff(self.lambda_grid) > 0):
            raise ValueError("lambda_grid must be strictly increasing")

        self.lambda_hat_: Optional[float] = None
        self.risk_curve_: Optional[np.ndarray] = None
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False
        self._premium_cal: Optional[np.ndarray] = None

    def calibrate(
        self,
        y_cal: Any,
        premium_cal: Any,
        exposure: Optional[Any] = None,
    ) -> "PremiumSufficiencyController":
        """
        Calibrate the shortfall controller on held-out claims data.

        Parameters
        ----------
        y_cal : array-like, shape (n,)
            Observed claims on the calibration set. Must be non-negative.
        premium_cal : array-like, shape (n,)
            Base premiums (model outputs or charged premiums) for calibration
            observations. Must be strictly positive. These are the denominators
            in the shortfall loss.
        exposure : array-like, shape (n,) or None
            Policy exposure in years. If provided, loss is computed as the
            annualised claim (y / exposure) vs premium, ensuring that a policy
            with 6 months' exposure is not penalised for having a 'small' claim.
            If None, y is used directly (assumes all policies are full-year).

        Returns
        -------
        self
        """
        y_cal = as_numpy(y_cal)
        premium_cal = as_numpy(premium_cal)

        if np.any(y_cal < 0):
            raise ValueError("y_cal contains negative values. Claims must be non-negative.")
        if np.any(premium_cal <= 0):
            raise ValueError("premium_cal must be strictly positive.")
        if len(y_cal) != len(premium_cal):
            raise ValueError(
                f"y_cal and premium_cal must have the same length, "
                f"got {len(y_cal)} and {len(premium_cal)}"
            )

        if exposure is not None:
            exposure = as_numpy(exposure)
            if np.any(exposure <= 0):
                raise ValueError("exposure must be strictly positive")
            y_cal = y_cal / exposure

        search = MonotoneLambdaSearch(
            loss_fn=scaled_shortfall_loss,
            lambda_grid=self.lambda_grid,
        )

        # compute_losses calls scaled_shortfall_loss(y, lambda_, base_premium=premium_cal)
        losses = search.compute_losses(y_cal, base_premium=premium_cal)

        try:
            lambda_hat, _, risk_curve = conformal_risk_calibration(
                losses, self.lambda_grid, alpha=self.alpha, B=self.B
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Calibration failed: {exc}. "
                f"Consider increasing alpha (current={self.alpha}) or extending "
                f"lambda_grid to cover larger multipliers."
            ) from exc

        self.lambda_hat_ = lambda_hat
        self.risk_curve_ = risk_curve
        self.n_calibration_ = len(y_cal)
        self._premium_cal = premium_cal
        self.is_calibrated_ = True

        return self

    def predict(
        self,
        premium_new: Any,
    ) -> pl.DataFrame:
        """
        Compute risk-controlled upper bounds for new policies.

        Returns a Polars DataFrame with columns:
        - ``base_premium``: input model premium
        - ``upper_bound``: lambda_hat * base_premium (risk-controlled bound)
        - ``safety_loading``: upper_bound / base_premium - 1 (relative uplift)
        - ``lambda_hat``: the calibrated multiplier (same for all rows)

        Parameters
        ----------
        premium_new : array-like, shape (n,)
            Model-predicted premiums for new policies.

        Returns
        -------
        pl.DataFrame
        """
        self._check_calibrated()
        premium_new = as_numpy(premium_new)

        if np.any(premium_new <= 0):
            raise ValueError("premium_new must be strictly positive.")

        upper_bound = self.lambda_hat_ * premium_new
        safety_loading = self.lambda_hat_ - 1.0

        return pl.DataFrame(
            {
                "base_premium": premium_new,
                "upper_bound": upper_bound,
                "safety_loading": np.full(len(premium_new), safety_loading),
                "lambda_hat": np.full(len(premium_new), self.lambda_hat_),
            }
        )

    def shortfall_report(
        self,
        y_cal: Any,
        premium_cal: Any,
        n_deciles: int = 10,
    ) -> pl.DataFrame:
        """
        Per-decile shortfall analysis on the calibration set.

        Shows where in the risk distribution the shortfall is concentrated.
        If high-premium policies have disproportionate shortfall, the marginal
        guarantee is masking a structural segment problem.

        Parameters
        ----------
        y_cal : array-like, shape (n,)
        premium_cal : array-like, shape (n,)
        n_deciles : int, default 10

        Returns
        -------
        pl.DataFrame
            Columns: decile, mean_premium, n_obs, mean_shortfall, max_shortfall,
            pct_policies_underpriced.
        """
        self._check_calibrated()
        y_cal = as_numpy(y_cal)
        premium_cal = as_numpy(premium_cal)
        upper = self.lambda_hat_ * premium_cal
        shortfall = np.maximum(y_cal - upper, 0.0) / premium_cal

        # Decile by premium
        quantiles = np.percentile(premium_cal, np.linspace(0, 100, n_deciles + 1))
        rows = []
        for i in range(n_deciles):
            lo, hi = quantiles[i], quantiles[i + 1]
            mask = (premium_cal >= lo) & (premium_cal <= hi)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "decile": i + 1,
                    "mean_premium": float(premium_cal[mask].mean()),
                    "n_obs": int(mask.sum()),
                    "mean_shortfall": float(shortfall[mask].mean()),
                    "max_shortfall": float(shortfall[mask].max()),
                    "pct_underpriced": float((y_cal[mask] > upper[mask]).mean()),
                }
            )

        return pl.DataFrame(rows)

    def risk_summary(self) -> dict:
        """
        Return a summary dict of calibration results for reporting.

        Returns
        -------
        dict with keys: lambda_hat, alpha, n_calibration, B,
        empirical_risk_at_lambda, lambda_grid_min, lambda_grid_max.
        """
        self._check_calibrated()
        # Find empirical risk at lambda_hat
        idx = np.searchsorted(self.lambda_grid, self.lambda_hat_)
        empirical_risk = float(self.risk_curve_[min(idx, len(self.risk_curve_) - 1)])

        return {
            "lambda_hat": self.lambda_hat_,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration_,
            "B": self.B,
            "corrected_risk_at_lambda": empirical_risk,
            "lambda_grid_min": float(self.lambda_grid[0]),
            "lambda_grid_max": float(self.lambda_grid[-1]),
        }

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Controller has not been calibrated. "
                "Call .calibrate(y_cal, premium_cal) first."
            )

    def __repr__(self) -> str:
        if self.is_calibrated_:
            return (
                f"PremiumSufficiencyController("
                f"alpha={self.alpha}, lambda_hat={self.lambda_hat_:.4f}, "
                f"n_cal={self.n_calibration_})"
            )
        return f"PremiumSufficiencyController(alpha={self.alpha}, not calibrated)"
