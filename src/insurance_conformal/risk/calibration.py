"""
Core conformal risk control calibration algorithm.

Implements Algorithm 1 from Angelopoulos, Bates, Fisch, Lei & Schuster (2024)
'Conformal Risk Control', ICLR 2024. arXiv:2208.02814.

The key result (Theorem 2.1): under exchangeability, if L_i(lambda) is
non-increasing in lambda and bounded in [0, B], then lambda_hat satisfies:

    E[L_{n+1}(lambda_hat)] <= alpha

The finite-sample correction is (n/(n+1)) * R_hat_n(lambda) + B/(n+1) <= alpha.
This is not optional. Dropping the correction produces invalid guarantees for
small calibration sets (n < 500 matters in practice).

The algorithm is trivially simple — 5 lines of numpy. The value of this
implementation is in the correct insurance-specific loss functions, the
monotone search helper, and the reporting infrastructure around it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def conformal_risk_calibration(
    losses: np.ndarray,
    lambdas: np.ndarray,
    alpha: float,
    B: float = 1.0,
) -> tuple[float, int, np.ndarray]:
    """
    Find the minimum lambda that controls expected risk at level alpha.

    Implements Algorithm 1 of Angelopoulos et al. (2024). The empirical
    risk is computed at each lambda value, then corrected for finite-sample
    bias before comparing to alpha.

    Parameters
    ----------
    losses : np.ndarray, shape (n, m) or (n,)
        Per-sample losses at each lambda value. If shape (n, m), column j
        corresponds to lambdas[j]. If shape (n,), treated as pre-computed
        losses at a single lambda (for single-point queries).
    lambdas : np.ndarray, shape (m,)
        Grid of lambda values, sorted in increasing order. Larger lambda
        must correspond to lower or equal loss (monotonicity requirement).
    alpha : float
        Target risk level. Must be in (0, B).
    B : float, default 1.0
        Upper bound on the loss function. The shortfall loss is bounded
        by 1.0 when normalised by premium. Width losses may need a larger
        B — set it to the maximum possible loss value.

    Returns
    -------
    lambda_hat : float
        Calibrated lambda. The smallest lambda such that the corrected
        empirical risk is at most alpha.
    lambda_idx : int
        Index into lambdas array.
    risk_curve : np.ndarray, shape (m,)
        Corrected empirical risk at each lambda. Useful for diagnostic plots.

    Raises
    ------
    ValueError
        If alpha is outside (0, B), or losses shape is inconsistent with lambdas.
    RuntimeError
        If no lambda controls risk at alpha (alpha too tight for this calibration
        set). Caller should catch this and expand the lambda grid or loosen alpha.

    Notes
    -----
    The correction (n/(n+1)) * R_hat + B/(n+1) <= alpha is the exact finite-
    sample guarantee from Theorem 2.1. It inflates the empirical risk slightly
    to account for the unseen test point. For large n (>10000) this correction
    is negligible, but for insurance calibration sets (n=500-2000) it matters.

    Examples
    --------
    >>> lambdas = np.linspace(0, 1, 200)
    >>> # losses[i, j] = loss for observation i at lambda j
    >>> losses = np.random.rand(500, 200) * (1 - lambdas)  # decreasing in lambda
    >>> lhat, idx, curve = conformal_risk_calibration(losses, lambdas, alpha=0.05)
    """
    if not (0 < alpha < B):
        raise ValueError(
            f"alpha must be in (0, B) = (0, {B}), got alpha={alpha}. "
            "If your loss is bounded by a value other than 1, pass B explicitly."
        )

    losses = np.asarray(losses, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)

    if losses.ndim == 1:
        if len(lambdas) != 1:
            raise ValueError(
                "losses is 1D (single lambda) but lambdas has multiple values. "
                "Pass losses as shape (n, m) for multiple lambda values."
            )
        losses = losses[:, np.newaxis]

    n, m = losses.shape
    if m != len(lambdas):
        raise ValueError(
            f"losses has {m} columns but lambdas has {len(lambdas)} elements. "
            "Each column of losses must correspond to one lambda value."
        )

    # Equation (1) from Algorithm 1: corrected empirical risk
    empirical_risk = losses.mean(axis=0)
    corrected_risk = (n / (n + 1)) * empirical_risk + B / (n + 1)

    # Find smallest lambda with corrected risk <= alpha
    valid = np.where(corrected_risk <= alpha)[0]

    if len(valid) == 0:
        raise RuntimeError(
            f"No lambda in the grid controls expected risk at alpha={alpha}. "
            f"Minimum corrected risk is {corrected_risk.min():.4f}. "
            "Either increase alpha, expand the lambda grid, or increase n_cal."
        )

    lambda_idx = int(valid[0])
    lambda_hat = float(lambdas[lambda_idx])

    return lambda_hat, lambda_idx, corrected_risk


class MonotoneLambdaSearch:
    """
    Vectorised loss computation over a lambda grid.

    Handles the bookkeeping of evaluating a per-sample loss function at
    many lambda values efficiently. The loss function must be non-increasing
    in lambda: larger lambda = more conservative prediction = lower loss.

    Parameters
    ----------
    loss_fn : callable
        Signature: loss_fn(y, predictions_at_lambda) -> np.ndarray of shape (n,).
        predictions_at_lambda is the model output adjusted by lambda. For
        premium sufficiency, this might be lambda * base_premium. For interval
        width, lambda might be the quantile level.
    lambda_grid : np.ndarray
        Grid of candidate lambda values. Should cover the full range from
        near-zero (aggressive) to near-maximum (conservative).

    Examples
    --------
    >>> def shortfall(y, upper): return np.maximum(y - upper, 0) / upper
    >>> search = MonotoneLambdaSearch(loss_fn=shortfall, lambda_grid=np.linspace(1, 3, 100))
    >>> # losses[i, j] = shortfall for obs i at upper = lambdas[j] * base_premium[i]
    >>> losses = search.compute_losses(y_cal, base_premiums=premium_cal)
    """

    def __init__(
        self,
        loss_fn: Callable,
        lambda_grid: np.ndarray,
    ) -> None:
        self.loss_fn = loss_fn
        self.lambda_grid = np.asarray(lambda_grid, dtype=float)

    def compute_losses(
        self,
        y: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Evaluate the loss function at every lambda in the grid.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Observed outcomes.
        **kwargs
            Additional arrays passed through to loss_fn at each lambda.
            For example, base_premiums=premium_cal.

        Returns
        -------
        np.ndarray, shape (n, m)
            losses[i, j] = loss for observation i at lambda_grid[j].
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        m = len(self.lambda_grid)
        losses = np.empty((n, m), dtype=float)

        for j, lam in enumerate(self.lambda_grid):
            losses[:, j] = self.loss_fn(y, lam, **kwargs)

        return losses

    def calibrate(
        self,
        y: np.ndarray,
        alpha: float,
        B: float = 1.0,
        **kwargs,
    ) -> tuple[float, np.ndarray]:
        """
        Compute losses and run conformal_risk_calibration in one call.

        Returns
        -------
        lambda_hat : float
        risk_curve : np.ndarray, shape (m,)
        """
        losses = self.compute_losses(y, **kwargs)
        lambda_hat, _, risk_curve = conformal_risk_calibration(
            losses, self.lambda_grid, alpha=alpha, B=B
        )
        return lambda_hat, risk_curve
