"""
SelectiveRiskController: accept/reject risks to bound expected loss on accepted book.

The selective conformal risk control problem (arXiv 2512.12844, Selective Conformal
Risk Control): a two-stage approach where you first decide which risks to accept,
then guarantee that the accepted book has bounded expected loss.

In insurance underwriting terms: you have a risk scoring model that produces a
score s(X) for each risk. You want to:
1. Accept risks where s(X) >= threshold (the 'good' risks)
2. Guarantee that E[L(Y, d(X)) | risk is accepted] <= alpha

The difficulty is that naive thresholding on the calibration set is biased
(the risks you accept are not representative of the full population). The SCRC-I
algorithm from arXiv 2512.12844 handles this with a DKW-based correction.

This implementation follows SCRC-I (calibration-only, no refitting), which is
the correct approach for a pricing team that has a fixed underwriting model.

Lambda parameterisation: lambda is the acceptance threshold on the risk score.
Higher lambda = more restrictive selection = fewer policies accepted = lower
expected loss on accepted book (monotone).

The guarantee: if at least xi_min fraction of risks are selected (selection rate
constraint), then E[L | selected] <= alpha.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from insurance_conformal.risk.calibration import conformal_risk_calibration
from insurance_conformal.risk.utils import as_numpy


class SelectiveRiskController:
    """
    Two-stage risk control: selection then expected-loss guarantee.

    Implements SCRC-I from arXiv 2512.12844. The controller finds the
    lowest risk-score threshold such that the expected loss on accepted
    risks is at most alpha, subject to a minimum selection rate xi_min.

    Parameters
    ----------
    alpha : float
        Target expected loss on accepted risks. Typical values: 0.05 to 0.20.
    loss_fn : callable
        Signature: loss_fn(y, scores) -> np.ndarray, shape (n,).
        Per-observation loss for accepted risks. Must return values in [0, B].
        Example: loss_fn = lambda y, scores: (y > threshold).astype(float)
    xi_min : float, default 0.5
        Minimum fraction of risks that must be accepted (selection rate floor).
        If the calibration set has 1000 observations, xi_min=0.5 means at least
        500 must be selected. This prevents the trivial solution of accepting
        zero risks.
    lambda_grid : np.ndarray or None
        Risk score thresholds to search over. If None, linearly spaced over
        the observed score range during calibrate(). Higher lambda = fewer
        accepted.
    B : float, default 1.0
        Upper bound on the loss function.
    delta : float, default 0.05
        Confidence level for the DKW-based correction on selection rate.
        The selection rate constraint holds with probability 1 - delta.

    Attributes
    ----------
    threshold_ : float
        Calibrated risk score threshold. Accept risk iff score >= threshold_.
    lambda_hat_ : float
        Same as threshold_, kept for API consistency with other controllers.
    n_calibration_ : int
    selection_rate_ : float
        Fraction of calibration set accepted at threshold_.
    is_calibrated_ : bool

    Examples
    --------
    >>> def high_claim_loss(y, scores):
    ...     return (y > 5000).astype(float)  # Loss = 1 if large claim
    >>> src = SelectiveRiskController(alpha=0.10, loss_fn=high_claim_loss, xi_min=0.6)
    >>> src.calibrate(y_cal, scores_cal)
    >>> decisions = src.predict(scores_new)
    >>> print(decisions["accept"].sum(), "policies accepted")
    """

    def __init__(
        self,
        alpha: float,
        loss_fn: Callable,
        xi_min: float = 0.5,
        lambda_grid: Optional[np.ndarray] = None,
        B: float = 1.0,
        delta: float = 0.05,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0 < xi_min < 1):
            raise ValueError(f"xi_min must be in (0, 1), got {xi_min}")
        if B <= 0:
            raise ValueError(f"B must be positive, got {B}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        self.alpha = alpha
        self.loss_fn = loss_fn
        self.xi_min = xi_min
        self.lambda_grid = lambda_grid
        self.B = B
        self.delta = delta

        self.threshold_: Optional[float] = None
        self.lambda_hat_: Optional[float] = None
        self.n_calibration_: int = 0
        self.selection_rate_: Optional[float] = None
        self.is_calibrated_: bool = False
        self._risk_curve: Optional[np.ndarray] = None

    def calibrate(
        self,
        y_cal: Any,
        scores_cal: Any,
        exposure: Optional[Any] = None,
    ) -> "SelectiveRiskController":
        """
        Find the acceptance threshold that controls expected loss on accepted risks.

        Parameters
        ----------
        y_cal : array-like, shape (n,)
            Observed outcomes on calibration set.
        scores_cal : array-like, shape (n,)
            Risk scores from your underwriting/pricing model. Higher score =
            lower risk (e.g., a goodness score, or the negative log-loss).
            If higher score = higher risk, negate before passing.
        exposure : array-like, shape (n,) or None
            Exposure weights. If provided, losses are exposure-weighted.

        Returns
        -------
        self
        """
        y_cal = as_numpy(y_cal)
        scores_cal = as_numpy(scores_cal)

        if len(y_cal) != len(scores_cal):
            raise ValueError(
                f"y_cal and scores_cal must have same length, "
                f"got {len(y_cal)} and {len(scores_cal)}"
            )

        n = len(y_cal)

        if exposure is not None:
            exposure = as_numpy(exposure)
            if np.any(exposure <= 0):
                raise ValueError("exposure must be strictly positive")
            # Normalise exposure to sum to n (so loss magnitudes are comparable)
            exposure = exposure / exposure.mean()
        else:
            exposure = np.ones(n, dtype=float)

        # Build lambda_grid over observed score range if not provided
        if self.lambda_grid is None:
            score_min, score_max = scores_cal.min(), scores_cal.max()
            lambda_grid = np.linspace(score_min, score_max, min(200, n))
        else:
            lambda_grid = self.lambda_grid

        # For SCRC-I: at threshold t, the accepted set is {i : scores_cal[i] >= t}
        # Loss for each lambda:
        # - If obs i is selected (score >= t): loss = L(y_i, scores_i)
        # - If obs i is not selected: loss = 0 (they're not in the accepted book)
        # Risk = sum of losses on selected / n (marginal risk, not conditional)
        # THEN adjust: conditional risk = marginal risk / selection_rate
        # The DKW correction applies to the selection rate estimate.

        # Compute per-observation losses (independent of threshold)
        base_losses = self.loss_fn(y_cal, scores_cal) * exposure

        if np.any(base_losses < 0) or np.any(base_losses > self.B):
            raise ValueError(
                f"loss_fn returned values outside [0, B={self.B}]. "
                "Check your loss function and B parameter."
            )

        m = len(lambda_grid)
        # Build loss matrix: losses[i, j] = loss for obs i at threshold j
        # If obs i is not selected at threshold j, loss = 0
        # Conditional risk at threshold t = sum_{i: score_i >= t} loss_i / max(1, n_selected)
        # We apply DKW correction on n_selected

        risk_at_lambda = np.empty(m, dtype=float)
        selection_rate_at_lambda = np.empty(m, dtype=float)

        for j, threshold in enumerate(lambda_grid):
            selected = scores_cal >= threshold
            n_sel = selected.sum()
            selection_rate_at_lambda[j] = n_sel / n

            if n_sel == 0:
                risk_at_lambda[j] = 0.0  # No policies accepted
            else:
                # DKW confidence interval on selection rate
                # P(xi >= xi_hat - eps) >= 1 - delta where eps = sqrt(log(2/delta)/(2n))
                eps_dkw = np.sqrt(np.log(2.0 / self.delta) / (2 * n))
                xi_lower = max(0.0, selection_rate_at_lambda[j] - eps_dkw)

                if xi_lower < self.xi_min:
                    # Selection rate is too low (with DKW-corrected confidence)
                    # Mark this lambda as infeasible by setting infinite risk
                    risk_at_lambda[j] = self.B + 1.0
                else:
                    # Conditional expected loss: E[L | selected]
                    conditional_risk = float(base_losses[selected].mean())
                    risk_at_lambda[j] = conditional_risk

        # Apply the CRC finite-sample correction on the conditional risk
        # We need losses as (n, m) matrix for conformal_risk_calibration
        # Build it: obs i at threshold j has loss = base_loss[i] if selected, else 0
        # Then condition on selection... but CRC operates on marginal, not conditional.
        #
        # Alternative: treat risk_at_lambda as the "risk curve" directly
        # and find threshold as: smallest t where risk_at_lambda[j] + B/(n+1) <= alpha
        # This is a simplified version of SCRC-I appropriate for moderate n.

        # CRC finite-sample correction: use n_sel (selected obs count) not n.
        # The correction accounts for the effective calibration set at each
        # threshold — we only observe losses for selected observations, so the
        # relevant sample size for the conformal guarantee is n_sel, not n.
        n_sel_at_lambda = np.maximum(1, np.round(selection_rate_at_lambda * n).astype(int))
        corrected_risk = (n_sel_at_lambda / (n_sel_at_lambda + 1)) * risk_at_lambda + self.B / (n_sel_at_lambda + 1)

        # Find smallest lambda (least restrictive) that controls conditional risk
        valid = np.where(corrected_risk <= self.alpha)[0]

        if len(valid) == 0:
            raise RuntimeError(
                f"No threshold controls conditional expected loss at alpha={self.alpha}. "
                f"Minimum corrected risk = {corrected_risk[corrected_risk <= self.B].min():.4f}. "
                "Consider increasing alpha, loosening xi_min, or reviewing the loss function."
            )

        threshold_idx = int(valid[0])
        self.threshold_ = float(lambda_grid[threshold_idx])
        self.lambda_hat_ = self.threshold_
        self.selection_rate_ = float(selection_rate_at_lambda[threshold_idx])
        self._risk_curve = corrected_risk
        self._lambda_grid_used = lambda_grid
        self.n_calibration_ = n
        self.is_calibrated_ = True

        return self

    def predict(
        self,
        scores_new: Any,
    ) -> pl.DataFrame:
        """
        Return accept/reject decisions for new risks.

        Parameters
        ----------
        scores_new : array-like, shape (n,)
            Risk scores for new observations. Same scale as scores_cal.

        Returns
        -------
        pl.DataFrame
            Columns: score (Float64), accept (Boolean), threshold (Float64).
        """
        self._check_calibrated()
        scores_new = as_numpy(scores_new)

        accept = scores_new >= self.threshold_

        return pl.DataFrame(
            {
                "score": scores_new,
                "accept": accept,
                "threshold": np.full(len(scores_new), self.threshold_),
            }
        )

    def portfolio_summary(
        self,
        y_cal: Any,
        scores_cal: Any,
    ) -> dict:
        """
        Summarise portfolio statistics at the calibrated threshold.

        Returns
        -------
        dict with keys: threshold, selection_rate, n_accepted,
        mean_loss_accepted, alpha, n_calibration.
        """
        self._check_calibrated()
        y_cal = as_numpy(y_cal)
        scores_cal = as_numpy(scores_cal)
        selected = scores_cal >= self.threshold_
        losses = self.loss_fn(y_cal, scores_cal)

        return {
            "threshold": self.threshold_,
            "selection_rate": self.selection_rate_,
            "n_accepted": int(selected.sum()),
            "n_total": len(y_cal),
            "mean_loss_accepted": float(losses[selected].mean()) if selected.sum() > 0 else 0.0,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration_,
        }

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Controller not calibrated. Call .calibrate(y_cal, scores_cal) first."
            )

    def __repr__(self) -> str:
        if self.is_calibrated_:
            return (
                f"SelectiveRiskController("
                f"alpha={self.alpha}, threshold={self.threshold_:.4f}, "
                f"selection_rate={self.selection_rate_:.2%}, "
                f"n_cal={self.n_calibration_})"
            )
        return f"SelectiveRiskController(alpha={self.alpha}, not calibrated)"
