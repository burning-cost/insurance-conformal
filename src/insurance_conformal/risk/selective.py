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



# ---------------------------------------------------------------------------
# SelectiveConformalRC — two-stage selective conformal risk control
# Xu, Guo & Wei (2025), arXiv:2512.12844, "Selective Conformal Risk Control"
# ---------------------------------------------------------------------------

import math
from dataclasses import dataclass, field


class InfeasibleSCRCError(Exception):
    """
    Raised when no (lambda_1, lambda_2) pair satisfies the SCRC constraints.

    This can happen when:
    - The minimum selection rate xi is too high for the score distribution
    - Alpha is too tight for the calibration set size
    - The DKW correction eps is large relative to xi (small n)

    Catch this error and either loosen xi, increase alpha, or collect more
    calibration data.
    """
    pass


@dataclass
class SCRCResult:
    """
    Result from SelectiveConformalRC calibration.

    All fields are populated after a successful calibrate() call. If
    calibrate() raises InfeasibleSCRCError, this object is never returned.

    Fields
    ------
    method : str
        'SCRC-I' (calibration-only, for real-time deployment) or
        'SCRC-T' (transductive, includes test point, for batch audits).
    lambda_1 : float
        First-stage selection threshold. Accept risk if selection_score >= lambda_1.
    lambda_2 : float
        Second-stage risk threshold. Conformal threshold on second-stage loss.
        For SCRC-I: the smallest lambda_2 such that the augmented sum of
        losses on selected calibration observations is within budget.
    selection_rate : float
        Fraction of calibration set accepted at lambda_1.
    conditional_risk : float
        Empirical conditional expected loss on the selected calibration subset.
    expected_set_size : float
        Mean loss on calibration points selected by stage one.
    n_calibration : int
        Number of calibration points used.
    n_selected : int
        Number of calibration points accepted by stage one.
    xi_lcb : float
        DKW lower confidence bound on the selection rate (SCRC-I only).
        xi_lcb = max(xi_hat - eps, 0) where eps = sqrt(log(2/delta) / (2*n)).
    pareto_frontier : list[dict]
        Pareto-optimal (lambda_1, lambda_2) pairs. Empty unless
        run_grid_search=True was passed to SelectiveConformalRC.
    feasible : bool
        True if calibration succeeded.
    infeasibility_reason : str or None
        Human-readable reason if feasible=False. None if feasible=True.
    """
    method: str
    lambda_1: float
    lambda_2: float
    selection_rate: float
    conditional_risk: float
    expected_set_size: float
    n_calibration: int
    n_selected: int
    xi_lcb: float
    pareto_frontier: list = field(default_factory=list)
    feasible: bool = True
    infeasibility_reason: Optional[str] = None


class SelectiveConformalRC:
    """
    Two-stage selective conformal risk control for insurance underwriting triage.

    Implements SCRC-I (calibration-only) and SCRC-T (transductive) from
    Xu, Guo & Wei (2025), arXiv:2512.12844.

    The two-stage structure:
    - Stage 1: selection gate. Accept risks with selection_score >= lambda_1.
    - Stage 2: risk control. For accepted risks, guarantee E[L | selected] <= alpha.

    Parameters
    ----------
    alpha : float
        Target expected loss on accepted risks. E[L | selected] <= alpha.
    xi : float
        Minimum required selection rate (fraction of risks accepted).
    delta : float, default 0.1
        Confidence level for SCRC-I DKW correction.
    B : float, default 1.0
        Upper bound on the second-stage loss function.
    method : str, default 'SCRC-I'
        'SCRC-I' for calibration-only (for production deployment).
        'SCRC-T' for transductive (requires test set at calibration time).
    lambda_1_grid : np.ndarray or None
        Grid of first-stage thresholds. If None, computed at calibrate() time.
    lambda_2_grid : np.ndarray or None
        Grid of second-stage thresholds. If None, uses row indices.
    run_grid_search : bool, default False
        If True, compute and store the Pareto frontier.

    Attributes
    ----------
    result_ : SCRCResult
        Populated after calibrate().
    is_calibrated_ : bool

    Examples
    --------
    >>> from insurance_conformal.risk import SelectiveConformalRC, build_pred_set_matrix
    >>> from insurance_conformal.risk.selection_scores import selection_score_msp
    >>>
    >>> scores_cal = selection_score_msp(model.predict_proba(X_cal))
    >>> pred_sets = build_pred_set_matrix(y_cal, lambda_2_grid, loss_fn)
    >>> scrc = SelectiveConformalRC(alpha=0.10, xi=0.80)
    >>> scrc.calibrate(y_cal, scores_cal, pred_sets)
    >>> decisions = scrc.predict(scores_new)
    """

    def __init__(
        self,
        alpha: float,
        xi: float,
        delta: float = 0.1,
        B: float = 1.0,
        method: str = "SCRC-I",
        lambda_1_grid: Optional[np.ndarray] = None,
        lambda_2_grid: Optional[np.ndarray] = None,
        run_grid_search: bool = False,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0 < xi < 1):
            raise ValueError(f"xi must be in (0, 1), got {xi}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if B <= 0:
            raise ValueError(f"B must be positive, got {B}")
        if alpha >= B:
            raise ValueError(f"alpha must be < B. Got alpha={alpha}, B={B}")
        if method not in ("SCRC-I", "SCRC-T"):
            raise ValueError(f"method must be 'SCRC-I' or 'SCRC-T', got '{method}'")

        self.alpha = alpha
        self.xi = xi
        self.delta = delta
        self.B = B
        self.method = method
        self.lambda_1_grid = lambda_1_grid
        self.lambda_2_grid = lambda_2_grid
        self.run_grid_search = run_grid_search

        self.result_: Optional[SCRCResult] = None
        self.is_calibrated_: bool = False
        self._lambda_1_grid_used: Optional[np.ndarray] = None
        self._lambda_2_grid_used: Optional[np.ndarray] = None

    def calibrate(
        self,
        y_cal: Any,
        scores_cal: Any,
        pred_sets_cal: Any,
        lambda_2_grid: Optional[Any] = None,
        x_test: Optional[Any] = None,
        score_test: Optional[Any] = None,
    ) -> "SelectiveConformalRC":
        """
        Calibrate both stages of the selective conformal risk controller.

        Parameters
        ----------
        y_cal : array-like, shape (n,)
            Calibration outcomes. Used only indirectly (already encoded in
            pred_sets_cal losses).
        scores_cal : array-like, shape (n,)
            First-stage selection scores for calibration observations.
        pred_sets_cal : array-like, shape (m, n)
            Second-stage loss matrix. pred_sets_cal[j, i] = loss for observation i
            at second-stage threshold j. Must be in [0, B]. Must be non-increasing
            along axis 0 (increasing lambda_2 = lower loss).
        lambda_2_grid : array-like or None
            Grid of second-stage threshold values corresponding to rows of
            pred_sets_cal. If provided, overrides the lambda_2_grid constructor
            argument. If None, uses row indices.
        x_test : array-like or None
            Covariates for test points (SCRC-T only). Currently reserved.
        score_test : array-like or None
            First-stage selection scores for test points (SCRC-T only).

        Returns
        -------
        self

        Raises
        ------
        InfeasibleSCRCError
            If no (lambda_1, lambda_2) pair satisfies the constraints.
        """
        scores_cal = as_numpy(scores_cal)
        pred_sets_cal = np.asarray(pred_sets_cal, dtype=float)
        n = len(scores_cal)

        if pred_sets_cal.ndim != 2:
            raise ValueError(
                f"pred_sets_cal must be 2D (m, n), got shape {pred_sets_cal.shape}"
            )

        m, n_check = pred_sets_cal.shape
        if n_check != n:
            raise ValueError(
                f"pred_sets_cal has {n_check} columns but scores_cal has {n} elements. "
                "pred_sets_cal shape must be (len(lambda_2_grid), len(scores_cal))."
            )

        # Build lambda_1 grid from observed score range if not provided
        if self.lambda_1_grid is None:
            score_min = scores_cal.min()
            score_max = scores_cal.max()
            lambda_1_grid = np.linspace(score_min, score_max, min(100, n))
        else:
            lambda_1_grid = np.asarray(self.lambda_1_grid, dtype=float)

        # Resolve lambda_2_grid: method arg > constructor arg > index
        if lambda_2_grid is not None:
            lam2_grid = np.asarray(lambda_2_grid, dtype=float)
            if len(lam2_grid) != m:
                raise ValueError(
                    f"lambda_2_grid has {len(lam2_grid)} elements but "
                    f"pred_sets_cal has {m} rows."
                )
        elif self.lambda_2_grid is not None:
            lam2_grid = np.asarray(self.lambda_2_grid, dtype=float)
            if len(lam2_grid) != m:
                raise ValueError(
                    f"lambda_2_grid has {len(lam2_grid)} elements but "
                    f"pred_sets_cal has {m} rows."
                )
        else:
            lam2_grid = np.arange(m, dtype=float)

        self._lambda_1_grid_used = lambda_1_grid
        self._lambda_2_grid_used = lam2_grid

        if self.method == "SCRC-I":
            result = self._calibrate_scrc_i(
                scores_cal, pred_sets_cal, lambda_1_grid, lam2_grid, n
            )
        else:
            if score_test is None:
                raise ValueError(
                    "score_test must be provided for SCRC-T (transductive method). "
                    "Pass the selection scores for your test batch."
                )
            score_test_arr = as_numpy(score_test)
            result = self._calibrate_scrc_t(
                scores_cal, pred_sets_cal, lambda_1_grid, lam2_grid, n, score_test_arr
            )

        self.result_ = result
        self.is_calibrated_ = True
        return self

    def _calibrate_scrc_i(
        self,
        scores_cal: np.ndarray,
        pred_sets_cal: np.ndarray,
        lambda_1_grid: np.ndarray,
        lambda_2_grid: np.ndarray,
        n: int,
    ) -> SCRCResult:
        """
        SCRC-I: calibration-only, PAC-style guarantee (Algorithm 1 of arXiv:2512.12844).

        1. Find smallest lambda_1 such that hat_xi(lambda_1) >= xi.
        2. DKW lower bound: xi_lcb = max(hat_xi - eps, 0).
        3. Budget k = ceil((n+1) * alpha * xi_lcb) - 1.
        4. Find smallest lambda_2 where sum of losses on selected <= budget k.
        """
        eps_dkw = math.sqrt(math.log(2.0 / self.delta) / (2 * n))

        # Find smallest lambda_1 (least restrictive) where selection rate >= xi.
        # lambda_1_grid is increasing, so higher index = more restrictive.
        # As lambda_1 increases, fewer risks are selected (rate decreases).
        # We iterate in increasing order and take the first feasible lambda_1.
        lambda_hat_1 = None

        for lam1 in lambda_1_grid:
            selected_mask = scores_cal >= lam1
            xi_hat = float(selected_mask.sum()) / n
            if xi_hat >= self.xi:
                lambda_hat_1 = lam1
                break

        if lambda_hat_1 is None:
            max_rate = float((scores_cal >= lambda_1_grid[0]).sum()) / n
            raise InfeasibleSCRCError(
                f"No lambda_1 in grid achieves selection rate >= xi={self.xi}. "
                f"Maximum achievable selection rate: {max_rate:.3f}. "
                "Reduce xi, expand the score range, or use a less restrictive model."
            )

        selected_mask = scores_cal >= lambda_hat_1
        n_selected = int(selected_mask.sum())
        xi_hat = float(n_selected) / n

        # DKW lower confidence bound
        xi_lcb = max(xi_hat - eps_dkw, 0.0)

        if xi_lcb == 0.0:
            raise InfeasibleSCRCError(
                f"DKW-corrected selection rate xi_lcb=0 (xi_hat={xi_hat:.3f}, "
                f"eps={eps_dkw:.3f}). "
                "Calibration set too small or xi too high. "
                "Increase n, reduce xi, or increase delta."
            )

        # Budget
        budget = math.ceil((n + 1) * self.alpha * xi_lcb) - 1

        if budget < 0:
            raise InfeasibleSCRCError(
                f"Budget k={budget} < 0 with n={n}, alpha={self.alpha}, "
                f"xi_lcb={xi_lcb:.4f}. Increase n or relax alpha."
            )

        # Second stage: find smallest lambda_2 where sum of selected losses <= budget
        lambda_hat_2 = None
        empirical_risk = None
        expected_set_size = None

        for j, lam2 in enumerate(lambda_2_grid):
            losses_selected = pred_sets_cal[j, selected_mask]
            if losses_selected.sum() <= budget:
                lambda_hat_2 = lam2
                empirical_risk = float(losses_selected.mean()) if n_selected > 0 else 0.0
                expected_set_size = empirical_risk
                break

        if lambda_hat_2 is None:
            min_sum = min(pred_sets_cal[j, selected_mask].sum() for j in range(len(lambda_2_grid)))
            raise InfeasibleSCRCError(
                f"No lambda_2 achieves sum of losses <= budget={budget} "
                f"on {n_selected} selected observations. "
                f"Minimum loss sum across grid: {min_sum:.2f}. "
                "Expand lambda_2_grid, increase alpha, or reduce xi."
            )

        pareto_frontier = []
        if self.run_grid_search:
            pareto_frontier = self._compute_pareto_frontier(
                scores_cal, pred_sets_cal, lambda_1_grid, lambda_2_grid, n, eps_dkw
            )

        return SCRCResult(
            method="SCRC-I",
            lambda_1=float(lambda_hat_1),
            lambda_2=float(lambda_hat_2),
            selection_rate=xi_hat,
            conditional_risk=float(empirical_risk),
            expected_set_size=float(expected_set_size),
            n_calibration=n,
            n_selected=n_selected,
            xi_lcb=xi_lcb,
            pareto_frontier=pareto_frontier,
            feasible=True,
            infeasibility_reason=None,
        )

    def _compute_pareto_frontier(
        self,
        scores_cal: np.ndarray,
        pred_sets_cal: np.ndarray,
        lambda_1_grid: np.ndarray,
        lambda_2_grid: np.ndarray,
        n: int,
        eps_dkw: float,
    ) -> list:
        """
        Enumerate feasible (lambda_1, lambda_2) pairs and return Pareto frontier.

        Pareto-optimal: no other feasible pair has both lower expected_set_size
        and higher selection_rate. Sorted by selection_rate descending.
        """
        candidates = []

        for lam1 in lambda_1_grid:
            selected_mask = scores_cal >= lam1
            n_sel = int(selected_mask.sum())
            xi_hat = float(n_sel) / n
            if xi_hat < self.xi:
                continue

            xi_lcb = max(xi_hat - eps_dkw, 0.0)
            if xi_lcb == 0.0:
                continue

            budget = math.ceil((n + 1) * self.alpha * xi_lcb) - 1
            if budget < 0:
                continue

            for j, lam2 in enumerate(lambda_2_grid):
                losses_sel = pred_sets_cal[j, selected_mask]
                if losses_sel.sum() <= budget:
                    ess = float(losses_sel.mean()) if n_sel > 0 else 0.0
                    candidates.append({
                        "lambda_1": float(lam1),
                        "lambda_2": float(lam2),
                        "selection_rate": xi_hat,
                        "expected_set_size": ess,
                    })
                    break

        if not candidates:
            return []

        # Pareto filter: sort by selection_rate descending, keep if ESS improves
        candidates.sort(key=lambda d: -d["selection_rate"])
        pareto = []
        best_ess = float("inf")
        for c in candidates:
            if c["expected_set_size"] < best_ess:
                pareto.append(c)
                best_ess = c["expected_set_size"]

        return pareto

    def _calibrate_scrc_t(
        self,
        scores_cal: np.ndarray,
        pred_sets_cal: np.ndarray,
        lambda_1_grid: np.ndarray,
        lambda_2_grid: np.ndarray,
        n: int,
        score_test: np.ndarray,
    ) -> SCRCResult:
        """
        SCRC-T: transductive, includes test points in first-stage threshold.

        For each test point: compute lambda_1 using combined cal+test scores,
        find smallest lambda_2 minimising ESS subject to budget ceil((n+1)*alpha)-1.
        """
        n_test = len(score_test)
        budget_second = math.ceil((n + 1) * self.alpha) - 1

        lambda_1_vals = []
        lambda_2_vals = []
        ess_vals = []

        for score_t in score_test:
            combined_scores = np.concatenate([scores_cal, [float(score_t)]])
            n_combined = n + 1

            lambda_hat_1_t = None
            for lam1 in lambda_1_grid:
                sel = combined_scores >= lam1
                xi_hat = float(sel.sum()) / n_combined
                if xi_hat >= self.xi:
                    lambda_hat_1_t = lam1
                    break

            if lambda_hat_1_t is None:
                continue

            selected_cal_mask = scores_cal >= lambda_hat_1_t

            best_lam2 = None
            best_ess_t = float("inf")

            for j, lam2 in enumerate(lambda_2_grid):
                losses_sel = pred_sets_cal[j, selected_cal_mask]
                if losses_sel.sum() <= budget_second:
                    n_sel_cal = int(selected_cal_mask.sum())
                    ess = float(losses_sel.mean()) if n_sel_cal > 0 else 0.0
                    if ess < best_ess_t:
                        best_ess_t = ess
                        best_lam2 = lam2

            if best_lam2 is None:
                continue

            lambda_1_vals.append(lambda_hat_1_t)
            lambda_2_vals.append(best_lam2)
            ess_vals.append(best_ess_t)

        if not lambda_1_vals:
            raise InfeasibleSCRCError(
                f"SCRC-T: no feasible (lambda_1, lambda_2) pair found for any of "
                f"{n_test} test points. n={n}, xi={self.xi}, alpha={self.alpha}. "
                "Try reducing xi, increasing alpha, or expanding the lambda grids."
            )

        lambda_1_out = float(np.median(lambda_1_vals))
        lambda_2_out = float(np.median(lambda_2_vals))

        selected_cal_mask = scores_cal >= lambda_1_out
        n_sel = int(selected_cal_mask.sum())
        xi_hat = float(n_sel) / n

        # Find the row of lambda_2_grid closest to the median lambda_2
        j_best = int(np.argmin(np.abs(lambda_2_grid - lambda_2_out)))
        losses_sel = pred_sets_cal[j_best, selected_cal_mask]
        empirical_risk = float(losses_sel.mean()) if n_sel > 0 else 0.0
        ess_out = float(np.mean(ess_vals))

        pareto_frontier = []
        if self.run_grid_search:
            pareto_frontier = self._compute_pareto_frontier(
                scores_cal, pred_sets_cal, lambda_1_grid, lambda_2_grid, n,
                eps_dkw=0.0,  # No DKW correction for SCRC-T
            )

        return SCRCResult(
            method="SCRC-T",
            lambda_1=lambda_1_out,
            lambda_2=lambda_2_out,
            selection_rate=xi_hat,
            conditional_risk=empirical_risk,
            expected_set_size=ess_out,
            n_calibration=n,
            n_selected=n_sel,
            xi_lcb=xi_hat,  # No DKW correction for SCRC-T
            pareto_frontier=pareto_frontier,
            feasible=True,
            infeasibility_reason=None,
        )

    def predict(
        self,
        scores_new: Any,
        x_new: Optional[Any] = None,
    ) -> pl.DataFrame:
        """
        Apply the calibrated two-stage controller to new risks.

        Parameters
        ----------
        scores_new : array-like, shape (n_new,)
            First-stage selection scores for new risks.
        x_new : array-like or None
            Reserved for covariate-conditional extensions.

        Returns
        -------
        pl.DataFrame
            Columns: score, selected, lambda_1, lambda_2.
        """
        self._check_calibrated()
        scores_new = as_numpy(scores_new)
        result = self.result_

        selected = scores_new >= result.lambda_1

        return pl.DataFrame(
            {
                "score": scores_new,
                "selected": selected,
                "lambda_1": np.full(len(scores_new), result.lambda_1),
                "lambda_2": np.full(len(scores_new), result.lambda_2),
            }
        )

    def pareto_report(self) -> pl.DataFrame:
        """
        Return the Pareto frontier as a Polars DataFrame.

        Only populated when run_grid_search=True. Returns Pareto-optimal
        (lambda_1, lambda_2) operating points, sorted by selection_rate descending.

        Raises
        ------
        RuntimeError
            If not calibrated or run_grid_search=False.
        """
        self._check_calibrated()
        if not self.run_grid_search:
            raise RuntimeError(
                "pareto_report() requires run_grid_search=True at construction time."
            )
        frontier = self.result_.pareto_frontier
        if not frontier:
            return pl.DataFrame(
                schema={
                    "lambda_1": pl.Float64,
                    "lambda_2": pl.Float64,
                    "selection_rate": pl.Float64,
                    "expected_set_size": pl.Float64,
                }
            )
        return pl.DataFrame(frontier).sort("selection_rate", descending=True)

    def underwriting_summary(self) -> dict:
        """
        Return a human-readable summary dict for underwriting management.

        Returns a dict with keys: method, alpha, xi, delta, lambda_1, lambda_2,
        selection_rate, xi_lcb, n_calibration, n_selected, conditional_risk,
        expected_set_size, feasible.

        Raises
        ------
        RuntimeError
            If not calibrated.
        """
        self._check_calibrated()
        r = self.result_
        return {
            "method": r.method,
            "alpha": self.alpha,
            "xi": self.xi,
            "delta": self.delta,
            "lambda_1": r.lambda_1,
            "lambda_2": r.lambda_2,
            "selection_rate": r.selection_rate,
            "xi_lcb": r.xi_lcb,
            "n_calibration": r.n_calibration,
            "n_selected": r.n_selected,
            "conditional_risk": r.conditional_risk,
            "expected_set_size": r.expected_set_size,
            "feasible": r.feasible,
        }

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "SelectiveConformalRC not calibrated. "
                "Call .calibrate(y_cal, scores_cal, pred_sets_cal) first."
            )

    def __repr__(self) -> str:
        if self.is_calibrated_:
            r = self.result_
            return (
                f"SelectiveConformalRC("
                f"method={r.method}, alpha={self.alpha}, xi={self.xi}, "
                f"lambda_1={r.lambda_1:.4f}, lambda_2={r.lambda_2:.4f}, "
                f"selection_rate={r.selection_rate:.2%}, n_cal={r.n_calibration})"
            )
        return f"SelectiveConformalRC(alpha={self.alpha}, xi={self.xi}, not calibrated)"
