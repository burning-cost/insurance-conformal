"""
insurance_conformal.risk: Conformal Risk Control for insurance pricing.

Standard conformal prediction (insurance_conformal) controls coverage probability:
P(Y in C(X)) >= 1 - alpha. That says nothing about the magnitude of errors.

This subpackage controls expected loss directly: E[L(C_lambda(X), Y)] <= alpha
for any bounded monotone loss L. This is Conformal Risk Control (CRC) from
Angelopoulos, Bates, Fisch, Lei & Schuster (2024), ICLR 2024.

The lead use case for UK pricing teams: premium sufficiency control.
Given a GBM that outputs a predicted pure premium p(X), find the smallest
loading factor lambda* such that:

    E[max(claim - lambda* * p(X), 0) / p(X)] <= alpha

This bounds expected shortfall from underpriced policies to alpha% of premium
income. No parametric assumptions. Finite-sample valid. Calibrate on one year's
claims, apply to next year's book.

Four controllers, each for a different use case:

**PremiumSufficiencyController** — underpricing risk::

    from insurance_conformal.risk import PremiumSufficiencyController

    psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
    psc.calibrate(y_cal, premium_cal)
    result = psc.predict(premium_new)
    # result["upper_bound"] = risk-controlled upper bound per policy

**IntervalWidthController** — efficient interval width::

    from insurance_conformal.risk import IntervalWidthController

    iwc = IntervalWidthController(width_target=500, scale=2000)
    iwc.calibrate_from_widths(widths_at_each_lambda)
    print(iwc.lambda_hat_)  # optimal quantile level

**SelectiveRiskController** — underwriting acceptance/rejection (simple)::

    from insurance_conformal.risk import SelectiveRiskController

    def high_claim_risk(y, scores):
        return (y > 5000).astype(float)

    src = SelectiveRiskController(alpha=0.08, loss_fn=high_claim_risk)
    src.calibrate(y_cal, scores_cal)
    decisions = src.predict(scores_new)

**SelectiveConformalRC** — two-stage selective conformal risk control (Xu et al. 2025)::

    from insurance_conformal.risk import SelectiveConformalRC
    from insurance_conformal.risk.selection_scores import selection_score_msp
    from insurance_conformal.risk.calibration import build_pred_set_matrix

    scores_cal = selection_score_msp(model.predict_proba(X_cal))
    pred_sets = build_pred_set_matrix(y_cal, lambda_2_grid, loss_fn)

    scrc = SelectiveConformalRC(alpha=0.10, xi=0.80)
    scrc.calibrate(y_cal, scores_cal, pred_sets, lambda_2_grid=lambda_2_grid)
    decisions = scrc.predict(scores_new)

References
----------
Angelopoulos et al. (2024). Conformal Risk Control. ICLR 2024.
arXiv:2208.02814. https://doi.org/10.48550/arXiv.2208.02814

Xu, Guo & Wei (2025). Selective Conformal Risk Control. arXiv:2512.12844.
"""

from insurance_conformal.risk.premium_sufficiency import PremiumSufficiencyController
from insurance_conformal.risk.interval_width import IntervalWidthController
from insurance_conformal.risk.selective import (
    SelectiveRiskController,
    SelectiveConformalRC,
    InfeasibleSCRCError,
    SCRCResult,
)
from insurance_conformal.risk.calibration import (
    conformal_risk_calibration,
    MonotoneLambdaSearch,
    build_pred_set_matrix,
)
from insurance_conformal.risk.losses import (
    shortfall_loss,
    scaled_shortfall_loss,
    coverage_loss,
    interval_width_loss,
    xl_recovery_loss,
    exposure_weighted_mean,
)
from insurance_conformal.risk.reporting import (
    premium_sufficiency_report,
    solvency_ii_model_error_note,
    risk_curve_dataframe,
)
from insurance_conformal.risk.selection_scores import (
    selection_score_msp,
    selection_score_margin,
    selection_score_entropy,
    selection_score_energy,
)

__all__ = [
    # Controllers
    "PremiumSufficiencyController",
    "IntervalWidthController",
    "SelectiveRiskController",
    "SelectiveConformalRC",
    # SCRC supporting types
    "InfeasibleSCRCError",
    "SCRCResult",
    # Core algorithm
    "conformal_risk_calibration",
    "MonotoneLambdaSearch",
    "build_pred_set_matrix",
    # Loss functions
    "shortfall_loss",
    "scaled_shortfall_loss",
    "coverage_loss",
    "interval_width_loss",
    "xl_recovery_loss",
    "exposure_weighted_mean",
    # Reporting
    "premium_sufficiency_report",
    "solvency_ii_model_error_note",
    "risk_curve_dataframe",
    # Selection scores
    "selection_score_msp",
    "selection_score_margin",
    "selection_score_entropy",
    "selection_score_energy",
]
