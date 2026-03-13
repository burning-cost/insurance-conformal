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

Three controllers, each for a different use case:

**PremiumSufficiencyController** — underpricing risk::

    from insurance_conformal.risk import PremiumSufficiencyController

    psc = PremiumSufficiencyController(alpha=0.05)
    psc.calibrate(y_cal, premium_cal)
    result = psc.predict(premium_new)
    # result["upper_bound"] = risk-controlled upper bound per policy

**IntervalWidthController** — efficient interval width::

    from insurance_conformal.risk import IntervalWidthController

    iwc = IntervalWidthController(width_target=500, scale=2000)
    iwc.calibrate_from_widths(widths_at_each_lambda)
    print(iwc.lambda_hat_)  # optimal quantile level

**SelectiveRiskController** — underwriting acceptance/rejection::

    from insurance_conformal.risk import SelectiveRiskController

    def high_claim_risk(y, scores):
        return (y > 5000).astype(float)

    src = SelectiveRiskController(alpha=0.08, loss_fn=high_claim_risk)
    src.calibrate(y_cal, scores_cal)
    decisions = src.predict(scores_new)

References
----------
Angelopoulos et al. (2024). Conformal Risk Control. ICLR 2024.
arXiv:2208.02814. https://doi.org/10.48550/arXiv.2208.02814

Selective CRC: arXiv:2512.12844 (2025).
"""

from insurance_conformal.risk.premium_sufficiency import PremiumSufficiencyController
from insurance_conformal.risk.interval_width import IntervalWidthController
from insurance_conformal.risk.selective import SelectiveRiskController
from insurance_conformal.risk.calibration import (
    conformal_risk_calibration,
    MonotoneLambdaSearch,
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

__all__ = [
    # Controllers
    "PremiumSufficiencyController",
    "IntervalWidthController",
    "SelectiveRiskController",
    # Core algorithm
    "conformal_risk_calibration",
    "MonotoneLambdaSearch",
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
]
