"""
insurance_conformal.claims: Conformal prediction intervals for insurance claims regression.

Three methodological pillars:

1. **Hong order-statistic shortcut** — model-free full conformal with O(n log n)
   computation. No distributional assumption. Finite-sample valid coverage for all
   exchangeable distributions. (Hong 2025 arXiv:2503.03659)

2. **h-transformation framework** — generalises the model-free approach to exploit
   any regression model h, producing narrower intervals when the model fits well.
   (Hong 2026 arXiv:2601.21153)

3. **Tweedie nonconformity scores** — insurance-specific scores using the Tweedie
   variance function V(mu) = mu^p, plus a two-stage locally weighted split conformal
   method that auto-fits a CatBoost spread model. (Manna et al. 2025 ASMBI asmb.70045)

Solvency II hook: alpha=0.005 gives a finite-sample valid 99.5% upper bound on
claims — directly interpretable as an SCR capital requirement under Solvency II
Article 101 / UK Solvency UK (PRA PS9/24).

Quick start
-----------
>>> import numpy as np
>>> from insurance_conformal.claims import HongConformal, SCRReport
>>>
>>> rng = np.random.default_rng(42)
>>> X_train = rng.normal(size=(500, 5))
>>> y_train = np.abs(rng.standard_gamma(2.0, size=500))
>>> X_new = rng.normal(size=(100, 5))
>>>
>>> hc = HongConformal()
>>> hc.fit(X_train, y_train)
>>> intervals = hc.predict_interval(X_new, alpha=0.005)
>>> print(f"99.5% interval upper bounds: mean={intervals[:,1].mean():.2f}")
>>>
>>> report = SCRReport(hc)
>>> scr = report.solvency_capital_requirement(X_new, alpha=0.005)
>>> print(f"SCR (99.5% VaR): {scr:.2f}")
"""

from insurance_conformal.claims.hong import HongConformal, HongTransformConformal
from insurance_conformal.claims.tweedie import (
    TweedePearsonScore,
    TweedieAnscombeScore,
    TweedieDevianceScore,
    TwoStageLWConformal,
)
from insurance_conformal.claims.scr import SCRReport
from insurance_conformal.claims.diagnostics import (
    calibration_plot,
    calibration_plot_data,
    conditional_coverage_gap,
    width_by_covariate,
)

__all__ = [
    # hong
    "HongConformal",
    "HongTransformConformal",
    # tweedie
    "TweedePearsonScore",
    "TweedieDevianceScore",
    "TweedieAnscombeScore",
    "TwoStageLWConformal",
    # scr
    "SCRReport",
    # diagnostics
    "conditional_coverage_gap",
    "calibration_plot",
    "calibration_plot_data",
    "width_by_covariate",
]
