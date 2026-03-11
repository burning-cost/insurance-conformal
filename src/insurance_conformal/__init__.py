"""
insurance-conformal: Distribution-free prediction intervals for insurance pricing models.

The core insight: for Tweedie/Poisson data, the correct non-conformity score is
the locally-weighted Pearson residual (y - yhat) / sqrt(yhat^p), not the raw
residual. This gives ~30% narrower intervals with identical coverage guarantees.

v0.2.0 adds:
- LocallyWeightedConformal: two-stage conformal with secondary spread model
  for adaptive-width intervals (~24% narrower than standard pearson_weighted)
- HongModelFree: model-free conformal (no regression model needed)
- HongTransformConformal: h-transformation conformal (any callable as predictor)
- SCRReport: Solvency II SCR upper bounds with conformal coverage guarantees
- ert_coverage_gap: ERT conditional coverage diagnostic
- subgroup_coverage: coverage by arbitrary grouping variable
- width_efficiency_comparison: compare multiple predictors on interval width

Based on Manna et al. (2025), Hong (2025, 2026), and arXiv 2507.06921.

Example usage::

    from insurance_conformal import InsuranceConformalPredictor

    cp = InsuranceConformalPredictor(
        model=fitted_catboost_tweedie,
        nonconformity="pearson_weighted",
        distribution="tweedie",
    )
    cp.calibrate(X_cal, y_cal, exposure=exposure_cal)
    intervals = cp.predict_interval(X_test, alpha=0.10)

For locally-adaptive intervals::

    from insurance_conformal import LocallyWeightedConformal

    lw = LocallyWeightedConformal(model=fitted_catboost, tweedie_power=1.5)
    lw.fit(X_train, y_train)
    lw.calibrate(X_cal, y_cal)
    intervals = lw.predict_interval(X_test, alpha=0.10)

For SCR reporting::

    from insurance_conformal import InsuranceConformalPredictor, SCRReport

    cp = InsuranceConformalPredictor(model=fitted_model)
    cp.calibrate(X_cal, y_cal)
    scr = SCRReport(predictor=cp)
    scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
    print(scr.to_markdown())
"""

from insurance_conformal.predictor import InsuranceConformalPredictor
from insurance_conformal.scores import (
    NonconformityScore,
    raw_score,
    pearson_score,
    pearson_weighted_score,
    deviance_score,
    anscombe_score,
)
from insurance_conformal.diagnostics import CoverageDiagnostics
from insurance_conformal.locally_weighted import LocallyWeightedConformal
from insurance_conformal.hong import HongModelFree, HongTransformConformal
from insurance_conformal.scr import SCRReport
from insurance_conformal.diagnostics_ext import (
    ert_coverage_gap,
    subgroup_coverage,
    width_efficiency_comparison,
)

__all__ = [
    # Core (v0.1)
    "InsuranceConformalPredictor",
    "NonconformityScore",
    "raw_score",
    "pearson_score",
    "pearson_weighted_score",
    "deviance_score",
    "anscombe_score",
    "CoverageDiagnostics",
    # v0.2 additions
    "LocallyWeightedConformal",
    "HongModelFree",
    "HongTransformConformal",
    "SCRReport",
    "ert_coverage_gap",
    "subgroup_coverage",
    "width_efficiency_comparison",
]

__version__ = "0.2.1"
