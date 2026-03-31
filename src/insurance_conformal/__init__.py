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

v0.3.0 adds:
- insurance_conformal.risk subpackage: Conformal Risk Control (CRC) for insurance.
  Controls expected loss directly — E[L(C_lambda(X), Y)] <= alpha — rather than
  just coverage probability. Lead use case: premium sufficiency control.
  See insurance_conformal.risk for PremiumSufficiencyController,
  IntervalWidthController, SelectiveRiskController, and supporting loss functions.

v0.4.0 adds:
- insurance_conformal.claims subpackage: Conformal prediction for claims regression.
  Hong order-statistic shortcut (O(n log n) model-free full conformal), Tweedie
  nonconformity scores, two-stage locally weighted conformal, and Solvency II SCR
  reporting. See insurance_conformal.claims for HongConformal, HongTransformConformal,
  TweedePearsonScore, TwoStageLWConformal, and SCRReport.
- insurance_conformal.multivariate subpackage: Joint multi-output conformal prediction.
  Simultaneous intervals for frequency/severity and other multi-output insurance models.
  Implements Fan & Sesia (2025) LWC/GWC methods with insurance-specific additions.
  See insurance_conformal.multivariate for JointConformalPredictor,
  SolvencyCapitalEstimator, and supporting calibration/diagnostic tools.

v0.5.0 adds SelectiveConformalRC (risk-controlled selective prediction).

v0.5.1 adds:
- LightGBM backend for LocallyWeightedConformal
- FrequencySeverityConformal (Graziadei et al. 2307.13124)
- TweediePearsonScore alias (corrects TweedePearsonScore typo)

v0.6.0 adds:
- RetroAdj: Online Conformal Inference with Retrospective Adjustment (Jun & Ohn 2025,
  arXiv:2511.04275). Jackknife+ intervals over a KRR base model with rank-one matrix
  updates that retroactively correct all leave-one-out residuals in the sliding window
  at each step. Recovers from abrupt distribution shifts (UK motor +30% inflation,
  Ogden rate changes, CAT events) within 1-3 steps versus ~200 steps for ACI at
  gamma=0.005. Supports ACI and SFOGD alpha updates, symmetric and asymmetric intervals,
  residual-only mode for external GLM/GBM models, and periodic numerical reset.
  See insurance_conformal.retro_adj for RetroAdj.

v0.6.2 adds:
- ConformalisedQuantileRegression: split CQR (Romano, Patterson & Candès, NeurIPS 2019,
  arXiv:1905.03222). Wraps a pair of pre-fitted quantile models and applies a conformal
  calibration correction to guarantee marginal coverage >= 1 - alpha regardless of
  quantile model misspecification. Produces heteroscedastic intervals — wider for
  high-variance risks, narrower for stable ones — which is directly useful for
  personal lines motor (young drivers, high-value vehicles) and commercial property
  (cat-exposed risks). Works with any quantile objective: CatBoost Quantile:alpha=,
  LightGBM objective=quantile, sklearn GradientBoostingRegressor loss=quantile.
  See ConformalisedQuantileRegression.

v0.7.0 adds:
- TweedieConformPredictor: unified split conformal for Tweedie models. Single
  entry point consolidating all four score types (pearson, lw_pearson, deviance,
  anscombe), exposure weighting, score selection, and coverage diagnostics.
  The key genuine gap vs existing code: exposure_weighted mode adjusts the
  score denominator to (e*mu)^{p/2}, appropriate for rate models with variable
  policy terms. See TweedieConformPredictor.

v0.6.3 adds:
- solvency_capital_range(): lightweight functional API for Solvency II SCR bounds.
  Returns a SolvencyCapitalRange dataclass with per-risk scr_estimate, lower_bound,
  upper_bound, interval_width, coverage_level, total_scr, and mean_interval_width.
  Works with any conformal predictor. Complements SCRReport (the class-based API
  for regulatory reporting); use this when you need SCR estimates inside a pipeline.
  See solvency_capital_range.

v0.7.1 adds:
- ConditionalCoverageERT: full ERT test for conditional coverage violations
  (Braun et al. arXiv:2512.11779). Trains a LightGBM classifier on coverage
  indicators Z_i = 1{y_i in interval_i} ~ X_i using KFold CV and measures how
  much it beats the constant predictor. Three loss variants (L1-ERT, L2-ERT,
  KL-ERT), directional variants (both/under/over) for insurance use cases, and
  bootstrap CIs. subgroup_coverage() method provides per-feature binned coverage
  reports for FCA/TCF documentation.
  See insurance_conformal.conditional_coverage for ConditionalCoverageERT.

Based on Manna et al. (2025), Hong (2025, 2026), arXiv 2507.06921,
Angelopoulos et al. (2024) Conformal Risk Control (ICLR 2024, arXiv:2208.02814),
Fan & Sesia (2025) arXiv:2512.15383, Jun & Ohn (2025) arXiv:2511.04275,
Romano, Patterson & Candès (2019) arXiv:1905.03222, and
Braun et al. (2024) arXiv:2512.11779.

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

For conformal risk control (premium sufficiency)::

    from insurance_conformal.risk import PremiumSufficiencyController

    psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
    psc.calibrate(y_cal, premium_cal)
    result = psc.predict(premium_new)
    # result["upper_bound"] = risk-controlled loading factor per policy

For claims conformal prediction::

    from insurance_conformal.claims import HongConformal, SCRReport

    hc = HongConformal()
    hc.fit(X_train, y_train)
    intervals = hc.predict_interval(X_test, alpha=0.005)

For joint frequency/severity conformal prediction::

    from insurance_conformal.multivariate import JointConformalPredictor

    predictor = JointConformalPredictor(
        models={'frequency': freq_glm, 'severity': sev_gbm},
        alpha=0.05,
        method='lwc',
    )
    predictor.calibrate(X_cal, Y_cal)
    joint_set = predictor.predict(X_test)

For online conformal with retrospective adjustment (distribution shift)::

    from insurance_conformal import RetroAdj

    model = RetroAdj(bandwidth=1.0, lambda_reg=0.1, window_size=250)
    model.fit(y_train, X_train)
    lower, upper = model.predict_interval(y_test, X_test, alpha=0.10)
    # Or residual-only mode for GLM/GBM residuals:
    resid_train = y_train - glm.predict(X_train)
    resid_test  = y_test  - glm.predict(X_test)
    model2 = RetroAdj(window_size=250)
    model2.fit(resid_train)
    lower_r, upper_r = model2.predict_interval(resid_test, alpha=0.10)

For conformalized quantile regression (heteroscedastic intervals)::

    from catboost import CatBoostRegressor
    from insurance_conformal import ConformalisedQuantileRegression

    lo = CatBoostRegressor(loss_function="Quantile:alpha=0.05", ...)
    hi = CatBoostRegressor(loss_function="Quantile:alpha=0.95", ...)
    lo.fit(X_train, y_train)
    hi.fit(X_train, y_train)

    cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi)
    cqr.calibrate(X_cal, y_cal)
    intervals = cqr.predict_interval(X_test, alpha=0.10)
    # intervals["lower"], intervals["upper"] carry the coverage guarantee
    # intervals["q_lo"], intervals["q_hi"] are the raw quantile model outputs

For conditional coverage testing (FCA/TCF diagnostics)::

    from insurance_conformal.conditional_coverage import ConditionalCoverageERT

    ert = ConditionalCoverageERT(loss="l1", direction="under", n_splits=5)
    result = ert.evaluate(X_test, y_lower, y_upper, y_true, alpha=0.10)
    print(result)  # ERTResult(ert=0.023***, CI=[0.008, 0.041], ...)
    subgroups = ert.subgroup_coverage(X_test, y_lower, y_upper, y_true, alpha=0.10,
                                       feature_names=["age", "vehicle_group", "region"])
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
from insurance_conformal.retro_adj import RetroAdj
from insurance_conformal.cqr import ConformalisedQuantileRegression
from insurance_conformal._solvency import solvency_capital_range, SolvencyCapitalRange
from insurance_conformal.tweedie_conform import TweedieConformPredictor
from insurance_conformal.conditional_coverage import ConditionalCoverageERT

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
    # v0.3: risk subpackage (import from insurance_conformal.risk)
    # v0.4: claims subpackage (import from insurance_conformal.claims)
    # v0.4: multivariate subpackage (import from insurance_conformal.multivariate)
    # v0.6 additions
    "RetroAdj",
    # v0.6.2 additions
    "ConformalisedQuantileRegression",
    # v0.6.3 additions
    "solvency_capital_range",
    "SolvencyCapitalRange",
    # v0.7.0 additions
    "TweedieConformPredictor",
    # v0.7.1 additions
    "ConditionalCoverageERT",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-conformal")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
