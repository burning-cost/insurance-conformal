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

v0.8.0 adds:
- ConditionalValidityIndex: scalar measure of conditional coverage quality,
  decomposed into CVI_U (undercoverage risk) and CVI_O (overcoverage cost).
  Fits a LightGBM classifier Z ~ X via KFold CV with isotonic probability
  calibration to estimate local coverage probabilities eta_hat(x). Based on
  Zhou et al. arXiv:2603.27189. Complements ConditionalCoverageERT: ERT is a
  hypothesis test, CVI is a continuous score for ranking and selection.
- CCSelect: selects the best conformal predictor from a set by minimising mean
  CVI across random train/eval splits. Picks the predictor with the most uniform
  conditional coverage — the right criterion when regulatory and commercial
  exposure is concentrated in tails. Predictors must implement .calibrate(X, y)
  and .predict_interval(X, alpha) matching the standard interface.
  See ConditionalValidityIndex, CCSelect, CVIResult, CCSelectResult.


v0.9.0 adds:
- ShapeAdaptiveCP: MOPI (Minimax Optimization for Predictive Inference) for
  conditional coverage conformal prediction (Bao et al., arXiv:2603.23374).
  Solves a minimax problem to achieve near-uniform conditional coverage across
  subgroups Z, reducing mean squared conditional coverage error (MSCE) vs
  standard split conformal and CQR. Key insurance feature: Z (age band, rating
  cell, gender) can be used at calibration time but masked at deployment — the
  prediction intervals are functions of X only. Two modes: group (finite
  categorical Z, pure numpy O(K) per step) and rkhs (continuous Z, Gaussian
  kernel, O(n^3) setup). See ShapeAdaptiveCP.

v1.0.0 adds:
- LoBoostCP: model-native local conformal prediction for gradient-boosted trees
  (Santos, Izbicki & Stern, 2025, arXiv:2602.22432). Uses the leaf structure of
  fitted GBMs to create multiscale local calibration groups — no retraining.
  Supports CatBoost, XGBoost, LightGBM, and sklearn GradientBoosting. Produces
  prediction intervals with local (per-risk-type) calibration, narrower where the
  model has genuine local resolution. Two score types: absolute (|y - ŷ|) and
  normalized (|y - ŷ|/ŷ) for heteroscedastic Tweedie/Gamma severity data.
  See LoBoostCP.

v1.1.0 adds:
- KMMConformalPredictor: covariate-shift-robust conformal prediction via Kernel
  Mean Matching (Laghuvarapu, Deb & Sun 2026, arXiv:2603.26415). Wraps any
  fitted sklearn-compatible model. Solves a QP (scipy SLSQP) to find calibration
  weights that minimise RKHS moment discrepancy between the reweighted calibration
  distribution and the target distribution, then applies the Tibshirani (2019)
  weighted conformal quantile. Selective mode (SKMM) abstains on out-of-support
  risks. Primary insurance use cases: portfolio transfers, post-COVID cohort shift,
  channel expansion, scheme changes. See KMMConformalPredictor.

v1.2.0 adds:
- ConformalisedSurvival: doubly robust conformalized survival analysis for
  right-censored data (Sesia & Svetnik, arXiv:2412.09729). Produces a lower
  prediction bound P[T >= L_hat(X)] >= 1 - alpha using IPCW-weighted conformal
  inference after Monte Carlo imputation of unobserved censoring times. Double
  robustness: coverage holds asymptotically if either the survival model or the
  censoring model is correctly specified. Insurance use cases: mortality bounds
  for protection pricing, lapse timing bounds, time-to-claim closure bounds.
  Includes KaplanMeierCensoringModel (marginal censoring), LifelinesCoxCensoringAdapter,
  and SksurvCoxCensoringAdapter. Optional dependency: pip install insurance-conformal[survival].
  See ConformalisedSurvival, KaplanMeierCensoringModel, SurvivalModelProtocol.
- LCPModelSelector: locally adaptive conformal model and score selection
  (Wang & Wang, Feb 2026, arXiv:2602.19284). Fits K pre-trained models and
  selects the locally best model/score for each test point using kernel-weighted
  LOO validity checks. Marginal coverage is guaranteed regardless of which model
  is selected. LocalScoreSelector is a convenience wrapper for single-model
  multi-score selection. Insurance use cases: score selection across pearson/deviance/
  anscombe, model selection between GLM and GBM, and portfolio mixture localisation.
  See LCPModelSelector, LocalScoreSelector.

v1.3.1 adds:
- ConditionalCoverageAssessor: CVI assessment with sklearn GradientBoostingClassifier.
  No LightGBM dependency required. Separable fit/score/select API: fit() trains the
  reliability estimator once on calibration data; score() applies it to any test set
  in one forward pass; select() picks the best predictor from a dict of prediction
  sets. Useful when comparing multiple conformal predictors — one fit(), K score()
  calls. Based on Zhou et al. arXiv:2603.27189. See ConditionalCoverageAssessor.

Based on Manna et al. (2025), Hong (2025, 2026), arXiv 2507.06921,
Angelopoulos et al. (2024) Conformal Risk Control (ICLR 2024, arXiv:2208.02814),
Fan & Sesia (2025) arXiv:2512.15383, Jun & Ohn (2025) arXiv:2511.04275,
Romano, Patterson & Candès (2019) arXiv:1905.03222,
Braun et al. (2024) arXiv:2512.11779,
Zhou et al. (2026) arXiv:2603.27189, and
Sesia & Svetnik (2024) arXiv:2412.09729.

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
    from insurance_conformal.covariate_shift import KMMConformalPredictor
from insurance_conformal.lcp_model_selector import LCPModelSelector, LocalScoreSelector

    ert = ConditionalCoverageERT(loss="l1", direction="under", n_splits=5)
    result = ert.evaluate(X_test, y_lower, y_upper, y_true, alpha=0.10)
    print(result)  # ERTResult(ert=0.023***, CI=[0.008, 0.041], ...)
    subgroups = ert.subgroup_coverage(X_test, y_lower, y_upper, y_true, alpha=0.10,
                                       feature_names=["age", "vehicle_group", "region"])

For conditional coverage diagnostics and predictor selection (v0.8.0)::

    from insurance_conformal import ConditionalValidityIndex, CCSelect

    # Evaluate conditional coverage quality of a single predictor
    cvi = ConditionalValidityIndex(gamma=0.1, n_splits=5)
    result = cvi.evaluate(X_test, lower, upper, y_test, alpha=0.10)
    print(result.cvi)    # total CVI
    print(result.cvi_u)  # undercoverage component
    print(result.cvi_o)  # overcoverage component

    # Select the best predictor from multiple candidates
    selector = CCSelect(n_splits=5, split_ratio=0.5, random_state=42)
    sel_result = selector.fit(
        {"pearson": cp_pearson, "locally_weighted": cp_lw},
        X_cal, y_cal, alpha=0.10
    )
    print(sel_result.best_predictor)
    print(selector.compare())

For conformalized survival analysis with right-censored data (v1.2.0)::

    from insurance_conformal import ConformalisedSurvival, KaplanMeierCensoringModel

    # Fit a survival model with predict_quantile(X, alpha) interface
    # (or use SksurvCoxCensoringAdapter / LifelinesCoxCensoringAdapter)
    km_censoring = KaplanMeierCensoringModel()
    km_censoring.fit(t_train, e_train)

    cs = ConformalisedSurvival(
        survival_model=fitted_survival_model,
        censoring_model=km_censoring,
        method="fixed_cutoff",
        alpha=0.10,
        n_impute=10,
    )
    cs.calibrate(X_cal, t_cal, e_cal)
    bounds = cs.predict_lower_bound(X_test)
    # bounds["lower_bound"]: 90% LPB on survival time
    # bounds["q_alpha_model"]: raw model quantile before conformal correction
    diag = cs.coverage_diagnostics(X_test, t_test, e_test)
    print(diag)
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
from insurance_conformal.conditional_coverage import (
    ConditionalCoverageERT,
    ConditionalValidityIndex,
    CCSelect,
    CVIResult,
    CCSelectResult,
)
from insurance_conformal.mopi import ShapeAdaptiveCP
from insurance_conformal.loboost import LoBoostCP
from insurance_conformal.survival import (
    ConformalisedSurvival,
    KaplanMeierCensoringModel,
    SurvivalModelProtocol,
    CensoringModelProtocol,
    LifelinesCoxCensoringAdapter,
    SksurvCoxCensoringAdapter,
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
    # v0.8.0 additions
    "ConditionalValidityIndex",
    "CCSelect",
    "CVIResult",
    "CCSelectResult",
    # v0.9.0 additions
    "ShapeAdaptiveCP",
    # v1.0.0 additions
    "LoBoostCP",
    # v1.1.0 additions
    "KMMConformalPredictor",
    # v1.2.0 additions
    "ConformalisedSurvival",
    "KaplanMeierCensoringModel",
    "SurvivalModelProtocol",
    "CensoringModelProtocol",
    "LifelinesCoxCensoringAdapter",
    "SksurvCoxCensoringAdapter",
    "LCPModelSelector",
    "LocalScoreSelector",
    # v1.3.1 additions
    "ConditionalCoverageAssessor",
    "CVIAssessmentResult",
    "CVISelectResult",
]

from insurance_conformal.covariate_shift import KMMConformalPredictor
from insurance_conformal.lcp_model_selector import LCPModelSelector, LocalScoreSelector
from insurance_conformal.assessment import (
    ConditionalCoverageAssessor,
    CVIAssessmentResult,
    CVISelectResult,
)

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-conformal")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
