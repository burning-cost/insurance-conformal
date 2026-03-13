"""
insurance_conformal.multivariate: Joint multi-output conformal prediction for insurance.

The core problem: a UK motor pricing team runs separate GLMs for claim
frequency (Poisson) and claim severity (Gamma). The point estimates lambda_hat
and mu_hat are useful, but actuaries need joint uncertainty quantification —
simultaneous intervals for both outputs that hold jointly with 95% probability.

Standard conformal prediction is univariate. Running it separately on each
output gives marginal coverage (95% per dimension), not joint coverage. The
distinction matters: if the frequency interval and severity interval each have
5% miscoverage probability, the joint miscoverage could be up to 10%.

This subpackage implements joint multi-output conformal prediction (Fan & Sesia,
arXiv:2512.15383) with insurance-specific additions:
- Coordinate-wise standardization for Poisson/Gamma scale mismatch
- Zero-claim masking for severity (N=0 policies have unobserved severity)
- Exposure offset support for Poisson frequency models
- Solvency II SCR mode (one-sided 99.5% joint upper bound)
- Consumer Duty monitoring support (joint outcome thresholds)

Quick start
-----------
    from insurance_conformal.multivariate import JointConformalPredictor

    predictor = JointConformalPredictor(
        models={'frequency': freq_glm, 'severity': sev_gbm},
        alpha=0.05,
        method='lwc',
    )
    predictor.calibrate(X_cal, Y_cal, zero_claim_mask=zero_mask)
    joint_set = predictor.predict(X_test)

    print(joint_set.lower['frequency'])  # Lower frequency bounds
    print(joint_set.upper['severity'])   # Upper severity bounds
    print(joint_set.joint_coverage_check(Y_test))  # >= 0.95

For SCR (Solvency II):
    from insurance_conformal.multivariate import SolvencyCapitalEstimator
    scr = SolvencyCapitalEstimator(models={...}, alpha=0.005)
    scr.calibrate(X_cal, Y_cal)
    result = scr.estimate(X_portfolio)
    print(result.aggregate_scr)

Coverage guarantee
------------------
Under exchangeability of (calibration, test) data:
    P(Y_new in C(X_new)) >= 1 - alpha    [finite-sample, no distributional assumption]

Methods
-------
- 'lwc': Local worst-case (Fan & Sesia Algorithm 2). Tightest valid method.
  O(d^2 n log n). Recommended default.
- 'gwc': Global worst-case. O(dn). Slightly wider than LWC but simpler.
- 'bonferroni': Classic Bonferroni correction. Valid but most conservative.
- 'sidak': Sidak correction. Valid only under output independence (rare in insurance).

References
----------
Fan & Sesia (2025). Interpretable Multivariate Conformal Prediction with Fast
Transductive Standardization. arXiv:2512.15383.

Hong (2025). Conformal prediction of future insurance claims in the regression
problem. arXiv:2503.03659.
"""

from .predictor import JointConformalPredictor
from .regions import JointPredictionSet
from .scr import SolvencyCapitalEstimator, SCRResult, scr_report
from .calibration import CalibratedScores, calibrate
from .diagnostics import coverage_report, width_by_dimension, compare_methods

__all__ = [
    "JointConformalPredictor",
    "JointPredictionSet",
    "SolvencyCapitalEstimator",
    "SCRResult",
    "scr_report",
    "CalibratedScores",
    "calibrate",
    "coverage_report",
    "width_by_dimension",
    "compare_methods",
]
