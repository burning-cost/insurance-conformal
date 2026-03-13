# insurance-conformal
[![Tests](https://github.com/burning-cost/insurance-conformal/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-conformal/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Distribution-free prediction intervals for insurance GBM and GLM pricing models. For pricing actuaries who need uncertainty quantification that doesn't rely on the model being correctly specified.

---

## The problem

Your Tweedie GBM gives point estimates. A pricing actuary needs to know the uncertainty around those estimates - not as a parametric confidence interval that depends on distributional assumptions, but as a guarantee: *this interval will contain the actual loss at least 90% of the time, for any data distribution*.

Conformal prediction provides that guarantee. The catch is that the choice of non-conformity score determines interval width. Most conformal implementations use the raw absolute residual `|y - yhat|`. For insurance data, that is wrong: it treats a 1-unit error on a £100 risk identically to a 1-unit error on a £10,000 risk, producing intervals that are too wide on low-risk policies and too narrow on large risks.

---

## The solution

For Tweedie/Poisson models, Var(Y) ~ mu^p. The correct non-conformity score is the locally-weighted Pearson residual:

```
score(y, yhat) = |y - yhat| / yhat^(p/2)
```

This accounts for the inherent heteroscedasticity of insurance claims. The result: ~30% narrower intervals with identical coverage guarantees. Based on Manna et al. (2025) and [arXiv 2507.06921](https://arxiv.org/abs/2507.06921).

---

## Blog post

[Conformal Prediction Intervals for Insurance Pricing Models](https://burning-cost.github.io/2026/03/06/conformal-prediction-intervals-for-insurance-pricing/)

---

## Installation

```bash
uv add insurance-conformal

# With CatBoost support:
uv add "insurance-conformal[catboost]"

# With plotting:
uv add "insurance-conformal[all]"
```

---

## Quick start

```python
from insurance_conformal import InsuranceConformalPredictor

# Fit your model however you normally would
import catboost
model = catboost.CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0,
)
model.fit(X_train_pd, y_train)

# Wrap it
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",  # default, recommended for insurance
    distribution="tweedie",
    tweedie_power=1.5,
)

# Calibrate on held-out data (must not overlap with training set)
cp.calibrate(X_cal, y_cal)

# Generate 90% prediction intervals
intervals = cp.predict_interval(X_test, alpha=0.10)
# DataFrame with columns: lower, point, upper

print(intervals.head())
#       lower   point    upper
# 0    0.0121  0.0845   0.3291
# 1    0.0034  0.0231   0.0901
# 2    0.1820  1.2742   4.9621
```

---

## Coverage diagnostics

The marginal coverage guarantee means `P(y in interval) >= 1 - alpha` averaged over all observations. In insurance, you also need to check that coverage is uniform across risk deciles - a model can achieve 90% overall while only covering 65% of high-risk policies.

```python
# THE key diagnostic
diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
print(diag)
#    decile  mean_predicted  n_obs  coverage  target_coverage
# 0       1          0.0234    400     0.923             0.90
# 1       2          0.0512    400     0.910             0.90
# ...
# 9      10          2.3410    400     0.905             0.90

# Full summary: marginal coverage + decile breakdown
cp.summary(X_test, y_test, alpha=0.10)

# Matplotlib plot with confidence bands
fig = cp.coverage_plot(X_test, y_test, alpha=0.10)
fig.savefig("coverage_by_decile.png", dpi=150)

# Interval width distribution
fig = cp.interval_width_distribution(X_test, alpha=0.10)
```

---

## Non-conformity scores

| Score | Formula | When to use |
|---|---|---|
| `pearson_weighted` | `\|y - yhat\| / yhat^(p/2)` | **Default.** Tweedie/Poisson pricing models. |
| `pearson` | `\|y - yhat\| / sqrt(yhat)` | Pure Poisson frequency models (p=1). |
| `deviance` | Deviance residual | When you want exact statistical optimality; slower. |
| `anscombe` | Anscombe transform | Variance-stabilising alternative to deviance. |
| `raw` | `\|y - yhat\|` | Baseline only. Not appropriate for insurance data. |

The score hierarchy for interval width (narrowest first, coverage identical):
`pearson_weighted >= deviance >= anscombe > pearson > raw`

---

## Temporal calibration

In insurance, you should calibrate on recent data to capture current loss trends, not a random subsample of all years:

```python
from insurance_conformal.utils import temporal_split

# Split by date - calibration gets the most recent 20%
X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
    X, y,
    calibration_frac=0.20,
    date_col="accident_year",  # column in X DataFrame
)

model.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)
```

Use [insurance-cv](https://github.com/burning-cost/insurance-cv) if you need full walk-forward cross-validation respecting IBNR development structure.

---

## Coverage guarantee

Split conformal prediction provides the following guarantee for exchangeable data:

```
P(y_test in [lower, upper]) >= 1 - alpha
```

This is distribution-free - it holds regardless of the true data distribution, model misspecification, or covariate shift (as long as calibration and test data are exchangeable). The only assumption is that the calibration set is held out from model training.

"Exchangeable" roughly means "drawn from the same distribution in the same order". For insurance, this means you should not calibrate on year 5 and test on year 1. Use temporal splits.

---

## Design choices

**Split conformal, not cross-conformal.** Cross-conformal is more statistically efficient but requires refitting the model on each calibration fold. For GBMs that take hours to train, this is not practical. Split conformal trains once, calibrates once.

**No MAPIE dependency.** MAPIE is excellent but it does not expose the insurance-specific scores implemented here. The split conformal algorithm is simple enough to own: 20 lines of code for `conformal_quantile()` plus the score functions.

**Lower bound clipped at 0.** Insurance losses are non-negative. Prediction intervals with negative lower bounds are nonsensical. We clip at 0 unconditionally.

**Auto-detection of Tweedie power.** For CatBoost, the power parameter is read from the loss function string. For sklearn `TweedieRegressor`, from `model.power`. If detection fails, we warn and default to p=1.5. Pass `tweedie_power=` explicitly if you know the correct value.

---

## References

- Manna, S. et al. (2025). "Distribution-free prediction sets for Tweedie regression." *arXiv:2507.06921*.
- Angelopoulos, A. N., & Bates, S. (2023). "Conformal prediction: A gentle introduction." *Foundations and Trends in Machine Learning*, 16(4), 494-591.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer.

---

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [credibility](https://github.com/burning-cost/credibility) | Bühlmann-Straub credibility weighting |
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Full conditional distribution per risk: mean, variance, CoV |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [rate-optimiser](https://github.com/burning-cost/rate-optimiser) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

**Spatial**

| Library | Description |
|---------|-------------|
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial territory ratemaking for UK personal lines |

[All libraries](https://burning-cost.github.io)

---

## Licence

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests welcome at [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal).

---

## Conformal Risk Control

Standard conformal prediction controls coverage probability: P(Y in C(X)) >= 1 - alpha. That guarantees a fraction of intervals contain the true outcome — but says nothing about how badly wrong the misses are. For insurance pricing, the question that matters is different: how much are we underpriced, in expectation?

The `insurance_conformal.risk` subpackage implements **Conformal Risk Control** (CRC, Angelopoulos et al., ICLR 2024), which controls expected loss directly:

```
E[L(C_lambda(X), Y)] <= alpha
```

for any bounded monotone loss L. No parametric assumptions. Finite-sample valid.

### Lead use case: premium sufficiency control

Given a GBM that outputs predicted pure premium p(X), find the smallest loading factor lambda* such that the expected shortfall from underpriced policies is bounded:

```python
from insurance_conformal.risk import PremiumSufficiencyController

psc = PremiumSufficiencyController(alpha=0.05)
psc.calibrate(y_cal, premium_cal)   # calibrate on held-out year
result = psc.predict(premium_new)   # apply to next year's book
# result["upper_bound"]: risk-controlled loading factor per policy
# result["lambda_hat"]: the single lambda* that achieves E[shortfall] <= 5%
```

### Three controllers

| Controller | Use case |
|---|---|
| `PremiumSufficiencyController` | Bound expected underpricing shortfall: E[max(claim - lambda * premium, 0) / premium] <= alpha |
| `IntervalWidthController` | Find the most efficient conformal quantile level that still bounds expected interval width |
| `SelectiveRiskController` | Accept/reject risks to bound expected loss on the accepted book |

### Import path

```python
from insurance_conformal.risk import (
    PremiumSufficiencyController,
    IntervalWidthController,
    SelectiveRiskController,
    conformal_risk_calibration,
    shortfall_loss,
    premium_sufficiency_report,
)
```

### References

- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal Risk Control. ICLR 2024. arXiv:2208.02814.
- Selective CRC: arXiv:2512.12844 (2025).
