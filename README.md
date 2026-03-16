# insurance-conformal

[![PyPI](https://img.shields.io/pypi/v/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-conformal/blob/main/notebooks/quickstart.ipynb)

Distribution-free prediction intervals for insurance GBM and GLM pricing models — for pricing actuaries who need uncertainty quantification that holds regardless of model specification, without the coverage failures that parametric intervals produce on heterogeneous motor books.

---

## Why bother

Benchmarked against naive parametric intervals (Poisson GLM residual sigma) on synthetic UK motor data — 50,000 policies, temporal 60/20/20 train/calibration/test split. Same CatBoost Poisson point forecast for both methods.

| Metric | Naive parametric | Conformal (split) | Conformal (LW) |
|--------|-----------------|-------------------|----------------|
| Coverage (90% target) | Often misses in high-risk tail | Meets by construction | Meets by construction |
| Worst-decile coverage | Can be 70-80% | Near target | Near target |
| Mean interval width | Reference | Comparable | ~10-20% narrower |
| Calibration overhead | ~0s | ~1s | +2-5 min (secondary GBM) |
| Adaptive width | No | Partial (Pearson) | Yes |

The 10-20 percentage point undercoverage in the top decile is the problem this library solves. Conformal intervals meet the stated target by construction — the only requirement is an exchangeable calibration set, which any temporal split provides.

---

[Run on Databricks](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/conformal_prediction_intervals.py)

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
import numpy as np
from insurance_conformal import InsuranceConformalPredictor

# Synthetic data: 50k training, 10k calibration, 10k test
rng = np.random.default_rng(42)
n_train, n_cal, n_test = 50_000, 10_000, 10_000
n_features = 6
X_train = rng.standard_normal((n_train, n_features))
X_cal   = rng.standard_normal((n_cal,   n_features))
X_test  = rng.standard_normal((n_test,  n_features))
y_train = rng.gamma(shape=1.5, scale=500, size=n_train)
y_cal   = rng.gamma(shape=1.5, scale=500, size=n_cal)
y_test  = rng.gamma(shape=1.5, scale=500, size=n_test)

# Fit your model however you normally would
import catboost
model = catboost.CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0,
)
model.fit(X_train, y_train)

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

## Worked Example

[`conformal_prediction_intervals.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/examples/conformal_prediction_intervals.py) compares Tweedie conformal prediction intervals against a parametric bootstrap baseline on a synthetic motor book, then drills into per-segment coverage analysis across risk deciles and vehicle groups. It shows exactly where the bootstrap fails to meet its stated 90% coverage target — and confirms that the conformal approach holds by construction.

A Databricks-importable version is also available: [Databricks notebook](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/conformal_prediction_intervals.py).


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

# Matplotlib plots - use CoverageDiagnostics for coverage_plot and interval_width_distribution
from insurance_conformal import CoverageDiagnostics
intervals_for_diag = cp.predict_interval(X_test, alpha=0.10)
diag_tool = CoverageDiagnostics(
    y_true=y_test,
    y_lower=intervals_for_diag["lower"].to_numpy(),
    y_upper=intervals_for_diag["upper"].to_numpy(),
    y_pred=intervals_for_diag["point"].to_numpy(),
    alpha=0.10,
)
fig = diag_tool.coverage_plot()
fig.savefig("coverage_by_decile.png", dpi=150)

# Interval width distribution
fig = diag_tool.interval_width_distribution()
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
`pearson_weighted <= deviance <= anscombe < pearson < raw`

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

This is distribution-free — it holds regardless of the true data distribution or model misspecification. The core assumption is exchangeability: calibration and test observations must be drawn from the same distribution and be interchangeable in order. Temporal covariate shift — where the risk profile of test data differs from calibration data — violates this assumption and can degrade coverage in practice. Use temporal calibration splits (calibrate on the most recent accident year before the test period) to minimise the distribution gap. The `temporal_split` utility is provided for this purpose.

"Exchangeable" roughly means "drawn from the same distribution in the same order". For insurance, this means you should not calibrate on year 5 and test on year 1. Use temporal splits.

---

## Design choices

**Split conformal, not cross-conformal.** Cross-conformal is more statistically efficient but requires refitting the model on each calibration fold. For GBMs that take hours to train, this is not practical. Split conformal trains once, calibrates once.

**No MAPIE dependency.** MAPIE is excellent but it does not expose the insurance-specific scores implemented here. The split conformal algorithm is simple enough to own: 20 lines of code for `conformal_quantile()` plus the score functions.

**Lower bound clipped at 0.** Insurance losses are non-negative. Prediction intervals with negative lower bounds are nonsensical. We clip at 0 unconditionally.

**Auto-detection of Tweedie power.** For CatBoost, the power parameter is read from the loss function string. For sklearn `TweedieRegressor`, from `model.power`. If detection fails, we warn and default to p=1.5. Pass `tweedie_power=` explicitly if you know the correct value.

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

psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
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

---

## SCRReport

`SCRReport` wraps a calibrated conformal predictor and produces per-risk 99.5% upper bounds suitable for internal stress-testing and model validation.

> **Disclaimer**: SCRReport is an internal stress-testing tool. Solvency II SCR calculations for regulatory purposes require sign-off under an approved internal model or the standard formula. Do not use this output in regulatory returns without appropriate actuarial review, governance sign-off, and alignment with your firm's approved methodology.

```python
from insurance_conformal.scr import SCRReport

scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
val_table = scr.coverage_validation_table(X_test, y_test)
print(scr.to_markdown())
```

---

## Limitations

**Exchangeability assumption.** Split conformal requires calibration and test data to be exchangeable. Temporal covariate shift — changes in portfolio mix, inflation, or risk profile between calibration and test periods — weakens this assumption. Use temporal calibration splits and monitor coverage drift over time.

**IBNR on recent accident years.** For severity and pure premium models, calibrating on the most recent accident year means calibrating on incomplete claims. IBNR (incurred but not reported) development causes non-conformity scores to be computed on understated y_cal values, producing intervals that are too narrow for open development periods. Recommend using only fully-developed accident years (typically 3+ years prior) for calibration, or applying a development factor to y_cal before calibration.

**Marginal vs. conditional coverage.** The conformal guarantee is marginal: it holds on average across all observations. High-risk subgroups can still be systematically under-covered if the non-conformity score does not fully account for heteroscedasticity. Always check `coverage_by_decile()` after calibration.

**Score choice matters.** The `raw` score produces valid but very wide intervals on insurance data. Use `pearson_weighted` for Tweedie/Poisson models. If you switch scores, recalibrate.

---

## References

- Manna, S. et al. (2025). "Distribution-free prediction sets for Tweedie regression." *arXiv:2507.06921*.
- Angelopoulos, A. N., & Bates, S. (2023). "Conformal prediction: A gentle introduction." *Foundations and Trends in Machine Learning*, 16(4), 494-591.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer.

---

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Temporal cross-validation — provides the calibration splits conformal prediction requires to maintain coverage guarantees |
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Parametric severity distributions — alternative when closed-form tail quantities are needed rather than distribution-free intervals |
| [insurance-quantile](https://github.com/burning-cost/insurance-quantile) | Quantile GBM for tail risk — feeds directly into conformalized quantile regression for distribution-free coverage |

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
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility weighting |
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Full conditional distribution per risk: mean, variance, CoV |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |
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
