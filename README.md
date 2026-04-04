# insurance-conformal

[![Tests](https://github.com/burning-cost/insurance-conformal/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-conformal/actions/workflows/tests.yml) [![PyPI](https://img.shields.io/pypi/v/insurance-conformal)](https://pypi.org/project/insurance-conformal/) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/insurance-conformal/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-conformal/blob/main/LICENSE)

**Distribution-free prediction intervals for insurance pricing models — 13% narrower than parametric Tweedie, with a finite-sample coverage guarantee.**

**Blog post:** [Conformal Prediction Intervals for Insurance Pricing Models](https://burning-cost.github.io/2026/03/06/conformal-prediction-intervals-for-insurance-pricing/)

---

## The problem

Your pricing model gives point estimates. Your parametric prediction intervals assume variance scales as mu^p across the whole book — an assumption that breaks exactly where the stakes are highest: large, unusual risks.

On a heterogeneous UK motor portfolio, parametric Tweedie intervals over-cover low-risk policies (unnecessary width) and under-cover the top risk decile — which is what drives reinsurance attachment, reserving, and SCR calculations.

Conformal prediction fixes this. The guarantee is `P(y in interval) >= 1 - alpha` for any data distribution, as long as calibration and test data are exchangeable. No parametric family required.

The non-obvious implementation detail: most conformal libraries use raw absolute residuals `|y - yhat|`. For insurance data that is wrong — a £1 error on a £100 risk is not the same as a £1 error on a £10,000 risk. The correct score for Tweedie models is `|y - yhat| / yhat^(p/2)`, which normalises by the Tweedie standard deviation and produces exchangeable scores across risk levels. That is what this library implements.

---

## Quick start

```python
from insurance_conformal import InsuranceConformalPredictor

# Wrap any fitted sklearn-compatible model
cp = InsuranceConformalPredictor(
    model=fitted_gbm,
    nonconformity="pearson_weighted",  # correct default for Tweedie
    tweedie_power=1.5,
)

# Calibrate on held-out data (must not overlap training)
cp.calibrate(X_cal, y_cal)

# 90% prediction intervals — polars DataFrame: lower, point, upper
intervals = cp.predict_interval(X_test, alpha=0.10)

# Always check per-decile coverage (marginal != conditional)
print(cp.coverage_by_decile(X_test, y_test, alpha=0.10))
```

For locally-adaptive intervals (narrower on low-variance risks, wider on high-variance risks):

```python
from insurance_conformal import LocallyWeightedConformal

lw = LocallyWeightedConformal(model=fitted_gbm, tweedie_power=1.5)
lw.fit(X_train, y_train)
lw.calibrate(X_cal, y_cal)
intervals = lw.predict_interval(X_test, alpha=0.10)
```

---

## Why a pricing actuary should care

**Accuracy where it matters.** Parametric Tweedie intervals produce 93% aggregate coverage at a 90% target — fine in aggregate, but that surplus width sits on low-risk policies. The top-risk decile that drives reinsurance and reserving gets marginal coverage at best, and on books with more pronounced tail heteroscedasticity it will miss the target.

**Regulatory defensibility.** The distribution-free guarantee does not rely on model fit. You can write "P(claim in interval) >= 90%, finite-sample valid, no parametric assumptions" in a PRA SS1/23 validation pack. You cannot write that for a parametric bootstrap interval.

**SCR calculations.** `SCRReport` produces per-risk 99.5% upper bounds with a coverage validation table — exactly the format needed for internal model stress-testing documentation.

**Premium sufficiency control.** `PremiumSufficiencyController` finds the smallest loading factor such that expected underpricing shortfall is bounded at alpha. A direct regulatory argument, not a statistical artefact.

---

## Performance on a realistic motor book

CatBoost Tweedie(p=1.5), 50,000 synthetic UK motor policies, heteroskedastic Gamma DGP, temporal 60/20/20 split.

| | Parametric Tweedie | Conformal (`pearson_weighted`) | Locally-weighted conformal |
|---|---|---|---|
| Distribution assumption | Tweedie Var ~ mu^p | None | None |
| Aggregate coverage @ 90% target | 93.1% (over-covers) | 90.2% | 90.3% |
| Top-decile coverage @ 90% target | 90.4% | 87.9% | 90.6% |
| Mean interval width | £4,393 | £3,806 (−13.4%) | £3,881 (−11.7%) |
| Width adapts per risk segment | No | Partial | Yes |
| Finite-sample valid guarantee | No | Yes | Yes |

The locally-weighted variant meets the 90% target in the top decile by construction — the parametric baseline only coincidentally passes it on this dataset. Run the validation: import `notebooks/databricks_validation.py` into Databricks.

---

## Installation

```bash
pip install insurance-conformal

# With CatBoost support:
pip install "insurance-conformal[catboost]"

# With LightGBM support:
pip install "insurance-conformal[lightgbm]"

# With everything (CatBoost, LightGBM, plotting):
pip install "insurance-conformal[all]"
```

Or with uv:

```bash
uv add insurance-conformal
```

**Dependencies:** polars and pandas are both required. Polars is the primary output format — all prediction and diagnostic methods return `pl.DataFrame`. Pandas is required for binning utilities and for accepting pandas DataFrame inputs. Both install automatically.

---

## Worked examples

### 1. Motor frequency-severity model with per-decile coverage audit

```python
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from insurance_conformal.claims import FrequencySeverityConformal
from insurance_conformal import subgroup_coverage

fs = FrequencySeverityConformal(
    freq_model=PoissonRegressor(),
    sev_model=GammaRegressor(),
)
fs.fit(X_train, d_train, y_train)   # d_train = observed claim counts
fs.calibrate(X_cal, d_cal, y_cal)
intervals = fs.predict_interval(X_test, alpha=0.10)

# Coverage by vehicle group
sg = subgroup_coverage(
    predictor=fs,
    X_test=X_test,
    y_test=y_test,
    alpha=0.10,
    groups=vehicle_group_band,
    group_name="vehicle_group_band",
)
print(sg)
```

The calibration subtlety here: using the observed claim count in the severity model at calibration time creates a distributional mismatch that breaks the coverage guarantee. `FrequencySeverityConformal` feeds the predicted frequency (not the observed count) into the severity model at both calibration and test time. See Graziadei et al. (2023) for the proof.

### 2. Premium sufficiency control — bound expected underpricing

Useful when a pricing review requires a documented guarantee that expected shortfall from underpriced policies stays below a threshold.

```python
from insurance_conformal.risk import PremiumSufficiencyController

psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
psc.calibrate(y_cal, premium_cal)   # calibrate on held-out year
result = psc.predict(premium_new)   # apply to next year's book

# result["lambda_hat"]: the loading factor such that E[shortfall] <= 5%
# result["upper_bound"]: risk-controlled loaded premium per policy
print(f"Required loading: {result['lambda_hat']:.3f}")
```

### 3. SCR bounds for internal model documentation

```python
from insurance_conformal import InsuranceConformalPredictor, SCRReport

cp = InsuranceConformalPredictor(model=fitted_model)
cp.calibrate(X_cal, y_cal)

scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
val_table  = scr.coverage_validation_table(X_test, y_test)
print(scr.to_markdown())
```

> **Disclaimer:** `SCRReport` is an internal stress-testing tool. Solvency II SCR calculations for regulatory purposes require sign-off under an approved internal model or the standard formula. Do not use this output in regulatory returns without appropriate actuarial review, governance sign-off, and alignment with your firm's approved methodology.

### 4. Recovering from mid-year claims inflation (Ogden rate change, CAT event)

Standard conformal with a static calibration set breaks when the book shifts mid-year. `RetroAdj` recovers within 1–3 steps by retroactively correcting all leave-one-out residuals in the sliding window simultaneously.

```python
from insurance_conformal import RetroAdj

# Residual-only mode: wrap an existing GLM or GBM
resid_train = y_train - glm.predict(X_train)
resid_test  = y_test  - glm.predict(X_test)

model = RetroAdj(window_size=250, gamma=0.005)
model.fit(resid_train)
lower_r, upper_r = model.predict_interval(resid_test, alpha=0.10)

lower_claims = lower_r + glm.predict(X_test)
upper_claims = upper_r + glm.predict(X_test)
```

| Metric | RetroAdj | Standard ACI |
|--------|----------|-----|
| Steps to recover 90% coverage after +30% inflation shock | ~15–30 | ~80–150 |
| Post-shift coverage (full window) | ~88–91% | ~80–87% |

---

## Features

- `InsuranceConformalPredictor` — split conformal prediction wrapping any sklearn-compatible model. Non-conformity scores: `pearson_weighted`, `pearson`, `deviance`, `anscombe`, `raw`.
- `LocallyWeightedConformal` — two-stage conformal with a secondary spread model. Meets per-decile coverage targets that standard conformal misses.
- `ConformalisedQuantileRegression` — split CQR (Romano et al., 2019). Wraps pre-fitted quantile models. Works with CatBoost `Quantile:alpha=`, LightGBM `objective=quantile`.
- `FrequencySeverityConformal` — correct conformity scoring for two-stage frequency-severity models (Graziadei et al., 2023).
- `SCRReport` — per-risk 99.5% upper bounds with coverage validation table. For PRA SS1/23 model documentation.
- `solvency_capital_range()` — functional API for SCR bounds inside pipelines.
- `insurance_conformal.risk` — Conformal Risk Control (Angelopoulos et al., ICLR 2024). `PremiumSufficiencyController`, `IntervalWidthController`, `SelectiveRiskController`.
- `RetroAdj` — online conformal with retrospective adjustment (Jun & Ohn, 2025). Recovers from abrupt distribution shifts within 1–3 steps.
- `CoverageDiagnostics` — coverage-by-decile plots, interval width distributions, subgroup coverage by arbitrary segment.
- `insurance_conformal.multivariate` — joint multi-output conformal for simultaneous frequency/severity intervals.

---

## Non-conformity scores

| Score | Formula | When to use |
|---|---|---|
| `pearson_weighted` | `\|y - yhat\| / yhat^(p/2)` | **Default.** Tweedie/Poisson pricing models. |
| `pearson` | `\|y - yhat\| / sqrt(yhat)` | Pure Poisson frequency models (p=1). |
| `deviance` | Deviance residual | When you want exact statistical optimality; slower. |
| `anscombe` | Anscombe transform | Variance-stabilising alternative to deviance. |
| `raw` | `\|y - yhat\|` | Baseline only. Not appropriate for insurance data. |

Width hierarchy (narrowest first, coverage identical): `pearson_weighted <= deviance <= anscombe < pearson < raw`.

---

## Temporal calibration

Calibrate on recent data to capture current loss trends:

```python
from insurance_conformal.utils import temporal_split

X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
    X, y,
    calibration_frac=0.20,
    date_col="accident_year",
)

model.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)
```

Target `n_cal >= 2,000` for stable production use. The guarantee holds for any `n_cal >= 1`, but below 500 interval widths are materially wider and more variable.

---

## Coverage guarantee

Split conformal provides:

```
P(y_test in [lower, upper]) >= 1 - alpha
```

Distribution-free — holds regardless of the true data distribution or model misspecification. The assumption is exchangeability: calibration and test observations drawn from the same distribution. Temporal covariate shift violates this — use temporal calibration splits and monitor coverage via `RetroAdj` if abrupt shifts are expected.

---


---

## Conditional Validity Index (CVI) assessment

Marginal coverage is a necessary but insufficient guarantee for regulatory purposes. A conformal predictor covering 90% of the total book but only 82% of young drivers is a Consumer Duty exposure — the portfolio-level number obscures the segment-level failure.

**CVI** makes this measurable. It trains a reliability estimator on calibration data to predict per-instance coverage probability from features. If coverage is uniform, no classifier should predict it from X — the signal will be noise. If it can, those predictions quantify the problem.

CVI decomposes into:

- **CVI_U** (safety): expected undercoverage shortfall for instances where the estimator predicts local coverage below (1-gamma)*(1-alpha). This is the dangerous component — false confidence.
- **CVI_O** (efficiency): expected overcoverage excess. Premium capacity wasted on unnecessarily wide intervals.
- **CVI = CVI_U + CVI_O**

### Two implementations

Both implement the same maths from Zhou et al. (arXiv:2603.27189). Choose based on your dependencies and workflow:

**`ConditionalValidityIndex`** (in `conditional_coverage`): LightGBM backend, single `evaluate()` call, cross-validation inside the call. Requires `pip install insurance-conformal[lightgbm]`.

**`ConditionalCoverageAssessor`** (in `assessment`): sklearn GradientBoostingClassifier — no optional dependencies. Separable `fit`/`score`/`select` API: train the reliability estimator once, then score multiple prediction sets without retraining.

```python
from insurance_conformal.assessment import ConditionalCoverageAssessor

# Train the reliability estimator on your calibration data
assessor = ConditionalCoverageAssessor(alpha=0.10, gamma=0.1)
assessor.fit(X_cal, y_cal, (lower_cal, upper_cal))

# Score a single predictor on the test set
result = assessor.score(X_test, y_test, (lower_test, upper_test))
print(result.cvi_u)     # undercoverage risk
print(result.pi_minus)  # fraction of instances below coverage target

# Select the best predictor from multiple candidates — one fit, K score calls
sel = assessor.select(X_test, y_test, {
    "pearson_weighted": (lo_pw, hi_pw),
    "cqr": (lo_cqr, hi_cqr),
    "locally_weighted": (lo_lw, hi_lw),
})
print(sel.best_key)
print(sel.compare())  # polars DataFrame: predictors ranked by CVI
```

### Consumer Duty use case

A pricing team runs three CQR variants (different quantile model depths) and needs to document which has the most uniform conditional coverage across policyholder segments before production deployment.

The workflow:

1. `assessor.fit(X_cal, y_cal, prediction_sets_cal)` — one classifier training run.
2. `assessor.select(X_test, y_test, {"depth_4": ..., "depth_6": ..., "depth_8": ...})` — three forward passes, no refitting.
3. `sel.compare()` gives a ranked Polars DataFrame; write `cvi_u` per predictor into the model governance pack.

The `cvi_u` value is interpretable: among the policyholders where this predictor systematically under-performs, how large is the average coverage shortfall? A `cvi_u` of 0.03 means those policyholders are on average 3 percentage points below their 90% target. That is a number a pricing actuary can argue about in a model review; "the classifier's ERT is 0.023***" is not.

**Based on:** Zhou, Z., Zhang, X., Tao, C. & Yang, Y. (2026). arXiv:2603.27189.

## Design choices

**Split conformal, not cross-conformal.** Cross-conformal is more statistically efficient but requires refitting the model on each calibration fold. For GBMs that take hours to train, this is not practical. Split conformal trains once, calibrates once.

**No MAPIE dependency.** MAPIE is excellent but does not expose the insurance-specific scores implemented here. The split conformal algorithm is simple enough to own: 20 lines of code for `conformal_quantile()` plus the score functions.

**Polars-native output.** All prediction and diagnostic methods return `pl.DataFrame`. Pandas inputs are accepted.

**Lower bound clipped at zero.** Insurance losses are non-negative. Intervals with negative lower bounds are nonsensical. We clip at zero unconditionally.

**Auto-detection of Tweedie power.** For CatBoost, read from the loss function string. For sklearn `TweedieRegressor`, from `model.power`. Pass `tweedie_power=` explicitly to override.

---

## Limitations

- Coverage is marginal, not conditional. The guarantee holds on average. High-risk subgroups can be systematically under-covered even when aggregate coverage meets the target. Always run `coverage_by_decile()` after calibration.
- Exchangeability is violated by portfolio drift. Mid-year claims inflation, Ogden rate changes, or significant portfolio mix shifts break the exchangeability assumption. Use temporal calibration splits and monitor via `RetroAdj`.
- IBNR on recent accident years produces intervals that are too narrow. Calibrating on development-year 0 or 1 data means non-conformity scores are computed on understated claim totals. Use only accident years with at least 3 years of development, or apply IBNR chain-ladder factors to `y_cal` before calibration.
- `RetroAdj` full method requires kernel ridge regression as the base model. Use residual-only mode for existing GLMs or GBMs.

---

## Part of the Burning Cost stack

Takes any fitted model — Tweedie GBM, GAM, GLM, or the output of [insurance-gam](https://github.com/burning-cost/insurance-gam) or [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity). Feeds distribution-free prediction intervals into [insurance-optimise](https://github.com/burning-cost/insurance-optimise) (uncertainty-aware pricing) and [insurance-governance](https://github.com/burning-cost/insurance-governance) (PRA SS1/23 validation packs). → [See the full stack](https://burning-cost.github.io/stack/)

---

## References

**Foundational theory**

- **Vovk, V., Gammerman, A. & Shafer, G. (2005).** *Algorithmic Learning in a Random World.* Springer. (Foundational text establishing conformal prediction and coverage guarantees.)
- **Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. & Wasserman, L. (2018).** "Distribution-Free Predictive Inference for Regression." *Journal of the American Statistical Association*, 113(523), 1094–1111. [doi:10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116)
- **Angelopoulos, A.N. & Bates, S. (2023).** "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *Foundations and Trends in Machine Learning*, 16(4). [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
- **Romano, Y., Patterson, E. & Candes, E. (2019).** "Conformalized Quantile Regression." *NeurIPS 2019*. [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)
- **Tibshirani, R.J., Barber, R.F., Candes, E. & Ramdas, A. (2019).** "Conformal Prediction Under Covariate Shift." *NeurIPS 2019*. [arXiv:1904.06019](https://arxiv.org/abs/1904.06019)

**Insurance-specific applications**

- **Hong, L. (2025).** "Conformal prediction of future insurance claims in the regression problem." [arXiv:2503.03659](https://arxiv.org/abs/2503.03659)
- **Hong, L. (2026).** "A new strategy for finite-sample valid prediction of future insurance claims in the regression setting." [arXiv:2601.21153](https://arxiv.org/abs/2601.21153)
- **Graziadei, H., Janett, C., Embrechts, P. & Bucher, A. (2023).** "Conformal Prediction for Insurance Data." [arXiv:2307.13124](https://arxiv.org/abs/2307.13124)
- **Angelopoulos, A.N., Bates, S. et al. (2024).** "Conformal Risk Control." *ICLR 2024*. [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)
- **Jun, J. & Ohn, I. (2025).** "Online Conformal Inference with Retrospective Adjustment." [arXiv:2511.04275](https://arxiv.org/abs/2511.04275)

---

## Related libraries

| Library | Description |
|---------|-------------|
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — track coverage stability over time |
| [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts) | Conformal prediction for non-exchangeable claims time series |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | GAM pricing models that feed directly into this library |

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Full conditional distribution per risk: mean, variance, CoV |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries](https://burning-cost.github.io)

---

## Training Course

Want structured learning? [Insurance Pricing in Python](https://burning-cost.github.io/course) is a 12-module course covering the full pricing workflow. Module 11 covers conformal prediction — split conformal, CQR, and coverage guarantees for pricing models. £97 one-time.

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-conformal/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-conformal/issues)
- **Blog & tutorials:** [burning-cost.github.io](https://burning-cost.github.io)

## Licence

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests welcome at [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal).

---

Need help implementing this? [See our consulting services](https://burning-cost.github.io/work-with-us/).
