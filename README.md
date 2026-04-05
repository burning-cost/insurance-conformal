# insurance-conformal

Distribution-free prediction intervals for insurance pricing models — Tweedie non-conformity scores, finite-sample coverage guarantees, SCR bounds, and per-decile coverage diagnostics.

[![Tests](https://github.com/burning-cost/insurance-conformal/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-conformal/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
[![License](https://img.shields.io/pypi/l/insurance-conformal)](https://pypi.org/project/insurance-conformal/)

**Blog post:** [Conformal Prediction Intervals for Insurance Pricing Models](https://burning-cost.github.io/2026/03/06/conformal-prediction-intervals-for-insurance-pricing/)

---

## The problem

Your pricing model gives point estimates. Parametric prediction intervals assume variance scales as mu^p across the whole book — an assumption that breaks exactly where the stakes are highest: large, unusual risks.

On a heterogeneous UK motor portfolio, parametric Tweedie intervals over-cover low-risk policies (unnecessary width) and under-cover the top risk decile, which drives reinsurance attachment, reserving, and SCR calculations.

Conformal prediction fixes this. The guarantee is `P(y in interval) >= 1 - alpha` for any data distribution, as long as calibration and test data are exchangeable. No parametric family required.

The non-obvious implementation detail: most conformal libraries use raw absolute residuals `|y - yhat|`. For insurance data that is wrong — a £1 error on a £100 risk is not the same as a £1 error on a £10,000 risk. The correct score for Tweedie models is `|y - yhat| / yhat^(p/2)`, normalising by the Tweedie standard deviation. That is what this library implements.

---

## Quickstart

```bash
pip install insurance-conformal
```

```python
import numpy as np
from catboost import CatBoostRegressor
from insurance_conformal import InsuranceConformalPredictor

rng = np.random.default_rng(42)
n = 10_000
vehicle_age  = rng.integers(1, 15, n).astype(float)
driver_age   = rng.integers(25, 75, n).astype(float)
ncd_years    = rng.integers(0, 9, n).astype(float)
X = np.column_stack([vehicle_age, driver_age, ncd_years])

# Gamma-distributed claims — heteroskedastic (variance grows with mean)
true_rate  = 200 + 15 * vehicle_age + 3 * np.maximum(30 - driver_age, 0) - 8 * ncd_years
y = rng.gamma(shape=1.5, scale=true_rate / 1.5)

# Temporal split: train / calibrate / test
i_train, i_cal, i_test = n * 6 // 10, n * 8 // 10, n
X_train, y_train = X[:i_train], y[:i_train]
X_cal,   y_cal   = X[i_train:i_cal], y[i_train:i_cal]
X_test,  y_test  = X[i_cal:], y[i_cal:]

# Fit any sklearn-compatible model
fitted_gbm = CatBoostRegressor(iterations=200, loss_function="Tweedie:variance_power=1.5", verbose=0)
fitted_gbm.fit(X_train, y_train)

# Wrap with conformal — calibrate on held-out data
cp = InsuranceConformalPredictor(
    model=fitted_gbm,
    nonconformity="pearson_weighted",  # correct default for Tweedie
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)

# 90% prediction intervals
intervals = cp.predict_interval(X_test, alpha=0.10)

# Always check per-decile coverage — marginal != conditional
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

**Accuracy where it matters.** Parametric Tweedie intervals produce 93% aggregate coverage at a 90% target — fine in aggregate, but that surplus width sits on low-risk policies. The top-risk decile that drives reinsurance and reserving gets marginal coverage.

**Regulatory defensibility.** The distribution-free guarantee does not rely on model fit. You can write "P(claim in interval) >= 90%, finite-sample valid, no parametric assumptions" in a model validation pack under Solvency II Article 120-126 or PRA CP6/24 (insurance model risk). You cannot write that for a parametric bootstrap interval.

**SCR calculations.** `SCRReport` produces per-risk 99.5% upper bounds with a coverage validation table — for internal model stress-testing documentation.

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

The locally-weighted variant meets the 90% target in the top decile by construction. Run the validation: import [`notebooks/databricks_validation.py`](notebooks/databricks_validation.py) into Databricks.

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

**Dependencies:** polars and pandas are both required. Polars is the primary output format — all prediction and diagnostic methods return `pl.DataFrame`. Pandas is required for binning utilities and for accepting pandas DataFrame inputs.

---

## Features

- `InsuranceConformalPredictor` — split conformal wrapping any sklearn-compatible model. Non-conformity scores: `pearson_weighted`, `pearson`, `deviance`, `anscombe`, `raw`.
- `LocallyWeightedConformal` — two-stage conformal with a secondary spread model. Meets per-decile coverage targets that standard conformal misses.
- `ConformalisedQuantileRegression` — split CQR (Romano et al., 2019). Wraps pre-fitted quantile models from CatBoost or LightGBM.
- `FrequencySeverityConformal` — correct conformity scoring for two-stage frequency-severity models (Graziadei et al., 2023).
- `SCRReport` — per-risk 99.5% upper bounds with coverage validation table; for model documentation under Solvency II Article 120-126.
- `insurance_conformal.risk` — Conformal Risk Control (Angelopoulos et al., ICLR 2024): `PremiumSufficiencyController`, `IntervalWidthController`, `SelectiveRiskController`.
- `RetroAdj` — online conformal with retrospective adjustment (Jun & Ohn, 2025). Recovers from abrupt distribution shifts within 1–3 steps.
- `CoverageDiagnostics` — coverage-by-decile plots, interval width distributions, subgroup coverage by arbitrary segment.
- `ConditionalCoverageAssessor` / `ConditionalValidityIndex` — quantify conditional coverage failures across policyholder segments; interpretable CVI_U (undercoverage risk) and CVI_O (overcoverage waste).

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

## Worked examples

### Frequency-severity model with per-decile coverage audit

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

sg = subgroup_coverage(
    predictor=fs, X_test=X_test, y_test=y_test,
    alpha=0.10, groups=vehicle_group_band, group_name="vehicle_group_band",
)
print(sg)
```

The calibration subtlety: feeding observed claim counts into the severity model at calibration time creates a distributional mismatch that breaks the coverage guarantee. `FrequencySeverityConformal` feeds predicted frequency into the severity model at both calibration and test time. See Graziadei et al. (2023) for the proof.

### SCR bounds for internal model documentation

```python
from insurance_conformal import InsuranceConformalPredictor, SCRReport

cp = InsuranceConformalPredictor(model=fitted_model)
cp.calibrate(X_cal, y_cal)

scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
val_table  = scr.coverage_validation_table(X_test, y_test)
print(scr.to_markdown())
```

> **Disclaimer:** `SCRReport` is an internal stress-testing tool. Solvency II SCR calculations for regulatory purposes require sign-off under an approved internal model or the standard formula. Do not use this output in regulatory returns without actuarial review and governance sign-off.

### Premium sufficiency control

```python
from insurance_conformal.risk import PremiumSufficiencyController

psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
psc.calibrate(y_cal, premium_cal)
result = psc.predict(premium_new)

print(f"Required loading: {result['lambda_hat']:.3f}")
# result["upper_bound"]: risk-controlled loaded premium per policy
```

### Recovering from mid-year claims inflation

Standard conformal with a static calibration set breaks when the book shifts mid-year. `RetroAdj` recovers within 1–3 steps:

```python
from insurance_conformal import RetroAdj

resid_train = y_train - glm.predict(X_train)
resid_test  = y_test  - glm.predict(X_test)

model = RetroAdj(window_size=250, gamma=0.005)
model.fit(resid_train)
lower_r, upper_r = model.predict_interval(resid_test, alpha=0.10)
```

| Metric | RetroAdj | Standard ACI |
|--------|----------|-----|
| Steps to recover 90% coverage after +30% inflation shock | ~15–30 | ~80–150 |
| Post-shift coverage (full window) | ~88–91% | ~80–87% |

---

## Conditional coverage assessment

Marginal coverage is a necessary but insufficient guarantee for regulatory purposes. A predictor covering 90% of the total book but only 82% of young drivers is a Consumer Duty exposure.

`ConditionalCoverageAssessor` trains a reliability estimator on calibration data to predict per-instance coverage probability from features. If coverage is uniform, no classifier should predict it from X.

```python
from insurance_conformal.assessment import ConditionalCoverageAssessor

assessor = ConditionalCoverageAssessor(alpha=0.10, gamma=0.1)
assessor.fit(X_cal, y_cal, (lower_cal, upper_cal))

result = assessor.score(X_test, y_test, (lower_test, upper_test))
print(result.cvi_u)     # undercoverage risk
print(result.pi_minus)  # fraction of instances below coverage target

# Select the best predictor from multiple candidates — one fit, K score calls
sel = assessor.select(X_test, y_test, {
    "pearson_weighted": (lo_pw, hi_pw),
    "cqr":              (lo_cqr, hi_cqr),
    "locally_weighted": (lo_lw, hi_lw),
})
print(sel.best_key)
print(sel.compare())   # polars DataFrame: predictors ranked by CVI
```

The `cvi_u` value is interpretable: among the policyholders where this predictor systematically under-performs, how large is the average coverage shortfall? A `cvi_u` of 0.03 means those policyholders are on average 3 percentage points below their 90% target.

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

Target `n_cal >= 2,000` for stable production use. The guarantee holds for any `n_cal >= 1`, but below 500 interval widths are materially wider.

---

## Coverage guarantee

Split conformal provides:

```
P(y_test in [lower, upper]) >= 1 - alpha
```

Distribution-free — holds regardless of the true data distribution or model misspecification. The assumption is exchangeability: calibration and test observations drawn from the same distribution. Temporal covariate shift violates this — use temporal calibration splits and monitor coverage via `RetroAdj`.

---

## Design choices

**Split conformal, not cross-conformal.** Cross-conformal is more statistically efficient but requires refitting the model on each calibration fold. For GBMs that take hours to train, this is not practical. Split conformal trains once, calibrates once.

**No MAPIE dependency.** MAPIE is excellent but does not expose the insurance-specific scores implemented here. The split conformal algorithm is simple enough to own.

**Polars-native output.** All prediction and diagnostic methods return `pl.DataFrame`. Pandas inputs are accepted.

**Lower bound clipped at zero.** Insurance losses are non-negative. Intervals with negative lower bounds are nonsensical.

---

## Limitations

- **Coverage is marginal, not conditional.** The guarantee holds on average. High-risk subgroups can be systematically under-covered. Always run `coverage_by_decile()` after calibration.
- **Exchangeability is violated by portfolio drift.** Mid-year claims inflation, Ogden rate changes, or significant portfolio mix shifts break the assumption. Use temporal calibration splits and monitor via `RetroAdj`.
- **IBNR on recent accident years produces intervals that are too narrow.** Calibrating on development-year 0 or 1 data means non-conformity scores are computed on understated claim totals. Use accident years with at least 3 years of development, or apply IBNR chain-ladder factors to `y_cal` before calibration.
- **`RetroAdj` full method requires kernel ridge regression as the base model.** Use residual-only mode for existing GLMs or GBMs.

---

## References

**Foundational theory**

- Angelopoulos, A.N. & Bates, S. (2023). "A Gentle Introduction to Conformal Prediction." *Foundations and Trends in Machine Learning*, 16(4). [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
- Romano, Y., Patterson, E. & Candes, E. (2019). "Conformalized Quantile Regression." *NeurIPS 2019*. [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)

**Insurance-specific applications**

- Hong, L. (2025). "Conformal prediction of future insurance claims." [arXiv:2503.03659](https://arxiv.org/abs/2503.03659)
- Graziadei, H., Janett, C., Embrechts, P. & Bucher, A. (2023). "Conformal Prediction for Insurance Data." [arXiv:2307.13124](https://arxiv.org/abs/2307.13124)
- Angelopoulos, A.N., Bates, S. et al. (2024). "Conformal Risk Control." *ICLR 2024*. [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)
- Jun, J. & Ohn, I. (2025). "Online Conformal Inference with Retrospective Adjustment." [arXiv:2511.04275](https://arxiv.org/abs/2511.04275)

---

## Part of the Burning Cost stack

Takes any fitted model — Tweedie GBM, GAM, GLM, or the output of [insurance-gam](https://github.com/burning-cost/insurance-gam) or [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity). Feeds distribution-free prediction intervals into [insurance-optimise](https://github.com/burning-cost/insurance-optimise) (uncertainty-aware pricing) and [insurance-governance](https://github.com/burning-cost/insurance-governance) (model validation documentation). → [See the full stack](https://burning-cost.github.io/stack/)

## Related libraries

| Library | Description |
|---------|-------------|
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — track coverage stability over time |
| [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts) | Conformal prediction for non-exchangeable claims time series |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | GAM pricing models that feed directly into this library |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |

[All libraries](https://burning-cost.github.io) | [Discussions](https://github.com/burning-cost/insurance-conformal/discussions) | [Issues](https://github.com/burning-cost/insurance-conformal/issues)

---

## Training course

[Insurance Pricing in Python](https://burning-cost.github.io/course) is a 12-module course covering the full pricing workflow. Module 11 covers conformal prediction — split conformal, CQR, and coverage guarantees for pricing models. £97 one-time.

## Licence

MIT. See [LICENSE](LICENSE).
