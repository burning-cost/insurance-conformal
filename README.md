# insurance-conformal

![Tests](https://github.com/burning-cost/insurance-conformal/actions/workflows/tests.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-conformal) [![Downloads](https://img.shields.io/pypi/dm/insurance-conformal)](https://pypi.org/project/insurance-conformal/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/burning-cost-examples/blob/main/notebooks/burning-cost-in-30-minutes.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-conformal/blob/main/notebooks/quickstart.ipynb)

Your Tweedie GBM's prediction intervals assume variance scales as mu^p across the whole book — an assumption that breaks on heterogeneous UK motor portfolios where high-mean risks are genuinely more dispersed than the parametric family predicts. insurance-conformal gives you distribution-free prediction intervals with a finite-sample coverage guarantee, 13–14% narrower than the parametric baseline on a heteroskedastic motor book.

**Blog post:** [Conformal Prediction Intervals for Insurance Pricing Models](https://burning-cost.github.io/2026/03/06/conformal-prediction-intervals-for-insurance-pricing/)

---

Your pricing model gives point estimates. A reserving actuary needs uncertainty bounds. A capital team needs 99.5th percentile estimates for the SCR. A reinsurer wants to know whether your stated 90% interval actually holds in the top risk decile. Parametric approaches answer these questions by assuming a distribution — and those assumptions break exactly where the stakes are highest: large, unusual risks.

Conformal prediction provides a finite-sample valid alternative. The guarantee is `P(y in interval) >= 1 - alpha` for any data distribution, as long as calibration and test data are exchangeable. No parametric family required.

The non-obvious part is the non-conformity score. Most conformal implementations use the raw absolute residual `|y - yhat|`. For insurance data that is wrong: it treats a £1 error on a £100 risk identically to a £1 error on a £10,000 risk. The correct score for Tweedie models is the Pearson-weighted residual `|y - yhat| / yhat^(p/2)`, which normalises by the Tweedie standard deviation and produces exchangeable scores across risk levels. That is what this library implements.

---

## Part of the Burning Cost stack

Takes any fitted model — Tweedie GBM, GAM, GLM, or the output of [insurance-gam](https://github.com/burning-cost/insurance-gam) or [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity). Feeds distribution-free prediction intervals into [insurance-optimise](https://github.com/burning-cost/insurance-optimise) (uncertainty-aware pricing) and [insurance-governance](https://github.com/burning-cost/insurance-governance) (PRA SS1/23 validation packs). → [See the full stack](https://burning-cost.github.io/stack/)

---

## Why use this?

- Parametric Tweedie prediction intervals use a single dispersion parameter estimated from the calibration set. On a heterogeneous UK motor book, this over-covers low-risk policies (unnecessary width) and under-covers high-risk policies — which is exactly where getting it wrong is most expensive.
- Conformal prediction removes the distributional assumption. The only requirement is exchangeable calibration and test data. On 50,000 synthetic UK motor policies, conformal intervals are 13–14% narrower than parametric while meeting the 90% target, and the locally-weighted variant also meets it in the top risk decile.
- The insurance-specific non-conformity scores (`pearson_weighted`: `|y - yhat| / yhat^(p/2)`) account for Tweedie heteroscedasticity. Using the raw absolute residual is the wrong default for insurance data.
- Conformal Risk Control (`insurance_conformal.risk`) controls expected shortfall directly — find the smallest loading factor such that expected underpricing is bounded. A direct regulatory argument, not a statistical artefact.
- `SCRReport` wraps any calibrated predictor and produces per-risk 99.5% upper bounds for internal stress-testing and model validation documentation.
- The coverage guarantee is distribution-free and finite-sample valid. Suitable for PRA SS1/23 model validation packs.

---

## Comparison: conformal vs parametric bootstrap

| | Parametric Tweedie | `pearson_weighted` conformal | Locally-weighted conformal |
|---|---|---|---|
| Distribution assumption | Tweedie Var ~ mu^p | None | None |
| Aggregate coverage @ 90% target | 93.1% (over-covers) | 90.2% | 90.3% |
| Top-decile coverage @ 90% target | 90.4% | 87.9% | 90.6% |
| Mean interval width | £4,393 | £3,806 (−13.4%) | £3,881 (−11.7%) |
| Width adapts per risk segment | No | Partial | Yes |
| Finite-sample valid guarantee | No | Yes | Yes |
| Requires refitting on calibration | No | No | No |

Benchmark: CatBoost Tweedie(p=1.5), 50k synthetic UK motor policies, heteroskedastic Gamma DGP, temporal 60/20/20 split, seed=42. Run: `benchmarks/benchmark_gbm.py`.

The parametric aggregate of 93.1% at a 90% target signals over-width on low-risk policies. The top decile just barely meets target at 90.4% — but that is coincidental; on a book with more pronounced tail heteroscedasticity it would miss. The locally-weighted conformal variant learns which features predict large residuals and adapts width accordingly: it meets the 90% target in the top decile by construction.

---

## Quick start

```python
import numpy as np
from insurance_conformal import InsuranceConformalPredictor

# --- 1. Your fitted model (any sklearn-compatible predictor) ---
# Here we use a simple example; in production this is your Tweedie GBM
from sklearn.linear_model import TweedieRegressor
rng = np.random.default_rng(42)
n_train, n_cal, n_test = 10_000, 2_000, 2_000
X_train = rng.standard_normal((n_train, 5))
y_train = rng.gamma(shape=1.5, scale=300, size=n_train)
X_cal   = rng.standard_normal((n_cal, 5))
y_cal   = rng.gamma(shape=1.5, scale=300, size=n_cal)
X_test  = rng.standard_normal((n_test, 5))

model = TweedieRegressor(power=1.5, link="log")
model.fit(X_train, y_train)

# --- 2. Wrap with conformal predictor ---
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",  # recommended for Tweedie models
    distribution="tweedie",
    tweedie_power=1.5,
)

# --- 3. Calibrate on held-out data (must not overlap with training) ---
cp.calibrate(X_cal, y_cal)

# --- 4. Generate 90% prediction intervals ---
intervals = cp.predict_interval(X_test, alpha=0.10)
# polars DataFrame with columns: lower, point, upper

# --- 5. Check coverage by decile (always do this) ---
y_test = rng.gamma(shape=1.5, scale=300, size=n_test)
print(cp.coverage_by_decile(X_test, y_test, alpha=0.10))
```

For locally-adaptive intervals (narrower on low-variance risks, wider on high-variance risks):

```python
from insurance_conformal import LocallyWeightedConformal

lw = LocallyWeightedConformal(model=model, tweedie_power=1.5)
lw.fit(X_train, y_train)
lw.calibrate(X_cal, y_cal)
intervals = lw.predict_interval(X_test, alpha=0.10)
```

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

## Features

- `InsuranceConformalPredictor` — split conformal prediction wrapping any sklearn-compatible model. Supports `pearson_weighted`, `pearson`, `deviance`, `anscombe`, and `raw` non-conformity scores.
- `LocallyWeightedConformal` — two-stage conformal with a secondary spread model. Learns which features predict large residuals. Meets per-decile coverage targets that standard conformal misses. Supports CatBoost and LightGBM spread models.
- `ConformalisedQuantileRegression` — split CQR (Romano, Patterson & Candes 2019). Wraps pre-fitted quantile models with a conformal calibration correction. Works with CatBoost `Quantile:alpha=`, LightGBM `objective=quantile`, sklearn `loss=quantile`.
- `FrequencySeverityConformal` — correct conformity scoring protocol for two-stage frequency-severity models (Graziadei et al. 2023). Feeds predicted frequency (not observed count) into the severity model at calibration time, preserving exchangeability.
- `SCRReport` — per-risk 99.5% upper bounds with coverage validation table. For internal stress-testing and PRA SS1/23 model validation documentation.
- `solvency_capital_range()` — lightweight functional API for SCR bounds, returns a `SolvencyCapitalRange` dataclass. Use inside pipelines when you don't need the full `SCRReport`.
- `insurance_conformal.risk` — Conformal Risk Control (Angelopoulos et al., ICLR 2024). Controls expected loss directly. `PremiumSufficiencyController`: find the smallest loading factor such that expected underpricing shortfall is bounded. `IntervalWidthController`, `SelectiveRiskController`.
- `RetroAdj` — online conformal with retrospective adjustment (Jun & Ohn 2025). Recovers from abrupt distribution shifts (claims inflation, Ogden rate changes) within 1–3 steps. KRR base model or residual-only mode for existing GBMs.
- `CoverageDiagnostics` — coverage-by-decile plots and interval width distribution. `subgroup_coverage()` for coverage by arbitrary segment (age band, vehicle group, area).
- `insurance_conformal.multivariate` — joint multi-output conformal prediction for simultaneous frequency/severity intervals. `JointConformalPredictor`, `SolvencyCapitalEstimator`.

---

## Expected Performance

On a 50,000-policy heteroskedastic Gamma UK motor book (CatBoost Tweedie(p=1.5), temporal 60/20/20 split, seed=42):

| Metric | Parametric Tweedie | Conformal (pearson_weighted) | LW Conformal |
|--------|-------------------|------------------------------|--------------|
| Aggregate coverage @ 90% | 0.931 | 0.902 | 0.903 |
| Top-decile coverage @ 90% | 0.904 | 0.879 | 0.906 |
| Mean interval width (£) | 4,393 | 3,806 | 3,881 |
| Width vs parametric | ref | −13.4% | −11.7% |
| Distribution-free guarantee | No | Yes | Yes |

The parametric aggregate of 93.1% at a 90% target signals over-width on low-risk policies. Conformal is 13.4% narrower with a valid coverage guarantee. LW conformal also meets the 90% target in the top decile — the one that drives reinsurance attachment and reserving decisions.

Run the validation: import `notebooks/databricks_validation.py` into Databricks.

---

## Coverage diagnostics

The marginal coverage guarantee means `P(y in interval) >= 1 - alpha` averaged over all observations. In insurance, you also need to check that coverage is uniform across risk deciles — a model can achieve 90% overall while only covering 65% of high-risk policies.

```python
# Check coverage by decile (always run this after calibration)
diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
print(diag)
#    decile  mean_predicted  n_obs  coverage  target_coverage
# 0       1          0.0234    400     0.923             0.90
# ...
# 9      10          2.3410    400     0.905             0.90

# Full summary
cp.summary(X_test, y_test, alpha=0.10)

# Coverage by arbitrary segment (age band, vehicle group, area)
from insurance_conformal import subgroup_coverage
sg = subgroup_coverage(
    predictor=cp,
    X_test=X_test,
    y_test=y_test,
    alpha=0.10,
    groups=vehicle_group_band,
    group_name="vehicle_group_band",
)

# Matplotlib plots
from insurance_conformal import CoverageDiagnostics
intervals = cp.predict_interval(X_test, alpha=0.10)
diag_tool = CoverageDiagnostics(
    y_true=y_test,
    y_lower=intervals["lower"].to_numpy(),
    y_upper=intervals["upper"].to_numpy(),
    y_pred=intervals["point"].to_numpy(),
    alpha=0.10,
)
fig = diag_tool.coverage_plot()
fig2 = diag_tool.interval_width_distribution()
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

Width hierarchy (narrowest first, coverage identical): `pearson_weighted <= deviance <= anscombe < pearson < raw`. Ordering is approximate and depends on Tweedie power.

---

## Temporal calibration

In insurance, calibrate on recent data to capture current loss trends:

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

---

## Conformal Risk Control

Standard conformal controls coverage probability. For insurance pricing the question that matters is different: how much are we underpriced, in expectation?

`insurance_conformal.risk` implements Conformal Risk Control (CRC, Angelopoulos et al., ICLR 2024), which controls expected loss directly: `E[L(C_lambda(X), Y)] <= alpha` for any bounded monotone loss L. Finite-sample valid, no parametric assumptions.

**Lead use case: premium sufficiency control.** Find the smallest loading factor lambda* such that expected shortfall from underpriced policies is bounded:

```python
from insurance_conformal.risk import PremiumSufficiencyController

psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
psc.calibrate(y_cal, premium_cal)   # calibrate on held-out year
result = psc.predict(premium_new)   # apply to next year's book
# result["upper_bound"]: risk-controlled loading factor per policy
# result["lambda_hat"]: the single lambda* that achieves E[shortfall] <= 5%
```

| Controller | Use case |
|---|---|
| `PremiumSufficiencyController` | Bound expected underpricing shortfall: E[max(claim - lambda * premium, 0) / premium] <= alpha |
| `IntervalWidthController` | Find the most efficient conformal quantile level that still bounds expected interval width |
| `SelectiveRiskController` | Accept/reject risks to bound expected loss on the accepted book |

---

## FrequencySeverityConformal

Conformal prediction intervals for frequency-severity insurance models (Graziadei et al. arXiv:2307.13124).

The calibration subtlety: using the observed claim count in the severity model at calibration time creates a distributional mismatch that breaks the coverage guarantee. The correct approach is to feed the predicted frequency (not the observed count) into the severity model at both calibration and test time.

```python
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from insurance_conformal.claims import FrequencySeverityConformal

fs = FrequencySeverityConformal(
    freq_model=PoissonRegressor(),
    sev_model=GammaRegressor(),
)

fs.fit(X_train, d_train, y_train)   # d_train = observed claim counts
fs.calibrate(X_cal, d_cal, y_cal)   # d_cal for validation only; scores use mu_hat(x)
intervals = fs.predict_interval(X_test, alpha=0.10)
# polars DataFrame with columns: lower, point, upper
```

---

## SCRReport

Per-risk 99.5% upper bounds for internal stress-testing and model validation.

```python
from insurance_conformal import InsuranceConformalPredictor, SCRReport

cp = InsuranceConformalPredictor(model=fitted_model)
cp.calibrate(X_cal, y_cal)
scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
val_table = scr.coverage_validation_table(X_test, y_test)
print(scr.to_markdown())
```

Or use the functional API inside a pipeline:

```python
from insurance_conformal import solvency_capital_range

result = solvency_capital_range(predictor=cp, X=X_test, alpha=0.005)
# result.scr_estimate, result.lower_bound, result.upper_bound
# result.total_scr, result.mean_interval_width
```

> **Disclaimer**: SCRReport is an internal stress-testing tool. Solvency II SCR calculations for regulatory purposes require sign-off under an approved internal model or the standard formula. Do not use this output in regulatory returns without appropriate actuarial review, governance sign-off, and alignment with your firm's approved methodology.

---

## RetroAdj: online conformal for distribution shifts

Standard conformal with a static calibration set breaks when the book shifts mid-year — claims inflation, Ogden rate changes, CAT events. ACI (Adaptive Conformal Inference) adapts by nudging the miscoverage level one step at a time, but at gamma=0.005 it needs ~200 steps to reprice. That is 17 years of monthly data — not adaptation.

`RetroAdj` (Jun & Ohn 2025, arXiv:2511.04275) fixes this by retroactively correcting all leave-one-out residuals in the sliding window simultaneously at each step. After an abrupt shift, coverage recovers within 1–3 steps.

```python
from insurance_conformal import RetroAdj

# KRR base model
model = RetroAdj(bandwidth=1.0, lambda_reg=0.1, window_size=250, gamma=0.005)
model.fit(y_train, X_train)
lower, upper = model.predict_interval(y_test, X_test, alpha=0.10)

# Residual-only mode: wrap an existing GLM or GBM
resid_train = y_train - glm.predict(X_train)
resid_test  = y_test  - glm.predict(X_test)
model2 = RetroAdj(window_size=250)
model2.fit(resid_train)
lower_r, upper_r = model2.predict_interval(resid_test, alpha=0.10)
lower_claims = lower_r + glm.predict(X_test)
upper_claims = upper_r + glm.predict(X_test)
```

**Hard constraint:** the base model must be kernel ridge regression or another self-stable linear smoother. GLMs and GBMs do not qualify for the full method — use residual-only mode.

**Recovery speed after a +30% claims inflation event (gamma=0.005, window=200):**

| Metric | RetroAdj | ACI |
|--------|----------|-----|
| Steps to recover 90% coverage | ~15–30 | ~80–150 |
| Post-shift coverage (full window) | ~88–91% | ~80–87% |
| Speedup | 3–8x faster | baseline |

---

## Coverage guarantee

Split conformal prediction provides the following guarantee for exchangeable data:

```
P(y_test in [lower, upper]) >= 1 - alpha
```

This is distribution-free — it holds regardless of the true data distribution or model misspecification. The assumption is exchangeability: calibration and test observations must be drawn from the same distribution. Temporal covariate shift violates this assumption and can degrade coverage in practice. Use temporal calibration splits (calibrate on the most recent accident year before the test period) to minimise the distribution gap.

For calibration set size: target n_cal >= 2,000 for stable production use. The guarantee holds for any n_cal >= 1, but with n_cal < 500 the quantile estimate has high variance and intervals will be materially wider and more variable.

---

## Design choices

**Split conformal, not cross-conformal.** Cross-conformal is more statistically efficient but requires refitting the model on each calibration fold. For GBMs that take hours to train, this is not practical. Split conformal trains once, calibrates once.

**No MAPIE dependency.** MAPIE is excellent but does not expose the insurance-specific scores implemented here. The split conformal algorithm is simple enough to own: 20 lines of code for `conformal_quantile()` plus the score functions.

**Polars-native output.** All prediction and diagnostic methods return `pl.DataFrame`. Pandas inputs are accepted.

**Lower bound clipped at zero.** Insurance losses are non-negative. Prediction intervals with negative lower bounds are nonsensical. We clip at zero unconditionally.

**Auto-detection of Tweedie power.** For CatBoost, the power parameter is read from the loss function string. For sklearn `TweedieRegressor`, from `model.power`. If detection fails, we warn and default to p=1.5. Pass `tweedie_power=` explicitly if you know the correct value.

---

## Limitations

- Coverage is marginal, not conditional. The conformal guarantee holds on average. High-risk subgroups can be systematically under-covered even when aggregate coverage meets the target. Always run `coverage_by_decile()` after calibration.
- Exchangeability is violated by portfolio drift. Mid-year claims inflation, Ogden rate changes, or significant portfolio mix shifts break the exchangeability assumption. Use temporal calibration splits and monitor coverage via `RetroAdj` if abrupt shifts are expected.
- IBNR on recent accident years produces intervals that are too narrow. Calibrating on development-year 0 or 1 data means non-conformity scores are computed on understated claim totals. Use only accident years with at least 3 years of development, or apply IBNR chain-ladder factors to `y_cal` before calibration.
- `RetroAdj` requires kernel ridge regression as the base model. Use residual-only mode for existing GLMs or GBMs.

---

## References

- **Hong, L. (2025).** "Conformal prediction of future insurance claims in the regression problem." arXiv:2503.03659.
- **Hong, L. (2026).** "A new strategy for finite-sample valid prediction of future insurance claims in the regression setting." arXiv:2601.21153.
- **Graziadei, H., Janett, C., Embrechts, P. & Bucher, A. (2023).** "Conformal Prediction for Insurance Data." arXiv:2307.13124.
- **Manna, S. et al. (2025).** "Conformal Prediction Inference in Regularized Insurance Models." Wiley ASMB; arXiv:2507.06921.
- **Angelopoulos, A. N., Bates, S. et al. (2024).** "Conformal Risk Control." ICLR 2024. arXiv:2208.02814.
- **Jun, J. & Ohn, I. (2025).** "Online Conformal Inference with Retrospective Adjustment." arXiv:2511.04275.
- **Romano, Y., Patterson, E. & Candes, E. (2019).** "Conformalized Quantile Regression." NeurIPS 2019. arXiv:1905.03222.

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

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-conformal/discussions). Found it useful? A star helps others find it.

## Licence

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests welcome at [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal).

---

Need help implementing this? [See our consulting services](https://burning-cost.github.io/work-with-us/).
