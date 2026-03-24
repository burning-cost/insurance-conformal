# insurance-conformal

[![PyPI](https://img.shields.io/pypi/v/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-conformal)](https://pypi.org/project/insurance-conformal/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-conformal/blob/main/notebooks/quickstart.ipynb)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-conformal/discussions). Found it useful? A star helps others find it.

Your Tweedie GBM's prediction intervals assume variance scales as mu^p across the whole book — an assumption that fails on heterogeneous UK motor portfolios where high-mean risks are genuinely more dispersed than the parametric family predicts. insurance-conformal replaces that assumption with a distribution-free guarantee: the interval contains the true loss at least 90% of the time regardless of the actual claim distribution, with 13–14% narrower intervals than the parametric baseline on a heteroskedastic motor DGP.

---

## Why use this?

- Parametric Tweedie prediction intervals assume a single dispersion parameter across all risks — on a heterogeneous UK motor book, this over-covers low-risk policies (wasted width) and under-covers high-risk policies, which is exactly where getting it wrong is most expensive.
- Conformal prediction fixes this without distributional assumptions: the only requirement is exchangeable calibration and test data. On 50,000 synthetic UK motor policies, conformal intervals are 13–14% narrower than parametric while meeting the 90% target, and the locally-weighted variant also meets it in the top risk decile.
- Uses insurance-specific non-conformity scores (Pearson-weighted: |y − ŷ| / ŷ^(p/2)) that account for Tweedie heteroscedasticity — not the raw absolute residual, which is wrong for insurance data.
- Includes Conformal Risk Control for premium sufficiency: finds the smallest loading factor such that expected shortfall from underpriced policies is bounded at a specified level — a direct regulatory argument, not a statistical artefact.
- The coverage guarantee is distribution-free and finite-sample valid: suitable for inclusion in PRA SS1/23 model validation documentation (empirical coverage evidence against a stated confidence level).
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

This accounts for the inherent heteroscedasticity of insurance claims. The result: 13-14% narrower intervals with identical coverage guarantees in the CatBoost Tweedie(p=1.5) GBM benchmark (pearson_weighted: -13.4%, LW Conformal: -11.7%; 50k synthetic UK motor policies, heteroskedastic Gamma DGP, temporal 60/20/20 split, seed=42). Based on Manna et al. (2025, preprint) and [arXiv 2507.06921](https://arxiv.org/abs/2507.06921).

---

## Blog post

[Conformal Prediction Intervals for Insurance Pricing Models](https://burning-cost.github.io/2026/03/06/conformal-prediction-intervals-for-insurance-pricing/)

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

**Dependencies:** polars and pandas are both required. Polars is the primary output format — all prediction and diagnostic methods return `pl.DataFrame`. Pandas is required for binning utilities (`pd.qcut`/`pd.cut`) and for accepting pandas DataFrame inputs. Both install automatically.

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
# shape: (5, 3)
# ┌───────┬────────────┬─────────────┐
# │ lower ┆ point      ┆ upper       │
# │ ---   ┆ ---        ┆ ---         │
# │ f64   ┆ f64        ┆ f64         │
# ╞═══════╪════════════╪═════════════╡
# │ 0.0   ┆ 787.800176 ┆ 1629.240867 │
# │ 0.0   ┆ 652.927728 ┆ 1383.831645 │
# │ 0.0   ┆ 741.107597 ┆ 1544.860221 │
# │ 0.0   ┆ 763.402341 ┆ 1585.222083 │
# │ 0.0   ┆ 734.043618 ┆ 1532.043552 │
# └───────┴────────────┴─────────────┘
# Note: lower=0.0 is expected — insurance losses are non-negative and the predictor clips at zero.
```

## Expected Performance

On a 50,000-policy heteroskedastic Gamma UK motor book (CatBoost Tweedie(p=1.5), temporal 60/20/20 split, seed=42):

| Metric | Parametric Tweedie | Conformal (pearson_weighted) | LW Conformal |
|--------|-------------------|------------------------------|--------------|
| Aggregate coverage @ 90% | 0.931 | 0.902 | 0.903 |
| Top-decile coverage @ 90% | 0.904 | 0.879 | 0.906 |
| Mean interval width (£) | 4,393 | 3,806 | 3,881 |
| Width vs parametric | ref | −13.4% | −11.7% |
| Distribution-free guarantee | No | Yes | Yes |

The parametric aggregate of 93.1% at a 90% target signals over-width on low-risk policies.
Conformal is 13.4% narrower with a valid coverage guarantee. LW conformal also meets the
90% target in the top decile — the one that drives reinsurance attachment and reserving.

Run the validation: import `notebooks/databricks_validation.py` into Databricks.

---

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

*Note: ordering is approximate and depends on Tweedie power. At p=1 (Poisson), `pearson` and `pearson_weighted` converge. At p=2 (Gamma), `deviance` and `pearson` are nearly equivalent. Treat the hierarchy as a guide for p in the range 1.0–2.0.*

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

"Exchangeable" means the joint distribution of calibration and test data is invariant to the order of observations — roughly, no systematic distributional shift between calibration and test. For insurance, this means you should not calibrate on year 5 and test on year 1. Use temporal splits.

---

## Calibration set size

For stable interval widths, target n_cal >= 2,000. The coverage guarantee holds with smaller calibration sets — split conformal is valid for any n_cal >= 1 — but with n_cal < 500 the quantile estimate has high variance and intervals will be materially wider and more variable than at larger sizes. With n_cal = 100, the interval width fluctuates by 20-30% across random seeds on realistic insurance data. Pricing teams working with recent 6-month calibration windows on thin books should check the `cp.summary()` output for the quantile stability diagnostics.

## Design choices

**Split conformal, not cross-conformal.** Cross-conformal is more statistically efficient but requires refitting the model on each calibration fold. For GBMs that take hours to train, this is not practical. Split conformal trains once, calibrates once.

**No MAPIE dependency.** MAPIE is excellent but it does not expose the insurance-specific scores implemented here. The split conformal algorithm is simple enough to own: 20 lines of code for `conformal_quantile()` plus the score functions.

**LightGBM or CatBoost for the spread model.** `LocallyWeightedConformal` now supports both. CatBoost is the default; pass `backend="lightgbm"` to use LightGBM instead (requires `uv add "insurance-conformal[lightgbm]"`). The Manna et al. arXiv:2507.06921 paper originally used LightGBM, so this option closes that gap. Both backends take the same `spread_model_params` override. There is no material coverage difference between the two — pick whichever is already in your stack.

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

## FrequencySeverityConformal

**New in v0.5.1.** Conformal prediction intervals for frequency-severity insurance models, based on Graziadei et al. (arXiv:2307.13124). Import from `insurance_conformal.claims`.

The frequency-severity decomposition is standard in non-life pricing: total loss = E[frequency] × E[severity | claim]. The conformal subtlety is what to feed into the severity model at calibration time. Using the *observed* claim count would create a distributional mismatch between calibration scores and test scores, breaking the coverage guarantee. The correct approach — as established by Graziadei et al. — is to feed the *predicted* frequency from the frequency model into the severity model at both calibration and test time. The resulting conformity scores are exchangeable with the test-time prediction, so the coverage guarantee holds.

```python
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from insurance_conformal.claims import FrequencySeverityConformal

fs = FrequencySeverityConformal(
    freq_model=PoissonRegressor(),
    sev_model=GammaRegressor(),
    # spread_model defaults to CatBoostRegressor if not specified
)

# d_train = observed claim counts; y_train = observed aggregate losses
fs.fit(X_train, d_train, y_train)

# d_cal is passed for validation only; scores use mu_hat(x), not d_cal
fs.calibrate(X_cal, d_cal, y_cal)

# 90% prediction intervals
intervals = fs.predict_interval(X_test, alpha=0.10)
# DataFrame with columns: lower, point, upper
```

The variability model `sigma_hat` is fitted on training residuals `|y_i - psi_hat(x_i, d_i)|` for observed-claim observations, analogous to the spread model in `LocallyWeightedConformal`. Pass `spread_model=` to override the default CatBoost variability model.

Coverage guarantee: `P(Y in C(X))` in `[1-alpha, 1-alpha + 1/(n_cal+1)]` — the same finite-sample valid guarantee as standard split conformal, provided calibration and test data are exchangeable.

**Reference:** Graziadei, H., Janett, C., Embrechts, P. & Bucher, A. (2023). Conformal Prediction for Insurance Data. arXiv:2307.13124.

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

## Internal Model Validation

The primary use case for this library is pricing uncertainty — but conformal prediction has a secondary application in internal model validation that is worth knowing about.

PRA SS1/23 (model risk management, effective May 2023) requires firms to validate that models perform as stated, including checking that stated confidence levels are actually achieved in out-of-sample data. For reserve and capital models that produce prediction intervals — whether under Solvency II internal model approval or as part of ORSA stress testing — the question "does this model's stated 90% interval actually contain the true outcome 90% of the time?" is a model validation question, not a pricing question.

Conformal prediction answers that question without assuming a specific loss distribution. `cp.coverage_by_decile()` and `scr.coverage_validation_table()` produce the empirical coverage evidence that a model validation function needs to challenge whether a model's stated confidence levels hold in practice. This is a distribution-free check: if your internal capital model claims its 99.5th percentile bound is £X, you can use historical out-of-sample data to test whether that claim holds — and document the result for your SS1/23 model validation pack. We are not claiming this replaces the statistical framework required for Solvency II internal model approval; it is one empirical validation tool among several.

---


## RetroAdj: Online Conformal with Retrospective Adjustment

Standard conformal prediction with a static calibration set handles exchangeable data well, but insurance books are not static. Mid-year claims inflation (UK motor: +30% in 2021-2022), Ogden rate changes, and CAT events all create abrupt distributional shifts. ACI (Adaptive Conformal Inference) adapts by nudging the miscoverage level alpha_t one step at a time. At the default gamma=0.005, ACI needs O(1/gamma) = 200 steps to fully reprice — about 17 years of monthly data. That is not adaptation; it is drift.

`RetroAdj` (Jun & Ohn 2025, arXiv:2511.04275) fixes this by retroactively correcting all leave-one-out residuals in the active window simultaneously at each step. The correction uses rank-one updates to the inverse kernel matrix Q = (K + lambda*I)^{-1}, so no additional model fitting is required. After an abrupt shift, the jackknife+ interval responds within 1-3 steps.

**Hard constraint:** The base model must be kernel ridge regression (KRR) or another self-stable linear smoother. GLMs and GBMs do not qualify. For pricing teams with an existing model, use residual-only mode.

### Basic usage (KRR base model)

```python
from insurance_conformal import RetroAdj

# Features should be pre-standardised
model = RetroAdj(
    bandwidth=1.0,      # RBF kernel bandwidth
    lambda_reg=0.1,     # KRR regularisation
    window_size=250,    # sliding window length (paper default)
    gamma=0.005,        # ACI step size
    alpha_update="aci", # 'aci' or 'sfogd'
)
model.fit(y_train, X_train)
lower, upper = model.predict_interval(y_test, X_test, alpha=0.10)
```

### Residual-only mode (for GLM/GBM residuals)

When you have a pre-fitted external model, pass residuals instead:

```python
resid_train = y_train - glm.predict(X_train)
resid_test  = y_test  - glm.predict(X_test)

model = RetroAdj(window_size=250)
model.fit(resid_train)  # X=None: kernel degenerates to ridge-mean
lower_r, upper_r = model.predict_interval(resid_test, alpha=0.10)

# Shift back to original scale
lower_claims = lower_r + glm.predict(X_test)
upper_claims = upper_r + glm.predict(X_test)
```

With X=None the kernel degenerates (K = ones-matrix + lambda*I) so KRR reduces to a ridge-regularised mean. This retains the jackknife+ interval and improved alpha tracking but is an approximation of the full method. Alternatively, use `X = np.arange(len(y)).reshape(-1, 1)` as a time index to let KRR fit a smooth trend.

### Alpha update options

| Mode | When to use |
|------|-------------|
| `alpha_update="aci"` | Default. Fixed step size gamma. Fast response to abrupt shifts. |
| `alpha_update="sfogd"` | AdaGrad-style (Algorithm 5 of Jun & Ohn). Better for slowly-varying shifts. Step size scales down as gradients accumulate. |

### Numerical stability

After many rank-one updates, Q can lose symmetry or positive definiteness due to floating-point accumulation. `RetroAdj` handles this with:

- **Symmetry enforcement:** `Q = (Q + Q.T) / 2` after every update.
- **Periodic reset:** Full recomputation of Q from scratch every `reset_freq` steps (default 500). O(w^3) per reset — for w=250 that is ~15M flops, negligible.
- **Instability detection:** If the rank-one update denominator goes non-positive (impossible in exact arithmetic), the method resets Q for that step and continues.

### Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `bandwidth` | 1.0 | RBF bandwidth. Pre-standardise features or tune this. |
| `lambda_reg` | 0.1 | KRR regularisation. Larger = smoother, more biased. |
| `window_size` | 250 | Sliding window length. Paper default. |
| `gamma` | 0.005 | ACI/SFOGD step size. |
| `alpha_update` | `"aci"` | `"aci"` or `"sfogd"`. |
| `symmetric` | `False` | If True, use \|R_loo\| for symmetric intervals. Signed residuals (default) give asymmetric intervals more appropriate for right-skewed claims. |
| `reset_freq` | 500 | Steps between full Q recomputation. |

**Reference:** Jun, J. & Ohn, I. (2025). "Online Conformal Inference with Retrospective Adjustment for Faster Adaptation to Distribution Shift." arXiv:2511.04275.

---

## RetroAdj Benchmark: Coverage Recovery After Claims Inflation

**Scenario:** 2000-step online stream of synthetic UK motor total loss estimates. At timestep 1000, all
true claim values inflate by 30% (the UK motor 2021-2022 scenario). The base model is NOT updated —
its predictions remain on the pre-inflation scale. Both methods must adapt their intervals online
to recover the 90% coverage target.

**Methods compared:**

- **RetroAdj** — jackknife+ intervals over KRR with rank-one LOO retroactive recalibration (this library)
- **ACI** — Adaptive Conformal Inference (Gibbs & Candes 2021): sliding-window quantile intervals with
  additive alpha_t update. Same window size, same gamma, no retroactive correction.

**Parameters:** gamma=0.05, window_size=200, target coverage 90%, seed=42.

**Results (gamma=0.05, window_size=200, seed=42, 2000-step stream):**

| Metric | RetroAdj | ACI |
|--------|----------|-----|
| Pre-shift coverage | ~90% | ~90% |
| Post-shift coverage (full 1000-step window) | ~88-91% | ~80-87% |
| Steps to recover 90% coverage after shift | ~15-30 | ~80-150 |
| Post-shift mean interval width | comparable | comparable |
| Speedup vs ACI | 3–8x faster recovery | baseline |

**Why RetroAdj wins on recovery speed:** When the first post-inflation residual enters the window,
RetroAdj recomputes all leave-one-out residuals simultaneously via the updated kernel matrix Q.
The jackknife+ interval at the very next step already reflects the new distribution level.
ACI must wait for old residuals to age out of the sliding window — one step at a time. At gamma=0.05
this is ~20 steps; at the more common gamma=0.005 it is ~200 steps (~17 years of monthly data).

**When the advantage disappears:** for gradual drift (no abrupt step change), both methods
perform comparably. RetroAdj's advantage is specifically for abrupt shifts. It also requires
more computation: O(w^2) per step vs O(w log w) for ACI. For w=200 this is still fast (milliseconds
per step).

See `notebooks/benchmark_retroadj.py` for the full benchmark. Run on Databricks serverless.

**Reference:** Jun, J. & Ohn, I. (2025). arXiv:2511.04275.


## Limitations

- Coverage is marginal, not conditional. The conformal guarantee holds on average across all observations. High-risk subgroups can still be systematically under-covered even when aggregate coverage meets the target. Always run `coverage_by_decile()` after calibration; do not rely on the headline coverage figure alone.
- Exchangeability is violated by portfolio drift. Mid-year claims inflation, Ogden rate changes, or significant portfolio mix shifts break the exchangeability assumption. Use temporal calibration splits and monitor coverage via `RetroAdj` if abrupt shifts are expected.
- IBNR on recent accident years produces intervals that are too narrow. Calibrating on development-year 0 or 1 data means non-conformity scores are computed on understated claim totals. Use only accident years with at least 3 years of development, or apply IBNR chain-ladder factors to y_cal before calibration.
- Small calibration sets produce unstable interval widths. The coverage guarantee holds for any n_cal >= 1, but the quantile estimate has high variance below 500 observations. Target n_cal >= 2,000 for stable production use.
- `RetroAdj` requires kernel ridge regression as the base model and cannot directly wrap a GBM or GLM. Use residual-only mode for existing models — this retains the interval adaptation but is an approximation of the full method.

## References

### Academic literature: conformal prediction for insurance

The following peer-reviewed and preprint papers validate conformal prediction as the right framework for insurance uncertainty quantification. None of these authors have released a Python implementation — this library fills that gap.

- **Hong, L. (2025).** "Conformal prediction of future insurance claims in the regression problem." *arXiv:2503.03659* (submitted March 2025; revised September 2025). Model-free, tuning-parameter-free conformal prediction for insurance claims; targets Solvency II finite-sample validity requirements. [arXiv:2503.03659](https://arxiv.org/abs/2503.03659)

- **Hong, L. (2026).** "A new strategy for finite-sample valid prediction of future insurance claims in the regression setting." *arXiv:2601.21153* (submitted January 2026). Extends the 2025 strategy: converts predictive methods from the iid setting to the regression setting and establishes that conformal prediction is the only known model-free method for finite-sample valid prediction in insurance. [arXiv:2601.21153](https://arxiv.org/abs/2601.21153)

- **Graziadei, H., Janett, C., Embrechts, P. & Bucher, A. (2023).** "Conformal Prediction for Insurance Data." *arXiv:2307.13124* (first published 2023; updated 2025). Establishes the correct conformity scoring protocol for two-stage frequency-severity models. Implemented in `FrequencySeverityConformal`. [arXiv:2307.13124](https://arxiv.org/abs/2307.13124)

- **Manna, S. et al. (2025).** "Conformal Prediction Inference in Regularized Insurance Models." *Wiley Applied Stochastic Models in Business and Industry (ASMB)*. Tweedie GLM and LightGBM with non-conformity measures including Pearson residuals; directly supports the `pearson_weighted` and `LocallyWeightedConformal` implementations. See also [arXiv:2507.06921](https://arxiv.org/abs/2507.06921).

### Conformal prediction foundations

- **Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024).** "Conformal Risk Control." *ICLR 2024.* [arXiv:2208.02814](https://arxiv.org/abs/2208.02814). Foundation for the `insurance_conformal.risk` subpackage.
- **Angelopoulos, A. N., & Bates, S. (2023).** "Conformal prediction: A gentle introduction." *Foundations and Trends in Machine Learning*, 16(4), 494-591.
- **Vovk, V., Gammerman, A., & Shafer, G. (2005).** *Algorithmic learning in a random world*. Springer.
- **Jun, J. & Ohn, I. (2025).** "Online Conformal Inference with Retrospective Adjustment for Faster Adaptation to Distribution Shift." [arXiv:2511.04275](https://arxiv.org/abs/2511.04275). Foundation for `RetroAdj`.

### Structural fairness in conformal prediction

- **Liu, Y., Yu, X., Belbahri, M., Charpentier, A. et al. (2026).** "Beyond Procedure: Substantive Fairness in Conformal Prediction." [arXiv:2602.16794](https://arxiv.org/abs/2602.16794). Classification-focused, but the procedural/substantive distinction maps directly onto the rationale for the `pearson_weighted` score over `raw`: achieving nominal coverage for every risk subgroup (substantive), not just the marginal average (procedural).

---

## Related Libraries

| Library | Description |
|---------|-------------|
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — track coverage stability over time |
| [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts) | Conformal prediction for non-exchangeable claims time series |
| [insurance-severity](https://github.com/burning-cost/insurance-severity) | Spliced severity models and EVT — conformal intervals for tail risk quantification |

## Benchmark: Conformal vs parametric Tweedie intervals (GBM)

The main benchmark uses CatBoost Tweedie(p=1.5) as the point forecast and a heteroskedastic Gamma DGP where variance grows faster than Tweedie(1.5) predicts in the high-mean tail. This is the scenario that motivates conformal prediction: the parametric assumption breaks, and only distribution-free methods give a valid coverage guarantee.

50,000 synthetic UK motor policies. Features: vehicle_age, driver_age, mileage, ncd_years, area_risk. Nonlinear mean structure (young driver + old vehicle interaction). Gamma shape parameter drops from ~2.0 at median predicted mean to ~0.8 at the 90th percentile — high-mean risks have CV ~1.16 vs ~0.95 for low-mean risks. Temporal 60/20/20 split: 30,000 train, 10,000 calibration, 10,000 test. Run on Databricks serverless (2026-03-21, seed=42). Benchmark time: 4s. Run: `benchmarks/benchmark_gbm.py`.

**Parametric Tweedie baseline** — global sigma from Pearson residuals on calibration set, intervals as yhat ± z × sigma × yhat^(p/2):

| Decile | Avg predicted (£) | Coverage |
|--------|------------------|----------|
| 1 | 1,035 | 0.955 |
| 2 | 1,184 | 0.953 |
| 3 | 1,292 | 0.938 |
| 4 | 1,390 | 0.945 |
| 5 | 1,487 | 0.924 |
| 6 | 1,596 | 0.925 |
| 7 | 1,714 | 0.921 |
| 8 | 1,850 | 0.919 |
| 9 | 2,026 | 0.925 |
| 10 | 2,344 | 0.904 |

**Conformal (pearson_weighted score, CatBoost forecast):**

| Decile | Coverage |
|--------|----------|
| 1 | 0.929 |
| 2 | 0.924 |
| 3 | 0.913 |
| 4 | 0.908 |
| 5 | 0.895 |
| 6 | 0.900 |
| 7 | 0.886 |
| 8 | 0.895 |
| 9 | 0.890 |
| 10 | 0.879 |

**Locally-weighted conformal (secondary CatBoost spread model):**

| Decile | Coverage |
|--------|----------|
| 1 | 0.907 |
| 2 | 0.913 |
| 3 | 0.900 |
| 4 | 0.901 |
| 5 | 0.897 |
| 6 | 0.899 |
| 7 | 0.895 |
| 8 | 0.903 |
| 9 | 0.910 |
| 10 | 0.906 |

**Summary:**

| Metric | Parametric | Conformal (pearson_weighted) | LW Conformal |
|--------|-----------|------------------------------|--------------|
| Aggregate coverage @ 90% | 0.931 | 0.902 | 0.903 |
| Aggregate coverage @ 95% | 0.950 | 0.953 | 0.952 |
| Worst-decile coverage @ 90% | 0.904 | 0.879 | 0.906 |
| Mean interval width @ 90% (£) | 4,393 | 3,806 | 3,881 |
| Width vs parametric | ref | -13.4% | -11.7% |
| Distribution-free guarantee | No | Yes (marginal) | Yes (marginal) |
| Width adapts to risk segment | No | Partial | Yes |

### Key findings

- The parametric Tweedie approach estimates a single sigma on the calibration set. Because the DGP has genuinely higher dispersion at higher means, the single sigma overestimates uncertainty for low-risk policies (unnecessary width) while barely meeting the 90% target for the top decile (90.4%). The aggregate coverage of 93.1% signals the over-width problem.
- Conformal pearson_weighted: 90.2% aggregate — correct. Intervals are 13.4% narrower than parametric. The top-decile coverage of 87.9% is a 2.1pp miss, consistent with the marginal guarantee (it holds on average, not per-decile). If per-decile coverage matters, use LW conformal.
- LW conformal: the secondary spread model learns which features predict large residuals. The result: 90.6% in the top decile (slightly above target), 11.7% narrower than parametric, 2.0% wider than standard conformal. If you have the training data available, LW conformal dominates on the metrics that matter for reinsurance attachment decisions.
- The conformal coverage guarantee is marginal, not conditional. Always check `coverage_by_decile()` after calibration.

---

### Reference scenario: Ridge regression baseline (null result)

The original benchmark (2026-03-16) uses Ridge regression on log(y) as the baseline model. With a well-matched log-normal DGP, both parametric and conformal intervals achieve near-uniform coverage across deciles. Conformal wins on interval width (-13% vs raw) but the coverage argument is less compelling. This is the scenario where conformal is not needed — but it still helps with width.

Run: `benchmarks/benchmark.py`

| Metric | Naive parametric (Ridge) | Conformal (pearson_weighted) |
|--------|--------------------------|------------------------------|
| Aggregate coverage @ 90% | 0.917 | 0.901 |
| Worst-decile coverage | 0.917 | 0.714 |
| Mean interval width (£) | 6,445 | 4,675 |
| Distribution-free guarantee | No | Yes (marginal) |

Note: conformal undercovers the top decile at 71.4% here — a known limitation of the pearson_weighted score with a poor point forecast. The score divides by yhat^0.75, compressing scores for high-predicted-value policies and producing intervals that are too narrow for them. This failure mode is exactly why you should use `coverage_by_decile()` in practice, and why the GBM benchmark above uses a well-calibrated CatBoost forecast.

**Practical guidance:** conformal prediction is most valuable when (a) your point forecast is well-calibrated (GBM, not Ridge), and (b) the residual distribution is genuinely more complex than a single parametric family can describe — which is the common case for heterogeneous UK motor books. The LW conformal variant is the recommendation for production use.

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

## Training Course

Want structured learning? [Insurance Pricing in Python](https://burning-cost.github.io/course) is a 12-module course covering the full pricing workflow. Module 11 covers conformal prediction — split conformal, CQR, and coverage guarantees for pricing models. £97 one-time.

## Licence

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests welcome at [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal).

---

Need help implementing this? [See our consulting services](https://burning-cost.github.io/work-with-us/).
