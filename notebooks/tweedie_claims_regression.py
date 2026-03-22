# Databricks notebook source
# MAGIC %md
# MAGIC # Tweedie Claims Regression with Conformal Prediction Intervals
# MAGIC
# MAGIC **The problem:** Your LightGBM Tweedie model produces point estimates. You need
# MAGIC prediction intervals for actuarial review, reinsurance treaty negotiation, and
# MAGIC Solvency II SCR calculations. The standard approach — fit a global sigma on
# MAGIC calibration residuals and add/subtract — breaks down on heterogeneous books
# MAGIC because variance is not constant across risk segments.
# MAGIC
# MAGIC **The solution:** Conformal prediction with the Pearson-weighted score:
# MAGIC `|y - yhat| / yhat^(p/2)`. This normalises residuals by the Tweedie standard
# MAGIC deviation, producing roughly exchangeable scores across the risk spectrum.
# MAGIC The finite-sample coverage guarantee holds regardless of model specification.
# MAGIC
# MAGIC ## What this notebook demonstrates
# MAGIC
# MAGIC 1. Synthetic UK motor portfolio with a heteroscedastic compound Poisson-Gamma DGP
# MAGIC    (variance grows faster in the high-risk tail than Tweedie(1.5) predicts)
# MAGIC 2. LightGBM Tweedie(p=1.5) pure premium model
# MAGIC 3. Conformal calibration with `InsuranceConformalPredictor`
# MAGIC 4. Coverage by risk segment: age band, vehicle group, area — the diagnostics
# MAGIC    that matter to pricing actuaries, not just decile-of-predicted-value
# MAGIC 5. Width efficiency comparison across all non-conformity scores
# MAGIC 6. SCR regulatory use case: finite-sample valid 99.5% upper bounds
# MAGIC
# MAGIC **References:**
# MAGIC - Manna et al. (2025) ASMBI asmb.70045 — Pearson score for Tweedie models
# MAGIC - Hong (2025) arXiv:2503.03659 — model-free conformal for insurance claims
# MAGIC - Hong (2026) arXiv:2601.21153 — h-transform residuals, regression setting

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install "insurance-conformal[lightgbm]" polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import polars as pl
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

from insurance_conformal import (
    InsuranceConformalPredictor,
    subgroup_coverage,
    width_efficiency_comparison,
)

print(f"insurance-conformal loaded")
print(f"LightGBM {lgb.__version__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic UK motor portfolio
# MAGIC
# MAGIC We generate 40,000 policies with a compound Poisson-Gamma DGP.
# MAGIC The key ingredient: variance grows faster than Tweedie(1.5) predicts for
# MAGIC high-risk policies (high vehicle group + urban area). This is the scenario
# MAGIC where parametric intervals break down on real UK motor books.

# COMMAND ----------

RNG = np.random.default_rng(42)
N = 40_000
ALPHA = 0.10     # 90% prediction intervals
TWEEDIE_P = 1.5

# Features
driver_age = RNG.integers(18, 80, N).astype(float)
ncd_years = RNG.integers(0, 9, N).astype(float)
vehicle_group = RNG.integers(1, 21, N).astype(float)   # 1-20, industry standard
area = RNG.integers(0, 6, N).astype(float)             # 0-5, urban/rural
annual_mileage = RNG.uniform(3_000, 25_000, N)
conviction_pts = RNG.integers(0, 5, N).astype(float)
exposure = RNG.uniform(0.25, 1.0, N)                   # policy years

# True mean (log-linear multiplicative)
log_mu = (
    5.5
    + 0.006 * np.maximum(0, 25 - driver_age)      # young driver loading
    + 0.003 * np.maximum(0, driver_age - 60)       # older driver loading
    - 0.10 * ncd_years                             # NCD discount
    + 0.06 * vehicle_group                         # vehicle group loading
    + 0.10 * area                                  # area loading
    + 0.22 * conviction_pts                        # conviction loading
    + 0.000012 * annual_mileage
)
mu_base = np.exp(log_mu)
mu = mu_base * exposure   # expected cost per policy year

# Heteroscedastic dispersion: high-risk segments have extra variance
phi_base = 2.5
phi = phi_base * (1.0 + 0.6 * (vehicle_group / 20) + 0.4 * (area / 5))

# Compound Poisson-Gamma simulation
lambda_pois = mu ** (2 - TWEEDIE_P) / (phi * (2 - TWEEDIE_P))
n_claims = RNG.poisson(lambda_pois)
alpha_gamma = (2 - TWEEDIE_P) / (TWEEDIE_P - 1)
y = np.zeros(N)
for i in range(N):
    if n_claims[i] > 0:
        beta_g = phi * (TWEEDIE_P - 1) * mu[i] ** (TWEEDIE_P - 1)
        y[i] = RNG.gamma(shape=alpha_gamma * n_claims[i], scale=1.0 / beta_g)

print(f"Portfolio: {N:,} policies")
print(f"Zero claims: {(y == 0).mean():.1%}")
print(f"Mean pure premium: £{y.mean():.2f}")
print(f"95th percentile: £{np.percentile(y, 95):.2f}")
print(f"Max: £{y.max():.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Train/calibration/test split (temporal)
# MAGIC
# MAGIC Always split temporally in insurance. Random splits violate the exchangeability
# MAGIC assumption that underpins conformal coverage guarantees. Train on earlier years,
# MAGIC calibrate on the most recent completed year, report on the current year.

# COMMAND ----------

FEATURE_NAMES = [
    "driver_age", "ncd_years", "vehicle_group", "area",
    "annual_mileage", "conviction_pts", "exposure",
]
X = np.column_stack([
    driver_age, ncd_years, vehicle_group, area,
    annual_mileage, conviction_pts, exposure,
])

n_train = int(0.60 * N)
n_cal = int(0.20 * N)
n_test = N - n_train - n_cal

X_train, y_train = X[:n_train], y[:n_train]
X_cal, y_cal = X[n_train:n_train + n_cal], y[n_train:n_train + n_cal]
X_test, y_test = X[n_train + n_cal:], y[n_train + n_cal:]

# Segment labels for test set diagnostics
driver_age_test = driver_age[n_train + n_cal:]
vehicle_group_test = vehicle_group[n_train + n_cal:]
area_test = area[n_train + n_cal:]

print(f"Train: {n_train:,}  |  Calibration: {n_cal:,}  |  Test: {n_test:,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. LightGBM Tweedie model
# MAGIC
# MAGIC `tweedie_variance_power=1.5` is the canonical pure premium model for UK motor.
# MAGIC We wrap the booster in a minimal sklearn-compatible class so it works with
# MAGIC `InsuranceConformalPredictor` — the predictor just needs a `.predict(X)` method.

# COMMAND ----------

class LGBWrapper:
    """Thin sklearn-compatible wrapper around a trained LightGBM Booster."""

    def __init__(self, num_boost_round: int = 400) -> None:
        self.num_boost_round = num_boost_round
        self._booster = None

    def fit(self, X, y):
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": TWEEDIE_P,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 30,
            "verbose": -1,
            "seed": 42,
        }
        ds = lgb.Dataset(X, label=y)
        self._booster = lgb.train(params, ds, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X):
        return self._booster.predict(X)


lgb_model = LGBWrapper(num_boost_round=400)
lgb_model.fit(X_train, y_train)

yhat_test = lgb_model.predict(X_test)
print(f"Test predictions: mean=£{yhat_test.mean():.2f}  max=£{yhat_test.max():.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Baseline: Naive percentile intervals
# MAGIC
# MAGIC The simplest approach: compute the 5th and 95th percentile of training
# MAGIC residuals and shift them by the point prediction. This is additive — the
# MAGIC same pound margin applies to a £50 risk and a £5,000 risk. It over-covers
# MAGIC low-risk policies (interval too wide) and under-covers high-risk ones.

# COMMAND ----------

yhat_cal = lgb_model.predict(X_cal)
resid_cal = y_cal - yhat_cal

lo_resid = np.percentile(resid_cal, ALPHA / 2 * 100)
hi_resid = np.percentile(resid_cal, (1 - ALPHA / 2) * 100)

naive_lower = np.maximum(0.0, yhat_test + lo_resid)
naive_upper = yhat_test + hi_resid

naive_covered = (y_test >= naive_lower) & (y_test <= naive_upper)
naive_coverage = float(naive_covered.mean())
naive_width = float((naive_upper - naive_lower).mean())

print(f"Calibration residuals: 5th=£{lo_resid:.2f}  95th=£{hi_resid:.2f}")
print(f"Aggregate coverage: {naive_coverage:.3f}  (target: {1-ALPHA:.2f})")
print(f"Mean interval width: £{naive_width:,.2f}")
print()

# Show coverage by vehicle group to expose the failure mode
def veh_band(vg):
    bands = np.empty(len(vg), dtype=object)
    bands[vg <= 7] = "01-07 (low-cost)"
    bands[(vg >= 8) & (vg <= 14)] = "08-14 (mid)"
    bands[vg >= 15] = "15-20 (high-cost)"
    return bands

veh_bands_test = veh_band(vehicle_group_test)

print("Naive coverage by vehicle group:")
for band in ["01-07 (low-cost)", "08-14 (mid)", "15-20 (high-cost)"]:
    mask = veh_bands_test == band
    cov = float(naive_covered[mask].mean())
    n = int(mask.sum())
    print(f"  {band:>20}  n={n:,}  coverage={cov:.3f}  {'<<<PROBLEM' if cov < 0.85 else ''}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Conformal prediction: Pearson-weighted score
# MAGIC
# MAGIC `InsuranceConformalPredictor` wraps the fitted model. The `pearson_weighted`
# MAGIC score divides each residual by `yhat^(p/2)` — the standard deviation of a
# MAGIC Tweedie(p) distribution with mean `yhat` (up to dispersion). This makes
# MAGIC residuals from high-risk and low-risk policies directly comparable,
# MAGIC producing coverage that is uniform across the risk spectrum.

# COMMAND ----------

cp = InsuranceConformalPredictor(
    model=lgb_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=TWEEDIE_P,
)
cp.calibrate(X_cal, y_cal)

intervals = cp.predict_interval(X_test, alpha=ALPHA)

pw_lower = intervals["lower"].to_numpy()
pw_upper = intervals["upper"].to_numpy()
pw_point = intervals["point"].to_numpy()

pw_covered = (y_test >= pw_lower) & (y_test <= pw_upper)
pw_coverage = float(pw_covered.mean())
pw_width = float((pw_upper - pw_lower).mean())

print(f"Conformal calibration quantile (alpha={ALPHA}): {cp._get_quantile(ALPHA):.4f}")
print(f"Aggregate coverage: {pw_coverage:.3f}  (target: {1-ALPHA:.2f})")
print(f"Mean interval width: £{pw_width:,.2f}")
print()
display(intervals.head(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Coverage by risk segment
# MAGIC
# MAGIC `coverage_by_decile()` bins by predicted value. For communicating to
# MAGIC underwriters and actuaries, business-relevant segments matter more:
# MAGIC age band, vehicle group, area. Use `subgroup_coverage()` for that.

# COMMAND ----------

# Coverage by decile of predicted pure premium
decile_table = cp.coverage_by_decile(X_test, y_test, alpha=ALPHA)
print("Coverage by decile of predicted pure premium:")
display(decile_table)

# COMMAND ----------

# Driver age band
def age_band(age):
    bands = np.empty(len(age), dtype=object)
    bands[(age >= 18) & (age < 25)] = "17-24 (young)"
    bands[(age >= 25) & (age < 35)] = "25-34"
    bands[(age >= 35) & (age < 50)] = "35-49"
    bands[(age >= 50) & (age < 70)] = "50-69"
    bands[age >= 70]                = "70+ (older)"
    return bands

age_bands_test = age_band(driver_age_test)

sg_age = subgroup_coverage(
    predictor=cp,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
    groups=age_bands_test,
    group_name="age_band",
)
print("Coverage by driver age band:")
display(sg_age.select(["age_band", "n_obs", "empirical_coverage", "coverage_gap", "mean_width"]))

# COMMAND ----------

# Vehicle group band
sg_veh = subgroup_coverage(
    predictor=cp,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
    groups=veh_bands_test,
    group_name="vehicle_group_band",
)
print("Coverage by vehicle group band:")
display(sg_veh.select(["vehicle_group_band", "n_obs", "empirical_coverage", "coverage_gap", "mean_width"]))

# COMMAND ----------

# Area type (urban vs rural)
area_bands_test = np.where(area_test <= 2, "rural/suburban (0-2)", "urban (3-5)")
sg_area = subgroup_coverage(
    predictor=cp,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
    groups=area_bands_test,
    group_name="area_type",
)
print("Coverage by area type:")
display(sg_area.select(["area_type", "n_obs", "empirical_coverage", "coverage_gap", "mean_width"]))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Width efficiency: all non-conformity scores vs naive
# MAGIC
# MAGIC All five scores achieve valid marginal coverage. The practical question is
# MAGIC which produces the narrowest intervals — tighter SCR estimates, less wasteful
# MAGIC reinsurance attachment. `pearson_weighted` is typically best for Tweedie models
# MAGIC because it accounts for the Tweedie variance structure directly.
# MAGIC
# MAGIC Note: `deviance` and `anscombe` are correct but slower to invert for general p.
# MAGIC `pearson_weighted` gives a good balance of correctness, speed, and interpretability.

# COMMAND ----------

score_predictors = {}
for score_name in ["raw", "pearson", "pearson_weighted", "deviance", "anscombe"]:
    cp_s = InsuranceConformalPredictor(
        model=lgb_model,
        nonconformity=score_name,
        distribution="tweedie",
        tweedie_power=TWEEDIE_P,
    )
    cp_s.calibrate(X_cal, y_cal)
    score_predictors[score_name] = cp_s

comparison = width_efficiency_comparison(
    predictors=score_predictors,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
)
print("Conformal scores (sorted by mean interval width):")
display(comparison)

print(f"\nNaive percentile baseline:")
print(f"  coverage={naive_coverage:.3f}  mean_width=£{naive_width:,.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. SCR use case: 99.5% conformal upper bounds
# MAGIC
# MAGIC Solvency II Article 101 / UK Solvency UK (PRA PS9/24) requires the SCR to
# MAGIC cover a 1-in-200 year loss — equivalently, a 99.5% upper bound on claims.
# MAGIC Conformal prediction provides a finite-sample valid bound with no distributional
# MAGIC assumption. Set alpha=0.005 to get a 99.5% prediction interval.

# COMMAND ----------

cp_scr = InsuranceConformalPredictor(
    model=lgb_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=TWEEDIE_P,
)
cp_scr.calibrate(X_cal, y_cal)

intervals_scr = cp_scr.predict_interval(X_test, alpha=0.005)
scr_upper = intervals_scr["upper"].to_numpy()
scr_covered = float((y_test <= scr_upper).mean())

print(f"99.5% conformal upper bounds:")
print(f"  Empirical coverage: {scr_covered:.4f}  (target: 0.995)")
print(f"  Mean upper bound: £{scr_upper.mean():,.2f}")
print(f"  Mean point estimate: £{yhat_test.mean():,.2f}")
print(f"  SCR loading factor: {scr_upper.mean() / yhat_test.mean():.3f}x")
print(f"  Aggregate SCR (sum of upper bounds): £{scr_upper.sum():,.0f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Two-stage locally weighted conformal (adaptive-width)
# MAGIC
# MAGIC `TwoStageLWConformal` fits a secondary spread model on training Pearson
# MAGIC residuals, producing intervals that are narrower in low-variance segments
# MAGIC and wider in high-variance ones. This is the Manna et al. (2025) two-stage
# MAGIC locally weighted method (Section 3.2, Theorem 1).
# MAGIC
# MAGIC The `predict_interval()` method returns a numpy array of shape (n, 2)
# MAGIC rather than a polars DataFrame — note the different API from
# MAGIC `InsuranceConformalPredictor`.

# COMMAND ----------

from insurance_conformal.claims import TwoStageLWConformal

# Use a simple spread model (Ridge) to avoid CatBoost dependency here
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

spread_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])

lw = TwoStageLWConformal(
    mean_model=lgb_model,
    p=TWEEDIE_P,
    spread_model=spread_model,
)
lw.fit(X_train, y_train)
lw.calibrate(X_cal, y_cal)

lw_intervals = lw.predict_interval(X_test, alpha=ALPHA)  # shape (n, 2)
lw_covered = (y_test >= lw_intervals[:, 0]) & (y_test <= lw_intervals[:, 1])
lw_coverage = float(lw_covered.mean())
lw_width = float((lw_intervals[:, 1] - lw_intervals[:, 0]).mean())

print(f"TwoStageLWConformal (spread=Ridge):")
print(f"  Coverage: {lw_coverage:.3f}  (target: {1-ALPHA:.2f})")
print(f"  Mean width: £{lw_width:,.2f}  vs pearson_weighted: £{pw_width:,.2f}")
print(f"  Width change vs pearson_weighted: {(lw_width - pw_width) / pw_width * 100:+.1f}%")
print()
print(lw)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Method | Coverage | Mean width (£) | Distribution-free? |
# MAGIC |--------|----------|----------------|--------------------|
# MAGIC | Naive percentile (baseline) | variable | widest | No |
# MAGIC | Conformal raw | >=90% | wide | Yes |
# MAGIC | Conformal pearson | >=90% | narrower | Yes |
# MAGIC | Conformal pearson_weighted | >=90% | narrowest | Yes |
# MAGIC | Conformal deviance | >=90% | similar to pearson_weighted | Yes |
# MAGIC | Conformal anscombe | >=90% | similar to pearson_weighted | Yes |
# MAGIC | TwoStageLW (adaptive) | >=90% | adaptive by segment | Yes |
# MAGIC
# MAGIC Key takeaways:
# MAGIC
# MAGIC 1. The naive percentile approach achieves acceptable aggregate coverage but
# MAGIC    under-covers high-risk segments (high vehicle group, urban area) where
# MAGIC    the DGP variance exceeds what the global margin assumes.
# MAGIC
# MAGIC 2. Conformal `pearson_weighted` achieves valid coverage uniformly across
# MAGIC    segments, including the high-risk tail. The coverage gap by vehicle group
# MAGIC    and age band is substantially smaller.
# MAGIC
# MAGIC 3. All conformal scores give identical marginal coverage guarantees.
# MAGIC    `pearson_weighted` typically produces the narrowest intervals for Tweedie
# MAGIC    models — tighter SCR estimates, better reinsurance attachment.
# MAGIC
# MAGIC 4. `TwoStageLWConformal` provides adaptive-width intervals: narrower for
# MAGIC    low-variance policies, wider for genuinely uncertain ones. Best for large
# MAGIC    books where per-segment width efficiency matters.
# MAGIC
# MAGIC 5. Setting alpha=0.005 gives Solvency II-valid 99.5% upper bounds. The
# MAGIC    finite-sample guarantee means coverage is at least 99.5% by construction,
# MAGIC    not just in expectation under parametric assumptions.
