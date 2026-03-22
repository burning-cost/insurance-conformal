"""
Tweedie claims regression with conformal prediction intervals.

This example is the practical answer to a question that comes up in every motor
pricing review: "You have a Tweedie GBM. What's the uncertainty around that
pure premium estimate, and does it hold for high-risk segments?"

Parametric Tweedie intervals use a single dispersion parameter estimated from
the calibration set. That breaks down when the true residual variance grows
faster than mu^p — which is the common case on heterogeneous UK motor books,
especially for older vehicles and high-mileage drivers.

Conformal prediction with the Pearson-weighted score fixes this by making no
distributional assumption. The only requirement: calibration and test data are
exchangeable. For motor, the natural approach is temporal: train on years 1-4,
calibrate on year 5, report on year 6.

This example demonstrates:
    1. Generating a realistic synthetic UK motor pure premium dataset
       (compound Poisson-Gamma DGP with heteroscedastic variance — the
       variance grows faster in the high-mean tail than Tweedie(1.5) predicts)
    2. Fitting a LightGBM model with Tweedie objective (p=1.5)
    3. Calibrating conformal intervals with InsuranceConformalPredictor
    4. Coverage by risk segment: age band, vehicle group, area (the
       diagnostics that matter to pricing actuaries and reinsurers)
    5. Comparing three approaches: naive percentile intervals, conformal raw,
       conformal pearson_weighted — with explicit interval width and coverage
    6. Using width_efficiency_comparison to summarise all scores in one table
    7. Pointing to TwoStageLWConformal for adaptive-width intervals

Run this example on Databricks to avoid crashing the local Pi:
    python examples/tweedie_claims_regression.py

Or run from the Databricks notebook at:
    notebooks/tweedie_claims_regression.py

References
----------
Manna et al. (2025) ASMBI asmb.70045 — Pearson score design for Tweedie models
Hong (2025) arXiv:2503.03659 — model-free conformal for insurance claims
Hong (2026) arXiv:2601.21153 — h-transform residuals, regression setting
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

t0 = time.time()

print("=" * 72)
print("Tweedie claims regression with conformal prediction intervals")
print("UK motor pure premium — heteroskedastic compound Poisson-Gamma DGP")
print("=" * 72)
print()

# ---------------------------------------------------------------------------
# Imports: light requirements — sklearn + lightgbm + insurance-conformal
# ---------------------------------------------------------------------------

try:
    from insurance_conformal import (
        InsuranceConformalPredictor,
        subgroup_coverage,
        width_efficiency_comparison,
    )
    print("insurance-conformal imported")
except ImportError as exc:
    print(f"ERROR: {exc}")
    print("Install: pip install insurance-conformal")
    sys.exit(1)

try:
    import lightgbm as lgb
    print(f"LightGBM {lgb.__version__}")
except ImportError:
    print("ERROR: lightgbm required for this example.")
    print("Install: pip install lightgbm  or  pip install 'insurance-conformal[lightgbm]'")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# 1. Synthetic UK motor pure premium data
#
# DGP: compound Poisson-Gamma, as in UK motor frequency-severity modelling.
# The variance deliberately grows faster than Tweedie(1.5) predicts for
# high-risk policies (high vehicle group, older driver, urban area). This
# is the failure mode that breaks parametric Tweedie intervals in practice.
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 40_000
ALPHA = 0.10   # 90% prediction intervals
TWEEDIE_P = 1.5

print("Generating synthetic portfolio...")

# --- Features -----------------------------------------------------------
driver_age = RNG.integers(18, 80, N).astype(float)
ncd_years = RNG.integers(0, 9, N).astype(float)
vehicle_group = RNG.integers(1, 21, N).astype(float)    # 1-20 (industry standard)
area = RNG.integers(0, 6, N).astype(float)              # 0-5 (urban/rural)
annual_mileage = RNG.uniform(3_000, 25_000, N)
conviction_pts = RNG.integers(0, 5, N).astype(float)
exposure = RNG.uniform(0.25, 1.0, N)                    # policy years

# --- True mean (log-linear multiplicative) --------------------------------
log_mu = (
    5.5                                             # base: ~£245/yr
    + 0.006 * np.maximum(0, 25 - driver_age)       # young driver loading
    + 0.003 * np.maximum(0, driver_age - 60)       # older driver loading
    - 0.10 * ncd_years                             # NCD discount
    + 0.06 * vehicle_group                         # group loading
    + 0.10 * area                                  # area loading
    + 0.22 * conviction_pts                        # conviction loading
    + 0.000012 * annual_mileage                    # mileage
)
mu_base = np.exp(log_mu)
mu = mu_base * exposure                            # expected claim cost per policy year

# --- Heteroscedastic compound Poisson-Gamma DGP ---------------------------
# The key: high-risk segments (high vehicle_group, urban area) have extra
# variance beyond what Tweedie(1.5) predicts. phi grows with risk level.
# This is what kills parametric Tweedie intervals in the top decile.
phi_base = 2.5
phi = phi_base * (1.0 + 0.6 * (vehicle_group / 20) + 0.4 * (area / 5))

lambda_pois = mu ** (2 - TWEEDIE_P) / (phi * (2 - TWEEDIE_P))
n_claims = RNG.poisson(lambda_pois)

alpha_gamma = (2 - TWEEDIE_P) / (TWEEDIE_P - 1)
y = np.zeros(N)
for i in range(N):
    if n_claims[i] > 0:
        # Each claim is Gamma-distributed; total is sum of n_claims[i] Gammas
        beta_g = phi * (TWEEDIE_P - 1) * mu[i] ** (TWEEDIE_P - 1)
        y[i] = RNG.gamma(
            shape=alpha_gamma * n_claims[i],
            scale=1.0 / beta_g,
        )

zero_frac = (y == 0).mean()
print(f"  Policies: {N:,}")
print(f"  Zero claims: {zero_frac:.1%}  (realistic for UK motor)")
print(f"  Mean pure premium: £{y.mean():.2f}")
print(f"  Std dev: £{y.std():.2f}")
print(f"  95th pct: £{np.percentile(y, 95):.2f}")
print(f"  Max: £{y.max():.2f}")
print()

# ---------------------------------------------------------------------------
# 2. Feature matrix and temporal split
#
# Always split temporally in insurance. A random split violates exchangeability
# (risks written in Q1 vs Q4 are not exchangeable due to seasonality, rate
# changes, etc.). Index-based split is the minimal approximation here.
# ---------------------------------------------------------------------------

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

# Segment labels for later diagnostics
driver_age_test = driver_age[n_train + n_cal:]
vehicle_group_test = vehicle_group[n_train + n_cal:]
area_test = area[n_train + n_cal:]

print(f"Split: {n_train:,} train / {n_cal:,} calibration / {n_test:,} test")
print()

# ---------------------------------------------------------------------------
# 3. Fit LightGBM Tweedie model
#
# LightGBM with tweedie_variance_power=1.5 is the canonical UK motor pure
# premium model. The log-link is implicit; predictions are on the response
# scale (£ per exposure-year). No need to transform y.
# ---------------------------------------------------------------------------

print("Fitting LightGBM Tweedie model (p=1.5)...")

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_cal, label=y_cal, reference=lgb_train)

lgb_params = {
    "objective": "tweedie",
    "tweedie_variance_power": TWEEDIE_P,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "n_estimators": 400,
    "verbose": -1,
    "seed": 42,
}

# Wrap in sklearn API so InsuranceConformalPredictor can call .predict()
class LGBWrapper:
    """Thin sklearn-compatible wrapper around a trained LightGBM Booster."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self._booster = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBWrapper":
        ds = lgb.Dataset(X, label=y)
        self._booster = lgb.train(
            {k: v for k, v in self.params.items() if k != "n_estimators"},
            ds,
            num_boost_round=self.params.get("n_estimators", 400),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(X)


lgb_model = LGBWrapper(lgb_params)
lgb_model.fit(X_train, y_train)
yhat_test_lgb = lgb_model.predict(X_test)

print(f"  Test predictions: mean=£{yhat_test_lgb.mean():.2f}  "
      f"range=[£{yhat_test_lgb.min():.2f}, £{yhat_test_lgb.max():.2f}]")
print()

# ---------------------------------------------------------------------------
# 4. BASELINE: Naive percentile intervals
#
# The simplest possible approach: compute the empirical (5th, 95th) percentile
# of training residuals and add/subtract from point predictions. This is what
# many teams use before they think carefully about heteroscedasticity.
#
# Problem: applies the same additive margin to a £50 risk and a £5,000 risk.
# Covers the low-risk segment excessively (interval too wide) and under-covers
# the high-risk segment (not wide enough relative to actual variance).
# ---------------------------------------------------------------------------

print("-" * 72)
print("BASELINE: Naive percentile intervals (global training residuals)")
print("-" * 72)
print()

yhat_cal_lgb = lgb_model.predict(X_cal)
resid_cal = y_cal - yhat_cal_lgb

# Global 5th/95th percentile of residuals (symmetric around zero by construction)
lo_resid = np.percentile(resid_cal, ALPHA / 2 * 100)      # ~5th pct
hi_resid = np.percentile(resid_cal, (1 - ALPHA / 2) * 100) # ~95th pct

naive_lower = np.maximum(0.0, yhat_test_lgb + lo_resid)
naive_upper = yhat_test_lgb + hi_resid

naive_covered = (y_test >= naive_lower) & (y_test <= naive_upper)
naive_coverage = float(naive_covered.mean())
naive_width = float((naive_upper - naive_lower).mean())

print(f"  Calibration residuals: 5th={lo_resid:.2f}  95th={hi_resid:.2f}")
print(f"  Aggregate coverage: {naive_coverage:.3f}  (target: {1-ALPHA:.2f})")
print(f"  Mean interval width: £{naive_width:,.2f}")
print()

# Coverage by age band — the naive method's failure mode is visible here
def age_band(age: np.ndarray) -> np.ndarray:
    """Bucket driver ages into 5 standard UK motor rating bands."""
    bands = np.empty(len(age), dtype=object)
    bands[(age >= 18) & (age < 25)] = "17-24 (young)"
    bands[(age >= 25) & (age < 35)] = "25-34"
    bands[(age >= 35) & (age < 50)] = "35-49"
    bands[(age >= 50) & (age < 70)] = "50-69"
    bands[age >= 70]                = "70+ (older)"
    return bands

age_bands_test = age_band(driver_age_test)

print("  Naive coverage by driver age band:")
print(f"  {'Age band':>15}  {'n':>6}  {'coverage':>10}")
for band in ["17-24 (young)", "25-34", "35-49", "50-69", "70+ (older)"]:
    mask = age_bands_test == band
    if not mask.any():
        continue
    cov = float(naive_covered[mask].mean())
    n = int(mask.sum())
    flag = " <<< under" if cov < 1 - ALPHA - 0.05 else ""
    print(f"  {band:>15}  {n:>6,}  {cov:>10.3f}{flag}")
print()

# Coverage by vehicle group band (groups 1-7, 8-14, 15-20)
def veh_band(vg: np.ndarray) -> np.ndarray:
    bands = np.empty(len(vg), dtype=object)
    bands[vg <= 7]                     = "01-07 (low-cost)"
    bands[(vg >= 8) & (vg <= 14)]     = "08-14 (mid)"
    bands[vg >= 15]                    = "15-20 (high-cost)"
    return bands

veh_bands_test = veh_band(vehicle_group_test)

print("  Naive coverage by vehicle group band:")
print(f"  {'Vehicle group':>20}  {'n':>6}  {'coverage':>10}")
for band in ["01-07 (low-cost)", "08-14 (mid)", "15-20 (high-cost)"]:
    mask = veh_bands_test == band
    if not mask.any():
        continue
    cov = float(naive_covered[mask].mean())
    n = int(mask.sum())
    flag = " <<< under" if cov < 1 - ALPHA - 0.05 else ""
    print(f"  {band:>20}  {n:>6,}  {cov:>10.3f}{flag}")
print()

# ---------------------------------------------------------------------------
# 5. Conformal prediction: pearson_weighted score
#
# InsuranceConformalPredictor wraps the fitted LightGBM model. The score
# |y - yhat| / yhat^(p/2) normalises residuals by the Tweedie standard
# deviation, producing roughly exchangeable scores across risk levels.
#
# The conformal quantile is: Q_{ceil((1-alpha)(n_cal+1))/n_cal}(scores_cal)
# Finite-sample guarantee: P(y_test in interval) >= 1 - alpha exactly.
# ---------------------------------------------------------------------------

print("-" * 72)
print("CONFORMAL: InsuranceConformalPredictor (pearson_weighted score)")
print("-" * 72)
print()

cp = InsuranceConformalPredictor(
    model=lgb_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=TWEEDIE_P,
)
cp.calibrate(X_cal, y_cal)

intervals_pw = cp.predict_interval(X_test, alpha=ALPHA)
pw_lower = intervals_pw["lower"].to_numpy()
pw_upper = intervals_pw["upper"].to_numpy()
pw_point = intervals_pw["point"].to_numpy()

pw_covered = (y_test >= pw_lower) & (y_test <= pw_upper)
pw_coverage = float(pw_covered.mean())
pw_width = float((pw_upper - pw_lower).mean())

print(f"  Calibration quantile (alpha={ALPHA}): {cp._get_quantile(ALPHA):.4f}")
print(f"  Aggregate coverage: {pw_coverage:.3f}  (target: {1-ALPHA:.2f})")
print(f"  Mean interval width: £{pw_width:,.2f}")
print()

# Coverage by decile using the built-in method
decile_table = cp.coverage_by_decile(X_test, y_test, alpha=ALPHA)
print("  Coverage by decile of predicted pure premium:")
print(f"  {'Decile':>7}  {'Mean pred (£)':>13}  {'n_obs':>6}  {'coverage':>10}")
for row in decile_table.iter_rows(named=True):
    flag = " *" if abs(row["coverage"] - (1 - ALPHA)) > 0.05 else ""
    print(f"  {row['decile']:>7}  {row['mean_predicted']:>13,.2f}  {row['n_obs']:>6}  {row['coverage']:>10.3f}{flag}")
print()

# Coverage by business segment using subgroup_coverage
print("  Coverage by driver age band:")
sg_age = subgroup_coverage(
    predictor=cp,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
    groups=age_bands_test,
    group_name="age_band",
)
print(sg_age.select(["age_band", "n_obs", "empirical_coverage", "coverage_gap", "mean_width"]))
print()

print("  Coverage by vehicle group band:")
sg_veh = subgroup_coverage(
    predictor=cp,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
    groups=veh_bands_test,
    group_name="vehicle_group_band",
)
print(sg_veh.select(["vehicle_group_band", "n_obs", "empirical_coverage", "coverage_gap", "mean_width"]))
print()

# Area coverage (urban areas have higher dispersion in the DGP)
area_bands_test = np.where(area_test <= 2, "rural/suburban (0-2)", "urban (3-5)")
print("  Coverage by area type:")
sg_area = subgroup_coverage(
    predictor=cp,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
    groups=area_bands_test,
    group_name="area_type",
)
print(sg_area.select(["area_type", "n_obs", "empirical_coverage", "coverage_gap", "mean_width"]))
print()

# ---------------------------------------------------------------------------
# 6. Width efficiency comparison across all non-conformity scores
#
# All five scores deliver valid marginal coverage. The question is which
# produces the narrowest intervals — tighter SCR estimates, less wasteful
# reinsurance attachment. pearson_weighted is typically best for Tweedie.
# ---------------------------------------------------------------------------

print("-" * 72)
print("WIDTH EFFICIENCY: All non-conformity scores vs naive percentile")
print("-" * 72)
print()

# Build all conformal predictors
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

# width_efficiency_comparison expects predictors with predict_interval(X, alpha)
# returning a polars DataFrame with "lower" and "upper" columns
comparison = width_efficiency_comparison(
    predictors=score_predictors,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
)
print("  Conformal scores (sorted by mean interval width):")
print(comparison)
print()

# Add naive baseline row for context
naive_mean_width = float((naive_upper - naive_lower).mean())
naive_median_width = float(np.median(naive_upper - naive_lower))
print(f"  Naive percentile baseline:")
print(f"    coverage={naive_coverage:.3f}  mean_width=£{naive_mean_width:,.2f}  "
      f"median_width=£{naive_median_width:,.2f}")
print()

# Width reduction vs naive
pw_mean_width = float(comparison.filter(pl.col("predictor_name") == "pearson_weighted")["mean_width"][0])
width_reduction_pct = (naive_mean_width - pw_mean_width) / naive_mean_width * 100
print(f"  pearson_weighted vs naive: {width_reduction_pct:+.1f}% change in mean interval width")
print()

# ---------------------------------------------------------------------------
# 7. SCR alpha comparison (regulatory use case)
#
# Solvency II Article 101 requires the SCR to cover a 1-in-200 year loss.
# In practice: fit a 99.5% upper confidence bound on future claims costs.
# Conformal gives a finite-sample valid bound with no distributional assumption.
# ---------------------------------------------------------------------------

print("-" * 72)
print("REGULATORY: Conformal prediction at SCR level (alpha=0.005, 99.5%)")
print("-" * 72)
print()

cp_scr = InsuranceConformalPredictor(
    model=lgb_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=TWEEDIE_P,
)
cp_scr.calibrate(X_cal, y_cal)

intervals_scr = cp_scr.predict_interval(X_test, alpha=0.005)
scr_upper = intervals_scr["upper"].to_numpy()
scr_covered = (y_test <= scr_upper).mean()

print(f"  99.5% upper bound: mean=£{scr_upper.mean():,.2f}")
print(f"  Empirical coverage at 99.5%: {scr_covered:.4f} (target: 0.995)")
print(f"  Aggregate SCR estimate (sum of upper bounds): £{scr_upper.sum():,.0f}")
print(f"  Point estimate (sum): £{yhat_test_lgb.sum():,.0f}")
print(f"  SCR loading factor: {scr_upper.mean() / yhat_test_lgb.mean():.3f}x point estimate")
print()

# ---------------------------------------------------------------------------
# 8. Summary table
# ---------------------------------------------------------------------------

print("=" * 72)
print("SUMMARY")
print("=" * 72)
print()

print(f"  {'Method':<35} {'Coverage':>10} {'Width (£)':>12} {'Valid?':>8}")
print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*8}")
print(f"  {'Naive percentile':35} {naive_coverage:>10.3f} {naive_mean_width:>12,.2f} {'NO':>8}")

for row in comparison.iter_rows(named=True):
    name = f"Conformal {row['predictor_name']}"
    print(f"  {name:<35} {row['marginal_coverage']:>10.3f} {row['mean_width']:>12,.2f} {'YES':>8}")

print()
print("  Notes:")
print(f"    - Target coverage: {1-ALPHA:.0%}  (alpha={ALPHA})")
print(f"    - n_cal={n_cal:,}  n_test={n_test:,}  DGP: heteroskedastic compound Poisson-Gamma")
print(f"    - 'Valid' = distribution-free finite-sample guarantee (conformal) vs.")
print(f"      parametric assumption (naive)")
print()
print("  For adaptive-width intervals (narrower in low-variance segments):")
print("    from insurance_conformal.claims import TwoStageLWConformal")
print("    lw = TwoStageLWConformal(mean_model=lgb_model, p=1.5)")
print("    lw.fit(X_train, y_train)")
print("    lw.calibrate(X_cal, y_cal)")
print("    intervals = lw.predict_interval(X_test, alpha=0.10)  # shape (n, 2)")
print()
print("  For per-segment coverage diagnostics:")
print("    from insurance_conformal import subgroup_coverage")
print("    sg = subgroup_coverage(cp, X_test, y_test, alpha=0.10,")
print("                           groups=veh_bands_test, group_name='vehicle_group_band')")
print()

elapsed = time.time() - t0
print(f"Completed in {elapsed:.1f}s")
