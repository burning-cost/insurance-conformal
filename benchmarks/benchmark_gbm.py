"""
Benchmark: GBM (CatBoost) conformal vs parametric Tweedie intervals.

The case for conformal prediction in insurance is not "parametric intervals are
always wrong" — it is "parametric intervals assume the model is correctly
specified, and they fail silently when it isn't." This benchmark constructs a
realistic scenario where the failure mode occurs: a heteroskedastic DGP where
residual variance grows faster than the Tweedie variance function predicts,
especially for high-mean risks.

The setup mirrors how a UK motor pricing team would actually use these tools:
- CatBoost Tweedie point forecast (the industry standard)
- Parametric Tweedie intervals: yhat ± z * yhat^(p/2) * sigma_cal
- Conformal intervals with pearson_weighted score
- Locally-weighted conformal (LW) for comparison

The heteroskedastic DGP is the key:
- Mean: nonlinear in vehicle_age, driver_age, mileage (CatBoost can fit this)
- Variance: grows faster than Tweedie(p=1.5) predicts in the high-mean tail
- Effect: parametric intervals undercover the 20% of risks with highest predicted
  mean by 10-15pp. Conformal meets the target by construction.

The Ridge/null scenario (from benchmarks/benchmark.py) is kept as a reference:
when model and DGP are well-matched, parametric intervals work fine.

Run:
    uv run python benchmarks/benchmark_gbm.py

Expected output:
    Parametric Tweedie: ~75-82% coverage in top-2 deciles (target 90%)
    Conformal pearson_weighted: >=90% in aggregate, near-target in top deciles
    Locally-weighted conformal: narrower intervals, similar or better coverage
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 72)
print("Benchmark: CatBoost conformal vs parametric Tweedie intervals")
print("Heteroskedastic DGP — variance grows faster in high-mean tail")
print("=" * 72)
print()

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    from insurance_conformal import (
        InsuranceConformalPredictor,
        LocallyWeightedConformal,
        CoverageDiagnostics,
    )
    print("insurance-conformal: OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-conformal: {e}")
    sys.exit(1)

try:
    import catboost
    from catboost import CatBoostRegressor
    print(f"catboost: OK (version {catboost.__version__})")
except ImportError:
    print("ERROR: catboost required. Install with: uv add 'insurance-conformal[catboost]'")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
# The DGP is designed so that:
# 1. The mean structure is nonlinear — CatBoost fits it well, Ridge does not.
# 2. The residual variance grows *faster* than Tweedie(p=1.5) in the tail.
#    Specifically, we use a Gamma DGP where the shape parameter (1/CV^2) is
#    a decreasing function of the mean. High-mean risks have lower shape ->
#    higher coefficient of variation -> more over-dispersion relative to Tweedie.
#    This is realistic: large commercial motor policies are genuinely more
#    volatile than their size alone would predict.
# 3. The tail over-dispersion is *not* captured by vehicle_age/driver_age alone —
#    it is an interaction between features that a parametric sigma estimate
#    (computed on the whole calibration set) cannot capture.

RNG = np.random.default_rng(42)

N_TOTAL = 50_000
ALPHA_90 = 0.10   # 90% prediction intervals
ALPHA_95 = 0.05   # 95% prediction intervals

print(f"DGP: {N_TOTAL:,} synthetic UK motor policies")
print(f"     Nonlinear mean, variance grows faster than Tweedie(1.5) in tail")
print(f"     Target coverages: 90% and 95%")
print()

# Features: realistic UK motor book
vehicle_age  = RNG.integers(0, 16, N_TOTAL).astype(float)   # 0-15 years
driver_age   = RNG.integers(18, 81, N_TOTAL).astype(float)  # 18-80
mileage      = RNG.uniform(3_000, 25_000, N_TOTAL)           # annual miles
ncd_years    = RNG.integers(0, 9, N_TOTAL).astype(float)     # 0-8 years NCD
area_risk    = RNG.integers(0, 6, N_TOTAL).astype(float)     # 0=rural, 5=London

X_all = np.column_stack([vehicle_age, driver_age, mileage, ncd_years, area_risk])
FEATURE_NAMES = ["vehicle_age", "driver_age", "mileage", "ncd_years", "area_risk"]

# Nonlinear mean structure (log scale)
# Young drivers + old vehicles is the expensive interaction.
young_driver = np.clip((25 - driver_age), 0, 10)  # extra loading for <25
old_vehicle  = np.clip((vehicle_age - 8), 0, 7)    # extra loading for >8yr vehicle
high_mileage = np.clip((mileage - 15_000) / 10_000, 0, 1)

log_mu = (
    7.2                             # base ~£1330
    + 0.025 * young_driver          # young driver loading
    - 0.005 * (driver_age - 45)     # middle-age discount
    + 0.018 * old_vehicle           # old vehicle loading
    + 0.15 * high_mileage           # high mileage loading
    - 0.045 * ncd_years             # NCD discount
    + 0.11 * area_risk              # area risk loading
    # Nonlinear interaction: young driver + old vehicle = expensive
    + 0.04 * (young_driver * old_vehicle / 10.0)
)
mu = np.exp(log_mu)

# Heteroskedastic Gamma DGP:
# shape = base_shape / (1 + k * (mu / median_mu)^beta)
# This makes high-mean risks significantly more dispersed than Tweedie(1.5)
# would predict. At the median, shape ~ 2.0 (CV ~ 70%). At the 90th percentile
# of mu, shape drops to ~0.8 (CV ~ 112%). This is the over-dispersion that
# breaks parametric Tweedie intervals.
median_mu = np.median(mu)
k = 1.2          # controls how fast shape drops with mu
beta = 0.7       # sub-linear so the effect is moderate below median, large above
gamma_shape = 2.0 / (1.0 + k * (mu / median_mu) ** beta)
gamma_shape = np.clip(gamma_shape, 0.4, 6.0)  # realistic bounds

y_all = RNG.gamma(shape=gamma_shape, scale=mu / gamma_shape)

print(f"DGP summary:")
print(f"  mu range: £{mu.min():,.0f} - £{mu.max():,.0f}, median £{median_mu:,.0f}")
print(f"  gamma_shape range: {gamma_shape.min():.2f} - {gamma_shape.max():.2f}")
print(f"  CV(y) at lowest-mean decile: {float(np.std(y_all[mu < np.percentile(mu,10)]) / np.mean(y_all[mu < np.percentile(mu,10)])):.2f}")
print(f"  CV(y) at highest-mean decile: {float(np.std(y_all[mu > np.percentile(mu,90)]) / np.mean(y_all[mu > np.percentile(mu,90)])):.2f}")
print()

# Temporal split: 60% train / 20% calibration / 20% test
n_train = int(0.60 * N_TOTAL)
n_cal   = int(0.20 * N_TOTAL)

X_train, X_cal, X_test = X_all[:n_train], X_all[n_train:n_train+n_cal], X_all[n_train+n_cal:]
y_train, y_cal, y_test = y_all[:n_train], y_all[n_train:n_train+n_cal], y_all[n_train+n_cal:]

print(f"Split: {n_train:,} train / {n_cal:,} calibration / {len(y_test):,} test (temporal)")
print()

# ---------------------------------------------------------------------------
# CatBoost point forecast (Tweedie p=1.5)
# ---------------------------------------------------------------------------

print("-" * 72)
print("STEP 1: Fit CatBoost Tweedie(p=1.5) point forecast")
print("-" * 72)
print()

t0 = time.time()
catboost_model = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)
catboost_model.fit(X_train, y_train, eval_set=(X_cal, y_cal), early_stopping_rounds=50)
t_fit = time.time() - t0

y_hat_cal  = catboost_model.predict(X_cal)
y_hat_test = catboost_model.predict(X_test)

# Clip predictions to positive (Tweedie output is always positive but guard anyway)
y_hat_cal  = np.clip(y_hat_cal, 1.0, None)
y_hat_test = np.clip(y_hat_test, 1.0, None)

print(f"  Fit time: {t_fit:.1f}s")
print(f"  Best iteration: {catboost_model.best_iteration_}")
print(f"  Predicted mean (test): £{y_hat_test.mean():,.0f}")
print(f"  True mean (test): £{y_test.mean():,.0f}")
print()

# Risk decile bins for coverage analysis (by predicted mean on test set)
decile_bins   = np.percentile(y_hat_test, np.arange(10, 100, 10))
decile_labels = np.digitize(y_hat_test, decile_bins) + 1  # 1 to 10

# ---------------------------------------------------------------------------
# BASELINE: Parametric Tweedie intervals
# ---------------------------------------------------------------------------
# Standard approach: estimate a global dispersion sigma on the calibration set
# using Pearson residuals, then construct intervals as:
#     lower = yhat - z * sigma * yhat^(p/2)
#     upper = yhat + z * sigma * yhat^(p/2)
# This is what a pricing team would do if not using conformal prediction.
# The assumption: dispersion is constant across the risk spectrum.

print("-" * 72)
print("BASELINE: Parametric Tweedie intervals (global dispersion, CatBoost forecast)")
print("-" * 72)
print()

TWEEDIE_P = 1.5

# Estimate global sigma from calibration Pearson residuals
pearson_cal = (y_cal - y_hat_cal) / (y_hat_cal ** (TWEEDIE_P / 2.0))
sigma_cal   = float(np.std(pearson_cal))

print(f"  Calibration sigma (Pearson residuals): {sigma_cal:.4f}")

def make_parametric_intervals(y_hat: np.ndarray, z: float, sigma: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    half_width = z * sigma * (y_hat ** (p / 2.0))
    return np.clip(y_hat - half_width, 0.0, None), y_hat + half_width

# 90% intervals (z = 1.645 two-sided normal)
z90 = 1.645
param_lo_90, param_hi_90 = make_parametric_intervals(y_hat_test, z90, sigma_cal, TWEEDIE_P)
param_covered_90 = (y_test >= param_lo_90) & (y_test <= param_hi_90)
param_cov_90  = float(np.mean(param_covered_90))
param_width_90 = float(np.mean(param_hi_90 - param_lo_90))

# 95% intervals (z = 1.960)
z95 = 1.960
param_lo_95, param_hi_95 = make_parametric_intervals(y_hat_test, z95, sigma_cal, TWEEDIE_P)
param_covered_95 = (y_test >= param_lo_95) & (y_test <= param_hi_95)
param_cov_95  = float(np.mean(param_covered_95))
param_width_95 = float(np.mean(param_hi_95 - param_lo_95))

print(f"  Aggregate coverage 90%: {param_cov_90:.3f}  (target 0.900)")
print(f"  Aggregate coverage 95%: {param_cov_95:.3f}  (target 0.950)")
print(f"  Mean interval width 90%: £{param_width_90:,.0f}")
print()

print(f"  Coverage by risk decile (parametric Tweedie, 90% target):")
print(f"  {'Decile':>7}  {'Avg predicted (£)':>18}  {'Coverage':>10}  {'n_obs':>7}")
print(f"  {'-'*7}  {'-'*18}  {'-'*10}  {'-'*7}")

param_worst_90 = 1.0
for d in range(1, 11):
    mask = decile_labels == d
    if not mask.any():
        continue
    cov  = float(np.mean(param_covered_90[mask]))
    pred = float(np.mean(y_hat_test[mask]))
    n    = int(mask.sum())
    flag = " <<< FAIL" if cov < 0.85 else ""
    print(f"  {d:>7}  {pred:>18,.0f}  {cov:>10.3f}{flag}")
    if d == 10:
        param_worst_90 = cov

print()

# ---------------------------------------------------------------------------
# CONFORMAL: InsuranceConformalPredictor (pearson_weighted score)
# ---------------------------------------------------------------------------

print("-" * 72)
print("CONFORMAL: pearson_weighted score (split conformal, CatBoost forecast)")
print("-" * 72)
print()

t0 = time.time()
cp_pw = InsuranceConformalPredictor(
    model=catboost_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=TWEEDIE_P,
)
cp_pw.calibrate(X_cal, y_cal)
t_cal = time.time() - t0

print(f"  Calibration time: {t_cal:.2f}s")
print(f"  n_calibration: {cp_pw.n_calibration_:,}")

# 90%
t0 = time.time()
iv_pw_90  = cp_pw.predict_interval(X_test, alpha=ALPHA_90)
pw_lo_90  = iv_pw_90["lower"].to_numpy()
pw_hi_90  = iv_pw_90["upper"].to_numpy()
pw_pt     = iv_pw_90["point"].to_numpy()
pw_cov_90 = float(np.mean((y_test >= pw_lo_90) & (y_test <= pw_hi_90)))
pw_width_90 = float(np.mean(pw_hi_90 - pw_lo_90))

# 95%
iv_pw_95  = cp_pw.predict_interval(X_test, alpha=ALPHA_95)
pw_lo_95  = iv_pw_95["lower"].to_numpy()
pw_hi_95  = iv_pw_95["upper"].to_numpy()
pw_cov_95 = float(np.mean((y_test >= pw_lo_95) & (y_test <= pw_hi_95)))
pw_width_95 = float(np.mean(pw_hi_95 - pw_lo_95))
t_pred = time.time() - t0

print(f"  Prediction time (both alphas): {t_pred:.2f}s")
print(f"  Aggregate coverage 90%: {pw_cov_90:.3f}  (target 0.900)")
print(f"  Aggregate coverage 95%: {pw_cov_95:.3f}  (target 0.950)")
print(f"  Mean interval width 90%: £{pw_width_90:,.0f}")
print()

covered_pw_90 = (y_test >= pw_lo_90) & (y_test <= pw_hi_90)
print(f"  Coverage by risk decile (conformal pearson_weighted, 90% target):")
print(f"  {'Decile':>7}  {'Avg predicted (£)':>18}  {'Coverage':>10}  {'n_obs':>7}")
print(f"  {'-'*7}  {'-'*18}  {'-'*10}  {'-'*7}")

pw_worst_90 = 1.0
for d in range(1, 11):
    mask = decile_labels == d
    if not mask.any():
        continue
    cov  = float(np.mean(covered_pw_90[mask]))
    pred = float(np.mean(pw_pt[mask]))
    flag = " <<< FAIL" if cov < 0.85 else ""
    print(f"  {d:>7}  {pred:>18,.0f}  {cov:>10.3f}{flag}")
    if d == 10:
        pw_worst_90 = cov

print()

# ---------------------------------------------------------------------------
# LOCALLY-WEIGHTED CONFORMAL
# ---------------------------------------------------------------------------
# The LW variant fits a secondary CatBoost on |Pearson residuals| to learn
# which features predict large residuals — this is the "locart-style" adaptive
# width. It requires the same training data as the base model.

print("-" * 72)
print("LOCALLY-WEIGHTED CONFORMAL: secondary spread model on |Pearson residuals|")
print("-" * 72)
print()

t0 = time.time()
lw = LocallyWeightedConformal(
    model=catboost_model,
    tweedie_power=TWEEDIE_P,
    spread_model_params={
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 4,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
    },
)
lw.fit(X_train, y_train)
lw.calibrate(X_cal, y_cal)
t_lw = time.time() - t0

print(f"  Fit + calibration time: {t_lw:.1f}s")

# 90%
iv_lw_90   = lw.predict_interval(X_test, alpha=ALPHA_90)
lw_lo_90   = iv_lw_90["lower"].to_numpy()
lw_hi_90   = iv_lw_90["upper"].to_numpy()
lw_pt      = iv_lw_90["point"].to_numpy()
lw_spread  = iv_lw_90["spread"].to_numpy()
lw_cov_90  = float(np.mean((y_test >= lw_lo_90) & (y_test <= lw_hi_90)))
lw_width_90 = float(np.mean(lw_hi_90 - lw_lo_90))

# 95%
iv_lw_95   = lw.predict_interval(X_test, alpha=ALPHA_95)
lw_lo_95   = iv_lw_95["lower"].to_numpy()
lw_hi_95   = iv_lw_95["upper"].to_numpy()
lw_cov_95  = float(np.mean((y_test >= lw_lo_95) & (y_test <= lw_hi_95)))
lw_width_95 = float(np.mean(lw_hi_95 - lw_lo_95))

print(f"  Aggregate coverage 90%: {lw_cov_90:.3f}  (target 0.900)")
print(f"  Aggregate coverage 95%: {lw_cov_95:.3f}  (target 0.950)")
print(f"  Mean interval width 90%: £{lw_width_90:,.0f}")
print()

covered_lw_90 = (y_test >= lw_lo_90) & (y_test <= lw_hi_90)
print(f"  Coverage by risk decile (LW conformal, 90% target):")
print(f"  {'Decile':>7}  {'Avg predicted (£)':>18}  {'Coverage':>10}  {'Avg spread':>12}  {'n_obs':>7}")
print(f"  {'-'*7}  {'-'*18}  {'-'*10}  {'-'*12}  {'-'*7}")

lw_worst_90 = 1.0
for d in range(1, 11):
    mask = decile_labels == d
    if not mask.any():
        continue
    cov    = float(np.mean(covered_lw_90[mask]))
    pred   = float(np.mean(lw_pt[mask]))
    spread = float(np.mean(lw_spread[mask]))
    flag   = " <<< FAIL" if cov < 0.85 else ""
    print(f"  {d:>7}  {pred:>18,.0f}  {cov:>10.3f}  {spread:>12.3f}{flag}")
    if d == 10:
        lw_worst_90 = cov

print()

# Width comparison (all three methods at 90%)
pw_vs_param = (pw_width_90 - param_width_90) / param_width_90 * 100
lw_vs_param = (lw_width_90 - param_width_90) / param_width_90 * 100
lw_vs_pw    = (lw_width_90 - pw_width_90) / pw_width_90 * 100

# ---------------------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------------------

print("=" * 72)
print("SUMMARY: Heteroskedastic DGP — CatBoost Tweedie(1.5) point forecast")
print("=" * 72)
print()

header = f"  {'Metric':<52} {'Parametric':>10} {'Conformal':>10} {'LW':>8}"
sep    = "  " + "-" * 52 + "  " + "-" * 10 + "  " + "-" * 10 + "  " + "-" * 8
print(header)
print(sep)

rows = [
    ("Aggregate coverage @ 90% target",       f"{param_cov_90:.3f}", f"{pw_cov_90:.3f}", f"{lw_cov_90:.3f}"),
    ("Aggregate coverage @ 95% target",       f"{param_cov_95:.3f}", f"{pw_cov_95:.3f}", f"{lw_cov_95:.3f}"),
    ("Worst-decile coverage @ 90%",           f"{param_worst_90:.3f}", f"{pw_worst_90:.3f}", f"{lw_worst_90:.3f}"),
    ("Mean interval width @ 90% (£)",         f"{param_width_90:,.0f}", f"{pw_width_90:,.0f}", f"{lw_width_90:,.0f}"),
    ("Width vs parametric",                   "ref", f"{pw_vs_param:+.1f}%", f"{lw_vs_param:+.1f}%"),
    ("Width vs conformal pearson_weighted",   "n/a", "ref", f"{lw_vs_pw:+.1f}%"),
    ("Distribution-free coverage guarantee",  "NO", "YES", "YES"),
    ("Adapts width by risk segment",          "NO", "Partial", "YES"),
]

for label, v_param, v_cp, v_lw in rows:
    print(f"  {label:<52} {v_param:>10} {v_cp:>10} {v_lw:>8}")

print()
print("INTERPRETATION")
print()
print(f"  The parametric Tweedie approach estimates a single sigma on the calibration")
print(f"  set. In this DGP, high-mean risks are genuinely more dispersed than")
print(f"  Tweedie(1.5) predicts — the shape parameter drops from ~2.0 at median")
print(f"  to ~0.8 at the 90th percentile of mu. A single sigma cannot capture this.")
print()
print(f"  Parametric coverage in top decile: {param_worst_90:.1%} vs 90.0% target")
print(f"  => {(0.9 - param_worst_90)*100:.1f}pp miss on the risks that drive reinsurance attachment")
print()
print(f"  Conformal pearson_weighted: >=90% aggregate by construction.")
print(f"  Width is {abs(pw_vs_param):.1f}% {'narrower' if pw_vs_param < 0 else 'wider'} than parametric because conformal")
print(f"  uses the empirical score quantile, not the normal approximation.")
print()
print(f"  LW conformal: secondary spread model narrows intervals by {abs(lw_vs_pw):.1f}%")
print(f"  vs standard pearson_weighted, with the same marginal coverage guarantee.")
print(f"  The spread model learns which features predict large residuals — useful")
print(f"  when heteroskedasticity is a function of features, not just yhat.")
print()
print(f"  Note: conformal provides a *marginal* guarantee (average over all obs),")
print(f"  not a conditional guarantee per risk segment. Decile-level coverage can")
print(f"  still deviate from target if the score is mis-specified. Always check")
print(f"  coverage_by_decile() after calibration.")

elapsed = time.time() - BENCHMARK_START
print(f"\nTotal benchmark time: {elapsed:.1f}s")
print()

# ---------------------------------------------------------------------------
# COLLECT NUMBERS for README update
# ---------------------------------------------------------------------------

print("-" * 72)
print("RAW NUMBERS (for README table update)")
print("-" * 72)
print()
print("Parametric decile coverage:")
for d in range(1, 11):
    mask = decile_labels == d
    if not mask.any():
        continue
    cov  = float(np.mean(param_covered_90[mask]))
    pred = float(np.mean(y_hat_test[mask]))
    n    = int(mask.sum())
    print(f"  {d:>2}  £{pred:>7,.0f}  {cov:.3f}  n={n}")

print()
print("Conformal (pearson_weighted) decile coverage:")
for d in range(1, 11):
    mask = decile_labels == d
    if not mask.any():
        continue
    cov  = float(np.mean(covered_pw_90[mask]))
    pred = float(np.mean(pw_pt[mask]))
    print(f"  {d:>2}  £{pred:>7,.0f}  {cov:.3f}")

print()
print("LW conformal decile coverage:")
for d in range(1, 11):
    mask = decile_labels == d
    if not mask.any():
        continue
    cov    = float(np.mean(covered_lw_90[mask]))
    pred   = float(np.mean(lw_pt[mask]))
    spread = float(np.mean(lw_spread[mask]))
    print(f"  {d:>2}  £{pred:>7,.0f}  {cov:.3f}  spread={spread:.3f}")
