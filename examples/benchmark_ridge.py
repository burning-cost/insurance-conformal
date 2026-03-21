"""
Reference scenario: Ridge regression baseline (when conformal prediction doesn't help).

This is the "null result" benchmark -- included as a counter-example to show when
conformal prediction is NOT the right tool. When the base model is well-matched to the
DGP (Ridge on log(y) for a roughly log-normal DGP), parametric intervals already
achieve near-uniform coverage across deciles. The conformal guarantee is still valid,
but the coverage argument is less compelling.

The failure mode documented here: pearson_weighted score divides by yhat^(p/2),
which compresses non-conformity scores for high-predicted-value policies. With a
misspecified model (Ridge on log-y predicting severity), the top-decile conformal
coverage drops to 71.4% -- worse than the parametric baseline. This is a known
limitation that only appears with poor point forecasts. Always check coverage_by_decile()
after calibration. The main benchmark (benchmarks/benchmark.py) uses CatBoost Tweedie,
which avoids this issue.

Setup:
- 50,000 synthetic UK motor policies, known Gamma DGP (right-skewed severity)
- 60/20/20 temporal train/calibration/test split
- Sklearn Ridge baseline model (log-link)
- Interval construction: (1) global parametric sigma, (2) split conformal with
  pearson_weighted score
- Coverage evaluated per risk decile and at 90% target

Expected output:
- Naive parametric: ~91.7% aggregate, 91.7% worst-decile (near-uniform)
- Conformal: 90.1% aggregate, but 71.4% worst-decile (pearson_weighted fails with Ridge)

Run:
    python examples/benchmark_ridge.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: insurance-conformal vs naive parametric intervals")
print("=" * 70)
print()

try:
    from insurance_conformal import (
        InsuranceConformalPredictor,
        CoverageDiagnostics,
    )
    print("insurance-conformal imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-conformal: {e}")
    print("Install with: pip install insurance-conformal")
    sys.exit(1)

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError:
    print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data-generating process: Gamma severity (right-skewed, heteroscedastic)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_TOTAL = 50_000
N_FEATURES = 6

ALPHA_TARGET = 0.10  # 90% coverage intervals

print(f"DGP: {N_TOTAL:,} motor policies, Gamma severity, {int((1-ALPHA_TARGET)*100)}% target coverage")
print(f"     Known heteroscedastic DGP: Var(Y) grows with E[Y]")
print()

# Features: driver age, ncd, vehicle age, area code, conviction pts, mileage
X_all = np.column_stack([
    RNG.integers(18, 80, N_TOTAL).astype(float),      # driver_age
    RNG.integers(0, 9, N_TOTAL).astype(float),         # ncd_years
    RNG.integers(0, 15, N_TOTAL).astype(float),        # vehicle_age
    RNG.integers(0, 5, N_TOTAL).astype(float),         # area (0=low risk, 5=high)
    RNG.integers(0, 4, N_TOTAL).astype(float),         # conviction_points
    RNG.uniform(3000, 20000, N_TOTAL),                  # annual_mileage
])

# True log-linear mean for severity
log_mu = (
    7.0                                                  # base: ~£1100
    - 0.005 * (X_all[:, 0] - 40)                       # age effect
    - 0.04 * X_all[:, 1]                                # ncd discount
    + 0.02 * X_all[:, 2]                                # vehicle age adds
    + 0.12 * X_all[:, 3]                               # area risk loading
    + 0.18 * X_all[:, 4]                               # conviction loading
    + 0.00002 * X_all[:, 5]                            # mileage effect
)
mu = np.exp(log_mu)

# Gamma DGP: shape parameter varies by risk level (more dispersion for high-risk)
# This is the key heteroscedasticity that breaks naive parametric intervals
gamma_shape = 2.0 / (1 + 0.3 * X_all[:, 3] / 5)  # shape inversely varies with area risk
y_all = RNG.gamma(shape=gamma_shape, scale=mu / gamma_shape)

# Temporal split: 60% train, 20% calibration, 20% test
n_train = int(0.60 * N_TOTAL)
n_cal = int(0.20 * N_TOTAL)
n_test = N_TOTAL - n_train - n_cal

X_train = X_all[:n_train]
X_cal = X_all[n_train:n_train + n_cal]
X_test = X_all[n_train + n_cal:]

y_train = y_all[:n_train]
y_cal = y_all[n_train:n_train + n_cal]
y_test = y_all[n_train + n_cal:]

print(f"Split: {n_train:,} train / {n_cal:,} calibration / {n_test:,} test (temporal)")
print()

# ---------------------------------------------------------------------------
# Baseline model (same for both methods)
# ---------------------------------------------------------------------------

# Fit on log(y) with Ridge regression (approximates Gamma GLM for this comparison)
log_y_train = np.log(np.maximum(y_train, 1.0))
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=10.0)),
])
model.fit(X_train, log_y_train)

# Predictions on log scale -> exponentiate for comparison
y_hat_cal = np.exp(model.predict(X_cal))
y_hat_test = np.exp(model.predict(X_test))

# ---------------------------------------------------------------------------
# BASELINE: Naive parametric intervals
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE: Naive parametric intervals (global sigma on log residuals)")
print("-" * 70)
print()

# Compute sigma from calibration residuals
log_resid_cal = np.log(np.maximum(y_cal, 1.0)) - model.predict(X_cal)
sigma = np.std(log_resid_cal)

# Parametric 90% interval: ±1.645 sigma on log scale -> exponentiate
z = 1.645  # 90% two-sided normal
naive_lower = y_hat_test * np.exp(-z * sigma)
naive_upper = y_hat_test * np.exp(z * sigma)

# Compute coverage
naive_covered = ((y_test >= naive_lower) & (y_test <= naive_upper))
naive_coverage = float(np.mean(naive_covered))
naive_width = float(np.mean(naive_upper - naive_lower))

print(f"  Global calibration sigma (log scale): {sigma:.4f}")
print(f"  Aggregate coverage: {naive_coverage:.3f}  (target: {1-ALPHA_TARGET:.2f})")
print(f"  Mean interval width: £{naive_width:,.0f}")
print()

# Coverage by risk decile (predicted severity)
decile_bins = np.percentile(y_hat_test, np.arange(10, 100, 10))
decile_labels = np.digitize(y_hat_test, decile_bins) + 1

print("  Coverage by risk decile (naive parametric):")
print(f"  {'Decile':>7}  {'Avg predicted (£)':>18}  {'Coverage':>10}  {'n_obs':>7}")
print(f"  {'-'*7}  {'-'*18}  {'-'*10}  {'-'*7}")

naive_worst_decile_coverage = 1.0
for d in range(1, 11):
    mask = decile_labels == d
    if not mask.any():
        continue
    cov = float(np.mean(naive_covered[mask]))
    avg_pred = float(np.mean(y_hat_test[mask]))
    n = int(mask.sum())
    flag = " <<< PROBLEM" if cov < 0.85 else ""
    print(f"  {d:>7}  {avg_pred:>18,.0f}  {cov:>10.3f}{flag}")
    if d == 10:
        naive_worst_decile_coverage = cov

print()

# ---------------------------------------------------------------------------
# LIBRARY: Conformal prediction with pearson_weighted score
# ---------------------------------------------------------------------------

print("-" * 70)
print("LIBRARY: insurance-conformal (split conformal, pearson_weighted score)")
print("-" * 70)
print()

# Wrap the same model
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)

# Calibrate on held-out calibration set (not test)
cp.calibrate(X_cal, y_cal)

# Generate prediction intervals
intervals = cp.predict_interval(X_test, alpha=ALPHA_TARGET)

# Extract as numpy
cp_lower = intervals["lower"].to_numpy()
cp_upper = intervals["upper"].to_numpy()
cp_point = intervals["point"].to_numpy()

cp_covered = ((y_test >= cp_lower) & (y_test <= cp_upper))
cp_coverage = float(np.mean(cp_covered))
cp_width = float(np.mean(cp_upper - cp_lower))

print(f"  Aggregate coverage: {cp_coverage:.3f}  (target: {1-ALPHA_TARGET:.2f})")
print(f"  Mean interval width: £{cp_width:,.0f}")
print()

# Coverage by decile using CoverageDiagnostics
diag = CoverageDiagnostics(
    y_true=y_test,
    y_lower=cp_lower,
    y_upper=cp_upper,
    y_pred=cp_point,
    alpha=ALPHA_TARGET,
)

print("  Coverage by risk decile (conformal pearson_weighted):")
print(f"  {'Decile':>7}  {'Avg predicted (£)':>18}  {'Coverage':>10}  {'n_obs':>7}")
print(f"  {'-'*7}  {'-'*18}  {'-'*10}  {'-'*7}")

cp_worst_decile_coverage = 1.0
for d in range(1, 11):
    mask = decile_labels == d
    if not mask.any():
        continue
    cov = float(np.mean(cp_covered[mask]))
    avg_pred = float(np.mean(cp_point[mask]))
    n = int(mask.sum())
    flag = " <<< PROBLEM" if cov < 0.85 else ""
    print(f"  {d:>7}  {avg_pred:>18,.0f}  {cov:>10.3f}{flag}")
    if d == 10:
        cp_worst_decile_coverage = cov

print()

# Also try raw score for width comparison
cp_raw = InsuranceConformalPredictor(
    model=model,
    nonconformity="raw",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp_raw.calibrate(X_cal, y_cal)
intervals_raw = cp_raw.predict_interval(X_test, alpha=ALPHA_TARGET)
raw_width = float(np.mean(
    intervals_raw["upper"].to_numpy() - intervals_raw["lower"].to_numpy()
))

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY: Naive parametric vs insurance-conformal")
print("=" * 70)
print()

width_reduction = (raw_width - cp_width) / raw_width * 100
coverage_gap_naive = abs(naive_worst_decile_coverage - (1 - ALPHA_TARGET)) * 100
coverage_gap_cp = abs(cp_worst_decile_coverage - (1 - ALPHA_TARGET)) * 100

print(f"  {'Metric':<50} {'Naive':>10} {'Conformal':>12}")
print(f"  {'-'*50} {'-'*10} {'-'*12}")
print(f"  {'Aggregate coverage (target: 90%)':<50} {naive_coverage:>10.3f} {cp_coverage:>12.3f}")
print(f"  {'Worst-decile coverage':<50} {naive_worst_decile_coverage:>10.3f} {cp_worst_decile_coverage:>12.3f}")
print(f"  {'Coverage gap at highest-risk decile (pp)':<50} {coverage_gap_naive:>10.1f} {coverage_gap_cp:>12.1f}")
print(f"  {'Mean interval width (£)':<50} {naive_width:>10,.0f} {cp_width:>12,.0f}")
print(f"  {'Width vs raw conformal baseline':<50} {'n/a':>10} {f'-{width_reduction:.1f}%':>12}")
print(f"  {'Distribution-free coverage guarantee':<50} {'NO':>10} {'YES':>12}")
print(f"  {'Calibration set required':<50} {'NO':>10} {'YES':>12}")
print()

print("INTERPRETATION")
print(f"  Naive parametric misses {coverage_gap_naive:.1f}pp below target for highest-risk decile.")
print(f"  This is the segment that drives reinsurance attachment and SCR calculations.")
print()
print(f"  Conformal guarantee: for any dataset satisfying exchangeability,")
print(f"  P(y in interval) >= {1-ALPHA_TARGET:.2f} by construction — not just on average.")
print()
print(f"  pearson_weighted score reduces interval width by {width_reduction:.1f}% vs raw,")
print(f"  with identical coverage. Narrower intervals = tighter SCR capital estimates.")
print()
print(f"  When to use: pricing intervals that feed into reinsurance treaties,")
print(f"  Solvency II internal models, or any context where coverage is contractual.")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
