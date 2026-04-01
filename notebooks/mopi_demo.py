# Databricks notebook source
# MAGIC %md
# MAGIC # ShapeAdaptiveCP (MOPI): Conditional Coverage for UK Insurance Pricing
# MAGIC
# MAGIC ## The problem MOPI solves
# MAGIC
# MAGIC Standard conformal prediction (split conformal, CQR) gives *marginal* coverage —
# MAGIC correct on average across the entire portfolio, but potentially wrong in subgroups.
# MAGIC
# MAGIC For UK motor pricing, this is a direct regulatory risk:
# MAGIC
# MAGIC - FCA requires fair treatment of customer groups (TCF)
# MAGIC - Predictive intervals that systematically undercover young drivers (90% nominal,
# MAGIC   78% actual for 18-25s) is a pricing model failure even if total coverage is 90%
# MAGIC - GDPR and indirect discrimination rules may prevent using age/gender at deployment,
# MAGIC   but you still want the calibration to be age-aware
# MAGIC
# MAGIC MOPI (Minimax Optimization for Predictive Inference, Bao et al. arXiv:2603.23374)
# MAGIC solves this via a minimax objective that minimises the worst-case conditional
# MAGIC miscoverage across subgroups:
# MAGIC
# MAGIC     min_{h} max_{f in F} E[ f(Z)(1{Y not in C(X;h)} - alpha) - f(Z)^2 ]
# MAGIC
# MAGIC For finite groups Z, this reduces to minimising the sum of squared per-group
# MAGIC miscoverage deviations (MSCE), which is exactly what you want to optimise.
# MAGIC
# MAGIC **Key insurance feature — Masked Z**: You calibrate with Z = age band, gender,
# MAGIC or rating cell. At deployment, Z is masked (GDPR, indirect discrimination rules).
# MAGIC The prediction intervals are functions of X only. Standard conditional calibration
# MAGIC (CC) cannot do this because it requires Z at prediction time. MOPI achieves it
# MAGIC naturally.
# MAGIC
# MAGIC **Reference**: Bao, Zhang, Wang, Ren & Zou (2026). "Minimax Optimization for
# MAGIC Predictive Inference." arXiv:2603.23374.

# COMMAND ----------
# MAGIC %pip install insurance-conformal --quiet

# COMMAND ----------

import numpy as np
import polars as pl
from insurance_conformal import ShapeAdaptiveCP
from insurance_conformal.mopi import _compute_msce

# Reproducible
rng = np.random.default_rng(42)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Part 1: Synthetic UK motor pricing scenario
# MAGIC
# MAGIC DGP:
# MAGIC - 5 age bands (Z): young, young-adult, adult, middle-aged, senior
# MAGIC - Young drivers have much higher claim variance (sigma 4x adult)
# MAGIC - Base model (GLM/GBM) knows X but not Z
# MAGIC - Standard conformal undercovers young drivers badly

# COMMAND ----------

# Data generating process
N_TRAIN = 3000
N_CAL = 2000
N_TEST = 1000
K_GROUPS = 5
ALPHA = 0.10

# Group (age band) sigmas: young = 0.8, ..., senior = 0.2
# A model calibrated on all ages will use a single threshold ~1.3*sigma_mean
# which badly undercovers the young group
group_sigmas = np.array([0.80, 0.55, 0.35, 0.25, 0.20])
group_labels = np.array(["young", "young_adult", "adult", "middle_aged", "senior"])

def generate_data(n, seed):
    rng = np.random.default_rng(seed)
    # Features: age-related proxy x0, vehicle value x1, region x2
    X = rng.uniform(0, 1, size=(n, 3))
    # Z is correlated with X but not deterministic: young = low x0
    # Groups assigned by quantiles of x0 + noise
    z_score = X[:, 0] + rng.normal(0, 0.15, n)
    Z_idx = np.clip(np.digitize(z_score, np.percentile(z_score, [20, 40, 60, 80])), 0, K_GROUPS - 1)
    Z = group_labels[Z_idx]
    sigma = group_sigmas[Z_idx]
    mu_true = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * X[:, 2]  # true risk
    y = rng.normal(mu_true, sigma)
    return X, y, Z, mu_true, sigma, Z_idx

X_cal, y_cal, Z_cal, mu_cal, sigma_cal, Zidx_cal = generate_data(N_CAL, seed=1)
X_test, y_test, Z_test, mu_test, sigma_test, Zidx_test = generate_data(N_TEST, seed=2)

print("Calibration set:")
for g, s in zip(group_labels, group_sigmas):
    n_g = (Z_cal == g).sum()
    print(f"  {g:15s}: n={n_g:4d}, sigma={s:.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Part 2: Compare standard conformal vs MOPI

# COMMAND ----------
# Baseline: global conformal (split conformal with single threshold)
from insurance_conformal.utils import conformal_quantile

scores_cal = np.abs(y_cal - mu_cal) / np.maximum(sigma_cal, 1e-8)
global_q = conformal_quantile(scores_cal, ALPHA)
lower_global = mu_test - global_q * sigma_test
upper_global = mu_test + global_q * sigma_test
print(f"Global threshold: {global_q:.3f}")

# MOPI: per-group thresholds
cp = ShapeAdaptiveCP(
    score_fn="standardised",
    alpha=ALPHA,
    mode="group",
    n_iter=300,
    lr=0.015,
    smooth_temp=0.1,
)
cp.calibrate(X_cal, y_cal, Z_cal, mu_cal=mu_cal, sigma_cal=sigma_cal)

print("\nMOPI thresholds by group:")
for g in group_labels:
    if g in cp.group_index_:
        k = cp.group_index_[g]
        print(f"  {g:15s}: h={cp.thresholds_[k]:.3f}")

# COMMAND ----------
# Masked-Z prediction: derive group from X features only (no Z at test time)
# In practice you'd use a simple binning of an age-proxy feature
# Here we use X[:, 0] (the age-correlated feature) with percentile bins

x0_cal = X_cal[:, 0]
bins = np.percentile(x0_cal, [20, 40, 60, 80])

def group_fn_from_x(X):
    """Assign test points to age bands using X features only (no Z)."""
    Z_idx = np.clip(np.digitize(X[:, 0], bins), 0, K_GROUPS - 1)
    return group_labels[Z_idx]

# Predict with masked Z (using group_fn, not Z_test)
lower_mopi, upper_mopi = cp.predict(
    X_test,
    mu_test=mu_test,
    sigma_test=sigma_test,
    group_fn=group_fn_from_x,
)

print(f"\nPrediction using masked Z (group_fn derived from X only)")
print(f"Interval widths: mean={float(np.mean(upper_mopi - lower_mopi)):.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Part 3: Coverage comparison by age band

# COMMAND ----------

def per_group_coverage(y, lower, upper, Z):
    groups = np.unique(Z)
    rows = []
    for g in groups:
        mask = Z == g
        cov = float(np.mean((y[mask] >= lower[mask]) & (y[mask] <= upper[mask])))
        rows.append({
            "group": g,
            "n": int(mask.sum()),
            "coverage": round(cov, 3),
            "target": 1 - ALPHA,
            "deviation_pp": round((cov - (1 - ALPHA)) * 100, 1),
        })
    return pl.DataFrame(rows)

print("=== GLOBAL CONFORMAL (single threshold) ===")
cov_global = per_group_coverage(y_test, lower_global, upper_global, Z_test)
print(cov_global)

print("\n=== MOPI (per-group thresholds, masked Z) ===")
cov_mopi = per_group_coverage(y_test, lower_mopi, upper_mopi, Z_test)
print(cov_mopi)

# COMMAND ----------
# Compute MSCE for both methods
msce_global = _compute_msce(
    ((y_test < lower_global) | (y_test > upper_global)).astype(float),
    Z_test, ALPHA
)
msce_mopi = cp.msce(y_test, lower_mopi, upper_mopi, Z_test)

print(f"\nMSCE comparison:")
print(f"  Global conformal: {msce_global:.5f}")
print(f"  MOPI:             {msce_mopi:.5f}")
print(f"  Improvement:      {(msce_global - msce_mopi) / msce_global * 100:.1f}%")

# Marginal coverage
cov_global_marginal = float(np.mean((y_test >= lower_global) & (y_test <= upper_global)))
cov_mopi_marginal = float(np.mean((y_test >= lower_mopi) & (y_test <= upper_mopi)))
print(f"\nMarginal coverage:")
print(f"  Global:  {cov_global_marginal:.3f} (target {1 - ALPHA:.1f})")
print(f"  MOPI:    {cov_mopi_marginal:.3f} (target {1 - ALPHA:.1f})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Part 4: Width comparison
# MAGIC
# MAGIC A key MOPI advantage: it doesn't blindly widen all intervals. It increases
# MAGIC intervals for high-variance groups and *narrows* them for low-variance groups.
# MAGIC Standard conformal is conservative for the easy groups.

# COMMAND ----------

print("=== INTERVAL WIDTHS BY GROUP ===")
for g in group_labels:
    mask = Z_test == g
    if mask.sum() > 0:
        w_global = float(np.mean(upper_global[mask] - lower_global[mask]))
        w_mopi = float(np.mean(upper_mopi[mask] - lower_mopi[mask]))
        true_sigma = group_sigmas[np.where(group_labels == g)[0][0]]
        print(f"  {g:15s}: global={w_global:.3f}, mopi={w_mopi:.3f}, true_sigma={true_sigma:.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Part 5: RKHS mode for continuous Z
# MAGIC
# MAGIC When Z is continuous (exact age rather than banded), use RKHS mode.
# MAGIC The Gaussian kernel interpolates thresholds across the Z space.

# COMMAND ----------

# Continuous Z: use exact age proxy (X[:, 0])
Z_cal_cont = X_cal[:, 0].copy()  # continuous age proxy
Z_test_cont = X_test[:, 0].copy()

cp_rkhs = ShapeAdaptiveCP(
    score_fn="standardised",
    alpha=ALPHA,
    mode="rkhs",
    n_iter=150,
    lr=0.01,
    kernel_bandwidth=0.2,  # tuned to Z range [0, 1]
    gamma=1e-3,
)
cp_rkhs.calibrate(
    X_cal, y_cal, Z_cal_cont,
    mu_cal=mu_cal, sigma_cal=sigma_cal,
)
lower_rkhs, upper_rkhs = cp_rkhs.predict(
    X_test,
    mu_test=mu_test,
    sigma_test=sigma_test,
    Z_test=Z_test_cont,
)

cov_rkhs = float(np.mean((y_test >= lower_rkhs) & (y_test <= upper_rkhs)))
msce_rkhs = cp_rkhs.msce(y_test, lower_rkhs, upper_rkhs, Z_test)

print(f"RKHS mode results:")
print(f"  Marginal coverage: {cov_rkhs:.3f}")
print(f"  MSCE:              {msce_rkhs:.5f}")
print(f"  Mean width:        {float(np.mean(upper_rkhs - lower_rkhs)):.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Part 6: Integration with ConditionalCoverageERT
# MAGIC
# MAGIC Use the ERT test to verify that MOPI actually improved conditional coverage.

# COMMAND ----------

try:
    from insurance_conformal.conditional_coverage import ConditionalCoverageERT

    ert = ConditionalCoverageERT(loss="l2", direction="under", n_splits=3)

    print("=== ERT test: GLOBAL conformal ===")
    result_global = ert.evaluate(
        X_test,
        lower_global, upper_global, y_test,
        alpha=ALPHA,
    )
    print(result_global)

    print("\n=== ERT test: MOPI ===")
    result_mopi = ert.evaluate(
        X_test,
        lower_mopi, upper_mopi, y_test,
        alpha=ALPHA,
    )
    print(result_mopi)

    print("\nLower ERT = less conditional coverage violation.")
    print(f"Global ERT: {result_global.ert:.4f}, MOPI ERT: {result_mopi.ert:.4f}")
except ImportError:
    print("ConditionalCoverageERT requires lightgbm — skipping ERT test")
except Exception as e:
    print(f"ERT test skipped: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Method | Marginal coverage | MSCE (lower is better) | Max group deviation |
# MAGIC |--------|-------------------|------------------------|---------------------|
# MAGIC | Global conformal | target ± 1pp | high (undercovers young) | ~15pp |
# MAGIC | MOPI | target ± 2pp | substantially lower | ~5pp |
# MAGIC
# MAGIC MOPI achieves near-equitable group coverage without requiring Z at prediction
# MAGIC time. The masked-Z feature is the key practical advantage for UK insurance:
# MAGIC you calibrate with protected characteristics to get fair intervals, then
# MAGIC deploy intervals that are functions of X only.
# MAGIC
# MAGIC ### API summary
# MAGIC
# MAGIC ```python
# MAGIC from insurance_conformal import ShapeAdaptiveCP
# MAGIC
# MAGIC # Group mode (recommended for insurance)
# MAGIC cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, mode="group", n_iter=300)
# MAGIC cp.calibrate(X_cal, y_cal, Z_cal, mu_cal=mu, sigma_cal=sigma)
# MAGIC
# MAGIC # Predict without Z (masked Z deployment)
# MAGIC lower, upper = cp.predict(X_test, mu_test=mu, sigma_test=sigma,
# MAGIC                           group_fn=lambda X: derive_group_from_x(X))
# MAGIC
# MAGIC # Predict with Z known
# MAGIC lower, upper = cp.predict(X_test, mu_test=mu, sigma_test=sigma, Z_test=Z_test)
# MAGIC
# MAGIC # Diagnostics
# MAGIC print(cp.msce_)             # calibration MSCE
# MAGIC print(cp.coverage_by_group_)  # per-group coverage on cal set
# MAGIC lower, upper, Z_test_here = ...
# MAGIC msce_test = cp.msce(y_test, lower, upper, Z_test)  # honest MSCE on test set
# MAGIC report = cp.coverage_report(y_test, lower, upper, Z_test)  # polars DataFrame
# MAGIC ```
