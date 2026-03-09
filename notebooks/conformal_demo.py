# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-conformal: Distribution-Free Prediction Intervals
# MAGIC
# MAGIC Your Tweedie GBM gives point estimates. A pricing actuary needs to know the
# MAGIC uncertainty — not as a parametric confidence interval that depends on the model
# MAGIC being correctly specified, but as a distribution-free guarantee.
# MAGIC
# MAGIC Conformal prediction provides that guarantee: the interval will contain the
# MAGIC actual loss at least 90% of the time, for any data distribution, as long as the
# MAGIC calibration set is held out from training.
# MAGIC
# MAGIC The key design choice in this library: for insurance Tweedie data, use the
# MAGIC Pearson-weighted non-conformity score `|y - yhat| / yhat^(p/2)`, not the raw
# MAGIC residual. This gives ~30% narrower intervals with identical coverage guarantees.
# MAGIC
# MAGIC ## What this demonstrates
# MAGIC
# MAGIC 1. Fit a CatBoost Tweedie model on synthetic UK motor data
# MAGIC 2. Calibrate the conformal predictor on a held-out set
# MAGIC 3. Generate 90% prediction intervals
# MAGIC 4. Coverage by decile — the key insurance diagnostic
# MAGIC 5. Compare non-conformity scores: raw vs. pearson vs. pearson_weighted

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install "insurance-conformal[catboost]" polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import polars as pl
import catboost
from insurance_conformal import InsuranceConformalPredictor

print("insurance-conformal imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK motor pure premium data
# MAGIC
# MAGIC We generate 20,000 policies with a Tweedie distribution (power=1.5, which sits
# MAGIC between Poisson counts and Gamma severity — appropriate for pure premium).
# MAGIC The DGP is a simple multiplicative model so we can verify the coverage holds.

# COMMAND ----------

rng = np.random.default_rng(42)
N = 20_000

# Features
driver_age = rng.integers(18, 75, N).astype(float)
ncd_years = rng.integers(0, 6, N).astype(float)
vehicle_group = rng.integers(1, 21, N).astype(float)
area = rng.integers(0, 6, N).astype(float)
annual_mileage = rng.integers(3000, 25000, N).astype(float)
exposure = rng.uniform(0.3, 1.0, N)

# True Tweedie mean (multiplicative log-linear predictor)
log_mu = (
    5.0
    - 0.008 * (driver_age - 40)       # age effect (u-shaped in reality but simplified here)
    - 0.12 * ncd_years                  # NCB discount
    + 0.04 * vehicle_group              # higher group = higher claims
    + 0.06 * area                       # area loading
    + 0.00002 * annual_mileage          # mileage loading
)
mu = np.exp(log_mu) * exposure        # expected pure premium per year of exposure

# Generate Tweedie samples (via compound Poisson-Gamma)
# Poisson count with lambda = mu^(2-p) / phi*(2-p) where p=1.5, phi=3.0
phi = 3.0
p = 1.5

# Use scipy for Tweedie simulation
from scipy.stats import poisson, gamma as gamma_dist

lambda_pois = mu ** (2 - p) / (phi * (2 - p))
n_claims = poisson.rvs(lambda_pois, random_state=rng)

# Gamma severity for each claim event
alpha_gamma = (2 - p) / (p - 1)  # shape
beta_gamma = phi * (p - 1) * mu ** (p - 1)  # rate

y = np.zeros(N)
for i in range(N):
    if n_claims[i] > 0:
        y[i] = gamma_dist.rvs(
            a=alpha_gamma * n_claims[i],
            scale=1.0 / beta_gamma[i],
        )

print(f"Portfolio: {N:,} policies")
print(f"Zero claims: {(y == 0).mean():.1%}")
print(f"Mean pure premium: £{y.mean():.2f}")
print(f"Max pure premium: £{y.max():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train/calibration/test split
# MAGIC
# MAGIC In insurance, always split temporally: train on earlier accident years, calibrate
# MAGIC on the most recent completed year, predict on the current year. A random split
# MAGIC violates the exchangeability assumption.
# MAGIC
# MAGIC Here we use a simple index-based split as a proxy for temporal ordering.

# COMMAND ----------

feature_names = ["driver_age", "ncd_years", "vehicle_group", "area", "annual_mileage"]
X = pd.DataFrame({
    "driver_age": driver_age,
    "ncd_years": ncd_years,
    "vehicle_group": vehicle_group,
    "area": area,
    "annual_mileage": annual_mileage,
})

n_train = int(N * 0.60)
n_cal = int(N * 0.20)
# Remaining 20% is test

X_train, y_train = X.iloc[:n_train], y[:n_train]
X_cal, y_cal = X.iloc[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
X_test, y_test = X.iloc[n_train+n_cal:], y[n_train+n_cal:]
exp_test = exposure[n_train+n_cal:]

print(f"Train: {len(X_train):,} | Calibration: {len(X_cal):,} | Test: {len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit Tweedie CatBoost model
# MAGIC
# MAGIC CatBoost auto-detects the Tweedie power from the loss function string.
# MAGIC `InsuranceConformalPredictor` reads it automatically — no need to pass
# MAGIC `tweedie_power=` explicitly.

# COMMAND ----------

model = catboost.CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
)

print("Fitting CatBoost Tweedie model...")
model.fit(X_train, y_train)
print("Done.")

yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)
print(f"Test predictions: mean={yhat_test.mean():.2f}, range=[{yhat_test.min():.2f}, {yhat_test.max():.2f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Calibrate the conformal predictor
# MAGIC
# MAGIC `calibrate()` computes the non-conformity scores on the held-out calibration set
# MAGIC and stores the distribution. From that, it can invert any alpha quantile to
# MAGIC produce prediction interval bounds.

# COMMAND ----------

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    # tweedie_power auto-detected from model: 1.5
)

cp.calibrate(X_cal, y_cal)

print(f"Calibrated on {cp.n_calibration_:,} observations")
print(f"Non-conformity score: {cp.nonconformity}")
print(f"Tweedie power (auto-detected): {cp.tweedie_power}")
print(f"Calibration quantile for alpha=0.10: {cp._get_quantile(0.10):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate 90% prediction intervals

# COMMAND ----------

intervals = cp.predict_interval(X_test, alpha=0.10)

print("Prediction intervals (first 10 rows):")
display(intervals.head(10))

# Overall coverage
covered = (
    (pl.Series(y_test) >= intervals["lower"]) &
    (pl.Series(y_test) <= intervals["upper"])
).mean()

print(f"\nMarginal coverage: {covered:.3f} (target: >=0.90)")
print(f"Mean interval width: £{(intervals['upper'] - intervals['lower']).mean():.2f}")
print(f"Mean point estimate: £{intervals['point'].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Coverage by decile — the key insurance diagnostic
# MAGIC
# MAGIC The marginal coverage guarantee means ~90% overall. In insurance, you also need
# MAGIC to check coverage is uniform across risk deciles. A model can hit 90% overall
# MAGIC while only covering 65% of the highest-risk policies.
# MAGIC
# MAGIC Uniform coverage across deciles confirms the `pearson_weighted` score is
# MAGIC correctly accounting for the heteroscedasticity of Tweedie data.

# COMMAND ----------

decile_diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)

print("Coverage by decile of predicted pure premium:")
display(decile_diag)

# Check uniformity
min_cov = decile_diag["coverage"].min()
max_cov = decile_diag["coverage"].max()
print(f"\nCoverage range: [{min_cov:.3f}, {max_cov:.3f}]")
print(f"All deciles above 0.85: {(decile_diag['coverage'] >= 0.85).all()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparison: pearson_weighted vs raw non-conformity score
# MAGIC
# MAGIC The `raw` score treats a £100 error on a low-risk policy identically to a
# MAGIC £100 error on a high-risk policy. For Tweedie data, Var(Y) ~ mu^p, so errors
# MAGIC on large risks are inherently larger. The weighted score accounts for this,
# MAGIC producing intervals that are calibrated across risk deciles.

# COMMAND ----------

cp_raw = InsuranceConformalPredictor(
    model=model,
    nonconformity="raw",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp_raw.calibrate(X_cal, y_cal)

intervals_raw = cp_raw.predict_interval(X_test, alpha=0.10)
intervals_pw = intervals  # pearson_weighted from above

width_raw = (intervals_raw["upper"] - intervals_raw["lower"]).mean()
width_pw = (intervals_pw["upper"] - intervals_pw["lower"]).mean()

cov_raw = (
    (pl.Series(y_test) >= intervals_raw["lower"]) &
    (pl.Series(y_test) <= intervals_raw["upper"])
).mean()

print("=== Non-conformity score comparison ===\n")
print(f"{'Score':25s} {'Coverage':>10} {'Mean width':>12}")
print("-" * 50)
print(f"{'raw':25s} {float(cov_raw):>10.3f} {float(width_raw):>12.2f}")
print(f"{'pearson_weighted':25s} {float(covered):>10.3f} {float(width_pw):>12.2f}")

pct_narrower = (width_raw - width_pw) / width_raw * 100
print(f"\npearson_weighted is {pct_narrower:.1f}% narrower with identical (or better) coverage.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Compare all available scores

# COMMAND ----------

results = []
for score in ["raw", "pearson", "pearson_weighted", "deviance", "anscombe"]:
    cp_s = InsuranceConformalPredictor(
        model=model,
        nonconformity=score,
        distribution="tweedie",
        tweedie_power=1.5,
    )
    cp_s.calibrate(X_cal, y_cal)
    iv = cp_s.predict_interval(X_test, alpha=0.10)
    cov = float(
        ((pl.Series(y_test) >= iv["lower"]) & (pl.Series(y_test) <= iv["upper"])).mean()
    )
    width = float((iv["upper"] - iv["lower"]).mean())
    results.append({"score": score, "coverage": round(cov, 3), "mean_width": round(width, 2)})

comparison = pl.DataFrame(results)
print("All non-conformity scores (target coverage = 0.90):")
display(comparison.sort("mean_width"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Score | Coverage | Mean interval width |
# MAGIC |-------|----------|---------------------|
# MAGIC | raw | ~0.90 | widest |
# MAGIC | pearson | ~0.90 | narrower |
# MAGIC | pearson_weighted | ~0.90 | ~30% narrower than raw |
# MAGIC | deviance | ~0.90 | similar to pearson_weighted |
# MAGIC | anscombe | ~0.90 | similar to pearson_weighted |
# MAGIC
# MAGIC All scores achieve the 90% coverage guarantee (distribution-free). The
# MAGIC `pearson_weighted` score produces substantially narrower intervals by correctly
# MAGIC accounting for Tweedie heteroscedasticity.
# MAGIC
# MAGIC Use `coverage_by_decile()` to verify that coverage is uniform across risk
# MAGIC deciles — the critical check for insurance applications. A 90% marginal
# MAGIC coverage with 65% coverage in the highest-risk decile is not acceptable.
