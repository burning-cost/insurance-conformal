# Databricks notebook source
# MAGIC %md
# MAGIC # ConformalisedSurvival: Doubly Robust Conformal Survival Analysis
# MAGIC
# MAGIC ## The problem
# MAGIC
# MAGIC Right-censored survival data appears throughout insurance but rarely gets
# MAGIC treated properly in pricing models:
# MAGIC
# MAGIC - **Protection/life**: you observe either the death date or the last policy
# MAGIC   anniversary before the study closed — but not both.
# MAGIC - **Lapse**: you observe either when a customer left or the current date
# MAGIC   for active policies.
# MAGIC - **Liability development**: you observe claim closure dates for settled
# MAGIC   claims, but not for open claims.
# MAGIC
# MAGIC The challenge: standard conformal prediction requires fully observed outcomes.
# MAGIC When the outcome (event time T) is censored, the standard approach breaks.
# MAGIC
# MAGIC ## The solution
# MAGIC
# MAGIC `ConformalisedSurvival` implements Sesia & Svetnik (arXiv:2412.09729):
# MAGIC
# MAGIC 1. **Impute** unobserved censoring times via a censoring model
# MAGIC 2. **Filter** calibration points to those informative at the target cutoff
# MAGIC 3. **Weight** by inverse probability of censoring (IPCW)
# MAGIC 4. **Apply** weighted conformal quantile correction
# MAGIC
# MAGIC The result: a lower prediction bound L_hat(x) with asymptotic guarantee
# MAGIC P[T >= L_hat(X)] >= 1 - alpha. Doubly robust: coverage holds if either
# MAGIC the survival model or the censoring model is correctly specified.

# COMMAND ----------

# MAGIC %pip install insurance-conformal>=1.2.0

# COMMAND ----------

# DBTITLE 1,Imports and setup
import numpy as np
import polars as pl

from insurance_conformal import (
    ConformalisedSurvival,
    KaplanMeierCensoringModel,
)

print(f"insurance-conformal version: {__import__('insurance_conformal').__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate synthetic right-censored data
# MAGIC
# MAGIC We simulate a UK protection book with:
# MAGIC - Event times T ~ Exponential (mean ~5 years, roughly 20% annual death rate)
# MAGIC - Censoring times C ~ Exponential (mean ~3.3 years, 30% annual lapse rate)
# MAGIC - Observed: T_tilde = min(T, C), E = 1{T <= C}
# MAGIC
# MAGIC In practice, you'd replace this with your actual policy-level data.

# COMMAND ----------

# DBTITLE 1,Synthetic data generation
def generate_protection_data(
    n: int,
    lambda_t: float = 0.20,   # event rate (deaths)
    lambda_c: float = 0.30,   # censoring rate (lapses)
    n_features: int = 5,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic right-censored protection book.

    Features X are drawn from a multivariate normal to represent
    risk factors like age, BMI, smoking status, sum assured, policy duration.

    Returns a dict with X, t (observed times), e (event indicator), T_true.
    """
    rng = np.random.default_rng(seed)

    # Event and censoring times
    T = rng.exponential(1.0 / lambda_t, size=n)
    C = rng.exponential(1.0 / lambda_c, size=n)
    t = np.minimum(T, C)
    e = (T <= C).astype(float)

    # Risk features (standardised)
    X = rng.standard_normal((n, n_features))

    return {"X": X, "t": t, "e": e, "T_true": T}


# Training and calibration data (used to fit and calibrate the models)
data_train = generate_protection_data(n=2000, seed=1)
data_cal   = generate_protection_data(n=800,  seed=2)
data_test  = generate_protection_data(n=500,  seed=3)

print(f"Training set:    {len(data_train['t'])} observations")
print(f"Calibration set: {len(data_cal['t'])} observations")
print(f"Test set:        {len(data_test['t'])} observations")
print()
print(f"Event rate (calibration):  {data_cal['e'].mean():.1%}")
print(f"Censoring rate (cal):      {(1 - data_cal['e']).mean():.1%}")
print(f"Median observed time (cal): {np.median(data_cal['t']):.2f} years")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fit a survival model
# MAGIC
# MAGIC The `ConformalisedSurvival` class accepts any model implementing
# MAGIC `predict_quantile(X, alpha) -> np.ndarray`. Here we use an oracle model
# MAGIC that knows the true distribution. In practice, you'd use:
# MAGIC
# MAGIC - `scikit-survival` `CoxPHSurvivalAnalysis` or `RandomSurvivalForest`
# MAGIC - `lifelines` `CoxPHFitter` or `WeibullAFTFitter`
# MAGIC - Any custom model wrapped to implement the protocol

# COMMAND ----------

# DBTITLE 1,Oracle survival model (for demonstration)
class OracleSurvivalModel:
    """
    For demonstration: oracle model with known exponential distribution.

    In production, replace with a fitted sklearn-survival or lifelines model
    wrapped via SksurvCoxCensoringAdapter or LifelinesCoxCensoringAdapter.
    """
    def __init__(self, lambda_t: float):
        self.lambda_t = lambda_t

    def predict_quantile(self, X: np.ndarray, alpha: float) -> np.ndarray:
        # alpha-quantile of Exp(lambda): -ln(1-alpha) / lambda
        q = -np.log(1.0 - alpha) / self.lambda_t
        return np.full(len(X), float(q))


# In practice:
# from sksurv.linear_model import CoxPHSurvivalAnalysis
# from insurance_conformal import SksurvCoxCensoringAdapter
# cox = CoxPHSurvivalAnalysis()
# structured_y = np.array([(bool(e), t) for e, t in zip(data_train['e'], data_train['t'])],
#                          dtype=[('event', bool), ('time', float)])
# cox.fit(data_train['X'], structured_y)
# surv_model = ...  # wrap to implement predict_quantile

surv_model = OracleSurvivalModel(lambda_t=0.20)
print("Survival model ready.")
print(f"  10th percentile (alpha=0.10): {surv_model.predict_quantile(np.zeros((1, 5)), 0.10)[0]:.3f} years")
print(f"  Median (alpha=0.50):          {surv_model.predict_quantile(np.zeros((1, 5)), 0.50)[0]:.3f} years")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Fit the censoring model
# MAGIC
# MAGIC The censoring model estimates P[C > t | X=x] — the probability that
# MAGIC a policyholder is still active at time t.
# MAGIC
# MAGIC `KaplanMeierCensoringModel` is the simplest option: it assumes censoring
# MAGIC is independent of covariates (i.e. lapses happen uniformly across the book).
# MAGIC
# MAGIC For a UK book where lapse propensity correlates with risk factors
# MAGIC (adverse selection), use `LifelinesCoxCensoringAdapter` or
# MAGIC `SksurvCoxCensoringAdapter` with a model fitted on the reverse event.

# COMMAND ----------

# DBTITLE 1,Fit KM censoring model
km_censoring = KaplanMeierCensoringModel(random_state=42)
km_censoring.fit(data_cal["t"], data_cal["e"])

# Check: survival curve at a few time points
check_times = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
X_dummy = np.zeros((5, 5))
surv_at_check = km_censoring.predict_survival_at(X_dummy, check_times)

print("KM censoring survival curve P[C > t]:")
for ti, si in zip(check_times, surv_at_check):
    print(f"  t={ti:.1f} years: {si:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Calibrate ConformalisedSurvival
# MAGIC
# MAGIC The `.calibrate()` method runs the full DR-COSARC algorithm:
# MAGIC 1. Imputes censoring times for event-observed points (Algorithm 1)
# MAGIC 2. Filters to points where imputed censoring time >= c_0
# MAGIC 3. Computes IPCW weights
# MAGIC 4. Computes the weighted conformal quantile

# COMMAND ----------

# DBTITLE 1,Calibrate
cs = ConformalisedSurvival(
    survival_model=surv_model,
    censoring_model=km_censoring,
    method="fixed_cutoff",
    cutoff=None,          # uses median observed time
    alpha=0.10,           # 90% lower prediction bound
    n_impute=20,          # Monte Carlo draws per event point
    random_state=42,
)

cs.calibrate(
    X_cal=data_cal["X"],
    t_cal=data_cal["t"],
    e_cal=data_cal["e"],
)

print(repr(cs))
print()

# Inspect calibration summary
summary = cs.calibration_summary()
print("Calibration summary:")
print(summary.to_pandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Produce lower prediction bounds for test data

# COMMAND ----------

# DBTITLE 1,Predict lower bounds
bounds = cs.predict_lower_bound(data_test["X"])
print("Lower prediction bounds (first 10 test observations):")
print(bounds.head(10))

print(f"\nLower bound statistics:")
print(f"  Mean LPB:    {bounds['lower_bound'].mean():.3f} years")
print(f"  Median LPB:  {bounds['lower_bound'].median():.3f} years")
print(f"  Min LPB:     {bounds['lower_bound'].min():.3f} years")
print(f"  Max LPB:     {bounds['lower_bound'].max():.3f} years")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Coverage diagnostics
# MAGIC
# MAGIC Coverage is evaluated only on event-observed (uncensored) test points,
# MAGIC because we don't know the true event time for censored observations.

# COMMAND ----------

# DBTITLE 1,Coverage diagnostics
diag = cs.coverage_diagnostics(
    X_test=data_test["X"],
    t_test=data_test["t"],
    e_test=data_test["e"],
)

print("Coverage diagnostics on test set:")
print(diag.to_pandas().to_string(index=False))

# Also evaluate against true event times (available in simulation, not in practice)
lb = bounds["lower_bound"].to_numpy()
T_true = data_test["T_true"]
coverage_oracle = (lb <= T_true).mean()
print(f"\nOracle coverage (against true T, not observable in practice): {coverage_oracle:.1%}")
print("Note: the conformal guarantee is on P[T >= L_hat(X)], which matches this.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Varying alpha (confidence levels)
# MAGIC
# MAGIC In protection pricing and Solvency II contexts, you might need different
# MAGIC confidence levels. The alpha parameter controls the miscoverage rate.

# COMMAND ----------

# DBTITLE 1,Multiple alpha levels
# Note: each alpha requires its own calibration — the stored quantile is alpha-specific
results = {}
for alpha in [0.05, 0.10, 0.20]:
    cs_a = ConformalisedSurvival(
        survival_model=surv_model,
        censoring_model=km_censoring,
        alpha=alpha,
        n_impute=20,
        random_state=42,
    )
    cs_a.calibrate(data_cal["X"], data_cal["t"], data_cal["e"])
    diag_a = cs_a.coverage_diagnostics(data_test["X"], data_test["t"], data_test["e"])
    results[alpha] = {
        "target": 1 - alpha,
        "empirical": float(diag_a["empirical_coverage"][0]),
        "n_filtered": cs_a.n_filtered_,
    }

print(f"{'Alpha':<8} {'Target':<10} {'Empirical':<12} {'n_filtered':<12}")
print("-" * 44)
for alpha, r in results.items():
    print(f"{alpha:<8.2f} {r['target']:<10.1%} {r['empirical']:<12.1%} {r['n_filtered']:<12}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Double robustness illustration
# MAGIC
# MAGIC Even with a misspecified survival model, coverage should be maintained
# MAGIC provided the censoring model is correct — the 'doubly robust' property.

# COMMAND ----------

# DBTITLE 1,Double robustness demonstration
class MisspecifiedSurvivalModel:
    """Survival model that systematically underpredicts survival times."""
    def __init__(self, lambda_t: float, bias_factor: float = 1.5):
        self.lambda_t = lambda_t * bias_factor  # biased lambda -> smaller quantiles

    def predict_quantile(self, X: np.ndarray, alpha: float) -> np.ndarray:
        q = -np.log(1.0 - alpha) / self.lambda_t
        return np.full(len(X), float(q))


biased_model = MisspecifiedSurvivalModel(lambda_t=0.20, bias_factor=1.5)

cs_biased = ConformalisedSurvival(
    survival_model=biased_model,
    censoring_model=km_censoring,  # censoring model is still correct
    alpha=0.10,
    n_impute=20,
    random_state=42,
)
cs_biased.calibrate(data_cal["X"], data_cal["t"], data_cal["e"])

diag_biased = cs_biased.coverage_diagnostics(data_test["X"], data_test["t"], data_test["e"])
diag_oracle = cs.coverage_diagnostics(data_test["X"], data_test["t"], data_test["e"])

print("Coverage with oracle survival model:      "
      f"{float(diag_oracle['empirical_coverage'][0]):.1%}")
print("Coverage with misspecified survival model: "
      f"{float(diag_biased['empirical_coverage'][0]):.1%}")
print()
print("Double robustness: the conformal correction compensates for survival model bias,")
print("provided the censoring model is correctly specified.")
print("Both should be >= 90% (the 1 - alpha = 0.90 target).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC `ConformalisedSurvival` provides:
# MAGIC
# MAGIC - **Valid coverage** on right-censored data: P[T >= L_hat(X)] >= 1 - alpha
# MAGIC - **Double robustness**: works even with misspecified survival model
# MAGIC - **IPCW correction**: correct for biased calibration due to censoring
# MAGIC - **Clean interface**: matches the rest of the insurance-conformal library
# MAGIC
# MAGIC **When to use this**:
# MAGIC - Protection book: lower bound on time to death = you are X% confident the
# MAGIC   policyholder won't die in the next L_hat years
# MAGIC - Lapse: lower bound on time to lapse = you are X% confident the policy
# MAGIC   will remain in force for at least L_hat years
# MAGIC - Liability: lower bound on time to claim closure
# MAGIC
# MAGIC **Limitations to document for actuarial sign-off**:
# MAGIC - Coverage guarantee is asymptotic (not finite-sample like standard CP)
# MAGIC - Requires T perp C | X (conditional independence of event and censoring)
# MAGIC - Produces lower bounds only (not upper bounds or intervals)
# MAGIC
# MAGIC Paper: Sesia & Svetnik (2024), arXiv:2412.09729
