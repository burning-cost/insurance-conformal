"""
Synthetic dataset generators for testing and demonstration.

These generators produce insurance-realistic data with known data-generating
processes (DGPs), making it possible to validate coverage guarantees empirically.

DGP for motor frequency-severity:
- Features: age, vehicle_value, no_claims_years, region (4 categories)
- Frequency: Poisson(lambda_i * exposure_i) where lambda_i = exp(beta @ x_i)
- Severity: Gamma(shape=2, scale=mu_i/2) where mu_i = exp(gamma @ x_i)
- Correlation: induced via shared latent factor on high-risk policies

DGP for home multi-peril (flood, fire, subsidence):
- 3 output dimensions with very different base rates
- Flood frequency ~0.01-0.03, fire ~0.003-0.008, subsidence ~0.001-0.003
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def make_motor_frequency_severity(
    n: int = 2000,
    n_features: int = 5,
    freq_intercept: float = -2.5,
    sev_intercept: float = 7.5,
    exposure_range: Tuple[float, float] = (0.1, 1.0),
    correlation_strength: float = 0.3,
    random_state: Optional[int] = 42,
) -> Dict[str, NDArray]:
    """
    Synthetic motor insurance dataset: correlated frequency and severity.

    Generates policies with:
    - Poisson claim frequency (lambda ~ 0.05-0.3 at full exposure)
    - Gamma claim severity (mu ~ £500-£8,000)
    - Mild positive correlation between freq and sev (shared risk factor)

    Parameters
    ----------
    n : int
        Number of policies.
    n_features : int
        Number of rating factors (features).
    freq_intercept : float
        Controls base frequency. -2.5 gives lambda ~ 0.08.
    sev_intercept : float
        Controls base severity. 7.5 gives mu ~ £1,800.
    exposure_range : (float, float)
        Min/max exposure in years.
    correlation_strength : float
        How much the shared latent factor adds to both freq and sev (0=independent).
    random_state : int or None

    Returns
    -------
    dict with:
        X : ndarray (n, n_features) — feature matrix
        freq : ndarray (n,) — observed claim count per policy
        sev : ndarray (n,) — observed claim severity (0 for zero-claim policies)
        exposure : ndarray (n,) — exposure in years
        lambda_true : ndarray (n,) — true Poisson rate (unexposed)
        mu_true : ndarray (n,) — true Gamma severity mean
        zero_claim_mask : ndarray of bool (n,) — True where freq == 0
    """
    rng = np.random.default_rng(random_state)

    # Feature matrix: standardized
    X = rng.standard_normal((n, n_features))

    # Coefficients
    freq_coef = rng.standard_normal(n_features) * 0.3
    sev_coef = rng.standard_normal(n_features) * 0.4

    # Shared latent risk factor (induces positive correlation)
    latent = rng.standard_normal(n) * correlation_strength

    # Linear predictors
    freq_eta = freq_intercept + X @ freq_coef + latent
    sev_eta = sev_intercept + X @ sev_coef + latent * 0.5

    # True rates
    lambda_true = np.exp(freq_eta)
    mu_true = np.exp(sev_eta)

    # Exposure
    exposure = rng.uniform(exposure_range[0], exposure_range[1], size=n)

    # Simulate claims
    freq = rng.poisson(lambda_true * exposure)

    # Severity: only observed for policies with claims
    # Gamma parameterisation: shape=2, scale=mu/2 gives E[C]=mu, Var[C]=mu^2/2
    sev_shape = 2.0
    sev = np.zeros(n)
    claim_mask = freq > 0
    if np.any(claim_mask):
        sev[claim_mask] = rng.gamma(
            shape=sev_shape,
            scale=mu_true[claim_mask] / sev_shape,
            size=np.sum(claim_mask),
        )

    zero_claim_mask = ~claim_mask

    return {
        "X": X,
        "freq": freq.astype(float),
        "sev": sev,
        "exposure": exposure,
        "lambda_true": lambda_true,
        "mu_true": mu_true,
        "zero_claim_mask": zero_claim_mask,
    }


def make_home_multi_peril(
    n: int = 5000,
    n_features: int = 6,
    random_state: Optional[int] = 42,
) -> Dict[str, NDArray]:
    """
    Synthetic home insurance dataset: flood, fire, subsidence (d=3).

    Base frequencies:
    - Flood: lambda_flood ~ 0.01-0.04 (geography dependent)
    - Fire: lambda_fire ~ 0.003-0.008 (relatively stable)
    - Subsidence: lambda_sub ~ 0.001-0.003 (rarest, most volatile)

    This means the zero-claim masking challenge is severe for subsidence —
    in a typical portfolio year, 99.7-99.9% of policies have no subsidence claim.

    Returns
    -------
    dict with:
        X : ndarray (n, n_features)
        freq_flood, freq_fire, freq_sub : ndarray (n,) — binary (0/1) indicators
        sev_flood, sev_fire, sev_sub : ndarray (n,) — severity (0 for no claim)
        zero_claim_mask : ndarray of bool (n, 3) — per-peril zero-claim indicators
    """
    rng = np.random.default_rng(random_state)

    X = rng.standard_normal((n, n_features))

    # Peril-specific coefficients
    coef_flood = rng.standard_normal(n_features) * 0.2
    coef_fire = rng.standard_normal(n_features) * 0.15
    coef_sub = rng.standard_normal(n_features) * 0.25

    # Log rates (Bernoulli approximation for rare events)
    lambda_flood = np.exp(-4.2 + X @ coef_flood)   # base ~0.015
    lambda_fire = np.exp(-5.5 + X @ coef_fire)      # base ~0.004
    lambda_sub = np.exp(-6.5 + X @ coef_sub)        # base ~0.0015

    # Clip rates (Poisson)
    freq_flood = rng.poisson(lambda_flood).astype(float)
    freq_fire = rng.poisson(lambda_fire).astype(float)
    freq_sub = rng.poisson(lambda_sub).astype(float)

    # Severity parameters (in £)
    mu_flood = np.exp(9.5 + X @ rng.standard_normal(n_features) * 0.3)  # ~£13k
    mu_fire = np.exp(8.5 + X @ rng.standard_normal(n_features) * 0.2)   # ~£5k
    mu_sub = np.exp(8.0 + X @ rng.standard_normal(n_features) * 0.35)   # ~£3k

    sev_flood = np.where(
        freq_flood > 0,
        rng.gamma(shape=2.0, scale=mu_flood / 2.0),
        0.0,
    )
    sev_fire = np.where(
        freq_fire > 0,
        rng.gamma(shape=2.0, scale=mu_fire / 2.0),
        0.0,
    )
    sev_sub = np.where(
        freq_sub > 0,
        rng.gamma(shape=2.0, scale=mu_sub / 2.0),
        0.0,
    )

    zero_claim_mask = np.stack([
        freq_flood == 0,
        freq_fire == 0,
        freq_sub == 0,
    ], axis=1)  # shape (n, 3)

    return {
        "X": X,
        "freq_flood": freq_flood,
        "freq_fire": freq_fire,
        "freq_sub": freq_sub,
        "sev_flood": sev_flood,
        "sev_fire": sev_fire,
        "sev_sub": sev_sub,
        "zero_claim_mask": zero_claim_mask,
        "lambda_flood": lambda_flood,
        "lambda_fire": lambda_fire,
        "lambda_sub": lambda_sub,
    }


class _SimpleLinearModel:
    """
    Trivial sklearn-compatible linear model for testing.

    Fits OLS and predicts on new data. Not a good insurance model but
    useful for constructing end-to-end tests without real GLMs.
    """

    def __init__(self, positive: bool = False):
        self.positive = positive
        self.coef_: Optional[NDArray] = None
        self.intercept_: float = 0.0

    def fit(self, X: NDArray, y: NDArray) -> "_SimpleLinearModel":
        X_ = np.column_stack([np.ones(len(X)), X])
        # OLS via pseudoinverse
        coef = np.linalg.lstsq(X_, y, rcond=None)[0]
        self.intercept_ = coef[0]
        self.coef_ = coef[1:]
        return self

    def predict(self, X: NDArray) -> NDArray:
        pred = X @ self.coef_ + self.intercept_
        if self.positive:
            pred = np.maximum(pred, 1e-8)
        return pred


class _SimpleExpModel:
    """
    Log-linear model: predicts exp(X @ coef + intercept).

    Better approximation of a Poisson GLM link function for testing.
    """

    def __init__(self):
        self.coef_: Optional[NDArray] = None
        self.intercept_: float = 0.0

    def fit(self, X: NDArray, y: NDArray) -> "_SimpleExpModel":
        y_safe = np.maximum(y, 1e-8)
        log_y = np.log(y_safe)
        X_ = np.column_stack([np.ones(len(X)), X])
        coef = np.linalg.lstsq(X_, log_y, rcond=None)[0]
        self.intercept_ = coef[0]
        self.coef_ = coef[1:]
        return self

    def predict(self, X: NDArray) -> NDArray:
        eta = X @ self.coef_ + self.intercept_
        return np.exp(eta)


def make_fitted_models(
    data: Dict[str, NDArray],
    freq_key: str = "freq",
    sev_key: str = "sev",
    use_exp_model: bool = True,
) -> Dict[str, Any]:
    """
    Fit simple models for testing purposes.

    Returns {'frequency': fitted_model, 'severity': fitted_model}.
    For severity, only uses non-zero-claim observations.
    """
    X = data["X"]
    freq = data[freq_key]
    sev = data[sev_key]

    if use_exp_model:
        freq_model = _SimpleExpModel().fit(X, np.maximum(freq, 0.01))
        sev_model = _SimpleExpModel()
    else:
        freq_model = _SimpleLinearModel(positive=True).fit(X, freq)
        sev_model = _SimpleLinearModel(positive=True)

    # Fit severity only on claim observations
    claim_mask = freq > 0
    if np.sum(claim_mask) > 5:
        sev_model.fit(X[claim_mask], sev[claim_mask])
    else:
        # Fallback: fit on all with small offset
        sev_model.fit(X, np.maximum(sev, 1.0))

    return {"frequency": freq_model, "severity": sev_model}
