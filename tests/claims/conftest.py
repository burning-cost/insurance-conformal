"""
Shared fixtures for insurance-conformal-claims tests.

Synthetic data follows a Tweedie compound Poisson-Gamma DGP:
    lambda = exp(beta_0 + X @ beta_lam)    (claim frequency)
    gamma_mean = exp(beta_0 + X @ beta_sev) (claim severity)
    Y = sum_{k=1}^{N} G_k  where N ~ Poisson(lambda), G_k ~ Gamma
"""

from __future__ import annotations

import numpy as np
import pytest


def make_tweedie_data(
    n: int = 1000,
    p: int = 5,
    seed: int = 42,
    p_tweedie: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Tweedie compound Poisson-Gamma data.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of covariates.
    seed : int
    p_tweedie : float
        Tweedie power (1 < p < 2 for compound Poisson-Gamma).

    Returns
    -------
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,) — non-negative responses
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 0] = np.abs(X[:, 0])  # one positive covariate (e.g. exposure)

    # Log-linear mean
    beta = rng.normal(size=p) * 0.3
    log_mu = 1.5 + X @ beta
    mu = np.exp(log_mu)

    # Compound Poisson-Gamma simulation
    # For Tweedie(p), phi=1: lambda = mu^{2-p}/(2-p), alpha_g = (2-p)/(p-1), beta_g = phi/(mu^{p-1}*(p-1))
    # Simplified: use direct np.random calls
    alpha_g = (2.0 - p_tweedie) / (p_tweedie - 1.0)
    y = np.zeros(n)
    for i in range(n):
        lam = mu[i] ** (2.0 - p_tweedie) / (2.0 - p_tweedie)
        k = rng.poisson(lam)
        if k > 0:
            scale = mu[i] ** (p_tweedie - 1.0) * (p_tweedie - 1.0)
            y[i] = rng.gamma(alpha_g * k, scale)
    return X, y


@pytest.fixture(scope="session")
def small_dataset():
    """50 training observations — fast for brute-force comparison tests."""
    X, y = make_tweedie_data(n=50, p=3, seed=0)
    return X, y


@pytest.fixture(scope="session")
def medium_dataset():
    """500 training, 200 test observations — for coverage checks."""
    X, y = make_tweedie_data(n=700, p=4, seed=1)
    return X[:500], y[:500], X[500:], y[500:]


@pytest.fixture(scope="session")
def large_dataset():
    """1500 training, 300 cal, 300 test — for TwoStageLWConformal."""
    X, y = make_tweedie_data(n=2100, p=5, seed=2)
    return X[:1500], y[:1500], X[1500:1800], y[1500:1800], X[1800:], y[1800:]
