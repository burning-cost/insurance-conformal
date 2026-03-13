"""
Shared test fixtures for insurance-multivariate-conformal.

The key fixture is a synthetic frequency-severity dataset with:
- Known DGP (so we can validate coverage against truth)
- Reasonable insurance parameters (lambda ~ 0.1, mu ~ £2000)
- Fitted simple models that approximate the true response
- Pre-split calibration and test sets
"""

import numpy as np
import pytest

from insurance_conformal.multivariate.datasets import (
    make_motor_frequency_severity,
    _SimpleExpModel,
    _SimpleLinearModel,
    make_fitted_models,
)


def _make_split_data(
    n_train: int = 400,
    n_cal: int = 300,
    n_test: int = 300,
    n_features: int = 4,
    random_state: int = 0,
):
    """
    Generate and split synthetic motor data into train/cal/test.
    Returns a namespace-like dict with all splits.
    """
    total = n_train + n_cal + n_test
    data = make_motor_frequency_severity(
        n=total,
        n_features=n_features,
        random_state=random_state,
    )

    X = data["X"]
    freq = data["freq"]
    sev = data["sev"]
    exposure = data["exposure"]
    zero_mask = data["zero_claim_mask"]

    # Split
    tr = slice(0, n_train)
    cal = slice(n_train, n_train + n_cal)
    te = slice(n_train + n_cal, None)

    # Fit simple models on training set
    freq_model = _SimpleExpModel().fit(X[tr], np.maximum(freq[tr], 0.01))

    # Severity: fit on claim obs only
    claim_tr = freq[tr] > 0
    if claim_tr.sum() > 5:
        sev_model = _SimpleExpModel().fit(X[tr][claim_tr], sev[tr][claim_tr])
    else:
        sev_model = _SimpleExpModel().fit(X[tr], np.maximum(sev[tr], 1.0))

    models = {"frequency": freq_model, "severity": sev_model}

    return {
        "models": models,
        "X_train": X[tr],
        "X_cal": X[cal],
        "X_test": X[te],
        "freq_train": freq[tr],
        "freq_cal": freq[cal],
        "freq_test": freq[te],
        "sev_train": sev[tr],
        "sev_cal": sev[cal],
        "sev_test": sev[te],
        "exposure_cal": exposure[cal],
        "exposure_test": exposure[te],
        "zero_mask_cal": zero_mask[cal],
        "zero_mask_test": zero_mask[te],
        "Y_cal": {"frequency": freq[cal], "severity": sev[cal]},
        "Y_test": {"frequency": freq[te], "severity": sev[te]},
        "n_cal": n_cal,
        "n_test": n_test,
    }


@pytest.fixture(scope="session")
def motor_data():
    """Standard motor dataset used across most tests."""
    return _make_split_data(
        n_train=500, n_cal=400, n_test=400, n_features=4, random_state=42
    )


@pytest.fixture(scope="session")
def small_motor_data():
    """Small dataset for edge-case tests (n_cal=50)."""
    return _make_split_data(
        n_train=200, n_cal=50, n_test=200, n_features=3, random_state=7
    )


@pytest.fixture(scope="session")
def large_motor_data():
    """Larger dataset for statistical coverage tests."""
    return _make_split_data(
        n_train=1000, n_cal=1000, n_test=500, n_features=5, random_state=99
    )


@pytest.fixture(scope="session")
def simple_residuals_2d():
    """Simple 2D residual matrix for algorithm unit tests."""
    rng = np.random.default_rng(1)
    # Simulate: freq residuals ~ N(0.1, 0.3^2), sev residuals ~ N(500, 200^2)
    n = 200
    freq_res = rng.normal(0.1, 0.3, n)
    sev_res = rng.normal(500, 200, n)
    residuals = np.column_stack([np.abs(freq_res), np.abs(sev_res)])
    return residuals


@pytest.fixture(scope="session")
def simple_residuals_3d():
    """3D residual matrix for multi-peril tests."""
    rng = np.random.default_rng(2)
    n = 200
    r1 = np.abs(rng.normal(0.1, 0.3, n))
    r2 = np.abs(rng.normal(500, 200, n))
    r3 = np.abs(rng.normal(50, 30, n))
    return np.column_stack([r1, r2, r3])
