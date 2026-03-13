"""
Nonconformity score functions for multi-output insurance models.

Each score takes true values and predictions, both shaped (n, d) or (n,) for
a single dimension, and returns residuals shaped (n, d). The residuals are then
passed to calibration routines that standardize across dimensions.

Insurance-specific considerations:
- Poisson frequency residuals cluster near zero (lambda ~ 0.05-0.3); most
  policies have zero claims. Residuals are |0 - lambda_hat| for the majority.
- Gamma severity residuals are only meaningful for observed claims (N > 0).
  For zero-claim observations, the caller should apply a zero-claim mask.
- Tweedie combines both; treat as single-output.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def absolute_residual_score(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    |y - y_hat| per element.

    Works on any shape. For multi-output, y_true and y_pred should be (n, d).
    Returns same shape as inputs.

    This is the standard split conformal nonconformity score. It works well
    for symmetric residuals but does not account for heteroskedasticity.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_true - y_pred)


def normalized_residual_score(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    |y - y_hat| / sigma per element.

    sigma should be a local scale estimate (e.g., from a variance model or
    prediction standard error). sigma must be strictly positive.

    For insurance: sigma might come from a quantile regression on the
    calibration set, or from a local neighborhood variance estimate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive everywhere")
    return np.abs(y_true - y_pred) / sigma


def poisson_deviance_score(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    clip_pred: float = 1e-8,
) -> NDArray[np.floating]:
    """
    Poisson deviance contribution: 2 * [y*log(y/mu) - (y - mu)].

    For y_true = 0: deviance = 2 * mu (special case, limit as y -> 0).
    For y_true > 0: standard Poisson deviance.

    This is the natural nonconformity score for Poisson frequency models —
    it matches the loss function the GLM was fitted on. Deviance residuals
    are closer to symmetric than raw residuals for Poisson data.

    clip_pred: minimum value for y_pred to avoid log(0).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, clip_pred)

    # Compute deviance element-wise
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(y_true > 0, np.log(y_true / y_pred), 0.0)
    deviance = 2.0 * (y_true * log_ratio - (y_true - y_pred))
    return np.maximum(deviance, 0.0)  # numerical safety


def gamma_deviance_score(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    clip_pred: float = 1e-8,
    clip_true: float = 1e-8,
) -> NDArray[np.floating]:
    """
    Gamma deviance contribution: 2 * [log(mu/y) + (y - mu)/mu].

    This is the natural score for Gamma severity models. Gamma deviance
    penalises relative errors, appropriate when severity spans orders of
    magnitude (e.g., £200 fender-bender vs £20,000 total loss).

    For zero-claim observations, severity is unobserved; apply zero-claim
    masking before using this score.

    clip_pred, clip_true: minimum values to avoid log(0) / division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, clip_pred)
    y_true = np.maximum(y_true, clip_true)

    deviance = 2.0 * (np.log(y_pred / y_true) + (y_true - y_pred) / y_pred)
    return np.maximum(deviance, 0.0)


def tweedie_deviance_score(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    p: float,
    clip_pred: float = 1e-8,
) -> NDArray[np.floating]:
    """
    Tweedie deviance contribution for power parameter p.

    p=0: Normal (MSE). p=1: Poisson. p=2: Gamma. p in (1,2): compound Poisson-Gamma.

    The compound Poisson-Gamma case (p ~ 1.5) is the standard Tweedie used
    for pure premium modelling in motor/home insurance.

    For the multi-output library, Tweedie applies when a single combined
    model is used instead of separate frequency and severity models. In that
    case the library reduces to single-output conformal (use
    insurance-conformal for this case).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, clip_pred)

    if p == 0:
        # Normal
        return (y_true - y_pred) ** 2
    elif p == 1:
        return poisson_deviance_score(y_true, y_pred, clip_pred=clip_pred)
    elif p == 2:
        return gamma_deviance_score(y_true, y_pred, clip_pred=clip_pred)
    else:
        # General Tweedie
        # D(y, mu) = 2 * [y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p)]
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = np.where(
                y_true > 0,
                y_true ** (2 - p) / ((1 - p) * (2 - p)),
                0.0,
            )
        term2 = y_true * y_pred ** (1 - p) / (1 - p)
        term3 = y_pred ** (2 - p) / (2 - p)
        deviance = 2.0 * (term1 - term2 + term3)
        return np.maximum(deviance, 0.0)
