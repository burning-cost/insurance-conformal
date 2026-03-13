"""
Insurance-specific loss functions for conformal risk control.

All loss functions here must satisfy the two conditions in Angelopoulos et al. (2024):
1. Non-increasing in lambda: larger lambda = more conservative = lower loss.
2. Bounded: loss in [0, B] for known B.

The normalisation choices matter. shortfall_loss divides by premium rather than
by the upper bound, which gives the financially meaningful quantity: 'what fraction
of expected premium income is lost to underpriced policies?' This is directly
comparable to a loss ratio target.

For width_loss, the bound B depends on the scale of your predictions. Pass
scale=max_expected_width to keep B=1 interpretable.
"""

from __future__ import annotations

import numpy as np


def shortfall_loss(
    y: np.ndarray,
    upper: np.ndarray,
    premium: np.ndarray,
) -> np.ndarray:
    """
    Normalised premium shortfall loss.

    L(y, upper, premium) = max(y - upper, 0) / premium

    Bounded by max(y/premium) in the calibration set. For claims bounded
    by a policy limit (which should always be true), this is well-defined.

    The natural use: y is the actual claim, upper is the predicted upper bound
    (or the premium itself), premium is the charged premium. The loss measures
    what fraction of premium is unrecovered by the upper bound.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Observed claims.
    upper : np.ndarray, shape (n,)
        Upper prediction bounds (or premiums, for premium sufficiency).
    premium : np.ndarray, shape (n,)
        Charged premiums. Must be strictly positive.

    Returns
    -------
    np.ndarray, shape (n,)
        Per-policy normalised shortfall. Zero when y <= upper.
    """
    y = np.asarray(y, dtype=float)
    upper = np.asarray(upper, dtype=float)
    premium = np.asarray(premium, dtype=float)

    if np.any(premium <= 0):
        raise ValueError("premium must be strictly positive (found non-positive values)")

    return np.maximum(y - upper, 0.0) / premium


def scaled_shortfall_loss(
    y: np.ndarray,
    lambda_: float,
    base_premium: np.ndarray,
) -> np.ndarray:
    """
    Shortfall loss when lambda scales the base premium.

    upper_i(lambda) = lambda * base_premium_i

    As lambda increases, the upper bound expands and the shortfall decreases.
    This is the monotone parameterisation used by PremiumSufficiencyController.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Observed claims.
    lambda_ : float
        Scaling factor applied to base_premium. lambda=1 means no uplift;
        lambda=1.5 means a 50% safety loading on the predicted premium.
    base_premium : np.ndarray, shape (n,)
        Model-predicted pure premiums. Must be strictly positive.

    Returns
    -------
    np.ndarray, shape (n,)
    """
    upper = lambda_ * np.asarray(base_premium, dtype=float)
    return shortfall_loss(y, upper, base_premium)


def coverage_loss(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    0-1 miscoverage indicator loss.

    L(y, [lower, upper]) = 1{y < lower or y > upper}

    This recovers standard conformal prediction as a special case of CRC
    when B=1. Useful for checking that CRC with this loss gives the same
    lambda as standard conformal (sanity check).

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    lower : np.ndarray, shape (n,)
    upper : np.ndarray, shape (n,)

    Returns
    -------
    np.ndarray, shape (n,)
        1.0 if y outside interval, 0.0 otherwise.
    """
    y = np.asarray(y, dtype=float)
    return ((y < lower) | (y > upper)).astype(float)


def interval_width_loss(
    lower: np.ndarray,
    upper: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Normalised interval width loss.

    L = (upper - lower) / scale

    Bounded by max(upper - lower) / scale. Set scale to the maximum
    plausible interval width to keep the loss in [0, 1].

    Parameters
    ----------
    lower : np.ndarray, shape (n,)
    upper : np.ndarray, shape (n,)
    scale : float, default 1.0
        Normalisation constant. Set to the largest expected interval width.

    Returns
    -------
    np.ndarray, shape (n,)
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return (np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)) / scale


def xl_recovery_loss(
    y: np.ndarray,
    excess: float,
    limit: float,
    estimated_recovery: np.ndarray,
) -> np.ndarray:
    """
    Excess-of-loss reinsurance recovery shortfall.

    For a per-risk XL treaty with excess point `excess` and limit `limit`,
    the actual recovery on claim y is:

        recovery(y) = min(max(y - excess, 0), limit)

    The loss is the shortfall of actual recovery above the estimated recovery,
    normalised by the limit:

        L = max(recovery(y) - estimated_recovery, 0) / limit

    Bounded by 1.0 (cannot overshoot by more than the limit).

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Observed gross claims.
    excess : float
        Excess (attachment) point of the XL layer.
    limit : float
        Maximum recovery under the XL layer.
    estimated_recovery : np.ndarray, shape (n,)
        Model's estimated recovery for each risk.

    Returns
    -------
    np.ndarray, shape (n,)
    """
    if excess < 0:
        raise ValueError(f"excess must be non-negative, got {excess}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    y = np.asarray(y, dtype=float)
    actual_recovery = np.minimum(np.maximum(y - excess, 0.0), limit)
    return np.maximum(actual_recovery - np.asarray(estimated_recovery, dtype=float), 0.0) / limit


def exposure_weighted_mean(
    losses: np.ndarray,
    exposure: np.ndarray,
) -> float:
    """
    Exposure-weighted mean loss.

    For portfolio-level risk control, weighting by exposure ensures that
    a policy with 0.5 years' exposure contributes half as much to the
    empirical risk estimate as one with a full year.

    Parameters
    ----------
    losses : np.ndarray, shape (n,)
    exposure : np.ndarray, shape (n,)
        Must be strictly positive.

    Returns
    -------
    float
        Exposure-weighted mean loss.
    """
    exposure = np.asarray(exposure, dtype=float)
    if np.any(exposure <= 0):
        raise ValueError("exposure must be strictly positive")
    weights = exposure / exposure.sum()
    return float(np.dot(weights, np.asarray(losses, dtype=float)))
