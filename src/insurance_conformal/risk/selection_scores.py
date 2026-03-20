"""
Selection score functions for two-stage selective conformal risk control.

In SCRC (Xu, Guo & Wei 2025, arXiv:2512.12844), the first stage selects risks
using a selection score. Higher score = more likely to select the risk. The
threshold lambda_1 is applied such that risks with selection_score >= lambda_1
are accepted by the underwriter.

These functions convert raw model outputs (softmax probabilities, logits,
energy scores) into scalar selection scores. All return scores where higher
values indicate risks the underwriter wants to accept — low-risk, high-quality
risks. If your model convention is reversed (higher = riskier), negate the
output before passing to SelectiveConformalRC.

Each function works on 2D probability arrays (shape (n, K) where K is the
number of classes or severity bands) or 1D probability vectors for binary
problems. The functions are intentionally simple — the literature shows that
the choice of selection score matters less than the quality of the underlying
model.

Insurance context: for motor underwriting, the model outputs probabilities over
claim severity bands [0, small, medium, large, XL]. The selection score ranks
risks by how likely they are to fall in the low/small bands. MSP (maximum
softmax probability) and energy score are the two most practically useful here.
"""

from __future__ import annotations

import numpy as np


def selection_score_msp(probs: np.ndarray) -> np.ndarray:
    """
    Maximum softmax probability (MSP) selection score.

    Returns the maximum class probability for each observation. For risks
    with a clear dominant class, MSP is high — the model is confident. For
    ambiguous risks spread across severity bands, MSP is low.

    In underwriting triage: high-MSP risks are easy decisions (the model is
    confident they're either good risks or bad risks). You then apply a second
    stage to ensure the accepted easy-decisions have bounded expected loss.

    Parameters
    ----------
    probs : np.ndarray, shape (n, K) or (n,)
        Class probabilities. Each row must sum to 1 (or close to it; this is
        not enforced). For binary problems, pass the probability of the
        positive class as shape (n,) or (n, 1).

    Returns
    -------
    np.ndarray, shape (n,)
        MSP score in [0, 1]. Higher = model more confident.

    Examples
    --------
    >>> probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]])
    >>> selection_score_msp(probs)
    array([0.8 , 0.4 ])
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        return probs
    return probs.max(axis=1)


def selection_score_margin(probs: np.ndarray) -> np.ndarray:
    """
    Margin selection score (top-1 minus top-2 probability).

    The margin measures how far ahead the top class is from the runner-up.
    A large margin means the model clearly prefers one class; a small margin
    means two classes are nearly tied — an uncertain prediction.

    For binary problems the margin equals 2*p - 1 (where p is the probability
    of the positive class), which is equivalent to MSP up to a linear transform.
    The interesting case is multiclass (several severity bands).

    Parameters
    ----------
    probs : np.ndarray, shape (n, K) or (n,)
        Class probabilities. Rows should sum to 1. K >= 2 recommended.

    Returns
    -------
    np.ndarray, shape (n,)
        Margin score in [0, 1]. Higher = clearer model preference.

    Examples
    --------
    >>> probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.39, 0.21]])
    >>> selection_score_margin(probs)
    array([0.7 , 0.01])
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        # Binary: margin = |p - (1-p)| = |2p - 1|
        return np.abs(2 * probs - 1)

    # Multiclass: sort each row descending and take top1 - top2
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    if sorted_probs.shape[1] == 1:
        return sorted_probs[:, 0]
    return sorted_probs[:, 0] - sorted_probs[:, 1]


def selection_score_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Negative entropy selection score.

    Shannon entropy H(p) = -sum_k p_k * log(p_k) is maximum when the
    distribution is uniform (high uncertainty) and minimum (0) when all
    probability mass is on one class (certainty). We return the negative
    entropy so that higher values correspond to more certain, more selectable
    risks — consistent with the other score functions in this module.

    The output is normalised to [0, 1] by dividing by log(K), so the score
    is comparable across models with different numbers of output classes.

    Parameters
    ----------
    probs : np.ndarray, shape (n, K) or (n,)
        Class probabilities. For numerical stability, values < 1e-15 are
        clipped before computing log. Rows should sum to 1.

    Returns
    -------
    np.ndarray, shape (n,)
        Negative normalised entropy in [0, 1]. Higher = lower entropy = more
        certain. Score of 1.0 means one class has all probability; 0.0 means
        uniform distribution.

    Examples
    --------
    >>> probs = np.array([[0.9, 0.05, 0.05], [1/3, 1/3, 1/3]])
    >>> selection_score_entropy(probs)  # doctest: +SKIP
    array([0.613, 0.   ])
    """
    probs = np.asarray(probs, dtype=float)

    if probs.ndim == 1:
        # Binary: normalise by log(2)
        p = np.clip(probs, 1e-15, 1 - 1e-15)
        q = 1 - p
        h = -(p * np.log(p) + q * np.log(q))
        return 1.0 - h / np.log(2)

    K = probs.shape[1]
    if K == 1:
        return np.ones(probs.shape[0])

    p = np.clip(probs, 1e-15, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    # Normalise: max entropy for K classes is log(K)
    return 1.0 - h / np.log(K)


def selection_score_energy(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Energy-based selection score (Liu et al. 2020).

    The energy score is E(x; T) = -T * log(sum_k exp(f_k(x) / T)), where
    f_k are the raw logits and T is a temperature. Lower energy corresponds
    to in-distribution (known-type) risks — the model assigns high total
    probability mass. We return the negative energy so that higher scores
    mean more selectable (more in-distribution) risks.

    The energy score is more robust than MSP for out-of-distribution detection
    because it uses the full logit magnitude rather than just the maximum
    softmax probability. In underwriting terms: an unusual risk profile (one the
    model was not trained on) has high energy (low score) and should be
    referred for manual review.

    Parameters
    ----------
    logits : np.ndarray, shape (n, K) or (n,)
        Raw model outputs before softmax. Do not pass probabilities here —
        this function expects unnormalised logits.
    temperature : float, default 1.0
        Temperature for energy computation. T=1 is the standard choice.
        Higher T smooths the energy distribution.

    Returns
    -------
    np.ndarray, shape (n,)
        Negative energy. Higher = lower energy = more in-distribution risk.
        Values are in (-inf, 0] by construction (energy is always positive).

    Examples
    --------
    >>> logits = np.array([[2.0, 1.0, -1.0], [0.1, 0.0, 0.1]])
    >>> selection_score_energy(logits)
    array([-0.693..., -1.099...])
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    logits = np.asarray(logits, dtype=float)

    if logits.ndim == 1:
        logits = logits[:, np.newaxis]

    # Energy: E = -T * log(sum_k exp(f_k / T))
    # Use log-sum-exp trick for numerical stability
    scaled = logits / temperature
    # log-sum-exp: max + log(sum(exp(x - max)))
    max_logits = scaled.max(axis=1, keepdims=True)
    lse = max_logits.squeeze(1) + np.log(np.sum(np.exp(scaled - max_logits), axis=1))
    energy = -temperature * lse

    # Return negative energy so higher = more in-distribution = more selectable
    return -energy
