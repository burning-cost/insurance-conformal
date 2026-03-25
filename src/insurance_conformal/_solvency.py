"""
Functional API for Solvency II SCR interval estimation via conformal prediction.

The SCRReport class (scr.py) handles per-risk reporting and regulatory tables
for analysts who want the full workflow in one object. This module provides a
lighter functional interface: call solvency_capital_range() with a fitted
predictor and get back a structured result you can embed in a pipeline or feed
into downstream capital aggregation code.

The distinction matters in practice. SCRReport is the right choice when you
are producing a regulatory submission with coverage validation tables and
markdown output. solvency_capital_range() is the right choice when you need
to call the SCR estimate from inside a larger modelling pipeline — a reserving
system, a reinsurance optimiser, or a stress-testing loop — and you just want
the numbers without the ceremony.

The upper bound of a conformal prediction interval at alpha=0.005 is a valid
99.5% prediction bound under the split conformal guarantee: at least 99.5% of
individual risks will have their actual loss fall below this bound, for any
exchangeable data generating process and without any distributional assumptions.
This makes it a technically rigorous, model-free complement to parametric SCR
methods (lognormal, Pareto, internal model VaR).

interval_width is the difference between upper and lower bounds. In a capital
context, it measures how much uncertainty the conformal procedure is carrying:
a narrow interval on a large upper bound means the SCR estimate is well-
determined; a wide interval means there is genuine estimation uncertainty in
the tail — the sort of uncertainty that should be disclosed to a CRO.

Exposure weighting: when individual policy exposures differ (different policy
lengths, different vehicle years, etc.), pass them via the exposure parameter.
The SCR bounds are scaled by exposure so that a policy with half the exposure
contributes proportionally less to the aggregated SCR.

Regulatory note: Solvency II Article 101 sets the SCR at the 99.5% VaR of the
basic own funds over a one-year period. Conformal prediction provides a
per-risk bound at this quantile; aggregating to portfolio level requires
additional assumptions about the dependence structure (which conformal
prediction does not model). Treat solvency_capital_range() as a model-free
validation tool for individual risk SCR components, not as a replacement for
full internal model aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from insurance_conformal.utils import as_numpy


@dataclass
class SolvencyCapitalRange:
    """
    Per-risk SCR bounds from a conformal prediction interval.

    Attributes
    ----------
    scr_estimate : np.ndarray
        SCR component per risk: max(0, upper_bound - expected_loss). Shape (n,).
        This is the excess of the tail bound over the point forecast.
    lower_bound : np.ndarray
        Lower bound of the conformal prediction interval. Shape (n,).
        Clipped at zero for non-negative insurance losses.
    upper_bound : np.ndarray
        Upper bound of the conformal prediction interval at the target
        coverage level. This is the SCR-relevant tail quantile. Shape (n,).
    interval_width : np.ndarray
        Width of the prediction interval (upper_bound - lower_bound). Shape (n,).
        Measures estimation uncertainty in the tail.
    coverage_level : float
        Target coverage probability. 0.995 for alpha=0.005 (Solvency II).
    n_risks : int
        Number of risks (observations) in the result.
    alpha : float
        Miscoverage rate used to produce the intervals.
    total_scr : float
        Sum of scr_estimate across all risks. Provided for convenience;
        does not account for dependence between risks.
    mean_interval_width : float
        Mean interval width across risks. Use this to assess whether the
        conformal intervals are well-determined across the portfolio.
    """

    scr_estimate: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    interval_width: np.ndarray
    coverage_level: float
    n_risks: int
    alpha: float
    total_scr: float
    mean_interval_width: float

    def __repr__(self) -> str:
        return (
            f"SolvencyCapitalRange("
            f"n_risks={self.n_risks}, "
            f"coverage_level={self.coverage_level:.3f}, "
            f"total_scr={self.total_scr:.4f}, "
            f"mean_interval_width={self.mean_interval_width:.4f})"
        )


def solvency_capital_range(
    predictor: Any,
    X: Any,
    alpha: float = 0.005,
    exposure: Optional[Any] = None,
) -> SolvencyCapitalRange:
    """
    Compute model-free Solvency II SCR bounds using conformal prediction.

    Returns the 99.5% (or custom alpha) prediction interval for each risk,
    with the upper bound serving as the SCR tail estimate and the interval
    width quantifying estimation uncertainty in the capital figure.

    The coverage guarantee is finite-sample and distribution-free: at least
    (1 - alpha) of individual risks will have their actual loss fall below
    upper_bound, regardless of the data generating process, provided the
    predictor was calibrated on exchangeable data.

    This function is the lightweight functional counterpart to SCRReport. Use
    SCRReport when you need coverage validation tables and markdown reporting
    for regulatory submissions. Use solvency_capital_range() when you need
    SCR estimates as part of a larger pipeline.

    Parameters
    ----------
    predictor : conformal predictor
        Any fitted and calibrated conformal predictor that implements
        ``predict_interval(X, alpha)`` returning a pl.DataFrame with at least
        ``lower``, ``point``, and ``upper`` columns. Compatible with:
        InsuranceConformalPredictor, LocallyWeightedConformal,
        HongTransformConformal, ConformalisedQuantileRegression.
    X : array-like of shape (n, p)
        Risk features for the policies to price. Accepts numpy arrays,
        pandas DataFrames, or polars DataFrames.
    alpha : float, default 0.005
        Miscoverage rate. The default 0.005 produces 99.5% intervals, which
        is the Solvency II Article 101 requirement. Use alpha=0.01 for 99%
        (Lloyd's stress test) or alpha=0.001 for 99.9% (extreme stress).
    exposure : array-like of shape (n,) or None, default None
        Policy exposure weights (e.g. years of cover, vehicle-years). When
        provided, all bounds and estimates are scaled by exposure so that a
        policy with 0.5 years of cover contributes half as much to total_scr
        as a full-year policy. Exposure values must be strictly positive.

    Returns
    -------
    SolvencyCapitalRange
        A dataclass with fields:

        - ``scr_estimate`` (np.ndarray): per-risk SCR component,
          max(0, upper_bound - expected_loss). Shape (n,).
        - ``lower_bound`` (np.ndarray): lower bound of the conformal
          interval. Shape (n,).
        - ``upper_bound`` (np.ndarray): upper bound at the target coverage
          level — the SCR tail quantile. Shape (n,).
        - ``interval_width`` (np.ndarray): upper_bound - lower_bound, a
          measure of estimation uncertainty. Shape (n,).
        - ``coverage_level`` (float): 1 - alpha.
        - ``n_risks`` (int): number of observations.
        - ``alpha`` (float): the miscoverage rate used.
        - ``total_scr`` (float): sum of scr_estimate across all risks.
        - ``mean_interval_width`` (float): mean interval width.

    Raises
    ------
    TypeError
        If predictor does not implement predict_interval(X, alpha).
    ValueError
        If alpha is not in (0, 1).
    ValueError
        If exposure contains non-positive values or has a different length
        than X.

    Examples
    --------
    Basic usage with InsuranceConformalPredictor::

        from insurance_conformal import InsuranceConformalPredictor
        from insurance_conformal import solvency_capital_range

        cp = InsuranceConformalPredictor(model=fitted_gbm, tweedie_power=1.5)
        cp.calibrate(X_cal, y_cal)

        result = solvency_capital_range(cp, X_test)
        print(result.total_scr)
        print(result.upper_bound[:5])

    With exposure scaling::

        result = solvency_capital_range(cp, X_test, exposure=years_on_cover)

    With CQR for heteroscedastic tails::

        from insurance_conformal import ConformalisedQuantileRegression
        from insurance_conformal import solvency_capital_range

        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi)
        cqr.calibrate(X_cal, y_cal)

        result = solvency_capital_range(cqr, X_test, alpha=0.005)
        # result.upper_bound carries the 99.5% coverage guarantee

    Notes
    -----
    The ``point`` column used to compute scr_estimate is extracted from the
    predict_interval output. For ConformalisedQuantileRegression, which does
    not have a ``point`` column, the midpoint (q_lo + q_hi) / 2 is used as
    the expected loss estimate.

    Conformal coverage is a marginal guarantee. Check subgroup coverage
    (coverage_by_decile, subgroup_coverage) to verify that the 99.5% bound
    holds for high-risk subgroups as well as on average.
    """
    if not hasattr(predictor, "predict_interval"):
        raise TypeError(
            "predictor must implement predict_interval(X, alpha). "
            f"Got: {type(predictor)}"
        )

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(X) if hasattr(X, "__len__") else len(as_numpy(X))

    if exposure is not None:
        exp = as_numpy(exposure).ravel()
        if len(exp) != n:
            raise ValueError(
                f"exposure length {len(exp)} does not match X length {n}"
            )
        if np.any(exp <= 0):
            raise ValueError("All exposure values must be strictly positive.")
    else:
        exp = None

    intervals = predictor.predict_interval(X, alpha=alpha)

    lower = intervals["lower"].to_numpy()
    upper = intervals["upper"].to_numpy()

    # point forecast: use "point" if present, else midpoint of quantile bounds
    if "point" in intervals.columns:
        expected = intervals["point"].to_numpy()
    elif "q_lo" in intervals.columns and "q_hi" in intervals.columns:
        expected = (intervals["q_lo"].to_numpy() + intervals["q_hi"].to_numpy()) / 2.0
    else:
        # Fall back: use midpoint of the conformally corrected bounds
        expected = (lower + upper) / 2.0

    if exp is not None:
        lower = lower * exp
        upper = upper * exp
        expected = expected * exp

    scr = np.maximum(0.0, upper - expected)
    width = upper - lower

    return SolvencyCapitalRange(
        scr_estimate=scr,
        lower_bound=lower,
        upper_bound=upper,
        interval_width=width,
        coverage_level=1.0 - alpha,
        n_risks=n,
        alpha=alpha,
        total_scr=float(scr.sum()),
        mean_interval_width=float(width.mean()),
    )
