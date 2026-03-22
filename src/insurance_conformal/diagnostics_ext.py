"""
Extended diagnostics for conformal prediction quality.

Two problems the core CoverageDiagnostics class doesn't address:

1. ERT (Expected Residual Test) conditional coverage gap (arXiv:2512.11779):
   The standard marginal coverage guarantee says nothing about conditional
   coverage P(Y in I(X) | X=x). A predictor could have 90% marginal coverage
   while having 0% coverage on a specific subgroup. The ERT test quantifies
   the worst-case conditional coverage gap via a test statistic that has power
   against conditional miscoverage.

2. Arbitrary subgroup coverage: coverage_by_decile() bins by predicted value,
   which is the right first diagnostic. But practitioners also need coverage
   by business-relevant groupings: vehicle class, policy holder age band,
   region, claim history flag. subgroup_coverage() handles arbitrary groupings.

References:
    Angelopoulos et al. (2024) "Conformal Risk Control", arXiv:2208.02814
    ERT test: arXiv:2512.11779 (Expected Residual Test for conditional coverage)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy


def ert_coverage_gap(
    predictor: Any,
    X_test: Any,
    y_test: Any,
    alpha: float = 0.05,
    n_bins: int = 10,
    random_state: Optional[int] = None,
) -> pl.DataFrame:
    """
    ERT conditional coverage metric.

    Estimates the conditional coverage gap using a regression-based approach:
    fit a predictor for the indicator I(y in interval) ~ X and measure how
    much the conditional coverage varies across the feature space. A well-
    calibrated predictor should have coverage close to (1-alpha) everywhere.

    The method:
    1. Compute coverage indicators: Z_i = 1 if y_i in [lo_i, hi_i] else 0
    2. Bin by predicted value (as proxy for X) using n_bins bins
    3. For each bin, compute empirical coverage and SE
    4. Report: max coverage gap, mean absolute gap, and whether any bin
       fails by more than 2 SE (a simple conditional miscoverage test)

    This is a pragmatic approximation of the full ERT test from arXiv:2512.11779.
    The full test requires fitting a classifier on Z ~ X, which we avoid here
    to keep the dependency footprint small. For a production implementation
    with power against complex conditional miscoverage patterns, use a
    gradient-boosted classifier on the Z_i indicators.

    Parameters
    ----------
    predictor : conformal predictor
        Must implement predict_interval(X, alpha) returning pl.DataFrame
        with "lower" and "upper" columns.
    X_test : array-like
        Test features.
    y_test : array-like
        Observed test outcomes.
    alpha : float, default 0.05
        Miscoverage rate used to generate intervals.
    n_bins : int, default 10
        Number of bins for coverage aggregation (by predicted value).
    random_state : int, optional
        Not currently used; reserved for future bootstrap-based extensions.

    Returns
    -------
    pl.DataFrame
        Columns: bin, mean_predicted, n_obs, empirical_coverage,
        target_coverage, coverage_gap, coverage_se, exceeds_2se.
        Plus a summary row appended at the end with bin=-1.

    Examples
    --------
    >>> gap_table = ert_coverage_gap(predictor=cp, X_test=X_test,
    ...                              y_test=y_test, alpha=0.10)
    >>> print(gap_table.filter(pl.col("exceeds_2se")))
    """
    import pandas as pd

    y = as_numpy(y_test).ravel()
    intervals = predictor.predict_interval(X_test, alpha=alpha)
    lower = intervals["lower"].to_numpy()
    upper = intervals["upper"].to_numpy()
    yhat = intervals["point"].to_numpy() if "point" in intervals.columns else (lower + upper) / 2.0
    covered = np.asarray((y >= lower) & (y <= upper), dtype=float)
    target = 1.0 - alpha

    # Bin by yhat
    try:
        bin_labels = pd.qcut(yhat, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bin_labels = pd.cut(yhat, bins=n_bins, labels=False, duplicates="drop")

    rows = []
    bin_labels = np.asarray(bin_labels, dtype=float)
    unique_bins = np.unique(bin_labels[~np.isnan(bin_labels)])
    for b in unique_bins:
        mask = bin_labels == b
        n_b = int(mask.sum())
        cov_b = float(covered[mask].mean())
        se_b = float(np.sqrt(cov_b * (1 - cov_b) / max(n_b, 1)))
        gap = float(target - cov_b)

        rows.append(
            {
                "bin": int(b) + 1,
                "mean_predicted": float(yhat[mask].mean()),
                "n_obs": n_b,
                "empirical_coverage": cov_b,
                "target_coverage": target,
                "coverage_gap": gap,
                "coverage_se": se_b,
                "exceeds_2se": bool(abs(gap) > 2 * se_b),
            }
        )

    # Summary row
    gaps = np.array([r["coverage_gap"] for r in rows])
    n_exceed = sum(1 for r in rows if r["exceeds_2se"])
    rows.append(
        {
            "bin": -1,  # sentinel for summary
            "mean_predicted": float(yhat.mean()),
            "n_obs": len(y),
            "empirical_coverage": float(covered.mean()),
            "target_coverage": target,
            "coverage_gap": float(np.mean(np.abs(gaps))),  # MAE of gap
            "coverage_se": float(np.max(np.abs(gaps))),    # max gap (reused field)
            "exceeds_2se": bool(n_exceed > 0),
        }
    )

    return pl.DataFrame(rows)


def subgroup_coverage(
    predictor: Any,
    X_test: Any,
    y_test: Any,
    alpha: float,
    groups: Any,
    group_name: str = "group",
) -> pl.DataFrame:
    """
    Coverage broken down by an arbitrary grouping variable.

    Use this for business-relevant coverage diagnostics: "does the model
    cover 90% of motor claims? 90% of property claims? 90% of high-value
    policies?" These slices matter more than decile-of-predicted-value for
    communicating model validity to underwriters.

    Parameters
    ----------
    predictor : conformal predictor
        Must implement predict_interval(X, alpha).
    X_test : array-like
        Test features.
    y_test : array-like
        Observed test outcomes.
    alpha : float
        Miscoverage rate.
    groups : array-like of shape (n,)
        Group labels for each test observation. Can be strings, integers,
        or any hashable type. All unique values will be reported.
    group_name : str, default "group"
        Column name for the group label in the output DataFrame.

    Returns
    -------
    pl.DataFrame
        Columns: {group_name}, n_obs, empirical_coverage, target_coverage,
        coverage_gap, mean_lower, mean_upper, mean_width, mean_predicted.
        Ordered by group label.

    Examples
    --------
    >>> vehicle_class = df["vehicle_class"].to_numpy()
    >>> cov = subgroup_coverage(cp, X_test, y_test, alpha=0.10,
    ...                         groups=vehicle_class, group_name="vehicle_class")
    >>> print(cov)
    """
    y = as_numpy(y_test).ravel()
    groups_arr = as_numpy(groups).ravel()

    if len(groups_arr) != len(y):
        raise ValueError(
            f"groups length {len(groups_arr)} must match y_test length {len(y)}"
        )

    intervals = predictor.predict_interval(X_test, alpha=alpha)
    lower = intervals["lower"].to_numpy()
    upper = intervals["upper"].to_numpy()
    yhat = intervals["point"].to_numpy() if "point" in intervals.columns else (lower + upper) / 2.0
    covered = (y >= lower) & (y <= upper)
    widths = upper - lower
    target = 1.0 - alpha

    unique_groups = np.unique(groups_arr)
    rows = []

    for g in unique_groups:
        mask = groups_arr == g
        n_g = int(mask.sum())
        cov_g = float(covered[mask].mean())

        rows.append(
            {
                group_name: g,
                "n_obs": n_g,
                "empirical_coverage": cov_g,
                "target_coverage": target,
                "coverage_gap": float(target - cov_g),
                "mean_lower": float(lower[mask].mean()),
                "mean_upper": float(upper[mask].mean()),
                "mean_width": float(widths[mask].mean()),
                "mean_predicted": float(yhat[mask].mean()),
            }
        )

    return pl.DataFrame(rows).sort(group_name)


def width_efficiency_comparison(
    predictors: Dict[str, Any],
    X_test: Any,
    y_test: Any,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """
    Compare prediction interval efficiency across multiple predictors.

    A useful companion to coverage diagnostics: once you've confirmed all
    predictors achieve valid coverage, rank them by interval width. Narrower
    intervals at the same coverage = better.

    Parameters
    ----------
    predictors : dict of {name: predictor}
        Dictionary mapping predictor names to fitted conformal predictors.
        Each must implement predict_interval(X, alpha).
    X_test : array-like
        Test features.
    y_test : array-like
        Observed test outcomes.
    alpha : float, default 0.05
        Miscoverage rate.

    Returns
    -------
    pl.DataFrame
        Columns: predictor_name, marginal_coverage, target_coverage,
        mean_width, median_width, width_90th_pct, width_relative_to_widest.
        Sorted by mean_width ascending.

    Examples
    --------
    >>> results = width_efficiency_comparison(
    ...     {
    ...         "standard": cp_standard,
    ...         "locally_weighted": cp_lw,
    ...     },
    ...     X_test, y_test, alpha=0.10
    ... )
    >>> print(results)
    """
    y = as_numpy(y_test).ravel()
    target = 1.0 - alpha

    rows = []
    for name, pred in predictors.items():
        intervals = pred.predict_interval(X_test, alpha=alpha)
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        covered = (y >= lower) & (y <= upper)
        widths = upper - lower

        rows.append(
            {
                "predictor_name": name,
                "marginal_coverage": float(covered.mean()),
                "target_coverage": target,
                "mean_width": float(widths.mean()),
                "median_width": float(np.median(widths)),
                "width_90th_pct": float(np.percentile(widths, 90)),
            }
        )

    df = pl.DataFrame(rows).sort("mean_width")

    # Add relative width column (relative to the widest, i.e., the baseline)
    max_width = df["mean_width"].max()
    df = df.with_columns(
        (pl.col("mean_width") / max_width).alias("width_relative_to_widest")
    )

    return df
