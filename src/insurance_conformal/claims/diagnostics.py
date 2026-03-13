"""
Coverage diagnostics for conformal prediction intervals on insurance claims.

These tools diagnose the gap between marginal and conditional coverage — a known
limitation of split and full conformal methods. Marginal coverage (P(Y in I) >= 1-alpha)
is guaranteed. Conditional coverage (P(Y in I | X=x) >= 1-alpha) is not, and will
fail in regions where the model is poorly calibrated or the data is sparse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def conditional_coverage_gap(
    predictor: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    groups: np.ndarray,
) -> dict:
    """Compute per-group coverage and report deviations from nominal.

    Marginal conformal prediction guarantees P(Y in I) >= 1-alpha. Conditional
    coverage — coverage within subgroups defined by covariates or policy segments —
    may be lower. This function diagnoses the conditional coverage gap.

    Parameters
    ----------
    predictor : object
        Fitted predictor with predict_interval(X, alpha) method.
    X_test : np.ndarray, shape (n_test, p)
    y_test : np.ndarray, shape (n_test,)
    alpha : float
        Miscoverage rate.
    groups : np.ndarray, shape (n_test,)
        Group labels for each test observation. Strings or integers.
        Example: policy type, accident year, vehicle class.

    Returns
    -------
    dict with keys:
        overall_coverage : float
        nominal_coverage : float
        group_results : pd.DataFrame with columns:
            group, n, coverage, width, coverage_gap
        max_gap : float — worst-case conditional coverage gap
        min_gap : float — best-case (may be positive if coverage exceeds nominal)

    Examples
    --------
    >>> groups = X_test[:, 0].astype(int)  # first covariate as group
    >>> result = conditional_coverage_gap(predictor, X_test, y_test, 0.05, groups)
    >>> result["group_results"]
    """
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    groups = np.asarray(groups)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    intervals = predictor.predict_interval(X_test, alpha=alpha)
    covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
    widths = intervals[:, 1] - intervals[:, 0]
    nominal = 1.0 - alpha

    overall_coverage = float(covered.mean())
    unique_groups = np.unique(groups)

    rows = []
    for g in unique_groups:
        mask = groups == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        cov_g = float(covered[mask].mean())
        width_g = float(widths[mask].mean())
        rows.append(
            {
                "group": g,
                "n": n_g,
                "coverage": cov_g,
                "width": width_g,
                "coverage_gap": cov_g - nominal,
            }
        )

    group_df = pd.DataFrame(rows)
    gaps = group_df["coverage_gap"].values if len(group_df) > 0 else np.array([0.0])

    return {
        "overall_coverage": overall_coverage,
        "nominal_coverage": nominal,
        "group_results": group_df,
        "max_gap": float(gaps.max()) if len(gaps) > 0 else 0.0,
        "min_gap": float(gaps.min()) if len(gaps) > 0 else 0.0,
    }


def calibration_plot_data(
    predictor: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute nominal vs empirical coverage across a grid of alpha values.

    Returns a DataFrame suitable for plotting. If matplotlib is available,
    use calibration_plot() for a ready-made figure.

    Parameters
    ----------
    predictor : object
        Fitted predictor with predict_interval(X, alpha) method.
    X_test : np.ndarray, shape (n_test, p)
    y_test : np.ndarray, shape (n_test,)
    alpha_grid : np.ndarray, optional
        Alpha values to evaluate. Default: 50 values from 0.005 to 0.50.

    Returns
    -------
    pd.DataFrame with columns: alpha, nominal_coverage, empirical_coverage,
        mean_width, coverage_gap
    """
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    if alpha_grid is None:
        alpha_grid = np.linspace(0.005, 0.50, 50)

    rows = []
    for a in alpha_grid:
        intervals = predictor.predict_interval(X_test, alpha=float(a))
        covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
        widths = intervals[:, 1] - intervals[:, 0]
        emp_cov = float(covered.mean())
        rows.append(
            {
                "alpha": float(a),
                "nominal_coverage": float(1.0 - a),
                "empirical_coverage": emp_cov,
                "mean_width": float(widths.mean()),
                "coverage_gap": emp_cov - (1.0 - a),
            }
        )

    return pd.DataFrame(rows)


def calibration_plot(
    predictor: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: np.ndarray | None = None,
    ax: Any = None,
) -> Any:
    """Plot nominal vs empirical coverage (calibration plot).

    Requires matplotlib. A well-calibrated conformal predictor should lie on
    or above the diagonal (empirical >= nominal).

    Parameters
    ----------
    predictor : object
        Fitted predictor with predict_interval(X, alpha) method.
    X_test : np.ndarray, shape (n_test, p)
    y_test : np.ndarray, shape (n_test,)
    alpha_grid : np.ndarray, optional
        Alpha values. Default: 50 values from 0.005 to 0.50.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for calibration_plot. "
            "Install it with: pip install matplotlib"
        ) from exc

    df = calibration_plot_data(predictor, X_test, y_test, alpha_grid)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        df["nominal_coverage"],
        df["empirical_coverage"],
        "b-o",
        ms=3,
        lw=1.5,
        label="Conformal predictor",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.fill_between(
        df["nominal_coverage"],
        df["nominal_coverage"],
        df["empirical_coverage"],
        where=df["empirical_coverage"] >= df["nominal_coverage"],
        alpha=0.15,
        color="green",
        label="Over-coverage (conservative)",
    )
    ax.fill_between(
        df["nominal_coverage"],
        df["nominal_coverage"],
        df["empirical_coverage"],
        where=df["empirical_coverage"] < df["nominal_coverage"],
        alpha=0.15,
        color="red",
        label="Under-coverage (invalid)",
    )
    ax.set_xlabel("Nominal coverage (1 - alpha)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal Prediction Calibration")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Mark SCR point
    ax.axvline(0.995, color="red", lw=0.8, ls=":", alpha=0.7, label="SCR (99.5%)")
    ax.axhline(0.995, color="red", lw=0.8, ls=":", alpha=0.7)

    return ax


def width_by_covariate(
    predictor: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    covariate_idx: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Interval width as a function of a covariate (heteroscedasticity diagnostic).

    A model-free predictor (HongConformal) will show roughly constant width
    per covariate. A locally weighted predictor (TwoStageLWConformal) should
    show narrower intervals in low-variance regions.

    Parameters
    ----------
    predictor : object
        Fitted predictor with predict_interval(X, alpha) method.
    X_test : np.ndarray, shape (n_test, p)
    y_test : np.ndarray, shape (n_test,)
    alpha : float
    covariate_idx : int
        Column index of the covariate to bin by.
    n_bins : int
        Number of quantile bins. Default: 10.

    Returns
    -------
    pd.DataFrame with columns: bin_centre, mean_width, mean_coverage,
        n, covariate_q_low, covariate_q_high
    """
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    intervals = predictor.predict_interval(X_test, alpha=alpha)
    covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
    widths = intervals[:, 1] - intervals[:, 0]
    covariate = X_test[:, covariate_idx]

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(covariate, quantiles)

    rows = []
    for i in range(n_bins):
        lo_edge = bin_edges[i]
        hi_edge = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (covariate >= lo_edge) & (covariate <= hi_edge)
        else:
            mask = (covariate >= lo_edge) & (covariate < hi_edge)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        rows.append(
            {
                "bin_centre": float((lo_edge + hi_edge) / 2.0),
                "mean_width": float(widths[mask].mean()),
                "mean_coverage": float(covered[mask].mean()),
                "n": n_bin,
                "covariate_q_low": float(lo_edge),
                "covariate_q_high": float(hi_edge),
            }
        )

    return pd.DataFrame(rows)
