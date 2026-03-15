"""
Utility functions for insurance-conformal.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import polars as pl


def extract_tweedie_power(model: Any) -> Optional[float]:
    """
    Attempt to extract the Tweedie power parameter from a fitted model.

    Supports CatBoost and sklearn TweedieRegressor. Returns None if
    the parameter cannot be determined automatically.

    Parameters
    ----------
    model : fitted model
        A fitted sklearn-compatible model.

    Returns
    -------
    float or None
        The Tweedie power parameter, or None if not detectable.
    """
    # CatBoost - read from loss_function string
    try:
        import catboost

        if isinstance(model, (catboost.CatBoostRegressor, catboost.CatBoost)):
            params = model.get_params()
            loss = params.get("loss_function", "")
            if loss == "Poisson":
                return 1.0
            elif loss == "MAPE":
                return None
            elif "Tweedie" in loss:
                # e.g. "Tweedie:variance_power=1.5"
                if "variance_power=" in loss:
                    try:
                        return float(loss.split("variance_power=")[1].split(":")[0])
                    except (IndexError, ValueError):
                        return 1.5
                return 1.5
            elif loss == "Gamma":
                return 2.0
    except ImportError:
        pass

    # sklearn TweedieRegressor
    try:
        from sklearn.linear_model import TweedieRegressor

        if isinstance(model, TweedieRegressor):
            return float(model.power)
    except ImportError:
        pass

    # Wrapped model (e.g. Pipeline) - try the last step
    try:
        if hasattr(model, "steps"):
            return extract_tweedie_power(model.steps[-1][1])
    except Exception:
        pass

    return None


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the split conformal quantile from calibration scores.

    For n calibration samples, the conformal quantile is the
    ceil((n+1)(1-alpha))/n quantile of the calibration scores. This
    guarantees marginal coverage >= 1-alpha for any exchangeable data.

    This is the finite-sample correction from Venn prediction / split conformal
    prediction - the (n+1) rather than n in the numerator is essential.

    Parameters
    ----------
    scores : np.ndarray
        Non-conformity scores from the calibration set.
    alpha : float
        Miscoverage rate. Must be in (0, 1).

    Returns
    -------
    float
        The calibration quantile.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute quantile from empty calibration set.")

    # The ceil((n+1)(1-alpha))/n quantile
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def apply_exposure(
    y: np.ndarray,
    yhat: np.ndarray,
    exposure: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply exposure offset to observed and predicted values.

    Divides both y and yhat by exposure to produce annualised (per-unit)
    values. This is the correct adjustment for frequency models where:
    - y is a count (e.g. 1 claim in 6 months of cover)
    - yhat is an expected count incorporating the exposure offset
    - The nonconformity score should operate on annualised rates

    For example, a policy with 0.5 years of exposure and 1 claim should
    be treated as a 2-claim-per-year risk, not a 1-claim-per-half-year risk.
    Dividing by exposure annualises both y and yhat consistently.

    Parameters
    ----------
    y : np.ndarray
        Observed values (counts or amounts, not yet annualised).
    yhat : np.ndarray
        Predicted values from model.predict() (incorporating exposure).
    exposure : np.ndarray or None
        Exposure weights (e.g. years of cover). If None, no adjustment is made.

    Returns
    -------
    y_adj, yhat_adj : both as np.ndarray
        Annualised values (divided by exposure).
    """
    if exposure is None:
        return y, yhat

    exposure = np.asarray(exposure, dtype=float)
    if np.any(exposure <= 0):
        raise ValueError("All exposure values must be positive.")

    return y / exposure, yhat / exposure


def as_numpy(x: Any) -> np.ndarray:
    """
    Convert pandas Series/DataFrame, Polars Series/DataFrame, or list to
    a numpy array.

    For polars inputs, uses .to_numpy() which is a zero-copy operation where
    possible. For pandas, uses .to_numpy(). Lists and numpy arrays pass
    through np.asarray() as before.
    """
    if isinstance(x, (pl.Series, pl.DataFrame)):
        return x.to_numpy()
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    return np.asarray(x)


def temporal_split(
    X: Any,
    y: Any,
    calibration_frac: float,
    date_col: Optional[Union[str, np.ndarray]] = None,
    exposure: Optional[Any] = None,
) -> tuple:
    """
    Split data into training and calibration sets.

    If date_col is provided, the calibration set is the most recent
    calibration_frac of observations (by date). This matches the insurance
    use-case where you calibrate on recent data to capture recent loss trends.

    If date_col is None, uses a simple tail split (last calibration_frac rows).

    Accepts both Polars and Pandas DataFrames for X. Internally converts to
    pandas for indexing, since sklearn estimators expect pandas or numpy at
    the fit boundary.

    Parameters
    ----------
    X : array-like, pl.DataFrame, or pd.DataFrame
        Features.
    y : array-like
        Target.
    calibration_frac : float
        Fraction of data to use as calibration set.
    date_col : str or array-like, optional
        Date column name (if X is a DataFrame) or array of dates.
        If provided, calibration uses the most recent observations.
    exposure : array-like, optional
        Exposure weights.

    Returns
    -------
    X_train, X_cal, y_train, y_cal, exp_train, exp_cal
    """
    # Normalise X to pandas for uniform iloc-based indexing
    if isinstance(X, pl.DataFrame):
        X_pd = X.to_pandas()
    elif isinstance(X, pd.DataFrame):
        X_pd = X
    else:
        X_pd = pd.DataFrame(X)

    y = as_numpy(y)
    n = len(y)
    cal_n = max(1, int(np.ceil(n * calibration_frac)))

    if date_col is not None:
        if isinstance(date_col, str):
            dates = X_pd[date_col].to_numpy()
        else:
            dates = as_numpy(date_col)

        order = np.argsort(dates)
        train_idx = order[:-cal_n]
        cal_idx = order[-cal_n:]
    else:
        # Simple tail split - assumes data is in temporal order
        train_idx = np.arange(n - cal_n)
        cal_idx = np.arange(n - cal_n, n)

    X_train = X_pd.iloc[train_idx]
    X_cal = X_pd.iloc[cal_idx]
    y_train = y[train_idx]
    y_cal = y[cal_idx]

    if exposure is not None:
        exp = as_numpy(exposure)
        exp_train = exp[train_idx]
        exp_cal = exp[cal_idx]
    else:
        exp_train = exp_cal = None

    return X_train, X_cal, y_train, y_cal, exp_train, exp_cal
