"""
Conformalized Quantile Regression (CQR) for insurance pricing models.

Standard conformal prediction wraps a point forecast and produces symmetric
intervals scaled by the Tweedie variance function. That works, but it relies
entirely on the base model's variance structure being correct.

CQR (Romano, Patterson & Candès, 2019 NeurIPS) is different: you train two
quantile models directly — one for the lower tail, one for the upper — and then
use conformal calibration to correct for any systematic miscoverage. The result
is heteroscedastic intervals that adapt to the actual data distribution, not
just the model-assumed variance function.

For insurance this matters in two ways:
1. High-risk policies (large vehicles, commercial fleet, young drivers) tend to
   have heavier tails than Tweedie assumes. CQR captures this because the upper
   quantile model can learn "this feature combination always misses high".
2. Zero-inflation: personal lines motor has a mass of zero claims. A Tweedie
   model with p=1.5 cannot simultaneously fit the zero mass and the continuous
   severity well. CQR's lower quantile will naturally sit at zero for low-risk
   policies.

Algorithm (split CQR):
    1. Fit lower quantile model: q_lo_hat(x) ~ alpha_lo quantile of Y|X=x
    2. Fit upper quantile model: q_hi_hat(x) ~ alpha_hi quantile of Y|X=x
       where alpha_lo = alpha/2, alpha_hi = 1 - alpha/2 (or asymmetric)
    3. On calibration set, compute nonconformity score:
       s_i = max(q_lo_hat(x_i) - y_i, y_i - q_hi_hat(x_i))
       This is positive when y falls outside [q_lo, q_hi], negative inside.
    4. Calibration quantile: q_hat = Quantile_{(1-alpha)(1+1/n)}(s_1,...,s_n)
    5. Prediction: [q_lo_hat(x) - q_hat, q_hi_hat(x) + q_hat]

The conformal correction q_hat adjusts the raw quantile model outputs so that
the final intervals achieve marginal coverage >= 1-alpha regardless of model
misspecification.

Reference:
    Romano, Patterson & Candès (2019). "Conformalized Quantile Regression."
    Advances in Neural Information Processing Systems 32.
    https://arxiv.org/abs/1905.03222
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy, conformal_quantile


class ConformalisedQuantileRegression:
    """
    Split Conformalized Quantile Regression for insurance pricing models.

    Wraps a pair of pre-fitted quantile regression models (lower and upper
    quantile) and calibrates the correction term using a held-out calibration
    set. Produces heteroscedastic prediction intervals that are wider for
    high-variance risks and narrower for stable risks.

    Unlike ``InsuranceConformalPredictor``, CQR does not require a point
    forecast. It operates directly on quantile model outputs. This makes it
    natural for GBM quantile objectives (CatBoost ``Quantile:alpha=0.05``,
    LightGBM ``alpha=0.05`` with ``objective="quantile"``).

    Parameters
    ----------
    model_lo : fitted sklearn-compatible model
        Trained lower quantile model. Must implement ``predict(X)`` returning
        the conditional lower quantile estimate. The training quantile level
        should be ``alpha/2`` for symmetric intervals (e.g. 0.05 for 90%
        intervals).
    model_hi : fitted sklearn-compatible model
        Trained upper quantile model. Must implement ``predict(X)`` returning
        the conditional upper quantile estimate. The training quantile level
        should be ``1 - alpha/2`` (e.g. 0.95 for 90% intervals).
    clip_lower : float, default 0.0
        Clip the lower prediction interval bound at this value. Use 0.0 for
        non-negative targets (insurance claims). Set to ``-np.inf`` to disable.

    Attributes
    ----------
    cal_scores_ : np.ndarray
        CQR nonconformity scores from the calibration set. Set after
        ``calibrate()``.
    cal_quantile_ : dict[float, float]
        Cached conformal correction quantiles by alpha level.
    n_calibration_ : int
        Number of calibration observations.
    is_calibrated_ : bool
        Whether ``calibrate()`` has been called.

    Notes
    -----
    The quantile models should be trained on a disjoint training set, not the
    calibration set. Fitting quantile models on the calibration set and then
    calibrating on the same data invalidates the coverage guarantee.

    CQR produces marginal coverage guarantees, not conditional. See
    ``coverage_by_decile()`` to check whether coverage is uniform across
    risk bands.

    Examples
    --------
    Using CatBoost quantile objectives::

        from catboost import CatBoostRegressor
        from insurance_conformal import ConformalisedQuantileRegression

        lo = CatBoostRegressor(loss_function="Quantile:alpha=0.05", ...)
        hi = CatBoostRegressor(loss_function="Quantile:alpha=0.95", ...)
        lo.fit(X_train, y_train)
        hi.fit(X_train, y_train)

        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi)
        cqr.calibrate(X_cal, y_cal)
        intervals = cqr.predict_interval(X_test, alpha=0.10)

    Using sklearn GradientBoostingRegressor::

        from sklearn.ensemble import GradientBoostingRegressor
        from insurance_conformal import ConformalisedQuantileRegression

        lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, ...)
        hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, ...)
        lo.fit(X_train, y_train)
        hi.fit(X_train, y_train)

        cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi)
        cqr.calibrate(X_cal, y_cal)
        intervals = cqr.predict_interval(X_test, alpha=0.10)
    """

    def __init__(
        self,
        model_lo: Any,
        model_hi: Any,
        clip_lower: float = 0.0,
    ) -> None:
        self.model_lo = model_lo
        self.model_hi = model_hi
        self.clip_lower = float(clip_lower)

        self.cal_scores_: Optional[np.ndarray] = None
        self.cal_quantile_: dict[float, float] = {}
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

    def calibrate(
        self,
        X_cal: Any,
        y_cal: Any,
    ) -> "ConformalisedQuantileRegression":
        """
        Compute CQR nonconformity scores on a held-out calibration set.

        The calibration set must be independent of the training data used to
        fit ``model_lo`` and ``model_hi``. In a temporal insurance setting,
        train on years 1–4, calibrate on year 5, predict on year 6.

        Parameters
        ----------
        X_cal : array-like of shape (n, p)
            Calibration features. Accepts numpy arrays, pandas DataFrames, or
            polars DataFrames.
        y_cal : array-like of shape (n,)
            Observed values for the calibration period.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        y = as_numpy(y_cal)
        if len(X_cal) != len(y):
            raise ValueError(
                f"X has {len(X_cal)} samples but y has {len(y)} — lengths must match"
            )

        q_lo = self._predict_lo(X_cal)
        q_hi = self._predict_hi(X_cal)

        self.cal_scores_ = _cqr_score(y, q_lo, q_hi)
        self.n_calibration_ = len(self.cal_scores_)
        self.cal_quantile_ = {}
        self.is_calibrated_ = True

        return self

    def predict_interval(
        self,
        X_test: Any,
        alpha: float = 0.10,
    ) -> pl.DataFrame:
        """
        Produce conformalized prediction intervals for new observations.

        Returns a Polars DataFrame with columns ``lower``, ``q_lo``, ``q_hi``,
        and ``upper``. ``q_lo`` and ``q_hi`` are the raw quantile model
        outputs before the conformal correction; ``lower`` and ``upper`` are
        the corrected bounds that carry the coverage guarantee.

        Parameters
        ----------
        X_test : array-like of shape (n, p)
            Test features.
        alpha : float, default 0.10
            Miscoverage rate. ``alpha=0.10`` gives 90% prediction intervals.

        Returns
        -------
        pl.DataFrame
            Columns: lower (Float64), q_lo (Float64), q_hi (Float64),
            upper (Float64). The ``lower`` and ``upper`` columns carry the
            coverage guarantee; ``q_lo``/``q_hi`` are diagnostic.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_calibrated()

        correction = self._get_quantile(alpha)
        q_lo = self._predict_lo(X_test)
        q_hi = self._predict_hi(X_test)

        lower = np.clip(q_lo - correction, self.clip_lower, None)
        upper = q_hi + correction

        return pl.DataFrame(
            {
                "lower": lower,
                "q_lo": q_lo,
                "q_hi": q_hi,
                "upper": upper,
            }
        )

    def coverage_by_decile(
        self,
        X_test: Any,
        y_test: Any,
        alpha: float = 0.10,
        n_bins: int = 10,
    ) -> pl.DataFrame:
        """
        Compute empirical coverage by decile of the midpoint prediction.

        CQR's coverage guarantee is marginal — correct on average but
        potentially uneven across risk bands. This diagnostic reveals whether
        high-risk policies (top decile of ``(q_lo + q_hi) / 2``) are covered
        at the same rate as low-risk ones.

        Parameters
        ----------
        X_test : array-like of shape (n, p)
            Test features.
        y_test : array-like of shape (n,)
            Observed values.
        alpha : float, default 0.10
            Miscoverage rate.
        n_bins : int, default 10
            Number of equal-frequency bins.

        Returns
        -------
        pl.DataFrame
            Columns: decile (Int32), mean_midpoint (Float64), mean_width
            (Float64), n_obs (Int32), coverage (Float64),
            target_coverage (Float64).
        """
        import pandas as pd

        intervals = self.predict_interval(X_test, alpha=alpha)
        y = as_numpy(y_test)

        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        q_lo = intervals["q_lo"].to_numpy()
        q_hi = intervals["q_hi"].to_numpy()
        midpoint = (q_lo + q_hi) / 2.0
        width = upper - lower
        covered = (y >= lower) & (y <= upper)

        try:
            decile_labels = pd.qcut(midpoint, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            decile_labels = pd.cut(midpoint, bins=n_bins, labels=False, duplicates="drop")

        decile_labels = np.asarray(decile_labels, dtype=float)
        rows = []
        for d in np.unique(decile_labels[~np.isnan(decile_labels)]):
            mask = decile_labels == d
            rows.append(
                {
                    "decile": int(d) + 1,
                    "mean_midpoint": float(midpoint[mask].mean()),
                    "mean_width": float(width[mask].mean()),
                    "n_obs": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "target_coverage": float(1 - alpha),
                }
            )
        return pl.DataFrame(rows)

    def correction_quantile(self, alpha: float) -> float:
        """
        Return the conformal correction quantile for a given miscoverage rate.

        This is the value added to (subtracted from) the upper (lower) quantile
        model outputs to achieve marginal coverage ``>= 1 - alpha``. Positive
        values indicate the raw quantile models undercover; negative values
        indicate they overcover.

        Parameters
        ----------
        alpha : float
            Miscoverage rate.

        Returns
        -------
        float
            The conformal correction term ``q_hat``.
        """
        return self._get_quantile(alpha)

    def _predict_lo(self, X: Any) -> np.ndarray:
        """Run the lower quantile model and return 1D numpy array."""
        X_np = self._to_numpy(X)
        return as_numpy(self.model_lo.predict(X_np)).ravel()

    def _predict_hi(self, X: Any) -> np.ndarray:
        """Run the upper quantile model and return 1D numpy array."""
        X_np = self._to_numpy(X)
        return as_numpy(self.model_hi.predict(X_np)).ravel()

    def _to_numpy(self, X: Any) -> np.ndarray:
        """Convert DataFrame inputs to numpy."""
        import pandas as pd
        if isinstance(X, pl.DataFrame):
            return X.to_numpy()
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.asarray(X)

    def _get_quantile(self, alpha: float) -> float:
        self._check_calibrated()
        if alpha not in self.cal_quantile_:
            self.cal_quantile_[alpha] = conformal_quantile(self.cal_scores_, alpha)
        return self.cal_quantile_[alpha]

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "CQR predictor has not been calibrated. "
                "Call .calibrate(X_cal, y_cal) first."
            )

    def __repr__(self) -> str:
        status = (
            f"calibrated on {self.n_calibration_} obs"
            if self.is_calibrated_
            else "not calibrated"
        )
        return f"ConformalisedQuantileRegression({status})"


def _cqr_score(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> np.ndarray:
    """
    Compute the CQR nonconformity score for a set of observations.

    The score is ``max(q_lo - y, y - q_hi)``. It is negative when ``y`` falls
    inside the raw interval ``[q_lo, q_hi]`` (a well-covered observation) and
    positive when ``y`` falls outside (a miscovered observation). The conformal
    quantile of these scores gives the correction needed to achieve the target
    coverage level.

    Parameters
    ----------
    y : np.ndarray
        Observed values, shape (n,).
    q_lo : np.ndarray
        Lower quantile predictions, shape (n,).
    q_hi : np.ndarray
        Upper quantile predictions, shape (n,).

    Returns
    -------
    np.ndarray
        CQR nonconformity scores, shape (n,).
    """
    y = np.asarray(y, dtype=float)
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)

    if not (y.shape == q_lo.shape == q_hi.shape):
        raise ValueError(
            f"y, q_lo, q_hi must have the same shape. "
            f"Got {y.shape}, {q_lo.shape}, {q_hi.shape}."
        )

    return np.maximum(q_lo - y, y - q_hi)
