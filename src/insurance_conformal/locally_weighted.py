"""
Locally-weighted conformal prediction for insurance models.

Standard split conformal with pearson_weighted scores adapts interval width
to the *model-implied* variance structure (yhat^p). But the true residual
variance may differ from what the Tweedie variance function predicts — either
because the model is misspecified or because there are additional covariates
that explain the residual variance.

This module fits a secondary CatBoost model on |Pearson residuals| ~ X.
The secondary model learns rho_hat(x) — how large the residuals tend to be
for observations with features x, regardless of what yhat says. Dividing by
rho_hat produces a more homogeneous score distribution, narrowing intervals
by ~24% compared to standard pearson_weighted (Manna et al. ASMBI 2025).

The two-stage formula:
    R_i_std = |y_i - f_hat(x_i)| / (f_hat(x_i)^(p/2) * rho_hat(x_i))

where rho_hat is fit on the training set (Stage 1: fit f_hat; Stage 2: fit
rho_hat on |R_i_Pearson|) and the final scores are computed on the calibration
set.

The secondary model must be fit on training data *not used for calibration*.
Fitting it on calibration data would leak information into the scores.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy, conformal_quantile


class LocallyWeightedConformal:
    """
    Two-stage conformal prediction with a secondary spread model.

    Stage 1: fit the base insurance model f_hat (passed in, already fitted).
    Stage 2: fit a secondary CatBoost model rho_hat on |Pearson residuals|
             from the training set.
    Calibration: compute locally-weighted scores on the calibration set.
    Prediction: use rho_hat(x) to produce adaptive-width intervals.

    The key design choice: use CatBoost for the secondary model, not a GLM.
    The residual spread may be a complex function of features — a GBM captures
    this better than a parametric spread model while remaining interpretable
    via SHAP. CatBoost is consistent with the rest of our stack.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        The base insurance pricing model. Must implement predict(X).
    tweedie_power : float, default 1.5
        Tweedie variance power p. Used in the Pearson score denominator
        yhat^(p/2). Set p=1.0 for Poisson, p=2.0 for Gamma.
    spread_model_params : dict, optional
        CatBoost parameters for the secondary spread model. Defaults to
        a sensible set: 300 trees, learning_rate=0.05, RMSE loss.
        Override to tune the secondary model — but be careful not to
        overfit it, since overfitting rho_hat will shrink calibration scores
        artificially and hurt coverage.
    clip_spread_at : float, default 0.01
        Minimum value for rho_hat. Prevents division-by-zero when the
        secondary model predicts near-zero spread.

    Attributes
    ----------
    spread_model_ : CatBoostRegressor
        Fitted secondary spread model. Available after fit().
    cal_scores_ : np.ndarray
        Locally-weighted non-conformity scores from calibration set.
    n_calibration_ : int
        Number of calibration observations.
    is_calibrated_ : bool
        Whether calibrate() has been called.

    Examples
    --------
    >>> lw = LocallyWeightedConformal(model=fitted_catboost, tweedie_power=1.5)
    >>> lw.fit(X_train, y_train)
    >>> lw.calibrate(X_cal, y_cal)
    >>> intervals = lw.predict_interval(X_test, alpha=0.10)
    """

    def __init__(
        self,
        model: Any,
        tweedie_power: float = 1.5,
        spread_model_params: Optional[dict] = None,
        clip_spread_at: float = 0.01,
    ) -> None:
        self.model = model
        self.tweedie_power = float(tweedie_power)
        self.clip_spread_at = float(clip_spread_at)

        default_params = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }
        if spread_model_params is not None:
            default_params.update(spread_model_params)
        self.spread_model_params = default_params

        self.spread_model_: Any = None
        self.cal_scores_: Optional[np.ndarray] = None
        self.cal_quantiles_: dict[float, float] = {}
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

    def fit(self, X_train: Any, y_train: Any) -> "LocallyWeightedConformal":
        """
        Fit the secondary spread model on training residuals.

        The base model (self.model) must already be fitted before calling
        this method. This method fits rho_hat on |Pearson residuals| computed
        from the training set.

        Do NOT pass calibration data here — the calibration set must be
        held out and used only in calibrate().

        Parameters
        ----------
        X_train : array-like
            Training features (same data used to fit the base model).
        y_train : array-like
            Training targets.

        Returns
        -------
        self
        """
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(
                "LocallyWeightedConformal requires CatBoost. "
                "Install with: uv pip install 'insurance-conformal[catboost]'"
            ) from e

        y = as_numpy(y_train)
        yhat = self._base_predict(X_train)

        # Pearson residuals on training set
        yhat_clipped = np.clip(yhat, 1e-8, None)
        pearson_resid = np.abs(y - yhat) / (yhat_clipped ** (self.tweedie_power / 2.0))

        # Fit secondary model on |Pearson residuals|
        X_np = self._to_numpy(X_train)
        self.spread_model_ = CatBoostRegressor(**self.spread_model_params)
        self.spread_model_.fit(X_np, pearson_resid)

        return self

    def calibrate(self, X_cal: Any, y_cal: Any) -> "LocallyWeightedConformal":
        """
        Compute locally-weighted non-conformity scores on the calibration set.

        Must be called after fit(). The calibration set must be independent
        of both the training data (used to fit the base and spread models)
        and the test data.

        Parameters
        ----------
        X_cal : array-like
            Calibration features.
        y_cal : array-like
            Calibration targets.

        Returns
        -------
        self
        """
        self._check_fitted()

        y = as_numpy(y_cal)
        yhat = self._base_predict(X_cal)
        rho_hat = self._spread_predict(X_cal)

        self.cal_scores_ = self._compute_lw_score(y, yhat, rho_hat)
        self.n_calibration_ = len(self.cal_scores_)
        self.cal_quantiles_ = {}
        self.is_calibrated_ = True

        return self

    def predict_interval(
        self, X_test: Any, alpha: float = 0.05
    ) -> pl.DataFrame:
        """
        Produce locally-adaptive prediction intervals.

        Interval width is proportional to both yhat^(p/2) (Tweedie variance)
        and rho_hat(x) (secondary spread model prediction). This means the
        same alpha gives narrower intervals where both the model is confident
        AND the residual spread is low for that type of risk.

        Parameters
        ----------
        X_test : array-like
            Test features.
        alpha : float, default 0.05
            Miscoverage rate. 0.05 gives 95% prediction intervals.

        Returns
        -------
        pl.DataFrame
            Columns: lower, point, upper, spread (rho_hat values).
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_calibrated()

        quantile = self._get_quantile(alpha)
        yhat = self._base_predict(X_test)
        rho_hat = self._spread_predict(X_test)

        yhat_clipped = np.clip(yhat, 1e-8, None)
        # Half-width: q * yhat^(p/2) * rho_hat
        half_width = quantile * (yhat_clipped ** (self.tweedie_power / 2.0)) * rho_hat

        lower = np.clip(yhat - half_width, 0.0, None)
        upper = yhat + half_width

        return pl.DataFrame(
            {
                "lower": lower,
                "point": yhat,
                "upper": upper,
                "spread": rho_hat,
            }
        )

    def coverage_diagnostics(
        self, X_test: Any, y_test: Any, alpha: float = 0.05
    ) -> pl.DataFrame:
        """
        Coverage by decile of predicted value and by spread quartile.

        Two slices are computed:
        1. Coverage by decile of yhat (exposes tail coverage issues).
        2. Coverage by quartile of rho_hat (exposes spread model calibration).

        Parameters
        ----------
        X_test : array-like
            Test features.
        y_test : array-like
            Observed values.
        alpha : float, default 0.05
            Miscoverage rate.

        Returns
        -------
        pl.DataFrame
            Columns: slice_type, bin, mean_predicted, mean_spread, n_obs,
            coverage, target_coverage.
        """
        import pandas as pd

        intervals = self.predict_interval(X_test, alpha=alpha)
        y = as_numpy(y_test)

        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        covered = np.asarray((y >= lower) & (y <= upper), dtype=bool)
        yhat = intervals["point"].to_numpy()
        spread = intervals["spread"].to_numpy()

        target = 1.0 - alpha
        rows = []

        # Slice 1: by decile of yhat
        try:
            decile_labels = pd.qcut(yhat, q=10, labels=False, duplicates="drop")
        except ValueError:
            decile_labels = pd.cut(yhat, bins=10, labels=False, duplicates="drop")

        decile_labels = np.asarray(decile_labels, dtype=float)
        unique_d = np.unique(decile_labels[~np.isnan(decile_labels)])
        for d in unique_d:
            mask = decile_labels == d
            rows.append(
                {
                    "slice_type": "pred_decile",
                    "bin": int(d) + 1,
                    "mean_predicted": float(yhat[mask].mean()),
                    "mean_spread": float(spread[mask].mean()),
                    "n_obs": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "target_coverage": target,
                }
            )

        # Slice 2: by quartile of spread (rho_hat)
        # Use np.percentile-based binning to avoid pandas Categorical issues
        spread_quantile_edges = np.percentile(spread, [0, 25, 50, 75, 100])
        spread_quantile_edges[-1] += 1e-10  # include right edge
        spread_bin_labels = np.digitize(spread, spread_quantile_edges[1:], right=False)
        spread_bin_labels = np.clip(spread_bin_labels, 0, 3)
        unique_s = np.arange(4)
        for s in unique_s:
            mask = spread_bin_labels == s
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "slice_type": "spread_quartile",
                    "bin": int(s) + 1,
                    "mean_predicted": float(yhat[mask].mean()),
                    "mean_spread": float(spread[mask].mean()),
                    "n_obs": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "target_coverage": target,
                }
            )

        return pl.DataFrame(rows)

    def _compute_lw_score(
        self, y: np.ndarray, yhat: np.ndarray, rho_hat: np.ndarray
    ) -> np.ndarray:
        """Compute locally-weighted nonconformity score."""
        yhat_clipped = np.clip(yhat, 1e-8, None)
        denominator = (yhat_clipped ** (self.tweedie_power / 2.0)) * rho_hat
        return np.abs(y - yhat) / denominator

    def _base_predict(self, X: Any) -> np.ndarray:
        """Run the base model and return 1D numpy."""
        X_np = self._to_numpy(X)
        return as_numpy(self.model.predict(X_np)).ravel()

    def _spread_predict(self, X: Any) -> np.ndarray:
        """Run the spread model and clip at minimum."""
        self._check_fitted()
        X_np = self._to_numpy(X)
        rho_raw = as_numpy(self.spread_model_.predict(X_np)).ravel()
        return np.clip(rho_raw, self.clip_spread_at, None)

    def _to_numpy(self, X: Any) -> np.ndarray:
        """Convert DataFrame inputs to numpy for CatBoost/sklearn."""
        import pandas as pd
        if isinstance(X, pl.DataFrame):
            return X.to_numpy()
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.asarray(X)

    def _get_quantile(self, alpha: float) -> float:
        self._check_calibrated()
        if alpha not in self.cal_quantiles_:
            self.cal_quantiles_[alpha] = conformal_quantile(self.cal_scores_, alpha)
        return self.cal_quantiles_[alpha]

    def _check_fitted(self) -> None:
        if self.spread_model_ is None:
            raise RuntimeError(
                "Spread model has not been fitted. Call .fit(X_train, y_train) first."
            )

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Predictor has not been calibrated. "
                "Call .calibrate(X_cal, y_cal) after .fit()."
            )

    def __repr__(self) -> str:
        status = (
            f"calibrated on {self.n_calibration_} obs"
            if self.is_calibrated_
            else ("fitted, not calibrated" if self.spread_model_ is not None else "not fitted")
        )
        return (
            f"LocallyWeightedConformal("
            f"tweedie_power={self.tweedie_power}, "
            f"{status})"
        )
