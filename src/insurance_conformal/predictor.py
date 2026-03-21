"""
InsuranceConformalPredictor: split conformal prediction for insurance pricing models.

Split conformal is the right approach here. Cross-conformal and full conformal are
more statistically efficient but require refitting the model on each calibration
fold - unacceptable for GBMs that take hours to train. Split conformal trains once,
calibrates once, and gives finite-sample marginal coverage guarantees.

The guarantee: P(y_test in [lower, upper]) >= 1 - alpha for exchangeable data.
This is distribution-free - no parametric assumptions on the error distribution.
The catch: it's a marginal guarantee, not conditional. See coverage_by_decile()
for checking whether coverage is uniform across risk deciles.

Academic foundation:
    Hong (2025, arXiv:2503.03659) and Hong (2026, arXiv:2601.21153) establish
    the theoretical case for applying conformal prediction to insurance claims
    in the regression setting, including the Solvency II finite-sample validity
    framing. This module is the implementation that those papers do not provide.
    The pearson_weighted non-conformity score is validated in Manna et al.
    (2025, arXiv:2507.06921) for Tweedie models.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from insurance_conformal.scores import NonconformityScore, compute_score, invert_score
from insurance_conformal.utils import (
    as_numpy,
    conformal_quantile,
    extract_tweedie_power,
    apply_exposure,
)


class InsuranceConformalPredictor:
    """
    Distribution-free prediction intervals for insurance pricing models.

    Wraps any fitted sklearn-compatible model and produces calibrated
    prediction intervals using split conformal prediction. The key design
    choice is the non-conformity score: for Tweedie/Poisson models, use
    "pearson_weighted" (the default), not "raw".

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Must implement predict(X). For CatBoost offset models, ensure
        predict() returns expected counts (i.e., includes the exposure offset).
    nonconformity : str, default "pearson_weighted"
        Non-conformity score type. One of:
        - "raw": |y - yhat| (not recommended for insurance data)
        - "pearson": |y - yhat| / sqrt(yhat) (Poisson models)
        - "pearson_weighted": |y - yhat| / yhat^(p/2) (Tweedie, DEFAULT)
        - "deviance": deviance residual (exact, but slower to invert)
        - "anscombe": Anscombe variance-stabilising residual
    distribution : str, default "tweedie"
        Distribution family. One of "poisson", "gamma", "tweedie".
        Used for deviance and anscombe scores, and for the default
        Tweedie power when auto-detection fails.
    tweedie_power : float or None, default None
        Tweedie variance power parameter. If None, attempts to auto-detect
        from the model (works for CatBoost and sklearn TweedieRegressor).
        If detection fails, defaults to 1.5.
    calibration_frac : float, default 0.2
        Fraction of training data to hold out for calibration, when using
        calibrate_from_train(). Not used when calibrate(X_cal, y_cal) is
        called directly with explicit calibration data.

    Attributes
    ----------
    cal_scores_ : np.ndarray
        Non-conformity scores from the calibration set. Set after calibrate().
    cal_quantiles_ : dict
        Cached quantiles by alpha level.
    n_calibration_ : int
        Number of calibration observations.
    is_calibrated_ : bool
        Whether the predictor has been calibrated.

    Examples
    --------
    >>> from insurance_conformal import InsuranceConformalPredictor
    >>> cp = InsuranceConformalPredictor(
    ...     model=fitted_catboost,
    ...     nonconformity="pearson_weighted",
    ...     distribution="tweedie",
    ... )
    >>> cp.calibrate(X_cal, y_cal)
    >>> intervals = cp.predict_interval(X_test, alpha=0.10)
    >>> print(intervals.head())
    """

    def __init__(
        self,
        model: Any,
        nonconformity: str = "pearson_weighted",
        distribution: str = "tweedie",
        tweedie_power: Optional[float] = None,
        calibration_frac: float = 0.2,
    ) -> None:
        # Validate inputs early
        try:
            NonconformityScore(nonconformity)
        except ValueError:
            valid = [s.value for s in NonconformityScore]
            raise ValueError(
                f"Unknown nonconformity score '{nonconformity}'. Choose from: {valid}"
            )

        if distribution.lower() not in ("poisson", "gamma", "tweedie"):
            raise ValueError(
                f"Unknown distribution '{distribution}'. "
                "Choose from: 'poisson', 'gamma', 'tweedie'."
            )

        if not 0 < calibration_frac < 1:
            raise ValueError(f"calibration_frac must be in (0, 1), got {calibration_frac}")

        self.model = model
        self.nonconformity = nonconformity
        self.distribution = distribution.lower()
        self.calibration_frac = calibration_frac

        # Resolve Tweedie power
        if tweedie_power is not None:
            self.tweedie_power = float(tweedie_power)
        else:
            detected = extract_tweedie_power(model)
            if detected is not None:
                self.tweedie_power = detected
            else:
                self.tweedie_power = 1.5
                if nonconformity in ("pearson_weighted", "deviance", "anscombe"):
                    warnings.warn(
                        "Could not auto-detect Tweedie power from model. "
                        "Defaulting to p=1.5. Pass tweedie_power= explicitly if this is wrong.",
                        UserWarning,
                        stacklevel=2,
                    )

        # State set during calibration
        self.cal_scores_: Optional[np.ndarray] = None
        self.cal_quantiles_: dict[float, float] = {}
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

    def calibrate(
        self,
        X_cal: Any,
        y_cal: Any,
        exposure: Optional[Any] = None,
    ) -> "InsuranceConformalPredictor":
        """
        Compute and store non-conformity scores on a held-out calibration set.

        The calibration set must be independent of both the training data used
        to fit the model and the test data you will predict on. In insurance,
        the natural split is temporal: train on years 1-4, calibrate on year 5,
        predict on year 6.

        Parameters
        ----------
        X_cal : array-like of shape (n, p)
            Calibration features. Accepts numpy arrays, pandas DataFrames,
            or polars DataFrames.
        y_cal : array-like of shape (n,)
            Observed losses/claims for the calibration period. Accepts numpy,
            pandas Series, or polars Series.
        exposure : array-like of shape (n,) or None
            Exposure for each calibration observation. Used for validation only
            - the model's predict() should already incorporate the exposure offset.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        y_cal = as_numpy(y_cal)
        yhat_cal = self._predict(X_cal)
        y_cal, yhat_cal = apply_exposure(y_cal, yhat_cal, exposure)

        self.cal_scores_ = compute_score(
            y=y_cal,
            yhat=yhat_cal,
            nonconformity=self.nonconformity,
            distribution=self.distribution,
            tweedie_power=self.tweedie_power,
        )
        self.n_calibration_ = len(self.cal_scores_)
        self.cal_quantiles_ = {}  # Clear cache when re-calibrating
        self.is_calibrated_ = True

        return self

    def _predict(self, X: Any) -> np.ndarray:
        """Call model.predict() and return as 1D numpy array."""
        # Convert polars to numpy before passing to sklearn-compatible models
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        yhat = self.model.predict(X)
        return as_numpy(yhat).ravel()

    def _get_quantile(self, alpha: float) -> float:
        """Get (or compute and cache) the calibration quantile for a given alpha."""
        self._check_calibrated()
        if alpha not in self.cal_quantiles_:
            self.cal_quantiles_[alpha] = conformal_quantile(self.cal_scores_, alpha)
        return self.cal_quantiles_[alpha]

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Predictor has not been calibrated. Call .calibrate(X_cal, y_cal) first."
            )

    def predict_interval(
        self,
        X_test: Any,
        alpha: float = 0.10,
    ) -> pl.DataFrame:
        """
        Produce prediction intervals for new observations.

        Returns a Polars DataFrame with columns "lower", "point", "upper".
        Lower bound is clipped at 0 since insurance losses are non-negative.

        Parameters
        ----------
        X_test : array-like of shape (n, p)
            Test features. Accepts numpy arrays, pandas DataFrames, or
            polars DataFrames.
        alpha : float, default 0.10
            Miscoverage rate. alpha=0.10 gives 90% prediction intervals.

        Returns
        -------
        pl.DataFrame
            Columns: lower (Float64), point (Float64), upper (Float64).
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        quantile = self._get_quantile(alpha)
        yhat = self._predict(X_test)

        lower, upper = invert_score(
            yhat=yhat,
            quantile=quantile,
            nonconformity=self.nonconformity,
            distribution=self.distribution,
            tweedie_power=self.tweedie_power,
            clip_lower=0.0,
        )

        return pl.DataFrame(
            {
                "lower": lower,
                "point": yhat,
                "upper": upper,
            }
        )

    def predict(self, X_test: Any) -> np.ndarray:
        """
        Return point predictions from the wrapped model.

        Parameters
        ----------
        X_test : array-like of shape (n, p)
            Test features.

        Returns
        -------
        np.ndarray
            Point predictions.
        """
        return self._predict(X_test)

    def coverage_by_decile(
        self,
        X_test: Any,
        y_test: Any,
        alpha: float = 0.10,
        n_bins: int = 10,
    ) -> pl.DataFrame:
        """
        Compute empirical coverage by decile of predicted value.

        This is the key insurance diagnostic. A model may have correct
        marginal coverage (say 90%) while being systematically miscovering
        for high-risk policies (say 65% coverage in the top decile). This
        method exposes that.

        Uniform coverage across deciles indicates the non-conformity score
        is well-matched to the heteroscedasticity of the data. If coverage
        is low in the tails, switch from "pearson" to "pearson_weighted"
        or "deviance".

        Parameters
        ----------
        X_test : array-like of shape (n, p)
            Test features.
        y_test : array-like of shape (n,)
            Observed values.
        alpha : float, default 0.10
            Miscoverage rate.
        n_bins : int, default 10
            Number of equal-frequency bins (deciles by default).

        Returns
        -------
        pl.DataFrame
            Columns: decile (Int32), mean_predicted (Float64),
            n_obs (Int32), coverage (Float64), target_coverage (Float64).
        """
        intervals = self.predict_interval(X_test, alpha=alpha)
        y = as_numpy(y_test)
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        covered = (y >= lower) & (y <= upper)
        yhat = intervals["point"].to_numpy()

        # Assign deciles by predicted value using pandas (pd.qcut is convenient)
        try:
            decile_labels = pd.qcut(yhat, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            decile_labels = pd.cut(yhat, bins=n_bins, labels=False, duplicates="drop")

        results = []
        unique_deciles = np.unique(decile_labels[~np.isnan(decile_labels.astype(float))])
        for d in unique_deciles:
            mask = decile_labels == d
            results.append(
                {
                    "decile": int(d) + 1,
                    "mean_predicted": float(yhat[mask].mean()),
                    "n_obs": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "target_coverage": float(1 - alpha),
                }
            )

        return pl.DataFrame(results)

    def summary(
        self,
        X_test: Any,
        y_test: Any,
        alpha: float = 0.10,
    ) -> None:
        """
        Print a summary of coverage and interval width diagnostics.

        Parameters
        ----------
        X_test : array-like of shape (n, p)
            Test features.
        y_test : array-like of shape (n,)
            Observed values.
        alpha : float, default 0.10
            Miscoverage rate.
        """
        intervals = self.predict_interval(X_test, alpha=alpha)
        y = as_numpy(y_test)

        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        covered = (y >= lower) & (y <= upper)
        widths = upper - lower

        print(f"InsuranceConformalPredictor summary")
        print(f"  Nonconformity score : {self.nonconformity}")
        print(f"  Distribution        : {self.distribution}")
        print(f"  Tweedie power       : {self.tweedie_power}")
        print(f"  Calibration n       : {self.n_calibration_}")
        print(f"  Target coverage     : {1 - alpha:.1%} (alpha={alpha})")
        print(f"  Marginal coverage   : {covered.mean():.3%}")
        print(f"  Mean interval width : {widths.mean():.4f}")
        print(f"  Median width        : {np.median(widths):.4f}")
        print(f"  Width 90th pct      : {np.percentile(widths, 90):.4f}")
        print()
        print("  Coverage by decile:")
        decile_df = self.coverage_by_decile(X_test, y_test, alpha=alpha)
        for row in decile_df.iter_rows(named=True):
            bar = "#" * int(row["coverage"] * 20)
            flag = " *" if abs(row["coverage"] - (1 - alpha)) > 0.05 else ""
            print(
                f"    Decile {row['decile']:2d} "
                f"(mean_pred={row['mean_predicted']:.3f}): "
                f"{row['coverage']:.1%} [{bar:<20}]{flag}"
            )
        coverages = decile_df["coverage"].to_numpy()
        if any(abs(coverages - (1 - alpha)) > 0.05):
            print()
            print(
                "  * Deciles flagged with * have coverage more than 5pp from target. "
                "Consider switching to 'pearson_weighted' or 'deviance' score."
            )

    def __repr__(self) -> str:
        status = f"calibrated on {self.n_calibration_} obs" if self.is_calibrated_ else "not calibrated"
        return (
            f"InsuranceConformalPredictor("
            f"nonconformity='{self.nonconformity}', "
            f"distribution='{self.distribution}', "
            f"tweedie_power={self.tweedie_power}, "
            f"{status})"
        )
