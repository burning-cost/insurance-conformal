"""
Model-free conformal prediction for insurance data.

Two classes from the Hong (2025/2026) research programme:

1. HongModelFree (arXiv:2503.03659): No regression model required.
   Constructs W_i = Y_i + sum_j (X_new_j - X_ij) / n as a feature-adjusted
   response, then takes an order-statistic interval. Works when Y is
   approximately linear in X in expectation. For insurance data with
   log-linear structure this is an approximation, but it provides a useful
   model-free baseline when you don't trust your GBM predictions.

2. HongTransformConformal (arXiv:2601.21153): h-transform residuals.
   Given any predictor h (could be your fitted GBM, a GLM, whatever), computes
   W_i = Y_i - h(X_i) and applies conformal to the residuals W_i, then shifts
   the interval by h(X_new). This is mathematically equivalent to standard
   residual conformal but Hong's framing makes explicit that ANY h works —
   even a bad one — because the coverage guarantee is distribution-free.
   A better h just gives narrower intervals.

The practical use case: you want to audit whether your GBM-based conformal
intervals are reasonable. HongTransformConformal with the same model gives
identical intervals to InsuranceConformalPredictor with nonconformity="raw",
which makes it easy to verify your implementation and to experiment with
different h functions.

References:
    Hong, L. (2025). "Conformal prediction of future insurance claims in the
        regression problem." arXiv:2503.03659.
        https://arxiv.org/abs/2503.03659
        Updated September 2025. Introduces the model-free approach (HongModelFree)
        and provides a Solvency II framing for conformal upper bounds. The paper
        establishes that the W_i feature adjustment gives valid coverage without
        any regression model, using only the exchangeability assumption.

    Hong, L. (2026). "A new strategy for finite-sample valid prediction of future
        insurance claims in the regression setting." arXiv:2601.21153.
        https://arxiv.org/abs/2601.21153
        Extends the approach to the regression setting. The h-transform formulation
        (HongTransformConformal) shows that coverage holds for ANY predictor h,
        making explicit what is implicit in standard residual conformal: the model
        quality only affects interval width, not validity.

    Graziadei, H., Janett, C., Embrechts, P. & Bucher, A. (2023, updated June 2025).
        "Conformal prediction for frequency-severity modeling." arXiv:2307.13124.
        https://arxiv.org/abs/2307.13124
        Related insurance application; complementary to Hong's regression framing.
        See insurance_conformal.claims.FrequencySeverityConformal.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy, conformal_quantile


class HongModelFree:
    """
    Model-free conformal prediction intervals.

    No regression model is needed. The approach uses a feature adjustment
    to the response: W_i = Y_i + sum_j (X_new_j - X_ij) / n, where the
    sum is over training features dimension-by-dimension. The interval is
    a one-sided upper bound (0 to k-th order statistic) clipped at 0.

    This is most appropriate when:
    - You don't have a fitted model (early exploratory phase)
    - You want a sanity check against your model-based intervals
    - The relationship between X and Y is approximately additive in logs

    Warning: the feature adjustment assumes Y ~ linear(X) in expectation.
    For insurance data where E[Y] = exp(X @ beta), this is a first-order
    Taylor approximation and will not be exact. Treat this as a benchmark,
    not a production interval.

    Parameters
    ----------
    clip_lower : float, default 0.0
        Clip interval lower bound at this value. Claims are non-negative.

    Examples
    --------
    >>> hmf = HongModelFree()
    >>> hmf.fit(X_train, y_train)
    >>> intervals = hmf.predict_interval(X_test, alpha=0.10)
    """

    def __init__(self, clip_lower: float = 0.0) -> None:
        self.clip_lower = clip_lower
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.n_train_: int = 0

    def fit(self, X: Any, y: Any) -> "HongModelFree":
        """
        Store training data for model-free conformal.

        No model fitting occurs. The training data is stored and used
        directly at prediction time to compute feature-adjusted responses.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Training features. Must be numeric.
        y : array-like of shape (n,)
            Training targets.

        Returns
        -------
        self
        """
        self.X_train_ = self._to_numpy(X)
        self.y_train_ = as_numpy(y).ravel()
        self.n_train_ = len(self.y_train_)

        if self.X_train_.ndim == 1:
            self.X_train_ = self.X_train_.reshape(-1, 1)

        return self

    def predict_interval(
        self, X_new: Any, alpha: float = 0.05
    ) -> pl.DataFrame:
        """
        Compute model-free conformal intervals for new observations.

        For each new observation x_new, computes feature-adjusted responses
        W_i = Y_i + mean(X_new - X_i) across all training points, then
        uses the k-th order statistic as the upper bound.

        The adjustment mean(X_new - X_i) is the average feature difference
        between the new point and all training points. It's a crude way to
        account for the fact that a richer-feature risk is likely to have
        higher expected claims.

        Parameters
        ----------
        X_new : array-like of shape (m, p)
            New observations to predict for.
        alpha : float, default 0.05
            Miscoverage rate.

        Returns
        -------
        pl.DataFrame
            Columns: lower, upper. No point column (model-free has no
            prediction; use the midpoint if you need one).
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_fitted()

        X_np = self._to_numpy(X_new)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n = self.n_train_
        k = min(n, int(np.floor((n + 1) * (1 - alpha))) + 1)

        lowers = []
        uppers = []

        for x_new_row in X_np:
            # W_i = Y_i + mean_j(x_new_j - X_ij) over all training points
            # = Y_i + (1/p) * sum_j(x_new_j - X_ij) / n * n
            # Hong's formula: W_i = Y_i + sum_j (x_new_j - X_ij) / n
            # where the sum is over features j, mean over training i
            feature_adjustment = np.mean(x_new_row - self.X_train_, axis=1)
            W = self.y_train_ + feature_adjustment

            W_sorted = np.sort(W)
            upper = float(W_sorted[k - 1]) if k <= n else float(np.inf)
            # clip_lower sets the minimum for the lower bound.
            # np.clip(0.0, clip_lower, None) always clips the constant 0.0, not
            # the upper bound — dead code. Use clip_lower directly as the floor.
            lower = float(self.clip_lower)

            lowers.append(lower)
            uppers.append(upper)

        return pl.DataFrame(
            {
                "lower": np.array(lowers),
                "upper": np.array(uppers),
            }
        )

    def _check_fitted(self) -> None:
        if self.X_train_ is None:
            raise RuntimeError("Call .fit(X_train, y_train) before predicting.")

    def _to_numpy(self, X: Any) -> np.ndarray:
        import pandas as pd
        if isinstance(X, pl.DataFrame):
            return X.to_numpy()
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.asarray(X, dtype=float)

    def __repr__(self) -> str:
        if self.X_train_ is None:
            return "HongModelFree(not fitted)"
        return f"HongModelFree(n_train={self.n_train_})"


class HongTransformConformal:
    """
    h-transformation conformal prediction (Hong arXiv:2601.21153).

    Given any callable h: X -> predictions, computes residuals
    W_i = Y_i - h(X_i) on a calibration set, then produces intervals
    [h(X_new) + W_(low), h(X_new) + W_(high)] using order statistics.

    The coverage guarantee holds for ANY h — even a terrible model. A better h
    just narrows the residual distribution and thus the interval width. This is
    the theoretical justification for why split conformal works regardless of
    model quality.

    Two-sided intervals: uses both lower and upper order statistics of W.
    Lower bound clipped at 0 for insurance data.

    When h is your fitted GBM and you use absolute residuals, this produces
    the same intervals as InsuranceConformalPredictor(nonconformity="raw").
    The value here is the flexibility: h can be anything.

    Parameters
    ----------
    h : callable
        Any function h(X) -> np.ndarray of predictions. Can be a fitted
        model's predict method, a simple mean, or a lambda.
    clip_lower : float, default 0.0
        Clip lower bounds at this value.

    Examples
    --------
    >>> htc = HongTransformConformal(h=fitted_model.predict)
    >>> htc.calibrate(X_cal, y_cal)
    >>> intervals = htc.predict_interval(X_test, alpha=0.10)

    Or with a simple constant predictor as a sanity-check baseline:
    >>> htc_baseline = HongTransformConformal(h=lambda X: np.full(len(X), y_mean))
    >>> htc_baseline.calibrate(X_cal, y_cal)
    >>> intervals = htc_baseline.predict_interval(X_test, alpha=0.10)
    """

    def __init__(
        self,
        h: Callable,
        clip_lower: float = 0.0,
    ) -> None:
        if not callable(h):
            raise ValueError("h must be callable (e.g. a model's .predict method)")
        self.h = h
        self.clip_lower = clip_lower

        self.cal_residuals_: Optional[np.ndarray] = None
        self.cal_quantiles_: dict[float, tuple[float, float]] = {}
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

    def calibrate(self, X_cal: Any, y_cal: Any) -> "HongTransformConformal":
        """
        Compute residuals W_i = Y_i - h(X_i) on calibration data.

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
        y = as_numpy(y_cal).ravel()
        h_cal = as_numpy(self.h(self._to_numpy(X_cal))).ravel()

        self.cal_residuals_ = y - h_cal
        self.n_calibration_ = len(self.cal_residuals_)
        self.cal_quantiles_ = {}
        self.is_calibrated_ = True

        return self

    def predict_interval(
        self, X_test: Any, alpha: float = 0.05
    ) -> pl.DataFrame:
        """
        Produce prediction intervals using h-transformed residuals.

        The interval is [h(X_new) + W_(lower_q), h(X_new) + W_(upper_q)]
        where W_(lower_q) and W_(upper_q) are quantiles of the calibration
        residuals.

        For a one-sided interval (upper bound only, appropriate when the main
        concern is reserve adequacy), set alpha directly — the upper bound is
        the (1-alpha) quantile of residuals shifted by h(X_new).

        Parameters
        ----------
        X_test : array-like
            Test features.
        alpha : float, default 0.05
            Miscoverage rate. Produces a two-sided (alpha/2, 1-alpha/2) interval.

        Returns
        -------
        pl.DataFrame
            Columns: lower, point, upper.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_calibrated()

        h_test = as_numpy(self.h(self._to_numpy(X_test))).ravel()
        lower_q, upper_q = self._get_residual_quantiles(alpha)

        lower = np.clip(h_test + lower_q, self.clip_lower, None)
        upper = h_test + upper_q

        return pl.DataFrame(
            {
                "lower": lower,
                "point": h_test,
                "upper": upper,
            }
        )

    def predict_upper_bound(self, X_test: Any, alpha: float = 0.05) -> np.ndarray:
        """
        Compute one-sided upper prediction bounds at level (1-alpha).

        This is the relevant output for reserve adequacy: what is the
        (1-alpha) upper bound on claims for this policy?

        Parameters
        ----------
        X_test : array-like
            Test features.
        alpha : float, default 0.05
            The upper bound covers with probability >= 1-alpha.

        Returns
        -------
        np.ndarray
            Upper bounds.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_calibrated()

        h_test = as_numpy(self.h(self._to_numpy(X_test))).ravel()
        n = self.n_calibration_
        k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
        sorted_resid = np.sort(self.cal_residuals_)
        upper_q = float(sorted_resid[k - 1]) if k <= n else float(np.inf)

        return h_test + upper_q

    def _get_residual_quantiles(self, alpha: float) -> tuple[float, float]:
        """Get (lower_q, upper_q) quantiles of calibration residuals."""
        if alpha not in self.cal_quantiles_:
            n = self.n_calibration_
            sorted_resid = np.sort(self.cal_residuals_)

            # Upper: ceil((n+1)(1-alpha/2)) / n quantile
            k_upper = min(n, int(np.ceil((n + 1) * (1 - alpha / 2))))
            upper_q = float(sorted_resid[k_upper - 1]) if k_upper <= n else float(np.inf)

            # Lower: floor((n+1)(alpha/2)) / n quantile
            k_lower = max(1, int(np.floor((n + 1) * (alpha / 2))))
            lower_q = float(sorted_resid[k_lower - 1])

            self.cal_quantiles_[alpha] = (lower_q, upper_q)

        return self.cal_quantiles_[alpha]

    def _to_numpy(self, X: Any) -> np.ndarray:
        import pandas as pd
        if isinstance(X, pl.DataFrame):
            return X.to_numpy()
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.asarray(X)

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Call .calibrate(X_cal, y_cal) before predicting."
            )

    def __repr__(self) -> str:
        status = (
            f"calibrated on {self.n_calibration_} obs"
            if self.is_calibrated_
            else "not calibrated"
        )
        return f"HongTransformConformal(h={self.h!r}, {status})"
