"""
Model-free full conformal prediction via Hong's order-statistic shortcut.

References
----------
Hong (2025) arXiv:2503.03659 — "Conformal Prediction of Future Insurance Claims
    in the Regression Problem"
Hong (2026) arXiv:2601.21153 — "A New Strategy for Finite-Sample Valid Prediction
    of Future Insurance Claims in the Regression Setting"
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class HongConformal:
    """Model-free full conformal prediction using Hong's order-statistic shortcut.

    Standard full conformal requires a grid search over candidate response values.
    Hong (2025) shows that for the nonconformity measure defined in Eq. 5 of
    arXiv:2503.03659, the prediction region reduces to a single order statistic,
    giving an O(n log n) algorithm with no model and no grid.

    The adjusted score for training observation i given a new point x_{n+1} is:

        W_i = Y_i + (1/n) * sum_j (X_{n+1,j} - X_{ij})

    The 100(1-alpha)% prediction region is (0, W_{(k)}) where:

        k = min(n, floor((n+1)(1-alpha) + 1))

    Coverage is guaranteed finite-sample for all exchangeable distributions (Theorem
    1 of arXiv:2503.03659). No distributional or model assumption required.

    Parameters
    ----------
    batch_size : int, optional
        Number of new points to process at once in predict_interval. Controls
        peak memory. At n=50_000 training observations and batch_size=1000,
        the W matrix is 50_000 x 1000 float64 = 400 MB. Reduce for limited RAM.
        Default: 500.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = np.abs(rng.normal(size=200))
    >>> hc = HongConformal()
    >>> hc.fit(X, y)
    HongConformal(n=200, p=3)
    >>> intervals = hc.predict_interval(X[:5], alpha=0.05)
    >>> intervals.shape
    (5, 2)
    """

    def __init__(self, batch_size: int = 500) -> None:
        self.batch_size = batch_size
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HongConformal":
        """Store training data. No model is fitted.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Training covariates.
        y : np.ndarray, shape (n,)
            Training responses. Must be non-negative (insurance claims).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        self._X_train = X
        self._y_train = y
        return self

    def _check_fitted(self) -> None:
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict_interval()")

    @property
    def n_(self) -> int:
        """Number of training observations."""
        self._check_fitted()
        return len(self._y_train)  # type: ignore[arg-type]

    def predict_interval(
        self, X_new: np.ndarray, alpha: float = 0.005
    ) -> np.ndarray:
        """Compute prediction intervals via the order-statistic shortcut.

        For each row x of X_new:

            W_i = y_i + (1/n) * sum_j (x_j - X_{ij}),  i = 1..n
            k   = min(n, floor((n+1)(1-alpha) + 1))
            upper = W_{(k)}   (k-th order statistic of W)

        Returns (0, upper) for each new point.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, p)
            New covariate matrix.
        alpha : float
            Miscoverage rate. alpha=0.005 gives 99.5% intervals (Solvency II SCR).

        Returns
        -------
        intervals : np.ndarray, shape (n_new, 2)
            Column 0 is always 0.0. Column 1 is the upper prediction bound.

        Notes
        -----
        Memory: peak usage is O(batch_size * n) float64 values per batch.
        For n=50_000 and batch_size=500, peak is ~200 MB.
        """
        self._check_fitted()
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        if X_new.shape[1] != self._X_train.shape[1]:  # type: ignore[union-attr]
            raise ValueError(
                f"Feature dimension mismatch: training has "
                f"{self._X_train.shape[1]} columns, X_new has {X_new.shape[1]}"  # type: ignore[union-attr]
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        n = self.n_
        k = min(n, math.floor((n + 1) * (1.0 - alpha) + 1)) - 1  # 0-indexed

        n_new = len(X_new)
        uppers = np.empty(n_new, dtype=float)

        X_train = self._X_train  # type: ignore[assignment]
        y_train = self._y_train  # type: ignore[assignment]

        # Process in batches to bound peak memory.
        for start in range(0, n_new, self.batch_size):
            end = min(start + self.batch_size, n_new)
            x_batch = X_new[start:end]  # (batch, p)

            # W[b, i] = y_i + mean_j(x_batch[b,j] - X_train[i,j])
            # = y_i + (x_batch[b,:] - X_train[i,:]).mean(axis=-1) — but note
            # the paper sums over features j=1..p and divides by n (sample size),
            # not by p. Re-reading Eq. 5 of 2503.03659: the correction is
            # (1/n) * sum_{j=1}^{p} (x_{n+1,j} - x_{ij}). So divide by n, not p.
            # Shape broadcast: x_batch (B,p), X_train (n,p)
            diff = x_batch[:, np.newaxis, :] - X_train[np.newaxis, :, :]  # (B,n,p)
            correction = diff.sum(axis=-1) / n  # (B, n)
            W = y_train[np.newaxis, :] + correction  # (B, n)

            # k-th order statistic (0-indexed). np.partition is O(n) average.
            partitioned = np.partition(W, k, axis=1)  # (B, n)
            uppers[start:end] = partitioned[:, k]

        lowers = np.zeros(n_new, dtype=float)
        return np.column_stack([lowers, uppers])

    def coverage_diagnostic(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float = 0.005,
    ) -> dict:
        """Compute empirical coverage on a held-out test set.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, p)
        y_test : np.ndarray, shape (n_test,)
        alpha : float

        Returns
        -------
        dict with keys: empirical_coverage, nominal_coverage, mean_width, n_test
        """
        y_test = np.asarray(y_test, dtype=float)
        intervals = self.predict_interval(X_test, alpha=alpha)
        covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
        widths = intervals[:, 1] - intervals[:, 0]
        return {
            "empirical_coverage": float(covered.mean()),
            "nominal_coverage": float(1.0 - alpha),
            "mean_width": float(widths.mean()),
            "n_test": len(y_test),
        }

    def __repr__(self) -> str:
        if self._X_train is None:
            return "HongConformal(not fitted)"
        n, p = self._X_train.shape
        return f"HongConformal(n={n}, p={p})"


class HongTransformConformal:
    """h-transformation extension of Hong's conformal prediction.

    Hong (2026) arXiv:2601.21153 decomposes Y = h(X) + W where h is any
    non-negative function. The residuals W_i = Y_i - h(X_i) are iid regardless
    of h (the exchangeability argument is unchanged). When h is a well-fitted
    regression model, W = Y - h(X) has smaller variance, producing narrower
    prediction intervals while preserving the finite-sample coverage guarantee.

    The prediction interval for a new point x is:

        I^h_alpha(x) = (max(0, h(x) + W_{(k_lo)}), h(x) + W_{(k_hi)})

    For a one-sided upper bound (insurance claims, SCR use):

        I^h_alpha(x) = (0, h(x) + W_{(k)})

    where k = min(n_cal, floor((n_cal+1)(1-alpha) + 1)).

    Parameters
    ----------
    h_model : sklearn estimator or callable, optional
        Regression model implementing fit(X, y) and predict(X). If None, uses
        column-mean prediction — equivalent to HongConformal on the calibration
        data. Any sklearn-compatible estimator works (LinearRegression,
        GradientBoostingRegressor, etc.).

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> htc = HongTransformConformal(h_model=Ridge())
    >>> htc.fit(X_train, y_train, X_cal, y_cal)
    HongTransformConformal(h=Ridge, n_cal=100)
    """

    def __init__(self, h_model: Any = None) -> None:
        self.h_model = h_model
        self._fitted_h: Any = None
        self._mean_y: float | None = None
        self._W_cal: np.ndarray | None = None
        self._n_cal: int = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
    ) -> "HongTransformConformal":
        """Fit h on training data and compute calibration residuals.

        If X_cal / y_cal are not provided, calibration residuals are computed
        on the training data (in-sample). This is optimistic — use a held-out
        calibration set where possible.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, p)
        y_train : np.ndarray, shape (n_train,)
        X_cal : np.ndarray, shape (n_cal, p), optional
        y_cal : np.ndarray, shape (n_cal,), optional

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        if self.h_model is None:
            # Column-mean prediction: h(x) = mean of y_train
            # Equivalent to HongConformal (model-free baseline)
            self._mean_y = float(y_train.mean())
            self._fitted_h = None
        else:
            h = self.h_model
            h.fit(X_train, y_train)
            self._fitted_h = h

        # Calibration residuals W = y_cal - h(X_cal)
        if X_cal is None or y_cal is None:
            X_cal_arr = X_train
            y_cal_arr = y_train
        else:
            X_cal_arr = np.asarray(X_cal, dtype=float)
            y_cal_arr = np.asarray(y_cal, dtype=float)
            if X_cal_arr.ndim == 1:
                X_cal_arr = X_cal_arr.reshape(-1, 1)

        h_cal = self._h_predict(X_cal_arr)
        self._W_cal = y_cal_arr - h_cal
        self._n_cal = len(self._W_cal)
        return self

    def _h_predict(self, X: np.ndarray) -> np.ndarray:
        """Apply h to X."""
        if self._fitted_h is None:
            # Mean-prediction model
            return np.full(len(X), self._mean_y)
        return np.asarray(self._fitted_h.predict(X), dtype=float)

    def _check_fitted(self) -> None:
        if self._W_cal is None:
            raise RuntimeError("Call fit() before predict_interval()")

    def predict_interval(
        self, X_new: np.ndarray, alpha: float = 0.005
    ) -> np.ndarray:
        """Compute prediction intervals using calibration residuals W.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, p)
        alpha : float
            Miscoverage rate.

        Returns
        -------
        intervals : np.ndarray, shape (n_new, 2)
            Lower bound clipped at 0. Upper bound = h(x) + W_{(k)}.
        """
        self._check_fitted()
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        n = self._n_cal
        k = min(n, math.floor((n + 1) * (1.0 - alpha) + 1)) - 1  # 0-indexed

        W_sorted = np.sort(self._W_cal)
        W_k = W_sorted[k]

        h_new = self._h_predict(X_new)
        uppers = h_new + W_k
        # Lower bound is 0 for insurance claims (non-negative by definition).
        # The previous symmetric lower-bound computation was dead code overwritten
        # by the next line, so it has been removed.
        lowers = np.zeros_like(uppers)

        # Warn if the upper bound is negative — this indicates a degenerate
        # calibration where the (1-alpha) residual quantile W_k is so negative
        # that h(x) + W_k < 0. The model h is a poor fit or the calibration
        # set is too small. Clipping silently would hide this problem.
        n_negative = int(np.sum(uppers < 0))
        if n_negative > 0:
            import warnings
            warnings.warn(
                f"{n_negative} upper bounds are negative (degenerate interval). "
                "This means h(x) + W_k < 0, which occurs when the calibration "
                "residual quantile is more negative than the model prediction. "
                "Check that the h model is a reasonable fit and that the "
                "calibration set is representative.",
                UserWarning,
                stacklevel=2,
            )
        uppers = np.maximum(0.0, uppers)

        return np.column_stack([lowers, uppers])

    def select_h(
        self,
        h_candidates: list,
        X_val: np.ndarray,
        y_val: np.ndarray,
        alpha: float = 0.005,
    ) -> Any:
        """Select the h model that minimises mean interval width subject to coverage.

        Evaluates each candidate h on the validation set (X_val, y_val). Filters
        to candidates achieving empirical_coverage >= 1-alpha, then returns the
        one with the smallest mean interval width.

        If no candidate achieves the coverage threshold, returns the one closest
        to nominal coverage.

        Parameters
        ----------
        h_candidates : list
            List of sklearn estimators or callables. Each will be cloned (if
            sklearn) or used as-is. Must already be fitted before calling this,
            or the caller must pass pre-fitted models.
        X_val : np.ndarray, shape (n_val, p)
        y_val : np.ndarray, shape (n_val,)
        alpha : float

        Returns
        -------
        best_h : the selected h model

        Notes
        -----
        Selection is done on the validation set — a distinct held-out set should
        be used for final performance evaluation to avoid optimistic width estimates.
        """
        self._check_fitted()
        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        if X_val.ndim == 1:
            X_val = X_val.reshape(1, -1)

        nominal = 1.0 - alpha
        results = []

        for h in h_candidates:
            original_fitted = self._fitted_h
            original_mean_y = self._mean_y
            # Temporarily swap in this candidate.
            # h=None means use the mean-prediction baseline.
            if h is None:
                self._fitted_h = None
                # Compute mean from calibration residuals (W + 0 = W; mean of W near 0
                # for well-fitted h). Use cached mean if available.
                if self._mean_y is None:
                    # Fallback: use W_cal centre as proxy for mean
                    self._mean_y = float(np.mean(self._W_cal + 0.0)) if self._W_cal is not None else 0.0
            else:
                self._fitted_h = h

            intervals = self.predict_interval(X_val, alpha=alpha)
            covered = (y_val >= intervals[:, 0]) & (y_val <= intervals[:, 1])
            cov = float(covered.mean())
            width = float((intervals[:, 1] - intervals[:, 0]).mean())
            results.append((h, cov, width))

            # Restore
            self._fitted_h = original_fitted
            self._mean_y = original_mean_y

        valid = [(h, cov, w) for h, cov, w in results if cov >= nominal]
        if valid:
            best = min(valid, key=lambda x: x[2])
        else:
            # No candidate meets coverage — pick closest to nominal
            best = max(results, key=lambda x: x[1])

        return best[0]

    def coverage_diagnostic(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float = 0.005,
    ) -> dict:
        """Compute empirical coverage on a held-out test set."""
        y_test = np.asarray(y_test, dtype=float)
        intervals = self.predict_interval(X_test, alpha=alpha)
        covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
        widths = intervals[:, 1] - intervals[:, 0]
        return {
            "empirical_coverage": float(covered.mean()),
            "nominal_coverage": float(1.0 - alpha),
            "mean_width": float(widths.mean()),
            "n_test": len(y_test),
        }

    def __repr__(self) -> str:
        if self._W_cal is None:
            return "HongTransformConformal(not fitted)"
        h_name = type(self._fitted_h).__name__ if self._fitted_h is not None else "mean"
        return f"HongTransformConformal(h={h_name}, n_cal={self._n_cal})"
