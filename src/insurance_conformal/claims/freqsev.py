"""
Frequency-severity conformal prediction for insurance aggregate loss models.

The frequency-severity decomposition is standard in non-life insurance pricing:
expected loss = E[frequency] * E[severity | claim occurred]. Conformal prediction
for this composed model has a subtlety: at calibration time, you don't know the
true frequency for each policy, only whether a claim occurred. The correct approach
is to feed the *predicted* frequency into the severity model — not the observed
claim count — so that calibration and prediction operate in the same way.

This is the key insight from Graziadei et al. (2307.13124): the conformity score
on the calibration set must use the composed prediction psi_hat(x, mu_hat(x)),
not psi_hat(x, d_i). Using observed claim counts at calibration would make the
score distribution systematically different from the test-time prediction, breaking
the exchangeability assumption that underpins conformal coverage guarantees.

Algorithm (Graziadei et al. 2307.13124, Section 3):
    1. Split data: training I_1, calibration I_2.
    2. Fit frequency model mu_hat on {(x_i, d_i) : i in I_1}.
    3. Fit severity model psi_hat on {(x_i, d_i, y_i) : i in I_1, d_i > 0}.
    4. Compute training residuals delta_i = |y_i - psi_hat(x_i, d_i)| for
       i in I_1 where d_i > 0.
    5. Fit variability model sigma_hat on {(x_i, d_i, delta_i) : i in I_1, d_i > 0}.
    6. On calibration I_2: compute conformity scores
       R_i = |y_i - psi_hat(x_i, mu_hat(x_i))| / sigma_hat(x_i, mu_hat(x_i))
       KEY: uses composed prediction psi_hat(x, mu_hat(x)).
    7. q_hat = ceil((|I_2|+1)(1-alpha)) / |I_2| quantile of {R_i}.
    8. Prediction interval:
       [psi_hat(x, mu_hat(x)) - q_hat * sigma_hat(x, mu_hat(x)),
        psi_hat(x, mu_hat(x)) + q_hat * sigma_hat(x, mu_hat(x))]

Coverage guarantee: P(Y in C(X)) in [1-alpha, 1-alpha + 1/(n_cal+1)].

References
----------
Graziadei et al. (2023) arXiv:2307.13124.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy, conformal_quantile


class FrequencySeverityConformal:
    """
    Conformal prediction intervals for frequency-severity insurance models.

    The frequency-severity decomposition is the workhorse of non-life pricing:
    total loss = frequency * severity. A key question for conformal prediction is
    what to feed into the severity model at calibration time — observed claim count
    or predicted frequency? The answer is predicted frequency: otherwise the score
    distribution at calibration differs from the test distribution, breaking the
    coverage guarantee.

    This class implements the Graziadei et al. (2307.13124) algorithm with a
    CatBoost variability model (sigma_hat) by default. The variability model
    estimates how large residuals tend to be for a given (x, mu_hat(x)) pair,
    analogous to the spread model in LocallyWeightedConformal.

    Parameters
    ----------
    freq_model : sklearn-compatible model
        Frequency model. Must implement predict(X) returning expected frequency.
        Typically a Poisson GLM or GBM. Does NOT need to be pre-fitted — fit()
        will train it on the training data.
    sev_model : sklearn-compatible model
        Severity model. Must implement fit(X, y) and predict(X). The feature
        matrix passed to it will include an appended column for (predicted or
        observed) claim frequency.
    spread_model : sklearn-compatible model, optional
        Variability model sigma_hat. Must implement fit(X, y) and predict(X).
        If None, defaults to CatBoostRegressor with RMSE loss.
    alpha : float, default 0.1
        Default miscoverage rate used in predict_interval(). Can be overridden
        at prediction time.

    Attributes
    ----------
    freq_model_ : fitted frequency model
        Available after fit().
    sev_model_ : fitted severity model
        Available after fit().
    spread_model_ : fitted variability model
        Available after fit().
    cal_scores_ : np.ndarray
        Conformity scores from calibration set. Available after calibrate().
    n_calibration_ : int
        Number of calibration observations.
    is_calibrated_ : bool
        Whether calibrate() has been called.

    Examples
    --------
    >>> from sklearn.linear_model import PoissonRegressor, GammaRegressor
    >>> fs = FrequencySeverityConformal(
    ...     freq_model=PoissonRegressor(),
    ...     sev_model=GammaRegressor(),
    ... )
    >>> fs.fit(X_train, d_train, y_train)
    >>> fs.calibrate(X_cal, d_cal, y_cal)
    >>> intervals = fs.predict_interval(X_test, alpha=0.10)

    References
    ----------
    Graziadei et al. (2023) arXiv:2307.13124.
    """

    def __init__(
        self,
        freq_model: Any,
        sev_model: Any,
        spread_model: Optional[Any] = None,
        alpha: float = 0.1,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.freq_model = freq_model
        self.sev_model = sev_model
        self.spread_model = spread_model
        self.alpha = float(alpha)

        self.freq_model_: Any = None
        self.sev_model_: Any = None
        self.spread_model_: Any = None

        self.cal_scores_: Optional[np.ndarray] = None
        self.cal_quantiles_: dict[float, float] = {}
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

    def fit(
        self,
        X_train: Any,
        d_train: Any,
        y_train: Any,
    ) -> "FrequencySeverityConformal":
        """
        Fit frequency, severity, and variability models on training data.

        Steps performed:
        1. Fit frequency model on (X_train, d_train).
        2. Fit severity model on (X_train, d_train, y_train) for observations
           where d_train > 0 (claims only). The severity model receives features
           [X, d] — the observed claim count appended to covariates.
        3. Compute training residuals for claim observations.
        4. Fit variability model on (X_train[claims], d_train[claims], residuals).

        Parameters
        ----------
        X_train : array-like, shape (n, p)
            Training features (policy characteristics).
        d_train : array-like, shape (n,)
            Observed claim counts.
        y_train : array-like, shape (n,)
            Observed aggregate loss amounts. Should be 0 when d_train == 0.

        Returns
        -------
        self
        """
        X = np.asarray(as_numpy(X_train), dtype=float)
        d = as_numpy(d_train).astype(float)
        y = as_numpy(y_train).astype(float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Step 2: fit frequency model
        self.freq_model_ = self.freq_model
        self.freq_model_.fit(X, d)

        # Step 3: fit severity model on claim observations
        claim_mask = d > 0
        if claim_mask.sum() == 0:
            raise ValueError(
                "No claim observations in training data (all d_train == 0). "
                "Cannot fit severity model."
            )

        X_claims = X[claim_mask]
        d_claims = d[claim_mask]
        y_claims = y[claim_mask]

        # Severity features: [X, d] — append observed claim count
        X_sev_train = self._append_freq(X_claims, d_claims)
        self.sev_model_ = self.sev_model
        self.sev_model_.fit(X_sev_train, y_claims)

        # Step 4: training residuals delta_i = |y_i - psi_hat(x_i, d_i)|
        sev_pred_train = as_numpy(self.sev_model_.predict(X_sev_train)).ravel()
        delta = np.abs(y_claims - sev_pred_train)

        # Step 5: fit variability model on (X_claims, d_claims, delta)
        X_var_train = self._append_freq(X_claims, d_claims)
        if self.spread_model is None:
            self.spread_model_ = self._make_default_spread_model()
        else:
            self.spread_model_ = self.spread_model
        self.spread_model_.fit(X_var_train, delta)

        return self

    def calibrate(
        self,
        X_cal: Any,
        d_cal: Any,
        y_cal: Any,
    ) -> "FrequencySeverityConformal":
        """
        Compute conformity scores on calibration data.

        Conformity score for each calibration observation:
            R_i = |y_i - psi_hat(x_i, mu_hat(x_i))| / sigma_hat(x_i, mu_hat(x_i))

        Note: mu_hat(x_i) is the PREDICTED frequency from the frequency model,
        not the observed d_i. This is essential — using observed d_i would
        create a distributional mismatch between calibration scores and test
        scores, breaking the coverage guarantee.

        Observations where both y_i == 0 and the prediction is near zero
        are still included with score = 0; they contribute to the calibration
        empirical distribution.

        Parameters
        ----------
        X_cal : array-like, shape (n_cal, p)
            Calibration features.
        d_cal : array-like, shape (n_cal,)
            Observed claim counts (used only to verify data, not in scores).
        y_cal : array-like, shape (n_cal,)
            Observed aggregate losses.

        Returns
        -------
        self
        """
        self._check_fitted()

        X = np.asarray(as_numpy(X_cal), dtype=float)
        y = as_numpy(y_cal).astype(float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Use composed prediction: feed mu_hat(x) into severity model
        mu_hat = self._freq_predict(X)  # predicted frequency
        composed_pred = self._composed_predict(X, mu_hat)
        sigma_hat = self._spread_predict(X, mu_hat)

        self.cal_scores_ = np.abs(y - composed_pred) / sigma_hat
        self.n_calibration_ = len(self.cal_scores_)
        self.cal_quantiles_ = {}
        self.is_calibrated_ = True

        return self

    def predict_interval(
        self,
        X_new: Any,
        alpha: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Produce frequency-severity conformal prediction intervals.

        For each new observation:
        1. Compute mu_hat(x) from frequency model.
        2. Compute point prediction: psi_hat(x, mu_hat(x)).
        3. Compute variability: sigma_hat(x, mu_hat(x)).
        4. Interval: point +/- q_hat * sigma_hat, lower clipped at 0.

        Parameters
        ----------
        X_new : array-like, shape (n_new, p)
            New policy features.
        alpha : float, optional
            Miscoverage rate. Defaults to self.alpha set at construction.
            0.10 gives 90% prediction intervals.

        Returns
        -------
        pl.DataFrame
            Columns: lower, point, upper.
            lower >= 0 always (losses can't be negative).
        """
        if alpha is None:
            alpha = self.alpha
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_calibrated()

        X = np.asarray(as_numpy(X_new), dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        q_hat = self._get_quantile(alpha)
        mu_hat = self._freq_predict(X)
        point = self._composed_predict(X, mu_hat)
        sigma_hat = self._spread_predict(X, mu_hat)

        half_width = q_hat * sigma_hat
        lower = np.clip(point - half_width, 0.0, None)
        upper = point + half_width

        return pl.DataFrame(
            {
                "lower": lower,
                "point": point,
                "upper": upper,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_freq(self, X: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """Append frequency column to feature matrix."""
        return np.column_stack([X, freq.reshape(-1, 1)])

    def _freq_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expected frequency from freq_model_."""
        return as_numpy(self.freq_model_.predict(X)).ravel()

    def _composed_predict(self, X: np.ndarray, mu_hat: np.ndarray) -> np.ndarray:
        """Predict severity using composed input psi_hat(x, mu_hat(x))."""
        X_sev = self._append_freq(X, mu_hat)
        return as_numpy(self.sev_model_.predict(X_sev)).ravel()

    def _spread_predict(self, X: np.ndarray, mu_hat: np.ndarray) -> np.ndarray:
        """Predict variability sigma_hat(x, mu_hat(x)), clipped at minimum."""
        X_var = self._append_freq(X, mu_hat)
        raw = as_numpy(self.spread_model_.predict(X_var)).ravel()
        # Clip at a small positive value to avoid division by zero
        return np.clip(raw, 1e-6, None)

    def _get_quantile(self, alpha: float) -> float:
        """Return the cached conformal quantile for given alpha."""
        self._check_calibrated()
        if alpha not in self.cal_quantiles_:
            self.cal_quantiles_[alpha] = conformal_quantile(self.cal_scores_, alpha)
        return self.cal_quantiles_[alpha]

    def _make_default_spread_model(self) -> Any:
        """Create default CatBoostRegressor for variability estimation."""
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "FrequencySeverityConformal requires CatBoost for the default "
                "spread model. Install with: pip install catboost\n"
                "Or pass spread_model= to use a different estimator."
            ) from exc
        return CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )

    def _check_fitted(self) -> None:
        if self.freq_model_ is None or self.sev_model_ is None or self.spread_model_ is None:
            raise RuntimeError(
                "Models have not been fitted. Call .fit(X_train, d_train, y_train) first."
            )

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Predictor has not been calibrated. "
                "Call .calibrate(X_cal, d_cal, y_cal) after .fit()."
            )

    def __repr__(self) -> str:
        freq_name = type(self.freq_model).__name__
        sev_name = type(self.sev_model).__name__
        status = (
            f"calibrated on {self.n_calibration_} obs"
            if self.is_calibrated_
            else ("fitted, not calibrated" if self.freq_model_ is not None else "not fitted")
        )
        return (
            f"FrequencySeverityConformal("
            f"freq={freq_name}, sev={sev_name}, "
            f"alpha={self.alpha}, {status})"
        )
