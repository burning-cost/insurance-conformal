"""
Insurance-specific nonconformity scores for Tweedie claims regression,
plus split conformal with a two-stage locally weighted Pearson score.

References
----------
Manna et al. (2025) ASMBI asmb.70045 — "Conformal Prediction Inference in
    Regularized Insurance Models"
Hong (2025) arXiv:2503.03659
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import scipy.optimize as opt


def _tweedie_deviance_element(y: float, mu: float, p: float) -> float:
    """Per-element Tweedie deviance (unit deviance, not scaled by phi).

    For p in (1, 2) (compound Poisson-Gamma):

        d(y, mu) = 2 * [y^{2-p} / ((2-p)(1-p)) - y * mu^{1-p} / (1-p) + mu^{2-p} / (2-p)]

    Edge cases:
        y = 0: term involving y^{2-p} vanishes (0^{2-p} = 0 for p < 2)
        p = 1: Poisson deviance  2 * [y*log(y/mu) - (y-mu)]
        p = 2: Gamma deviance    2 * [log(mu/y) + y/mu - 1]
    """
    if p == 1.0:
        if y == 0.0:
            return 2.0 * mu
        return 2.0 * (y * math.log(y / mu) - (y - mu))
    if p == 2.0:
        if y == 0.0:
            return float("inf")
        return 2.0 * (math.log(mu / y) + y / mu - 1.0)
    # General p in (1, 2)
    term1 = y * mu ** (1 - p) / (1 - p)
    term2 = (y ** (2 - p)) / ((1 - p) * (2 - p)) if y > 0 else 0.0
    term3 = mu ** (2 - p) / (2 - p)
    return 2.0 * (term2 - term1 + term3)


class TweedePearsonScore:
    """Tweedie Pearson nonconformity score for split conformal prediction.

    The nonconformity score is:

        R = |y - mu_hat| / mu_hat^{p/2}

    where mu_hat is the predicted mean and p is the Tweedie power parameter.
    For Tweedie compound Poisson-Gamma, p in (1, 2); Var(Y|X) = phi * mu^p,
    so mu^{p/2} is the standard deviation up to the dispersion parameter phi.

    Parameters
    ----------
    p : float
        Tweedie power. p=1 Poisson, p=2 Gamma, 1 < p < 2 compound Poisson-Gamma.
        Default: 1.5 (midpoint of compound Poisson-Gamma range).
    min_mu : float
        Lower clip for mu_hat to prevent division by zero. Default: 1e-6.

    References
    ----------
    Manna et al. (2025) ASMBI asmb.70045, Eq. (3).
    """

    def __init__(self, p: float = 1.5, min_mu: float = 1e-6) -> None:
        if not 1.0 <= p <= 2.0:
            warnings.warn(
                f"Tweedie power p={p} is outside the compound Poisson-Gamma range "
                f"[1, 2]. Scores may not be appropriate for insurance claims.",
                UserWarning,
                stacklevel=2,
            )
        self.p = float(p)
        self.min_mu = float(min_mu)

    def score(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Observed values.
        y_hat : np.ndarray, shape (n,)
            Predicted means from a fitted Tweedie model.

        Returns
        -------
        scores : np.ndarray, shape (n,)
            R_i = |y_i - y_hat_i| / y_hat_i^{p/2}
        """
        y = np.asarray(y, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        mu = np.clip(y_hat, self.min_mu, None)
        return np.abs(y - mu) / mu ** (self.p / 2.0)

    def inverse(
        self, s: float | np.ndarray, y_hat: np.ndarray, upper: bool = True
    ) -> np.ndarray:
        """Invert the score to obtain prediction bounds.

        Parameters
        ----------
        s : float or np.ndarray
            Conformal quantile (scalar or broadcastable to y_hat).
        y_hat : np.ndarray, shape (n,)
            Predicted means.
        upper : bool
            If True, return upper bound: y_hat + s * y_hat^{p/2}.
            If False, return lower bound: max(0, y_hat - s * y_hat^{p/2}).

        Returns
        -------
        bound : np.ndarray, shape (n,)
        """
        y_hat = np.asarray(y_hat, dtype=float)
        mu = np.clip(y_hat, self.min_mu, None)
        margin = np.asarray(s) * mu ** (self.p / 2.0)
        if upper:
            return mu + margin
        return np.maximum(0.0, mu - margin)

    def __repr__(self) -> str:
        return f"TweedePearsonScore(p={self.p})"


class TweedieDevianceScore:
    """Tweedie deviance nonconformity score for split conformal prediction.

    The nonconformity score is the signed deviance residual:

        R = sqrt(d(y, mu_hat))

    where d(y, mu) is the unit Tweedie deviance. Inverting this score requires
    solving d(y, mu_hat) = s^2 for y, which has no closed form when p != 1 or 2.
    scipy.optimize.brentq is used per element.

    Parameters
    ----------
    p : float
        Tweedie power. Default: 1.5.
    min_mu : float
        Lower clip for mu_hat. Default: 1e-6.

    References
    ----------
    Manna et al. (2025) ASMBI asmb.70045, Eq. (4).
    """

    def __init__(self, p: float = 1.5, min_mu: float = 1e-6) -> None:
        self.p = float(p)
        self.min_mu = float(min_mu)

    def score(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Compute deviance nonconformity scores.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
        y_hat : np.ndarray, shape (n,)

        Returns
        -------
        scores : np.ndarray, shape (n,)
            R_i = sqrt(d(y_i, y_hat_i))
        """
        y = np.asarray(y, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        mu = np.clip(y_hat, self.min_mu, None)
        n = len(y)
        scores = np.empty(n, dtype=float)
        for i in range(n):
            scores[i] = math.sqrt(max(0.0, _tweedie_deviance_element(y[i], mu[i], self.p)))
        return scores

    def inverse(
        self, s: float | np.ndarray, y_hat: np.ndarray, upper: bool = True
    ) -> np.ndarray:
        """Invert the deviance score to obtain prediction bounds.

        Solves sqrt(d(y, mu_hat)) = s for y using brentq. Upper bound: y > mu_hat.
        Lower bound: y < mu_hat (clipped at 0).

        Parameters
        ----------
        s : float or np.ndarray
        y_hat : np.ndarray, shape (n,)
        upper : bool

        Returns
        -------
        bound : np.ndarray, shape (n,)
        """
        y_hat = np.asarray(y_hat, dtype=float)
        s_arr = np.broadcast_to(np.asarray(s, dtype=float), y_hat.shape)
        mu = np.clip(y_hat, self.min_mu, None)
        n = len(mu)
        bound = np.empty(n, dtype=float)

        for i in range(n):
            mu_i = mu[i]
            s_i = float(s_arr.flat[i])
            target = s_i ** 2  # d(y, mu) = target

            if upper:
                # y > mu_i: search in [mu_i, mu_i * 1e6]
                hi = mu_i * 1e6
                # Expand bracket if deviance at hi is still less than target
                for _ in range(20):
                    if _tweedie_deviance_element(hi, mu_i, self.p) >= target:
                        break
                    hi *= 10.0
                try:
                    bound[i] = opt.brentq(
                        lambda y_: _tweedie_deviance_element(y_, mu_i, self.p) - target,
                        mu_i + 1e-10,
                        hi,
                        xtol=1e-6,
                        maxiter=100,
                    )
                except ValueError:
                    bound[i] = hi
            else:
                # y < mu_i: search in [0, mu_i]
                lo = max(0.0, mu_i - s_i * mu_i ** (self.p / 2.0) * 10.0)
                try:
                    bound[i] = opt.brentq(
                        lambda y_: _tweedie_deviance_element(y_, mu_i, self.p) - target,
                        max(lo, 1e-10),
                        mu_i - 1e-10,
                        xtol=1e-6,
                        maxiter=100,
                    )
                except ValueError:
                    bound[i] = 0.0

        return bound

    def __repr__(self) -> str:
        return f"TweedieDevianceScore(p={self.p})"


class TweedieAnscombeScore:
    """Tweedie Anscombe nonconformity score for split conformal prediction.

    The Anscombe transformation normalises the Tweedie response to approximate
    normality. The nonconformity score is:

        R = |y^{(2-p)/2} - mu^{(2-p)/2}| / mu^{(2-p)/2 - 1}

    This uses the variance-stabilising Anscombe transform A(y) = y^{(2-p)/2}
    whose derivative A'(mu) = (2-p)/2 * mu^{(2-p)/2 - 1}, consistent with the
    anscombe_score() function in scores.py.

    Note: the previous implementation used exponent 1-p/3 which is incorrect
    for the Tweedie Anscombe transform and inconsistent with scores.py.

    Parameters
    ----------
    p : float
        Tweedie power. Default: 1.5. Must not equal 2.
    min_mu : float
        Lower clip for mu_hat. Default: 1e-6.

    References
    ----------
    Manna et al. (2025) ASMBI asmb.70045, Eq. (5).
    """

    def __init__(self, p: float = 1.5, min_mu: float = 1e-6) -> None:
        if abs(p - 2.0) < 1e-9:
            raise ValueError("Anscombe score undefined for p=2 (exponent is 0; use Gamma/log transform)")
        self.p = float(p)
        self.min_mu = float(min_mu)
        # Variance-stabilising transform A(y) = y^{(2-p)/2}, matching scores.py
        self._exp_y = (2.0 - p) / 2.0   # exponent for y and mu in the transform
        self._exp_mu = self._exp_y - 1.0  # derivative exponent = (2-p)/2 - 1 = -p/2

    def score(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Compute Anscombe nonconformity scores.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
        y_hat : np.ndarray, shape (n,)

        Returns
        -------
        scores : np.ndarray, shape (n,)
            R_i = |y_i^{(2-p)/2} - mu_i^{(2-p)/2}| / mu_i^{(2-p)/2 - 1}
        """
        y = np.asarray(y, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        mu = np.clip(y_hat, self.min_mu, None)
        y_clipped = np.maximum(y, 0.0)
        y_transformed = np.where(y_clipped > 0, y_clipped ** self._exp_y, 0.0)
        mu_transformed = mu ** self._exp_y
        denom = mu ** self._exp_mu
        return np.abs(y_transformed - mu_transformed) / denom

    def inverse(
        self, s: float | np.ndarray, y_hat: np.ndarray, upper: bool = True
    ) -> np.ndarray:
        """Invert the Anscombe score to obtain prediction bounds.

        Solves |y^{(2-p)/2} - mu^{(2-p)/2}| / mu^{(2-p)/2 - 1} = s for y.

        Parameters
        ----------
        s : float or np.ndarray
        y_hat : np.ndarray, shape (n,)
        upper : bool

        Returns
        -------
        bound : np.ndarray, shape (n,)
        """
        y_hat = np.asarray(y_hat, dtype=float)
        mu = np.clip(y_hat, self.min_mu, None)
        s_arr = np.broadcast_to(np.asarray(s, dtype=float), y_hat.shape)

        # Invert: y^{(2-p)/2} = mu^{(2-p)/2} +/- s * mu^{(2-p)/2 - 1}
        # => y = (mu^{(2-p)/2} +/- s * mu^{(2-p)/2 - 1})^{2/(2-p)}
        sign = 1.0 if upper else -1.0
        exp_y = self._exp_y  # (2-p)/2
        inner = mu ** exp_y + sign * s_arr * mu ** self._exp_mu
        # inner must be positive for real solution; clip at 0
        inner = np.maximum(0.0, inner)
        # y = inner^{1/exp_y} = inner^{2/(2-p)}
        bound = inner ** (1.0 / exp_y)
        if not upper:
            bound = np.maximum(0.0, bound)
        return bound

    def __repr__(self) -> str:
        return f"TweedieAnscombeScore(p={self.p})"


class TwoStageLWConformal:
    """Split conformal prediction with a two-stage locally weighted Pearson score.

    Stage 1: Fit a mean model (any sklearn estimator) on training data.
    Stage 2: Fit a spread model (CatBoost by default) on training Pearson residuals
             to learn heteroscedasticity. The spread model predicts rho_hat(X_i),
             where rho_hat estimates E[R_{Pear}(X_i)].

    Locally weighted score on calibration set:

        R*_i = |y_i - mu_hat_i| / (rho_hat(X_i) * mu_hat_i^{p/2})

    Prediction interval for new point x:

        [max(0, mu_hat - q* * rho_hat(x) * mu_hat^{p/2}),
              mu_hat + q* * rho_hat(x) * mu_hat^{p/2}]

    where q* is the ceil((1-alpha)(n_cal+1)) / n_cal quantile of calibration scores.

    Coverage guarantee: P(Y in I) in [1-alpha, 1-alpha + 1/(n_cal+1)] (Theorem 1,
    Manna et al. 2025).

    Parameters
    ----------
    mean_model : sklearn estimator
        Fitted or unfitted. Must implement fit(X, y) and predict(X).
        Typical choices: CatBoostRegressor, sklearn GBM, GLM.
    p : float
        Tweedie power. Default: 1.5.
    spread_model : sklearn estimator, optional
        Model for predicting residual spread. Default: CatBoostRegressor with
        Tweedie loss (requires catboost package). Set to any sklearn regressor
        to override.
    min_mu : float
        Lower clip for predicted means. Default: 1e-6.
    min_rho : float
        Lower clip for spread predictions (prevents division by zero). Default: 1e-4.

    References
    ----------
    Manna et al. (2025) ASMBI asmb.70045, Section 3.2.
    """

    def __init__(
        self,
        mean_model: Any,
        p: float = 1.5,
        spread_model: Any = None,
        min_mu: float = 1e-6,
        min_rho: float = 1e-4,
    ) -> None:
        self.mean_model = mean_model
        self.p = float(p)
        self.spread_model = spread_model
        self.min_mu = float(min_mu)
        self.min_rho = float(min_rho)
        self._score = TweedePearsonScore(p=p, min_mu=min_mu)
        self._fitted_mean: Any = None
        self._fitted_spread: Any = None
        self._cal_scores: np.ndarray | None = None
        self._n_cal: int = 0

    def _make_default_spread_model(self) -> Any:
        """Create a default CatBoostRegressor for spread estimation."""
        try:
            from catboost import CatBoostRegressor  # type: ignore[import]
            return CatBoostRegressor(
                loss_function="Tweedie:variance_power=1.5",
                iterations=300,
                learning_rate=0.05,
                depth=4,
                verbose=0,
                random_seed=42,
            )
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for the default spread model. "
                "Install it with: pip install catboost\n"
                "Or pass spread_model= to use a different estimator."
            ) from exc

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TwoStageLWConformal":
        """Fit the mean model and the spread model on training data.

        The spread model is fitted on training Pearson residuals. These are
        in-sample residuals (optimistic) — fitting on a separate split or via
        cross-fitting would reduce overfitting of the spread estimate.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, p)
        y_train : np.ndarray, shape (n_train,)

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        # Stage 1: fit mean model
        self.mean_model.fit(X_train, y_train)
        self._fitted_mean = self.mean_model

        # Stage 2: fit spread model on Pearson residuals
        mu_train = np.clip(
            np.asarray(self._fitted_mean.predict(X_train), dtype=float),
            self.min_mu,
            None,
        )
        pearson_residuals = self._score.score(y_train, mu_train)

        if self.spread_model is None:
            spread = self._make_default_spread_model()
        else:
            spread = self.spread_model

        spread.fit(X_train, pearson_residuals)
        self._fitted_spread = spread
        return self

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> "TwoStageLWConformal":
        """Compute locally weighted nonconformity scores on calibration data.

        Parameters
        ----------
        X_cal : np.ndarray, shape (n_cal, p)
        y_cal : np.ndarray, shape (n_cal,)

        Returns
        -------
        self
        """
        if self._fitted_mean is None:
            raise RuntimeError("Call fit() before calibrate()")
        X_cal = np.asarray(X_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)
        if X_cal.ndim == 1:
            X_cal = X_cal.reshape(-1, 1)

        mu_cal = np.clip(
            np.asarray(self._fitted_mean.predict(X_cal), dtype=float),
            self.min_mu,
            None,
        )
        rho_cal = np.clip(
            np.asarray(self._fitted_spread.predict(X_cal), dtype=float),
            self.min_rho,
            None,
        )
        # Locally weighted Pearson: R* = |y - mu| / (rho * mu^{p/2})
        raw_scores = self._score.score(y_cal, mu_cal)
        self._cal_scores = raw_scores / rho_cal
        self._n_cal = len(self._cal_scores)
        return self

    def predict_interval(
        self, X_new: np.ndarray, alpha: float = 0.005
    ) -> np.ndarray:
        """Compute prediction intervals using the locally weighted Pearson score.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, p)
        alpha : float
            Miscoverage rate. Default: 0.005 (Solvency II 99.5%).

        Returns
        -------
        intervals : np.ndarray, shape (n_new, 2)
        """
        if self._cal_scores is None:
            raise RuntimeError("Call calibrate() before predict_interval()")
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        n_cal = self._n_cal
        # Conformal quantile with finite-sample correction
        k = math.ceil((1.0 - alpha) * (n_cal + 1))
        k = min(k, n_cal)
        q_star = float(np.sort(self._cal_scores)[k - 1])  # 1-indexed k, 0-indexed

        mu_new = np.clip(
            np.asarray(self._fitted_mean.predict(X_new), dtype=float),
            self.min_mu,
            None,
        )
        rho_new = np.clip(
            np.asarray(self._fitted_spread.predict(X_new), dtype=float),
            self.min_rho,
            None,
        )
        margin = q_star * rho_new * mu_new ** (self.p / 2.0)
        lowers = np.maximum(0.0, mu_new - margin)
        uppers = mu_new + margin
        return np.column_stack([lowers, uppers])

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
        mean_name = type(self.mean_model).__name__
        spread_name = (
            type(self._fitted_spread).__name__
            if self._fitted_spread is not None
            else "CatBoostRegressor(default)"
        )
        return (
            f"TwoStageLWConformal(mean={mean_name}, spread={spread_name}, "
            f"p={self.p}, n_cal={self._n_cal})"
        )
