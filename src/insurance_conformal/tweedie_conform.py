"""
TweedieConformPredictor: unified split conformal prediction for Tweedie insurance models.

This class consolidates the score infrastructure from insurance_conformal.claims.tweedie
and the locally-weighted spread model from insurance_conformal.locally_weighted into a
single, clean API. The design goal is a single entry point for practitioners who want
distribution-free prediction intervals on Tweedie GLM or GBM predictions.

Why a unified class rather than leaving the pieces separate?
The existing code has three different score classes (TweedePearsonScore,
TweedieDevianceScore, TweedieAnscombeScore), a two-stage locally weighted class
(TwoStageLWConformal, LocallyWeightedConformal), and InsuranceConformalPredictor.
A pricing analyst should not have to assemble these. TweedieConformPredictor gives
them a single class that handles all four score types, exposure weighting, and
score selection — and stays internally consistent.

Key design decisions:
- predict_interval() returns a (n, 2) numpy array (not Polars DataFrame) because
  this class is built for downstream pipeline integration.
- score='lw_pearson' triggers the two-stage spread model (delegates to the
  same _resolve_backend / GBM fitting logic as LocallyWeightedConformal).
- exposure_weighted modifies the score denominator to (e*mu)^{p/2}, which is
  appropriate when exposure varies substantially across policies and the model
  predicts the rate (mu_hat per unit exposure) not the total expected loss.
- select_score() evaluates all candidate scores on a validation set and returns
  the one with the shortest mean interval width at the specified alpha.
- coverage_diagnostic() breaks coverage by decile of mu_hat and by quartile of
  the spread model prediction (rho_hat), matching LocallyWeightedConformal.coverage_diagnostics().

References
----------
Manna et al. (2025) ASMBI asmb.70045 — Tweedie conformal scores
arXiv 2507.06921 — locally weighted Pearson, LightGBM spread model
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

from insurance_conformal.claims.tweedie import (
    TweedePearsonScore,
    TweedieDevianceScore,
    TweedieAnscombeScore,
)
from insurance_conformal.locally_weighted import _resolve_backend
from insurance_conformal.utils import as_numpy, conformal_quantile


_VALID_SCORES = ("pearson", "lw_pearson", "deviance", "anscombe")
_VALID_BACKENDS = ("auto", "catboost", "lightgbm", "sklearn")


class TweedieConformPredictor:
    """
    Unified conformal prediction intervals for Tweedie insurance models.

    Wraps any fitted sklearn-compatible Tweedie/Poisson/Gamma model and produces
    distribution-free prediction intervals via split conformal prediction. Supports
    four non-conformity score types, an optional two-stage locally-weighted spread
    model, and exposure weighting.

    The split conformal guarantee: P(y_test in interval) >= 1 - alpha for
    exchangeable data, regardless of the Tweedie distribution parameters or model
    misspecification.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Must implement predict(X) -> array of shape (n,). Predictions should be
        the expected loss (mu_hat), positive-valued. For offset/exposure models,
        predictions should already incorporate the exposure.
    p : float, default 1.5
        Tweedie variance power. Var(Y|X) ~ mu^p. Common values:
        1.0 (Poisson), 1.5 (compound Poisson-Gamma), 2.0 (Gamma).
    score : str, default 'pearson'
        Non-conformity score type. One of:
        - 'pearson': |y - mu| / mu^{p/2}. Fast, closed-form inversion.
          Appropriate for well-specified Tweedie models.
        - 'lw_pearson': |y - mu| / (mu^{p/2} * rho_hat(x)). Requires fitting
          a secondary spread model on training data. Produces adaptive-width
          intervals — narrower where the model is confident and the residual
          spread is low. Use when heteroscedasticity is not fully captured by
          the Tweedie variance function.
        - 'deviance': sqrt(Tweedie unit deviance). Exact score for the Tweedie
          family; asymmetric intervals. Slower to invert numerically (brentq
          per observation).
        - 'anscombe': Anscombe variance-stabilising residual. Similar to deviance
          but faster (closed-form inversion). Undefined at p=2.
    spread_backend : str, default 'auto'
        Backend for the lw_pearson spread model. One of 'auto', 'catboost',
        'lightgbm', 'sklearn'. 'auto' tries catboost -> lightgbm -> sklearn.
        Ignored unless score='lw_pearson'.
    exposure_weighted : bool, default False
        If True, adjust the non-conformity score denominator for exposure:
        R_i = |y_i - e_i * mu_i| / (e_i * mu_i)^{p/2}
        where e_i is the policy exposure. The model should predict the rate
        (mu_hat per unit exposure), not the expected count/amount. Pass the
        exposure to fit(), calibrate(), and predict_interval().
    min_mu : float, default 1e-6
        Minimum clip for predicted means to avoid division by zero.

    Attributes
    ----------
    spread_model_ : fitted model or None
        The fitted secondary spread model. Set after fit() when score='lw_pearson'.
    backend_ : str or None
        Resolved backend for the spread model (e.g. 'lightgbm' when backend='auto').
    cal_scores_ : np.ndarray or None
        Non-conformity scores from the calibration set. Set after calibrate().
    n_calibration_ : int
        Number of calibration observations.
    is_calibrated_ : bool
        Whether calibrate() has been called successfully.

    Examples
    --------
    Basic usage with Pearson score:

    >>> tcp = TweedieConformPredictor(model=fitted_glm, p=1.5, score='pearson')
    >>> tcp.calibrate(X_cal, y_cal)
    >>> intervals = tcp.predict_interval(X_test, alpha=0.05)
    >>> # intervals.shape == (n_test, 2); intervals[:, 0] are lower bounds

    With locally-weighted score (requires training data):

    >>> tcp = TweedieConformPredictor(model=fitted_gbm, p=1.5, score='lw_pearson')
    >>> tcp.fit(X_train, y_train)
    >>> tcp.calibrate(X_cal, y_cal)
    >>> intervals = tcp.predict_interval(X_test, alpha=0.10)

    With exposure weighting (rate models):

    >>> tcp = TweedieConformPredictor(model=rate_model, p=1.5, exposure_weighted=True)
    >>> tcp.calibrate(X_cal, y_cal, exposure_cal=exposure_cal)
    >>> intervals = tcp.predict_interval(X_test, alpha=0.05, exposure_new=exposure_test)

    Score selection (choose narrowest-width score automatically):

    >>> best_score = tcp.select_score(X_val, y_val, alpha=0.05)
    >>> print(best_score)  # e.g. 'lw_pearson'
    """

    def __init__(
        self,
        model: Any,
        p: float = 1.5,
        score: str = "pearson",
        spread_backend: str = "auto",
        exposure_weighted: bool = False,
        min_mu: float = 1e-6,
    ) -> None:
        if score not in _VALID_SCORES:
            raise ValueError(
                f"score must be one of {_VALID_SCORES}, got {score!r}"
            )
        if spread_backend not in _VALID_BACKENDS:
            raise ValueError(
                f"spread_backend must be one of {_VALID_BACKENDS}, got {spread_backend!r}"
            )
        if min_mu <= 0:
            raise ValueError(f"min_mu must be positive, got {min_mu}")
        if score == "anscombe" and abs(float(p) - 2.0) < 1e-9:
            raise ValueError(
                "Anscombe score is undefined for p=2 (exponent is 0; "
                "use score='deviance' instead)."
            )

        self.model = model
        self.p = float(p)
        self.score = score
        self.spread_backend = spread_backend
        self.exposure_weighted = bool(exposure_weighted)
        self.min_mu = float(min_mu)

        # State set during fit / calibrate
        self.spread_model_: Any = None
        self.backend_: Optional[str] = None
        self.cal_scores_: Optional[np.ndarray] = None
        self.cal_quantiles_: dict[float, float] = {}
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

        # Internal score objects — built once in _build_score_objects()
        self._pearson = TweedePearsonScore(p=self.p, min_mu=self.min_mu)
        self._deviance = TweedieDevianceScore(p=self.p, min_mu=self.min_mu)
        if abs(self.p - 2.0) > 1e-9:
            self._anscombe: Optional[TweedieAnscombeScore] = TweedieAnscombeScore(
                p=self.p, min_mu=self.min_mu
            )
        else:
            self._anscombe = None  # undefined at p=2

        # spread model params (same defaults as LocallyWeightedConformal)
        self._catboost_params: dict = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }
        self._lgbm_params: dict = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "objective": "regression",
            "random_state": 42,
            "verbose": -1,
        }
        self._sklearn_params: dict = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "random_state": 42,
        }
        self._clip_spread_at: float = 0.01

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        exposure_train: Optional[Any] = None,
    ) -> "TweedieConformPredictor":
        """
        Fit the secondary spread model (only required when score='lw_pearson').

        For all other scores this is a no-op and does not need to be called.
        The base model (self.model) must already be fitted before calling this.

        The spread model learns rho_hat(X) = E[|Pearson residual| | X] on the
        training data. This is used in predict_interval to produce adaptive-width
        intervals: wider for observations where the residual spread is typically
        large, narrower where it is small.

        Parameters
        ----------
        X_train : array-like of shape (n, p)
            Training features. Must be the same data used to fit the base model.
            Do NOT use calibration data here.
        y_train : array-like of shape (n,)
            Training targets.
        exposure_train : array-like of shape (n,) or None
            Training exposure. Required when exposure_weighted=True.

        Returns
        -------
        self
        """
        if self.score != "lw_pearson":
            # No-op for non-LW scores
            return self

        y = as_numpy(y_train)
        mu_hat = self._base_predict(X_train)
        X_np = self._to_numpy(X_train)

        if self.exposure_weighted:
            exposure = self._validate_exposure(exposure_train, len(y), "fit")
            effective_mu = exposure * mu_hat
        else:
            effective_mu = mu_hat

        effective_mu = np.clip(effective_mu, self.min_mu, None)
        pearson_resid = np.abs(y - effective_mu) / effective_mu ** (self.p / 2.0)

        resolved = _resolve_backend(self.spread_backend)
        self.backend_ = resolved
        self.spread_model_ = self._fit_spread_model(resolved, X_np, pearson_resid)

        return self

    def calibrate(
        self,
        X_cal: Any,
        y_cal: Any,
        exposure_cal: Optional[Any] = None,
    ) -> "TweedieConformPredictor":
        """
        Compute non-conformity scores on the calibration set.

        For score='lw_pearson', fit() must be called first.

        Parameters
        ----------
        X_cal : array-like of shape (n, p)
            Calibration features. Must be independent of training data.
        y_cal : array-like of shape (n,)
            Calibration targets.
        exposure_cal : array-like of shape (n,) or None
            Calibration exposure. Required when exposure_weighted=True.

        Returns
        -------
        self
        """
        if self.score == "lw_pearson" and self.spread_model_ is None:
            raise RuntimeError(
                "score='lw_pearson' requires fit() to be called before calibrate(). "
                "Call .fit(X_train, y_train) first."
            )

        y = as_numpy(y_cal)
        mu_hat = self._base_predict(X_cal)

        if len(y) != len(mu_hat):
            raise ValueError(
                f"X_cal has {len(mu_hat)} rows but y_cal has {len(y)} — lengths must match"
            )

        if self.exposure_weighted:
            exposure = self._validate_exposure(exposure_cal, len(y), "calibrate")
            effective_y = y
            effective_mu = exposure * mu_hat
        else:
            effective_y = y
            effective_mu = mu_hat

        self.cal_scores_ = self._compute_scores(
            effective_y, effective_mu, X_cal
        )
        self.n_calibration_ = len(self.cal_scores_)
        self.cal_quantiles_ = {}  # clear cache when recalibrating
        self.is_calibrated_ = True

        # Store raw arrays for select_score() to recompute scores under
        # alternative score types without re-running the base model.
        self._cal_y: np.ndarray = effective_y
        self._cal_mu: np.ndarray = effective_mu
        self._cal_X: np.ndarray = self._to_numpy(X_cal)

        return self

    def predict_interval(
        self,
        X_new: Any,
        alpha: float = 0.05,
        exposure_new: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Produce conformal prediction intervals for new observations.

        Parameters
        ----------
        X_new : array-like of shape (n, p)
            New features to predict.
        alpha : float, default 0.05
            Miscoverage rate. alpha=0.05 gives 95% prediction intervals.
        exposure_new : array-like of shape (n,) or None
            Exposure for new observations. Required when exposure_weighted=True.

        Returns
        -------
        intervals : np.ndarray of shape (n, 2)
            intervals[:, 0] are lower bounds (clipped at 0).
            intervals[:, 1] are upper bounds.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_calibrated()

        q_star = self._get_quantile(alpha)
        mu_hat = self._base_predict(X_new)
        n = len(mu_hat)

        if self.exposure_weighted:
            exposure = self._validate_exposure(exposure_new, n, "predict_interval")
            effective_mu = np.clip(exposure * mu_hat, self.min_mu, None)
        else:
            exposure = None
            effective_mu = np.clip(mu_hat, self.min_mu, None)

        if self.score == "lw_pearson":
            self._check_fitted()
            rho_hat = self._spread_predict(X_new)
            margin = q_star * effective_mu ** (self.p / 2.0) * rho_hat
            lower = np.maximum(0.0, effective_mu - margin)
            upper = effective_mu + margin
        elif self.score == "pearson":
            lower = self._pearson.inverse(q_star, effective_mu, upper=False)
            upper = self._pearson.inverse(q_star, effective_mu, upper=True)
        elif self.score == "deviance":
            lower = self._deviance.inverse(q_star, effective_mu, upper=False)
            upper = self._deviance.inverse(q_star, effective_mu, upper=True)
        elif self.score == "anscombe":
            if self._anscombe is None:
                raise ValueError("Anscombe score is undefined for p=2. Use score='deviance'.")
            lower = self._anscombe.inverse(q_star, effective_mu, upper=False)
            upper = self._anscombe.inverse(q_star, effective_mu, upper=True)
        else:
            raise ValueError(f"Unknown score: {self.score!r}")

        return np.column_stack([lower, upper])

    def select_score(
        self,
        X_val: Any,
        y_val: Any,
        alpha: float = 0.05,
        candidates: Optional[list[str]] = None,
    ) -> str:
        """
        Evaluate candidate scores on a validation set and return the best by interval width.

        For each candidate score type, this method re-calibrates a copy of this
        predictor with that score on the calibration data already stored, then
        evaluates mean interval width on the validation set. The score with the
        shortest mean width (while achieving >= 1-alpha coverage) wins.

        This requires that calibrate() has already been called.

        Parameters
        ----------
        X_val : array-like of shape (n, p)
            Validation features. Must be independent of calibration data.
        y_val : array-like of shape (n,)
            Validation targets.
        alpha : float, default 0.05
            Target coverage level.
        candidates : list of str or None
            Score types to evaluate. Defaults to all applicable types:
            ['pearson', 'deviance', 'anscombe'] (lw_pearson is included only
            if the spread model is already fitted).

        Returns
        -------
        str
            The score type with the best (narrowest) mean interval width
            among candidates achieving >= 1-alpha coverage.
        """
        self._check_calibrated()

        if candidates is None:
            candidates = ["pearson", "deviance"]
            if abs(self.p - 2.0) > 1e-9:
                candidates.append("anscombe")
            if self.spread_model_ is not None:
                candidates.append("lw_pearson")

        y = as_numpy(y_val)
        results: list[tuple[str, float, float]] = []  # (score, coverage, mean_width)

        for candidate in candidates:
            try:
                # Build a temporary predictor with this score
                tmp = self._clone_with_score(candidate)
                intervals = tmp.predict_interval(X_val, alpha=alpha)
                lower = intervals[:, 0]
                upper = intervals[:, 1]
                coverage = float(np.mean((y >= lower) & (y <= upper)))
                mean_width = float(np.mean(upper - lower))
                results.append((candidate, coverage, mean_width))
            except Exception as exc:
                warnings.warn(
                    f"select_score: score={candidate!r} raised {type(exc).__name__}: {exc}. "
                    "Skipping.",
                    UserWarning,
                    stacklevel=2,
                )

        if not results:
            raise RuntimeError("No candidate scores could be evaluated.")

        # Among scores that achieve the target coverage, pick narrowest width.
        # If none achieve it (can happen with very small calibration sets), fall
        # back to the one with coverage closest to target.
        valid = [(s, w) for s, cov, w in results if cov >= 1 - alpha]
        if valid:
            best = min(valid, key=lambda x: x[1])[0]
        else:
            best = min(results, key=lambda x: abs(x[1] - (1 - alpha)))[0]

        return best

    def coverage_diagnostic(
        self,
        X_test: Any,
        y_test: Any,
        alpha: float = 0.05,
    ) -> dict:
        """
        Compute coverage diagnostics on a test set.

        Returns empirical coverage, mean width, coverage by decile of mu_hat,
        and (if score='lw_pearson') coverage by quartile of rho_hat.

        Parameters
        ----------
        X_test : array-like of shape (n, p)
            Test features.
        y_test : array-like of shape (n,)
            Observed values.
        alpha : float, default 0.05
            Target miscoverage rate.

        Returns
        -------
        dict with keys:
            empirical_coverage : float
            mean_width : float
            by_decile : list of dicts
                Each dict has keys: decile, n_obs, mean_predicted, coverage.
            by_spread_quartile : list of dicts or None
                Coverage by rho_hat quartile. None when score != 'lw_pearson'.
        """
        import pandas as pd

        self._check_calibrated()
        y = as_numpy(y_test)
        intervals = self.predict_interval(X_test, alpha=alpha)
        lower = intervals[:, 0]
        upper = intervals[:, 1]
        covered = (y >= lower) & (y <= upper)
        widths = upper - lower
        mu_hat = self._base_predict(X_test)

        empirical_coverage = float(covered.mean())
        mean_width = float(widths.mean())

        # Coverage by decile of mu_hat
        try:
            decile_labels = pd.qcut(mu_hat, q=10, labels=False, duplicates="drop")
        except ValueError:
            decile_labels = pd.cut(mu_hat, bins=10, labels=False, duplicates="drop")

        decile_labels = np.asarray(decile_labels, dtype=float)
        by_decile = []
        for d in np.unique(decile_labels[~np.isnan(decile_labels)]):
            mask = decile_labels == d
            by_decile.append(
                {
                    "decile": int(d) + 1,
                    "n_obs": int(mask.sum()),
                    "mean_predicted": float(mu_hat[mask].mean()),
                    "coverage": float(covered[mask].mean()),
                }
            )

        # Coverage by spread quartile (only for lw_pearson)
        by_spread_quartile = None
        if self.score == "lw_pearson" and self.spread_model_ is not None:
            rho_hat = self._spread_predict(X_test)
            edges = np.percentile(rho_hat, [0, 25, 50, 75, 100])
            edges[-1] += 1e-10
            spread_bin = np.digitize(rho_hat, edges[1:], right=False)
            spread_bin = np.clip(spread_bin, 0, 3)
            by_spread_quartile = []
            for q in range(4):
                mask = spread_bin == q
                if mask.sum() == 0:
                    continue
                by_spread_quartile.append(
                    {
                        "spread_quartile": q + 1,
                        "n_obs": int(mask.sum()),
                        "mean_spread": float(rho_hat[mask].mean()),
                        "coverage": float(covered[mask].mean()),
                    }
                )

        return {
            "empirical_coverage": empirical_coverage,
            "mean_width": mean_width,
            "by_decile": by_decile,
            "by_spread_quartile": by_spread_quartile,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _base_predict(self, X: Any) -> np.ndarray:
        """Run the base model and return a clipped 1D numpy array."""
        X_np = self._to_numpy(X)
        mu = as_numpy(self.model.predict(X_np)).ravel()
        return np.clip(mu, self.min_mu, None)

    def _spread_predict(self, X: Any) -> np.ndarray:
        """Run the spread model and return clipped 1D numpy array."""
        X_np = self._to_numpy(X)
        rho_raw = as_numpy(self.spread_model_.predict(X_np)).ravel()
        return np.clip(rho_raw, self._clip_spread_at, None)

    def _to_numpy(self, X: Any) -> np.ndarray:
        """Convert DataFrames to numpy for sklearn-compatible models."""
        import pandas as pd
        import polars as pl
        if isinstance(X, pl.DataFrame):
            return X.to_numpy()
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.asarray(X)

    def _compute_scores(
        self,
        y: np.ndarray,
        effective_mu: np.ndarray,
        X: Any,
    ) -> np.ndarray:
        """Compute non-conformity scores for the configured score type."""
        effective_mu = np.clip(effective_mu, self.min_mu, None)

        if self.score == "pearson":
            return self._pearson.score(y, effective_mu)
        elif self.score == "lw_pearson":
            self._check_fitted()
            rho_hat = self._spread_predict(X)
            raw = self._pearson.score(y, effective_mu)
            return raw / rho_hat
        elif self.score == "deviance":
            return self._deviance.score(y, effective_mu)
        elif self.score == "anscombe":
            if self._anscombe is None:
                raise ValueError("Anscombe score is undefined for p=2. Use score='deviance'.")
            return self._anscombe.score(y, effective_mu)
        else:
            raise ValueError(f"Unknown score: {self.score!r}")

    def _fit_spread_model(
        self, resolved_backend: str, X_np: np.ndarray, targets: np.ndarray
    ) -> Any:
        """Fit and return the spread model using the resolved backend."""
        if resolved_backend == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as e:
                raise ImportError(
                    "spread_backend='catboost' requires CatBoost. "
                    "Install with: pip install 'insurance-conformal[catboost]'."
                ) from e
            spread = CatBoostRegressor(**self._catboost_params)
            spread.fit(X_np, targets)
        elif resolved_backend == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as e:
                raise ImportError(
                    "spread_backend='lightgbm' requires LightGBM. "
                    "Install with: pip install 'insurance-conformal[lightgbm]'."
                ) from e
            spread = LGBMRegressor(**self._lgbm_params)
            spread.fit(X_np, targets)
        elif resolved_backend == "sklearn":
            from sklearn.ensemble import GradientBoostingRegressor
            spread = GradientBoostingRegressor(**self._sklearn_params)
            spread.fit(X_np, targets)
        else:
            raise ValueError(f"Unknown resolved backend: {resolved_backend!r}")
        return spread

    def _get_quantile(self, alpha: float) -> float:
        """Return (and cache) the conformal quantile for alpha."""
        self._check_calibrated()
        if alpha not in self.cal_quantiles_:
            self.cal_quantiles_[alpha] = conformal_quantile(self.cal_scores_, alpha)
        return self.cal_quantiles_[alpha]

    def _clone_with_score(self, new_score: str) -> "TweedieConformPredictor":
        """
        Return a shallow copy of this predictor with a different score type.

        Used internally by select_score(). The clone shares the fitted model,
        spread model, and calibration scores but uses a different score function.
        Calibration scores are recomputed from the stored calibration predictions
        is not possible without storing them, so instead we compute a fresh clone
        that stores cal_scores_ from the *current* calibration data.

        Implementation note: we can't easily recompute cal_scores_ for a different
        score without storing the raw (y_cal, mu_cal) pairs. Instead, we create a
        proxy predictor that uses the *current* predictor's quantile infrastructure
        with a different inversion — but we do need properly calibrated scores.

        The pragmatic solution: the clone's cal_scores_ are *approximated* by
        re-applying the new score formula to self.cal_scores_ (which are Pearson
        residuals / rho_hat for lw_pearson). This is only valid for score selection,
        not for the final predictor — hence why select_score() returns a *name*,
        not a new predictor instance.

        Actually, that approximation would be wrong. The correct approach is to
        store (y_cal, mu_cal) so we can recompute. We do this by saving these
        arrays at calibrate() time in _cal_y and _cal_mu attributes.
        """
        if not hasattr(self, "_cal_y") or not hasattr(self, "_cal_mu"):
            raise RuntimeError(
                "select_score() requires that calibrate() was called on this instance. "
                "Internal _cal_y / _cal_mu arrays are not available."
            )

        clone = TweedieConformPredictor(
            model=self.model,
            p=self.p,
            score=new_score,
            spread_backend=self.spread_backend,
            exposure_weighted=self.exposure_weighted,
            min_mu=self.min_mu,
        )
        # Share the spread model (read-only)
        clone.spread_model_ = self.spread_model_
        clone.backend_ = self.backend_

        # Re-compute cal scores under new score type using stored arrays
        clone.cal_scores_ = clone._compute_scores(
            self._cal_y, self._cal_mu, self._cal_X
        )
        clone.n_calibration_ = len(clone.cal_scores_)
        clone.is_calibrated_ = True

        return clone

    @staticmethod
    def _validate_exposure(
        exposure: Any, expected_len: int, context: str
    ) -> np.ndarray:
        """Validate and return exposure as a numpy array."""
        if exposure is None:
            raise ValueError(
                f"exposure_weighted=True but no exposure provided to {context}(). "
                "Pass exposure_cal= (or exposure_train= / exposure_new=) explicitly."
            )
        e = np.asarray(exposure, dtype=float)
        if len(e) != expected_len:
            raise ValueError(
                f"exposure has {len(e)} elements but data has {expected_len} — "
                "lengths must match."
            )
        if np.any(e <= 0):
            raise ValueError("All exposure values must be positive.")
        return e

    def _check_fitted(self) -> None:
        if self.spread_model_ is None:
            raise RuntimeError(
                "spread_model_ is not fitted. "
                "Call .fit(X_train, y_train) before .calibrate() when score='lw_pearson'."
            )

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Predictor has not been calibrated. "
                "Call .calibrate(X_cal, y_cal) first."
            )

    def __repr__(self) -> str:
        status = (
            f"calibrated on {self.n_calibration_} obs"
            if self.is_calibrated_
            else "not calibrated"
        )
        exp_str = ", exposure_weighted" if self.exposure_weighted else ""
        backend_str = ""
        if self.score == "lw_pearson" and self.backend_ is not None:
            backend_str = f", backend={self.backend_!r}"
        return (
            f"TweedieConformPredictor("
            f"p={self.p}, "
            f"score={self.score!r}"
            f"{backend_str}"
            f"{exp_str}, "
            f"{status})"
        )
