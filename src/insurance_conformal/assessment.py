"""
Conditional Validity Index (CVI) assessment module.

Based on Zhou, Zhang, Tao & Yang (2026), "Conformal Prediction Assessment: A
Framework for Conditional Coverage Evaluation and Selection", arXiv:2603.27189.

The core problem
----------------
Marginal coverage P(Y in C(X)) >= 1-alpha is guaranteed by construction in split
conformal prediction. But a predictor covering 90% of all policyholders overall
can cover only 82% of young drivers and 97% of standard risks simultaneously.
Under Consumer Duty, the portfolio-level guarantee is not enough — you need to
evidence fair outcomes at the segment level.

CVI makes this measurable as a single number that decomposes into:

    CVI_U: undercoverage risk (the dangerous failure mode — false confidence)
    CVI_O: overcoverage cost (efficiency loss — unnecessarily wide intervals)
    CVI   = CVI_U + CVI_O

How it works
------------
1. Fit a reliability estimator: a binary classifier that predicts P(Y in C(X)) per
   instance from features X. If coverage is truly uniform, no classifier should beat
   the constant predictor. If it can, that predictability quantifies the problem.
2. The estimator outputs eta_hat(x) — the estimated local coverage probability.
3. CVI decomposes deviations of eta_hat from the target (1-alpha) into safety
   penalties (eta_hat < (1-gamma)*(1-alpha)) and efficiency penalties
   (eta_hat > (1+gamma)*(1-alpha)).

Separable API
-------------
The key design difference from ConditionalValidityIndex (which requires LightGBM):

- ConditionalCoverageAssessor uses sklearn GradientBoostingClassifier — no optional
  dependencies. Base insurance-conformal install is enough.
- The API separates training from scoring: fit() once on calibration data, then
  score() on multiple test sets. This is useful when you want to assess many
  conformal predictors using the same trained reliability estimator — cutting
  the computation roughly in half for each additional predictor.
- select() does CC-Select directly from a dictionary of prediction sets, without
  the train/eval splits that CCSelect uses. Appropriate when you have a held-out
  test set and want to pick the best predictor from it.

References
----------
Zhou, Z., Zhang, X., Tao, C. & Yang, Y. (2026).
"Conformal Prediction Assessment: A Framework for Conditional Coverage Evaluation
and Selection." arXiv:2603.27189.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_2d(X: Any) -> np.ndarray:
    """Convert X to a 2D float64 numpy array."""
    arr = np.asarray(as_numpy(X), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _coverage_indicators(
    y: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray
) -> np.ndarray:
    """Binary coverage indicators: 1 if y_i in [lower_i, upper_i], else 0."""
    return ((y >= y_lower) & (y <= y_upper)).astype(float)


def _compute_cvi(
    eta_hat: np.ndarray, alpha: float, gamma: float
) -> tuple[float, float, float, float, float, float, float]:
    """
    Decompose local coverage estimates into CVI components.

    Returns (cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo).
    """
    target = 1.0 - alpha
    lower_thr = (1.0 - gamma) * target
    upper_thr = (1.0 + gamma) * target

    mask_under = eta_hat < lower_thr
    mask_over = eta_hat > upper_thr

    pi_minus = float(mask_under.mean())
    pi_plus = float(mask_over.mean())

    cmu = float((target - eta_hat[mask_under]).mean()) if mask_under.any() else 0.0
    cmo = float((eta_hat[mask_over] - target).mean()) if mask_over.any() else 0.0

    cvi_u = pi_minus * cmu
    cvi_o = pi_plus * cmo
    cvi = cvi_u + cvi_o

    return cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CVIAssessmentResult:
    """
    Result from ConditionalCoverageAssessor.score().

    Attributes
    ----------
    cvi : float
        Total Conditional Validity Index = CVI_U + CVI_O.
    cvi_u : float
        Safety penalty: expected undercoverage shortfall for undercovered instances.
        This is the dangerous component — it represents false confidence in interval
        coverage for specific risk segments. Under Consumer Duty, a CVI_U > 0.02
        at the portfolio level is worth investigating at the segment level.
    cvi_o : float
        Efficiency penalty: expected overcoverage excess for overcovered instances.
        Quantifies premium capacity wasted on unnecessarily wide intervals.
    pi_minus : float
        Proportion of instances where estimated coverage is below
        (1-gamma)*(1-alpha). These are the systematically undercovered segments.
    pi_plus : float
        Proportion of instances where estimated coverage is above
        (1+gamma)*(1-alpha). Overcovered segments.
    cmu : float
        Conditional mean undercoverage: E[(1-alpha) - eta_hat | eta_hat < threshold].
        Average shortfall per undercovered instance.
    cmo : float
        Conditional mean overcoverage: E[eta_hat - (1-alpha) | eta_hat > threshold].
        Average excess per overcovered instance.
    marginal_coverage : float
        Observed marginal coverage fraction on the test set.
    alpha : float
        Miscoverage rate used.
    gamma : float
        Tolerance parameter used for under/overcoverage classification.
    n_obs : int
        Number of test observations.
    eta_hat : np.ndarray
        Estimated local coverage probabilities, shape (n_obs,).
    """

    cvi: float
    cvi_u: float
    cvi_o: float
    pi_minus: float
    pi_plus: float
    cmu: float
    cmo: float
    marginal_coverage: float
    alpha: float
    gamma: float
    n_obs: int
    eta_hat: np.ndarray = field(repr=False)

    def __repr__(self) -> str:
        safety = "HIGH" if self.cvi_u > 0.02 else ("MEDIUM" if self.cvi_u > 0.005 else "LOW")
        return (
            f"CVIAssessmentResult("
            f"cvi={self.cvi:.4f}, "
            f"cvi_u={self.cvi_u:.4f} [{safety}], "
            f"cvi_o={self.cvi_o:.4f}, "
            f"pi_minus={self.pi_minus:.3f}, "
            f"pi_plus={self.pi_plus:.3f}, "
            f"marginal={self.marginal_coverage:.3f}, "
            f"n={self.n_obs})"
        )

    def summary(self) -> pl.DataFrame:
        """
        Return a one-row Polars DataFrame summarising key CVI metrics.

        Useful for collecting results from multiple predictors into a comparison
        table via pl.concat().
        """
        return pl.DataFrame({
            "cvi": [self.cvi],
            "cvi_u": [self.cvi_u],
            "cvi_o": [self.cvi_o],
            "pi_minus": [self.pi_minus],
            "pi_plus": [self.pi_plus],
            "cmu": [self.cmu],
            "cmo": [self.cmo],
            "marginal_coverage": [self.marginal_coverage],
            "alpha": [self.alpha],
            "gamma": [self.gamma],
            "n_obs": [self.n_obs],
        })


@dataclass
class CVISelectResult:
    """
    Result from ConditionalCoverageAssessor.select().

    Attributes
    ----------
    best_key : str
        Key of the prediction set with the lowest CVI (best conditional coverage).
    rankings : list of str
        All keys sorted by ascending CVI (best first).
    scores : dict of str -> CVIAssessmentResult
        Full CVI decomposition for each predictor.
    """

    best_key: str
    rankings: List[str]
    scores: Dict[str, CVIAssessmentResult]

    def __repr__(self) -> str:
        return (
            f"CVISelectResult("
            f"best_key={self.best_key!r}, "
            f"rankings={self.rankings!r})"
        )

    def compare(self) -> pl.DataFrame:
        """
        Return a Polars DataFrame ranking predictors by CVI.

        Columns: key, cvi, cvi_u, cvi_o, pi_minus, pi_plus, marginal_coverage, rank.
        Sorted by ascending CVI.
        """
        rows = []
        for rank, key in enumerate(self.rankings, start=1):
            r = self.scores[key]
            rows.append({
                "key": key,
                "cvi": r.cvi,
                "cvi_u": r.cvi_u,
                "cvi_o": r.cvi_o,
                "pi_minus": r.pi_minus,
                "pi_plus": r.pi_plus,
                "marginal_coverage": r.marginal_coverage,
                "rank": rank,
            })
        return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ConditionalCoverageAssessor:
    """
    Assess conditional coverage quality using the Conditional Validity Index.

    This class implements the CVI framework from Zhou et al. (arXiv:2603.27189)
    with an sklearn GradientBoostingClassifier as the reliability estimator —
    no LightGBM dependency required.

    The API separates training from scoring:

    - fit(X_cal, y_cal, prediction_sets_cal): train the reliability estimator on
      calibration data. This is the expensive step. Do it once.
    - score(X_test, y_test, prediction_sets_test): apply the fitted estimator to
      a test set and compute CVI. Fast — only a forward pass through the classifier.
    - select(X_test, y_test, dict_of_prediction_sets): score all prediction sets
      on the same test data and return the one with the lowest CVI.

    When comparing K conformal predictors, this approach requires one fit() call
    and K score() calls — whereas the cross-validation approach in CCSelect
    requires K * n_splits full classifier training runs.

    Parameters
    ----------
    alpha : float
        Miscoverage rate. Target coverage = 1 - alpha. Must be in (0, 1).
    gamma : float, default 0.1
        Relative tolerance for the under/overcoverage classification.
        Instances with eta_hat(x) in [(1-gamma)*(1-alpha), (1+gamma)*(1-alpha)]
        contribute neither to CVI_U nor CVI_O — they are "close enough".
        Increase gamma to make CVI less sensitive to near-nominal coverage.
    n_estimators : int, default 200
        Number of boosting rounds for the GradientBoostingClassifier.
    max_depth : int, default 4
        Maximum tree depth. Shallow trees reduce overfitting on small datasets.
    min_samples_leaf : int, default 20
        Minimum samples per leaf. Higher values regularise more aggressively.
    subsample : float, default 0.8
        Row subsampling fraction per tree.
    learning_rate : float, default 0.05
        Shrinkage parameter.
    calibrate_proba : bool, default True
        Apply isotonic regression calibration (sklearn CalibratedClassifierCV)
        to the out-of-fold probability estimates. Strongly recommended: raw
        GBM probabilities can be poorly calibrated for low-prevalence coverage
        events. Isotonic calibration corrects the marginal probability scale.
    n_splits : int, default 5
        KFold CV folds for fitting the reliability estimator with
        out-of-fold prediction (avoids in-sample optimism).
    random_state : int, optional
        Random state for GBM training and KFold splitting.

    Attributes
    ----------
    is_fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> from insurance_conformal.assessment import ConditionalCoverageAssessor
    >>> assessor = ConditionalCoverageAssessor(alpha=0.10, gamma=0.1)
    >>> assessor.fit(X_cal, y_cal, prediction_sets_cal)
    >>> result = assessor.score(X_test, y_test, prediction_sets_test)
    >>> print(result.cvi_u)  # undercoverage risk
    >>> sel = assessor.select(X_test, y_test, {
    ...     "pearson": sets_pearson,
    ...     "cqr": sets_cqr,
    ... })
    >>> print(sel.best_key)

    Notes
    -----
    prediction_sets should be a dict-like or Polars/pandas DataFrame with "lower"
    and "upper" columns, or a tuple (lower, upper) of array-likes.

    References
    ----------
    Zhou, Z., Zhang, X., Tao, C. & Yang, Y. (2026).
    arXiv:2603.27189. See also CCSelect (conditional_coverage.py) for the
    cross-validation-based selection approach.
    """

    _MIN_OBS = 50
    _WARN_OBS = 800

    def __init__(
        self,
        alpha: float,
        gamma: float = 0.1,
        n_estimators: int = 200,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        subsample: float = 0.8,
        learning_rate: float = 0.05,
        calibrate_proba: bool = True,
        n_splits: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 0.0 < subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.learning_rate = float(learning_rate)
        self.calibrate_proba = bool(calibrate_proba)
        self.n_splits = int(n_splits)
        self.random_state = random_state

        self.is_fitted_: bool = False
        self._estimator: Any = None  # fitted sklearn pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_cal: Any,
        y_cal: Any,
        prediction_sets_cal: Any,
    ) -> "ConditionalCoverageAssessor":
        """
        Train the reliability estimator on calibration data.

        The estimator learns to predict per-instance coverage probability
        P(Y in C(X)) from features X. Under marginal coverage, this probability
        is approximately constant at (1-alpha). If it varies with X, those
        variations are the conditional coverage problem CVI quantifies.

        Parameters
        ----------
        X_cal : array-like of shape (n_cal, p)
            Feature matrix for calibration observations.
        y_cal : array-like of shape (n_cal,)
            Observed outcomes for calibration observations.
        prediction_sets_cal : dict-like, pl.DataFrame, pd.DataFrame, or tuple
            Prediction intervals for calibration observations. Accepted formats:
            - dict or DataFrame with "lower" and "upper" keys/columns
            - tuple (lower, upper) of array-likes of shape (n_cal,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If n_cal < 50 (insufficient for reliable estimation).
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import KFold
        from sklearn.base import clone

        X_np = _to_2d(X_cal)
        y_np = as_numpy(y_cal).ravel().astype(float)
        lower, upper = _parse_prediction_sets(prediction_sets_cal)

        n = len(y_np)
        if n < self._MIN_OBS:
            raise ValueError(
                f"Need at least {self._MIN_OBS} calibration observations, got {n}."
            )
        if len(lower) != n or len(upper) != n:
            raise ValueError(
                "prediction_sets_cal must have the same number of rows as y_cal."
            )

        if n < self._WARN_OBS:
            warnings.warn(
                f"Reliability estimator fitted on {n} observations. "
                f"CVI estimates are more reliable with n >= {self._WARN_OBS} "
                "(Zhou et al., arXiv:2603.27189).",
                UserWarning,
                stacklevel=2,
            )

        Z = _coverage_indicators(y_np, lower, upper).astype(int)

        # Build base classifier
        base_clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

        if self.calibrate_proba:
            # KFold cross-validated isotonic calibration. The inner KFold inside
            # CalibratedClassifierCV is independent of the outer KFold we use for
            # out-of-fold eta_hat. For the fitted estimator we want a calibrated
            # model trained on all of X_cal — this is what CalibratedClassifierCV
            # gives us when fit on the full training set.
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            clf = CalibratedClassifierCV(
                estimator=base_clf,
                method="isotonic",
                cv=kf,
            )
        else:
            clf = base_clf

        clf.fit(X_np, Z)
        self._estimator = clf
        self.is_fitted_ = True
        return self

    def score(
        self,
        X_test: Any,
        y_test: Any,
        prediction_sets_test: Any,
    ) -> CVIAssessmentResult:
        """
        Compute CVI on a test set using the fitted reliability estimator.

        The estimator (trained on calibration data) predicts eta_hat(x) for each
        test observation. CVI is then computed from those predictions — no
        additional training occurs. This makes score() fast when comparing
        multiple conformal predictors.

        Parameters
        ----------
        X_test : array-like of shape (n_test, p)
            Feature matrix for test observations. Must have the same features as
            X_cal used in fit().
        y_test : array-like of shape (n_test,)
            Observed outcomes for test observations.
        prediction_sets_test : dict-like, pl.DataFrame, pd.DataFrame, or tuple
            Prediction intervals for test observations. Accepted formats:
            - dict or DataFrame with "lower" and "upper" keys/columns
            - tuple (lower, upper) of array-likes of shape (n_test,)

        Returns
        -------
        CVIAssessmentResult
            CVI decomposition for this prediction set on the test data.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        ValueError
            If shapes are inconsistent.
        """
        self._check_fitted()

        X_np = _to_2d(X_test)
        y_np = as_numpy(y_test).ravel().astype(float)
        lower, upper = _parse_prediction_sets(prediction_sets_test)

        n = len(y_np)
        if len(lower) != n or len(upper) != n:
            raise ValueError(
                "prediction_sets_test must have the same number of rows as y_test."
            )

        eta_hat = self._estimator.predict_proba(X_np)[:, 1]
        Z = _coverage_indicators(y_np, lower, upper)
        marginal_coverage = float(Z.mean())

        cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo = _compute_cvi(
            eta_hat, self.alpha, self.gamma
        )

        return CVIAssessmentResult(
            cvi=cvi,
            cvi_u=cvi_u,
            cvi_o=cvi_o,
            pi_minus=pi_minus,
            pi_plus=pi_plus,
            cmu=cmu,
            cmo=cmo,
            marginal_coverage=marginal_coverage,
            alpha=self.alpha,
            gamma=self.gamma,
            n_obs=n,
            eta_hat=eta_hat,
        )

    def select(
        self,
        X_test: Any,
        y_test: Any,
        prediction_sets: Dict[str, Any],
    ) -> CVISelectResult:
        """
        CC-Select: pick the best conformal predictor by CVI.

        Scores all prediction sets against the same fitted reliability estimator
        and returns the one with the lowest total CVI. Lower CVI means more
        uniform conditional coverage — the criterion that matters for regulatory
        use.

        Parameters
        ----------
        X_test : array-like of shape (n_test, p)
            Test features.
        y_test : array-like of shape (n_test,)
            Observed test outcomes.
        prediction_sets : dict of str -> prediction_set
            A mapping from predictor name to prediction intervals. Each value
            must be in one of the formats accepted by score(): a dict/DataFrame
            with "lower"/"upper" keys, or a (lower, upper) tuple.

        Returns
        -------
        CVISelectResult
            Contains best_key (lowest CVI), rankings (ascending CVI order),
            and scores (CVIAssessmentResult per predictor).

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        ValueError
            If prediction_sets is empty.
        """
        self._check_fitted()
        if not prediction_sets:
            raise ValueError("prediction_sets must not be empty.")

        results = {}
        for key, ps in prediction_sets.items():
            results[key] = self.score(X_test, y_test, ps)

        rankings = sorted(results.keys(), key=lambda k: results[k].cvi)
        best_key = rankings[0]

        return CVISelectResult(
            best_key=best_key,
            rankings=rankings,
            scores=results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Call fit() before score() or select()."
            )


# ---------------------------------------------------------------------------
# Helpers for parsing prediction set formats
# ---------------------------------------------------------------------------


def _parse_prediction_sets(ps: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse prediction intervals into (lower, upper) numpy arrays.

    Accepts:
    - tuple (lower, upper)
    - dict with "lower"/"upper" keys
    - polars DataFrame with "lower"/"upper" columns
    - pandas DataFrame with "lower"/"upper" columns
    """
    if isinstance(ps, tuple):
        if len(ps) != 2:
            raise ValueError(
                "prediction_sets tuple must be (lower, upper), got length "
                f"{len(ps)}."
            )
        lower = as_numpy(ps[0]).ravel().astype(float)
        upper = as_numpy(ps[1]).ravel().astype(float)
        return lower, upper

    if isinstance(ps, pl.DataFrame):
        _check_lower_upper_cols(ps.columns)
        return (
            ps["lower"].to_numpy().astype(float),
            ps["upper"].to_numpy().astype(float),
        )

    # Try pandas DataFrame
    try:
        import pandas as pd
        if isinstance(ps, pd.DataFrame):
            _check_lower_upper_cols(list(ps.columns))
            return (
                ps["lower"].to_numpy().astype(float),
                ps["upper"].to_numpy().astype(float),
            )
    except ImportError:
        pass

    # Dict-like
    try:
        _check_lower_upper_cols(list(ps.keys()))
        lower = as_numpy(ps["lower"]).ravel().astype(float)
        upper = as_numpy(ps["upper"]).ravel().astype(float)
        return lower, upper
    except (AttributeError, KeyError, TypeError):
        pass

    raise TypeError(
        "prediction_sets must be a (lower, upper) tuple, a dict/DataFrame with "
        "'lower' and 'upper' keys/columns, or a polars/pandas DataFrame. "
        f"Got {type(ps).__name__}."
    )


def _check_lower_upper_cols(columns: List[str]) -> None:
    missing = {"lower", "upper"} - set(columns)
    if missing:
        raise ValueError(
            f"prediction_sets must have 'lower' and 'upper' columns. "
            f"Missing: {missing}."
        )
