"""
Conformal Prediction Assessment (CPA) — conditional coverage evaluation.

Based on Zhou, Zhang, Tao & Yang (2026), "Conformal Prediction Assessment:
A Framework for Conditional Coverage Evaluation and Selection",
arXiv:2603.27189.

Why this module exists
-----------------------
Standard conformal prediction gives marginal coverage: 90% of all policyholders
get covered. That hides a dangerous pattern — 82% of young drivers covered while
97% of standard risks are covered simultaneously. The portfolio-level guarantee
is met but specific segments are systematically under-protected.

Under FCA Consumer Duty (PS22/9) and EP25/2 (proxy discrimination), a pricing
team cannot simply report marginal coverage. They need to evidence that coverage
is reasonably uniform across protected and proxy characteristics.

CPA makes conditional coverage measurable with two numbers:

    CVI_U: undercoverage risk. Expected shortfall for under-protected segments.
    CVI_O: overcoverage cost. Expected excess for over-protected segments.
    CVI   = CVI_U + CVI_O

The key mechanism: fit a classifier that predicts P(Y in C(X) | X) from
features. If coverage is truly uniform, no classifier can do better than
the constant predictor. If it can, the predictability quantifies the problem.

Classes
-------
ReliabilityEstimator
    Standalone sklearn-compatible estimator for per-instance coverage
    probability. Accepts any sklearn binary classifier. Can be used
    independently from CPASelector — useful for segment-level reporting
    and visualisation.

CPASelector
    Select the best conformal predictor from a set using CVI scores.
    Takes pre-computed prediction intervals (not predictor objects).
    Builds on ReliabilityEstimator internally.

Result types
------------
CPAResult
    CVI decomposition for a single prediction set.
CPASelectResult
    Selection result across multiple prediction sets.

Relationship to other modules
------------------------------
- conditional_coverage.ConditionalValidityIndex: LightGBM-based CVI. Use this
  when you have lightgbm installed and want maximum predictive power.
- assessment.ConditionalCoverageAssessor: sklearn GBM-based CVI with a
  separable fit/score/select API. Use this when you want to fit once and
  score multiple test sets.
- cpa.ReliabilityEstimator + cpa.CPASelector: any sklearn classifier, most
  flexible. Use this when you want logistic regression (interpretable,
  fast, good starting point) or want to inject your own fitted classifier.

References
----------
Zhou, Z., Zhang, X., Tao, C. & Yang, Y. (2026).
"Conformal Prediction Assessment: A Framework for Conditional Coverage
Evaluation and Selection." arXiv:2603.27189.

FCA PS22/9 — Consumer Duty: fair value and outcomes.
FCA EP25/2 — Proxy discrimination in general insurance pricing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from insurance_conformal.utils import as_numpy


# ---------------------------------------------------------------------------
# Helpers shared across this module
# ---------------------------------------------------------------------------


def _to_2d(X: Any) -> np.ndarray:
    """Convert X to a 2D float64 numpy array."""
    arr = np.asarray(as_numpy(X), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _coverage_indicators(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Binary coverage: 1 if y_i in [lower_i, upper_i], else 0."""
    return ((y >= lower) & (y <= upper)).astype(float)


def _parse_intervals(ps: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse prediction intervals into (lower, upper) numpy arrays.

    Accepted formats:
    - (lower, upper) tuple of array-likes
    - dict with "lower" and "upper" keys
    - Polars DataFrame with "lower" and "upper" columns
    - pandas DataFrame with "lower" and "upper" columns
    """
    if isinstance(ps, tuple):
        if len(ps) != 2:
            raise ValueError(
                f"Interval tuple must be (lower, upper), got length {len(ps)}."
            )
        return (
            as_numpy(ps[0]).ravel().astype(float),
            as_numpy(ps[1]).ravel().astype(float),
        )

    try:
        import polars as pl
        if isinstance(ps, pl.DataFrame):
            _require_cols(ps.columns, {"lower", "upper"})
            return (
                ps["lower"].to_numpy().astype(float),
                ps["upper"].to_numpy().astype(float),
            )
    except ImportError:
        pass

    try:
        import pandas as pd
        if isinstance(ps, pd.DataFrame):
            _require_cols(list(ps.columns), {"lower", "upper"})
            return (
                ps["lower"].to_numpy().astype(float),
                ps["upper"].to_numpy().astype(float),
            )
    except ImportError:
        pass

    # Dict-like fallback
    try:
        _require_cols(list(ps.keys()), {"lower", "upper"})
        return (
            as_numpy(ps["lower"]).ravel().astype(float),
            as_numpy(ps["upper"]).ravel().astype(float),
        )
    except (AttributeError, KeyError, TypeError):
        pass

    raise TypeError(
        "prediction_intervals must be a (lower, upper) tuple, a dict with "
        "'lower'/'upper' keys, or a DataFrame with 'lower'/'upper' columns. "
        f"Got {type(ps).__name__}."
    )


def _require_cols(columns: List[str], required: set) -> None:
    missing = required - set(columns)
    if missing:
        raise ValueError(
            f"prediction_intervals is missing required column(s): {sorted(missing)}."
        )


def _compute_cvi(
    eta_hat: np.ndarray,
    alpha: float,
    gamma: float,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Decompose local coverage estimates into CVI components.

    CVI_U = pi_minus * E[(1-alpha) - eta | eta < (1-gamma)*(1-alpha)]
    CVI_O = pi_plus  * E[eta - (1-alpha) | eta > (1+gamma)*(1-alpha)]
    CVI   = CVI_U + CVI_O

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
class CPAResult:
    """
    CVI decomposition for a single set of prediction intervals.

    Attributes
    ----------
    cvi : float
        Total Conditional Validity Index = CVI_U + CVI_O. Lower is better.
    cvi_u : float
        Safety penalty: undercoverage risk. The dangerous component.
        Positive CVI_U means some segments are systematically under-covered.
        Under Consumer Duty / FCA EP25/2, CVI_U > 0.02 at portfolio level
        warrants segment-level investigation.
    cvi_o : float
        Efficiency penalty: overcoverage cost. Positive CVI_O means some
        segments have unnecessarily wide intervals.
    pi_minus : float
        Proportion of observations where estimated coverage probability is
        below (1-gamma)*(1-alpha). These are the undercovered segments.
    pi_plus : float
        Proportion of observations where estimated coverage probability is
        above (1+gamma)*(1-alpha). Overcovered segments.
    cmu : float
        Conditional mean undercoverage: E[(1-alpha) - eta | undercovered].
        Average shortfall per undercovered observation.
    cmo : float
        Conditional mean overcoverage: E[eta - (1-alpha) | overcovered].
        Average excess per overcovered observation.
    marginal_coverage : float
        Observed marginal coverage on the evaluation set.
    alpha : float
        Miscoverage rate used in CVI computation.
    gamma : float
        Relative tolerance for under/overcoverage classification.
    n_obs : int
        Number of observations in the evaluation set.
    eta_hat : np.ndarray
        Per-instance estimated coverage probabilities, shape (n_obs,).
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
        risk = "HIGH" if self.cvi_u > 0.02 else ("MEDIUM" if self.cvi_u > 0.005 else "LOW")
        return (
            f"CPAResult("
            f"cvi={self.cvi:.4f}, "
            f"cvi_u={self.cvi_u:.4f} [{risk}], "
            f"cvi_o={self.cvi_o:.4f}, "
            f"pi_minus={self.pi_minus:.3f}, "
            f"pi_plus={self.pi_plus:.3f}, "
            f"marginal={self.marginal_coverage:.3f}, "
            f"n={self.n_obs})"
        )


@dataclass
class CPASelectResult:
    """
    Selection result from CPASelector.select().

    Attributes
    ----------
    best_key : str
        Key of the prediction set with the lowest CVI.
    rankings : list of str
        All keys sorted by ascending CVI (best first).
    scores : dict of str -> CPAResult
        Full CVI decomposition for each prediction set.
    """

    best_key: str
    rankings: List[str]
    scores: Dict[str, CPAResult]

    def __repr__(self) -> str:
        return (
            f"CPASelectResult("
            f"best_key={self.best_key!r}, "
            f"rankings={self.rankings!r})"
        )

    def compare(self) -> Any:
        """
        Return a Polars DataFrame ranking prediction sets by CVI.

        Columns: key, cvi, cvi_u, cvi_o, pi_minus, pi_plus, marginal_coverage, rank.
        Sorted by ascending CVI.
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for compare(). Install with: pip install polars"
            )

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
# ReliabilityEstimator
# ---------------------------------------------------------------------------


class ReliabilityEstimator:
    """
    Predicts per-instance coverage probability P(Y in C(X) | X).

    The reliability estimator is the core mechanism behind CPA. It trains
    a binary classifier to predict whether each observation will be covered
    by the conformal interval. If coverage is truly uniform (the ideal case),
    the classifier can do no better than predicting the constant 1-alpha
    everywhere. When it does better — when coverage is predictable from
    features — those predictions quantify conditional miscoverage.

    This class is intentionally standalone. You can use it independently
    from CPASelector: for example, to produce eta_hat(x) maps for visualisation,
    or to identify which rating cells are systematically under-covered, or to
    feed coverage probability estimates into a downstream fairness audit.

    Parameters
    ----------
    estimator : sklearn-compatible binary classifier, optional
        Any classifier implementing fit(X, y) and predict_proba(X). If None,
        uses LogisticRegression(max_iter=500) — a fast, interpretable default
        that works reasonably well when the conditional coverage structure is
        approximately linear in the feature space (common for simple conformal
        predictors applied to tabular data). For highly non-linear conditional
        coverage patterns, pass GradientBoostingClassifier or a LightGBM
        classifier.
    calibrate : bool, default True
        Apply Platt scaling (sigmoid calibration) to the fitted classifier's
        probability outputs. Recommended when using non-probabilistic or
        poorly-calibrated classifiers. Has no effect on logistic regression
        (already calibrated by construction).
    random_state : int, optional
        Passed to LogisticRegression when estimator is None. Ignored when
        a custom estimator is provided.

    Attributes
    ----------
    is_fitted_ : bool
        True after fit() has been called.
    estimator_ : fitted sklearn classifier
        The fitted reliability estimator (after calibration if enabled).
    classes_ : np.ndarray
        Class labels [0, 1].

    Examples
    --------
    Basic usage with default logistic regression::

        from insurance_conformal.cpa import ReliabilityEstimator
        re = ReliabilityEstimator()
        re.fit(X_cal, y_cal, lower_cal, upper_cal)
        eta_hat = re.predict_coverage_proba(X_test)

    Using a gradient boosted classifier::

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4)
        re = ReliabilityEstimator(estimator=clf)
        re.fit(X_cal, y_cal, lower_cal, upper_cal)

    Segment-level coverage audit::

        eta_hat = re.predict_coverage_proba(X_test)
        young_mask = X_test[:, age_col] < 25
        print(f"Young driver coverage: {eta_hat[young_mask].mean():.3f}")
        print(f"Portfolio coverage:    {eta_hat.mean():.3f}")
    """

    _MIN_OBS = 30
    _WARN_OBS = 500

    def __init__(
        self,
        estimator: Optional[Any] = None,
        calibrate: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.calibrate = bool(calibrate)
        self.random_state = random_state

        self.is_fitted_: bool = False
        self.estimator_: Any = None
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(
        self,
        X: Any,
        y: Any,
        lower: Any,
        upper: Any,
    ) -> "ReliabilityEstimator":
        """
        Fit the reliability estimator on calibration data.

        Constructs binary coverage indicators Z_i = 1{y_i in [lower_i, upper_i]}
        and trains the classifier on (X, Z).

        Parameters
        ----------
        X : array-like of shape (n, p)
            Feature matrix. Can be a numpy array, pandas DataFrame, or Polars
            DataFrame.
        y : array-like of shape (n,)
            Observed outcomes.
        lower : array-like of shape (n,)
            Lower prediction interval bounds.
        upper : array-like of shape (n,)
            Upper prediction interval bounds.

        Returns
        -------
        self
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        X_np = _to_2d(X)
        y_np = as_numpy(y).ravel().astype(float)
        lo = as_numpy(lower).ravel().astype(float)
        hi = as_numpy(upper).ravel().astype(float)

        n = len(y_np)

        if n < self._MIN_OBS:
            raise ValueError(
                f"Need at least {self._MIN_OBS} observations to fit a reliability "
                f"estimator, got {n}."
            )
        if len(lo) != n or len(hi) != n:
            raise ValueError(
                "lower and upper must have the same length as y."
            )
        if n < self._WARN_OBS:
            warnings.warn(
                f"ReliabilityEstimator fitted on {n} observations. "
                f"CVI estimates are more stable with n >= {self._WARN_OBS}.",
                UserWarning,
                stacklevel=2,
            )

        Z = _coverage_indicators(y_np, lo, hi).astype(int)

        if self.estimator is None:
            base_clf = LogisticRegression(
                max_iter=500,
                random_state=self.random_state,
            )
        else:
            base_clf = self.estimator

        if self.calibrate:
            # Sigmoid calibration wraps any classifier. For logistic regression
            # this is redundant but harmless — calibrated probabilities match
            # the raw probabilities for GLMs.
            clf = CalibratedClassifierCV(
                estimator=base_clf,
                method="sigmoid",
                cv=5,
            )
        else:
            clf = base_clf

        clf.fit(X_np, Z)
        self.estimator_ = clf
        self.is_fitted_ = True
        return self

    def predict_coverage_proba(self, X: Any) -> np.ndarray:
        """
        Predict per-instance coverage probability.

        Returns the estimated probability P(Y in C(X) | X = x) for each
        row in X. A value near 1-alpha means the instance is covered at
        the nominal rate. Values below (1-gamma)*(1-alpha) indicate likely
        undercoverage; values above (1+gamma)*(1-alpha) indicate overcoverage.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Feature matrix. Same feature space as used in fit().

        Returns
        -------
        eta_hat : np.ndarray of shape (n,)
            Estimated coverage probabilities in [0, 1].

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        self._check_fitted()
        X_np = _to_2d(X)
        return self.estimator_.predict_proba(X_np)[:, 1]

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict_coverage_proba().")


# ---------------------------------------------------------------------------
# CPASelector
# ---------------------------------------------------------------------------


class CPASelector:
    """
    Select the best conformal predictor using Conditional Validity Index (CVI).

    CPASelector fits a ReliabilityEstimator on calibration data, then scores
    each candidate set of prediction intervals by CVI — decomposed into
    undercoverage risk (CVI_U) and overcoverage cost (CVI_O). The predictor
    with the lowest CVI wins.

    This implements the CC-Select algorithm from Zhou et al. (2026), adapted
    to work from pre-computed intervals rather than predictor objects. The key
    design choice: take intervals as input rather than conformal predictor
    objects. This makes CPASelector usable in any pipeline — including
    pipelines where the conformal predictor is not a Python object (e.g.,
    intervals computed in R or SQL).

    Parameters
    ----------
    alpha : float
        Miscoverage rate. Target coverage = 1 - alpha. Must be in (0, 1).
    gamma : float, default 0.1
        Relative tolerance for under/overcoverage classification. Instances
        with eta_hat in [(1-gamma)*(1-alpha), (1+gamma)*(1-alpha)] contribute
        to neither CVI_U nor CVI_O — they are "near-nominal". Increase gamma
        to make CVI less sensitive to near-nominal deviations.
    estimator : sklearn-compatible binary classifier, optional
        Passed to ReliabilityEstimator. If None, uses LogisticRegression.
        For large datasets with complex conditional coverage patterns, pass
        a GradientBoostingClassifier or LightGBM classifier.
    calibrate : bool, default True
        Apply probability calibration to the reliability estimator.
    random_state : int, optional
        Random state for the default LogisticRegression estimator.

    Attributes
    ----------
    reliability_estimator_ : ReliabilityEstimator
        The fitted reliability estimator, available after fit().
    is_fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    Fit on calibration data, then score test intervals::

        from insurance_conformal.cpa import CPASelector

        selector = CPASelector(alpha=0.10, gamma=0.1, random_state=42)
        selector.fit(X_cal, y_cal, lower_cal, upper_cal)
        result = selector.score(X_test, y_test, lower_test, upper_test)
        print(result.cvi_u)  # undercoverage risk

    Select the best predictor from multiple candidates::

        result = selector.select(
            X_test,
            y_test,
            {
                "pearson": (lower_pearson, upper_pearson),
                "cqr": (lower_cqr, upper_cqr),
                "locally_weighted": (lower_lw, upper_lw),
            }
        )
        print(result.best_key)   # "cqr" (lowest CVI)
        print(result.compare())  # Polars DataFrame ranked by CVI

    Using a gradient boosted reliability estimator::

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
        selector = CPASelector(alpha=0.10, estimator=clf)
        selector.fit(X_cal, y_cal, lower_cal, upper_cal)
    """

    def __init__(
        self,
        alpha: float,
        gamma: float = 0.1,
        estimator: Optional[Any] = None,
        calibrate: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}.")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.estimator = estimator
        self.calibrate = bool(calibrate)
        self.random_state = random_state

        self.is_fitted_: bool = False
        self.reliability_estimator_: Optional[ReliabilityEstimator] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_cal: Any,
        y_cal: Any,
        lower_cal: Any,
        upper_cal: Any,
    ) -> "CPASelector":
        """
        Fit the reliability estimator on calibration data.

        This is the expensive step. Do it once, then call score() or select()
        as many times as needed on different test sets or predictor candidates.

        Parameters
        ----------
        X_cal : array-like of shape (n_cal, p)
            Calibration feature matrix.
        y_cal : array-like of shape (n_cal,)
            Calibration outcomes.
        lower_cal : array-like of shape (n_cal,)
            Lower bounds of prediction intervals on calibration data.
        upper_cal : array-like of shape (n_cal,)
            Upper bounds of prediction intervals on calibration data.

        Returns
        -------
        self
        """
        re = ReliabilityEstimator(
            estimator=self.estimator,
            calibrate=self.calibrate,
            random_state=self.random_state,
        )
        re.fit(X_cal, y_cal, lower_cal, upper_cal)
        self.reliability_estimator_ = re
        self.is_fitted_ = True
        return self

    def score(
        self,
        X_test: Any,
        y_test: Any,
        prediction_intervals: Any,
    ) -> CPAResult:
        """
        Compute CVI for a single set of prediction intervals.

        Parameters
        ----------
        X_test : array-like of shape (n_test, p)
            Test feature matrix.
        y_test : array-like of shape (n_test,)
            Test outcomes.
        prediction_intervals : (lower, upper) tuple, dict, or DataFrame
            Prediction intervals for test observations. Accepted formats:
            - (lower, upper) tuple of array-likes of shape (n_test,)
            - dict with "lower" and "upper" keys
            - Polars or pandas DataFrame with "lower" and "upper" columns

        Returns
        -------
        CPAResult
            CVI decomposition for these prediction intervals.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        self._check_fitted()

        X_np = _to_2d(X_test)
        y_np = as_numpy(y_test).ravel().astype(float)
        lower, upper = _parse_intervals(prediction_intervals)

        n = len(y_np)
        if len(lower) != n or len(upper) != n:
            raise ValueError(
                "prediction_intervals must have the same number of rows as y_test."
            )

        eta_hat = self.reliability_estimator_.predict_coverage_proba(X_np)
        Z = _coverage_indicators(y_np, lower, upper)
        marginal = float(Z.mean())

        cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo = _compute_cvi(
            eta_hat, self.alpha, self.gamma
        )

        return CPAResult(
            cvi=cvi,
            cvi_u=cvi_u,
            cvi_o=cvi_o,
            pi_minus=pi_minus,
            pi_plus=pi_plus,
            cmu=cmu,
            cmo=cmo,
            marginal_coverage=marginal,
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
    ) -> CPASelectResult:
        """
        CC-Select: pick the best predictor by CVI from a dict of prediction sets.

        Scores every prediction set against the fitted reliability estimator
        and returns the one with the lowest total CVI. Equal CVI scores are
        broken by CVI_U (lower undercoverage risk wins) — the more conservative
        choice for regulatory use.

        Parameters
        ----------
        X_test : array-like of shape (n_test, p)
            Test features.
        y_test : array-like of shape (n_test,)
            Test outcomes.
        prediction_sets : dict of str -> prediction intervals
            A mapping from predictor name to prediction intervals. Each value
            must be in one of the formats accepted by score().

        Returns
        -------
        CPASelectResult
            Contains best_key, rankings sorted by ascending CVI, and per-
            predictor CPAResult scores.

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

        results: Dict[str, CPAResult] = {}
        for key, ps in prediction_sets.items():
            results[key] = self.score(X_test, y_test, ps)

        # Sort by (cvi, cvi_u) — ties broken by undercoverage risk
        rankings = sorted(
            results.keys(),
            key=lambda k: (results[k].cvi, results[k].cvi_u),
        )
        best_key = rankings[0]

        return CPASelectResult(
            best_key=best_key,
            rankings=rankings,
            scores=results,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before score() or select().")
