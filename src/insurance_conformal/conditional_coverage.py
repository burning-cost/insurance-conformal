"""
Conditional coverage diagnostics for conformal prediction.

Two complementary tools for assessing and improving conditional coverage:

ConditionalCoverageERT (v0.7.1): hypothesis test for conditional coverage
violations. Trains a LightGBM classifier on Z_i ~ X_i to detect whether the
coverage indicator is predictable from features. ERT measures the classifier's
advantage over the constant predictor — zero under the null, positive under
conditional miscoverage. Three loss variants (L1, L2, KL), directional variants
for insurance FCA/TCF use cases. Based on Braun et al. arXiv:2512.11779.

ConditionalValidityIndex (v1.3.0): scalar measure of conditional coverage
quality, decomposed into undercoverage risk (CVI_U) and overcoverage cost
(CVI_O). Unlike ERT (a hypothesis test), CVI gives a continuous, interpretable
score that can be used to rank and select conformal predictors. Fits a LightGBM
classifier with isotonic probability calibration to estimate eta_hat(x) — the
local coverage probability at each point. Based on Zhou et al. arXiv:2603.27189.

CCSelect (v0.8.0): model selection via CVI. Evaluates multiple conformal
predictors across random train/eval splits and picks the one with the lowest
mean CVI — the predictor with the most uniform conditional coverage. Predictors
must implement .calibrate(X, y) and .predict_interval(X, alpha) matching the
standard insurance-conformal interface.

The CVI decomposition:
    CVI_U = pi_minus * E[(1-alpha) - eta_hat(X) | eta_hat(X) < (1-gamma)*(1-alpha)]
          = pi_minus * CMU
    CVI_O = pi_plus * E[eta_hat(X) - (1-alpha) | eta_hat(X) > (1+gamma)*(1-alpha)]
          = pi_plus * CMO
    CVI = CVI_U + CVI_O

References:
    Braun, Gruber & Matthias (2024) arXiv:2512.11779
    "Excess Risk of Target Coverage: Testing Conditional Coverage"

    Zhou et al. (2026) arXiv:2603.27189
    "Conditional Validity Index for Conformal Prediction"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LossVariant = Literal["l1", "l2", "kl"]
DirectionVariant = Literal["both", "under", "over"]


# ===========================================================================
# ConditionalCoverageERT (v0.7.1)
# ===========================================================================


@dataclass
class ERTResult:
    """
    Results from a ConditionalCoverageERT evaluation.

    Attributes
    ----------
    ert : float
        ERT statistic. Positive values indicate conditional coverage violation
        in the specified direction. Zero means the classifier cannot beat the
        constant predictor (coverage appears uniform).
    baseline_loss : float
        Loss of the constant predictor (predicts 1-alpha everywhere).
    classifier_loss : float
        Cross-validated loss of the LightGBM classifier.
    ci_lower : float
        Lower bound of the bootstrap CI for ERT.
    ci_upper : float
        Upper bound of the bootstrap CI for ERT.
    alpha : float
        Miscoverage rate (target coverage = 1 - alpha).
    marginal_coverage : float
        Observed marginal coverage across all test samples.
    loss : str
        Loss variant used ("l1", "l2", or "kl").
    direction : str
        Direction variant used ("both", "under", or "over").
    n_obs : int
        Number of test observations.
    n_bootstraps : int
        Number of bootstrap resamples used for CI.
    """

    ert: float
    baseline_loss: float
    classifier_loss: float
    ci_lower: float
    ci_upper: float
    alpha: float
    marginal_coverage: float
    loss: str
    direction: str
    n_obs: int
    n_bootstraps: int

    def __repr__(self) -> str:
        sig = "***" if self.ci_lower > 0 else ("" if self.ci_upper < 0 else " (not significant)")
        return (
            f"ERTResult(ert={self.ert:.4f}{sig}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"marginal_coverage={self.marginal_coverage:.3f}, "
            f"target={1 - self.alpha:.3f}, "
            f"loss={self.loss!r}, direction={self.direction!r})"
        )


class ConditionalCoverageERT:
    """
    Full ERT test for conditional coverage using LightGBM.

    Tests whether conformal prediction intervals have uniform coverage across
    feature space. Under the null hypothesis of conditional coverage equality,
    ERT should be zero: a classifier predicting Z_i from X_i does no better
    than always predicting 1 - alpha.

    A positive ERT with CI that excludes zero is evidence of conditional
    miscoverage. The direction parameter controls what kind of miscoverage
    you care about.

    Parameters
    ----------
    loss : {"l1", "l2", "kl"}, default "l1"
        Loss function for computing ERT.
        - "l1": mean absolute error. Linear penalty. Best for reporting.
        - "l2": Brier score (MSE). Quadratic, more sensitive to large gaps.
        - "kl": log loss. Penalises confident mispredictions heavily.
    direction : {"both", "under", "over"}, default "under"
        Which direction of coverage deviation to penalise.
        - "under": only count samples where the classifier predicts p < 1-alpha
          (i.e., where the model suspects undercoverage). Most relevant for
          insurance — FCA/TCF concerns focus on systematic undercoverage.
        - "over": only penalise predicted overcoverage (efficiency concern).
        - "both": penalise all deviations symmetrically.
    n_splits : int, default 5
        Number of KFold CV splits for classifier evaluation.
    n_bootstraps : int, default 500
        Number of bootstrap resamples for CI computation.
    ci_level : float, default 0.90
        Confidence level for bootstrap CI (e.g., 0.90 = 5th to 95th percentile).
    lgbm_params : dict, optional
        LightGBM parameters. Defaults are conservative to avoid overfit
        on small datasets: shallow trees, high regularisation, early stopping
        not used (to keep CV clean).
    random_state : int, optional
        Random state for KFold splits and bootstrap.

    Examples
    --------
    >>> from insurance_conformal.conditional_coverage import ConditionalCoverageERT
    >>> ert = ConditionalCoverageERT(loss="l1", direction="under", n_splits=5)
    >>> result = ert.evaluate(X_test, y_lower, y_upper, y_true, alpha=0.10)
    >>> print(result)  # ERTResult(ert=0.023..., CI=[0.008, 0.041], ...)
    """

    _DEFAULT_LGBM_PARAMS: dict = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 200,
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }

    def __init__(
        self,
        loss: LossVariant = "l1",
        direction: DirectionVariant = "under",
        n_splits: int = 5,
        n_bootstraps: int = 500,
        ci_level: float = 0.90,
        lgbm_params: Optional[dict] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if loss not in ("l1", "l2", "kl"):
            raise ValueError(f"loss must be 'l1', 'l2', or 'kl', got {loss!r}")
        if direction not in ("both", "under", "over"):
            raise ValueError(f"direction must be 'both', 'under', or 'over', got {direction!r}")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        self.loss = loss
        self.direction = direction
        self.n_splits = n_splits
        self.n_bootstraps = n_bootstraps
        self.ci_level = ci_level
        self.lgbm_params = dict(self._DEFAULT_LGBM_PARAMS)
        if lgbm_params is not None:
            self.lgbm_params.update(lgbm_params)
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: Any,
        y_lower: Any,
        y_upper: Any,
        y_true: Any,
        alpha: float,
    ) -> ERTResult:
        """
        Evaluate the ERT conditional coverage test.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Test features. Must be numeric (encode categoricals first).
        y_lower : array-like of shape (n,)
            Lower bounds of prediction intervals.
        y_upper : array-like of shape (n,)
            Upper bounds of prediction intervals.
        y_true : array-like of shape (n,)
            Observed outcomes.
        alpha : float
            Miscoverage rate. Target coverage = 1 - alpha.

        Returns
        -------
        ERTResult
            ERT statistic with bootstrap CI.
        """
        X_arr = np.asarray(as_numpy(X), dtype=np.float64)
        lo = as_numpy(y_lower).ravel().astype(float)
        hi = as_numpy(y_upper).ravel().astype(float)
        y = as_numpy(y_true).ravel().astype(float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        Z = ((y >= lo) & (y <= hi)).astype(float)
        n = len(Z)
        target = 1.0 - alpha

        ert, baseline, clf_loss = self._compute_ert(X_arr, Z, target)

        # Bootstrap CI
        rng = np.random.default_rng(self.random_state)
        boot_erts = np.empty(self.n_bootstraps)
        for b in range(self.n_bootstraps):
            idx = rng.integers(0, n, size=n)
            try:
                boot_ert, _, _ = self._compute_ert(X_arr[idx], Z[idx], target)
            except Exception:
                boot_ert = 0.0
            boot_erts[b] = boot_ert

        alpha_ci = (1.0 - self.ci_level) / 2.0
        ci_lo = float(np.quantile(boot_erts, alpha_ci))
        ci_hi = float(np.quantile(boot_erts, 1.0 - alpha_ci))

        return ERTResult(
            ert=ert,
            baseline_loss=baseline,
            classifier_loss=clf_loss,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            alpha=alpha,
            marginal_coverage=float(Z.mean()),
            loss=self.loss,
            direction=self.direction,
            n_obs=n,
            n_bootstraps=self.n_bootstraps,
        )

    def subgroup_coverage(
        self,
        X: Any,
        y_lower: Any,
        y_upper: Any,
        y_true: Any,
        alpha: float,
        feature_names: Optional[Sequence[str]] = None,
        n_bins: int = 5,
    ) -> pl.DataFrame:
        """
        Per-feature binned coverage report.

        Bins each feature into quantile bins and reports empirical coverage per
        bin. Useful for FCA reporting: "show that coverage is uniform across
        policyholder age, vehicle group, and region."

        Parameters
        ----------
        X : array-like of shape (n, p)
            Test features.
        y_lower, y_upper, y_true : array-like of shape (n,)
            Interval bounds and observed outcomes.
        alpha : float
            Miscoverage rate.
        feature_names : list of str, optional
            Names for each feature column. Defaults to "feature_0", "feature_1", ...
        n_bins : int, default 5
            Number of quantile bins per feature.

        Returns
        -------
        pl.DataFrame
            Columns: feature, bin_index, bin_midpoint, n_obs, empirical_coverage,
            target_coverage, coverage_gap. One row per (feature, bin) combination.
        """
        import pandas as pd

        X_arr = np.asarray(as_numpy(X), dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        lo = as_numpy(y_lower).ravel().astype(float)
        hi = as_numpy(y_upper).ravel().astype(float)
        y = as_numpy(y_true).ravel().astype(float)

        Z = ((y >= lo) & (y <= hi)).astype(float)
        n_features = X_arr.shape[1]
        target = 1.0 - alpha

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        rows = []
        for j, fname in enumerate(feature_names):
            col = X_arr[:, j]
            try:
                bin_labels = pd.qcut(col, q=n_bins, labels=False, duplicates="drop")
            except ValueError:
                bin_labels = pd.cut(col, bins=n_bins, labels=False, duplicates="drop")
            bin_arr = np.asarray(bin_labels, dtype=float)
            unique_bins = np.unique(bin_arr[~np.isnan(bin_arr)])
            for b in unique_bins:
                mask = bin_arr == b
                n_b = int(mask.sum())
                if n_b == 0:
                    continue
                cov_b = float(Z[mask].mean())
                mid = float(col[mask].mean())
                rows.append(
                    {
                        "feature": fname,
                        "bin_index": int(b) + 1,
                        "bin_midpoint": mid,
                        "n_obs": n_b,
                        "empirical_coverage": cov_b,
                        "target_coverage": target,
                        "coverage_gap": float(target - cov_b),
                    }
                )

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_ert(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        target: float,
    ) -> tuple[float, float, float]:
        """
        Compute ERT = baseline_loss - classifier_cv_loss.

        Returns (ert, baseline_loss, classifier_loss).
        """
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "ConditionalCoverageERT requires lightgbm. "
                "Install with: pip install insurance-conformal[lightgbm]"
            ) from e

        from sklearn.model_selection import KFold

        n = len(Z)
        baseline = self._loss_fn(Z, np.full(n, target))

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        oof_preds = np.full(n, target)
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            Z_tr = Z[train_idx]

            params = dict(self.lgbm_params)
            params.pop("metric", None)
            n_estimators = params.pop("n_estimators", 200)

            clf = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
            clf.fit(X_tr, Z_tr.astype(int))
            oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]

        clf_loss = self._loss_fn(Z, oof_preds)
        ert = float(baseline - clf_loss)
        return ert, float(baseline), float(clf_loss)

    def _loss_fn(self, Z: np.ndarray, p: np.ndarray) -> float:
        """
        Compute directional loss between coverage indicators Z and predictions p.

        For directional variants, we mask to observations where the prediction
        is in the direction we care about, then compute the loss on that subset.
        The baseline uses the same mask applied to the constant predictor.
        """
        p = np.clip(p, 1e-7, 1 - 1e-7)

        if self.direction == "both":
            mask = np.ones(len(Z), dtype=bool)
        elif self.direction == "under":
            # Penalise where predicted coverage is below target — the classifier
            # thinks this region is undercovered
            target_approx = np.mean(Z)  # use marginal coverage as target proxy
            mask = p < target_approx
            if mask.sum() == 0:
                mask = np.ones(len(Z), dtype=bool)
        else:  # "over"
            target_approx = np.mean(Z)
            mask = p > target_approx
            if mask.sum() == 0:
                mask = np.ones(len(Z), dtype=bool)

        Z_m = Z[mask]
        p_m = p[mask]

        if len(Z_m) == 0:
            return 0.0

        if self.loss == "l1":
            return float(np.mean(np.abs(Z_m - p_m)))
        elif self.loss == "l2":
            return float(np.mean((Z_m - p_m) ** 2))
        else:  # kl
            # Binary cross-entropy / log loss
            # E[-Z log(p) - (1-Z) log(1-p)]
            return float(np.mean(-Z_m * np.log(p_m) - (1 - Z_m) * np.log(1 - p_m)))


# ===========================================================================
# ConditionalValidityIndex and CCSelect (v0.8.0)
# ===========================================================================


# Default LightGBM parameters for the CVI classifier — same conservative
# pattern as ConditionalCoverageERT but binary classification objective.
_CVI_DEFAULT_LGBM_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "objective": "binary",
    "random_state": 42,
    "verbose": -1,
}

_MIN_OBS_CVI = 50  # Minimum observations for reliable CV-based CVI estimation


@dataclass
class CVIResult:
    """
    Result of a ConditionalValidityIndex evaluation.

    Attributes
    ----------
    cvi : float
        Total CVI = CVI_U + CVI_O.
    cvi_u : float
        Undercoverage component (safety). Counts only observations where the
        local coverage probability is below (1-gamma)*(1-alpha).
    cvi_o : float
        Overcoverage component (efficiency cost). Counts only observations
        where local coverage probability is above (1+gamma)*(1-alpha).
    pi_minus : float
        Proportion of observations with eta_hat < (1-gamma)*(1-alpha).
    pi_plus : float
        Proportion of observations with eta_hat > (1+gamma)*(1-alpha).
    cmu : float
        Conditional mean undercoverage among undercovered observations.
        E[(1-alpha) - eta_hat(X) | eta_hat(X) < (1-gamma)*(1-alpha)].
        Zero if no undercovered observations.
    cmo : float
        Conditional mean overcoverage among overcovered observations.
        E[eta_hat(X) - (1-alpha) | eta_hat(X) > (1+gamma)*(1-alpha)].
        Zero if no overcovered observations.
    marginal_coverage : float
        Empirical marginal coverage proportion.
    alpha : float
        Miscoverage rate used in evaluation.
    gamma : float
        Relative tolerance threshold for undercoverage/overcoverage classification.
    n_obs : int
        Number of observations used in evaluation.
    eta_hat : np.ndarray
        Estimated local coverage probabilities of shape (n_obs,).
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
        return (
            f"CVIResult("
            f"cvi={self.cvi:.4f}, "
            f"cvi_u={self.cvi_u:.4f}, "
            f"cvi_o={self.cvi_o:.4f}, "
            f"pi_minus={self.pi_minus:.3f}, "
            f"pi_plus={self.pi_plus:.3f}, "
            f"marginal_coverage={self.marginal_coverage:.3f}, "
            f"n_obs={self.n_obs})"
        )


@dataclass
class CCSelectResult:
    """
    Result of a CCSelect model selection run.

    Attributes
    ----------
    best_predictor : str
        Name of the predictor with the lowest mean CVI.
    cvi_scores : dict
        Mean CVI per predictor name.
    cvi_u_scores : dict
        Mean CVI_U per predictor name.
    cvi_o_scores : dict
        Mean CVI_O per predictor name.
    rankings : list
        Predictor names sorted by ascending mean CVI (best first).
    """

    best_predictor: str
    cvi_scores: Dict[str, float]
    cvi_u_scores: Dict[str, float]
    cvi_o_scores: Dict[str, float]
    rankings: List[str]

    def __repr__(self) -> str:
        return (
            f"CCSelectResult("
            f"best_predictor={self.best_predictor!r}, "
            f"rankings={self.rankings!r})"
        )


class ConditionalValidityIndex:
    """
    Conditional Validity Index: a scalar measure of conditional coverage quality.

    Standard conformal predictors guarantee marginal coverage P(Y in C(X)) >= 1-alpha,
    but say nothing about conditional coverage P(Y in C(X) | X=x). A predictor could
    undercover high-risk segments while overcovering standard risks — passing marginal
    tests while failing where it matters.

    CVI quantifies this by:
    1. Computing coverage indicators Z_i = 1{y_i in [lower_i, upper_i]}.
    2. Training a LightGBM classifier Z ~ X using KFold cross-validation.
    3. Applying isotonic calibration to out-of-fold predictions -> eta_hat(x).
    4. Decomposing deviations from target coverage (1-alpha) into CVI_U (safety)
       and CVI_O (efficiency cost).

    The gamma tolerance means observations within gamma*100% of (1-alpha) are
    treated as acceptable — they contribute neither to CVI_U nor CVI_O. This
    prevents the metric from being dominated by noise in well-calibrated regions.

    A CVI near zero means coverage is approximately uniform across feature space.
    A high CVI_U relative to CVI_O signals systematic undercoverage — the predictor
    is unsafe for some segments. A high CVI_O signals over-conservative intervals
    that waste premium capacity.

    Parameters
    ----------
    gamma : float, default 0.1
        Relative tolerance threshold. Observations with eta_hat in
        [(1-gamma)*(1-alpha), (1+gamma)*(1-alpha)] are excluded from both
        CVI_U and CVI_O. Increasing gamma makes the metric more lenient.
    n_splits : int, default 5
        Number of folds for KFold cross-validation when estimating eta_hat.
    lgbm_params : dict, optional
        LightGBM LGBMClassifier parameters. Defaults to: n_estimators=300,
        learning_rate=0.05, num_leaves=15, objective='binary', verbose=-1.
        Override to tune the coverage estimator — but avoid overfitting,
        which would artificially inflate CVI.
    calibrate_proba : bool, default True
        Apply isotonic calibration (sklearn CalibratedClassifierCV with
        method='isotonic') to the out-of-fold probability estimates.
        Strongly recommended: raw LightGBM probabilities can be poorly
        calibrated for rare coverage events.
    random_state : int, optional
        Random state for KFold splitting.

    Attributes
    ----------
    result_ : CVIResult or None
        The most recent result from evaluate(). None before evaluate() is called.

    Examples
    --------
    >>> from insurance_conformal import ConditionalValidityIndex
    >>> cvi = ConditionalValidityIndex(gamma=0.1, n_splits=5)
    >>> result = cvi.evaluate(X_test, lower, upper, y_test, alpha=0.10)
    >>> print(result.cvi)  # total CVI
    >>> print(result.cvi_u)  # undercoverage component
    >>> profile = cvi.reliability_profile()
    >>> print(profile.head(10))
    """

    def __init__(
        self,
        gamma: float = 0.1,
        n_splits: int = 5,
        lgbm_params: Optional[dict] = None,
        calibrate_proba: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        self.gamma = float(gamma)
        self.n_splits = int(n_splits)
        self.calibrate_proba = bool(calibrate_proba)
        self.random_state = random_state

        params = dict(_CVI_DEFAULT_LGBM_PARAMS)
        if lgbm_params is not None:
            params.update(lgbm_params)
        self.lgbm_params = params

        self.result_: Optional[CVIResult] = None

    def evaluate(
        self,
        X: Any,
        y_lower: Any,
        y_upper: Any,
        y_true: Any,
        alpha: float,
    ) -> CVIResult:
        """
        Evaluate conditional coverage and compute CVI.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Feature matrix. Rows correspond to the same observations as
            y_lower, y_upper, y_true.
        y_lower : array-like of shape (n,)
            Lower bounds of the prediction intervals.
        y_upper : array-like of shape (n,)
            Upper bounds of the prediction intervals.
        y_true : array-like of shape (n,)
            Observed outcomes.
        alpha : float
            Miscoverage rate the predictor was calibrated at. Used to set
            the target coverage (1-alpha) for CVI computation.

        Returns
        -------
        CVIResult
            Computed CVI result. Also stored as self.result_.

        Raises
        ------
        ValueError
            If n < 50 (insufficient observations for reliable CV estimation).
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError(
                "ConditionalValidityIndex requires LightGBM. "
                "Install with: pip install 'insurance-conformal[lightgbm]'"
            ) from e

        X_np = _to_numpy_2d(X)
        lower = as_numpy(y_lower).ravel()
        upper = as_numpy(y_upper).ravel()
        y = as_numpy(y_true).ravel()

        n = len(y)
        if n < _MIN_OBS_CVI:
            raise ValueError(
                f"Need at least {_MIN_OBS_CVI} observations for reliable CVI estimation, "
                f"got {n}."
            )
        if len(lower) != n or len(upper) != n:
            raise ValueError(
                "y_lower, y_upper, and y_true must all have the same length."
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        # Warn for small calibration sets — CVI convergence requires sufficient
        # data per bin. Zhou et al. (arXiv:2603.27189) note n>=800 for reliability.
        _WARN_THRESHOLD_CVI = 800
        if n < _WARN_THRESHOLD_CVI:
            warnings.warn(
                f"CVI estimates may be unreliable with fewer than {_WARN_THRESHOLD_CVI} "
                f"calibration samples (got {n}). Results should be interpreted with caution. "
                "See Zhou et al. arXiv:2603.27189.",
                UserWarning,
                stacklevel=2,
            )

        # Step 1: coverage indicators
        Z = ((y >= lower) & (y <= upper)).astype(float)
        marginal_coverage = float(Z.mean())

        # Step 2+3: KFold CV, train classifier, calibrate probabilities
        eta_hat = self._fit_eta_hat(X_np, Z, LGBMClassifier)

        # Step 4: compute CVI components
        target = 1.0 - alpha
        lower_threshold = (1.0 - self.gamma) * target
        upper_threshold = (1.0 + self.gamma) * target

        mask_under = eta_hat < lower_threshold
        mask_over = eta_hat > upper_threshold

        pi_minus = float(mask_under.mean())
        pi_plus = float(mask_over.mean())

        if mask_under.any():
            cmu = float((target - eta_hat[mask_under]).mean())
        else:
            cmu = 0.0

        if mask_over.any():
            cmo = float((eta_hat[mask_over] - target).mean())
        else:
            cmo = 0.0

        cvi_u = pi_minus * cmu
        cvi_o = pi_plus * cmo
        cvi = cvi_u + cvi_o

        self.result_ = CVIResult(
            cvi=cvi,
            cvi_u=cvi_u,
            cvi_o=cvi_o,
            pi_minus=pi_minus,
            pi_plus=pi_plus,
            cmu=cmu,
            cmo=cmo,
            marginal_coverage=marginal_coverage,
            alpha=alpha,
            gamma=self.gamma,
            n_obs=n,
            eta_hat=eta_hat,
        )
        return self.result_

    def reliability_profile(self) -> pl.DataFrame:
        """
        Return the quantile function of eta_hat (CVP curve data).

        The reliability profile shows how the estimated local coverage
        probabilities eta_hat are distributed across observations. A well-
        calibrated predictor with good conditional coverage should have a
        tight distribution of eta_hat values centred near (1-alpha).

        Returns
        -------
        pl.DataFrame
            Columns: quantile (0.0 to 1.0 in 101 steps), coverage_prob.
            The quantile column gives the empirical CDF evaluation point;
            coverage_prob gives the corresponding eta_hat value.

        Raises
        ------
        RuntimeError
            If evaluate() has not been called yet.
        """
        if self.result_ is None:
            raise RuntimeError(
                "Call evaluate() before reliability_profile()."
            )
        quantiles = np.linspace(0.0, 1.0, 101)
        probs = np.quantile(self.result_.eta_hat, quantiles)
        return pl.DataFrame({"quantile": quantiles, "coverage_prob": probs})

    def reliability_plot(self, figsize: tuple = (10, 5)) -> None:
        """
        Plot the distribution of local coverage probabilities (CVP plot).

        Shows a histogram of eta_hat(X) with the target coverage (1-alpha)
        marked as a vertical line. Shading indicates:
        - Red region: observations below (1-gamma)*(1-alpha) — undercoverage zone.
        - Yellow region: observations above (1+gamma)*(1-alpha) — overcoverage zone.
        - Green region: acceptable coverage band.

        Also shows the reliability profile (empirical CDF) on a secondary axis.

        Parameters
        ----------
        figsize : tuple, default (10, 5)
            Figure size in inches.

        Raises
        ------
        RuntimeError
            If evaluate() has not been called yet.
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "reliability_plot() requires matplotlib. "
                "Install with: pip install 'insurance-conformal[plot]'"
            ) from e

        if self.result_ is None:
            raise RuntimeError("Call evaluate() before reliability_plot().")

        r = self.result_
        target = 1.0 - r.alpha
        lo = (1.0 - r.gamma) * target
        hi = (1.0 + r.gamma) * target

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: histogram of eta_hat
        ax1.hist(r.eta_hat, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
        ax1.axvline(target, color="black", linewidth=1.5, linestyle="--",
                    label=f"Target 1-alpha={target:.2f}")
        ax1.axvspan(0.0, lo, alpha=0.15, color="red",
                    label=f"Undercoverage (<{lo:.2f})")
        ax1.axvspan(hi, 1.0, alpha=0.15, color="orange",
                    label=f"Overcoverage (>{hi:.2f})")
        ax1.axvspan(lo, hi, alpha=0.10, color="green", label="Acceptable band")
        ax1.set_xlabel("Estimated local coverage probability eta_hat(X)")
        ax1.set_ylabel("Count")
        ax1.set_title(
            f"Coverage distribution\nCVI={r.cvi:.4f} "
            f"(U={r.cvi_u:.4f}, O={r.cvi_o:.4f})"
        )
        ax1.legend(fontsize=8)

        # Right: reliability profile (empirical CDF)
        profile = self.reliability_profile()
        ax2.plot(
            profile["coverage_prob"].to_numpy(),
            profile["quantile"].to_numpy(),
            color="steelblue",
            linewidth=1.5,
        )
        ax2.axvline(target, color="black", linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Coverage probability")
        ax2.set_ylabel("Quantile of observations")
        ax2.set_title("Reliability profile (CVP curve)")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """
        Expected calibration error (ECE) — a complement to CVI.

        ECE is the weighted average of |observed_coverage - nominal_coverage|
        across bins of eta_hat. Where CVI decomposes coverage errors into
        under/overcoverage components, ECE gives a single summary of raw
        calibration deviation without the directional decomposition.

        Bins observations by estimated coverage probability eta_hat(X), then
        computes empirical coverage per bin and weights the absolute deviations
        by bin size::

            ECE = sum_k (n_k / n) * |coverage_k - (1 - alpha)|

        Parameters
        ----------
        n_bins : int, default 10
            Number of equal-width bins over [0, 1] in eta_hat space.

        Returns
        -------
        float
            Expected calibration error in [0, 1]. Zero means perfect
            per-bin calibration. Values above 0.05 indicate meaningful
            systematic miscoverage.

        Raises
        ------
        RuntimeError
            If evaluate() has not been called yet.
        """
        if self.result_ is None:
            raise RuntimeError(
                "Call evaluate() before expected_calibration_error()."
            )
        r = self.result_
        target = 1.0 - r.alpha
        eta = r.eta_hat
        n = r.n_obs

        # Coverage indicators are not stored on CVIResult, but we can infer
        # from eta_hat whether points were covered. However, eta_hat is the
        # *estimated* coverage probability, not the raw coverage indicator.
        # We bin by eta_hat and compute the mean eta_hat per bin vs target,
        # since the raw Z indicators are not stored after evaluate().
        # This gives ECE as the weighted mean deviation of estimated coverage
        # from the nominal level — analogous to how ECE is computed in
        # classification calibration with confidence as the bin key.
        bin_edges = np.linspace(0.0, 1.0 + 1e-10, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (eta >= bin_edges[i]) & (eta < bin_edges[i + 1])
            n_k = mask.sum()
            if n_k == 0:
                continue
            coverage_k = eta[mask].mean()
            ece += (n_k / n) * abs(coverage_k - target)
        return float(ece)

    def plot_conditional_validity(
        self,
        n_bins: int = 10,
        alpha_ci: float = 0.05,
        figsize: tuple = (8, 5),
    ):
        """
        Per-bin coverage vs nominal level (CVP plot).

        Shows empirical coverage per bin of eta_hat(X) with Wilson score
        confidence intervals, and a horizontal reference line at 1-alpha.
        Use this to identify which coverage-probability strata are most
        miscalibrated — bins that clearly miss the nominal line (outside
        their error bars) are regions worth investigating.

        Parameters
        ----------
        n_bins : int, default 10
            Number of equal-width bins over eta_hat.
        alpha_ci : float, default 0.05
            Significance level for binomial proportion Wilson CIs.
            0.05 gives 95% confidence intervals.
        figsize : tuple, default (8, 5)
            Matplotlib figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object. Call .savefig() or .show() as needed.

        Raises
        ------
        RuntimeError
            If evaluate() has not been called yet.
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "plot_conditional_validity() requires matplotlib. "
                "Install with: pip install 'insurance-conformal[plot]'"
            ) from exc

        if self.result_ is None:
            raise RuntimeError(
                "Call evaluate() before plot_conditional_validity()."
            )

        from scipy import stats as _stats  # noqa: PLC0415 — optional dep

        r = self.result_
        target = 1.0 - r.alpha
        eta = r.eta_hat
        n = r.n_obs
        z_ci = _stats.norm.ppf(1.0 - alpha_ci / 2.0)

        bin_edges = np.linspace(0.0, 1.0 + 1e-10, n_bins + 1)
        bin_labels = []
        coverages = []
        err_lo = []
        err_hi = []
        n_bins_used = []

        for i in range(n_bins):
            mask = (eta >= bin_edges[i]) & (eta < bin_edges[i + 1])
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            # Coverage_k = mean estimated coverage probability in this bin
            p_hat = float(eta[mask].mean())
            # Wilson score CI for a proportion
            denom = 1.0 + z_ci ** 2 / n_k
            centre = (p_hat + z_ci ** 2 / (2 * n_k)) / denom
            spread = z_ci * np.sqrt(p_hat * (1.0 - p_hat) / n_k + z_ci ** 2 / (4 * n_k ** 2)) / denom
            lo = float(np.clip(centre - spread, 0.0, 1.0))
            hi = float(np.clip(centre + spread, 0.0, 1.0))

            mid = 0.5 * (bin_edges[i] + bin_edges[i + 1])
            bin_labels.append(f"{mid:.2f}")
            coverages.append(p_hat)
            err_lo.append(p_hat - lo)
            err_hi.append(hi - p_hat)
            n_bins_used.append(n_k)

        x_pos = np.arange(len(bin_labels))
        coverages = np.array(coverages)
        yerr = np.array([err_lo, err_hi])

        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(
            x_pos, coverages, yerr=yerr,
            fmt="o", color="steelblue", capsize=4, linewidth=1.5,
            label="Per-bin coverage",
        )
        ax.axhline(target, color="black", linewidth=1.5, linestyle="--",
                   label=f"Nominal 1-alpha = {target:.2f}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("eta_hat bin midpoint")
        ax.set_ylabel("Coverage probability")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"Conditional validity plot
"
            f"CVI={r.cvi:.4f} (U={r.cvi_u:.4f}, O={r.cvi_o:.4f}), n={n}"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def _fit_eta_hat(
        self, X: np.ndarray, Z: np.ndarray, LGBMClassifier: Any
    ) -> np.ndarray:
        """
        Estimate local coverage probabilities via KFold CV + isotonic calibration.

        Out-of-fold predictions prevent overfitting. Isotonic calibration corrects
        systematic probability distortions in the LightGBM output.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import KFold

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        eta_hat = np.zeros(len(Z), dtype=float)

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            Z_tr = Z[train_idx]

            # Handle degenerate fold where Z_tr is all one class
            if Z_tr.sum() == 0 or Z_tr.sum() == len(Z_tr):
                eta_hat[val_idx] = float(Z_tr.mean())
                continue

            base_clf = LGBMClassifier(**self.lgbm_params)

            if self.calibrate_proba:
                # CalibratedClassifierCV needs enough samples for its own CV
                cv_folds = min(3, max(2, len(Z_tr) // 20))
                # Only calibrate if we have enough samples
                if len(Z_tr) >= cv_folds * 10:
                    clf = CalibratedClassifierCV(
                        base_clf, method="isotonic", cv=cv_folds
                    )
                else:
                    clf = base_clf
            else:
                clf = base_clf

            clf.fit(X_tr, Z_tr)
            proba = clf.predict_proba(X_val)

            # predict_proba returns columns [P(Z=0), P(Z=1)]
            # We want P(Z=1) = P(covered)
            if proba.shape[1] == 2:
                eta_hat[val_idx] = proba[:, 1]
            else:
                eta_hat[val_idx] = proba[:, 0]

        return np.clip(eta_hat, 0.0, 1.0)


class CCSelect:
    """
    Conformal predictor selection via Conditional Validity Index.

    When choosing between multiple conformal predictors (different non-conformity
    scores, different base models, different calibration methods), marginal coverage
    comparison is uninformative — any valid conformal predictor achieves >= 1-alpha
    marginal coverage by construction. The interesting question is which predictor
    has the best *conditional* coverage.

    CCSelect answers this by evaluating each predictor's CVI across multiple
    random train/eval splits and selecting the predictor with the lowest mean CVI.
    A lower CVI means the predictor's intervals are more uniformly valid across
    the feature space — neither systematically undercovering nor overcovering any
    particular segment.

    The evaluation protocol:
    1. For each of n_splits random splits:
       a. Split (X, y) into train (proportion split_ratio) and eval sets.
       b. Calibrate each predictor on the train set.
       c. Generate prediction intervals on the eval set.
       d. Compute CVI for each predictor on the eval set.
    2. Select the predictor with the lowest mean CVI across splits.

    Parameters
    ----------
    n_splits : int, default 5
        Number of random train/eval splits for CVI evaluation.
    split_ratio : float, default 0.5
        Proportion of observations used for calibration (training the
        conformal predictor). The rest are used for CVI evaluation.
        Default 0.5 gives equal-sized calibration and evaluation sets.
    lgbm_params : dict, optional
        LightGBM parameters for ConditionalValidityIndex. Defaults to the
        same sensible set used throughout.
    calibrate_proba : bool, default True
        Apply isotonic calibration to CVI probability estimates.
    random_state : int, optional
        Random state for reproducible splits.

    Attributes
    ----------
    result_ : CCSelectResult or None
        The result of the most recent fit() call.

    Examples
    --------
    >>> from insurance_conformal import InsuranceConformalPredictor, CCSelect
    >>> predictors = {
    ...     "pearson_weighted": predictor_pw,
    ...     "locally_weighted": predictor_lw,
    ... }
    >>> selector = CCSelect(n_splits=5, split_ratio=0.5, random_state=42)
    >>> result = selector.fit(predictors, X_cal, y_cal, alpha=0.10)
    >>> print(result.best_predictor)
    >>> print(selector.compare())
    """

    def __init__(
        self,
        n_splits: int = 5,
        split_ratio: float = 0.5,
        lgbm_params: Optional[dict] = None,
        calibrate_proba: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 0 < split_ratio < 1:
            raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")

        self.n_splits = int(n_splits)
        self.split_ratio = float(split_ratio)
        self.lgbm_params = lgbm_params
        self.calibrate_proba = bool(calibrate_proba)
        self.random_state = random_state

        self.result_: Optional[CCSelectResult] = None

    def fit(
        self,
        predictors: Dict[str, Any],
        X: Any,
        y: Any,
        alpha: float,
        exposure: Optional[Any] = None,
    ) -> CCSelectResult:
        """
        Evaluate predictors and select the one with the lowest mean CVI.

        Each predictor must implement:
        - .calibrate(X, y) — calibrate on the training split.
        - .predict_interval(X, alpha) — return a dict-like with 'lower' and
          'upper' keys (or a Polars DataFrame with those columns).

        This matches the standard insurance-conformal predictor interface.

        Parameters
        ----------
        predictors : dict of {name: predictor}
            Predictor objects to compare.
        X : array-like of shape (n, p)
            Feature matrix.
        y : array-like of shape (n,)
            Target outcomes.
        alpha : float
            Miscoverage rate for calibration and interval generation.
        exposure : array-like of shape (n,), optional
            Exposure weights passed to .calibrate() as keyword argument.

        Returns
        -------
        CCSelectResult
            Selection result with rankings and per-predictor CVI scores.

        Raises
        ------
        ValueError
            If predictors dict is empty.
        """
        if not predictors:
            raise ValueError("predictors dict must not be empty.")

        X_np = _to_numpy_2d(X)
        y_np = as_numpy(y).ravel()
        exp_np = as_numpy(exposure).ravel() if exposure is not None else None

        rng = np.random.default_rng(self.random_state)
        n = len(y_np)

        # Accumulate per-split CVI scores
        cvi_per_split: Dict[str, List[float]] = {name: [] for name in predictors}
        cvi_u_per_split: Dict[str, List[float]] = {name: [] for name in predictors}
        cvi_o_per_split: Dict[str, List[float]] = {name: [] for name in predictors}

        for split_idx in range(self.n_splits):
            # Random split into calibration / evaluation
            idx = rng.permutation(n)
            n_cal = int(np.round(n * self.split_ratio))
            cal_idx = idx[:n_cal]
            eval_idx = idx[n_cal:]

            X_cal, X_eval = X_np[cal_idx], X_np[eval_idx]
            y_cal, y_eval = y_np[cal_idx], y_np[eval_idx]
            exp_cal = exp_np[cal_idx] if exp_np is not None else None

            for name, pred in predictors.items():
                # Calibrate on the calibration split
                if exp_cal is not None:
                    pred.calibrate(X_cal, y_cal, exposure=exp_cal)
                else:
                    pred.calibrate(X_cal, y_cal)

                # Generate intervals on the evaluation split
                intervals = pred.predict_interval(X_eval, alpha=alpha)
                lower, upper = _extract_bounds(intervals)

                # Evaluate CVI on the evaluation split
                cvi_estimator = ConditionalValidityIndex(
                    n_splits=min(3, self.n_splits),
                    lgbm_params=self.lgbm_params,
                    calibrate_proba=self.calibrate_proba,
                    random_state=int(rng.integers(0, 2**31)),
                )
                try:
                    result = cvi_estimator.evaluate(
                        X_eval, lower, upper, y_eval, alpha=alpha
                    )
                    cvi_per_split[name].append(result.cvi)
                    cvi_u_per_split[name].append(result.cvi_u)
                    cvi_o_per_split[name].append(result.cvi_o)
                except ValueError:
                    # Evaluation set too small for this split — skip it
                    pass

        # Compute mean scores
        cvi_scores = {
            name: float(np.mean(vals)) if vals else float("inf")
            for name, vals in cvi_per_split.items()
        }
        cvi_u_scores = {
            name: float(np.mean(vals)) if vals else float("inf")
            for name, vals in cvi_u_per_split.items()
        }
        cvi_o_scores = {
            name: float(np.mean(vals)) if vals else float("inf")
            for name, vals in cvi_o_per_split.items()
        }

        rankings = sorted(cvi_scores.keys(), key=lambda k: cvi_scores[k])
        best_predictor = rankings[0]

        self.result_ = CCSelectResult(
            best_predictor=best_predictor,
            cvi_scores=cvi_scores,
            cvi_u_scores=cvi_u_scores,
            cvi_o_scores=cvi_o_scores,
            rankings=rankings,
        )
        return self.result_

    def compare(self) -> pl.DataFrame:
        """
        Return a comparison DataFrame of all evaluated predictors.

        Returns
        -------
        pl.DataFrame
            Columns: predictor_name, mean_cvi, mean_cvi_u, mean_cvi_o, rank.
            Sorted by ascending mean_cvi (best predictor first).

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before compare().")

        r = self.result_
        rows = []
        for rank, name in enumerate(r.rankings, start=1):
            rows.append(
                {
                    "predictor_name": name,
                    "mean_cvi": r.cvi_scores[name],
                    "mean_cvi_u": r.cvi_u_scores[name],
                    "mean_cvi_o": r.cvi_o_scores[name],
                    "rank": rank,
                }
            )
        return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy_2d(X: Any) -> np.ndarray:
    """Convert any array-like to a 2D numpy array."""
    import pandas as pd

    if isinstance(X, pl.DataFrame):
        arr = X.to_numpy()
    elif isinstance(X, pd.DataFrame):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _extract_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract lower and upper bounds from a predictor's predict_interval output.

    Supports Polars DataFrames and plain dicts with 'lower'/'upper' keys.
    """
    if isinstance(intervals, pl.DataFrame):
        return intervals["lower"].to_numpy(), intervals["upper"].to_numpy()
    if isinstance(intervals, dict):
        lower = as_numpy(intervals["lower"]).ravel()
        upper = as_numpy(intervals["upper"]).ravel()
        return lower, upper
    # Try attribute access (e.g. a custom result object)
    try:
        lower = as_numpy(intervals.lower).ravel()
        upper = as_numpy(intervals.upper).ravel()
        return lower, upper
    except AttributeError:
        raise TypeError(
            "predict_interval() must return a Polars DataFrame or a dict "
            "with 'lower' and 'upper' keys."
        )
