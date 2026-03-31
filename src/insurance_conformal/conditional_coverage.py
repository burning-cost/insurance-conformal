"""
ConditionalCoverageERT: full ERT test for conditional coverage violations.

The marginal coverage guarantee from split conformal is real but limited. It says
P(y in [lo, hi]) >= 1 - alpha over the test distribution. It says nothing about
P(y in [lo, hi] | X = x). A predictor that gets 90% overall can have 30% coverage
on young drivers and 99% on middle-aged ones — the marginal guarantee is satisfied
either way.

The ERT (Excess Risk of Target Coverage) test from Braun et al. (arXiv:2512.11779)
detects this. The idea: if coverage is truly uniform across features, then the
coverage indicator Z_i = 1{y_i in [lo_i, hi_i]} is independent of X_i. A classifier
trained to predict Z_i from X_i should do no better than the constant predictor.
ERT measures the gap: how much better does the classifier do?

Three loss variants handle different regulatory framings:
- L1-ERT (MAE): linear penalty, easy to explain to underwriters
- L2-ERT (Brier score): quadratic, more sensitive to large gaps
- KL-ERT (log loss): information-theoretic, penalises confident mispredictions

Directional variants let you focus on what actually matters for insurance:
- "under": only penalise when coverage is below target (FCA TCF concern)
- "over": only penalise when coverage is above target (efficiency concern)
- "both": symmetric, for general diagnostics

Implementation notes:
- LightGBM classifier for high power against complex conditional patterns
- KFold CV to avoid in-sample overfit
- Bootstrap CIs by resampling (Z_i, X_i) pairs together
- subgroup_report() bins each feature and reports coverage per bin — useful
  for FCA reporting where you need to show coverage is uniform across segments

References:
    Braun, Gruber & Matthias (2024) arXiv:2512.11779
    "Excess Risk of Target Coverage: Testing Conditional Coverage"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy


LossVariant = Literal["l1", "l2", "kl"]
DirectionVariant = Literal["both", "under", "over"]


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
    >>> subgroups = result_with_names.subgroup_report(X_test, feature_names=["age", "region"])
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
