"""
LoBoostCP: Model-native local conformal prediction for gradient-boosted trees.

Standard conformal prediction treats the GBM as a black box — same nonconformity
threshold for every test point. But GBTs are not black boxes: their leaf structure
encodes the model's own uncertainty partitioning. Risks that end up in the same
leaf are already judged similar by the model. LoBoost (Santos et al., 2025,
arXiv:2602.22432) exploits this.

The core idea: for each calibration point, record which leaf it falls into across
all trees. Points that share many leaves share a local context. Use the calibration
residuals from that context — not from the entire calibration set — to set the
conformal threshold.

Concretely, at prediction time for a new risk x:
1. Extract x's leaf assignments from all T trees.
2. Find the calibration points that share the same leaf assignment in *at least
   one tree* (locality set).
3. If the locality set is too small (< min_samples), fall back to the full
   calibration set (marginal conformal). Intermediate fallback to tree-level
   groupings is also possible.
4. Compute the conformal quantile of nonconformity scores from that local set.
5. Return lower/upper bounds centred on the point prediction.

This requires NO retraining. The leaf structure is a by-product of the already-
fitted model. Calibration is a single forward pass.

Score options:
- absolute:  |y - ŷ|   (symmetric, standard regression)
- normalized: |y - ŷ| / ŷ  (multiplicative errors, appropriate for Tweedie/Poisson
  where the variance scales with the mean — roughly equivalent to Pearson residuals
  when p=2, the Gamma case)

Supported backends: CatBoost, XGBoost, LightGBM, and sklearn GradientBoosting
(the sklearn backend uses apply() which returns leaf indices directly).

Design choice: locality grouping is done by majority-leaf overlap rather than full
leaf-vector equality. Full equality is too strict — with T=500 trees, exact match
probability collapses to near-zero. We use a minimum overlap fraction (overlap_frac)
to define the locality set. Default 0.5 means a calibration point must share >=50%
of leaf assignments with the test point. This is a tunable locality-vs-sample-size
tradeoff.

Reference:
    Santos, Izbicki & Stern (2025). "LoBoost: Locally Boosted Conformal Prediction."
    arXiv:2602.22432.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy, conformal_quantile


# ---------------------------------------------------------------------------
# Leaf extraction dispatch
# ---------------------------------------------------------------------------


def _get_leaf_indices(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Extract per-tree leaf assignments for all rows in X.

    Returns an array of shape (n_samples, n_trees) where entry [i, t]
    is the leaf index that sample i falls into in tree t.

    Supports CatBoost, XGBoost, LightGBM, and sklearn
    GradientBoostingRegressor/GradientBoostingClassifier.

    Parameters
    ----------
    model : fitted GBM model
        One of CatBoost, XGBoost Booster, LightGBM Booster, or sklearn GBM.
    X : np.ndarray
        Input features, shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_trees), dtype int32.

    Raises
    ------
    TypeError
        If the model type is not recognised.
    """
    # --- CatBoost ---
    try:
        import catboost

        if isinstance(model, (catboost.CatBoost, catboost.CatBoostRegressor, catboost.CatBoostClassifier)):
            from catboost import Pool

            pool = Pool(X)
            # calc_leaf_indexes returns (n_samples, n_trees) int array
            leaf_idx = model.calc_leaf_indexes(pool)
            return np.asarray(leaf_idx, dtype=np.int32)
    except ImportError:
        pass

    # --- XGBoost ---
    try:
        import xgboost as xgb

        if isinstance(model, (xgb.Booster, xgb.XGBRegressor, xgb.XGBClassifier)):
            if isinstance(model, xgb.Booster):
                booster = model
                dmat = xgb.DMatrix(X)
            else:
                booster = model.get_booster()
                dmat = xgb.DMatrix(X)
            leaf_idx = booster.predict(dmat, pred_leaf=True)
            # XGBoost returns (n, n_trees) or (n, n_trees * n_classes) for multi-class
            if leaf_idx.ndim == 1:
                leaf_idx = leaf_idx.reshape(-1, 1)
            return np.asarray(leaf_idx, dtype=np.int32)
    except ImportError:
        pass

    # --- LightGBM ---
    try:
        import lightgbm as lgb

        if isinstance(model, (lgb.Booster, lgb.LGBMRegressor, lgb.LGBMClassifier)):
            if isinstance(model, lgb.Booster):
                booster = model
            else:
                booster = model.booster_
            # LightGBM returns (n_samples, n_trees), int
            leaf_idx = booster.predict(X, pred_leaf=True)
            return np.asarray(leaf_idx, dtype=np.int32)
    except ImportError:
        pass

    # --- sklearn GradientBoosting ---
    try:
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

        if isinstance(model, (GradientBoostingRegressor, GradientBoostingClassifier)):
            # apply() returns (n_samples, n_estimators, 1) — squeeze last dim
            leaf_idx = model.apply(X)
            if leaf_idx.ndim == 3:
                leaf_idx = leaf_idx[:, :, 0]
            return np.asarray(leaf_idx, dtype=np.int32)
    except ImportError:
        pass

    raise TypeError(
        f"Unrecognised model type: {type(model).__name__}. "
        "LoBoostCP supports CatBoost, XGBoost (Booster/XGBRegressor), "
        "LightGBM (Booster/LGBMRegressor), and sklearn GradientBoosting."
    )


# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------


def _nonconformity_score(
    y: np.ndarray,
    yhat: np.ndarray,
    score_type: str,
) -> np.ndarray:
    """
    Compute nonconformity scores.

    Parameters
    ----------
    y : np.ndarray
        Observed values.
    yhat : np.ndarray
        Point predictions.
    score_type : str
        One of "absolute" or "normalized".

    Returns
    -------
    np.ndarray
        Non-negative nonconformity scores.
    """
    resid = np.abs(y - yhat)
    if score_type == "absolute":
        return resid
    elif score_type == "normalized":
        denom = np.clip(np.abs(yhat), 1e-8, None)
        return resid / denom
    else:
        raise ValueError(
            f"score_type must be 'absolute' or 'normalized', got {score_type!r}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LoBoostCP:
    """
    Model-native local conformal prediction for gradient-boosted trees.

    Wraps an already-fitted GBM and uses its leaf structure to compute
    prediction intervals with local (rather than global) calibration. No
    retraining is required.

    The locality for each test point is defined by leaf-assignment overlap
    with calibration points: a calibration point is in the local context if
    it shares at least ``overlap_frac`` of its leaf assignments with the test
    point. If the resulting local set has fewer than ``min_samples`` points,
    the predictor falls back to the full calibration set (marginal conformal).

    Parameters
    ----------
    model : fitted GBM
        Supported: CatBoost, XGBoost (Booster or XGBRegressor), LightGBM
        (Booster or LGBMRegressor), sklearn GradientBoostingRegressor.
    alpha : float, default 0.1
        Miscoverage rate. 0.1 gives 90% prediction intervals.
    score_type : {"absolute", "normalized"}, default "absolute"
        Nonconformity score type.

        - "absolute": |y - ŷ|. Appropriate for symmetric Gaussian-like
          residuals. Interval width is on the same scale as the target.
        - "normalized": |y - ŷ| / ŷ. Appropriate for heteroscedastic
          (Tweedie/Poisson) data where the residual variance scales with
          the mean. Equivalent to a Gamma-type Pearson residual (p=2).
          Use this for claim severity models.

    min_samples : int, default 30
        Minimum number of calibration points in the local context before
        falling back to the global calibration set. Lower values allow
        tighter local intervals at the cost of higher variance; higher
        values make the method closer to standard marginal conformal.
        30 gives roughly ±0.15 finite-sample correction on the quantile
        level for alpha=0.1.
    overlap_frac : float, default 0.5
        Fraction of trees that must assign the same leaf for two points to
        be considered locally similar. At 1.0, the local set is exact leaf-
        vector matches (very sparse). At 0.0, every calibration point is
        in every local set (equivalent to marginal conformal). 0.5 is a
        sensible default that gives locality sets of useful size across
        typical ensemble sizes (100-1000 trees).

    Attributes
    ----------
    cal_leaves_ : np.ndarray
        Leaf assignments for the calibration set, shape (n_cal, n_trees).
    cal_scores_ : np.ndarray
        Nonconformity scores for the calibration set, shape (n_cal,).
    cal_yhat_ : np.ndarray
        Point predictions on calibration set.
    n_calibration_ : int
        Number of calibration points.
    is_calibrated_ : bool
        Whether calibrate() has been called.

    Examples
    --------
    Wrap a fitted CatBoost model:

    >>> lcp = LoBoostCP(model=fitted_catboost, alpha=0.1)
    >>> lcp.calibrate(X_cal, y_cal)
    >>> intervals = lcp.predict(X_test)
    >>> intervals.columns
    ['lower', 'upper', 'point_pred']

    Normalized score for Tweedie severity model:

    >>> lcp = LoBoostCP(
    ...     model=fitted_catboost_gamma,
    ...     alpha=0.1,
    ...     score_type="normalized",
    ... )
    >>> lcp.calibrate(X_cal, y_cal)
    >>> intervals = lcp.predict(X_test)

    Coverage report:

    >>> report = lcp.coverage_report(X_test, y_test)
    >>> print(report)

    References
    ----------
    Santos, Izbicki & Stern (2025). "LoBoost: Locally Boosted Conformal
    Prediction." arXiv:2602.22432.
    """

    def __init__(
        self,
        model: Any,
        alpha: float = 0.1,
        score_type: Literal["absolute", "normalized"] = "absolute",
        min_samples: int = 30,
        overlap_frac: float = 0.5,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if score_type not in ("absolute", "normalized"):
            raise ValueError(
                f"score_type must be 'absolute' or 'normalized', got {score_type!r}"
            )
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if not 0.0 <= overlap_frac <= 1.0:
            raise ValueError(f"overlap_frac must be in [0, 1], got {overlap_frac}")

        self.model = model
        self.alpha = float(alpha)
        self.score_type = score_type
        self.min_samples = int(min_samples)
        self.overlap_frac = float(overlap_frac)

        self.cal_leaves_: Optional[np.ndarray] = None
        self.cal_scores_: Optional[np.ndarray] = None
        self.cal_yhat_: Optional[np.ndarray] = None
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, X_cal: Any, y_cal: Any) -> "LoBoostCP":
        """
        Calibrate using a held-out calibration set.

        Computes and stores leaf assignments and nonconformity scores for the
        calibration set. These are used at prediction time to find the local
        context for each test point.

        Parameters
        ----------
        X_cal : array-like
            Calibration features (must not overlap with training data).
        y_cal : array-like
            Calibration targets.

        Returns
        -------
        self
        """
        X_np = self._to_numpy(X_cal)
        y_np = as_numpy(y_cal).ravel()

        if len(X_np) != len(y_np):
            raise ValueError(
                f"X_cal has {len(X_np)} rows but y_cal has {len(y_np)} entries. "
                "Lengths must match."
            )

        yhat = self._base_predict(X_np)
        leaves = _get_leaf_indices(self.model, X_np)

        self.cal_leaves_ = leaves  # (n_cal, n_trees)
        self.cal_scores_ = _nonconformity_score(y_np, yhat, self.score_type)
        self.cal_yhat_ = yhat
        self.n_calibration_ = len(y_np)
        self.is_calibrated_ = True

        return self

    def predict(self, X_test: Any) -> pl.DataFrame:
        """
        Produce prediction intervals for new observations.

        For each test point, finds the local calibration context (those
        calibration points sharing at least ``overlap_frac`` of leaf
        assignments), computes the conformal quantile from that local set,
        and constructs the interval.

        Parameters
        ----------
        X_test : array-like
            Test features.

        Returns
        -------
        pl.DataFrame
            Columns: lower, upper, point_pred.
            All float64. lower is clipped at 0 for non-negative targets
            (use score_type="normalized" for this to be meaningful).
        """
        self._check_calibrated()

        X_np = self._to_numpy(X_test)
        n_test = len(X_np)

        yhat = self._base_predict(X_np)
        test_leaves = _get_leaf_indices(self.model, X_np)  # (n_test, n_trees)

        n_trees = self.cal_leaves_.shape[1]
        min_overlap = int(np.ceil(self.overlap_frac * n_trees))

        lower = np.empty(n_test, dtype=np.float64)
        upper = np.empty(n_test, dtype=np.float64)

        # Vectorised overlap computation: broadcast (n_test, n_trees) against
        # (n_cal, n_trees) — for large n_cal this may be memory-intensive.
        # We chunk if needed, but for typical calibration sizes (1k-50k) this
        # is fine (50k * 500 trees * 4 bytes = 100 MB, manageable).
        cal_leaves = self.cal_leaves_  # (n_cal, n_trees)
        cal_scores = self.cal_scores_

        for i in range(n_test):
            # Overlap count: how many trees assign the same leaf to test[i] and
            # each calibration point. Shape: (n_cal,)
            overlap = (test_leaves[i : i + 1, :] == cal_leaves).sum(axis=1)
            local_mask = overlap >= min_overlap

            if local_mask.sum() >= self.min_samples:
                local_scores = cal_scores[local_mask]
            else:
                # Fall back to global calibration set
                local_scores = cal_scores
                warnings.warn(
                    f"Test point {i}: local context has {local_mask.sum()} samples "
                    f"(< min_samples={self.min_samples}). "
                    "Falling back to global calibration set for this point.",
                    UserWarning,
                    stacklevel=2,
                )

            q = conformal_quantile(local_scores, self.alpha)
            half = self._half_width(yhat[i], q)
            lower[i] = max(0.0, yhat[i] - half)
            upper[i] = yhat[i] + half

        return pl.DataFrame(
            {
                "lower": lower,
                "upper": upper,
                "point_pred": yhat,
            }
        )

    def coverage_report(self, X_test: Any, y_test: Any) -> pl.DataFrame:
        """
        Marginal and conditional coverage diagnostics.

        Reports:
        1. Marginal coverage: overall fraction of y_test falling in the interval.
        2. Coverage by decile of point prediction.
        3. Coverage by quartile of interval width.

        These three slices expose undercoverage in tails (decile slice) and
        systematic bias in interval width (width slice).

        Parameters
        ----------
        X_test : array-like
            Test features.
        y_test : array-like
            Observed values on the test set.

        Returns
        -------
        pl.DataFrame
            Columns: slice_type, bin, n_obs, coverage, target_coverage,
            mean_point_pred, mean_width.
        """
        self._check_calibrated()

        import pandas as pd

        intervals = self.predict(X_test)
        y = as_numpy(y_test).ravel()

        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        yhat = intervals["point_pred"].to_numpy()
        width = upper - lower
        covered = (y >= lower) & (y <= upper)
        target = 1.0 - self.alpha

        rows = []

        # 1. Marginal
        rows.append(
            {
                "slice_type": "marginal",
                "bin": 0,
                "n_obs": len(y),
                "coverage": float(covered.mean()),
                "target_coverage": target,
                "mean_point_pred": float(yhat.mean()),
                "mean_width": float(width.mean()),
            }
        )

        # 2. By decile of point prediction
        try:
            decile_labels = pd.qcut(yhat, q=10, labels=False, duplicates="drop")
        except ValueError:
            decile_labels = pd.cut(yhat, bins=10, labels=False, duplicates="drop")
        decile_labels = np.asarray(decile_labels, dtype=float)

        for d in np.unique(decile_labels[~np.isnan(decile_labels)]):
            mask = decile_labels == d
            rows.append(
                {
                    "slice_type": "pred_decile",
                    "bin": int(d) + 1,
                    "n_obs": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "target_coverage": target,
                    "mean_point_pred": float(yhat[mask].mean()),
                    "mean_width": float(width[mask].mean()),
                }
            )

        # 3. By quartile of interval width
        edges = np.percentile(width, [0, 25, 50, 75, 100])
        edges[-1] += 1e-10
        bin_labels = np.clip(np.digitize(width, edges[1:], right=False), 0, 3)
        for q_bin in range(4):
            mask = bin_labels == q_bin
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "slice_type": "width_quartile",
                    "bin": int(q_bin) + 1,
                    "n_obs": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "target_coverage": target,
                    "mean_point_pred": float(yhat[mask].mean()),
                    "mean_width": float(width[mask].mean()),
                }
            )

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_predict(self, X: np.ndarray) -> np.ndarray:
        """Run model.predict() and return a 1D float64 array."""
        return as_numpy(self.model.predict(X)).ravel().astype(np.float64)

    def _half_width(self, yhat_i: float, quantile: float) -> float:
        """
        Convert a quantile of nonconformity scores to an interval half-width.

        For score_type="absolute": half_width = quantile (scores are in Y-space).
        For score_type="normalized": scores are |y - ŷ| / ŷ, so half_width = quantile * ŷ.
        """
        if self.score_type == "absolute":
            return float(quantile)
        else:  # normalized
            return float(quantile * max(yhat_i, 1e-8))

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        """Convert DataFrame/Series to numpy, preserving 2D structure."""
        import pandas as pd

        if isinstance(X, pl.DataFrame):
            return X.to_numpy()
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.asarray(X)

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "LoBoostCP has not been calibrated. "
                "Call .calibrate(X_cal, y_cal) before .predict()."
            )

    def __repr__(self) -> str:
        status = (
            f"calibrated on {self.n_calibration_} obs"
            if self.is_calibrated_
            else "not calibrated"
        )
        return (
            f"LoBoostCP("
            f"alpha={self.alpha}, "
            f"score_type={self.score_type!r}, "
            f"min_samples={self.min_samples}, "
            f"overlap_frac={self.overlap_frac}, "
            f"{status})"
        )
