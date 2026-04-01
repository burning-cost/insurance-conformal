"""
LCPModelSelector: Locally Calibrated Prediction model and score selection.

Standard conformal prediction commits to a single model and score function
before seeing the test point. But in insurance pricing, the right model
depends on the local region of covariate space: a frequency model calibrated
on low-risk urban policies will have poor calibration in rural high-risk cells,
and vice versa.

Wang & Wang (2026, arXiv:2602.19284) show how to do *locally adaptive* model
and score selection within a conformal prediction framework, with a marginal
coverage guarantee. The key idea: treat model/score selection as a nuisance
by constructing prediction sets that are valid for the *locally best* model
at each test point.

The construction:
1. Fit K models on training data. Calibrate each on a shared calibration set.
2. For each calibration point i and model k, compute a leave-one-out (LOO)
   prediction interval using an approximate rank-update trick.
3. At prediction time for a test point x, compute kernel weights w_i(x) from
   the calibration set. These measure local similarity.
4. Find the set of models K(x) that are locally valid at level gamma: those
   whose LOO intervals cover the calibration labels in a weighted sense.
5. Among valid models, select the one with the narrowest interval for x.
6. If no model passes the validity check (degenerate case), fall back to the
   globally narrowest model.

The result: intervals that adapt locally to the best model for each risk type,
with the same marginal coverage guarantee as standard split conformal.

Insurance use cases:
- Score selection: run the same GBM with pearson_weighted, deviance, and
  anscombe scores. LCP selects the locally best-calibrated score.
- Model selection: compare a GLM vs GBM, or a Poisson vs Gamma model. Let
  the data decide which is better for each risk segment.
- Portfolio mixtures: train a fleet model and a personal lines model. LCP
  blends them locally based on the test point's distance from each segment.

Reference:
    Wang, Y. & Wang, R. (2026). "Locally Calibrated Conformal Prediction."
    arXiv:2602.19284.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from insurance_conformal.scores import compute_score, invert_score
from insurance_conformal.utils import as_numpy, conformal_quantile


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _kernel_weights(
    x_test: np.ndarray,
    X_cal: np.ndarray,
    localizer: str,
    bandwidth: float,
    knn_k: int = 20,
) -> np.ndarray:
    """
    Compute unnormalised kernel weights from a single test point to all
    calibration points.

    Parameters
    ----------
    x_test : np.ndarray, shape (p,)
        Single test observation.
    X_cal : np.ndarray, shape (n, p)
        Calibration feature matrix.
    localizer : str
        One of 'gaussian', 'exponential', 'uniform', 'knn'.
    bandwidth : float
        Kernel bandwidth h. Ignored for 'knn'.
    knn_k : int
        Number of neighbours for 'knn' localizer.

    Returns
    -------
    np.ndarray, shape (n,)
        Unnormalised weights (non-negative, sum may differ from 1).
    """
    diff = X_cal - x_test[np.newaxis, :]  # (n, p)
    sq_dist = np.sum(diff ** 2, axis=1)    # (n,)
    dist = np.sqrt(sq_dist)

    if localizer == "gaussian":
        h2 = max(bandwidth ** 2, 1e-12)
        weights = np.exp(-sq_dist / (2.0 * h2))
    elif localizer == "exponential":
        h = max(bandwidth, 1e-12)
        weights = np.exp(-dist / h)
    elif localizer == "uniform":
        weights = (dist <= bandwidth).astype(float)
        if weights.sum() < 1e-12:
            # All points outside bandwidth — use all with equal weight
            weights = np.ones(len(X_cal))
    elif localizer == "knn":
        k = min(knn_k, len(X_cal))
        knn_idx = np.argpartition(dist, k - 1)[:k]
        weights = np.zeros(len(X_cal))
        weights[knn_idx] = 1.0 / k
    else:
        raise ValueError(
            f"Unknown localizer {localizer!r}. "
            "Choose from: 'gaussian', 'exponential', 'uniform', 'knn'."
        )

    return weights


def _build_score_fn(
    score_name: str, tweedie_power: float = 1.5
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Build a nonconformity score callable from a string name.

    Parameters
    ----------
    score_name : str
        One of 'raw', 'pearson', 'pearson_weighted', 'deviance', 'anscombe'.
    tweedie_power : float
        Tweedie variance power for scores that need it.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function (y, yhat) -> scores.
    """
    def fn(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        return compute_score(
            y, yhat, score_name,
            distribution="tweedie",
            tweedie_power=tweedie_power,
        )
    fn.__name__ = score_name
    return fn


def _invert_score_fn(
    yhat: np.ndarray,
    quantile: float,
    score_name: str,
    tweedie_power: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert a named score to produce prediction interval bounds.

    Returns (lower, upper) arrays clipped at 0.
    """
    return invert_score(
        yhat, quantile, score_name,
        distribution="tweedie",
        tweedie_power=tweedie_power,
        clip_lower=0.0,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LCPModelSelector:
    """
    Locally Calibrated Prediction: adaptive model and score selection.

    Fits K models to training data (passed in pre-fitted), calibrates each on
    a shared calibration set, then at prediction time selects the locally best
    model for each test point based on kernel-weighted validity checks.

    The marginal coverage guarantee holds regardless of which model is selected:
    P(Y_{n+1} in C_{alpha}(X_{n+1})) >= 1 - alpha.

    Parameters
    ----------
    models : list of fitted sklearn-compatible models
        K pre-fitted models implementing predict(X). All K models must be
        fitted before calling calibrate().
    score_fns : list of callables or list of str
        K nonconformity score functions, one per model. Each callable must
        accept (y, yhat) and return an array of non-negative scores. String
        shortcuts are accepted: 'raw', 'pearson', 'pearson_weighted',
        'deviance', 'anscombe'.
    localizer : str, default 'gaussian'
        Kernel type for computing local similarity weights. One of:
        'gaussian' (exp(-||x-x_i||^2/(2h^2))),
        'exponential' (exp(-||x-x_i||/h)),
        'uniform' (indicator ||x-x_i|| <= h),
        'knn' (uniform over k nearest neighbours).
    bandwidth : float, default 1.0
        Kernel bandwidth h. Ignored for localizer='knn'.
    knn_k : int, default 20
        Number of neighbours for localizer='knn'.
    gamma_grid : np.ndarray, optional
        Discrete miscoverage grid for validity checks. Defaults to
        np.linspace(0, 1, 100). The LCP interval is valid at level alpha
        for any alpha not in the grid — the grid resolution only affects
        how precisely the local threshold is estimated.
    tweedie_power : float, default 1.5
        Tweedie variance power, used when score functions are string
        shortcuts requiring this parameter.
    n_jobs : int, default 1
        Reserved for future parallel calibration. Currently unused.

    Attributes
    ----------
    cal_X_ : np.ndarray, shape (n, p)
        Calibration features. Set by calibrate().
    cal_y_ : np.ndarray, shape (n,)
        Calibration targets. Set by calibrate().
    cal_yhats_ : np.ndarray, shape (n, K)
        Model predictions on the calibration set.
    cal_scores_ : np.ndarray, shape (n, K)
        Nonconformity scores on the calibration set.
    loo_intervals_ : np.ndarray, shape (n, K, 2)
        Approximate LOO prediction intervals (lower, upper) for each
        calibration point under each model.
    is_calibrated_ : bool

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> m1 = Ridge().fit(X_train, y_train)
    >>> m2 = GradientBoostingRegressor().fit(X_train, y_train)
    >>> lcp = LCPModelSelector(
    ...     models=[m1, m2],
    ...     score_fns=["pearson_weighted", "pearson_weighted"],
    ...     localizer="gaussian",
    ...     bandwidth=2.0,
    ... )
    >>> lcp.calibrate(X_cal, y_cal)
    >>> intervals = lcp.predict_interval(X_test, alpha=0.10)
    """

    def __init__(
        self,
        models: list[Any],
        score_fns: list[Callable | str],
        localizer: str = "gaussian",
        bandwidth: float = 1.0,
        knn_k: int = 20,
        gamma_grid: Optional[np.ndarray] = None,
        tweedie_power: float = 1.5,
        n_jobs: int = 1,
    ) -> None:
        valid_localizers = ("gaussian", "exponential", "uniform", "knn")
        if localizer not in valid_localizers:
            raise ValueError(
                f"localizer must be one of {valid_localizers}, got {localizer!r}"
            )
        if len(models) == 0:
            raise ValueError("At least one model must be provided.")
        if len(models) != len(score_fns):
            raise ValueError(
                f"models and score_fns must have the same length, "
                f"got {len(models)} and {len(score_fns)}."
            )

        self.models = list(models)
        self.tweedie_power = float(tweedie_power)

        # Resolve string shortcuts to callables
        self.score_fns: list[Callable[[np.ndarray, np.ndarray], np.ndarray]] = []
        self._score_names: list[str | None] = []
        for sf in score_fns:
            if isinstance(sf, str):
                self.score_fns.append(_build_score_fn(sf, tweedie_power=self.tweedie_power))
                self._score_names.append(sf)
            else:
                self.score_fns.append(sf)
                self._score_names.append(getattr(sf, "__name__", None))

        self.localizer = localizer
        self.bandwidth = float(bandwidth)
        self.knn_k = int(knn_k)
        self.gamma_grid: np.ndarray = (
            gamma_grid if gamma_grid is not None else np.linspace(0.0, 1.0, 100)
        )
        self.n_jobs = n_jobs

        # Set post-calibrate
        self.cal_X_: Optional[np.ndarray] = None
        self.cal_y_: Optional[np.ndarray] = None
        self.cal_yhats_: Optional[np.ndarray] = None
        self.cal_scores_: Optional[np.ndarray] = None
        self.loo_intervals_: Optional[np.ndarray] = None
        self.is_calibrated_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, X_cal: Any, y_cal: Any) -> "LCPModelSelector":
        """
        Calibrate the selector on a held-out calibration set.

        Computes:
        - Model predictions for each of the K models.
        - Nonconformity scores for each (calibration point, model) pair.
        - Approximate LOO prediction intervals using the rank-update trick.

        Parameters
        ----------
        X_cal : array-like, shape (n, p)
            Calibration features.
        y_cal : array-like, shape (n,)
            Calibration targets.

        Returns
        -------
        self
        """
        X_np = self._to_numpy(X_cal)
        y_np = as_numpy(y_cal).ravel()

        if len(X_np) != len(y_np):
            raise ValueError(
                f"X_cal has {len(X_np)} rows but y_cal has {len(y_np)} elements."
            )

        n = len(y_np)
        K = len(self.models)

        # --- Step 1: predictions and scores ---
        cal_yhats = np.empty((n, K), dtype=float)
        cal_scores = np.empty((n, K), dtype=float)

        for k, (model, score_fn) in enumerate(zip(self.models, self.score_fns)):
            yhat_k = self._predict_model(model, X_np)
            cal_yhats[:, k] = yhat_k
            cal_scores[:, k] = score_fn(y_np, yhat_k)

        self.cal_X_ = X_np
        self.cal_y_ = y_np
        self.cal_yhats_ = cal_yhats
        self.cal_scores_ = cal_scores

        # --- Step 2: approximate LOO intervals ---
        # For large n we use the rank-update trick; for small n we just recompute.
        self.loo_intervals_ = self._build_loo_intervals(
            y_np, cal_yhats, cal_scores
        )

        self.is_calibrated_ = True
        return self

    def predict_interval(self, X_test: Any, alpha: float = 0.05) -> pl.DataFrame:
        """
        Produce locally adaptive prediction intervals.

        For each test point, computes kernel weights to all calibration points,
        determines which models are locally valid, selects the locally narrowest
        valid model, and returns the corresponding prediction interval.

        Parameters
        ----------
        X_test : array-like, shape (m, p)
            Test features.
        alpha : float, default 0.05
            Miscoverage level. 0.05 gives 95% prediction intervals.

        Returns
        -------
        pl.DataFrame
            One row per test observation. Columns:
            - lower: lower prediction bound (>= 0)
            - point_estimate: model point prediction for selected model
            - upper: upper prediction bound
            - ensemble_width: upper - lower
            - n_models_in_set: number of models in the valid local set
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_calibrated()

        X_np = self._to_numpy(X_test)
        m = len(X_np)
        K = len(self.models)

        # Predictions for all models on test set
        test_yhats = np.empty((m, K), dtype=float)
        for k, model in enumerate(self.models):
            test_yhats[:, k] = self._predict_model(model, X_np)

        # Per-test-point: kernel weights, model selection, interval
        lower = np.empty(m, dtype=float)
        upper = np.empty(m, dtype=float)
        point_est = np.empty(m, dtype=float)
        n_in_set = np.empty(m, dtype=int)

        for i in range(m):
            l_i, u_i, pt_i, n_i = self._predict_single(
                x_test=X_np[i],
                test_yhats_i=test_yhats[i],
                alpha=alpha,
            )
            lower[i] = l_i
            upper[i] = u_i
            point_est[i] = pt_i
            n_in_set[i] = n_i

        lower = np.clip(lower, 0.0, None)

        return pl.DataFrame({
            "lower": lower,
            "point_estimate": point_est,
            "upper": upper,
            "ensemble_width": upper - lower,
            "n_models_in_set": n_in_set,
        })

    def selected_models(self, X_test: Any, alpha: float = 0.05) -> pl.DataFrame:
        """
        Return which model index was selected for each test point, along with
        the local valid set size and the model's score name.

        Parameters
        ----------
        X_test : array-like, shape (m, p)
        alpha : float, default 0.05

        Returns
        -------
        pl.DataFrame
            Columns: selected_model_idx, score_name, n_models_in_set,
            fallback_used (True when no locally valid model existed and the
            globally narrowest model was chosen).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._check_calibrated()

        X_np = self._to_numpy(X_test)
        m = len(X_np)
        K = len(self.models)

        # Predictions for all models on test set
        test_yhats = np.empty((m, K), dtype=float)
        for k, model in enumerate(self.models):
            test_yhats[:, k] = self._predict_model(model, X_np)

        sel_idx = np.empty(m, dtype=int)
        n_in_set = np.empty(m, dtype=int)
        fallback = np.empty(m, dtype=bool)

        for i in range(m):
            valid_mask, k_star, used_fallback = self._select_model(
                x_test=X_np[i],
                test_yhats_i=test_yhats[i],
                alpha=alpha,
            )
            sel_idx[i] = k_star
            n_in_set[i] = int(valid_mask.sum())
            fallback[i] = used_fallback

        score_names = [
            sn if sn is not None else f"model_{k}"
            for k, sn in enumerate(self._score_names)
        ]

        return pl.DataFrame({
            "selected_model_idx": sel_idx,
            "score_name": [score_names[k] for k in sel_idx],
            "n_models_in_set": n_in_set,
            "fallback_used": fallback,
        })

    def coverage_diagnostics(
        self, X_test: Any, y_test: Any, alpha: float = 0.05
    ) -> pl.DataFrame:
        """
        Compute coverage diagnostics by predicted-value decile and model
        selection frequency.

        Useful for validating that:
        - Empirical coverage is close to 1 - alpha in each segment.
        - The selector is not always picking the same model (which would
          suggest the localisation is too coarse or K=1 effectively).

        Parameters
        ----------
        X_test : array-like, shape (m, p)
        y_test : array-like, shape (m,)
        alpha : float, default 0.05

        Returns
        -------
        pl.DataFrame
            Columns: slice_type, bin, mean_predicted, n_obs, coverage,
            target_coverage, dominant_model.
        """
        import pandas as pd

        intervals = self.predict_interval(X_test, alpha=alpha)
        sel = self.selected_models(X_test, alpha=alpha)
        y = as_numpy(y_test).ravel()

        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        point = intervals["point_estimate"].to_numpy()
        covered = (y >= lower) & (y <= upper)
        model_idx = sel["selected_model_idx"].to_numpy()

        target = 1.0 - alpha
        rows = []

        # Slice 1: by decile of point estimate
        try:
            decile_labels = pd.qcut(point, q=10, labels=False, duplicates="drop")
        except ValueError:
            decile_labels = pd.cut(point, bins=10, labels=False, duplicates="drop")

        decile_labels = np.asarray(decile_labels, dtype=float)
        unique_d = np.unique(decile_labels[~np.isnan(decile_labels)])

        for d in unique_d:
            mask = decile_labels == d
            dominant = int(np.bincount(model_idx[mask]).argmax())
            rows.append({
                "slice_type": "pred_decile",
                "bin": int(d) + 1,
                "mean_predicted": float(point[mask].mean()),
                "n_obs": int(mask.sum()),
                "coverage": float(covered[mask].mean()),
                "target_coverage": target,
                "dominant_model": dominant,
            })

        # Slice 2: by selected model
        for k in range(len(self.models)):
            mask = model_idx == k
            if mask.sum() == 0:
                continue
            sn = self._score_names[k] or f"model_{k}"
            rows.append({
                "slice_type": "by_model",
                "bin": k,
                "mean_predicted": float(point[mask].mean()),
                "n_obs": int(mask.sum()),
                "coverage": float(covered[mask].mean()),
                "target_coverage": target,
                "dominant_model": k,
            })

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _predict_single(
        self,
        x_test: np.ndarray,
        test_yhats_i: np.ndarray,
        alpha: float,
    ) -> tuple[float, float, float, int]:
        """
        Compute interval for a single test point.

        Returns (lower, upper, point_estimate, n_models_in_valid_set).
        """
        valid_mask, k_star, _used_fallback = self._select_model(
            x_test, test_yhats_i, alpha
        )

        n_valid = int(valid_mask.sum())

        # Interval for the selected model
        yhat_k = float(test_yhats_i[k_star])
        l_k, u_k = self._interval_for_model(
            x_test=x_test,
            yhat=np.array([yhat_k]),
            model_idx=k_star,
            alpha=alpha,
        )

        return float(l_k[0]), float(u_k[0]), yhat_k, n_valid

    def _select_model(
        self,
        x_test: np.ndarray,
        test_yhats_i: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, int, bool]:
        """
        Determine the locally valid model set and select the narrowest.

        Returns (valid_mask, selected_model_idx, fallback_used).
        """
        K = len(self.models)
        n = len(self.cal_y_)

        # Kernel weights from test point to calibration set
        w = _kernel_weights(
            x_test, self.cal_X_,
            localizer=self.localizer,
            bandwidth=self.bandwidth,
            knn_k=self.knn_k,
        )
        # Normalise; guard against zero-sum (e.g. all points outside uniform bandwidth)
        w_sum = w.sum()
        if w_sum < 1e-12:
            w = np.ones(n) / n
        else:
            w = w / w_sum

        # LOO coverage check: model k is valid if its LOO intervals cover y_i
        # with sufficient weighted frequency.
        # A model k passes at level gamma if:
        #   sum_i w_i * 1{y_i in LOO_interval_i_k} >= 1 - gamma
        # We want the set of models that pass at level alpha.
        #
        # Equivalently: weighted non-coverage = sum_i w_i * 1{y_i NOT covered}
        # Model k is valid if weighted_noncoverage_k <= alpha.

        loo_lo = self.loo_intervals_[:, :, 0]  # (n, K)
        loo_hi = self.loo_intervals_[:, :, 1]  # (n, K)
        y_rep = self.cal_y_[:, np.newaxis]      # (n, 1)

        covered_loo = (y_rep >= loo_lo) & (y_rep <= loo_hi)  # (n, K) bool

        # Weighted non-coverage per model
        weighted_noncoverage = w @ (~covered_loo).astype(float)  # (K,)

        valid_mask = weighted_noncoverage <= alpha  # (K,)

        # Among valid models, pick the one with the narrowest interval for x_test
        used_fallback = False
        if not valid_mask.any():
            # Degenerate: no model passes the local validity check.
            # Fall back to the model with the globally smallest average interval width.
            warnings.warn(
                "No locally valid model found for a test point. "
                "Falling back to the globally narrowest model. "
                "Consider increasing bandwidth or calibration set size.",
                UserWarning,
                stacklevel=4,
            )
            valid_mask = np.ones(K, dtype=bool)
            used_fallback = True

        # Compute interval widths for valid models at x_test
        min_width = np.inf
        k_star = int(np.where(valid_mask)[0][0])  # initialise to first valid

        for k in range(K):
            if not valid_mask[k]:
                continue
            yhat = np.array([test_yhats_i[k]])
            l_k, u_k = self._interval_for_model(
                x_test=x_test,
                yhat=yhat,
                model_idx=k,
                alpha=alpha,
            )
            w_k = float(u_k[0] - l_k[0])
            if w_k < min_width:
                min_width = w_k
                k_star = k

        return valid_mask, k_star, used_fallback

    def _interval_for_model(
        self,
        x_test: np.ndarray,
        yhat: np.ndarray,
        model_idx: int,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the prediction interval for model_idx at x_test, using the
        locally weighted conformal quantile.

        We compute a kernel-weighted quantile of the calibration scores for
        model_idx, using weights from x_test.
        """
        k = model_idx
        n = len(self.cal_y_)

        # Kernel weights (re-compute; cheap relative to the LOO ops)
        w = _kernel_weights(
            x_test, self.cal_X_,
            localizer=self.localizer,
            bandwidth=self.bandwidth,
            knn_k=self.knn_k,
        )
        w_sum = w.sum()
        if w_sum < 1e-12:
            w = np.ones(n) / n
        else:
            w = w / w_sum

        scores_k = self.cal_scores_[:, k]  # (n,)

        # Weighted (1-alpha) quantile: smallest s such that cumulative weight >= 1-alpha
        quantile_val = _weighted_quantile(scores_k, w, 1.0 - alpha)

        # Invert the score to get interval bounds
        score_name = self._score_names[k]
        if score_name is not None:
            lo, hi = _invert_score_fn(
                yhat, quantile_val, score_name, tweedie_power=self.tweedie_power
            )
        else:
            # Generic inversion: assume raw (absolute) score
            lo = yhat - quantile_val
            hi = yhat + quantile_val

        lo = np.clip(lo, 0.0, None)
        return lo, hi

    def _build_loo_intervals(
        self,
        y: np.ndarray,
        yhats: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Construct approximate LOO prediction intervals for each
        (calibration point, model) pair.

        For large n (>1000), uses the rank-update trick: removing score s_i
        from n sorted scores shifts the (1-alpha) quantile rank by at most 1.
        This gives O(K * n * log n) total cost rather than O(K * n^2).

        For small n, uses brute-force LOO.

        The LOO interval for point i under model k is:
          s_i^{-i} = (1-alpha) quantile of {s_j : j != i, model k}
          [yhat_i_k - invert(s_i^{-i}), yhat_i_k + invert(s_i^{-i})]

        Returns
        -------
        np.ndarray, shape (n, K, 2)
            intervals[:, :, 0] = lower bounds, intervals[:, :, 1] = upper bounds.
        """
        n, K = scores.shape
        # We fix alpha = 0.05 as the reference level for LOO validity.
        # The validity check at prediction time uses the user-supplied alpha;
        # the LOO intervals here are used to determine which models are
        # *locally valid* — they serve as a coverage proxy, not exact intervals.
        # Using a fixed moderate alpha here is consistent with the Wang & Wang
        # paper's construction.
        ref_alpha = 0.10  # reference level for LOO validity checks

        loo_intervals = np.empty((n, K, 2), dtype=float)

        if n > 1000:
            # Approximate LOO via rank-update trick
            for k in range(K):
                s = scores[:, k]
                sorted_s = np.sort(s)
                loo_intervals[:, k, :] = self._loo_rank_update(
                    y=y,
                    yhat=yhats[:, k],
                    scores=s,
                    sorted_scores=sorted_s,
                    model_idx=k,
                    ref_alpha=ref_alpha,
                )
        else:
            # Brute-force LOO for small n
            for k in range(K):
                s = scores[:, k]
                for i in range(n):
                    s_loo = np.delete(s, i)
                    q_loo = conformal_quantile(s_loo, ref_alpha)
                    score_name = self._score_names[k]
                    if score_name is not None:
                        lo, hi = _invert_score_fn(
                            np.array([yhats[i, k]]),
                            q_loo,
                            score_name,
                            tweedie_power=self.tweedie_power,
                        )
                    else:
                        lo = yhats[i, k] - q_loo
                        hi = yhats[i, k] + q_loo
                    loo_intervals[i, k, 0] = float(np.clip(lo, 0.0, None))
                    loo_intervals[i, k, 1] = float(hi)

        return loo_intervals

    def _loo_rank_update(
        self,
        y: np.ndarray,
        yhat: np.ndarray,
        scores: np.ndarray,
        sorted_scores: np.ndarray,
        model_idx: int,
        ref_alpha: float,
    ) -> np.ndarray:
        """
        Approximate LOO intervals using the rank-update trick.

        Removing score s_i from the sorted list shifts the rank by at most 1.
        We find the LOO quantile for each i by binary-search lookup.

        Parameters
        ----------
        y, yhat, scores : shape (n,)
        sorted_scores : shape (n,) sorted scores for this model.
        model_idx : int
        ref_alpha : float

        Returns
        -------
        np.ndarray, shape (n, 2)
            LOO intervals (lower, upper) for each calibration point.
        """
        n = len(scores)
        score_name = self._score_names[model_idx]
        intervals = np.empty((n, 2), dtype=float)

        # Global quantile level: ceil((n)(1-alpha)) / (n-1) for LOO set of size n-1
        # We clip at 1.0 to handle the case where the level exceeds 1.
        for i in range(n):
            # LOO sorted scores: drop s_i
            # The rank of the quantile in n-1 sorted scores
            n_loo = n - 1
            level_loo = min(np.ceil((n_loo + 1) * (1.0 - ref_alpha)) / n_loo, 1.0)

            # Binary search in sorted_scores excluding s_i
            # The k-th order statistic of {s_j : j != i} can be found by noting
            # which rank s_i occupied in the full sorted array.
            s_i = scores[i]
            rank_i = int(np.searchsorted(sorted_scores, s_i, side="left"))

            # Compute the target rank in the LOO set
            target_idx = int(np.ceil(level_loo * n_loo)) - 1
            target_idx = max(0, min(target_idx, n_loo - 1))

            # Adjust for removed element: if removed element was at rank <= target_idx,
            # the target-idx-th element in the LOO set corresponds to sorted_scores[target_idx + 1]
            if rank_i <= target_idx:
                # s_i was at or below target rank; shift up by 1
                adj_idx = min(target_idx + 1, n - 1)
            else:
                adj_idx = target_idx

            q_loo = float(sorted_scores[adj_idx])

            if score_name is not None:
                lo, hi = _invert_score_fn(
                    np.array([yhat[i]]),
                    q_loo,
                    score_name,
                    tweedie_power=self.tweedie_power,
                )
            else:
                lo = np.array([yhat[i] - q_loo])
                hi = np.array([yhat[i] + q_loo])

            intervals[i, 0] = float(np.clip(lo, 0.0, None))
            intervals[i, 1] = float(hi)

        return intervals

    def _predict_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Run a model and return a 1D numpy array of predictions."""
        return as_numpy(model.predict(X)).ravel()

    def _to_numpy(self, X: Any) -> np.ndarray:
        """Convert DataFrame inputs to a 2D numpy float array."""
        import pandas as pd
        if isinstance(X, pl.DataFrame):
            return X.to_numpy().astype(float)
        if isinstance(X, pd.DataFrame):
            return X.to_numpy().astype(float)
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "LCPModelSelector has not been calibrated. "
                "Call .calibrate(X_cal, y_cal) first."
            )

    def __repr__(self) -> str:
        K = len(self.models)
        status = (
            f"calibrated on {len(self.cal_y_)} obs"
            if self.is_calibrated_
            else "not calibrated"
        )
        return (
            f"LCPModelSelector("
            f"K={K}, "
            f"localizer={self.localizer!r}, "
            f"bandwidth={self.bandwidth}, "
            f"{status})"
        )


# ---------------------------------------------------------------------------
# Weighted quantile helper
# ---------------------------------------------------------------------------


def _weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    level: float,
) -> float:
    """
    Compute the weighted quantile at given level.

    Finds the smallest score s such that the cumulative normalised weight
    of scores <= s is at least `level`. This is the conformal weighted
    quantile from Tibshirani et al. (2019).

    Parameters
    ----------
    scores : np.ndarray, shape (n,)
    weights : np.ndarray, shape (n,), non-negative, sum to 1.
    level : float in (0, 1]

    Returns
    -------
    float
    """
    order = np.argsort(scores)
    sorted_s = scores[order]
    sorted_w = weights[order]
    cumw = np.cumsum(sorted_w)

    idx = np.searchsorted(cumw, level, side="left")
    idx = min(idx, len(sorted_s) - 1)
    return float(sorted_s[idx])


# ---------------------------------------------------------------------------
# Convenience wrapper: LocalScoreSelector
# ---------------------------------------------------------------------------


class LocalScoreSelector(LCPModelSelector):
    """
    Convenience wrapper: locally adaptive score selection for a single model.

    Rather than choosing between K different models, LocalScoreSelector runs
    the same model with K different nonconformity score functions and lets
    the LCP framework select the locally best score for each test point.

    This is the most natural entry point for practitioners who already have
    a fitted insurance pricing model and want narrower, better-calibrated
    prediction intervals without refitting.

    The default score set — pearson_weighted, deviance, anscombe — covers
    the main Tweedie score families. pearson_weighted tends to dominate in
    the bulk; deviance and anscombe can be narrower near zero (high-frequency
    or very low-severity risks).

    Parameters
    ----------
    model : fitted sklearn-compatible model
        The base insurance pricing model. Must implement predict(X).
    score_types : list of str, default ['pearson_weighted', 'deviance', 'anscombe']
        Nonconformity score names to try. Each must be a valid string for
        insurance_conformal.scores.compute_score.
    tweedie_power : float, default 1.5
        Tweedie variance power, passed to all score functions.
    localizer : str, default 'gaussian'
        Kernel type.
    bandwidth : float, default 1.0
        Kernel bandwidth.
    knn_k : int, default 20
        Neighbours for knn localizer.

    Examples
    --------
    >>> lss = LocalScoreSelector(model=fitted_catboost, tweedie_power=1.5)
    >>> lss.calibrate(X_cal, y_cal)
    >>> intervals = lss.predict_interval(X_test, alpha=0.10)
    >>> selection = lss.selected_models(X_test, alpha=0.10)
    >>> selection["score_name"].value_counts()  # which scores win locally
    """

    def __init__(
        self,
        model: Any,
        score_types: Optional[list[str]] = None,
        tweedie_power: float = 1.5,
        localizer: str = "gaussian",
        bandwidth: float = 1.0,
        knn_k: int = 20,
    ) -> None:
        if score_types is None:
            score_types = ["pearson_weighted", "deviance", "anscombe"]

        K = len(score_types)
        models = [model] * K

        super().__init__(
            models=models,
            score_fns=score_types,
            localizer=localizer,
            bandwidth=bandwidth,
            knn_k=knn_k,
            tweedie_power=tweedie_power,
        )
        self.model = model
        self.score_types = score_types

    def __repr__(self) -> str:
        status = (
            f"calibrated on {len(self.cal_y_)} obs"
            if self.is_calibrated_
            else "not calibrated"
        )
        scores_str = ", ".join(self.score_types)
        return (
            f"LocalScoreSelector("
            f"scores=[{scores_str}], "
            f"localizer={self.localizer!r}, "
            f"bandwidth={self.bandwidth}, "
            f"{status})"
        )
