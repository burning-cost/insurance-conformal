"""
KMMConformalPredictor: covariate-shift-robust conformal prediction via Kernel
Mean Matching.

Standard split conformal prediction assumes exchangeability between calibration
and test data. In insurance, this fails routinely: portfolio transfers bring
acquired books with different feature distributions, post-COVID cohorts look
nothing like 2019 calibration data, and broker books skew differently from
direct channel books.

Tibshirani et al. (2019) fix this with importance-weighted conformal: weight
calibration points by w(x) = p_target(x) / p_source(x). The problem is that
direct density ratio estimation is unstable when support overlap is limited —
exactly the case in real portfolio transfers. Near-zero denominator densities
produce exploding weights and degraded coverage.

KMM (Gretton et al., 2009) sidesteps this entirely. Instead of estimating the
density ratio, it finds weights beta_i on calibration samples that minimise the
Maximum Mean Discrepancy (MMD) between the reweighted calibration distribution
and the target distribution in an RKHS. The result is stable weights without
the instability of ratio estimation.

The selective extension (SKMM) identifies calibration points far from the target
support and excludes them from the QP. Test points with low kernel similarity to
the target receive NaN intervals — the correct actuarial response when a policy
is genuinely outside the model's valid range.

Paper: Laghuvarapu, Deb & Sun (2026). "KMM-CP: Practical Conformal Prediction
under Covariate Shift via Selective Kernel Mean Matching." arXiv:2603.26415.

Foundation: Tibshirani et al. (2019). "Conformal Prediction Under Covariate
Shift." NeurIPS 32. arXiv:1904.06019.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import polars as pl

from insurance_conformal.scores import NonconformityScore, compute_score, invert_score
from insurance_conformal.utils import as_numpy, apply_exposure, extract_tweedie_power


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _median_bandwidth(X_cal: np.ndarray, X_target: np.ndarray) -> float:
    """
    Compute RBF bandwidth via the median heuristic.

    gamma = 1 / (2 * median_sq_dist^2) where median_sq_dist is the median
    pairwise squared Euclidean distance between X_cal and X_target. This is
    the standard heuristic for kernel methods — it places the kernel in the
    range where it is neither near-zero (too narrow) nor near-constant (too
    wide) across the data.

    Parameters
    ----------
    X_cal : np.ndarray, shape (n, p)
    X_target : np.ndarray, shape (m, p)

    Returns
    -------
    float
        Positive gamma value.
    """
    from sklearn.metrics import pairwise_distances

    # Subsample for speed when data is large — median is robust to subsampling
    n_sub = min(500, len(X_cal))
    m_sub = min(500, len(X_target))
    rng = np.random.default_rng(0)
    idx_c = rng.choice(len(X_cal), n_sub, replace=False) if len(X_cal) > n_sub else np.arange(len(X_cal))
    idx_t = rng.choice(len(X_target), m_sub, replace=False) if len(X_target) > m_sub else np.arange(len(X_target))

    D2 = pairwise_distances(X_cal[idx_c], X_target[idx_t], metric="sqeuclidean")
    med = np.median(D2)
    if med < 1e-12:
        return 1.0  # fallback: unit gamma
    return float(1.0 / (2.0 * med))


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between two sets of points.

    k(x, x') = exp(-gamma * ||x - x'||^2)

    Uses sklearn pairwise_distances for numerical stability (avoids the
    ||x||^2 - 2x.x' + ||x'||^2 expansion which can produce negative values
    from floating point errors).

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
    X2 : np.ndarray, shape (n2, p)
    gamma : float

    Returns
    -------
    np.ndarray, shape (n1, n2)
    """
    from sklearn.metrics import pairwise_distances

    D2 = pairwise_distances(X1, X2, metric="sqeuclidean")
    return np.exp(-gamma * D2)


def _solve_kmm_qp(
    K_cc: np.ndarray,
    kappa: np.ndarray,
    B: float,
    epsilon: float,
) -> np.ndarray:
    """
    Solve the KMM quadratic program.

    min_beta  (1/2) beta^T K_cc beta - kappa^T beta
    s.t.      0 <= beta_i <= B  for all i
              |sum(beta) / n - 1| <= epsilon

    This finds weights beta such that the reweighted calibration distribution
    has minimal MMD to the target distribution.

    Parameters
    ----------
    K_cc : np.ndarray, shape (n, n)
        Calibration-calibration kernel matrix (regularised with 1e-8*I).
    kappa : np.ndarray, shape (n,)
        Calibration-target cross-kernel means: kappa_i = (n/m) sum_j k(x_i, x'_j).
    B : float
        Upper bound on each weight.
    epsilon : float
        Tolerance on the normalisation constraint.

    Returns
    -------
    np.ndarray, shape (n,)
        Clipped weight vector.
    """
    from scipy.optimize import minimize

    n = len(kappa)

    def objective(beta: np.ndarray) -> float:
        return 0.5 * float(beta @ K_cc @ beta) - float(kappa @ beta)

    def grad(beta: np.ndarray) -> np.ndarray:
        return K_cc @ beta - kappa

    constraints = [
        {
            "type": "ineq",
            "fun": lambda b: (1.0 + epsilon) * n - b.sum(),
            "jac": lambda b: -np.ones(n),
        },
        {
            "type": "ineq",
            "fun": lambda b: b.sum() - (1.0 - epsilon) * n,
            "jac": lambda b: np.ones(n),
        },
    ]
    bounds = [(0.0, B)] * n
    beta0 = np.ones(n)  # warm start: uniform weights satisfy the normalisation constraint

    result = minimize(
        objective,
        beta0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        warnings.warn(
            f"KMM QP did not converge: {result.message}. "
            "Using last iterate. Check weight_bound and epsilon settings.",
            UserWarning,
            stacklevel=3,
        )
    return np.clip(result.x, 0.0, B)


def _kernel_smooth_test_weights(
    X_test: np.ndarray,
    X_cal: np.ndarray,
    beta: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Estimate test-point weights via Nadaraya-Watson kernel smoothing.

    w(x_test) = sum_i k(x_test, x_i) * beta_i / sum_i k(x_test, x_i)

    This is the most principled approach for estimating w(x_test) in the
    Tibshirani weighted quantile. It preserves locality — test points near
    high-weight calibration points get higher weight estimates — and uses
    the RBF kernel already computed during calibration.

    Parameters
    ----------
    X_test : np.ndarray, shape (n_test, p)
    X_cal : np.ndarray, shape (n_cal, p)
    beta : np.ndarray, shape (n_cal,)
    gamma : float

    Returns
    -------
    np.ndarray, shape (n_test,)
    """
    # K_tc shape: (n_test, n_cal)
    K_tc = _rbf_kernel(X_test, X_cal, gamma)
    denom = K_tc.sum(axis=1)  # shape (n_test,)
    # Avoid division by zero — fall back to mean(beta) for very distant points
    safe_denom = np.where(denom < 1e-12, 1.0, denom)
    numer = K_tc @ beta  # shape (n_test,)
    fallback = float(np.mean(beta))
    w = np.where(denom < 1e-12, fallback, numer / safe_denom)
    return w


def _knn_test_weights(
    X_test: np.ndarray,
    X_cal: np.ndarray,
    beta: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Estimate test-point weights via k-nearest-neighbour averaging.

    For each test point, find the k nearest calibration points by Euclidean
    distance and average their KMM weights.

    Parameters
    ----------
    X_test : np.ndarray, shape (n_test, p)
    X_cal : np.ndarray, shape (n_cal, p)
    beta : np.ndarray, shape (n_cal,)
    k : int

    Returns
    -------
    np.ndarray, shape (n_test,)
    """
    from sklearn.metrics import pairwise_distances

    D = pairwise_distances(X_test, X_cal, metric="euclidean")
    k_eff = min(k, len(beta))
    idx = np.argpartition(D, k_eff, axis=1)[:, :k_eff]
    return np.mean(beta[idx], axis=1)


def _weighted_conformal_quantile(
    scores: np.ndarray,
    beta: np.ndarray,
    beta_test: float,
    alpha: float,
) -> float:
    """
    Compute the Tibshirani (2019) weighted conformal quantile.

    q = inf{ r : sum_{i: R_i <= r} beta_i / (sum_j beta_j + beta_test) >= 1-alpha }

    The mass beta_test / (sum_j beta_j + beta_test) is assigned to +inf, so
    if the cumulative mass never reaches 1-alpha, we return inf (valid but
    uninformative interval).

    Parameters
    ----------
    scores : np.ndarray, shape (n,)
        Non-conformity scores.
    beta : np.ndarray, shape (n,)
        KMM weights (calibration points).
    beta_test : float
        Estimated weight for the test point.
    alpha : float
        Miscoverage rate.

    Returns
    -------
    float
        Weighted quantile. May be inf.
    """
    total = float(beta.sum()) + beta_test
    if total < 1e-12:
        return float(np.max(scores))

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_beta = beta[order]
    cummass = np.cumsum(sorted_beta) / total

    idx = np.searchsorted(cummass, 1.0 - alpha)
    if idx >= len(sorted_scores):
        return np.inf
    return float(sorted_scores[idx])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class KMMConformalPredictor:
    """
    Covariate-shift-robust conformal prediction via Kernel Mean Matching.

    Wraps any fitted sklearn-compatible model and produces prediction intervals
    that remain valid under covariate shift — when the calibration and test
    feature distributions differ. The standard use cases in UK insurance are
    portfolio transfers, post-COVID cohort drift, and channel expansion.

    The algorithm uses KMM (Gretton et al., 2009) to find weights on calibration
    samples that minimise MMD between the reweighted calibration distribution and
    the target distribution. These weights feed directly into the Tibshirani (2019)
    weighted conformal quantile. The key advantage over direct density ratio
    estimation is stability: KMM weights are bounded by construction, whereas
    density ratios can explode when support overlap is limited.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Must implement predict(X). Any model that works with
        InsuranceConformalPredictor works here.
    nonconformity : str, default "pearson_weighted"
        Non-conformity score type. Same options as InsuranceConformalPredictor:
        "raw", "pearson", "pearson_weighted", "deviance", "anscombe".
    distribution : str, default "tweedie"
        Distribution family for deviance and anscombe scores.
    tweedie_power : float or None, default None
        Tweedie variance power. Auto-detected from model if None.
    gamma : float or None, default None
        RBF bandwidth parameter. If None, uses the median heuristic:
        gamma = 1 / (2 * median_sq_dist) where median_sq_dist is the median
        pairwise squared distance between X_cal and X_target.
    weight_bound : float, default 10.0
        Upper bound B on individual KMM weights. Higher B allows stronger
        reweighting but risks instability. B=1 disables upweighting. Values
        5-20 are typical for insurance covariate shift.
    epsilon : float, default 0.05
        Normalisation tolerance: |mean(beta) - 1| <= epsilon. 5% is usually
        appropriate; tighten for near-perfect overlap, loosen for strong shift.
    test_weight_method : str, default "kernel_smooth"
        How to estimate the test-point weight for the Tibshirani quantile:
        - "kernel_smooth": Nadaraya-Watson (principled, recommended)
        - "mean": constant mean(beta) proxy (fast, less accurate)
        - "knn": k-nearest-neighbour average
    knn_k : int, default 10
        Number of neighbours for test_weight_method="knn".
    selective : bool, default False
        Enable selective prediction (SKMM). Calibration points with low kernel
        similarity to the target are excluded from the QP. Test points with low
        target similarity receive NaN intervals — flagged for manual review.
    selective_threshold : float or None, default None
        Threshold on per-point kernel similarity for inclusion in the support set.
        None uses the 10th percentile of calibration similarities.
    max_cal_n : int, default 2000
        Maximum calibration set size for the KMM QP. The QP is O(n^3); for
        n > max_cal_n a subsample is drawn with a warning.

    Attributes
    ----------
    kmm_weights_ : np.ndarray, shape (n_cal,) or (n_support,)
        KMM weights for (support-filtered) calibration points.
    cal_scores_ : np.ndarray
        Non-conformity scores for calibration points in the support set.
    X_cal_ : np.ndarray
        Feature matrix used in calibration (after support filtering and
        subsampling). Retained for test weight estimation.
    X_target_ : np.ndarray
        Target feature matrix. Retained for selective support scoring.
    support_mask_ : np.ndarray of bool or None
        Calibration support mask. None if selective=False.
    support_scores_ : np.ndarray or None
        Per-calibration-point kernel similarity to target. None if selective=False.
    selective_threshold_ : float or None
        Resolved threshold value. None if selective=False.
    n_calibration_ : int
        Total calibration points (before selective filtering).
    n_support_ : int
        Calibration points in the support set (== n_calibration_ if not selective).
    is_calibrated_ : bool
    gamma_ : float
        Effective gamma used (resolved from median heuristic if gamma=None).

    References
    ----------
    Laghuvarapu, Deb & Sun (2026). arXiv:2603.26415.
    Tibshirani et al. (2019). NeurIPS 32. arXiv:1904.06019.
    Gretton et al. (2009). Dataset Shift in Machine Learning. MIT Press.
    """

    def __init__(
        self,
        model: Any,
        nonconformity: str = "pearson_weighted",
        distribution: str = "tweedie",
        tweedie_power: Optional[float] = None,
        gamma: Optional[float] = None,
        weight_bound: float = 10.0,
        epsilon: float = 0.05,
        test_weight_method: str = "kernel_smooth",
        knn_k: int = 10,
        selective: bool = False,
        selective_threshold: Optional[float] = None,
        max_cal_n: int = 2000,
    ) -> None:
        # Validate nonconformity score early
        try:
            NonconformityScore(nonconformity)
        except ValueError:
            valid = [s.value for s in NonconformityScore]
            raise ValueError(
                f"Unknown nonconformity score '{nonconformity}'. Choose from: {valid}"
            )

        if distribution.lower() not in ("poisson", "gamma", "tweedie"):
            raise ValueError(
                f"Unknown distribution '{distribution}'. "
                "Choose from: 'poisson', 'gamma', 'tweedie'."
            )

        valid_methods = ("kernel_smooth", "mean", "knn")
        if test_weight_method not in valid_methods:
            raise ValueError(
                f"Unknown test_weight_method '{test_weight_method}'. "
                f"Choose from: {valid_methods}"
            )

        if weight_bound <= 0:
            raise ValueError(f"weight_bound must be positive, got {weight_bound}")

        if not 0 < epsilon < 1:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")

        self.model = model
        self.nonconformity = nonconformity
        self.distribution = distribution.lower()
        self.gamma = gamma
        self.weight_bound = weight_bound
        self.epsilon = epsilon
        self.test_weight_method = test_weight_method
        self.knn_k = knn_k
        self.selective = selective
        self.selective_threshold = selective_threshold
        self.max_cal_n = max_cal_n

        # Resolve Tweedie power
        if tweedie_power is not None:
            self.tweedie_power = float(tweedie_power)
        else:
            detected = extract_tweedie_power(model)
            if detected is not None:
                self.tweedie_power = detected
            else:
                self.tweedie_power = 1.5
                if nonconformity in ("pearson_weighted", "deviance", "anscombe"):
                    warnings.warn(
                        "Could not auto-detect Tweedie power from model. "
                        "Defaulting to p=1.5. Pass tweedie_power= explicitly if this is wrong.",
                        UserWarning,
                        stacklevel=2,
                    )

        # State set during calibration
        self.kmm_weights_: Optional[np.ndarray] = None
        self.cal_scores_: Optional[np.ndarray] = None
        self.X_cal_: Optional[np.ndarray] = None
        self.X_target_: Optional[np.ndarray] = None
        self.support_mask_: Optional[np.ndarray] = None
        self.support_scores_: Optional[np.ndarray] = None
        self.selective_threshold_: Optional[float] = None
        self.n_calibration_: int = 0
        self.n_support_: int = 0
        self.is_calibrated_: bool = False
        self.gamma_: float = 1.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict(self, X: Any) -> np.ndarray:
        """Call model.predict() and return as 1D numpy array."""
        import polars as pl

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        yhat = self.model.predict(X)
        return as_numpy(yhat).ravel()

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Predictor has not been calibrated. "
                "Call .calibrate(X_cal, y_cal, X_target) first."
            )

    def _subsample_calibration(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        exposure: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Subsample calibration set to max_cal_n if needed."""
        n = len(X_cal)
        if n <= self.max_cal_n:
            return X_cal, y_cal, exposure
        warnings.warn(
            f"Calibration set has {n} points, which exceeds max_cal_n={self.max_cal_n}. "
            f"Subsampling to {self.max_cal_n} points. KMM weights will be approximate. "
            "Increase max_cal_n or pass a smaller calibration set to disable subsampling.",
            UserWarning,
            stacklevel=3,
        )
        rng = np.random.default_rng(0)
        idx = rng.choice(n, self.max_cal_n, replace=False)
        exp_sub = exposure[idx] if exposure is not None else None
        return X_cal[idx], y_cal[idx], exp_sub

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        X_cal: Any,
        y_cal: Any,
        X_target: Any,
        exposure: Optional[Any] = None,
    ) -> "KMMConformalPredictor":
        """
        Fit KMM weights and compute calibration scores.

        Parameters
        ----------
        X_cal : array-like of shape (n_cal, p)
            Calibration features (source distribution). Must be numeric; drop
            categorical columns before passing.
        y_cal : array-like of shape (n_cal,)
            Observed losses on the calibration set.
        X_target : array-like of shape (m, p)
            Feature sample from the target distribution. Does NOT need to be
            the test set — any representative sample of the target book works.
            In practice, pass X_test when available. More target samples means
            better KMM weight estimation.
        exposure : array-like of shape (n_cal,) or None
            Exposure for calibration set (e.g. years of cover). If provided,
            both y_cal and model predictions are divided by exposure before
            computing non-conformity scores.

        Returns
        -------
        self
        """
        X_cal_np = np.asarray(as_numpy(X_cal) if not isinstance(X_cal, np.ndarray) else X_cal, dtype=float)
        y_cal_np = as_numpy(y_cal).astype(float).ravel()
        X_target_np = np.asarray(as_numpy(X_target) if not isinstance(X_target, np.ndarray) else X_target, dtype=float)

        if len(X_cal_np) != len(y_cal_np):
            raise ValueError(
                f"X_cal has {len(X_cal_np)} samples but y_cal has {len(y_cal_np)} — lengths must match."
            )

        exp_np = as_numpy(exposure).astype(float).ravel() if exposure is not None else None

        self.n_calibration_ = len(X_cal_np)

        # Warn on small target sample
        m = len(X_target_np)
        if m < 0.05 * self.n_calibration_:
            warnings.warn(
                f"X_target has only {m} samples compared to {self.n_calibration_} "
                "calibration points. KMM kappa estimates will be noisy. "
                "Provide more target samples for reliable weights.",
                UserWarning,
                stacklevel=2,
            )

        # Subsample calibration if too large
        X_cal_np, y_cal_np, exp_np = self._subsample_calibration(X_cal_np, y_cal_np, exp_np)
        n = len(X_cal_np)

        # Resolve gamma
        if self.gamma is not None:
            gamma = float(self.gamma)
        else:
            gamma = _median_bandwidth(X_cal_np, X_target_np)
        self.gamma_ = gamma

        # Selective extension: compute support scores and filter calibration
        if self.selective:
            K_ct = _rbf_kernel(X_cal_np, X_target_np, gamma)  # (n, m)
            support_scores = K_ct.mean(axis=1)  # (n,)

            if self.selective_threshold is not None:
                tau = float(self.selective_threshold)
            else:
                tau = float(np.percentile(support_scores, 10))

            self.support_mask_ = support_scores >= tau
            self.support_scores_ = support_scores
            self.selective_threshold_ = tau

            X_cal_fit = X_cal_np[self.support_mask_]
            y_cal_fit = y_cal_np[self.support_mask_]
            exp_fit = exp_np[self.support_mask_] if exp_np is not None else None
        else:
            X_cal_fit = X_cal_np
            y_cal_fit = y_cal_np
            exp_fit = exp_np
            self.support_mask_ = None
            self.support_scores_ = None
            self.selective_threshold_ = None

        self.n_support_ = len(X_cal_fit)

        if self.n_support_ == 0:
            raise ValueError(
                "All calibration points were excluded by the selective threshold. "
                "Try a lower selective_threshold or set selective=False."
            )

        # Warn on large n — QP is O(n^3)
        if self.n_support_ > 2000:
            warnings.warn(
                f"KMM QP has {self.n_support_} variables after support filtering. "
                "SLSQP may be slow (O(n^3)). Consider passing a smaller calibration set "
                "or setting max_cal_n.",
                UserWarning,
                stacklevel=2,
            )

        # Build KMM kernel matrices
        K_cc = _rbf_kernel(X_cal_fit, X_cal_fit, gamma)
        K_cc += 1e-8 * np.eye(self.n_support_)  # numerical regularisation

        K_ct_fit = _rbf_kernel(X_cal_fit, X_target_np, gamma)  # (n_support, m)
        kappa = (float(self.n_support_) / float(m)) * K_ct_fit.sum(axis=1)

        # Check condition number of K_cc
        cond = np.linalg.cond(K_cc)
        if cond > 1e10:
            extra_reg = 1e-4 * np.trace(K_cc) / self.n_support_
            K_cc += extra_reg * np.eye(self.n_support_)
            warnings.warn(
                f"K_cc condition number is {cond:.2e}, which may cause instability. "
                "Added extra regularisation. Consider adjusting gamma.",
                UserWarning,
                stacklevel=2,
            )

        # Solve KMM QP
        beta = _solve_kmm_qp(K_cc, kappa, self.weight_bound, self.epsilon)

        # Check effective sample size — warn on strong shift
        ess = float(beta.sum() ** 2) / float((beta**2).sum() + 1e-12)
        if ess / self.n_support_ < 0.2:
            warnings.warn(
                f"KMM effective sample size is very low ({ess:.1f} / {self.n_support_}, "
                f"ratio={ess/self.n_support_:.2f}). Coverage guarantees may be unreliable. "
                "Consider setting selective=True to exclude out-of-support calibration points, "
                "or retraining the model on target data.",
                UserWarning,
                stacklevel=2,
            )

        self.kmm_weights_ = beta

        # Compute non-conformity scores on the support calibration set
        yhat_cal = self._predict(X_cal_fit)
        y_adj, yhat_adj = apply_exposure(y_cal_fit, yhat_cal, exp_fit)

        self.cal_scores_ = compute_score(
            y=y_adj,
            yhat=yhat_adj,
            nonconformity=self.nonconformity,
            distribution=self.distribution,
            tweedie_power=self.tweedie_power,
        )

        # Retain these for predict_interval
        self.X_cal_ = X_cal_fit
        self.X_target_ = X_target_np

        self.is_calibrated_ = True
        return self

    def predict_interval(
        self,
        X_test: Any,
        alpha: float = 0.10,
        return_weights: bool = False,
    ) -> pl.DataFrame:
        """
        Produce covariate-shift-robust prediction intervals.

        For each test point, estimates its KMM weight using the method specified
        by test_weight_method, then computes the Tibshirani (2019) weighted
        conformal quantile and inverts the non-conformity score to get bounds.

        Parameters
        ----------
        X_test : array-like of shape (n_test, p)
            Test features (from the target distribution).
        alpha : float, default 0.10
            Miscoverage rate. alpha=0.10 gives 90% prediction intervals.
        return_weights : bool, default False
            If True, include a w_test column with the per-test-point weight
            estimate. Useful for diagnosing which test points are poorly supported.

        Returns
        -------
        pl.DataFrame
            Columns: lower (Float64), point (Float64), upper (Float64).
            If return_weights=True, also: w_test (Float64).
            If selective=True and a test point is out-of-support:
            lower=NaN, upper=NaN (point is still returned).
        """
        self._check_calibrated()

        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        X_test_np = np.asarray(
            as_numpy(X_test) if not isinstance(X_test, np.ndarray) else X_test,
            dtype=float,
        )
        n_test = len(X_test_np)

        # Point predictions
        yhat = self._predict(X_test_np)

        # Estimate test weights
        beta = self.kmm_weights_
        gamma = self.gamma_

        if self.test_weight_method == "kernel_smooth":
            w_test = _kernel_smooth_test_weights(X_test_np, self.X_cal_, beta, gamma)
        elif self.test_weight_method == "mean":
            w_test = np.full(n_test, float(np.mean(beta)))
        elif self.test_weight_method == "knn":
            w_test = _knn_test_weights(X_test_np, self.X_cal_, beta, self.knn_k)
        else:
            raise ValueError(f"Unknown test_weight_method: {self.test_weight_method}")

        # Compute per-test-point quantiles and invert
        lower = np.empty(n_test)
        upper = np.empty(n_test)

        for i in range(n_test):
            q = _weighted_conformal_quantile(
                scores=self.cal_scores_,
                beta=beta,
                beta_test=float(w_test[i]),
                alpha=alpha,
            )
            lo_i, up_i = invert_score(
                yhat=np.array([yhat[i]]),
                quantile=q,
                nonconformity=self.nonconformity,
                distribution=self.distribution,
                tweedie_power=self.tweedie_power,
                clip_lower=0.0,
            )
            lower[i] = lo_i[0]
            upper[i] = up_i[0]

        # Selective: mark out-of-support test points with NaN
        if self.selective and self.selective_threshold_ is not None:
            K_test_target = _rbf_kernel(X_test_np, self.X_target_, gamma)
            test_support_scores = K_test_target.mean(axis=1)
            out_of_support = test_support_scores < self.selective_threshold_
            lower[out_of_support] = np.nan
            upper[out_of_support] = np.nan
            if return_weights:
                w_test[out_of_support] = 0.0

        result: dict[str, Any] = {
            "lower": lower,
            "point": yhat,
            "upper": upper,
        }
        if return_weights:
            result["w_test"] = w_test

        return pl.DataFrame(result)

    def shift_diagnostics(self) -> pl.DataFrame:
        """
        Summary of the KMM calibration: weight distribution and effective
        sample size.

        The key metric is coverage_shrinkage = effective_n / n_support. This
        quantifies how much the distributional shift has degraded the effective
        calibration sample. A value of 0.4 means the shift has more than halved
        the calibration sample — a concrete signal for the retraining decision.

        Returns
        -------
        pl.DataFrame
            Columns: statistic (String), value (Float64).
            Rows: n_cal, n_support, mean_weight, max_weight, weight_std,
                  min_weight, effective_n, coverage_shrinkage, gamma_used.
        """
        self._check_calibrated()

        beta = self.kmm_weights_
        beta_sum_sq = float((beta**2).sum())
        ess = float(beta.sum() ** 2) / (beta_sum_sq + 1e-12)

        rows = [
            ("n_cal", float(self.n_calibration_)),
            ("n_support", float(self.n_support_)),
            ("mean_weight", float(np.mean(beta))),
            ("max_weight", float(np.max(beta))),
            ("min_weight", float(np.min(beta))),
            ("weight_std", float(np.std(beta))),
            ("effective_n", ess),
            ("coverage_shrinkage", ess / self.n_support_),
            ("gamma_used", float(self.gamma_)),
        ]

        return pl.DataFrame(
            {
                "statistic": [r[0] for r in rows],
                "value": [r[1] for r in rows],
            }
        )

    def __repr__(self) -> str:
        status = (
            f"calibrated on {self.n_support_} support points "
            f"(of {self.n_calibration_} cal)"
            if self.is_calibrated_
            else "not calibrated"
        )
        return (
            f"KMMConformalPredictor("
            f"nonconformity='{self.nonconformity}', "
            f"distribution='{self.distribution}', "
            f"tweedie_power={self.tweedie_power}, "
            f"weight_bound={self.weight_bound}, "
            f"selective={self.selective}, "
            f"{status})"
        )
