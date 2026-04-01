"""
ShapeAdaptiveCP: MOPI (Minimax Optimization for Predictive Inference) for
conditional coverage conformal prediction.

Standard conformal prediction (split conformal, CQR) gives marginal coverage
guarantees — correct on average, but systematically wrong in subgroups. For UK
pricing this is a direct regulatory problem: FCA requires fair treatment across
age, gender, and vehicle group. A predictor that underpays young drivers by 8pp
on average is not acceptable even if total coverage is 90%.

MOPI (Bao et al., arXiv:2603.23374, March 2026) fixes this via minimax
optimisation. The core idea: instead of choosing thresholds to minimise
marginal miscoverage, solve

    min_{h} max_{f in F} E[ f(Z)(1{Y not in C(X;h)} - alpha) - f(Z)^2 ]

The inner maximisation over weight functions f finds the worst-case subgroup.
The outer minimisation over thresholds h forces per-group coverage to be as
uniform as possible.

For finite categorical Z (groups), the inner max has a closed-form solution
(Lemma 3.1, Bao et al.):

    max_{f in F} = sum_k (delta_k)^2 / (4 * n_k)

where delta_k = sum_{i in group k} (1{y_i not in C(x_i;h)} - alpha). This is
exactly the mean-squared conditional coverage error (MSCE), the right quantity
to minimise for fair prediction.

KEY INSURANCE FEATURE — Masked Z:
You calibrate with Z = age band or rating cell. At deployment, Z is masked
(GDPR, FCA indirect discrimination rules). The prediction intervals are
functions of X only. MOPI achieves this naturally: Z enters only the calibration
objective, not the prediction function. Standard conditional calibration (CC,
Gibbs et al. 2025) cannot do this because it requires Z at prediction time.

Algorithm (group mode):
1. Compute nonconformity scores s_i = |y_i - mu_i| / sigma_i
2. Initialise h_k = (1-alpha) quantile of {s_i : Z_i = k}
3. Iterate:
   a. Smoothed indicator: phi_i = sigmoid((s_i - h_{Z_i}) / smooth_temp)
   b. Group miscoverage: delta_k = mean(phi_i for Z_i=k) - alpha
   c. Loss = sum_k delta_k^2 (closed-form inner max, rescaled)
   d. Gradient via autograd of smoothed sigmoid
   e. Adam update on h_k
4. Store thresholds_[k] for each group k

Reference:
    Bao, Zhang, Wang, Ren & Zou (2026). "Minimax Optimization for Predictive
    Inference." arXiv:2603.23374.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy


# ---------------------------------------------------------------------------
# Adam optimiser — pure numpy, no external dependencies
# ---------------------------------------------------------------------------


class _Adam:
    """
    Minimal Adam optimiser for a flat parameter vector.

    Uses the standard Adam update (Kingma & Ba, 2015) with numerical stability
    epsilon. No weight decay, no learning rate schedule.
    """

    def __init__(self, params: np.ndarray, lr: float = 0.01) -> None:
        self.params = params.copy().astype(float)
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)
        self.t = 0

    def step(self, grads: np.ndarray) -> np.ndarray:
        """Apply one Adam update and return new parameters."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return self.params.copy()


# ---------------------------------------------------------------------------
# Core smoothing function
# ---------------------------------------------------------------------------


def _sigmoid(u: np.ndarray, temp: float) -> np.ndarray:
    """
    Numerically stable sigmoid: 1 / (1 + exp(-u / temp)).

    This is the smooth surrogate for the indicator 1{s > h}. When temp is
    small (0.1, default), this is a tight approximation. When temp is large,
    the approximation is smoother but introduces more bias.

    Parameters
    ----------
    u : np.ndarray
        Input values (s_i - h_k per observation).
    temp : float
        Smoothing temperature. Paper default is 0.1.
    """
    x = u / temp
    # Clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_grad(u: np.ndarray, temp: float) -> np.ndarray:
    """
    Gradient of sigmoid w.r.t. u: sigma(u) * (1 - sigma(u)) / temp.
    """
    sig = _sigmoid(u, temp)
    return sig * (1.0 - sig) / temp


# ---------------------------------------------------------------------------
# ShapeAdaptiveCP
# ---------------------------------------------------------------------------


class ShapeAdaptiveCP:
    """
    Shape-adaptive conformal predictor via MOPI minimax optimisation.

    Produces prediction intervals that achieve near-uniform conditional
    coverage across user-defined subgroups Z. Compared to Mondrian conformal
    (which partitions the calibration set per group), MOPI uses all calibration
    data jointly via the minimax objective — avoiding the sparse-cell problem
    when some groups have few calibration observations.

    Two modes:

    ``mode="group"`` (recommended for insurance):
        Z is a categorical array of group labels (e.g. age band, rating cell).
        The inner maximisation is closed-form: O(K) per iteration where K is
        the number of groups. Uses pure numpy throughout. Suitable for up to
        K ~ 100 groups.

    ``mode="rkhs"`` (continuous Z):
        Z is a continuous array. The inner maximisation uses a kernel matrix
        (Gaussian RBF). O(n^3) setup (Cholesky), O(n^2) per iteration. Use
        when you want coverage conditional on a continuous covariate (e.g. age
        as a continuous variable rather than banded).

    The masked Z feature is the key selling point for UK insurance. Pass Z at
    calibration time (e.g. gender or exact age), then predict using only X
    features. The calibration objective forced coverage to be equitable across
    Z groups; the prediction function does not require Z at deployment time.
    For group mode this means you need a ``group_fn`` that assigns test points
    to groups based on X alone.

    Parameters
    ----------
    score_fn : str or callable, default "absolute"
        How to compute nonconformity scores.
        - ``"absolute"``: s_i = |y_i - mu_i|
        - ``"standardised"``: s_i = |y_i - mu_i| / sigma_i (requires
          sigma estimates)
        - callable: ``score_fn(y, mu, sigma) -> np.ndarray`` (shape n,)
    alpha : float, default 0.1
        Miscoverage rate. ``alpha=0.1`` targets 90% conditional coverage.
    mode : {"group", "rkhs"}, default "group"
        Conditioning regime. See class docstring.
    n_iter : int, default 200
        Number of Adam gradient steps for the outer minimisation.
    lr : float, default 0.01
        Adam learning rate for threshold optimisation.
    smooth_temp : float, default 0.1
        Sigmoid smoothing temperature r. Smaller = tighter approximation,
        but harder to optimise. Paper default is 0.1; stable in [0.01, 0.5].
    kernel_bandwidth : float, default 1.0
        Bandwidth theta for the Gaussian RBF kernel in RKHS mode.
        K(z, z') = exp(-||z - z'||^2 / (2 * theta^2)).
    gamma : float, default 0.0
        RKHS regularisation parameter. Set to 0 for finite Z (group mode).
        For RKHS mode, use a small positive value (e.g. 1e-3) for stability.
    random_state : int, default 42
        Random seed for reproducibility.

    Attributes
    ----------
    thresholds_ : np.ndarray
        Per-group thresholds after calibration. Shape (K,) for group mode.
        In group mode, ``thresholds_[k]`` is the threshold for group k.
    groups_ : np.ndarray
        Unique group labels, in sorted order. Shape (K,). Group mode only.
    group_index_ : dict
        Maps group label to integer index in ``thresholds_``. Group mode only.
    msce_ : float
        Mean squared conditional coverage error on the calibration set after
        optimisation. Lower is better. Computed on the calibration set so it
        is biased downward — use a held-out test set for honest evaluation.
    coverage_by_group_ : dict
        Per-group empirical coverage on the calibration set. Keys are group
        labels; values are float coverage rates. Diagnostic only.
    n_calibration_ : int
        Number of calibration observations.
    is_calibrated_ : bool
        Whether ``calibrate()`` has been called.

    Examples
    --------
    Group-conditional coverage (UK motor, age-band conditioned)::

        from insurance_conformal import ShapeAdaptiveCP

        # Z_cal = age bands (used at calibration, masked at test time)
        # group_fn assigns test points to bands from X features only
        cp = ShapeAdaptiveCP(score_fn="standardised", alpha=0.1, mode="group")
        cp.calibrate(
            X_cal, y_cal, Z_cal,
            mu_cal=glm.predict(X_cal),
            sigma_cal=np.sqrt(glm.predict(X_cal)),
        )
        # At deployment: no age column in X_test, but group_fn derives band
        lower, upper = cp.predict(
            X_test,
            mu_test=glm.predict(X_test),
            sigma_test=np.sqrt(glm.predict(X_test)),
            group_fn=lambda X: np.digitize(X[:, age_col], age_bins) - 1,
        )

    RKHS mode for continuous Z::

        cp = ShapeAdaptiveCP(
            score_fn="standardised",
            alpha=0.1,
            mode="rkhs",
            kernel_bandwidth=0.5,
            gamma=1e-3,
        )
        cp.calibrate(X_cal, y_cal, Z_cal, mu_cal=mu_cal, sigma_cal=sigma_cal)
        lower, upper = cp.predict(X_test, mu_test=mu_test, sigma_test=sigma_test)

    References
    ----------
    Bao, Zhang, Wang, Ren & Zou (2026). "Minimax Optimization for Predictive
    Inference." arXiv:2603.23374.
    """

    def __init__(
        self,
        score_fn: Union[str, Callable] = "absolute",
        alpha: float = 0.1,
        mode: Literal["group", "rkhs"] = "group",
        n_iter: int = 200,
        lr: float = 0.01,
        smooth_temp: float = 0.1,
        kernel_bandwidth: float = 1.0,
        gamma: float = 0.0,
        random_state: int = 42,
    ) -> None:
        if mode not in ("group", "rkhs"):
            raise ValueError(f"mode must be 'group' or 'rkhs', got {mode!r}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if smooth_temp <= 0:
            raise ValueError(f"smooth_temp must be positive, got {smooth_temp}")

        self.score_fn = score_fn
        self.alpha = alpha
        self.mode = mode
        self.n_iter = n_iter
        self.lr = lr
        self.smooth_temp = smooth_temp
        self.kernel_bandwidth = kernel_bandwidth
        self.gamma = gamma
        self.random_state = random_state

        # Set after calibrate()
        self.thresholds_: Optional[np.ndarray] = None
        self.groups_: Optional[np.ndarray] = None
        self.group_index_: Optional[Dict] = None
        self.msce_: Optional[float] = None
        self.coverage_by_group_: Optional[Dict] = None
        self.n_calibration_: int = 0
        self.is_calibrated_: bool = False

        # RKHS internal state
        self._K_inv: Optional[np.ndarray] = None
        self._Z_cal: Optional[np.ndarray] = None
        self._scores_cal: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def calibrate(
        self,
        X_cal: Any,
        y_cal: Any,
        Z_cal: Any,
        mu_cal: Optional[Any] = None,
        sigma_cal: Optional[Any] = None,
    ) -> "ShapeAdaptiveCP":
        """
        Fit MOPI thresholds on the calibration set.

        Runs the minimax optimisation: outer Adam on per-group thresholds,
        inner max computed in closed form (group mode) or via kernel matrix
        inversion (RKHS mode).

        Parameters
        ----------
        X_cal : array-like of shape (n, p)
            Calibration features. Not used directly in group mode (only Z_cal
            determines group membership), but kept for interface consistency
            and future use in RKHS mode.
        y_cal : array-like of shape (n,)
            Calibration responses.
        Z_cal : array-like of shape (n,) or (n, d)
            Conditioning variable. For group mode, must be 1D categorical
            (integer or string labels). For RKHS mode, can be continuous
            (1D or multi-dimensional).
        mu_cal : array-like of shape (n,), optional
            Point predictions from a pre-fitted model. Required if
            ``score_fn="standardised"`` or ``score_fn="absolute"``.
            If None and score_fn is a string, raises an error.
        sigma_cal : array-like of shape (n,), optional
            Scale estimates (standard deviation predictions) from the model.
            Required if ``score_fn="standardised"``. Ignored for
            ``score_fn="absolute"``.

        Returns
        -------
        self
        """
        y = as_numpy(y_cal).ravel()
        Z = as_numpy(Z_cal)
        n = len(y)

        mu = as_numpy(mu_cal).ravel() if mu_cal is not None else None
        sigma = as_numpy(sigma_cal).ravel() if sigma_cal is not None else None

        if len(Z) != n:
            raise ValueError(
                f"Z_cal length {len(Z)} does not match y_cal length {n}"
            )

        # Compute nonconformity scores
        scores = self._compute_scores(y, mu, sigma)

        if self.mode == "group":
            self._calibrate_group(scores, Z, n)
        else:
            self._calibrate_rkhs(scores, Z, n)

        self.n_calibration_ = n
        self.is_calibrated_ = True
        return self

    def predict(
        self,
        X_test: Any,
        mu_test: Optional[Any] = None,
        sigma_test: Optional[Any] = None,
        group_fn: Optional[Callable] = None,
        Z_test: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce MOPI prediction intervals for new observations.

        Returns (lower, upper) arrays. For group mode, each test point is
        assigned to a group either via ``Z_test`` (direct group label) or
        ``group_fn(X_test)`` (derived group assignment from features, enabling
        masked Z deployment).

        Parameters
        ----------
        X_test : array-like of shape (m, p)
            Test features. Used by ``group_fn`` if provided.
        mu_test : array-like of shape (m,), optional
            Point predictions. Required if ``score_fn`` is a string and
            intervals are centred on predictions (the default).
            If None, intervals are returned as threshold values only (centred
            at zero), which is only meaningful for custom score functions.
        sigma_test : array-like of shape (m,), optional
            Scale estimates. Required if ``score_fn="standardised"``.
            If None and mode is standardised, sigma_test defaults to ones
            (threshold in raw score units).
        group_fn : callable, optional
            Function ``X_test -> group_labels`` that assigns test points to
            groups from features alone. Use this when Z is masked at
            deployment. Mutually exclusive with ``Z_test``.
        Z_test : array-like of shape (m,), optional
            Direct group labels for test points. Mutually exclusive with
            ``group_fn``. If both are None in group mode, raises an error.

        Returns
        -------
        lower : np.ndarray of shape (m,)
            Lower prediction interval bounds.
        upper : np.ndarray of shape (m,)
            Upper prediction interval bounds.

        Notes
        -----
        For RKHS mode, ``Z_test`` is required (RKHS extrapolation uses the
        kernel between test Z and calibration Z). If you want masked-Z
        prediction in RKHS mode, you need a model for Z|X to derive Z_test
        from X_test.
        """
        self._check_calibrated()

        X_np = _to_numpy_2d(X_test)
        m = len(X_np)

        mu = as_numpy(mu_test).ravel() if mu_test is not None else np.zeros(m)
        sigma = as_numpy(sigma_test).ravel() if sigma_test is not None else np.ones(m)

        if self.mode == "group":
            return self._predict_group(X_np, mu, sigma, group_fn, Z_test)
        else:
            return self._predict_rkhs(X_np, mu, sigma, Z_test)

    def msce(
        self,
        y_test: Any,
        lower: Any,
        upper: Any,
        Z_test: Any,
    ) -> float:
        """
        Compute mean squared conditional coverage error on a test set.

        MSCE = E[(P(Y not in C(X) | Z) - alpha)^2]. Estimated by binning on
        groups Z_test and computing per-group empirical miscoverage.

        This is the honest evaluation metric — use a held-out test set, not
        the calibration set. The ``msce_`` attribute is computed on the
        calibration set and is biased downward.

        Parameters
        ----------
        y_test : array-like of shape (n,)
            Observed responses.
        lower : array-like of shape (n,)
            Lower prediction interval bounds.
        upper : array-like of shape (n,)
            Upper prediction interval bounds.
        Z_test : array-like of shape (n,)
            Group labels for the test set.

        Returns
        -------
        float
            Estimated MSCE (lower is better).
        """
        y = as_numpy(y_test).ravel()
        lo = as_numpy(lower).ravel()
        hi = as_numpy(upper).ravel()
        Z = as_numpy(Z_test)

        covered = (y >= lo) & (y <= hi)
        miscovered = ~covered
        return _compute_msce(miscovered.astype(float), Z, self.alpha)

    def coverage_report(
        self,
        y_test: Any,
        lower: Any,
        upper: Any,
        Z_test: Any,
    ) -> pl.DataFrame:
        """
        Per-group coverage report on a test set.

        Parameters
        ----------
        y_test : array-like of shape (n,)
            Observed responses.
        lower : array-like of shape (n,)
            Lower prediction interval bounds.
        upper : array-like of shape (n,)
            Upper prediction interval bounds.
        Z_test : array-like of shape (n,)
            Group labels for the test set.

        Returns
        -------
        pl.DataFrame
            Columns: group, n_obs, coverage, target_coverage,
            miscoverage_deviation (= coverage - target).
        """
        y = as_numpy(y_test).ravel()
        lo = as_numpy(lower).ravel()
        hi = as_numpy(upper).ravel()
        Z = as_numpy(Z_test)

        covered = (y >= lo) & (y <= hi)
        groups = np.unique(Z)
        rows = []
        for g in groups:
            mask = Z == g
            cov = float(covered[mask].mean()) if mask.sum() > 0 else float("nan")
            rows.append(
                {
                    "group": str(g),
                    "n_obs": int(mask.sum()),
                    "coverage": cov,
                    "target_coverage": float(1.0 - self.alpha),
                    "miscoverage_deviation": cov - (1.0 - self.alpha),
                }
            )
        return pl.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Internal: score computation
    # -----------------------------------------------------------------------

    def _compute_scores(
        self,
        y: np.ndarray,
        mu: Optional[np.ndarray],
        sigma: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute nonconformity scores from y, mu, sigma."""
        if callable(self.score_fn):
            return self.score_fn(y, mu, sigma)

        if self.score_fn == "absolute":
            if mu is None:
                raise ValueError(
                    "score_fn='absolute' requires mu_cal (point predictions)."
                )
            return np.abs(y - mu)

        if self.score_fn == "standardised":
            if mu is None:
                raise ValueError(
                    "score_fn='standardised' requires mu_cal (point predictions)."
                )
            if sigma is None:
                raise ValueError(
                    "score_fn='standardised' requires sigma_cal (scale predictions)."
                )
            sigma_safe = np.where(sigma > 0, sigma, 1e-8)
            return np.abs(y - mu) / sigma_safe

        raise ValueError(
            f"score_fn must be 'absolute', 'standardised', or a callable. "
            f"Got {self.score_fn!r}."
        )

    # -----------------------------------------------------------------------
    # Internal: group mode
    # -----------------------------------------------------------------------

    def _calibrate_group(
        self,
        scores: np.ndarray,
        Z: np.ndarray,
        n: int,
    ) -> None:
        """MOPI calibration for finite categorical Z."""
        groups = np.unique(Z)
        K = len(groups)
        group_index = {g: k for k, g in enumerate(groups)}

        # Map Z to integer indices
        z_idx = np.array([group_index[z] for z in Z], dtype=int)

        # Initialise thresholds at (1-alpha) quantile per group — this is
        # Mondrian conformal, and a good warm start for MOPI.
        h_init = np.zeros(K)
        for k in range(K):
            mask = z_idx == k
            if mask.sum() > 0:
                h_init[k] = float(np.quantile(scores[mask], 1.0 - self.alpha))
            else:
                h_init[k] = float(np.quantile(scores, 1.0 - self.alpha))

        # Adam optimisation
        optim = _Adam(h_init, lr=self.lr)

        for _ in range(self.n_iter):
            h = optim.params

            # Smoothed indicators: phi_i = sigmoid((s_i - h_{z_i}) / temp)
            residuals = scores - h[z_idx]  # shape (n,)
            phi = _sigmoid(residuals, self.smooth_temp)  # smooth 1{s > h_k}

            # Per-group miscoverage: delta_k = mean(phi in group k) - alpha
            delta = np.zeros(K)
            n_k = np.zeros(K)
            for k in range(K):
                mask = z_idx == k
                n_k[k] = mask.sum()
                if n_k[k] > 0:
                    delta[k] = phi[mask].mean() - self.alpha

            # Gradient: d_loss/d_h_k where loss = sum_k delta_k^2
            # d_loss/d_h_k = 2 * delta_k * d_delta_k/d_h_k
            # d_delta_k/d_h_k = mean( -d_phi_i/d_h_k for i in group k)
            #                 = mean( -d_phi_i/d_residuals_i * d_residuals_i/d_h_k )
            #                 = mean( -d_sigmoid/d_u * (-1) ) for i in group k
            #                 = mean( d_sigmoid/d_u ) for i in group k
            # but d_sigmoid/d_u = sigma(u)*(1-sigma(u))/temp
            dphi_dresiduals = _sigmoid_grad(residuals, self.smooth_temp)  # shape (n,)

            grads = np.zeros(K)
            for k in range(K):
                mask = z_idx == k
                if n_k[k] > 0:
                    # d_delta_k/d_h_k = +mean(dphi/dresiduals) since d_residuals/d_h_k = -1
                    # => the chain rule gives d_delta_k/d_h_k = -mean(dphi * (-1)/...) ...
                    # Let's be precise:
                    # residual_i = s_i - h_k  => d_residual_i/d_h_k = -1
                    # phi_i = sigmoid(residual_i / temp)
                    # d_phi_i/d_h_k = (d_phi/d_residual) * (-1) = -dphi_dresiduals_i
                    # delta_k = mean(phi_i for i in k) - alpha
                    # d_delta_k/d_h_k = mean(-dphi_dresiduals_i for i in k)
                    d_delta_dh = -dphi_dresiduals[mask].mean()
                    grads[k] = 2.0 * delta[k] * d_delta_dh

            optim.step(grads)

        h_final = optim.params

        # Store results
        self.thresholds_ = h_final
        self.groups_ = groups
        self.group_index_ = group_index

        # Compute calibration-set MSCE and per-group coverage (hard indicators)
        covered_hard = (scores <= h_final[z_idx]).astype(float)
        self.msce_ = _compute_msce(1.0 - covered_hard, Z, self.alpha)

        cov_by_group: Dict = {}
        for k, g in enumerate(groups):
            mask = z_idx == k
            if mask.sum() > 0:
                cov_by_group[g] = float(covered_hard[mask].mean())
        self.coverage_by_group_ = cov_by_group

    def _predict_group(
        self,
        X: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        group_fn: Optional[Callable],
        Z_test: Optional[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction for group mode."""
        if group_fn is not None and Z_test is not None:
            raise ValueError(
                "Provide either group_fn or Z_test, not both."
            )

        if group_fn is not None:
            groups_test = as_numpy(group_fn(X)).ravel()
        elif Z_test is not None:
            groups_test = as_numpy(Z_test).ravel()
        else:
            raise ValueError(
                "For mode='group', must provide either Z_test or group_fn "
                "to assign test points to groups."
            )

        m = len(X)
        thresholds_test = np.zeros(m)
        fallback_threshold = float(np.median(self.thresholds_))

        for i, g in enumerate(groups_test):
            if g in self.group_index_:
                k = self.group_index_[g]
                thresholds_test[i] = self.thresholds_[k]
            else:
                # Unseen group at test time: use median threshold as fallback
                thresholds_test[i] = fallback_threshold

        # Interval: [mu - h*sigma, mu + h*sigma]  (score_fn="standardised")
        # or [mu - h, mu + h]  (score_fn="absolute")
        if self.score_fn == "standardised":
            half_width = thresholds_test * sigma
        elif self.score_fn == "absolute":
            half_width = thresholds_test
        else:
            # Custom score_fn: return threshold as half-width (best effort)
            half_width = thresholds_test * sigma

        lower = mu - half_width
        upper = mu + half_width
        return lower, upper

    # -----------------------------------------------------------------------
    # Internal: RKHS mode
    # -----------------------------------------------------------------------

    def _calibrate_rkhs(
        self,
        scores: np.ndarray,
        Z: np.ndarray,
        n: int,
    ) -> None:
        """
        MOPI calibration for continuous Z via RKHS.

        The inner maximisation (Lemma 3.2) is:
            max_{f in F} = (1/4) phi_n^T K_n (K_n/n + gamma*I)^{-1} phi_n

        where phi_i = (1/n)(1{s_i > h_i} - alpha) and K_n is the Gram matrix
        of Z under the Gaussian kernel.

        The outer minimisation is over h — in RKHS mode, h is a scalar per
        calibration point (not per group). We optimise the smoothed version.
        """
        Z_2d = Z.reshape(n, -1) if Z.ndim == 1 else Z

        # Compute Gram matrix: K[i,j] = exp(-||z_i - z_j||^2 / (2*theta^2))
        K = _rbf_kernel(Z_2d, Z_2d, self.kernel_bandwidth)

        # Cholesky factored solve: (K/n + gamma*I)^{-1}
        reg_matrix = K / n + self.gamma * np.eye(n)
        try:
            L = np.linalg.cholesky(reg_matrix)
        except np.linalg.LinAlgError:
            # Add jitter if not PD
            reg_matrix += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(reg_matrix)

        # Cache for prediction
        self._K = K
        self._L = L
        self._Z_cal = Z_2d
        self._scores_cal = scores

        # Initialise h (scalar per calibration point) = global (1-alpha) quantile
        h_global = float(np.quantile(scores, 1.0 - self.alpha))
        h_init = np.full(n, h_global)

        optim = _Adam(h_init, lr=self.lr)

        for _ in range(self.n_iter):
            h = optim.params

            # Smoothed phi_i = sigmoid((s_i - h_i) / temp)
            residuals = scores - h
            phi_smooth = _sigmoid(residuals, self.smooth_temp)

            # phi_n vector (1/n) * (phi_i - alpha)
            phi_n = (phi_smooth - self.alpha) / n

            # Solve (K/n + gamma*I)^{-1} phi_n via Cholesky
            # reg_matrix * w = phi_n  => w = L^{-T} L^{-1} phi_n
            w = np.linalg.solve(L.T, np.linalg.solve(L, phi_n))

            # RKHS inner max value: (1/4) phi_n^T K w = (1/4) phi_n @ K @ w
            # For gradient, we need d_loss/d_h_i
            # loss = (1/4) phi_n^T K_n (K/n + gamma*I)^{-1} phi_n
            # d_loss/d_h_i = (1/2) (K_n w)_i * d_phi_n_i/d_h_i
            # d_phi_n_i/d_h_i = (1/n) * (-d_sigmoid/d_residuals_i)
            #                 = -(1/n) * dphi_dresiduals_i

            Kw = K @ w  # shape (n,)
            dphi_dresiduals = _sigmoid_grad(residuals, self.smooth_temp)
            d_phi_n_d_h = -(1.0 / n) * dphi_dresiduals

            grads = 0.5 * Kw * d_phi_n_d_h

            optim.step(grads)

        h_final = optim.params
        self.thresholds_ = h_final
        self._h_cal = h_final

        # Calibration-set diagnostics (approximate: treat each point as a group)
        covered_hard = (scores <= h_final).astype(float)
        # MSCE requires groups; for RKHS, use a coarse binning of Z for diagnostics
        if Z.ndim == 1 and np.issubdtype(Z.dtype, np.integer):
            msce_Z = Z
        else:
            # Bin continuous Z into 10 percentile groups for diagnostic MSCE
            z_1d = Z_2d[:, 0] if Z_2d.ndim > 1 else Z_2d.ravel()
            msce_Z = np.digitize(
                z_1d, np.percentile(z_1d, np.linspace(0, 100, 11)[1:-1])
            )
        self.msce_ = _compute_msce(1.0 - covered_hard, msce_Z, self.alpha)
        self.coverage_by_group_ = None  # not applicable for RKHS

    def _predict_rkhs(
        self,
        X: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        Z_test: Optional[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction for RKHS mode."""
        if Z_test is None:
            raise ValueError(
                "For mode='rkhs', Z_test is required to compute kernel "
                "distances to calibration points."
            )

        Z_test_np = as_numpy(Z_test)
        m = len(Z_test_np)
        Z_test_2d = Z_test_np.reshape(m, -1) if Z_test_np.ndim == 1 else Z_test_np

        # Kernel between test Z and calibration Z: shape (m, n)
        K_test = _rbf_kernel(Z_test_2d, self._Z_cal, self.kernel_bandwidth)

        # Interpolate calibration thresholds to test points via kernel smoothing
        # h_test_i = K_test[i, :] @ (K_cal + lambda*I)^{-1} @ h_cal
        # We already have L from calibration: solve (K/n + gamma*I)^{-1} h_cal
        w_h = np.linalg.solve(
            self._L.T, np.linalg.solve(self._L, self._h_cal)
        )
        h_test = K_test @ w_h  # shape (m,)

        # Ensure thresholds are non-negative
        h_test = np.maximum(h_test, 0.0)

        if self.score_fn == "standardised":
            half_width = h_test * sigma
        elif self.score_fn == "absolute":
            half_width = h_test
        else:
            half_width = h_test * sigma

        lower = mu - half_width
        upper = mu + half_width
        return lower, upper

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "ShapeAdaptiveCP has not been calibrated. "
                "Call .calibrate(X_cal, y_cal, Z_cal) first."
            )

    def __repr__(self) -> str:
        status = (
            f"calibrated on {self.n_calibration_} obs, "
            f"msce={self.msce_:.4f}"
            if self.is_calibrated_
            else "not calibrated"
        )
        return (
            f"ShapeAdaptiveCP("
            f"mode={self.mode!r}, "
            f"alpha={self.alpha}, "
            f"score_fn={self.score_fn!r}, "
            f"{status})"
        )


# ---------------------------------------------------------------------------
# Helpers (module-level, used by ShapeAdaptiveCP and potentially by tests)
# ---------------------------------------------------------------------------


def _compute_msce(
    miscoverage: np.ndarray,
    Z: np.ndarray,
    alpha: float,
) -> float:
    """
    Estimate MSCE = sum_k n_k/n * (delta_k)^2 where delta_k is the mean
    group miscoverage deviation from alpha.

    Parameters
    ----------
    miscoverage : np.ndarray of shape (n,)
        Binary indicators: 1 if observation is NOT covered.
    Z : np.ndarray of shape (n,)
        Group labels.
    alpha : float
        Target miscoverage rate.

    Returns
    -------
    float
        Estimated MSCE.
    """
    n = len(miscoverage)
    groups = np.unique(Z)
    msce = 0.0
    for g in groups:
        mask = Z == g
        n_k = mask.sum()
        if n_k > 0:
            delta_k = float(miscoverage[mask].mean()) - alpha
            msce += (n_k / n) * delta_k ** 2
    return float(msce)


def _rbf_kernel(
    Z1: np.ndarray,
    Z2: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    Gaussian RBF kernel: K(z1, z2) = exp(-||z1 - z2||^2 / (2 * bw^2)).

    Parameters
    ----------
    Z1 : np.ndarray of shape (n1, d)
    Z2 : np.ndarray of shape (n2, d)
    bandwidth : float

    Returns
    -------
    np.ndarray of shape (n1, n2)
    """
    # Pairwise squared distances via |z1|^2 + |z2|^2 - 2 z1@z2^T
    sq1 = np.sum(Z1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    sq2 = np.sum(Z2 ** 2, axis=1, keepdims=True)  # (n2, 1)
    sq_dist = sq1 + sq2.T - 2.0 * Z1 @ Z2.T
    sq_dist = np.maximum(sq_dist, 0.0)  # numerical safety
    return np.exp(-sq_dist / (2.0 * bandwidth ** 2))


def _to_numpy_2d(X: Any) -> np.ndarray:
    """Convert to 2D numpy array."""
    import pandas as pd

    if isinstance(X, pl.DataFrame):
        return X.to_numpy()
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr
