"""
RetroAdj: Online Conformal Inference with Retrospective Adjustment.

Source: Jun, J. & Ohn, I. (2025). "Online Conformal Inference with
Retrospective Adjustment for Faster Adaptation to Distribution Shift."
arXiv:2511.04275.

The core insight is that ACI and EnbPI adapt one step at a time. After an
abrupt shift, ACI needs O(1/gamma) steps to fully reprice. At gamma=0.005
that is ~200 steps — about 17 years for a monthly series. RetroAdj fixes
this by retroactively correcting all leave-one-out residuals in the active
window simultaneously at each step, using rank-one updates to an inverse
kernel matrix Q = (K + lambda*I)^{-1}.

For insurance:
- Mid-year claims inflation (UK motor: +30% in 2021-2022): intervals widen
  within 1-3 months post-shift rather than ~50 months for ACI at standard gamma.
- Catastrophe events: faster widening AND faster recovery.
- Regulatory regime changes (Ogden rate): recalibrates within the same year.

Hard constraint: the base model must be kernel ridge regression (KRR) or
another self-stable linear smoother. GLMs and GBMs do not qualify. For the
common case where you have an existing model, use residual-only mode (X=None)
— see the RetroAdj class docstring.

No new dependencies beyond numpy (already a project requirement).

Implementation note on the LOO formula
---------------------------------------
The LOO leverage score for KRR is H[i,i] = diag(K Q)[i] = K[i,:] @ Q[:,i],
where H = K (K + lambda I)^{-1} is the hat matrix. This equals the i-th
diagonal of K Q. The LOO prediction at a test point x_t is:
    f_loo[i](x_t) = f_hat(x_t) - (Q @ k_t)[i] * R_loo[i]
where Q @ k_t is the Q-weighted test kernel vector and R_loo[i] = raw_resid[i]
/ (1 - H[i,i]) is the LOO residual. The spec (arXiv:2511.04275) uses a
slightly different notation involving K_self @ Q @ K_self that corresponds to
a different (and incorrect) formula. The derivation above is verified
empirically against brute-force LOO re-fitting.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------


def _rbf_kernel(X: np.ndarray, X2: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute the RBF (Gaussian) kernel matrix between X and X2.

    Parameters
    ----------
    X:
        Shape (n, p) or (p,) for a single point.
    X2:
        Shape (m, p) or (p,) for a single point.
    bandwidth:
        Kernel bandwidth sigma. kappa(x, x') = exp(-||x-x'||^2 / (2*bw^2)).

    Returns
    -------
    np.ndarray
        Shape (n, m) kernel matrix. All entries in [0, 1].
    """
    X = np.atleast_2d(X).astype(np.float64)
    X2 = np.atleast_2d(X2).astype(np.float64)
    # Squared Euclidean distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a^T b
    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        + np.sum(X2 ** 2, axis=1, keepdims=True).T
        - 2.0 * X @ X2.T
    )
    sq_dists = np.maximum(sq_dists, 0.0)  # clip numerical negatives
    return np.exp(-sq_dists / (2.0 * bandwidth ** 2))


# ---------------------------------------------------------------------------
# Rank-one matrix update and downdate (Lemma 3 & 4 of Jun & Ohn 2025)
# ---------------------------------------------------------------------------


def _rank_one_update(
    Q: np.ndarray,
    k_new: np.ndarray,
    kappa_self_val: float,
    lambda_reg: float,
) -> np.ndarray:
    """Add a new point to Q via Schur complement (Lemma 4).

    Given Q = (K_old + lambda*I)^{-1} of shape (n, n) and k_new = kernel
    vector between the new point and existing window points of shape (n,),
    computes Q_new = (K_new + lambda*I)^{-1} of shape (n+1, n+1).

    Parameters
    ----------
    Q:
        Current inverse, shape (n, n).
    k_new:
        Kernel vector between new point and existing window, shape (n,).
    kappa_self_val:
        kappa(x_new, x_new). Always 1.0 for RBF.
    lambda_reg:
        KRR regularisation parameter.

    Returns
    -------
    np.ndarray
        Updated inverse, shape (n+1, n+1).
    """
    n = len(Q)
    Qk = Q @ k_new  # (n,)
    denom = kappa_self_val + lambda_reg - k_new @ Qk
    if denom <= 0:
        raise _NumericalInstabilityError(
            f"Rank-one update denominator non-positive ({denom:.6g}). "
            "Q has drifted numerically — a periodic reset will recover."
        )
    delta = 1.0 / denom

    Q_new = np.empty((n + 1, n + 1), dtype=np.float64)
    Q_new[:n, :n] = Q + delta * np.outer(Qk, Qk)
    Q_new[:n, n] = -delta * Qk
    Q_new[n, :n] = -delta * Qk
    Q_new[n, n] = delta
    return Q_new


def _rank_one_downdate(Q: np.ndarray) -> np.ndarray:
    """Remove the first (oldest) point from Q via Schur complement (Lemma 3).

    Given Q = (K + lambda*I)^{-1} of shape (n, n), partition as:
        Q = [[q11, q12^T], [q12, Q22]]
    then (K22 + lambda*I)^{-1} = Q22 - outer(q12, q12) / q11.

    Parameters
    ----------
    Q:
        Current inverse, shape (n, n).

    Returns
    -------
    np.ndarray
        Updated inverse after removing index 0, shape (n-1, n-1).
    """
    q11 = Q[0, 0]
    if abs(q11) < 1e-15:
        raise _NumericalInstabilityError(
            f"Downdate pivot q11 near zero ({q11:.6g}). "
            "Q has drifted numerically — a periodic reset will recover."
        )
    q12 = Q[1:, 0]  # (n-1,)
    Q22 = Q[1:, 1:]  # (n-1, n-1)
    return Q22 - np.outer(q12, q12) / q11


# ---------------------------------------------------------------------------
# LOO residuals and predictions (correct formulas for KRR hat matrix)
# ---------------------------------------------------------------------------


def _loo_residuals_and_leverages(
    Q: np.ndarray,
    K_self: np.ndarray,
    Y_window: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute LOO residuals, hat matrix diagonal, and valid-point mask.

    The KRR hat matrix is H = K (K + lambda I)^{-1} = K Q.
    The diagonal H[i,i] = K[i,:] @ Q[:,i] = diag(K Q)[i].
    The LOO residual is R_loo[i] = (Y[i] - f_hat[i]) / (1 - H[i,i]).

    Note: the original paper (arXiv:2511.04275) uses notation involving
    K @ Q @ K which computes diag(K Q K), NOT diag(K Q). The correct
    formula is diag(K Q) — verified against brute-force LOO re-fitting.

    Parameters
    ----------
    Q:
        Inverse kernel matrix Q = (K + lambda I)^{-1}, shape (n, n).
    K_self:
        Kernel matrix of window against itself, shape (n, n).
    Y_window:
        Window responses, shape (n,).
    eps:
        Threshold for |1 - H[i,i]| below which the point is masked
        (near-interpolating, numerically unstable LOO).

    Returns
    -------
    R_loo:
        LOO residuals, shape (n,). np.inf for masked points.
    H_diag:
        Hat matrix diagonal H[i,i] = diag(K Q), shape (n,).
    mask:
        Boolean mask of numerically valid points, shape (n,).
    """
    # Hat matrix diagonal: H[i,i] = K[i,:] @ Q[:,i] = diag(K Q)
    # einsum("ij,ji->i", K_self, Q) computes sum_j K[i,j] * Q[j,i] = (K Q)[i,i]
    H_diag = np.einsum("ij,ji->i", K_self, Q)  # shape (n,)

    # Full fitted values: f_hat = K Q Y = H Y
    QY = Q @ Y_window  # (n,)
    fitted_all = K_self @ QY  # (n,) — same as H @ Y
    raw_resid = Y_window - fitted_all  # (n,)

    denominator = 1.0 - H_diag
    mask = np.abs(denominator) > eps

    R_loo = np.full_like(raw_resid, np.inf)
    R_loo[mask] = raw_resid[mask] / denominator[mask]

    return R_loo, H_diag, mask


def _loo_predictions_at_test(
    Q: np.ndarray,
    K_self: np.ndarray,
    k_test: np.ndarray,
    Y_window: np.ndarray,
    R_loo: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute LOO-adjusted KRR predictions at the test point.

    For each window point i, the prediction at x_test from the model trained
    without point i is:

        f_loo[i](x_test) = f_hat(x_test) - (Q @ k_test)[i] * R_loo[i]

    where f_hat(x_test) = k_test @ Q @ Y is the full-model prediction and
    (Q @ k_test)[i] is the i-th component of the Q-weighted test kernel vector.

    Parameters
    ----------
    Q:
        Inverse kernel matrix, shape (n, n).
    K_self:
        Window self-kernel matrix, shape (n, n). Used to compute f_hat.
    k_test:
        Kernel vector between test point and window, shape (n,).
    Y_window:
        Window responses, shape (n,).
    R_loo:
        LOO residuals from _loo_residuals_and_leverages, shape (n,).
        np.inf for masked points.
    mask:
        Valid-point mask, shape (n,).

    Returns
    -------
    np.ndarray
        LOO predictions at test point, shape (n,). Masked points get
        the full-model prediction as a fallback.
    """
    # Full prediction at test point
    QY = Q @ Y_window  # (n,)
    f_hat_scalar = float(k_test @ QY)  # scalar

    # Q-weighted test kernel vector: Q_k_test[i] = (Q @ k_test)[i]
    Q_k_test = Q @ k_test  # (n,)

    # LOO prediction for each window point i:
    # f_loo[i] = f_hat - Q_k_test[i] * R_loo[i]
    f_loo = np.full(len(Y_window), f_hat_scalar)
    f_loo[mask] = f_hat_scalar - Q_k_test[mask] * R_loo[mask]

    return f_loo


# ---------------------------------------------------------------------------
# Jackknife+ interval (Equation 2 of Jun & Ohn 2025)
# ---------------------------------------------------------------------------


def _jackknife_plus_interval(
    f_loo: np.ndarray,
    R_loo: np.ndarray,
    alpha_t: float,
    mask: np.ndarray,
    symmetric: bool = False,
) -> tuple[float, float]:
    """Compute jackknife+ prediction interval.

    Uses signed residuals by default for asymmetric (insurance-appropriate)
    intervals. Set symmetric=True to use |R_loo| instead.

    Parameters
    ----------
    f_loo:
        LOO predictions at test point, shape (n,).
    R_loo:
        LOO residuals, shape (n,). np.inf for masked points.
    alpha_t:
        Current miscoverage level (ACI-adjusted).
    mask:
        Valid-point mask, shape (n,).
    symmetric:
        If True, use |R_loo| for symmetric intervals.

    Returns
    -------
    (lower, upper):
        Prediction interval bounds.
    """
    n_valid = mask.sum()
    if n_valid == 0:
        return (-np.inf, np.inf)

    f_valid = f_loo[mask]
    r_valid = R_loo[mask]

    if symmetric:
        r_valid = np.abs(r_valid)

    # Jackknife+ quantile levels (Barber et al. 2021, Eq. 1):
    # Upper: quantile_{(1-alpha)(1+1/n)} of {f_loo + R_loo}
    # Lower: quantile_{alpha(1+1/n)} of {f_loo - R_loo}
    # Both capped at [0, 1] to handle extreme alpha_t values during adaptation.
    level_upper = float(np.clip((1.0 - alpha_t) * (1.0 + 1.0 / n_valid), 0.0, 1.0))
    level_lower = float(np.clip(alpha_t * (1.0 + 1.0 / n_valid), 0.0, 1.0))

    # Jackknife+ upper: quantile_{level_upper} of {f_loo[i] + R_loo[i]}
    upper_vals = f_valid + r_valid
    # Jackknife+ lower: quantile_{level_lower} of {f_loo[i] - R_loo[i]}
    lower_vals = f_valid - r_valid

    upper = float(np.quantile(upper_vals, level_upper))
    lower = float(np.quantile(lower_vals, level_lower))

    return lower, upper


# ---------------------------------------------------------------------------
# SFOGD alpha tracker (Algorithm 5 of Jun & Ohn 2025)
# ---------------------------------------------------------------------------


class _SFOGDAlphaTracker:
    """AdaGrad-style (SFOGD) alpha update for slowly-varying shifts.

    Implements Algorithm 5 from Jun & Ohn (2025). The AdaGrad denominator
    sqrt(sum of squared gradients) produces smaller effective step sizes as
    gradients accumulate — preferable when shifts are slow and gradual.

    The update rule is:
        err = alpha_target - (0.0 if covered else 1.0)
        alpha_t -= gamma * err / sqrt(sum_sq_grad)

    Semantic direction (same as ACI):
        - Consistently missing (covered=False): err < 0, alpha_t increases
          (wider intervals needed).
        - Consistently covering (covered=True): err > 0, alpha_t decreases
          (intervals can narrow, we're over-conservative).

    Parameters
    ----------
    alpha_init:
        Starting miscoverage level.
    gamma:
        Base step size (equivalent to ACI gamma).
    """

    def __init__(self, alpha_init: float, gamma: float) -> None:
        self.alpha = float(alpha_init)
        self.gamma = float(gamma)
        self._cumulative_sq_grad = 0.0

    def update(self, covered: bool, alpha_target: float) -> float:
        """Update alpha_t after observing coverage at step t.

        Parameters
        ----------
        covered:
            Whether Y_t was covered by the interval.
        alpha_target:
            Target miscoverage rate (1 - coverage target).

        Returns
        -------
        float
            Updated alpha_t.
        """
        err = alpha_target - (0.0 if covered else 1.0)
        self._cumulative_sq_grad += err ** 2
        if self._cumulative_sq_grad > 0:
            self.alpha = self.alpha - self.gamma * err / math.sqrt(self._cumulative_sq_grad)
        else:
            # First step with no gradient signal — small nudge toward alpha_target
            self.alpha = self.alpha + self.gamma * alpha_target
        self.alpha = float(np.clip(self.alpha, 1e-6, 1.0 - 1e-6))
        return self.alpha


# ---------------------------------------------------------------------------
# Internal sentinel exception for numerical failures
# ---------------------------------------------------------------------------


class _NumericalInstabilityError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# RetroAdj: main class
# ---------------------------------------------------------------------------


class RetroAdj:
    """Online Conformal Inference with Retrospective Adjustment (Jun & Ohn 2025).

    Wraps jackknife+ conformal intervals over a kernel ridge regression (KRR)
    base model with incremental matrix updates (rank-one update and downdate)
    that retroactively correct leave-one-out residuals at each step.

    This gives faster coverage recovery after abrupt distribution shifts than
    ACI or EnbPI, at the cost of requiring a KRR base model. At gamma=0.005,
    ACI needs ~200 steps to reprice after a shift; RetroAdj responds within
    1-3 steps for abrupt shifts because all window LOO residuals are corrected
    simultaneously.

    Parameters
    ----------
    bandwidth:
        RBF kernel bandwidth. Controls KRR smoothness. Features should be
        pre-standardised before passing. Default 1.0.
    lambda_reg:
        KRR regularisation. Larger = smoother, more biased. Default 0.1.
    window_size:
        Sliding window length w. Paper default: 250. Must be >= 5.
    gamma:
        ACI step size for alpha_t update. Default 0.005 (paper default).
    alpha_update:
        'aci' (default) or 'sfogd'. SFOGD uses AdaGrad-style step scaling
        and is better for slowly-varying shifts.
    symmetric:
        If True, use absolute residuals for symmetric intervals. Default False
        (signed residuals, giving asymmetric intervals more appropriate for
        right-skewed insurance claims).
    reset_freq:
        Number of steps between full Q recomputation from scratch (to prevent
        floating-point drift). Default 500. Each reset is O(w^3); for w=250
        that is ~15M flops — negligible in practice.

    Notes
    -----
    **KRR constraint / residual-only mode:**
    When you have a pre-fitted external model (GLM, GBM), pass X=None
    throughout and supply residuals y = actual - model_predict instead of
    raw actuals. With X=None the kernel degenerates to a constant 1 for all
    cross-point evaluations (K = ones-matrix + lambda*I), so KRR reduces to a
    ridge-regularised mean. This retains the jackknife+ interval and improved
    alpha tracking but is an approximation. Alternatively, use
    X = np.arange(len(y)).reshape(-1, 1) to let KRR fit a smooth trend.

    References
    ----------
    Jun, J. & Ohn, I. (2025). arXiv:2511.04275.

    Examples
    --------
    Basic usage with explicit features:

    >>> model = RetroAdj(bandwidth=1.0, lambda_reg=0.1, window_size=100)
    >>> model.fit(y_train, X_train)
    >>> lower, upper = model.predict_interval(y_test, X_test, alpha=0.1)

    Residual-only mode (for GLM/GBM residuals):

    >>> resid_train = y_train - glm.predict(X_train)
    >>> resid_test  = y_test  - glm.predict(X_test)
    >>> model = RetroAdj(window_size=100)
    >>> model.fit(resid_train)
    >>> lower, upper = model.predict_interval(resid_test, alpha=0.1)
    >>> # Reconstruct: shift intervals back by GLM predictions
    >>> lower += glm.predict(X_test)
    >>> upper += glm.predict(X_test)
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        lambda_reg: float = 0.1,
        window_size: int = 250,
        gamma: float = 0.005,
        alpha_update: str = "aci",
        symmetric: bool = False,
        reset_freq: int = 500,
    ) -> None:
        if window_size < 5:
            raise ValueError(f"window_size must be >= 5, got {window_size}")
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth}")
        if lambda_reg <= 0:
            raise ValueError(f"lambda_reg must be > 0, got {lambda_reg}")
        if alpha_update not in ("aci", "sfogd"):
            raise ValueError(f"alpha_update must be 'aci' or 'sfogd', got {alpha_update!r}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")

        self.bandwidth = float(bandwidth)
        self.lambda_reg = float(lambda_reg)
        self.window_size = int(window_size)
        self.gamma = float(gamma)
        self.alpha_update = alpha_update
        self.symmetric = symmetric
        self.reset_freq = int(reset_freq)

        # State set during fit / predict_interval
        self._Q: Optional[np.ndarray] = None
        self._X_window: Optional[np.ndarray] = None  # None when X=None mode
        self._Y_window: Optional[np.ndarray] = None
        self._x_none_mode: bool = False
        self._is_fitted: bool = False
        self._steps_since_reset: int = 0

    # ------------------------------------------------------------------
    # Kernel evaluation helpers (handle X=None degeneracy)
    # ------------------------------------------------------------------

    def _build_K_self(self, n: int) -> np.ndarray:
        """Build kernel matrix of the current window against itself.

        In residual-only mode (X=None), returns an n x n ones-matrix.
        """
        if self._x_none_mode:
            return np.ones((n, n), dtype=np.float64)
        return _rbf_kernel(self._X_window, self._X_window, self.bandwidth)

    def _build_k_test(self, x_t: Optional[np.ndarray], n: int) -> np.ndarray:
        """Kernel vector between test point and current window, shape (n,).

        Returns ones(n) in residual-only mode.
        """
        if self._x_none_mode or x_t is None:
            return np.ones(n, dtype=np.float64)
        return _rbf_kernel(x_t.reshape(1, -1), self._X_window, self.bandwidth).ravel()

    def _build_k_new(self, x_t: Optional[np.ndarray], n_prev: int) -> np.ndarray:
        """Kernel vector between new point and existing window (size n_prev).

        Returns ones(n_prev) in residual-only mode.
        """
        if self._x_none_mode or x_t is None:
            return np.ones(n_prev, dtype=np.float64)
        return _rbf_kernel(x_t.reshape(1, -1), self._X_window, self.bandwidth).ravel()

    # ------------------------------------------------------------------
    # Q recomputation (periodic numerical reset)
    # ------------------------------------------------------------------

    def _recompute_Q(self) -> None:
        """Recompute Q from scratch to prevent floating-point drift."""
        n = len(self._Y_window)
        K = self._build_K_self(n)
        self._Q = np.linalg.inv(K + self.lambda_reg * np.eye(n, dtype=np.float64))
        self._steps_since_reset = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "RetroAdj":
        """Initialise the sliding window from training data.

        Takes the last min(len(y), window_size) observations as the initial
        window and computes Q = (K + lambda*I)^{-1} directly. No initial LOO
        residuals are seeded — calibration builds organically from the first
        test step.

        Parameters
        ----------
        y:
            Training responses (or residuals in residual-only mode), shape (n,).
        X:
            Training features, shape (n, p). Pass None for residual-only mode
            (see class docstring). Default None.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64).ravel()

        if X is not None:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if len(X) != len(y):
                raise ValueError(
                    f"X and y must have same length, got {len(X)} and {len(y)}"
                )

        if X is None:
            warnings.warn(
                "RetroAdj: X=None — running in residual-only mode. "
                "KRR degenerates to ridge-regularised mean. "
                "This is an approximation; pass y = actual - model_predict "
                "and optionally X = time_index.reshape(-1,1) for better fit.",
                UserWarning,
                stacklevel=2,
            )
            self._x_none_mode = True
        else:
            self._x_none_mode = False

        # Take last window_size observations
        n_init = min(len(y), self.window_size)
        self._Y_window = y[-n_init:].copy()
        self._X_window = X[-n_init:].copy() if X is not None else None

        self._recompute_Q()
        self._is_fitted = True
        self._steps_since_reset = 0

        return self

    def predict_interval(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        alpha: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce online prediction intervals, updating state after each step.

        Processes y (and X if provided) sequentially. At each step t:
        1. Computes LOO residuals from the current window via Q diagonal.
        2. Computes jackknife+ interval at the test point.
        3. Observes Y_t, updates alpha_t (ACI or SFOGD).
        4. Updates Q via rank-one downdate (if full) and rank-one update.

        Parameters
        ----------
        y:
            Test responses, shape (T,). Used after interval construction to
            update alpha_t and the window.
        X:
            Test features, shape (T, p). Must be None when fit(X=None) was used.
            Default None.
        alpha:
            Target miscoverage rate. 0.1 gives 90% prediction intervals.
            The actual alpha_t adapts step by step; this is the starting value.

        Returns
        -------
        (lower, upper):
            Interval bounds, each shape (T,).

        Notes
        -----
        When X=None, intervals are over the residual space. Add back your model
        predictions to get intervals on the original scale:

            lower_claims = lower + model.predict(X_test)
            upper_claims = upper + model.predict(X_test)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_interval().")

        y = np.asarray(y, dtype=np.float64).ravel()
        T = len(y)

        if X is not None:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if len(X) != T:
                raise ValueError(
                    f"X and y must have same length, got {len(X)} and {T}"
                )
            if self._x_none_mode:
                raise ValueError(
                    "fit() was called with X=None (residual-only mode). "
                    "Call predict_interval() with X=None too."
                )

        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        # Initialise alpha tracker
        alpha_t = float(alpha)
        if self.alpha_update == "sfogd":
            tracker = _SFOGDAlphaTracker(alpha_init=alpha_t, gamma=self.gamma)

        lower_out = np.empty(T, dtype=np.float64)
        upper_out = np.empty(T, dtype=np.float64)

        for t in range(T):
            x_t = X[t] if X is not None else None  # shape (p,) or None
            y_t = float(y[t])

            # --- Step 1: Build K_self and k_test for current window ---
            n_w = len(self._Y_window)

            K_self = self._build_K_self(n_w)          # (n_w, n_w)
            k_test = self._build_k_test(x_t, n_w)     # (n_w,)

            # --- Step 2: LOO residuals and hat matrix diagonal ---
            R_loo, H_diag, mask = _loo_residuals_and_leverages(
                self._Q, K_self, self._Y_window
            )

            # --- Step 3: LOO predictions at test point ---
            f_loo = _loo_predictions_at_test(
                self._Q, K_self, k_test, self._Y_window, R_loo, mask
            )

            # --- Step 4: Jackknife+ interval ---
            lo, hi = _jackknife_plus_interval(
                f_loo, R_loo, alpha_t, mask, symmetric=self.symmetric
            )
            lower_out[t] = lo
            upper_out[t] = hi

            # --- Step 5: Observe Y_t, update alpha_t ---
            covered = (lo <= y_t <= hi)
            if self.alpha_update == "aci":
                err = alpha - (0.0 if covered else 1.0)
                alpha_t = float(np.clip(alpha_t + self.gamma * err, 1e-6, 1.0 - 1e-6))
            else:  # sfogd
                alpha_t = tracker.update(covered, alpha)

            # --- Step 6: Update Q ---
            # Periodic reset to prevent numerical drift
            self._steps_since_reset += 1
            if self._steps_since_reset >= self.reset_freq:
                # Add new point first, then reset
                self._Y_window = np.append(self._Y_window, y_t)
                if x_t is not None:
                    self._X_window = np.vstack([self._X_window, x_t.reshape(1, -1)])
                # Remove oldest if over window size
                if len(self._Y_window) > self.window_size:
                    self._Y_window = self._Y_window[1:]
                    if self._X_window is not None:
                        self._X_window = self._X_window[1:]
                self._recompute_Q()
                continue

            try:
                # Downdate: remove oldest point if window is full
                if len(self._Y_window) >= self.window_size:
                    self._Q = _rank_one_downdate(self._Q)
                    self._Y_window = self._Y_window[1:]
                    if self._X_window is not None:
                        self._X_window = self._X_window[1:]

                # Update: add new point
                n_prev = len(self._Y_window)
                k_new = self._build_k_new(x_t, n_prev)

                self._Q = _rank_one_update(
                    self._Q, k_new,
                    kappa_self_val=1.0,  # RBF kappa(x,x)=1; also 1 in degenerate mode
                    lambda_reg=self.lambda_reg,
                )
                self._Y_window = np.append(self._Y_window, y_t)
                if x_t is not None:
                    self._X_window = np.vstack([self._X_window, x_t.reshape(1, -1)])

                # Enforce symmetry to combat floating-point drift
                self._Q = (self._Q + self._Q.T) / 2.0

            except _NumericalInstabilityError:
                # Recovery: add new point and recompute Q from scratch
                self._Y_window = np.append(self._Y_window, y_t)
                if x_t is not None:
                    self._X_window = np.vstack([self._X_window, x_t.reshape(1, -1)])
                if len(self._Y_window) > self.window_size:
                    self._Y_window = self._Y_window[1:]
                    if self._X_window is not None:
                        self._X_window = self._X_window[1:]
                self._recompute_Q()

        return lower_out, upper_out

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        mode = "residual-only" if self._x_none_mode else "full KRR"
        return (
            f"RetroAdj("
            f"bandwidth={self.bandwidth}, "
            f"lambda_reg={self.lambda_reg}, "
            f"window_size={self.window_size}, "
            f"gamma={self.gamma}, "
            f"alpha_update='{self.alpha_update}', "
            f"mode={mode}, "
            f"{status})"
        )
