"""
Core conformal calibration: fitting nonconformity scores on calibration data.

Design decisions:
1. Always use absolute residuals |y - y_hat| for interval construction.
   This ensures half-widths are in prediction units (claims, £, etc.).

2. Zero-claim masking: for severity dimensions, zero-claim obs have unobserved
   severity. We handle this by:
   - Computing mu_hat/sigma_hat for severity from claim-only obs (N>0).
   - For zero-claim calibration obs, the severity score is excluded from the
     max-score (they only contribute via the frequency score).
   - This avoids deflating sigma_hat with artificially zero residuals.

3. Coordinate-wise standardization handles Poisson/Gamma scale mismatch.
   Frequency residuals ~ 0.05-2 (claims/year), severity ~ £200-£5,000.
   After standardization both are z-scores — directly comparable for
   max-score aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .methods import (
    gwc_quantile,
    lwc_quantile_exact,
    bonferroni_quantile,
    sidak_quantile,
)


ModelLike = Any  # Any object with .predict(X) -> ndarray


def _get_predictions(
    model: ModelLike,
    X: NDArray,
    exposure: Optional[NDArray] = None,
) -> NDArray[np.floating]:
    """
    Get predictions from a model, optionally using exposure.

    Tries predict_with_exposure(X, exposure) first, then predict(X, exposure),
    then predict(X). Allows sklearn-style and custom wrappers.
    """
    if exposure is not None:
        if hasattr(model, "predict_with_exposure"):
            return np.asarray(model.predict_with_exposure(X, exposure), dtype=float)
        try:
            return np.asarray(model.predict(X, exposure), dtype=float)
        except TypeError:
            pass
    return np.asarray(model.predict(X), dtype=float)


def _normalise_models(
    models: Union[Dict[str, ModelLike], List[ModelLike], ModelLike],
) -> Dict[str, ModelLike]:
    """Normalise models to a dict keyed by dimension name."""
    if isinstance(models, dict):
        return models
    if isinstance(models, (list, tuple)):
        return {str(i): m for i, m in enumerate(models)}
    return {"0": models}


def _normalise_y(
    Y: Union[Dict[str, NDArray], NDArray],
    model_keys: Sequence[str],
) -> Dict[str, NDArray]:
    """Normalise Y to a dict keyed by dimension name."""
    if isinstance(Y, dict):
        return {k: np.asarray(v, dtype=float) for k, v in Y.items()}
    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 1:
        return {model_keys[0]: arr}
    if arr.ndim == 2:
        assert arr.shape[1] == len(model_keys), (
            f"Y has {arr.shape[1]} columns but {len(model_keys)} models"
        )
        return {k: arr[:, i] for i, k in enumerate(model_keys)}
    raise ValueError(f"Y must be 1D or 2D, got shape {arr.shape}")


def _compute_standardization_stats(
    residuals: NDArray[np.floating],
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
    severity_dim: Optional[int] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute per-dimension mu_hat and sigma_hat.

    For the severity dimension (if specified), compute stats from claim-only
    observations (where zero_claim_mask is False). This avoids deflating
    sigma_hat with artificially zero residuals.

    Returns (mu_hat, sigma_hat), each shape (d,).
    """
    n, d = residuals.shape
    mu_hat = np.zeros(d)
    sigma_hat = np.ones(d)

    for j in range(d):
        if (zero_claim_mask is not None
                and severity_dim is not None
                and j == severity_dim):
            # Use only claim observations for severity standardization
            claim_mask = ~zero_claim_mask
            if claim_mask.sum() < 2:
                # Not enough claim obs — fall back to all obs
                res_j = residuals[:, j]
            else:
                res_j = residuals[claim_mask, j]
        else:
            res_j = residuals[:, j]

        mu_hat[j] = np.mean(res_j)
        sigma_j = np.std(res_j)
        sigma_hat[j] = max(sigma_j, 1e-8)

    return mu_hat, sigma_hat


def _compute_masked_max_scores(
    residuals: NDArray[np.floating],
    mu_hat: NDArray[np.floating],
    sigma_hat: NDArray[np.floating],
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
    severity_dim: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Compute max standardized score per calibration observation.

    For zero-claim observations (where severity is unobserved), exclude
    the severity dimension from the max-score (use only frequency score).
    """
    n, d = residuals.shape
    standardized = (residuals - mu_hat) / sigma_hat

    if zero_claim_mask is None or severity_dim is None:
        return np.max(standardized, axis=1)

    # For zero-claim obs: max over all dims except severity
    max_scores = np.zeros(n)
    for i in range(n):
        if zero_claim_mask[i]:
            # Exclude severity dimension
            dims = [j for j in range(d) if j != severity_dim]
            if dims:
                max_scores[i] = np.max(standardized[i, dims])
            else:
                max_scores[i] = 0.0
        else:
            max_scores[i] = np.max(standardized[i])

    return max_scores


@dataclass
class CalibratedScores:
    """
    Calibration output: residuals, standardization statistics, and thresholds.

    Attributes
    ----------
    dimensions : list of str
        Dimension names (e.g. ['frequency', 'severity']).
    residuals : ndarray, shape (n, d)
        Absolute calibration residuals |y - y_hat| per dimension.
        In original prediction units (claims, £, etc.).
    mu_hat : ndarray, shape (d,)
        Per-dimension mean of calibration residuals (standardization mean).
    sigma_hat : ndarray, shape (d,)
        Per-dimension std of calibration residuals (standardization std).
        For severity, computed from claim-only observations when masked.
    n_cal : int
        Number of calibration points.
    method : str
        Conformal method: 'bonferroni', 'sidak', 'gwc', 'lwc'.
    alpha : float
        Joint miscoverage level.
    zero_claim_mask : ndarray of bool or None
        Which calibration observations had zero claims (severity excluded).
    severity_dim : int or None
        Index of severity dimension in residuals (for masking).

    Computed thresholds:
    thresholds : ndarray, shape (d,) or None
        For GWC/LWC: per-dimension thresholds in standardized units.
    per_dim_quantiles : ndarray, shape (d,) or None
        For Bonferroni/Sidak: per-dimension raw quantiles in original units.
    """

    dimensions: List[str]
    residuals: NDArray[np.floating]
    mu_hat: NDArray[np.floating]
    sigma_hat: NDArray[np.floating]
    n_cal: int
    method: str
    alpha: float
    zero_claim_mask: Optional[NDArray[np.bool_]] = None
    severity_dim: Optional[int] = None

    thresholds: Optional[NDArray[np.floating]] = field(default=None, repr=False)
    per_dim_quantiles: Optional[NDArray[np.floating]] = field(default=None, repr=False)

    def standardized_residuals(self) -> NDArray[np.floating]:
        """Return (residuals - mu_hat) / sigma_hat, shape (n, d)."""
        return (self.residuals - self.mu_hat) / self.sigma_hat

    def max_scores(self) -> NDArray[np.floating]:
        """Return max standardized score per calibration observation, shape (n,)."""
        return _compute_masked_max_scores(
            self.residuals, self.mu_hat, self.sigma_hat,
            self.zero_claim_mask, self.severity_dim,
        )

    def interval_half_widths(self) -> NDArray[np.floating]:
        """
        Per-dimension interval half-widths in original prediction units.

        For GWC/LWC: hw_j = threshold_j * sigma_hat_j + mu_hat_j
        For Bonferroni/Sidak: hw_j = per_dim_quantiles_j (raw quantile)

        Returns ndarray, shape (d,).
        """
        if self.method in ("gwc", "lwc"):
            if self.thresholds is None:
                raise RuntimeError("thresholds not set — internal error")
            return self.thresholds * self.sigma_hat + self.mu_hat
        else:  # bonferroni or sidak
            if self.per_dim_quantiles is None:
                raise RuntimeError("per_dim_quantiles not set — internal error")
            return self.per_dim_quantiles


def _compute_absolute_residuals(
    models: Dict[str, ModelLike],
    X_cal: NDArray,
    Y_cal: Dict[str, NDArray],
    exposure: Optional[NDArray] = None,
) -> NDArray[np.floating]:
    """
    Compute absolute residuals |y - y_hat| per dimension, shape (n, d).
    Always in original prediction units — correct for interval construction.
    """
    keys = list(models.keys())
    n = X_cal.shape[0]
    d = len(keys)
    residuals = np.zeros((n, d), dtype=float)

    for j, key in enumerate(keys):
        model = models[key]
        y_true = Y_cal[key]
        y_pred = _get_predictions(model, X_cal, exposure)
        residuals[:, j] = np.abs(y_true - y_pred)

    return residuals


def compute_residuals(
    models: Dict[str, ModelLike],
    X_cal: NDArray,
    Y_cal: Dict[str, NDArray],
    score_fn: str = "absolute",
    exposure: Optional[NDArray] = None,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.floating]:
    """
    Compute nonconformity residuals for each dimension.

    Returns absolute residuals |y - y_hat|, shape (n, d).
    zero_claim_mask and score_fn accepted for API compatibility.
    """
    return _compute_absolute_residuals(models, X_cal, Y_cal, exposure=exposure)


def _gwc_with_masked_scores(
    residuals: NDArray[np.floating],
    alpha: float,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
    severity_dim: Optional[int] = None,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """
    GWC with optional zero-claim masking.

    Computes mu_hat/sigma_hat from appropriate subsets, then finds the
    conformal quantile of the masked max-scores.
    """
    mu_hat, sigma_hat = _compute_standardization_stats(
        residuals, zero_claim_mask=zero_claim_mask, severity_dim=severity_dim
    )
    max_scores = _compute_masked_max_scores(
        residuals, mu_hat, sigma_hat,
        zero_claim_mask=zero_claim_mask, severity_dim=severity_dim,
    )
    n = len(max_scores)
    k = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)
    q = np.sort(max_scores)[k - 1]
    return q, mu_hat, sigma_hat


def _lwc_with_masked_scores(
    residuals: NDArray[np.floating],
    alpha: float,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
    severity_dim: Optional[int] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    LWC with optional zero-claim masking.

    Uses the exact LWC per-group quantile with masked max-scores.
    """
    n, d = residuals.shape
    mu_hat, sigma_hat = _compute_standardization_stats(
        residuals, zero_claim_mask=zero_claim_mask, severity_dim=severity_dim
    )
    max_scores = _compute_masked_max_scores(
        residuals, mu_hat, sigma_hat,
        zero_claim_mask=zero_claim_mask, severity_dim=severity_dim,
    )
    standardized = (residuals - mu_hat) / sigma_hat

    # Argmax dimension per observation (using masked max scores)
    # For zero-claim obs, argmax excludes severity_dim
    argmax_dims = np.zeros(n, dtype=int)
    for i in range(n):
        if (zero_claim_mask is not None
                and severity_dim is not None
                and zero_claim_mask[i]):
            dims = [j for j in range(d) if j != severity_dim]
            if dims:
                argmax_dims[i] = dims[int(np.argmax(standardized[i, dims]))]
            else:
                argmax_dims[i] = 0
        else:
            argmax_dims[i] = int(np.argmax(standardized[i]))

    # Global quantile
    k_global = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)
    q_global = np.sort(max_scores)[k_global - 1]

    thresholds = np.full(d, q_global)
    for j in range(d):
        group_mask = argmax_dims == j
        n_j = int(np.sum(group_mask))
        if n_j == 0:
            continue
        group_scores = max_scores[group_mask]
        k_j = min(int(np.ceil((n_j + 1) * (1.0 - alpha))), n_j)
        t_j = np.sort(group_scores)[k_j - 1]
        thresholds[j] = min(t_j, q_global)

    return thresholds, mu_hat, sigma_hat


def calibrate(
    models: Union[Dict[str, ModelLike], List[ModelLike]],
    X_cal: NDArray,
    Y_cal: Union[Dict[str, NDArray], NDArray],
    alpha: float = 0.05,
    method: str = "lwc",
    score_fn: str = "absolute",
    exposure: Optional[NDArray] = None,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
) -> CalibratedScores:
    """
    Calibrate a joint conformal predictor on held-out data.

    Parameters
    ----------
    models : dict or list
        Fitted base models. Must implement .predict(X) -> ndarray shape (n,).
        Dict: {'frequency': glm, 'severity': gbm}
        List: [model_0, model_1, ...]
    X_cal : ndarray, shape (n, p)
        Calibration feature matrix (must be held-out from training).
    Y_cal : dict or ndarray
        Calibration targets. Dict keys must match model keys.
    alpha : float
        Joint miscoverage level. 0.05 = 95% joint coverage.
    method : str
        'bonferroni': per-dimension quantile at 1-alpha/d. Conservative, always valid.
        'sidak': per-dimension at 1-(1-alpha)^(1/d). Valid under independence only.
        'gwc': global worst-case max-score. Valid, O(dn).
        'lwc': local worst-case max-score. Valid, tightest, O(d^2 n log n).
    score_fn : str
        Accepted for API compatibility. Currently 'absolute' used for all methods.
    exposure : ndarray, shape (n,), optional
        Exposure values for Poisson frequency models.
    zero_claim_mask : ndarray of bool, shape (n,), optional
        True where N=0 (severity unobserved). For GWC/LWC, severity standardization
        uses claim-only obs and zero-claim obs contribute only via frequency score.
        For Bonferroni/Sidak, severity quantile computed from claim-only obs.

    Returns
    -------
    CalibratedScores
    """
    models_dict = _normalise_models(models)
    keys = list(models_dict.keys())
    Y_dict = _normalise_y(Y_cal, keys)

    n = X_cal.shape[0]
    for k in keys:
        if k not in Y_dict:
            raise ValueError(f"Y_cal missing key '{k}' (available: {list(Y_dict.keys())})")
        if Y_dict[k].shape[0] != n:
            raise ValueError(
                f"Y_cal['{k}'] has {Y_dict[k].shape[0]} rows, X_cal has {n}"
            )

    if zero_claim_mask is not None:
        zero_claim_mask = np.asarray(zero_claim_mask, dtype=bool)
        if zero_claim_mask.shape[0] != n:
            raise ValueError(
                f"zero_claim_mask has {zero_claim_mask.shape[0]} entries, expected {n}"
            )

    residuals = _compute_absolute_residuals(
        models_dict, X_cal, Y_dict, exposure=exposure
    )
    d = len(keys)

    # Find severity dimension index for masking
    severity_dim: Optional[int] = None
    if zero_claim_mask is not None and "severity" in keys:
        severity_dim = keys.index("severity")

    if method == "bonferroni":
        # Per-dimension quantile at 1 - alpha/d
        # For severity: compute quantile from claim-only observations
        alpha_per_dim = alpha / d
        level = 1.0 - alpha_per_dim
        per_dim_q = np.zeros(d)
        for j, key in enumerate(keys):
            if zero_claim_mask is not None and key == "severity":
                claim_mask = ~zero_claim_mask
                if claim_mask.sum() >= 1:
                    res_j = residuals[claim_mask, j]
                else:
                    res_j = residuals[:, j]
            else:
                res_j = residuals[:, j]
            n_j = len(res_j)
            k_j = min(int(np.ceil((n_j + 1) * level)), n_j)
            per_dim_q[j] = np.sort(res_j)[k_j - 1]

        mu_hat, sigma_hat = _compute_standardization_stats(
            residuals, zero_claim_mask=zero_claim_mask, severity_dim=severity_dim
        )
        cal = CalibratedScores(
            dimensions=keys, residuals=residuals, mu_hat=mu_hat, sigma_hat=sigma_hat,
            n_cal=n, method=method, alpha=alpha, zero_claim_mask=zero_claim_mask,
            severity_dim=severity_dim, per_dim_quantiles=per_dim_q,
        )

    elif method == "sidak":
        # Per-dimension quantile at 1 - (1-(1-alpha))^(1/d)
        alpha_per_dim = 1.0 - (1.0 - alpha) ** (1.0 / d)
        level = 1.0 - alpha_per_dim
        per_dim_q = np.zeros(d)
        for j, key in enumerate(keys):
            if zero_claim_mask is not None and key == "severity":
                claim_mask = ~zero_claim_mask
                if claim_mask.sum() >= 1:
                    res_j = residuals[claim_mask, j]
                else:
                    res_j = residuals[:, j]
            else:
                res_j = residuals[:, j]
            n_j = len(res_j)
            k_j = min(int(np.ceil((n_j + 1) * level)), n_j)
            per_dim_q[j] = np.sort(res_j)[k_j - 1]

        mu_hat, sigma_hat = _compute_standardization_stats(
            residuals, zero_claim_mask=zero_claim_mask, severity_dim=severity_dim
        )
        cal = CalibratedScores(
            dimensions=keys, residuals=residuals, mu_hat=mu_hat, sigma_hat=sigma_hat,
            n_cal=n, method=method, alpha=alpha, zero_claim_mask=zero_claim_mask,
            severity_dim=severity_dim, per_dim_quantiles=per_dim_q,
        )

    elif method == "gwc":
        q_scalar, mu_hat, sigma_hat = _gwc_with_masked_scores(
            residuals, alpha,
            zero_claim_mask=zero_claim_mask, severity_dim=severity_dim,
        )
        cal = CalibratedScores(
            dimensions=keys, residuals=residuals, mu_hat=mu_hat, sigma_hat=sigma_hat,
            n_cal=n, method=method, alpha=alpha, zero_claim_mask=zero_claim_mask,
            severity_dim=severity_dim, thresholds=np.full(d, q_scalar),
        )

    elif method == "lwc":
        thresholds, mu_hat, sigma_hat = _lwc_with_masked_scores(
            residuals, alpha,
            zero_claim_mask=zero_claim_mask, severity_dim=severity_dim,
        )
        cal = CalibratedScores(
            dimensions=keys, residuals=residuals, mu_hat=mu_hat, sigma_hat=sigma_hat,
            n_cal=n, method=method, alpha=alpha, zero_claim_mask=zero_claim_mask,
            severity_dim=severity_dim, thresholds=thresholds,
        )

    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose 'bonferroni', 'sidak', 'gwc', 'lwc'."
        )

    return cal
