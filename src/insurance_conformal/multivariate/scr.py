"""
Solvency II SCR estimation via joint conformal prediction.

Solvency II Article 101 requires the SCR to cover losses at the 99.5% VaR
level. Standard formula uses prescribed correlation matrices — a parametric
approach that may underestimate joint tail risk.

The conformal approach (Hong 2025, arXiv:2503.03659):
1. Run a one-sided joint conformal predictor at alpha=0.005 (99.5% coverage).
2. The upper bounds [U_freq, U_sev] form a conservative joint upper bound on
   the one-year loss.
3. SCR_conformal = U_freq * U_sev (product of upper bounds = conservative
   estimate of joint worst-case pure premium).

The coverage guarantee is finite-sample: P(loss <= SCR_conformal) >= 99.5%
with no distributional assumption. This is strictly stronger than the parametric
normal/lognormal assumptions in internal models.

Practical note: at n=999 calibration points, the SCR guarantee is
P >= 99.5% exactly (with at most 0.1% excess conservatism). For n=199,
the excess conservatism is 0.45% — still valid but wider intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .predictor import JointConformalPredictor
from .regions import JointPredictionSet


@dataclass
class SCRResult:
    """
    Result of SolvencyCapitalEstimator.estimate().

    Attributes
    ----------
    joint_upper : dict[str, ndarray]
        Per-dimension 99.5% upper bounds, shape (n_policies,).
        E.g. joint_upper['frequency'] = upper bound on claim frequency.
    scr_per_policy : ndarray, shape (n_policies,)
        Conservative SCR per policy = product of joint upper bounds.
        For freq+sev: scr = U_freq * U_sev.
    aggregate_scr : float
        Sum of per-policy SCR (conservative portfolio aggregate).
    coverage_guarantee : float
        Nominal coverage guarantee = 1 - alpha (e.g. 0.995).
    n_cal : int
        Calibration set size (determines tightness of guarantee).
    finite_sample_bound : float
        Exact finite-sample upper bound on miscoverage: 1/(n_cal+1).
    method : str
        Conformal method used.
    """
    joint_upper: Dict[str, NDArray]
    scr_per_policy: NDArray[np.floating]
    aggregate_scr: float
    coverage_guarantee: float
    n_cal: int
    finite_sample_bound: float
    method: str
    alpha: float

    def bootstrap_ci(
        self,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """
        Bootstrap confidence interval for aggregate_scr.

        Resamples per-policy SCR values with replacement to give a CI on
        the portfolio-level aggregate SCR.

        Returns dict with 'lower', 'upper', 'mean'.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        n = len(self.scr_per_policy)
        boot_means = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = np.sum(self.scr_per_policy[idx])
        lo = float(np.quantile(boot_means, (1 - ci_level) / 2))
        hi = float(np.quantile(boot_means, 1 - (1 - ci_level) / 2))
        return {"lower": lo, "upper": hi, "mean": float(np.mean(boot_means))}

    def summary(self) -> Dict[str, object]:
        return {
            "aggregate_scr": self.aggregate_scr,
            "coverage_guarantee": self.coverage_guarantee,
            "finite_sample_bound": self.finite_sample_bound,
            "n_cal": self.n_cal,
            "method": self.method,
            "alpha": self.alpha,
            "per_policy_stats": {
                "mean": float(np.mean(self.scr_per_policy)),
                "p50": float(np.median(self.scr_per_policy)),
                "p95": float(np.quantile(self.scr_per_policy, 0.95)),
                "p99": float(np.quantile(self.scr_per_policy, 0.99)),
                "max": float(np.max(self.scr_per_policy)),
            },
        }


class SolvencyCapitalEstimator:
    """
    Solvency II SCR estimator using joint conformal prediction.

    Constructs a one-sided joint prediction set at alpha=0.005 (99.5% coverage).
    Returns conservative SCR bounds per policy and portfolio aggregate.

    Parameters
    ----------
    models : dict
        {'frequency': freq_model, 'severity': sev_model} or similar.
    alpha : float
        Miscoverage level. Default 0.005 (99.5% VaR for Solvency II).
    method : str
        'gwc' (recommended for regulatory use — more conservative) or 'lwc'.
    score_fn : str
        Score function. Default 'auto'.
    """

    def __init__(
        self,
        models: Union[Dict[str, Any], List[Any]],
        alpha: float = 0.005,
        method: str = "gwc",
        score_fn: str = "auto",
    ):
        if alpha > 0.01:
            import warnings
            warnings.warn(
                f"alpha={alpha} is larger than 0.01. For Solvency II SCR use alpha=0.005.",
                UserWarning,
                stacklevel=2,
            )
        self._predictor = JointConformalPredictor(
            models=models,
            alpha=alpha,
            method=method,
            score_fn=score_fn,
            one_sided=True,
        )
        self.alpha = alpha
        self.method = method

    def calibrate(
        self,
        X_cal: NDArray,
        Y_cal: Union[Dict[str, NDArray], NDArray],
        exposure: Optional[NDArray] = None,
        zero_claim_mask: Optional[NDArray[np.bool_]] = None,
    ) -> "SolvencyCapitalEstimator":
        """
        Calibrate on held-out data.

        Parameters
        ----------
        X_cal : ndarray, shape (n, p)
        Y_cal : dict or ndarray
        exposure : ndarray, shape (n,), optional
        zero_claim_mask : ndarray of bool, shape (n,), optional

        Returns
        -------
        self
        """
        self._predictor.calibrate(
            X_cal=X_cal,
            Y_cal=Y_cal,
            exposure=exposure,
            zero_claim_mask=zero_claim_mask,
        )
        return self

    def estimate(
        self,
        X_new: NDArray,
        exposure: Optional[NDArray] = None,
    ) -> SCRResult:
        """
        Estimate SCR for a portfolio of policies.

        Parameters
        ----------
        X_new : ndarray, shape (m, p)
            Policy feature matrix.
        exposure : ndarray, shape (m,), optional

        Returns
        -------
        SCRResult
        """
        if not self._predictor.is_calibrated():
            raise RuntimeError("Must call .calibrate() before .estimate()")

        joint_set = self._predictor.predict(X_new, exposure=exposure)

        # SCR per policy = product of upper bounds across dimensions
        # For freq + sev: SCR_i = U_freq_i * U_sev_i
        # More generally: product over all output dimensions.
        dims = joint_set.dimensions
        scr_per_policy = np.ones(joint_set.n_obs, dtype=float)
        for k in dims:
            scr_per_policy *= np.maximum(joint_set.upper[k], 0.0)

        n_cal = self._predictor.calibrated_scores_.n_cal
        finite_sample_bound = 1.0 / (n_cal + 1)

        return SCRResult(
            joint_upper={k: joint_set.upper[k] for k in dims},
            scr_per_policy=scr_per_policy,
            aggregate_scr=float(np.sum(scr_per_policy)),
            coverage_guarantee=1.0 - self.alpha,
            n_cal=n_cal,
            finite_sample_bound=finite_sample_bound,
            method=self.method,
            alpha=self.alpha,
        )


def scr_report(
    estimator: SolvencyCapitalEstimator,
    X_portfolio: NDArray,
    exposure: Optional[NDArray] = None,
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Full SCR report with bootstrap confidence interval.

    Parameters
    ----------
    estimator : SolvencyCapitalEstimator
        A calibrated SCR estimator.
    X_portfolio : ndarray, shape (m, p)
        Full portfolio feature matrix.
    exposure : ndarray, shape (m,), optional
    n_bootstrap : int
        Bootstrap replications for CI on aggregate SCR.
    ci_level : float
        CI level (default 0.95 = 95% CI).

    Returns
    -------
    dict with aggregate_scr, bootstrap_ci, per_policy stats, coverage_guarantee.
    """
    result = estimator.estimate(X_portfolio, exposure=exposure)
    boot_ci = result.bootstrap_ci(n_bootstrap=n_bootstrap, ci_level=ci_level)
    summary = result.summary()
    summary["bootstrap_ci"] = boot_ci
    summary["n_portfolio"] = X_portfolio.shape[0]
    return summary
