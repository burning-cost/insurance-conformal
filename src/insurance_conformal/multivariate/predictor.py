"""
Main prediction API: JointConformalPredictor.

This is the class users interact with. It wraps the calibration and prediction
steps and handles the scikit-learn-compatible interface.

Usage pattern:
    predictor = JointConformalPredictor(
        models={'frequency': freq_glm, 'severity': sev_gbm},
        alpha=0.05,
        method='lwc',
    )
    predictor.calibrate(X_cal, Y_cal, exposure=exposure_cal)
    joint_set = predictor.predict(X_test, exposure=exposure_test)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .calibration import CalibratedScores, calibrate, _normalise_models, _normalise_y, _get_predictions
from .regions import JointPredictionSet


class JointConformalPredictor:
    """
    Joint multi-output conformal predictor with coordinate-wise standardization.

    Produces hyperrectangular prediction sets with finite-sample joint coverage
    guarantee: P(Y ∈ Ĉ(X)) >= 1 - alpha, under exchangeability of calibration
    and test data.

    Parameters
    ----------
    models : dict or list
        Fitted base models. Must implement .predict(X) returning shape (n,).
        If dict: {'frequency': model_freq, 'severity': model_sev}
        If list: [model_0, model_1, ...]
        Each model handles one output dimension.
    alpha : float
        Joint miscoverage level. 0.05 gives 95% joint coverage. 0.005 for SCR.
    method : str
        'bonferroni': Bonferroni-corrected per-dimension intervals (valid, conservative).
        'sidak': Sidak-corrected (valid only under independence — use carefully).
        'gwc': Global worst-case max-score (Fan & Sesia, valid, moderate width).
        'lwc': Local worst-case max-score (Fan & Sesia, valid, tightest).
    score_fn : str
        'auto': poisson_deviance for 'frequency', gamma_deviance for 'severity'.
        'absolute': |y - y_hat| for all dimensions.
        'poisson_deviance', 'gamma_deviance': applied to all dimensions.
    one_sided : bool
        If True, lower bounds are clamped to 0. Used for SCR / VaR applications
        where you only need the upper tail (loss is non-negative).

    Attributes
    ----------
    calibrated_scores_ : CalibratedScores or None
        Set after .calibrate(). Contains residuals and thresholds.
    dimensions_ : list of str
        Dimension names, set after .calibrate().
    """

    def __init__(
        self,
        models: Union[Dict[str, Any], List[Any]],
        alpha: float = 0.05,
        method: str = "lwc",
        score_fn: str = "auto",
        one_sided: bool = False,
    ):
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        valid_methods = ("bonferroni", "sidak", "gwc", "lwc")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

        self.models = _normalise_models(models)
        self.alpha = alpha
        self.method = method
        self.score_fn = score_fn
        self.one_sided = one_sided

        self.calibrated_scores_: Optional[CalibratedScores] = None
        self.dimensions_: Optional[List[str]] = None

    def calibrate(
        self,
        X_cal: NDArray,
        Y_cal: Union[Dict[str, NDArray], NDArray],
        exposure: Optional[NDArray] = None,
        zero_claim_mask: Optional[NDArray[np.bool_]] = None,
    ) -> "JointConformalPredictor":
        """
        Fit conformal calibration on held-out calibration data.

        Parameters
        ----------
        X_cal : ndarray, shape (n, p)
            Calibration feature matrix (NOT the training set — must be held out).
        Y_cal : dict or ndarray
            Calibration targets. Dict keys must match model dict keys.
        exposure : ndarray, shape (n,), optional
            Exposure values for Poisson frequency models.
        zero_claim_mask : ndarray of bool, shape (n,), optional
            True where severity is unobserved (zero-claim observations).
            These rows get severity residual = 0 in calibration (conservative).

        Returns
        -------
        self
        """
        self.calibrated_scores_ = calibrate(
            models=self.models,
            X_cal=X_cal,
            Y_cal=Y_cal,
            alpha=self.alpha,
            method=self.method,
            score_fn=self.score_fn,
            exposure=exposure,
            zero_claim_mask=zero_claim_mask,
        )
        self.dimensions_ = self.calibrated_scores_.dimensions
        return self

    def predict(
        self,
        X_new: NDArray,
        exposure: Optional[NDArray] = None,
    ) -> JointPredictionSet:
        """
        Produce joint prediction sets for new observations.

        Parameters
        ----------
        X_new : ndarray, shape (m, p)
            Test feature matrix.
        exposure : ndarray, shape (m,), optional
            Exposure values for Poisson frequency predictions.

        Returns
        -------
        JointPredictionSet
            Joint prediction set with lower/upper bounds per dimension.
            Bounds are in the original (raw) scale of each output.
        """
        if self.calibrated_scores_ is None:
            raise RuntimeError("Must call .calibrate() before .predict()")

        cal = self.calibrated_scores_
        m = X_new.shape[0]

        # Get point predictions per dimension
        lower = {}
        upper = {}

        # Get per-dimension interval half-widths (in original units)
        half_widths = cal.interval_half_widths()

        for j, key in enumerate(cal.dimensions):
            model = self.models[key]
            y_pred = _get_predictions(model, X_new, exposure)

            hw = half_widths[j]
            lo = y_pred - hw
            hi = y_pred + hw

            if self.one_sided:
                lo = np.zeros_like(lo)

            lower[key] = lo
            upper[key] = hi

        return JointPredictionSet(
            lower=lower,
            upper=upper,
            dimensions=cal.dimensions,
            alpha=self.alpha,
            method=self.method,
            one_sided=self.one_sided,
        )

    def calibration_summary(self) -> Dict[str, object]:
        """Return a summary of the calibration statistics."""
        if self.calibrated_scores_ is None:
            raise RuntimeError("Not yet calibrated")
        cal = self.calibrated_scores_
        hw = cal.interval_half_widths()
        return {
            "dimensions": cal.dimensions,
            "n_cal": cal.n_cal,
            "method": cal.method,
            "alpha": cal.alpha,
            "mu_hat": dict(zip(cal.dimensions, cal.mu_hat.tolist())),
            "sigma_hat": dict(zip(cal.dimensions, cal.sigma_hat.tolist())),
            "half_widths": dict(zip(cal.dimensions, hw.tolist())),
        }

    def is_calibrated(self) -> bool:
        """True if .calibrate() has been called."""
        return self.calibrated_scores_ is not None

    def __repr__(self) -> str:
        status = "calibrated" if self.is_calibrated() else "not calibrated"
        dims = list(self.models.keys())
        return (
            f"JointConformalPredictor("
            f"dims={dims}, alpha={self.alpha}, "
            f"method='{self.method}', {status})"
        )
