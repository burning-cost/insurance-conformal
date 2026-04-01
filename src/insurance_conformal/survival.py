"""
ConformalisedSurvival: doubly robust conformalized survival analysis for
right-censored insurance data.

Right-censored survival data appears throughout insurance in forms that are not
always labelled as 'survival analysis': time to lapse, time to death in life and
protection books, time to claim closure, time to large-loss development. In each
case you observe either the event time T_i (if the event occurred before
observation ended) or a censored time C_i (the point at which the policy was
still active when you stopped watching). Standard conformal prediction fails
here because it requires fully observed outcomes.

The fix, due to Sesia & Svetnik (arXiv:2412.09729), is to impute the unobserved
censoring times for event-observed points using the conditional censoring
distribution, then apply importance-weighted conformal inference with IPCW
weights. The result is an asymptotically valid lower prediction bound (LPB):

    P[T >= L_hat(X)] >= 1 - alpha

provided at least one of the two models (survival or censoring) is correctly
specified — the 'doubly robust' property. With both models misspecified, coverage
degrades gracefully rather than collapsing.

The insurance interpretation: if you produce a lower bound of, say, 5 years for a
protection policyholder, you are saying with 90% confidence that the policyholder
will not die (or lapse) within 5 years. Directly useful for reserve adequacy and
pricing stress-testing.

Algorithm references:
    Algorithm 1: Censoring time imputation (Section 2.2 of Sesia & Svetnik 2024)
    Algorithm 2: Fixed-cutoff calibration with IPCW weights (Section 2.3)

Paper: Sesia, M. & Svetnik, V. (2024). 'Doubly Robust Conformalized Survival
Analysis with Right-Censored Data.' arXiv:2412.09729.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy, ipcw_weighted_quantile


# ---------------------------------------------------------------------------
# Protocols and adapters
# ---------------------------------------------------------------------------


@runtime_checkable
class SurvivalModelProtocol(Protocol):
    """
    Protocol for survival models used with ConformalisedSurvival.

    Any model that implements predict_quantile(X, alpha) is compatible.
    The thin adapters below wrap lifelines and scikit-survival to this interface.
    """

    def predict_quantile(self, X: np.ndarray, alpha: float) -> np.ndarray:
        """
        Predict the alpha-quantile of the survival time distribution.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Feature matrix.
        alpha : float
            Quantile level in (0, 1). alpha=0.10 gives the 10th percentile
            of the survival distribution (i.e. the lower 10% bound).

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted quantile values. Positive.
        """
        ...


@runtime_checkable
class CensoringModelProtocol(Protocol):
    """
    Protocol for censoring models used with ConformalisedSurvival.

    The censoring model must provide survival function estimates
    P[C > t | X=x] for arbitrary t values. This is used both for IPCW
    weight computation and for the truncated sampling in Algorithm 1.
    """

    def predict_survival_at(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Predict survival probability P[C > t | X=x] for each (x_i, t_i) pair.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Feature matrix.
        times : np.ndarray, shape (n,)
            Times at which to evaluate survival probability.

        Returns
        -------
        np.ndarray, shape (n,)
            Survival probabilities in [0, 1].
        """
        ...

    def sample_truncated(
        self,
        X: np.ndarray,
        t_min: np.ndarray,
        n_draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample censoring times from the truncated distribution P[C | C > t_min, X].

        For event-observed points (E_i=1), we know T_i < C_i but do not observe
        C_i. This method samples C_i from the conditional distribution
        f_{C|X}(c|x) / P[C > T_i | X=x] restricted to c > T_i.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Feature matrix.
        t_min : np.ndarray, shape (n,)
            Lower truncation bounds. One per row of X.
        n_draws : int
            Number of Monte Carlo draws per point.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray, shape (n,)
            Mean of n_draws samples from the truncated distribution per point.
        """
        ...


class KaplanMeierCensoringModel:
    """
    Non-covariate censoring model using the Kaplan-Meier estimator on the
    censoring indicator (i.e. KM on 1 - E, treating censoring as the event).

    This is the simplest possible censoring model: it assumes censoring is
    independent of covariates (covariate-independent censoring). Valid when
    administrative censoring dominates (e.g. study end date), but questionable
    when policyholders lapse for reasons correlated with risk.

    Implements CensoringModelProtocol via discrete-time inverse transform sampling.

    Parameters
    ----------
    random_state : int or None
        Passed to np.random.default_rng. Not used here (deterministic KM).
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._times: Optional[np.ndarray] = None
        self._survival: Optional[np.ndarray] = None

    def fit(self, t: np.ndarray, e: np.ndarray) -> "KaplanMeierCensoringModel":
        """
        Fit the Kaplan-Meier estimator on the censoring event indicator.

        Treats censoring (E=0) as the event of interest and event-observed
        points (E=1) as censored for the censoring KM. This is the standard
        reverse-KM approach for censoring distribution estimation.

        Parameters
        ----------
        t : np.ndarray, shape (n,)
            Observed times (min(T, C) for each observation).
        e : np.ndarray, shape (n,)
            Event indicators. 1 = event observed, 0 = censored.

        Returns
        -------
        self
        """
        t = np.asarray(t, dtype=float)
        e = np.asarray(e, dtype=float)

        # For the censoring KM: censoring event = 1-e, at-risk indicator reversed
        c_event = 1 - e  # 1 if censored, 0 if event-observed
        unique_times = np.sort(np.unique(t))

        survival = np.ones(len(unique_times) + 1)
        times_with_zero = np.concatenate([[0.0], unique_times])

        n = len(t)
        at_risk = n

        for i, ti in enumerate(unique_times):
            mask = t == ti
            d_i = int(c_event[mask].sum())  # censoring events at time ti
            n_i = int((t >= ti).sum())       # at-risk at time ti

            if n_i > 0 and d_i > 0:
                survival[i + 1] = survival[i] * (1.0 - d_i / n_i)
            else:
                survival[i + 1] = survival[i]

        self._times = times_with_zero
        self._survival = survival
        return self

    def _survival_at(self, t_query: float) -> float:
        """Evaluate KM survival curve at a single time point."""
        assert self._times is not None
        idx = np.searchsorted(self._times, t_query, side="right") - 1
        idx = int(np.clip(idx, 0, len(self._survival) - 1))
        return float(self._survival[idx])

    def predict_survival_at(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Predict P[C > t] ignoring covariates (KM marginal estimate)."""
        assert self._times is not None, "Call fit() before predict_survival_at()."
        return np.array([self._survival_at(float(ti)) for ti in times])

    def sample_truncated(
        self,
        X: np.ndarray,
        t_min: np.ndarray,
        n_draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample from P[C | C > t_min] via inverse transform sampling on the KM grid.

        For each point i, draws n_draws values from the conditional KM
        distribution and returns the mean.
        """
        assert self._times is not None, "Call fit() before sample_truncated()."
        times = self._times
        survival = self._survival

        result = np.empty(len(t_min))
        for i, t0 in enumerate(t_min):
            # Find the portion of the KM curve above t0
            idx_start = int(np.searchsorted(times, t0, side="right"))
            valid_times = times[idx_start:]
            valid_surv = survival[idx_start:]

            if len(valid_times) == 0 or valid_surv[0] < 1e-12:
                # Truncation point is beyond the support — return t0
                result[i] = float(t0)
                continue

            # Normalise the conditional distribution
            s0 = valid_surv[0]
            cdf = 1.0 - valid_surv / s0  # conditional CDF at each time step
            cdf = np.clip(cdf, 0.0, 1.0)

            # Inverse transform sampling: sample uniform, find corresponding time
            u = rng.uniform(0.0, 1.0, size=n_draws)
            draws = np.empty(n_draws)
            for k, u_k in enumerate(u):
                idx_k = int(np.searchsorted(cdf, u_k))
                if idx_k >= len(valid_times):
                    idx_k = len(valid_times) - 1
                draws[k] = float(valid_times[idx_k])

            result[i] = float(draws.mean())

        return result


class LifelinesCoxCensoringAdapter:
    """
    Adapter wrapping a fitted lifelines CoxPHFitter as a CensoringModelProtocol.

    The lifelines CoxPHFitter can be used as the censoring model by fitting
    it on the reverse event indicator (censoring as the event). This adapter
    converts the lifelines API to the CensoringModelProtocol interface.

    Parameters
    ----------
    model : lifelines.CoxPHFitter
        A fitted lifelines CoxPHFitter. Must have been fitted with the
        censoring indicator as the event column (1 - original_event_col).

    Notes
    -----
    Requires lifelines>=0.27. Install with:
        pip install insurance-conformal[survival]
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_survival_at(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        Predict P[C > t_i | X=x_i] using the fitted Cox model.

        Evaluates the survival function at each (x_i, t_i) pair by
        calling predict_survival_function on the lifelines model.
        """
        import pandas as pd

        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        sf = self.model.predict_survival_function(X_df)
        # sf is a DataFrame: index=times, columns=observations
        result = np.empty(len(times))
        for i, ti in enumerate(times):
            col = sf.iloc[:, i]
            idx = np.searchsorted(col.index.values, float(ti), side="right") - 1
            idx = int(np.clip(idx, 0, len(col) - 1))
            result[i] = float(col.iloc[idx])
        return result

    def sample_truncated(
        self,
        X: np.ndarray,
        t_min: np.ndarray,
        n_draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample from the truncated Cox censoring distribution via ITS on the
        discrete survival grid.
        """
        import pandas as pd

        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        sf = self.model.predict_survival_function(X_df)
        grid_times = sf.index.values.astype(float)

        result = np.empty(len(t_min))
        for i, t0 in enumerate(t_min):
            surv_i = sf.iloc[:, i].values

            # Restrict to times > t0
            mask = grid_times > t0
            if not mask.any():
                result[i] = float(t0)
                continue

            valid_times = grid_times[mask]
            valid_surv = surv_i[mask]
            s0_idx = np.searchsorted(grid_times, t0, side="right") - 1
            s0_idx = int(np.clip(s0_idx, 0, len(grid_times) - 1))
            s0 = float(surv_i[s0_idx])

            if s0 < 1e-12:
                result[i] = float(t0)
                continue

            cdf = np.clip(1.0 - valid_surv / s0, 0.0, 1.0)
            u = rng.uniform(0.0, 1.0, size=n_draws)
            draws = np.empty(n_draws)
            for k, u_k in enumerate(u):
                idx_k = int(np.searchsorted(cdf, u_k))
                if idx_k >= len(valid_times):
                    idx_k = len(valid_times) - 1
                draws[k] = float(valid_times[idx_k])

            result[i] = float(draws.mean())

        return result


class SksurvCoxCensoringAdapter:
    """
    Adapter wrapping a scikit-survival CoxPHSurvivalAnalysis or similar model.

    Converts the scikit-survival API (predict_survival_function returning
    StepFunction objects) to the CensoringModelProtocol interface.

    Parameters
    ----------
    model : fitted sksurv survival model
        Must have been fitted with the censoring indicator as the event column.

    Notes
    -----
    Requires scikit-survival>=0.22. Install with:
        pip install insurance-conformal[survival]
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_survival_at(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        Predict P[C > t_i | X=x_i] using the fitted scikit-survival model.

        Each StepFunction in the returned list is evaluated at the
        corresponding time.
        """
        step_fns = self.model.predict_survival_function(X)
        result = np.empty(len(times))
        for i, (sf, ti) in enumerate(zip(step_fns, times)):
            # StepFunction.__call__ extrapolates with last value by default
            t_clipped = float(min(float(ti), sf.domain[1]))
            result[i] = float(sf(t_clipped))
        return result

    def sample_truncated(
        self,
        X: np.ndarray,
        t_min: np.ndarray,
        n_draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample from the truncated scikit-survival censoring distribution via ITS.
        """
        step_fns = self.model.predict_survival_function(X)
        result = np.empty(len(t_min))

        for i, (sf, t0) in enumerate(zip(step_fns, t_min)):
            grid_times = sf.x.astype(float)
            surv_vals = sf.y.astype(float)

            mask = grid_times > float(t0)
            if not mask.any():
                result[i] = float(t0)
                continue

            valid_times = grid_times[mask]
            valid_surv = surv_vals[mask]

            idx0 = np.searchsorted(grid_times, float(t0), side="right") - 1
            idx0 = int(np.clip(idx0, 0, len(grid_times) - 1))
            s0 = float(surv_vals[idx0])

            if s0 < 1e-12:
                result[i] = float(t0)
                continue

            cdf = np.clip(1.0 - valid_surv / s0, 0.0, 1.0)
            u = rng.uniform(0.0, 1.0, size=n_draws)
            draws = np.empty(n_draws)
            for k, u_k in enumerate(u):
                idx_k = int(np.searchsorted(cdf, u_k))
                if idx_k >= len(valid_times):
                    idx_k = len(valid_times) - 1
                draws[k] = float(valid_times[idx_k])

            result[i] = float(draws.mean())

        return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ConformalisedSurvival:
    """
    Doubly robust conformalized survival analysis for right-censored data.

    Produces a lower prediction bound L_hat(x) such that:

        P[T >= L_hat(X)] >= 1 - alpha  (asymptotically)

    under conditional independence of T and C given X (T perp C | X). The
    double robustness guarantee means the asymptotic coverage holds as long
    as at least one of the survival model and the censoring model is correctly
    specified.

    The typical UK insurance use cases are:

    - **Protection/life pricing**: lower bound on time to death or critical
      illness, used to bound reserve adequacy.
    - **Lapse analysis**: lower bound on time to policy lapse, used to bound
      expected premium income.
    - **Liability development**: lower bound on time to claim closure.

    Parameters
    ----------
    survival_model : SurvivalModelProtocol
        A fitted model implementing predict_quantile(X, alpha) -> np.ndarray.
        Use the SksurvCoxCensoringAdapter or similar adapter if your model
        does not natively implement this interface.
    censoring_model : CensoringModelProtocol or KaplanMeierCensoringModel
        A fitted model for the conditional censoring distribution f_{C|X}.
        Must implement predict_survival_at(X, times) and sample_truncated().
        Use KaplanMeierCensoringModel for covariate-independent censoring.
        Use LifelinesCoxCensoringAdapter or SksurvCoxCensoringAdapter for
        covariate-dependent censoring (recommended for insurance data).
    method : str, default 'fixed_cutoff'
        Calibration algorithm to use. Currently supported:
        - 'fixed_cutoff': Algorithm 2 from Sesia & Svetnik (2024). Calibrates
          on all points where the imputed censoring time exceeds c_0.
    cutoff : float or None, default None
        The fixed cutoff c_0 for 'fixed_cutoff' method. Calibration points are
        restricted to those with imputed censoring time >= c_0. If None, uses
        the median observed time in the calibration set. Smaller c_0 retains
        more calibration points but includes points where the censoring model
        is less reliable.
    alpha : float, default 0.10
        Miscoverage level. alpha=0.10 gives a 90% lower prediction bound.
        In Solvency II contexts, you may want alpha=0.005 for a 99.5% bound.
    n_impute : int, default 10
        Number of Monte Carlo draws per event-observed calibration point in
        Algorithm 1. Higher values reduce imputation variance at the cost of
        more censoring model evaluations. 10 is usually sufficient; increase
        to 50 for high-precision applications.
    random_state : int or None, default None
        Seed for the Monte Carlo imputation RNG.

    Attributes
    ----------
    cal_scores_ : np.ndarray
        Conformal non-conformity scores for the filtered calibration set.
    ipcw_weights_ : np.ndarray
        IPCW weights (1 / P[C > c_0 | X=x]) for each filtered calibration
        point. Normalised to sum to the number of filtered points.
    calibration_quantile_ : float
        The weighted conformal quantile stored after calibrate().
    cutoff_ : float
        The resolved cutoff c_0 (median observed time if cutoff=None).
    n_calibration_ : int
        Total number of calibration points passed to calibrate().
    n_filtered_ : int
        Number of calibration points after the c_0 filter.
    is_calibrated_ : bool

    References
    ----------
    Sesia, M. & Svetnik, V. (2024). 'Doubly Robust Conformalized Survival
    Analysis with Right-Censored Data.' arXiv:2412.09729.
    """

    def __init__(
        self,
        survival_model: Any,
        censoring_model: Any,
        method: str = "fixed_cutoff",
        cutoff: Optional[float] = None,
        alpha: float = 0.10,
        n_impute: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        valid_methods = ("fixed_cutoff",)
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {valid_methods}. "
                "Note: 'adaptive_cutoff' (Algorithm 3) is not yet implemented."
            )

        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        if n_impute < 1:
            raise ValueError(f"n_impute must be >= 1, got {n_impute}")

        if cutoff is not None and cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")

        self.survival_model = survival_model
        self.censoring_model = censoring_model
        self.method = method
        self.cutoff = cutoff
        self.alpha = alpha
        self.n_impute = n_impute
        self.random_state = random_state

        # State set during calibration
        self.cal_scores_: Optional[np.ndarray] = None
        self.ipcw_weights_: Optional[np.ndarray] = None
        self.calibration_quantile_: Optional[float] = None
        self.cutoff_: Optional[float] = None
        self.n_calibration_: int = 0
        self.n_filtered_: int = 0
        self.is_calibrated_: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "ConformalisedSurvival has not been calibrated. "
                "Call .calibrate(X_cal, t_cal, e_cal) first."
            )

    def _impute_censoring_times(
        self,
        X_cal: np.ndarray,
        t_cal: np.ndarray,
        e_cal: np.ndarray,
    ) -> np.ndarray:
        """
        Algorithm 1: impute censoring times for all calibration points.

        For censored points (E_i=0): C_hat_i = T_tilde_i (the observed censored
        time is the censoring time by definition).

        For event-observed points (E_i=1): C_i is unobserved but we know
        C_i > T_i. We sample n_impute draws from the truncated censoring
        distribution f_{C|X}(c|x_i) / P[C > T_i | X=x_i] for c > T_i
        and take the mean. This Monte Carlo average reduces imputation variance.

        Parameters
        ----------
        X_cal : np.ndarray, shape (n, p)
        t_cal : np.ndarray, shape (n,)
            Observed times T_tilde_i = min(T_i, C_i).
        e_cal : np.ndarray, shape (n,)
            Event indicators. 1=event, 0=censored.

        Returns
        -------
        np.ndarray, shape (n,)
            Imputed censoring times C_hat_i.
        """
        rng = np.random.default_rng(self.random_state)
        n = len(t_cal)
        c_hat = np.empty(n)

        # Censored points: C_hat = T_tilde (the censoring time is directly observed)
        censored_mask = e_cal == 0
        c_hat[censored_mask] = t_cal[censored_mask]

        # Event-observed points: sample from truncated censoring distribution
        event_mask = e_cal == 1
        n_event = int(event_mask.sum())
        if n_event == 0:
            return c_hat

        X_event = X_cal[event_mask]
        t_event = t_cal[event_mask]

        c_hat[event_mask] = self.censoring_model.sample_truncated(
            X=X_event,
            t_min=t_event,
            n_draws=self.n_impute,
            rng=rng,
        )
        return c_hat

    def _ipcw_weights(
        self,
        X_cal: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """
        Compute IPCW weights: 1 / P[C > c_0 | X=x_i] for each calibration point.

        These weights correct for the fact that we have filtered to only those
        calibration points with C_hat_i >= c_0 — higher-censoring-probability
        points are upweighted to compensate for their lower inclusion probability.

        The weights are clipped at a maximum of 100 to prevent degenerate weight
        concentration from poorly-calibrated censoring models.

        Parameters
        ----------
        X_cal : np.ndarray, shape (n_filtered, p)
            Features for the filtered calibration points.
        cutoff : float
            The c_0 threshold.

        Returns
        -------
        np.ndarray, shape (n_filtered,)
            IPCW weights, clipped and raw (not normalised).
        """
        times = np.full(len(X_cal), cutoff)
        surv_prob = self.censoring_model.predict_survival_at(X_cal, times)

        # Clip survival probabilities away from zero to avoid exploding weights
        surv_prob_clipped = np.clip(surv_prob, 1e-2, 1.0)
        weights = 1.0 / surv_prob_clipped

        # Cap individual weights — prevents a single observation from dominating
        max_weight = 100.0
        n_capped = int((weights > max_weight).sum())
        if n_capped > 0:
            warnings.warn(
                f"{n_capped} IPCW weights exceeded {max_weight} and were capped. "
                "This suggests the censoring model assigns very low P[C > c_0] for "
                "some covariate regions. Check censoring model calibration or increase "
                "the cutoff c_0.",
                UserWarning,
                stacklevel=3,
            )
        weights = np.minimum(weights, max_weight)
        return weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        X_cal: Any,
        t_cal: Any,
        e_cal: Any,
    ) -> "ConformalisedSurvival":
        """
        Run the doubly robust conformal calibration procedure.

        Implements Algorithm 2 (fixed-cutoff) from Sesia & Svetnik (2024):

        1. Run Algorithm 1 to impute censoring times C_hat_i for all calibration
           points (observed for censored points, sampled for event points).
        2. Resolve the cutoff c_0 (median observed time if not specified).
        3. Filter to {i : C_hat_i >= c_0} — only points whose censoring time
           exceeds the cutoff are informative at the LPB level c_0.
        4. Compute IPCW weights 1/P[C > c_0 | X=x_i] for filtered points.
        5. Compute nonconformity scores s_i = q_hat_alpha(x_i) - t_i, where
           q_hat_alpha(x_i) is the survival model's alpha-quantile prediction.
        6. Compute the IPCW-weighted conformal quantile of the scores.
        7. Store the calibration quantile for use in predict_lower_bound().

        Parameters
        ----------
        X_cal : array-like of shape (n, p)
            Calibration features.
        t_cal : array-like of shape (n,)
            Observed times T_tilde_i = min(T_i, C_i).
        e_cal : array-like of shape (n,)
            Event indicators. 1=event observed, 0=censored.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the filtered calibration set is empty (all imputed censoring
            times fall below c_0).
        """
        X_cal_np = np.asarray(as_numpy(X_cal), dtype=float)
        t_cal_np = as_numpy(t_cal).astype(float).ravel()
        e_cal_np = as_numpy(e_cal).astype(float).ravel()

        n = len(X_cal_np)
        if len(t_cal_np) != n or len(e_cal_np) != n:
            raise ValueError(
                f"X_cal, t_cal, e_cal must have the same length. "
                f"Got {len(X_cal_np)}, {len(t_cal_np)}, {len(e_cal_np)}."
            )

        if not np.all((e_cal_np == 0) | (e_cal_np == 1)):
            raise ValueError(
                "e_cal must contain only 0 (censored) and 1 (event observed)."
            )

        self.n_calibration_ = n

        # Step 1: Impute censoring times
        c_hat = self._impute_censoring_times(X_cal_np, t_cal_np, e_cal_np)

        # Step 2: Resolve cutoff
        if self.cutoff is not None:
            cutoff = float(self.cutoff)
        else:
            cutoff = float(np.median(t_cal_np))
        self.cutoff_ = cutoff

        # Step 3: Filter to points where imputed censoring time >= c_0
        filter_mask = c_hat >= cutoff
        n_filtered = int(filter_mask.sum())

        if n_filtered == 0:
            raise ValueError(
                f"No calibration points remain after filtering to C_hat >= {cutoff:.4f}. "
                "All imputed censoring times are below the cutoff. "
                "Try a smaller cutoff or providing more calibration data."
            )

        if n_filtered < 20:
            warnings.warn(
                f"Only {n_filtered} calibration points remain after the c_0 filter. "
                f"Coverage guarantees require a larger filtered set (>=50 recommended). "
                "Consider reducing the cutoff value.",
                UserWarning,
                stacklevel=2,
            )

        self.n_filtered_ = n_filtered
        X_filt = X_cal_np[filter_mask]
        t_filt = t_cal_np[filter_mask]

        # Step 4: IPCW weights
        weights = self._ipcw_weights(X_filt, cutoff)
        self.ipcw_weights_ = weights

        # Step 5: Nonconformity scores
        # s_i = q_hat_alpha(x_i) - t_i
        # Positive score means the model predicted a quantile ABOVE the observed time
        # (good — the LPB is below the event). Negative means the model under-predicted.
        q_preds = self.survival_model.predict_quantile(X_filt, self.alpha)
        scores = q_preds - t_filt
        self.cal_scores_ = scores

        # Step 6: IPCW-weighted conformal quantile
        self.calibration_quantile_ = ipcw_weighted_quantile(
            scores=scores,
            weights=weights,
            alpha=self.alpha,
        )

        self.is_calibrated_ = True
        return self

    def predict_lower_bound(
        self,
        X_test: Any,
        alpha: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Produce lower prediction bounds for new observations.

        The lower prediction bound is:

            L_hat(x) = q_hat_alpha(x) - Q_hat

        where q_hat_alpha(x) is the survival model's alpha-quantile prediction
        and Q_hat is the weighted conformal correction from calibration. The
        bound is clipped at zero.

        Parameters
        ----------
        X_test : array-like of shape (n_test, p)
            Test features.
        alpha : float or None, default None
            Miscoverage rate. If None, uses the alpha from the constructor.
            Note: changing alpha at prediction time re-uses the existing
            calibration quantile (which was computed at the constructor alpha).
            For a different alpha, re-calibrate.

        Returns
        -------
        pl.DataFrame
            Columns:
            - lower_bound (Float64): the conformalized LPB, clipped at 0.
            - q_alpha_model (Float64): the raw survival model quantile prediction
              before the conformal correction.

        Raises
        ------
        RuntimeError
            If calibrate() has not been called.
        """
        self._check_calibrated()

        if alpha is not None and alpha != self.alpha:
            warnings.warn(
                f"alpha={alpha} passed to predict_lower_bound differs from the "
                f"calibration alpha={self.alpha}. The stored calibration_quantile_ "
                f"was computed at alpha={self.alpha}. Re-run calibrate() with the "
                f"desired alpha for correct coverage.",
                UserWarning,
                stacklevel=2,
            )

        X_test_np = np.asarray(as_numpy(X_test), dtype=float)
        eff_alpha = alpha if alpha is not None else self.alpha

        # Raw model quantile predictions
        q_model = self.survival_model.predict_quantile(X_test_np, eff_alpha)

        # Apply conformal correction: subtract the calibration quantile
        lower_bound = q_model - self.calibration_quantile_
        lower_bound = np.maximum(lower_bound, 0.0)

        return pl.DataFrame(
            {
                "lower_bound": lower_bound,
                "q_alpha_model": q_model,
            }
        )

    def coverage_diagnostics(
        self,
        X_test: Any,
        t_test: Any,
        e_test: Any,
        alpha: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Compute empirical coverage against uncensored (event-observed) test points.

        Coverage is evaluated only on points with e_test=1 (event observed) because
        for censored points we do not know the true event time T — the lower bound
        may or may not be valid. This is the standard diagnostic for survival
        conformal methods.

        Parameters
        ----------
        X_test : array-like of shape (n, p)
        t_test : array-like of shape (n,)
            Observed times (T_tilde = min(T, C)).
        e_test : array-like of shape (n,)
            Event indicators. 1=event, 0=censored.
        alpha : float or None
            Miscoverage rate. None uses the constructor alpha.

        Returns
        -------
        pl.DataFrame
            Columns:
            - n_total (Int32): total test points.
            - n_uncensored (Int32): uncensored test points (e==1).
            - empirical_coverage (Float64): fraction of uncensored points where
              T >= L_hat(X) (i.e. the LPB is at or below the event time).
            - target_coverage (Float64): 1 - alpha.
            - coverage_gap (Float64): target_coverage - empirical_coverage.
              Negative means overcoverage (conservative); positive means
              undercoverage (bad).

        Raises
        ------
        RuntimeError
            If calibrate() has not been called.
        """
        self._check_calibrated()

        t_test_np = as_numpy(t_test).astype(float).ravel()
        e_test_np = as_numpy(e_test).astype(float).ravel()

        intervals = self.predict_lower_bound(X_test, alpha=alpha)
        lower = intervals["lower_bound"].to_numpy()

        uncensored = e_test_np == 1
        n_total = len(t_test_np)
        n_uncensored = int(uncensored.sum())

        if n_uncensored == 0:
            warnings.warn(
                "No uncensored test observations. Coverage cannot be evaluated.",
                UserWarning,
                stacklevel=2,
            )
            coverage = float("nan")
        else:
            # Coverage: LPB is valid when T >= L_hat, i.e. lower <= T
            covered = lower[uncensored] <= t_test_np[uncensored]
            coverage = float(covered.mean())

        eff_alpha = alpha if alpha is not None else self.alpha
        target = 1.0 - eff_alpha

        return pl.DataFrame(
            {
                "n_total": [n_total],
                "n_uncensored": [n_uncensored],
                "empirical_coverage": [coverage],
                "target_coverage": [target],
                "coverage_gap": [target - coverage],
            }
        )

    def calibration_summary(self) -> pl.DataFrame:
        """
        Summary of the calibration state.

        Returns
        -------
        pl.DataFrame
            Columns: statistic (String), value (Float64).
        """
        self._check_calibrated()

        weights = self.ipcw_weights_
        ess = float(weights.sum() ** 2) / float((weights**2).sum() + 1e-12)

        rows = [
            ("n_calibration", float(self.n_calibration_)),
            ("n_filtered", float(self.n_filtered_)),
            ("cutoff_c0", float(self.cutoff_)),
            ("alpha", float(self.alpha)),
            ("calibration_quantile", float(self.calibration_quantile_)),
            ("mean_ipcw_weight", float(np.mean(weights))),
            ("max_ipcw_weight", float(np.max(weights))),
            ("min_ipcw_weight", float(np.min(weights))),
            ("effective_n", float(ess)),
            ("coverage_shrinkage", float(ess / self.n_filtered_)),
        ]

        return pl.DataFrame(
            {
                "statistic": [r[0] for r in rows],
                "value": [r[1] for r in rows],
            }
        )

    def __repr__(self) -> str:
        if self.is_calibrated_:
            status = (
                f"calibrated on {self.n_filtered_} filtered points "
                f"(of {self.n_calibration_} cal, cutoff={self.cutoff_:.4f})"
            )
        else:
            status = "not calibrated"
        return (
            f"ConformalisedSurvival("
            f"method='{self.method}', "
            f"alpha={self.alpha}, "
            f"n_impute={self.n_impute}, "
            f"{status})"
        )
