"""
Solvency Capital Requirement (SCR) reporting with conformal coverage guarantees.

Solvency II requires insurers to hold capital sufficient to cover a 1-in-200
year loss event — the 99.5th percentile of the loss distribution. Standard
SCR calculations use parametric models (lognormal, Pareto) for the tail, which
introduces model risk.

Conformal prediction provides a distribution-free alternative: the upper bound
of a conformal interval at alpha=0.005 is a valid 99.5% prediction upper bound
for any individual risk, with finite-sample coverage guarantees. This doesn't
replace actuarial capital modelling but provides a statistically rigorous
sanity check on individual risk SCR components.

The SCRReport class wraps any fitted+calibrated conformal predictor and adds:
1. Per-risk SCR upper bounds at alpha=0.005
2. Multi-alpha coverage validation tables for regulatory submission
3. Markdown report rendering

Note on conservatism: conformal prediction gives a MARGINAL coverage guarantee
— at least (1-alpha) of individual predictions are covered, on average. For
regulatory purposes, a margin for conservatism should be applied: consider
using alpha=0.003 or alpha=0.001 to have additional buffer. The
solvency_capital_requirement() method accepts any alpha.

Regulatory context: EIOPA has not issued formal guidance on conformal methods
as of 2025. Present these outputs as supplementary technical validation, not
as primary SCR figures, until regulatory acceptance is established.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import polars as pl

from insurance_conformal.utils import as_numpy


class SCRReport:
    """
    Solvency II SCR upper bounds with conformal coverage guarantees.

    Wraps any conformal predictor that implements predict_interval(X, alpha)
    and returns a Polars DataFrame with lower/point/upper columns.

    Parameters
    ----------
    predictor : conformal predictor
        Any object with a predict_interval(X, alpha) method returning a
        pl.DataFrame with at least "upper" and "point" columns. Works with
        InsuranceConformalPredictor, LocallyWeightedConformal, or
        HongTransformConformal.
    policy_ids : array-like, optional
        Policy identifiers to include in output tables. If None, integer
        indices are used.

    Examples
    --------
    >>> from insurance_conformal import InsuranceConformalPredictor
    >>> from insurance_conformal.scr import SCRReport
    >>>
    >>> cp = InsuranceConformalPredictor(model=fitted_model, tweedie_power=1.5)
    >>> cp.calibrate(X_cal, y_cal)
    >>>
    >>> scr = SCRReport(predictor=cp)
    >>> scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
    >>> val_table = scr.coverage_validation_table(X_test, y_test)
    >>> print(scr.to_markdown())
    """

    def __init__(
        self,
        predictor: Any,
        policy_ids: Optional[Any] = None,
    ) -> None:
        if not hasattr(predictor, "predict_interval"):
            raise TypeError(
                "predictor must implement predict_interval(X, alpha). "
                f"Got: {type(predictor)}"
            )
        self.predictor = predictor
        self._policy_ids = policy_ids
        self._last_scr_result: Optional[pl.DataFrame] = None
        self._last_coverage_table: Optional[pl.DataFrame] = None

    @classmethod
    def from_conformal(
        cls, predictor: Any, policy_ids: Optional[Any] = None
    ) -> "SCRReport":
        """
        Construct an SCRReport from any conformal predictor.

        Parameters
        ----------
        predictor : conformal predictor
            Fitted and calibrated conformal predictor.
        policy_ids : array-like, optional
            Policy identifiers.

        Returns
        -------
        SCRReport
        """
        return cls(predictor=predictor, policy_ids=policy_ids)

    def solvency_capital_requirement(
        self,
        X: Any,
        alpha: float = 0.005,
    ) -> pl.DataFrame:
        """
        Compute 99.5% (or custom alpha) SCR upper bounds per risk.

        The upper bound from predict_interval(X, alpha=0.005) is a valid
        (1-alpha) = 99.5% prediction upper bound for each risk, guaranteed
        with finite-sample coverage by split conformal theory.

        The SCR component for a risk is max(0, upper_bound - expected_loss),
        i.e., the excess of the tail bound over the expected value. This is
        conservative relative to a parametric VaR estimate.

        Parameters
        ----------
        X : array-like
            Risk features.
        alpha : float, default 0.005
            Miscoverage rate. Default 0.005 gives 99.5% upper bounds (Solvency II).
            Use alpha=0.01 for 99% or alpha=0.001 for 99.9%.

        Returns
        -------
        pl.DataFrame
            Columns: policy_id (if provided), expected_loss, upper_bound_99_5,
            scr_component, coverage_level.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        intervals = self.predictor.predict_interval(X, alpha=alpha)

        expected = intervals["point"].to_numpy()
        upper = intervals["upper"].to_numpy()
        scr = np.maximum(0.0, upper - expected)
        n = len(expected)

        col_name = f"upper_bound_{int((1 - alpha) * 1000) / 10:.1f}pct"

        data: dict = {
            "expected_loss": expected,
            col_name: upper,
            "scr_component": scr,
            "coverage_level": np.full(n, 1.0 - alpha),
        }

        if self._policy_ids is not None:
            ids = as_numpy(self._policy_ids)
            if len(ids) != n:
                raise ValueError(
                    f"policy_ids length {len(ids)} does not match X length {n}"
                )
            data = {"policy_id": ids, **data}

        result = pl.DataFrame(data)
        self._last_scr_result = result
        return result

    def coverage_validation_table(
        self,
        X_test: Any,
        y_test: Any,
        alphas: List[float] = None,
    ) -> pl.DataFrame:
        """
        Multi-alpha coverage table for regulatory submission.

        Tests whether the conformal predictor achieves its stated coverage
        at multiple alpha levels. This is the empirical validation a regulator
        would want to see: that the method's coverage claims are borne out
        on held-out data.

        Parameters
        ----------
        X_test : array-like
            Test features (held-out, not used for calibration or training).
        y_test : array-like
            Observed test outcomes.
        alphas : list of float, optional
            Alpha levels to test. Defaults to [0.005, 0.01, 0.05, 0.10, 0.20].

        Returns
        -------
        pl.DataFrame
            Columns: alpha, target_coverage, empirical_coverage, n_covered,
            n_total, coverage_shortfall, meets_requirement.
            Rows ordered by alpha descending (most stringent first).
        """
        if alphas is None:
            alphas = [0.005, 0.01, 0.05, 0.10, 0.20]

        y = as_numpy(y_test).ravel()
        n_total = len(y)
        rows = []

        for alpha in sorted(alphas):
            intervals = self.predictor.predict_interval(X_test, alpha=alpha)
            lower = intervals["lower"].to_numpy()
            upper = intervals["upper"].to_numpy()
            covered = (y >= lower) & (y <= upper)
            n_covered = int(covered.sum())
            empirical = float(covered.mean())
            target = 1.0 - alpha
            shortfall = max(0.0, target - empirical)

            rows.append(
                {
                    "alpha": alpha,
                    "target_coverage": target,
                    "empirical_coverage": empirical,
                    "n_covered": n_covered,
                    "n_total": n_total,
                    "coverage_shortfall": shortfall,
                    "meets_requirement": bool(empirical >= target - 0.02),
                }
            )

        # Most stringent first (smallest alpha = highest coverage requirement)
        table = pl.DataFrame(rows).sort("alpha")
        self._last_coverage_table = table
        return table

    def to_markdown(self) -> str:
        """
        Render coverage validation results as a markdown report.

        Returns the most recently computed coverage validation table in
        a markdown format suitable for regulatory submissions, model
        governance documentation, or Confluence pages.

        Returns
        -------
        str
            Markdown-formatted report string.

        Raises
        ------
        RuntimeError
            If coverage_validation_table() has not been called yet.
        """
        if self._last_coverage_table is None:
            raise RuntimeError(
                "Call coverage_validation_table(X_test, y_test) before to_markdown()."
            )

        table = self._last_coverage_table
        lines = [
            "## SCR Coverage Validation Report",
            "",
            "Coverage validation for conformal prediction intervals at multiple ",
            "alpha levels. The 'Meets Requirement' column uses a 2pp tolerance ",
            "for finite-sample variation.",
            "",
            "| Alpha | Target | Empirical | N Covered | N Total | Shortfall | Meets Req |",
            "|-------|--------|-----------|-----------|---------|-----------|-----------|",
        ]

        for row in table.iter_rows(named=True):
            alpha_str = f"{row['alpha']:.3f}"
            target_str = f"{row['target_coverage']:.1%}"
            empirical_str = f"{row['empirical_coverage']:.3%}"
            shortfall_str = f"{row['coverage_shortfall']:.3%}"
            meets_str = "Yes" if row["meets_requirement"] else "**No**"

            lines.append(
                f"| {alpha_str} | {target_str} | {empirical_str} | "
                f"{row['n_covered']} | {row['n_total']} | "
                f"{shortfall_str} | {meets_str} |"
            )

        lines.extend(
            [
                "",
                "> **Note**: Conformal coverage is a marginal guarantee. "
                "Coverage by risk subgroup should be validated separately using "
                "`CoverageDiagnostics.coverage_by_decile()`.",
                "",
                "> These outputs are supplementary technical validation. "
                "Consult your actuarial team before using conformal bounds "
                "as primary SCR inputs.",
            ]
        )

        return "\n".join(lines)

    def aggregate_scr(self) -> dict:
        """
        Aggregate SCR components to portfolio level.

        Returns a dict with total expected loss, total SCR (sum of components),
        and SCR ratio (SCR / expected loss). Call after solvency_capital_requirement().

        Returns
        -------
        dict
            Keys: total_expected_loss, total_scr, scr_ratio, n_risks, alpha.
        """
        if self._last_scr_result is None:
            raise RuntimeError(
                "Call solvency_capital_requirement(X, alpha) first."
            )

        df = self._last_scr_result
        total_expected = float(df["expected_loss"].sum())
        total_scr = float(df["scr_component"].sum())
        scr_ratio = total_scr / total_expected if total_expected > 0 else float("nan")
        alpha = 1.0 - float(df["coverage_level"][0])

        return {
            "total_expected_loss": total_expected,
            "total_scr": total_scr,
            "scr_ratio": scr_ratio,
            "n_risks": len(df),
            "alpha": alpha,
        }

    def __repr__(self) -> str:
        return f"SCRReport(predictor={self.predictor!r})"
