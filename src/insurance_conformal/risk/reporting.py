"""
Reporting utilities: Solvency II / PRA-format output for risk control results.

Actuaries and risk managers need outputs in formats that map to regulatory
reporting requirements, not just raw numbers. This module translates
conformal risk control results into structures that fit the Solvency II SCR
narrative and PRA SS1/23 risk appetite reporting.

The key framing: conformal risk control provides distribution-free bounds on
expected financial shortfall. In Solvency II terms, this is a contribution to
the Underwriting Risk module — specifically, the model error component of the
premium risk charge.

We do not claim CRC replaces the full SCR calculation (it doesn't — that
requires 99.5% VaR, not expected shortfall). But CRC provides a rigorous
bound on the expected model error component that feeds into premium risk.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl


def premium_sufficiency_report(
    lambda_hat: float,
    alpha: float,
    n_calibration: int,
    B: float,
    corrected_risk: float,
    portfolio_gwp: Optional[float] = None,
) -> pl.DataFrame:
    """
    Format PremiumSufficiencyController results as a regulatory-style table.

    Parameters
    ----------
    lambda_hat : float
        Calibrated loading factor.
    alpha : float
        Target expected shortfall level.
    n_calibration : int
        Calibration set size.
    B : float
        Loss bound used in calibration.
    corrected_risk : float
        Corrected empirical risk at lambda_hat.
    portfolio_gwp : float or None
        If provided, computes expected monetary shortfall = alpha * portfolio_gwp.

    Returns
    -------
    pl.DataFrame
        Single-row summary with named metrics.
    """
    row = {
        "metric": [
            "Calibrated loading factor (lambda*)",
            "Target expected shortfall (alpha)",
            "Achieved corrected risk",
            "Calibration set size (n)",
            "Loss bound (B)",
            "Finite-sample correction (B/(n+1))",
        ],
        "value": [
            f"{lambda_hat:.4f}",
            f"{alpha:.4f} ({alpha:.1%})",
            f"{corrected_risk:.4f} ({corrected_risk:.1%})",
            str(n_calibration),
            f"{B:.4f}",
            f"{B / (n_calibration + 1):.6f}",
        ],
        "interpretation": [
            "Upper bound = lambda* x model premium",
            "Max expected shortfall as % of premium income",
            "Empirical risk after finite-sample correction",
            "Observations used to calibrate lambda*",
            "Maximum possible loss per policy",
            "Inflation for unseen test point (smaller = better)",
        ],
    }

    if portfolio_gwp is not None:
        row["metric"].append("Expected monetary shortfall (alpha x GWP)")
        row["value"].append(f"£{alpha * portfolio_gwp:,.0f}")
        row["interpretation"].append("Upper bound on expected unrecovered loss")

    return pl.DataFrame(row)


def solvency_ii_model_error_note(
    alpha: float,
    lambda_hat: float,
    n_calibration: int,
    delta: float = 0.05,
) -> str:
    """
    Generate a brief narrative suitable for Solvency II ORSA / SFCR model error section.

    Parameters
    ----------
    alpha : float
        Expected shortfall target.
    lambda_hat : float
        Calibrated loading factor.
    n_calibration : int
        Calibration observations.
    delta : float, default 0.05
        Not directly used in text but included for completeness.

    Returns
    -------
    str
        Plain-text narrative paragraph.
    """
    return (
        f"Model error in the pricing GBM is quantified using conformal risk control "
        f"(Angelopoulos et al., 2024, ICLR). The method provides a distribution-free "
        f"guarantee: the expected shortfall from underpriced policies — claims exceeding "
        f"the risk-controlled upper bound — is at most {alpha:.1%} of expected premium "
        f"income. This guarantee is finite-sample valid under exchangeability of the "
        f"calibration and test distributions; no parametric assumptions are required. "
        f"Calibration used {n_calibration:,} held-out policies. The calibrated loading "
        f"factor is {lambda_hat:.3f}x the model pure premium. In Solvency II terms, "
        f"this bounds the model error component of the premium risk charge under the "
        f"standard formula (Article 105, Directive 2009/138/EC)."
    )


def risk_curve_dataframe(
    lambdas: np.ndarray,
    corrected_risk: np.ndarray,
    alpha: float,
    lambda_hat: float,
) -> pl.DataFrame:
    """
    Format the risk curve for plotting or export.

    Parameters
    ----------
    lambdas : np.ndarray, shape (m,)
    corrected_risk : np.ndarray, shape (m,)
    alpha : float
    lambda_hat : float

    Returns
    -------
    pl.DataFrame
        Columns: lambda, corrected_risk, below_target, is_lambda_hat.
    """
    return pl.DataFrame(
        {
            "lambda": lambdas,
            "corrected_risk": corrected_risk,
            "below_target": corrected_risk <= alpha,
            "is_lambda_hat": np.abs(lambdas - lambda_hat) < 1e-10,
        }
    )
