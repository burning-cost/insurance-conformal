"""
Solvency II / UK Solvency UK SCR reporting from any conformal predictor.

Under Solvency II Article 101 (retained as UK Solvency UK via PRA PS9/24),
the Solvency Capital Requirement is calibrated to a VaR at 99.5% confidence
over a one-year time horizon.

Conformal prediction provides a finite-sample valid 99.5% upper bound on claims
without requiring a distributional assumption. This module wraps any conformal
predictor in an SCR-flavoured reporting interface.

References
----------
Hong (2025) arXiv:2503.03659, Section 5 and Table 7.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


class SCRReport:
    """Solvency II / UK Solvency UK SCR reporting wrapper.

    Accepts any object with a ``predict_interval(X, alpha)`` method returning
    an (n, 2) array of [lower, upper] prediction bounds.

    Parameters
    ----------
    predictor : object
        Any fitted predictor implementing predict_interval(X, alpha) -> np.ndarray.
        Typical: HongConformal, HongTransformConformal, TwoStageLWConformal.

    Examples
    --------
    >>> report = SCRReport(hong_conformal)
    >>> scr = report.solvency_capital_requirement(X_portfolio, alpha=0.005)
    >>> print(f"SCR upper bound: £{scr:,.0f}")
    """

    def __init__(self, predictor: Any) -> None:
        if not hasattr(predictor, "predict_interval"):
            raise TypeError(
                "predictor must implement predict_interval(X, alpha) -> np.ndarray"
            )
        self.predictor = predictor
        self._last_coverage_table: pd.DataFrame | None = None

    def solvency_capital_requirement(
        self, X: np.ndarray, alpha: float = 0.005
    ) -> float:
        """Compute the SCR upper bound at the 99.5% confidence level.

        Returns the maximum upper prediction bound across all risks in the
        portfolio. For Solvency II compliance, this is the aggregate claim
        amount that will not be exceeded with probability 1-alpha.

        Parameters
        ----------
        X : np.ndarray, shape (n_risks, p)
            Portfolio covariate matrix. Each row represents one risk (policy
            or claim).
        alpha : float
            Miscoverage rate. Default: 0.005 (99.5% coverage, Solvency II).

        Returns
        -------
        scr : float
            Maximum upper prediction bound across all portfolio risks.

        Notes
        -----
        Coverage guarantee: P(Y_i <= SCR for all i) >= 1-alpha marginally,
        not jointly. For joint SCR, see the portfolio-level aggregation methods
        in insurance-multivariate-conformal.

        This is the portfolio-level maximum. For reserving purposes, the sum
        of upper bounds across policies gives a conservative aggregate reserve.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        intervals = self.predictor.predict_interval(X, alpha=alpha)
        return float(np.max(intervals[:, 1]))

    def aggregate_reserve(
        self, X: np.ndarray, alpha: float = 0.005
    ) -> float:
        """Sum of upper prediction bounds across all risks (conservative reserve).

        Parameters
        ----------
        X : np.ndarray, shape (n_risks, p)
        alpha : float

        Returns
        -------
        aggregate : float
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        intervals = self.predictor.predict_interval(X, alpha=alpha)
        return float(np.sum(intervals[:, 1]))

    def coverage_table(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alphas: list[float] | None = None,
    ) -> pd.DataFrame:
        """Compute empirical vs nominal coverage at multiple alpha levels.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, p)
        y_test : np.ndarray, shape (n_test,)
        alphas : list of float, optional
            Default: [0.005, 0.01, 0.05, 0.10, 0.20].

        Returns
        -------
        pd.DataFrame with columns:
            alpha, nominal_coverage, empirical_coverage, mean_width,
            mean_lower, mean_upper, n_test
        """
        if alphas is None:
            alphas = [0.005, 0.01, 0.05, 0.10, 0.20]

        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        rows = []
        for a in alphas:
            intervals = self.predictor.predict_interval(X_test, alpha=a)
            covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
            widths = intervals[:, 1] - intervals[:, 0]
            rows.append(
                {
                    "alpha": a,
                    "nominal_coverage": 1.0 - a,
                    "empirical_coverage": float(covered.mean()),
                    "mean_width": float(widths.mean()),
                    "mean_lower": float(intervals[:, 0].mean()),
                    "mean_upper": float(intervals[:, 1].mean()),
                    "n_test": len(y_test),
                }
            )

        df = pd.DataFrame(rows)
        self._last_coverage_table = df
        return df

    def to_html(
        self,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        alphas: list[float] | None = None,
        title: str = "Solvency Capital Requirement — Conformal Prediction Report",
    ) -> str:
        """Render an HTML coverage report.

        Parameters
        ----------
        X_test : np.ndarray, optional
            If provided, recomputes coverage table. Otherwise uses cached table.
        y_test : np.ndarray, optional
        alphas : list of float, optional
        title : str

        Returns
        -------
        html : str
        """
        if X_test is not None and y_test is not None:
            df = self.coverage_table(X_test, y_test, alphas)
        elif self._last_coverage_table is not None:
            df = self._last_coverage_table
        else:
            raise ValueError(
                "Provide X_test and y_test, or call coverage_table() first."
            )

        predictor_name = type(self.predictor).__name__

        def fmt_pct(v: float) -> str:
            return f"{v * 100:.1f}%"

        def fmt_num(v: float) -> str:
            return f"{v:,.2f}"

        rows_html = ""
        for _, row in df.iterrows():
            nom = fmt_pct(row["nominal_coverage"])
            emp = fmt_pct(row["empirical_coverage"])
            gap = row["empirical_coverage"] - row["nominal_coverage"]
            gap_str = f"+{gap*100:.1f}pp" if gap >= 0 else f"{gap*100:.1f}pp"
            colour = "#2d6a4f" if gap >= 0 else "#d62828"
            scr_flag = " &#9733;" if abs(row["alpha"] - 0.005) < 1e-9 else ""
            rows_html += (
                f"<tr>"
                f"<td>{row['alpha']:.3f}{scr_flag}</td>"
                f"<td>{nom}</td>"
                f"<td style='color:{colour};font-weight:bold'>{emp}</td>"
                f"<td style='color:{colour}'>{gap_str}</td>"
                f"<td>{fmt_num(row['mean_width'])}</td>"
                f"<td>{fmt_num(row['mean_upper'])}</td>"
                f"<td>{int(row['n_test'])}</td>"
                f"</tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #222; }}
  h1 {{ font-size: 1.4em; color: #1a1a2e; }}
  p.meta {{ color: #666; font-size: 0.9em; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
  th {{ background: #1a1a2e; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #e0e0e0; }}
  tr:hover {{ background: #f5f5f5; }}
  .note {{ margin-top: 20px; font-size: 0.85em; color: #555; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Predictor: <strong>{predictor_name}</strong> &nbsp;|&nbsp;
Observations: <strong>{int(df['n_test'].iloc[0])}</strong></p>
<table>
<thead>
<tr>
  <th>Alpha (&#9733; = SCR)</th>
  <th>Nominal Coverage</th>
  <th>Empirical Coverage</th>
  <th>Coverage Gap</th>
  <th>Mean Width</th>
  <th>Mean Upper Bound</th>
  <th>N Test</th>
</tr>
</thead>
<tbody>
{rows_html}</tbody>
</table>
<p class="note">Coverage guarantee (Theorem 1, Hong 2025 arXiv:2503.03659):
Empirical coverage &ge; Nominal coverage for all n and all exchangeable
distributions. &#9733; Alpha = 0.005 corresponds to Solvency II Article 101
(99.5% VaR, one-year horizon). UK Solvency UK maintained under PRA PS9/24.</p>
</body>
</html>"""
        return html

    def to_json(
        self,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        alphas: list[float] | None = None,
    ) -> str:
        """Render a JSON coverage report.

        Parameters
        ----------
        X_test : np.ndarray, optional
        y_test : np.ndarray, optional
        alphas : list of float, optional

        Returns
        -------
        json_str : str
        """
        if X_test is not None and y_test is not None:
            df = self.coverage_table(X_test, y_test, alphas)
        elif self._last_coverage_table is not None:
            df = self._last_coverage_table
        else:
            raise ValueError(
                "Provide X_test and y_test, or call coverage_table() first."
            )

        payload = {
            "predictor": type(self.predictor).__name__,
            "n_test": int(df["n_test"].iloc[0]),
            "coverage_table": df.to_dict(orient="records"),
            "methodology": {
                "reference": "Hong (2025) arXiv:2503.03659",
                "framework": "Solvency II Article 101 / UK Solvency UK PRA PS9/24",
                "scr_alpha": 0.005,
                "coverage_type": "marginal",
            },
        }
        return json.dumps(payload, indent=2, default=float)

    def __repr__(self) -> str:
        predictor_name = type(self.predictor).__name__
        return f"SCRReport(predictor={predictor_name})"
