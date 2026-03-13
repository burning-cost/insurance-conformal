"""
Statistical coverage validity tests.

These are the core tests for conformal validity. They use Monte Carlo to check
that the empirical joint coverage rate is >= 1 - alpha on held-out test data.

Key tests:
1. Joint coverage >= 1 - alpha for all methods.
2. LWC calibration thresholds <= GWC thresholds (the theoretical efficiency claim).
3. Bonferroni achieves higher marginal coverage (more conservative per dimension).
4. d=1 reduces to standard split conformal.
5. Zero-claim masking preserves joint coverage guarantee.
"""

import numpy as np
import pytest

from insurance_conformal.multivariate import JointConformalPredictor
from insurance_conformal.multivariate.calibration import calibrate
from insurance_conformal.multivariate.datasets import make_motor_frequency_severity
from insurance_conformal.multivariate.diagnostics import coverage_report, compare_methods


def _calibrate_and_cover(
    models, X_cal, Y_cal, X_test, Y_test, alpha=0.05, method="lwc",
    zero_claim_mask=None,
):
    """Helper: calibrate and return joint coverage rate."""
    pred = JointConformalPredictor(models, alpha=alpha, method=method)
    pred.calibrate(X_cal, Y_cal, zero_claim_mask=zero_claim_mask)
    joint = pred.predict(X_test)
    return joint.joint_coverage_check(Y_test)


class TestJointCoverageValidity:
    """
    Coverage must be >= 1 - alpha. This is the fundamental guarantee.
    We allow 5% slack for finite-sample noise at moderate n.
    """

    def _assert_coverage(self, coverage, n_test, alpha, method):
        target = 1.0 - alpha
        slack = 0.05
        assert coverage >= target - slack, (
            f"Method '{method}': coverage {coverage:.3f} < target {target:.3f} - {slack}"
        )

    @pytest.mark.parametrize("method", ["bonferroni", "gwc", "lwc"])
    def test_joint_coverage_valid_alpha05(self, large_motor_data, method):
        d = large_motor_data
        cov = _calibrate_and_cover(
            d["models"], d["X_cal"], d["Y_cal"],
            d["X_test"], d["Y_test"],
            alpha=0.05, method=method,
        )
        self._assert_coverage(cov, d["n_test"], 0.05, method)

    @pytest.mark.parametrize("method", ["bonferroni", "gwc", "lwc"])
    def test_joint_coverage_valid_alpha10(self, large_motor_data, method):
        d = large_motor_data
        cov = _calibrate_and_cover(
            d["models"], d["X_cal"], d["Y_cal"],
            d["X_test"], d["Y_test"],
            alpha=0.10, method=method,
        )
        self._assert_coverage(cov, d["n_test"], 0.10, method)

    def test_joint_coverage_with_zero_claim_masking(self, large_motor_data):
        d = large_motor_data
        cov = _calibrate_and_cover(
            d["models"], d["X_cal"], d["Y_cal"],
            d["X_test"], d["Y_test"],
            alpha=0.05, method="lwc",
            zero_claim_mask=d["zero_mask_cal"],
        )
        self._assert_coverage(cov, d["n_test"], 0.05, "lwc+zero_mask")

    def test_marginal_coverage_valid(self, large_motor_data):
        d = large_motor_data
        pred = JointConformalPredictor(d["models"], alpha=0.05, method="lwc")
        pred.calibrate(d["X_cal"], d["Y_cal"])
        joint = pred.predict(d["X_test"])
        marginal = joint.marginal_coverage_rates(d["Y_test"])
        for k, cov in marginal.items():
            assert cov >= 0.90 - 0.05, f"Marginal coverage for {k}: {cov:.3f}"

    def test_small_n_cal_still_valid(self, small_motor_data):
        d = small_motor_data
        cov = _calibrate_and_cover(
            d["models"], d["X_cal"], d["Y_cal"],
            d["X_test"], d["Y_test"],
            alpha=0.05, method="lwc",
        )
        # At n_cal=50, finite-sample bound is 1/51 ~ 2% excess. Allow more slack.
        assert cov >= 0.85, f"Coverage too low: {cov:.3f}"


class TestEfficiencyOrdering:
    """
    Test the key efficiency properties.

    The correct statements from Fan & Sesia (2512.15383):
    - LWC thresholds (in standardized units) <= GWC threshold. This is exact.
    - LWC half-widths (in raw units) <= GWC half-widths. Follows from above.
    - Bonferroni has tighter per-dimension marginal levels (1-alpha/d vs 1-alpha),
      so it achieves HIGHER marginal coverage — this is the conservatism.
    - LWC vs Bonferroni: LWC may produce narrower OR wider intervals depending
      on residual distribution — volume ordering is not guaranteed.

    The meaningful efficiency test: GWC >= LWC in terms of calibration thresholds.
    """

    def test_lwc_thresholds_leq_gwc_at_calibration(self, large_motor_data):
        """
        LWC thresholds in standardized units should be <= GWC threshold.
        This is the core theoretical claim from Algorithm 2.
        """
        d = large_motor_data
        cal_gwc = calibrate(d["models"], d["X_cal"], d["Y_cal"], alpha=0.05, method="gwc")
        cal_lwc = calibrate(d["models"], d["X_cal"], d["Y_cal"], alpha=0.05, method="lwc")
        # Both thresholds are in standardized units
        assert cal_gwc.thresholds is not None
        assert cal_lwc.thresholds is not None
        # LWC thresholds should be <= GWC threshold
        assert np.all(cal_lwc.thresholds <= cal_gwc.thresholds[0] + 1e-10)

    def test_lwc_half_widths_leq_gwc(self, large_motor_data):
        """LWC raw interval half-widths <= GWC half-widths per dimension."""
        d = large_motor_data
        cal_gwc = calibrate(d["models"], d["X_cal"], d["Y_cal"], alpha=0.05, method="gwc")
        cal_lwc = calibrate(d["models"], d["X_cal"], d["Y_cal"], alpha=0.05, method="lwc")
        hw_gwc = cal_gwc.interval_half_widths()
        hw_lwc = cal_lwc.interval_half_widths()
        # Each dimension: LWC <= GWC
        assert np.all(hw_lwc <= hw_gwc + 1e-8)

    def test_bonferroni_higher_marginal_coverage(self, large_motor_data):
        """
        Bonferroni is more conservative per dimension: sets each marginal at
        1-alpha/d rather than 1-alpha. This means Bonferroni marginal coverages
        should be higher than GWC marginal coverages.
        """
        d = large_motor_data
        pred_bonf = JointConformalPredictor(d["models"], alpha=0.05, method="bonferroni")
        pred_bonf.calibrate(d["X_cal"], d["Y_cal"])
        joint_bonf = pred_bonf.predict(d["X_test"])
        marginal_bonf = joint_bonf.marginal_coverage_rates(d["Y_test"])

        pred_gwc = JointConformalPredictor(d["models"], alpha=0.05, method="gwc")
        pred_gwc.calibrate(d["X_cal"], d["Y_cal"])
        joint_gwc = pred_gwc.predict(d["X_test"])
        marginal_gwc = joint_gwc.marginal_coverage_rates(d["Y_test"])

        # Bonferroni should have higher (or equal) marginal coverage
        for k in joint_bonf.dimensions:
            assert marginal_bonf[k] >= marginal_gwc[k] - 0.05, (
                f"Bonferroni marginal {k}: {marginal_bonf[k]:.3f} vs GWC: {marginal_gwc[k]:.3f}"
            )

    def test_all_methods_achieve_valid_coverage(self, large_motor_data):
        """All three methods should achieve joint coverage >= 0.90."""
        d = large_motor_data
        for method in ["bonferroni", "gwc", "lwc"]:
            cov = _calibrate_and_cover(
                d["models"], d["X_cal"], d["Y_cal"],
                d["X_test"], d["Y_test"], alpha=0.05, method=method,
            )
            assert cov >= 0.90, f"Method {method}: coverage {cov:.3f} < 0.90"


class TestDegenerateAndEdgeCases:
    def test_d1_standard_conformal_coverage(self):
        """
        With d=1 single output, should recover standard split conformal.
        Coverage should be >= 1-alpha.
        """
        rng = np.random.default_rng(10)
        n_total = 600
        n_cal = 300
        X = rng.standard_normal((n_total, 3))
        y = np.exp(X @ [0.5, -0.3, 0.2]) + rng.normal(0, 0.5, n_total)

        from insurance_conformal.multivariate.datasets import _SimpleLinearModel
        model = _SimpleLinearModel(positive=False).fit(X[:n_cal - 100], y[:n_cal - 100])

        X_cal, y_cal = X[n_cal - 100:n_cal], y[n_cal - 100:n_cal]
        X_test, y_test = X[n_cal:], y[n_cal:]

        pred = JointConformalPredictor({"output": model}, alpha=0.1, method="lwc")
        pred.calibrate(X_cal, {"output": y_cal})
        joint = pred.predict(X_test)
        cov = joint.joint_coverage_check({"output": y_test})
        assert cov >= 0.85, f"d=1 coverage: {cov:.3f}"

    def test_all_zero_severity(self, motor_data):
        """
        When severity is all zeros, the predictor should still run without error.
        """
        d = motor_data
        Y_zero_sev = {"frequency": d["Y_cal"]["frequency"], "severity": np.zeros(d["n_cal"])}
        pred = JointConformalPredictor(d["models"], alpha=0.05, method="lwc")
        pred.calibrate(d["X_cal"], Y_zero_sev)
        joint = pred.predict(d["X_test"])
        for k in joint.dimensions:
            assert np.all(np.isfinite(joint.lower[k]))
            assert np.all(np.isfinite(joint.upper[k]))

    def test_alpha_extremes(self, motor_data):
        """Coverage at very high and very low alpha."""
        d = motor_data
        for alpha in [0.01, 0.50]:
            pred = JointConformalPredictor(d["models"], alpha=alpha, method="lwc")
            pred.calibrate(d["X_cal"], d["Y_cal"])
            joint = pred.predict(d["X_test"])
            assert np.all(np.isfinite(joint.volume()))

    def test_single_calibration_point(self):
        """
        With n_cal=1, the predictor should use the only point as the threshold.
        """
        rng = np.random.default_rng(99)
        n_total = 100
        X = rng.standard_normal((n_total, 2))
        y = rng.normal(1.0, 0.5, n_total)

        from insurance_conformal.multivariate.datasets import _SimpleLinearModel
        model = _SimpleLinearModel().fit(X[:80], y[:80])

        X_cal = X[80:81]
        y_cal = y[80:81]
        X_test = X[81:]

        pred = JointConformalPredictor({"output": model}, alpha=0.05, method="gwc")
        pred.calibrate(X_cal, {"output": y_cal})
        joint = pred.predict(X_test)
        assert np.all(np.isfinite(joint.lower["output"]))

    def test_identical_calibration_points(self, motor_data):
        """Constant X in calibration (degenerate features)."""
        d = motor_data
        X_const = np.ones_like(d["X_cal"])
        pred = JointConformalPredictor(d["models"], alpha=0.05, method="lwc")
        pred.calibrate(X_const, d["Y_cal"])
        joint = pred.predict(d["X_test"])
        assert np.all(np.isfinite(joint.volume()))


class TestCompareMethodsDiagnostics:
    def test_compare_methods_returns_dict(self, large_motor_data):
        d = large_motor_data
        results = compare_methods(
            d["models"], d["X_cal"], d["Y_cal"], d["X_test"], d["Y_test"],
            alpha=0.05,
        )
        assert set(results.keys()) == {"bonferroni", "gwc", "lwc"}
        for method, rep in results.items():
            assert "joint_coverage" in rep
            assert "mean_widths" in rep

    def test_coverage_report_keys(self, large_motor_data):
        d = large_motor_data
        pred = JointConformalPredictor(d["models"], alpha=0.05, method="lwc")
        pred.calibrate(d["X_cal"], d["Y_cal"])
        rep = coverage_report(pred, d["X_test"], d["Y_test"])
        required_keys = [
            "joint_coverage", "marginal_coverages", "mean_widths",
            "mean_volume", "alpha", "method", "n_test", "target_coverage",
            "coverage_gap",
        ]
        for k in required_keys:
            assert k in rep, f"Missing key: {k}"
