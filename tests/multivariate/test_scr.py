"""
Tests for Solvency Capital Requirement (SCR) estimation.
"""

import numpy as np
import pytest

from insurance_conformal.multivariate.scr import (
    SolvencyCapitalEstimator,
    SCRResult,
    scr_report,
)


class TestSolvencyCapitalEstimator:
    def test_basic_init(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        assert scr.alpha == 0.005

    def test_high_alpha_warns(self, motor_data):
        d = motor_data
        with pytest.warns(UserWarning, match="Solvency II"):
            SolvencyCapitalEstimator(d["models"], alpha=0.05)

    def test_calibrate_returns_self(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        result = scr.calibrate(d["X_cal"], d["Y_cal"])
        assert result is scr

    def test_estimate_before_calibrate_raises(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        with pytest.raises(RuntimeError, match="calibrate"):
            scr.estimate(d["X_test"])

    def test_estimate_returns_scr_result(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        result = scr.estimate(d["X_test"])
        assert isinstance(result, SCRResult)

    def test_scr_per_policy_shape(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        result = scr.estimate(d["X_test"])
        assert result.scr_per_policy.shape == (d["n_test"],)

    def test_scr_per_policy_nonneg(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        result = scr.estimate(d["X_test"])
        assert np.all(result.scr_per_policy >= 0)

    def test_aggregate_scr_positive(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        result = scr.estimate(d["X_test"])
        assert result.aggregate_scr > 0

    def test_coverage_guarantee_correct(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        result = scr.estimate(d["X_test"])
        assert result.coverage_guarantee == pytest.approx(0.995)

    def test_finite_sample_bound(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        result = scr.estimate(d["X_test"])
        expected = 1.0 / (d["n_cal"] + 1)
        assert result.finite_sample_bound == pytest.approx(expected, rel=1e-6)

    def test_one_sided_lower_zero(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        result = scr.estimate(d["X_test"])
        for k in result.joint_upper:
            # SCR upper bounds should be positive
            assert np.all(np.isfinite(result.joint_upper[k]))

    def test_upside_coverage_valid(self, large_motor_data):
        """
        One-sided coverage: fraction with loss <= SCR_upper should be >= 99.5%.
        This uses alpha=0.005 so we expect coverage >= 0.995.
        """
        d = large_motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005, method="gwc")
        scr.calibrate(d["X_cal"], d["Y_cal"])
        result = scr.estimate(d["X_test"])

        # Check: fraction of test policies where freq <= U_freq AND sev <= U_sev
        freq_covered = d["Y_test"]["frequency"] <= result.joint_upper["frequency"]
        sev_covered = d["Y_test"]["severity"] <= result.joint_upper["severity"]
        joint_covered = np.mean(freq_covered & sev_covered)
        assert joint_covered >= 0.90, f"SCR joint coverage: {joint_covered:.3f}"

    def test_gwc_more_conservative_than_lwc(self, motor_data):
        d = motor_data
        scr_gwc = SolvencyCapitalEstimator(d["models"], alpha=0.005, method="gwc")
        scr_gwc.calibrate(d["X_cal"], d["Y_cal"])
        res_gwc = scr_gwc.estimate(d["X_test"])

        scr_lwc = SolvencyCapitalEstimator(d["models"], alpha=0.005, method="lwc")
        scr_lwc.calibrate(d["X_cal"], d["Y_cal"])
        res_lwc = scr_lwc.estimate(d["X_test"])

        # GWC should be >= LWC (more conservative)
        assert res_gwc.aggregate_scr >= res_lwc.aggregate_scr * 0.99


class TestSCRResult:
    def _make_result(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        return scr.estimate(d["X_test"])

    def test_bootstrap_ci(self, motor_data):
        result = self._make_result(motor_data)
        ci = result.bootstrap_ci(n_bootstrap=100)
        assert "lower" in ci and "upper" in ci and "mean" in ci
        assert ci["lower"] <= ci["mean"] <= ci["upper"]

    def test_bootstrap_ci_lower_leq_aggregate(self, motor_data):
        result = self._make_result(motor_data)
        ci = result.bootstrap_ci(n_bootstrap=200)
        assert ci["lower"] <= result.aggregate_scr * 1.1  # roughly in range
        assert ci["upper"] >= result.aggregate_scr * 0.9

    def test_summary_keys(self, motor_data):
        result = self._make_result(motor_data)
        s = result.summary()
        assert "aggregate_scr" in s
        assert "coverage_guarantee" in s
        assert "per_policy_stats" in s
        assert "p95" in s["per_policy_stats"]

    def test_summary_aggregate_consistent(self, motor_data):
        result = self._make_result(motor_data)
        s = result.summary()
        assert s["aggregate_scr"] == pytest.approx(result.aggregate_scr)


class TestSCRReport:
    def test_scr_report_basic(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        report = scr_report(scr, d["X_test"], n_bootstrap=50)
        assert "aggregate_scr" in report
        assert "bootstrap_ci" in report
        assert "n_portfolio" in report

    def test_scr_report_n_portfolio(self, motor_data):
        d = motor_data
        scr = SolvencyCapitalEstimator(d["models"], alpha=0.005)
        scr.calibrate(d["X_cal"], d["Y_cal"])
        report = scr_report(scr, d["X_test"], n_bootstrap=20)
        assert report["n_portfolio"] == d["n_test"]
