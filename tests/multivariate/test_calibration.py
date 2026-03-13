"""
Tests for calibration module.
"""

import numpy as np
import pytest

from insurance_conformal.multivariate.calibration import (
    calibrate,
    compute_residuals,
    CalibratedScores,
    _normalise_models,
    _normalise_y,
)
from insurance_conformal.multivariate.datasets import _SimpleExpModel


class TestNormaliseModels:
    def test_dict_passthrough(self):
        m = {"a": 1, "b": 2}
        assert _normalise_models(m) == m

    def test_list_to_dict(self):
        m_list = ["m0", "m1"]
        result = _normalise_models(m_list)
        assert list(result.keys()) == ["0", "1"]
        assert result["0"] == "m0"

    def test_single_model(self):
        result = _normalise_models("model")
        assert result == {"0": "model"}


class TestNormaliseY:
    def test_dict_passthrough(self):
        Y = {"freq": np.array([1.0, 2.0]), "sev": np.array([100.0, 200.0])}
        result = _normalise_y(Y, ["freq", "sev"])
        np.testing.assert_array_equal(result["freq"], Y["freq"])

    def test_1d_array(self):
        Y = np.array([1.0, 2.0, 3.0])
        result = _normalise_y(Y, ["freq"])
        assert "freq" in result
        np.testing.assert_array_equal(result["freq"], Y)

    def test_2d_array(self):
        Y = np.column_stack([np.array([1.0, 2.0]), np.array([10.0, 20.0])])
        result = _normalise_y(Y, ["a", "b"])
        np.testing.assert_array_equal(result["a"], [1.0, 2.0])
        np.testing.assert_array_equal(result["b"], [10.0, 20.0])

    def test_mismatch_raises(self):
        Y = np.ones((10, 3))
        with pytest.raises(AssertionError):
            _normalise_y(Y, ["a", "b"])  # 2 keys but 3 columns


class TestComputeResiduals:
    def test_basic_shape(self, motor_data):
        d = motor_data
        res = compute_residuals(d["models"], d["X_cal"], d["Y_cal"])
        assert res.shape == (d["n_cal"], 2)

    def test_nonnegative(self, motor_data):
        d = motor_data
        res = compute_residuals(d["models"], d["X_cal"], d["Y_cal"])
        assert np.all(res >= 0)

    def test_absolute_residuals_are_true_differences(self, motor_data):
        d = motor_data
        res = compute_residuals(d["models"], d["X_cal"], d["Y_cal"])
        # Spot-check: residual_j should be |y_j - y_hat_j|
        freq_pred = d["models"]["frequency"].predict(d["X_cal"])
        expected_freq_res = np.abs(d["Y_cal"]["frequency"] - freq_pred)
        np.testing.assert_allclose(res[:, 0], expected_freq_res, rtol=1e-6)

    def test_zero_claim_mask_ignored_by_compute_residuals(self, motor_data):
        """
        compute_residuals() always returns raw absolute residuals — masking
        is handled at calibration level, not residual computation level.
        """
        d = motor_data
        mask = d["zero_mask_cal"]
        res_no_mask = compute_residuals(d["models"], d["X_cal"], d["Y_cal"])
        res_with_mask = compute_residuals(
            d["models"], d["X_cal"], d["Y_cal"], zero_claim_mask=mask
        )
        # Both should return identical residuals — mask has no effect here
        np.testing.assert_array_equal(res_no_mask, res_with_mask)


class TestCalibrate:
    @pytest.mark.parametrize("method", ["bonferroni", "sidak", "gwc", "lwc"])
    def test_returns_calibrated_scores(self, motor_data, method):
        d = motor_data
        cal = calibrate(
            d["models"], d["X_cal"], d["Y_cal"],
            alpha=0.05, method=method
        )
        assert isinstance(cal, CalibratedScores)
        assert cal.n_cal == d["n_cal"]
        assert cal.alpha == 0.05
        assert cal.method == method

    def test_dimensions_match_model_keys(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        assert set(cal.dimensions) == {"frequency", "severity"}

    def test_residuals_shape(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        assert cal.residuals.shape == (d["n_cal"], 2)

    def test_sigma_positive(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        assert np.all(cal.sigma_hat > 0)

    def test_lwc_thresholds_leq_gwc(self, motor_data):
        d = motor_data
        cal_gwc = calibrate(d["models"], d["X_cal"], d["Y_cal"], method="gwc")
        cal_lwc = calibrate(d["models"], d["X_cal"], d["Y_cal"], method="lwc")
        hw_gwc = cal_gwc.interval_half_widths()
        hw_lwc = cal_lwc.interval_half_widths()
        # LWC should produce <= half-widths than GWC (tighter or equal)
        assert np.all(hw_lwc <= hw_gwc + 1e-10)

    def test_interval_half_widths_positive(self, motor_data):
        d = motor_data
        for method in ["bonferroni", "gwc", "lwc"]:
            cal = calibrate(d["models"], d["X_cal"], d["Y_cal"], method=method)
            hw = cal.interval_half_widths()
            assert np.all(hw > 0), f"Non-positive half-widths for method={method}"

    def test_half_widths_in_original_units(self, motor_data):
        """
        Frequency hw should be in claim units (small).
        Severity hw should be in £ (hundreds to thousands).
        """
        d = motor_data
        for method in ["bonferroni", "gwc", "lwc"]:
            cal = calibrate(d["models"], d["X_cal"], d["Y_cal"], method=method)
            hw = cal.interval_half_widths()
            assert hw[0] < 100, f"Freq half-width {hw[0]} seems wrong for method={method}"
            assert hw[1] > 10, f"Sev half-width {hw[1]} seems wrong for method={method}"

    def test_zero_claim_masking_sets_severity_dim(self, motor_data):
        """
        With zero_claim_mask, severity_dim should be set and sigma_hat[sev]
        should be computed from claim-only obs (should differ from no-mask version).
        """
        d = motor_data
        cal_no_mask = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        cal_masked = calibrate(
            d["models"], d["X_cal"], d["Y_cal"],
            zero_claim_mask=d["zero_mask_cal"]
        )
        # severity_dim should be set
        assert cal_masked.severity_dim == 1  # severity is second dimension
        # sigma_hat for severity may differ between masked and unmasked
        # (masked uses only claim obs — typically higher variance)
        assert cal_masked.sigma_hat[1] > 0

    def test_zero_claim_masking_produces_valid_half_widths(self, motor_data):
        d = motor_data
        cal_masked = calibrate(
            d["models"], d["X_cal"], d["Y_cal"],
            zero_claim_mask=d["zero_mask_cal"]
        )
        hw = cal_masked.interval_half_widths()
        assert np.all(hw > 0)
        # Severity hw with proper claim-only sigma should be reasonable (£ scale)
        assert hw[1] > 100, f"Severity hw {hw[1]} too small after masking"

    def test_unknown_method_raises(self, motor_data):
        d = motor_data
        with pytest.raises(ValueError, match="Unknown method"):
            calibrate(d["models"], d["X_cal"], d["Y_cal"], method="bad")

    def test_shape_mismatch_raises(self, motor_data):
        d = motor_data
        Y_bad = {"frequency": np.ones(5), "severity": np.ones(d["n_cal"])}
        with pytest.raises(ValueError, match="rows"):
            calibrate(d["models"], d["X_cal"], Y_bad)

    def test_zero_claim_mask_shape_mismatch_raises(self, motor_data):
        d = motor_data
        bad_mask = np.ones(5, dtype=bool)
        with pytest.raises(ValueError, match="zero_claim_mask"):
            calibrate(d["models"], d["X_cal"], d["Y_cal"], zero_claim_mask=bad_mask)

    def test_list_models_input(self, motor_data):
        d = motor_data
        models_list = list(d["models"].values())
        Y_arr = np.column_stack([d["Y_cal"]["frequency"], d["Y_cal"]["severity"]])
        cal = calibrate(models_list, d["X_cal"], Y_arr)
        assert cal.n_cal == d["n_cal"]

    def test_standardized_residuals(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        std = cal.standardized_residuals()
        assert std.shape == (d["n_cal"], 2)

    def test_max_scores(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        ms = cal.max_scores()
        assert ms.shape == (d["n_cal"],)

    def test_bonferroni_quantile_is_abs_residual_quantile(self, motor_data):
        """
        Bonferroni per-dim quantile should match manual computation
        of conformal quantile on |y - y_hat| at 1-alpha/d level.
        """
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"], alpha=0.05, method="bonferroni")
        hw = cal.interval_half_widths()

        # Manual: frequency quantile at 1 - alpha/2 = 0.975 level
        n = d["n_cal"]
        k = min(int(np.ceil((n + 1) * (1.0 - 0.05 / 2))), n)
        freq_res = cal.residuals[:, 0]
        expected_hw = np.sort(freq_res)[k - 1]
        assert hw[0] == pytest.approx(expected_hw, rel=1e-6)
