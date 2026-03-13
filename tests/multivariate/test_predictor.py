"""
Tests for JointConformalPredictor API.
"""

import numpy as np
import pytest

from insurance_conformal.multivariate import JointConformalPredictor, JointPredictionSet


class TestJointConformalPredictorInit:
    def test_basic_init(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"], alpha=0.05, method="lwc")
        assert pred.alpha == 0.05
        assert pred.method == "lwc"
        assert not pred.is_calibrated()

    def test_invalid_alpha_raises(self, motor_data):
        d = motor_data
        with pytest.raises(ValueError, match="alpha"):
            JointConformalPredictor(d["models"], alpha=1.5)

    def test_alpha_zero_raises(self, motor_data):
        d = motor_data
        with pytest.raises(ValueError, match="alpha"):
            JointConformalPredictor(d["models"], alpha=0.0)

    def test_invalid_method_raises(self, motor_data):
        d = motor_data
        with pytest.raises(ValueError, match="method"):
            JointConformalPredictor(d["models"], method="oracle")

    def test_repr_before_calibration(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        r = repr(pred)
        assert "not calibrated" in r

    def test_repr_after_calibration(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        pred.calibrate(d["X_cal"], d["Y_cal"])
        r = repr(pred)
        assert "calibrated" in r


class TestJointConformalPredictorCalibrate:
    def test_calibrate_returns_self(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        result = pred.calibrate(d["X_cal"], d["Y_cal"])
        assert result is pred

    def test_is_calibrated_after_calibrate(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        pred.calibrate(d["X_cal"], d["Y_cal"])
        assert pred.is_calibrated()

    def test_dimensions_set(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        pred.calibrate(d["X_cal"], d["Y_cal"])
        assert set(pred.dimensions_) == {"frequency", "severity"}

    @pytest.mark.parametrize("method", ["bonferroni", "sidak", "gwc", "lwc"])
    def test_all_methods_calibrate(self, motor_data, method):
        d = motor_data
        pred = JointConformalPredictor(d["models"], method=method)
        pred.calibrate(d["X_cal"], d["Y_cal"])
        assert pred.is_calibrated()

    def test_calibration_summary(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        pred.calibrate(d["X_cal"], d["Y_cal"])
        s = pred.calibration_summary()
        assert "half_widths" in s
        assert "n_cal" in s
        assert s["n_cal"] == d["n_cal"]


class TestJointConformalPredictorPredict:
    def test_predict_before_calibrate_raises(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        with pytest.raises(RuntimeError, match="calibrate"):
            pred.predict(d["X_test"])

    def test_returns_joint_prediction_set(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        pred.calibrate(d["X_cal"], d["Y_cal"])
        joint = pred.predict(d["X_test"])
        assert isinstance(joint, JointPredictionSet)

    def test_output_shape(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        pred.calibrate(d["X_cal"], d["Y_cal"])
        joint = pred.predict(d["X_test"])
        n_test = d["n_test"]
        for k in ["frequency", "severity"]:
            assert joint.lower[k].shape == (n_test,)
            assert joint.upper[k].shape == (n_test,)

    def test_lower_leq_upper(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        pred.calibrate(d["X_cal"], d["Y_cal"])
        joint = pred.predict(d["X_test"])
        for k in joint.dimensions:
            assert np.all(joint.lower[k] <= joint.upper[k])

    def test_one_sided_lower_zero(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"], one_sided=True)
        pred.calibrate(d["X_cal"], d["Y_cal"])
        joint = pred.predict(d["X_test"])
        for k in joint.dimensions:
            np.testing.assert_array_equal(joint.lower[k], np.zeros(d["n_test"]))

    def test_single_observation_predict(self, motor_data):
        d = motor_data
        pred = JointConformalPredictor(d["models"])
        pred.calibrate(d["X_cal"], d["Y_cal"])
        joint = pred.predict(d["X_test"][:1])
        assert joint.n_obs == 1

    @pytest.mark.parametrize("method", ["bonferroni", "gwc", "lwc"])
    def test_all_methods_predict(self, motor_data, method):
        d = motor_data
        pred = JointConformalPredictor(d["models"], method=method)
        pred.calibrate(d["X_cal"], d["Y_cal"])
        joint = pred.predict(d["X_test"])
        assert joint.n_obs == d["n_test"]


class TestJointPredictionSetMethods:
    def _make_pred(self, motor_data, method="lwc"):
        d = motor_data
        pred = JointConformalPredictor(d["models"], alpha=0.05, method=method)
        pred.calibrate(d["X_cal"], d["Y_cal"])
        return pred.predict(d["X_test"]), d

    def test_volume_positive(self, motor_data):
        joint, d = self._make_pred(motor_data)
        vol = joint.volume()
        assert vol.shape == (d["n_test"],)
        assert np.all(vol > 0)

    def test_contains_own_prediction(self, motor_data):
        # The prediction interval center (point prediction) should always
        # be inside the interval — lower <= pred <= upper
        d = motor_data
        pred_obj = JointConformalPredictor(d["models"], alpha=0.05)
        pred_obj.calibrate(d["X_cal"], d["Y_cal"])
        joint = pred_obj.predict(d["X_test"])
        # Midpoints
        mid = {
            k: (joint.lower[k] + joint.upper[k]) / 2
            for k in joint.dimensions
        }
        covered = joint.contains(mid)
        assert np.all(covered)

    def test_contains_returns_bool_array(self, motor_data):
        joint, d = self._make_pred(motor_data)
        Y = d["Y_test"]
        covered = joint.contains(Y)
        assert covered.dtype == bool
        assert covered.shape == (d["n_test"],)

    def test_joint_coverage_check(self, motor_data):
        joint, d = self._make_pred(motor_data)
        cov = joint.joint_coverage_check(d["Y_test"])
        assert 0.0 <= cov <= 1.0

    def test_marginal_coverage_rates(self, motor_data):
        joint, d = self._make_pred(motor_data)
        rates = joint.marginal_coverage_rates(d["Y_test"])
        assert set(rates.keys()) == {"frequency", "severity"}
        for v in rates.values():
            assert 0.0 <= v <= 1.0

    def test_to_polars(self, motor_data):
        joint, d = self._make_pred(motor_data)
        try:
            df = joint.to_polars()
            import polars as pl
            assert isinstance(df, pl.DataFrame)
            assert df.shape[0] == d["n_test"]
            expected_cols = {"frequency_lower", "frequency_upper", "severity_lower", "severity_upper"}
            assert set(df.columns) == expected_cols
        except ImportError:
            pytest.skip("polars not available")

    def test_summary_keys(self, motor_data):
        joint, _ = self._make_pred(motor_data)
        s = joint.summary()
        assert "n_obs" in s
        assert "mean_volume" in s
        assert "mean_widths" in s

    def test_marginal_intervals(self, motor_data):
        joint, d = self._make_pred(motor_data)
        widths = joint.marginal_intervals()
        for k in joint.dimensions:
            assert k in widths
            assert np.all(widths[k] >= 0)

    def test_repr(self, motor_data):
        joint, _ = self._make_pred(motor_data)
        r = repr(joint)
        assert "JointPredictionSet" in r

    def test_contains_ndarray_input(self, motor_data):
        joint, d = self._make_pred(motor_data)
        Y_arr = np.column_stack([d["Y_test"]["frequency"], d["Y_test"]["severity"]])
        covered = joint.contains(Y_arr)
        assert covered.shape == (d["n_test"],)
