"""
Tests for IntervalWidthController.

Key design point: width_target must satisfy 0 < width_target/scale < B.
The natural usage is to set scale = max_expected_width so that
width_target/scale is a fraction in (0, B).
"""

import numpy as np
import pytest

from insurance_conformal.risk import IntervalWidthController


def make_synthetic_widths(n: int, m: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic interval widths. widths[i, j] increases with j (wider at higher lambda).
    widths are in the range [50, 1500] — use scale appropriately.
    """
    rng = np.random.default_rng(seed)
    lambdas = np.linspace(0.5, 0.99, m)
    # base widths in [100, 1000]
    base_widths = rng.uniform(100, 1000, size=n)
    # Wider at higher lambda
    widths = np.outer(base_widths, 0.5 + lambdas)
    return widths, lambdas


def get_scale(widths: np.ndarray) -> float:
    """Natural scale: max possible width."""
    return float(widths.max()) + 1.0


class TestIntervalWidthControllerBasic:

    def test_calibrate_returns_self(self):
        widths, lambdas = make_synthetic_widths(300, 50)
        scale = get_scale(widths)
        # width_target as fraction of scale: target 60% of max width
        controller = IntervalWidthController(
            width_target=0.60 * scale, scale=scale, lambda_grid=lambdas
        )
        result = controller.calibrate_from_widths(widths)
        assert result is controller

    def test_is_calibrated_flag(self):
        widths, lambdas = make_synthetic_widths(200, 30)
        scale = get_scale(widths)
        controller = IntervalWidthController(
            width_target=0.70 * scale, scale=scale, lambda_grid=lambdas
        )
        assert not controller.is_calibrated_
        controller.calibrate_from_widths(widths)
        assert controller.is_calibrated_

    def test_lambda_hat_in_grid(self):
        widths, lambdas = make_synthetic_widths(400, 50)
        scale = get_scale(widths)
        controller = IntervalWidthController(
            width_target=0.65 * scale, scale=scale, lambda_grid=lambdas
        )
        controller.calibrate_from_widths(widths)
        assert lambdas[0] <= controller.lambda_hat_ <= lambdas[-1]

    def test_tighter_target_gives_smaller_lambda(self):
        """A tighter width constraint should select a smaller lambda (narrower intervals)."""
        widths, lambdas = make_synthetic_widths(500, 50, seed=7)
        scale = get_scale(widths)

        # Tighter target: want expected width <= 40% of max
        c_tight = IntervalWidthController(
            width_target=0.40 * scale, scale=scale, lambda_grid=lambdas
        )
        c_tight.calibrate_from_widths(widths)

        # Looser target: want expected width <= 80% of max
        c_loose = IntervalWidthController(
            width_target=0.80 * scale, scale=scale, lambda_grid=lambdas
        )
        c_loose.calibrate_from_widths(widths)

        # Tighter width target => smaller lambda (narrower intervals)
        assert c_tight.lambda_hat_ <= c_loose.lambda_hat_ + 0.05

    def test_n_calibration_recorded(self):
        n = 250
        widths, lambdas = make_synthetic_widths(n, 40)
        scale = get_scale(widths)
        controller = IntervalWidthController(
            width_target=0.70 * scale, scale=scale, lambda_grid=lambdas
        )
        controller.calibrate_from_widths(widths)
        assert controller.n_calibration_ == n

    def test_repr_after_calibrate(self):
        widths, lambdas = make_synthetic_widths(200, 30)
        scale = get_scale(widths)
        controller = IntervalWidthController(
            width_target=0.60 * scale, scale=scale, lambda_grid=lambdas
        )
        controller.calibrate_from_widths(widths)
        r = repr(controller)
        assert "lambda_hat" in r

    def test_repr_before_calibrate(self):
        controller = IntervalWidthController(width_target=0.5, scale=1.0)
        assert "not calibrated" in repr(controller)

    def test_normalised_widths_in_unit_interval(self):
        """When scale = max_width, normalised losses should all be in [0, 1]."""
        widths, lambdas = make_synthetic_widths(300, 40)
        scale = widths.max()
        normalised = widths / scale
        assert np.all(normalised >= 0)
        assert np.all(normalised <= 1.0 + 1e-9)


class TestIntervalWidthControllerEdgeCases:

    def test_negative_widths_raises(self):
        widths, lambdas = make_synthetic_widths(100, 20)
        scale = get_scale(widths)
        widths[0, 0] = -1.0
        controller = IntervalWidthController(
            width_target=0.5 * scale, scale=scale, lambda_grid=lambdas
        )
        with pytest.raises(ValueError, match="negative"):
            controller.calibrate_from_widths(widths)

    def test_wrong_column_count_raises(self):
        widths = np.ones((100, 10))
        lambdas = np.linspace(0.5, 0.99, 20)
        controller = IntervalWidthController(
            width_target=0.5, scale=1.0, lambda_grid=lambdas
        )
        with pytest.raises(ValueError):
            controller.calibrate_from_widths(widths)

    def test_1d_input_raises(self):
        widths = np.ones(100)
        lambdas = np.linspace(0.5, 0.99, 50)
        controller = IntervalWidthController(
            width_target=0.5, scale=1.0, lambda_grid=lambdas
        )
        with pytest.raises(ValueError):
            controller.calibrate_from_widths(widths)

    def test_zero_width_target_raises(self):
        with pytest.raises(ValueError, match="width_target"):
            IntervalWidthController(width_target=0.0)

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            IntervalWidthController(width_target=0.5, scale=0.0)

    def test_non_increasing_grid_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            IntervalWidthController(
                width_target=0.5,
                lambda_grid=np.array([0.9, 0.8, 0.7])
            )

    def test_target_too_tight_raises(self):
        """Width target as fraction of scale that's impossible to achieve."""
        n, m = 100, 30
        lambdas = np.linspace(0.5, 0.99, m)
        # Widths all above 900 (when scale=1000, normalised > 0.9)
        widths = 950 + np.random.default_rng(42).uniform(0, 40, (n, m))
        scale = 1000.0

        controller = IntervalWidthController(
            width_target=0.001 * scale,  # Wants widths < 1 — impossible
            scale=scale,
            B=1.0,
            lambda_grid=lambdas
        )
        with pytest.raises(RuntimeError):
            controller.calibrate_from_widths(widths)

    def test_risk_summary_keys(self):
        widths, lambdas = make_synthetic_widths(200, 30)
        scale = get_scale(widths)
        controller = IntervalWidthController(
            width_target=0.65 * scale, scale=scale, lambda_grid=lambdas
        )
        controller.calibrate_from_widths(widths)
        summary = controller.risk_summary()
        assert "lambda_hat" in summary
        assert "width_target" in summary
        assert "n_calibration" in summary

    def test_alpha_out_of_range_raises(self):
        """width_target must be < scale * B."""
        widths, lambdas = make_synthetic_widths(100, 20)
        scale = 1.0
        # width_target = 2.0 > scale * B = 1.0 -> alpha = 2.0 which is out of (0, 1)
        controller = IntervalWidthController(
            width_target=2.0, scale=scale, B=1.0, lambda_grid=lambdas
        )
        with pytest.raises(ValueError):
            controller.calibrate_from_widths(widths)
