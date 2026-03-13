"""
Tests for SelectiveRiskController.
"""

import numpy as np
import pytest

from insurance_conformal.risk import SelectiveRiskController


def high_claim_loss(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Binary loss: 1 if claim > 5000, else 0."""
    return (y > 5000).astype(float)


def proportional_loss(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Normalised loss: min(y / 10000, 1)."""
    return np.minimum(y / 10000.0, 1.0)


def make_scored_data(n: int, seed: int = 42):
    """
    Synthetic scored portfolio. High-score risks have lower expected claims.
    score ~ Uniform(0, 1), claim ~ Gamma(shape=1, scale=2000*(1 - 0.8*score))
    Designed so that at high score thresholds, claim frequency above 5000 is < 30%.
    """
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, size=n)
    # Higher score = lower risk = smaller claims
    scale = 2000 * (1 - scores * 0.8)
    y = rng.gamma(1, scale, size=n)
    return y, scores


class TestSelectiveRiskBasic:

    def test_calibrate_returns_self(self):
        y, scores = make_scored_data(500)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        result = src.calibrate(y, scores)
        assert result is src

    def test_is_calibrated_flag(self):
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        assert not src.is_calibrated_
        y, scores = make_scored_data(500)
        src.calibrate(y, scores)
        assert src.is_calibrated_

    def test_threshold_in_score_range(self):
        y, scores = make_scored_data(500)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        src.calibrate(y, scores)
        assert scores.min() <= src.threshold_ <= scores.max()

    def test_predict_output_structure(self):
        y, scores = make_scored_data(500)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        src.calibrate(y, scores)

        new_scores = np.array([0.2, 0.5, 0.8, 0.9])
        result = src.predict(new_scores)

        assert "score" in result.columns
        assert "accept" in result.columns
        assert "threshold" in result.columns
        assert result.shape[0] == 4

    def test_accept_above_threshold(self):
        y, scores = make_scored_data(500)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        src.calibrate(y, scores)

        threshold = src.threshold_
        # Create test scores clearly above and below threshold
        new_scores = np.array([min(threshold - 0.15, scores.min() + 0.01),
                                max(threshold + 0.15, scores.max() - 0.01)])
        result = src.predict(new_scores)

        accepts = result["accept"].to_numpy()
        assert not accepts[0]
        assert accepts[1]

    def test_selection_rate_recorded(self):
        y, scores = make_scored_data(500)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        src.calibrate(y, scores)
        assert 0 < src.selection_rate_ <= 1

    def test_n_calibration(self):
        n = 300
        y, scores = make_scored_data(n)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        src.calibrate(y, scores)
        assert src.n_calibration_ == n

    def test_higher_alpha_accepts_more(self):
        """Higher loss tolerance -> threshold can be lower -> more accepted."""
        y, scores = make_scored_data(1000, seed=7)

        src_tight = SelectiveRiskController(alpha=0.25, loss_fn=proportional_loss, xi_min=0.3)
        src_tight.calibrate(y, scores)

        src_loose = SelectiveRiskController(alpha=0.50, loss_fn=proportional_loss, xi_min=0.3)
        src_loose.calibrate(y, scores)

        assert src_loose.threshold_ <= src_tight.threshold_ + 0.1

    def test_portfolio_summary_keys(self):
        y, scores = make_scored_data(500)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        src.calibrate(y, scores)
        summary = src.portfolio_summary(y, scores)
        expected = {"threshold", "selection_rate", "n_accepted", "n_total", "mean_loss_accepted"}
        assert expected.issubset(set(summary.keys()))

    def test_repr_before_calibrate(self):
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        assert "not calibrated" in repr(src)

    def test_repr_after_calibrate(self):
        y, scores = make_scored_data(300)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        src.calibrate(y, scores)
        r = repr(src)
        assert "threshold" in r


class TestSelectiveRiskEdgeCases:

    def test_predict_before_calibrate_raises(self):
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        with pytest.raises(RuntimeError, match="not calibrated"):
            src.predict(np.array([0.5]))

    def test_mismatched_lengths_raises(self):
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        with pytest.raises(ValueError, match="same length"):
            src.calibrate(np.array([100.0, 200.0]), np.array([0.5]))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            SelectiveRiskController(alpha=0.0, loss_fn=high_claim_loss)

    def test_invalid_xi_min_raises(self):
        with pytest.raises(ValueError, match="xi_min must be in"):
            SelectiveRiskController(alpha=0.1, loss_fn=high_claim_loss, xi_min=1.5)

    def test_zero_exposure_raises(self):
        y, scores = make_scored_data(200)
        src = SelectiveRiskController(alpha=0.35, loss_fn=high_claim_loss)
        exposure = np.ones(len(y))
        exposure[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            src.calibrate(y, scores, exposure=exposure)

    def test_very_tight_alpha_raises(self):
        """Alpha so tight no threshold achieves it -> RuntimeError."""
        rng = np.random.default_rng(42)
        # All claims very large -> high_claim_loss always 1
        y = rng.uniform(10000, 20000, size=200)
        scores = rng.uniform(0, 1, size=200)
        src = SelectiveRiskController(
            alpha=0.001,
            loss_fn=high_claim_loss,
            xi_min=0.5,
        )
        with pytest.raises(RuntimeError):
            src.calibrate(y, scores)

    def test_custom_lambda_grid(self):
        y, scores = make_scored_data(400)
        custom_grid = np.linspace(0.3, 0.9, 50)
        src = SelectiveRiskController(
            alpha=0.40,
            loss_fn=high_claim_loss,
            lambda_grid=custom_grid,
        )
        src.calibrate(y, scores)
        assert src.is_calibrated_

    def test_exposure_weighting(self):
        """Exposure-weighted calibration should produce a valid threshold."""
        rng = np.random.default_rng(55)
        n = 400
        y, scores = make_scored_data(n, seed=55)
        exposure = rng.uniform(0.5, 1.5, size=n)

        src = SelectiveRiskController(alpha=0.35, loss_fn=proportional_loss)
        src.calibrate(y, scores, exposure=exposure)
        assert src.is_calibrated_
        assert src.threshold_ is not None

    def test_loss_out_of_bounds_raises(self):
        """If loss_fn returns values > B, calibrate should raise."""
        def bad_loss(y, scores):
            return y / 100.0  # Could easily exceed B=1.0

        rng = np.random.default_rng(42)
        y = rng.uniform(200, 500, size=200)  # y/100 = 2-5, exceeds B=1
        scores = rng.uniform(0, 1, size=200)

        src = SelectiveRiskController(alpha=0.35, loss_fn=bad_loss, B=1.0)
        with pytest.raises(ValueError, match="outside.*B"):
            src.calibrate(y, scores)

    def test_proportional_loss_calibration(self):
        """Proportional loss (bounded by 1) should calibrate successfully."""
        y, scores = make_scored_data(500)
        src = SelectiveRiskController(alpha=0.40, loss_fn=proportional_loss, xi_min=0.4)
        src.calibrate(y, scores)
        assert src.is_calibrated_
