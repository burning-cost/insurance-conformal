"""
Direct unit tests for insurance_conformal.risk.selection_scores.

The selection_scores module provides four functions that convert raw model
outputs into scalar scores used in the first stage of SelectiveConformalRC.
Convention: higher score = more selectable (underwriter wants to accept).

Test philosophy: verify exact values on small hand-computable inputs, verify
shapes and types, verify edge cases, and verify monotonicity/ordering
properties that are required by the SCRC algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal.risk.selection_scores import (
    selection_score_msp,
    selection_score_margin,
    selection_score_entropy,
    selection_score_energy,
)


# ---------------------------------------------------------------------------
# selection_score_msp
# ---------------------------------------------------------------------------


class TestSelectionScoreMSP:
    def test_basic_multiclass(self):
        """Returns max probability per row."""
        probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]])
        scores = selection_score_msp(probs)
        assert scores.shape == (2,)
        assert scores[0] == pytest.approx(0.8)
        assert scores[1] == pytest.approx(0.4)

    def test_1d_passthrough(self):
        """1-D input is returned unchanged (binary case)."""
        probs = np.array([0.7, 0.3, 0.9])
        scores = selection_score_msp(probs)
        np.testing.assert_array_equal(scores, probs)

    def test_uniform_distribution(self):
        """Uniform over K classes returns 1/K."""
        K = 5
        probs = np.ones((4, K)) / K
        scores = selection_score_msp(probs)
        np.testing.assert_allclose(scores, 1.0 / K)

    def test_concentrated_distribution(self):
        """All mass on one class: score = 1.0."""
        probs = np.zeros((3, 4))
        probs[0, 2] = 1.0
        probs[1, 0] = 1.0
        probs[2, 3] = 1.0
        scores = selection_score_msp(probs)
        np.testing.assert_allclose(scores, 1.0)

    def test_single_observation(self):
        """Works with a single row."""
        probs = np.array([[0.5, 0.3, 0.2]])
        scores = selection_score_msp(probs)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(0.5)

    def test_output_range(self):
        """MSP is always in [0, 1] for valid probability inputs."""
        rng = np.random.default_rng(0)
        raw = rng.dirichlet(np.ones(5), size=200)
        scores = selection_score_msp(raw)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_ordering_preserved(self):
        """More concentrated distribution has higher MSP than less concentrated."""
        concentrated = np.array([[0.9, 0.05, 0.05]])
        diffuse = np.array([[0.4, 0.35, 0.25]])
        assert selection_score_msp(concentrated)[0] > selection_score_msp(diffuse)[0]


# ---------------------------------------------------------------------------
# selection_score_margin
# ---------------------------------------------------------------------------


class TestSelectionScoreMargin:
    def test_basic_multiclass(self):
        """Top-1 minus top-2 probability."""
        probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.39, 0.21]])
        scores = selection_score_margin(probs)
        assert scores[0] == pytest.approx(0.7)
        assert scores[1] == pytest.approx(0.01)

    def test_1d_binary_case(self):
        """1-D input: margin = |2p - 1|."""
        probs = np.array([0.9, 0.5, 0.3])
        scores = selection_score_margin(probs)
        assert scores[0] == pytest.approx(0.8)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(0.4)

    def test_tied_top_two_classes(self):
        """Equal top-2 probabilities give margin = 0."""
        probs = np.array([[0.5, 0.5, 0.0]])
        scores = selection_score_margin(probs)
        assert scores[0] == pytest.approx(0.0)

    def test_single_class(self):
        """Single-class array: margin = the single probability."""
        probs = np.array([[0.7], [0.3]])
        scores = selection_score_margin(probs)
        assert scores[0] == pytest.approx(0.7)
        assert scores[1] == pytest.approx(0.3)

    def test_output_range(self):
        """Margin is in [0, 1] for valid probabilities."""
        rng = np.random.default_rng(1)
        raw = rng.dirichlet(np.ones(4), size=200)
        scores = selection_score_margin(raw)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_margin_less_than_or_equal_msp(self):
        """Margin <= MSP always (margin uses top1 - top2, MSP uses top1)."""
        rng = np.random.default_rng(2)
        probs = rng.dirichlet(np.ones(5), size=100)
        assert (selection_score_margin(probs) <= selection_score_msp(probs) + 1e-10).all()

    def test_single_observation(self):
        probs = np.array([[0.6, 0.25, 0.15]])
        scores = selection_score_margin(probs)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# selection_score_entropy
# ---------------------------------------------------------------------------


class TestSelectionScoreEntropy:
    def test_uniform_gives_zero(self):
        """Uniform distribution: maximum entropy, score = 0."""
        K = 4
        probs = np.ones((3, K)) / K
        scores = selection_score_entropy(probs)
        np.testing.assert_allclose(scores, 0.0, atol=1e-10)

    def test_concentrated_gives_one(self):
        """All mass on one class: minimum entropy, score = 1."""
        probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scores = selection_score_entropy(probs)
        np.testing.assert_allclose(scores, 1.0, atol=1e-10)

    def test_known_value_binary(self):
        """Binary case: score = 1 - H(p,q)/log(2). Verify at p=0.9."""
        # H(0.9, 0.1) = -(0.9*log(0.9) + 0.1*log(0.1)) in nats
        p = 0.9
        q = 0.1
        h = -(p * np.log(p) + q * np.log(q))
        expected = 1.0 - h / np.log(2)
        probs = np.array([0.9, 0.1])
        scores = selection_score_entropy(probs)
        assert scores[0] == pytest.approx(expected, abs=1e-10)

    def test_1d_binary_symmetric(self):
        """p=0.5 gives score 0; p=1.0 gives score 1."""
        probs = np.array([0.5, 1.0 - 1e-12])
        scores = selection_score_entropy(probs)
        assert scores[0] == pytest.approx(0.0, abs=1e-10)
        assert scores[1] == pytest.approx(1.0, abs=1e-6)

    def test_single_class_column(self):
        """K=1: only one possible class, entropy = 0, score = 1."""
        probs = np.array([[1.0], [1.0], [1.0]])
        scores = selection_score_entropy(probs)
        np.testing.assert_allclose(scores, 1.0)

    def test_output_range(self):
        """Score is always in [0, 1]."""
        rng = np.random.default_rng(3)
        raw = rng.dirichlet(np.ones(6), size=200)
        scores = selection_score_entropy(raw)
        assert (scores >= 0 - 1e-10).all()
        assert (scores <= 1 + 1e-10).all()

    def test_monotone_in_concentration(self):
        """More concentrated dist has higher entropy score than uniform."""
        uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
        semi_concentrated = np.array([[0.7, 0.1, 0.1, 0.1]])
        assert selection_score_entropy(semi_concentrated)[0] > selection_score_entropy(uniform)[0]

    def test_single_observation(self):
        probs = np.array([[0.6, 0.2, 0.2]])
        scores = selection_score_entropy(probs)
        assert scores.shape == (1,)
        assert 0 <= scores[0] <= 1


# ---------------------------------------------------------------------------
# selection_score_energy
# ---------------------------------------------------------------------------


class TestSelectionScoreEnergy:
    def test_basic_shape(self):
        """Returns 1-D array of length n."""
        logits = np.array([[2.0, 1.0, -1.0], [0.1, 0.0, 0.1]])
        scores = selection_score_energy(logits)
        assert scores.shape == (2,)

    def test_values_are_non_positive(self):
        """Negative energy is always <= 0 (energy is positive)."""
        rng = np.random.default_rng(4)
        logits = rng.standard_normal((100, 5))
        scores = selection_score_energy(logits)
        assert (scores <= 1e-10).all()

    def test_1d_input(self):
        """1-D input is treated as n observations with one class each."""
        logits = np.array([2.0, 1.0, 3.0])
        scores = selection_score_energy(logits)
        assert scores.shape == (3,)

    def test_temperature_1_known_value(self):
        """Manual check: logits=[0, 0], T=1, energy = -log(2)."""
        logits = np.array([[0.0, 0.0]])
        # E = -1 * log(exp(0) + exp(0)) = -log(2)
        # score = -E = log(2)
        scores = selection_score_energy(logits, temperature=1.0)
        assert scores[0] == pytest.approx(np.log(2.0), abs=1e-10)

    def test_temperature_effect(self):
        """Higher temperature reduces the spread of energy scores."""
        logits = np.array([[3.0, 0.0, -3.0]])
        scores_t1 = selection_score_energy(logits, temperature=1.0)
        scores_t10 = selection_score_energy(logits, temperature=10.0)
        # With higher T, the dominant logit matters less; energy spread smaller
        # Both should be valid; just check they differ
        assert scores_t1[0] != pytest.approx(scores_t10[0])

    def test_invalid_temperature_raises(self):
        """temperature <= 0 must raise ValueError."""
        logits = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="temperature"):
            selection_score_energy(logits, temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            selection_score_energy(logits, temperature=-1.0)

    def test_high_dominant_logit_gives_higher_score(self):
        """A risk with a clearly dominant logit has higher energy score."""
        # In-distribution risk: one logit dominates
        logits_in = np.array([[10.0, -5.0, -5.0]])
        # Out-of-distribution: logits spread more evenly
        logits_out = np.array([[-1.0, -1.0, -1.0]])
        score_in = selection_score_energy(logits_in)[0]
        score_out = selection_score_energy(logits_out)[0]
        assert score_in > score_out

    def test_single_observation(self):
        logits = np.array([[1.0, 2.0, 0.5]])
        scores = selection_score_energy(logits)
        assert scores.shape == (1,)

    def test_numerical_stability_large_logits(self):
        """log-sum-exp trick prevents overflow with large logits."""
        logits = np.array([[1000.0, 999.0, 998.0]])
        scores = selection_score_energy(logits)
        assert np.isfinite(scores[0])
