"""
Tests for SelectiveConformalRC (Xu, Guo & Wei 2025, arXiv:2512.12844).

Test philosophy: validate the statistical guarantees, not just the mechanics.
The PAC guarantee is the point. We simulate data where the guarantee is
checkable and assert it holds across repeated trials.

Data convention used throughout:
- selection scores: higher = better risk (underwriter wants to accept)
- pred_sets_cal: shape (m, n) — row j = losses at lambda_2_grid[j]
- loss fn: 0-1 indicator (large claim = 1), so expected_set_size == E[large claim | selected]
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from insurance_conformal.risk import (
    SelectiveConformalRC,
    InfeasibleSCRCError,
    SCRCResult,
    build_pred_set_matrix,
)
from insurance_conformal.risk.selection_scores import (
    selection_score_msp,
    selection_score_margin,
    selection_score_entropy,
    selection_score_energy,
)


# ---------------------------------------------------------------------------
# Test fixtures / data generators
# ---------------------------------------------------------------------------

def make_underwriting_data(
    n: int = 800,
    seed: int = 42,
    good_risk_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic two-class underwriting dataset.

    Good risks: selection_score ~ U(0.5, 1.0), large_claim_prob = 0.05
    Bad risks:  selection_score ~ U(0.0, 0.5), large_claim_prob = 0.40

    Returns (y_cal, scores_cal, pred_sets_cal, lambda_2_grid).
    pred_sets_cal uses a single binary loss (large claim indicator).
    """
    rng = np.random.default_rng(seed)
    n_good = int(n * good_risk_frac)
    n_bad = n - n_good

    # Selection scores (stage 1)
    scores_good = rng.uniform(0.5, 1.0, size=n_good)
    scores_bad = rng.uniform(0.0, 0.5, size=n_bad)
    scores_cal = np.concatenate([scores_good, scores_bad])

    # Claims (stage 2 loss: 1 if large claim)
    large_good = rng.binomial(1, 0.05, size=n_good).astype(float)
    large_bad = rng.binomial(1, 0.40, size=n_bad).astype(float)
    y_cal = np.concatenate([large_good, large_bad])  # 0/1 directly

    # Single threshold: lambda_2 ∈ {0, 1}. Row 0: threshold=0 (accept all losses),
    # Row 1: threshold=1 (loss=0 for everyone, trivially). This models a
    # prediction set that at lambda_2=0 exposes all loss, at lambda_2=1 covers all.
    # For a richer test: use actual loss matrix that varies across lambda_2.
    #
    # We use threshold on y itself: loss(y, lam2) = (y > lam2).astype(float)
    # For y ∈ {0,1} and lam2 ∈ {-0.5, 0.5}: at lam2=-0.5 all get loss 1 or 0
    # depending on y; at lam2=0.5 only y=1 incur loss.
    lambda_2_grid = np.array([-0.5, 0.5])
    def loss_fn(y: np.ndarray, lam: float) -> np.ndarray:
        return (y > lam).astype(float)

    pred_sets_cal = build_pred_set_matrix(y_cal, lambda_2_grid, loss_fn, B=1.0)

    return y_cal, scores_cal, pred_sets_cal, lambda_2_grid


def make_graded_data(
    n: int = 600,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dataset with graded selection scores and a richer lambda_2_grid.
    Used for testing grid search and Pareto frontier.
    """
    rng = np.random.default_rng(seed)
    scores_cal = rng.uniform(0, 1, size=n)
    # Claim probability decreases with score
    p_claim = 0.5 * (1 - scores_cal)
    y_cal = rng.binomial(1, p_claim).astype(float)

    lambda_2_grid = np.array([-0.5, 0.0, 0.5, 1.0])

    def loss_fn(y: np.ndarray, lam: float) -> np.ndarray:
        return (y > lam).astype(float)

    pred_sets_cal = build_pred_set_matrix(y_cal, lambda_2_grid, loss_fn, B=1.0)
    return y_cal, scores_cal, pred_sets_cal, lambda_2_grid


# ---------------------------------------------------------------------------
# Test 1: SCRC-I controls risk at alpha (PAC guarantee)
# ---------------------------------------------------------------------------

class TestSCRCIRiskControl:

    def test_scrc_i_controls_risk_at_alpha(self):
        """
        PAC guarantee: fraction of trials where conditional risk > alpha < delta.

        We run multiple calibration/test splits. In each, we check whether the
        conditional expected loss on the test set (selected risks) exceeds alpha.
        The fraction of violations should be <= delta (plus Monte Carlo error).

        This is the core guarantee from Theorem 3.1 of arXiv:2512.12844.
        """
        alpha = 0.15
        xi = 0.5
        delta = 0.1
        n_trials = 50
        n_cal = 600
        n_test = 300

        violations = 0
        rng = np.random.default_rng(77)

        for trial in range(n_trials):
            seed = int(rng.integers(1, 10000))
            y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data(
                n=n_cal, seed=seed
            )
            y_test, scores_test, _, _ = make_underwriting_data(n=n_test, seed=seed + 1000)

            try:
                scrc = SelectiveConformalRC(
                    alpha=alpha, xi=xi, delta=delta, method="SCRC-I"
                )
                scrc.calibrate(y_cal, scores_cal, pred_sets_cal,
                               lambda_2_grid=lambda_2_grid)
            except InfeasibleSCRCError:
                continue

            r = scrc.result_
            # Apply stage 1 to test set
            selected_test = scores_test >= r.lambda_1
            if selected_test.sum() == 0:
                continue

            # Conditional risk on test set: fraction with y_test > lambda_2
            y_sel = y_test[selected_test]
            cond_risk = float((y_sel > r.lambda_2).mean())
            if cond_risk > alpha:
                violations += 1

        # Violation rate should be <= delta + Monte Carlo slack
        violation_rate = violations / n_trials
        # Generous allowance: delta + 3*std (binomial std = sqrt(delta*(1-delta)/n_trials))
        allowed = delta + 3 * math.sqrt(delta * (1 - delta) / n_trials) + 0.05
        assert violation_rate <= allowed, (
            f"Violation rate {violation_rate:.3f} > allowed {allowed:.3f} "
            f"(alpha={alpha}, xi={xi}, delta={delta})"
        )

    def test_scrc_i_result_is_scrc_result(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=0.1)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        assert isinstance(scrc.result_, SCRCResult)
        assert scrc.result_.method == "SCRC-I"
        assert scrc.result_.feasible

    def test_scrc_i_lambda_1_in_score_range(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=0.1)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        r = scrc.result_
        assert scores_cal.min() <= r.lambda_1 <= scores_cal.max()

    def test_scrc_i_selection_rate_above_xi(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=0.1)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        r = scrc.result_
        assert r.selection_rate >= 0.5

    def test_scrc_i_n_calibration_correct(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data(n=500)
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=0.1)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        assert scrc.result_.n_calibration == 500


# ---------------------------------------------------------------------------
# Test 2: InfeasibleSCRCError raised when no feasible lambda_1
# ---------------------------------------------------------------------------

class TestSCRCIInfeasibility:

    def test_scrc_i_rejects_when_no_feasible_l1(self):
        """
        A custom lambda_1_grid that starts above all scores means no threshold
        achieves the required selection rate xi.
        """
        rng = np.random.default_rng(1)
        n = 400
        scores_cal = rng.uniform(0.0, 0.5, size=n)
        y_cal = rng.binomial(1, 0.3, size=n).astype(float)
        lambda_2_grid = np.array([-0.5, 0.5])
        pred_sets_cal = build_pred_set_matrix(
            y_cal, lambda_2_grid,
            lambda y, lam: (y > lam).astype(float), B=1.0
        )

        # lambda_1_grid starts above all observed scores: no threshold selects anyone
        custom_l1_grid = np.linspace(0.6, 1.0, 50)
        scrc = SelectiveConformalRC(
            alpha=0.20, xi=0.5, delta=0.1,
            lambda_1_grid=custom_l1_grid,
        )
        with pytest.raises(InfeasibleSCRCError, match="selection rate"):
            scrc.calibrate(y_cal, scores_cal, pred_sets_cal,
                           lambda_2_grid=lambda_2_grid)

    def test_scrc_i_rejects_when_xi_lcb_zero(self):
        """
        Very small n with very small delta => eps = sqrt(log(2/delta)/(2n)) > 1
        => xi_lcb = max(xi_hat - eps, 0) = 0 => InfeasibleSCRCError.

        With n=10 and delta=1e-10: eps = sqrt(log(2e10)/20) ≈ 1.09 > 1 >= xi_hat.
        """
        rng = np.random.default_rng(5)
        n = 10  # Very small calibration set
        scores_cal = rng.uniform(0.5, 1.0, size=n)
        y_cal = rng.binomial(1, 0.1, size=n).astype(float)
        lambda_2_grid = np.array([-0.5, 0.5])
        pred_sets_cal = build_pred_set_matrix(
            y_cal, lambda_2_grid,
            lambda y, lam: (y > lam).astype(float), B=1.0
        )
        # Very small delta => very large DKW eps => xi_lcb = 0 guaranteed
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=1e-10)
        with pytest.raises(InfeasibleSCRCError):
            scrc.calibrate(y_cal, scores_cal, pred_sets_cal,
                           lambda_2_grid=lambda_2_grid)

    def test_scrc_i_raises_before_calibrate(self):
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5)
        with pytest.raises(RuntimeError, match="not calibrated"):
            scrc.predict(np.array([0.5, 0.6]))

    def test_scrc_t_requires_score_test(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, method="SCRC-T")
        with pytest.raises(ValueError, match="score_test must be provided"):
            scrc.calibrate(y_cal, scores_cal, pred_sets_cal,
                           lambda_2_grid=lambda_2_grid)


# ---------------------------------------------------------------------------
# Test 3: SCRC-T tighter than SCRC-I
# ---------------------------------------------------------------------------

class TestSCRCTVsSCRCI:

    def test_scrc_t_tighter_than_scrc_i(self):
        """
        SCRC-T is transductive: it uses test data, giving tighter (smaller)
        expected_set_size than SCRC-I on the same calibration set.

        This tests the qualitative property: SCRC-T <= SCRC-I in expected set size
        at least on average across random seeds. We allow one exception since
        SCRC-T uses a different budget formula.
        """
        alpha = 0.20
        xi = 0.5
        n_cal = 600
        n_test = 50
        n_trials = 10

        scrc_i_ess_list = []
        scrc_t_ess_list = []

        rng = np.random.default_rng(101)
        for trial in range(n_trials):
            seed = int(rng.integers(1, 5000))
            y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data(
                n=n_cal, seed=seed
            )
            _, scores_test, _, _ = make_underwriting_data(n=n_test, seed=seed + 5000)

            try:
                scrc_i = SelectiveConformalRC(alpha=alpha, xi=xi, method="SCRC-I")
                scrc_i.calibrate(y_cal, scores_cal, pred_sets_cal,
                                 lambda_2_grid=lambda_2_grid)

                scrc_t = SelectiveConformalRC(alpha=alpha, xi=xi, method="SCRC-T")
                scrc_t.calibrate(y_cal, scores_cal, pred_sets_cal,
                                 lambda_2_grid=lambda_2_grid,
                                 score_test=scores_test)
            except InfeasibleSCRCError:
                continue

            scrc_i_ess_list.append(scrc_i.result_.expected_set_size)
            scrc_t_ess_list.append(scrc_t.result_.expected_set_size)

        assert len(scrc_i_ess_list) >= 5, "Too few successful trials for comparison"
        mean_i = float(np.mean(scrc_i_ess_list))
        mean_t = float(np.mean(scrc_t_ess_list))
        # SCRC-T should have equal or smaller expected set size on average
        # Allow 10% slack for randomness
        assert mean_t <= mean_i * 1.1, (
            f"SCRC-T mean ESS {mean_t:.4f} not tighter than SCRC-I {mean_i:.4f}"
        )

    def test_scrc_t_method_label(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scores_test = scores_cal[:50]
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, method="SCRC-T")
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal,
                       lambda_2_grid=lambda_2_grid, score_test=scores_test)
        assert scrc.result_.method == "SCRC-T"


# ---------------------------------------------------------------------------
# Test 4: DKW correction effect
# ---------------------------------------------------------------------------

class TestDKWCorrection:

    def test_scrc_i_dkw_correction_effect(self):
        """
        Higher delta => smaller DKW eps => larger xi_lcb.

        eps = sqrt(log(2/delta) / (2n)). As delta increases, log(2/delta) decreases,
        so eps decreases, and xi_lcb = max(xi_hat - eps, 0) increases.
        We verify that higher delta leads to larger xi_lcb.
        """
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data(
            n=500, seed=7
        )

        scrc_low_delta = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=0.01)
        scrc_high_delta = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=0.30)

        try:
            scrc_low_delta.calibrate(y_cal, scores_cal, pred_sets_cal,
                                     lambda_2_grid=lambda_2_grid)
            scrc_high_delta.calibrate(y_cal, scores_cal, pred_sets_cal,
                                      lambda_2_grid=lambda_2_grid)
        except InfeasibleSCRCError:
            pytest.skip("Infeasible for this seed — not a test failure")

        # eps = sqrt(log(2/delta)/(2n)). Higher delta => log(2/delta) smaller
        # => eps smaller => xi_lcb = xi_hat - eps is LARGER for higher delta.
        assert scrc_high_delta.result_.xi_lcb >= scrc_low_delta.result_.xi_lcb

    def test_xi_lcb_formula(self):
        """
        xi_lcb should equal max(xi_hat - eps, 0) where eps = sqrt(log(2/delta)/(2n)).
        """
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data(
            n=600, seed=13
        )
        delta = 0.10
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=delta)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)

        n = scrc.result_.n_calibration
        xi_hat = scrc.result_.selection_rate
        eps = math.sqrt(math.log(2.0 / delta) / (2 * n))
        expected_xi_lcb = max(xi_hat - eps, 0.0)

        assert abs(scrc.result_.xi_lcb - expected_xi_lcb) < 1e-10


# ---------------------------------------------------------------------------
# Test 5: Selection score functions
# ---------------------------------------------------------------------------

class TestSelectionScoreFunctions:

    def test_selection_score_msp_shape(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.9, 0.05, 0.05]])
        scores = selection_score_msp(probs)
        assert scores.shape == (3,)
        np.testing.assert_allclose(scores, [0.7, 0.4, 0.9])

    def test_selection_score_msp_1d(self):
        probs = np.array([0.8, 0.3, 0.6])
        scores = selection_score_msp(probs)
        np.testing.assert_allclose(scores, [0.8, 0.3, 0.6])

    def test_selection_score_margin_multiclass(self):
        probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.39, 0.21]])
        scores = selection_score_margin(probs)
        assert scores.shape == (2,)
        np.testing.assert_allclose(scores, [0.7, 0.01], atol=1e-10)

    def test_selection_score_margin_binary(self):
        probs = np.array([0.9, 0.5, 0.1])
        scores = selection_score_margin(probs)
        # |2p - 1|
        expected = np.abs(2 * probs - 1)
        np.testing.assert_allclose(scores, expected)

    def test_selection_score_entropy_uniform_is_zero(self):
        """Uniform distribution has max entropy => negative entropy = 0."""
        probs = np.array([[1/3, 1/3, 1/3], [0.25, 0.25, 0.25, 0.25]])
        # For 3-class: normalised entropy = 1, so score = 0
        s3 = selection_score_entropy(np.array([[1/3, 1/3, 1/3]]))
        assert abs(s3[0]) < 1e-10

    def test_selection_score_entropy_deterministic_is_one(self):
        """Deterministic distribution has zero entropy => score = 1."""
        probs = np.array([[1.0, 0.0, 0.0]])
        scores = selection_score_entropy(probs)
        np.testing.assert_allclose(scores, [1.0], atol=1e-6)

    def test_selection_score_entropy_monotone(self):
        """More concentrated = higher entropy score."""
        p_concentrated = np.array([[0.9, 0.05, 0.05]])
        p_spread = np.array([[0.5, 0.3, 0.2]])
        s_conc = selection_score_entropy(p_concentrated)[0]
        s_spread = selection_score_entropy(p_spread)[0]
        assert s_conc > s_spread

    def test_selection_score_energy_shape(self):
        logits = np.array([[2.0, 1.0, -1.0], [0.1, 0.0, 0.1]])
        scores = selection_score_energy(logits)
        assert scores.shape == (2,)

    def test_selection_score_energy_higher_logits_higher_score(self):
        """
        Higher logit magnitudes => lower energy => higher neg-energy score.
        A risk with large positive logits is more in-distribution.
        """
        logits_strong = np.array([[5.0, 4.0, 3.0]])
        logits_weak = np.array([[0.1, 0.0, -0.1]])
        s_strong = selection_score_energy(logits_strong)[0]
        s_weak = selection_score_energy(logits_weak)[0]
        assert s_strong > s_weak

    def test_selection_score_energy_negative_temperature_raises(self):
        logits = np.ones((3, 3))
        with pytest.raises(ValueError, match="temperature"):
            selection_score_energy(logits, temperature=-1.0)

    def test_selection_score_energy_1d_input(self):
        """1D logits (single class) should return shape (n,)."""
        logits = np.array([1.0, 2.0, -1.0])
        scores = selection_score_energy(logits)
        assert scores.shape == (3,)

    def test_all_score_functions_return_float64(self):
        probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
        logits = np.array([[1.0, 0.5, -0.5], [0.2, 0.1, -0.3]])
        for fn in [selection_score_msp, selection_score_margin, selection_score_entropy]:
            s = fn(probs)
            assert s.dtype == np.float64, f"{fn.__name__} returned {s.dtype}"
        s_energy = selection_score_energy(logits)
        assert s_energy.dtype == np.float64


# ---------------------------------------------------------------------------
# Test 6: underwriting_summary format
# ---------------------------------------------------------------------------

class TestUnderwritingSummary:

    def test_underwriting_summary_format(self):
        """
        underwriting_summary() must return a dict with all required keys
        and correct value types.
        """
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=0.1)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)

        summary = scrc.underwriting_summary()

        required_keys = {
            "method", "alpha", "xi", "delta",
            "lambda_1", "lambda_2",
            "selection_rate", "xi_lcb",
            "n_calibration", "n_selected",
            "conditional_risk", "expected_set_size",
            "feasible",
        }
        assert required_keys == set(summary.keys()), (
            f"Missing: {required_keys - set(summary.keys())}, "
            f"Extra: {set(summary.keys()) - required_keys}"
        )

        # Type checks
        assert isinstance(summary["method"], str)
        assert isinstance(summary["alpha"], float)
        assert isinstance(summary["xi"], float)
        assert isinstance(summary["delta"], float)
        assert isinstance(summary["lambda_1"], float)
        assert isinstance(summary["lambda_2"], float)
        assert isinstance(summary["selection_rate"], float)
        assert isinstance(summary["xi_lcb"], float)
        assert isinstance(summary["n_calibration"], int)
        assert isinstance(summary["n_selected"], int)
        assert isinstance(summary["conditional_risk"], float)
        assert isinstance(summary["expected_set_size"], float)
        assert isinstance(summary["feasible"], bool)

    def test_underwriting_summary_values_in_range(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5, delta=0.1)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        summary = scrc.underwriting_summary()

        assert 0 < summary["selection_rate"] <= 1
        assert 0 <= summary["xi_lcb"] <= 1
        assert 0 <= summary["conditional_risk"] <= 1
        assert summary["n_selected"] <= summary["n_calibration"]
        assert summary["feasible"] is True

    def test_underwriting_summary_raises_before_calibrate(self):
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5)
        with pytest.raises(RuntimeError, match="not calibrated"):
            scrc.underwriting_summary()


# ---------------------------------------------------------------------------
# Test 7: Pareto frontier monotone
# ---------------------------------------------------------------------------

class TestParetoFrontier:

    def test_pareto_frontier_monotone(self):
        """
        Pareto frontier must have selection_rate monotonically decreasing
        as lambda_1 increases. Higher lambda_1 = fewer accepted = lower ESS.
        """
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_graded_data(n=600)

        scrc = SelectiveConformalRC(
            alpha=0.25, xi=0.4, delta=0.1,
            run_grid_search=True
        )
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)

        df = scrc.pareto_report()
        assert isinstance(df, pl.DataFrame)

        if len(df) < 2:
            pytest.skip("Pareto frontier has < 2 points — can't test monotonicity")

        sel_rates = df["selection_rate"].to_numpy()
        ess_vals = df["expected_set_size"].to_numpy()

        # selection_rate is sorted descending
        for i in range(len(sel_rates) - 1):
            assert sel_rates[i] >= sel_rates[i + 1] - 1e-10, (
                f"Pareto frontier selection_rate not descending at index {i}: "
                f"{sel_rates[i]:.4f} < {sel_rates[i+1]:.4f}"
            )

        # expected_set_size should be non-increasing as selection_rate decreases
        # (more restrictive selection => lower expected loss on selected book)
        for i in range(len(ess_vals) - 1):
            assert ess_vals[i] >= ess_vals[i + 1] - 1e-10, (
                f"Pareto frontier ESS not non-increasing at index {i}: "
                f"{ess_vals[i]:.4f} < {ess_vals[i+1]:.4f}"
            )

    def test_pareto_report_raises_without_grid_search(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_graded_data(n=400)
        scrc = SelectiveConformalRC(alpha=0.25, xi=0.4, run_grid_search=False)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        with pytest.raises(RuntimeError, match="run_grid_search"):
            scrc.pareto_report()

    def test_pareto_report_returns_dataframe(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_graded_data(n=400)
        scrc = SelectiveConformalRC(
            alpha=0.25, xi=0.4, delta=0.1, run_grid_search=True
        )
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        df = scrc.pareto_report()
        assert isinstance(df, pl.DataFrame)
        if len(df) > 0:
            assert "lambda_1" in df.columns
            assert "lambda_2" in df.columns
            assert "selection_rate" in df.columns
            assert "expected_set_size" in df.columns


# ---------------------------------------------------------------------------
# Test 8: predict() output format
# ---------------------------------------------------------------------------

class TestPredict:

    def test_predict_output_columns(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)

        scores_new = np.array([0.2, 0.5, 0.8, 0.95])
        df = scrc.predict(scores_new)

        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"score", "selected", "lambda_1", "lambda_2"}
        assert df.shape[0] == 4

    def test_predict_high_scores_selected(self):
        """
        Risks with score well above lambda_1 should be selected.
        """
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)

        lam1 = scrc.result_.lambda_1
        scores_new = np.array([lam1 + 0.3, lam1 - 0.3])
        df = scrc.predict(scores_new)
        selected = df["selected"].to_numpy()
        assert selected[0] is True or bool(selected[0])
        assert selected[1] is False or not bool(selected[1])

    def test_predict_lambda_values_constant(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        df = scrc.predict(np.array([0.1, 0.5, 0.9]))
        # All rows have same lambda_1 and lambda_2
        assert df["lambda_1"].n_unique() == 1
        assert df["lambda_2"].n_unique() == 1


# ---------------------------------------------------------------------------
# Test 9: build_pred_set_matrix
# ---------------------------------------------------------------------------

class TestBuildPredSetMatrix:

    def test_build_pred_set_matrix_shape(self):
        y = np.array([1000, 5000, 2000, 8000])
        lambdas = np.array([3000.0, 5000.0, 7000.0])
        result = build_pred_set_matrix(y, lambdas, lambda y, lam: (y > lam).astype(float))
        assert result.shape == (3, 4)

    def test_build_pred_set_matrix_values(self):
        y = np.array([1000.0, 5000.0, 2000.0, 8000.0])
        lambdas = np.array([3000.0, 5000.0, 7000.0])
        result = build_pred_set_matrix(y, lambdas, lambda y, lam: (y > lam).astype(float))
        expected = np.array([
            [0., 1., 0., 1.],  # threshold=3000: 5000>3000, 8000>3000
            [0., 0., 0., 1.],  # threshold=5000: only 8000>5000
            [0., 0., 0., 1.],  # threshold=7000: only 8000>7000
        ])
        np.testing.assert_array_equal(result, expected)

    def test_build_pred_set_matrix_bad_loss_raises(self):
        y = np.array([1.0, 2.0])
        lambdas = np.array([0.5])
        with pytest.raises(ValueError, match="outside"):
            build_pred_set_matrix(
                y, lambdas,
                lambda y, lam: y * 5.0,  # Returns values > B=1
                B=1.0
            )

    def test_build_pred_set_matrix_wrong_shape_raises(self):
        y = np.array([1.0, 2.0, 3.0])
        lambdas = np.array([0.5])
        with pytest.raises(ValueError):
            build_pred_set_matrix(
                y, lambdas,
                lambda y, lam: np.array([0.0, 0.5]),  # Wrong shape
            )


# ---------------------------------------------------------------------------
# Test 10: repr and constructor validation
# ---------------------------------------------------------------------------

class TestConstructorAndRepr:

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            SelectiveConformalRC(alpha=0.0, xi=0.5)

    def test_invalid_xi_raises(self):
        with pytest.raises(ValueError, match="xi"):
            SelectiveConformalRC(alpha=0.20, xi=1.5)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            SelectiveConformalRC(alpha=0.20, xi=0.5, delta=-0.1)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            SelectiveConformalRC(alpha=0.20, xi=0.5, method="SCRC-X")

    def test_alpha_ge_B_raises(self):
        with pytest.raises(ValueError, match="alpha must be < B"):
            SelectiveConformalRC(alpha=0.50, xi=0.5, B=0.3)

    def test_repr_before_calibrate(self):
        scrc = SelectiveConformalRC(alpha=0.15, xi=0.6)
        r = repr(scrc)
        assert "not calibrated" in r
        assert "0.15" in r

    def test_repr_after_calibrate(self):
        y_cal, scores_cal, pred_sets_cal, lambda_2_grid = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5)
        scrc.calibrate(y_cal, scores_cal, pred_sets_cal, lambda_2_grid=lambda_2_grid)
        r = repr(scrc)
        assert "SCRC-I" in r
        assert "lambda_1" in r

    def test_pred_sets_wrong_ndim_raises(self):
        y_cal, scores_cal, _, _ = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5)
        with pytest.raises(ValueError, match="2D"):
            scrc.calibrate(y_cal, scores_cal, np.ones(len(scores_cal)))

    def test_pred_sets_wrong_columns_raises(self):
        y_cal, scores_cal, _, _ = make_underwriting_data()
        scrc = SelectiveConformalRC(alpha=0.20, xi=0.5)
        wrong_pred_sets = np.ones((3, len(scores_cal) + 5))
        with pytest.raises(ValueError, match="columns"):
            scrc.calibrate(y_cal, scores_cal, wrong_pred_sets)
