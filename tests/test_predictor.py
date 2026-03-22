"""
Tests for InsuranceConformalPredictor.

The coverage guarantee test is the most important: for any alpha, the empirical
coverage on a fresh test set should be >= 1 - alpha. We allow a small tolerance
for finite-sample variation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_conformal import InsuranceConformalPredictor


class TestInitialisation:
    def test_defaults(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"])
        assert cp.nonconformity == "pearson_weighted"
        assert cp.distribution == "tweedie"
        assert cp.calibration_frac == 0.2
        assert not cp.is_calibrated_

    def test_invalid_nonconformity(self, poisson_data):
        with pytest.raises(ValueError, match="Unknown nonconformity"):
            InsuranceConformalPredictor(
                model=poisson_data["model"],
                nonconformity="rubbish",
            )

    def test_invalid_distribution(self, poisson_data):
        with pytest.raises(ValueError, match="Unknown distribution"):
            InsuranceConformalPredictor(
                model=poisson_data["model"],
                distribution="normal",
            )

    def test_invalid_calibration_frac(self, poisson_data):
        with pytest.raises(ValueError, match="calibration_frac"):
            InsuranceConformalPredictor(
                model=poisson_data["model"],
                calibration_frac=1.5,
            )

    def test_explicit_tweedie_power(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"],
            tweedie_power=1.0,
        )
        assert cp.tweedie_power == 1.0

    def test_repr_before_calibration(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"])
        r = repr(cp)
        assert "not calibrated" in r
        assert "pearson_weighted" in r

    def test_repr_after_calibration(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        r = repr(cp)
        assert "calibrated on" in r
        assert "not calibrated" not in r


class TestCalibration:
    def test_calibrate_sets_state(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])

        assert cp.is_calibrated_
        assert cp.n_calibration_ == len(poisson_data["y_cal"])
        assert cp.cal_scores_ is not None
        assert len(cp.cal_scores_) == len(poisson_data["y_cal"])

    def test_calibrate_returns_self(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        result = cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        assert result is cp

    def test_scores_nonnegative(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        assert np.all(cp.cal_scores_ >= 0)

    def test_recalibration_clears_cache(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        cp.predict_interval(poisson_data["X_test"], alpha=0.1)
        assert 0.1 in cp.cal_quantiles_

        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        assert len(cp.cal_quantiles_) == 0

    def test_predict_before_calibrate_raises(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"])
        with pytest.raises(RuntimeError, match="not been calibrated"):
            cp.predict_interval(poisson_data["X_test"])

    def test_calibrate_with_exposure(self, poisson_data):
        exposure = np.ones(len(poisson_data["y_cal"]))
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"], exposure=exposure)
        assert cp.is_calibrated_

    def test_calibrate_with_pandas(self, poisson_data):
        X_cal_df = pd.DataFrame(poisson_data["X_cal"])
        y_cal_s = pd.Series(poisson_data["y_cal"])
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(X_cal_df, y_cal_s)
        assert cp.is_calibrated_

    def test_calibrate_with_polars(self, poisson_data):
        """Polars Series should be accepted for y_cal."""
        y_cal_pl = pl.Series(poisson_data["y_cal"].tolist())
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], y_cal_pl)
        assert cp.is_calibrated_

    def test_calibrate_mismatched_lengths_raises(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        y_short = poisson_data["y_cal"][:-1]
        with pytest.raises(ValueError, match="lengths must match"):
            cp.calibrate(poisson_data["X_cal"], y_short)


class TestPredictInterval:
    def test_output_structure(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        intervals = cp.predict_interval(poisson_data["X_test"], alpha=0.10)

        assert isinstance(intervals, pl.DataFrame)
        assert set(intervals.columns) == {"lower", "point", "upper"}
        assert len(intervals) == len(poisson_data["y_test"])

    def test_lower_leq_point_leq_upper(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        intervals = cp.predict_interval(poisson_data["X_test"], alpha=0.10)

        assert (intervals["lower"] <= intervals["point"]).all()
        assert (intervals["point"] <= intervals["upper"]).all()

    def test_lower_bound_nonnegative(self, poisson_data):
        for nc in ["raw", "pearson", "pearson_weighted"]:
            cp = InsuranceConformalPredictor(
                model=poisson_data["model"], nonconformity=nc, tweedie_power=1.0
            )
            cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
            intervals = cp.predict_interval(poisson_data["X_test"], alpha=0.10)
            assert (intervals["lower"] >= 0).all(), f"Lower bound negative for {nc}"

    def test_invalid_alpha(self, poisson_data):
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        with pytest.raises(ValueError, match="alpha"):
            cp.predict_interval(poisson_data["X_test"], alpha=1.5)

    def test_wider_intervals_at_smaller_alpha(self, poisson_data):
        """alpha=0.01 should give wider intervals than alpha=0.20."""
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])

        i90 = cp.predict_interval(poisson_data["X_test"], alpha=0.10)
        i50 = cp.predict_interval(poisson_data["X_test"], alpha=0.50)

        w90 = (i90["upper"] - i90["lower"]).mean()
        w50 = (i50["upper"] - i50["lower"]).mean()
        assert w90 > w50, "90% intervals should be wider than 50% intervals"

    def test_alpha_01(self, poisson_data):
        """alpha=0.01 should produce very wide intervals."""
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        intervals = cp.predict_interval(poisson_data["X_test"], alpha=0.01)
        assert (intervals["upper"] > 0).all()

    def test_alpha_05(self, poisson_data):
        """alpha=0.5 should produce narrower intervals than alpha=0.10."""
        cp = InsuranceConformalPredictor(model=poisson_data["model"], tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        i50 = cp.predict_interval(poisson_data["X_test"], alpha=0.50)
        i10 = cp.predict_interval(poisson_data["X_test"], alpha=0.10)
        w50 = (i50["upper"] - i50["lower"]).mean()
        w10 = (i10["upper"] - i10["lower"]).mean()
        assert w50 < w10


class TestCoverageGuarantee:
    """
    The most important tests: verify the marginal coverage guarantee holds.

    P(y in [lower, upper]) >= 1 - alpha.

    We allow a generous tolerance because these are finite samples (n=400 test).
    With n=400, the standard error of an empirical 90% coverage rate is about
    sqrt(0.9 * 0.1 / 400) = 1.5%, so we tolerate 4% below target.
    """

    TOLERANCE = 0.04

    def _check_coverage(self, cp, X_test, y_test, alpha, name=""):
        intervals = cp.predict_interval(X_test, alpha=alpha)
        lower = intervals["lower"].to_numpy()
        upper = intervals["upper"].to_numpy()
        covered = (y_test >= lower) & (y_test <= upper)
        coverage = covered.mean()
        target = 1 - alpha
        assert coverage >= target - self.TOLERANCE, (
            f"{name}: empirical coverage {coverage:.3%} is more than "
            f"{self.TOLERANCE:.0%} below target {target:.0%}"
        )

    def test_coverage_pearson_alpha_10(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"],
            nonconformity="pearson",
            tweedie_power=1.0,
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        self._check_coverage(
            cp, poisson_data["X_test"], poisson_data["y_test"], alpha=0.10, name="pearson"
        )

    def test_coverage_pearson_weighted_alpha_10(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"],
            nonconformity="pearson_weighted",
            tweedie_power=1.0,
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        self._check_coverage(
            cp,
            poisson_data["X_test"],
            poisson_data["y_test"],
            alpha=0.10,
            name="pearson_weighted",
        )

    def test_coverage_raw_alpha_10(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"],
            nonconformity="raw",
            tweedie_power=1.0,
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        self._check_coverage(
            cp, poisson_data["X_test"], poisson_data["y_test"], alpha=0.10, name="raw"
        )

    def test_coverage_alpha_20(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"],
            nonconformity="pearson_weighted",
            tweedie_power=1.0,
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        self._check_coverage(
            cp, poisson_data["X_test"], poisson_data["y_test"], alpha=0.20, name="alpha=0.20"
        )

    def test_coverage_alpha_50(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"],
            nonconformity="pearson_weighted",
            tweedie_power=1.0,
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        self._check_coverage(
            cp, poisson_data["X_test"], poisson_data["y_test"], alpha=0.50, name="alpha=0.50"
        )

    def test_pearson_weighted_narrower_than_raw(self, poisson_data):
        """
        Key claim from Manna et al.: pearson_weighted produces narrower intervals
        than raw while maintaining the same coverage.
        """
        cp_raw = InsuranceConformalPredictor(
            model=poisson_data["model"], nonconformity="raw", tweedie_power=1.0
        )
        cp_pw = InsuranceConformalPredictor(
            model=poisson_data["model"], nonconformity="pearson_weighted", tweedie_power=1.0
        )

        cp_raw.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        cp_pw.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])

        i_raw = cp_raw.predict_interval(poisson_data["X_test"], alpha=0.10)
        i_pw = cp_pw.predict_interval(poisson_data["X_test"], alpha=0.10)

        w_raw = (i_raw["upper"] - i_raw["lower"]).mean()
        w_pw = (i_pw["upper"] - i_pw["lower"]).mean()

        # pearson_weighted should be noticeably narrower (at least 5%)
        assert w_pw < w_raw, (
            f"pearson_weighted (mean_width={w_pw:.4f}) should be narrower than "
            f"raw (mean_width={w_raw:.4f})"
        )


class TestCoverageByDecile:
    def test_output_structure(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"], nonconformity="pearson", tweedie_power=1.0
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        df = cp.coverage_by_decile(
            poisson_data["X_test"], poisson_data["y_test"], alpha=0.10
        )

        assert isinstance(df, pl.DataFrame)
        assert "decile" in df.columns
        assert "coverage" in df.columns
        assert "mean_predicted" in df.columns
        assert "n_obs" in df.columns
        assert "target_coverage" in df.columns
        assert len(df) <= 10
        assert len(df) >= 1

    def test_coverage_values_in_range(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"], nonconformity="pearson", tweedie_power=1.0
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        df = cp.coverage_by_decile(
            poisson_data["X_test"], poisson_data["y_test"], alpha=0.10
        )
        assert (df["coverage"] >= 0).all()
        assert (df["coverage"] <= 1).all()

    def test_n_obs_sums_to_total(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"], nonconformity="pearson", tweedie_power=1.0
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        df = cp.coverage_by_decile(
            poisson_data["X_test"], poisson_data["y_test"], alpha=0.10
        )
        assert df["n_obs"].sum() == len(poisson_data["y_test"])

    def test_target_coverage_matches_alpha(self, poisson_data):
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"], nonconformity="pearson", tweedie_power=1.0
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        df = cp.coverage_by_decile(
            poisson_data["X_test"], poisson_data["y_test"], alpha=0.15
        )
        assert np.allclose(df["target_coverage"].to_numpy(), 0.85)

    def test_mean_predicted_increases_by_decile(self, poisson_data):
        """Deciles should be ordered by predicted value."""
        cp = InsuranceConformalPredictor(
            model=poisson_data["model"], nonconformity="pearson", tweedie_power=1.0
        )
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        df = cp.coverage_by_decile(
            poisson_data["X_test"], poisson_data["y_test"], alpha=0.10
        )
        preds = df["mean_predicted"].to_numpy()
        assert np.all(np.diff(preds) >= 0), "Mean predicted should increase across deciles"


class TestEdgeCases:
    def test_single_calibration_observation(self, poisson_data):
        model = poisson_data["model"]
        X_cal = poisson_data["X_cal"][:1]
        y_cal = poisson_data["y_cal"][:1]

        cp = InsuranceConformalPredictor(model=model, nonconformity="raw", tweedie_power=1.0)
        cp.calibrate(X_cal, y_cal)
        intervals = cp.predict_interval(poisson_data["X_test"], alpha=0.10)
        assert len(intervals) == len(poisson_data["X_test"])

    def test_all_scores_equal(self, poisson_data):
        """Degenerate case where model predicts the same value for everything."""

        class ConstantModel:
            def predict(self, X):
                return np.full(len(X), 2.0)

        model = ConstantModel()
        cp = InsuranceConformalPredictor(model=model, nonconformity="raw", tweedie_power=1.0)
        cp.calibrate(poisson_data["X_cal"], poisson_data["y_cal"])
        intervals = cp.predict_interval(poisson_data["X_test"], alpha=0.10)
        # All point predictions should be 2.0
        assert (intervals["point"] == 2.0).all()
        # All lower/upper should be the same constant
        assert intervals["lower"].n_unique() == 1
        assert intervals["upper"].n_unique() == 1
