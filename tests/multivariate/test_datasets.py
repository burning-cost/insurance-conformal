"""
Tests for synthetic dataset generators and helper models.
"""

import numpy as np
import pytest

from insurance_conformal.multivariate.datasets import (
    make_motor_frequency_severity,
    make_home_multi_peril,
    make_fitted_models,
    _SimpleLinearModel,
    _SimpleExpModel,
)


class TestMakeMotorFrequencySeverity:
    def test_shapes(self):
        data = make_motor_frequency_severity(n=500, n_features=4)
        assert data["X"].shape == (500, 4)
        assert data["freq"].shape == (500,)
        assert data["sev"].shape == (500,)
        assert data["exposure"].shape == (500,)

    def test_freq_nonnegative(self):
        data = make_motor_frequency_severity(n=200)
        assert np.all(data["freq"] >= 0)

    def test_sev_nonneg(self):
        data = make_motor_frequency_severity(n=200)
        assert np.all(data["sev"] >= 0)

    def test_sev_zero_where_no_claims(self):
        data = make_motor_frequency_severity(n=500)
        zero_mask = data["zero_claim_mask"]
        assert np.all(data["sev"][zero_mask] == 0)

    def test_sev_positive_where_claims(self):
        data = make_motor_frequency_severity(n=1000)
        claim_mask = ~data["zero_claim_mask"]
        assert np.all(data["sev"][claim_mask] > 0)

    def test_exposure_in_range(self):
        data = make_motor_frequency_severity(n=200, exposure_range=(0.5, 0.5))
        np.testing.assert_allclose(data["exposure"], 0.5)

    def test_random_state_reproducibility(self):
        d1 = make_motor_frequency_severity(n=100, random_state=42)
        d2 = make_motor_frequency_severity(n=100, random_state=42)
        np.testing.assert_array_equal(d1["freq"], d2["freq"])
        np.testing.assert_array_equal(d1["sev"], d2["sev"])

    def test_different_seeds_different_data(self):
        d1 = make_motor_frequency_severity(n=100, random_state=1)
        d2 = make_motor_frequency_severity(n=100, random_state=2)
        assert not np.array_equal(d1["freq"], d2["freq"])

    def test_zero_claim_mask_boolean(self):
        data = make_motor_frequency_severity(n=200)
        assert data["zero_claim_mask"].dtype == bool

    def test_zero_claim_rate_reasonable(self):
        # At lambda ~ 0.08-0.12, zero-claim rate should be ~e^-lambda ~ 0.88-0.92
        # With exposure ~0.55 average, expected_lambda ~ 0.05
        data = make_motor_frequency_severity(n=2000, random_state=0)
        zero_rate = data["zero_claim_mask"].mean()
        assert 0.70 <= zero_rate <= 0.99, f"Unexpected zero claim rate: {zero_rate}"


class TestMakeHomeMultiPeril:
    def test_shapes(self):
        data = make_home_multi_peril(n=500)
        assert data["X"].shape == (500, 6)
        assert data["freq_flood"].shape == (500,)
        assert data["freq_fire"].shape == (500,)
        assert data["freq_sub"].shape == (500,)

    def test_sev_zero_where_no_claims(self):
        data = make_home_multi_peril(n=500)
        np.testing.assert_array_equal(
            data["sev_flood"][data["freq_flood"] == 0], 0.0
        )
        np.testing.assert_array_equal(
            data["sev_fire"][data["freq_fire"] == 0], 0.0
        )

    def test_subsidence_rarest(self):
        data = make_home_multi_peril(n=5000, random_state=0)
        # Subsidence should be rarest
        rate_flood = (data["freq_flood"] > 0).mean()
        rate_sub = (data["freq_sub"] > 0).mean()
        assert rate_sub < rate_flood

    def test_zero_claim_mask_shape(self):
        data = make_home_multi_peril(n=100)
        assert data["zero_claim_mask"].shape == (100, 3)
        assert data["zero_claim_mask"].dtype == bool

    def test_nonnegative_severities(self):
        data = make_home_multi_peril(n=200)
        assert np.all(data["sev_flood"] >= 0)
        assert np.all(data["sev_fire"] >= 0)
        assert np.all(data["sev_sub"] >= 0)


class TestSimpleModels:
    def test_linear_model_fit_predict(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = X @ [1.0, -0.5, 0.2] + rng.normal(0, 0.1, 100)
        model = _SimpleLinearModel().fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (100,)
        # Should fit well (low noise)
        assert np.corrcoef(y, pred)[0, 1] > 0.9

    def test_linear_model_positive(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 2))
        y = np.abs(rng.normal(1, 5, 50))
        model = _SimpleLinearModel(positive=True).fit(X, y)
        pred = model.predict(rng.standard_normal((20, 2)))
        assert np.all(pred > 0)

    def test_exp_model_fit_predict(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((100, 3))
        y = np.exp(X @ [0.5, -0.3, 0.1]) * rng.lognormal(0, 0.2, 100)
        model = _SimpleExpModel().fit(X, y)
        pred = model.predict(X)
        assert np.all(pred > 0)
        assert pred.shape == (100,)

    def test_exp_model_predictions_positive(self):
        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((50, 2))
        y = np.exp(X_train @ [1.0, -0.5]) + 0.01
        model = _SimpleExpModel().fit(X_train, y)
        X_test = rng.standard_normal((20, 2))
        pred = model.predict(X_test)
        assert np.all(pred > 0)


class TestMakeFittedModels:
    def test_returns_dict(self):
        data = make_motor_frequency_severity(n=300)
        models = make_fitted_models(data)
        assert "frequency" in models
        assert "severity" in models

    def test_models_predict(self):
        data = make_motor_frequency_severity(n=300)
        models = make_fitted_models(data)
        X = data["X"][:10]
        freq_pred = models["frequency"].predict(X)
        sev_pred = models["severity"].predict(X)
        assert freq_pred.shape == (10,)
        assert sev_pred.shape == (10,)

    def test_frequency_predictions_positive(self):
        data = make_motor_frequency_severity(n=500)
        models = make_fitted_models(data)
        pred = models["frequency"].predict(data["X"])
        assert np.all(pred > 0)
