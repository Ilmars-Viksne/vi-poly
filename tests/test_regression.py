"""Tests for PolynomialRegressor and RegressionMetrics modules."""

import unittest
from unittest.mock import patch

import numpy as np
import pytest

from polynomial_regression.features import PolynomialFeatureTransformer
from polynomial_regression.metrics import RegressionMetrics
from polynomial_regression.regression import PolynomialRegressor


class TestRegression(unittest.TestCase):
    def test_ols_recovery_of_synthetic_coefficients(self):
        # Noise-free quadratic synthetic data: y = 3 + 2x - 0.5x^2
        x = np.linspace(-5, 5, 50)
        true_beta = np.array([3.0, 2.0, -0.5])
        y = true_beta[0] + true_beta[1] * x + true_beta[2] * (x**2)

        X = PolynomialFeatureTransformer(degree=2, scale_features=False).fit_transform(
            x
        )
        regressor = PolynomialRegressor(l2_lambda=0.0).fit(X, y)

        np.testing.assert_allclose(regressor.beta, true_beta, atol=1e-10)
        self.assertEqual(regressor.fit_details.solver_used, "lstsq_ols")
        self.assertTrue(regressor.fit_details.full_rank)
        self.assertEqual(regressor.fit_details.rank, 3)

    def test_ols_does_not_call_solve(self):
        x = np.linspace(-3, 3, 20)
        X = PolynomialFeatureTransformer(degree=2).fit_transform(x)
        y = 1.0 + x - x**2

        with patch(
            "numpy.linalg.solve",
            side_effect=AssertionError("np.linalg.solve should not be called"),
        ):
            regressor = PolynomialRegressor(l2_lambda=0.0).fit(X, y)

        self.assertEqual(regressor.fit_details.solver_used, "lstsq_ols")

    @pytest.mark.filterwarnings(
        "ignore:Design matrix solver system is rank deficient.*:UserWarning"
    )
    @pytest.mark.filterwarnings("ignore:Solver matrix condition number.*:UserWarning")
    def test_rank_deficient_ols(self):
        # Create rank-deficient matrix with identical columns
        X_singular = np.ones((10, 3), dtype=np.float64)
        y = np.ones(10, dtype=np.float64)

        with pytest.warns(UserWarning, match="rank deficient"):
            regressor = PolynomialRegressor(l2_lambda=0.0).fit(X_singular, y)

        self.assertEqual(regressor.fit_details.solver_used, "lstsq_ols")
        self.assertEqual(regressor.fit_details.rank, 1)
        self.assertFalse(regressor.fit_details.full_rank)
        self.assertTrue(np.all(np.isfinite(regressor.beta)))

    def test_ridge_augmented_formulation_matches_reference(self):
        x = np.linspace(-2, 2, 25)
        X = PolynomialFeatureTransformer(degree=3).fit_transform(x)
        y = 0.5 * x**3 - x + 2.0
        l2 = 0.5

        regressor = PolynomialRegressor(l2_lambda=l2).fit(X, y)
        self.assertEqual(regressor.fit_details.solver_used, "lstsq_augmented_ridge")

        # Reference calculation
        num_features = X.shape[1]
        sqrt_lambda = np.sqrt(l2)
        penalty = np.eye(num_features, dtype=np.float64)
        penalty[0, 0] = 0.0

        X_aug = np.vstack([X, sqrt_lambda * penalty])
        y_aug = np.concatenate([y, np.zeros(num_features, dtype=np.float64)])
        expected_beta, _, _, _ = np.linalg.lstsq(X_aug, y_aug, rcond=None)

        np.testing.assert_allclose(
            regressor.beta, expected_beta, rtol=1e-12, atol=1e-12
        )
        self.assertEqual(
            regressor.fit_details.solver_condition_number, float(np.linalg.cond(X_aug))
        )
        self.assertEqual(
            regressor.fit_details.condition_number,
            regressor.fit_details.solver_condition_number,
        )

    def test_ridge_intercept_not_regularized(self):
        x = np.linspace(-1, 1, 20)
        y = 50.0 + 2.0 * x

        X = PolynomialFeatureTransformer(degree=1, scale_features=False).fit_transform(
            x
        )
        regressor = PolynomialRegressor(l2_lambda=1e6).fit(X, y)

        np.testing.assert_allclose(regressor.beta[0], 50.0, atol=1e-2)
        self.assertLess(abs(regressor.beta[1]), 0.1)

    def test_fit_input_validation(self):
        regressor = PolynomialRegressor(l2_lambda=0.0)

        # 1D X
        with self.assertRaises(ValueError):
            regressor.fit(np.ones(10), np.ones(10))

        # 3D X
        with self.assertRaises(ValueError):
            regressor.fit(np.ones((10, 2, 2)), np.ones(10))

        # 2D y
        with self.assertRaises(ValueError):
            regressor.fit(np.ones((10, 2)), np.ones((10, 1)))

        # Mismatched lengths
        with self.assertRaises(ValueError):
            regressor.fit(np.ones((10, 2)), np.ones(9))

        # Empty array
        with self.assertRaises(ValueError):
            regressor.fit(np.empty((0, 2)), np.empty(0))

        # Zero column X
        with self.assertRaises(ValueError):
            regressor.fit(np.ones((10, 0)), np.ones(10))

        # NaN in X
        X_nan = np.ones((10, 2))
        X_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            regressor.fit(X_nan, np.ones(10))

        # Inf in y
        y_inf = np.ones(10)
        y_inf[0] = np.inf
        with self.assertRaises(ValueError):
            regressor.fit(np.ones((10, 2)), y_inf)

        # Negative lambda
        with self.assertRaises(ValueError):
            PolynomialRegressor(l2_lambda=-1.0)

        # Non-finite threshold
        with self.assertRaises(ValueError):
            PolynomialRegressor(condition_warning_threshold=np.nan)

    @pytest.mark.filterwarnings(
        "ignore:Design matrix solver system is rank deficient.*:UserWarning"
    )
    @pytest.mark.filterwarnings("ignore:Solver matrix condition number.*:UserWarning")
    def test_predict_input_validation(self):
        regressor = PolynomialRegressor().fit(np.ones((10, 2)), np.ones(10))

        # Wrong number of features
        with self.assertRaises(ValueError):
            regressor.predict(np.ones((10, 3)))

        # 1D array
        with self.assertRaises(ValueError):
            regressor.predict(np.ones(10))

        # Non-finite inputs
        X_nan = np.ones((5, 2))
        X_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            regressor.predict(X_nan)

    def test_predict_before_fit_raises_error(self):
        regressor = PolynomialRegressor()
        with self.assertRaises(RuntimeError):
            regressor.predict(np.ones((5, 2)))


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = RegressionMetrics.calculate(y_true, y_pred, num_predictors=1)
        self.assertAlmostEqual(metrics.mse, 0.0)
        self.assertAlmostEqual(metrics.rmse, 0.0)
        self.assertAlmostEqual(metrics.mae, 0.0)
        self.assertAlmostEqual(metrics.r_squared, 1.0)
        self.assertAlmostEqual(metrics.adjusted_r_squared, 1.0)

    def test_residual_sign_convention(self):
        y_true = np.array([10.0])
        y_pred = np.array([8.0])
        metrics = RegressionMetrics.calculate(y_true, y_pred)
        self.assertEqual(metrics.mae, 2.0)

    def test_constant_target_edge_case(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        metrics = RegressionMetrics.calculate(y_true, y_pred)
        self.assertEqual(metrics.r_squared, 1.0)

        y_pred_bad = np.array([4.0, 6.0, 5.0])
        metrics_bad = RegressionMetrics.calculate(y_true, y_pred_bad)
        self.assertEqual(metrics_bad.r_squared, 0.0)

    def test_metrics_input_validation(self):
        # Empty arrays
        with self.assertRaises(ValueError):
            RegressionMetrics.calculate(np.array([]), np.array([]))

        # Mismatched lengths
        with self.assertRaises(ValueError):
            RegressionMetrics.calculate(np.array([1.0, 2.0]), np.array([1.0]))

        # Non 1D
        with self.assertRaises(ValueError):
            RegressionMetrics.calculate(np.array([[1.0]]), np.array([[1.0]]))

        # Non finite
        with self.assertRaises(ValueError):
            RegressionMetrics.calculate(np.array([1.0, np.nan]), np.array([1.0, 2.0]))

        # Negative num_predictors
        with self.assertRaises(ValueError):
            RegressionMetrics.calculate(
                np.array([1.0, 2.0]), np.array([1.0, 2.0]), num_predictors=-1
            )


if __name__ == "__main__":
    unittest.main()
