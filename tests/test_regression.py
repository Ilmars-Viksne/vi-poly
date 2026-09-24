"""Tests for PolynomialRegressor and RegressionMetrics modules."""

import unittest

import numpy as np

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

    def test_ridge_intercept_not_regularized(self):
        # Even with huge lambda, intercept should adapt to shift in y
        x = np.linspace(-1, 1, 20)
        y = 50.0 + 2.0 * x  # Shifted mean y = 50

        X = PolynomialFeatureTransformer(degree=1, scale_features=False).fit_transform(
            x
        )
        # Large L2 lambda will shrink slope x coefficient, but intercept should remain ~50.0
        regressor = PolynomialRegressor(l2_lambda=1e6).fit(X, y)

        np.testing.assert_allclose(regressor.beta[0], 50.0, atol=1e-2)
        # Non-intercept beta[1] shrunk close to zero
        self.assertLess(abs(regressor.beta[1]), 0.1)

    def test_predict_before_fit_raises_error(self):
        regressor = PolynomialRegressor()
        with self.assertRaises(RuntimeError):
            regressor.predict(np.ones((5, 2)))

    def test_singular_matrix_fallback(self):
        # Create design matrix with identical duplicate columns
        X_singular = np.ones((10, 3))
        y = np.ones(10)

        regressor = PolynomialRegressor(l2_lambda=0.0)
        # Should fit successfully using fallback solver
        regressor.fit(X_singular, y)
        self.assertEqual(regressor.fit_details.solver_used, "lstsq")


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
        # residual = actual - predicted = 10 - 8 = 2
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


if __name__ == "__main__":
    unittest.main()
