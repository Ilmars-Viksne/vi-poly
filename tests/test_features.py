"""Tests for PolynomialFeatureTransformer module."""

import unittest
import numpy as np

from polynomial_regression.features import PolynomialFeatureTransformer


class TestPolynomialFeatureTransformer(unittest.TestCase):
    def test_unscaled_design_matrix(self):
        x = np.array([1.0, 2.0, 3.0])
        transformer = PolynomialFeatureTransformer(degree=3, scale_features=False)
        X = transformer.transform(x)

        expected = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0, 8.0],
            [1.0, 3.0, 9.0, 27.0],
        ])
        np.testing.assert_allclose(X, expected)

    def test_scaled_design_matrix(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transformer = PolynomialFeatureTransformer(degree=2, scale_features=True)
        X_scaled = transformer.fit_transform(x)

        # Intercept column is unscaled 1.0s
        np.testing.assert_allclose(X_scaled[:, 0], 1.0)

        # Non-intercept columns have mean 0 and std 1
        np.testing.assert_allclose(np.mean(X_scaled[:, 1]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(X_scaled[:, 1]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.mean(X_scaled[:, 2]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(X_scaled[:, 2]), 1.0, atol=1e-12)

    def test_prevention_of_data_leakage(self):
        x_train = np.array([1.0, 2.0, 3.0])
        x_test = np.array([10.0, 20.0])

        transformer = PolynomialFeatureTransformer(degree=1, scale_features=True)
        transformer.fit(x_train)

        # Transform test data using TRAIN statistics
        X_test_scaled = transformer.transform(x_test)

        mean_train = np.mean(x_train)
        std_train = np.std(x_train)

        expected_x_test_col1 = (x_test - mean_train) / std_train
        np.testing.assert_allclose(X_test_scaled[:, 1], expected_x_test_col1)

    def test_coefficient_conversion_and_verification(self):
        x = np.linspace(-3, 3, 50)
        transformer = PolynomialFeatureTransformer(degree=3, scale_features=True)
        transformer.fit(x)

        # Arbitrary scaled coefficients: beta_0, beta_1, beta_2, beta_3
        beta_scaled = np.array([10.5, -2.3, 4.1, 0.8])

        beta_orig = transformer.convert_coefficients_to_original_basis(beta_scaled)

        # Verification must pass without error
        is_valid = PolynomialFeatureTransformer.verify_coefficient_conversion(
            x, transformer, beta_scaled, beta_orig, rtol=1e-10, atol=1e-12
        )
        self.assertTrue(is_valid)

    def test_constant_feature_handling(self):
        x_constant = np.array([2.0, 2.0, 2.0, 2.0])
        transformer = PolynomialFeatureTransformer(degree=2, scale_features=True)
        X_scaled = transformer.fit_transform(x_constant)

        # Constant feature x^1 and x^2 should scale to 0.0
        np.testing.assert_allclose(X_scaled[:, 1], 0.0)
        np.testing.assert_allclose(X_scaled[:, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
