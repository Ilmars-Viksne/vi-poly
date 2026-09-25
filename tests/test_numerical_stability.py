"""Numerical stability and input validation unit tests."""

import unittest

import numpy as np

from polynomial_regression.features import (
    PolynomialFeatureTransformer,
)
from polynomial_regression.selection import HoldoutModelSelector, KFoldModelSelector


class TestFeatureTransformerNumericalStability(unittest.TestCase):
    def test_default_degree_construction(self):
        x = np.linspace(-2.0, 2.0, 20)
        transformer = PolynomialFeatureTransformer(degree=10)
        X = transformer.fit_transform(x)
        self.assertEqual(X.shape, (20, 11))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_degree_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        transformer = PolynomialFeatureTransformer(degree=0)
        X = transformer.fit_transform(x)
        np.testing.assert_array_equal(X, np.ones((3, 1)))

    def test_constant_zero_input(self):
        x = np.zeros(10)
        transformer = PolynomialFeatureTransformer(degree=5, scale_features=True)
        X = transformer.fit_transform(x)
        self.assertTrue(np.all(np.isfinite(X)))
        np.testing.assert_array_equal(X[:, 0], np.ones(10))

    def test_input_abs_le_one(self):
        x = np.linspace(-1.0, 1.0, 30)
        transformer = PolynomialFeatureTransformer(degree=50)
        X = transformer.fit_transform(x)
        self.assertTrue(np.all(np.isfinite(X)))

    def test_degree_exceeds_max_degree_raises_value_error(self):
        with self.assertRaises(ValueError):
            PolynomialFeatureTransformer(degree=51, max_degree=50)

    def test_negative_degree_raises_value_error(self):
        with self.assertRaises(ValueError):
            PolynomialFeatureTransformer(degree=-1)

    def test_larger_explicit_max_degree(self):
        x = np.linspace(-0.5, 0.5, 10)
        transformer = PolynomialFeatureTransformer(degree=60, max_degree=100)
        X = transformer.fit_transform(x)
        self.assertEqual(X.shape, (10, 61))

    def test_preflight_overflow_check_raises_floating_point_error(self):
        x = np.array([1e10])
        # 1e10 ** 50 = 1e500 > float max (~1.8e308)
        transformer = PolynomialFeatureTransformer(degree=50)
        with self.assertRaises(FloatingPointError) as cm:
            transformer.transform(x)
        self.assertIn("overflow estimated", str(cm.exception))

    def test_scaling_enabled_does_not_bypass_overflow(self):
        x = np.array([1e10, 1e10])
        transformer = PolynomialFeatureTransformer(degree=50, scale_features=True)
        with self.assertRaises(FloatingPointError):
            transformer.fit(x)

    def test_invalid_x_fit_transform_raises_value_error(self):
        transformer = PolynomialFeatureTransformer(degree=2)

        # 2D x
        with self.assertRaises(ValueError):
            transformer.fit(np.ones((5, 2)))

        # Empty x
        with self.assertRaises(ValueError):
            transformer.fit(np.array([]))

        # NaN x
        with self.assertRaises(ValueError):
            transformer.fit(np.array([1.0, np.nan]))

        # Inf x
        with self.assertRaises(ValueError):
            transformer.fit(np.array([1.0, np.inf]))

    def test_coefficient_conversion_overflow_raises_floating_point_error(self):
        x = np.linspace(100.0, 200.0, 20)
        transformer = PolynomialFeatureTransformer(degree=10, scale_features=True)
        transformer.fit(x)

        # Create massive scaled coefficients that cause conversion overflow
        beta_scaled = np.ones(11) * 1e300
        with self.assertRaises(FloatingPointError):
            transformer.convert_coefficients_to_original_basis(beta_scaled)


class TestSelectorInputValidation(unittest.TestCase):
    def test_selector_empty_hyperparameters_raises_value_error(self):
        with self.assertRaises(ValueError):
            HoldoutModelSelector(degrees=[], l2_lambdas=[0.0])

        with self.assertRaises(ValueError):
            KFoldModelSelector(degrees=[1], l2_lambdas=[])

    def test_selector_degree_out_of_bounds_raises_value_error(self):
        with self.assertRaises(ValueError):
            HoldoutModelSelector(degrees=[55], l2_lambdas=[0.0], max_degree=50)

    def test_selector_overlapping_indices_raises_value_error(self):
        x = np.linspace(-1, 1, 20)
        y = x.copy()
        selector = HoldoutModelSelector(degrees=[1], l2_lambdas=[0.0])

        # train and val overlap
        with self.assertRaises(ValueError):
            selector.select(
                x,
                y,
                np.array([0, 1, 2]),
                np.array([2, 3, 4]),
                np.array([5, 6]),
                np.array([0, 1, 2, 3, 4]),
            )


if __name__ == "__main__":
    unittest.main()
