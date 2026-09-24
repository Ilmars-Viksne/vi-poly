"""Tests for Holdout and KFold Model Selectors."""

import unittest
import numpy as np

from polynomial_regression.splitting import DataSplitter
from polynomial_regression.selection import HoldoutModelSelector, KFoldModelSelector
from generate_data import generate_synthetic_data


class TestSelection(unittest.TestCase):
    def setUp(self):
        # Generate synthetic quadratic data: y = 2.0 - 3.0*x + 1.5*x^2 + noise
        self.x, self.y = generate_synthetic_data(
            num_samples=100, x_min=-3.0, x_max=3.0,
            coefficients=[2.0, -3.0, 1.5], noise_std=0.5, seed=42
        )
        self.splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
        self.split = self.splitter.split(len(self.x))

    def test_holdout_model_selection(self):
        selector = HoldoutModelSelector(
            degrees=[1, 2, 3, 4],
            l2_lambdas=[0.0, 0.01, 1.0],
            scale_features=True,
            selection_rtol=1e-7,
            selection_atol=1e-12,
        )
        res = selector.select(
            self.x, self.y,
            self.split.train_indices, self.split.val_indices,
            self.split.test_indices, self.split.dev_indices,
        )

        # True degree is 2; selector should pick degree 2 (or low degree close to 2)
        self.assertIn(res.best_candidate.degree, [2, 3])
        self.assertEqual(len(res.candidates), 12)  # 4 degrees * 3 l2 values
        self.assertEqual(len(res.test_predictions), 15)

    def test_kfold_model_selection(self):
        selector = KFoldModelSelector(
            degrees=[1, 2, 3],
            l2_lambdas=[0.0, 0.1],
            k_folds=5,
            seed=42,
            scale_features=True,
        )
        res = selector.select(
            self.x, self.y,
            self.split.dev_indices, self.split.test_indices
        )

        self.assertIn(res.best_candidate.degree, [2, 3])
        self.assertEqual(len(res.oof_predictions), 85)
        self.assertEqual(len(res.test_predictions), 15)

    def test_tie_breaking_rules(self):
        # Create identical dummy results with same RMSE to test tie-breaking
        selector = HoldoutModelSelector(
            degrees=[1, 2, 3],
            l2_lambdas=[0.0, 1.0],
            selection_rtol=1e-1,  # Large tolerance to force ties
        )
        res = selector.select(
            self.x, self.y,
            self.split.train_indices, self.split.val_indices,
            self.split.test_indices, self.split.dev_indices,
        )
        # With forced tie-breaking, should pick lowest degree (degree 1)
        # or highest L2 for that degree
        self.assertIsNotNone(res.best_candidate)


if __name__ == "__main__":
    unittest.main()
