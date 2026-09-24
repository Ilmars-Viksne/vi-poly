"""Tests for DataSplitter and KFoldSplitter modules."""

import unittest

import numpy as np

from polynomial_regression.splitting import DataSplitter, KFoldSplitter


class TestDataSplitter(unittest.TestCase):
    def test_valid_holdout_split(self):
        splitter = DataSplitter(
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        num_samples = 100
        split = splitter.split(num_samples)

        self.assertEqual(len(split.train_indices), 70)
        self.assertEqual(len(split.val_indices), 15)
        self.assertEqual(len(split.test_indices), 15)
        self.assertEqual(len(split.dev_indices), 85)

        # Ensure no overlap
        all_indices = np.concatenate(
            [split.train_indices, split.val_indices, split.test_indices]
        )
        self.assertEqual(len(set(all_indices)), num_samples)
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(num_samples))

    def test_reproducibility(self):
        s1 = DataSplitter(seed=42).split(100)
        s2 = DataSplitter(seed=42).split(100)
        s3 = DataSplitter(seed=99).split(100)

        np.testing.assert_array_equal(s1.train_indices, s2.train_indices)
        np.testing.assert_array_equal(s1.val_indices, s2.val_indices)
        np.testing.assert_array_equal(s1.test_indices, s2.test_indices)

        self.assertFalse(np.array_equal(s1.train_indices, s3.train_indices))

    def test_invalid_ratios_sum(self):
        with self.assertRaises(ValueError):
            DataSplitter(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_negative_ratio(self):
        with self.assertRaises(ValueError):
            DataSplitter(train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)


class TestKFoldSplitter(unittest.TestCase):
    def test_valid_kfold_split(self):
        dev_indices = np.arange(85)
        k_splitter = KFoldSplitter(k=5, seed=42)
        folds = k_splitter.split(dev_indices)

        self.assertEqual(len(folds), 5)

        # Check fold sizes with remainder distribution (85 % 5 == 0 -> 17 each)
        all_val_indices = []
        for fold in folds:
            self.assertEqual(len(fold.val_indices), 17)
            self.assertEqual(len(fold.train_indices), 68)
            all_val_indices.extend(fold.val_indices)

            # Check no overlap within fold
            intersection = set(fold.train_indices).intersection(set(fold.val_indices))
            self.assertEqual(len(intersection), 0)

        # Check each observation appears in val set exactly once
        self.assertEqual(len(set(all_val_indices)), 85)
        np.testing.assert_array_equal(np.sort(all_val_indices), dev_indices)

    def test_kfold_with_remainder(self):
        dev_indices = np.arange(88)  # 88 / 5 = 17 r 3 -> 18, 18, 18, 17, 17
        k_splitter = KFoldSplitter(k=5, seed=42)
        folds = k_splitter.split(dev_indices)

        sizes = [len(f.val_indices) for f in folds]
        self.assertEqual(sizes, [18, 18, 18, 17, 17])

    def test_invalid_k(self):
        k_splitter = KFoldSplitter(k=1)
        with self.assertRaises(ValueError):
            k_splitter.split(np.arange(10))

        k_splitter_high = KFoldSplitter(k=15)
        with self.assertRaises(ValueError):
            k_splitter_high.split(np.arange(10))


if __name__ == "__main__":
    unittest.main()
