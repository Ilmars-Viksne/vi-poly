"""Data splitting utilities for Holdout and K-Fold cross-validation."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class HoldoutSplit:
    """Container for indices of holdout dataset splitting."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    dev_indices: np.ndarray


@dataclass
class FoldSplit:
    """Container for indices of a single K-Fold split."""

    fold_index: int
    train_indices: np.ndarray
    val_indices: np.ndarray


class DataSplitter:
    """Splits dataset indices into train, validation, and test subsets."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.seed = int(seed)

        self._validate_ratios()

    def _validate_ratios(self) -> None:
        if not (
            np.isfinite(self.train_ratio)
            and np.isfinite(self.val_ratio)
            and np.isfinite(self.test_ratio)
        ):
            raise ValueError("Split ratios must be finite numbers.")

        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError(
                f"All split ratios must be positive (> 0). Got: "
                f"train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}"
            )
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"Split ratios must sum to 1.0 within tolerance. Sum = {total:.6f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

    def split(self, num_samples: int) -> HoldoutSplit:
        """Generates holdout split indices for a dataset of size num_samples.

        Args:
            num_samples: Total number of observations.

        Returns:
            HoldoutSplit with shuffled train, validation, test, and dev indices.
        """
        if not isinstance(num_samples, (int, np.integer)) or num_samples < 3:
            raise ValueError(
                f"Number of samples ({num_samples}) is too small or invalid for train-val-test split."
            )

        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(num_samples)

        num_test = round(num_samples * self.test_ratio)
        num_val = round(num_samples * self.val_ratio)
        num_train = num_samples - num_test - num_val

        if num_train < 1 or num_val < 1 or num_test < 1:
            raise ValueError(
                f"Sample size {num_samples} is too small to populate all splits with ratios "
                f"({self.train_ratio}, {self.val_ratio}, {self.test_ratio}). "
                f"Calculated counts: train={num_train}, val={num_val}, test={num_test}."
            )

        train_idx = shuffled_indices[:num_train]
        val_idx = shuffled_indices[num_train : num_train + num_val]
        test_idx = shuffled_indices[num_train + num_val :]

        dev_idx = shuffled_indices[: num_train + num_val]

        assert len(train_idx) + len(val_idx) + len(test_idx) == num_samples
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(set(val_idx).intersection(set(test_idx))) == 0

        return HoldoutSplit(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            dev_indices=dev_idx,
        )


class KFoldSplitter:
    """Splits development dataset indices into K non-overlapping folds."""

    def __init__(self, k: int = 5, seed: int = 42):
        self.k = k
        self.seed = seed

    def split(self, dev_indices: np.ndarray) -> list[FoldSplit]:
        """Splits dev_indices into K folds.

        Args:
            dev_indices: 1D array of development indices to split into folds.

        Returns:
            List of FoldSplit objects, one per fold.
        """
        if not isinstance(self.k, (int, np.integer)) or self.k < 2:
            raise ValueError(
                f"Number of folds K must be an integer >= 2, got {self.k}."
            )

        try:
            dev_arr = np.asarray(dev_indices, dtype=np.int64)
        except (ValueError, TypeError) as exc:
            raise ValueError("dev_indices must be integer array.") from exc

        if dev_arr.ndim != 1:
            raise ValueError(
                f"dev_indices must be a 1D array; received shape {dev_arr.shape}."
            )

        if dev_arr.size == 0:
            raise ValueError("dev_indices must not be empty.")

        if len(set(dev_arr)) != len(dev_arr):
            raise ValueError("dev_indices must not contain duplicate values.")

        num_samples = len(dev_arr)
        if self.k > num_samples:
            raise ValueError(
                f"Number of folds K ({self.k}) cannot exceed number of development samples ({num_samples})."
            )

        rng = np.random.default_rng(self.seed)
        shuffled_dev = rng.permutation(dev_arr)

        base_size = num_samples // self.k
        remainder = num_samples % self.k

        folds_val_indices: list[np.ndarray] = []
        start = 0
        for i in range(self.k):
            fold_size = base_size + (1 if i < remainder else 0)
            end = start + fold_size
            val_idx = shuffled_dev[start:end]
            folds_val_indices.append(val_idx)
            start = end

        fold_splits: list[FoldSplit] = []
        for i in range(self.k):
            val_idx = folds_val_indices[i]
            train_folds = [folds_val_indices[j] for j in range(self.k) if j != i]
            train_idx = (
                np.concatenate(train_folds) if train_folds else np.array([], dtype=int)
            )

            assert len(set(train_idx).intersection(set(val_idx))) == 0
            assert len(train_idx) + len(val_idx) == num_samples

            fold_splits.append(
                FoldSplit(
                    fold_index=i,
                    train_indices=train_idx,
                    val_indices=val_idx,
                )
            )

        return fold_splits
