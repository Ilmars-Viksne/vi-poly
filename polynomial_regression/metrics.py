"""Manual evaluation metrics for regression models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationMetrics:
    """Container for regression performance metrics."""

    mse: float
    rmse: float
    mae: float
    r_squared: float
    adjusted_r_squared: float | None = None


class RegressionMetrics:
    """Calculates evaluation metrics manually using NumPy."""

    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_predictors: int | None = None,
    ) -> EvaluationMetrics:
        """Calculates MSE, RMSE, MAE, R-squared, and optionally Adjusted R-squared.

        Residual is defined as: residual = actual - predicted.

        Args:
            y_true: 1D NumPy array of ground truth target values.
            y_pred: 1D NumPy array of predicted target values.
            num_predictors: Number of non-intercept predictor variables (degree d) for adjusted R2.

        Returns:
            EvaluationMetrics dataclass instance.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Shape mismatch: y_true len ({len(y_true)}) != y_pred len ({len(y_pred)})"
            )

        N = len(y_true)
        if N == 0:
            raise ValueError("Cannot calculate metrics on empty arrays.")

        residuals = y_true - y_pred

        mse = float(np.mean(residuals**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residuals)))

        # R-squared calculation
        # R2 = 1 - (SS_res / SS_tot)
        ss_res = float(np.sum(residuals**2))
        y_mean = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))

        if ss_tot < 1e-15:
            # Constant target edge case
            if ss_res < 1e-15:
                r_squared = 1.0  # Perfect prediction of constant target
            else:
                r_squared = 0.0  # Residual non-zero for constant target
        else:
            r_squared = float(1.0 - (ss_res / ss_tot))

        # Adjusted R-squared calculation
        # R2_adj = 1 - (1 - R2) * (N - 1) / (N - p - 1)
        adj_r_squared: float | None = None
        if num_predictors is not None:
            p = num_predictors
            if N > p + 1:
                adj_r_squared = float(1.0 - (1.0 - r_squared) * (N - 1) / (N - p - 1))

        return EvaluationMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
        )
