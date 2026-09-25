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
        try:
            y_true_arr = np.asarray(y_true, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("y_true must be numeric.") from exc

        try:
            y_pred_arr = np.asarray(y_pred, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("y_pred must be numeric.") from exc

        if y_true_arr.ndim != 1:
            raise ValueError(
                f"y_true must be a 1D array; received shape {y_true_arr.shape}."
            )

        if y_pred_arr.ndim != 1:
            raise ValueError(
                f"y_pred must be a 1D array; received shape {y_pred_arr.shape}."
            )

        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                f"Shape mismatch: y_true len ({len(y_true_arr)}) != y_pred len ({len(y_pred_arr)})"
            )

        N = len(y_true_arr)
        if N == 0:
            raise ValueError("Cannot calculate regression metrics for empty arrays.")

        if not np.all(np.isfinite(y_true_arr)):
            non_finite_count = int(np.sum(~np.isfinite(y_true_arr)))
            raise ValueError(f"y_true contains {non_finite_count} non-finite value(s).")

        if not np.all(np.isfinite(y_pred_arr)):
            non_finite_count = int(np.sum(~np.isfinite(y_pred_arr)))
            raise ValueError(f"y_pred contains {non_finite_count} non-finite value(s).")

        if num_predictors is not None and (
            not isinstance(num_predictors, (int, np.integer)) or num_predictors < 0
        ):
            raise ValueError(
                f"num_predictors must be a non-negative integer; got {num_predictors}."
            )

        residuals = y_true_arr - y_pred_arr

        mse = float(np.mean(residuals**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residuals)))

        ss_res = float(np.sum(residuals**2))
        y_mean = float(np.mean(y_true_arr))
        ss_tot = float(np.sum((y_true_arr - y_mean) ** 2))

        if ss_tot < 1e-15:
            if ss_res < 1e-15:
                r_squared = 1.0
            else:
                r_squared = 0.0
        else:
            r_squared = float(1.0 - (ss_res / ss_tot))

        adj_r_squared: float | None = None
        if num_predictors is not None:
            p = int(num_predictors)
            if N > p + 1:
                adj_r_squared = float(1.0 - (1.0 - r_squared) * (N - 1) / (N - p - 1))

        if not np.all(np.isfinite([mse, rmse, mae, r_squared])):
            raise FloatingPointError("Calculated metrics contain non-finite values.")

        if adj_r_squared is not None and not np.isfinite(adj_r_squared):
            raise FloatingPointError("Calculated adjusted R-squared is non-finite.")

        return EvaluationMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
        )
