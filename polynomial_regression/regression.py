"""Polynomial Regressor module supporting OLS and L2 Ridge regularization."""

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ModelFitDetails:
    """Details about fitted regression model."""

    coefficients: np.ndarray
    condition_number: float
    solver_used: str
    condition_warning: bool


class PolynomialRegressor:
    """Polynomial Regressor supporting Ordinary Least Squares (OLS) and L2 regularization (Ridge).

    Fits beta solving:
        (X^T X + lambda * R) beta = X^T y
    where R is identity matrix with R[0, 0] = 0.0 (no regularization on intercept).
    """

    def __init__(
        self,
        l2_lambda: float = 0.0,
        condition_warning_threshold: float = 1e12,
    ):
        if l2_lambda < 0.0:
            raise ValueError(
                f"L2 regularization lambda must be >= 0.0, got {l2_lambda}."
            )
        self.l2_lambda = l2_lambda
        self.condition_warning_threshold = condition_warning_threshold

        self.beta: np.ndarray | None = None
        self.fit_details: ModelFitDetails | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegressor":
        """Fits coefficient vector beta from design matrix X and target y.

        Args:
            X: 2D NumPy array of shape (N, num_features).
            y: 1D NumPy array of shape (N,).

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        N, num_features = X.shape

        if N < num_features:
            warnings.warn(
                f"Number of observations ({N}) is smaller than number of features ({num_features}). "
                "The system is underdetermined.",
                UserWarning,
            )

        # Compute condition number of design matrix X
        cond_num = float(np.linalg.cond(X))
        cond_warning = cond_num > self.condition_warning_threshold

        if cond_warning:
            warnings.warn(
                f"Design matrix condition number ({cond_num:.4e}) exceeds threshold "
                f"({self.condition_warning_threshold:.4e}). Model estimation may be unstable.",
                UserWarning,
            )

        # Form normal equations matrix: A = X^T X + lambda * R
        XT_X = X.T @ X
        XT_y = X.T @ y

        R = np.eye(num_features, dtype=np.float64)
        R[0, 0] = 0.0  # Crucial: Do NOT regularize intercept

        A = XT_X + self.l2_lambda * R

        solver_used = "solve"
        try:
            # First attempt solve
            beta = np.linalg.solve(A, XT_y)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if singular
            solver_used = "lstsq"
            beta, _, _, _ = np.linalg.lstsq(A, XT_y, rcond=None)

        self.beta = beta
        self.fit_details = ModelFitDetails(
            coefficients=beta,
            condition_number=cond_num,
            solver_used=solver_used,
            condition_warning=cond_warning,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values given design matrix X.

        Args:
            X: 2D NumPy array of shape (N, num_features).

        Returns:
            1D NumPy array of predictions.
        """
        if self.beta is None:
            raise RuntimeError(
                "Cannot predict with unfitted PolynomialRegressor. Call fit() first."
            )

        X = np.asarray(X, dtype=np.float64)
        predictions = X @ self.beta

        # Check for overflow / non-finite predictions
        if not np.all(np.isfinite(predictions)):
            warnings.warn(
                "Predictions contain non-finite values (NaN or Inf).", UserWarning
            )

        return predictions
