"""Polynomial Regressor module supporting OLS and L2 Ridge regularization."""

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelFitDetails:
    """Details about fitted regression model."""

    coefficients: np.ndarray
    condition_number: float
    solver_used: str
    condition_warning: bool
    rank: int
    full_rank: bool
    design_condition_number: float
    solver_condition_number: float


class PolynomialRegressor:
    """Polynomial Regressor supporting Ordinary Least Squares (OLS) and L2 regularization (Ridge).

    Fits beta solving:
        OLS (l2_lambda == 0): np.linalg.lstsq(X, y)
        Ridge (l2_lambda > 0): np.linalg.lstsq(X_aug, y_aug) with unregularized intercept.
    """

    def __init__(
        self,
        l2_lambda: float = 0.0,
        condition_warning_threshold: float = 1e12,
    ):
        try:
            l2_val = float(l2_lambda)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"L2 lambda must be numeric, got {l2_lambda}.") from exc

        if not np.isfinite(l2_val) or l2_val < 0.0:
            raise ValueError(
                f"L2 regularization lambda must be finite and >= 0.0, got {l2_lambda}."
            )

        try:
            thresh_val = float(condition_warning_threshold)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Condition warning threshold must be numeric, got {condition_warning_threshold}."
            ) from exc

        if not np.isfinite(thresh_val) or thresh_val <= 0.0:
            raise ValueError(
                f"Condition warning threshold must be finite and strictly positive (> 0.0), got {condition_warning_threshold}."
            )

        self.l2_lambda = l2_val
        self.condition_warning_threshold = thresh_val

        self.beta: np.ndarray | None = None
        self.fit_details: ModelFitDetails | None = None

    def _validate_fit_inputs(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validates input matrices for fit()."""
        try:
            X_arr = np.asarray(X, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("X must be numeric.") from exc

        try:
            y_arr = np.asarray(y, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("y must be numeric.") from exc

        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be a two-dimensional array; received shape {X_arr.shape}."
            )

        if y_arr.ndim != 1:
            raise ValueError(
                f"y must be a one-dimensional array; received shape {y_arr.shape}."
            )

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must contain the same number of observations; received {X_arr.shape[0]} and {y_arr.shape[0]}."
            )

        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one observation.")

        if X_arr.shape[1] == 0:
            raise ValueError("X must contain at least one feature column.")

        if not np.all(np.isfinite(X_arr)):
            non_finite_count = int(np.sum(~np.isfinite(X_arr)))
            raise ValueError(f"X contains {non_finite_count} non-finite value(s).")

        if not np.all(np.isfinite(y_arr)):
            non_finite_count = int(np.sum(~np.isfinite(y_arr)))
            raise ValueError(f"y contains {non_finite_count} non-finite value(s).")

        return X_arr, y_arr

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegressor":
        """Fits coefficient vector beta from design matrix X and target y.

        Args:
            X: 2D NumPy array of shape (N, num_features).
            y: 1D NumPy array of shape (N,).

        Returns:
            self
        """
        X_arr, y_arr = self._validate_fit_inputs(X, y)
        N, num_features = X_arr.shape

        if N < num_features:
            warnings.warn(
                f"Number of observations ({N}) is smaller than number of features ({num_features}). "
                "The system is underdetermined.",
                UserWarning,
            )

        design_cond = float(np.linalg.cond(X_arr))

        if self.l2_lambda == 0.0:
            solver_used = "lstsq_ols"
            beta, _, rank, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
            solver_cond = design_cond
        else:
            solver_used = "lstsq_augmented_ridge"
            sqrt_lambda = np.sqrt(self.l2_lambda)
            penalty = np.eye(num_features, dtype=np.float64)
            penalty[0, 0] = 0.0  # Do NOT regularize intercept

            X_aug = np.vstack([X_arr, sqrt_lambda * penalty])
            y_aug = np.concatenate([y_arr, np.zeros(num_features, dtype=np.float64)])

            beta, _residuals, rank, _singular_values = np.linalg.lstsq(
                X_aug, y_aug, rcond=None
            )
            solver_cond = float(np.linalg.cond(X_aug))

        rank_val = int(rank)
        full_rank = bool(rank_val == num_features)

        if not full_rank:
            warnings.warn(
                f"Design matrix solver system is rank deficient (effective rank {rank_val} < {num_features} features).",
                UserWarning,
            )

        cond_warning = solver_cond > self.condition_warning_threshold
        if cond_warning:
            warnings.warn(
                f"Solver matrix condition number ({solver_cond:.4e}) exceeds threshold "
                f"({self.condition_warning_threshold:.4e}). Model estimation may be unstable.",
                UserWarning,
            )

        if not np.all(np.isfinite(beta)):
            raise FloatingPointError(
                "Fitted model coefficients contain non-finite values."
            )

        self.beta = beta
        self.fit_details = ModelFitDetails(
            coefficients=beta,
            condition_number=solver_cond,
            solver_used=solver_used,
            condition_warning=cond_warning,
            rank=rank_val,
            full_rank=full_rank,
            design_condition_number=design_cond,
            solver_condition_number=solver_cond,
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

        try:
            X_arr = np.asarray(X, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("Prediction matrix X must be numeric.") from exc

        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be a two-dimensional array; received shape {X_arr.shape}."
            )

        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one row for prediction.")

        if X_arr.shape[1] != len(self.beta):
            raise ValueError(
                f"Feature dimension mismatch: model expected {len(self.beta)} features, got {X_arr.shape[1]}."
            )

        if not np.all(np.isfinite(X_arr)):
            non_finite_count = int(np.sum(~np.isfinite(X_arr)))
            raise ValueError(f"X contains {non_finite_count} non-finite value(s).")

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                predictions = X_arr @ self.beta
            except FloatingPointError as exc:
                raise FloatingPointError(
                    "Model prediction computation overflowed."
                ) from exc

        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError(
                "Predictions contain non-finite values (NaN or Inf)."
            )

        return predictions
