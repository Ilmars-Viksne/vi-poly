"""Polynomial Regressor module supporting OLS, L1 (Lasso), and L2 (Ridge) regularization."""

import warnings
from dataclasses import dataclass

import numpy as np

REGULARIZATION_NONE = "none"
REGULARIZATION_L1 = "l1"
REGULARIZATION_L2 = "l2"

SUPPORTED_REGULARIZATIONS = {
    REGULARIZATION_NONE,
    REGULARIZATION_L1,
    REGULARIZATION_L2,
}

DEFAULT_REGULARIZATION_STRENGTHS = [
    0.0,
    1e-6,
    1e-4,
    1e-2,
    0.1,
    1.0,
    10.0,
    100.0,
]

DEFAULT_L1_MAX_ITERATIONS = 10_000
DEFAULT_L1_TOLERANCE = 1e-8
DEFAULT_L1_INITIALIZATION = "zeros"


def soft_threshold(value: float, threshold: float) -> float:
    """Soft-thresholding operator S(value, threshold).

    S(v, t) = sign(v) * max(|v| - t, 0)
    """
    try:
        val = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Value must be numeric, got {value}.") from exc

    try:
        thresh = float(threshold)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Threshold must be numeric, got {threshold}.") from exc

    if not np.isfinite(val):
        raise ValueError("Value for soft_threshold must be finite.")

    if not np.isfinite(thresh) or thresh < 0.0:
        raise ValueError(
            f"Threshold for soft_threshold must be finite and non-negative (>= 0.0), got {threshold}."
        )

    if val > thresh:
        return val - thresh
    if val < -thresh:
        return val + thresh
    return 0.0


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
    requested_regularization: str
    effective_regularization: str
    regularization_strength: float
    iterations: int | None = None
    converged: bool | None = None
    final_objective: float | None = None
    max_coefficient_change: float | None = None
    nonzero_coefficient_count: int = 0
    l1_initialization: str | None = None


class PolynomialRegressor:
    """Polynomial Regressor supporting OLS, L1 (Lasso), and L2 (Ridge) regularization.

    Fits beta solving:
        OLS (regularization='none' or strength==0): np.linalg.lstsq(X, y)
        Ridge (regularization='l2' and strength>0): np.linalg.lstsq(X_aug, y_aug) with unregularized intercept.
        Lasso (regularization='l1' and strength>0): cyclic coordinate descent with unregularized intercept.
    """

    def __init__(
        self,
        regularization: str = REGULARIZATION_NONE,
        regularization_strength: float = 0.0,
        l2_lambda: float | None = None,
        l1_max_iterations: int = DEFAULT_L1_MAX_ITERATIONS,
        l1_tolerance: float = DEFAULT_L1_TOLERANCE,
        l1_initialization: str = DEFAULT_L1_INITIALIZATION,
        condition_warning_threshold: float = 1e12,
        coefficient_zero_tolerance: float = 1e-12,
    ):
        if l2_lambda is not None:
            reg_type = REGULARIZATION_L2
            reg_strength = l2_lambda
        else:
            reg_type = regularization
            reg_strength = regularization_strength

        if not isinstance(reg_type, str):
            raise TypeError(
                f"Regularization type must be a string, got {type(reg_type).__name__}."
            )

        norm_reg = reg_type.strip().lower()
        if norm_reg not in SUPPORTED_REGULARIZATIONS:
            raise ValueError(
                f"Unsupported regularization type '{reg_type}'. Supported choices: {sorted(SUPPORTED_REGULARIZATIONS)}."
            )

        try:
            strength_val = float(reg_strength)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Regularization strength must be numeric, got {reg_strength}."
            ) from exc

        if not np.isfinite(strength_val) or strength_val < 0.0:
            raise ValueError(
                f"Regularization strength must be finite and >= 0.0, got {reg_strength}."
            )

        if norm_reg == REGULARIZATION_NONE and strength_val != 0.0:
            raise ValueError(
                f"Regularization strength must be 0.0 for 'none' regularization; received {strength_val}."
            )

        try:
            max_iters_val = int(l1_max_iterations)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"L1 max iterations must be an integer, got {l1_max_iterations}."
            ) from exc

        if max_iters_val < 1:
            raise ValueError(
                f"L1 maximum iterations must be a positive integer (>= 1); received {l1_max_iterations}."
            )

        try:
            tol_val = float(l1_tolerance)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"L1 tolerance must be numeric, got {l1_tolerance}."
            ) from exc

        if not np.isfinite(tol_val) or tol_val <= 0.0:
            raise ValueError(
                f"L1 convergence tolerance must be finite and strictly positive (> 0.0); received {l1_tolerance}."
            )

        if not isinstance(l1_initialization, str):
            raise TypeError(
                f"L1 initialization must be a string, got {type(l1_initialization).__name__}."
            )

        norm_init = l1_initialization.strip().lower()
        if norm_init not in {"zeros", "ols"}:
            raise ValueError(
                f"Unsupported L1 initialization '{l1_initialization}'. Supported choices: 'zeros', 'ols'."
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

        try:
            zero_tol_val = float(coefficient_zero_tolerance)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Coefficient zero tolerance must be numeric, got {coefficient_zero_tolerance}."
            ) from exc

        if not np.isfinite(zero_tol_val) or zero_tol_val < 0.0:
            raise ValueError(
                f"Coefficient zero tolerance must be finite and >= 0.0, got {coefficient_zero_tolerance}."
            )

        self.regularization = norm_reg
        self.regularization_strength = strength_val
        self.l1_max_iterations = max_iters_val
        self.l1_tolerance = tol_val
        self.l1_initialization = norm_init
        self.condition_warning_threshold = thresh_val
        self.coefficient_zero_tolerance = zero_tol_val

        self.beta: np.ndarray | None = None
        self.fit_details: ModelFitDetails | None = None

    @property
    def l2_lambda(self) -> float:
        if self.regularization != REGULARIZATION_L2:
            raise AttributeError("l2_lambda is available only for L2 models.")
        return self.regularization_strength

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
        requested_reg = self.regularization
        strength = self.regularization_strength

        iterations: int | None = None
        converged: bool | None = None
        max_coeff_change: float | None = None
        l1_init_used: str | None = None

        if requested_reg == REGULARIZATION_NONE or strength == 0.0:
            effective_reg = REGULARIZATION_NONE
            solver_used = "lstsq_ols"
            beta, _, rank, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
            solver_cond = design_cond
            iterations = None
            converged = True
            max_coeff_change = None
            final_obj = 0.5 * float(np.sum((X_arr @ beta - y_arr) ** 2))
        elif requested_reg == REGULARIZATION_L2:
            effective_reg = REGULARIZATION_L2
            solver_used = "lstsq_augmented_ridge"
            sqrt_lambda = np.sqrt(strength)
            penalty = np.eye(num_features, dtype=np.float64)
            penalty[0, 0] = 0.0  # Do NOT regularize intercept

            X_aug = np.vstack([X_arr, sqrt_lambda * penalty])
            y_aug = np.concatenate([y_arr, np.zeros(num_features, dtype=np.float64)])

            beta, _residuals, rank, _singular_values = np.linalg.lstsq(
                X_aug, y_aug, rcond=None
            )
            solver_cond = float(np.linalg.cond(X_aug))
            iterations = None
            converged = True
            max_coeff_change = None
            final_obj = 0.5 * float(
                np.sum((X_arr @ beta - y_arr) ** 2)
            ) + 0.5 * strength * float(np.sum(beta[1:] ** 2))
        elif requested_reg == REGULARIZATION_L1:
            effective_reg = REGULARIZATION_L1
            solver_used = "coordinate_descent_l1"
            l1_init_used = self.l1_initialization
            solver_cond = design_cond
            rank = np.linalg.matrix_rank(X_arr)

            if self.l1_initialization == "zeros":
                beta = np.zeros(num_features, dtype=np.float64)
            else:  # 'ols'
                beta, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)

            col_norm_sq = np.sum(X_arr**2, axis=0)
            residual = y_arr - X_arr @ beta
            converged = False
            last_max_change = 0.0

            for iteration in range(1, self.l1_max_iterations + 1):
                max_change = 0.0
                for j in range(num_features):
                    denom = col_norm_sq[j]
                    if denom == 0.0:
                        new_beta_j = 0.0
                    else:
                        residual += X_arr[:, j] * beta[j]
                        rho = X_arr[:, j] @ residual
                        if j == 0:
                            new_beta_j = rho / denom
                        else:
                            new_beta_j = soft_threshold(rho, strength) / denom
                        residual -= X_arr[:, j] * new_beta_j

                    change = abs(new_beta_j - beta[j])
                    max_change = max(max_change, change)
                    beta[j] = new_beta_j

                last_max_change = max_change
                if max_change <= self.l1_tolerance:
                    converged = True
                    iterations = iteration
                    break
            else:
                iterations = self.l1_max_iterations

            max_coeff_change = float(last_max_change)

            if not converged:
                warnings.warn(
                    f"L1 coordinate descent did not converge within {self.l1_max_iterations} iterations. "
                    f"Final maximum coefficient change: {max_coeff_change:.4e}.",
                    UserWarning,
                )

            final_obj = 0.5 * float(
                np.sum((X_arr @ beta - y_arr) ** 2)
            ) + strength * float(np.sum(np.abs(beta[1:])))
        else:
            raise ValueError(f"Unhandled regularization type '{requested_reg}'.")

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

        if not np.isfinite(final_obj):
            raise FloatingPointError("Final objective value is non-finite.")

        nonzero_count = int(
            np.count_nonzero(np.abs(beta[1:]) > self.coefficient_zero_tolerance)
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
            requested_regularization=requested_reg,
            effective_regularization=effective_reg,
            regularization_strength=strength,
            iterations=iterations,
            converged=converged,
            final_objective=final_obj,
            max_coefficient_change=max_coeff_change,
            nonzero_coefficient_count=nonzero_count,
            l1_initialization=l1_init_used,
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
