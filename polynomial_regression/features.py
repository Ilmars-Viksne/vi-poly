"""Polynomial feature transformer and feature scaling module."""

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_DEGREE = 50


@dataclass
class ScalingParameters:
    """Container for polynomial feature scaling statistics."""

    means: np.ndarray  # mean per polynomial degree (p >= 1)
    stds: np.ndarray  # std per polynomial degree (p >= 1)


class PolynomialFeatureTransformer:
    """Generates polynomial design matrices and optionally scales non-intercept features."""

    def __init__(
        self,
        degree: int,
        scale_features: bool = False,
        max_degree: int = DEFAULT_MAX_DEGREE,
    ):
        if not isinstance(max_degree, (int, np.integer)) or max_degree < 0:
            raise ValueError(
                f"max_degree must be a non-negative integer, got {max_degree}."
            )
        if (
            not isinstance(degree, (int, np.integer))
            or degree < 0
            or degree > max_degree
        ):
            raise ValueError(
                f"Polynomial degree must be an integer between 0 and configured maximum {max_degree}, got {degree}."
            )

        self.degree = int(degree)
        self.scale_features = scale_features
        self.max_degree = int(max_degree)
        self.scaling_params: ScalingParameters | None = None

    def _validate_x(self, x: np.ndarray, param_name: str = "x") -> np.ndarray:
        """Validates that x is a 1D, non-empty, finite numeric array."""
        try:
            x_arr = np.asarray(x, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Input '{param_name}' must be numeric.") from exc

        if x_arr.ndim != 1:
            raise ValueError(
                f"Input '{param_name}' must be a 1D array; received shape {x_arr.shape}."
            )

        if x_arr.size == 0:
            raise ValueError(
                f"Input '{param_name}' must contain at least one observation."
            )

        if not np.all(np.isfinite(x_arr)):
            non_finite_count = int(np.sum(~np.isfinite(x_arr)))
            raise ValueError(
                f"Input '{param_name}' contains {non_finite_count} non-finite value(s)."
            )

        return x_arr

    def _preflight_power_check(self, x: np.ndarray) -> None:
        """Estimates whether generating powers up to degree will overflow float64 capacity."""
        if self.degree == 0:
            return

        max_abs_x = float(np.max(np.abs(x)))
        if max_abs_x > 1.0:
            log_max_power = self.degree * math.log(max_abs_x)
            log_float_max = math.log(np.finfo(np.float64).max)

            if log_max_power > log_float_max:
                raise FloatingPointError(
                    f"Polynomial feature generation overflow estimated for degree {self.degree} "
                    f"with max(abs(x))={max_abs_x:.6e}. "
                    f"Feature scaling was enabled: {self.scale_features}, but scaling occurs "
                    "after power construction. Reduce the polynomial degree or rescale X."
                )

    def _build_unscaled_design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Constructs unscaled design matrix [1, x, x^2, ..., x^degree] with overflow guards."""
        self._preflight_power_check(x)

        n_samples = x.shape[0]
        X_design = np.empty((n_samples, self.degree + 1), dtype=np.float64)
        X_design[:, 0] = 1.0

        if self.degree > 0:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                try:
                    for power in range(1, self.degree + 1):
                        X_design[:, power] = X_design[:, power - 1] * x
                except FloatingPointError as exc:
                    max_abs_x = float(np.max(np.abs(x)))
                    raise FloatingPointError(
                        f"Polynomial feature generation overflowed at power {power} for degree {self.degree} "
                        f"with max(abs(x))={max_abs_x:.6e}. "
                        f"Feature scaling was enabled: {self.scale_features}, but scaling occurs "
                        "after power construction. Reduce the polynomial degree or rescale X."
                    ) from exc

        if not np.all(np.isfinite(X_design)):
            max_abs_x = float(np.max(np.abs(x)))
            raise FloatingPointError(
                f"Polynomial feature generation produced non-finite values for degree {self.degree} "
                f"with max(abs(x))={max_abs_x:.6e}. Reduce the polynomial degree or rescale X."
            )

        return X_design

    def fit(self, x: np.ndarray) -> "PolynomialFeatureTransformer":
        """Computes scaling parameters (means and stds) from x for degree >= 1.

        Args:
            x: 1D NumPy array of feature values.

        Returns:
            self
        """
        x_arr = self._validate_x(x)

        if not self.scale_features or self.degree == 0:
            self.scaling_params = None
            return self

        X_unscaled = self._build_unscaled_design_matrix(x_arr)

        means = np.zeros(self.degree, dtype=np.float64)
        stds = np.ones(self.degree, dtype=np.float64)

        for p in range(1, self.degree + 1):
            col = X_unscaled[:, p]
            m = float(np.mean(col))
            s = float(np.std(col, ddof=0))
            means[p - 1] = m
            if not np.isfinite(s) or s < 1e-15:
                stds[p - 1] = 1.0
            else:
                stds[p - 1] = s

        self.scaling_params = ScalingParameters(means=means, stds=stds)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Constructs the polynomial design matrix [1, x, x^2, ..., x^degree].

        Applies Z-score scaling to non-intercept columns if scale_features is True.

        Args:
            x: 1D NumPy array of feature values.

        Returns:
            2D NumPy array of shape (N, degree + 1).
        """
        x_arr = self._validate_x(x)
        X_design = self._build_unscaled_design_matrix(x_arr)

        if self.scale_features and self.degree > 0:
            if self.scaling_params is None:
                raise RuntimeError(
                    "Feature scaling is enabled but transformer is not fitted. Call fit() first."
                )
            if (
                len(self.scaling_params.means) != self.degree
                or len(self.scaling_params.stds) != self.degree
            ):
                raise ValueError(
                    "Scaling parameters dimension mismatch for current degree."
                )

            for p in range(1, self.degree + 1):
                m = self.scaling_params.means[p - 1]
                s = self.scaling_params.stds[p - 1]
                if s < 1e-15:
                    X_design[:, p] = 0.0
                else:
                    X_design[:, p] = (X_design[:, p] - m) / s

            if not np.all(np.isfinite(X_design)):
                raise FloatingPointError(
                    "Scaled polynomial feature generation produced non-finite values."
                )

        return X_design

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fits scaling parameters on x and returns transformed design matrix."""
        return self.fit(x).transform(x)

    def convert_coefficients_to_original_basis(
        self, beta_scaled: np.ndarray
    ) -> np.ndarray:
        """Converts coefficients fitted on scaled features back to the original polynomial basis.

        Args:
            beta_scaled: 1D NumPy array of fitted coefficients on scaled design matrix.

        Returns:
            1D NumPy array of equivalent coefficients on original unscaled basis.
        """
        try:
            beta_arr = np.asarray(beta_scaled, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("beta_scaled must be numeric.") from exc

        if beta_arr.ndim != 1:
            raise ValueError(
                f"beta_scaled must be a 1D array; received shape {beta_arr.shape}."
            )

        if len(beta_arr) != self.degree + 1:
            raise ValueError(
                f"beta_scaled length mismatch: expected {self.degree + 1} for degree {self.degree}, got {len(beta_arr)}."
            )

        if not np.all(np.isfinite(beta_arr)):
            non_finite_count = int(np.sum(~np.isfinite(beta_arr)))
            raise ValueError(
                f"beta_scaled contains {non_finite_count} non-finite value(s)."
            )

        if not self.scale_features or self.degree == 0 or self.scaling_params is None:
            return beta_arr.copy()

        beta_orig = np.zeros_like(beta_arr)
        beta_orig[0] = beta_arr[0]

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for p in range(1, self.degree + 1):
                    m = self.scaling_params.means[p - 1]
                    s = self.scaling_params.stds[p - 1]

                    if not np.isfinite(m) or not np.isfinite(s) or s <= 0.0:
                        raise ValueError(
                            f"Invalid scaling parameters for degree {p}: mean={m}, std={s}."
                        )

                    if s < 1e-15:
                        beta_orig[p] = 0.0
                    else:
                        beta_orig[p] = beta_arr[p] / s
                        beta_orig[0] -= beta_arr[p] * m / s
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f"Conversion of scaled coefficients to original monomial basis overflowed for degree {self.degree}. "
                    "The scaled model exists but cannot be represented safely in the original monomial basis."
                ) from exc

        if not np.all(np.isfinite(beta_orig)):
            raise FloatingPointError(
                f"Conversion of scaled coefficients to original monomial basis produced non-finite values for degree {self.degree}. "
                "The scaled model exists but cannot be represented safely in the original monomial basis."
            )

        return beta_orig

    @staticmethod
    def verify_coefficient_conversion(
        x: np.ndarray,
        transformer: "PolynomialFeatureTransformer",
        beta_scaled: np.ndarray,
        beta_orig: np.ndarray,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> bool:
        """Verifies that predictions from scaled features and original basis match.

        Raises:
            ValueError: If predictions differ beyond specified numerical tolerance.
        """
        # Scaled prediction: X_scaled @ beta_scaled
        X_scaled = transformer.transform(x)
        pred_scaled = X_scaled @ beta_scaled

        # Original prediction: X_unscaled @ beta_orig
        unscaled_transformer = PolynomialFeatureTransformer(
            degree=transformer.degree,
            scale_features=False,
            max_degree=transformer.max_degree,
        )
        X_unscaled = unscaled_transformer.transform(x)
        pred_orig = X_unscaled @ beta_orig

        if not np.allclose(pred_scaled, pred_orig, rtol=rtol, atol=atol):
            diff = np.abs(pred_scaled - pred_orig)
            max_diff = float(np.max(diff)) if len(diff) > 0 else 0.0
            raise ValueError(
                f"Coefficient conversion verification failed! "
                f"Max absolute prediction difference: {max_diff:.6e} "
                f"(tolerance rtol={rtol}, atol={atol})."
            )

        return True
