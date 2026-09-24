"""Polynomial feature transformer and feature scaling module."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ScalingParameters:
    """Container for polynomial feature scaling statistics."""

    means: np.ndarray  # mean per polynomial degree (p >= 1)
    stds: np.ndarray   # std per polynomial degree (p >= 1)


class PolynomialFeatureTransformer:
    """Generates polynomial design matrices and optionally scales non-intercept features."""

    def __init__(self, degree: int, scale_features: bool = False):
        if degree < 0:
            raise ValueError(f"Polynomial degree must be >= 0, got {degree}.")
        self.degree = degree
        self.scale_features = scale_features
        self.scaling_params: Optional[ScalingParameters] = None

    def fit(self, x: np.ndarray) -> "PolynomialFeatureTransformer":
        """Computes scaling parameters (means and stds) from x for degree >= 1.

        Args:
            x: 1D NumPy array of feature values.

        Returns:
            self
        """
        if not self.scale_features or self.degree == 0:
            self.scaling_params = None
            return self

        x = np.asarray(x, dtype=np.float64)
        means = np.zeros(self.degree, dtype=np.float64)
        stds = np.ones(self.degree, dtype=np.float64)

        for p in range(1, self.degree + 1):
            xp = x ** p
            m = np.mean(xp)
            s = np.std(xp, ddof=0)
            means[p - 1] = m
            # Handle constant or near-zero standard deviation safely
            if s < 1e-15 or np.isnan(s):
                stds[p - 1] = 1.0  # Avoid division by zero; (x^p - m)/1.0 = 0 if constant
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
        x = np.asarray(x, dtype=np.float64)
        N = len(x)

        # Generate unscaled design matrix X_unscaled: shape (N, degree + 1)
        # x_p columns for p = 0, 1, ..., degree
        powers = np.arange(self.degree + 1, dtype=np.float64)
        X_design = x[:, None] ** powers  # Vectorized broadcasting

        if self.scale_features and self.degree > 0:
            if self.scaling_params is None:
                raise RuntimeError(
                    "Feature scaling is enabled but transformer is not fitted. Call fit() first."
                )
            # Scale non-intercept columns (column 1 to degree)
            for p in range(1, self.degree + 1):
                m = self.scaling_params.means[p - 1]
                s = self.scaling_params.stds[p - 1]
                if s < 1e-15:
                    X_design[:, p] = 0.0
                else:
                    X_design[:, p] = (X_design[:, p] - m) / s

        return X_design

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fits scaling parameters on x and returns transformed design matrix."""
        return self.fit(x).transform(x)

    def convert_coefficients_to_original_basis(
        self, beta_scaled: np.ndarray
    ) -> np.ndarray:
        """Converts coefficients fitted on scaled features back to the original polynomial basis.

        Formula:
            y_hat = beta_scaled_0 + sum_{p=1}^d beta_scaled_p * (x^p - mean_p) / std_p
            beta_orig_p = beta_scaled_p / std_p   (for p >= 1)
            beta_orig_0 = beta_scaled_0 - sum_{p=1}^d (beta_scaled_p * mean_p / std_p)

        Args:
            beta_scaled: 1D NumPy array of fitted coefficients on scaled design matrix.

        Returns:
            1D NumPy array of equivalent coefficients on original unscaled basis.
        """
        beta_scaled = np.asarray(beta_scaled, dtype=np.float64)
        if not self.scale_features or self.degree == 0 or self.scaling_params is None:
            return beta_scaled.copy()

        beta_orig = np.zeros_like(beta_scaled)
        beta_orig[0] = beta_scaled[0]

        for p in range(1, self.degree + 1):
            m = self.scaling_params.means[p - 1]
            s = self.scaling_params.stds[p - 1]

            if s < 1e-15:
                beta_orig[p] = 0.0
            else:
                beta_orig[p] = beta_scaled[p] / s
                beta_orig[0] -= beta_scaled[p] * m / s

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
            degree=transformer.degree, scale_features=False
        )
        X_unscaled = unscaled_transformer.transform(x)
        pred_orig = X_unscaled @ beta_orig

        if not np.allclose(pred_scaled, pred_orig, rtol=rtol, atol=atol):
            diff = np.abs(pred_scaled - pred_orig)
            max_diff = np.max(diff) if len(diff) > 0 else 0.0
            raise ValueError(
                f"Coefficient conversion verification failed! "
                f"Max absolute prediction difference: {max_diff:.6e} "
                f"(tolerance rtol={rtol}, atol={atol})."
            )

        return True
