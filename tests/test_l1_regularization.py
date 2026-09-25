"""Comprehensive tests for L1 regularization, coordinate descent, CLI options, and reporting."""

import json
import pathlib
import tempfile
import unittest

import numpy as np
import pytest

from main import parse_args, run_pipeline
from polynomial_regression.features import PolynomialFeatureTransformer
from polynomial_regression.regression import (
    PolynomialRegressor,
    soft_threshold,
)
from polynomial_regression.selection import HoldoutModelSelector


class TestSoftThreshold(unittest.TestCase):
    def test_soft_threshold_scalar_values(self):
        self.assertEqual(soft_threshold(5.0, 2.0), 3.0)
        self.assertEqual(soft_threshold(-5.0, 2.0), -3.0)
        self.assertEqual(soft_threshold(1.0, 2.0), 0.0)
        self.assertEqual(soft_threshold(-1.0, 2.0), 0.0)
        self.assertEqual(soft_threshold(2.0, 2.0), 0.0)
        self.assertEqual(soft_threshold(-2.0, 2.0), 0.0)
        self.assertEqual(soft_threshold(0.0, 0.0), 0.0)

    def test_soft_threshold_invalid_inputs(self):
        with self.assertRaises(ValueError):
            soft_threshold(5.0, -1.0)
        with self.assertRaises(ValueError):
            soft_threshold(np.nan, 1.0)
        with self.assertRaises(ValueError):
            soft_threshold(1.0, np.inf)


class TestL1Regressor(unittest.TestCase):
    def test_l1_solver_and_diagnostics(self):
        x = np.linspace(-2, 2, 40)
        y = 3.0 + 1.5 * x - 0.5 * x**2
        X = PolynomialFeatureTransformer(degree=2, scale_features=True).fit_transform(x)

        reg = PolynomialRegressor(
            regularization="l1",
            regularization_strength=0.1,
            l1_max_iterations=1000,
            l1_tolerance=1e-8,
        ).fit(X, y)

        details = reg.fit_details
        self.assertEqual(details.solver_used, "coordinate_descent_l1")
        self.assertTrue(details.converged)
        self.assertIsNotNone(details.iterations)
        self.assertLessEqual(details.iterations, 1000)
        self.assertIsNotNone(details.final_objective)
        self.assertIsNotNone(details.max_coefficient_change)
        self.assertLessEqual(details.max_coefficient_change, 1e-8)
        self.assertTrue(np.all(np.isfinite(reg.beta)))

    def test_l1_zero_strength_matches_ols(self):
        x = np.linspace(-2, 2, 30)
        y = 2.0 - x + 3.0 * x**2
        X = PolynomialFeatureTransformer(degree=2, scale_features=False).fit_transform(
            x
        )

        reg_l1_zero = PolynomialRegressor(
            regularization="l1", regularization_strength=0.0
        ).fit(X, y)

        reg_ols = PolynomialRegressor(regularization="none").fit(X, y)

        np.testing.assert_allclose(
            reg_l1_zero.beta, reg_ols.beta, rtol=1e-10, atol=1e-10
        )
        self.assertEqual(reg_l1_zero.fit_details.solver_used, "lstsq_ols")
        self.assertEqual(reg_l1_zero.fit_details.requested_regularization, "l1")
        self.assertEqual(reg_l1_zero.fit_details.effective_regularization, "none")

    def test_unregularized_intercept(self):
        x = np.linspace(-1, 1, 100)
        y = 50.0 + 2.0 * x
        X = PolynomialFeatureTransformer(degree=1, scale_features=False).fit_transform(
            x
        )

        reg = PolynomialRegressor(regularization="l1", regularization_strength=1e4).fit(
            X, y
        )

        # Intercept beta[0] should remain ~50.0, slope beta[1] should be driven to 0.0
        np.testing.assert_allclose(reg.beta[0], 50.0, atol=1e-2)
        self.assertEqual(reg.beta[1], 0.0)

    def test_l1_kkt_conditions(self):
        x = np.linspace(-2, 2, 50)
        y = 1.0 + 2.0 * x - x**2 + 0.5 * x**3
        X = PolynomialFeatureTransformer(degree=3, scale_features=True).fit_transform(x)
        strength = 0.5

        reg = PolynomialRegressor(
            regularization="l1",
            regularization_strength=strength,
            l1_max_iterations=10000,
            l1_tolerance=1e-10,
        ).fit(X, y)

        beta = reg.beta
        residual = y - X @ beta

        # Unpenalized intercept: X_0^T residual == 0
        grad_0 = X[:, 0] @ residual
        self.assertAlmostEqual(grad_0, 0.0, delta=1e-5)

        # Penalized features j >= 1
        for j in range(1, len(beta)):
            grad_j = X[:, j] @ residual
            bj = beta[j]
            if abs(bj) > 1e-8:
                # grad_j = strength * sign(bj)
                expected_grad = strength * np.sign(bj)
                self.assertAlmostEqual(grad_j, expected_grad, delta=1e-4)
            else:
                # |grad_j| <= strength
                self.assertLessEqual(abs(grad_j), strength + 1e-4)

    def test_l1_non_convergence_warning(self):
        x = np.linspace(-2, 2, 50)
        y = np.sin(x)
        X = PolynomialFeatureTransformer(degree=5, scale_features=False).fit_transform(
            x
        )

        with pytest.warns(UserWarning, match="L1 coordinate descent did not converge"):
            reg = PolynomialRegressor(
                regularization="l1",
                regularization_strength=0.01,
                l1_max_iterations=1,  # Force non-convergence
                l1_tolerance=1e-15,
            ).fit(X, y)

        self.assertFalse(reg.fit_details.converged)
        self.assertEqual(reg.fit_details.iterations, 1)


@pytest.mark.filterwarnings(
    "ignore:L1 regularization is sensitive to feature scale.*:UserWarning"
)
class TestCLIRegularizationOptions(unittest.TestCase):
    def test_regularization_cli_options(self):
        parsed = parse_args(
            [
                "dummy.csv",
                "--regularization",
                "l1",
                "--regularization-values",
                "0.0",
                "0.1",
                "1.0",
                "--l1-max-iterations",
                "5000",
                "--l1-tolerance",
                "1e-6",
                "--l1-initialization",
                "ols",
            ]
        )
        self.assertEqual(parsed.regularization, "l1")
        self.assertEqual(parsed.regularization_values, [0.0, 0.1, 1.0])
        self.assertEqual(parsed.l1_max_iterations, 5000)
        self.assertEqual(parsed.l1_tolerance, 1e-6)
        self.assertEqual(parsed.l1_initialization, "ols")

    def test_regularization_none_forces_zero(self):
        parsed = parse_args(
            [
                "dummy.csv",
                "--regularization",
                "none",
            ]
        )
        self.assertEqual(parsed.regularization, "none")
        self.assertEqual(parsed.regularization_values, [0.0])

    def test_regularization_none_rejects_nonzero(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "dummy.csv",
                    "--regularization",
                    "none",
                    "--regularization-values",
                    "0.1",
                ]
            )

    def test_conflicting_l2_and_regularization_values(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "dummy.csv",
                    "--regularization-values",
                    "0.1",
                    "--l2-values",
                    "0.1",
                ]
            )

    def test_l2_values_with_l1_regularization_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "dummy.csv",
                    "--regularization",
                    "l1",
                    "--l2-values",
                    "0.1",
                ]
            )


class TestL1SelectionAndEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp())
        self.csv_file = self.tmp_dir / "data.csv"
        x = np.linspace(-3, 3, 60)
        y = 1.0 + 2.0 * x - 0.5 * x**2
        lines = ["X,Y"] + [f"{xi},{yi}" for xi, yi in zip(x, y)]
        self.csv_file.write_text("\n".join(lines), encoding="utf-8")

    @pytest.mark.filterwarnings(
        "ignore:L1 coordinate descent did not converge.*:UserWarning"
    )
    def test_all_non_converged_candidates_raises_runtime_error(self):
        x = np.linspace(-3, 3, 50)
        y = x**3 - x
        train_idx = np.arange(30)
        val_idx = np.arange(30, 40)
        test_idx = np.arange(40, 50)
        dev_idx = np.arange(40)

        selector = HoldoutModelSelector(
            degrees=[2, 3],
            regularization="l1",
            regularization_strengths=[0.1],
            l1_max_iterations=1,  # Impossible to converge
            l1_tolerance=1e-15,
        )

        with self.assertRaises(RuntimeError) as ctx:
            selector.select(x, y, train_idx, val_idx, test_idx, dev_idx)
        self.assertIn("No L1 candidate converged", str(ctx.exception))

    @pytest.mark.filterwarnings(
        "ignore:L1 regularization is sensitive to feature scale.*:UserWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:L1 coordinate descent did not converge.*:UserWarning"
    )
    def test_end_to_end_combinations(self):
        modes = ["holdout", "kfold", "both"]
        regs = ["none", "l1", "l2"]

        for mode in modes:
            for reg in regs:
                out_dir = self.tmp_dir / f"results_{mode}_{reg}"
                cmd = [
                    str(self.csv_file),
                    "--output-dir",
                    str(out_dir),
                    "--mode",
                    mode,
                    "--regularization",
                    reg,
                    "--no-plots",
                ]
                if reg == "none":
                    cmd.extend(["--regularization-values", "0.0"])
                else:
                    cmd.extend(["--regularization-values", "0.0", "0.1"])

                args = parse_args(cmd)
                run_pipeline(args)

                if mode in ["holdout", "both"]:
                    self.assertTrue((out_dir / "holdout" / "final_model.json").exists())
                if mode in ["kfold", "both"]:
                    self.assertTrue((out_dir / "kfold" / "final_model.json").exists())

    def test_end_to_end_l1_bootstrap_and_scaling(self):
        out_dir = self.tmp_dir / "results_l1_bs"
        args = parse_args(
            [
                str(self.csv_file),
                "--output-dir",
                str(out_dir),
                "--mode",
                "holdout",
                "--regularization",
                "l1",
                "--regularization-values",
                "0.0",
                "0.01",
                "0.1",
                "--scale-features",
                "--bootstrap-samples",
                "10",
            ]
        )

        run_pipeline(args)

        final_model_file = out_dir / "holdout" / "final_model.json"
        self.assertTrue(final_model_file.exists())
        with open(final_model_file, "r", encoding="utf-8") as f:
            fm = json.load(f)

        self.assertEqual(fm["selected_regularization"], "l1")
        self.assertTrue(fm["scale_features_enabled"])
        self.assertIn("scaling_parameters", fm)


if __name__ == "__main__":
    unittest.main()
