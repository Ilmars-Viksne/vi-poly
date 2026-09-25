"""Tests for RegressionVisualizer and ReportGenerator modules."""

import pathlib
import tempfile
import unittest

import numpy as np

import warnings

from polynomial_regression.metrics import EvaluationMetrics
from polynomial_regression.reporting import ReportGenerator
from polynomial_regression.splitting import KFoldSplitter
from polynomial_regression.visualization import (
    WORKFLOW_DISPLAY_NAMES,
    RegressionVisualizer,
)


class TestVisualizationAndReporting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reporting_generators(self):
        reporter = ReportGenerator(self.output_dir)

        # Split summary
        split_file = reporter.write_split_summary(
            seed=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            total_samples=100,
            train_indices=np.arange(70),
            val_indices=np.arange(70, 85),
            test_indices=np.arange(85, 100),
            dev_indices=np.arange(85),
        )
        self.assertTrue(split_file.exists())
        self.assertGreater(split_file.stat().st_size, 0)

        # Plot manifest
        dummy_plot = self.output_dir / "test.png"
        dummy_plot.write_bytes(b"dummy")
        manifest_file = reporter.write_plot_manifest(
            [
                {
                    "plot_filename": "test.png",
                    "plot_title": "Test",
                    "workflow": "common",
                    "source_numerical_result_file": "split_summary.json",
                }
            ]
        )
        self.assertTrue(manifest_file.exists())

    def test_visualizer_creation_and_non_empty(self):
        visualizer = RegressionVisualizer(
            output_dir=self.output_dir,
            plot_format="png",
            plot_dpi=100,
            show_plots=False,
        )

        x = np.linspace(-5, 5, 20)
        y = x**2 + np.random.randn(20)

        plot_path = visualizer.plot_01_original_data(x, y)

        self.assertTrue(plot_path.exists())
        self.assertGreater(plot_path.stat().st_size, 0)
        self.assertEqual(len(visualizer.manifest_entries), 1)

    def test_metrics_summary_plot(self):
        visualizer = RegressionVisualizer(self.output_dir, show_plots=False)
        metrics = EvaluationMetrics(mse=1.2, rmse=1.095, mae=0.9, r_squared=0.95)

        plot_path = visualizer.plot_17_final_metrics(metrics, workflow="Holdout")
        self.assertTrue(plot_path.exists())
        self.assertGreater(plot_path.stat().st_size, 0)

    def test_workflow_naming_conventions(self):
        visualizer = RegressionVisualizer(self.output_dir, show_plots=False)

        # Common plot
        p1 = visualizer.plot_01_original_data(
            np.array([1, 2, 3]), np.array([1, 4, 9])
        )
        self.assertEqual(p1.name, "01_original_data.png")
        self.assertNotIn("holdout", p1.name)
        self.assertNotIn("kfold", p1.name)
        self.assertEqual(visualizer.manifest_entries[0]["workflow"], "common")

        # Workflow specific plots
        metrics = EvaluationMetrics(mse=1.0, rmse=1.0, mae=1.0, r_squared=0.8)
        p_holdout = visualizer.plot_17_final_metrics(metrics, workflow="holdout")
        self.assertIn("holdout", p_holdout.name)
        self.assertIn(
            "Holdout", visualizer.manifest_entries[-1]["plot_title"]
        )

        p_kfold = visualizer.plot_17_final_metrics(metrics, workflow="kfold")
        self.assertIn("kfold", p_kfold.name)
        self.assertIn("K-Fold", visualizer.manifest_entries[-1]["plot_title"])
        self.assertNotIn("Kfold", visualizer.manifest_entries[-1]["plot_title"])

        # Check no filename collision if both stored in same directory
        self.assertNotEqual(p_holdout.name, p_kfold.name)

    def test_bootstrap_band_terminology_and_metadata(self):
        visualizer = RegressionVisualizer(self.output_dir, show_plots=False)
        x_dev = np.linspace(-3, 3, 20)
        y_dev = x_dev**2
        x_grid = np.linspace(-3, 3, 50)
        y_grid = x_grid**2
        lower_b = y_grid - 0.5
        upper_b = y_grid + 0.5

        p = visualizer.plot_13b_fitted_curve_uncertainty_band(
            x_dev,
            y_dev,
            x_dev[:5],
            y_dev[:5],
            x_grid,
            y_grid,
            lower_b,
            upper_b,
            degree=2,
            l2=0.01,
            num_bootstraps=100,
            workflow="holdout",
        )

        self.assertIn("fitted_curve_uncertainty_band", p.name)
        entry = visualizer.manifest_entries[-1]
        self.assertIn("Bootstrap Fitted-Curve Uncertainty Band", entry["plot_title"])
        self.assertEqual(
            entry["parameters_represented"]["interval_interpretation"],
            "fitted_curve_uncertainty",
        )
        self.assertEqual(
            entry["parameters_represented"]["lower_percentile"], 2.5
        )
        self.assertEqual(
            entry["parameters_represented"]["upper_percentile"], 97.5
        )

        # Test deprecated method wrapper
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            visualizer.plot_13b_final_polynomial_bootstrap_band(
                x_dev,
                y_dev,
                x_dev[:5],
                y_dev[:5],
                x_grid,
                y_grid,
                lower_b,
                upper_b,
                degree=2,
                l2=0.01,
                num_bootstraps=100,
                workflow="holdout",
            )
            self.assertTrue(any(issubclass(warn.category, DeprecationWarning) for warn in w))

    def test_kfold_membership_matrix(self):
        k_splitter = KFoldSplitter(k=5, seed=42)
        dev_indices = np.arange(25)
        fold_splits = k_splitter.split(dev_indices)

        matrix = RegressionVisualizer.build_kfold_membership_matrix(25, fold_splits)
        self.assertEqual(matrix.shape, (5, 25))

        # Every column must sum to exactly 1 (validation = 1)
        np.testing.assert_array_equal(matrix.sum(axis=0), np.ones(25, dtype=int))

        # Every column must have exactly K - 1 training cells
        np.testing.assert_array_equal(
            (matrix == 0).sum(axis=0), np.full(25, 4, dtype=int)
        )

        # Each fold validation count matches fold val_indices length
        for fold in fold_splits:
            self.assertEqual(
                np.sum(matrix[fold.fold_index] == 1), len(fold.val_indices)
            )

        # Test invalid inputs
        with self.assertRaises(ValueError):
            RegressionVisualizer.build_kfold_membership_matrix(0, fold_splits)
        with self.assertRaises(ValueError):
            RegressionVisualizer.build_kfold_membership_matrix(25, [])
        with self.assertRaises(ValueError):
            RegressionVisualizer.build_kfold_membership_matrix(25, fold_splits[:1])

        # Test plot creation
        visualizer = RegressionVisualizer(self.output_dir, show_plots=False)
        p = visualizer.plot_08_kfold_assignments(dev_indices, fold_splits)
        self.assertTrue(p.exists())
        self.assertEqual(p.name, "08_kfold_assignments.png")
        entry = visualizer.manifest_entries[-1]
        self.assertEqual(entry["plot_type"], "Binary Membership Matrix")
        self.assertIn("K-Fold", entry["plot_title"])


if __name__ == "__main__":
    unittest.main()
