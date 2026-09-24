"""Tests for RegressionVisualizer and ReportGenerator modules."""

import pathlib
import tempfile
import unittest
import numpy as np

from polynomial_regression.visualization import RegressionVisualizer
from polynomial_regression.reporting import ReportGenerator
from polynomial_regression.metrics import EvaluationMetrics


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
            seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            total_samples=100, train_indices=np.arange(70), val_indices=np.arange(70, 85),
            test_indices=np.arange(85, 100), dev_indices=np.arange(85)
        )
        self.assertTrue(split_file.exists())
        self.assertGreater(split_file.stat().st_size, 0)

        # Plot manifest
        manifest_file = reporter.write_plot_manifest([{"plot_filename": "test.png"}])
        self.assertTrue(manifest_file.exists())

    def test_visualizer_creation_and_non_empty(self):
        visualizer = RegressionVisualizer(
            output_dir=self.output_dir,
            plot_format="png",
            plot_dpi=100,
            show_plots=False,
        )

        x = np.linspace(-5, 5, 20)
        y = x ** 2 + np.random.randn(20)

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


if __name__ == "__main__":
    unittest.main()
