"""End-to-end tests for main.py CLI pipeline."""

import pathlib
import tempfile
import unittest

from generate_data import generate_synthetic_data, save_to_csv
from main import parse_args, run_pipeline


class TestEndToEndCLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = pathlib.Path(self.temp_dir.name)

        # Generate synthetic CSV
        self.csv_file = self.temp_path / "test_data.csv"
        x, y = generate_synthetic_data(
            100, -5.0, 5.0, [2.0, -1.0, 0.5], noise_std=0.2, seed=42
        )
        save_to_csv(self.csv_file, x, y)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_end_to_end_holdout_mode(self):
        out_dir = self.temp_path / "results_holdout"
        args = parse_args(
            [
                str(self.csv_file),
                "--mode",
                "holdout",
                "--degrees",
                "1",
                "2",
                "3",
                "--l2-values",
                "0",
                "0.01",
                "--output-dir",
                str(out_dir),
                "--scale-features",
            ]
        )
        run_pipeline(args)

        self.assertTrue((out_dir / "split_summary.json").exists())
        self.assertTrue((out_dir / "holdout_results.csv").exists())
        self.assertTrue((out_dir / "final_model.json").exists())
        self.assertTrue((out_dir / "test_predictions.csv").exists())
        self.assertTrue((out_dir / "plot_manifest.json").exists())

    def test_end_to_end_kfold_mode(self):
        out_dir = self.temp_path / "results_kfold"
        args = parse_args(
            [
                str(self.csv_file),
                "--mode",
                "kfold",
                "--degrees",
                "1",
                "2",
                "--l2-values",
                "0",
                "0.1",
                "--folds",
                "3",
                "--output-dir",
                str(out_dir),
            ]
        )
        run_pipeline(args)

        self.assertTrue((out_dir / "kfold_fold_results.csv").exists())
        self.assertTrue((out_dir / "kfold_summary_results.csv").exists())
        self.assertTrue((out_dir / "out_of_fold_predictions.csv").exists())

    def test_end_to_end_both_mode_with_bootstrap(self):
        out_dir = self.temp_path / "results_both"
        args = parse_args(
            [
                str(self.csv_file),
                "--mode",
                "both",
                "--degrees",
                "1",
                "2",
                "--l2-values",
                "0",
                "0.01",
                "--folds",
                "3",
                "--output-dir",
                str(out_dir),
                "--bootstrap-samples",
                "10",
            ]
        )
        run_pipeline(args)

        self.assertTrue((out_dir / "13b_final_polynomial_bootstrap_band.png").exists())


if __name__ == "__main__":
    unittest.main()
