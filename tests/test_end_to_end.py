"""End-to-end tests for main.py CLI pipeline and correctness fixes."""

import json
import pathlib
import tempfile
import unittest

import numpy as np

from generate_data import generate_synthetic_data, save_to_csv
from main import parse_args, run_pipeline
from polynomial_regression.regression import PolynomialRegressor
from polynomial_regression.selection import HoldoutModelSelector, KFoldModelSelector
from polynomial_regression.splitting import DataSplitter


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

        self.assertTrue((out_dir / "common" / "split_summary.json").exists())
        self.assertTrue((out_dir / "common" / "plot_manifest.json").exists())
        self.assertTrue((out_dir / "holdout" / "holdout_results.csv").exists())
        self.assertTrue((out_dir / "holdout" / "final_model.json").exists())
        self.assertTrue((out_dir / "holdout" / "test_predictions.csv").exists())
        self.assertTrue((out_dir / "holdout" / "plot_manifest.json").exists())

        # Ensure kfold and comparison.json are NOT created
        self.assertFalse((out_dir / "kfold").exists())
        self.assertFalse((out_dir / "comparison.json").exists())

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

        self.assertTrue((out_dir / "common" / "split_summary.json").exists())
        self.assertTrue((out_dir / "common" / "plot_manifest.json").exists())
        self.assertTrue((out_dir / "kfold" / "kfold_fold_results.csv").exists())
        self.assertTrue((out_dir / "kfold" / "kfold_summary_results.csv").exists())
        self.assertTrue((out_dir / "kfold" / "final_model.json").exists())
        self.assertTrue((out_dir / "kfold" / "test_predictions.csv").exists())
        self.assertTrue((out_dir / "kfold" / "out_of_fold_predictions.csv").exists())
        self.assertTrue((out_dir / "kfold" / "plot_manifest.json").exists())

        # Ensure holdout and comparison.json are NOT created
        self.assertFalse((out_dir / "holdout").exists())
        self.assertFalse((out_dir / "comparison.json").exists())

    def test_end_to_end_both_mode_and_manifest_uniqueness(self):
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

        # Check directory structures
        self.assertTrue((out_dir / "common").exists())
        self.assertTrue((out_dir / "holdout").exists())
        self.assertTrue((out_dir / "kfold").exists())

        # Distinct artifact paths
        holdout_model = out_dir / "holdout" / "final_model.json"
        kfold_model = out_dir / "kfold" / "final_model.json"
        self.assertTrue(holdout_model.exists())
        self.assertTrue(kfold_model.exists())
        self.assertNotEqual(holdout_model.resolve(), kfold_model.resolve())

        holdout_preds = out_dir / "holdout" / "test_predictions.csv"
        kfold_preds = out_dir / "kfold" / "test_predictions.csv"
        self.assertTrue(holdout_preds.exists())
        self.assertTrue(kfold_preds.exists())
        self.assertNotEqual(holdout_preds.resolve(), kfold_preds.resolve())

        # Final plots exist simultaneously
        self.assertTrue(
            (out_dir / "holdout" / "13b_final_polynomial_bootstrap_band.png").exists()
        )
        self.assertTrue(
            (out_dir / "kfold" / "13b_final_polynomial_bootstrap_band.png").exists()
        )

        # Check manifests and verify no duplicate plot path references exist
        for subdir in ["common", "holdout", "kfold"]:
            manifest_file = out_dir / subdir / "plot_manifest.json"
            self.assertTrue(manifest_file.exists())
            with open(manifest_file, "r") as f:
                entries = json.load(f)
            paths = [e["plot_filename"] for e in entries]
            self.assertEqual(len(paths), len(set(paths)), "Duplicate plot filename in manifest")
            for p in paths:
                self.assertTrue((out_dir / subdir / p).exists())

        # Check comparison.json
        comp_file = out_dir / "comparison.json"
        self.assertTrue(comp_file.exists())
        with open(comp_file, "r") as f:
            comp_data = json.load(f)
        self.assertEqual(comp_data["mode"], "both")
        self.assertIsNone(comp_data["overall_best_workflow"])
        self.assertIn("development data only", comp_data["selection_policy"])

    def test_final_fit_metadata_propagation(self):
        out_dir = self.temp_path / "results_fit_metadata"
        args = parse_args(
            [
                str(self.csv_file),
                "--mode",
                "holdout",
                "--degrees",
                "1",
                "2",
                "--l2-values",
                "0",
                "--output-dir",
                str(out_dir),
            ]
        )
        run_pipeline(args)

        model_json = out_dir / "holdout" / "final_model.json"
        with open(model_json, "r") as f:
            data = json.load(f)

        self.assertIn("solver_used", data)
        self.assertIn("condition_number", data)
        self.assertIn("condition_warning", data)
        self.assertEqual(data["solver_used"], "solve")
        self.assertIsInstance(data["condition_number"], float)
        self.assertIsInstance(data["condition_warning"], bool)

    def test_solver_fallback_and_metadata(self):
        # Construct rank-deficient design matrix with identical columns to force singular matrix in solver
        X_singular = np.ones((10, 3), dtype=np.float64)
        y = np.ones(10, dtype=np.float64)

        regressor = PolynomialRegressor(l2_lambda=0.0).fit(X_singular, y)
        self.assertEqual(regressor.fit_details.solver_used, "lstsq")

    def test_neutral_workflow_selection(self):
        out_dir = self.temp_path / "results_neutral"
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
                "--output-dir",
                str(out_dir),
            ]
        )
        run_pipeline(args)

        comp_file = out_dir / "comparison.json"
        with open(comp_file, "r") as f:
            data = json.load(f)

        self.assertIsNone(data["overall_best_workflow"])
        self.assertIn("workflow_selection", data)
        self.assertIn("holdout", data["workflow_selection"])
        self.assertIn("kfold", data["workflow_selection"])


if __name__ == "__main__":
    unittest.main()
