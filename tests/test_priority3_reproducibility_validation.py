"""Tests for Priority 3: Reproducibility, CLI Validation, Search-Grid Canonicalization, and CSV Provenance."""

import argparse
import json
import pathlib
import unittest

import numpy as np

from main import parse_args, run_pipeline, validate_args
from polynomial_regression.data import CSVDataLoader
from polynomial_regression.reporting import _validate_equal_lengths
from polynomial_regression.selection import HoldoutModelSelector


class TestCLIValidation(unittest.TestCase):
    """Unit tests for CLI argument validation and boundary cases."""

    def test_rejection_negative_degree(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--degrees", "-1"])

    def test_rejection_degree_above_max_degree(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--degrees", "10", "--max-degree", "5"])

    def test_rejection_negative_max_degree(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--max-degree", "-5"])

    def test_rejection_negative_l2(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--l2-values", "-0.1"])

    def test_rejection_nan_l2(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--l2-values", "nan"])

    def test_rejection_infinite_l2(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--l2-values", "inf"])

    def test_rejection_folds_less_than_two(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--folds", "1"])
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--folds", "0"])

    def test_rejection_curve_points_less_than_two(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--curve-points", "1"])
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--curve-points", "0"])

    def test_rejection_negative_bootstrap_samples(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--bootstrap-samples", "-5"])

    def test_rejection_plot_dpi_non_positive(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--plot-dpi", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--plot-dpi", "-100"])

    def test_rejection_non_positive_condition_threshold(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--condition-warning-threshold", "0.0"])

    def test_rejection_negative_selection_tolerance(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--selection-rtol", "-1e-5"])

    def test_rejection_invalid_residual_bins(self):
        with self.assertRaises(SystemExit):
            parse_args(["dummy.csv", "--residual-bins", "0"])

    def test_acceptance_boundary_values(self):
        parsed = parse_args(
            [
                "dummy.csv",
                "--degrees",
                "0",
                "--max-degree",
                "0",
                "--l2-values",
                "0.0",
                "--folds",
                "2",
                "--curve-points",
                "2",
                "--bootstrap-samples",
                "0",
                "--plot-dpi",
                "1",
            ]
        )
        self.assertEqual(parsed.degrees, [0])
        self.assertEqual(parsed.max_degree, 0)
        self.assertEqual(parsed.l2_values, [0.0])
        self.assertEqual(parsed.folds, 2)
        self.assertEqual(parsed.curve_points, 2)
        self.assertEqual(parsed.bootstrap_samples, 0)
        self.assertEqual(parsed.plot_dpi, 1)

    def test_programmatic_validation_raises_value_error(self):
        ns = argparse.Namespace(
            csv_file="data.csv",
            x_column="X",
            y_column="Y",
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            folds=1,  # Invalid
            degrees=[1, 2],
            max_degree=50,
            l2_values=[0.0],
            curve_points=500,
            bootstrap_samples=0,
            bootstrap_seed=123,
            plot_dpi=150,
            condition_warning_threshold=1e12,
            selection_rtol=1e-7,
            selection_atol=1e-12,
            residual_bins=None,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_args(ns, parser=None)
        self.assertIn("Number of folds must be at least 2", str(ctx.exception))


class TestSearchGridCanonicalization(unittest.TestCase):
    """Tests for deduplication and sorting of degree and L2 search grids."""

    def test_deduplication_and_sorting(self):
        parsed = parse_args(
            [
                "dummy.csv",
                "--degrees",
                "3",
                "1",
                "2",
                "3",
                "1",
                "--l2-values",
                "1.0",
                "0.0",
                "0.1",
                "1.0",
                "0.0",
            ]
        )
        self.assertEqual(parsed._requested_degrees, [3, 1, 2, 3, 1])
        self.assertEqual(parsed._requested_l2_values, [1.0, 0.0, 0.1, 1.0, 0.0])
        self.assertEqual(parsed.degrees, [1, 2, 3])
        self.assertEqual(parsed.l2_values, [0.0, 0.1, 1.0])

    def test_nearly_equal_floats_remain_distinct(self):
        ns = argparse.Namespace(
            csv_file="data.csv",
            x_column="X",
            y_column="Y",
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            folds=5,
            degrees=[1],
            max_degree=50,
            l2_values=[0.1, 0.1000000001],
            curve_points=500,
            bootstrap_samples=0,
            bootstrap_seed=123,
            plot_dpi=150,
            condition_warning_threshold=1e12,
            selection_rtol=1e-7,
            selection_atol=1e-12,
            residual_bins=None,
        )
        validated = validate_args(ns)
        self.assertEqual(len(validated.l2_values), 2)
        self.assertEqual(validated.l2_values, [0.1, 0.1000000001])

    def test_candidate_counts_equal_unique_combinations(self):
        selector = HoldoutModelSelector(
            degrees=[3, 1, 2, 3],
            l2_lambdas=[1.0, 0.0, 1.0],
            max_degree=10,
        )
        self.assertEqual(selector.degrees, [1, 2, 3])
        self.assertEqual(selector.l2_lambdas, [0.0, 1.0])


class TestCSVProvenance(unittest.TestCase):
    """Tests for zero-based observation indexing and physical CSV line numbers."""

    def test_provenance_tracking_with_skipped_rows(self, tmp_path=None):
        if tmp_path is None:
            import tempfile

            tmp_dir = pathlib.Path(tempfile.mkdtemp())
        else:
            tmp_dir = tmp_path

        csv_file = tmp_dir / "test_data.csv"
        csv_file.write_text(
            "X,Y\n"
            "1.0,2.0\n"  # line 2, obs_idx 0 (valid)
            "invalid,3.0\n"  # line 3, obs_idx 1 (skipped)
            "2.0,4.0\n"  # line 4, obs_idx 2 (valid)
            "3.0,NaN\n"  # line 5, obs_idx 3 (skipped)
            "4.0,8.0\n",  # line 6, obs_idx 4 (valid)
            encoding="utf-8",
        )

        loader = CSVDataLoader(csv_file, skip_invalid_rows=True)
        loaded = loader.load()

        np.testing.assert_array_equal(loaded.x, [1.0, 2.0, 4.0])
        np.testing.assert_array_equal(loaded.y, [2.0, 4.0, 8.0])
        np.testing.assert_array_equal(loaded.observation_indices, [0, 2, 4])
        np.testing.assert_array_equal(loaded.csv_line_numbers, [2, 4, 6])
        np.testing.assert_array_equal(loaded.original_indices, [0, 2, 4])

        self.assertEqual(len(loaded.skipped_rows), 2)
        sr1, sr2 = loaded.skipped_rows
        self.assertEqual(sr1.observation_index, 1)
        self.assertEqual(sr1.csv_line_number, 3)
        self.assertEqual(sr2.observation_index, 3)
        self.assertEqual(sr2.csv_line_number, 5)

    def test_reporting_mismatched_array_lengths_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_equal_lengths(
                {
                    "a": np.array([1, 2, 3]),
                    "b": np.array([1, 2]),
                }
            )
        self.assertIn("equal lengths", str(ctx.exception))


class TestEndToEndPriority3(unittest.TestCase):
    """End-to-end integration tests verifying metadata and CSV outputs."""

    def test_run_metadata_and_prediction_schemas(self):
        import tempfile

        tmp_dir = pathlib.Path(tempfile.mkdtemp())
        csv_file = tmp_dir / "data.csv"
        csv_file.write_text(
            "X,Y\n"
            "1.0,2.0\n"
            "2.0,4.0\n"
            "3.0,6.0\n"
            "4.0,8.0\n"
            "5.0,10.0\n"
            "6.0,12.0\n"
            "7.0,14.0\n"
            "8.0,16.0\n"
            "9.0,18.0\n"
            "10.0,20.0\n",
            encoding="utf-8",
        )

        out_dir = tmp_dir / "results"
        args = parse_args(
            [
                str(csv_file),
                "--output-dir",
                str(out_dir),
                "--mode",
                "both",
                "--no-plots",
                "--degrees",
                "2",
                "1",
                "2",
                "--l2-values",
                "0.1",
                "0.0",
                "0.1",
                "--folds",
                "2",
            ]
        )

        run_pipeline(args)

        meta_file = out_dir / "common" / "run_metadata.json"
        self.assertTrue(meta_file.exists())
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["schema_version"], 1)
        self.assertIn("numpy", meta["packages"])
        self.assertIn("matplotlib", meta["packages"])
        self.assertEqual(meta["search_grid"]["effective"]["degrees"], [1, 2])
        self.assertEqual(meta["search_grid"]["effective"]["l2_values"], [0.0, 0.1])
        self.assertTrue(meta["search_grid"]["canonicalization"]["duplicates_removed"])
        self.assertIn("sha256", meta["input_file"])
        self.assertGreater(meta["input_file"]["size_bytes"], 0)

        # Check prediction CSV header
        test_pred_file = out_dir / "holdout" / "test_predictions.csv"
        self.assertTrue(test_pred_file.exists())
        header_line = test_pred_file.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(
            header_line,
            "observation_index,csv_line_number,X,actual_Y,predicted_Y,residual",
        )

        oof_pred_file = out_dir / "kfold" / "out_of_fold_predictions.csv"
        self.assertTrue(oof_pred_file.exists())
        oof_header = oof_pred_file.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(
            oof_header,
            "observation_index,csv_line_number,fold_number,X,actual_Y,oof_predicted_Y,residual",
        )

        # Check split summary
        split_file = out_dir / "common" / "split_summary.json"
        with open(split_file, "r", encoding="utf-8") as f:
            split_summary = json.load(f)
        self.assertIn("loaded_array_indices", split_summary["indices"])
        self.assertIn("observation_indices", split_summary["indices"])
        self.assertIn("csv_line_numbers", split_summary["indices"])


if __name__ == "__main__":
    unittest.main()
