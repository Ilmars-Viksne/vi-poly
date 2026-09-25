"""Tests for CSVDataLoader and generate_data module."""

import pathlib
import tempfile
import unittest

import numpy as np

from generate_data import generate_synthetic_data, save_to_csv
from polynomial_regression.data import CSVDataLoader


class TestCSVDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_csv(self):
        csv_file = self.temp_path / "valid.csv"
        x, y = generate_synthetic_data(
            50, -5.0, 5.0, [1.0, 2.0], noise_std=0.1, seed=42
        )
        save_to_csv(csv_file, x, y)

        loader = CSVDataLoader(csv_file, x_column="X", y_column="Y")
        loaded = loader.load()

        self.assertEqual(len(loaded.x), 50)
        self.assertEqual(len(loaded.y), 50)
        self.assertEqual(len(loaded.original_indices), 50)
        self.assertEqual(len(loaded.skipped_rows), 0)
        np.testing.assert_allclose(loaded.x, x, atol=1e-5)
        np.testing.assert_allclose(loaded.y, y, atol=1e-5)

    def test_missing_column(self):
        csv_file = self.temp_path / "missing_col.csv"
        with open(csv_file, "w") as f:
            f.write("A,B\n1,2\n3,4\n")

        loader = CSVDataLoader(csv_file, x_column="X", y_column="Y")
        with self.assertRaises(ValueError) as ctx:
            loader.load()
        self.assertIn("Required column(s)", str(ctx.exception))

    def test_invalid_values_raise_error(self):
        csv_file = self.temp_path / "invalid.csv"
        with open(csv_file, "w") as f:
            f.write("X,Y\n1.0,2.0\ninvalid,3.0\n3.0,nan\n4.0,inf\n5.0,6.0\n")

        loader = CSVDataLoader(csv_file, skip_invalid_rows=False)
        with self.assertRaises(ValueError) as ctx:
            loader.load()
        self.assertIn("Found 3 invalid row(s)", str(ctx.exception))

    def test_invalid_values_skipped(self):
        csv_file = self.temp_path / "invalid_skip.csv"
        with open(csv_file, "w") as f:
            f.write("X,Y\n1.0,2.0\ninvalid,3.0\n3.0,nan\n4.0,inf\n5.0,6.0\n")

        loader = CSVDataLoader(csv_file, skip_invalid_rows=True)
        loaded = loader.load()

        self.assertEqual(len(loaded.x), 2)
        self.assertEqual(len(loaded.y), 2)
        self.assertEqual(len(loaded.skipped_rows), 3)
        np.testing.assert_array_equal(loaded.x, [1.0, 5.0])
        np.testing.assert_array_equal(loaded.y, [2.0, 6.0])
        np.testing.assert_array_equal(loaded.original_indices, [0, 4])

    def test_whitespace_headers_and_custom_columns(self):
        csv_file = self.temp_path / "whitespace_headers.csv"
        with open(csv_file, "w") as f:
            f.write(" X , Y \n1.0,2.0\n3.0,4.0\n")

        loader = CSVDataLoader(csv_file, x_column=" X ", y_column="Y")
        loaded = loader.load()

        self.assertEqual(len(loaded.x), 2)
        np.testing.assert_array_equal(loaded.x, [1.0, 3.0])
        np.testing.assert_array_equal(loaded.y, [2.0, 4.0])

    def test_duplicate_headers_after_normalization(self):
        csv_file = self.temp_path / "dup_headers.csv"
        with open(csv_file, "w") as f:
            f.write("X, X ,Y\n1.0,2.0,3.0\n")

        loader = CSVDataLoader(csv_file, x_column="X", y_column="Y")
        with self.assertRaises(ValueError) as ctx:
            loader.load()
        self.assertIn("duplicate normalized column headers", str(ctx.exception))

    def test_empty_normalized_header(self):
        csv_file = self.temp_path / "empty_header.csv"
        with open(csv_file, "w") as f:
            f.write("X,  ,Y\n1.0,2.0,3.0\n")

        loader = CSVDataLoader(csv_file, x_column="X", y_column="Y")
        with self.assertRaises(ValueError) as ctx:
            loader.load()
        self.assertIn("empty or whitespace-only column header", str(ctx.exception))

    def test_empty_requested_column_name(self):
        csv_file = self.temp_path / "valid.csv"
        with open(csv_file, "w") as f:
            f.write("X,Y\n1.0,2.0\n")

        loader = CSVDataLoader(csv_file, x_column="   ", y_column="Y")
        with self.assertRaises(ValueError) as ctx:
            loader.load()
        self.assertIn("Requested X column name cannot be empty", str(ctx.exception))

    def test_missing_requested_column_after_normalization(self):
        csv_file = self.temp_path / "valid.csv"
        with open(csv_file, "w") as f:
            f.write("X,Y\n1.0,2.0\n")

        loader = CSVDataLoader(csv_file, x_column="Z", y_column="Y")
        with self.assertRaises(ValueError) as ctx:
            loader.load()
        self.assertIn("Required column(s)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
