"""CSV Data Loader module for polynomial regression."""

import csv
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoadedData:
    """Class representing loaded and parsed numerical data from CSV."""

    x: np.ndarray
    y: np.ndarray
    original_indices: np.ndarray
    skipped_rows: list[tuple[int, str, str]] = field(
        default_factory=list
    )  # (row_num, col, reason)


class CSVDataLoader:
    """Loads X and Y numerical columns from a CSV file."""

    def __init__(
        self,
        filepath: str | pathlib.Path,
        x_column: str = "X",
        y_column: str = "Y",
        skip_invalid_rows: bool = False,
    ):
        self.filepath = pathlib.Path(filepath)
        self.x_column = x_column
        self.y_column = y_column
        self.skip_invalid_rows = skip_invalid_rows

    def load(self) -> LoadedData:
        """Reads CSV and parses X and Y columns into NumPy arrays.

        Returns:
            LoadedData container with numpy arrays x, y, original row indices,
            and skipped row details.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If columns are missing or if invalid values exist and skip_invalid_rows is False.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")

        valid_x: list[float] = []
        valid_y: list[float] = []
        original_indices: list[int] = []
        skipped_rows: list[tuple[int, str, str]] = []
        invalid_row_errors: list[str] = []

        with open(self.filepath, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(
                    f"CSV file '{self.filepath}' is empty or header is missing."
                )

            fieldnames = [field.strip() if field else "" for field in reader.fieldnames]

            if self.x_column not in fieldnames or self.y_column not in fieldnames:
                missing = []
                if self.x_column not in fieldnames:
                    missing.append(self.x_column)
                if self.y_column not in fieldnames:
                    missing.append(self.y_column)
                raise ValueError(
                    f"Required column(s) {missing} not found in CSV '{self.filepath}'. "
                    f"Available columns: {reader.fieldnames}"
                )

            # DictReader uses 1-based line numbers; line 1 is header, so data starts at row 2
            for row_idx, row in enumerate(reader, start=2):
                x_str = row.get(self.x_column)
                y_str = row.get(self.y_column)

                val_x, err_x = self._parse_float(x_str)
                val_y, err_y = self._parse_float(y_str)

                if err_x is not None or err_y is not None:
                    reasons = []
                    col_affected = []
                    if err_x:
                        reasons.append(f"X ({self.x_column}): {err_x}")
                        col_affected.append(self.x_column)
                    if err_y:
                        reasons.append(f"Y ({self.y_column}): {err_y}")
                        col_affected.append(self.y_column)

                    reason_str = "; ".join(reasons)
                    cols_str = ", ".join(col_affected)
                    skipped_rows.append((row_idx, cols_str, reason_str))
                    invalid_row_errors.append(f"Row {row_idx}: {reason_str}")
                    continue

                valid_x.append(val_x)
                valid_y.append(val_y)
                # Save 0-based data row index or 1-based CSV row number.
                # Preserving 0-based index of observation relative to full dataset or CSV row number.
                # Let's preserve 0-based observation index in CSV (row_idx - 2) or CSV row number.
                # The specification says "original CSV row indices". Keeping 0-based row index (row_idx - 2).
                original_indices.append(row_idx - 2)

        if invalid_row_errors and not self.skip_invalid_rows:
            error_details = "\n  ".join(invalid_row_errors)
            raise ValueError(
                f"Found {len(invalid_row_errors)} invalid row(s) in '{self.filepath}':\n  {error_details}"
            )

        if len(valid_x) == 0:
            raise ValueError(f"No valid data rows found in CSV file '{self.filepath}'.")

        return LoadedData(
            x=np.array(valid_x, dtype=np.float64),
            y=np.array(valid_y, dtype=np.float64),
            original_indices=np.array(original_indices, dtype=np.int64),
            skipped_rows=skipped_rows,
        )

    @staticmethod
    def _parse_float(val_str: str | None) -> tuple[float, str | None]:
        if val_str is None or val_str.strip() == "":
            return 0.0, "Missing or empty value"
        try:
            val = float(val_str.strip())
            if math.isnan(val):
                return 0.0, "Value is NaN"
            if math.isinf(val):
                return 0.0, "Value is infinite"
            return val, None
        except ValueError:
            return 0.0, f"Cannot convert '{val_str}' to float"
