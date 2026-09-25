"""CSV Data Loader module for polynomial regression."""

import csv
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SkippedRow:
    """Provenance details for a skipped invalid row in CSV."""

    observation_index: int
    csv_line_number: int
    columns: str
    reason: str


@dataclass
class LoadedData:
    """Class representing loaded and parsed numerical data from CSV."""

    x: np.ndarray
    y: np.ndarray
    observation_indices: np.ndarray
    csv_line_numbers: np.ndarray
    skipped_rows: list[SkippedRow] = field(default_factory=list)

    @property
    def original_indices(self) -> np.ndarray:
        """Compatibility alias for zero-based observation indices."""
        return self.observation_indices


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
            LoadedData container with numpy arrays x, y, observation_indices,
            csv_line_numbers, and skipped row details.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If columns are missing or if invalid values exist and skip_invalid_rows is False.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")

        valid_x: list[float] = []
        valid_y: list[float] = []
        observation_indices: list[int] = []
        csv_line_numbers: list[int] = []
        skipped_rows: list[SkippedRow] = []
        invalid_row_errors: list[str] = []

        x_col_norm = self.x_column.strip() if self.x_column else ""
        if not x_col_norm:
            raise ValueError(
                "Requested X column name cannot be empty or whitespace-only."
            )

        y_col_norm = self.y_column.strip() if self.y_column else ""
        if not y_col_norm:
            raise ValueError(
                "Requested Y column name cannot be empty or whitespace-only."
            )

        with open(self.filepath, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None or len(reader.fieldnames) == 0:
                raise ValueError(
                    f"CSV file '{self.filepath}' is empty or header is missing."
                )

            seen_headers: dict[str, list[str]] = {}
            normalized_fieldnames: list[str] = []

            for orig in reader.fieldnames:
                orig_str = orig if orig is not None else ""
                norm = orig_str.strip()
                if not norm:
                    raise ValueError(
                        f"CSV file '{self.filepath}' contains an empty or whitespace-only column header. "
                        f"Original headers: {reader.fieldnames}"
                    )
                if norm in seen_headers:
                    seen_headers[norm].append(orig_str)
                else:
                    seen_headers[norm] = [orig_str]
                normalized_fieldnames.append(norm)

            duplicates = {
                norm: origs for norm, origs in seen_headers.items() if len(origs) > 1
            }
            if duplicates:
                dup_details = ", ".join(
                    f"'{norm}' (from original: {origs})"
                    for norm, origs in duplicates.items()
                )
                raise ValueError(
                    f"CSV file '{self.filepath}' contains duplicate normalized column headers: {dup_details}. "
                    f"Available normalized columns: {list(seen_headers.keys())}"
                )

            if x_col_norm not in seen_headers or y_col_norm not in seen_headers:
                missing = []
                if x_col_norm not in seen_headers:
                    missing.append(x_col_norm)
                if y_col_norm not in seen_headers:
                    missing.append(y_col_norm)
                raise ValueError(
                    f"Required column(s) {missing} not found in CSV '{self.filepath}'. "
                    f"Available normalized columns: {list(seen_headers.keys())}"
                )

            reader.fieldnames = normalized_fieldnames

            for obs_idx, row in enumerate(reader):
                line_num = reader.line_num
                x_str = row.get(x_col_norm)
                y_str = row.get(y_col_norm)

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
                    skipped_rows.append(
                        SkippedRow(
                            observation_index=obs_idx,
                            csv_line_number=line_num,
                            columns=cols_str,
                            reason=reason_str,
                        )
                    )
                    invalid_row_errors.append(
                        f"Observation {obs_idx} (Line {line_num}): {reason_str}"
                    )
                    continue

                valid_x.append(val_x)
                valid_y.append(val_y)
                observation_indices.append(obs_idx)
                csv_line_numbers.append(line_num)

        if invalid_row_errors and not self.skip_invalid_rows:
            error_details = "\n  ".join(invalid_row_errors)
            raise ValueError(
                f"Found {len(invalid_row_errors)} invalid row(s) in '{self.filepath}':\n  {error_details}"
            )

        if len(valid_x) == 0:
            raise ValueError(f"No valid data rows found in CSV file '{self.filepath}'.")

        x_arr = np.array(valid_x, dtype=np.float64)
        y_arr = np.array(valid_y, dtype=np.float64)
        obs_arr = np.array(observation_indices, dtype=np.int64)
        line_arr = np.array(csv_line_numbers, dtype=np.int64)

        # Validate invariants
        if not (len(x_arr) == len(y_arr) == len(obs_arr) == len(line_arr)):
            raise ValueError("Loaded data array lengths must be equal.")
        if obs_arr.ndim != 1 or line_arr.ndim != 1:
            raise ValueError("Provenance arrays must be 1D.")
        if not np.issubdtype(obs_arr.dtype, np.integer) or not np.issubdtype(
            line_arr.dtype, np.integer
        ):
            raise ValueError("Provenance arrays must have integer dtype.")
        if len(set(obs_arr)) != len(obs_arr):
            raise ValueError("Observation indices must be unique.")
        if np.any(obs_arr < 0):
            raise ValueError("Observation indices must be non-negative.")
        if np.any(line_arr < 1):
            raise ValueError("CSV line numbers must be positive.")

        return LoadedData(
            x=x_arr,
            y=y_arr,
            observation_indices=obs_arr,
            csv_line_numbers=line_arr,
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
