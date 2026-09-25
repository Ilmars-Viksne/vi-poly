"""Serialization and reporting generator for numerical outputs and plot manifests."""

import argparse
import csv
import hashlib
import json
import pathlib
import platform
import sys
from typing import Any

import matplotlib
import numpy as np

from .metrics import EvaluationMetrics
from .regression import ModelFitDetails
from .selection import HoldoutSelectionResult, KFoldSelectionResult


def _validate_equal_lengths(arrays: dict[str, np.ndarray]) -> int:
    """Validates that all supplied 1D arrays have identical non-zero lengths and finite numeric values."""
    lengths = {name: len(values) for name, values in arrays.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(f"Report arrays must have equal lengths; received {lengths}.")

    n = next(iter(unique_lengths))
    if n == 0:
        raise ValueError("Report arrays must not be empty.")

    for name, arr in arrays.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim != 1:
            raise ValueError(
                f"Array '{name}' must be 1D; received shape {arr_np.shape}."
            )
        if not np.all(np.isfinite(arr_np)):
            non_finite_count = int(np.sum(~np.isfinite(arr_np)))
            raise ValueError(
                f"Array '{name}' contains {non_finite_count} non-finite value(s)."
            )

    return n


class ReportGenerator:
    """Handles serialization of numerical outputs into CSV and JSON files."""

    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(
        self,
        args: argparse.Namespace,
        csv_filepath: str | pathlib.Path,
    ) -> pathlib.Path:
        filepath = self.output_dir / "run_metadata.json"
        csv_path = pathlib.Path(csv_filepath)

        input_file_meta = None
        if csv_path.exists() and csv_path.is_file():
            size_bytes = csv_path.stat().st_size
            digest = hashlib.sha256()
            with open(csv_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            input_file_meta = {
                "path": str(csv_path),
                "size_bytes": size_bytes,
                "sha256": digest.hexdigest(),
            }

        raw_argv = getattr(args, "_raw_argv", None)
        if raw_argv is not None:
            argv = ["main.py", *raw_argv]
        else:
            argv = sys.argv[:]

        parsed_args_dict = {}
        for k, v in vars(args).items():
            if not k.startswith("_"):
                parsed_args_dict[k] = v

        req_degrees = getattr(args, "_requested_degrees", getattr(args, "degrees", []))
        req_l2 = getattr(args, "_requested_l2_values", getattr(args, "l2_values", []))
        eff_degrees = getattr(args, "degrees", [])
        eff_l2 = getattr(args, "l2_values", [])

        duplicates_removed = (
            len(req_degrees) != len(eff_degrees)
            or len(req_l2) != len(eff_l2)
            or list(req_degrees) != list(eff_degrees)
            or list(req_l2) != list(eff_l2)
        )

        metadata = {
            "schema_version": 1,
            "command": {
                "argv": argv,
                "arguments": parsed_args_dict,
            },
            "runtime": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
            "packages": {
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "search_grid": {
                "requested": {
                    "degrees": req_degrees,
                    "l2_values": req_l2,
                },
                "effective": {
                    "degrees": eff_degrees,
                    "l2_values": eff_l2,
                },
                "canonicalization": {
                    "duplicates_removed": duplicates_removed,
                    "ordering": "ascending",
                    "float_deduplication": "exact_equality",
                },
            },
            "random_seeds": {
                "split_seed": getattr(args, "seed", None),
                "kfold_seed": getattr(args, "seed", None),
                "bootstrap_seed": getattr(args, "bootstrap_seed", None),
            },
            "input_file": input_file_meta,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return filepath

    def write_split_summary(
        self,
        seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        total_samples: int,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        dev_indices: np.ndarray,
        train_observation_indices: np.ndarray | None = None,
        val_observation_indices: np.ndarray | None = None,
        test_observation_indices: np.ndarray | None = None,
        dev_observation_indices: np.ndarray | None = None,
        train_csv_line_numbers: np.ndarray | None = None,
        val_csv_line_numbers: np.ndarray | None = None,
        test_csv_line_numbers: np.ndarray | None = None,
        dev_csv_line_numbers: np.ndarray | None = None,
        fold_assignments: dict[str, list[int]] | None = None,
        skipped_rows: list[Any] | None = None,
    ) -> pathlib.Path:
        filepath = self.output_dir / "split_summary.json"

        # Fall back to loaded_array_indices if provenance arrays are omitted
        tr_obs = (
            train_observation_indices.tolist()
            if train_observation_indices is not None
            else train_indices.tolist()
        )
        v_obs = (
            val_observation_indices.tolist()
            if val_observation_indices is not None
            else val_indices.tolist()
        )
        te_obs = (
            test_observation_indices.tolist()
            if test_observation_indices is not None
            else test_indices.tolist()
        )
        d_obs = (
            dev_observation_indices.tolist()
            if dev_observation_indices is not None
            else dev_indices.tolist()
        )

        tr_line = (
            train_csv_line_numbers.tolist()
            if train_csv_line_numbers is not None
            else (train_indices + 2).tolist()
        )
        v_line = (
            val_csv_line_numbers.tolist()
            if val_csv_line_numbers is not None
            else (val_indices + 2).tolist()
        )
        te_line = (
            test_csv_line_numbers.tolist()
            if test_csv_line_numbers is not None
            else (test_indices + 2).tolist()
        )
        d_line = (
            dev_csv_line_numbers.tolist()
            if dev_csv_line_numbers is not None
            else (dev_indices + 2).tolist()
        )

        formatted_skipped = []
        for r in skipped_rows or []:
            if hasattr(r, "observation_index") and hasattr(r, "csv_line_number"):
                formatted_skipped.append(
                    {
                        "observation_index": r.observation_index,
                        "csv_line_number": r.csv_line_number,
                        "columns": r.columns,
                        "reason": r.reason,
                    }
                )
            elif isinstance(r, dict):
                formatted_skipped.append(
                    {
                        "observation_index": r.get(
                            "observation_index", r.get("row_num", 0)
                        ),
                        "csv_line_number": r.get(
                            "csv_line_number", r.get("row_num", 1)
                        ),
                        "columns": r.get("columns", ""),
                        "reason": r.get("reason", ""),
                    }
                )
            elif isinstance(r, (list, tuple)):
                row_n = r[0]
                formatted_skipped.append(
                    {
                        "observation_index": row_n - 2 if row_n >= 2 else 0,
                        "csv_line_number": row_n,
                        "columns": r[1] if len(r) > 1 else "",
                        "reason": r[2] if len(r) > 2 else "",
                    }
                )

        summary = {
            "random_seed": seed,
            "split_ratios": {
                "train_ratio": train_ratio,
                "validation_ratio": val_ratio,
                "test_ratio": test_ratio,
            },
            "counts": {
                "total_loaded_samples": total_samples,
                "train_count": len(train_indices),
                "validation_count": len(val_indices),
                "test_count": len(test_indices),
                "development_count": len(dev_indices),
            },
            "indices": {
                "loaded_array_indices": {
                    "train": train_indices.tolist(),
                    "validation": val_indices.tolist(),
                    "test": test_indices.tolist(),
                    "development": dev_indices.tolist(),
                },
                "observation_indices": {
                    "train": tr_obs,
                    "validation": v_obs,
                    "test": te_obs,
                    "development": d_obs,
                },
                "csv_line_numbers": {
                    "train": tr_line,
                    "validation": v_line,
                    "test": te_line,
                    "development": d_line,
                },
            },
            "fold_assignments": fold_assignments or {},
            "skipped_rows": formatted_skipped,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return filepath

    def write_holdout_results(
        self, holdout_res: HoldoutSelectionResult
    ) -> pathlib.Path:
        filepath = self.output_dir / "holdout_results.csv"
        fieldnames = [
            "degree",
            "l2_lambda",
            "train_mse",
            "train_rmse",
            "train_mae",
            "train_r_squared",
            "val_mse",
            "val_rmse",
            "val_mae",
            "val_r_squared",
            "condition_number",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cand in holdout_res.candidates:
                writer.writerow(
                    {
                        "degree": cand.degree,
                        "l2_lambda": cand.l2_lambda,
                        "train_mse": cand.train_metrics.mse,
                        "train_rmse": cand.train_metrics.rmse,
                        "train_mae": cand.train_metrics.mae,
                        "train_r_squared": cand.train_metrics.r_squared,
                        "val_mse": cand.val_metrics.mse,
                        "val_rmse": cand.val_metrics.rmse,
                        "val_mae": cand.val_metrics.mae,
                        "val_r_squared": cand.val_metrics.r_squared,
                        "condition_number": cand.condition_number,
                    }
                )
        return filepath

    def write_kfold_results(
        self, kfold_res: KFoldSelectionResult
    ) -> tuple[pathlib.Path, pathlib.Path]:
        fold_file = self.output_dir / "kfold_fold_results.csv"
        summary_file = self.output_dir / "kfold_summary_results.csv"

        fold_fields = [
            "degree",
            "l2_lambda",
            "fold_number",
            "train_mse",
            "train_rmse",
            "train_mae",
            "train_r_squared",
            "val_mse",
            "val_rmse",
            "val_mae",
            "val_r_squared",
            "condition_number",
        ]

        summary_fields = [
            "degree",
            "l2_lambda",
            "mean_val_mse",
            "mean_val_rmse",
            "mean_val_mae",
            "mean_val_r_squared",
            "std_val_mse",
            "std_val_rmse",
            "std_val_mae",
            "std_val_r_squared",
            "mean_condition_number",
        ]

        with open(fold_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fold_fields)
            writer.writeheader()
            for cand in kfold_res.candidates:
                for fr in cand.fold_results:
                    writer.writerow(
                        {
                            "degree": cand.degree,
                            "l2_lambda": cand.l2_lambda,
                            "fold_number": fr.fold_index + 1,
                            "train_mse": fr.train_metrics.mse,
                            "train_rmse": fr.train_metrics.rmse,
                            "train_mae": fr.train_metrics.mae,
                            "train_r_squared": fr.train_metrics.r_squared,
                            "val_mse": fr.val_metrics.mse,
                            "val_rmse": fr.val_metrics.rmse,
                            "val_mae": fr.val_metrics.mae,
                            "val_r_squared": fr.val_metrics.r_squared,
                            "condition_number": fr.condition_number,
                        }
                    )

        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            for cand in kfold_res.candidates:
                writer.writerow(
                    {
                        "degree": cand.degree,
                        "l2_lambda": cand.l2_lambda,
                        "mean_val_mse": cand.mean_val_metrics.mse,
                        "mean_val_rmse": cand.mean_val_metrics.rmse,
                        "mean_val_mae": cand.mean_val_metrics.mae,
                        "mean_val_r_squared": cand.mean_val_metrics.r_squared,
                        "std_val_mse": cand.std_val_metrics.mse,
                        "std_val_rmse": cand.std_val_metrics.rmse,
                        "std_val_mae": cand.std_val_metrics.mae,
                        "std_val_r_squared": cand.std_val_metrics.r_squared,
                        "mean_condition_number": cand.mean_condition_number,
                    }
                )

        return fold_file, summary_file

    def write_final_model(
        self,
        workflow_name: str,
        best_candidate: Any,
        beta_scaled: np.ndarray,
        beta_orig: np.ndarray,
        transformer: Any,
        test_metrics: EvaluationMetrics,
        fit_details: ModelFitDetails,
        selection_rtol: float,
        selection_atol: float,
    ) -> pathlib.Path:
        filepath = self.output_dir / "final_model.json"

        scaling_params = None
        if transformer.scale_features and transformer.scaling_params is not None:
            scaling_params = {
                "means": transformer.scaling_params.means.tolist(),
                "stds": transformer.scaling_params.stds.tolist(),
            }

        model_dict = {
            "selected_workflow": workflow_name,
            "selected_degree": best_candidate.degree,
            "selected_l2_lambda": best_candidate.l2_lambda,
            "coefficients_ordering_convention": "Ascending polynomial order: [beta_0, beta_1*x, beta_2*x^2, ...]",
            "coefficients_scaled_basis": beta_scaled.tolist(),
            "coefficients_original_basis": beta_orig.tolist(),
            "scale_features_enabled": transformer.scale_features,
            "scaling_parameters": scaling_params,
            "solver_used": fit_details.solver_used,
            "condition_number": fit_details.condition_number,
            "condition_warning": fit_details.condition_warning,
            "rank": fit_details.rank,
            "full_rank": fit_details.full_rank,
            "design_condition_number": fit_details.design_condition_number,
            "solver_condition_number": fit_details.solver_condition_number,
            "selection_tolerances": {
                "rtol": selection_rtol,
                "atol": selection_atol,
            },
            "final_test_metrics": {
                "mse": test_metrics.mse,
                "rmse": test_metrics.rmse,
                "mae": test_metrics.mae,
                "r_squared": test_metrics.r_squared,
                "adjusted_r_squared": test_metrics.adjusted_r_squared,
            },
            "run_metadata_file": "../common/run_metadata.json",
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(model_dict, f, indent=2)

        return filepath

    def write_test_predictions(
        self,
        observation_indices: np.ndarray,
        csv_line_numbers: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        test_preds: np.ndarray,
        test_residuals: np.ndarray,
    ) -> pathlib.Path:
        filepath = self.output_dir / "test_predictions.csv"
        _validate_equal_lengths(
            {
                "observation_indices": observation_indices,
                "csv_line_numbers": csv_line_numbers,
                "x_test": x_test,
                "y_test": y_test,
                "test_preds": test_preds,
                "test_residuals": test_residuals,
            }
        )

        obs_arr = np.asarray(observation_indices, dtype=np.int64)
        line_arr = np.asarray(csv_line_numbers, dtype=np.int64)
        if np.any(obs_arr < 0):
            raise ValueError("observation_indices must contain non-negative integers.")
        if np.any(line_arr < 1):
            raise ValueError("csv_line_numbers must contain positive integers.")

        fieldnames = [
            "observation_index",
            "csv_line_number",
            "X",
            "actual_Y",
            "predicted_Y",
            "residual",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for obs_idx, line_num, xi, yi, pi, ri in zip(
                obs_arr, line_arr, x_test, y_test, test_preds, test_residuals
            ):
                writer.writerow(
                    {
                        "observation_index": int(obs_idx),
                        "csv_line_number": int(line_num),
                        "X": float(xi),
                        "actual_Y": float(yi),
                        "predicted_Y": float(pi),
                        "residual": float(ri),
                    }
                )

        return filepath

    def write_oof_predictions(
        self,
        observation_indices: np.ndarray,
        csv_line_numbers: np.ndarray,
        dev_fold_assignments: np.ndarray,
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        oof_preds: np.ndarray,
        oof_residuals: np.ndarray,
    ) -> pathlib.Path:
        filepath = self.output_dir / "out_of_fold_predictions.csv"
        _validate_equal_lengths(
            {
                "observation_indices": observation_indices,
                "csv_line_numbers": csv_line_numbers,
                "dev_fold_assignments": dev_fold_assignments,
                "x_dev": x_dev,
                "y_dev": y_dev,
                "oof_preds": oof_preds,
                "oof_residuals": oof_residuals,
            }
        )

        obs_arr = np.asarray(observation_indices, dtype=np.int64)
        line_arr = np.asarray(csv_line_numbers, dtype=np.int64)
        fold_arr = np.asarray(dev_fold_assignments, dtype=np.int64)

        if np.any(obs_arr < 0):
            raise ValueError("observation_indices must contain non-negative integers.")
        if np.any(line_arr < 1):
            raise ValueError("csv_line_numbers must contain positive integers.")
        if np.any(fold_arr < 1):
            raise ValueError("fold_number values must be positive integers.")

        fieldnames = [
            "observation_index",
            "csv_line_number",
            "fold_number",
            "X",
            "actual_Y",
            "oof_predicted_Y",
            "residual",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for obs_idx, line_num, fold_num, xi, yi, pi, ri in zip(
                obs_arr,
                line_arr,
                fold_arr,
                x_dev,
                y_dev,
                oof_preds,
                oof_residuals,
            ):
                writer.writerow(
                    {
                        "observation_index": int(obs_idx),
                        "csv_line_number": int(line_num),
                        "fold_number": int(fold_num),
                        "X": float(xi),
                        "actual_Y": float(yi),
                        "oof_predicted_Y": float(pi),
                        "residual": float(ri),
                    }
                )

        return filepath

    def write_plot_manifest(
        self, manifest_entries: list[dict[str, Any]]
    ) -> pathlib.Path:
        filepath = self.output_dir / "plot_manifest.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, indent=2)
        return filepath

    def write_comparison(
        self,
        holdout_res: HoldoutSelectionResult,
        kfold_res: KFoldSelectionResult,
    ) -> pathlib.Path:
        filepath = self.output_dir / "comparison.json"
        data = {
            "mode": "both",
            "selection_policy": (
                "Each workflow selects hyperparameters using development data only. "
                "Test results are reported independently and are not used to select a workflow."
            ),
            "workflow_selection": {
                "holdout": {
                    "selection_metric": "validation_rmse",
                    "selected_degree": holdout_res.best_candidate.degree,
                    "selected_l2_lambda": holdout_res.best_candidate.l2_lambda,
                },
                "kfold": {
                    "selection_metric": "mean_cross_validation_rmse",
                    "selected_degree": kfold_res.best_candidate.degree,
                    "selected_l2_lambda": kfold_res.best_candidate.l2_lambda,
                },
            },
            "overall_best_workflow": None,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath
