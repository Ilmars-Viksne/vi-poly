"""Serialization and reporting generator for numerical outputs and plot manifests."""

import csv
import json
import pathlib
from typing import Any

import numpy as np

from .metrics import EvaluationMetrics
from .selection import HoldoutSelectionResult, KFoldSelectionResult


class ReportGenerator:
    """Handles serialization of numerical outputs into CSV and JSON files."""

    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        fold_assignments: dict[int, list[int]] | None = None,
        skipped_rows: list[tuple] | None = None,
    ) -> pathlib.Path:
        filepath = self.output_dir / "split_summary.json"
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
                "train_indices": train_indices.tolist(),
                "validation_indices": val_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "development_indices": dev_indices.tolist(),
            },
            "fold_assignments": fold_assignments or {},
            "skipped_rows": [
                {"row_num": r[0], "columns": r[1], "reason": r[2]}
                for r in (skipped_rows or [])
            ],
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
        solver_used: str,
        condition_number: float,
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
            "solver_used": solver_used,
            "condition_number": condition_number,
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
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(model_dict, f, indent=2)

        return filepath

    def write_test_predictions(
        self,
        original_indices: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        test_preds: np.ndarray,
        test_residuals: np.ndarray,
    ) -> pathlib.Path:
        filepath = self.output_dir / "test_predictions.csv"
        fieldnames = ["original_csv_index", "X", "actual_Y", "predicted_Y", "residual"]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for orig_idx, xi, yi, pi, ri in zip(
                original_indices, x_test, y_test, test_preds, test_residuals
            ):
                writer.writerow(
                    {
                        "original_csv_index": orig_idx,
                        "X": xi,
                        "actual_Y": yi,
                        "predicted_Y": pi,
                        "residual": ri,
                    }
                )

        return filepath

    def write_oof_predictions(
        self,
        original_dev_indices: np.ndarray,
        dev_fold_assignments: np.ndarray,
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        oof_preds: np.ndarray,
        oof_residuals: np.ndarray,
    ) -> pathlib.Path:
        filepath = self.output_dir / "out_of_fold_predictions.csv"
        fieldnames = [
            "original_csv_index",
            "fold_number",
            "X",
            "actual_Y",
            "oof_predicted_Y",
            "residual",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for orig_idx, fold_num, xi, yi, pi, ri in zip(
                original_dev_indices,
                dev_fold_assignments,
                x_dev,
                y_dev,
                oof_preds,
                oof_residuals,
            ):
                writer.writerow(
                    {
                        "original_csv_index": orig_idx,
                        "fold_number": fold_num,
                        "X": xi,
                        "actual_Y": yi,
                        "oof_predicted_Y": pi,
                        "residual": ri,
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
