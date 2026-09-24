"""Scientific visualization module for polynomial regression using Matplotlib."""

import pathlib
from typing import Any

import matplotlib

# Force non-interactive Agg backend if no display requested
import matplotlib.pyplot as plt
import numpy as np

from .metrics import EvaluationMetrics


class RegressionVisualizer:
    """Creates scientific plots for polynomial regression analysis."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        plot_format: str = "png",
        plot_dpi: int = 150,
        plot_style: str | None = None,
        show_plots: bool = False,
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_format = plot_format.lower().lstrip(".")
        self.plot_dpi = plot_dpi
        self.show_plots = show_plots

        if not self.show_plots:
            matplotlib.use("Agg")

        if plot_style and plot_style in plt.style.available:
            plt.style.use(plot_style)

        # Color-blind friendly, accessible palette
        self.colors = {
            "train": "#0072B2",  # Blue
            "val": "#E69F00",  # Orange
            "test": "#009E73",  # Green/Teal
            "dev": "#56B4E9",  # Light Blue
            "fit": "#D55E00",  # Red/Vermilion
            "reference": "#000000",  # Black
            "bootstrap": "#CC79A7",  # Pink/Purple
        }

        self.manifest_entries: list[dict[str, Any]] = []

    def _save_and_close(
        self,
        fig: plt.Figure,
        filename_stem: str,
        title: str,
        plot_type: str,
        workflow: str,
        parameters: dict[str, Any],
        source_file: str,
    ) -> pathlib.Path:
        filename = f"{filename_stem}.{self.plot_format}"
        filepath = self.output_dir / filename

        fig.savefig(filepath, dpi=self.plot_dpi, bbox_inches="tight")

        if self.show_plots:
            plt.show()
        plt.close(fig)

        self.manifest_entries.append(
            {
                "plot_filename": filename,
                "plot_title": title,
                "plot_type": plot_type,
                "workflow": workflow,
                "parameters_represented": parameters,
                "source_numerical_result_file": source_file,
                "plot_format": self.plot_format,
                "dpi": self.plot_dpi
                if self.plot_format in ["png", "jpg", "jpeg"]
                else None,
            }
        )

        return filepath

    # 1. Original Data
    def plot_01_original_data(self, x: np.ndarray, y: np.ndarray) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            x,
            y,
            color=self.colors["train"],
            alpha=0.7,
            edgecolors="none",
            s=25,
            label="Observations",
        )
        ax.set_title(f"01 Original Dataset (N = {len(x)})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        return self._save_and_close(
            fig,
            "01_original_data",
            "Original Dataset",
            "Scatter",
            "General",
            {"sample_count": len(x)},
            "split_summary.json",
        )

    # 2. Train-Val-Test Split
    def plot_02_data_splits(
        self,
        x: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            x[train_idx],
            y[train_idx],
            color=self.colors["train"],
            marker="o",
            alpha=0.7,
            s=25,
            label=f"Train ({len(train_idx)})",
        )
        ax.scatter(
            x[val_idx],
            y[val_idx],
            color=self.colors["val"],
            marker="s",
            alpha=0.8,
            s=30,
            label=f"Validation ({len(val_idx)})",
        )
        ax.scatter(
            x[test_idx],
            y[test_idx],
            color=self.colors["test"],
            marker="^",
            alpha=0.9,
            s=35,
            label=f"Test ({len(test_idx)})",
        )

        ax.set_title("02 Train / Validation / Test Data Splits")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        return self._save_and_close(
            fig,
            "02_data_splits",
            "Data Splits",
            "Scatter",
            "Holdout",
            {
                "train_count": len(train_idx),
                "val_count": len(val_idx),
                "test_count": len(test_idx),
            },
            "split_summary.json",
        )

    # 3. Split Sizes Bar Chart
    def plot_03_split_sizes(
        self, train_count: int, val_count: int, test_count: int
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(6, 4))
        total = train_count + val_count + test_count
        categories = ["Train", "Validation", "Test"]
        counts = [train_count, val_count, test_count]
        colors = [self.colors["train"], self.colors["val"], self.colors["test"]]

        bars = ax.bar(
            categories,
            counts,
            color=colors,
            width=0.5,
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, count in zip(bars, counts):
            pct = (count / total) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
            )

        ax.set_ylim(0, max(counts) * 1.2)
        ax.set_title("03 Subset Sample Allocation")
        ax.set_ylabel("Number of Samples")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        return self._save_and_close(
            fig,
            "03_split_sizes",
            "Split Sizes",
            "Bar Chart",
            "General",
            {"total_samples": total},
            "split_summary.json",
        )

    # 4. Holdout Validation RMSE Curves
    def plot_04_holdout_validation_rmse(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        degrees = sorted({c.degree for c in candidates})
        l2_values = sorted({c.l2_lambda for c in candidates})

        for l2 in l2_values:
            cands_l2 = [c for c in candidates if c.l2_lambda == l2]
            cands_l2.sort(key=lambda c: c.degree)
            degs = [c.degree for c in cands_l2]
            rmses = [c.val_metrics.rmse for c in cands_l2]

            label_str = f"L2 = {l2}" if l2 > 0 else "OLS (L2 = 0)"
            ax.plot(degs, rmses, marker="o", label=label_str, alpha=0.8)

        # Mark selected
        ax.plot(
            best_degree,
            next(
                c.val_metrics.rmse
                for c in candidates
                if c.degree == best_degree and c.l2_lambda == best_l2
            ),
            marker="*",
            markersize=14,
            color="red",
            label=f"Selected (deg={best_degree}, L2={best_l2})",
        )

        ax.set_title("04 Holdout Validation RMSE vs. Polynomial Degree")
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("Validation RMSE")
        ax.set_xticks(degrees)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize="small")

        return self._save_and_close(
            fig,
            "04_holdout_validation_rmse",
            "Holdout Validation RMSE",
            "Line Plot",
            "Holdout",
            {"selected_degree": best_degree, "selected_l2": best_l2},
            "holdout_results.csv",
        )

    # 5. Holdout Train vs Validation Error
    def plot_05_train_validation_error(
        self, candidates: list[Any], selected_l2: float, best_degree: int
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        cands_l2 = [c for c in candidates if c.l2_lambda == selected_l2]
        cands_l2.sort(key=lambda c: c.degree)

        degs = [c.degree for c in cands_l2]
        train_rmse = [c.train_metrics.rmse for c in cands_l2]
        val_rmse = [c.val_metrics.rmse for c in cands_l2]

        ax.plot(
            degs, train_rmse, "o--", color=self.colors["train"], label="Training RMSE"
        )
        ax.plot(degs, val_rmse, "s-", color=self.colors["val"], label="Validation RMSE")

        ax.axvline(
            x=best_degree,
            color="red",
            linestyle=":",
            label=f"Selected Degree ({best_degree})",
        )

        ax.set_title(f"05 Bias-Variance Trade-off (L2 = {selected_l2})")
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("RMSE")
        ax.set_xticks(degs)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        return self._save_and_close(
            fig,
            "05_train_validation_error",
            "Train vs Validation Error",
            "Line Plot",
            "Holdout",
            {"l2_lambda": selected_l2, "selected_degree": best_degree},
            "holdout_results.csv",
        )

    # 6. Holdout RMSE Heatmap
    def plot_06_holdout_rmse_heatmap(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        return self._plot_rmse_heatmap(
            candidates,
            best_degree,
            best_l2,
            title="06 Holdout Validation RMSE Heatmap",
            filename="06_holdout_rmse_heatmap",
            workflow="Holdout",
            rmse_extractor=lambda c: c.val_metrics.rmse,
        )

    # 7. Candidate Polynomial Comparison
    def plot_07_candidate_models(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        candidate_curves: list[
            tuple[int, float, np.ndarray, np.ndarray]
        ],  # (deg, l2, x_grid, y_grid)
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(
            x_train,
            y_train,
            color=self.colors["train"],
            alpha=0.5,
            s=20,
            label="Train Data",
        )
        ax.scatter(
            x_val, y_val, color=self.colors["val"], alpha=0.5, s=20, label="Val Data"
        )

        for deg, l2, x_grid, y_grid in candidate_curves:
            ax.plot(x_grid, y_grid, linewidth=2, label=f"Degree {deg} (L2={l2})")

        ax.set_title("07 Candidate Polynomial Models Comparison")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        return self._save_and_close(
            fig,
            "07_candidate_models",
            "Candidate Models Comparison",
            "Line Plot",
            "Holdout",
            {"candidate_degrees": [c[0] for c in candidate_curves]},
            "holdout_results.csv",
        )

    # 8. K-Fold Assignments Plot
    def plot_08_kfold_assignments(
        self, dev_indices: np.ndarray, fold_splits: list[Any]
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(9, 5))
        k = len(fold_splits)

        for fold in fold_splits:
            fold_num = fold.fold_index + 1
            val_idx = fold.val_indices
            ax.scatter(
                val_idx,
                np.full_like(val_idx, fold_num),
                color=self.colors["val"],
                marker="s",
                s=30,
                label="Validation" if fold_num == 1 else "",
            )

        ax.set_yticks(range(1, k + 1))
        ax.set_yticklabels([f"Fold {i}" for i in range(1, k + 1)])
        ax.set_xlabel("Development Sample Index")
        ax.set_title(f"08 K-Fold Validation Membership Matrix (K = {k})")
        ax.grid(True, linestyle="--", alpha=0.5)

        return self._save_and_close(
            fig,
            "08_kfold_assignments",
            "K-Fold Assignments",
            "Scatter Matrix",
            "KFold",
            {"num_folds": k, "dev_samples": len(dev_indices)},
            "split_summary.json",
        )

    # 9. Cross-Validation Mean RMSE Plot with Error Bars
    def plot_09_kfold_mean_rmse(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        degrees = sorted({c.degree for c in candidates})
        l2_values = sorted({c.l2_lambda for c in candidates})

        for l2 in l2_values:
            cands_l2 = [c for c in candidates if c.l2_lambda == l2]
            cands_l2.sort(key=lambda c: c.degree)
            degs = [c.degree for c in cands_l2]
            means = [c.mean_val_metrics.rmse for c in cands_l2]
            stds = [c.std_val_metrics.rmse for c in cands_l2]

            label_str = f"L2 = {l2}" if l2 > 0 else "OLS (L2 = 0)"
            ax.errorbar(
                degs, means, yerr=stds, fmt="-o", capsize=4, label=label_str, alpha=0.8
            )

        # Mark selected
        sel_cand = next(
            c for c in candidates if c.degree == best_degree and c.l2_lambda == best_l2
        )
        ax.plot(
            best_degree,
            sel_cand.mean_val_metrics.rmse,
            marker="*",
            markersize=14,
            color="red",
            label=f"Selected (deg={best_degree}, L2={best_l2})",
        )

        ax.set_title("09 K-Fold Cross-Validation Mean RMSE ± 1 Std")
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("Mean Validation RMSE")
        ax.set_xticks(degrees)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize="small")

        return self._save_and_close(
            fig,
            "09_kfold_mean_rmse",
            "Cross-Validation Mean RMSE",
            "Errorbar Plot",
            "KFold",
            {"selected_degree": best_degree, "selected_l2": best_l2},
            "kfold_summary_results.csv",
        )

    # 10. K-Fold RMSE Heatmap
    def plot_10_kfold_rmse_heatmap(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        return self._plot_rmse_heatmap(
            candidates,
            best_degree,
            best_l2,
            title="10 K-Fold Validation Mean RMSE Heatmap",
            filename="10_kfold_rmse_heatmap",
            workflow="KFold",
            rmse_extractor=lambda c: c.mean_val_metrics.rmse,
        )

    # 10b. K-Fold RMSE Std Heatmap
    def plot_10b_kfold_rmse_std_heatmap(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        return self._plot_rmse_heatmap(
            candidates,
            best_degree,
            best_l2,
            title="10b K-Fold Validation RMSE Standard Deviation Heatmap",
            filename="10b_kfold_rmse_std_heatmap",
            workflow="KFold",
            rmse_extractor=lambda c: c.std_val_metrics.rmse,
        )

    # 11. Fold-by-Fold Metrics Plot
    def plot_11_fold_metrics(
        self,
        fold_results: list[Any],
        mean_rmse: float,
        std_rmse: float,
        degree: int,
        l2: float,
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        folds = [fr.fold_index + 1 for fr in fold_results]
        rmses = [fr.val_metrics.rmse for fr in fold_results]

        ax.bar(
            folds,
            rmses,
            color=self.colors["val"],
            width=0.4,
            edgecolor="black",
            alpha=0.85,
            label="Fold Validation RMSE",
        )
        ax.axhline(
            mean_rmse,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean RMSE ({mean_rmse:.3f})",
        )
        ax.axhspan(
            mean_rmse - std_rmse,
            mean_rmse + std_rmse,
            color="red",
            alpha=0.15,
            label="±1 Std Dev Band",
        )

        ax.set_xticks(folds)
        ax.set_xlabel("Fold Number")
        ax.set_ylabel("Validation RMSE")
        ax.set_title(f"11 Fold-by-Fold Performance (Degree={degree}, L2={l2})")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        return self._save_and_close(
            fig,
            "11_fold_metrics",
            "Fold-by-Fold Metrics",
            "Bar Chart",
            "KFold",
            {"degree": degree, "l2_lambda": l2, "mean_rmse": mean_rmse},
            "kfold_fold_results.csv",
        )

    # 12. Out-of-Fold Predictions Plot
    def plot_12_out_of_fold_predictions(
        self, y_dev: np.ndarray, oof_preds: np.ndarray, metrics: EvaluationMetrics
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(
            y_dev,
            oof_preds,
            color=self.colors["dev"],
            alpha=0.7,
            edgecolors="k",
            s=30,
            label="Out-of-Fold Predictions",
        )

        # 45-degree reference line
        min_val = min(np.min(y_dev), np.min(oof_preds))
        max_val = max(np.max(y_dev), np.max(oof_preds))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=1.5,
            label="Ideal 1:1 Line",
        )

        text_str = f"OOF RMSE: {metrics.rmse:.3f}\nOOF R²: {metrics.r_squared:.3f}"
        ax.text(
            0.05,
            0.90,
            text_str,
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_title("12 Out-of-Fold Actual vs. Predicted Y")
        ax.set_xlabel("Actual Y (Development Set)")
        ax.set_ylabel("Predicted Y (Out-of-Fold)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        return self._save_and_close(
            fig,
            "12_out_of_fold_predictions",
            "Out-of-Fold Predictions",
            "Scatter Plot",
            "KFold",
            {"oof_rmse": metrics.rmse, "oof_r2": metrics.r_squared},
            "out_of_fold_predictions.csv",
        )

    # 13. Final Fitted Polynomial Plot
    def plot_13_final_polynomial_fit(
        self,
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        degree: int,
        l2: float,
        test_metrics: EvaluationMetrics,
        workflow: str,
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(
            x_dev, y_dev, color=self.colors["dev"], alpha=0.6, s=25, label="Dev Data"
        )
        ax.scatter(
            x_test,
            y_test,
            color=self.colors["test"],
            marker="^",
            s=35,
            label="Untouched Test Data",
        )

        ax.plot(
            x_grid,
            y_grid,
            color=self.colors["fit"],
            linewidth=2.5,
            label=f"Final Polynomial Fit (deg={degree}, L2={l2})",
        )

        ax.set_title(
            f"13 Final Polynomial Fit on Development Data\nTest RMSE: {test_metrics.rmse:.3f} | Test R²: {test_metrics.r_squared:.3f}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        return self._save_and_close(
            fig,
            "13_final_polynomial_fit",
            "Final Polynomial Fit",
            "Line Plot",
            workflow,
            {
                "degree": degree,
                "l2": l2,
                "test_rmse": test_metrics.rmse,
                "test_r2": test_metrics.r_squared,
            },
            "final_model.json",
        )

    # 13b. Bootstrap Prediction Band (Optional)
    def plot_13b_final_polynomial_bootstrap_band(
        self,
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        lower_band: np.ndarray,
        upper_band: np.ndarray,
        degree: int,
        l2: float,
        num_bootstraps: int,
        workflow: str,
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(
            x_dev, y_dev, color=self.colors["dev"], alpha=0.5, s=20, label="Dev Data"
        )
        ax.scatter(
            x_test,
            y_test,
            color=self.colors["test"],
            marker="^",
            s=30,
            label="Test Data",
        )

        ax.plot(
            x_grid,
            y_grid,
            color=self.colors["fit"],
            linewidth=2,
            label="Final Fitted Curve",
        )
        ax.fill_between(
            x_grid,
            lower_band,
            upper_band,
            color=self.colors["bootstrap"],
            alpha=0.3,
            label=f"Bootstrap Model Fit Uncertainty Band ({num_bootstraps} resamples)",
        )

        ax.set_title(
            f"13b Final Polynomial Fit with Bootstrap Uncertainty (deg={degree}, L2={l2})"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        return self._save_and_close(
            fig,
            "13b_final_polynomial_bootstrap_band",
            "Bootstrap Uncertainty Band",
            "Line Band Plot",
            workflow,
            {"degree": degree, "l2": l2, "bootstrap_samples": num_bootstraps},
            "final_model.json",
        )

    # 14. Actual vs. Predicted Test Plot
    def plot_14_test_actual_vs_predicted(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        metrics: EvaluationMetrics,
        workflow: str,
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(
            y_test,
            y_pred,
            color=self.colors["test"],
            alpha=0.8,
            edgecolors="k",
            s=35,
            label="Test Predictions",
        )

        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=1.5,
            label="Ideal 1:1 Line",
        )

        text_str = f"Test RMSE: {metrics.rmse:.3f}\nTest MAE: {metrics.mae:.3f}\nTest R²: {metrics.r_squared:.3f}"
        ax.text(
            0.05,
            0.88,
            text_str,
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_title("14 Test Set Actual vs. Predicted Y")
        ax.set_xlabel("Actual Test Y")
        ax.set_ylabel("Predicted Test Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        return self._save_and_close(
            fig,
            "14_test_actual_vs_predicted",
            "Test Actual vs Predicted",
            "Scatter Plot",
            workflow,
            {
                "test_rmse": metrics.rmse,
                "test_mae": metrics.mae,
                "test_r2": metrics.r_squared,
            },
            "test_predictions.csv",
        )

    # 15. Residual vs Predicted Plot
    def plot_15_test_residuals(
        self, y_pred: np.ndarray, residuals: np.ndarray, workflow: str
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            y_pred,
            residuals,
            color=self.colors["test"],
            alpha=0.8,
            edgecolors="k",
            s=35,
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))
        ax.text(
            0.05,
            0.90,
            f"Residual Mean: {res_mean:.3f}\nResidual Std: {res_std:.3f}",
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_title("15 Residuals vs. Predicted Values (Diagnostic)")
        ax.set_xlabel("Predicted Test Y")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.grid(True, linestyle="--", alpha=0.5)

        return self._save_and_close(
            fig,
            "15_test_residuals",
            "Test Residuals vs Predicted",
            "Scatter Plot",
            workflow,
            {"residual_mean": res_mean, "residual_std": res_std},
            "test_predictions.csv",
        )

    # 16. Residual Histogram
    def plot_16_test_residual_histogram(
        self, residuals: np.ndarray, bins: int | None, workflow: str
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        counts, _bin_edges, _ = ax.hist(
            residuals,
            bins=bins or "auto",
            color=self.colors["test"],
            edgecolor="black",
            alpha=0.7,
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Residual")

        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))
        ax.text(
            0.05,
            0.88,
            f"Residual Mean: {res_mean:.3f}\nResidual Std: {res_std:.3f}",
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_title("16 Final Test Residual Distribution")
        ax.set_xlabel("Residual Value (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        return self._save_and_close(
            fig,
            "16_test_residual_histogram",
            "Test Residual Histogram",
            "Histogram",
            workflow,
            {"bins": len(counts), "residual_mean": res_mean, "residual_std": res_std},
            "test_predictions.csv",
        )

    # 17. Metrics Summary Plot
    def plot_17_final_metrics(
        self, metrics: EvaluationMetrics, workflow: str
    ) -> pathlib.Path:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [3, 1]}
        )

        names_err = ["MSE", "RMSE", "MAE"]
        vals_err = [metrics.mse, metrics.rmse, metrics.mae]

        bars = ax1.bar(
            names_err,
            vals_err,
            color=[self.colors["train"], self.colors["val"], self.colors["fit"]],
            edgecolor="black",
            width=0.5,
        )
        for bar in bars:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(vals_err),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
            )

        ax1.set_ylabel("Error Metric Value")
        ax1.set_title("Error Metrics")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # R2 Subplot
        bar2 = ax2.bar(
            ["R²"],
            [metrics.r_squared],
            color=self.colors["test"],
            edgecolor="black",
            width=0.4,
        )
        ax2.text(
            bar2[0].get_x() + bar2[0].get_width() / 2,
            metrics.r_squared + 0.02,
            f"{metrics.r_squared:.3f}",
            ha="center",
            va="bottom",
        )
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_title("Goodness of Fit")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        fig.suptitle("17 Final Test Performance Metrics Summary", fontsize=12)

        return self._save_and_close(
            fig,
            "17_final_metrics",
            "Final Metrics Summary",
            "Bar Chart",
            workflow,
            {
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r_squared,
            },
            "final_model.json",
        )

    # 18. Model Coefficients Plot
    def plot_18_model_coefficients(
        self, beta: np.ndarray, is_original_basis: bool, workflow: str
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        degrees = np.arange(len(beta))
        labels = ["Intercept"] + [f"x^{d}" if d > 1 else "x" for d in degrees[1:]]

        bars = ax.bar(
            labels, beta, color=self.colors["fit"], edgecolor="black", width=0.5
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=1.0)

        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2e}",
                ha="center",
                va=va,
                fontsize=8,
            )

        basis_str = (
            "Original Basis (Unscaled)" if is_original_basis else "Scaled Feature Basis"
        )
        ax.set_title(f"18 Fitted Polynomial Coefficients ({basis_str})")
        ax.set_xlabel("Polynomial Term")
        ax.set_ylabel("Coefficient Value")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        return self._save_and_close(
            fig,
            "18_model_coefficients",
            "Model Coefficients",
            "Bar Chart",
            workflow,
            {"degree": len(beta) - 1, "is_original_basis": is_original_basis},
            "final_model.json",
        )

    # 19. Condition Numbers Plot
    def plot_19_condition_numbers(
        self,
        degrees: list[int],
        cond_numbers: list[float],
        threshold: float,
        workflow: str,
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            degrees,
            cond_numbers,
            "o-",
            color=self.colors["train"],
            linewidth=2,
            label="Design Matrix Condition Number",
        )
        ax.axhline(
            threshold,
            color="red",
            linestyle="--",
            label=f"Warning Threshold ({threshold:.0e})",
        )

        ax.set_yscale("log")
        ax.set_xticks(degrees)
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("Condition Number (Log Scale)")
        ax.set_title("19 Design Matrix Condition Number vs. Degree")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        return self._save_and_close(
            fig,
            "19_condition_numbers",
            "Condition Numbers",
            "Line Plot (Log)",
            workflow,
            {"threshold": threshold, "max_cond": max(cond_numbers)},
            "holdout_results.csv",
        )

    # 20. Combined Results Dashboard
    def plot_20_results_dashboard(
        self,
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        y_test_pred: np.ndarray,
        test_residuals: np.ndarray,
        candidates: list[Any],
        best_degree: int,
        best_l2: float,
        test_metrics: EvaluationMetrics,
        workflow: str,
    ) -> pathlib.Path:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Final Polynomial Fit
        ax1.scatter(
            x_dev, y_dev, color=self.colors["dev"], alpha=0.5, s=15, label="Dev"
        )
        ax1.scatter(
            x_test, y_test, color=self.colors["test"], marker="^", s=25, label="Test"
        )
        ax1.plot(
            x_grid,
            y_grid,
            color=self.colors["fit"],
            linewidth=2,
            label="Polynomial Fit",
        )
        ax1.set_title(f"1. Final Fit (deg={best_degree}, L2={best_l2})")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="best", fontsize="small")

        # Panel 2: Actual vs Predicted
        ax2.scatter(
            y_test,
            y_test_pred,
            color=self.colors["test"],
            alpha=0.8,
            edgecolors="k",
            s=25,
        )
        min_v = min(np.min(y_test), np.min(y_test_pred))
        max_v = max(np.max(y_test), np.max(y_test_pred))
        ax2.plot([min_v, max_v], [min_v, max_v], "r--")
        ax2.set_title(f"2. Actual vs. Predicted (RMSE={test_metrics.rmse:.3f})")
        ax2.set_xlabel("Actual Test Y")
        ax2.set_ylabel("Predicted Test Y")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Panel 3: Residuals vs Predictions
        ax3.scatter(
            y_test_pred,
            test_residuals,
            color=self.colors["test"],
            alpha=0.8,
            edgecolors="k",
            s=25,
        )
        ax3.axhline(0, color="red", linestyle="--")
        ax3.set_title("3. Residuals vs. Predictions")
        ax3.set_xlabel("Predicted Test Y")
        ax3.set_ylabel("Residual (Actual - Predicted)")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Panel 4: Model Selection Curve
        cands_l2 = [c for c in candidates if c.l2_lambda == best_l2]
        cands_l2.sort(key=lambda c: c.degree)
        degs = [c.degree for c in cands_l2]

        if hasattr(cands_l2[0], "val_metrics"):
            rmses = [c.val_metrics.rmse for c in cands_l2]
        else:
            rmses = [c.mean_val_metrics.rmse for c in cands_l2]

        ax4.plot(degs, rmses, "o-", color=self.colors["train"])
        ax4.axvline(best_degree, color="red", linestyle=":")
        ax4.set_title(f"4. Selection Curve (L2={best_l2})")
        ax4.set_xlabel("Polynomial Degree")
        ax4.set_ylabel("Validation RMSE")
        ax4.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(
            f"20 Polynomial Regression Dashboard ({workflow.capitalize()} Mode)",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        return self._save_and_close(
            fig,
            "20_results_dashboard",
            "Combined Results Dashboard",
            "Dashboard (4-panel)",
            workflow,
            {"degree": best_degree, "l2": best_l2, "test_rmse": test_metrics.rmse},
            "final_model.json",
        )

    # Helper for Heatmaps
    def _plot_rmse_heatmap(
        self,
        candidates: list[Any],
        best_degree: int,
        best_l2: float,
        title: str,
        filename: str,
        workflow: str,
        rmse_extractor: Any,
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(8, 6))

        degrees = sorted({c.degree for c in candidates})
        l2_values = sorted({c.l2_lambda for c in candidates})

        heatmap_data = np.zeros((len(l2_values), len(degrees)), dtype=np.float64)

        for i, l2 in enumerate(l2_values):
            for j, deg in enumerate(degrees):
                cand = next(c for c in candidates if c.degree == deg and c.l2_lambda == l2)
                heatmap_data[i, j] = rmse_extractor(cand)

        im = ax.imshow(heatmap_data, cmap="viridis_r", aspect="auto")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Validation RMSE")

        # Annotate text
        if len(degrees) * len(l2_values) <= 100:
            for i in range(len(l2_values)):
                for j in range(len(degrees)):
                    val = heatmap_data[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="w" if val > np.mean(heatmap_data) else "k",
                        fontsize=8,
                    )

        # Labels
        ax.set_xticks(np.arange(len(degrees)))
        ax.set_yticks(np.arange(len(l2_values)))

        ax.set_xticklabels(degrees)
        ax.set_yticklabels([f"{l2}" if l2 > 0 else "0 (OLS)" for l2 in l2_values])

        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("L2 Regularization Strength")
        ax.set_title(title)

        # Highlight best cell
        best_i = l2_values.index(best_l2)
        best_j = degrees.index(best_degree)
        ax.add_patch(
            plt.Rectangle(
                (best_j - 0.45, best_i - 0.45),
                0.9,
                0.9,
                fill=False,
                edgecolor="red",
                linewidth=2.5,
            )
        )

        return self._save_and_close(
            fig,
            filename,
            title,
            "Heatmap",
            workflow,
            {"selected_degree": best_degree, "selected_l2": best_l2},
            "holdout_results.csv"
            if workflow == "Holdout"
            else "kfold_summary_results.csv",
        )
