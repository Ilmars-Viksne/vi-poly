"""Scientific visualization module for polynomial regression using Matplotlib."""

import pathlib
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .metrics import EvaluationMetrics

WORKFLOW_DISPLAY_NAMES = {
    "common": "Common",
    "holdout": "Holdout",
    "kfold": "K-Fold",
}


def _workflow_filename(number: str, workflow: str, description: str) -> str:
    wf_norm = workflow.lower().strip()
    if wf_norm not in WORKFLOW_DISPLAY_NAMES:
        raise ValueError(f"Unsupported workflow name: {workflow}")
    if wf_norm == "common":
        return f"{number}_{description}"
    return f"{number}_{wf_norm}_{description}"


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

    @staticmethod
    def _r_squared_axis_limits(value: float) -> tuple[float, float]:
        if not np.isfinite(value):
            raise ValueError(f"R-squared must be finite; received {value}.")

        lower_reference = min(0.0, value)
        upper_reference = max(1.0, value)

        span = upper_reference - lower_reference
        margin = max(0.1, 0.1 * span)

        return lower_reference - margin, upper_reference + margin

    @staticmethod
    def build_kfold_membership_matrix(
        development_sample_count: int,
        fold_splits: list[Any],
    ) -> np.ndarray:
        if (
            not isinstance(development_sample_count, (int, np.integer))
            or development_sample_count <= 0
        ):
            raise ValueError(
                f"development_sample_count must be a positive integer; received {development_sample_count}."
            )
        if not isinstance(fold_splits, list) or len(fold_splits) < 2:
            raise ValueError(
                f"fold_splits must contain at least 2 folds; received {len(fold_splits) if isinstance(fold_splits, list) else fold_splits}."
            )

        num_folds = len(fold_splits)
        VALIDATION = 1

        membership = np.zeros((num_folds, development_sample_count), dtype=np.uint8)
        seen_fold_indices = set()

        for fold in fold_splits:
            f_idx = getattr(fold, "fold_index", None)
            if f_idx is None:
                raise ValueError("FoldSplit object missing fold_index attribute.")
            if (
                not isinstance(f_idx, (int, np.integer))
                or f_idx < 0
                or f_idx >= num_folds
            ):
                raise ValueError(f"Invalid fold_index: {f_idx}.")
            if f_idx in seen_fold_indices:
                raise ValueError(f"Duplicate fold_index encountered: {f_idx}.")
            seen_fold_indices.add(f_idx)

            val_indices = getattr(fold, "val_indices", None)
            train_indices = getattr(fold, "train_indices", None)
            if val_indices is None or train_indices is None:
                raise ValueError(
                    "FoldSplit object missing val_indices or train_indices."
                )

            val_arr = np.asarray(val_indices, dtype=int)
            train_arr = np.asarray(train_indices, dtype=int)

            if val_arr.ndim != 1 or train_arr.ndim != 1:
                raise ValueError("Fold indices must be 1D arrays.")

            if len(val_arr) == 0 or len(train_arr) == 0:
                raise ValueError(
                    "Train and validation sets within each fold must not be empty."
                )

            if np.any(val_arr < 0) or np.any(val_arr >= development_sample_count):
                raise ValueError(
                    f"Validation indices out of bounds for development sample count {development_sample_count}."
                )
            if np.any(train_arr < 0) or np.any(train_arr >= development_sample_count):
                raise ValueError(
                    f"Train indices out of bounds for development sample count {development_sample_count}."
                )

            if len(np.unique(val_arr)) != len(val_arr):
                raise ValueError(f"Fold {f_idx} validation indices contain duplicates.")
            if len(np.unique(train_arr)) != len(train_arr):
                raise ValueError(f"Fold {f_idx} train indices contain duplicates.")

            if len(np.intersect1d(val_arr, train_arr)) > 0:
                raise ValueError(f"Fold {f_idx} train and validation indices overlap.")

            if (
                len(val_arr) + len(train_arr) != development_sample_count
                or len(np.union1d(val_arr, train_arr)) != development_sample_count
            ):
                raise ValueError(
                    f"Fold {f_idx} train and validation indices do not cover all development samples."
                )

            membership[f_idx, val_arr] = VALIDATION

        col_val_sums = membership.sum(axis=0)
        if not np.all(col_val_sums == 1):
            raise ValueError(
                "Fold assignment invariant violated: each sample must belong to validation in exactly 1 fold."
            )

        return membership

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
        wf_norm = workflow.lower().strip()
        if wf_norm not in WORKFLOW_DISPLAY_NAMES:
            raise ValueError(f"Unsupported workflow: {workflow}")

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
                "workflow": wf_norm,
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
        title = f"01 Original Dataset (N = {len(x)})"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        filename_stem = _workflow_filename("01", "common", "original_data")
        return self._save_and_close(
            fig,
            filename_stem,
            "Original Dataset",
            "Scatter",
            "common",
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

        title = "02 Train / Validation / Test Data Splits"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        filename_stem = _workflow_filename("02", "common", "data_splits")
        return self._save_and_close(
            fig,
            filename_stem,
            "Data Splits",
            "Scatter",
            "common",
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
        title = "03 Subset Sample Allocation"
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        filename_stem = _workflow_filename("03", "common", "split_sizes")
        return self._save_and_close(
            fig,
            filename_stem,
            "Split Sizes",
            "Bar Chart",
            "common",
            {"total_samples": total},
            "split_summary.json",
        )

    # 4. Holdout Validation RMSE Curves
    def plot_04_holdout_validation_rmse(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        wf_norm = "holdout"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"04 {wf_disp}: Validation RMSE vs. Polynomial Degree"
        ax.set_title(title)
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("Validation RMSE")
        ax.set_xticks(degrees)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize="small")

        filename_stem = _workflow_filename("04", wf_norm, "validation_rmse")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Line Plot",
            wf_norm,
            {"selected_degree": best_degree, "selected_l2": best_l2},
            "holdout_results.csv",
        )

    # 5. Holdout Train vs Validation Error
    def plot_05_train_validation_error(
        self, candidates: list[Any], selected_l2: float, best_degree: int
    ) -> pathlib.Path:
        wf_norm = "holdout"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"05 {wf_disp}: Bias-Variance Trade-off (L2 = {selected_l2})"
        ax.set_title(title)
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("RMSE")
        ax.set_xticks(degs)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        filename_stem = _workflow_filename("05", wf_norm, "train_validation_error")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Line Plot",
            wf_norm,
            {"l2_lambda": selected_l2, "selected_degree": best_degree},
            "holdout_results.csv",
        )

    # 6. Holdout RMSE Heatmap
    def plot_06_holdout_rmse_heatmap(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        wf_norm = "holdout"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]
        title = f"06 {wf_disp}: Validation RMSE Heatmap"
        filename_stem = _workflow_filename("06", wf_norm, "rmse_heatmap")
        return self._plot_rmse_heatmap(
            candidates,
            best_degree,
            best_l2,
            title=title,
            filename=filename_stem,
            workflow=wf_norm,
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
        wf_norm = "holdout"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"07 {wf_disp}: Candidate Polynomial Models Comparison"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        filename_stem = _workflow_filename("07", wf_norm, "candidate_models")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Line Plot",
            wf_norm,
            {"candidate_degrees": [c[0] for c in candidate_curves]},
            "holdout_results.csv",
        )

    # 8. K-Fold Assignments Plot
    def plot_08_kfold_assignments(
        self, dev_indices: np.ndarray, fold_splits: list[Any]
    ) -> pathlib.Path:
        wf_norm = "kfold"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

        dev_count = len(dev_indices)
        membership = self.build_kfold_membership_matrix(dev_count, fold_splits)
        k = len(fold_splits)

        fig, ax = plt.subplots(figsize=(9, 5))

        TRAINING_COLOR = "#BBD7E8"
        VALIDATION_COLOR = "#E69F00"

        cmap = ListedColormap([TRAINING_COLOR, VALIDATION_COLOR])

        ax.imshow(
            membership,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            origin="upper",
        )

        ax.set_yticks(range(k))
        ax.set_yticklabels([f"Fold {i + 1}" for i in range(k)])

        max_ticks = 10
        tick_positions = np.linspace(
            0, dev_count - 1, min(max_ticks, dev_count), dtype=int
        )
        tick_positions = np.unique(tick_positions)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(p) for p in tick_positions])

        ax.set_xlabel("Development Sample Position")
        ax.set_ylabel("Cross-Validation Fold")

        title = f"08 {wf_disp}: Training and Validation Membership Matrix (K = {k})"
        ax.set_title(title)

        legend_handles = [
            Patch(facecolor=TRAINING_COLOR, edgecolor="none", label="Training"),
            Patch(facecolor=VALIDATION_COLOR, edgecolor="none", label="Validation"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
        )

        filename_stem = _workflow_filename("08", wf_norm, "assignments")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Binary Membership Matrix",
            wf_norm,
            {
                "num_folds": k,
                "development_sample_count": dev_count,
                "training_code": 0,
                "validation_code": 1,
                "validation_assignments_per_sample": 1,
            },
            "split_summary.json",
        )

    # 9. Cross-Validation Mean RMSE Plot with Error Bars
    def plot_09_kfold_mean_rmse(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        wf_norm = "kfold"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"09 {wf_disp}: Cross-Validation Mean RMSE ± 1 Std"
        ax.set_title(title)
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("Mean Validation RMSE")
        ax.set_xticks(degrees)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize="small")

        filename_stem = _workflow_filename("09", wf_norm, "mean_rmse")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Errorbar Plot",
            wf_norm,
            {"selected_degree": best_degree, "selected_l2": best_l2},
            "kfold_summary_results.csv",
        )

    # 10. K-Fold RMSE Heatmap
    def plot_10_kfold_rmse_heatmap(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        wf_norm = "kfold"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]
        title = f"10 {wf_disp}: Validation Mean RMSE Heatmap"
        filename_stem = _workflow_filename("10", wf_norm, "rmse_heatmap")
        return self._plot_rmse_heatmap(
            candidates,
            best_degree,
            best_l2,
            title=title,
            filename=filename_stem,
            workflow=wf_norm,
            rmse_extractor=lambda c: c.mean_val_metrics.rmse,
        )

    # 10b. K-Fold RMSE Std Heatmap
    def plot_10b_kfold_rmse_std_heatmap(
        self, candidates: list[Any], best_degree: int, best_l2: float
    ) -> pathlib.Path:
        wf_norm = "kfold"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]
        title = f"10b {wf_disp}: Validation RMSE Standard Deviation Heatmap"
        filename_stem = _workflow_filename("10b", wf_norm, "rmse_std_heatmap")
        return self._plot_rmse_heatmap(
            candidates,
            best_degree,
            best_l2,
            title=title,
            filename=filename_stem,
            workflow=wf_norm,
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
        wf_norm = "kfold"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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
        title = f"11 {wf_disp}: Fold-by-Fold Performance (Degree={degree}, L2={l2})"
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        filename_stem = _workflow_filename("11", wf_norm, "fold_metrics")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Bar Chart",
            wf_norm,
            {"degree": degree, "l2_lambda": l2, "mean_rmse": mean_rmse},
            "kfold_fold_results.csv",
        )

    # 12. Out-of-Fold Predictions Plot
    def plot_12_out_of_fold_predictions(
        self, y_dev: np.ndarray, oof_preds: np.ndarray, metrics: EvaluationMetrics
    ) -> pathlib.Path:
        wf_norm = "kfold"
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"12 {wf_disp}: Out-of-Fold Actual vs. Predicted Y"
        ax.set_title(title)
        ax.set_xlabel("Actual Y (Development Set)")
        ax.set_ylabel("Predicted Y (Out-of-Fold)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        filename_stem = _workflow_filename("12", wf_norm, "out_of_fold_predictions")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Scatter Plot",
            wf_norm,
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
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"13 {wf_disp}: Final Polynomial Fit on Development Data\nTest RMSE: {test_metrics.rmse:.3f} | Test R²: {test_metrics.r_squared:.3f}"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        filename_stem = _workflow_filename("13", wf_norm, "final_polynomial_fit")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Line Plot",
            wf_norm,
            {
                "degree": degree,
                "l2": l2,
                "test_rmse": test_metrics.rmse,
                "test_r2": test_metrics.r_squared,
            },
            "final_model.json",
        )

    # 13b. Bootstrap Fitted-Curve Uncertainty Band
    def plot_13b_fitted_curve_uncertainty_band(
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
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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
            label=f"95% bootstrap fitted-curve uncertainty band ({num_bootstraps} resamples)",
        )

        title = f"13b {wf_disp}: Bootstrap Fitted-Curve Uncertainty Band (deg={degree}, L2={l2})"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        filename_stem = _workflow_filename(
            "13b", wf_norm, "fitted_curve_uncertainty_band"
        )
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Line Band Plot",
            wf_norm,
            {
                "degree": degree,
                "l2": l2,
                "bootstrap_samples": num_bootstraps,
                "lower_percentile": 2.5,
                "upper_percentile": 97.5,
                "interval_interpretation": "fitted_curve_uncertainty",
            },
            "final_model.json",
        )

    def plot_13b_final_polynomial_bootstrap_band(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> pathlib.Path:
        warnings.warn(
            "plot_13b_final_polynomial_bootstrap_band() is deprecated; "
            "use plot_13b_fitted_curve_uncertainty_band().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plot_13b_fitted_curve_uncertainty_band(*args, **kwargs)

    # 14. Actual vs. Predicted Test Plot
    def plot_14_test_actual_vs_predicted(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        metrics: EvaluationMetrics,
        workflow: str,
    ) -> pathlib.Path:
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"14 {wf_disp}: Test Set Actual vs. Predicted Y"
        ax.set_title(title)
        ax.set_xlabel("Actual Test Y")
        ax.set_ylabel("Predicted Test Y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        filename_stem = _workflow_filename("14", wf_norm, "test_actual_vs_predicted")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Scatter Plot",
            wf_norm,
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
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"15 {wf_disp}: Residuals vs. Predicted Values (Diagnostic)"
        ax.set_title(title)
        ax.set_xlabel("Predicted Test Y")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.grid(True, linestyle="--", alpha=0.5)

        filename_stem = _workflow_filename("15", wf_norm, "test_residuals")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Scatter Plot",
            wf_norm,
            {"residual_mean": res_mean, "residual_std": res_std},
            "test_predictions.csv",
        )

    # 16. Residual Histogram
    def plot_16_test_residual_histogram(
        self, residuals: np.ndarray, bins: int | None, workflow: str
    ) -> pathlib.Path:
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"16 {wf_disp}: Final Test Residual Distribution"
        ax.set_title(title)
        ax.set_xlabel("Residual Value (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        filename_stem = _workflow_filename("16", wf_norm, "test_residual_histogram")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Histogram",
            wf_norm,
            {"bins": len(counts), "residual_mean": res_mean, "residual_std": res_std},
            "test_predictions.csv",
        )

    # 17. Metrics Summary Plot
    def plot_17_final_metrics(
        self, metrics: EvaluationMetrics, workflow: str
    ) -> pathlib.Path:
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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
        y_min, y_max = self._r_squared_axis_limits(metrics.r_squared)
        ax2.set_ylim(y_min, y_max)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)

        offset = 0.02 * (y_max - y_min)
        if metrics.r_squared < 0:
            label_y = metrics.r_squared - offset
            va = "top"
        else:
            label_y = metrics.r_squared + offset
            va = "bottom"

        ax2.text(
            bar2[0].get_x() + bar2[0].get_width() / 2,
            label_y,
            f"{metrics.r_squared:.3f}",
            ha="center",
            va=va,
        )
        ax2.set_title("Goodness of Fit")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        title = f"17 {wf_disp}: Final Test Performance Metrics Summary"
        fig.suptitle(title, fontsize=12)

        filename_stem = _workflow_filename("17", wf_norm, "final_metrics")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Bar Chart",
            wf_norm,
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
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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
        title = f"18 {wf_disp}: Fitted Polynomial Coefficients ({basis_str})"
        ax.set_title(title)
        ax.set_xlabel("Polynomial Term")
        ax.set_ylabel("Coefficient Value")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        filename_stem = _workflow_filename("18", wf_norm, "model_coefficients")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Bar Chart",
            wf_norm,
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
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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
        title = f"19 {wf_disp}: Design Matrix Condition Number vs. Degree"
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        filename_stem = _workflow_filename("19", wf_norm, "condition_numbers")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Line Plot (Log)",
            wf_norm,
            {"threshold": threshold, "max_cond": max(cond_numbers)},
            "holdout_results.csv"
            if wf_norm == "holdout"
            else "kfold_summary_results.csv",
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
        wf_norm = workflow.lower().strip()
        wf_disp = WORKFLOW_DISPLAY_NAMES[wf_norm]

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

        title = f"20 {wf_disp}: Polynomial Regression Dashboard"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        filename_stem = _workflow_filename("20", wf_norm, "results_dashboard")
        return self._save_and_close(
            fig,
            filename_stem,
            title,
            "Dashboard (4-panel)",
            wf_norm,
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
        wf_norm = workflow.lower().strip()

        fig, ax = plt.subplots(figsize=(8, 6))

        degrees = sorted({c.degree for c in candidates})
        l2_values = sorted({c.l2_lambda for c in candidates})

        heatmap_data = np.zeros((len(l2_values), len(degrees)), dtype=np.float64)

        for i, l2 in enumerate(l2_values):
            for j, deg in enumerate(degrees):
                cand = next(
                    c for c in candidates if c.degree == deg and c.l2_lambda == l2
                )
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
            wf_norm,
            {"selected_degree": best_degree, "selected_l2": best_l2},
            "holdout_results.csv"
            if wf_norm == "holdout"
            else "kfold_summary_results.csv",
        )
