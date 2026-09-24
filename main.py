"""Main CLI script for Polynomial Regression Workflows."""

import argparse
import pathlib
import sys
from typing import Any

import numpy as np

from polynomial_regression.data import CSVDataLoader
from polynomial_regression.features import PolynomialFeatureTransformer
from polynomial_regression.metrics import RegressionMetrics
from polynomial_regression.regression import PolynomialRegressor
from polynomial_regression.reporting import ReportGenerator
from polynomial_regression.selection import HoldoutModelSelector, KFoldModelSelector
from polynomial_regression.splitting import DataSplitter, KFoldSplitter
from polynomial_regression.visualization import RegressionVisualizer


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polynomial Regression pipeline using NumPy and Matplotlib."
    )
    # Data input options
    parser.add_argument("csv_file", type=str, help="Path to input CSV file.")
    parser.add_argument(
        "--x-column", type=str, default="X", help="Name of X column in CSV."
    )
    parser.add_argument(
        "--y-column", type=str, default="Y", help="Name of Y column in CSV."
    )
    parser.add_argument(
        "--skip-invalid-rows",
        action="store_true",
        help="Skip rows with invalid/missing values instead of raising an error.",
    )

    # Split options
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Testing set ratio (default: 0.15).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible splitting."
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of folds for K-fold CV."
    )

    # Hyperparameter search space
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Candidate polynomial degrees.",
    )
    parser.add_argument(
        "--l2-values",
        type=float,
        nargs="+",
        default=[0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0],
        help="Candidate L2 regularization strengths.",
    )
    parser.add_argument(
        "--scale-features",
        action="store_true",
        help="Opt-in to Z-score feature scaling for polynomial terms.",
    )

    # Workflow mode
    parser.add_argument(
        "--mode",
        choices=["holdout", "kfold", "both"],
        default="both",
        help="Execution mode: holdout, kfold, or both.",
    )

    # Visualizations
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save numerical and plot outputs.",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Disable all plot generation."
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Plot format.",
    )
    parser.add_argument(
        "--plot-dpi", type=int, default=150, help="DPI resolution for raster plots."
    )
    parser.add_argument(
        "--plot-style", type=str, default=None, help="Matplotlib plot style."
    )
    parser.add_argument(
        "--curve-points",
        type=int,
        default=500,
        help="Number of points in dense X grid for plotting.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Number of bootstrap samples for uncertainty band (0=disabled).",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=123, help="Seed for bootstrap sampling."
    )
    parser.add_argument(
        "--residual-bins",
        type=int,
        default=None,
        help="Bin count for residual histogram.",
    )

    # Numerical robustness & tie-breaking
    parser.add_argument(
        "--condition-warning-threshold",
        type=float,
        default=1e12,
        help="Condition number warning threshold.",
    )
    parser.add_argument(
        "--selection-rtol",
        type=float,
        default=1e-7,
        help="Relative tolerance for model selection tie-breaking.",
    )
    parser.add_argument(
        "--selection-atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for model selection tie-breaking.",
    )

    return parser.parse_args(args)


def run_pipeline(args: argparse.Namespace) -> None:
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    loader = CSVDataLoader(
        filepath=args.csv_file,
        x_column=args.x_column,
        y_column=args.y_column,
        skip_invalid_rows=args.skip_invalid_rows,
    )
    loaded_data = loader.load()
    x_all, y_all = loaded_data.x, loaded_data.y
    N = len(x_all)

    # 2. Perform Partitioning
    splitter = DataSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split = splitter.split(N)

    kfold_assignments = {}
    if args.mode in ["kfold", "both"]:
        k_splitter = KFoldSplitter(k=args.folds, seed=args.seed)
        fold_splits = k_splitter.split(np.arange(len(split.dev_indices)))
        for fold in fold_splits:
            kfold_assignments[f"fold_{fold.fold_index + 1}"] = split.dev_indices[
                fold.val_indices
            ].tolist()

    # 3. Initialize Reporter and Visualizer
    reporter = ReportGenerator(output_path)
    reporter.write_split_summary(
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        total_samples=N,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
        dev_indices=split.dev_indices,
        fold_assignments=kfold_assignments,
        skipped_rows=loaded_data.skipped_rows,
    )

    visualizer = None
    if not args.no_plots:
        visualizer = RegressionVisualizer(
            output_dir=output_path,
            plot_format=args.plot_format,
            plot_dpi=args.plot_dpi,
            plot_style=args.plot_style,
            show_plots=args.show_plots,
        )
        visualizer.plot_01_original_data(x_all, y_all)
        visualizer.plot_02_data_splits(
            x_all, y_all, split.train_indices, split.val_indices, split.test_indices
        )
        visualizer.plot_03_split_sizes(
            len(split.train_indices), len(split.val_indices), len(split.test_indices)
        )

    # Dense X Grid for smooth curve plotting across observed X range
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    x_grid = np.linspace(x_min, x_max, args.curve_points)

    summary_outputs = []

    # 4. Execute Workflows
    # --- HOLDOUT WORKFLOW ---
    if args.mode in ["holdout", "both"]:
        holdout_selector = HoldoutModelSelector(
            degrees=args.degrees,
            l2_lambdas=args.l2_values,
            scale_features=args.scale_features,
            selection_rtol=args.selection_rtol,
            selection_atol=args.selection_atol,
            condition_warning_threshold=args.condition_warning_threshold,
        )
        holdout_res = holdout_selector.select(
            x_all,
            y_all,
            split.train_indices,
            split.val_indices,
            split.test_indices,
            split.dev_indices,
        )

        reporter.write_holdout_results(holdout_res)
        reporter.write_final_model(
            workflow_name="holdout",
            best_candidate=holdout_res.best_candidate,
            beta_scaled=holdout_res.final_refitted_beta_scaled,
            beta_orig=holdout_res.final_refitted_beta_orig,
            transformer=holdout_res.final_refitted_transformer,
            test_metrics=holdout_res.final_test_metrics,
            solver_used="solve",
            condition_number=holdout_res.best_candidate.condition_number,
            selection_rtol=args.selection_rtol,
            selection_atol=args.selection_atol,
        )
        reporter.write_test_predictions(
            loaded_data.original_indices[split.test_indices],
            x_all[split.test_indices],
            y_all[split.test_indices],
            holdout_res.test_predictions,
            holdout_res.test_residuals,
        )

        if visualizer is not None:
            best_deg = holdout_res.best_candidate.degree
            best_l2 = holdout_res.best_candidate.l2_lambda

            visualizer.plot_04_holdout_validation_rmse(
                holdout_res.candidates, best_deg, best_l2
            )
            visualizer.plot_05_train_validation_error(
                holdout_res.candidates, best_l2, best_deg
            )
            visualizer.plot_06_holdout_rmse_heatmap(
                holdout_res.candidates, best_deg, best_l2
            )

            # Candidate Model Curves (min, mid, selected, max)
            cand_degs = sorted({c.degree for c in holdout_res.candidates})
            select_degs = sorted(
                {
                            cand_degs[0],
                            cand_degs[len(cand_degs) // 2],
                            best_deg,
                            cand_degs[-1],
                        }
            )
            cand_curves = []
            for cd in select_degs:
                c_trans = PolynomialFeatureTransformer(
                    degree=cd, scale_features=args.scale_features
                )
                X_tr = c_trans.fit_transform(x_all[split.train_indices])
                c_reg = PolynomialRegressor(l2_lambda=best_l2).fit(
                    X_tr, y_all[split.train_indices]
                )
                X_g = c_trans.transform(x_grid)
                cand_curves.append((cd, best_l2, x_grid, c_reg.predict(X_g)))

            visualizer.plot_07_candidate_models(
                x_all[split.train_indices],
                y_all[split.train_indices],
                x_all[split.val_indices],
                y_all[split.val_indices],
                cand_curves,
            )

            # Final Fit & Diagnostic Plots
            X_grid_dev = holdout_res.final_refitted_transformer.transform(x_grid)
            y_grid_pred = X_grid_dev @ holdout_res.final_refitted_beta_scaled

            visualizer.plot_13_final_polynomial_fit(
                x_all[split.dev_indices],
                y_all[split.dev_indices],
                x_all[split.test_indices],
                y_all[split.test_indices],
                x_grid,
                y_grid_pred,
                best_deg,
                best_l2,
                holdout_res.final_test_metrics,
                workflow="holdout",
            )

            # Optional Bootstrap Band
            if args.bootstrap_samples > 0:
                lower_b, upper_b = _compute_bootstrap_bands(
                    x_all[split.dev_indices],
                    y_all[split.dev_indices],
                    x_grid,
                    best_deg,
                    best_l2,
                    args.scale_features,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                )
                visualizer.plot_13b_final_polynomial_bootstrap_band(
                    x_all[split.dev_indices],
                    y_all[split.dev_indices],
                    x_all[split.test_indices],
                    y_all[split.test_indices],
                    x_grid,
                    y_grid_pred,
                    lower_b,
                    upper_b,
                    best_deg,
                    best_l2,
                    args.bootstrap_samples,
                    workflow="holdout",
                )

            visualizer.plot_14_test_actual_vs_predicted(
                y_all[split.test_indices],
                holdout_res.test_predictions,
                holdout_res.final_test_metrics,
                workflow="holdout",
            )
            visualizer.plot_15_test_residuals(
                holdout_res.test_predictions,
                holdout_res.test_residuals,
                workflow="holdout",
            )
            visualizer.plot_16_test_residual_histogram(
                holdout_res.test_residuals, args.residual_bins, workflow="holdout"
            )
            visualizer.plot_17_final_metrics(
                holdout_res.final_test_metrics, workflow="holdout"
            )
            visualizer.plot_18_model_coefficients(
                holdout_res.final_refitted_beta_orig
                if args.scale_features
                else holdout_res.final_refitted_beta_scaled,
                is_original_basis=True,
                workflow="holdout",
            )

            cond_degs = sorted(
                {c.degree for c in holdout_res.candidates if c.l2_lambda == best_l2}
            )
            cond_nums = [
                c.condition_number
                for c in holdout_res.candidates
                if c.l2_lambda == best_l2
            ]
            visualizer.plot_19_condition_numbers(
                cond_degs,
                cond_nums,
                args.condition_warning_threshold,
                workflow="holdout",
            )

            visualizer.plot_20_results_dashboard(
                x_all[split.dev_indices],
                y_all[split.dev_indices],
                x_all[split.test_indices],
                y_all[split.test_indices],
                x_grid,
                y_grid_pred,
                holdout_res.test_predictions,
                holdout_res.test_residuals,
                holdout_res.candidates,
                best_deg,
                best_l2,
                holdout_res.final_test_metrics,
                workflow="holdout",
            )

        summary_outputs.append(
            (
                "Holdout",
                holdout_res.best_candidate.degree,
                holdout_res.best_candidate.l2_lambda,
                holdout_res.final_test_metrics,
            )
        )

    # --- KFOLD WORKFLOW ---
    if args.mode in ["kfold", "both"]:
        kfold_selector = KFoldModelSelector(
            degrees=args.degrees,
            l2_lambdas=args.l2_values,
            k_folds=args.folds,
            seed=args.seed,
            scale_features=args.scale_features,
            selection_rtol=args.selection_rtol,
            selection_atol=args.selection_atol,
            condition_warning_threshold=args.condition_warning_threshold,
        )
        kfold_res = kfold_selector.select(
            x_all, y_all, split.dev_indices, split.test_indices
        )

        reporter.write_kfold_results(kfold_res)
        reporter.write_final_model(
            workflow_name="kfold",
            best_candidate=kfold_res.best_candidate,
            beta_scaled=kfold_res.final_refitted_beta_scaled,
            beta_orig=kfold_res.final_refitted_beta_orig,
            transformer=kfold_res.final_refitted_transformer,
            test_metrics=kfold_res.final_test_metrics,
            solver_used="solve",
            condition_number=kfold_res.best_candidate.mean_condition_number,
            selection_rtol=args.selection_rtol,
            selection_atol=args.selection_atol,
        )
        reporter.write_test_predictions(
            loaded_data.original_indices[split.test_indices],
            x_all[split.test_indices],
            y_all[split.test_indices],
            kfold_res.test_predictions,
            kfold_res.test_residuals,
        )

        # Reconstruct fold assignment array for dev samples
        dev_k_splitter = KFoldSplitter(k=args.folds, seed=args.seed)
        dev_fold_splits = dev_k_splitter.split(np.arange(len(split.dev_indices)))
        dev_fold_nums = np.zeros(len(split.dev_indices), dtype=int)
        for fold in dev_fold_splits:
            dev_fold_nums[fold.val_indices] = fold.fold_index + 1

        reporter.write_oof_predictions(
            loaded_data.original_indices[split.dev_indices],
            dev_fold_nums,
            x_all[split.dev_indices],
            y_all[split.dev_indices],
            kfold_res.oof_predictions,
            kfold_res.oof_residuals,
        )

        if visualizer is not None:
            best_deg = kfold_res.best_candidate.degree
            best_l2 = kfold_res.best_candidate.l2_lambda

            visualizer.plot_08_kfold_assignments(split.dev_indices, dev_fold_splits)
            visualizer.plot_09_kfold_mean_rmse(kfold_res.candidates, best_deg, best_l2)
            visualizer.plot_10_kfold_rmse_heatmap(
                kfold_res.candidates, best_deg, best_l2
            )
            visualizer.plot_10b_kfold_rmse_std_heatmap(
                kfold_res.candidates, best_deg, best_l2
            )

            sel_cand = kfold_res.best_candidate
            visualizer.plot_11_fold_metrics(
                sel_cand.fold_results,
                sel_cand.mean_val_metrics.rmse,
                sel_cand.std_val_metrics.rmse,
                best_deg,
                best_l2,
            )

            oof_metrics = RegressionMetrics.calculate(
                y_all[split.dev_indices],
                kfold_res.oof_predictions,
                num_predictors=best_deg,
            )
            visualizer.plot_12_out_of_fold_predictions(
                y_all[split.dev_indices], kfold_res.oof_predictions, oof_metrics
            )

            X_grid_dev = kfold_res.final_refitted_transformer.transform(x_grid)
            y_grid_pred = X_grid_dev @ kfold_res.final_refitted_beta_scaled

            visualizer.plot_13_final_polynomial_fit(
                x_all[split.dev_indices],
                y_all[split.dev_indices],
                x_all[split.test_indices],
                y_all[split.test_indices],
                x_grid,
                y_grid_pred,
                best_deg,
                best_l2,
                kfold_res.final_test_metrics,
                workflow="kfold",
            )

            if args.bootstrap_samples > 0:
                lower_b, upper_b = _compute_bootstrap_bands(
                    x_all[split.dev_indices],
                    y_all[split.dev_indices],
                    x_grid,
                    best_deg,
                    best_l2,
                    args.scale_features,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                )
                visualizer.plot_13b_final_polynomial_bootstrap_band(
                    x_all[split.dev_indices],
                    y_all[split.dev_indices],
                    x_all[split.test_indices],
                    y_all[split.test_indices],
                    x_grid,
                    y_grid_pred,
                    lower_b,
                    upper_b,
                    best_deg,
                    best_l2,
                    args.bootstrap_samples,
                    workflow="kfold",
                )

            visualizer.plot_14_test_actual_vs_predicted(
                y_all[split.test_indices],
                kfold_res.test_predictions,
                kfold_res.final_test_metrics,
                workflow="kfold",
            )
            visualizer.plot_15_test_residuals(
                kfold_res.test_predictions, kfold_res.test_residuals, workflow="kfold"
            )
            visualizer.plot_16_test_residual_histogram(
                kfold_res.test_residuals, args.residual_bins, workflow="kfold"
            )
            visualizer.plot_17_final_metrics(
                kfold_res.final_test_metrics, workflow="kfold"
            )
            visualizer.plot_18_model_coefficients(
                kfold_res.final_refitted_beta_orig
                if args.scale_features
                else kfold_res.final_refitted_beta_scaled,
                is_original_basis=True,
                workflow="kfold",
            )

            cond_degs = sorted(
                {c.degree for c in kfold_res.candidates if c.l2_lambda == best_l2}
            )
            cond_nums = [
                c.mean_condition_number
                for c in kfold_res.candidates
                if c.l2_lambda == best_l2
            ]
            visualizer.plot_19_condition_numbers(
                cond_degs, cond_nums, args.condition_warning_threshold, workflow="kfold"
            )

            visualizer.plot_20_results_dashboard(
                x_all[split.dev_indices],
                y_all[split.dev_indices],
                x_all[split.test_indices],
                y_all[split.test_indices],
                x_grid,
                y_grid_pred,
                kfold_res.test_predictions,
                kfold_res.test_residuals,
                kfold_res.candidates,
                best_deg,
                best_l2,
                kfold_res.final_test_metrics,
                workflow="kfold",
            )

        summary_outputs.append(
            (
                "K-Fold",
                kfold_res.best_candidate.degree,
                kfold_res.best_candidate.l2_lambda,
                kfold_res.final_test_metrics,
            )
        )

    # Write Manifest if plots generated
    if visualizer is not None:
        reporter.write_plot_manifest(visualizer.manifest_entries)

    # 5. Print Terminal Summary
    _print_terminal_summary(args, N, split, summary_outputs, output_path)


def _compute_bootstrap_bands(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_grid: np.ndarray,
    degree: int,
    l2_lambda: float,
    scale_features: bool,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(x_dev)
    grid_preds = np.zeros((num_samples, len(x_grid)), dtype=np.float64)

    for i in range(num_samples):
        boot_idx = rng.choice(n, size=n, replace=True)
        xb, yb = x_dev[boot_idx], y_dev[boot_idx]

        transformer = PolynomialFeatureTransformer(
            degree=degree, scale_features=scale_features
        )
        Xb = transformer.fit_transform(xb)
        regressor = PolynomialRegressor(l2_lambda=l2_lambda).fit(Xb, yb)

        Xg = transformer.transform(x_grid)
        grid_preds[i, :] = regressor.predict(Xg)

    lower_band = np.percentile(grid_preds, 2.5, axis=0)
    upper_band = np.percentile(grid_preds, 97.5, axis=0)
    return lower_band, upper_band


def _print_terminal_summary(
    args: argparse.Namespace,
    total_loaded: int,
    split: Any,
    summary_outputs: list[tuple],
    output_path: pathlib.Path,
) -> None:
    print("=" * 70)
    print(" POLYNOMIAL REGRESSION PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Loaded Observations: {total_loaded}")
    print(
        f"Dataset Split Sizes: Train={len(split.train_indices)}, Val={len(split.val_indices)}, Test={len(split.test_indices)}"
    )
    print(f"Candidate Degrees:   {args.degrees}")
    print(f"Candidate L2 Values: {args.l2_values}")
    print(f"Feature Scaling:     {'Enabled' if args.scale_features else 'Disabled'}")
    print("-" * 70)

    for mode_name, deg, l2, metrics in summary_outputs:
        print(f"[{mode_name.upper()} WORKFLOW SELECTION]")
        print(f"  Selected Degree:   {deg}")
        print(f"  Selected L2:       {l2}")
        print(f"  Final Test RMSE:   {metrics.rmse:.6f}")
        print(f"  Final Test MAE:    {metrics.mae:.6f}")
        print(f"  Final Test R²:     {metrics.r_squared:.6f}")
        print("-" * 70)

    print(f"Output Directory:    {output_path.resolve()}")
    print("=" * 70)


def main():
    args = parse_args()
    try:
        run_pipeline(args)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
