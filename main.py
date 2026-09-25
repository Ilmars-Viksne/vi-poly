"""Main CLI script for Polynomial Regression Workflows."""

import argparse
import math
import pathlib
import sys
import warnings
from typing import Any

import numpy as np

from polynomial_regression.data import CSVDataLoader
from polynomial_regression.features import (
    DEFAULT_MAX_DEGREE,
    PolynomialFeatureTransformer,
)
from polynomial_regression.metrics import RegressionMetrics
from polynomial_regression.regression import (
    DEFAULT_L1_INITIALIZATION,
    DEFAULT_L1_MAX_ITERATIONS,
    DEFAULT_L1_TOLERANCE,
    DEFAULT_REGULARIZATION_STRENGTHS,
    REGULARIZATION_L2,
    REGULARIZATION_NONE,
    SUPPORTED_REGULARIZATIONS,
    PolynomialRegressor,
)
from polynomial_regression.reporting import ReportGenerator
from polynomial_regression.selection import HoldoutModelSelector, KFoldModelSelector
from polynomial_regression.splitting import DataSplitter, KFoldSplitter


def configure_matplotlib_backend(show_plots: bool) -> None:
    import matplotlib

    if not show_plots:
        matplotlib.use("Agg", force=True)


def _validate_configuration(args: argparse.Namespace) -> None:
    """Validates pipeline configuration settings and canonicalizes search grids."""

    # Avoid re-validation conflict if already validated
    if getattr(args, "_validated", False):
        return

    # 1. Input path & column validation
    if not hasattr(args, "csv_file") or not str(args.csv_file).strip():
        raise ValueError("Input CSV file path must be a non-empty string.")

    if not hasattr(args, "x_column") or not str(args.x_column).strip():
        raise ValueError("Requested X column name cannot be empty or whitespace-only.")

    if not hasattr(args, "y_column") or not str(args.y_column).strip():
        raise ValueError("Requested Y column name cannot be empty or whitespace-only.")

    # 2. Regularization type validation & deprecated option handling
    reg = getattr(args, "regularization", REGULARIZATION_L2)
    if not isinstance(reg, str):
        raise ValueError(f"Regularization type must be a string; received {type(reg)}.")
    norm_reg = reg.strip().lower()
    if norm_reg not in SUPPORTED_REGULARIZATIONS:
        raise ValueError(
            f"Unsupported regularization type '{reg}'. Supported choices: {sorted(SUPPORTED_REGULARIZATIONS)}."
        )

    l2_vals_provided = hasattr(args, "l2_values") and args.l2_values is not None
    reg_vals_provided = (
        hasattr(args, "regularization_values") and args.regularization_values is not None
    )

    if l2_vals_provided and reg_vals_provided:
        raise ValueError(
            "Cannot specify both --l2-values and --regularization-values."
        )

    if l2_vals_provided:
        if hasattr(args, "regularization_explicit") and args.regularization_explicit and norm_reg != REGULARIZATION_L2:
            raise ValueError(
                f"--l2-values cannot be used with --regularization '{norm_reg}'. Use --regularization-values instead."
            )
        warnings.warn(
            "--l2-values is deprecated; use --regularization-values instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        norm_reg = REGULARIZATION_L2
        raw_strengths = args.l2_values
    elif reg_vals_provided:
        raw_strengths = args.regularization_values
    else:
        if norm_reg == REGULARIZATION_NONE:
            raw_strengths = [0.0]
        else:
            raw_strengths = DEFAULT_REGULARIZATION_STRENGTHS

    args.regularization = norm_reg

    # 3. Regularization strengths validation
    if not raw_strengths:
        raise ValueError("At least one regularization strength value must be supplied.")

    validated_strengths: list[float] = []
    for str_val in raw_strengths:
        try:
            val = float(str_val)
        except (ValueError, TypeError):
            raise ValueError(
                f"Regularization strength values must be numeric; received {str_val}."
            )
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(
                f"Regularization strength values must be finite and non-negative; received {str_val}."
            )
        validated_strengths.append(val)

    if norm_reg == REGULARIZATION_NONE:
        if any(v != 0.0 for v in validated_strengths):
            raise ValueError(
                f"Regularization strengths must be 0.0 for 'none' regularization; received {validated_strengths}."
            )
        validated_strengths = [0.0]

    # 4. L1 solver parameters validation
    max_iters = getattr(args, "l1_max_iterations", DEFAULT_L1_MAX_ITERATIONS)
    try:
        max_iters_val = int(max_iters)
    except (ValueError, TypeError):
        raise ValueError(
            f"L1 maximum iterations must be a positive integer; received {max_iters}."
        )
    if max_iters_val < 1:
        raise ValueError(
            f"L1 maximum iterations must be a positive integer; received {max_iters}."
        )
    args.l1_max_iterations = max_iters_val

    tol = getattr(args, "l1_tolerance", DEFAULT_L1_TOLERANCE)
    try:
        tol_val = float(tol)
    except (ValueError, TypeError):
        raise ValueError(
            f"L1 convergence tolerance must be finite and strictly positive; received {tol}."
        )
    if not np.isfinite(tol_val) or tol_val <= 0.0:
        raise ValueError(
            f"L1 convergence tolerance must be finite and strictly positive; received {tol}."
        )
    args.l1_tolerance = tol_val

    l1_init = getattr(args, "l1_initialization", DEFAULT_L1_INITIALIZATION)
    if not isinstance(l1_init, str):
        raise ValueError(
            f"L1 initialization must be a string; received {type(l1_init)}."
        )
    norm_l1_init = l1_init.strip().lower()
    if norm_l1_init not in {"zeros", "ols"}:
        raise ValueError(
            f"Unsupported L1 initialization '{l1_init}'. Supported choices: 'zeros', 'ols'."
        )
    args.l1_initialization = norm_l1_init

    # Feature scaling warning for L1
    if norm_reg == "l1" and not getattr(args, "scale_features", False):
        warnings.warn(
            "L1 regularization is sensitive to feature scale. Consider enabling "
            "--scale-features so polynomial terms are penalized on comparable scales.",
            UserWarning,
            stacklevel=2,
        )

    # 5. Max degree validation
    if not hasattr(args, "max_degree") or not isinstance(
        args.max_degree, (int, np.integer)
    ):
        raise ValueError("Maximum polynomial degree must be an integer.")
    max_deg = int(args.max_degree)
    if max_deg < 0:
        raise ValueError(
            f"Maximum polynomial degree must be non-negative; received {max_deg}."
        )

    # 6. Degrees validation
    if not hasattr(args, "degrees") or not args.degrees:
        raise ValueError("At least one polynomial degree must be supplied.")

    validated_degrees: list[int] = []
    for deg in args.degrees:
        if not isinstance(deg, (int, np.integer)):
            raise TypeError(f"Polynomial degrees must be integers; received {deg}.")
        d_int = int(deg)
        if d_int < 0:
            raise ValueError(
                f"Polynomial degrees must be non-negative integers; received {d_int}."
            )
        if d_int > max_deg:
            raise ValueError(
                f"Polynomial degree {d_int} exceeds the configured maximum degree {max_deg}."
            )
        validated_degrees.append(d_int)

    # 7. Fold count validation
    if not hasattr(args, "folds") or not isinstance(args.folds, (int, np.integer)):
        raise ValueError("Number of folds must be an integer.")
    if int(args.folds) < 2:
        raise ValueError(f"Number of folds must be at least 2; received {args.folds}.")

    # 8. Split ratios validation
    train_ratio = float(args.train_ratio)
    val_ratio = float(args.validation_ratio)
    test_ratio = float(args.test_ratio)
    if not (
        np.isfinite(train_ratio) and np.isfinite(val_ratio) and np.isfinite(test_ratio)
    ):
        raise ValueError("Split ratios must be finite numbers.")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError(
            f"All split ratios must be positive (> 0); received train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0 within tolerance; sum is {total_ratio:.6f}."
        )

    # 9. Curve points validation
    if not hasattr(args, "curve_points") or not isinstance(
        args.curve_points, (int, np.integer)
    ):
        raise ValueError("Curve point count must be an integer.")
    if int(args.curve_points) < 2:
        raise ValueError(
            f"Curve point count must be at least 2; received {args.curve_points}."
        )

    # 10. Bootstrap samples validation
    if not hasattr(args, "bootstrap_samples") or not isinstance(
        args.bootstrap_samples, (int, np.integer)
    ):
        raise ValueError("Bootstrap sample count must be an integer.")
    if int(args.bootstrap_samples) < 0:
        raise ValueError(
            f"Bootstrap sample count must be non-negative; received {args.bootstrap_samples}."
        )

    # 11. Plot DPI validation
    if not hasattr(args, "plot_dpi") or not isinstance(
        args.plot_dpi, (int, np.integer)
    ):
        raise ValueError("Plot DPI must be an integer.")
    if int(args.plot_dpi) <= 0:
        raise ValueError(
            f"Plot DPI must be a positive integer; received {args.plot_dpi}."
        )

    # 12. Condition warning threshold
    cond_thresh = float(args.condition_warning_threshold)
    if not np.isfinite(cond_thresh) or cond_thresh <= 0.0:
        raise ValueError(
            f"Condition warning threshold must be finite and strictly positive (> 0); received {cond_thresh}."
        )

    # 13. Selection tolerances
    rtol = float(args.selection_rtol)
    atol = float(args.selection_atol)
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(
            f"Selection relative tolerance must be finite and non-negative; received {rtol}."
        )
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(
            f"Selection absolute tolerance must be finite and non-negative; received {atol}."
        )

    # 14. Residual bins
    if args.residual_bins is not None and (
        not isinstance(args.residual_bins, (int, np.integer))
        or int(args.residual_bins) <= 0
    ):
        raise ValueError(
            f"Residual bins count must be a positive integer; received {args.residual_bins}."
        )

    # 15. Seeds validation
    if not isinstance(args.seed, (int, np.integer)):
        raise TypeError("Random seed must be an integer.")
    if not isinstance(args.bootstrap_seed, (int, np.integer)):
        raise TypeError("Bootstrap seed must be an integer.")

    # Store raw requested grids before canonicalization
    if not hasattr(args, "_requested_degrees") or args._requested_degrees is None:
        args._requested_degrees = list(args.degrees)
    if (
        not hasattr(args, "_requested_regularization_values")
        or args._requested_regularization_values is None
    ):
        args._requested_regularization_values = list(raw_strengths)
    args._requested_l2_values = list(raw_strengths)

    # Store canonicalized effective grids
    args.degrees = sorted(set(validated_degrees))
    args.regularization_values = sorted(set(validated_strengths))
    args.l2_values = args.regularization_values
    args._validated = True


def validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    try:
        _validate_configuration(args)
    except ValueError as exc:
        if parser is not None:
            parser.error(str(exc))
        raise
    return args


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
        "--regularization",
        choices=["none", "l1", "l2"],
        default=REGULARIZATION_L2,
        help="Regularization family: 'none', 'l1', or 'l2' (default: l2).",
    )
    parser.add_argument(
        "--regularization-values",
        type=float,
        nargs="+",
        default=None,
        help="Candidate regularization strengths.",
    )
    parser.add_argument(
        "--l2-values",
        type=float,
        nargs="+",
        default=None,
        help="[Deprecated] Candidate L2 regularization strengths.",
    )
    parser.add_argument(
        "--l1-max-iterations",
        type=int,
        default=DEFAULT_L1_MAX_ITERATIONS,
        help=f"Maximum coordinate-descent iterations for L1 solver (default: {DEFAULT_L1_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--l1-tolerance",
        type=float,
        default=DEFAULT_L1_TOLERANCE,
        help=f"Convergence tolerance for L1 solver (default: {DEFAULT_L1_TOLERANCE}).",
    )
    parser.add_argument(
        "--l1-initialization",
        choices=["zeros", "ols"],
        default=DEFAULT_L1_INITIALIZATION,
        help=f"Initialization method for L1 coordinate descent (default: {DEFAULT_L1_INITIALIZATION}).",
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Candidate polynomial degrees.",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=DEFAULT_MAX_DEGREE,
        help=f"Maximum allowed polynomial degree (default: {DEFAULT_MAX_DEGREE}).",
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
        help="Number of bootstrap resamples used to estimate the fitted-curve uncertainty band; 0 disables the band.",
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

    raw_args = list(args) if args is not None else sys.argv[1:]
    parsed = parser.parse_args(args)
    parsed._raw_argv = raw_args
    parsed.regularization_explicit = any(
        a == "--regularization" or a.startswith("--regularization=") for a in raw_args
    )
    return validate_args(parsed, parser)


def run_pipeline(args: argparse.Namespace) -> None:
    # Programmatic configuration validation (raises ValueError on failure)
    validate_args(args, parser=None)

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

    if args.folds > len(split.dev_indices):
        raise ValueError(
            f"Number of folds {args.folds} exceeds the number of development samples {len(split.dev_indices)}."
        )

    kfold_assignments = {}
    if args.mode in ["kfold", "both"]:
        k_splitter = KFoldSplitter(k=args.folds, seed=args.seed)
        fold_splits = k_splitter.split(np.arange(len(split.dev_indices)))
        for fold in fold_splits:
            kfold_assignments[f"fold_{fold.fold_index + 1}"] = split.dev_indices[
                fold.val_indices
            ].tolist()

    # 3. Common Outputs
    common_dir = output_path / "common"
    common_reporter = ReportGenerator(common_dir)
    common_reporter.write_run_metadata(args, args.csv_file)
    common_reporter.write_split_summary(
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        total_samples=N,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
        dev_indices=split.dev_indices,
        train_observation_indices=loaded_data.observation_indices[split.train_indices],
        val_observation_indices=loaded_data.observation_indices[split.val_indices],
        test_observation_indices=loaded_data.observation_indices[split.test_indices],
        dev_observation_indices=loaded_data.observation_indices[split.dev_indices],
        train_csv_line_numbers=loaded_data.csv_line_numbers[split.train_indices],
        val_csv_line_numbers=loaded_data.csv_line_numbers[split.val_indices],
        test_csv_line_numbers=loaded_data.csv_line_numbers[split.test_indices],
        dev_csv_line_numbers=loaded_data.csv_line_numbers[split.dev_indices],
        fold_assignments=kfold_assignments,
        skipped_rows=loaded_data.skipped_rows,
    )

    if not args.no_plots:
        configure_matplotlib_backend(args.show_plots)
        from polynomial_regression.visualization import RegressionVisualizer

        common_visualizer = RegressionVisualizer(
            output_dir=common_dir,
            plot_format=args.plot_format,
            plot_dpi=args.plot_dpi,
            plot_style=args.plot_style,
            show_plots=args.show_plots,
        )
        common_visualizer.plot_01_original_data(x_all, y_all)
        common_visualizer.plot_02_data_splits(
            x_all, y_all, split.train_indices, split.val_indices, split.test_indices
        )
        common_visualizer.plot_03_split_sizes(
            len(split.train_indices), len(split.val_indices), len(split.test_indices)
        )
        common_reporter.write_plot_manifest(common_visualizer.manifest_entries)

    # Dense X Grid for smooth curve plotting across observed X range
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    x_grid = np.linspace(x_min, x_max, args.curve_points)

    summary_outputs = []
    holdout_res = None
    kfold_res = None

    # 4. Execute Workflows
    # --- HOLDOUT WORKFLOW ---
    if args.mode in ["holdout", "both"]:
        holdout_dir = output_path / "holdout"
        holdout_reporter = ReportGenerator(holdout_dir)

        holdout_selector = HoldoutModelSelector(
            degrees=args.degrees,
            regularization=args.regularization,
            regularization_strengths=args.regularization_values,
            l1_max_iterations=args.l1_max_iterations,
            l1_tolerance=args.l1_tolerance,
            l1_initialization=args.l1_initialization,
            scale_features=args.scale_features,
            selection_rtol=args.selection_rtol,
            selection_atol=args.selection_atol,
            condition_warning_threshold=args.condition_warning_threshold,
            max_degree=args.max_degree,
        )
        holdout_res = holdout_selector.select(
            x_all,
            y_all,
            split.train_indices,
            split.val_indices,
            split.test_indices,
            split.dev_indices,
        )

        holdout_reporter.write_holdout_results(holdout_res)
        holdout_reporter.write_final_model(
            workflow_name="holdout",
            best_candidate=holdout_res.best_candidate,
            beta_scaled=holdout_res.final_refitted_beta_scaled,
            beta_orig=holdout_res.final_refitted_beta_orig,
            transformer=holdout_res.final_refitted_transformer,
            test_metrics=holdout_res.final_test_metrics,
            fit_details=holdout_res.final_fit_details,
            selection_rtol=args.selection_rtol,
            selection_atol=args.selection_atol,
        )
        holdout_reporter.write_test_predictions(
            loaded_data.observation_indices[split.test_indices],
            loaded_data.csv_line_numbers[split.test_indices],
            x_all[split.test_indices],
            y_all[split.test_indices],
            holdout_res.test_predictions,
            holdout_res.test_residuals,
        )

        if not args.no_plots:
            holdout_visualizer = RegressionVisualizer(
                output_dir=holdout_dir,
                plot_format=args.plot_format,
                plot_dpi=args.plot_dpi,
                plot_style=args.plot_style,
                show_plots=args.show_plots,
            )
            best_cand = holdout_res.best_candidate
            best_deg = best_cand.degree
            best_strength = best_cand.regularization_strength

            holdout_visualizer.plot_04_holdout_validation_rmse(
                holdout_res.candidates, best_deg, best_strength
            )
            holdout_visualizer.plot_05_train_validation_error(
                holdout_res.candidates, best_strength, best_deg
            )
            holdout_visualizer.plot_06_holdout_rmse_heatmap(
                holdout_res.candidates, best_deg, best_strength
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
                    degree=cd,
                    scale_features=args.scale_features,
                    max_degree=args.max_degree,
                )
                X_tr = c_trans.fit_transform(x_all[split.train_indices])
                c_reg = PolynomialRegressor(
                    regularization=best_cand.regularization,
                    regularization_strength=best_strength,
                    l1_max_iterations=args.l1_max_iterations,
                    l1_tolerance=args.l1_tolerance,
                    l1_initialization=args.l1_initialization,
                ).fit(X_tr, y_all[split.train_indices])
                X_g = c_trans.transform(x_grid)
                cand_curves.append((cd, best_strength, x_grid, c_reg.predict(X_g)))

            holdout_visualizer.plot_07_candidate_models(
                x_all[split.train_indices],
                y_all[split.train_indices],
                x_all[split.val_indices],
                y_all[split.val_indices],
                cand_curves,
            )

            # Final Fit & Diagnostic Plots
            X_grid_dev = holdout_res.final_refitted_transformer.transform(x_grid)
            y_grid_pred = X_grid_dev @ holdout_res.final_refitted_beta_scaled

            holdout_visualizer.plot_13_final_polynomial_fit(
                x_all[split.dev_indices],
                y_all[split.dev_indices],
                x_all[split.test_indices],
                y_all[split.test_indices],
                x_grid,
                y_grid_pred,
                best_deg,
                best_strength,
                holdout_res.final_test_metrics,
                workflow="holdout",
                regularization=best_cand.regularization,
            )

            # Optional Bootstrap Band
            if args.bootstrap_samples > 0:
                lower_b, upper_b = _compute_bootstrap_bands(
                    x_all[split.dev_indices],
                    y_all[split.dev_indices],
                    x_grid,
                    best_deg,
                    regularization=best_cand.regularization,
                    regularization_strength=best_strength,
                    l1_max_iterations=args.l1_max_iterations,
                    l1_tolerance=args.l1_tolerance,
                    l1_initialization=args.l1_initialization,
                    scale_features=args.scale_features,
                    num_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed,
                    max_degree=args.max_degree,
                )
                holdout_visualizer.plot_13b_fitted_curve_uncertainty_band(
                    x_all[split.dev_indices],
                    y_all[split.dev_indices],
                    x_all[split.test_indices],
                    y_all[split.test_indices],
                    x_grid,
                    y_grid_pred,
                    lower_b,
                    upper_b,
                    best_deg,
                    best_strength,
                    args.bootstrap_samples,
                    workflow="holdout",
                    regularization=best_cand.regularization,
                )

            holdout_visualizer.plot_14_test_actual_vs_predicted(
                y_all[split.test_indices],
                holdout_res.test_predictions,
                holdout_res.final_test_metrics,
                workflow="holdout",
            )
            holdout_visualizer.plot_15_test_residuals(
                holdout_res.test_predictions,
                holdout_res.test_residuals,
                workflow="holdout",
            )
            holdout_visualizer.plot_16_test_residual_histogram(
                holdout_res.test_residuals, args.residual_bins, workflow="holdout"
            )
            holdout_visualizer.plot_17_final_metrics(
                holdout_res.final_test_metrics, workflow="holdout"
            )
            holdout_visualizer.plot_18_model_coefficients(
                holdout_res.final_refitted_beta_orig
                if args.scale_features
                else holdout_res.final_refitted_beta_scaled,
                is_original_basis=True,
                workflow="holdout",
                regularization=best_cand.regularization,
                regularization_strength=best_strength,
                nonzero_coefficient_count=holdout_res.final_fit_details.nonzero_coefficient_count,
            )

            cond_degs = sorted(
                {
                    c.degree
                    for c in holdout_res.candidates
                    if c.regularization_strength == best_strength
                }
            )
            cond_nums = [
                c.condition_number
                for c in holdout_res.candidates
                if c.regularization_strength == best_strength
            ]
            holdout_visualizer.plot_19_condition_numbers(
                cond_degs,
                cond_nums,
                args.condition_warning_threshold,
                workflow="holdout",
                regularization=best_cand.regularization,
            )

            holdout_visualizer.plot_20_results_dashboard(
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
                best_strength,
                holdout_res.final_test_metrics,
                workflow="holdout",
            )
            holdout_reporter.write_plot_manifest(holdout_visualizer.manifest_entries)

        summary_outputs.append(
            (
                "Holdout",
                holdout_res.best_candidate.degree,
                holdout_res.best_candidate.regularization,
                holdout_res.best_candidate.regularization_strength,
                holdout_res.final_test_metrics,
            )
        )

    # --- KFOLD WORKFLOW ---
    if args.mode in ["kfold", "both"]:
        kfold_dir = output_path / "kfold"
        kfold_reporter = ReportGenerator(kfold_dir)

        kfold_selector = KFoldModelSelector(
            degrees=args.degrees,
            regularization=args.regularization,
            regularization_strengths=args.regularization_values,
            l1_max_iterations=args.l1_max_iterations,
            l1_tolerance=args.l1_tolerance,
            l1_initialization=args.l1_initialization,
            k_folds=args.folds,
            seed=args.seed,
            scale_features=args.scale_features,
            selection_rtol=args.selection_rtol,
            selection_atol=args.selection_atol,
            condition_warning_threshold=args.condition_warning_threshold,
            max_degree=args.max_degree,
        )
        kfold_res = kfold_selector.select(
            x_all, y_all, split.dev_indices, split.test_indices
        )

        kfold_reporter.write_kfold_results(kfold_res)
        kfold_reporter.write_final_model(
            workflow_name="kfold",
            best_candidate=kfold_res.best_candidate,
            beta_scaled=kfold_res.final_refitted_beta_scaled,
            beta_orig=kfold_res.final_refitted_beta_orig,
            transformer=kfold_res.final_refitted_transformer,
            test_metrics=kfold_res.final_test_metrics,
            fit_details=kfold_res.final_fit_details,
            selection_rtol=args.selection_rtol,
            selection_atol=args.selection_atol,
        )
        kfold_reporter.write_test_predictions(
            loaded_data.observation_indices[split.test_indices],
            loaded_data.csv_line_numbers[split.test_indices],
            x_all[split.test_indices],
            y_all[split.test_indices],
            kfold_res.test_predictions,
            kfold_res.test_residuals,
        )

        dev_k_splitter = KFoldSplitter(k=args.folds, seed=args.seed)
        dev_fold_splits = dev_k_splitter.split(np.arange(len(split.dev_indices)))
        dev_fold_nums = np.zeros(len(split.dev_indices), dtype=int)
        for fold in dev_fold_splits:
            dev_fold_nums[fold.val_indices] = fold.fold_index + 1

        kfold_reporter.write_oof_predictions(
            loaded_data.observation_indices[split.dev_indices],
            loaded_data.csv_line_numbers[split.dev_indices],
            dev_fold_nums,
            x_all[split.dev_indices],
            y_all[split.dev_indices],
            kfold_res.oof_predictions,
            kfold_res.oof_residuals,
        )

        if not args.no_plots:
            kfold_visualizer = RegressionVisualizer(
                output_dir=kfold_dir,
                plot_format=args.plot_format,
                plot_dpi=args.plot_dpi,
                plot_style=args.plot_style,
                show_plots=args.show_plots,
            )
            best_cand = kfold_res.best_candidate
            best_deg = best_cand.degree
            best_strength = best_cand.regularization_strength

            kfold_visualizer.plot_08_kfold_assignments(
                split.dev_indices, dev_fold_splits
            )
            kfold_visualizer.plot_09_kfold_mean_rmse(
                kfold_res.candidates, best_deg, best_strength
            )
            kfold_visualizer.plot_10_kfold_rmse_heatmap(
                kfold_res.candidates, best_deg, best_strength
            )
            kfold_visualizer.plot_10b_kfold_rmse_std_heatmap(
                kfold_res.candidates, best_deg, best_strength
            )

            sel_cand = kfold_res.best_candidate
            kfold_visualizer.plot_11_fold_metrics(
                sel_cand.fold_results,
                sel_cand.mean_val_metrics.rmse,
                sel_cand.std_val_metrics.rmse,
                best_deg,
                best_strength,
                regularization=best_cand.regularization,
            )

            oof_metrics = RegressionMetrics.calculate(
                y_all[split.dev_indices],
                kfold_res.oof_predictions,
                num_predictors=best_deg,
            )
            kfold_visualizer.plot_12_out_of_fold_predictions(
                y_all[split.dev_indices], kfold_res.oof_predictions, oof_metrics
            )

            X_grid_dev = kfold_res.final_refitted_transformer.transform(x_grid)
            y_grid_pred = X_grid_dev @ kfold_res.final_refitted_beta_scaled

            kfold_visualizer.plot_13_final_polynomial_fit(
                x_all[split.dev_indices],
                y_all[split.dev_indices],
                x_all[split.test_indices],
                y_all[split.test_indices],
                x_grid,
                y_grid_pred,
                best_deg,
                best_strength,
                kfold_res.final_test_metrics,
                workflow="kfold",
                regularization=best_cand.regularization,
            )

            if args.bootstrap_samples > 0:
                lower_b, upper_b = _compute_bootstrap_bands(
                    x_all[split.dev_indices],
                    y_all[split.dev_indices],
                    x_grid,
                    best_deg,
                    regularization=best_cand.regularization,
                    regularization_strength=best_strength,
                    l1_max_iterations=args.l1_max_iterations,
                    l1_tolerance=args.l1_tolerance,
                    l1_initialization=args.l1_initialization,
                    scale_features=args.scale_features,
                    num_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed,
                    max_degree=args.max_degree,
                )
                kfold_visualizer.plot_13b_fitted_curve_uncertainty_band(
                    x_all[split.dev_indices],
                    y_all[split.dev_indices],
                    x_all[split.test_indices],
                    y_all[split.test_indices],
                    x_grid,
                    y_grid_pred,
                    lower_b,
                    upper_b,
                    best_deg,
                    best_strength,
                    args.bootstrap_samples,
                    workflow="kfold",
                    regularization=best_cand.regularization,
                )

            kfold_visualizer.plot_14_test_actual_vs_predicted(
                y_all[split.test_indices],
                kfold_res.test_predictions,
                kfold_res.final_test_metrics,
                workflow="kfold",
            )
            kfold_visualizer.plot_15_test_residuals(
                kfold_res.test_predictions,
                kfold_res.test_residuals,
                workflow="kfold",
            )
            kfold_visualizer.plot_16_test_residual_histogram(
                kfold_res.test_residuals, args.residual_bins, workflow="kfold"
            )
            kfold_visualizer.plot_17_final_metrics(
                kfold_res.final_test_metrics, workflow="kfold"
            )
            kfold_visualizer.plot_18_model_coefficients(
                kfold_res.final_refitted_beta_orig
                if args.scale_features
                else kfold_res.final_refitted_beta_scaled,
                is_original_basis=True,
                workflow="kfold",
                regularization=best_cand.regularization,
                regularization_strength=best_strength,
                nonzero_coefficient_count=kfold_res.final_fit_details.nonzero_coefficient_count,
            )

            cond_degs = sorted(
                {
                    c.degree
                    for c in kfold_res.candidates
                    if c.regularization_strength == best_strength
                }
            )
            cond_nums = [
                c.mean_condition_number
                for c in kfold_res.candidates
                if c.regularization_strength == best_strength
            ]
            kfold_visualizer.plot_19_condition_numbers(
                cond_degs,
                cond_nums,
                args.condition_warning_threshold,
                workflow="kfold",
                regularization=best_cand.regularization,
            )

            kfold_visualizer.plot_20_results_dashboard(
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
                best_strength,
                kfold_res.final_test_metrics,
                workflow="kfold",
            )
            kfold_reporter.write_plot_manifest(kfold_visualizer.manifest_entries)

        summary_outputs.append(
            (
                "K-Fold",
                kfold_res.best_candidate.degree,
                kfold_res.best_candidate.regularization,
                kfold_res.best_candidate.regularization_strength,
                kfold_res.final_test_metrics,
            )
        )

    # Top-level comparison for 'both' mode
    if args.mode == "both" and holdout_res is not None and kfold_res is not None:
        top_reporter = ReportGenerator(output_path)
        top_reporter.write_comparison(holdout_res, kfold_res)

    # 5. Print Terminal Summary
    _print_terminal_summary(args, N, split, summary_outputs, output_path)


def _compute_bootstrap_bands(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_grid: np.ndarray,
    degree: int,
    regularization: str,
    regularization_strength: float,
    l1_max_iterations: int,
    l1_tolerance: float,
    l1_initialization: str,
    scale_features: bool,
    num_samples: int,
    seed: int,
    max_degree: int = DEFAULT_MAX_DEGREE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(x_dev)
    grid_preds = np.zeros((num_samples, len(x_grid)), dtype=np.float64)

    for i in range(num_samples):
        boot_idx = rng.choice(n, size=n, replace=True)
        xb, yb = x_dev[boot_idx], y_dev[boot_idx]

        transformer = PolynomialFeatureTransformer(
            degree=degree, scale_features=scale_features, max_degree=max_degree
        )
        Xb = transformer.fit_transform(xb)
        regressor = PolynomialRegressor(
            regularization=regularization,
            regularization_strength=regularization_strength,
            l1_max_iterations=l1_max_iterations,
            l1_tolerance=l1_tolerance,
            l1_initialization=l1_initialization,
        ).fit(Xb, yb)

        if regressor.fit_details.converged is False:
            raise RuntimeError(
                f"L1 bootstrap refit {i} of {num_samples} failed to converge within {l1_max_iterations} iterations. "
                "The fitted-curve uncertainty band was not generated."
            )

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
    print(f"Regularization Type: {args.regularization}")
    print(f"Candidate Strengths: {args.regularization_values}")
    print(f"Feature Scaling:     {'Enabled' if args.scale_features else 'Disabled'}")
    print("-" * 70)

    for mode_name, deg, reg_type, strength, metrics in summary_outputs:
        print(f"[{mode_name.upper()} WORKFLOW SELECTION]")
        print(f"  Selected Degree:   {deg}")
        print(f"  Selected Reg Type: {reg_type}")
        print(f"  Selected Strength: {strength}")
        print(f"  Final Test RMSE:   {metrics.rmse:.6f}")
        print(f"  Final Test MAE:    {metrics.mae:.6f}")
        print(f"  Final Test R²:     {metrics.r_squared:.6f}")
        print("-" * 70)

    if args.mode == "both":
        print("No workflow was selected using test-set performance.")
        print("-" * 70)

    print(f"Output Directory:    {output_path.resolve()}")
    print("=" * 70)


def main():
    args = parse_args()
    try:
        run_pipeline(args)
    except (OSError, RuntimeError, ValueError, FloatingPointError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
