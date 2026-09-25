# Polynomial Regression with NumPy and Matplotlib

A complete, maintainable, modular Python framework for polynomial regression analysis, model selection, hyperparameter tuning (L2 regularization and polynomial degree), and scientific visualization built strictly using NumPy, Matplotlib, and the Python Standard Library.

---

## Conceptual Foundations & Overview

### Train, Validation, and Test Sets
When fitting predictive models, evaluating performance on the data used for training leads to overly optimistic performance estimates due to overfitting. To reliably estimate real-world generalization:
- **Training Set (70% default)**: Used strictly to solve for model parameters ($\beta$ coefficients) and fit feature scaling parameters ($\mu, \sigma$).
- **Validation Set (15% default)**: Used to tune hyperparameters (polynomial degree $d$, L2 regularization strength $\lambda$) and compare competing candidate models.
- **Test Set (15% default)**: Reserved and kept **completely untouched** during feature scaling and hyperparameter selection. Evaluated exactly once at the end to provide an unbiased estimate of generalization error.

### Why K-Fold Cross-Validation Still Requires a Test Set
K-Fold cross-validation provides a robust out-of-fold estimate of validation error on development data. However, using cross-validation performance to select the best hyperparameter combination introduces a subtle form of optimization bias ("selection bias"). Therefore, reserving a separate test set remains essential to verify that the final selected model generalizes well to unseen data.

---

## Mathematical Formulation & Least-Squares Solvers

### Polynomial Regression Equation
For a scalar feature $x$, polynomial regression models the target $y$ as:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d + \epsilon$$

where:
- $\beta_0$ is the intercept term. Degree zero ($d=0$) remains valid as an intercept-only constant model.
- $\beta_1, \beta_2, \dots, \beta_d$ are the polynomial coefficients in **ascending order of power**.
- $d$ is the polynomial degree.
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is random Gaussian noise.

### Design Matrix & Feature Scaling
Given $N$ samples, the design matrix $\mathbf{X}$ is constructed as:

$$\mathbf{X} = \begin{bmatrix}
1 & x_1 & x_1^2 & \dots & x_1^d \\
1 & x_2 & x_2^2 & \dots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_N & x_N^2 & \dots & x_N^d
\end{bmatrix}$$

#### Z-Score Feature Scaling (Opt-In)
When `--scale-features` is enabled, every non-intercept column $p \ge 1$ is standardized:

$$z_{i, p} = \frac{x_i^p - \mu_p}{\sigma_p}$$

The intercept column $x^0 = 1$ is left unscaled. Standardizing polynomial terms prevents ill-conditioning of the Vandermonde matrix as degree $d$ increases. Note that feature scaling is applied **after** calculating polynomial powers, so scaling itself does not prevent floating-point overflow during initial power construction.

#### Prevention of Data Leakage
- In Holdout mode, feature scaling parameters ($\mu_p, \sigma_p$) are calculated from the **Training Set only**.
- In K-Fold CV, scaling parameters are calculated independently inside each fold's **Training Partition**.
- When refitting on development data, scaling statistics are recomputed from the development dataset.
- **The Test Set is never used to compute scaling parameters.**

#### Conversion to Original Basis Coefficients
When feature scaling is active, coefficients $\beta^{\text{scaled}}$ are converted back to the original polynomial basis $\beta^{\text{orig}}$ via:

$$\beta^{\text{orig}}_p = \frac{\beta^{\text{scaled}}_p}{\sigma_p} \quad \text{for } p \ge 1$$

$$\beta^{\text{orig}}_0 = \beta^{\text{scaled}}_0 - \sum_{p=1}^d \frac{\beta^{\text{scaled}}_p \cdot \mu_p}{\sigma_p}$$

This conversion is verified numerically using `np.allclose(pred_scaled, pred_orig, rtol=1e-10, atol=1e-12)`.

---

## Least-Squares Formulations & Numerical Stability

To avoid the loss of precision associated with forming normal equations ($\mathbf{X}^T \mathbf{X}$), all regression models are solved directly using singular value decomposition or QR factorization via `np.linalg.lstsq`.

### Ordinary Least Squares (OLS)
When $\lambda = 0.0$, the model parameters $\boldsymbol{\beta}$ are solved directly from the original system:

$$\boldsymbol{\beta}, \text{residuals}, \text{rank}, \text{singular\_values} = \text{np.linalg.lstsq}(\mathbf{X}, \mathbf{y}, \text{rcond}=\text{None})$$

- **Solver Label**: `lstsq_ols`
- **Normal Equations**: Not formed or solved.
- **Rank Deficient Handling**: `np.linalg.lstsq` returns the minimum-norm solution for rank-deficient systems and issues a `UserWarning`.

### L2 Regularization (Ridge) via Augmented System
When $\lambda > 0.0$, Ridge regression is solved using an augmented least-squares formulation:

$$\mathbf{X}_{\text{aug}} = \begin{bmatrix} \mathbf{X} \\ \sqrt{\lambda} \mathbf{L} \end{bmatrix}, \quad \mathbf{y}_{\text{aug}} = \begin{bmatrix} \mathbf{y} \\ \mathbf{0}_{p+1} \end{bmatrix}$$

$$\boldsymbol{\beta} = \text{np.linalg.lstsq}(\mathbf{X}_{\text{aug}}, \mathbf{y}_{\text{aug}}, \text{rcond}=\text{None})[0]$$

where $\mathbf{L} = \text{diag}(0, 1, 1, \dots, 1)$. **The intercept $\beta_0$ is explicitly excluded from regularization** ($L_{00} = 0$).

- **Solver Label**: `lstsq_augmented_ridge`
- **Normal Equations**: Neither $\mathbf{X}^T \mathbf{X} + \lambda \mathbf{R}$ nor `np.linalg.solve` is used.

### Condition Number & Solver Diagnostics
`ModelFitDetails` records comprehensive numerical diagnostics for the final model refit:
- `design_condition_number`: Condition number of the original polynomial design matrix $\mathbf{X}$.
- `solver_condition_number`: Condition number of the matrix actually passed to `np.linalg.lstsq` ($\mathbf{X}$ for OLS, $\mathbf{X}_{\text{aug}}$ for Ridge).
- `condition_number`: Alias for `solver_condition_number` for backward compatibility.
- `rank` & `full_rank`: Effective rank and full-rank status returned by the solver.

---

## Input Validation & Search Grid Canonicalization

### Command-Line Arguments Validation
All CLI configurations undergo strict pre-flight validation to prevent execution with invalid or ambiguous parameters:
- **Degrees (`--degrees`)**: Non-empty list of non-negative integers satisfying $0 \le d \le \text{max\_degree}$. Degree 0 is valid.
- **Maximum Degree (`--max-degree`)**: Non-negative integer (default: 50). Excessive degrees raise `ValueError`.
- **L2 Regularization (`--l2-values`)**: Non-empty list of finite, non-negative floats ($\lambda \ge 0.0$). Negative, NaN, or infinite values are strictly rejected.
- **Fold Count (`--folds`)**: Integer $\ge 2$. Must not exceed the total number of development samples after data partitioning.
- **Curve Points (`--curve-points`)**: Integer $\ge 2$. Validated consistently even with `--no-plots`.
- **Bootstrap Samples (`--bootstrap-samples`)**: Non-negative integer ($0 = \text{disabled}$).
- **Plot DPI (`--plot-dpi`)**: Positive integer ($> 0$).
- **Split Ratios (`--train-ratio`, `--validation-ratio`, `--test-ratio`)**: Finite, strictly positive numbers summing to 1.0 within tolerance (`1e-5`).
- **Selection Tolerances (`--selection-rtol`, `--selection-atol`)**: Finite, non-negative floats.
- **Condition Warning Threshold (`--condition-warning-threshold`)**: Finite, strictly positive float ($> 0.0$).
- **Residual Bins (`--residual-bins`)**: Positive integer if supplied.

Validation uses a two-level exception strategy:
- CLI calls via `parse_args()` trigger `parser.error()` and exit with a non-zero status via `SystemExit`.
- Direct Python API calls to `run_pipeline(args)` or model selectors raise `ValueError` before creating output directories or artifacts.

### Search Grid Canonicalization
Search grids specified via `--degrees` and `--l2-values` are canonicalized before model fitting:
1. Every requested value is validated prior to deduplication.
2. Duplicate entries are removed via exact Python equality (not using `np.isclose`).
3. Effective grids are sorted in ascending order (`sorted(set(...))`).
4. Both raw requested grids and canonical effective grids are stored in reproducibility metadata (`run_metadata.json`).

### Power Construction Guards
1. **Logarithmic Preflight Check**: Before calculating powers, $\text{degree} \cdot \log(\max |x|)$ is compared against $\log(\text{float64\_max})$ to catch overflow before execution.
2. **Guarded Multiplication**: Features are generated iteratively ($x^p = x^{p-1} \cdot x$) inside NumPy error states (`np.errstate(over="raise")`).
3. **Finite Array Enforcement**: All numerical arrays (`x`, `X`, `y`, predictions, converted coefficients) are strictly validated for finiteness, non-emptiness, and dimension constraints. Non-finite values or computational overflows raise `FloatingPointError`.

---

## CSV Data Traceability & Provenance System

The pipeline enforces complete observation traceability from raw CSV input through split summaries and prediction outputs using three distinct index views:

1. **`loaded_array_index`**: Zero-based array index in the filtered loaded arrays (`loaded_data.x`, `loaded_data.y`). Describes observation position after skipping invalid records.
2. **`observation_index`**: Zero-based record counter among all CSV data records following the header. Skipped invalid rows do not renumber subsequent observation indices.
3. **`csv_line_number`**: One-based physical line number in the CSV file, starting at line 2 for the first data row under a single-line header (retrieved from `reader.line_num`).

### Compatibility Alias
For backward compatibility with existing code and tests, `LoadedData.original_indices` is preserved as a `@property` returning `observation_indices`.

### Updated Prediction CSV Schemas
Prediction CSV files explicitly disambiguate provenance:
- **`test_predictions.csv`**:
  `observation_index, csv_line_number, X, actual_Y, predicted_Y, residual`
- **`out_of_fold_predictions.csv`**:
  `observation_index, csv_line_number, fold_number, X, actual_Y, oof_predicted_Y, residual`

### Skipped Rows & Split Summary
In `split_summary.json`, skipped invalid rows are reported as structured `SkippedRow` objects:
```json
{
  "observation_index": 4,
  "csv_line_number": 6,
  "columns": "X",
  "reason": "Value is NaN"
}
```
Subset indices in `split_summary.json` record all three index systems (`loaded_array_indices`, `observation_indices`, and `csv_line_numbers`).

---

## Residual Conventions & Diagnostics

- **Residual Definition**: Residual is defined consistently as $\text{residual} = y_{\text{actual}} - y_{\text{predicted}}$.
- **Diagnostic Usage**: Residual vs. Predicted plots and Residual Histograms serve as visual diagnostic tools to inspect homoscedasticity, non-linearity, and outlier influence.

---

## Scientific Visualizations & Presentation Enhancements

### Matplotlib Backend Selection
- **Centralized Configuration**: Matplotlib backend selection is handled centrally in `main.py` prior to importing `RegressionVisualizer` or `matplotlib.pyplot`.
- **Headless Execution**: Non-interactive execution uses the `Agg` backend (`matplotlib.use("Agg", force=True)`), ensuring figure rendering is completely safe in headless CI and server environments.
- **Interactive Display**: Specifying `--show-plots` preserves interactive GUI rendering capabilities without mutating backends per visualizer instance.

### Adaptive $R^2$ Axis Limits & Negative $R^2$ Visualization
- **Dynamic Bounds**: Vertical axis limits for $R^2$ plots (such as Plot 17) adapt dynamically via `_r_squared_axis_limits` to accommodate finite negative $R^2$ values without clipping.
- **Zero Reference Line**: A horizontal reference line is plotted at $R^2 = 0$.
- **Dynamic Text Annotation**: Value labels are positioned above positive bars with `va="bottom"` and below negative bars with `va="top"`.

### Workflow-Specific Plot Filenames, Titles & Manifests
- **Common Plots**: Plots 01 through 03 remain workflow-neutral (`01_original_data.png`, `02_data_splits.png`, `03_split_sizes.png`) and report `"workflow": "common"` in manifests.
- **Workflow Tokens in Filenames**: All workflow-specific plots (04–20) incorporate normalized lowercase workflow tokens into their filenames (e.g., `13_holdout_final_polynomial_fit.png`, `13_kfold_final_polynomial_fit.png`).
- **Clear Display Titles**: Plot titles explicitly display the human-readable workflow name `Holdout` or `K-Fold` (e.g., `13 Holdout: Final Polynomial Fit`, `13 K-Fold: Final Polynomial Fit`).

### Bootstrap Fitted-Curve Uncertainty Band
- **Interpretation**: The bootstrap band (`13b_<workflow>_fitted_curve_uncertainty_band.png`) represents variation in the fitted polynomial curve across case-resampled development datasets.
- **Statistical Distinction**: The band does not include new observation noise and must not be interpreted as a prediction interval for future individual observations.
- **Manifest Metadata**: Plot manifest entries record percentile bounds (`lower_percentile`: 2.5, `upper_percentile`: 97.5) and `interval_interpretation`: `"fitted_curve_uncertainty"`.

### K-Fold Binary Membership Matrix (Plot 08)
- **Matrix Representation**: Plot 08 (`08_kfold_assignments.png`) displays a binary training-validation membership matrix of shape $(K, N_{\text{dev}})$ using `imshow()`.
- **Color Category Mapping**: Light blue (`#BBD7E8`) represents Training membership (0), and orange (`#E69F00`) represents Validation membership (1).
- **Validation Invariants**: Each development sample is assigned to validation in exactly 1 fold and to training in the remaining $K - 1$ folds.
- **Axis Semantics**: The X-axis represents zero-based development set sample positions ($0$ to $N_{\text{dev}}-1$), and the Y-axis represents cross-validation folds (`Fold 1` at top to `Fold K`).

---

## Installation & Environment Setup

### Requirements
The project relies solely on standard Python libraries plus NumPy and Matplotlib:

```
numpy>=1.22.0
matplotlib>=3.5.0
```

### Installation
Clone the repository and install requirements:

```bash
pip install -r requirements.txt
```

---

## Execution & Command-Line Usage

### Running Synthetic Data Generation
Generate synthetic data with optional polynomial coefficients and Gaussian noise:

```bash
python generate_data.py \
    --output polynomial_data.csv \
    --num-samples 200 \
    --x-min -10.0 \
    --x-max 10.0 \
    --coefficients 1.0 -2.0 0.5 \
    --noise-std 5.0 \
    --seed 42
```

### Running the Main Pipeline
Run the pipeline in `holdout`, `kfold`, or `both` mode:

```bash
python main.py polynomial_data.csv \
    --x-column X \
    --y-column Y \
    --train-ratio 0.70 \
    --validation-ratio 0.15 \
    --test-ratio 0.15 \
    --degrees 1 2 3 4 5 6 7 8 9 10 \
    --max-degree 50 \
    --l2-values 0 0.0001 0.01 0.1 1 10 \
    --folds 5 \
    --seed 42 \
    --scale-features \
    --mode both \
    --plot-format png \
    --plot-dpi 150 \
    --output-dir results
```

---

## Summary of Generated Output Files

Numerical results and visual artifacts are written to `--output-dir` in workflow-isolated subdirectories:

### Output Layout (`--mode both`)

```text
<output-dir>/
├── common/
│   ├── run_metadata.json
│   ├── split_summary.json
│   ├── 01_original_data.<format>
│   ├── 02_data_splits.<format>
│   ├── 03_split_sizes.<format>
│   └── plot_manifest.json
├── holdout/
│   ├── holdout_results.csv
│   ├── final_model.json
│   ├── test_predictions.csv
│   ├── plot_manifest.json
│   └── workflow-specific plots
├── kfold/
│   ├── kfold_fold_results.csv
│   ├── kfold_summary_results.csv
│   ├── final_model.json
│   ├── test_predictions.csv
│   ├── out_of_fold_predictions.csv
│   ├── plot_manifest.json
│   └── workflow-specific plots
└── comparison.json
```

- In `--mode holdout`, only `common/` and `holdout/` are created.
- In `--mode kfold`, only `common/` and `kfold/` are created.
- `comparison.json` is generated only when `--mode both` is selected. Neither workflow selection nor hyperparameter choice uses test-set metrics.

### Reproducibility Metadata Schema (`common/run_metadata.json`)
The `run_metadata.json` file records environment versions, command-line arguments, search grids, seeds, and an input CSV fingerprint:

```json
{
  "schema_version": 1,
  "command": {
    "argv": [
      "main.py",
      "polynomial_data.csv",
      "--mode",
      "both"
    ],
    "arguments": {
      "csv_file": "polynomial_data.csv",
      "x_column": "X",
      "y_column": "Y",
      "train_ratio": 0.7,
      "validation_ratio": 0.15,
      "test_ratio": 0.15,
      "seed": 42,
      "folds": 5,
      "degrees": [1, 2, 3],
      "l2_values": [0.0, 0.01, 1.0],
      "scale_features": true,
      "mode": "both"
    }
  },
  "runtime": {
    "python_version": "3.12.13",
    "python_implementation": "CPython",
    "platform": "Linux-..."
  },
  "packages": {
    "numpy": "2.5.3",
    "matplotlib": "3.11.2"
  },
  "search_grid": {
    "requested": {
      "degrees": [3, 1, 2, 3],
      "l2_values": [1.0, 0.0, 0.01, 1.0]
    },
    "effective": {
      "degrees": [1, 2, 3],
      "l2_values": [0.0, 0.01, 1.0]
    },
    "canonicalization": {
      "duplicates_removed": true,
      "ordering": "ascending",
      "float_deduplication": "exact_equality"
    }
  },
  "random_seeds": {
    "split_seed": 42,
    "kfold_seed": 42,
    "bootstrap_seed": 123
  },
  "input_file": {
    "path": "polynomial_data.csv",
    "size_bytes": 12345,
    "sha256": "3a8b..."
  }
}
```

### Final Model Metadata Schema (`final_model.json`)
The `final_model.json` artifact includes full numerical details from the final refit:

```json
{
  "selected_workflow": "holdout",
  "selected_degree": 2,
  "selected_l2_lambda": 0.0,
  "coefficients_ordering_convention": "Ascending polynomial order: [beta_0, beta_1*x, beta_2*x^2, ...]",
  "coefficients_scaled_basis": [2.0, -1.0, 0.5],
  "coefficients_original_basis": [2.0, -1.0, 0.5],
  "scale_features_enabled": false,
  "scaling_parameters": null,
  "solver_used": "lstsq_ols",
  "condition_number": 234.56,
  "condition_warning": false,
  "rank": 3,
  "full_rank": true,
  "design_condition_number": 234.56,
  "solver_condition_number": 234.56,
  "selection_tolerances": {
    "rtol": 1e-07,
    "atol": 1e-12
  },
  "final_test_metrics": {
    "mse": 0.042,
    "rmse": 0.205,
    "mae": 0.162,
    "r_squared": 0.998,
    "adjusted_r_squared": 0.998
  },
  "run_metadata_file": "../common/run_metadata.json"
}
```
