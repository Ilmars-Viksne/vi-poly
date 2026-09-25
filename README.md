# Polynomial Regression with NumPy and Matplotlib

A complete, maintainable, modular Python framework for polynomial regression analysis, model selection, hyperparameter tuning (None, L1 Lasso, and L2 Ridge regularization; polynomial degree), and scientific visualization built strictly using NumPy, Matplotlib, and the Python Standard Library.

---

## Conceptual Foundations & Overview

### Train, Validation, and Test Sets
When fitting predictive models, evaluating performance on the data used for training leads to overly optimistic performance estimates due to overfitting. To reliably estimate real-world generalization:
- **Training Set (70% default)**: Used strictly to solve for model parameters ($\beta$ coefficients) and fit feature scaling parameters ($\mu, \sigma$).
- **Validation Set (15% default)**: Used to tune hyperparameters (polynomial degree $d$, regularization family, and regularization strength) and compare competing candidate models.
- **Test Set (15% default)**: Reserved and kept **completely untouched** during feature scaling and hyperparameter selection. Evaluated exactly once at the end to provide an unbiased estimate of generalization error.

### Why K-Fold Cross-Validation Still Requires a Test Set
K-Fold cross-validation provides a robust out-of-fold estimate of validation error on development data. However, using cross-validation performance to select the best hyperparameter combination introduces a subtle form of optimization bias ("selection bias"). Therefore, reserving a separate test set remains essential to verify that the final selected model generalizes well to unseen data.

---

## Mathematical Formulation & Solvers

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

#### Z-Score Feature Scaling (Opt-In & Recommended for L1)
When `--scale-features` is enabled, every non-intercept column $p \ge 1$ is standardized:

$$z_{i, p} = \frac{x_i^p - \mu_p}{\sigma_p}$$

The intercept column $x^0 = 1$ is left unscaled. Standardizing polynomial terms prevents ill-conditioning of the Vandermonde matrix as degree $d$ increases and ensures polynomial terms are penalized on comparable scales. **`--scale-features` is strongly recommended when using L1 regularization.** Note that feature scaling is applied **after** calculating polynomial powers.

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

## Regularization Objectives & Solvers

The user can select from three regularization strategies via `--regularization`:
- `none`: Unregularized Ordinary Least Squares (OLS)
- `l1`: L1 Regularization (Lasso)
- `l2`: L2 Regularization (Ridge, default)

### 1. Ordinary Least Squares (OLS)
For `regularization="none"` or zero strength ($\lambda = 0.0$):

$$\min_\beta \frac{1}{2}\|X\beta - y\|_2^2$$

Solved directly using singular value decomposition or QR factorization via:

```python
np.linalg.lstsq(X, y, rcond=None)
```

- **Solver Label**: `lstsq_ols`
- **Normal Equations**: Not formed or solved.
- **Rank Deficient Handling**: `np.linalg.lstsq` returns the minimum-norm solution for rank-deficient systems and issues a `UserWarning`.

### 2. L2 Regularization (Ridge) via Augmented System
For `regularization="l2"` and strength $\lambda > 0.0$:

$$\min_\beta \frac{1}{2}\|X\beta - y\|_2^2 + \frac{\lambda}{2}\sum_{j=1}^p \beta_j^2$$

**The intercept $\beta_0$ is explicitly excluded from regularization.** Solved using an augmented least-squares formulation:

$$\mathbf{X}_{\text{aug}} = \begin{bmatrix} \mathbf{X} \\ \sqrt{\lambda} \mathbf{L} \end{bmatrix}, \quad \mathbf{y}_{\text{aug}} = \begin{bmatrix} \mathbf{y} \\ \mathbf{0}_{p+1} \end{bmatrix}$$

$$\boldsymbol{\beta} = \text{np.linalg.lstsq}(\mathbf{X}_{\text{aug}}, \mathbf{y}_{\text{aug}}, \text{rcond}=\text{None})[0]$$

where $\mathbf{L} = \text{diag}(0, 1, 1, \dots, 1)$.

- **Solver Label**: `lstsq_augmented_ridge`

### 3. L1 Regularization (Lasso) via Coordinate Descent
For `regularization="l1"` and strength $\lambda > 0.0$:

$$\min_\beta \frac{1}{2}\|X\beta - y\|_2^2 + \lambda \sum_{j=1}^p |\beta_j|$$

**The intercept $\beta_0$ is explicitly excluded from penalization.** Solved using deterministic cyclic coordinate descent with soft thresholding:

For penalized coefficients $j \ge 1$:

$$r_j = y - X\beta + X_j \beta_j, \quad \rho_j = X_j^T r_j, \quad \beta_j = \frac{S(\rho_j, \lambda)}{X_j^T X_j}$$

where $S(v, t) = \text{sign}(v) \max(|v| - t, 0)$ is the soft-thresholding operator. L1 regularization can drive penalized coefficients exactly to zero, performing implicit feature selection.

For the unpenalized intercept $j = 0$:

$$\beta_0 = \frac{X_0^T r_0}{X_0^T X_0}$$

- **Solver Label**: `coordinate_descent_l1`
- **Convergence Controls**: Configurable via `--l1-max-iterations` (default: 10,000) and `--l1-tolerance` (default: 1e-8).
- **Initialization**: Configurable via `--l1-initialization` (`zeros` default, or `ols`).
- **Non-Convergence Policy**: If coordinate descent reaches `--l1-max-iterations` without satisfying the tolerance, `converged=False` is set and a `UserWarning` is issued. Non-converged candidates are excluded from selection when converged candidates exist. If all candidates fail to converge, a `RuntimeError` is raised.

---

## Condition Number & Diagnostics

`ModelFitDetails` records numerical diagnostics for the final model refit:
- `design_condition_number`: Condition number of the original polynomial design matrix $\mathbf{X}$.
- `solver_condition_number`: Condition number of the matrix passed to the linear solver. For L1 models, this reflects the design matrix condition number.
- `requested_regularization` & `effective_regularization`: Requested family vs effective execution family (`none` when strength is 0.0).
- `iterations`, `converged`, `final_objective`, `max_coefficient_change`: L1 coordinate-descent solver diagnostics.
- `nonzero_coefficient_count`: Count of non-zero penalized coefficients ($j \ge 1$).

---

## Input Validation & CLI Arguments

### Command-Line Arguments
- **Regularization Family (`--regularization`)**: Choice of `none`, `l1`, or `l2` (default: `l2`).
- **Regularization Strengths (`--regularization-values`)**: Non-empty list of finite, non-negative floats ($\ge 0.0$). Default: `[0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]`. For `--regularization none`, strength must be `[0.0]`.
- **Deprecated L2 Option (`--l2-values`)**: Retained for backward compatibility. Emits a `DeprecationWarning` and maps to `--regularization l2`. Cannot be combined with `--regularization-values` or non-L2 regularization.
- **L1 Solver Controls**:
  - `--l1-max-iterations`: Positive integer $\ge 1$ (default: 10000).
  - `--l1-tolerance`: Finite, strictly positive float $> 0.0$ (default: 1e-8).
  - `--l1-initialization`: Choice of `zeros` or `ols` (default: `zeros`).
- **Degrees (`--degrees`)**: Non-empty list of non-negative integers satisfying $0 \le d \le \text{max\_degree}$.
- **Maximum Degree (`--max-degree`)**: Non-negative integer (default: 50).
- **Fold Count (`--folds`)**: Integer $\ge 2$.
- **Split Ratios (`--train-ratio`, `--validation-ratio`, `--test-ratio`)**: Summing to 1.0.

---

## CSV Data Traceability & Provenance System

The pipeline enforces complete observation traceability from raw CSV input through split summaries and prediction outputs using three distinct index views:

1. **`loaded_array_index`**: Zero-based array index in filtered loaded arrays.
2. **`observation_index`**: Zero-based record counter among all CSV data records following the header.
3. **`csv_line_number`**: One-based physical line number in the CSV file (from `reader.line_num`).

---

## Scientific Visualizations & Manifests

All plot filenames and titles incorporate workflow tokens (`common`, `holdout`, `kfold`) and generic regularization labels (`OLS`, `L1 = ...`, `L2 = ...`).

Generated plots include:
- `01_original_data`: Original scatter dataset.
- `02_data_splits`: Train / Validation / Test partitions.
- `03_split_sizes`: Bar chart of sample allocations.
- `04_<workflow>_validation_rmse`: Validation RMSE across degrees and strengths.
- `05_<workflow>_train_validation_error`: Bias-variance trade-off curves.
- `06_<workflow>_rmse_heatmap`: Validation RMSE heatmap.
- `07_<workflow>_candidate_models`: Candidate polynomial fits overlaid on data.
- `08_kfold_assignments`: Binary membership matrix visualization.
- `09_kfold_mean_rmse`: Cross-validation mean RMSE with $\pm 1$ std error bars.
- `10_kfold_rmse_heatmap`: K-Fold validation mean RMSE heatmap.
- `11_kfold_fold_metrics`: Fold-by-fold RMSE performance.
- `12_kfold_out_of_fold_predictions`: Out-of-fold actual vs predicted scatter plot.
- `13_<workflow>_final_polynomial_fit`: Final refitted curve on development data.
- `13b_<workflow>_fitted_curve_uncertainty_band`: 95% bootstrap uncertainty band.
- `14_<workflow>_test_actual_vs_predicted`: Test set actual vs predicted scatter plot.
- `15_<workflow>_test_residuals`: Test residuals vs predicted values.
- `16_<workflow>_test_residual_histogram`: Test residual distribution.
- `17_<workflow>_final_metrics`: Final test metrics summary bar chart.
- `18_<workflow>_model_coefficients`: Fitted polynomial coefficients in original basis (highlights zero coefficients for L1).
- `19_<workflow>_condition_numbers`: Design matrix condition number vs degree.
- `20_<workflow>_results_dashboard`: Combined 4-panel results dashboard.

---

## Execution & Command-Line Usage

### Running Synthetic Data Generation
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

### Running Pipeline with L1 Regularization
```bash
python main.py polynomial_data.csv \
    --mode both \
    --regularization l1 \
    --regularization-values 0.0 0.001 0.01 0.1 1.0 \
    --l1-max-iterations 10000 \
    --l1-tolerance 1e-8 \
    --scale-features \
    --output-dir results_l1
```

### Running Pipeline with L2 Regularization
```bash
python main.py polynomial_data.csv \
    --mode both \
    --regularization l2 \
    --regularization-values 0.0 0.001 0.01 0.1 1.0 \
    --scale-features \
    --output-dir results_l2
```

---

## Summary of Generated Output Files

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

### Final Model Metadata Schema (`final_model.json`)
```json
{
  "selected_workflow": "holdout",
  "selected_degree": 2,
  "selected_regularization": "l1",
  "selected_regularization_strength": 0.01,
  "requested_regularization": "l1",
  "effective_regularization": "l1",
  "coefficients_ordering_convention": "Ascending polynomial order: [beta_0, beta_1*x, beta_2*x^2, ...]",
  "coefficients_scaled_basis": [2.0, -1.0, 0.0],
  "coefficients_original_basis": [2.0, -0.5, 0.0],
  "scale_features_enabled": true,
  "scaling_parameters": {
    "means": [0.0, 2.5],
    "stds": [1.0, 2.0]
  },
  "solver_used": "coordinate_descent_l1",
  "converged": true,
  "iterations": 142,
  "final_objective": 12.345,
  "max_coefficient_change": 8.2e-9,
  "nonzero_coefficient_count": 1,
  "l1_initialization": "zeros",
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
