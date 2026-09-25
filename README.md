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

## Mathematical Formulation

### Polynomial Regression Equation
For a scalar feature $x$, polynomial regression models the target $y$ as:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d + \epsilon$$

where:
- $\beta_0$ is the intercept term.
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

The intercept column $x^0 = 1$ is left unscaled. Standardizing polynomial terms prevents ill-conditioning of the Vandermonde matrix as degree $d$ increases.

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

### L2 Regularization (Ridge) Objective
The coefficients $\boldsymbol{\beta}$ are obtained by solving the regularized normal equations:

$$(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{R}) \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$$

where $\mathbf{R} = \text{diag}(0, 1, 1, \dots, 1)$. **The intercept $\beta_0$ is unregularized** ($R_{00} = 0$) so that shifting the mean of $y$ does not distort penalty calculations.

---

## Residual Conventions & Diagnostics

- **Residual Definition**: Residual is defined consistently as $\text{residual} = y_{\text{actual}} - y_{\text{predicted}}$.
- **Diagnostic Usage**: Residual vs. Predicted plots and Residual Histograms serve as visual diagnostic tools to inspect homoscedasticity, non-linearity, and outlier influence. They assist in diagnosing model inadequacy rather than acting as definitive formal statistical tests.

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
    --l2-values 0 0.0001 0.01 0.1 1 10 \
    --folds 5 \
    --seed 42 \
    --scale-features \
    --mode both \
    --plot-format png \
    --plot-dpi 150 \
    --output-dir results
```

### Headless Execution
To run in a headless environment (e.g. CI/CD or server without display), use `--no-plots` or standard execution (which defaults to non-interactive Matplotlib `Agg` backend unless `--show-plots` is set):

```bash
python main.py polynomial_data.csv --mode both --output-dir results
```

---

## Summary of Generated Output Files

Numerical results and visual artifacts are written to `--output-dir` in workflow-isolated subdirectories:

### Output Layout (`--mode both`)

```text
<output-dir>/
├── common/
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
- `comparison.json` is generated only when `--mode both` is selected.

### Numerical Results & Manifests
1. `common/split_summary.json`: Split proportions, sample counts, row indices, fold memberships, and skipped row details.
2. `holdout/holdout_results.csv`: Per-candidate train and validation MSE, RMSE, MAE, R², and condition numbers.
3. `kfold/kfold_fold_results.csv`: Per-fold train and validation metrics across folds.
4. `kfold/kfold_summary_results.csv`: Mean and standard deviation of validation metrics across folds.
5. `<workflow>/final_model.json`: Selected degree, L2 strength, scaled & original-basis coefficients, scaling statistics, solver used, condition number, condition warning, and final test metrics.
6. `<workflow>/test_predictions.csv`: Original CSV row index, X, actual Y, predicted Y, and residual on untouched test set.
7. `kfold/out_of_fold_predictions.csv`: Cross-validated out-of-fold predictions for development data.
8. `<dir>/plot_manifest.json`: Manifest recording every figure generated in that directory.
9. `comparison.json`: Neutral comparison summary when running `--mode both`.

### Visual Outputs
- `common/01_original_data.png`: Input scatter plot before splitting.
- `common/02_data_splits.png`: Scatter plot showing train, validation, and test subsets.
- `common/03_split_sizes.png`: Bar chart summarizing subset sample allocation.
- `holdout/04_holdout_validation_rmse.png`: Validation RMSE curves across candidate degrees and L2 values.
- `holdout/05_train_validation_error.png`: Bias-variance trade-off curves for selected L2 value.
- `holdout/06_holdout_rmse_heatmap.png`: Validation RMSE heatmap.
- `holdout/07_candidate_models.png`: Fitted polynomial curves for representative degrees on training data.
- `kfold/08_kfold_assignments.png`: Validation fold membership matrix.
- `kfold/09_kfold_mean_rmse.png`: Cross-validation mean RMSE with error bars.
- `kfold/10_kfold_rmse_heatmap.png`: Cross-validation mean RMSE heatmap.
- `kfold/10b_kfold_rmse_std_heatmap.png`: Cross-validation RMSE standard deviation heatmap.
- `kfold/11_fold_metrics.png`: Per-fold validation RMSE bar chart.
- `kfold/12_out_of_fold_predictions.png`: Out-of-fold actual vs. predicted scatter plot.
- `<workflow>/13_final_polynomial_fit.png`: Final polynomial curve fitted on development data against test points.
- `<workflow>/13b_final_polynomial_bootstrap_band.png` (Optional): Bootstrap model-fit uncertainty band.
- `<workflow>/14_test_actual_vs_predicted.png`: Final test actual vs. predicted scatter plot.
- `<workflow>/15_test_residuals.png`: Residual vs. predicted diagnostic scatter plot.
- `<workflow>/16_test_residual_histogram.png`: Residual distribution histogram.
- `<workflow>/17_final_metrics.png`: Clean summary bar chart of final test performance metrics.
- `<workflow>/18_model_coefficients.png`: Fitted polynomial coefficients bar chart.
- `<workflow>/19_condition_numbers.png`: Design matrix condition number vs. degree.
- `<workflow>/20_results_dashboard.png`: Presentation-ready 4-panel summary dashboard.
