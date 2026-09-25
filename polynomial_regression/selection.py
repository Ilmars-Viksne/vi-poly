"""Model selection logic for Holdout and K-Fold cross-validation workflows."""

from dataclasses import dataclass

import numpy as np

from .features import DEFAULT_MAX_DEGREE, PolynomialFeatureTransformer
from .metrics import EvaluationMetrics, RegressionMetrics
from .regression import (
    DEFAULT_L1_INITIALIZATION,
    DEFAULT_L1_MAX_ITERATIONS,
    DEFAULT_L1_TOLERANCE,
    DEFAULT_REGULARIZATION_STRENGTHS,
    REGULARIZATION_L2,
    REGULARIZATION_NONE,
    SUPPORTED_REGULARIZATIONS,
    ModelFitDetails,
    PolynomialRegressor,
)
from .splitting import KFoldSplitter


@dataclass
class ModelCandidateResult:
    """Evaluation result for a specific degree and regularization configuration."""

    degree: int
    regularization: str
    regularization_strength: float
    train_metrics: EvaluationMetrics
    val_metrics: EvaluationMetrics
    condition_number: float
    beta_scaled: np.ndarray
    beta_orig: np.ndarray
    transformer: PolynomialFeatureTransformer
    solver_used: str
    converged: bool | None
    iterations: int | None
    nonzero_coefficient_count: int

    @property
    def l2_lambda(self) -> float:
        if self.regularization != REGULARIZATION_L2:
            raise AttributeError("l2_lambda is available only for L2 candidates.")
        return self.regularization_strength


@dataclass
class HoldoutSelectionResult:
    """Summary result from Holdout model selection."""

    candidates: list[ModelCandidateResult]
    best_candidate: ModelCandidateResult
    final_refitted_beta_scaled: np.ndarray
    final_refitted_beta_orig: np.ndarray
    final_refitted_transformer: PolynomialFeatureTransformer
    final_fit_details: ModelFitDetails
    final_test_metrics: EvaluationMetrics
    test_predictions: np.ndarray
    test_residuals: np.ndarray


@dataclass
class FoldMetricsResult:
    """Metrics for a single fold in K-Fold CV."""

    fold_index: int
    train_metrics: EvaluationMetrics
    val_metrics: EvaluationMetrics
    condition_number: float
    converged: bool | None
    iterations: int | None
    nonzero_coefficient_count: int


@dataclass
class KFoldCandidateResult:
    """Aggregate K-Fold CV evaluation result for a hyperparameter combination."""

    degree: int
    regularization: str
    regularization_strength: float
    fold_results: list[FoldMetricsResult]
    mean_val_metrics: EvaluationMetrics
    std_val_metrics: EvaluationMetrics
    mean_condition_number: float
    oof_predictions: np.ndarray  # Out-of-fold predictions in dev_indices order
    converged_fold_count: int
    mean_iterations: float | None
    max_iterations: int | None
    mean_nonzero_coefficient_count: float
    converged: bool

    @property
    def l2_lambda(self) -> float:
        if self.regularization != REGULARIZATION_L2:
            raise AttributeError("l2_lambda is available only for L2 candidates.")
        return self.regularization_strength


@dataclass
class KFoldSelectionResult:
    """Summary result from K-Fold model selection."""

    candidates: list[KFoldCandidateResult]
    best_candidate: KFoldCandidateResult
    final_refitted_beta_scaled: np.ndarray
    final_refitted_beta_orig: np.ndarray
    final_refitted_transformer: PolynomialFeatureTransformer
    final_fit_details: ModelFitDetails
    final_test_metrics: EvaluationMetrics
    test_predictions: np.ndarray
    test_residuals: np.ndarray
    oof_predictions: np.ndarray
    oof_residuals: np.ndarray


def _validate_hyperparameters(
    degrees: list[int],
    regularization_strengths: list[float],
    max_degree: int,
) -> tuple[list[int], list[float]]:
    if not degrees:
        raise ValueError("Candidate degrees list must not be empty.")
    if not regularization_strengths:
        raise ValueError("Candidate regularization strengths list must not be empty.")

    validated_degrees = []
    for deg in degrees:
        if not isinstance(deg, (int, np.integer)):
            raise TypeError(f"Degree values must be integers, got {type(deg)}.")
        d_int = int(deg)
        if d_int < 0 or d_int > max_degree:
            raise ValueError(
                f"Polynomial degree {d_int} is outside allowed range [0, {max_degree}]."
            )
        validated_degrees.append(d_int)

    validated_strengths = []
    for str_val in regularization_strengths:
        try:
            val = float(str_val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Regularization strength must be numeric, got {str_val}."
            ) from exc
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(
                f"Regularization strength must be finite and non-negative, got {val}."
            )
        validated_strengths.append(val)

    return sorted(set(validated_degrees)), sorted(set(validated_strengths))


def _validate_index_array(idx: np.ndarray, name: str, n_all: int) -> np.ndarray:
    try:
        arr = np.asarray(idx, dtype=np.int64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Index array '{name}' must be integer array.") from exc

    if arr.ndim != 1:
        raise ValueError(f"Index array '{name}' must be 1D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"Index array '{name}' must not be empty.")
    if np.any(arr < 0) or np.any(arr >= n_all):
        raise ValueError(
            f"Index array '{name}' contains out-of-bounds indices for sample size {n_all}."
        )
    if len(set(arr)) != len(arr):
        raise ValueError(f"Index array '{name}' contains duplicate indices.")
    return arr


class HoldoutModelSelector:
    """Evaluates candidate models on holdout train/val sets and refits on dev set."""

    def __init__(
        self,
        degrees: list[int],
        regularization: str = REGULARIZATION_L2,
        regularization_strengths: list[float] | None = None,
        l2_lambdas: list[float] | None = None,
        l1_max_iterations: int = DEFAULT_L1_MAX_ITERATIONS,
        l1_tolerance: float = DEFAULT_L1_TOLERANCE,
        l1_initialization: str = DEFAULT_L1_INITIALIZATION,
        scale_features: bool = False,
        selection_rtol: float = 1e-7,
        selection_atol: float = 1e-12,
        condition_warning_threshold: float = 1e12,
        max_degree: int = DEFAULT_MAX_DEGREE,
    ):
        if l2_lambdas is not None and regularization_strengths is not None:
            raise ValueError("Cannot specify both l2_lambdas and regularization_strengths.")

        strengths = (
            l2_lambdas
            if l2_lambdas is not None
            else (
                regularization_strengths
                if regularization_strengths is not None
                else DEFAULT_REGULARIZATION_STRENGTHS
            )
        )

        norm_reg = regularization.strip().lower() if isinstance(regularization, str) else regularization
        if norm_reg not in SUPPORTED_REGULARIZATIONS:
            raise ValueError(
                f"Unsupported regularization type '{regularization}'. Supported choices: {sorted(SUPPORTED_REGULARIZATIONS)}."
            )

        if norm_reg == REGULARIZATION_NONE:
            strengths = [0.0]

        self.max_degree = int(max_degree)
        self.regularization = norm_reg
        self.degrees, self.regularization_strengths = _validate_hyperparameters(
            degrees, strengths, self.max_degree
        )
        self.l1_max_iterations = int(l1_max_iterations)
        self.l1_tolerance = float(l1_tolerance)
        self.l1_initialization = l1_initialization
        self.scale_features = scale_features
        self.selection_rtol = float(selection_rtol)
        self.selection_atol = float(selection_atol)
        self.condition_warning_threshold = float(condition_warning_threshold)

    @property
    def l2_lambdas(self) -> list[float]:
        return self.regularization_strengths

    def select(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        dev_idx: np.ndarray,
    ) -> HoldoutSelectionResult:
        n_all = len(x_all)
        tr_idx = _validate_index_array(train_idx, "train_idx", n_all)
        v_idx = _validate_index_array(val_idx, "val_idx", n_all)
        te_idx = _validate_index_array(test_idx, "test_idx", n_all)
        d_idx = _validate_index_array(dev_idx, "dev_idx", n_all)

        if len(set(tr_idx).intersection(set(v_idx))) > 0:
            raise ValueError("train_idx and val_idx must not overlap.")
        if len(set(tr_idx).intersection(set(te_idx))) > 0:
            raise ValueError("train_idx and test_idx must not overlap.")
        if len(set(v_idx).intersection(set(te_idx))) > 0:
            raise ValueError("val_idx and test_idx must not overlap.")

        x_train, y_train = x_all[tr_idx], y_all[tr_idx]
        x_val, y_val = x_all[v_idx], y_all[v_idx]
        x_test, y_test = x_all[te_idx], y_all[te_idx]
        x_dev, y_dev = x_all[d_idx], y_all[d_idx]

        candidates: list[ModelCandidateResult] = []

        for deg in self.degrees:
            for strength in self.regularization_strengths:
                transformer = PolynomialFeatureTransformer(
                    degree=deg,
                    scale_features=self.scale_features,
                    max_degree=self.max_degree,
                )
                X_train = transformer.fit_transform(x_train)
                X_val = transformer.transform(x_val)

                regressor = PolynomialRegressor(
                    regularization=self.regularization,
                    regularization_strength=strength,
                    l1_max_iterations=self.l1_max_iterations,
                    l1_tolerance=self.l1_tolerance,
                    l1_initialization=self.l1_initialization,
                    condition_warning_threshold=self.condition_warning_threshold,
                ).fit(X_train, y_train)

                beta_scaled = regressor.beta
                beta_orig = transformer.convert_coefficients_to_original_basis(
                    beta_scaled
                )

                pred_train = regressor.predict(X_train)
                pred_val = regressor.predict(X_val)

                train_metrics = RegressionMetrics.calculate(
                    y_train, pred_train, num_predictors=deg
                )
                val_metrics = RegressionMetrics.calculate(
                    y_val, pred_val, num_predictors=deg
                )

                details = regressor.fit_details

                candidates.append(
                    ModelCandidateResult(
                        degree=deg,
                        regularization=self.regularization,
                        regularization_strength=strength,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        condition_number=details.condition_number,
                        beta_scaled=beta_scaled,
                        beta_orig=beta_orig,
                        transformer=transformer,
                        solver_used=details.solver_used,
                        converged=details.converged,
                        iterations=details.iterations,
                        nonzero_coefficient_count=details.nonzero_coefficient_count,
                    )
                )

        best_candidate = self._select_best_candidate(candidates)

        refitted_transformer = PolynomialFeatureTransformer(
            degree=best_candidate.degree,
            scale_features=self.scale_features,
            max_degree=self.max_degree,
        )
        X_dev = refitted_transformer.fit_transform(x_dev)
        refitted_regressor = PolynomialRegressor(
            regularization=best_candidate.regularization,
            regularization_strength=best_candidate.regularization_strength,
            l1_max_iterations=self.l1_max_iterations,
            l1_tolerance=self.l1_tolerance,
            l1_initialization=self.l1_initialization,
            condition_warning_threshold=self.condition_warning_threshold,
        ).fit(X_dev, y_dev)

        if refitted_regressor.fit_details.converged is False:
            raise RuntimeError(
                "Final refitted model on development set failed to converge."
            )

        final_beta_scaled = refitted_regressor.beta
        final_beta_orig = refitted_transformer.convert_coefficients_to_original_basis(
            final_beta_scaled
        )

        PolynomialFeatureTransformer.verify_coefficient_conversion(
            x_dev, refitted_transformer, final_beta_scaled, final_beta_orig
        )

        X_test = refitted_transformer.transform(x_test)
        test_pred = refitted_regressor.predict(X_test)
        test_metrics = RegressionMetrics.calculate(
            y_test, test_pred, num_predictors=best_candidate.degree
        )
        test_residuals = y_test - test_pred

        return HoldoutSelectionResult(
            candidates=candidates,
            best_candidate=best_candidate,
            final_refitted_beta_scaled=final_beta_scaled,
            final_refitted_beta_orig=final_beta_orig,
            final_refitted_transformer=refitted_transformer,
            final_fit_details=refitted_regressor.fit_details,
            final_test_metrics=test_metrics,
            test_predictions=test_pred,
            test_residuals=test_residuals,
        )

    def _select_best_candidate(
        self, candidates: list[ModelCandidateResult]
    ) -> ModelCandidateResult:
        converged_candidates = [c for c in candidates if c.converged is not False]
        if not converged_candidates:
            raise RuntimeError(
                f"No L1 candidate converged within {self.l1_max_iterations} coordinate-descent iterations. "
                f"Evaluated {len(candidates)} candidates across {len(self.degrees)} degrees and {len(self.regularization_strengths)} regularization strengths. "
                "Increase --l1-max-iterations, relax --l1-tolerance, enable --scale-features, or revise the search grid."
            )

        best = converged_candidates[0]
        for candidate in converged_candidates[1:]:
            cand_rmse = candidate.val_metrics.rmse
            best_rmse = best.val_metrics.rmse

            if np.isclose(
                cand_rmse, best_rmse, rtol=self.selection_rtol, atol=self.selection_atol
            ):
                if candidate.degree < best.degree or (
                    candidate.degree == best.degree
                    and candidate.regularization_strength > best.regularization_strength
                ):
                    best = candidate
            elif cand_rmse < best_rmse:
                best = candidate

        return best


class KFoldModelSelector:
    """Evaluates candidate models using K-Fold cross validation on development data."""

    def __init__(
        self,
        degrees: list[int],
        regularization: str = REGULARIZATION_L2,
        regularization_strengths: list[float] | None = None,
        l2_lambdas: list[float] | None = None,
        l1_max_iterations: int = DEFAULT_L1_MAX_ITERATIONS,
        l1_tolerance: float = DEFAULT_L1_TOLERANCE,
        l1_initialization: str = DEFAULT_L1_INITIALIZATION,
        k_folds: int = 5,
        seed: int = 42,
        scale_features: bool = False,
        selection_rtol: float = 1e-7,
        selection_atol: float = 1e-12,
        condition_warning_threshold: float = 1e12,
        max_degree: int = DEFAULT_MAX_DEGREE,
    ):
        if l2_lambdas is not None and regularization_strengths is not None:
            raise ValueError("Cannot specify both l2_lambdas and regularization_strengths.")

        strengths = (
            l2_lambdas
            if l2_lambdas is not None
            else (
                regularization_strengths
                if regularization_strengths is not None
                else DEFAULT_REGULARIZATION_STRENGTHS
            )
        )

        norm_reg = regularization.strip().lower() if isinstance(regularization, str) else regularization
        if norm_reg not in SUPPORTED_REGULARIZATIONS:
            raise ValueError(
                f"Unsupported regularization type '{regularization}'. Supported choices: {sorted(SUPPORTED_REGULARIZATIONS)}."
            )

        if norm_reg == REGULARIZATION_NONE:
            strengths = [0.0]

        self.max_degree = int(max_degree)
        self.regularization = norm_reg
        self.degrees, self.regularization_strengths = _validate_hyperparameters(
            degrees, strengths, self.max_degree
        )
        self.l1_max_iterations = int(l1_max_iterations)
        self.l1_tolerance = float(l1_tolerance)
        self.l1_initialization = l1_initialization
        self.k_folds = int(k_folds)
        self.seed = int(seed)
        self.scale_features = scale_features
        self.selection_rtol = float(selection_rtol)
        self.selection_atol = float(selection_atol)
        self.condition_warning_threshold = float(condition_warning_threshold)

    @property
    def l2_lambdas(self) -> list[float]:
        return self.regularization_strengths

    def select(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        dev_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> KFoldSelectionResult:
        n_all = len(x_all)
        d_idx = _validate_index_array(dev_idx, "dev_idx", n_all)
        te_idx = _validate_index_array(test_idx, "test_idx", n_all)

        if len(set(d_idx).intersection(set(te_idx))) > 0:
            raise ValueError("dev_idx and test_idx must not overlap.")

        x_dev, y_dev = x_all[d_idx], y_all[d_idx]
        x_test, y_test = x_all[te_idx], y_all[te_idx]

        splitter = KFoldSplitter(k=self.k_folds, seed=self.seed)
        folds = splitter.split(np.arange(len(d_idx)))

        candidates: list[KFoldCandidateResult] = []

        for deg in self.degrees:
            for strength in self.regularization_strengths:
                fold_results: list[FoldMetricsResult] = []
                oof_preds = np.zeros(len(d_idx), dtype=np.float64)

                fold_mses, fold_rmses, fold_maes, fold_r2s = [], [], [], []
                cond_nums = []
                fold_converged_list = []
                fold_iters_list = []
                fold_nonzeros_list = []

                for fold in folds:
                    f_train_idx = fold.train_indices
                    f_val_idx = fold.val_indices

                    xf_train, yf_train = x_dev[f_train_idx], y_dev[f_train_idx]
                    xf_val, yf_val = x_dev[f_val_idx], y_dev[f_val_idx]

                    transformer = PolynomialFeatureTransformer(
                        degree=deg,
                        scale_features=self.scale_features,
                        max_degree=self.max_degree,
                    )
                    Xf_train = transformer.fit_transform(xf_train)
                    Xf_val = transformer.transform(xf_val)

                    regressor = PolynomialRegressor(
                        regularization=self.regularization,
                        regularization_strength=strength,
                        l1_max_iterations=self.l1_max_iterations,
                        l1_tolerance=self.l1_tolerance,
                        l1_initialization=self.l1_initialization,
                        condition_warning_threshold=self.condition_warning_threshold,
                    ).fit(Xf_train, yf_train)

                    pred_train = regressor.predict(Xf_train)
                    pred_val = regressor.predict(Xf_val)

                    oof_preds[f_val_idx] = pred_val

                    f_train_metrics = RegressionMetrics.calculate(
                        yf_train, pred_train, num_predictors=deg
                    )
                    f_val_metrics = RegressionMetrics.calculate(
                        yf_val, pred_val, num_predictors=deg
                    )

                    details = regressor.fit_details
                    cond_nums.append(details.condition_number)
                    fold_converged_list.append(details.converged)
                    if details.iterations is not None:
                        fold_iters_list.append(details.iterations)
                    fold_nonzeros_list.append(details.nonzero_coefficient_count)

                    fold_results.append(
                        FoldMetricsResult(
                            fold_index=fold.fold_index,
                            train_metrics=f_train_metrics,
                            val_metrics=f_val_metrics,
                            condition_number=details.condition_number,
                            converged=details.converged,
                            iterations=details.iterations,
                            nonzero_coefficient_count=details.nonzero_coefficient_count,
                        )
                    )

                    fold_mses.append(f_val_metrics.mse)
                    fold_rmses.append(f_val_metrics.rmse)
                    fold_maes.append(f_val_metrics.mae)
                    fold_r2s.append(f_val_metrics.r_squared)

                mean_val_metrics = EvaluationMetrics(
                    mse=float(np.mean(fold_mses)),
                    rmse=float(np.mean(fold_rmses)),
                    mae=float(np.mean(fold_maes)),
                    r_squared=float(np.mean(fold_r2s)),
                )
                std_val_metrics = EvaluationMetrics(
                    mse=float(np.std(fold_mses, ddof=0)),
                    rmse=float(np.std(fold_rmses, ddof=0)),
                    mae=float(np.std(fold_maes, ddof=0)),
                    r_squared=float(np.std(fold_r2s, ddof=0)),
                )

                converged_fold_count = sum(1 for c in fold_converged_list if c is not False)
                cand_converged = converged_fold_count == len(folds)
                mean_iters = float(np.mean(fold_iters_list)) if fold_iters_list else None
                max_iters = max(fold_iters_list) if fold_iters_list else None
                mean_nonzeros = float(np.mean(fold_nonzeros_list))

                candidates.append(
                    KFoldCandidateResult(
                        degree=deg,
                        regularization=self.regularization,
                        regularization_strength=strength,
                        fold_results=fold_results,
                        mean_val_metrics=mean_val_metrics,
                        std_val_metrics=std_val_metrics,
                        mean_condition_number=float(np.mean(cond_nums)),
                        oof_predictions=oof_preds,
                        converged_fold_count=converged_fold_count,
                        mean_iterations=mean_iters,
                        max_iterations=max_iters,
                        mean_nonzero_coefficient_count=mean_nonzeros,
                        converged=cand_converged,
                    )
                )

        best_candidate = self._select_best_candidate(candidates)

        refitted_transformer = PolynomialFeatureTransformer(
            degree=best_candidate.degree,
            scale_features=self.scale_features,
            max_degree=self.max_degree,
        )
        X_dev = refitted_transformer.fit_transform(x_dev)
        refitted_regressor = PolynomialRegressor(
            regularization=best_candidate.regularization,
            regularization_strength=best_candidate.regularization_strength,
            l1_max_iterations=self.l1_max_iterations,
            l1_tolerance=self.l1_tolerance,
            l1_initialization=self.l1_initialization,
            condition_warning_threshold=self.condition_warning_threshold,
        ).fit(X_dev, y_dev)

        if refitted_regressor.fit_details.converged is False:
            raise RuntimeError(
                "Final refitted model on development set failed to converge."
            )

        final_beta_scaled = refitted_regressor.beta
        final_beta_orig = refitted_transformer.convert_coefficients_to_original_basis(
            final_beta_scaled
        )

        PolynomialFeatureTransformer.verify_coefficient_conversion(
            x_dev, refitted_transformer, final_beta_scaled, final_beta_orig
        )

        X_test = refitted_transformer.transform(x_test)
        test_pred = refitted_regressor.predict(X_test)
        test_metrics = RegressionMetrics.calculate(
            y_test, test_pred, num_predictors=best_candidate.degree
        )
        test_residuals = y_test - test_pred

        oof_residuals = y_dev - best_candidate.oof_predictions

        return KFoldSelectionResult(
            candidates=candidates,
            best_candidate=best_candidate,
            final_refitted_beta_scaled=final_beta_scaled,
            final_refitted_beta_orig=final_beta_orig,
            final_refitted_transformer=refitted_transformer,
            final_fit_details=refitted_regressor.fit_details,
            final_test_metrics=test_metrics,
            test_predictions=test_pred,
            test_residuals=test_residuals,
            oof_predictions=best_candidate.oof_predictions,
            oof_residuals=oof_residuals,
        )

    def _select_best_candidate(
        self, candidates: list[KFoldCandidateResult]
    ) -> KFoldCandidateResult:
        converged_candidates = [c for c in candidates if c.converged]
        if not converged_candidates:
            raise RuntimeError(
                f"No L1 candidate converged within {self.l1_max_iterations} coordinate-descent iterations. "
                f"Evaluated {len(candidates)} candidates across {len(self.degrees)} degrees and {len(self.regularization_strengths)} regularization strengths. "
                "Increase --l1-max-iterations, relax --l1-tolerance, enable --scale-features, or revise the search grid."
            )

        best = converged_candidates[0]
        for candidate in converged_candidates[1:]:
            cand_rmse = candidate.mean_val_metrics.rmse
            best_rmse = best.mean_val_metrics.rmse

            if np.isclose(
                cand_rmse, best_rmse, rtol=self.selection_rtol, atol=self.selection_atol
            ):
                if candidate.degree < best.degree or (
                    candidate.degree == best.degree
                    and candidate.regularization_strength > best.regularization_strength
                ):
                    best = candidate
            elif cand_rmse < best_rmse:
                best = candidate

        return best
