"""Main script for resolution time prediction using machine learning models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.models.baseline import evaluate_model, train_baseline_model
from src.models.feature_prep import prepare_ml_features
from src.models.lightgbm_model import (
    evaluate_lightgbm,
    get_lightgbm_feature_importance,
    train_lightgbm,
)
from src.models.extra_trees import (
    evaluate_extra_trees,
    get_extra_trees_feature_importance,
    train_extra_trees,
)
from src.models.random_forest import (
    evaluate_random_forest,
    get_feature_importance,
    train_random_forest,
)
from src.models.xgboost_model import evaluate_xgboost, get_xgboost_feature_importance, train_xgboost
from src.utils.logger import get_logger
from src.visualization.model_plots import (
    plot_feature_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
)

logger = get_logger(__name__)


def load_tuned_params(filename: str) -> dict[str, object] | None:
    """Load tuned hyperparameters from outputs/tuning if available."""
    path = Path("outputs/tuning") / filename
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        best_params = data.get("best_params")
        logger.info(f"Loaded tuned params from {path}: {best_params}")
        return best_params if isinstance(best_params, dict) else None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"Failed to load tuned params from {path}: {exc}")
        return None


def main() -> None:
    """Run resolution time prediction pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Resolution time prediction")
    parser.add_argument(
        "--sample",
        type=float,
        default=0.1,
        help="Sampling percentage for Random Forest (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration for LightGBM (requires GPU-enabled LightGBM)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lr,rf,lgbm,xgb,extra,ensemble",
        help="Comma-separated models to run (choices: lr,rf,lgbm,xgb,extra,ensemble). Default: all.",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RESOLUTION TIME PREDICTION - PHASE 2")
    logger.info("=" * 80)
    selected_models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    if not selected_models:
        logger.warning("No models selected via --models; defaulting to all.")
        selected_models = {"lr", "rf", "lgbm", "xgb", "extra", "ensemble"}

    logger.info(f"Random Forest sampling: {args.sample*100}%")
    logger.info(f"GPU acceleration: {'Enabled' if args.gpu else 'Disabled'}")
    logger.info(f"Models selected: {sorted(selected_models)}")

    # Create output directory
    output_dir = Path("outputs/figures/resolution_time")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    df = load_data()

    if df.empty:
        logger.error("No data loaded!")
        return

    df = preprocess_data(df)

    # Prepare features
    logger.info("Preparing features for machine learning")
    X_train, X_test, y_train, y_test, encoders = prepare_ml_features(df)
    feature_names = X_train.columns.tolist()

    # Load tuned hyperparameters if available
    # Random Forest: tuned params performed worse; keep tuned params empty for now.
    rf_tuned_params: dict[str, object] = {}
    lgbm_tuned_params = load_tuned_params("lightgbm_tuning.json") or {}
    xgb_tuned_params = load_tuned_params("xgboost_tuning.json") or {}
    extra_tuned_params = load_tuned_params("extra_trees_tuning.json") or {}

    if not rf_tuned_params:
        logger.info("No tuned Random Forest params found; using defaults.")
    if not lgbm_tuned_params:
        logger.info("No tuned LightGBM params found; using defaults.")
    if not xgb_tuned_params:
        logger.info("No tuned XGBoost params found; using defaults.")
    if not extra_tuned_params:
        logger.info("No tuned ExtraTrees params found; using defaults.")

    if args.gpu:
        # Ensure CPU-specific params from tuning do not override GPU usage if user opts in
        lgbm_tuned_params.pop("device", None)
        xgb_tuned_params.pop("device", None)

    # Default parameter baselines (overridable by tuned params)
    rf_base_params: dict[str, object] = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": 0,
    }
    lgbm_base_params: dict[str, object] = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    xgb_base_params: dict[str, object] = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }
    extra_base_params: dict[str, object] = {
        "n_estimators": 100,
        "max_depth": 20,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": 4,
        "bootstrap": True,
        "max_samples": 0.3,
    }

    rf_params = {**rf_base_params, **rf_tuned_params}
    lgbm_params = {**lgbm_base_params, **lgbm_tuned_params}
    xgb_params = {**xgb_base_params, **xgb_tuned_params}
    extra_params = {**extra_base_params, **extra_tuned_params}

    # Train and evaluate models
    results = {}

    lr_pred = None
    rf_pred = None
    lgbm_pred = None
    xgb_pred = None
    extra_pred = None

    # Model 1: Linear Regression Baseline
    if "lr" in selected_models:
        logger.info("=" * 80)
        logger.info("MODEL: Linear Regression Baseline")
        logger.info("=" * 80)
        lr_model = train_baseline_model(X_train, y_train)
        lr_mae, lr_r2, lr_pred = evaluate_model(lr_model, X_test, y_test)
        results["Linear Regression"] = {"mae": lr_mae, "r2": lr_r2}

    # Model 2: Random Forest (with sampling)
    if "rf" in selected_models or "ensemble" in selected_models:
        logger.info("=" * 80)
        logger.info("MODEL: Random Forest")
        logger.info("=" * 80)

        # Sample training data for Random Forest
        if args.sample < 1.0:
            logger.info(f"Sampling {args.sample*100}% of training data for Random Forest")
            X_train_rf = X_train.sample(frac=args.sample, random_state=42)
            y_train_rf = y_train.loc[X_train_rf.index]
            logger.info(f"RF training set: {len(X_train_rf)} samples")
        else:
            X_train_rf = X_train
            y_train_rf = y_train

        rf_model = train_random_forest(X_train_rf, y_train_rf, params=rf_params)
        rf_mae, rf_r2, rf_pred = evaluate_random_forest(rf_model, X_test, y_test)
        rf_importance = get_feature_importance(rf_model, feature_names, top_n=15)
        results["Random Forest"] = {"mae": rf_mae, "r2": rf_r2}

    # Model 3: LightGBM
    if "lgbm" in selected_models or "ensemble" in selected_models:
        logger.info("=" * 80)
        logger.info("MODEL: LightGBM (Production)")
        logger.info("=" * 80)
        lgbm_model = train_lightgbm(X_train, y_train, use_gpu=args.gpu, params=lgbm_params)
        lgbm_mae, lgbm_r2, lgbm_pred = evaluate_lightgbm(lgbm_model, X_test, y_test)
        lgbm_importance = get_lightgbm_feature_importance(lgbm_model, feature_names, top_n=15)
        results["LightGBM"] = {"mae": lgbm_mae, "r2": lgbm_r2}

    # Model 4: XGBoost GPU
    if "xgb" in selected_models or "ensemble" in selected_models:
        logger.info("=" * 80)
        logger.info("MODEL: XGBoost GPU")
        logger.info("=" * 80)
        xgb_model = train_xgboost(X_train, y_train, use_gpu=args.gpu, params=xgb_params)
        xgb_mae, xgb_r2, xgb_pred = evaluate_xgboost(xgb_model, X_test, y_test)
        xgb_importance = get_xgboost_feature_importance(xgb_model, feature_names, top_n=15)
        results["XGBoost"] = {"mae": xgb_mae, "r2": xgb_r2}

    # Model 5: ExtraTrees
    if "extra" in selected_models or "ensemble" in selected_models:
        logger.info("=" * 80)
        logger.info("MODEL: ExtraTrees")
        logger.info("=" * 80)
        extra_model = train_extra_trees(X_train, y_train, params=extra_params)
        extra_mae, extra_r2, extra_pred = evaluate_extra_trees(extra_model, X_test, y_test)
        extra_importance = get_extra_trees_feature_importance(extra_model, feature_names, top_n=15)
        results["ExtraTrees"] = {"mae": extra_mae, "r2": extra_r2}

    # Ensemble (simple average of available tree-based models)
    if "ensemble" in selected_models:
        logger.info("=" * 80)
        logger.info("MODEL: Ensemble (average of selected tree models)")
        logger.info("=" * 80)
        tree_preds = [p for p in [rf_pred, lgbm_pred, xgb_pred, extra_pred] if p is not None]
        if tree_preds:
            ensemble_pred = sum(tree_preds) / len(tree_preds)
            ensemble_mae = (ensemble_pred - y_test).abs().mean()
            ensemble_r2 = 1 - ((y_test - ensemble_pred) ** 2).sum() / (
                (y_test - y_test.mean()) ** 2
            ).sum()
            results["Ensemble (avg)"] = {"mae": float(ensemble_mae), "r2": float(ensemble_r2)}
        else:
            logger.warning("No tree models available for ensemble; skipping ensemble.")

    # Generate visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    # Feature importance plots
    if "rf" in selected_models or ("ensemble" in selected_models and rf_pred is not None):
        plot_feature_importance(
            rf_importance, str(output_dir / "feature_importance_rf.png"), "Random Forest"
        )
    if "lgbm" in selected_models or ("ensemble" in selected_models and lgbm_pred is not None):
        plot_feature_importance(
            lgbm_importance, str(output_dir / "feature_importance_lgbm.png"), "LightGBM"
        )
    if "xgb" in selected_models or ("ensemble" in selected_models and xgb_pred is not None):
        plot_feature_importance(
            xgb_importance, str(output_dir / "feature_importance_xgb.png"), "XGBoost"
        )
    if "extra" in selected_models or ("ensemble" in selected_models and extra_pred is not None):
        plot_feature_importance(
            extra_importance,
            str(output_dir / "feature_importance_extra_trees.png"),
            "ExtraTrees",
        )

    # Predicted vs actual plots
    if lr_pred is not None and "lr" in selected_models:
        plot_predicted_vs_actual(
            y_test,
            lr_pred,
            str(output_dir / "predicted_vs_actual_lr.png"),
            "Linear Regression",
            lr_mae,
            lr_r2,
        )
    if rf_pred is not None and ("rf" in selected_models or "ensemble" in selected_models):
        plot_predicted_vs_actual(
            y_test,
            rf_pred,
            str(output_dir / "predicted_vs_actual_rf.png"),
            "Random Forest",
            rf_mae,
            rf_r2,
        )
    if lgbm_pred is not None and ("lgbm" in selected_models or "ensemble" in selected_models):
        plot_predicted_vs_actual(
            y_test,
            lgbm_pred,
            str(output_dir / "predicted_vs_actual_lgbm.png"),
            "LightGBM",
            lgbm_mae,
            lgbm_r2,
        )
    if xgb_pred is not None and ("xgb" in selected_models or "ensemble" in selected_models):
        plot_predicted_vs_actual(
            y_test,
            xgb_pred,
            str(output_dir / "predicted_vs_actual_xgb.png"),
            "XGBoost",
            xgb_mae,
            xgb_r2,
        )
    if extra_pred is not None and ("extra" in selected_models or "ensemble" in selected_models):
        plot_predicted_vs_actual(
            y_test,
            extra_pred,
            str(output_dir / "predicted_vs_actual_extra_trees.png"),
            "ExtraTrees",
            extra_mae,
            extra_r2,
        )
    if "ensemble" in selected_models and "Ensemble (avg)" in results:
        plot_predicted_vs_actual(
            y_test,
            ensemble_pred,
            str(output_dir / "predicted_vs_actual_ensemble.png"),
            "Ensemble (avg)",
            results["Ensemble (avg)"]["mae"],
            results["Ensemble (avg)"]["r2"],
        )

    # Model comparison plot
    plot_model_comparison(results, str(output_dir / "model_comparison.png"))

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY - MODEL PERFORMANCE")
    logger.info("=" * 80)
    for model_name, metrics in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  MAE: {metrics['mae']:.2f} days")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info("")

    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]["mae"])
    logger.info(f"Best Model (lowest MAE): {best_model[0]}")
    logger.info(f"  MAE: {best_model[1]['mae']:.2f} days")
    logger.info(f"  R²: {best_model[1]['r2']:.4f}")

    logger.info("=" * 80)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
