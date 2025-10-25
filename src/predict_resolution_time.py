"""Main script for resolution time prediction using machine learning models."""

from __future__ import annotations

import argparse
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
from src.models.random_forest import (
    evaluate_random_forest,
    get_feature_importance,
    train_random_forest,
)
from src.utils.logger import get_logger
from src.visualization.model_plots import (
    plot_feature_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
)

logger = get_logger(__name__)


def main() -> None:
    """Run resolution time prediction pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Resolution time prediction")
    parser.add_argument(
        "--sample",
        type=float,
        default=0.1,
        help="Sampling percentage for Random Forest (default: 0.1 = 10%%)"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RESOLUTION TIME PREDICTION - PHASE 2")
    logger.info("=" * 80)
    logger.info(f"Random Forest sampling: {args.sample*100}%")

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

    # Train and evaluate models
    results = {}

    # Model 1: Linear Regression Baseline
    logger.info("=" * 80)
    logger.info("MODEL: Linear Regression Baseline")
    logger.info("=" * 80)
    lr_model = train_baseline_model(X_train, y_train)
    lr_mae, lr_r2, lr_pred = evaluate_model(lr_model, X_test, y_test)
    results["Linear Regression"] = {"mae": lr_mae, "r2": lr_r2}

    # Model 2: Random Forest (with sampling)
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

    rf_model = train_random_forest(X_train_rf, y_train_rf, n_estimators=100)
    rf_mae, rf_r2, rf_pred = evaluate_random_forest(rf_model, X_test, y_test)
    rf_importance = get_feature_importance(rf_model, feature_names, top_n=15)
    results["Random Forest"] = {"mae": rf_mae, "r2": rf_r2}

    # Model 3: LightGBM
    logger.info("=" * 80)
    logger.info("MODEL: LightGBM (Production)")
    logger.info("=" * 80)
    lgbm_model = train_lightgbm(X_train, y_train, n_estimators=100)
    lgbm_mae, lgbm_r2, lgbm_pred = evaluate_lightgbm(lgbm_model, X_test, y_test)
    lgbm_importance = get_lightgbm_feature_importance(lgbm_model, feature_names, top_n=15)
    results["LightGBM"] = {"mae": lgbm_mae, "r2": lgbm_r2}

    # Generate visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    # Feature importance plots
    plot_feature_importance(
        rf_importance,
        str(output_dir / "feature_importance_rf.png"),
        "Random Forest"
    )
    plot_feature_importance(
        lgbm_importance,
        str(output_dir / "feature_importance_lgbm.png"),
        "LightGBM"
    )

    # Predicted vs actual plots
    plot_predicted_vs_actual(
        y_test, lr_pred,
        str(output_dir / "predicted_vs_actual_lr.png"),
        "Linear Regression", lr_mae, lr_r2
    )
    plot_predicted_vs_actual(
        y_test, rf_pred,
        str(output_dir / "predicted_vs_actual_rf.png"),
        "Random Forest", rf_mae, rf_r2
    )
    plot_predicted_vs_actual(
        y_test, lgbm_pred,
        str(output_dir / "predicted_vs_actual_lgbm.png"),
        "LightGBM", lgbm_mae, lgbm_r2
    )

    # Model comparison plot
    plot_model_comparison(results, str(output_dir / "model_comparison.png"))

    # Summary
    logger.info("\n" + "=" * 80)
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

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
