"""Visualization functions for model evaluation plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str,
    model_name: str = "Model"
) -> None:
    """Plot horizontal bar chart of top feature importances."""
    logger.info(f"Creating feature importance plot for {model_name}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot horizontal bars
    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        hue="feature",
        ax=ax,
        palette="viridis",
        legend=False
    )

    ax.set_title(f"Top {len(importance_df)} Feature Importances - {model_name}", fontsize=14, pad=20)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved feature importance plot to {output_path}")


def plot_predicted_vs_actual(
    y_test: pd.Series,
    y_pred: pd.Series,
    output_path: str,
    model_name: str = "Model",
    mae: float | None = None,
    r2: float | None = None
) -> None:
    """Plot scatter plot of predicted vs actual values."""
    logger.info(f"Creating predicted vs actual plot for {model_name}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors="k", linewidths=0.5)

    # Add diagonal line (perfect prediction)
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Prediction")

    # Add metrics to plot
    if mae is not None and r2 is not None:
        ax.text(
            0.05, 0.95,
            f"MAE: {mae:.2f} days\nR²: {r2:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

    ax.set_title(f"Predicted vs Actual Resolution Time - {model_name}", fontsize=14, pad=20)
    ax.set_xlabel("Actual Resolution Time (days)", fontsize=12)
    ax.set_ylabel("Predicted Resolution Time (days)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved predicted vs actual plot to {output_path}")


def plot_model_comparison(
    results: dict,
    output_path: str
) -> None:
    """Plot comparison of multiple models (MAE and R² scores)."""
    logger.info("Creating model comparison plot")

    models = list(results.keys())
    mae_values = [results[m]["mae"] for m in models]
    r2_values = [results[m]["r2"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MAE comparison
    ax1.barh(models, mae_values, color="coral")
    ax1.set_xlabel("MAE (days)", fontsize=12)
    ax1.set_title("Model Comparison - MAE", fontsize=14)
    ax1.invert_yaxis()

    # R² comparison
    ax2.barh(models, r2_values, color="skyblue")
    ax2.set_xlabel("R² Score", fontsize=12)
    ax2.set_title("Model Comparison - R²", fontsize=14)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved model comparison plot to {output_path}")
