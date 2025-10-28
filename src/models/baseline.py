"""Baseline Linear Regression model for resolution time prediction."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train a baseline Linear Regression model."""
    logger.info("Training baseline Linear Regression model")

    model = LinearRegression()
    model.fit(X_train, y_train)

    logger.info("Baseline model training complete")

    return model


def evaluate_model(
    model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, float, pd.Series]:
    """Evaluate model and return MAE, R², and predictions."""
    logger.info("Evaluating model on test set")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Baseline Model Performance:")
    logger.info(f"  MAE: {mae:.2f} days")
    logger.info(f"  R² Score: {r2:.4f}")

    return mae, r2, pd.Series(y_pred, index=y_test.index)
