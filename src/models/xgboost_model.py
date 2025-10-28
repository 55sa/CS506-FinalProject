"""XGBoost GPU model for resolution time prediction."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42,
    use_gpu: bool = True,
) -> xgb.XGBRegressor:
    """Train XGBoost regression model with optional GPU support."""
    if use_gpu:
        logger.info(f"Training GPU XGBoost model with {n_estimators} trees")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            tree_method="hist",
            device="cuda:0",
            n_jobs=-1,
            verbosity=0,
        )
    else:
        logger.info(f"Training CPU XGBoost model with {n_estimators} trees")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )

    model.fit(X_train, y_train)
    logger.info("XGBoost training complete")

    return model


def evaluate_xgboost(
    model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, float, pd.Series]:
    """Evaluate XGBoost model and return MAE, R², and predictions."""
    logger.info("Evaluating XGBoost model")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"XGBoost Performance:")
    logger.info(f"  MAE: {mae:.2f} days")
    logger.info(f"  R² Score: {r2:.4f}")

    return mae, r2, pd.Series(y_pred, index=y_test.index)


def get_xgboost_feature_importance(
    model: xgb.XGBRegressor, feature_names: list, top_n: int = 15
) -> pd.DataFrame:
    """Extract top N feature importances from trained XGBoost model."""
    logger.info(f"Extracting top {top_n} feature importances")

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    logger.info("Top 5 features:")
    for idx, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return importance_df
