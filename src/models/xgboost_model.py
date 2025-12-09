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
    use_gpu: bool = True,
    params: dict[str, object] | None = None,
) -> xgb.XGBRegressor:
    """Train XGBoost regression model with optional GPU support."""
    if params is None:
        raise ValueError("XGBoost params must be provided by caller")

    if use_gpu:
        logger.info("Training GPU XGBoost model")
        params["device"] = "cuda:0"
    else:
        logger.info("Training CPU XGBoost model")
        params["device"] = "cpu"

    logger.info(f"XGBoost params: {params}")
    model = xgb.XGBRegressor(**params)

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
