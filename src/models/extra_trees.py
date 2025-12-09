"""ExtraTrees regressor for resolution time prediction."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def train_extra_trees(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, object] | None = None
) -> ExtraTreesRegressor:
    """Train ExtraTrees regression model."""
    params = params or {}
    logger.info(f"Training ExtraTrees model with params: {params}")
    model = ExtraTreesRegressor(**params)
    model.fit(X_train, y_train)
    logger.info("ExtraTrees training complete")
    return model


def evaluate_extra_trees(
    model: ExtraTreesRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, float, pd.Series]:
    """Evaluate ExtraTrees model and return MAE, R², and predictions."""
    logger.info("Evaluating ExtraTrees model")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info("ExtraTrees Performance:")
    logger.info(f"  MAE: {mae:.2f} days")
    logger.info(f"  R² Score: {r2:.4f}")
    return mae, r2, pd.Series(y_pred, index=y_test.index)


def get_extra_trees_feature_importance(
    model: ExtraTreesRegressor, feature_names: list, top_n: int = 15
) -> pd.DataFrame:
    """Extract top N feature importances from trained ExtraTrees model."""
    logger.info(f"Extracting top {top_n} feature importances (ExtraTrees)")
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    logger.info("Top 5 features:")
    for _, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    return importance_df
