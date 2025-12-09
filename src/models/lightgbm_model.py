"""LightGBM production model for resolution time prediction."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_gpu: bool = False,
    params: dict[str, object] | None = None,
) -> LGBMRegressor:
    """Train LightGBM regression model."""
    device_type = "gpu" if use_gpu else "cpu"
    logger.info(f"Training LightGBM model on {device_type.upper()}")

    if params is None:
        raise ValueError("LightGBM params must be provided by caller")

    # Add GPU parameters if enabled
    if use_gpu:
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
    else:
        params["device"] = "cpu"

    logger.info(f"LightGBM params: {params}")

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    logger.info("LightGBM training complete")

    return model


def evaluate_lightgbm(
    model: LGBMRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, float, pd.Series]:
    """Evaluate LightGBM model and return MAE, R², and predictions."""
    logger.info("Evaluating LightGBM model")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"LightGBM Performance:")
    logger.info(f"  MAE: {mae:.2f} days")
    logger.info(f"  R² Score: {r2:.4f}")

    return mae, r2, pd.Series(y_pred, index=y_test.index)


def get_lightgbm_feature_importance(
    model: LGBMRegressor, feature_names: list, top_n: int = 15
) -> pd.DataFrame:
    """Extract top N feature importances from trained LightGBM model."""
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
