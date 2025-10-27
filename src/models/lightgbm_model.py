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
    n_estimators: int = 2504,
    learning_rate: float = 0.08664631853295084,
    num_leaves: int = 5,
    reg_alpha: float = 0.005144845893441943,
    reg_lambda: float = 0.05549669560219034,
    max_bin: int = 1024,
    colsample_bytree: float = 1.0,
    min_child_samples: int = 3,
    random_state: int = 42,
    use_gpu: bool = True
) -> LGBMRegressor:
    """Train LightGBM regression model with AutoML-optimized parameters."""
    logger.info(f"Training LightGBM model with {n_estimators} trees (AutoML optimized)")
    logger.info(f"AutoML parameters: max_bin={max_bin}, num_leaves={num_leaves}, learning_rate={learning_rate:.6f}")

    # Base parameters with AutoML optimization
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "max_bin": max_bin,
        "colsample_bytree": colsample_bytree,
        "min_child_samples": min_child_samples,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1
    }

    # Add GPU parameters if enabled
    if use_gpu:
        params.update({
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0
        })

    model = LGBMRegressor(**params)

    model.fit(X_train, y_train)
    logger.info("LightGBM training complete")

    return model


def evaluate_lightgbm(
    model: LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
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
    model: LGBMRegressor,
    feature_names: list,
    top_n: int = 15
) -> pd.DataFrame:
    """Extract top N feature importances from trained LightGBM model."""
    logger.info(f"Extracting top {top_n} feature importances")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(top_n)

    logger.info("Top 5 features:")
    for idx, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return importance_df
