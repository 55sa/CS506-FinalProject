"""
Daily request volume forecasting using Prophet, SARIMA, and LightGBM.

This is a standalone script similar to the resolution time pipeline. It:
1) Loads and preprocesses data
2) Aggregates daily counts
3) Trains three models (Prophet, SARIMA, LightGBM)
4) Evaluates on a holdout horizon and saves forecast plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def aggregate_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily request counts."""
    daily = df.groupby("date").size().reset_index(name="y")
    daily = daily.sort_values("date").rename(columns={"date": "ds"})
    daily["ds"] = pd.to_datetime(daily["ds"])
    return daily


def train_prophet(train: pd.DataFrame) -> Prophet:
    """Train Prophet model."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    model.fit(train)
    return model


def forecast_prophet(model: Prophet, horizon: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)[["ds", "yhat"]]
    return forecast


def train_sarima(train: pd.Series) -> SARIMAX:
    """Train SARIMA model with a simple seasonal spec."""
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False)
    results = model.fit(disp=False)
    return results


def create_lgbm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and lag features for LightGBM."""
    df_feat = df.copy()
    df_feat["year"] = df_feat["ds"].dt.year
    df_feat["month"] = df_feat["ds"].dt.month
    df_feat["day"] = df_feat["ds"].dt.day
    df_feat["dayofweek"] = df_feat["ds"].dt.dayofweek
    df_feat["weekofyear"] = df_feat["ds"].dt.isocalendar().week.astype(int)

    for lag in [1, 7, 14]:
        df_feat[f"lag_{lag}"] = df_feat["y"].shift(lag)

    for win in [7, 14]:
        df_feat[f"roll_mean_{win}"] = df_feat["y"].shift(1).rolling(win).mean()

    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat


def train_lgbm(train_feat: pd.DataFrame, feature_cols: list[str]) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_feat[feature_cols], train_feat["y"])
    return model


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    """Return MAE and RMSE (manual sqrt to support older sklearn)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def plot_forecast(
    actual: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(actual["ds"], actual["y"], label="Actual", color="black", linewidth=2)

    for name, df_pred in forecasts.items():
        plt.plot(df_pred["ds"], df_pred["yhat"], label=name, linewidth=1.5)

    plt.xlabel("Date")
    plt.ylabel("Requests per day")
    plt.title("Daily Request Volume Forecast")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily request volume forecasting")
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Holdout horizon in days (default: 30)",
    )
    args = parser.parse_args()

    output_dir = Path("outputs/figures/forecast_requests")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and preprocessing data...")
    df = preprocess_data(load_data())

    logger.info("Aggregating daily counts...")
    daily = aggregate_daily_counts(df)

    if len(daily) <= args.horizon + 14:
        logger.error("Not enough data for the requested horizon.")
        return

    train = daily.iloc[:-args.horizon].reset_index(drop=True)
    test = daily.iloc[-args.horizon :].reset_index(drop=True)

    # Prophet
    logger.info("Training Prophet...")
    prophet_model = train_prophet(train)
    prophet_forecast = forecast_prophet(prophet_model, args.horizon)
    prophet_pred = prophet_forecast.tail(args.horizon).reset_index(drop=True)
    prophet_pred.columns = ["ds", "yhat"]
    prophet_mae, prophet_rmse = evaluate(test["y"], prophet_pred["yhat"])

    # SARIMA
    logger.info("Training SARIMA...")
    sarima_model = train_sarima(train["y"].set_axis(train["ds"]))
    sarima_pred_vals = sarima_model.forecast(args.horizon)
    sarima_pred = pd.DataFrame({"ds": test["ds"], "yhat": sarima_pred_vals.values})
    sarima_mae, sarima_rmse = evaluate(test["y"], sarima_pred["yhat"])

    # LightGBM
    logger.info("Training LightGBM...")
    lgbm_data = create_lgbm_features(daily)
    split_idx = len(lgbm_data) - args.horizon
    feature_cols = [c for c in lgbm_data.columns if c not in {"ds", "y"}]
    lgbm_train = lgbm_data.iloc[:split_idx]
    lgbm_test = lgbm_data.iloc[split_idx:]
    lgbm_model = train_lgbm(lgbm_train, feature_cols)
    lgbm_pred_vals = lgbm_model.predict(lgbm_test[feature_cols])
    lgbm_pred = pd.DataFrame({"ds": lgbm_test["ds"], "yhat": lgbm_pred_vals})
    lgbm_mae, lgbm_rmse = evaluate(lgbm_test["y"], lgbm_pred["yhat"])

    # Log metrics
    logger.info("=== Forecast Metrics (Holdout) ===")
    logger.info(f"Prophet  - MAE: {prophet_mae:.2f}, RMSE: {prophet_rmse:.2f}")
    logger.info(f"SARIMA   - MAE: {sarima_mae:.2f}, RMSE: {sarima_rmse:.2f}")
    logger.info(f"LightGBM - MAE: {lgbm_mae:.2f}, RMSE: {lgbm_rmse:.2f}")

    # Plot forecasts
    forecasts = {
        "Prophet": prophet_pred,
        "SARIMA": sarima_pred,
        "LightGBM": lgbm_pred,
    }
    plot_forecast(test, forecasts, output_dir / "forecast_comparison.png")
    logger.info(f"Saved forecast plot to {output_dir / 'forecast_comparison.png'}")


if __name__ == "__main__":
    main()
