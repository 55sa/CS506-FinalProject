"""Optuna hyperparameter tuning for LightGBM regression (CPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.models.feature_prep import prepare_ml_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_dataset(sample: float) -> Tuple[pd.DataFrame, pd.Series]:
    """Load, preprocess, engineer features, and optionally subsample."""
    df = load_data()
    if df.empty:
        raise ValueError("No data loaded. Please run download_data.py first.")

    df = preprocess_data(df)
    X_train, X_test, y_train, y_test, _ = prepare_ml_features(df)
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    if sample < 1.0:
        X_full, y_full = sample_data(X_full, y_full, sample)
        logger.info(f"Using sampled dataset: {len(X_full)} rows")
    else:
        logger.info(f"Using full dataset: {len(X_full)} rows")

    return X_full, y_full


def sample_data(X: pd.DataFrame, y: pd.Series, frac: float) -> Tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic sample of the dataset."""
    sampled_idx = X.sample(frac=frac, random_state=42).index
    return X.loc[sampled_idx], y.loc[sampled_idx]


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective minimizing MAE."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "device": "cpu",
    }

    model = LGBMRegressor(**params)

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    mae = float(-np.mean(scores))
    trial.report(mae, step=0)
    return mae


def save_results(best_mae: float, best_params: dict, args: argparse.Namespace) -> None:
    """Persist best study results to outputs/tuning."""
    output_dir = Path("outputs/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lightgbm_tuning.json"

    payload = {
        "model": "LightGBMRegressor",
        "best_mae": best_mae,
        "best_params": best_params,
        "n_trials": args.trials,
        "sample_fraction": args.sample,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Saved best results to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Tune LightGBM with Optuna (MAE)")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (20-50 suggested)")
    parser.add_argument(
        "--sample",
        type=float,
        default=0.1,
        help="Sampling fraction for tuning dataset (default: 0.1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=" * 80)
    logger.info("LightGBM Hyperparameter Tuning (MAE, 3-fold CV, CPU)")
    logger.info(f"Trials: {args.trials} | Sample: {args.sample*100:.0f}%")
    logger.info("=" * 80)

    X, y = prepare_dataset(sample=args.sample)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=args.trials, show_progress_bar=True)

    logger.info(f"Best MAE: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    save_results(study.best_value, study.best_params, args)


if __name__ == "__main__":
    main()
