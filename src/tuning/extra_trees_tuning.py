"""Optuna hyperparameter tuning for ExtraTrees regressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
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
        sampled_idx = X_full.sample(frac=sample, random_state=42).index
        X_full = X_full.loc[sampled_idx]
        y_full = y_full.loc[sampled_idx]
        logger.info(f"Using sampled dataset: {len(X_full)} rows")
    else:
        logger.info(f"Using full dataset: {len(X_full)} rows")

    return X_full, y_full


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective minimizing MAE."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 10, 40),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "max_samples": trial.suggest_float("max_samples", 0.2, 0.8)
        if trial.params.get("bootstrap", True)
        else None,
        "random_state": 42,
        "n_jobs": -1,
    }

    # Respect sklearn constraint: max_samples only valid if bootstrap=True
    if not params["bootstrap"]:
        params.pop("max_samples", None)

    model = ExtraTreesRegressor(**params)
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
    output_path = output_dir / "extra_trees_tuning.json"

    payload = {
        "model": "ExtraTreesRegressor",
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
    parser = argparse.ArgumentParser(description="Tune ExtraTrees with Optuna (MAE)")
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
    logger.info("ExtraTrees Hyperparameter Tuning (MAE, 3-fold CV)")
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
