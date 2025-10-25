"""Feature preparation utilities for machine learning models."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def prepare_ml_features(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Prepare features for machine learning models.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed 311 data with all temporal and categorical features
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Testing target
    encoders : dict
        Dictionary of LabelEncoders for categorical features
    """
    logger.info("Preparing features for machine learning")

    # Filter to only cases with valid resolution time
    df_ml = df[df["resolution_time_days"].notna()].copy()
    df_ml = df_ml[df_ml["resolution_time_days"] >= 0]  # Remove negative times

    logger.info(f"Filtered to {len(df_ml)} cases with valid resolution time")

    # Define categorical columns to encode
    categorical_cols = [
        "subject", "reason", "type", "queue", "department",
        "neighborhood", "location_zipcode", "source", "closure_reason",
        "fire_district", "pwd_district", "police_district",
        "city_council_district", "season"
    ]

    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        if col in df_ml.columns:
            # Fill missing values
            df_ml[col] = df_ml[col].fillna("Unknown").astype(str)

            # Encode
            encoder = LabelEncoder()
            df_ml[f"{col}_encoded"] = encoder.fit_transform(df_ml[col])
            encoders[col] = encoder

    # Define feature columns
    feature_cols = [
        # Temporal features
        "year", "month", "day_of_week_num", "hour", "day_of_month",
        "is_holiday", "is_weekend",
        # Encoded categorical features
        "subject_encoded", "reason_encoded", "type_encoded",
        "queue_encoded", "department_encoded",
        "neighborhood_encoded", "location_zipcode_encoded",
        "source_encoded", "closure_reason_encoded",
        "fire_district_encoded", "pwd_district_encoded",
        "police_district_encoded", "city_council_district_encoded",
        "season_encoded"
    ]

    # Filter to only available features
    available_features = [f for f in feature_cols if f in df_ml.columns]
    logger.info(f"Using {len(available_features)} features for modeling")

    # Prepare X and y
    X = df_ml[available_features].copy()
    y = df_ml["resolution_time_days"].copy()

    # Handle any remaining NaN values
    X = X.fillna(X.median())

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Target range: {y.min():.2f} to {y.max():.2f} days")
    logger.info(f"Target mean: {y.mean():.2f} days, median: {y.median():.2f} days")

    return X_train, X_test, y_train, y_test, encoders
