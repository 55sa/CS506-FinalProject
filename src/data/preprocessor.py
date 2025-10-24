"""Data preprocessing for Boston 311 requests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to proper types and handle invalid dates."""
    logger.info("Cleaning datetime columns")

    df["open_dt"] = pd.to_datetime(df["open_dt"], errors="coerce")
    null_open = df["open_dt"].isna().sum()
    if null_open > 0:
        logger.warning(f"Found {null_open} invalid OPEN_DT records")

    df["closed_dt"] = pd.to_datetime(df["closed_dt"], errors="coerce")
    null_closed = df["closed_dt"].isna().sum()
    logger.info(f"Found {null_closed} null CLOSED_DT (open/missing)")

    return df


def derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive temporal features from open_dt timestamp.

    Adds columns: year, month, day_of_week, hour, date.
    """
    logger.info("Deriving temporal features from OPEN_DT")

    df["year"] = df["open_dt"].dt.year
    df["month"] = df["open_dt"].dt.month
    df["day_of_week"] = df["open_dt"].dt.day_name()
    df["hour"] = df["open_dt"].dt.hour
    df["date"] = df["open_dt"].dt.date

    years = sorted(df["year"].dropna().unique())
    logger.info(f"Derived features for years: {list(years)}")

    return df


def calculate_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate resolution time in hours for closed cases."""
    logger.info("Calculating resolution times")

    df["resolution_hours"] = (
        (df["closed_dt"] - df["open_dt"]).dt.total_seconds() / 3600
    )

    resolved = df["resolution_hours"].notna().sum()
    logger.info(f"Calculated resolution times for {resolved} closed cases")

    if resolved > 0:
        mean_res = df["resolution_hours"].mean()
        median_res = df["resolution_hours"].median()
        logger.info(f"Mean resolution time: {mean_res:.2f} hours")
        logger.info(f"Median resolution time: {median_res:.2f} hours")

    return df


def standardize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize categorical columns by stripping whitespace and handling nulls."""
    logger.info("Standardizing categorical columns")

    categorical_cols = [
        "neighborhood",
        "subject",
        "reason",
        "type",
        "queue",
        "source",
        "case_status",
    ]

    for col in categorical_cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found")
            continue

        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["nan", "NaN", "None", ""], np.nan)

        nulls = df[col].isna().sum()
        unique = df[col].nunique()
        logger.info(f"{col}: {unique} unique values, {nulls} nulls")

    return df


def validate_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Generate data quality report."""
    logger.info("Validating data quality")

    quality_report = {
        "total_records": len(df),
        "duplicate_case_ids": (
            df.duplicated(subset=["case_enquiry_id"]).sum()
            if "case_enquiry_id" in df.columns
            else "N/A"
        ),
        "missing_open_dt": df["open_dt"].isna().sum(),
        "missing_closed_dt": df["closed_dt"].isna().sum(),
        "missing_neighborhood": (
            df["neighborhood"].isna().sum()
            if "neighborhood" in df.columns
            else "N/A"
        ),
        "case_status_breakdown": (
            df["case_status"].value_counts().to_dict()
            if "case_status" in df.columns
            else "N/A"
        ),
        "year_range": (
            f"{int(df['year'].min())}-{int(df['year'].max())}"
            if "year" in df.columns
            else "N/A"
        ),
    }

    logger.info("Data Quality Report:")
    for key, value in quality_report.items():
        logger.info(f"  {key}: {value}")

    return quality_report


def remove_invalid_records(
    df: pd.DataFrame, drop_missing_open_dt: bool = True
) -> pd.DataFrame:
    """Remove invalid records based on data quality rules."""
    logger.info("Removing invalid records")
    initial = len(df)

    if drop_missing_open_dt:
        df = df.dropna(subset=["open_dt"])
        logger.info(f"Dropped {initial - len(df)} records with missing OPEN_DT")

    if "case_enquiry_id" in df.columns:
        initial = len(df)
        df = df.drop_duplicates(subset=["case_enquiry_id"], keep="first")
        if (dup_count := initial - len(df)) > 0:
            logger.info(f"Dropped {dup_count} duplicate case IDs")

    logger.info(f"Final record count: {len(df)}")
    return df


def preprocess_data(
    df: pd.DataFrame, drop_invalid: bool = True
) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to raw 311 data.

    Args:
        df: Raw DataFrame loaded from CSV files.
        drop_invalid: Whether to drop invalid records.

    Returns:
        Fully preprocessed DataFrame ready for analysis.

    Examples:
        >>> raw_df = load_data()
        >>> clean_df = preprocess_data(raw_df)
    """
    logger.info("Starting preprocessing pipeline")

    df = clean_datetime_columns(df)

    if drop_invalid:
        df = remove_invalid_records(df)

    df = standardize_categorical_columns(df)
    df = derive_temporal_features(df)
    df = calculate_resolution_time(df)

    validate_data_quality(df)

    logger.info("Preprocessing complete")
    return df


def main() -> None:
    """Run preprocessing workflow: load, clean, and save data."""
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_data, save_processed_data

    logger.info("Starting preprocessing workflow")

    raw_df = load_data()

    if not raw_df.empty:
        processed_df = preprocess_data(raw_df)
        save_processed_data(processed_df, "data/processed/311_cleaned.csv")
        logger.info("Preprocessing workflow completed successfully")
    else:
        logger.error("No data to preprocess. Check data/raw/")


if __name__ == "__main__":
    main()
