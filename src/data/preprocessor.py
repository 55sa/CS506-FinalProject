"""
Data preprocessing module for Boston 311 Service Request Analysis.

This module handles data cleaning, validation, and feature derivation.
Does not perform analysis - only prepares data for downstream analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Setup logging
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def clean_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime columns to proper datetime types and handle invalid dates.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'open_dt' and 'closed_dt' columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned datetime columns
    """
    logger.info("Cleaning datetime columns")

    # Convert open_dt to datetime
    df["open_dt"] = pd.to_datetime(df["open_dt"], errors="coerce")
    open_dt_series: pd.Series[Any] = df["open_dt"]
    null_open_count: int = int(open_dt_series.isna().sum())
    if null_open_count > 0:
        logger.warning(f"Found {null_open_count} records with invalid OPEN_DT")

    # Convert closed_dt to datetime (many will be null for open cases)
    df["closed_dt"] = pd.to_datetime(df["closed_dt"], errors="coerce")
    closed_dt_series: pd.Series[Any] = df["closed_dt"]
    null_closed_count: int = int(closed_dt_series.isna().sum())
    logger.info(
        f"Found {null_closed_count} records with null CLOSED_DT (open cases or missing data)"
    )

    return df


def derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive temporal features from open_dt timestamp.

    Adds columns: year, month, day_of_week, hour, date

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'open_dt' column (must be datetime type)

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional temporal feature columns
    """
    logger.info("Deriving temporal features from OPEN_DT")

    # Extract temporal components
    df["year"] = df["open_dt"].dt.year
    df["month"] = df["open_dt"].dt.month
    df["day_of_week"] = df["open_dt"].dt.day_name()
    df["hour"] = df["open_dt"].dt.hour
    df["date"] = df["open_dt"].dt.date

    years_list: list[Any] = sorted(df["year"].dropna().unique().tolist())
    logger.info(f"Derived features for years: {years_list}")

    return df


def calculate_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate resolution time in hours for closed cases.

    Adds column: resolution_hours (null for open cases)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'open_dt' and 'closed_dt' columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'resolution_hours' column added
    """
    logger.info("Calculating resolution times")

    # Calculate resolution time only where both dates exist
    resolution_time = (df["closed_dt"] - df["open_dt"]).dt.total_seconds() / 3600

    df["resolution_hours"] = resolution_time

    # Count how many cases have resolution times
    resolution_series: pd.Series[Any] = df["resolution_hours"]
    resolved_count: int = int(resolution_series.notna().sum())
    logger.info(f"Calculated resolution times for {resolved_count} closed cases")

    # Log statistics
    if resolved_count > 0:
        mean_resolution: float = float(df["resolution_hours"].mean())
        median_resolution: float = float(df["resolution_hours"].median())
        logger.info(f"Mean resolution time: {mean_resolution:.2f} hours")
        logger.info(f"Median resolution time: {median_resolution:.2f} hours")

    return df


def standardize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize categorical columns by stripping whitespace and handling nulls.

    Cleans columns: neighborhood, subject, reason, type, queue, source, case_status

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with categorical columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with standardized categorical columns
    """
    logger.info("Standardizing categorical columns")

    categorical_columns: list[str] = [
        "neighborhood",
        "subject",
        "reason",
        "type",
        "queue",
        "source",
        "case_status",
    ]

    for col in categorical_columns:
        if col in df.columns:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()

            # Replace 'nan' string with actual NaN
            df[col] = df[col].replace(["nan", "NaN", "None", ""], np.nan)

            col_series: pd.Series[Any] = df[col]
            null_count: int = int(col_series.isna().sum())
            unique_count: int = int(col_series.nunique())

            logger.info(f"{col}: {unique_count} unique values, {null_count} nulls")
        else:
            logger.warning(f"Column {col} not found in DataFrame")

    return df


def validate_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate data quality report.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate

    Returns:
    --------
    dict[str, Any]
        Dictionary containing data quality metrics
    """
    logger.info("Validating data quality")

    # Extract series for type checking
    open_dt_col: pd.Series[Any] = df["open_dt"]
    closed_dt_col: pd.Series[Any] = df["closed_dt"]

    quality_report: dict[str, Any] = {
        "total_records": len(df),
        "duplicate_case_ids": (
            int(df.duplicated(subset=["case_enquiry_id"]).sum())
            if "case_enquiry_id" in df.columns
            else "N/A"
        ),
        "missing_open_dt": int(open_dt_col.isna().sum()),
        "missing_closed_dt": int(closed_dt_col.isna().sum()),
        "missing_neighborhood": (
            int(df["neighborhood"].isna().sum()) if "neighborhood" in df.columns else "N/A"
        ),
        "case_status_breakdown": (
            df["case_status"].value_counts().to_dict() if "case_status" in df.columns else "N/A"
        ),
        "year_range": (
            f"{int(df['year'].min())}-{int(df['year'].max())}" if "year" in df.columns else "N/A"
        ),
    }

    # Log the report
    logger.info("Data Quality Report:")
    for key, value in quality_report.items():
        logger.info(f"  {key}: {value}")

    return quality_report


def remove_invalid_records(df: pd.DataFrame, drop_missing_open_dt: bool = True) -> pd.DataFrame:
    """
    Remove invalid records based on data quality rules.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to clean
    drop_missing_open_dt : bool
        Whether to drop records with missing open_dt (default: True)

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    logger.info("Removing invalid records")
    initial_count: int = len(df)

    # Drop records with missing open_dt if specified
    if drop_missing_open_dt:
        df = df.dropna(subset=["open_dt"])
        logger.info(f"Dropped {initial_count - len(df)} records with missing OPEN_DT")

    # Drop duplicate case_enquiry_id if it exists
    if "case_enquiry_id" in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["case_enquiry_id"], keep="first")
        duplicates_dropped: int = initial_count - len(df)
        if duplicates_dropped > 0:
            logger.info(f"Dropped {duplicates_dropped} duplicate case IDs")

    logger.info(f"Final record count: {len(df)}")

    return df


def preprocess_data(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """
    Main preprocessing pipeline - applies all cleaning and feature derivation steps.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame loaded from CSV files
    drop_invalid : bool
        Whether to drop invalid records (default: True)

    Returns:
    --------
    pd.DataFrame
        Fully preprocessed DataFrame ready for analysis

    Examples:
    ---------
    >>> from src.data.loader import load_data
    >>> from src.data.preprocessor import preprocess_data
    >>>
    >>> # Load raw data
    >>> raw_df = load_data()
    >>>
    >>> # Preprocess
    >>> clean_df = preprocess_data(raw_df)
    """
    logger.info("Starting preprocessing pipeline")

    # Step 1: Clean datetime columns
    df = clean_datetime_columns(df)

    # Step 2: Remove invalid records if specified
    if drop_invalid:
        df = remove_invalid_records(df)

    # Step 3: Standardize categorical columns
    df = standardize_categorical_columns(df)

    # Step 4: Derive temporal features
    df = derive_temporal_features(df)

    # Step 5: Calculate resolution times
    df = calculate_resolution_time(df)

    # Step 6: Validate data quality
    quality_report: dict[str, Any] = validate_data_quality(df)

    logger.info("Preprocessing complete")

    return df


if __name__ == "__main__":
    # Add parent directory to path to import loader
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_data, save_processed_data

    logger.info("Starting preprocessing workflow")

    # Load raw data
    raw_df: pd.DataFrame = load_data()

    if not raw_df.empty:
        # Preprocess
        processed_df: pd.DataFrame = preprocess_data(raw_df)

        # Save processed data
        save_processed_data(processed_df, "data/processed/311_cleaned.csv")

        logger.info("Preprocessing workflow completed successfully")
    else:
        logger.error("No data to preprocess. Please add CSV files to data/raw/")
