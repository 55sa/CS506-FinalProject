"""Data loader for Boston 311 requests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_single_year(filepath: Path) -> pd.DataFrame:
    """Load single year CSV file and return DataFrame."""
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded {len(df)} records from {filepath.name}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise


def load_all_years(
    data_dir: Path, years: Optional[list[int]] = None
) -> pd.DataFrame:
    """
    Load and merge multiple years of 311 data.

    Args:
        data_dir: Directory containing yearly CSV files.
        years: Specific years to load (loads all if None).

    Returns:
        Combined DataFrame with all years' data.
    """
    logger.info(f"Loading data from directory: {data_dir}")

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Get CSV files to load
    if years:
        csv_files = [
            data_dir / f"311_requests_{year}.csv" for year in years
        ]
        csv_files = [f for f in csv_files if f.exists()]
    else:
        csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(csv_files)} CSV files to load")

    # Load and concatenate
    dataframes = []
    for filepath in csv_files:
        try:
            df = load_single_year(filepath)
            dataframes.append(df)
        except Exception as e:
            logger.warning(f"Skipping {filepath.name}: {e}")
            continue

    if not dataframes:
        logger.error("No data loaded successfully")
        return pd.DataFrame()

    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataset contains {len(combined_df)} total records")

    return combined_df


def load_data(
    data_dir: str = "data/raw", years: Optional[list[int]] = None
) -> pd.DataFrame:
    """
    Load 311 service request data from CSV files.

    Args:
        data_dir: Path to directory with CSV files (default: "data/raw").
        years: Specific years to load (loads all if None).

    Returns:
        Combined DataFrame with all requested data.

    Examples:
        >>> df = load_data()  # Load all years
        >>> df = load_data(years=[2020, 2021, 2022])  # Specific years
    """
    return load_all_years(Path(data_dir), years)


def save_processed_data(
    df: pd.DataFrame, output_path: str = "data/processed/311_combined.csv"
) -> None:
    """Save processed DataFrame to CSV file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(df)} records to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved successfully")


def main() -> None:
    """Load all available data and save to processed directory."""
    logger.info("Starting data loading process")

    df = load_data()

    if not df.empty:
        logger.info(f"Successfully loaded {len(df):,} total records")
        logger.info(f"Columns: {list(df.columns)[:5]}...")
        save_processed_data(df)
    else:
        logger.warning("No data loaded. Check data/raw/ directory.")


if __name__ == "__main__":
    main()
