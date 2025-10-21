"""
Data loader module for Boston 311 Service Request Analysis.

This module handles loading and merging yearly CSV files from the raw data directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Setup logging
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_single_year(filepath: Path) -> pd.DataFrame:
    """
    Load a single year's CSV file of 311 service requests.

    Parameters:
    -----------
    filepath : Path
        Path to the CSV file to load

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the year's 311 requests
    """
    logger.info(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded {len(df)} records from {filepath.name}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        raise


def load_all_years(data_dir: Path, years: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Load and merge multiple years of 311 service request data.

    Parameters:
    -----------
    data_dir : Path
        Directory containing the yearly CSV files
    years : Optional[List[int]]
        List of specific years to load. If None, attempts to load all CSV files
        in the directory.

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame containing all years' data
    """
    logger.info(f"Loading data from directory: {data_dir}")

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Get list of CSV files to load
    csv_files: list[Path]
    if years:
        csv_files = [data_dir / f"311_requests_{year}.csv" for year in years]
        # Filter to only existing files
        csv_files = [f for f in csv_files if f.exists()]
    else:
        csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(csv_files)} CSV files to load")

    # Load and concatenate all files
    dataframes: list[pd.DataFrame] = []
    for filepath in csv_files:
        try:
            df: pd.DataFrame = load_single_year(filepath)
            dataframes.append(df)
        except Exception as e:
            logger.warning(f"Skipping {filepath.name} due to error: {str(e)}")
            continue

    if not dataframes:
        logger.error("No data loaded successfully")
        return pd.DataFrame()

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataset contains {len(combined_df)} total records")

    return combined_df


def load_data(data_dir: str = "data/raw", years: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Main function to load 311 service request data.

    This is a convenience wrapper around load_all_years that accepts string paths.

    Parameters:
    -----------
    data_dir : str
        Path to directory containing CSV files (default: "data/raw")
    years : Optional[List[int]]
        Specific years to load. If None, loads all available years.

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame containing all requested data

    Examples:
    ---------
    >>> # Load all available years
    >>> df = load_data()

    >>> # Load specific years
    >>> df = load_data(years=[2020, 2021, 2022])

    >>> # Load from custom directory
    >>> df = load_data(data_dir="path/to/custom/data")
    """
    data_path = Path(data_dir)
    return load_all_years(data_path, years)


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/311_combined.csv") -> None:
    """
    Save processed DataFrame to CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    output_path : str
        Path where the CSV file should be saved
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(df)} records to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved successfully to {output_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("Starting data loading process")

    # Load all available data
    main_df: pd.DataFrame = load_data()

    if not main_df.empty:
        logger.info(f"Successfully loaded {len(main_df)} total records")
        logger.info(f"Columns: {main_df.columns.tolist()}")

        # Save combined data
        save_processed_data(main_df)
    else:
        logger.warning("No data loaded. Please ensure CSV files are in data/raw/")
