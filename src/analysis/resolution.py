"""
Resolution time analysis module for Boston 311 Service Request Analysis.

Handles calculation and analysis of case resolution times by various dimensions.
No visualization - only computes metrics and aggregations.
"""

from __future__ import annotations


import logging
import pandas as pd
import numpy as np
from typing import Any

# Setup logging
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
HOURS_PER_DAY = 24
OUTLIER_THRESHOLD_DAYS = 365  # Consider resolutions over 1 year as outliers


def calculate_average_resolution_by_queue(df: pd.DataFrame,
                                         exclude_outliers: bool = True,
                                         outlier_days: int = OUTLIER_THRESHOLD_DAYS) -> pd.DataFrame:
    """
    Calculate average resolution time by processing queue.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'queue' and 'resolution_hours' columns
    exclude_outliers : bool
        Whether to exclude extreme outliers (default: True)
    outlier_days : int
        Threshold in days for outlier exclusion (default: 365)

    Returns:
    --------
    pd.DataFrame
        Statistics by queue (mean, median, count)
    """
    logger.info("Calculating average resolution time by queue")

    # Filter to resolved cases
    resolved = df[df['resolution_hours'].notna()].copy()

    # Optionally exclude outliers
    if exclude_outliers:
        outlier_hours = outlier_days * HOURS_PER_DAY
        initial_count = len(resolved)
        resolved = resolved[resolved['resolution_hours'] <= outlier_hours]
        excluded = initial_count - len(resolved)
        if excluded > 0:
            logger.info(f"Excluded {excluded} outliers (>{outlier_days} days)")

    # Calculate statistics by queue
    queue_stats = (resolved
                   .groupby('queue')['resolution_hours']
                   .agg(['mean', 'median', 'count'])
                   .sort_values('mean'))

    # Convert to days for readability
    queue_stats['mean_days'] = queue_stats['mean'] / HOURS_PER_DAY
    queue_stats['median_days'] = queue_stats['median'] / HOURS_PER_DAY

    logger.info(f"Analyzed {len(queue_stats)} different queues")
    logger.info(f"Fastest queue: {queue_stats.index[0]} ({queue_stats.iloc[0]['mean_days']:.1f} days avg)")
    logger.info(f"Slowest queue: {queue_stats.index[-1]} ({queue_stats.iloc[-1]['mean_days']:.1f} days avg)")

    return queue_stats


def calculate_average_resolution_by_neighborhood(df: pd.DataFrame,
                                                exclude_outliers: bool = True,
                                                outlier_days: int = OUTLIER_THRESHOLD_DAYS) -> pd.DataFrame:
    """
    Calculate average resolution time by neighborhood.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'neighborhood' and 'resolution_hours' columns
    exclude_outliers : bool
        Whether to exclude extreme outliers (default: True)
    outlier_days : int
        Threshold in days for outlier exclusion (default: 365)

    Returns:
    --------
    pd.DataFrame
        Statistics by neighborhood (mean, median, count)
    """
    logger.info("Calculating average resolution time by neighborhood")

    # Filter to resolved cases
    resolved = df[df['resolution_hours'].notna()].copy()

    # Optionally exclude outliers
    if exclude_outliers:
        outlier_hours = outlier_days * HOURS_PER_DAY
        initial_count = len(resolved)
        resolved = resolved[resolved['resolution_hours'] <= outlier_hours]
        excluded = initial_count - len(resolved)
        if excluded > 0:
            logger.info(f"Excluded {excluded} outliers (>{outlier_days} days)")

    # Calculate statistics by neighborhood
    neighborhood_stats = (resolved
                         .groupby('neighborhood')['resolution_hours']
                         .agg(['mean', 'median', 'count'])
                         .sort_values('mean'))

    # Convert to days for readability
    neighborhood_stats['mean_days'] = neighborhood_stats['mean'] / HOURS_PER_DAY
    neighborhood_stats['median_days'] = neighborhood_stats['median'] / HOURS_PER_DAY

    logger.info(f"Analyzed {len(neighborhood_stats)} neighborhoods")
    logger.info(f"Fastest neighborhood: {neighborhood_stats.index[0]} ({neighborhood_stats.iloc[0]['mean_days']:.1f} days avg)")
    logger.info(f"Slowest neighborhood: {neighborhood_stats.index[-1]} ({neighborhood_stats.iloc[-1]['mean_days']:.1f} days avg)")

    return neighborhood_stats


def calculate_resolution_by_type(df: pd.DataFrame,
                                 exclude_outliers: bool = True,
                                 outlier_days: int = OUTLIER_THRESHOLD_DAYS,
                                 top_n: int = 20) -> pd.DataFrame:
    """
    Calculate average resolution time by request type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'type' and 'resolution_hours' columns
    exclude_outliers : bool
        Whether to exclude extreme outliers (default: True)
    outlier_days : int
        Threshold in days for outlier exclusion (default: 365)
    top_n : int
        Number of most common types to analyze (default: 20)

    Returns:
    --------
    pd.DataFrame
        Statistics by request type (mean, median, count)
    """
    logger.info(f"Calculating average resolution time by request type (top {top_n})")

    # Filter to resolved cases
    resolved = df[df['resolution_hours'].notna()].copy()

    # Optionally exclude outliers
    if exclude_outliers:
        outlier_hours = outlier_days * HOURS_PER_DAY
        initial_count = len(resolved)
        resolved = resolved[resolved['resolution_hours'] <= outlier_hours]
        excluded = initial_count - len(resolved)
        if excluded > 0:
            logger.info(f"Excluded {excluded} outliers (>{outlier_days} days)")

    # Get top N most common types
    top_types = resolved['type'].value_counts().head(top_n).index
    resolved_filtered = resolved[resolved['type'].isin(top_types)]

    # Calculate statistics by type
    type_stats = (resolved_filtered
                 .groupby('type')['resolution_hours']
                 .agg(['mean', 'median', 'count'])
                 .sort_values('mean'))

    # Convert to days for readability
    type_stats['mean_days'] = type_stats['mean'] / HOURS_PER_DAY
    type_stats['median_days'] = type_stats['median'] / HOURS_PER_DAY

    logger.info(f"Fastest type: {type_stats.index[0]} ({type_stats.iloc[0]['mean_days']:.1f} days avg)")
    logger.info(f"Slowest type: {type_stats.index[-1]} ({type_stats.iloc[-1]['mean_days']:.1f} days avg)")

    return type_stats


def calculate_resolution_percentiles(df: pd.DataFrame,
                                     exclude_outliers: bool = True,
                                     outlier_days: int = OUTLIER_THRESHOLD_DAYS) -> pd.Series:
    """
    Calculate resolution time percentiles.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'resolution_hours' column
    exclude_outliers : bool
        Whether to exclude extreme outliers (default: True)
    outlier_days : int
        Threshold in days for outlier exclusion (default: 365)

    Returns:
    --------
    pd.Series
        Percentile values (10th, 25th, 50th, 75th, 90th, 95th, 99th)
    """
    logger.info("Calculating resolution time percentiles")

    # Filter to resolved cases
    resolved = df[df['resolution_hours'].notna()].copy()

    # Optionally exclude outliers
    if exclude_outliers:
        outlier_hours = outlier_days * HOURS_PER_DAY
        initial_count = len(resolved)
        resolved = resolved[resolved['resolution_hours'] <= outlier_hours]
        excluded = initial_count - len(resolved)
        if excluded > 0:
            logger.info(f"Excluded {excluded} outliers (>{outlier_days} days)")

    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = resolved['resolution_hours'].quantile([p/100 for p in percentiles])
    percentile_values.index = [f'{p}th' for p in percentiles]

    # Convert to days
    percentile_days = percentile_values / HOURS_PER_DAY

    logger.info("Resolution time percentiles (days):")
    for label, value in percentile_days.items():
        logger.info(f"  {label}: {value:.1f} days")

    return percentile_days


def calculate_resolution_rate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate what percentage of cases are resolved each year.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year' and 'resolution_hours' columns

    Returns:
    --------
    pd.DataFrame
        Year-by-year resolution rates
    """
    logger.info("Calculating resolution rate by year")

    # Group by year
    yearly_stats = df.groupby('year').agg({
        'resolution_hours': ['count', lambda x: x.notna().sum()]
    })

    yearly_stats.columns = ['total_cases', 'resolved_cases']
    yearly_stats['resolution_rate'] = (yearly_stats['resolved_cases'] /
                                       yearly_stats['total_cases'] * 100)

    logger.info("Resolution rates by year:")
    for year, row in yearly_stats.iterrows():
        logger.info(f"  {year}: {row['resolution_rate']:.1f}% ({row['resolved_cases']}/{row['total_cases']})")

    return yearly_stats


def calculate_resolution_trends_by_queue(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Calculate year-over-year resolution time trends for top queues.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year', 'queue', and 'resolution_hours' columns
    top_n : int
        Number of top queues to analyze (default: 5)

    Returns:
    --------
    pd.DataFrame
        Pivot table with years as rows, queues as columns (median resolution in days)
    """
    logger.info(f"Calculating resolution time trends for top {top_n} queues")

    # Filter to resolved cases
    resolved = df[df['resolution_hours'].notna()].copy()

    # Get top N queues by volume
    top_queues = resolved['queue'].value_counts().head(top_n).index
    resolved_filtered = resolved[resolved['queue'].isin(top_queues)]

    # Convert to days
    resolved_filtered['resolution_days'] = resolved_filtered['resolution_hours'] / HOURS_PER_DAY

    # Calculate median by year and queue
    trends = (resolved_filtered
              .groupby(['year', 'queue'])['resolution_days']
              .median()
              .unstack())

    logger.info(f"Calculated trends for {len(trends)} years")

    return trends


def calculate_resolution_summary(df: pd.DataFrame,
                                exclude_outliers: bool = True,
                                outlier_days: int = OUTLIER_THRESHOLD_DAYS) -> dict[str, Any]:
    """
    Generate comprehensive resolution time summary statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame with resolution_hours
    exclude_outliers : bool
        Whether to exclude extreme outliers (default: True)
    outlier_days : int
        Threshold in days for outlier exclusion (default: 365)

    Returns:
    --------
    dict[str, Any]
        Dictionary containing resolution metrics
    """
    logger.info("Calculating comprehensive resolution summary")

    # Filter to resolved cases
    resolved = df[df['resolution_hours'].notna()].copy()

    total_cases = len(df)
    resolved_cases = len(resolved)
    resolution_rate = (resolved_cases / total_cases) * 100

    # Optionally exclude outliers
    if exclude_outliers:
        outlier_hours = outlier_days * HOURS_PER_DAY
        resolved_clean = resolved[resolved['resolution_hours'] <= outlier_hours]
        outliers = len(resolved) - len(resolved_clean)
    else:
        resolved_clean = resolved
        outliers = 0

    summary = {
        'total_cases': total_cases,
        'resolved_cases': resolved_cases,
        'resolution_rate': resolution_rate,
        'outliers_excluded': outliers if exclude_outliers else 0,
        'mean_hours': resolved_clean['resolution_hours'].mean(),
        'mean_days': resolved_clean['resolution_hours'].mean() / HOURS_PER_DAY,
        'median_hours': resolved_clean['resolution_hours'].median(),
        'median_days': resolved_clean['resolution_hours'].median() / HOURS_PER_DAY,
        'min_hours': resolved_clean['resolution_hours'].min(),
        'max_hours': resolved_clean['resolution_hours'].max(),
        'std_hours': resolved_clean['resolution_hours'].std()
    }

    logger.info("Resolution summary:")
    logger.info(f"  Resolution rate: {summary['resolution_rate']:.1f}%")
    logger.info(f"  Mean resolution time: {summary['mean_days']:.1f} days")
    logger.info(f"  Median resolution time: {summary['median_days']:.1f} days")

    return summary


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_data
    from src.data.preprocessor import preprocess_data

    logger.info("Running resolution time analysis")

    # Load and preprocess data
    raw_df = load_data()

    if not raw_df.empty:
        df = preprocess_data(raw_df)

        # Run analyses
        queue_stats = calculate_average_resolution_by_queue(df)
        neighborhood_stats = calculate_average_resolution_by_neighborhood(df)
        type_stats = calculate_resolution_by_type(df)
        percentiles = calculate_resolution_percentiles(df)
        summary = calculate_resolution_summary(df)

        logger.info("Resolution analysis complete")
    else:
        logger.error("No data available for analysis")
