"""
Temporal analysis module for Boston 311 Service Request Analysis.

Handles year-over-year trends, daily averages, and time-based patterns.
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
DAYS_PER_YEAR = 365


def calculate_requests_per_year(df: pd.DataFrame) -> pd.Series:
    """
    Calculate total number of 311 requests per year.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year' column

    Returns:
    --------
    pd.Series
        Index: year, Values: request count
    """
    logger.info("Calculating requests per year")

    yearly_counts = df.groupby('year').size()

    logger.info(f"Found data for years: {yearly_counts.index.tolist()}")
    logger.info(f"Total requests range: {yearly_counts.min()} to {yearly_counts.max()}")

    return yearly_counts


def calculate_average_daily_contacts(df: pd.DataFrame) -> pd.Series:
    """
    Calculate average daily contacts per year.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year' column

    Returns:
    --------
    pd.Series
        Index: year, Values: average daily contacts
    """
    logger.info("Calculating average daily contacts by year")

    # Count requests per year
    yearly_counts = df.groupby('year').size()

    # Calculate average daily contacts (assuming 365 days per year)
    avg_daily = yearly_counts / DAYS_PER_YEAR

    logger.info(f"Average daily contacts range: {avg_daily.min():.1f} to {avg_daily.max():.1f}")

    return avg_daily


def calculate_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly request trends across all years.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year' and 'month' columns

    Returns:
    --------
    pd.DataFrame
        Pivot table with years as rows, months as columns
    """
    logger.info("Calculating monthly trends")

    # Group by year and month
    monthly_counts = (df
                      .groupby(['year', 'month'])
                      .size()
                      .reset_index(name='count'))

    # Pivot to create year x month matrix
    pivot = monthly_counts.pivot(index='year', columns='month', values='count')

    logger.info(f"Monthly trends calculated for {len(pivot)} years")

    return pivot


def calculate_day_of_week_patterns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate request volume by day of week (aggregated across all years).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'day_of_week' column

    Returns:
    --------
    pd.Series
        Index: day name, Values: total requests
    """
    logger.info("Calculating day of week patterns")

    # Define proper day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Count by day of week
    dow_counts = df['day_of_week'].value_counts()

    # Reindex to proper order
    dow_counts = dow_counts.reindex(day_order)

    logger.info(f"Busiest day: {dow_counts.idxmax()} ({dow_counts.max():,} requests)")
    logger.info(f"Quietest day: {dow_counts.idxmin()} ({dow_counts.min():,} requests)")

    return dow_counts


def calculate_hourly_patterns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate request volume by hour of day (aggregated across all years).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'hour' column

    Returns:
    --------
    pd.Series
        Index: hour (0-23), Values: total requests
    """
    logger.info("Calculating hourly patterns")

    hourly_counts = (df
                     .groupby('hour')
                     .size()
                     .sort_index())

    peak_hour = hourly_counts.idxmax()
    logger.info(f"Peak hour: {peak_hour}:00 ({hourly_counts.max():,} requests)")

    return hourly_counts


def calculate_yearly_growth_rate(df: pd.DataFrame) -> pd.Series:
    """
    Calculate year-over-year growth rate in request volume.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year' column

    Returns:
    --------
    pd.Series
        Index: year, Values: percent change from previous year
    """
    logger.info("Calculating year-over-year growth rates")

    yearly_counts = df.groupby('year').size()

    # Calculate percent change
    growth_rate = yearly_counts.pct_change() * 100

    # Log significant changes
    for year, rate in growth_rate.items():
        if pd.notna(rate):
            direction = "increase" if rate > 0 else "decrease"
            logger.info(f"{year}: {abs(rate):.1f}% {direction}")

    return growth_rate


def calculate_seasonal_patterns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Calculate seasonal patterns by grouping months into seasons.

    Winter: Dec, Jan, Feb
    Spring: Mar, Apr, May
    Summer: Jun, Jul, Aug
    Fall: Sep, Oct, Nov

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'month' and 'year' columns

    Returns:
    --------
    dict[str, pd.Series]
        Dictionary with 'overall' (all years) and 'by_year' breakdowns
    """
    logger.info("Calculating seasonal patterns")

    # Define season mapping
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }

    df_temp = df.copy()
    df_temp['season'] = df_temp['month'].map(season_map)

    # Overall seasonal counts
    overall = df_temp['season'].value_counts()
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    overall = overall.reindex(season_order)

    # Seasonal counts by year
    by_year = (df_temp
               .groupby(['year', 'season'])
               .size()
               .unstack(fill_value=0))

    logger.info(f"Busiest season overall: {overall.idxmax()} ({overall.max():,} requests)")

    return {
        'overall': overall,
        'by_year': by_year
    }


def calculate_trend_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate comprehensive temporal statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame with temporal features

    Returns:
    --------
    dict[str, Any]
        Dictionary containing various temporal metrics
    """
    logger.info("Calculating comprehensive trend statistics")

    stats = {
        'total_requests': len(df),
        'date_range': f"{df['open_dt'].min()} to {df['open_dt'].max()}",
        'years_covered': df['year'].nunique(),
        'yearly_average': df.groupby('year').size().mean(),
        'yearly_median': df.groupby('year').size().median(),
        'peak_year': df['year'].value_counts().idxmax(),
        'peak_year_count': df['year'].value_counts().max(),
        'busiest_month': df['month'].value_counts().idxmax(),
        'busiest_day_of_week': df['day_of_week'].value_counts().idxmax(),
        'busiest_hour': df['hour'].value_counts().idxmax()
    }

    logger.info("Temporal statistics summary:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    return stats


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_data
    from src.data.preprocessor import preprocess_data

    logger.info("Running temporal analysis")

    # Load and preprocess data
    raw_df = load_data()

    if not raw_df.empty:
        df = preprocess_data(raw_df)

        # Run analyses
        yearly_counts = calculate_requests_per_year(df)
        avg_daily = calculate_average_daily_contacts(df)
        dow_patterns = calculate_day_of_week_patterns(df)
        hourly_patterns = calculate_hourly_patterns(df)
        seasonal_patterns = calculate_seasonal_patterns(df)
        stats = calculate_trend_statistics(df)

        logger.info("Temporal analysis complete")
    else:
        logger.error("No data available for analysis")
