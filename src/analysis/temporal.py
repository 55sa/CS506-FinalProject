"""Temporal analysis for Boston 311 requests."""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_requests_per_year(df: pd.DataFrame) -> pd.Series:
    """Return total 311 requests grouped by year."""
    logger.info("Calculating requests per year")
    yearly_counts = df.groupby("year").size()
    logger.info(
        f"Years: {yearly_counts.index.tolist()}, "
        f"Range: {yearly_counts.min():,} to {yearly_counts.max():,}"
    )
    return yearly_counts


def calculate_average_daily_contacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return average daily contacts per year using actual unique dates.

    Args:
        df: DataFrame with 'year' and 'date' columns.

    Returns:
        DataFrame with 'year' and 'avg_daily_requests' columns.
    """
    logger.info("Calculating average daily contacts by year")

    daily_avg = (
        df.groupby("year")
        .apply(lambda x: len(x) / x["date"].nunique(), include_groups=False)
        .reset_index()
    )
    daily_avg.columns = ["year", "avg_daily_requests"]

    logger.info(
        f"Daily contacts range: {daily_avg['avg_daily_requests'].min():.1f} "
        f"to {daily_avg['avg_daily_requests'].max():.1f}"
    )
    return daily_avg
