"""Resolution time analysis for Boston 311 requests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

HOURS_PER_DAY = 24
OUTLIER_THRESHOLD_DAYS = 365


def calculate_average_resolution_by_queue(
    df: pd.DataFrame,
    exclude_outliers: bool = True,
    outlier_days: int = OUTLIER_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """
    Return average resolution time statistics by queue.

    Args:
        df: DataFrame with 'queue' and 'resolution_hours' columns.
        exclude_outliers: Whether to exclude cases over outlier_days.
        outlier_days: Threshold in days for outlier exclusion.

    Returns:
        DataFrame with queue stats (mean, median, count, mean_days, median_days).
    """
    logger.info("Calculating average resolution by queue")
    resolved = df[df["resolution_hours"].notna()].copy()

    if exclude_outliers:
        outlier_hours = outlier_days * HOURS_PER_DAY
        initial = len(resolved)
        resolved = resolved[resolved["resolution_hours"] <= outlier_hours]
        excluded = initial - len(resolved)
        if excluded > 0:
            logger.info(f"Excluded {excluded} outliers (>{outlier_days} days)")

    queue_stats = (
        resolved.groupby("queue")["resolution_hours"]
        .agg(["mean", "median", "count"])
        .sort_values("mean")
    )
    queue_stats["mean_days"] = queue_stats["mean"] / HOURS_PER_DAY
    queue_stats["median_days"] = queue_stats["median"] / HOURS_PER_DAY

    logger.info(
        f"Analyzed {len(queue_stats)} queues, "
        f"fastest: {queue_stats.index[0]} ({queue_stats.iloc[0]['mean_days']:.1f}d)"
    )
    return queue_stats


def calculate_resolution_heatmap_data(
    df: pd.DataFrame,
    top_queues: int = 10,
    top_neighborhoods: int = 10,
    exclude_outliers: bool = True,
    outlier_days: int = OUTLIER_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """
    Return resolution time heatmap data (queue × neighborhood).

    Args:
        df: DataFrame with 'queue', 'neighborhood', 'resolution_hours'.
        top_queues: Number of top queues to include.
        top_neighborhoods: Number of top neighborhoods to include.
        exclude_outliers: Whether to exclude extreme outliers.
        outlier_days: Threshold in days for outlier exclusion.

    Returns:
        Pivot table with neighborhoods as rows, queues as columns,
        average resolution days as values.
    """
    logger.info(f"Calculating heatmap for top {top_queues}×{top_neighborhoods}")
    resolved = df[df["resolution_hours"].notna()].copy()
    resolved["resolution_days"] = resolved["resolution_hours"] / HOURS_PER_DAY

    if exclude_outliers:
        initial = len(resolved)
        resolved = resolved[resolved["resolution_days"] <= outlier_days]
        if (excluded := initial - len(resolved)) > 0:
            logger.info(f"Excluded {excluded} outliers")

    top_queue_list = resolved["queue"].value_counts().head(top_queues).index
    top_hood_list = resolved["neighborhood"].value_counts().head(top_neighborhoods).index

    heatmap_df = resolved[
        resolved["queue"].isin(top_queue_list) & resolved["neighborhood"].isin(top_hood_list)
    ]

    heatmap_data = (
        heatmap_df.groupby(["neighborhood", "queue"])["resolution_days"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    logger.info(f"Heatmap shape: {heatmap_data.shape}")
    return heatmap_data
