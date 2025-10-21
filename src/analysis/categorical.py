"""
Categorical analysis module for Boston 311 Service Request Analysis.

Handles analysis of request types, neighborhoods, departments, and sources.
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
TOP_N_DEFAULT = 10


def calculate_request_types_overall(df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.Series:
    """
    Calculate most common request types overall.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'type' column
    top_n : int
        Number of top types to return (default: 10)

    Returns:
    --------
    pd.Series
        Top N request types with counts
    """
    logger.info(f"Calculating top {top_n} request types overall")

    type_counts = df['type'].value_counts().head(top_n)

    logger.info(f"Most common request type: {type_counts.index[0]} ({type_counts.iloc[0]:,} requests)")

    return type_counts


def calculate_request_types_by_neighborhood(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Calculate most common request types for each neighborhood.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'neighborhood' and 'type' columns
    top_n : int
        Number of top types per neighborhood (default: 5)

    Returns:
    --------
    pd.DataFrame
        Crosstab of neighborhoods vs request types
    """
    logger.info("Calculating request types by neighborhood")

    # Create crosstab
    crosstab = pd.crosstab(df['neighborhood'], df['type'])

    # Sort by total requests per neighborhood
    neighborhood_totals = crosstab.sum(axis=1).sort_values(ascending=False)

    logger.info(f"Analyzed {len(crosstab)} neighborhoods")
    logger.info(f"Most active neighborhood: {neighborhood_totals.index[0]} ({neighborhood_totals.iloc[0]:,} requests)")

    return crosstab


def calculate_top_neighborhoods(df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.Series:
    """
    Calculate neighborhoods with most 311 requests.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'neighborhood' column
    top_n : int
        Number of top neighborhoods to return (default: 10)

    Returns:
    --------
    pd.Series
        Top N neighborhoods with request counts
    """
    logger.info(f"Calculating top {top_n} neighborhoods by request volume")

    neighborhood_counts = df['neighborhood'].value_counts().head(top_n)

    total_requests = df['neighborhood'].notna().sum()
    top_n_percentage = (neighborhood_counts.sum() / total_requests) * 100

    logger.info(f"Top {top_n} neighborhoods account for {top_n_percentage:.1f}% of all requests")

    return neighborhood_counts


def calculate_subject_distribution(df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.Series:
    """
    Calculate distribution of requests by subject (department).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'subject' column
    top_n : int
        Number of top subjects to return (default: 10)

    Returns:
    --------
    pd.Series
        Top N subjects with request counts
    """
    logger.info(f"Calculating top {top_n} subjects (departments)")

    subject_counts = df['subject'].value_counts().head(top_n)

    logger.info(f"Most common subject: {subject_counts.index[0]} ({subject_counts.iloc[0]:,} requests)")

    return subject_counts


def calculate_reason_distribution(df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.Series:
    """
    Calculate distribution of requests by reason.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'reason' column
    top_n : int
        Number of top reasons to return (default: 10)

    Returns:
    --------
    pd.Series
        Top N reasons with request counts
    """
    logger.info(f"Calculating top {top_n} reasons")

    reason_counts = df['reason'].value_counts().head(top_n)

    logger.info(f"Most common reason: {reason_counts.index[0]} ({reason_counts.iloc[0]:,} requests)")

    return reason_counts


def calculate_queue_distribution(df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.Series:
    """
    Calculate distribution of requests by processing queue.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'queue' column
    top_n : int
        Number of top queues to return (default: 10)

    Returns:
    --------
    pd.Series
        Top N queues with request counts
    """
    logger.info(f"Calculating top {top_n} processing queues")

    queue_counts = df['queue'].value_counts().head(top_n)

    logger.info(f"Most common queue: {queue_counts.index[0]} ({queue_counts.iloc[0]:,} requests)")

    return queue_counts


def calculate_source_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Calculate distribution of requests by submission source.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'source' column

    Returns:
    --------
    pd.Series
        All sources with request counts (sorted by count)
    """
    logger.info("Calculating submission source distribution")

    source_counts = df['source'].value_counts()

    logger.info(f"Found {len(source_counts)} different submission sources")
    logger.info(f"Most common source: {source_counts.index[0]} ({source_counts.iloc[0]:,} requests, {source_counts.iloc[0]/len(df)*100:.1f}%)")

    return source_counts


def calculate_case_status_breakdown(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate percentage breakdown of case statuses (closed/open/null).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'case_status' column

    Returns:
    --------
    dict[str, Any]
        Dictionary with counts and percentages for each status
    """
    logger.info("Calculating case status breakdown")

    total = len(df)

    # Count each status (including null)
    status_counts = df['case_status'].value_counts(dropna=False)

    # Calculate percentages
    breakdown = {}
    for status, count in status_counts.items():
        status_label = 'null' if pd.isna(status) else status
        breakdown[status_label] = {
            'count': count,
            'percentage': (count / total) * 100
        }

    logger.info("Case status breakdown:")
    for status, data in breakdown.items():
        logger.info(f"  {status}: {data['count']:,} ({data['percentage']:.1f}%)")

    return breakdown


def calculate_trends_by_subject(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Calculate year-over-year trends for top subjects.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year' and 'subject' columns
    top_n : int
        Number of top subjects to analyze (default: 5)

    Returns:
    --------
    pd.DataFrame
        Pivot table with years as rows, subjects as columns
    """
    logger.info(f"Calculating trends for top {top_n} subjects")

    # Get top N subjects
    top_subjects = df['subject'].value_counts().head(top_n).index

    # Filter to top subjects
    df_filtered = df[df['subject'].isin(top_subjects)]

    # Create pivot table
    trends = (df_filtered
              .groupby(['year', 'subject'])
              .size()
              .unstack(fill_value=0))

    logger.info(f"Calculated trends for {len(trends)} years")

    return trends


def calculate_trends_by_queue(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Calculate year-over-year trends for top queues.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year' and 'queue' columns
    top_n : int
        Number of top queues to analyze (default: 5)

    Returns:
    --------
    pd.DataFrame
        Pivot table with years as rows, queues as columns
    """
    logger.info(f"Calculating trends for top {top_n} queues")

    # Get top N queues
    top_queues = df['queue'].value_counts().head(top_n).index

    # Filter to top queues
    df_filtered = df[df['queue'].isin(top_queues)]

    # Create pivot table
    trends = (df_filtered
              .groupby(['year', 'queue'])
              .size()
              .unstack(fill_value=0))

    logger.info(f"Calculated trends for {len(trends)} years")

    return trends


def calculate_neighborhood_yearly_trends(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Calculate year-over-year trends for top neighborhoods.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'year' and 'neighborhood' columns
    top_n : int
        Number of top neighborhoods to analyze (default: 5)

    Returns:
    --------
    pd.DataFrame
        Pivot table with years as rows, neighborhoods as columns
    """
    logger.info(f"Calculating trends for top {top_n} neighborhoods")

    # Get top N neighborhoods
    top_neighborhoods = df['neighborhood'].value_counts().head(top_n).index

    # Filter to top neighborhoods
    df_filtered = df[df['neighborhood'].isin(top_neighborhoods)]

    # Create pivot table
    trends = (df_filtered
              .groupby(['year', 'neighborhood'])
              .size()
              .unstack(fill_value=0))

    logger.info(f"Calculated trends for {len(trends)} years")

    return trends


def calculate_categorical_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate comprehensive categorical analysis summary.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame

    Returns:
    --------
    dict[str, Any]
        Dictionary containing all categorical metrics
    """
    logger.info("Generating comprehensive categorical summary")

    summary = {
        'unique_neighborhoods': df['neighborhood'].nunique(),
        'unique_types': df['type'].nunique(),
        'unique_subjects': df['subject'].nunique(),
        'unique_reasons': df['reason'].nunique(),
        'unique_queues': df['queue'].nunique(),
        'unique_sources': df['source'].nunique(),
        'top_type': df['type'].value_counts().index[0],
        'top_neighborhood': df['neighborhood'].value_counts().index[0],
        'top_subject': df['subject'].value_counts().index[0],
        'top_source': df['source'].value_counts().index[0]
    }

    logger.info("Categorical summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    return summary


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_data
    from src.data.preprocessor import preprocess_data

    logger.info("Running categorical analysis")

    # Load and preprocess data
    raw_df = load_data()

    if not raw_df.empty:
        df = preprocess_data(raw_df)

        # Run analyses
        top_types = calculate_request_types_overall(df, top_n=10)
        top_neighborhoods = calculate_top_neighborhoods(df, top_n=10)
        subject_dist = calculate_subject_distribution(df)
        source_dist = calculate_source_distribution(df)
        status_breakdown = calculate_case_status_breakdown(df)
        summary = calculate_categorical_summary(df)

        logger.info("Categorical analysis complete")
    else:
        logger.error("No data available for analysis")
