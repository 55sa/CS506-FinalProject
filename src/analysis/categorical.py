"""Categorical analysis for Boston 311 requests."""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_request_types_overall(
    df: pd.DataFrame, top_n: int = 10
) -> pd.Series:
    """Return top N most common request types."""
    logger.info(f"Calculating top {top_n} request types")
    type_counts = df["type"].value_counts().head(top_n)
    logger.info(
        f"Most common: {type_counts.index[0]} ({type_counts.iloc[0]:,})"
    )
    return type_counts


def calculate_top_neighborhoods(
    df: pd.DataFrame, top_n: int = 10
) -> pd.Series:
    """Return top N neighborhoods by request volume."""
    logger.info(f"Calculating top {top_n} neighborhoods")
    neighborhood_counts = df["neighborhood"].value_counts().head(top_n)
    coverage = neighborhood_counts.sum() / df["neighborhood"].notna().sum()
    logger.info(f"Top {top_n} account for {coverage * 100:.1f}% of requests")
    return neighborhood_counts


def calculate_source_distribution(df: pd.DataFrame) -> pd.Series:
    """Return request counts by submission source."""
    logger.info("Calculating source distribution")
    source_counts = df["source"].value_counts()
    top_pct = source_counts.iloc[0] / len(df) * 100
    logger.info(
        f"{len(source_counts)} sources found, "
        f"top: {source_counts.index[0]} ({top_pct:.1f}%)"
    )
    return source_counts


def calculate_trends_by_subject(
    df: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """Return year-over-year trends for top N subjects."""
    logger.info(f"Calculating subject trends (top {top_n})")
    top_subjects = df["subject"].value_counts().head(top_n).index
    trends = (
        df[df["subject"].isin(top_subjects)]
        .groupby(["year", "subject"])
        .size()
        .unstack(fill_value=0)
    )
    logger.info(f"Calculated trends for {len(trends)} years")
    return trends


def calculate_trends_by_reason(
    df: pd.DataFrame, top_n: int = 8
) -> pd.DataFrame:
    """Return year-over-year trends for top N reasons."""
    logger.info(f"Calculating reason trends (top {top_n})")
    top_reasons = df["reason"].value_counts().head(top_n).index
    trends = (
        df[df["reason"].isin(top_reasons)]
        .groupby(["year", "reason"])
        .size()
        .unstack(fill_value=0)
    )
    logger.info(f"Calculated trends for {len(trends)} years")
    return trends


def calculate_trends_by_queue(
    df: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """Return year-over-year trends for top N queues."""
    logger.info(f"Calculating queue trends (top {top_n})")
    top_queues = df["queue"].value_counts().head(top_n).index
    trends = (
        df[df["queue"].isin(top_queues)]
        .groupby(["year", "queue"])
        .size()
        .unstack(fill_value=0)
    )
    logger.info(f"Calculated trends for {len(trends)} years")
    return trends


def calculate_request_types_by_neighborhood_data(
    df: pd.DataFrame, top_hoods: int = 5, top_types_per_hood: int = 10
) -> dict[str, pd.Series]:
    """
    Return top request types for each top neighborhood.

    Args:
        df: DataFrame with 'neighborhood' and 'type' columns.
        top_hoods: Number of top neighborhoods to analyze.
        top_types_per_hood: Number of top types per neighborhood.

    Returns:
        Dict mapping neighborhood name to Series of top request types.
    """
    logger.info(
        f"Calculating top {top_types_per_hood} types "
        f"for {top_hoods} neighborhoods"
    )
    top_neighborhoods = df["neighborhood"].value_counts().head(top_hoods).index

    result = {}
    for hood in top_neighborhoods:
        hood_df = df[df["neighborhood"] == hood]
        result[hood] = hood_df["type"].value_counts().head(top_types_per_hood)

    logger.info(f"Analyzed {len(result)} neighborhoods")
    return result


def calculate_top_types_by_year(
    df: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """Return yearly volume for top N request types."""
    logger.info(f"Calculating yearly trends for top {top_n} types")
    top_types = df["type"].value_counts().head(top_n).index
    trends = (
        df[df["type"].isin(top_types)]
        .groupby(["year", "type"])
        .size()
        .unstack(fill_value=0)
    )
    logger.info(f"Calculated trends for {len(trends)} years")
    return trends


def calculate_source_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return source distribution by year."""
    logger.info("Calculating source distribution by year")
    source_yearly = (
        df.groupby(["year", "source"]).size().unstack(fill_value=0)
    )
    logger.info(f"Calculated for {len(source_yearly)} years")
    return source_yearly


def calculate_status_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return case status breakdown by year."""
    logger.info("Calculating status breakdown by year")
    status_yearly = (
        df.groupby(["year", "case_status"]).size().unstack(fill_value=0)
    )
    logger.info(f"Calculated for {len(status_yearly)} years")
    return status_yearly
