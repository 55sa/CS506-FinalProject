#!/usr/bin/env python3
"""
Core Analysis Script for Boston 311 Service Request Data.

This script performs comprehensive analysis covering all core analytics goals:
1. Total volume of requests per year
2. Most common request types (overall and by neighborhood)
3. Trends by SUBJECT, REASON, and QUEUE
4. Case volume by submission channel (SOURCE)
5. Average daily contacts by year
6. Top 5 request types volume
7. Average resolution time by QUEUE
8. Average resolution time by QUEUE and neighborhood
9. Case status breakdown (Closed, Open, No Data)

Generates 15 publication-quality visualizations saved to outputs/figures/

This script uses the modular analysis functions from src/analysis/ and
visualization functions from src/visualization/ for clean separation of concerns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.analysis.temporal import (
    calculate_requests_per_year,
    calculate_average_daily_contacts,
)
from src.analysis.categorical import (
    calculate_request_types_overall,
    calculate_top_neighborhoods,
    calculate_trends_by_subject,
    calculate_trends_by_reason,
    calculate_trends_by_queue,
    calculate_source_distribution,
    calculate_request_types_by_neighborhood_data,
    calculate_top_types_by_year,
    calculate_source_by_year,
    calculate_status_by_year,
)
from src.analysis.resolution import (
    calculate_average_resolution_by_queue,
    calculate_resolution_heatmap_data,
)
from src.visualization.core_plots import (
    plot_yearly_requests,
    plot_top_request_types,
    plot_request_types_by_neighborhood,
    plot_trends_by_subject,
    plot_trends_by_reason,
    plot_trends_by_queue,
    plot_volume_by_source,
    plot_avg_daily_contacts,
    plot_top5_types_volume,
    plot_resolution_by_queue,
    plot_resolution_heatmap,
    plot_case_status_breakdown,
    plot_top_neighborhoods,
    plot_resolution_distribution,
    plot_status_yearly_trends,
)


def main() -> None:
    """Run comprehensive analysis covering all core analytics goals."""
    logger.info("=" * 80)
    logger.info("Boston 311 Comprehensive Analysis - Core Analytics Goals")
    logger.info("=" * 80)

    # Load and preprocess data
    logger.info("\nLoading all data (2011-2025)...")
    df = load_data()

    if df.empty:
        logger.error("No data loaded!")
        return

    logger.info(f"Loaded {len(df):,} records")

    logger.info("\nPreprocessing data...")
    df = preprocess_data(df)

    logger.info(f"After preprocessing: {len(df):,} records")
    logger.info(f"Date range: {df['open_dt'].min()} to {df['open_dt'].max()}")

    # Create output directory
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Run all analyses using modular functions
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING CORE ANALYSES")
    logger.info("=" * 80)

    # Analysis 1: Yearly requests
    yearly = calculate_requests_per_year(df)

    # Analysis 2: Top request types
    top_types = calculate_request_types_overall(df, top_n=20)

    # Analysis 3: Request types by neighborhood
    neighborhood_types = calculate_request_types_by_neighborhood_data(df, top_hoods=5, top_types_per_hood=10)

    # Analysis 4: Subject trends
    subject_trends = calculate_trends_by_subject(df, top_n=6)

    # Analysis 5: Reason trends
    reason_trends = calculate_trends_by_reason(df, top_n=8)

    # Analysis 6: Queue trends
    queue_trends = calculate_trends_by_queue(df, top_n=8)

    # Analysis 7: Source by year
    source_yearly = calculate_source_by_year(df)

    # Analysis 8: Average daily contacts
    daily_avg = calculate_average_daily_contacts(df)

    # Analysis 9: Top 5 types by year
    top5_yearly = calculate_top_types_by_year(df, top_n=5)

    # Analysis 10: Resolution by queue
    queue_resolution = calculate_average_resolution_by_queue(df, exclude_outliers=True)
    queue_resolution = queue_resolution[queue_resolution["count"] >= 100].sort_values("mean_days", ascending=False).head(15)

    # Analysis 11: Resolution heatmap
    heatmap_data = calculate_resolution_heatmap_data(df, top_queues=10, top_neighborhoods=10, exclude_outliers=True)

    # Analysis 12: Case status breakdown
    closed_count = len(df[df["case_status"] == "Closed"])
    open_count = len(df[df["case_status"] == "Open"])
    null_count = len(df[df["case_status"].isna()])

    status_data = pd.DataFrame({
        "Status": ["Closed", "Open", "No Data (Null)"],
        "Count": [closed_count, open_count, null_count],
        "Percentage": [
            closed_count / len(df) * 100,
            open_count / len(df) * 100,
            null_count / len(df) * 100
        ]
    })

    # Analysis 13: Top neighborhoods
    top_hoods = calculate_top_neighborhoods(df, top_n=20)

    # Analysis 14: Resolution distribution (box plot data)
    resolved_df = df[df["resolution_hours"].notna()].copy()
    resolved_df["resolution_days"] = resolved_df["resolution_hours"] / 24
    resolved_df = resolved_df[resolved_df["resolution_days"] <= 365]
    top_queues_box = queue_resolution.head(8).index.tolist()
    box_data = resolved_df[resolved_df["queue"].isin(top_queues_box)]

    # Analysis 15: Status by year
    status_yearly = calculate_status_by_year(df)

    # ========================================================================
    # Generate all visualizations
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80 + "\n")

    logger.info("[1/15] Total volume of requests per year...")
    plot_yearly_requests(yearly, output_dir)

    logger.info("[2/15] Most common request types overall...")
    plot_top_request_types(top_types, output_dir)

    logger.info("[3/15] Most common request types by neighborhood...")
    plot_request_types_by_neighborhood(neighborhood_types, output_dir)

    logger.info("[4/15] Request volume trends by SUBJECT (department)...")
    plot_trends_by_subject(subject_trends, output_dir)

    logger.info("[5/15] Request volume trends by REASON...")
    plot_trends_by_reason(reason_trends, output_dir)

    logger.info("[6/15] Request volume trends by QUEUE...")
    plot_trends_by_queue(queue_trends, output_dir)

    logger.info("[7/15] Case volume by submission channel (SOURCE)...")
    plot_volume_by_source(source_yearly, output_dir)

    logger.info("[8/15] Average daily contacts by year...")
    plot_avg_daily_contacts(daily_avg, output_dir)

    logger.info("[9/15] Volume of top 5 request types...")
    plot_top5_types_volume(top5_yearly, output_dir)

    logger.info("[10/15] Average resolution time by QUEUE...")
    plot_resolution_by_queue(queue_resolution, output_dir)

    logger.info("[11/15] Average resolution time by QUEUE and neighborhood...")
    plot_resolution_heatmap(heatmap_data, output_dir)

    logger.info("[12/15] Case status breakdown...")
    plot_case_status_breakdown(status_data, output_dir)

    logger.info("[13/15] Top neighborhoods by request volume...")
    plot_top_neighborhoods(top_hoods, output_dir)

    logger.info("[14/15] Resolution time distribution by top queues...")
    plot_resolution_distribution(box_data, output_dir)

    logger.info("[15/15] Year-over-year case status breakdown...")
    plot_status_yearly_trends(status_yearly, output_dir)

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE SUMMARY STATISTICS (2011-2025)")
    logger.info("=" * 80)

    logger.info(f"\n📊 DATASET OVERVIEW:")
    logger.info(f"  Total Records:          {len(df):,}")
    logger.info(f"  Date Range:             {df['open_dt'].min().date()} to {df['open_dt'].max().date()}")
    logger.info(f"  Years Covered:          {df['year'].min():.0f} - {df['year'].max():.0f}")
    logger.info(f"  Unique Neighborhoods:   {df['neighborhood'].nunique():,}")
    logger.info(f"  Unique Request Types:   {df['type'].nunique():,}")
    logger.info(f"  Unique Subjects:        {df['subject'].nunique():,}")
    logger.info(f"  Unique Queues:          {df['queue'].nunique():,}")

    logger.info(f"\n📈 VOLUME METRICS:")
    logger.info(f"  Peak Year:              {yearly.idxmax():.0f} ({yearly.max():,} requests)")
    logger.info(f"  Lowest Year:            {yearly.idxmin():.0f} ({yearly.min():,} requests)")
    logger.info(f"  Avg Requests/Year:      {yearly.mean():,.0f}")
    logger.info(f"  Avg Daily Requests:     {len(df) / df['date'].nunique():.0f}")

    logger.info(f"\n🔝 TOP REQUEST TYPES:")
    for i, (type_name, count) in enumerate(df["type"].value_counts().head(5).items(), 1):
        pct = count / len(df) * 100
        logger.info(f"  {i}. {type_name}: {count:,} ({pct:.1f}%)")

    logger.info(f"\n🏘️  TOP NEIGHBORHOODS:")
    for i, (hood, count) in enumerate(df["neighborhood"].value_counts().head(5).items(), 1):
        pct = count / len(df) * 100
        logger.info(f"  {i}. {hood}: {count:,} ({pct:.1f}%)")

    logger.info(f"\n📱 SUBMISSION CHANNELS:")
    source_dist = calculate_source_distribution(df)
    for source, count in source_dist.head(5).items():
        pct = count / len(df) * 100
        logger.info(f"  {source}: {count:,} ({pct:.1f}%)")

    logger.info(f"\n✅ CASE STATUS:")
    logger.info(f"  Closed:                 {closed_count:,} ({status_data.iloc[0]['Percentage']:.2f}%)")
    logger.info(f"  Open:                   {open_count:,} ({status_data.iloc[1]['Percentage']:.2f}%)")
    logger.info(f"  No Data (Null):         {null_count:,} ({status_data.iloc[2]['Percentage']:.2f}%)")

    logger.info(f"\n⏱️  RESOLUTION TIMES:")
    logger.info(f"  Cases with resolution:  {len(resolved_df):,}")
    logger.info(f"  Avg Resolution:         {resolved_df['resolution_days'].mean():.1f} days")
    logger.info(f"  Median Resolution:      {resolved_df['resolution_days'].median():.1f} days")
    logger.info(f"  Fastest Queue (avg):    {queue_resolution['mean_days'].idxmin()} ({queue_resolution['mean_days'].min():.1f} days)")
    logger.info(f"  Slowest Queue (avg):    {queue_resolution['mean_days'].idxmax()} ({queue_resolution['mean_days'].max():.1f} days)")

    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL 15 CORE ANALYTICS VISUALIZATIONS GENERATED SUCCESSFULLY!")
    logger.info(f"📊 Check {output_dir}/ for all PNG files")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
