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
visualization helpers for clean separation of concerns.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# Setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    calculate_case_status_breakdown,
    calculate_source_distribution,
)
from src.analysis.resolution import (
    calculate_average_resolution_by_queue,
)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100


def plot_yearly_requests(yearly: pd.Series, output_dir: Path) -> None:
    """Plot total requests per year."""
    plt.figure(figsize=(14, 7))
    plt.plot(yearly.index, yearly.values, marker="o", linewidth=3, markersize=12, color="#2E86AB")  # type: ignore
    plt.fill_between(yearly.index, yearly.values, alpha=0.2, color="#2E86AB")  # type: ignore
    plt.title("Total 311 Requests by Year (2011-2025)", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    for year, count in yearly.items():
        plt.text(year, count, f"{count:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "1_requests_per_year.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 1_requests_per_year.png")
    plt.close()


def plot_top_request_types(top_types: pd.Series, output_dir: Path) -> None:
    """Plot top request types."""
    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("viridis", len(top_types))
    plt.barh(range(len(top_types)), top_types.values, color=colors)
    plt.yticks(range(len(top_types)), top_types.index, fontsize=11)
    plt.xlabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.title("Top 20 Request Types (Overall)", fontsize=18, fontweight="bold", pad=20)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "2_top_request_types_overall.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 2_top_request_types_overall.png")
    plt.close()


def plot_request_types_by_neighborhood(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot request types for top neighborhoods."""
    top_5_hoods = calculate_top_neighborhoods(df, top_n=5).index.tolist()

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    axes = axes.flatten()

    for idx, hood in enumerate(top_5_hoods):
        hood_df = df[df["neighborhood"] == hood]
        top_types_hood = hood_df["type"].value_counts().head(10)

        axes[idx].barh(range(len(top_types_hood)), top_types_hood.values,
                       color=sns.color_palette("Set2", len(top_types_hood)))
        axes[idx].set_yticks(range(len(top_types_hood)))
        axes[idx].set_yticklabels(top_types_hood.index, fontsize=9)
        axes[idx].set_xlabel("Count", fontsize=11)
        axes[idx].set_title(f"{hood} (n={len(hood_df):,})", fontsize=12, fontweight="bold")
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis="x")

    fig.delaxes(axes[5])
    fig.suptitle("Top 10 Request Types by Neighborhood (Top 5 Neighborhoods)",
                 fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "3_request_types_by_neighborhood.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 3_request_types_by_neighborhood.png")
    plt.close()


def plot_trends_by_subject(subject_trends: pd.DataFrame, output_dir: Path) -> None:
    """Plot request volume trends by subject."""
    plt.figure(figsize=(16, 8))
    for subject in subject_trends.columns:
        plt.plot(subject_trends.index, subject_trends[subject], marker="o", linewidth=2.5,
                label=subject, markersize=8)
    plt.title("Request Volume by SUBJECT (Department) - Year over Year",
              fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Subject", fontsize=11, title_fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(output_dir / "4_trends_by_subject.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 4_trends_by_subject.png")
    plt.close()


def plot_trends_by_reason(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot request volume trends by reason."""
    top_reasons = df["reason"].value_counts().head(8).index.tolist()
    reason_yearly = df[df["reason"].isin(top_reasons)].groupby(["year", "reason"]).size().unstack(fill_value=0)

    plt.figure(figsize=(16, 9))
    for reason in reason_yearly.columns:
        plt.plot(reason_yearly.index, reason_yearly[reason], marker="s", linewidth=2.5,
                label=reason, markersize=7, alpha=0.8)
    plt.title("Request Volume by REASON - Year over Year (Top 8)",
              fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Reason", fontsize=10, title_fontsize=12, loc="best", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(output_dir / "5_trends_by_reason.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 5_trends_by_reason.png")
    plt.close()


def plot_trends_by_queue(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot request volume trends by queue."""
    top_queues = df["queue"].value_counts().head(8).index.tolist()
    queue_yearly = df[df["queue"].isin(top_queues)].groupby(["year", "queue"]).size().unstack(fill_value=0)

    plt.figure(figsize=(16, 9))
    for queue in queue_yearly.columns:
        plt.plot(queue_yearly.index, queue_yearly[queue], marker="^", linewidth=2.5,
                label=queue, markersize=7, alpha=0.8)
    plt.title("Request Volume by QUEUE - Year over Year (Top 8)",
              fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Queue", fontsize=10, title_fontsize=12, loc="best", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(output_dir / "6_trends_by_queue.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 6_trends_by_queue.png")
    plt.close()


def plot_volume_by_source(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot case volume by submission channel."""
    source_yearly = df.groupby(["year", "source"]).size().unstack(fill_value=0)

    plt.figure(figsize=(16, 8))
    source_yearly.plot(kind="bar", stacked=False, ax=plt.gca(),
                       colormap="tab10", width=0.8)  # type: ignore
    plt.title("Request Volume by Submission Channel (SOURCE) - Year over Year",
              fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Source", fontsize=11, title_fontsize=12, loc="best")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis="y")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(output_dir / "7_volume_by_source.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 7_volume_by_source.png")
    plt.close()


def plot_avg_daily_contacts(daily_avg: pd.DataFrame, output_dir: Path) -> None:
    """Plot average daily contacts by year."""
    plt.figure(figsize=(14, 7))
    plt.bar(daily_avg["year"], daily_avg["avg_daily_requests"],
            color=sns.color_palette("coolwarm", len(daily_avg)), edgecolor="black", linewidth=1.5)
    plt.title("Average Daily 311 Contacts by Year", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Average Daily Requests", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    for i, row in daily_avg.iterrows():
        plt.text(row["year"], row["avg_daily_requests"], f'{int(row["avg_daily_requests"]):,}',
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "8_avg_daily_contacts.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 8_avg_daily_contacts.png")
    plt.close()


def plot_top5_types_volume(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot volume of top 5 request types over time."""
    top_5_types = df["type"].value_counts().head(5).index.tolist()
    top5_yearly = df[df["type"].isin(top_5_types)].groupby(["year", "type"]).size().unstack(fill_value=0)

    plt.figure(figsize=(16, 8))
    top5_yearly.plot(kind="bar", ax=plt.gca(), width=0.85,
                     color=sns.color_palette("Set2", len(top_5_types)))  # type: ignore
    plt.title("Volume of Top 5 Request Types - Year over Year",
              fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Type", fontsize=11, title_fontsize=12, loc="best")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis="y")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(output_dir / "9_top5_types_volume.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 9_top5_types_volume.png")
    plt.close()


def plot_resolution_by_queue(queue_resolution: pd.DataFrame, output_dir: Path) -> None:
    """Plot average resolution time by queue."""
    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(queue_resolution))
    plt.barh(y_pos, queue_resolution["mean_days"], color=sns.color_palette("rocket", len(queue_resolution)))
    plt.yticks(y_pos, queue_resolution.index, fontsize=10)
    plt.xlabel("Average Resolution Time (Days)", fontsize=14, fontweight="bold")
    plt.title("Average Resolution Time by QUEUE (Top 15, min 100 cases)",
              fontsize=18, fontweight="bold", pad=20)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")
    for i, (idx, row) in enumerate(queue_resolution.iterrows()):
        plt.text(row["mean_days"], i, f' {row["mean_days"]:.1f}d (n={int(row["count"]):,})',
                va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "10_resolution_by_queue.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 10_resolution_by_queue.png")
    plt.close()


def plot_resolution_heatmap(df: pd.DataFrame, queue_resolution: pd.DataFrame, output_dir: Path) -> None:
    """Plot resolution time heatmap by queue and neighborhood."""
    # Get resolved cases with resolution_days
    resolved_df = df[df["resolution_hours"].notna()].copy()
    resolved_df["resolution_days"] = resolved_df["resolution_hours"] / 24
    resolved_df = resolved_df[resolved_df["resolution_days"] <= 365]

    top_queues_res = queue_resolution.head(10).index.tolist()
    top_hoods_res = df["neighborhood"].value_counts().head(10).index.tolist()

    heatmap_data = resolved_df[
        (resolved_df["queue"].isin(top_queues_res)) &
        (resolved_df["neighborhood"].isin(top_hoods_res))
    ].groupby(["neighborhood", "queue"])["resolution_days"].mean().unstack(fill_value=np.nan)

    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={"label": "Avg Resolution (Days)"}, linewidths=0.5)  # type: ignore
    plt.title("Average Resolution Time by QUEUE and Neighborhood (Days)",
              fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Queue", fontsize=14, fontweight="bold")
    plt.ylabel("Neighborhood", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "11_resolution_queue_neighborhood.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 11_resolution_queue_neighborhood.png")
    plt.close()


def plot_case_status_breakdown(status_data: pd.DataFrame, output_dir: Path) -> None:
    """Plot case status breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors_pie = ["#66C2A5", "#FC8D62", "#8DA0CB"]
    ax1.pie(status_data["Percentage"], labels=status_data["Status"], autopct="%1.2f%%",
            colors=colors_pie, startangle=90, textprops={"fontsize": 13, "fontweight": "bold"})  # type: ignore
    ax1.set_title("Case Status Distribution", fontsize=16, fontweight="bold", pad=20)

    ax2.bar(status_data["Status"], status_data["Count"], color=colors_pie, edgecolor="black", linewidth=2)
    ax2.set_ylabel("Number of Cases", fontsize=14, fontweight="bold")
    ax2.set_title("Case Status Counts", fontsize=16, fontweight="bold", pad=20)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax2.grid(True, alpha=0.3, axis="y")
    for i, row in status_data.iterrows():
        ax2.text(i, row["Count"], f'{int(row["Count"]):,}\n({row["Percentage"]:.2f}%)',
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "12_case_status_breakdown.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 12_case_status_breakdown.png")
    plt.close()


def plot_top_neighborhoods(top_hoods: pd.Series, output_dir: Path) -> None:
    """Plot top neighborhoods by request volume."""
    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("coolwarm", len(top_hoods))
    plt.barh(range(len(top_hoods)), top_hoods.values, color=colors)
    plt.yticks(range(len(top_hoods)), top_hoods.index, fontsize=11)
    plt.xlabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.title("Top 20 Neighborhoods by Request Volume (2011-2025)",
              fontsize=18, fontweight="bold", pad=20)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "13_top_neighborhoods.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 13_top_neighborhoods.png")
    plt.close()


def plot_resolution_distribution(df: pd.DataFrame, queue_resolution: pd.DataFrame, output_dir: Path) -> None:
    """Plot resolution time distribution by queue."""
    resolved_df = df[df["resolution_hours"].notna()].copy()
    resolved_df["resolution_days"] = resolved_df["resolution_hours"] / 24
    resolved_df = resolved_df[resolved_df["resolution_days"] <= 365]

    top_queues_box = queue_resolution.head(8).index.tolist()
    box_data = resolved_df[resolved_df["queue"].isin(top_queues_box)]

    plt.figure(figsize=(16, 8))
    sns.boxplot(data=box_data, x="queue", y="resolution_days", palette="Set3", hue="queue", legend=False)  # type: ignore
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.xlabel("Queue", fontsize=14, fontweight="bold")
    plt.ylabel("Resolution Time (Days)", fontsize=14, fontweight="bold")
    plt.title("Resolution Time Distribution by QUEUE (Top 8)",
              fontsize=18, fontweight="bold", pad=20)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "14_resolution_distribution.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 14_resolution_distribution.png")
    plt.close()


def plot_status_yearly_trends(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot year-over-year case status trends."""
    status_yearly = df.groupby(["year", "case_status"]).size().unstack(fill_value=0)
    status_yearly_pct = status_yearly.div(status_yearly.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    status_yearly.plot(kind="bar", stacked=True, ax=ax1,
                       color=["#66C2A5", "#FC8D62"], width=0.8)  # type: ignore
    ax1.set_title("Case Status by Year (Absolute Counts)", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Number of Cases", fontsize=13, fontweight="bold")
    ax1.legend(title="Status", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    status_yearly_pct.plot(kind="bar", stacked=True, ax=ax2,
                           color=["#66C2A5", "#FC8D62"], width=0.8)  # type: ignore
    ax2.set_title("Case Status by Year (Percentage)", fontsize=16, fontweight="bold", pad=15)
    ax2.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=13, fontweight="bold")
    ax2.legend(title="Status", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "15_status_yearly_trends.png", dpi=300, bbox_inches="tight")
    logger.info("  ✓ Saved: 15_status_yearly_trends.png")
    plt.close()


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

    # Use analysis modules
    yearly = calculate_requests_per_year(df)
    top_types = calculate_request_types_overall(df, top_n=20)
    top_hoods = calculate_top_neighborhoods(df, top_n=20)
    subject_trends = calculate_trends_by_subject(df, top_n=6)
    source_dist = calculate_source_distribution(df)

    # Calculate daily average (using groupby apply)
    daily_avg = df.groupby("year").apply(lambda x: len(x) / x["date"].nunique()).reset_index()
    daily_avg.columns = ["year", "avg_daily_requests"]

    # Resolution analysis
    queue_resolution = calculate_average_resolution_by_queue(df, exclude_outliers=True)
    queue_resolution = queue_resolution[queue_resolution["count"] >= 100].sort_values("mean_days", ascending=False).head(15)

    # Case status breakdown
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
    plot_request_types_by_neighborhood(df, output_dir)

    logger.info("[4/15] Request volume trends by SUBJECT (department)...")
    plot_trends_by_subject(subject_trends, output_dir)

    logger.info("[5/15] Request volume trends by REASON...")
    plot_trends_by_reason(df, output_dir)

    logger.info("[6/15] Request volume trends by QUEUE...")
    plot_trends_by_queue(df, output_dir)

    logger.info("[7/15] Case volume by submission channel (SOURCE)...")
    plot_volume_by_source(df, output_dir)

    logger.info("[8/15] Average daily contacts by year...")
    plot_avg_daily_contacts(daily_avg, output_dir)

    logger.info("[9/15] Volume of top 5 request types...")
    plot_top5_types_volume(df, output_dir)

    logger.info("[10/15] Average resolution time by QUEUE...")
    plot_resolution_by_queue(queue_resolution, output_dir)

    logger.info("[11/15] Average resolution time by QUEUE and neighborhood...")
    plot_resolution_heatmap(df, queue_resolution, output_dir)

    logger.info("[12/15] Case status breakdown...")
    plot_case_status_breakdown(status_data, output_dir)

    logger.info("[13/15] Top neighborhoods by request volume...")
    plot_top_neighborhoods(top_hoods, output_dir)

    logger.info("[14/15] Resolution time distribution by top queues...")
    plot_resolution_distribution(df, queue_resolution, output_dir)

    logger.info("[15/15] Year-over-year case status breakdown...")
    plot_status_yearly_trends(df, output_dir)

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
    for source, count in source_dist.head(5).items():
        pct = count / len(df) * 100
        logger.info(f"  {source}: {count:,} ({pct:.1f}%)")

    logger.info(f"\n✅ CASE STATUS:")
    logger.info(f"  Closed:                 {closed_count:,} ({status_data.iloc[0]['Percentage']:.2f}%)")
    logger.info(f"  Open:                   {open_count:,} ({status_data.iloc[1]['Percentage']:.2f}%)")
    logger.info(f"  No Data (Null):         {null_count:,} ({status_data.iloc[2]['Percentage']:.2f}%)")

    logger.info(f"\n⏱️  RESOLUTION TIMES:")
    resolved_df = df[df["resolution_hours"].notna()].copy()
    resolved_df["resolution_days"] = resolved_df["resolution_hours"] / 24
    resolved_df = resolved_df[resolved_df["resolution_days"] <= 365]
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
