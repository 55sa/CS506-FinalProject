"""Core visualization functions for the 15 required plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")

FIGURE_DPI = 300


def plot_yearly_requests(yearly: pd.Series, output_dir: Path) -> None:
    """Plot total requests per year with trend line."""
    plt.figure(figsize=(14, 7))
    plt.plot(
        yearly.index,
        yearly.values,
        marker="o",
        linewidth=3,
        markersize=12,
        color="#2E86AB",
    )
    plt.fill_between(yearly.index, yearly.values, alpha=0.2, color="#2E86AB")
    plt.title(
        "Total 311 Requests by Year (2011-2025)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    for year, count in yearly.items():
        plt.text(year, count, f"{count:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        output_dir / "1_requests_per_year.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    logger.info("✓ Saved: 1_requests_per_year.png")
    plt.close()


def plot_top_request_types(top_types: pd.Series, output_dir: Path) -> None:
    """Plot top request types as horizontal bar chart."""
    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("viridis", len(top_types))
    plt.barh(range(len(top_types)), top_types.values, color=colors)
    plt.yticks(range(len(top_types)), top_types.index, fontsize=11)
    plt.xlabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.title(
        "Top 20 Request Types (Overall)", fontsize=18, fontweight="bold", pad=20
    )
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        output_dir / "2_top_request_types_overall.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 2_top_request_types_overall.png")
    plt.close()


def plot_request_types_by_neighborhood(
    neighborhood_types: dict[str, pd.Series], output_dir: Path
) -> None:
    """Plot top request types for each top neighborhood."""
    neighborhoods = list(neighborhood_types.keys())
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    axes = axes.flatten()

    for idx, hood in enumerate(neighborhoods):
        types = neighborhood_types[hood]
        axes[idx].barh(
            range(len(types)),
            types.values,
            color=sns.color_palette("Set2", len(types)),
        )
        axes[idx].set_yticks(range(len(types)))
        axes[idx].set_yticklabels(types.index, fontsize=9)
        axes[idx].set_xlabel("Count", fontsize=11)
        axes[idx].set_title(
            f"{hood} (n={types.sum():,})", fontsize=12, fontweight="bold"
        )
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis="x")

    fig.delaxes(axes[5])
    fig.suptitle(
        "Top 10 Request Types by Neighborhood (Top 5 Neighborhoods)",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "3_request_types_by_neighborhood.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 3_request_types_by_neighborhood.png")
    plt.close()


def plot_trends_by_subject(
    subject_trends: pd.DataFrame, output_dir: Path
) -> None:
    """Plot subject (department) request trends over time."""
    plt.figure(figsize=(16, 8))
    for subject in subject_trends.columns:
        plt.plot(
            subject_trends.index,
            subject_trends[subject],
            marker="o",
            linewidth=2.5,
            label=subject,
            markersize=8,
        )
    plt.title(
        "Request Volume by SUBJECT (Department) - Year over Year",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Subject", fontsize=11, title_fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "4_trends_by_subject.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    logger.info("✓ Saved: 4_trends_by_subject.png")
    plt.close()


def plot_trends_by_reason(
    reason_trends: pd.DataFrame, output_dir: Path
) -> None:
    """Plot reason request trends over time."""
    plt.figure(figsize=(16, 9))
    for reason in reason_trends.columns:
        plt.plot(
            reason_trends.index,
            reason_trends[reason],
            marker="s",
            linewidth=2.5,
            label=reason,
            markersize=7,
            alpha=0.8,
        )
    plt.title(
        "Request Volume by REASON - Year over Year (Top 8)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(
        title="Reason", fontsize=10, title_fontsize=12, loc="best", ncol=2
    )
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "5_trends_by_reason.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    logger.info("✓ Saved: 5_trends_by_reason.png")
    plt.close()


def plot_trends_by_queue(queue_trends: pd.DataFrame, output_dir: Path) -> None:
    """Plot queue request trends over time."""
    plt.figure(figsize=(16, 9))
    for queue in queue_trends.columns:
        plt.plot(
            queue_trends.index,
            queue_trends[queue],
            marker="^",
            linewidth=2.5,
            label=queue,
            markersize=7,
            alpha=0.8,
        )
    plt.title(
        "Request Volume by QUEUE - Year over Year (Top 8)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Queue", fontsize=10, title_fontsize=12, loc="best", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "6_trends_by_queue.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    logger.info("✓ Saved: 6_trends_by_queue.png")
    plt.close()


def plot_volume_by_source(
    source_yearly: pd.DataFrame, output_dir: Path
) -> None:
    """Plot submission channel volume by year."""
    plt.figure(figsize=(16, 8))
    source_yearly.plot(
        kind="bar", stacked=False, ax=plt.gca(), colormap="tab10", width=0.8
    )
    plt.title(
        "Request Volume by Submission Channel (SOURCE) - Year over Year",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Source", fontsize=11, title_fontsize=12, loc="best")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis="y")
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "7_volume_by_source.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    logger.info("✓ Saved: 7_volume_by_source.png")
    plt.close()


def plot_avg_daily_contacts(daily_avg: pd.DataFrame, output_dir: Path) -> None:
    """Plot average daily contacts by year."""
    plt.figure(figsize=(14, 7))
    plt.bar(
        daily_avg["year"],
        daily_avg["avg_daily_requests"],
        color=sns.color_palette("coolwarm", len(daily_avg)),
        edgecolor="black",
        linewidth=1.5,
    )
    plt.title(
        "Average Daily 311 Contacts by Year",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Average Daily Requests", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    for _, row in daily_avg.iterrows():
        plt.text(
            row["year"],
            row["avg_daily_requests"],
            f'{int(row["avg_daily_requests"]):,}',
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(
        output_dir / "8_avg_daily_contacts.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 8_avg_daily_contacts.png")
    plt.close()


def plot_top5_types_volume(
    top5_yearly: pd.DataFrame, output_dir: Path
) -> None:
    """Plot top 5 request types volume over time."""
    plt.figure(figsize=(16, 8))
    top5_yearly.plot(
        kind="bar",
        ax=plt.gca(),
        width=0.85,
        color=sns.color_palette("Set2", len(top5_yearly.columns)),
    )
    plt.title(
        "Volume of Top 5 Request Types - Year over Year",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.legend(title="Type", fontsize=11, title_fontsize=12, loc="best")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis="y")
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "9_top5_types_volume.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 9_top5_types_volume.png")
    plt.close()


def plot_resolution_by_queue(
    queue_resolution: pd.DataFrame, output_dir: Path
) -> None:
    """Plot average resolution time by queue."""
    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(queue_resolution))
    plt.barh(
        y_pos,
        queue_resolution["mean_days"],
        color=sns.color_palette("rocket", len(queue_resolution)),
    )
    plt.yticks(y_pos, queue_resolution.index, fontsize=10)
    plt.xlabel("Average Resolution Time (Days)", fontsize=14, fontweight="bold")
    plt.title(
        "Average Resolution Time by QUEUE (Top 15, min 100 cases)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")
    for i, (idx, row) in enumerate(queue_resolution.iterrows()):
        plt.text(
            row["mean_days"],
            i,
            f' {row["mean_days"]:.1f}d (n={int(row["count"]):,})',
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(
        output_dir / "10_resolution_by_queue.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 10_resolution_by_queue.png")
    plt.close()


def plot_resolution_heatmap(
    heatmap_data: pd.DataFrame, output_dir: Path
) -> None:
    """Plot resolution time heatmap (queue × neighborhood)."""
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Avg Resolution (Days)"},
        linewidths=0.5,
    )
    plt.title(
        "Average Resolution Time by QUEUE and Neighborhood (Days)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Queue", fontsize=14, fontweight="bold")
    plt.ylabel("Neighborhood", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(
        output_dir / "11_resolution_queue_neighborhood.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 11_resolution_queue_neighborhood.png")
    plt.close()


def plot_case_status_breakdown(
    status_data: pd.DataFrame, output_dir: Path
) -> None:
    """Plot case status breakdown (pie + bar chart)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = ["#66C2A5", "#FC8D62", "#8DA0CB"]
    ax1.pie(
        status_data["Percentage"],
        labels=status_data["Status"],
        autopct="%1.2f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 13, "fontweight": "bold"},
    )
    ax1.set_title("Case Status Distribution", fontsize=16, fontweight="bold", pad=20)

    ax2.bar(
        status_data["Status"],
        status_data["Count"],
        color=colors,
        edgecolor="black",
        linewidth=2,
    )
    ax2.set_ylabel("Number of Cases", fontsize=14, fontweight="bold")
    ax2.set_title("Case Status Counts", fontsize=16, fontweight="bold", pad=20)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax2.grid(True, alpha=0.3, axis="y")
    for i, row in status_data.iterrows():
        ax2.text(
            i,
            row["Count"],
            f'{int(row["Count"]):,}\n({row["Percentage"]:.2f}%)',
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "12_case_status_breakdown.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 12_case_status_breakdown.png")
    plt.close()


def plot_top_neighborhoods(top_hoods: pd.Series, output_dir: Path) -> None:
    """Plot top neighborhoods by request volume."""
    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("coolwarm", len(top_hoods))
    plt.barh(range(len(top_hoods)), top_hoods.values, color=colors)
    plt.yticks(range(len(top_hoods)), top_hoods.index, fontsize=11)
    plt.xlabel("Number of Requests", fontsize=14, fontweight="bold")
    plt.title(
        "Top 20 Neighborhoods by Request Volume (2011-2025)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}")
    )
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        output_dir / "13_top_neighborhoods.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 13_top_neighborhoods.png")
    plt.close()


def plot_resolution_distribution(
    box_data: pd.DataFrame, output_dir: Path
) -> None:
    """Plot resolution time distribution as box plots."""
    plt.figure(figsize=(16, 8))
    sns.boxplot(
        data=box_data,
        x="queue",
        y="resolution_days",
        palette="Set3",
        hue="queue",
        legend=False,
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.xlabel("Queue", fontsize=14, fontweight="bold")
    plt.ylabel("Resolution Time (Days)", fontsize=14, fontweight="bold")
    plt.title(
        "Resolution Time Distribution by QUEUE (Top 8)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        output_dir / "14_resolution_distribution.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 14_resolution_distribution.png")
    plt.close()


def plot_status_yearly_trends(
    status_yearly: pd.DataFrame, output_dir: Path
) -> None:
    """Plot year-over-year case status trends."""
    status_yearly_pct = (
        status_yearly.div(status_yearly.sum(axis=1), axis=0) * 100
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    status_yearly.plot(
        kind="bar", stacked=True, ax=ax1, color=["#66C2A5", "#FC8D62"], width=0.8
    )
    ax1.set_title(
        "Case Status by Year (Absolute Counts)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Number of Cases", fontsize=13, fontweight="bold")
    ax1.legend(title="Status", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    status_yearly_pct.plot(
        kind="bar", stacked=True, ax=ax2, color=["#66C2A5", "#FC8D62"], width=0.8
    )
    ax2.set_title(
        "Case Status by Year (Percentage)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=13, fontweight="bold")
    ax2.legend(title="Status", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(
        output_dir / "15_status_yearly_trends.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    logger.info("✓ Saved: 15_status_yearly_trends.png")
    plt.close()
