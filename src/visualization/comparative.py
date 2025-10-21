"""
Comparative visualization module for Boston 311 Service Request Analysis.

Creates bar charts, scatter plots, and comparative visualizations.
No analysis logic - only visualization of pre-computed metrics.
"""

from __future__ import annotations


import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Any

# Setup logging
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set seaborn style
sns.set_theme(style='whitegrid')

# Constants
FIGURE_DPI = 300
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_COLOR_PALETTE = 'Set2'


def plot_top_request_types(type_counts: pd.Series,
                           top_n: int = 10,
                           output_path: Optional[str] = None,
                           show_plot: bool = False) -> None:
    """
    Create horizontal bar chart of top request types.

    Parameters:
    -----------
    type_counts : pd.Series
        Series with request types as index, counts as values
    top_n : int
        Number of types to display (default: 10)
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info(f"Creating top {top_n} request types bar chart")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Take top N and reverse for better readability
    data = type_counts.head(top_n).sort_values()

    # Create horizontal bar chart
    colors = sns.color_palette(DEFAULT_COLOR_PALETTE, len(data))
    ax.barh(data.index, data.values, color=colors)

    # Formatting
    ax.set_title(f'Top {top_n} Request Types', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Requests', fontsize=12)
    ax.set_ylabel('Request Type', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/top_request_types.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_top_neighborhoods(neighborhood_counts: pd.Series,
                          top_n: int = 10,
                          output_path: Optional[str] = None,
                          show_plot: bool = False) -> None:
    """
    Create horizontal bar chart of top neighborhoods by request volume.

    Parameters:
    -----------
    neighborhood_counts : pd.Series
        Series with neighborhoods as index, counts as values
    top_n : int
        Number of neighborhoods to display (default: 10)
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info(f"Creating top {top_n} neighborhoods bar chart")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Take top N and reverse for better readability
    data = neighborhood_counts.head(top_n).sort_values()

    # Create horizontal bar chart
    colors = sns.color_palette('viridis', len(data))
    ax.barh(data.index, data.values, color=colors)

    # Formatting
    ax.set_title(f'Top {top_n} Neighborhoods by Request Volume', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Requests', fontsize=12)
    ax.set_ylabel('Neighborhood', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/top_neighborhoods.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_source_distribution(source_counts: pd.Series,
                            output_path: Optional[str] = None,
                            show_plot: bool = False) -> None:
    """
    Create pie chart of submission source distribution.

    Parameters:
    -----------
    source_counts : pd.Series
        Series with sources as index, counts as values
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating source distribution pie chart")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pie chart
    colors = sns.color_palette(DEFAULT_COLOR_PALETTE, len(source_counts))
    wedges, texts, autotexts = ax.pie(source_counts.values,
                                       labels=source_counts.index,
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90)

    # Formatting
    ax.set_title('311 Requests by Submission Source', fontsize=16, fontweight='bold')

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/source_distribution.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_case_status_breakdown(status_breakdown: dict[str, Any],
                               output_path: Optional[str] = None,
                               show_plot: bool = False) -> None:
    """
    Create stacked bar chart of case status breakdown.

    Parameters:
    -----------
    status_breakdown : dict[str, Any]
        Dictionary with status as key, dict with count and percentage as value
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating case status breakdown chart")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    statuses = list(status_breakdown.keys())
    counts = [status_breakdown[s]['count'] for s in statuses]
    percentages = [status_breakdown[s]['percentage'] for s in statuses]

    # Create bar chart
    colors = sns.color_palette(DEFAULT_COLOR_PALETTE, len(statuses))
    bars = ax.bar(statuses, counts, color=colors)

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%',
               ha='center', va='bottom', fontweight='bold')

    # Formatting
    ax.set_title('Case Status Breakdown', fontsize=16, fontweight='bold')
    ax.set_xlabel('Case Status', fontsize=12)
    ax.set_ylabel('Number of Cases', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/case_status_breakdown.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_subject_trends(subject_trends: pd.DataFrame,
                       output_path: Optional[str] = None,
                       show_plot: bool = False) -> None:
    """
    Create line chart of subject (department) trends over years.

    Parameters:
    -----------
    subject_trends : pd.DataFrame
        DataFrame with years as rows, subjects as columns
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating subject trends line chart")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each subject
    for column in subject_trends.columns:
        ax.plot(subject_trends.index, subject_trends[column],
               marker='o', linewidth=2, label=column)

    # Formatting
    ax.set_title('Request Trends by Subject (Department)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/subject_trends.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_queue_trends(queue_trends: pd.DataFrame,
                     output_path: Optional[str] = None,
                     show_plot: bool = False) -> None:
    """
    Create line chart of queue trends over years.

    Parameters:
    -----------
    queue_trends : pd.DataFrame
        DataFrame with years as rows, queues as columns
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating queue trends line chart")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each queue
    for column in queue_trends.columns:
        ax.plot(queue_trends.index, queue_trends[column],
               marker='o', linewidth=2, label=column)

    # Formatting
    ax.set_title('Request Trends by Queue', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/queue_trends.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_resolution_by_neighborhood(neighborhood_stats: pd.DataFrame,
                                   top_n: int = 15,
                                   output_path: Optional[str] = None,
                                   show_plot: bool = False) -> None:
    """
    Create horizontal bar chart of median resolution times by neighborhood.

    Parameters:
    -----------
    neighborhood_stats : pd.DataFrame
        DataFrame with neighborhoods as index, stats as columns (must have 'median_days')
    top_n : int
        Number of neighborhoods to display (default: 15)
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info(f"Creating resolution time by neighborhood chart (top {top_n})")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Take top N by request count, sorted by median resolution time
    data = neighborhood_stats.nlargest(top_n, 'count').sort_values('median_days')

    # Create horizontal bar chart
    colors = sns.color_palette('coolwarm', len(data))
    ax.barh(data.index, data['median_days'], color=colors)

    # Formatting
    ax.set_title(f'Median Resolution Time by Neighborhood (Top {top_n})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Median Resolution Time (days)', fontsize=12)
    ax.set_ylabel('Neighborhood', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/resolution_by_neighborhood.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_resolution_by_queue(queue_stats: pd.DataFrame,
                            output_path: Optional[str] = None,
                            show_plot: bool = False) -> None:
    """
    Create box plot of resolution times by queue.

    Parameters:
    -----------
    queue_stats : pd.DataFrame
        DataFrame with queues as index, stats as columns (must have 'median_days')
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating resolution time by queue chart")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Sort by median and take top 15
    data = queue_stats.nlargest(15, 'count').sort_values('median_days')

    # Create horizontal bar chart
    colors = sns.color_palette('plasma', len(data))
    ax.barh(data.index, data['median_days'], color=colors)

    # Formatting
    ax.set_title('Median Resolution Time by Queue', fontsize=16, fontweight='bold')
    ax.set_xlabel('Median Resolution Time (days)', fontsize=12)
    ax.set_ylabel('Queue', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/resolution_by_queue.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_data
    from src.data.preprocessor import preprocess_data
    from src.analysis.categorical import (
        calculate_request_types_overall,
        calculate_top_neighborhoods,
        calculate_source_distribution,
        calculate_case_status_breakdown,
        calculate_trends_by_subject,
        calculate_trends_by_queue
    )
    from src.analysis.resolution import (
        calculate_average_resolution_by_neighborhood,
        calculate_average_resolution_by_queue
    )

    logger.info("Running comparative visualizations")

    # Load and preprocess data
    raw_df = load_data()

    if not raw_df.empty:
        df = preprocess_data(raw_df)

        # Calculate metrics
        top_types = calculate_request_types_overall(df, top_n=10)
        top_neighborhoods = calculate_top_neighborhoods(df, top_n=10)
        source_dist = calculate_source_distribution(df)
        status_breakdown = calculate_case_status_breakdown(df)
        subject_trends = calculate_trends_by_subject(df, top_n=5)
        queue_trends = calculate_trends_by_queue(df, top_n=5)
        neighborhood_resolution = calculate_average_resolution_by_neighborhood(df)
        queue_resolution = calculate_average_resolution_by_queue(df)

        # Create visualizations
        plot_top_request_types(top_types)
        plot_top_neighborhoods(top_neighborhoods)
        plot_source_distribution(source_dist)
        plot_case_status_breakdown(status_breakdown)
        plot_subject_trends(subject_trends)
        plot_queue_trends(queue_trends)
        plot_resolution_by_neighborhood(neighborhood_resolution)
        plot_resolution_by_queue(queue_resolution)

        logger.info("All comparative visualizations created successfully")
    else:
        logger.error("No data available for visualization")
