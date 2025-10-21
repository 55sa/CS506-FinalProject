"""
Temporal visualization module for Boston 311 Service Request Analysis.

Creates time series plots and temporal visualizations.
No analysis logic - only visualization of pre-computed metrics.
"""

from __future__ import annotations


import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional

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
DEFAULT_FIGSIZE = (12, 6)
DEFAULT_COLOR_PALETTE = 'Set2'


def plot_requests_per_year(yearly_counts: pd.Series,
                           output_path: Optional[str] = None,
                           show_plot: bool = False) -> None:
    """
    Create line chart of total requests per year.

    Parameters:
    -----------
    yearly_counts : pd.Series
        Series with years as index, request counts as values
    output_path : Optional[str]
        Path to save the figure (default: outputs/figures/requests_per_year.png)
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating yearly requests line chart")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Create line plot
    ax.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8)

    # Formatting
    ax.set_title('Total 311 Service Requests by Year', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/requests_per_year.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_average_daily_contacts(avg_daily: pd.Series,
                                output_path: Optional[str] = None,
                                show_plot: bool = False) -> None:
    """
    Create line chart of average daily contacts by year.

    Parameters:
    -----------
    avg_daily : pd.Series
        Series with years as index, average daily contacts as values
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating average daily contacts line chart")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Create line plot
    ax.plot(avg_daily.index, avg_daily.values, marker='o', linewidth=2, markersize=8, color='steelblue')

    # Formatting
    ax.set_title('Average Daily 311 Contacts by Year', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Daily Contacts', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/avg_daily_contacts.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_day_of_week_pattern(dow_counts: pd.Series,
                             output_path: Optional[str] = None,
                             show_plot: bool = False) -> None:
    """
    Create bar chart of request volume by day of week.

    Parameters:
    -----------
    dow_counts : pd.Series
        Series with day names as index, request counts as values
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating day of week pattern bar chart")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Create bar plot
    colors = sns.color_palette(DEFAULT_COLOR_PALETTE, len(dow_counts))
    ax.bar(dow_counts.index, dow_counts.values, color=colors)

    # Formatting
    ax.set_title('311 Requests by Day of Week', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Total Requests', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/day_of_week_pattern.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_hourly_pattern(hourly_counts: pd.Series,
                       output_path: Optional[str] = None,
                       show_plot: bool = False) -> None:
    """
    Create line chart of request volume by hour of day.

    Parameters:
    -----------
    hourly_counts : pd.Series
        Series with hours (0-23) as index, request counts as values
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating hourly pattern line chart")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Create line plot with area fill
    ax.plot(hourly_counts.index, hourly_counts.values, linewidth=2, color='coral')
    ax.fill_between(hourly_counts.index, hourly_counts.values, alpha=0.3, color='coral')

    # Formatting
    ax.set_title('311 Requests by Hour of Day', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Total Requests', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks
    ax.set_xticks(range(0, 24, 2))

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/hourly_pattern.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_monthly_heatmap(monthly_pivot: pd.DataFrame,
                         output_path: Optional[str] = None,
                         show_plot: bool = False) -> None:
    """
    Create heatmap of monthly request patterns across years.

    Parameters:
    -----------
    monthly_pivot : pd.DataFrame
        Pivot table with years as rows, months as columns
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating monthly heatmap")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    sns.heatmap(monthly_pivot, annot=False, fmt='d', cmap='viridis',
                cbar_kws={'label': 'Number of Requests'}, ax=ax)

    # Formatting
    ax.set_title('311 Requests Heatmap: Month by Year', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    # Set month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_labels)

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/monthly_heatmap.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_seasonal_patterns(seasonal_overall: pd.Series,
                          output_path: Optional[str] = None,
                          show_plot: bool = False) -> None:
    """
    Create bar chart of seasonal request patterns.

    Parameters:
    -----------
    seasonal_overall : pd.Series
        Series with seasons as index, request counts as values
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating seasonal patterns bar chart")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot
    colors = sns.color_palette(DEFAULT_COLOR_PALETTE, len(seasonal_overall))
    ax.bar(seasonal_overall.index, seasonal_overall.values, color=colors)

    # Formatting
    ax.set_title('311 Requests by Season', fontsize=16, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Total Requests', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/seasonal_patterns.png'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_yearly_growth_rate(growth_rate: pd.Series,
                           output_path: Optional[str] = None,
                           show_plot: bool = False) -> None:
    """
    Create bar chart of year-over-year growth rates.

    Parameters:
    -----------
    growth_rate : pd.Series
        Series with years as index, percent change as values
    output_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot (default: False)
    """
    logger.info("Creating yearly growth rate bar chart")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Create bar plot with colors based on positive/negative
    colors = ['green' if x > 0 else 'red' for x in growth_rate.values]
    ax.bar(growth_rate.index, growth_rate.values, color=colors, alpha=0.7)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # Formatting
    ax.set_title('Year-over-Year Growth Rate in 311 Requests', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Growth Rate (%)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/yearly_growth_rate.png'

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
    from src.analysis.temporal import (
        calculate_requests_per_year,
        calculate_average_daily_contacts,
        calculate_day_of_week_patterns,
        calculate_hourly_patterns,
        calculate_monthly_trends,
        calculate_seasonal_patterns,
        calculate_yearly_growth_rate
    )

    logger.info("Running temporal visualizations")

    # Load and preprocess data
    raw_df = load_data()

    if not raw_df.empty:
        df = preprocess_data(raw_df)

        # Calculate metrics
        yearly_counts = calculate_requests_per_year(df)
        avg_daily = calculate_average_daily_contacts(df)
        dow_patterns = calculate_day_of_week_patterns(df)
        hourly_patterns = calculate_hourly_patterns(df)
        monthly_trends = calculate_monthly_trends(df)
        seasonal = calculate_seasonal_patterns(df)
        growth_rate = calculate_yearly_growth_rate(df)

        # Create visualizations
        plot_requests_per_year(yearly_counts)
        plot_average_daily_contacts(avg_daily)
        plot_day_of_week_pattern(dow_patterns)
        plot_hourly_pattern(hourly_patterns)
        plot_monthly_heatmap(monthly_trends)
        plot_seasonal_patterns(seasonal['overall'])
        plot_yearly_growth_rate(growth_rate)

        logger.info("All temporal visualizations created successfully")
    else:
        logger.error("No data available for visualization")
