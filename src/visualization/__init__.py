"""
Visualization module for Boston 311 analysis.
"""

from __future__ import annotations

from .comparative import (
    plot_case_status_breakdown,
    plot_source_distribution,
    plot_top_neighborhoods,
    plot_top_request_types,
)
from .maps import create_neighborhood_choropleth, create_request_heatmap
from .temporal import (
    plot_average_daily_contacts,
    plot_day_of_week_pattern,
    plot_hourly_pattern,
    plot_requests_per_year,
)

__all__ = [
    "plot_requests_per_year",
    "plot_average_daily_contacts",
    "plot_day_of_week_pattern",
    "plot_hourly_pattern",
    "plot_top_request_types",
    "plot_top_neighborhoods",
    "plot_source_distribution",
    "plot_case_status_breakdown",
    "create_request_heatmap",
    "create_neighborhood_choropleth",
]
