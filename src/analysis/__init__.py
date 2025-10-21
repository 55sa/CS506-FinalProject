"""
Analysis module for Boston 311 service requests.
"""

from __future__ import annotations

from .categorical import (
    calculate_case_status_breakdown,
    calculate_request_types_overall,
    calculate_source_distribution,
    calculate_top_neighborhoods,
)
from .resolution import (
    calculate_average_resolution_by_neighborhood,
    calculate_average_resolution_by_queue,
    calculate_resolution_summary,
)
from .temporal import (
    calculate_average_daily_contacts,
    calculate_day_of_week_patterns,
    calculate_hourly_patterns,
    calculate_requests_per_year,
    calculate_trend_statistics,
)

__all__ = [
    "calculate_requests_per_year",
    "calculate_average_daily_contacts",
    "calculate_day_of_week_patterns",
    "calculate_hourly_patterns",
    "calculate_trend_statistics",
    "calculate_request_types_overall",
    "calculate_top_neighborhoods",
    "calculate_source_distribution",
    "calculate_case_status_breakdown",
    "calculate_average_resolution_by_queue",
    "calculate_average_resolution_by_neighborhood",
    "calculate_resolution_summary",
]
