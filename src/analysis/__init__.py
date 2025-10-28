"""
Analysis module for Boston 311 service requests.
"""

from __future__ import annotations

from .categorical import (
    calculate_request_types_by_neighborhood_data,
    calculate_request_types_overall,
    calculate_source_by_year,
    calculate_source_distribution,
    calculate_status_by_year,
    calculate_top_neighborhoods,
    calculate_top_types_by_year,
    calculate_trends_by_queue,
    calculate_trends_by_reason,
    calculate_trends_by_subject,
)
from .resolution import (
    calculate_average_resolution_by_queue,
    calculate_resolution_heatmap_data,
)
from .temporal import (
    calculate_average_daily_contacts,
    calculate_requests_per_year,
)

__all__ = [
    "calculate_requests_per_year",
    "calculate_average_daily_contacts",
    "calculate_request_types_overall",
    "calculate_top_neighborhoods",
    "calculate_source_distribution",
    "calculate_trends_by_subject",
    "calculate_trends_by_reason",
    "calculate_trends_by_queue",
    "calculate_request_types_by_neighborhood_data",
    "calculate_top_types_by_year",
    "calculate_source_by_year",
    "calculate_status_by_year",
    "calculate_average_resolution_by_queue",
    "calculate_resolution_heatmap_data",
]
