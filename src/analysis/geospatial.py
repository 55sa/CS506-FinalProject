"""Geospatial analysis utilities for Boston 311 requests."""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_zip_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return request counts aggregated by 5-digit ZIP code."""
    if "location_zipcode" not in df.columns:
        logger.warning("location_zipcode column not found; returning empty ZIP counts.")
        return pd.DataFrame(columns=["zipcode", "count"])

    zips = (
        df["location_zipcode"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .dropna()
        .astype(float)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    counts = zips.value_counts().rename_axis("zipcode").reset_index(name="count")
    logger.info(f"Calculated ZIP counts for {len(counts)} ZIP codes")
    if not counts.empty:
        logger.info(f"Top ZIPs: {counts.head(3).to_dict(orient='records')}")
    return counts


def calculate_zip_resolution_medians(
    df: pd.DataFrame, max_days: float = 365.0
) -> pd.DataFrame:
    """Return median resolution time (days) per ZIP."""
    if "location_zipcode" not in df.columns or "resolution_time_days" not in df.columns:
        logger.warning(
            "Required columns missing for ZIP resolution stats; returning empty DataFrame."
        )
        return pd.DataFrame(columns=["zipcode", "median_resolution_days"])

    # Normalize ZIP
    zips = (
        df["location_zipcode"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .dropna()
        .astype(float)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    df_local = df.copy()
    df_local["zipcode"] = zips

    # Filter valid resolution times
    res = df_local[df_local["resolution_time_days"].notna()].copy()
    res = res[res["resolution_time_days"] <= max_days]

    medians = (
        res.groupby("zipcode")["resolution_time_days"].median().reset_index(name="median_resolution_days")
    )

    logger.info(f"Calculated ZIP median resolution for {len(medians)} ZIP codes (<= {max_days} days)")
    return medians


def extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with latitude/longitude columns, filtered to valid bounds."""
    lat_cols = [c for c in df.columns if c.lower() in {"latitude", "lat"}]
    lon_cols = [c for c in df.columns if c.lower() in {"longitude", "lon", "lng"}]

    if not lat_cols or not lon_cols:
        logger.warning("Latitude/longitude columns not found; returning empty DataFrame.")
        return pd.DataFrame(columns=["latitude", "longitude"])

    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    coords = df[[lat_col, lon_col]].rename(columns={lat_col: "latitude", lon_col: "longitude"})
    coords = coords.dropna()

    # Basic Boston bounding box filter to remove outliers
    coords = coords[
        (coords["latitude"].between(41.0, 43.0)) & (coords["longitude"].between(-72.0, -69.5))
    ]

    logger.info(f"Extracted {len(coords)} coordinate pairs for heatmap.")
    return coords
