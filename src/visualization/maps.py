"""Folium-based geospatial visualizations."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import HeatMap

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_GEOJSON_URL = (
    "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/"
    "ma_massachusetts_zip_codes_geo.min.json"
)


def ensure_geojson(path: Path, url: str = DEFAULT_GEOJSON_URL) -> Path:
    """Ensure a GeoJSON exists at the given path; download if missing."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"GeoJSON not found at {path}, downloading from {url}")
    urlretrieve(url, path)
    logger.info(f"Downloaded GeoJSON to {path}")
    return path


def plot_zip_choropleth(
    zip_counts: pd.DataFrame,
    geojson_path: Path,
    output_path: Path,
    zip_field: str = "ZCTA5CE10",
) -> None:
    """Create an interactive ZIP choropleth using Folium."""
    if zip_counts.empty:
        logger.warning("ZIP counts are empty; skipping choropleth.")
        return

    geojson_path = ensure_geojson(geojson_path)

    logger.info(f"Loading GeoJSON from {geojson_path}")
    geo_df = gpd.read_file(geojson_path)
    geo_df[zip_field] = geo_df[zip_field].astype(str).str.zfill(5)

    merged = geo_df.merge(zip_counts, left_on=zip_field, right_on="zipcode", how="left").fillna(
        {"count": 0}
    )

    m = folium.Map(location=(42.3601, -71.0589), zoom_start=11, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=merged,
        data=merged,
        columns=[zip_field, "count"],
        key_on=f"feature.properties.{zip_field}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_color="lightgray",
        legend_name="311 Requests by ZIP",
    ).add_to(m)

    folium.GeoJson(
        merged,
        name="ZIPs",
        style_function=lambda _: {"weight": 0.4, "color": "black", "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(fields=[zip_field, "count"], aliases=["ZIP", "Requests"]),
    ).add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"✓ Saved ZIP choropleth to {output_path}")


def plot_density_heatmap(
    coords: pd.DataFrame,
    output_path: Path,
    radius: int = 12,
    blur: int = 25,
    max_val: float | None = None,
) -> None:
    """Create an interactive density heatmap from coordinates."""
    if coords.empty:
        logger.warning("Coordinate data empty; skipping density heatmap.")
        return

    gradient = {
        0.0: "transparent",
        0.2: "blue",
        0.4: "lime",
        0.6: "yellow",
        0.8: "orange",
        1.0: "red",
    }

    if max_val is None:
        max_val = max(1, int(len(coords) * 0.001))

    m = folium.Map(location=(42.3601, -71.0589), zoom_start=11, tiles="cartodbpositron")
    HeatMap(
        data=coords[["latitude", "longitude"]].values.tolist(),
        radius=radius,
        blur=blur,
        max_zoom=13,
        gradient=gradient,
        max_val=max_val,
    ).add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"✓ Saved density heatmap to {output_path}")
