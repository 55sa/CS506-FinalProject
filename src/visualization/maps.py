"""
Geographic visualization module for Boston 311 Service Request Analysis.

Creates choropleth maps and heatmaps using folium and plotly.
No analysis logic - only visualization of pre-computed metrics.
"""

from __future__ import annotations


import logging
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
from pathlib import Path
from typing import Any

# Setup logging
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
BOSTON_CENTER = [42.3601, -71.0589]  # Boston coordinates
DEFAULT_ZOOM = 12


def create_request_heatmap(df: pd.DataFrame,
                           sample_size: Optional[int] = 10000,
                           output_path: Optional[str] = None) -> folium.Map:
    """
    Create interactive heatmap of 311 request locations.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with location data (must have lat/long or separate latitude/longitude columns)
    sample_size : Optional[int]
        Number of points to sample for performance (default: 10000).
        Set to None to use all data.
    output_path : Optional[str]
        Path to save the HTML map

    Returns:
    --------
    folium.Map
        Folium map object with heatmap layer
    """
    logger.info("Creating request location heatmap")

    # Filter to records with location data
    # Try common column name variations
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat_col, lon_col = 'latitude', 'longitude'
    elif 'lat' in df.columns and 'lon' in df.columns:
        lat_col, lon_col = 'lat', 'lon'
    elif 'lat' in df.columns and 'long' in df.columns:
        lat_col, lon_col = 'lat', 'long'
    else:
        logger.error("Location columns not found in DataFrame")
        return None

    # Filter to valid coordinates
    df_with_location = df[[lat_col, lon_col]].dropna()

    # Sample if needed for performance
    if sample_size and len(df_with_location) > sample_size:
        logger.info(f"Sampling {sample_size} points from {len(df_with_location)} total")
        df_with_location = df_with_location.sample(n=sample_size, random_state=42)

    logger.info(f"Creating heatmap with {len(df_with_location)} points")

    # Create base map
    boston_map = folium.Map(
        location=BOSTON_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles='OpenStreetMap'
    )

    # Prepare heatmap data
    heat_data = [[row[lat_col], row[lon_col]] for idx, row in df_with_location.iterrows()]

    # Add heatmap layer
    HeatMap(heat_data,
            min_opacity=0.3,
            max_zoom=15,
            radius=15,
            blur=20,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1.0: 'red'}
            ).add_to(boston_map)

    # Save map
    if output_path is None:
        output_path = 'outputs/figures/request_heatmap.html'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    boston_map.save(output_path)
    logger.info(f"Saved heatmap to {output_path}")

    return boston_map


def create_neighborhood_choropleth(neighborhood_counts: pd.Series,
                                   output_path: Optional[str] = None) -> None:
    """
    Create choropleth map of request volume by neighborhood.

    Note: This function creates a simple bar chart representation.
    For true choropleth, GeoJSON boundaries would be needed.

    Parameters:
    -----------
    neighborhood_counts : pd.Series
        Series with neighborhoods as index, request counts as values
    output_path : Optional[str]
        Path to save the figure
    """
    logger.info("Creating neighborhood choropleth visualization")
    logger.warning("True choropleth requires GeoJSON boundaries. Creating bar chart representation.")

    # Create a simple plotly bar chart as placeholder
    # In production, this would use actual GeoJSON boundaries
    fig = px.bar(
        x=neighborhood_counts.values,
        y=neighborhood_counts.index,
        orientation='h',
        title='311 Requests by Neighborhood',
        labels={'x': 'Number of Requests', 'y': 'Neighborhood'},
        color=neighborhood_counts.values,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=800,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/neighborhood_choropleth.html'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Saved choropleth to {output_path}")


def create_clustered_map(df: pd.DataFrame,
                        sample_size: Optional[int] = 1000,
                        output_path: Optional[str] = None) -> folium.Map:
    """
    Create interactive map with clustered markers for 311 requests.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with location data and request information
    sample_size : Optional[int]
        Number of points to sample for performance (default: 1000)
    output_path : Optional[str]
        Path to save the HTML map

    Returns:
    --------
    folium.Map
        Folium map object with marker clusters
    """
    logger.info("Creating clustered marker map")

    # Try to find location columns
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat_col, lon_col = 'latitude', 'longitude'
    elif 'lat' in df.columns and 'lon' in df.columns:
        lat_col, lon_col = 'lat', 'lon'
    elif 'lat' in df.columns and 'long' in df.columns:
        lat_col, lon_col = 'lat', 'long'
    else:
        logger.error("Location columns not found in DataFrame")
        return None

    # Filter to valid coordinates and required columns
    required_cols = [lat_col, lon_col]
    if 'type' in df.columns:
        required_cols.append('type')
    if 'neighborhood' in df.columns:
        required_cols.append('neighborhood')

    df_with_location = df[required_cols].dropna(subset=[lat_col, lon_col])

    # Sample if needed
    if sample_size and len(df_with_location) > sample_size:
        logger.info(f"Sampling {sample_size} points from {len(df_with_location)} total")
        df_with_location = df_with_location.sample(n=sample_size, random_state=42)

    logger.info(f"Creating clustered map with {len(df_with_location)} points")

    # Create base map
    boston_map = folium.Map(
        location=BOSTON_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles='OpenStreetMap'
    )

    # Note: Marker clustering requires folium.plugins.MarkerCluster
    # For simplicity, adding individual markers (in production, use MarkerCluster)
    from folium.plugins import MarkerCluster

    marker_cluster = MarkerCluster().add_to(boston_map)

    # Add markers
    for idx, row in df_with_location.head(sample_size if sample_size else len(df_with_location)).iterrows():
        popup_text = ""
        if 'type' in df.columns and pd.notna(row.get('type')):
            popup_text += f"Type: {row['type']}<br>"
        if 'neighborhood' in df.columns and pd.notna(row.get('neighborhood')):
            popup_text += f"Neighborhood: {row['neighborhood']}"

        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=popup_text if popup_text else "311 Request"
        ).add_to(marker_cluster)

    # Save map
    if output_path is None:
        output_path = 'outputs/figures/clustered_map.html'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    boston_map.save(output_path)
    logger.info(f"Saved clustered map to {output_path}")

    return boston_map


def create_neighborhood_comparison_map(df: pd.DataFrame,
                                      metric: str = 'count',
                                      output_path: Optional[str] = None) -> None:
    """
    Create interactive bar chart comparing neighborhoods.

    Parameters:
    -----------
    df : pd.DataFrame
        Aggregated data by neighborhood
    metric : str
        Metric to display (e.g., 'count', 'resolution_time')
    output_path : Optional[str]
        Path to save the HTML figure
    """
    logger.info(f"Creating neighborhood comparison map for metric: {metric}")

    if 'neighborhood' not in df.columns:
        logger.error("DataFrame must have 'neighborhood' column")
        return

    # Group by neighborhood
    if metric == 'count':
        neighborhood_data = df['neighborhood'].value_counts().sort_values(ascending=False)
    else:
        logger.warning(f"Metric {metric} not implemented, using count")
        neighborhood_data = df['neighborhood'].value_counts().sort_values(ascending=False)

    # Create interactive plotly chart
    fig = px.bar(
        x=neighborhood_data.index,
        y=neighborhood_data.values,
        title=f'311 Requests by Neighborhood',
        labels={'x': 'Neighborhood', 'y': 'Number of Requests'},
        color=neighborhood_data.values,
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        showlegend=False
    )

    # Save figure
    if output_path is None:
        output_path = 'outputs/figures/neighborhood_comparison.html'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Saved neighborhood comparison to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_data
    from src.data.preprocessor import preprocess_data
    from src.analysis.categorical import calculate_top_neighborhoods

    logger.info("Running geographic visualizations")

    # Load and preprocess data
    raw_df = load_data()

    if not raw_df.empty:
        df = preprocess_data(raw_df)

        # Create visualizations
        logger.info("Note: Geographic visualizations require location data in the dataset")

        # Heatmap (if location data available)
        if any(col in df.columns for col in ['latitude', 'lat']):
            create_request_heatmap(df, sample_size=10000)
            create_clustered_map(df, sample_size=1000)
        else:
            logger.warning("No location columns found. Skipping heatmap and clustered map.")

        # Neighborhood analysis
        neighborhood_counts = calculate_top_neighborhoods(df, top_n=25)
        create_neighborhood_choropleth(neighborhood_counts)
        create_neighborhood_comparison_map(df)

        logger.info("Geographic visualizations completed")
    else:
        logger.error("No data available for visualization")
