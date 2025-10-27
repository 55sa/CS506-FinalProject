#!/usr/bin/env python3
"""
Extract 25% of data from the large CSV file for faster processing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add src to path for logger import
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_batch_data(
    input_file: str = "data/processed/311_cleaned.csv",
    output_file: str = "data/processed/batch.csv",
    batch_size: int = 10000
) -> None:
    """
    Extract a batch of data from the large CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        batch_size: Number of records to extract (default: 10000)
    """
    logger.info("=" * 60)
    logger.info("EXTRACTING BATCH DATA SAMPLE")
    logger.info("=" * 60)
    
    # Check if input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please run data preprocessing first:")
        logger.info("  python -m src.data.preprocessor")
        return
    
    logger.info(f"Loading data from: {input_file}")
    
    try:
        # Load the data
        df = pd.read_csv(input_file, low_memory=False)
        logger.info(f"Original data size: {len(df):,} records")
        
        # Extract batch of data
        actual_batch_size = min(batch_size, len(df))
        df_batch = df.sample(n=actual_batch_size, random_state=42)
        
        logger.info(f"Batch data size: {len(df_batch):,} records")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the batch
        df_batch.to_csv(output_file, index=False)
        logger.info(f"Batch data saved to: {output_file}")
        
        # Display batch statistics
        logger.info("\n" + "=" * 60)
        logger.info("BATCH DATA STATISTICS")
        logger.info("=" * 60)
        
        if 'resolution_time_days' in df_batch.columns:
            resolved = df_batch['resolution_time_days'].notna().sum()
            logger.info(f"Records with resolution time: {resolved:,}")
            if resolved > 0:
                mean_res = df_batch['resolution_time_days'].mean()
                median_res = df_batch['resolution_time_days'].median()
                logger.info(f"Mean resolution time: {mean_res:.2f} days")
                logger.info(f"Median resolution time: {median_res:.2f} days")
        
        if 'year' in df_batch.columns:
            year_range = f"{df_batch['year'].min():.0f}-{df_batch['year'].max():.0f}"
            logger.info(f"Year range: {year_range}")
        
        if 'neighborhood' in df_batch.columns:
            unique_hoods = df_batch['neighborhood'].nunique()
            logger.info(f"Unique neighborhoods: {unique_hoods:,}")
        
        if 'type' in df_batch.columns:
            unique_types = df_batch['type'].nunique()
            logger.info(f"Unique request types: {unique_types:,}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ BATCH DATA EXTRACTION COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Output file: {output_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


def main() -> None:
    """Main function to extract batch data sample."""
    extract_batch_data()


if __name__ == "__main__":
    main()
