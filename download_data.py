#!/usr/bin/env python3
"""
Download Boston 311 Service Request data files (2011-2025).

This script downloads all yearly CSV files from the Boston Open Data portal
and saves them to the data/raw directory with standardized names.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

# Download URLs for each year (2011-2025)
DOWNLOAD_URLS: dict[int, str] = {
    2025: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/9d7c2214-4709-478a-a2e8-fb2020a5bb94/download/tmpfd97jjsj.csv",
    2024: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/dff4d804-5031-443a-8409-8344efd0e5c8/download/tmpm461rr5o.csv",
    2023: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/e6013a93-1321-4f2a-bf91-8d8a02f1e62f/download/tmpwbgyud93.csv",
    2022: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/81a7b022-f8fc-4da5-80e4-b160058ca207/download/tmpfm8veglw.csv",
    2021: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/f53ebccd-bc61-49f9-83db-625f209c95f5/download/tmp88p9g82n.csv",
    2020: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/6ff6a6fd-3141-4440-a880-6f60a37fe789/download/tmpcv_10m2s.csv",
    2019: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/ea2e4696-4a2d-429c-9807-d02eb92e0222/download/tmpcje3ep_w.csv",
    2018: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/2be28d90-3a90-4af1-a3f6-f28c1e25880a/download/tmp7602cia8.csv",
    2017: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/30022137-709d-465e-baae-ca155b51927d/download/tmpzccn8u4q.csv",
    2016: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/b7ea6b1b-3ca4-4c5b-9713-6dc1db52379a/download/tmpzxzxeqfb.csv",
    2015: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/c9509ab4-6f6d-4b97-979a-0cf2a10c922b/download/tmphrybkxuh.csv",
    2014: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/bdae89c8-d4ce-40e9-a6e1-a5203953a2e0/download/tmp8afxvko_.csv",
    2013: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/407c5cd0-f764-4a41-adf8-054ff535049e/download/tmpyzk_wmya.csv",
    2012: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/382e10d9-1864-40ba-bef6-4eea3c75463c/download/tmpeyvgdt5u.csv",
    2011: "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/94b499d9-712a-4d2a-b790-7ceec5c9c4b1/download/tmp_9ogynu0.csv",
}

# Output directory
DATA_DIR = Path("data/raw")

# Download settings
CHUNK_SIZE = 8192  # 8KB chunks
TIMEOUT = 30  # seconds
MAX_RETRIES = 3


def get_session() -> requests.Session:
    """
    Create a requests session with retry logic.

    Returns:
    --------
    requests.Session
        Configured session with retry adapter
    """
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def download_file(
    url: str,
    output_path: Path,
    session: Optional[requests.Session] = None
) -> bool:
    """
    Download a file from URL to output path with progress indication.

    Parameters:
    -----------
    url : str
        URL to download from
    output_path : Path
        Path where file should be saved
    session : Optional[requests.Session]
        Requests session to use (creates new if None)

    Returns:
    --------
    bool
        True if download successful, False otherwise
    """
    if session is None:
        session = get_session()

    try:
        logger.info(f"Downloading {output_path.name}...")

        # Stream the download
        response = session.get(url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress every 10MB
                    if total_size and downloaded % (10 * 1024 * 1024) < CHUNK_SIZE:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"  Progress: {progress:.1f}%")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {output_path.name} ({file_size_mb:.2f} MB)")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Failed to download {output_path.name}: {str(e)}")

        # Clean up partial download
        if output_path.exists():
            output_path.unlink()

        return False


def download_all_data(
    years: Optional[list[int]] = None,
    skip_existing: bool = True
) -> dict[str, bool]:
    """
    Download all 311 service request data files.

    Parameters:
    -----------
    years : Optional[list[int]]
        Specific years to download. If None, downloads all years.
    skip_existing : bool
        If True, skip files that already exist (default: True)

    Returns:
    --------
    dict[str, bool]
        Dictionary mapping year to download success status
    """
    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which years to download
    years_to_download = years if years else sorted(DOWNLOAD_URLS.keys())

    logger.info(f"Starting download of {len(years_to_download)} files")
    logger.info(f"Output directory: {DATA_DIR.absolute()}")
    logger.info("-" * 70)

    # Create session for reuse
    session = get_session()

    results: dict[int, bool] = {}
    successful = 0
    skipped = 0
    failed = 0

    for year in years_to_download:
        if year not in DOWNLOAD_URLS:
            logger.warning(f"No URL available for year {year}, skipping")
            results[year] = False
            failed += 1
            continue

        output_path = DATA_DIR / f"311_requests_{year}.csv"

        # Skip if file exists
        if skip_existing and output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"⊘ Skipping {year} (already exists, {file_size_mb:.2f} MB)")
            results[year] = True
            skipped += 1
            continue

        # Download the file
        url = DOWNLOAD_URLS[year]
        success = download_file(url, output_path, session)
        results[year] = success

        if success:
            successful += 1
        else:
            failed += 1

        # Be nice to the server - small delay between downloads
        if year != years_to_download[-1]:  # Don't sleep after last download
            time.sleep(1)

    # Print summary
    logger.info("-" * 70)
    logger.info("Download Summary:")
    logger.info(f"  ✓ Successfully downloaded: {successful}")
    logger.info(f"  ⊘ Skipped (existing): {skipped}")
    logger.info(f"  ✗ Failed: {failed}")
    logger.info(f"  Total files: {successful + skipped}")

    return results


def main() -> None:
    """Main function to run the download script."""
    logger.info("=" * 70)
    logger.info("Boston 311 Service Request Data Download")
    logger.info("=" * 70)

    # Download all years
    results = download_all_data()

    # Check if all downloads were successful
    all_successful = all(results.values())

    if all_successful:
        logger.info("\n✅ All data files downloaded successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Run preprocessing: python src/data/preprocessor.py")
        logger.info("  2. Run analysis: python src/analysis/temporal.py")
        logger.info("  3. Generate visualizations: python src/visualization/temporal.py")
    else:
        failed_years = [year for year, success in results.items() if not success]
        logger.warning(f"\n⚠️  Some downloads failed: {failed_years}")
        logger.info("You can retry failed downloads by running this script again.")

    logger.info(f"\nData location: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
