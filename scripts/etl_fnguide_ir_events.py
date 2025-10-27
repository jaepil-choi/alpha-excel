"""
ETL: FnGuide IR Events Data Crawler

This script:
1. Crawls FnGuide IR events API from 2025-09 backwards
2. Fetches data month by month with 1 second delay between requests
3. Saves each month immediately (incremental save)
4. Stops after 10 consecutive failures (data no longer available)
5. Saves to Parquet with hive partitioning (year/month)

API Endpoint:
- https://comp.fnguide.com/SVO2/json/data/05_01/{YYYYMM}.json?_={timestamp}

Input:  FnGuide API (JSON)
Output: data/fnguide/ir_events/**/*.parquet
"""

import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from curl_cffi import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Column name mapping (Korean → English)
COLUMN_MAP = {
    'KEY': 'ir_event_key',
    '일련번호': 'serial_number',
    '기준일자': 'reference_date_code',
    '기업명': 'company_name',
    '활동코드': 'activity_code',
    '이벤트명': 'event_name',
    '이벤트코드': 'event_code',
    '일자': 'event_date',
    '종목명': 'symbol',
    '주식구분': 'stock_type',
    '종류': 'category',
    '변동주식수': 'changed_shares',
    '발행가': 'issue_price',
    '변동후자본금': 'capital_after_change',
    '총발행주식수': 'total_issued_shares',
    '신주상장일': 'new_listing_date',
    '권리락일': 'ex_rights_date',
    '납입일': 'payment_date',
    '배정기준일': 'allocation_base_date',
    '배정비율': 'allocation_ratio',
    '할인비율': 'discount_ratio',
    '비고': 'remarks'
}


def print_section(title: str):
    """Print a formatted section header."""
    logger.info("=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)


def fetch_month_data(year: int, month: int) -> tuple[bool, dict | None]:
    """Fetch IR events data for a given year-month.

    Args:
        year: Year (e.g., 2025)
        month: Month (1-12)

    Returns:
        Tuple of (success: bool, data: dict | None)
        - success=True, data=dict: Successfully fetched JSON data
        - success=False, data=None: HTML error or JSON parse error (no data available)
    """
    year_month = f"{year:04d}{month:02d}"
    timestamp = int(time.time() * 1000)
    url = f"https://comp.fnguide.com/SVO2/json/data/05_01/{year_month}.json?_{timestamp}"

    logger.info(f"  Fetching {year:04d}-{month:02d}...")
    logger.info(f"    URL: {url}")

    try:
        # Use curl_cffi with Chrome impersonation
        response = requests.get(url, impersonate="chrome", timeout=30)

        logger.info(f"    Status: {response.status_code}")
        logger.info(f"    Content-Type: {response.headers.get('Content-Type', 'N/A')}")

        if response.status_code != 200:
            logger.warning(f"    Non-200 status code: {response.status_code}")
            return False, None

        # Check if response is JSON (not HTML error page)
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            logger.warning(f"    Not JSON response (likely HTML error page)")
            return False, None

        # Parse JSON
        try:
            data = response.json()
        except UnicodeDecodeError:
            # Try EUC-KR encoding
            content_text = response.content.decode('euc-kr')
            data = json.loads(content_text)
        except json.JSONDecodeError as e:
            logger.warning(f"    JSON parse error: {e}")
            return False, None

        # Validate structure
        if 'comp' not in data:
            logger.warning(f"    No 'comp' key in response")
            return False, None

        records = data['comp']
        logger.info(f"    Records: {len(records)}")

        return True, data

    except Exception as e:
        logger.error(f"    Request failed: {e}")
        return False, None


def transform_data(data: dict, year: int, month: int) -> pd.DataFrame:
    """Transform raw JSON data to DataFrame with English column names.

    Args:
        data: Raw JSON data with 'comp' array
        year: Year for partition column
        month: Month for partition column

    Returns:
        DataFrame with transformed data and partition columns
    """
    records = data['comp']

    if len(records) == 0:
        logger.warning(f"    No records to transform")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(records)

    logger.info(f"  Transforming {len(df)} records...")

    # Rename columns using map
    df = df.rename(columns=COLUMN_MAP)

    # Add partition columns
    df['year'] = year
    df['month'] = month

    logger.info(f"    Columns after renaming: {list(df.columns)}")
    logger.info(f"    Sample event_date values: {df['event_date'].head(3).tolist()}")

    return df


def save_month_data(df: pd.DataFrame, output_dir: Path, year: int, month: int):
    """Save month data to Parquet with hive partitioning.

    Args:
        df: DataFrame with IR events data
        output_dir: Base output directory
        year: Year for partition
        month: Month for partition
    """
    if len(df) == 0:
        logger.warning(f"  Skipping save (no data)")
        return

    logger.info(f"  Saving to Parquet...")

    # Create partition directory
    partition_dir = output_dir / f"year={year}" / f"month={month:02d}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Drop partition columns before saving (hive partitioning handles them)
    df_to_save = df.drop(columns=['year', 'month'])

    # Save to Parquet
    output_file = partition_dir / "data.parquet"
    df_to_save.to_parquet(output_file, index=False, engine='pyarrow')

    logger.info(f"    Saved {len(df)} records to {output_file}")
    logger.info(f"    File size: {output_file.stat().st_size / 1024:.1f} KB")


def main(start_year: int, start_month: int, test_mode: bool = False):
    """Main ETL workflow.

    Fetches data from start_year/start_month backwards until:
    - 10 consecutive failures (data no longer available)
    - OR test_mode and 3 months fetched

    Args:
        start_year: Starting year (e.g., 2025)
        start_month: Starting month (e.g., 9)
        test_mode: If True, only fetch 3 months for testing
    """
    print_section("FnGuide IR Events ETL Pipeline")

    output_dir = Path('data/fnguide/ir_events')

    logger.info(f"Configuration:")
    logger.info(f"  Start: {start_year}-{start_month:02d}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Test mode: {test_mode}")
    logger.info(f"  Strategy: Fetch backwards, stop after 10 consecutive failures")

    # Initialize counters
    current_date = datetime(start_year, start_month, 1)
    consecutive_failures = 0
    max_consecutive_failures = 10
    total_fetched = 0
    total_records = 0
    months_processed = 0

    print_section("Starting Data Fetch")

    while True:
        year = current_date.year
        month = current_date.month

        logger.info(f"\n[Month {months_processed + 1}] Processing {year:04d}-{month:02d}")

        # Check if partition already exists (skip if already fetched)
        partition_dir = output_dir / f"year={year}" / f"month={month:02d}"
        partition_file = partition_dir / "data.parquet"

        if partition_file.exists():
            logger.info(f"  [SKIP] Data already exists: {partition_file}")
            logger.info(f"    File size: {partition_file.stat().st_size / 1024:.1f} KB")

            # Reset failure counter (partition exists = data available for this period)
            consecutive_failures = 0
            total_fetched += 1  # Count as fetched (already have it)

            # Read record count from existing file
            try:
                existing_df = pd.read_parquet(partition_file)
                existing_records = len(existing_df)
                total_records += existing_records
                logger.info(f"    Records: {existing_records}")
            except Exception as e:
                logger.warning(f"    Could not read existing file: {e}")

            months_processed += 1

            # Test mode: stop after 3 months
            if test_mode and months_processed >= 3:
                logger.info(f"\n[STOP] Test mode - processed 3 months")
                break

            # Move to previous month and continue
            current_date = current_date - relativedelta(months=1)
            logger.info(f"    Sleeping 1 second...")
            time.sleep(1)
            continue

        # Fetch data
        success, data = fetch_month_data(year, month)

        if success and data is not None:
            # Reset failure counter on success
            consecutive_failures = 0

            # Transform data
            df = transform_data(data, year, month)

            # Save immediately
            save_month_data(df, output_dir, year, month)

            total_fetched += 1
            total_records += len(df)
            logger.info(f"    [OK] Successfully saved {year:04d}-{month:02d}")

        else:
            # Increment failure counter
            consecutive_failures += 1
            logger.warning(f"    [SKIP] Failed to fetch {year:04d}-{month:02d} (consecutive failures: {consecutive_failures}/{max_consecutive_failures})")

            # Check if we should stop
            if consecutive_failures >= max_consecutive_failures:
                logger.info(f"\n[STOP] Reached {max_consecutive_failures} consecutive failures - data no longer available")
                break

        months_processed += 1

        # Test mode: stop after 3 months
        if test_mode and months_processed >= 3:
            logger.info(f"\n[STOP] Test mode - processed 3 months")
            break

        # Move to previous month
        current_date = current_date - relativedelta(months=1)

        # Sleep 1 second between requests (be respectful to server)
        logger.info(f"    Sleeping 1 second...")
        time.sleep(1)

    # Summary
    print_section("ETL COMPLETE")

    logger.info(f"\n[Statistics]")
    logger.info(f"  Months processed: {months_processed}")
    logger.info(f"  Successful fetches: {total_fetched}")
    logger.info(f"  Total records: {total_records:,}")
    logger.info(f"  Failed months: {months_processed - total_fetched}")
    logger.info(f"  Final consecutive failures: {consecutive_failures}")

    if total_fetched > 0:
        logger.info(f"\n[Output]")
        logger.info(f"  Directory: {output_dir}")
        logger.info(f"  Partitioning: year/month (hive format)")
        logger.info(f"  Format: Parquet")

        logger.info(f"\n[Next Steps]")
        logger.info(f"  1. Verify data in {output_dir}")
        logger.info(f"  2. Add field definition to config/data.yaml")
        logger.info(f"  3. Use Field('fnguide_earnings_announcement') in expressions")
    else:
        logger.warning(f"\n[WARNING] No data was fetched!")

    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ETL pipeline for FnGuide IR events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (fetch 3 months from 2025-09)
  poetry run python scripts/etl_fnguide_ir_events.py --test

  # Full mode (fetch all available data from 2025-09 backwards)
  poetry run python scripts/etl_fnguide_ir_events.py --no-test
  poetry run python scripts/etl_fnguide_ir_events.py -f

  # Custom start date
  poetry run python scripts/etl_fnguide_ir_events.py --year 2024 --month 12 -f
        """
    )

    parser.add_argument(
        '--year',
        type=int,
        default=2025,
        help='Start year (default: 2025)'
    )

    parser.add_argument(
        '--month',
        type=int,
        default=9,
        help='Start month (default: 9)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        default=True,
        help='Test mode: fetch only 3 months (default)'
    )

    parser.add_argument(
        '--no-test',
        dest='test',
        action='store_false',
        help='Full mode: fetch all available data'
    )

    parser.add_argument(
        '-f', '--full',
        dest='test',
        action='store_false',
        help='Shorthand for --no-test'
    )

    args = parser.parse_args()

    # Validate month
    if args.month < 1 or args.month > 12:
        logger.error(f"Invalid month: {args.month} (must be 1-12)")
        exit(1)

    # Run ETL
    main(
        start_year=args.year,
        start_month=args.month,
        test_mode=args.test
    )
