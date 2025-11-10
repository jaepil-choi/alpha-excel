"""
Generalized ETL Script for DataGuide Excel Files

This script transforms ANY DataGuide Excel file from wide format to clean,
relational Parquet format with hive partitioning.

All DataGuide files share the same structure:
- Header at row 8
- Metadata columns: Symbol, Symbol Name, Kind, Item, Item Name, Frequency
- Date columns in wide format
- Item Name field contains the actual data fields

The script is fully config-driven via FILE_CONFIGS dictionary.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATAGUIDE_HEADER_ROW = 8  # Header row is always at index 8

# Common metadata columns (shared across all DataGuide files)
COMMON_METADATA_MAP = {
    'Symbol': 'symbol',
    'Symbol Name': 'symbol_name',
    'Kind': 'kind',
    'Frequency': 'frequency'
}

# File-specific configurations
FILE_CONFIGS = {
    'groups': {
        'input_file': 'dataguide_groups.xlsx',
        'output_dir': 'groups',
        'partition_cols': ['year', 'month'],
        'column_map': {
            **COMMON_METADATA_MAP,
            'FnGuide Sector': 'fn_sector',
            'FnGuide Industry Group': 'fn_industry_group',
            'FnGuide Industry': 'fn_industry',
            'FnGuide Industry Group 27': 'fn_industry_group_27',
            '거래소 업종': 'krx_sector',
            '거래소 업종 (세부분류)': 'krx_sector_detail'
        },
        'preprocessing': None  # No preprocessing needed
    },
    'price': {
        'input_file': 'dataguide_price.xlsx',
        'output_dir': 'price',
        'partition_cols': ['year', 'month', 'day'],
        'column_map': {
            **COMMON_METADATA_MAP,
            '수정주가(원)': 'adj_close',
            '거래대금(원)': 'trading_value',
            '상장주식수 (보통)(주)': 'listed_shares_common',
            '유동주식수(주)': 'float_shares',
            '유동주식비율(%)': 'float_ratio_pct',
            '시가총액 (보통-상장예정주식수 포함)(백만원)': 'market_cap_million',  # Will convert
            '거래정지구분': 'trading_halt_status',  # Will convert to boolean
            '관리구분': 'management_classification'  # Will convert to boolean
        },
        'preprocessing': 'price_preprocessing'  # Special preprocessing function
    },
    'funda': {
        'input_file': 'dataguide_funda.xlsx',
        'output_dir': 'funda',
        'partition_cols': ['year', 'month'],
        'column_map': {
            **COMMON_METADATA_MAP,
            '보통주자본금(천원)': 'common_stock_capital_thousand',
            '자본잉여금(천원)': 'capital_surplus_thousand',
            '이익잉여금(천원)': 'retained_earnings_thousand',
            '자기주식(천원)': 'treasury_stock_thousand',
            '매출액(천원)': 'sales_thousand',
            '이연법인세부채(천원)': 'deferred_tax_liabilities_thousand'
        },
        'preprocessing': 'funda_preprocessing'  # Convert thousands to raw KRW
    },
    'price_monthly': {
        'input_file': 'dataguide_price_monthly.xlsx',
        'output_dir': 'price_monthly',
        'partition_cols': ['year', 'month'],
        'column_map': {
            **COMMON_METADATA_MAP,
            '수정시가(원)': 'monthly_adj_open',
            '수정주가(원)': 'monthly_adj_close',
            '수익률 (1개월)(%)': 'monthly_return_pct',
            '거래량(주)': 'monthly_trading_volume',
            '거래대금(원)': 'monthly_trading_value',
            '거래정지구분': 'monthly_trading_halt_status',
            '관리구분': 'monthly_management_classification',
            '시가총액 (보통-상장예정주식수 포함)(백만원)': 'monthly_market_cap_million',
            '상장주식수 (보통)(주)': 'monthly_listed_shares_common',
            '유동주식수(주)': 'monthly_float_shares'
        },
        'preprocessing': 'price_monthly_preprocessing'  # Convert market cap to raw won, handle status fields
    }
}


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List]:
    """Identify metadata columns vs date columns.

    Args:
        df: DataFrame with columns to identify

    Returns:
        Tuple of (metadata_columns, date_columns)
    """
    metadata_cols = []
    date_cols = []

    for col in df.columns:
        try:
            pd.to_datetime(col)
            date_cols.append(col)
        except:
            metadata_cols.append(col)

    return metadata_cols, date_cols


def melt_to_long(
    df: pd.DataFrame,
    metadata_cols: List[str],
    date_cols: List
) -> pd.DataFrame:
    """Transform wide format to long format.

    Args:
        df: Wide format DataFrame
        metadata_cols: List of metadata column names
        date_cols: List of date column objects

    Returns:
        Long format DataFrame with [metadata..., date, value]
    """
    logger.info(f"  Melting {len(date_cols)} date columns to long format...")

    df_melted = pd.melt(
        df,
        id_vars=metadata_cols,
        value_vars=date_cols,
        var_name='date',
        value_name='value'
    )

    # Convert date column to datetime
    df_melted['date'] = pd.to_datetime(df_melted['date'])

    # Remove rows with NaT dates
    df_melted = df_melted.dropna(subset=['date'])

    logger.info(f"  [OK] After melting: {df_melted.shape[0]:,} rows")

    return df_melted


def pivot_items(df_long: pd.DataFrame, item_col: str) -> pd.DataFrame:
    """Pivot Item Name to columns.

    Args:
        df_long: Long format DataFrame with item_col
        item_col: Name of the item column ('Item Name' or 'Item Name ')

    Returns:
        Pivoted DataFrame with one row per [date, symbol]
    """
    logger.info(f"  Pivoting '{item_col}' to columns...")

    # Use ORIGINAL column names (before renaming)
    index_cols = ['date', 'Symbol', 'Symbol Name', 'Kind']

    # Add Frequency if it exists and has NON-NULL values
    if 'Frequency' in df_long.columns:
        non_null_count = df_long['Frequency'].notna().sum()
        if non_null_count > 0:
            logger.info(f"    Including Frequency column ({non_null_count:,} non-null values)")
            index_cols.append('Frequency')
        else:
            logger.info(f"    Skipping Frequency column (all NaN)")

    df_pivoted = df_long.pivot_table(
        index=index_cols,
        columns=item_col,
        values='value',
        aggfunc='first'  # Use first in case of duplicates
    )

    # Reset index to make all columns regular
    df_pivoted = df_pivoted.reset_index()

    # Remove the columns name (artifact from pivot_table)
    df_pivoted.columns.name = None

    logger.info(f"  [OK] After pivoting: {df_pivoted.shape[0]:,} rows × {df_pivoted.shape[1]:,} columns")

    return df_pivoted


def rename_columns(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    """Rename columns using mapping dictionary.

    Args:
        df: DataFrame with original column names
        column_map: Mapping from original to new names

    Returns:
        DataFrame with renamed columns
    """
    logger.info(f"  Renaming columns (Korean → English)...")

    # Only rename columns that exist in both df and column_map
    existing_renames = {k: v for k, v in column_map.items() if k in df.columns}

    df_renamed = df.rename(columns=existing_renames)

    logger.info(f"  [OK] Renamed {len(existing_renames)} columns")

    return df_renamed


def preprocess_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess price data to ensure clean data for alpha-database.

    Transformations:
    1. Convert market_cap from millions to won (multiply by 1,000,000)
    2. Convert trading_halt_status to boolean is_trading_suspended
    3. Convert management_classification to boolean is_issue
    4. Enforce proper data types for all numeric fields
    5. Drop original columns after transformation

    Args:
        df: DataFrame with renamed columns

    Returns:
        Preprocessed DataFrame with clean data
    """
    logger.info(f"  Preprocessing price data...")

    df_clean = df.copy()

    # 1. Market Cap: Convert from millions to won
    if 'market_cap_million' in df_clean.columns:
        logger.info(f"    Converting market_cap from millions to won...")
        df_clean['market_cap'] = (df_clean['market_cap_million'] * 1_000_000).astype('Int64')
        df_clean = df_clean.drop(columns=['market_cap_million'])
        logger.info(f"    [OK] market_cap converted")

    # 2. Trading Halt Status: Convert to boolean
    if 'trading_halt_status' in df_clean.columns:
        logger.info(f"    Converting trading_halt_status to is_trading_suspended...")
        df_clean['is_trading_suspended'] = df_clean['trading_halt_status'] != '정상'
        df_clean = df_clean.drop(columns=['trading_halt_status'])

        n_suspended = df_clean['is_trading_suspended'].sum()
        n_total = len(df_clean)
        logger.info(f"    [OK] is_trading_suspended created ({n_suspended:,} / {n_total:,} suspended)")

    # 3. Management Classification: Convert to boolean
    if 'management_classification' in df_clean.columns:
        logger.info(f"    Converting management_classification to is_issue...")
        df_clean['is_issue'] = df_clean['management_classification'] != '일반'
        df_clean = df_clean.drop(columns=['management_classification'])

        n_issue = df_clean['is_issue'].sum()
        n_total = len(df_clean)
        logger.info(f"    [OK] is_issue created ({n_issue:,} / {n_total:,} with issues)")

    # 4. Enforce numeric data types
    logger.info(f"    Enforcing numeric data types...")

    numeric_conversions = {
        'adj_close': 'Int64',
        'trading_value': 'Int64',
        'listed_shares_common': 'Int64',
        'float_shares': 'Int64',
        'float_ratio_pct': 'float64',
        'market_cap': 'Int64'
    }

    for col, dtype in numeric_conversions.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(dtype)

    logger.info(f"  [OK] Price preprocessing complete")

    return df_clean


def preprocess_funda_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess fundamental data to convert thousands to raw KRW.

    Transformations:
    1. Convert all _thousand fields to raw KRW (multiply by 1,000)
    2. Rename by removing _thousand suffix
    3. Enforce proper data types

    Args:
        df: DataFrame with renamed columns

    Returns:
        Preprocessed DataFrame with values in raw KRW
    """
    logger.info(f"  Preprocessing fundamental data...")

    df_clean = df.copy()

    # Find all columns with _thousand suffix
    thousand_cols = [col for col in df_clean.columns if col.endswith('_thousand')]

    logger.info(f"    Converting {len(thousand_cols)} fields from thousands to raw KRW...")

    for col in thousand_cols:
        # Convert to raw KRW
        new_col = col.replace('_thousand', '')
        # Multiply by 1000, keep as float64 to preserve decimal precision
        df_clean[new_col] = (df_clean[col] * 1_000).astype('float64')

        # Drop original column
        df_clean = df_clean.drop(columns=[col])

        logger.info(f"      [OK] {col} → {new_col}")

    logger.info(f"  [OK] Fundamental preprocessing complete")

    return df_clean


def preprocess_price_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess monthly price data to ensure clean data for alpha-database.

    Transformations:
    1. Convert monthly_market_cap from millions to won (multiply by 1,000,000)
    2. Convert monthly_trading_halt_status to boolean monthly_is_trading_suspended
    3. Convert monthly_management_classification to boolean monthly_is_issue
    4. Enforce proper data types for all numeric fields

    Args:
        df: DataFrame with renamed columns

    Returns:
        Preprocessed DataFrame with clean data
    """
    logger.info(f"  Preprocessing monthly price data...")

    df_clean = df.copy()

    # 1. Market Cap: Convert from millions to won
    if 'monthly_market_cap_million' in df_clean.columns:
        logger.info(f"    Converting monthly_market_cap from millions to won...")
        df_clean['monthly_market_cap'] = (df_clean['monthly_market_cap_million'] * 1_000_000).astype('Int64')
        df_clean = df_clean.drop(columns=['monthly_market_cap_million'])
        logger.info(f"    [OK] monthly_market_cap converted")

    # 2. Trading Halt Status: Convert to boolean
    if 'monthly_trading_halt_status' in df_clean.columns:
        logger.info(f"    Converting monthly_trading_halt_status to monthly_is_trading_suspended...")
        df_clean['monthly_is_trading_suspended'] = df_clean['monthly_trading_halt_status'] != '정상'
        df_clean = df_clean.drop(columns=['monthly_trading_halt_status'])

        n_suspended = df_clean['monthly_is_trading_suspended'].sum()
        n_total = len(df_clean)
        logger.info(f"    [OK] monthly_is_trading_suspended created ({n_suspended:,} / {n_total:,} suspended)")

    # 3. Management Classification: Convert to boolean
    if 'monthly_management_classification' in df_clean.columns:
        logger.info(f"    Converting monthly_management_classification to monthly_is_issue...")
        df_clean['monthly_is_issue'] = df_clean['monthly_management_classification'] != '일반'
        df_clean = df_clean.drop(columns=['monthly_management_classification'])

        n_issue = df_clean['monthly_is_issue'].sum()
        n_total = len(df_clean)
        logger.info(f"    [OK] monthly_is_issue created ({n_issue:,} / {n_total:,} with issues)")

    # 4. Enforce numeric data types
    logger.info(f"    Enforcing numeric data types...")

    numeric_conversions = {
        'monthly_adj_open': 'Int64',
        'monthly_adj_close': 'Int64',
        'monthly_return_pct': 'float64',
        'monthly_trading_volume': 'Int64',
        'monthly_trading_value': 'Int64',
        'monthly_market_cap': 'Int64',
        'monthly_listed_shares_common': 'Int64',
        'monthly_float_shares': 'Int64'
    }

    for col, dtype in numeric_conversions.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(dtype)

    logger.info(f"  [OK] Monthly price preprocessing complete")

    return df_clean


def transform_dataguide_file(
    file_key: str,
    input_dir: Path,
    output_base: Path,
    test_mode: bool = False
):
    """Transform a DataGuide file based on its configuration.

    Args:
        file_key: Key in FILE_CONFIGS ('groups', 'price', 'funda', 'price_monthly')
        input_dir: Directory containing input Excel files
        output_base: Base output directory for Parquet files
        test_mode: If True, only process first 10,000 rows for testing
    """
    config = FILE_CONFIGS[file_key]

    input_path = input_dir / config['input_file']
    output_dir = output_base / config['output_dir']

    print_section(f"TRANSFORMING: {config['input_file']}")

    # 1. Load with header=8
    logger.info(f"[1/8] Loading Excel file...")
    if test_mode:
        logger.info(f"  [TEST MODE] Loading first 10,000 rows only")
        df = pd.read_excel(input_path, sheet_name=0, header=DATAGUIDE_HEADER_ROW, nrows=10000)
    else:
        if file_key == 'price':
            logger.info(f"  [LARGE FILE] ({input_path.stat().st_size / 1024 / 1024:.1f} MB) - may take 5-10 minutes...")
        df = pd.read_excel(input_path, sheet_name=0, header=DATAGUIDE_HEADER_ROW)
    logger.info(f"  [OK] Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    # 2. Identify columns
    logger.info(f"[2/8] Identifying column types...")
    metadata_cols, date_cols = identify_columns(df)
    logger.info(f"  [OK] Metadata columns: {len(metadata_cols)}")
    logger.info(f"  [OK] Date columns: {len(date_cols)}")

    # 3. Melt to long format
    logger.info(f"[3/8] Transforming wide → long format...")
    df_long = melt_to_long(df, metadata_cols, date_cols)

    # 4. Pivot Item Name to columns
    logger.info(f"[4/8] Pivoting Item Name to columns...")
    item_col = 'Item Name ' if 'Item Name ' in df_long.columns else 'Item Name'
    df_pivoted = pivot_items(df_long, item_col)

    # 5. Rename columns
    logger.info(f"[5/8] Renaming columns...")
    df_final = rename_columns(df_pivoted, config['column_map'])

    # Drop unnecessary columns (Item, Item Name columns)
    cols_to_drop = ['Item', 'Item Name', 'Item Name ']
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])

    # Add frequency column if missing
    if 'frequency' not in df_final.columns:
        if file_key == 'groups':
            df_final['frequency'] = pd.Series([None] * len(df_final), dtype='object')
            logger.info(f"  Added 'frequency' column (set to None)")
        elif file_key in ['price', 'funda', 'price_monthly']:
            freq_map = {'price': 'DAILY', 'funda': 'MONTHLY', 'price_monthly': 'MONTHLY'}
            df_final['frequency'] = freq_map[file_key]
            logger.info(f"  Added 'frequency' column (set to '{df_final['frequency'].iloc[0]}')")

    # 6. Apply preprocessing if specified
    logger.info(f"[6/8] Preprocessing data...")
    if config['preprocessing'] == 'price_preprocessing':
        df_final = preprocess_price_data(df_final)
    elif config['preprocessing'] == 'funda_preprocessing':
        df_final = preprocess_funda_data(df_final)
    elif config['preprocessing'] == 'price_monthly_preprocessing':
        df_final = preprocess_price_monthly_data(df_final)
    else:
        logger.info(f"  No preprocessing needed for {file_key}")

    # 7. Filter out non-existent securities (only for price data)
    logger.info(f"[7/8] Filtering data...")
    if file_key == 'price':
        rows_before = len(df_final)
        df_final = df_final.dropna(subset=['adj_close'])
        rows_after = len(df_final)
        rows_dropped = rows_before - rows_after
        logger.info(f"  [OK] Dropped {rows_dropped:,} rows without adj_close ({rows_dropped/rows_before*100:.1f}%)")
    else:
        logger.info(f"  No filtering needed for {file_key}")

    # 8. Add partition columns and save
    if test_mode:
        logger.info(f"[8/8] Preparing data (TEST MODE - not saving)...")
    else:
        logger.info(f"[8/8] Adding partition columns and saving...")

    # Add partition columns based on config
    if 'year' in config['partition_cols']:
        df_final['year'] = df_final['date'].dt.year
    if 'month' in config['partition_cols']:
        df_final['month'] = df_final['date'].dt.month
    if 'day' in config['partition_cols']:
        df_final['day'] = df_final['date'].dt.day

    # Convert date to string for storage
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')

    # Statistics
    n_partitions = df_final[config['partition_cols']].drop_duplicates().shape[0]

    if test_mode:
        # TEST MODE: Don't save, just show what would be created
        logger.info(f"  [TEST] Would save to: {output_dir}")
        logger.info(f"  [TEST] Would create {n_partitions} partitions")
        logger.info(f"  [TEST] Total rows: {df_final.shape[0]:,}")
        logger.info(f"  [TEST] Total columns: {df_final.shape[1]:,}")
    else:
        # PRODUCTION MODE: Save to parquet
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save with hive partitioning
        df_final.to_parquet(
            output_dir,
            partition_cols=config['partition_cols'],
            index=False,
            engine='pyarrow'
        )

        logger.info(f"  [OK] Saved to: {output_dir}")
        logger.info(f"  [OK] Created {n_partitions} partitions")
        logger.info(f"  [OK] Total rows: {df_final.shape[0]:,}")
        logger.info(f"  [OK] Total columns: {df_final.shape[1]:,}")

    # Show sample (both test and production)
    print("\n  Sample data (first 5 rows):")
    sample_df = df_final.drop(columns=config['partition_cols']).head()
    print(sample_df.to_string(index=False))


def main(test_mode: bool, files: Optional[List[str]] = None):
    """Main ETL workflow.

    Args:
        test_mode: If True, only process first 10,000 rows for testing
        files: List of file keys to process. If None, process all files.
    """
    print_section("DataGuide Generalized ETL Pipeline")

    if test_mode:
        print("\n[TEST MODE] - Processing first 10,000 rows only")
        print("   Use --no-test or -f flag to process entire files")

    # Define paths
    input_dir = Path("data/unprocessed/fnguide")
    output_base = Path("data/fnguide")

    # Determine which files to process
    if files is None:
        files_to_process = list(FILE_CONFIGS.keys())
    else:
        files_to_process = files
        # Validate file keys
        invalid_keys = [f for f in files_to_process if f not in FILE_CONFIGS]
        if invalid_keys:
            logger.error(f"Invalid file keys: {invalid_keys}")
            logger.error(f"Valid keys are: {list(FILE_CONFIGS.keys())}")
            return

    # Check input files exist
    print("\n[Checking input files...]")
    for file_key in files_to_process:
        config = FILE_CONFIGS[file_key]
        input_path = input_dir / config['input_file']

        if input_path.exists():
            size_mb = input_path.stat().st_size / 1024 / 1024
            print(f"  [OK] {config['input_file']} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {config['input_file']} NOT FOUND")
            logger.error(f"File not found: {input_path}")
            return

    # Transform each file
    for file_key in files_to_process:
        try:
            transform_dataguide_file(
                file_key=file_key,
                input_dir=input_dir,
                output_base=output_base,
                test_mode=test_mode
            )
        except Exception as e:
            logger.error(f"Failed to transform {file_key}: {e}")
            raise

    # Summary
    print_section("ETL COMPLETE")

    if test_mode:
        print("\n[WARNING] TEST MODE: Only first 10,000 rows were processed")
        print("   To process entire files, run: poetry run python scripts/etl_dataguide_generalized.py --no-test")
        print("   Or use shorthand: poetry run python scripts/etl_dataguide_generalized.py -f")

    print("\n[Output Structure]")
    print(f"  {output_base}/")
    for file_key in files_to_process:
        config = FILE_CONFIGS[file_key]
        partition_str = "/".join(config['partition_cols'])
        print(f"  ├── {config['output_dir']}/  (partitioned by {partition_str})")

    print("\n[Next Steps]")
    if test_mode:
        print("  1. Verify test output looks correct")
        print("  2. Re-run with --no-test flag to process all data")
        print("  3. Configure alpha-database to read from data/fnguide/")
    else:
        print("  1. Verify data quality in output directories")
        print("  2. Configure alpha-database to read from data/fnguide/")
        print("  3. Test queries with alpha-excel")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # CLI argument parser
    parser = argparse.ArgumentParser(
        description="Generalized ETL pipeline for DataGuide Excel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (first 10,000 rows only)
  python scripts/etl_dataguide_generalized.py --test

  # Process all data
  python scripts/etl_dataguide_generalized.py --no-test

  # Process specific files only
  python scripts/etl_dataguide_generalized.py --files groups funda

  # Short form
  python scripts/etl_dataguide_generalized.py        # test mode (default)
  python scripts/etl_dataguide_generalized.py -f     # full mode (no test)
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        default=True,
        help='Test mode: process first 10,000 rows only (default)'
    )

    parser.add_argument(
        '--no-test',
        dest='test',
        action='store_false',
        help='Full mode: process all rows'
    )

    parser.add_argument(
        '-f', '--full',
        dest='test',
        action='store_false',
        help='Shorthand for --no-test'
    )

    parser.add_argument(
        '--files',
        nargs='+',
        choices=['groups', 'price', 'funda', 'price_monthly'],
        help='Specific files to process (default: all)'
    )

    args = parser.parse_args()

    # Run ETL
    main(test_mode=args.test, files=args.files)
