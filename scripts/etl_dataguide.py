"""
ETL Script for DataGuide Excel Files

This script transforms DataGuide Excel files from wide format to clean,
relational Parquet format with hive partitioning.

Architecture:
- dataguide_groups.xlsx → data/fnguide/groups/ (year-month partitioning)
- dataguide_price.xlsx → data/fnguide/price/ (date partitioning)

Output format: One row per [date, symbol] with all fields as columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATAGUIDE_HEADER_ROW = 8  # Header row is always at index 8

# Column name mappings
COMMON_METADATA_MAP = {
    'Symbol': 'symbol',
    'Symbol Name': 'symbol_name',
    'Kind': 'kind',
    'Frequency': 'frequency'
}

GROUPS_COLUMN_MAP = {
    **COMMON_METADATA_MAP,
    'FnGuide Sector': 'fn_sector',
    'FnGuide Industry Group': 'fn_industry_group',
    'FnGuide Industry': 'fn_industry',
    'FnGuide Industry Group 27': 'fn_industry_group_27',
    '거래소 업종': 'exchange_sector', # TODO: krx_sector
    '거래소 업종 (세부분류)': 'exchange_sector_detail' # TODO: krx_sector_detail
}

PRICE_COLUMN_MAP = {
    **COMMON_METADATA_MAP,
    '수정주가(원)': 'adj_close',
    '거래대금(원)': 'trading_value',
    '상장주식수 (보통)(주)': 'listed_shares_common',
    '유동주식수(주)': 'float_shares',
    '유동주식비율(%)': 'float_ratio_pct',
    '시가총액 (보통-상장예정주식수 포함)(백만원)': 'market_cap_million',  # Temporary name, will convert to market_cap
    '거래정지구분': 'trading_halt_status',  # Temporary name, will convert to is_trading_suspended
    '관리구분': 'management_classification'  # Temporary name, will convert to is_issue
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
    
    logger.info(f"  ✓ After melting: {df_melted.shape[0]:,} rows")
    
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
    
    logger.info(f"  ✓ After pivoting: {df_pivoted.shape[0]:,} rows × {df_pivoted.shape[1]:,} columns")
    
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
    
    logger.info(f"  ✓ Renamed {len(existing_renames)} columns")
    
    return df_renamed


def preprocess_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess price data to ensure 100% clean data for alpha-database.
    
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
    logger.info(f"  Preprocessing data for alpha-database...")
    
    df_clean = df.copy()
    
    # 1. Market Cap: Convert from millions to won
    if 'market_cap_million' in df_clean.columns:
        logger.info(f"    Converting market_cap from millions to won...")
        df_clean['market_cap'] = (df_clean['market_cap_million'] * 1_000_000).astype('Int64')
        df_clean = df_clean.drop(columns=['market_cap_million'])
        logger.info(f"    ✓ market_cap converted (sample: {df_clean['market_cap'].iloc[0]:,} won)")
    
    # 2. Trading Halt Status: Convert to boolean
    if 'trading_halt_status' in df_clean.columns:
        logger.info(f"    Converting trading_halt_status to is_trading_suspended...")
        # '정상' = False (normal trading), '거래정지' = True (trading halted)
        df_clean['is_trading_suspended'] = df_clean['trading_halt_status'] != '정상'
        df_clean = df_clean.drop(columns=['trading_halt_status'])
        
        n_suspended = df_clean['is_trading_suspended'].sum()
        n_total = len(df_clean)
        logger.info(f"    ✓ is_trading_suspended created ({n_suspended:,} / {n_total:,} = {n_suspended/n_total*100:.1f}% suspended)")
    
    # 3. Management Classification: Convert to boolean
    if 'management_classification' in df_clean.columns:
        logger.info(f"    Converting management_classification to is_issue...")
        # '일반' = False (general, no issues), anything else = True (has issues)
        df_clean['is_issue'] = df_clean['management_classification'] != '일반'
        df_clean = df_clean.drop(columns=['management_classification'])
        
        n_issue = df_clean['is_issue'].sum()
        n_total = len(df_clean)
        logger.info(f"    ✓ is_issue created ({n_issue:,} / {n_total:,} = {n_issue/n_total*100:.1f}% with issues)")
    
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
    
    logger.info(f"    ✓ All numeric fields have correct data types")
    logger.info(f"  ✓ Preprocessing complete")
    
    return df_clean


def transform_groups(input_path: Path, output_dir: Path, test_mode: bool = False):
    """Transform groups (monthly) with year-month partitioning.
    
    Args:
        input_path: Path to dataguide_groups.xlsx
        output_dir: Output directory for parquet files
        test_mode: If True, only process first 10,000 rows for testing
    """
    print_section(f"TRANSFORMING: {input_path.name}")
    
    # 1. Load with header=8
    logger.info(f"[1/6] Loading Excel file...")
    if test_mode:
        logger.info(f"  🧪 TEST MODE: Loading first 10,000 rows only")
        df = pd.read_excel(input_path, sheet_name=0, header=DATAGUIDE_HEADER_ROW, nrows=3000)
    else:
        df = pd.read_excel(input_path, sheet_name=0, header=DATAGUIDE_HEADER_ROW)
    logger.info(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    
    # 2. Identify columns
    logger.info(f"[2/6] Identifying column types...")
    metadata_cols, date_cols = identify_columns(df)
    logger.info(f"  ✓ Metadata columns: {len(metadata_cols)}")
    logger.info(f"  ✓ Date columns: {len(date_cols)}")
    
    # 3. Melt to long format
    logger.info(f"[3/6] Transforming wide → long format...")
    df_long = melt_to_long(df, metadata_cols, date_cols)
    
    # 4. Pivot Item Name to columns
    logger.info(f"[4/6] Pivoting Item Name to columns...")
    item_col = 'Item Name ' if 'Item Name ' in df_long.columns else 'Item Name'
    df_pivoted = pivot_items(df_long, item_col)
    
    # 5. Rename columns
    logger.info(f"[5/6] Renaming columns...")
    df_final = rename_columns(df_pivoted, GROUPS_COLUMN_MAP)
    
    # Drop unnecessary columns (Item, Item Name columns)
    cols_to_drop = ['Item', 'Item Name', 'Item Name ']
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])
    
    # Add frequency column if missing (set to None as string type)
    if 'frequency' not in df_final.columns:
        df_final['frequency'] = pd.Series([None] * len(df_final), dtype='object')
        logger.info(f"  Added 'frequency' column (set to None with string type)")
    
    # 6. Add partition columns and save
    logger.info(f"[6/6] Adding partition columns and saving...")
    
    df_final['year'] = df_final['date'].dt.year
    df_final['month'] = df_final['date'].dt.month
    
    # Convert date to string for storage
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with hive partitioning
    df_final.to_parquet(
        output_dir,
        partition_cols=['year', 'month'],
        index=False,
        engine='pyarrow'
    )
    
    # Statistics
    n_partitions = df_final[['year', 'month']].drop_duplicates().shape[0]
    logger.info(f"  ✓ Saved to: {output_dir}")
    logger.info(f"  ✓ Created {n_partitions} partitions (year-month)")
    logger.info(f"  ✓ Total rows: {df_final.shape[0]:,}")
    logger.info(f"  ✓ Total columns: {df_final.shape[1]:,}")
    
    # Show sample
    print("\n  Sample data (first 5 rows):")
    print(df_final.drop(columns=['year', 'month']).head().to_string(index=False))


def transform_price(input_path: Path, output_dir: Path, test_mode: bool = False):
    """Transform price (daily) with date partitioning.
    
    Note: Price file is very large (505MB), so loading will take several minutes.
    
    Args:
        input_path: Path to dataguide_price.xlsx
        output_dir: Output directory for parquet files
        test_mode: If True, only process first 10,000 rows for testing
    """
    print_section(f"TRANSFORMING: {input_path.name}")
    
    # 1. Load with header=8
    logger.info(f"[1/6] Loading Excel file...")
    if test_mode:
        logger.info(f"  🧪 TEST MODE: Loading first 10,000 rows only")
        df = pd.read_excel(input_path, sheet_name=0, header=DATAGUIDE_HEADER_ROW, nrows=10000)
    else:
        logger.info(f"  ⏳ Large file detected ({input_path.stat().st_size / 1024 / 1024:.1f} MB) - this may take 5-10 minutes...")
        df = pd.read_excel(input_path, sheet_name=0, header=DATAGUIDE_HEADER_ROW)
    logger.info(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    
    # 2. Identify columns
    logger.info(f"[2/6] Identifying column types...")
    metadata_cols, date_cols = identify_columns(df)
    logger.info(f"  ✓ Metadata columns: {len(metadata_cols)}")
    logger.info(f"  ✓ Date columns: {len(date_cols)}")
    
    # 3. Melt to long format
    logger.info(f"[3/6] Transforming wide → long format...")
    df_long = melt_to_long(df, metadata_cols, date_cols)
    
    # 4. Pivot Item Name to columns
    logger.info(f"[4/6] Pivoting Item Name to columns...")
    item_col = 'Item Name ' if 'Item Name ' in df_long.columns else 'Item Name'
    df_pivoted = pivot_items(df_long, item_col)
    
    # 5. Rename columns
    logger.info(f"[5/7] Renaming columns...")
    df_final = rename_columns(df_pivoted, PRICE_COLUMN_MAP)
    
    # Drop unnecessary columns (Item, Item Name columns)
    cols_to_drop = ['Item', 'Item Name', 'Item Name ']
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])
    
    # Add frequency column if missing (though price data usually has it)
    if 'frequency' not in df_final.columns:
        df_final['frequency'] = 'DAILY'
        logger.info(f"  Added 'frequency' column (set to 'DAILY')")
    
    # 6. Preprocess data for alpha-database
    logger.info(f"[6/8] Preprocessing data...")
    df_final = preprocess_price_data(df_final)
    
    # 7. Drop rows without adj_close (non-existent securities on that date)
    logger.info(f"[7/8] Filtering non-existent securities...")
    rows_before = len(df_final)
    df_final = df_final.dropna(subset=['adj_close'])
    rows_after = len(df_final)
    rows_dropped = rows_before - rows_after
    logger.info(f"  ✓ Dropped {rows_dropped:,} rows without adj_close ({rows_dropped/rows_before*100:.1f}%)")
    logger.info(f"  ✓ Remaining: {rows_after:,} rows (securities that actually existed)")
    
    # 8. Add partition columns and save
    logger.info(f"[8/8] Adding partition columns and saving...")
    
    # Add year, month, day partition columns (hierarchical like groups)
    df_final['year'] = df_final['date'].dt.year
    df_final['month'] = df_final['date'].dt.month
    df_final['day'] = df_final['date'].dt.day
    
    # Convert date to string for storage
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with hierarchical hive partitioning: year/month/day
    df_final.to_parquet(
        output_dir,
        partition_cols=['year', 'month', 'day'],
        index=False,
        engine='pyarrow'
    )
    
    # Statistics
    n_partitions = df_final[['year', 'month', 'day']].drop_duplicates().shape[0]
    logger.info(f"  ✓ Saved to: {output_dir}")
    logger.info(f"  ✓ Created {n_partitions} partitions (year-month-day)")
    logger.info(f"  ✓ Total rows: {df_final.shape[0]:,}")
    logger.info(f"  ✓ Total columns: {df_final.shape[1]:,}")
    
    # Show sample
    print("\n  Sample data (first 5 rows):")
    sample_df = df_final.drop(columns=['year', 'month', 'day']).head()
    print(sample_df.to_string(index=False))


def main(test_mode: bool):
    """Main ETL workflow.
    
    Args:
        test_mode: If True, only process first 10,000 rows for testing
    """
    print_section("DataGuide ETL Pipeline")
    
    if test_mode:
        print("\n🧪 RUNNING IN TEST MODE - Processing first 10,000 rows only")
        print("   Use --no-test or -f flag to process entire files")
    
    # Define paths
    input_dir = Path("data/unprocessed/fnguide")
    output_base = Path("data/fnguide")
    
    files = {
        'groups': {
            'input': input_dir / "dataguide_groups.xlsx",
            'output': output_base / "groups"
        },
        'price': {
            'input': input_dir / "dataguide_price.xlsx",
            'output': output_base / "price"
        }
    }
    
    # Check input files exist
    print("\n[Checking input files...]")
    for name, paths in files.items():
        if paths['input'].exists():
            size_mb = paths['input'].stat().st_size / 1024 / 1024
            print(f"  ✓ {paths['input'].name} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {paths['input'].name} NOT FOUND")
            logger.error(f"File not found: {paths['input']}")
            return
    
    # Transform groups (monthly data)
    try:
        transform_groups(
            input_path=files['groups']['input'],
            output_dir=files['groups']['output'],
            test_mode=test_mode
        )
    except Exception as e:
        logger.error(f"Failed to transform groups: {e}")
        raise
    
    # Transform price (daily data)
    try:
        transform_price(
            input_path=files['price']['input'],
            output_dir=files['price']['output'],
            test_mode=test_mode
        )
    except Exception as e:
        logger.error(f"Failed to transform price: {e}")
        raise
    
    # Summary
    print_section("ETL COMPLETE")
    
    if test_mode:
        print("\n⚠️  TEST MODE: Only first 10,000 rows were processed")
        print("   To process entire files, run: poetry run python scripts/etl_dataguide.py --no-test")
        print("   Or use shorthand: poetry run python scripts/etl_dataguide.py -f")
    
    print("\n[Output Structure]")
    print(f"  {output_base}/")
    print(f"  ├── groups/  (monthly, partitioned by year/month)")
    print(f"  └── price/   (daily, partitioned by year/month/day)")
    
    print("\n[Next Steps]")
    if test_mode:
        print("  1. Verify test output looks correct")
        print("  2. Re-run with --no-test flag to process all data")
        print("  3. Configure alpha-database to read from data/fnguide/")
    else:
        print("  1. Verify data quality in output directories")
        print("  2. Configure alpha-database to read from data/fnguide/")
        print("  3. Test queries with alpha-canvas")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # CLI argument parser
    parser = argparse.ArgumentParser(
        description="ETL pipeline for DataGuide Excel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (first 10,000 rows only)
  python scripts/etl_dataguide.py --test
  
  # Process all data
  python scripts/etl_dataguide.py --no-test
  
  # Short form
  python scripts/etl_dataguide.py        # test mode (default)
  python scripts/etl_dataguide.py -f     # full mode (no test)
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
    
    args = parser.parse_args()
    
    # Run ETL
    main(test_mode=args.test)

