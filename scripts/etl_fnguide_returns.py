"""
ETL: Calculate Returns from FnGuide Adjusted Close Prices

This script:
1. Reads processed FnGuide price data (adj_close) from Parquet
2. Calculates daily returns per symbol: (price_t - price_t-1) / price_t-1
3. Saves returns to Parquet with same hive partitioning (year/month/day)

Input:  data/fnguide/price/**/*.parquet (adj_close field)
Output: data/fnguide/returns/**/*.parquet (return field)
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    logger.info("=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)


def load_price_data(input_dir: Path, test_mode: bool = False) -> pd.DataFrame:
    """Load adjusted close price data from Parquet files.
    
    Args:
        input_dir: Path to fnguide/price directory
        test_mode: If True, limit to first 10,000 rows
    
    Returns:
        DataFrame with columns: [date, symbol, adj_close, year, month, day]
    """
    logger.info(f"Loading price data from {input_dir}...")
    
    # Use DuckDB to read hive-partitioned Parquet
    query = f"""
        SELECT 
            date,
            symbol,
            adj_close,
            year,
            month,
            day
        FROM read_parquet('{input_dir}/**/*.parquet', hive_partitioning=true)
        ORDER BY symbol, date
    """
    
    if test_mode:
        query += " LIMIT 10000"
    
    logger.info(f"  Executing query...")
    df = duckdb.query(query).to_df()
    
    logger.info(f"  ✓ Loaded {len(df):,} rows")
    logger.info(f"    Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"    Unique symbols: {df['symbol'].nunique():,}")
    
    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns per symbol.
    
    Formula: return_t = (price_t - price_t-1) / price_t-1
    
    Args:
        df: DataFrame with [date, symbol, adj_close] columns
    
    Returns:
        DataFrame with [date, symbol, return] columns
        Note: First date per symbol will have NaN return (no previous price)
    """
    logger.info(f"Calculating returns...")
    
    # Sort by symbol and date (should already be sorted, but ensure)
    df = df.sort_values(['symbol', 'date']).copy()
    
    # Calculate returns using groupby + pct_change
    # pct_change() calculates (current - previous) / previous
    df['return'] = df.groupby('symbol')['adj_close'].pct_change()
    
    # Count NaN returns (first date per symbol)
    nan_count = df['return'].isna().sum()
    total_count = len(df)
    nan_pct = (nan_count / total_count) * 100
    
    logger.info(f"  ✓ Returns calculated")
    logger.info(f"    Total rows: {total_count:,}")
    logger.info(f"    NaN returns: {nan_count:,} ({nan_pct:.2f}%)")
    logger.info(f"    Valid returns: {total_count - nan_count:,}")
    
    # Show sample returns
    sample = df[df['return'].notna()].head(5)
    logger.info(f"    Sample returns:")
    for idx, row in sample.iterrows():
        logger.info(f"      {row['date']} | {row['symbol']} | {row['return']:.6f}")
    
    # Return statistics
    returns_series = df['return'].dropna()
    logger.info(f"    Return statistics:")
    logger.info(f"      Mean: {returns_series.mean():.6f}")
    logger.info(f"      Std: {returns_series.std():.6f}")
    logger.info(f"      Min: {returns_series.min():.6f}")
    logger.info(f"      Max: {returns_series.max():.6f}")
    
    return df


def save_returns(df: pd.DataFrame, output_dir: Path):
    """Save returns to Parquet with hive partitioning.
    
    Args:
        df: DataFrame with [date, symbol, return, year, month, day] columns
        output_dir: Path to output directory (data/fnguide/returns/)
    """
    logger.info(f"Saving returns to {output_dir}...")
    
    # Select only needed columns
    df_output = df[['date', 'symbol', 'return', 'year', 'month', 'day']].copy()
    
    # Group by partition columns
    partition_groups = df_output.groupby(['year', 'month', 'day'])
    
    logger.info(f"  Writing {len(partition_groups)} partitions...")
    
    saved_count = 0
    for (year, month, day), group in partition_groups:
        # Create partition directory
        partition_dir = output_dir / f"year={year}" / f"month={month:02d}" / f"day={day:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Drop partition columns (hive partitioning stores them in path)
        group_data = group.drop(columns=['year', 'month', 'day'])
        
        # Save to Parquet
        output_file = partition_dir / "data.parquet"
        group_data.to_parquet(output_file, index=False, engine='pyarrow')
        
        saved_count += len(group)
        
        if saved_count % 10000 < len(group):  # Log progress every ~10k rows
            logger.info(f"    Progress: {saved_count:,} / {len(df_output):,} rows saved")
    
    logger.info(f"  ✓ Saved {saved_count:,} rows to {len(partition_groups)} partitions")


def verify_output(output_dir: Path):
    """Verify saved returns data.
    
    Args:
        output_dir: Path to output directory
    """
    logger.info(f"Verifying output...")
    
    # Use DuckDB to read back and verify
    query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT date) as unique_dates,
            COUNT(DISTINCT symbol) as unique_symbols,
            MIN(date) as min_date,
            MAX(date) as max_date,
            AVG(return) as mean_return,
            STDDEV(return) as std_return
        FROM read_parquet('{output_dir}/**/*.parquet', hive_partitioning=true)
        WHERE return IS NOT NULL
    """
    
    result = duckdb.query(query).to_df()
    
    logger.info(f"  Verification results:")
    logger.info(f"    Total rows (non-null): {result['total_rows'].iloc[0]:,}")
    logger.info(f"    Unique dates: {result['unique_dates'].iloc[0]:,}")
    logger.info(f"    Unique symbols: {result['unique_symbols'].iloc[0]:,}")
    logger.info(f"    Date range: {result['min_date'].iloc[0]} to {result['max_date'].iloc[0]}")
    logger.info(f"    Mean return: {result['mean_return'].iloc[0]:.6f}")
    logger.info(f"    Std return: {result['std_return'].iloc[0]:.6f}")
    
    logger.info(f"  ✓ Verification complete")


def main(test_mode: bool = False):
    """Main ETL workflow.
    
    Args:
        test_mode: If True, process only first 10,000 rows
    """
    print_section("ETL: FnGuide Returns Calculation")
    
    # Define paths
    input_dir = Path('data/fnguide/price')
    output_dir = Path('data/fnguide/returns')
    
    logger.info(f"Configuration:")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Test mode: {test_mode}")
    
    # Validate input exists
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error(f"Please run etl_dataguide.py first to create price data")
        return
    
    # Step 1: Load price data
    print_section("Step 1: Load Adjusted Close Price Data")
    df = load_price_data(input_dir, test_mode=test_mode)
    
    # Step 2: Calculate returns
    print_section("Step 2: Calculate Daily Returns")
    df = calculate_returns(df)
    
    # Step 3: Save returns
    print_section("Step 3: Save Returns to Parquet")
    save_returns(df, output_dir)
    
    # Step 4: Verify output
    print_section("Step 4: Verify Output")
    verify_output(output_dir)
    
    # Final summary
    print_section("ETL COMPLETE")
    
    logger.info(f"✓ Returns calculation complete!")
    logger.info(f"  Output saved to: {output_dir}")
    logger.info(f"  Data is hive-partitioned by: year/month/day")
    logger.info(f"  Ready to use in alpha-database")
    
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Update config/data.yaml to point 'returns' field to this data")
    logger.info(f"  2. Use Field('returns') in AlphaCanvas expressions")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate returns from FnGuide price data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (first 10,000 rows)
  poetry run python scripts/etl_fnguide_returns.py --test
  
  # Full mode (all data)
  poetry run python scripts/etl_fnguide_returns.py --no-test
  poetry run python scripts/etl_fnguide_returns.py -f
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
    
    main(test_mode=args.test)

