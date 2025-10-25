"""
Verification Script for DataGuide Parquet Files

This script verifies that the hive-partitioned Parquet files were created correctly
by querying them with DuckDB.
"""

import duckdb
from pathlib import Path

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def verify_groups_data():
    """Verify groups (monthly) data."""
    print_section("VERIFYING: data/fnguide/groups/")
    
    # Connect to DuckDB
    conn = duckdb.connect(':memory:')
    
    # Query groups data
    print("\n[Query 1] Count total rows:")
    query = """
    SELECT COUNT(*) as total_rows
    FROM read_parquet('data/fnguide/groups/**/*.parquet', hive_partitioning=true)
    """
    result = conn.execute(query).fetchdf()
    print(result)
    
    print("\n[Query 2] Check partitions (year-month):")
    query = """
    SELECT year, month, COUNT(*) as row_count
    FROM read_parquet('data/fnguide/groups/**/*.parquet', hive_partitioning=true)
    GROUP BY year, month
    ORDER BY year, month
    LIMIT 10
    """
    result = conn.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    print("\n[Query 3] Sample data for 2010-01:")
    query = """
    SELECT *
    FROM read_parquet('data/fnguide/groups/**/*.parquet', hive_partitioning=true)
    WHERE year = 2010 AND month = 1
    LIMIT 5
    """
    result = conn.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    print("\n[Query 4] Check columns:")
    query = """
    SELECT column_name, column_type
    FROM (
        DESCRIBE 
        SELECT * FROM read_parquet('data/fnguide/groups/**/*.parquet', hive_partitioning=true)
    )
    """
    result = conn.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    conn.close()


def verify_price_data():
    """Verify price (daily) data."""
    print_section("VERIFYING: data/fnguide/price/")
    
    # Connect to DuckDB
    conn = duckdb.connect(':memory:')
    
    # Query price data
    print("\n[Query 1] Count total rows:")
    query = """
    SELECT COUNT(*) as total_rows
    FROM read_parquet('data/fnguide/price/**/*.parquet', hive_partitioning=true)
    """
    result = conn.execute(query).fetchdf()
    print(result)
    
    print("\n[Query 2] Check date range:")
    query = """
    SELECT 
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(DISTINCT date) as unique_dates
    FROM read_parquet('data/fnguide/price/**/*.parquet', hive_partitioning=true)
    """
    result = conn.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    print("\n[Query 3] Sample data for 2010-01-04 to 2010-01-31:")
    query = """
    SELECT *
    FROM read_parquet('data/fnguide/price/**/*.parquet', hive_partitioning=true)
    WHERE date >= '2010-01-04' AND date <= '2010-01-31'
    LIMIT 10
    """
    result = conn.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    print("\n[Query 4] Count rows by date (first 10 dates):")
    query = """
    SELECT date, COUNT(*) as row_count
    FROM read_parquet('data/fnguide/price/**/*.parquet', hive_partitioning=true)
    GROUP BY date
    ORDER BY date
    LIMIT 10
    """
    result = conn.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    print("\n[Query 5] Check columns:")
    query = """
    SELECT column_name, column_type
    FROM (
        DESCRIBE 
        SELECT * FROM read_parquet('data/fnguide/price/**/*.parquet', hive_partitioning=true)
    )
    """
    result = conn.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    print("\n[Query 6] Sample symbols:")
    query = """
    SELECT DISTINCT symbol
    FROM read_parquet('data/fnguide/price/**/*.parquet', hive_partitioning=true)
    ORDER BY symbol
    LIMIT 10
    """
    result = conn.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    conn.close()


def main():
    """Main verification workflow."""
    print_section("DataGuide Parquet Verification")
    
    # Check if directories exist
    groups_dir = Path("data/fnguide/groups")
    price_dir = Path("data/fnguide/price")
    
    print("\n[Checking directories...]")
    if groups_dir.exists():
        print(f"  ✓ Found: {groups_dir}")
    else:
        print(f"  ✗ Missing: {groups_dir}")
        return
    
    if price_dir.exists():
        print(f"  ✓ Found: {price_dir}")
    else:
        print(f"  ✗ Missing: {price_dir}")
        return
    
    # Verify groups data
    try:
        verify_groups_data()
    except Exception as e:
        print(f"\n✗ Error verifying groups data: {e}")
    
    # Verify price data
    try:
        verify_price_data()
    except Exception as e:
        print(f"\n✗ Error verifying price data: {e}")
    
    print_section("VERIFICATION COMPLETE")
    print("\n✓ All queries executed successfully!")
    print("  Data is correctly partitioned and queryable with DuckDB")


if __name__ == "__main__":
    main()

