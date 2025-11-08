"""
Experiment 39: Verify Fundamental Data Loading

This script directly reads the fundamental data parquet files
to verify that the ETL worked correctly and data is not empty.

Goals:
- Load fundamental data directly from parquet (no alpha-excel)
- Check data shape, non-null counts, value ranges
- Verify all 6 fundamental fields
- Print sample data for inspection
"""

import duckdb
import pandas as pd
from pathlib import Path

print("=" * 80)
print("  Experiment 39: Verify Fundamental Data Loading")
print("=" * 80)

# Define data path
funda_path = Path("data/fnguide/funda")

print(f"\n[1] Checking data directory...")
print(f"  Path: {funda_path}")
print(f"  Exists: {funda_path.exists()}")

if not funda_path.exists():
    print("\n  ERROR: Fundamental data directory not found!")
    exit(1)

# Count partition directories
partitions = list(funda_path.glob("year=*/month=*"))
print(f"  Partitions found: {len(partitions)}")

print(f"\n[2] Loading fundamental data with DuckDB...")

# Create DuckDB connection
con = duckdb.connect()

# Query to load all fundamental data
query = """
SELECT
    date,
    symbol,
    symbol_name,
    kind,
    frequency,
    sales,
    common_stock_capital,
    deferred_tax_liabilities,
    retained_earnings,
    treasury_stock,
    capital_surplus
FROM read_parquet('data/fnguide/funda/**/*.parquet', hive_partitioning=true)
ORDER BY date, symbol
"""

print(f"  Query: {query}")

# Execute query
df = con.execute(query).df()

print(f"\n[3] Data shape and basic info...")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n[4] Date range...")
print(f"  Min date: {df['date'].min()}")
print(f"  Max date: {df['date'].max()}")
print(f"  Unique dates: {df['date'].nunique():,}")

print(f"\n[5] Symbol info...")
print(f"  Unique symbols: {df['symbol'].nunique():,}")
print(f"  Sample symbols: {df['symbol'].unique()[:10].tolist()}")

print(f"\n[6] Data types...")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print(f"\n[7] Non-null counts for fundamental fields...")
fundamental_fields = [
    'sales',
    'common_stock_capital',
    'deferred_tax_liabilities',
    'retained_earnings',
    'treasury_stock',
    'capital_surplus'
]

for field in fundamental_fields:
    non_null = df[field].notna().sum()
    pct = (non_null / len(df)) * 100
    print(f"  {field}:")
    print(f"    Non-null: {non_null:,} / {len(df):,} ({pct:.1f}%)")

    if non_null > 0:
        # Show value statistics
        valid_values = df[field].dropna()
        print(f"    Min: {valid_values.min():,.0f}")
        print(f"    Max: {valid_values.max():,.0f}")
        print(f"    Mean: {valid_values.mean():,.0f}")
        print(f"    Median: {valid_values.median():,.0f}")

print(f"\n[8] Sample data (first 10 rows)...")
print(df.head(10).to_string(index=False))

print(f"\n[9] Sample data for a specific symbol (Samsung Electronics - A005930)...")
samsung = df[df['symbol'] == 'A005930'].sort_values('date')
if len(samsung) > 0:
    print(f"  Rows for Samsung: {len(samsung):,}")
    print(f"  Date range: {samsung['date'].min()} to {samsung['date'].max()}")
    print("\n  Latest 5 records:")
    print(samsung.tail(5).to_string(index=False))
else:
    print("  No data found for Samsung Electronics (A005930)")

print(f"\n[10] Data completeness by date...")
# Check how many symbols have data for each date
completeness = df.groupby('date').agg({
    'symbol': 'count',
    'sales': lambda x: x.notna().sum(),
    'common_stock_capital': lambda x: x.notna().sum(),
    'retained_earnings': lambda x: x.notna().sum()
}).rename(columns={
    'symbol': 'total_symbols',
    'sales': 'has_sales',
    'common_stock_capital': 'has_capital',
    'retained_earnings': 'has_earnings'
})

print("\n  Sample completeness (first 5 dates):")
print(completeness.head(5).to_string())

print("\n  Sample completeness (last 5 dates):")
print(completeness.tail(5).to_string())

print(f"\n[11] Frequency distribution...")
freq_counts = df['frequency'].value_counts()
print(f"  Frequency values:")
for freq, count in freq_counts.items():
    print(f"    {freq}: {count:,} rows")

print(f"\n[12] Kind distribution...")
kind_counts = df['kind'].value_counts()
print(f"  Kind values:")
for kind, count in kind_counts.items():
    print(f"    {kind}: {count:,} rows")

print(f"\n[13] Check for duplicate [date, symbol] pairs...")
duplicates = df.groupby(['date', 'symbol']).size()
duplicate_count = (duplicates > 1).sum()
print(f"  Duplicate [date, symbol] pairs: {duplicate_count}")
if duplicate_count > 0:
    print("  WARNING: Found duplicates! This indicates a problem with the ETL.")
    print(f"  Sample duplicates:")
    dup_pairs = duplicates[duplicates > 1].head(5)
    print(dup_pairs.to_string())

print(f"\n[14] Verify data types are correct (float64 for monetary values)...")
for field in fundamental_fields:
    dtype = df[field].dtype
    print(f"  {field}: {dtype}", end="")
    if dtype == 'float64':
        print(" [OK]")
    else:
        print(f" [ERROR] (Expected float64)")

print(f"\n[15] Check value ranges (should be in raw KRW, not thousands)...")
print("  Checking if values are in millions/billions range (not thousands)...")

for field in fundamental_fields:
    if df[field].notna().sum() > 0:
        valid_values = df[field].dropna()
        avg_value = valid_values.mean()

        print(f"\n  {field}:")
        print(f"    Average: {avg_value:,.0f}")

        # Check magnitude
        if avg_value > 1e9:
            print(f"    Magnitude: Billions (10^9) [OK]")
        elif avg_value > 1e6:
            print(f"    Magnitude: Millions (10^6) [OK]")
        elif avg_value > 1e3:
            print(f"    Magnitude: Thousands (10^3) - [WARNING] Values may still be in thousands!")
        else:
            print(f"    Magnitude: < 1000 - [WARNING] Unexpected magnitude!")

print("\n" + "=" * 80)
print("  Verification Complete!")
print("=" * 80)

# Close connection
con.close()
