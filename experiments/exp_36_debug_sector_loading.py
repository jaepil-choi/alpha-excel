"""
Experiment 36: Debug Sector Loading with Non-Trading-Day Timestamps

This experiment debugs the issue where sector data timestamped on non-trading days
(e.g., 2016-01-31 Sunday) results in NaN values when universe dates only include
trading days.

Expected behavior:
- Sector data at 2016-01-31 should forward-fill to 2016-02-01 and later dates
- Early dates (2016-01-28, 2016-01-29) will be NaN (before first data timestamp)
"""

import sys
sys.path.insert(0, 'C:\\Users\\chlje\\VSCodeProjects\\alpha-excel\\src')

import pandas as pd
from alpha_excel2.core.facade import AlphaExcel

print("=" * 80)
print("Experiment 36: Debug Sector Loading")
print("=" * 80)

# Initialize AlphaExcel
ae = AlphaExcel(
    start_time='2016-01-28',
    end_time='2016-03-31',  # Shorter period for debugging
    universe=None,
    config_path='C:\\Users\\chlje\\VSCodeProjects\\alpha-excel\\config'
)

print(f"\nTime Range: {ae._start_time} to {ae._end_time}")
print(f"Universe Shape: {ae._universe_mask._data.shape}")

# Get universe dates for inspection
universe_dates = ae._universe_mask._data.index
print(f"\nFirst 20 Universe Dates:")
for i, date in enumerate(universe_dates[:20]):
    print(f"  {i}: {date}")

# Load sector data
f = ae.field
sector = f('fnguide_industry_group')

print("\n" + "=" * 80)
print("Sector Data Loaded")
print("=" * 80)
print(f"Data type: {sector._data_type}")
print(f"Shape: {sector.to_df().shape}")

# Show first 20 rows, first 5 columns
sector_df = sector.to_df()
print(f"\nFirst 20 rows, first 5 columns:")
print(sector_df.iloc[:20, :5])

# Check for NaN counts
print(f"\n" + "=" * 80)
print("NaN Analysis")
print("=" * 80)
nan_counts_per_row = sector_df.isna().sum(axis=1)
print(f"\nNaN counts per date (first 20 rows):")
for date, nan_count in nan_counts_per_row.head(20).items():
    total_cols = sector_df.shape[1]
    print(f"  {date}: {nan_count}/{total_cols} NaN ({100*nan_count/total_cols:.1f}%)")

# Let's also check what the raw data looks like from DataSource
print("\n" + "=" * 80)
print("Raw Data from DataSource (before forward-fill)")
print("=" * 80)

# Access the data source directly
data_source = ae._field_loader._ds  # In alpha_excel2 it's _ds not _data_source
raw_data = data_source.load_field(
    'fnguide_industry_group',
    start_time='2016-01-01',  # In alpha_excel2 it's start_time not start_date
    end_time='2016-03-31'
)

print(f"Raw data shape: {raw_data.shape}")
print(f"Raw data index (first 20):")
for i, date in enumerate(raw_data.index[:20]):
    print(f"  {i}: {date}")

print(f"\nRaw data (first 20 rows, first 5 columns):")
print(raw_data.iloc[:20, :5])

# Check if 2016-01-31 is in raw data
if pd.Timestamp('2016-01-31') in raw_data.index:
    print(f"\n✓ 2016-01-31 IS in raw data index")
    row_2016_01_31 = raw_data.loc['2016-01-31']
    non_null_count = row_2016_01_31.notna().sum()
    print(f"  Non-null values at 2016-01-31: {non_null_count}/{len(row_2016_01_31)}")
    print(f"  Sample values: {row_2016_01_31.head()}")
else:
    print(f"\n✗ 2016-01-31 NOT in raw data index")

# Check if 2016-01-28 and 2016-01-29 are in universe dates
print(f"\n" + "=" * 80)
print("Date Alignment Check")
print("=" * 80)
print(f"2016-01-28 in universe_dates: {pd.Timestamp('2016-01-28') in universe_dates}")
print(f"2016-01-29 in universe_dates: {pd.Timestamp('2016-01-29') in universe_dates}")
print(f"2016-01-31 in universe_dates: {pd.Timestamp('2016-01-31') in universe_dates}")
print(f"2016-02-01 in universe_dates: {pd.Timestamp('2016-02-01') in universe_dates}")

print("\n" + "=" * 80)
print("Testing Forward-Fill Logic Manually")
print("=" * 80)

# Manually test the forward-fill logic
test_data = raw_data.copy()
print(f"Original data index (first 10): {list(test_data.index[:10])}")

# Apply the new forward-fill logic
all_dates = test_data.index.union(universe_dates).sort_values()
print(f"\nUnion of dates (first 20): {list(all_dates[:20])}")

# Reindex to union
test_data_union = test_data.reindex(all_dates)
print(f"\nAfter reindex to union (first 20 rows, first 3 cols):")
print(test_data_union.iloc[:20, :3])

# Forward fill
test_data_ffill = test_data_union.ffill()
print(f"\nAfter forward-fill (first 20 rows, first 3 cols):")
print(test_data_ffill.iloc[:20, :3])

# Reindex to universe dates
test_data_final = test_data_ffill.reindex(universe_dates)
print(f"\nAfter final reindex to universe (first 20 rows, first 3 cols):")
print(test_data_final.iloc[:20, :3])

# Check if 2016-02-01 has data now
if pd.Timestamp('2016-02-01') in test_data_final.index:
    row_2016_02_01 = test_data_final.loc['2016-02-01']
    non_null_count = row_2016_02_01.notna().sum()
    total = len(row_2016_02_01)
    print(f"\n2016-02-01 non-null values: {non_null_count}/{total} ({100*non_null_count/total:.1f}%)")

print("\n" + "=" * 80)
print("Experiment Complete")
print("=" * 80)
