"""
Experiment 36: Verify Sector Loading Fix

Simple verification that sector data loads correctly despite non-trading-day timestamps.
"""

import sys
sys.path.insert(0, 'C:\\Users\\chlje\\VSCodeProjects\\alpha-excel\\src')

from alpha_excel2.core.facade import AlphaExcel

print("=" * 80)
print("Verifying Sector Loading Fix")
print("=" * 80)

# Initialize AlphaExcel
ae = AlphaExcel(
    start_time='2016-01-28',
    end_time='2016-03-31',
    universe=None,
    config_path='C:\\Users\\chlje\\VSCodeProjects\\alpha-excel\\config'
)

# Load sector data
f = ae.field
sector = f('fnguide_industry_group')

print(f"\nSector Data Shape: {sector.to_df().shape}")
print(f"\n First 10 rows, first 5 columns:")
print(sector.to_df().iloc[:10, :5])

# Count NaNs per date
sector_df = sector.to_df()
nan_counts = sector_df.isna().sum(axis=1)
total_cols = sector_df.shape[1]

print(f"\n" + "=" * 80)
print("NaN Analysis (first 10 dates):")
print("=" * 80)
import pandas as pd
for date, nan_count in nan_counts.head(10).items():
    pct = 100 * nan_count / total_cols
    status = "[EXPECTED]" if pct == 100 and date < pd.Timestamp('2016-02-01') else "[OK]" if pct < 5 else "[PROBLEM]"
    print(f"{date.date()}: {nan_count}/{total_cols} NaN ({pct:.1f}%) {status}")

# Check that 2016-02-01 onwards has data
feb_1_data = sector_df.loc['2016-02-01']
non_null_count = feb_1_data.notna().sum()
print(f"\n" + "=" * 80)
print(f"2016-02-01 Data Check:")
print(f"  Non-null values: {non_null_count}/{total_cols} ({100*non_null_count/total_cols:.1f}%)")
print(f"  Sample values: {list(feb_1_data.dropna().head(3))}")
print(f"  Result: {'[PASS] Data loaded successfully!' if non_null_count > total_cols * 0.9 else '[FAIL] Missing data'}")
print("=" * 80)
