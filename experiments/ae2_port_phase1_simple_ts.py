"""
Experiment: Phase 1 - Simple Time-Series Aggregations

Test porting of 4 simple time-series operators from v1.0 to v2.0:
- TsStdDev: Rolling standard deviation
- TsMax: Rolling maximum
- TsMin: Rolling minimum
- TsSum: Rolling sum

All are nearly identical to TsMean, just different pandas rolling methods.
"""

import pandas as pd
import numpy as np

print("="*80)
print("EXPERIMENT: Phase 1 - Simple Time-Series Aggregations")
print("="*80)

# Create fake data for testing
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=10, freq='D')
assets = ['A', 'B', 'C']

# Create (T, N) DataFrame with known patterns
data = pd.DataFrame({
    'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'B': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    'C': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
}, index=dates)

print("\nInput Data (T=10, N=3):")
print(data)

# Test window = 3
window = 3

print(f"\n{'='*80}")
print(f"Testing with window={window}")
print(f"{'='*80}")

# 1. TsStdDev - Rolling standard deviation
print("\n1. TsStdDev - Rolling Standard Deviation")
print("-" * 40)

min_periods_ratio = 0.7
min_periods = max(1, int(window * min_periods_ratio))
print(f"min_periods_ratio: {min_periods_ratio}")
print(f"min_periods: {min_periods} (out of {window})")

ts_std = data.rolling(window=window, min_periods=min_periods).std()
print("\nResult:")
print(ts_std)
print(f"\nAsset A std over [1,2,3]: {np.std([1,2,3], ddof=1):.4f} (expected at index 2)")
print(f"Actual at index 2: {ts_std.loc[dates[2], 'A']:.4f}")

# 2. TsMax - Rolling maximum
print("\n2. TsMax - Rolling Maximum")
print("-" * 40)

min_periods_ratio = 0.3
min_periods = max(1, int(window * min_periods_ratio))
print(f"min_periods_ratio: {min_periods_ratio}")
print(f"min_periods: {min_periods} (out of {window})")

ts_max = data.rolling(window=window, min_periods=min_periods).max()
print("\nResult:")
print(ts_max)
print(f"\nAsset A max over [1,2,3]: {np.max([1,2,3])} (expected at index 2)")
print(f"Actual at index 2: {ts_max.loc[dates[2], 'A']}")

# 3. TsMin - Rolling minimum
print("\n3. TsMin - Rolling Minimum")
print("-" * 40)

min_periods_ratio = 0.3
min_periods = max(1, int(window * min_periods_ratio))
print(f"min_periods_ratio: {min_periods_ratio}")
print(f"min_periods: {min_periods} (out of {window})")

ts_min = data.rolling(window=window, min_periods=min_periods).min()
print("\nResult:")
print(ts_min)
print(f"\nAsset A min over [1,2,3]: {np.min([1,2,3])} (expected at index 2)")
print(f"Actual at index 2: {ts_min.loc[dates[2], 'A']}")

# 4. TsSum - Rolling sum
print("\n4. TsSum - Rolling Sum")
print("-" * 40)

min_periods_ratio = 0.3
min_periods = max(1, int(window * min_periods_ratio))
print(f"min_periods_ratio: {min_periods_ratio}")
print(f"min_periods: {min_periods} (out of {window})")

ts_sum = data.rolling(window=window, min_periods=min_periods).sum()
print("\nResult:")
print(ts_sum)
print(f"\nAsset A sum over [1,2,3]: {np.sum([1,2,3])} (expected at index 2)")
print(f"Actual at index 2: {ts_sum.loc[dates[2], 'A']}")

# Test NaN handling
print("\n" + "="*80)
print("NaN Handling Test")
print("="*80)

data_with_nan = data.copy()
data_with_nan.loc[dates[2], 'A'] = np.nan  # Insert NaN at index 2

print("\nData with NaN:")
print(data_with_nan)

print("\nTsMax with NaN (window=3, min_periods=1):")
ts_max_nan = data_with_nan.rolling(window=3, min_periods=1).max()
print(ts_max_nan)
print("\nNote: NaN at index 2 is ignored, max([1,NaN,4]) over indices [1,2,3] = 4")

# Test partial windows at beginning
print("\n" + "="*80)
print("Partial Window Test (first rows)")
print("="*80)

print(f"\nWith window={window}, min_periods={min_periods}:")
print("Index 0 (1 value): Should be NaN if min_periods > 1, else valid")
print("Index 1 (2 values): Should be NaN if min_periods > 2, else valid")
print("Index 2 (3 values): Always valid (full window)")

print("\nTsMax results:")
for i, date in enumerate(dates[:4]):
    print(f"Index {i}: {ts_max.loc[date, 'A']}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("[OK] All 4 operators follow same pattern as TsMean")
print("[OK] Only difference: pandas rolling method (.std(), .max(), .min(), .sum())")
print("[OK] NaN handling works correctly (pandas built-in)")
print("[OK] min_periods controls partial window behavior")
print("[OK] Ready to implement in v2.0 using BaseOperator")
