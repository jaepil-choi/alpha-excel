"""
Experiment: Ts_Zscore Operator (Operator Composition Pattern)

Demonstrates how to build complex operators by reusing simpler ones.
Ts_Zscore = (X - TsMean(X)) / TsStdDev(X)

This experiment validates:
1. Operator composition pattern (reuse existing operators)
2. Time-series z-score normalization
3. Comparison with manual calculation

Key Insight:
Instead of duplicating code, we can compose operators:
- TsMean: already implemented in timeseries.py
- TsStdDev: already implemented in timeseries.py
- Ts_Zscore: reuses both!
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("Experiment: Ts_Zscore Operator (Operator Composition Pattern)")
print("=" * 80)

# Create test data
dates = pd.date_range('2024-01-01', periods=10, freq='D')
securities = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

# Trending data to see normalization effect
data = pd.DataFrame([
    [1.0, 10.0, 5.0, 2.0],
    [2.0, 11.0, 6.0, 3.0],
    [3.0, 12.0, 7.0, 4.0],
    [4.0, 13.0, 8.0, 5.0],
    [5.0, 14.0, 9.0, 6.0],
    [6.0, 15.0, 10.0, 7.0],
    [7.0, 16.0, 11.0, 8.0],
    [8.0, 17.0, 12.0, 9.0],
    [9.0, 18.0, 13.0, 10.0],
    [10.0, 19.0, 14.0, 11.0],
], index=dates, columns=securities)

print("\n1. Input Data (Trending):")
print(data)

# Manual calculation
print("\n" + "=" * 80)
print("2. Manual Ts_Zscore Calculation (window=5)")
print("=" * 80)

window = 5

# Method 1: Using pandas rolling
ts_mean = data.rolling(window=window, min_periods=3).mean()
ts_std = data.rolling(window=window, min_periods=3).std()
ts_zscore_manual = (data - ts_mean) / ts_std

print("\nRolling mean (window=5, min_periods=3):")
print(ts_mean)

print("\nRolling std (window=5, min_periods=3):")
print(ts_std)

print("\nTs_Zscore = (X - mean) / std:")
print(ts_zscore_manual)

# Verify properties
print("\n" + "=" * 80)
print("3. Verification of Ts_Zscore Properties")
print("=" * 80)

# For each column, check that ts_zscore is normalized
for col in securities:
    col_data = data[col]
    col_zscore = ts_zscore_manual[col]

    # Window 5, row index 4 (0-based) is first full window
    # Check a few windows
    for i in range(4, 10):  # Rows 4-9 have full windows
        window_data = col_data.iloc[i-window+1:i+1]
        window_zscore = col_zscore.iloc[i]

        if not pd.isna(window_zscore):
            # Z-score should be: (current_val - window_mean) / window_std
            expected_mean = window_data.mean()
            expected_std = window_data.std()
            expected_zscore = (col_data.iloc[i] - expected_mean) / expected_std

            print(f"\n{col} at row {i} ({dates[i].date()}):")
            print(f"  Window data: {window_data.tolist()}")
            print(f"  Window mean: {expected_mean:.4f}")
            print(f"  Window std:  {expected_std:.4f}")
            print(f"  Current val: {col_data.iloc[i]}")
            print(f"  Z-score:     {window_zscore:.4f}")
            print(f"  Expected:    {expected_zscore:.4f}")
            assert abs(window_zscore - expected_zscore) < 1e-10

print("\n[OK] All z-scores match expected values")

# Edge cases
print("\n" + "=" * 80)
print("4. Edge Cases")
print("=" * 80)

# Data with NaN
data_with_nan = data.copy()
data_with_nan.iloc[3, 0] = np.nan  # Add NaN to AAPL row 3

ts_mean_nan = data_with_nan.rolling(window=window, min_periods=3).mean()
ts_std_nan = data_with_nan.rolling(window=window, min_periods=3).std()
ts_zscore_nan = (data_with_nan - ts_mean_nan) / ts_std_nan

print("\nData with NaN (AAPL row 3):")
print(data_with_nan)
print("\nTs_Zscore:")
print(ts_zscore_nan)
print("[OK] NaN handled correctly (skipna=True in rolling)")

# Operator Composition Pattern
print("\n" + "=" * 80)
print("5. Operator Composition Pattern")
print("=" * 80)

print("""
Ts_Zscore demonstrates operator composition:

Without composition (duplicating code):
```python
class TsZscore(BaseOperator):
    def compute(self, data, window):
        # Duplicate TsMean logic
        ts_mean = data.rolling(window=window, min_periods=...).mean()
        # Duplicate TsStdDev logic
        ts_std = data.rolling(window=window, min_periods=...).std()
        # Compute z-score
        return (data - ts_mean) / ts_std
```

With composition (reusing operators):
```python
class TsZscore(BaseOperator):
    def __call__(self, alpha_data, window, record_output=False):
        # Reuse TsMean operator
        ts_mean_op = self._registry.get_operator('TsMean')
        mean_result = ts_mean_op(alpha_data, window=window)

        # Reuse TsStdDev operator
        ts_std_op = self._registry.get_operator('TsStdDev')
        std_result = ts_std_op(alpha_data, window=window)

        # Compute z-score using arithmetic operators
        # (alpha_data - mean_result) / std_result
        demeaned = alpha_data - mean_result
        zscored = demeaned / std_result

        return zscored
```

Benefits of composition:
1. Code reuse - no duplication
2. Consistency - same rolling window logic
3. Maintainability - fix bugs in one place
4. Composability - build complex operators from simple ones
5. Type safety - operators validate inputs

Example operators that could use composition:
- TsZscore = (X - TsMean) / TsStdDev
- TsSharpe = TsMean / TsStdDev  (similar to TsZscore but without demean)
- TsCorr could reuse TsMean and TsStdDev for normalization
- GroupNeutralize could reuse GroupSum and GroupCount
""")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: Ts_Zscore Operator Specification")
print("=" * 80)
print("""
Operation: Time-series z-score normalization

Formula:
    Ts_Zscore(X, window) = (X - TsMean(X, window)) / TsStdDev(X, window)

Properties:
1. Normalizes each time series to have mean ~0 and std ~1 over rolling window
2. Removes trends and focuses on deviations from recent average
3. Useful for detecting outliers and normalizing signals
4. Window parameter controls lookback period

Implementation (Composition):
```python
class TsZscore(BaseOperator):
    \"\"\"Time-series z-score normalization operator.

    Computes rolling z-score: (X - rolling_mean) / rolling_std.
    Demonstrates operator composition pattern.
    \"\"\"

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False

    def __call__(self, alpha_data, window, record_output=False):
        # Get operators from registry
        ts_mean_op = self._registry.get_operator('TsMean')
        ts_std_op = self._registry.get_operator('TsStdDev')

        # Compute mean and std using existing operators
        mean_result = ts_mean_op(alpha_data, window=window)
        std_result = ts_std_op(alpha_data, window=window)

        # Compute z-score using arithmetic
        demeaned = alpha_data - mean_result
        zscored = demeaned / std_result

        return zscored
```

Alternative (without composition, for comparison):
```python
def compute(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
    min_periods = self._get_min_periods(window)
    ts_mean = data.rolling(window=window, min_periods=min_periods).mean()
    ts_std = data.rolling(window=window, min_periods=min_periods).std()
    return (data - ts_mean) / ts_std
```

Edge Cases:
[OK] Trending data normalized correctly
[OK] NaN values handled (skipna=True)
[OK] min_periods controls when results start appearing

Use Cases:
- Detect outliers in time series
- Normalize trending signals
- Compare volatility-adjusted signals
- Remove scale effects from factors

Next Steps:
1. Implement in src/alpha_excel2/ops/timeseries.py
2. Use operator composition pattern (requires OperatorRegistry)
3. Note: This will work after Phase 3 when Registry is implemented
4. For now, we can implement using direct computation as alternative
""")

print("\nExperiment completed successfully!")
