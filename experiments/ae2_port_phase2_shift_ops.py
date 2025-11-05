"""Experiment: Phase 2 - Simple Shift Operations (TsDelay, TsDelta)

This experiment validates the behavior of shift operations before implementation:
- TsDelay: Shift data by N periods (data.shift(window))
- TsDelta: Calculate difference from N periods ago (data - data.shift(window))

These are the simplest operators - no rolling windows, no config needed.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("PHASE 2 EXPERIMENT: SHIFT OPERATIONS (TsDelay, TsDelta)")
print("=" * 80)

# Create sample data
dates = pd.date_range('2024-01-01', periods=10, freq='D')
data = pd.DataFrame({
    'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'B': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
}, index=dates)

print("\n[ORIGINAL DATA]")
print(data)

# ============================================================================
# TsDelay Tests
# ============================================================================

print("\n" + "=" * 80)
print("OPERATOR 1: TsDelay (shift data by N periods)")
print("=" * 80)

print("\n[TEST 1: TsDelay with window=1]")
print("Expected: Each value shifted down by 1 row, first row becomes NaN")
result_delay1 = data.shift(1)
print(result_delay1)
print(f"First row should be NaN: {result_delay1.iloc[0, 0]}")
print(f"Second row should be 1.0: {result_delay1.iloc[1, 0]}")
print(f"Third row should be 2.0: {result_delay1.iloc[2, 0]}")

print("\n[TEST 2: TsDelay with window=3]")
print("Expected: Each value shifted down by 3 rows, first 3 rows become NaN")
result_delay3 = data.shift(3)
print(result_delay3)
print(f"First 3 rows should be NaN:")
print(f"  Row 0: {result_delay3.iloc[0, 0]}")
print(f"  Row 1: {result_delay3.iloc[1, 0]}")
print(f"  Row 2: {result_delay3.iloc[2, 0]}")
print(f"Fourth row should be 1.0: {result_delay3.iloc[3, 0]}")
print(f"Fifth row should be 2.0: {result_delay3.iloc[4, 0]}")

print("\n[TEST 3: TsDelay with existing NaNs]")
data_with_nans = data.copy()
data_with_nans.iloc[2, 0] = np.nan
print("Data with NaN at row 2:")
print(data_with_nans)
result_delay_nans = data_with_nans.shift(1)
print("\nAfter shift(1):")
print(result_delay_nans)
print(f"NaN at row 2 should appear at row 3: {result_delay_nans.iloc[3, 0]}")

# ============================================================================
# TsDelta Tests
# ============================================================================

print("\n" + "=" * 80)
print("OPERATOR 2: TsDelta (difference from N periods ago)")
print("=" * 80)

print("\n[TEST 1: TsDelta with window=1]")
print("Expected: data - data.shift(1) = difference from previous period")
result_delta1 = data - data.shift(1)
print(result_delta1)
print(f"First row should be NaN: {result_delta1.iloc[0, 0]}")
print(f"Second row should be 1.0 (2 - 1): {result_delta1.iloc[1, 0]}")
print(f"Third row should be 1.0 (3 - 2): {result_delta1.iloc[2, 0]}")
print("For asset A (increasing by 1 each period), all deltas should be 1.0")
print("For asset B (increasing by 10 each period), all deltas should be 10.0")

print("\n[TEST 2: TsDelta with window=3]")
print("Expected: data - data.shift(3) = difference from 3 periods ago")
result_delta3 = data - data.shift(3)
print(result_delta3)
print(f"First 3 rows should be NaN:")
print(f"  Row 0: {result_delta3.iloc[0, 0]}")
print(f"  Row 1: {result_delta3.iloc[1, 0]}")
print(f"  Row 2: {result_delta3.iloc[2, 0]}")
print(f"Fourth row should be 3.0 (4 - 1): {result_delta3.iloc[3, 0]}")
print(f"Fifth row should be 3.0 (5 - 2): {result_delta3.iloc[4, 0]}")
print("For asset A, all deltas should be 3.0 (constant increment)")

print("\n[TEST 3: TsDelta with non-uniform changes]")
data_nonuniform = pd.DataFrame({
    'A': [1.0, 3.0, 4.0, 10.0, 11.0, 13.0, 14.0, 20.0, 21.0, 23.0],
}, index=dates)
print("Data with non-uniform changes:")
print(data_nonuniform)
result_delta_nonuniform = data_nonuniform - data_nonuniform.shift(1)
print("\nDelta (window=1):")
print(result_delta_nonuniform)
print(f"Row 1: 3 - 1 = {result_delta_nonuniform.iloc[1, 0]}")
print(f"Row 2: 4 - 3 = {result_delta_nonuniform.iloc[2, 0]}")
print(f"Row 3: 10 - 4 = {result_delta_nonuniform.iloc[3, 0]}")

print("\n[TEST 4: TsDelta with NaNs]")
result_delta_nans = data_with_nans - data_with_nans.shift(1)
print("Delta with NaN at row 2:")
print(result_delta_nans)
print("When NaN is involved in subtraction, result is NaN")
print(f"Row 2: NaN - 2 = {result_delta_nans.iloc[2, 0]}")
print(f"Row 3: 4 - NaN = {result_delta_nans.iloc[3, 0]}")

# ============================================================================
# Validation Parameters
# ============================================================================

print("\n" + "=" * 80)
print("IMPLEMENTATION PARAMETERS")
print("=" * 80)

print("\n[VALIDATION RULES]")
print("1. window must be positive integer")
print("2. No min_periods needed (shift is straightforward)")
print("3. No config reading needed")
print("4. Both operators accept single AlphaData input")
print("5. Both return numeric type")

print("\n[IMPLEMENTATION PATTERN]")
print("TsDelay:")
print("  return data.shift(window)")
print("\nTsDelta:")
print("  return data - data.shift(window)")

print("\n[OK] Shift operations validated!")
print("Ready to implement TsDelay and TsDelta operators")
print("=" * 80)
